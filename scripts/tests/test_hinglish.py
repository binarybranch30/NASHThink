"""Tests for Hinglish-aware retrieval and Ask (scripts/semantic/hinglish.py, search.search_expanded,
scripts/api/memory_brief.py). Synthetic chats only. Retrieval runs against a real throwaway LanceDB table in
a temp directory (so the LIKE prefilter is exercised for real) with a deterministic bag-of-words stand-in
for the embedding model that, like the real one, matches Hinglish *style* words as strongly as topic words.
Run: .venv/bin/python scripts/tests/test_hinglish.py"""
import math
import os
import re
import sys
import tempfile
import unittest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.append(SCRIPT_DIR)
sys.path.append(os.path.join(REPO_ROOT, "scripts", "semantic"))
sys.path.append(os.path.join(REPO_ROOT, "scripts", "api"))

import hinglish  # noqa: E402
import memory_brief as mb  # noqa: E402
import search  # noqa: E402

try:
    import lancedb
    HAVE_LANCEDB = True
except ImportError:
    HAVE_LANCEDB = False

# ─── Synthetic corpus ─────────────────────────────────────────────────────────
DAY = "2024-03-0{}"


def chat(pid, platform, day, lines, title=None):
    body = "\n".join(f"[{DAY.format(day)} 09:0{k}:00] {a}: {t}" for k, (a, t) in enumerate(lines))
    return {"parent_id": str(pid), "channel_id": str(pid), "platform": platform, "title": title or f"Chat {pid}",
            "start_time": f"{DAY.format(day)}T09:00:00+00:00", "end_time": f"{DAY.format(day)}T10:00:00+00:00",
            "text": f"Platform: {platform.upper()}\nTitle: {title or f'Chat {pid}'}\n{body}"}


EN_SLEEP = chat(1, "reddit", 1, [("Me", "I have been sleeping badly at night for weeks and wake up tired"),
                                 ("Sam", "try no screens before bed, it helped my sleep")], "Sleep help")
HI_SLEEP = chat(2, "instagram", 2, [("Me", "yaar kal raat bhi neend nahi aayi, bas phone chalata raha"),
                                    ("Riya", "tu late tak jaagta hai isliye nind nahi aati")], "DM Riya")
HI_SLEEP_VARIANT = chat(3, "instagram", 3, [("Me", "aaj neendh bohot achhi aayi finally, 9 ghante so gaya"),
                                            ("Riya", "wah finally neendh poori hui")], "DM Riya 2")
HI_CHATTER = [chat(10 + k, "instagram", 4, [("Me", f"yaar mujhe nahi pata kya hua {w}, bas aise hi"),
                                            ("Dev", f"haan yaar mujhe bhi nahi aati samajh {w}")], f"DM Dev {k}")
              for k, w in enumerate(("kal", "aaj", "abhi", "phir", "waise"))]
EN_STUDY = chat(20, "reddit", 5, [("Me", "exam stress is killing me, I cannot focus on study at all"),
                                  ("Kay", "break the syllabus into small daily chunks")], "Exam season")
HI_STUDY = chat(21, "instagram", 6, [("Me", "padhai ka itna tension hai, pariksha agle hafte hai"),
                                     ("Riya", "tension mat le, roz thoda padh le")], "DM Riya 3")
TRAP = chat(30, "reddit", 7, [("Me", "for example the standard library has a nice example of this pattern"),
                              ("Kay", "that example is from the docs, pretty standard")], "Python example")
CAMERA = chat(31, "reddit", 8, [("Me", "which camera lens should I buy for street photography"),
                                ("Sam", "a 35mm prime lens is a great street camera lens")], "Camera lens")
ROWS = [EN_SLEEP, HI_SLEEP, HI_SLEEP_VARIANT, *HI_CHATTER, EN_STUDY, HI_STUDY, TRAP, CAMERA]
QUERIES = ["yaar mujhe neend nahi aati", "trouble sleeping at night", "padhai ka bahut tension hai", "exam stress",
           "which camera lens should I buy", "standard example"]


class BagOfWords:
    """Deterministic embedding stand-in: word counts over the corpus vocabulary + a constant bias dimension.
    Shared filler words (yaar, mujhe, nahi) pull Hinglish chats together just like topic words do."""

    def __init__(self, texts):
        vocab = sorted({w for t in texts for w in re.findall(r"[a-z]+", t.lower())})
        self.index = {w: i for i, w in enumerate(vocab)}
        self.dim = len(vocab) + 1
        self.queries = []

    def encode(self, text, convert_to_numpy=True):
        self.queries.append(text)
        v = [0.0] * self.dim
        v[-1] = 0.3
        for w in re.findall(r"[a-z]+", text.lower()):
            if w in self.index:
                v[self.index[w]] += 1.0
        n = math.sqrt(sum(x * x for x in v))
        return [x / n for x in v]


def body(row):
    return "\n".join(l for l in row["text"].splitlines() if not l.startswith(("Platform:", "Title:")))


# ─── Vocabulary ───────────────────────────────────────────────────────────────
class VocabularyTests(unittest.TestCase):
    def test_spelling_variants_normalise(self):
        for variant, canon in [("nhi", "nahi"), ("nai", "nahi"), ("nahin", "nahi"), ("bohot", "bahut"), ("bhot", "bahut"),
                               ("padhaai", "padhai"), ("parhai", "padhai"), ("nind", "neend"), ("neendh", "neend"),
                               ("pareeksha", "pariksha"), ("NHI", "nahi")]:
            self.assertEqual(hinglish.normalize(variant), canon, variant)
        for english in ("sleep", "camera", "example", "standard", "hay", "not"):
            self.assertEqual(hinglish.normalize(english), english, "English words are never rewritten")

    def test_concepts_in_both_languages(self):
        self.assertEqual(hinglish.concept_of("neend"), ("sleep", "hi"))
        self.assertEqual(hinglish.concept_of("nindh"), ("sleep", "hi"))
        self.assertEqual(hinglish.concept_of("Sleeping"), ("sleep", "en"))
        self.assertEqual(hinglish.concept_of("padhaai"), ("study", "hi"))
        self.assertEqual(hinglish.concept_of("exams"), ("exam", "en"))
        self.assertEqual(hinglish.concept_of("camera"), (None, None))

    def test_filler_and_negation_are_not_topics(self):
        for w in ("hai", "mujhe", "yaar", "bahut", "bohot", "kya", "nahi", "nhi"):
            self.assertTrue(hinglish.is_filler(w), w)
        for w in ("nahi", "nhi", "nai", "not", "never", "don't", "mat"):
            self.assertTrue(hinglish.is_negation(w), w)
        self.assertFalse(hinglish.is_filler("neend"))

    def test_ambiguous_short_words_only_filler_in_hinglish(self):
        eng = hinglish.words("main reason for the tab crash")
        self.assertFalse(hinglish.looks_hinglish(eng))
        self.assertFalse(hinglish.is_filler("main", hinglish.looks_hinglish(eng)))
        self.assertFalse(hinglish.is_filler("tab", False))
        self.assertTrue(hinglish.looks_hinglish(hinglish.words("main kya karu yaar")))
        self.assertTrue(hinglish.is_filler("main", True))

    def test_expansion_both_directions(self):
        en = hinglish.Expansion("trouble sleeping at night")
        self.assertIn("neend", en.terms)
        self.assertIn("nind", en.terms)
        self.assertNotIn("sleep", en.terms, "English words in the query are left to the vector search")
        self.assertNotIn("raat", en.terms, "night/raat is too generic to expand")
        hi = hinglish.Expansion("yaar mujhe neend nahi aati")
        self.assertIn("sleep", hi.terms)
        self.assertIn("nind", hi.terms, "Hinglish query words also bring their spelling variants")
        both = hinglish.Expansion("padhai ka bahut tension hai")
        self.assertTrue({"study", "stress", "padhaai"} <= set(both.terms))

    def test_expansion_is_conservative(self):
        for q in ("which camera lens should I buy", "feeling bored and missing friends", "yaar kya haal hai",
                  "mujhe nahi pata", "love my home"):
            self.assertFalse(hinglish.Expansion(q), q)
        self.assertIn("akela", hinglish.Expansion("feeling lonely").terms, "loneliness is a listed topic now")
        self.assertEqual(hinglish.Expansion("yaar mujhe neend nahi aati").concepts, [("sleep", "neend", "hi")])

    def test_expansion_found_needs_whole_words(self):
        exp = hinglish.Expansion("exam stress")
        self.assertIn("pariksha", exp.found("kal pariksha hai"))
        pain = hinglish.Expansion("back pain")
        self.assertEqual(pain.found("the standard library"), [], "dard inside 'standard' is not a match")
        self.assertEqual(pain.found("kamar mein dard hai"), ["dard"])

    def test_filter_expression_is_plain_literals(self):
        expr = hinglish.Expansion("yaar mujhe neend nahi aati").filter_expression()
        self.assertTrue(expr.startswith("(") and expr.endswith(")"))
        for t in re.findall(r"'%([^%]*)%'", expr):
            self.assertRegex(t, r"^[a-z]{3,}$")


# ─── Retrieval on a real temporary LanceDB table ──────────────────────────────
@unittest.skipUnless(HAVE_LANCEDB, "lancedb not installed")
class RetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.model = BagOfWords([r["text"] for r in ROWS] + QUERIES)
        data = [{**r, "vector": cls.model.encode(r["text"])} for r in ROWS]
        cls.table = lancedb.connect(cls.tmp.name).create_table("topics", data=data)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def ids(self, results):
        return [r["channel_id"] for r in results]

    def test_hinglish_query_reaches_topic_not_just_style(self):
        plain = search.search(self.table, self.model, "yaar mujhe neend nahi aati", 4)
        self.assertTrue(set(self.ids(plain)) <= {r["channel_id"] for r in HI_CHATTER} | {"2"},
                        "sanity: the stand-in model reproduces the style-over-topic problem")
        plain8 = self.ids(search.search(self.table, self.model, "yaar mujhe neend nahi aati", 8))
        got = self.ids(search.search_expanded(self.table, self.model, "yaar mujhe neend nahi aati", 8))
        self.assertIn("1", got, "Hinglish -> English: the English sleep chat is found")
        self.assertIn("3", got, "spelling variant 'neendh' is found")
        self.assertEqual(got[0], plain8[0], "expansion interleaves; it never displaces the top plain hit")
        self.assertIn("2", got[:4], "the first expansion hit lands right after the top plain hits")
        on_topic = lambda ids: sum(i in ("1", "2", "3") for i in ids)
        self.assertGreater(on_topic(got), on_topic(plain8))

    def test_english_query_reaches_hinglish(self):
        plain = self.ids(search.search(self.table, self.model, "trouble sleeping at night", 3))
        self.assertNotIn("2", plain)
        got = search.search_expanded(self.table, self.model, "trouble sleeping at night", 4)
        self.assertEqual(got[0]["channel_id"], "1", "the best English match keeps first place")
        hi = [r for r in got if r["channel_id"] in ("2", "3")]
        self.assertTrue(hi, "English -> Hinglish: a neend/neendh chat is found")
        self.assertTrue(all(r["matched_via"] in ("expansion", "both") for r in hi))
        self.assertTrue(all({"neend", "neendh", "nind"} & set(r["expansion_terms"]) for r in hi))

    def test_merge_deduplicates_and_keeps_metadata(self):
        got = search.search_expanded(self.table, self.model, "padhai ka bahut tension hai", 8)
        keys = [search.chunk_key(r) for r in got]
        self.assertEqual(len(keys), len(set(keys)), "a chunk found by both searches appears once")
        study = next(r for r in got if r["channel_id"] == "21")
        self.assertEqual(study["matched_via"], "both")
        self.assertTrue({"padhai", "tension"} & set(study["expansion_terms"]))
        plain = {r["channel_id"]: r for r in search.search(self.table, self.model, "padhai ka bahut tension hai", 20)}
        for r in got:
            self.assertEqual(r["similarity"], plain[r["channel_id"]]["similarity"], "similarity is the real cosine")
            self.assertEqual((r["platform"], r["title"], r["start_time"]),
                             (plain[r["channel_id"]]["platform"], plain[r["channel_id"]]["title"], plain[r["channel_id"]]["start_time"]))
        self.assertEqual([r["rank"] for r in got], list(range(1, len(got) + 1)))

    def test_unrelated_query_is_unchanged(self):
        for q in ("which camera lens should I buy", "standard example"):
            plain = search.search(self.table, self.model, q, 5)
            exp = search.search_expanded(self.table, self.model, q, 5)
            self.assertEqual(self.ids(exp), self.ids(plain), q)
            self.assertTrue(all(r["matched_via"] == "query" and r["expansion_terms"] == [] for r in exp))

    def test_expansion_does_not_promote_substring_matches(self):
        got = search.search_expanded(self.table, self.model, "exam stress", 10)
        trap = [r for r in got if r["channel_id"] == "30"]
        self.assertTrue(all(r["matched_via"] == "query" for r in trap), "'example' is not the exam topic")

    def test_filters_still_apply_to_expansion(self):
        where = search.filter_expression(["reddit"])
        got = search.search_expanded(self.table, self.model, "yaar mujhe neend nahi aati", 10, where=where)
        self.assertTrue(got)
        self.assertTrue(all(r["platform"] == "reddit" for r in got))

    def test_encodes_query_once(self):
        self.model.queries.clear()
        search.search_expanded(self.table, self.model, "trouble sleeping at night", 3)
        self.assertEqual(self.model.queries, ["trouble sleeping at night"])


# ─── Ask (memory_brief) ───────────────────────────────────────────────────────
def as_result(row, similarity, rank=1):
    return {**row, "rank": rank, "similarity": similarity, "distance": round(1 - similarity, 4), "node_id": f"T_{row['parent_id']}",
            "people": [], "summary": None, "snippet": ""}


class AskTests(unittest.TestCase):
    def test_hinglish_terms_skip_filler_and_negation(self):
        self.assertEqual(mb.content_terms("yaar mujhe neend nahi aati"), ["neend"])
        self.assertEqual(mb.content_terms("padhaai ka bohot tension hai"), ["padhai", "tension"])
        self.assertEqual(mb.content_terms("What was I stressed about during college?"), ["stressed", "college"],
                         "English questions keep their terms")
        self.assertEqual(mb.content_terms("main reason for the tab crash"), ["main", "reason", "tab", "crash"],
                         "ambiguous short words stay topics in English questions")

    def test_spelling_variant_counts_as_evidence(self):
        brief = mb.build_brief("mujhe neend nahi aati", [as_result(HI_SLEEP_VARIANT, 0.6)])
        self.assertEqual(brief["evidence"]["relevant"], 1)
        self.assertTrue(brief["sources"][0]["relevant"])
        self.assertEqual(brief["sources"][0]["matched_terms"], ["neend"], "neendh is the same word as neend")

    def test_english_question_uses_hinglish_evidence(self):
        brief = mb.build_brief("how was my sleep", [as_result(HI_SLEEP, 0.24)])
        src = brief["sources"][0]
        self.assertTrue(src["relevant"], "cross-language support passes the lower similarity floor")
        self.assertEqual(src["matched_terms"], ["neend"], "shows the word that was actually found")
        self.assertIn("neend", brief["answer"] + " ".join(brief["summary_points"]))

    def test_hinglish_question_uses_english_evidence(self):
        brief = mb.build_brief("neend kaisi thi", [as_result(EN_SLEEP, 0.3)])
        self.assertTrue(brief["sources"][0]["relevant"])
        self.assertIn(brief["sources"][0]["matched_terms"][0], {"sleep", "sleeping"})

    def test_cross_language_floor_is_still_a_floor(self):
        brief = mb.build_brief("how was my sleep", [as_result(HI_SLEEP, 0.15)])
        self.assertFalse(brief["sources"][0]["relevant"])
        self.assertEqual(brief["confidence"], "low")
        same = mb.build_brief("how was my sleep", [as_result(EN_SLEEP, 0.3)])
        self.assertFalse(same["sources"][0]["relevant"], "same-language evidence keeps the normal 0.35 floor")

    def test_negation_is_kept_in_quotes_and_answer(self):
        brief = mb.build_brief("neend", [as_result(HI_SLEEP, 0.6)])
        quote = brief["sources"][0]["snippet"]
        self.assertIn("nahi", quote, "the negation stays in the quoted sentence")
        self.assertIn("neend nahi aayi", brief["answer"], "answer quotes it verbatim, not reversed")
        self.assertNotIn("neend aayi", brief["answer"].replace("neend nahi aayi", ""))
        self.assertNotIn("nahi", brief["evidence"]["terms"])

    def test_filler_only_chat_is_not_evidence(self):
        results = [as_result(r, 0.8, k) for k, r in enumerate(HI_CHATTER, start=1)]
        brief = mb.build_brief("yaar mujhe neend nahi aati", results)
        self.assertEqual(brief["evidence"]["relevant"], 0, "sharing yaar/mujhe/nahi is not evidence")
        self.assertEqual(brief["confidence"], "low")
        self.assertIn("No relevant info found", brief["answer"])

    def test_unrelated_substring_is_not_evidence(self):
        brief = mb.build_brief("exam stress", [as_result(TRAP, 0.5)])
        self.assertFalse(brief["sources"][0]["relevant"], "'example' does not mention exams")
        pain = mb.build_brief("dard", [as_result(TRAP, 0.5)])
        self.assertFalse(pain["sources"][0]["relevant"], "'standard' does not mention dard")

    def test_english_behaviour_unchanged(self):
        words = mb.words_of("I was stressful about my photographs")
        self.assertTrue(mb.mentions(words, mb.term_key("stressed")))
        brief = mb.build_brief("which camera lens", [as_result(CAMERA, 0.6), as_result(TRAP, 0.6, 2)])
        self.assertEqual([s["relevant"] for s in brief["sources"]], [True, False])
        self.assertEqual(brief["sources"][0]["matched_terms"], ["camera", "lens"])


try:
    from unittest import mock
    from fastapi.testclient import TestClient
    import server
    HAVE_FASTAPI = True
except ImportError:
    HAVE_FASTAPI = False


@unittest.skipUnless(HAVE_LANCEDB and HAVE_FASTAPI, "lancedb/fastapi not installed")
class EndpointTests(unittest.TestCase):
    """/api/search and /api/ask over the temp table: expansion fields, citations and graph links."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.model = BagOfWords([r["text"] for r in ROWS] + QUERIES)
        cls.table = lancedb.connect(cls.tmp.name).create_table("topics", data=[{**r, "vector": cls.model.encode(r["text"])} for r in ROWS])

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def client(self):
        meta = {"table": "topics", "model": "fake-bow", "vector_dim": self.model.dim, "rows": len(ROWS), "created_at": "x"}
        self.enterContext(mock.patch.object(search, "open_table", return_value=self.table))
        self.enterContext(mock.patch.object(search, "table_model_metadata", return_value=meta))
        svc = server.SearchService(self.tmp.name, "topics", model_loader=lambda name: self.model)
        return TestClient(server.create_app(svc, graph_html=__file__, graph_dir=self.tmp.name))

    def test_search_reports_expansion_and_keeps_graph_links(self):
        body = self.client().post("/api/search", json={"query": "trouble sleeping at night", "limit": 4}).json()
        self.assertEqual(body["results"][0]["node_id"], "T_1")
        exp = [r for r in body["results"] if r["matched_via"] != "query"]
        self.assertTrue(exp)
        for r in body["results"]:
            self.assertEqual(r["node_id"], f"T_{r['channel_id']}")
            self.assertIn(r["matched_via"], ("query", "expansion", "both"))
        self.assertTrue(all(r["expansion_terms"] for r in exp))

    def test_ask_hinglish_question_cites_both_languages(self):
        body = self.client().post("/api/ask", json={"question": "yaar mujhe neend nahi aati", "limit": 8}).json()
        rel = [s for s in body["sources"] if s["relevant"]]
        self.assertEqual(body["evidence"]["terms"], ["neend"])
        cited = {s["node_id"]: s for s in body["sources"]}
        self.assertIn("T_2", {s["node_id"] for s in rel}, "the Hinglish sleep chat is evidence")
        self.assertTrue({"T_1", "T_3"} <= set(cited), "expansion brings the English and variant-spelling chats in")
        # With this stand-in model they share no words with the question (similarity ~0), so the similarity
        # floor keeps them as leads rather than evidence: expansion alone never makes something evidence.
        self.assertFalse(cited["T_1"]["relevant"])
        self.assertEqual(cited["T_1"]["matched_terms"], ["sleep"])
        self.assertFalse({"T_10", "T_11", "T_12", "T_13", "T_14"} & {s["node_id"] for s in rel}, "small talk is not")
        for s in rel:
            if "neend" in s["snippet"] and "nahi" in s["text"]:
                self.assertIn("nahi", s["snippet"])


class SafeClipTests(unittest.TestCase):
    def test_short_text_untouched(self):
        self.assertEqual(mb.safe_clip("neend nahi aati", 50), "neend nahi aati")

    def test_negation_after_cut_is_kept(self):
        text = "kal raat " + "bahut der tak phone pe baat hoti rahi aur phir " * 4 + "neend bilkul nahi aayi thi yaar sach mein"
        out = mb.safe_clip(text, 120)
        self.assertIn("neend bilkul nahi aayi", out)
        self.assertLessEqual(len(out), 241)
        plain = mb.clip(text, 120)
        self.assertNotIn("nahi", plain, "sanity: a plain clip would have dropped it")

    def test_english_negation_kept(self):
        text = "I thought the new routine would fix my sleep after all those weeks of trying things but honestly it did not work at all"
        self.assertIn("did not work", mb.safe_clip(text, 60))

    def test_negation_too_far_rejects_the_quote(self):
        text = "neend " + "x" * 10 + " " + "word " * 80 + "nahi aayi"
        self.assertIsNone(mb.safe_clip(text, 60, hard_max=120))

    def test_quote_is_never_negation_truncated(self):
        long = "yaar " + "aaj office mein kaam bahut tha aur ghar aake bhi laptop khula raha phir " * 3 + "neend nahi aayi raat bhar"
        row = chat(40, "instagram", 9, [("Me", long)])
        brief = mb.build_brief("neend", [as_result(row, 0.6)])
        quote = brief["sources"][0]["snippet"]
        if "neend" in quote:
            self.assertIn("nahi", quote)



# ─── Shorthand, spelling tolerance and the lexical channel ────────────────────
HI_FIGHT = chat(40, "discord", 1, [("Me", "kal ghar pe bahut ladaai hui mummy papa ke beech"),
                                   ("Ana", "arre yaar, phir se jhagda? tu theek hai?")], "DM Ana")
HI_FIGHT_VARIANT = chat(41, "discord", 2, [("Me", "unki ladayi dekh ke mera mood kharab ho gaya h"),
                                           ("Ana", "bhai chill kar")], "DM Ana 2")
HI_HOME_ONLY = chat(42, "discord", 3, [("Me", "ghar pe aaj pizza bana, sab khush the"),
                                       ("Ana", "wah party ho gayi fir")], "DM Ana 3")
JEE_TITLE = chat(43, "reddit", 4, [("Me", "kal ka match dekha kya, kya jeet thi bhai"),
                                   ("Kay", "haan yaar last over mein jeet gaye")], "r/JEENEETards (Thread 1)")
JEE_BODY = chat(44, "reddit", 5, [("Me", "jee ki tayyari ab serious karni padegi"),
                                  ("Kay", "haan taiyari abhi se shuru kar")], "r/JEENEETards (Thread 2)")
LEX_ROWS = [HI_FIGHT, HI_FIGHT_VARIANT, HI_HOME_ONLY, JEE_TITLE, JEE_BODY, *HI_CHATTER]


class ShorthandTests(unittest.TestCase):
    def test_chat_shorthand_normalises_in_hinglish_only(self):
        self.assertEqual(hinglish.normalize("ni"), "nahi")
        self.assertEqual(hinglish.normalize("h"), "hai")
        self.assertEqual(hinglish.normalize("clg"), "college")
        self.assertTrue(hinglish.is_negation("ni"), "'ni' is a negation, so quotes never drop it")
        self.assertTrue(hinglish.is_filler("h", True))
        self.assertFalse(hinglish.is_filler("m", False), "single letters are only shorthand in Hinglish text")

    def test_looks_hinglish(self):
        for q in ("jee ki tayyari", "exam ki tension", "akela feel hota hai", "paise ki problem"):
            self.assertTrue(hinglish.looks_hinglish(hinglish.words(q)), q)
        for q in ("main reason for the tab crash", "tension headache and the flu", "which camera lens should I buy"):
            self.assertFalse(hinglish.looks_hinglish(hinglish.words(q)), q)

    def test_skeleton_joins_spellings(self):
        for group in (("ladai", "ladaai", "ladayi", "laddai"), ("tayyari", "taiyari", "tayari", "taiyaari"),
                      ("padhai", "padhaai", "padai"), ("neend", "nind", "neendh")):
            self.assertEqual(len({hinglish.skeleton(w) for w in group}), 1, group)
        self.assertNotEqual(hinglish.skeleton("ladai"), hinglish.skeleton("ladki"))

    def test_word_pattern_is_whole_word_and_safe(self):
        rx = hinglish.whole_word_regex([hinglish.word_pattern("ladai")])
        for w in ("ladai", "ladaai", "ladayi", "LADAI"):
            self.assertTrue(rx.search(f"kal {w} hui"), w)
        for w in ("ladki", "xladai", "ladaix"):
            self.assertFalse(rx.search(f"kal {w} hui"), f"{w}: a different word, or ladai inside another word")
        akela = hinglish.whole_word_regex([hinglish.word_pattern("akela")])
        self.assertTrue(all(akela.search(w) for w in ("akela", "akele", "akeli")), "inflections")
        jee = hinglish.whole_word_regex([hinglish.word_pattern("jee")])
        self.assertTrue(jee.search("JEE mains"))
        self.assertFalse(any(jee.search(w) for w in ("jeet", "jeena", "jeeneetards")), "short words match exactly")
        college = hinglish.whole_word_regex([hinglish.word_pattern("college")])
        self.assertTrue(college.search("kal clg nahi gaya"), "listed variants are part of the pattern")
        for w in ("ladai", "tayyari", "college", "jee"):
            self.assertRegex(hinglish.word_pattern(w), r"^[a-z()?:+|\[\]{},0-9]+$", "only letters and regex syntax")

    def test_expansion_covers_unlisted_words_and_skips_generic_ones(self):
        e = hinglish.Expansion("jee ki tayyari")
        self.assertEqual([c for c, _, _ in e.concepts], ["prep"])
        self.assertEqual([n for n, _ in e.patterns], ["jee"], "an unlisted Hinglish word still gets a pattern")
        fight = hinglish.Expansion("ghar pe kya ladai hui")
        self.assertEqual(fight.patterns, [], "ghar is generic; the question is about the fight")
        dost = hinglish.Expansion("dost se baat nahi hui")
        self.assertEqual([n for n, _ in dost.patterns], ["dost"], "a generic word is used when it is all there is")
        self.assertIn("regexp_like(text,", dost.filter_expression())
        self.assertEqual(dost.found("mera dost aaya"), ["dost"])
        self.assertEqual(hinglish.Expansion("akela feel hota hai").patterns, [], "'feel' is scaffolding, not a topic")


@unittest.skipUnless(HAVE_LANCEDB, "lancedb not installed")
class LexicalRetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        qs = ["ghar pe kya ladai hui", "jee ki tayyari", "dost se baat nahi hui"]
        cls.model = BagOfWords([r["text"] for r in LEX_ROWS] + qs)
        data = [{**r, "vector": cls.model.encode(r["text"])} for r in LEX_ROWS]
        cls.table = lancedb.connect(cls.tmp.name).create_table("topics", data=data)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_regex_prefilter_runs_in_lancedb(self):
        exp = hinglish.Expansion("ghar pe kya ladai hui")
        rows = self.table.search().where(exp.filter_expression()).limit(50).to_list()
        self.assertEqual({r["channel_id"] for r in rows}, {"40", "41"}, "both spellings, and not the pizza chat")

    def test_every_spelling_of_the_topic_is_found(self):
        got = search.search_expanded(self.table, self.model, "ghar pe kya ladai hui", 6)
        ids = [r["channel_id"] for r in got]
        self.assertIn("40", ids)
        self.assertIn("41", ids, "ladayi is found for a question about ladai")
        via = {r["channel_id"]: r["expansion_terms"] for r in got}
        self.assertTrue({"ladaai", "jhagda"} & set(via["40"]) or via["40"])

    def test_title_alone_is_not_evidence(self):
        results = [as_result(JEE_TITLE, 0.7, 1), as_result(JEE_BODY, 0.6, 2)]
        brief = mb.build_brief("jee ki tayyari", results)
        rel = {s["title"]: s["relevant"] for s in brief["sources"]}
        self.assertFalse(rel["r/JEENEETards (Thread 1)"], "'jee' in the subreddit name and 'jeet' in the text don't count")
        self.assertTrue(rel["r/JEENEETards (Thread 2)"])

    def test_generic_word_alone_is_not_evidence(self):
        brief = mb.build_brief("ghar pe kya ladai hui", [as_result(HI_HOME_ONLY, 0.8, 1), as_result(HI_FIGHT_VARIANT, 0.6, 2)])
        rel = {s["title"]: s["relevant"] for s in brief["sources"]}
        self.assertFalse(rel["DM Ana 3"], "only 'ghar' matches: it doesn't answer a question about a fight")
        self.assertTrue(rel["DM Ana 2"], "'ladayi' is the same word as 'ladai'")


class RelevanceFixTests(unittest.TestCase):
    def test_negations_are_never_topics(self):
        self.assertEqual(mb.content_terms("I couldn't sleep"), ["sleep"])
        self.assertEqual(mb.content_terms("mujhe neend ni aati"), ["neend"])
        self.assertEqual(mb.content_terms("akela feel hota hai"), ["akela"])

    def test_long_chunk_shows_the_matching_lines(self):
        filler = "\n".join(f"[2024-03-01 09:{k:02d}:00] Dev: kuch bhi random baat {k} yaar" for k in range(60))
        row = chat(50, "discord", 1, [("Me", "placeholder")], "DM Dev long")
        row["text"] = filler + "\n[2024-03-01 10:30:00] Me: kal raat phir neend nahi aayi yaar\n" + filler
        brief = mb.build_brief("mujhe neend nahi aati", [as_result(row, 0.6)])
        text = brief["sources"][0]["text"]
        self.assertLessEqual(len(text), mb.SOURCE_TEXT_CHARS)
        self.assertIn("neend nahi aayi", text, "the matching line is kept even though it is past the first 2000 chars")
        self.assertIn(mb.GAP, text)

if __name__ == "__main__":
    unittest.main()
