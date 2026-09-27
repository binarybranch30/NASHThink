"""Tests for Ask Sarthink: POST /api/ask and scripts/api/memory_brief.py. Synthetic archive text only; a fake
model and table stand in for sentence-transformers and LanceDB, so nothing is downloaded, embedded or read
from the real index. Needs fastapi + httpx (.venv).
Run: .venv/bin/python scripts/tests/test_ask.py"""
import datetime as dt
import os
import re
import sys
import unittest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.append(SCRIPT_DIR)
sys.path.append(os.path.join(REPO_ROOT, "scripts", "semantic"))
sys.path.append(os.path.join(REPO_ROOT, "scripts", "api"))

import memory_brief as mb  # noqa: E402
import search  # noqa: E402
from test_api import HAVE_FASTAPI, MODEL, ApiTestCase, FakeTable, Loader  # noqa: E402

if HAVE_FASTAPI:
    import server  # noqa: E402

UTC = dt.timezone.utc


def row(pid, platform, day, lines, distance=0.4, title=None):
    """A topic chunk as embedder.py stores it: chat lines '[ts] Author: text' (or reddit post blocks)."""
    start = f"{day}T09:00:00+00:00"
    if platform == "reddit":
        body = "\n".join(f"[POST] u/{a} — {day} 09:0{k}:00\n{t}\n" for k, (a, t) in enumerate(lines))
    else:
        body = "\n".join(f"[{day} 09:0{k}:00] {a}: {t}" for k, (a, t) in enumerate(lines))
    return {"vector": [0.0, 0.0, 1.0], "parent_id": str(pid), "platform": platform, "title": title or f"Thread {pid}",
            "start_time": start, "end_time": f"{day}T10:00:00+00:00", "text": body, "_distance": distance}


SOURDOUGH = [
    row(1, "reddit", "2024-01-10", [("tester", "Started my first sourdough starter this week and it smells like apples."),
                                    ("other", "Feed it twice a day and keep it warm.")], 0.30, "Bread beginners"),
    row(2, "claude", "2024-03-05", [("Me", "How long should I proof sourdough baking in a cold kitchen?"),
                                    ("Claude", "Try an overnight proof in the fridge for a more open crumb.")], 0.35),
    row(3, "instagram", "2024-06-20", [("Me", "My sourdough baking finally produced a loaf with real ear and oven spring."),
                                       ("Friend", "That crust looks amazing, share the recipe!")], 0.40),
    row(4, "claude", "2024-06-21", [("Me", "Can sourdough baking work with whole wheat flour only, or is it too dense?"),
                                    ("Claude", "It works, but hydration needs to go up.")], 0.42),
]
NOISE = [
    row(20, "reddit", "2024-02-01", [("x", "lol")], 0.20),     # tiny chunk with a high similarity
    row(21, "instagram", "2024-02-02", [("y", "ok sure")], 0.22),
    row(22, "claude", "2024-02-03", [("Me", "Planning the weekend hiking route around the lake and the ridge.")], 0.55),
]


@unittest.skipUnless(HAVE_FASTAPI, "fastapi not installed (.venv/bin/pip install fastapi uvicorn)")
class AskApiTests(ApiTestCase):
    def ask(self, client, **payload):
        payload.setdefault("question", "What did I say about sourdough baking?")
        return client.post("/api/ask", json=payload)

    def assertGrounded(self, body):
        """Every quote in the answer and key points is verbatim text from a returned source."""
        haystack = " ".join(re.sub(r"\s+", " ", s["text"]) for s in body["sources"])
        titles = " ".join(s["title"] or "" for s in body["sources"])
        for chunk in [body["answer"], *body["summary_points"]]:
            for quote in re.findall(r"“([^”]+)”", chunk):
                quote = quote.rstrip("…")
                self.assertTrue(quote in haystack or quote.rstrip("…") in titles, f"ungrounded quote: {quote!r}")

    def test_grounded_response_shape(self):
        c = self.client(FakeTable(SOURDOUGH + NOISE))
        r = self.ask(c)
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        for key in ("question", "answer", "confidence", "summary_points", "timeline", "sources", "took_ms", "notes", "evidence"):
            self.assertIn(key, body)
        self.assertEqual(body["question"], "What did I say about sourdough baking?")
        self.assertIn(body["confidence"], ("high", "medium", "low"))
        self.assertEqual(body["confidence"], "high")
        self.assertIsInstance(body["took_ms"], int)
        self.assertGrounded(body)

        src = body["sources"]
        self.assertLessEqual(len(src), 8)
        for s in src:
            for key in ("node_id", "title", "platform", "date_start", "date_end", "similarity", "snippet", "text"):
                self.assertIn(key, s)
        relevant = [s for s in src if s["relevant"]]
        self.assertEqual({s["node_id"] for s in relevant}, {"T_1", "T_2", "T_3", "T_4"})
        self.assertTrue(all(s["relevant"] for s in src[:len(relevant)]), "evidence comes before the closest non-matches")
        self.assertNotIn("T_20", {s["node_id"] for s in relevant}, "tiny chunks are never evidence")
        self.assertNotIn("T_22", {s["node_id"] for s in relevant}, "similar but off-topic chunks are not evidence")
        # Dates come from the messages inside the chunk.
        self.assertEqual(next(s for s in src if s["node_id"] == "T_1")["date_start"], "2024-01-10T09:00:00+00:00")

        dates = [t["date"] for t in body["timeline"]]
        self.assertEqual(dates, sorted(dates))
        self.assertTrue({t["node_id"] for t in body["timeline"]} <= {s["node_id"] for s in src})
        for t in body["timeline"]:
            self.assertEqual(src[t["source"] - 1]["node_id"], t["node_id"])
        self.assertTrue(body["summary_points"])
        self.assertIn("sourdough", body["answer"])
        self.assertEqual(body["evidence"]["terms"], ["sourdough", "baking"])

    def test_answer_does_not_invent(self):
        body = self.ask(self.client(FakeTable(SOURDOUGH))).json()
        # The only feelings/people in the answer are ones quoted from sources.
        for word in ("stressed", "happy", "girlfriend", "sister", "loved"):
            self.assertNotIn(word, body["answer"].lower())

    def test_trend_question_quotes_earliest_and_latest(self):
        body = self.ask(self.client(FakeTable(SOURDOUGH)), question="How has my sourdough baking changed?").json()
        self.assertIn("Earliest (Jan 2024", body["answer"])
        self.assertIn("Latest (Jun 2024", body["answer"])
        self.assertGrounded(body)

    def test_duplicates_merged(self):
        dup = dict(SOURDOUGH[0], start_time="2024-01-10T11:00:00+00:00", _distance=0.31)  # overlapping session window
        body = self.ask(self.client(FakeTable([SOURDOUGH[0], dup, SOURDOUGH[1]]))).json()
        self.assertEqual([s["node_id"] for s in body["sources"]].count("T_1"), 1)
        self.assertTrue(any("Merged 1" in n for n in body["notes"]))

    def test_no_results(self):
        body = self.ask(self.client(FakeTable([]))).json()
        self.assertEqual(body["confidence"], "low")
        self.assertIn("couldn't find", body["answer"])
        self.assertEqual((body["sources"], body["summary_points"], body["timeline"]), ([], [], []))

    def test_weak_evidence_is_low_and_shows_closest(self):
        body = self.ask(self.client(FakeTable(NOISE)), question="What did I say about sourdough baking?").json()
        self.assertEqual(body["confidence"], "low")
        self.assertIn("evidence is weak", body["answer"])
        self.assertIn("“sourdough”", body["answer"])
        self.assertEqual(body["summary_points"], [])
        self.assertTrue(body["sources"], "closest sources are still shown as leads")
        self.assertFalse(any(s["relevant"] for s in body["sources"]))

    def test_few_sources_is_medium_with_caveat(self):
        body = self.ask(self.client(FakeTable([SOURDOUGH[2]] + NOISE))).json()
        self.assertEqual(body["confidence"], "medium")
        self.assertIn("limited to 1 source", body["answer"])

    def test_limit_caps_sources(self):
        body = self.ask(self.client(FakeTable(SOURDOUGH + NOISE)), limit=2).json()
        self.assertEqual(len(body["sources"]), 2)
        self.assertTrue(all(s["relevant"] for s in body["sources"]))

    def test_platform_filter(self):
        table = FakeTable(SOURDOUGH + NOISE)
        body = self.ask(self.client(table), platforms=["Claude", "claude"]).json()
        q = table.queries[0]
        self.assertEqual(q.filter, "platform IN ('claude')")
        self.assertTrue(q.prefilter)
        self.assertEqual(q.n, server.ASK_CANDIDATES)
        self.assertEqual({s["platform"] for s in body["sources"]}, {"claude"})
        self.assertEqual(body["filters"]["platforms"], ["claude"])
        self.assertIn("on Claude", body["answer"])

    def test_date_filter(self):
        table = FakeTable(SOURDOUGH + NOISE)
        body = self.ask(self.client(table), date_from="2024-03-01", date_to="2024-06-20").json()
        self.assertEqual(table.queries[0].filter,
                         "(start_time >= '2024-03-01T00:00:00+00:00' OR end_time >= '2024-03-01T00:00:00+00:00') "
                         "AND start_time < '2024-06-21T00:00:00+00:00'")
        self.assertEqual({s["node_id"] for s in body["sources"]}, {"T_2", "T_3"})
        self.assertEqual(body["filters"]["date_to"], "2024-06-21T00:00:00+00:00")

    def test_date_in_question_sets_window_and_is_not_embedded(self):
        c = self.client(FakeTable(SOURDOUGH))
        body = self.ask(c, question="What was I baking around June 2024?").json()
        self.assertEqual(body["filters"]["date_from"], "2024-05-01T00:00:00+00:00")
        self.assertEqual(body["filters"]["date_to"], "2024-08-01T00:00:00+00:00")
        self.assertEqual(self.loader.model.queries, ["What was I baking?"])
        self.assertTrue(any("around June 2024" in n for n in body["notes"]))
        self.assertEqual({s["node_id"] for s in body["sources"]}, {"T_3", "T_4"})

    def test_explicit_dates_override_question_dates(self):
        c = self.client(FakeTable(SOURDOUGH))
        body = self.ask(c, question="sourdough baking in June 2024", date_from="2024-01-01", date_to="2024-01-31").json()
        self.assertEqual({s["node_id"] for s in body["sources"]}, {"T_1"})
        self.assertEqual(self.loader.model.queries, ["sourdough baking in June 2024"])

    def test_model_reused_across_search_and_ask(self):
        c = self.client(FakeTable(SOURDOUGH))
        c.post("/api/search", json={"query": "bread"})
        for q in ("sourdough?", "baking?", "starter?"):
            self.assertEqual(self.ask(c, question=q).status_code, 200)
        self.assertEqual(self.loader.calls, [MODEL])
        self.assertEqual(len(self.loader.model.queries), 4)

    def test_index_unavailable_then_ready(self):
        c = self.client()
        r = self.ask(c)
        self.assertEqual(r.status_code, 503)
        self.assertEqual(r.json()["error"]["code"], "index_unavailable")
        self.assertEqual(self.loader.calls, [])
        from unittest import mock
        with mock.patch.object(search, "open_table", return_value=FakeTable(SOURDOUGH)), \
                mock.patch.object(search, "table_model_metadata", return_value={"model": MODEL, "vector_dim": 3}):
            self.assertEqual(self.ask(c).status_code, 200)

    def test_model_failure_is_500(self):
        c = self.client(FakeTable(SOURDOUGH), loader=Loader(error="boom"))
        r = self.ask(c)
        self.assertEqual((r.status_code, r.json()["error"]["code"]), (500, "model_unavailable"))

    def test_invalid_requests(self):
        c = self.client(FakeTable(SOURDOUGH))
        bad = [{"question": "  "}, {"question": ""}, {"limit": 0}, {"limit": server.ASK_MAX_LIMIT + 1},
               {"question": "x" * (server.MAX_QUERY_CHARS + 1)}, {"platforms": ["red dit'"]},
               {"platforms": ["reddit') OR ('1'='1"]}, {"date_from": "yesterday"}, {"date_to": "2024-13-01"},
               {"date_from": "2024-05-01", "date_to": "2024-04-01"}]
        for extra in bad:
            payload = {"question": "sourdough", **extra}
            with self.subTest(payload=payload):
                r = c.post("/api/ask", json=payload)
                self.assertEqual(r.status_code, 422, r.text)
                self.assertEqual(r.json()["error"]["code"], "invalid_request")
        self.assertEqual(c.post("/api/ask", json={}).status_code, 422)
        self.assertEqual(self.loader.calls, [], "invalid requests must not load the model")

    def test_cors_preflight_for_ask(self):
        r = self.client().options("/api/ask", headers={"Origin": "http://127.0.0.1:8080", "Access-Control-Request-Method": "POST",
                                                       "Access-Control-Request-Headers": "Content-Type"})
        self.assertEqual(r.headers.get("access-control-allow-origin"), "http://127.0.0.1:8080")
        r = self.client().options("/api/ask", headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"})
        self.assertIsNone(r.headers.get("access-control-allow-origin"))


class BriefUnitTests(unittest.TestCase):
    def test_filter_expression(self):
        self.assertIsNone(search.filter_expression())
        self.assertEqual(search.filter_expression(["reddit", "claude"]), "platform IN ('claude', 'reddit')")
        for bad in (["x'y"], ["A B"], ["a" * 40]):
            with self.assertRaises(search.SearchError):
                search.filter_expression(bad)
        with self.assertRaises(search.SearchError):
            search.filter_expression(date_from="2024-01-01' OR 1=1 --")

    def test_implied_window(self):
        a, b, phrase = mb.implied_window("What was I working on around August 2026?")
        self.assertEqual((a.isoformat(), b.isoformat()), ("2026-07-01T00:00:00+00:00", "2026-10-01T00:00:00+00:00"))
        self.assertEqual(phrase, "around August 2026")
        a, b, _ = mb.implied_window("photos in dec 2025")
        self.assertEqual((a.month, b.year, b.month), (12, 2026, 1))
        a, b, _ = mb.implied_window("exams in 2025")
        self.assertEqual((a.year, b.year), (2025, 2026))
        self.assertIsNone(mb.implied_window("What was I stressed about during college?"))

    def test_content_terms_drop_scaffolding(self):
        self.assertEqual(mb.content_terms("What was I stressed about during college?"), ["stressed", "college"])
        self.assertEqual(mb.content_terms("How has my interest in photography changed?"), ["photography"])
        self.assertEqual(mb.content_terms("What was I working on around August 2026?"), [])

    def test_term_matching_uses_prefixes(self):
        words = mb.words_of("I was stressful about my photographs")
        self.assertTrue(mb.mentions(words, mb.term_key("stressed")))
        self.assertTrue(mb.mentions(words, mb.term_key("photography")))
        self.assertFalse(mb.mentions(words, mb.term_key("college")))

    def test_parse_messages(self):
        text = ("Platform: REDDIT\nTitle: t\nParticipants: a, b\nTimeframe: x\n"
                "[POST] u/alice — 2024-01-02 03:04:05\nfirst line\nsecond line\n\n"
                "[2024-01-03 00:00:00] Bob: hi there: with colon")
        msgs = mb.parse_messages(text)
        self.assertEqual([(m["author"], m["content"]) for m in msgs], [("u/alice", "first line\nsecond line"), ("Bob", "hi there: with colon")])
        self.assertEqual(msgs[0]["ts"], dt.datetime(2024, 1, 2, 3, 4, 5, tzinfo=UTC))

    def test_best_quote_prefers_terms_and_skips_urls(self):
        msgs = mb.parse_messages("[2024-01-01 00:00:00] Me: Nice weather today for a long walk outside. "
                                 "I finally bought a film camera for street photography! https://x.example/y")
        q = mb.best_quote(msgs, [mb.term_key("photography")])
        self.assertEqual(q["text"], "I finally bought a film camera for street photography!")
        self.assertEqual(q["author"], "Me")

    def test_brief_is_deterministic(self):
        results = [dict(r, similarity=round(1 - r["_distance"], 4), node_id=f"T_{r['parent_id']}") for r in SOURDOUGH + NOISE]
        a = mb.build_brief("sourdough baking", results)
        b = mb.build_brief("sourdough baking", list(results))
        self.assertEqual(a, b)


@unittest.skipUnless(HAVE_FASTAPI, "fastapi not installed")
class AskPageTests(unittest.TestCase):
    def test_graph_html_wires_ask(self):
        from pathlib import Path
        html = Path(REPO_ROOT, "sarthink_graph.html").read_text(encoding="utf-8")
        for needle in ("'/api/ask'", 'id="omni-q"', 'id="omni-go"', "askSarthink", 'data-omode="ask"', 'data-omode="search"',
                       "ask-chip", "ask-conf", "ask-timeline"):
            self.assertIn(needle, html)
        # Archive text only ever reaches the DOM through esc()/highlight().
        block = html[html.index("function renderAsk"):html.index("async function askSarthink")]
        for field in ("data.answer", "data.question", "pt", "n", "t.label", "t.platform", "r.title", "r.snippet", "r.text"):
            self.assertNotRegex(block, r"\$\{" + re.escape(field) + r"[}.\s]", f"unescaped interpolation of {field} in renderAsk")


# Synthetic Discord chunks as chunk_builder writes them for a data package: only the owner's lines.
DISCORD = [
    row(501, "discord", "2025-04-02", [("Me", "Booked the synthetic zebra-kite workshop for Saturday morning."),
                                       ("Me", "Bringing the spare zebra-kite lines too.")], 0.25, "DM Test Friend"),
    row(502, "discord", "2025-04-03", [("Me", "zebra-kite photos from the workshop are in the drive")], 0.30, "#general (Test Server)"),
]


@unittest.skipUnless(HAVE_FASTAPI, "fastapi not installed (.venv/bin/pip install fastapi uvicorn)")
class DiscordSourceTests(ApiTestCase):
    """A Discord chunk reaches Search and Ask as a source that links to its graph conversation node."""

    def test_search_returns_clickable_discord_result(self):
        c = self.client(FakeTable(DISCORD + NOISE))
        body = c.post("/api/search", json={"query": "zebra-kite workshop", "limit": 5}).json()
        hit = next(r for r in body["results"] if r["platform"] == "discord")
        self.assertEqual((hit["node_id"], hit["title"]), ("T_501", "DM Test Friend"))
        self.assertEqual(hit["people"], ["Me"])
        self.assertEqual((hit["start_time"][:10], hit["end_time"][:10]), ("2025-04-02", "2025-04-02"))

    def test_ask_cites_discord_source_with_platform_filter(self):
        table = FakeTable(DISCORD + NOISE)
        c = self.client(table)
        r = c.post("/api/ask", json={"question": "What did I say about the zebra-kite workshop?", "platforms": ["discord"]})
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertIn("platform IN ('discord')", table.queries[0].filter)
        relevant = [s for s in body["sources"] if s["relevant"]]
        self.assertTrue(relevant)
        self.assertTrue(all(s["platform"] == "discord" and re.fullmatch(r"T_50[12]", s["node_id"]) for s in relevant))
        self.assertIn("zebra-kite", " ".join(s["text"] for s in relevant))


if __name__ == "__main__":
    unittest.main()
