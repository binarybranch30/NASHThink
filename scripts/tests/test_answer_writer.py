"""Tests for Ask Sarthink's written answers: POST /api/ask/stream, GET /api/llm and scripts/api/answer_writer.py.
A fake llama.cpp server (httpx.MockTransport) stands in for the local Llama, and the fake model/table from
test_api stand in for the index, so nothing is loaded, embedded or sent anywhere. Needs fastapi + httpx (.venv).
Run: .venv/bin/python scripts/tests/test_answer_writer.py"""
import asyncio
import json
import os
import sys
import unittest
from unittest import mock

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.append(SCRIPT_DIR)
sys.path.append(os.path.join(REPO_ROOT, "scripts", "semantic"))
sys.path.append(os.path.join(REPO_ROOT, "scripts", "api"))

import httpx  # noqa: E402

import answer_writer as aw  # noqa: E402
import llm_config  # noqa: E402
from test_api import HAVE_FASTAPI, ApiTestCase, FakeTable  # noqa: E402
from test_ask import NOISE, SOURDOUGH  # noqa: E402

if HAVE_FASTAPI:
    from fastapi.testclient import TestClient  # noqa: E402

    import insights  # noqa: E402
    import server  # noqa: E402

OWNER = ("Tester Persona", {"tester", "Tester Persona", "tester persona", "12345", "me@example.com"})
PROFILES = llm_config.load_profiles(path="/nonexistent")
# Never read a real DeepSeek key in tests: tests that need one set it explicitly.
os.environ.pop("DEEPSEEK_API_KEY", None)
llm_config.SECRETS_PATH = "/nonexistent/.env"
FAKE_KEY = "sk-test-not-a-real-key"


def sse_body(pieces, model="fake-llama", timings=None):
    chunks = [{"model": model, "choices": [{"delta": {"content": p}}]} for p in pieces]
    chunks.append({"model": model, "choices": [{"delta": {}, "finish_reason": "stop"}],
                   "timings": timings or {"prompt_n": 100, "predicted_n": len(pieces)}})
    return "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"


class FakeLlama:
    """Records chat requests; answers health probes and streams `pieces` (or fails with `status`)."""

    def __init__(self, pieces=("You baked ", "sourdough [1], ", "then a loaf [3][9]."), status=200, offline=False,
                 chars_per_token=3):
        self.pieces, self.status, self.offline = pieces, status, offline
        self.chars_per_token = chars_per_token   # None: no /tokenize endpoint (older server)
        self.requests = []
        self.tokenized = []

    def __call__(self, request):
        if self.offline:
            raise httpx.ConnectError("connection refused", request=request)
        if request.url.path == "/health":
            return httpx.Response(200 if self.status == 200 else self.status, json={"status": "ok"})
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": f"fake-{request.url.port}"}]})
        if request.url.path == "/tokenize":
            if self.chars_per_token is None:
                return httpx.Response(404, json={"error": "not found"})
            content = json.loads(request.content)["content"]
            self.tokenized.append(len(content))
            return httpx.Response(200, json={"tokens": list(range(int(len(content) / self.chars_per_token)))})
        self.requests.append(json.loads(request.content))
        if self.status != 200:
            return httpx.Response(self.status, json={"error": {"message": "Loading model"}})
        return httpx.Response(200, content=sse_body(self.pieces).encode(), headers={"Content-Type": "text/event-stream"})


def parse_sse(text):
    events = []
    for block in text.strip().split("\n\n"):
        lines = dict(l.split(": ", 1) for l in block.splitlines() if ": " in l)
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def src(rank, text, relevant=True, terms=("sourdough",)):
    return {"rank": rank, "title": f"Thread {rank}", "platform": "reddit", "date_start": "2024-01-10T09:00:00+00:00",
            "date_end": "2024-01-10T10:00:00+00:00", "people": ["tester", "other"], "text": text, "snippet": text[:50],
            "relevant": relevant, "matched_terms": list(terms)}


class PromptTests(unittest.TestCase):
    def test_only_evidence_sources_keep_their_ranks(self):
        brief = {"confidence": "medium", "evidence": {"terms": ["sourdough"]},
                 "sources": [src(1, "[2024-01-10 09:00:00] tester: my sourdough starter"),
                             src(2, "[2024-01-11 09:00:00] tester: sourdough loaf"),
                             src(3, "[2024-02-01 09:00:00] x: unrelated hiking", relevant=False)]}
        messages, allowed = aw.build_messages("What about sourdough?", brief, PROFILES["quick"], OWNER)
        self.assertEqual(allowed, [1, 2])
        user = messages[1]["content"]
        self.assertIn("[1] Thread 1 · Reddit · 2024-01-10 · people: tester, other", user)
        self.assertIn("[2] Thread 2", user)
        self.assertNotIn("hiking", user)
        self.assertTrue(user.rstrip().endswith("like [2]."))
        self.assertIn("Question: What about sourdough?", user)
        system = messages[0]["content"]
        self.assertIn('"Tester Persona", "tester"', system)
        self.assertNotIn("12345", system)          # numeric ids and emails are not names
        self.assertNotIn("me@example.com", system)

    def test_budget_trims_long_sources_around_mentions(self):
        filler = "\n".join(f"[2024-01-10 09:{k:02d}:00] other: filler line number {k} about nothing much" for k in range(60))
        text = filler + "\n[2024-01-10 10:00:00] tester: the sourdough finally rose\n" + filler
        brief = {"confidence": "high", "evidence": {"terms": ["sourdough"]}, "sources": [src(k, text) for k in range(1, 9)]}
        profile = PROFILES["quick"]
        messages, allowed = aw.build_messages("sourdough?", brief, profile, OWNER)
        self.assertEqual(allowed, [1, 2, 3, 4, 5])   # max_sources
        user = messages[1]["content"]
        self.assertEqual(user.count("the sourdough finally rose"), 5)
        self.assertIn(aw.GAP, user)
        for block in user.split("\n\n")[1:6]:
            self.assertLessEqual(len(block.split("\n", 1)[1]), profile["source_chars"] + 5)
        est_tokens = sum(len(m["content"]) for m in messages) / aw.CHARS_PER_TOKEN_EN
        self.assertLess(est_tokens, profile["prompt_tokens"] * 1.2)
        self.assertLess(est_tokens + profile["max_tokens"], profile["ctx"])

    def test_english_answers_and_quotes_are_required(self):
        brief = {"confidence": "high", "evidence": {"terms": ["sourdough"]}, "sources": [src(1, "tester: sourdough starter")]}
        system, user = (m["content"] for m in aw.build_messages("mujhe sourdough ke baare mein batao", brief, PROFILES["best"], OWNER)[0])
        self.assertIn("Always write in English", system)
        self.assertIn("exact quote", system)
        self.assertIn("Answer in English", user)
        self.assertNotIn("language of the question", system)

    def test_glossary_only_for_hinglish_sources(self):
        en = {"confidence": "high", "evidence": {"terms": []}, "sources": [src(1, "tester: my sourdough starter smells great")]}
        hi = {"confidence": "high", "evidence": {"terms": []}, "sources": [src(1, "tester: yaar neend nahi aa rha h\nother: kyu bhai kya hua")]}
        self.assertNotIn("shorthand", aw.build_messages("q", en, PROFILES["quick"], OWNER)[0][0]["content"])
        system = aw.build_messages("q", hi, PROFILES["quick"], OWNER)[0][0]["content"]
        self.assertIn("ni / nhi / nahi / na = not", system)
        self.assertIn("never drop it", system)

    def test_heavy_content_gets_strict_quoting(self):
        heavy = {"confidence": "high", "evidence": {"terms": []},
                 "sources": [src(1, "tester: synthetic example line, sab bekaar lag raha, mar jaana chahta hu")]}
        joke = {"confidence": "high", "evidence": {"terms": []}, "sources": [src(1, "tester: mar gaya yaar hasi se, too funny")]}
        self.assertIn("Never soften it", aw.build_messages("q", heavy, PROFILES["best"], OWNER)[0][0]["content"])
        self.assertNotIn("Never soften it", aw.build_messages("q", joke, PROFILES["best"], OWNER)[0][0]["content"])

    def test_hinglish_estimate_is_tighter(self):
        hi_text = "\n".join(f"tester: yaar kal raat bhi neend nahi aayi {k} bas phone chalata raha h" for k in range(80))
        en_text = "\n".join(f"tester: I slept badly again last night {k} and kept scrolling my phone" for k in range(80))
        mk = lambda t: {"confidence": "high", "evidence": {"terms": []}, "sources": [src(k, t) for k in range(1, 6)]}
        self.assertLess(aw.default_source_chars("q", mk(hi_text), PROFILES["quick"]),
                        aw.default_source_chars("q", mk(en_text), PROFILES["quick"]))

    def test_single_huge_line_is_cut(self):
        out = aw.excerpt("x" * 5000, ["sourdough"], 300)
        self.assertEqual(len(out), 300)
        self.assertTrue(out.endswith(aw.GAP))

    def test_weak_evidence_is_not_written(self):
        self.assertFalse(aw.should_write({"confidence": "low", "sources": [src(1, "a")]}))
        self.assertFalse(aw.should_write({"confidence": "medium", "sources": [src(1, "a", relevant=False)]}))
        self.assertTrue(aw.should_write({"confidence": "medium", "sources": [src(1, "a")]}))


class FitTests(unittest.TestCase):
    def brief(self):
        text = "\n".join(f"[2024-01-10 09:{k:02d}:00] tester: yaar sourdough ka starter phir se fail ho gaya {k} 😭" for k in range(60))
        return {"confidence": "high", "evidence": {"terms": ["sourdough"]}, "sources": [src(k, text) for k in range(1, 9)]}

    def test_prompt_is_shrunk_until_the_tokenizer_says_it_fits(self):
        llama = FakeLlama(chars_per_token=1.6)   # harsher than real Hinglish (~2): the first guess is too big
        profile = PROFILES["best"]
        messages, allowed, n, counted = asyncio.run(aw.fit_messages("sourdough?", self.brief(), profile, OWNER,
                                                                    httpx.MockTransport(llama)))
        self.assertTrue(counted)
        self.assertLessEqual(n, profile["prompt_tokens"])
        self.assertGreater(len(llama.tokenized), 1, "it re-measured after shrinking")
        self.assertTrue(allowed and allowed == list(range(1, len(allowed) + 1)), "only the lowest-ranked sources are dropped")
        self.assertIn("sourdough", messages[1]["content"])

    def test_estimate_when_the_server_cannot_count(self):
        messages, _, n, counted = asyncio.run(aw.fit_messages("sourdough?", self.brief(), PROFILES["quick"], OWNER,
                                                              httpx.MockTransport(FakeLlama(chars_per_token=None))))
        self.assertFalse(counted)
        self.assertLess(n, PROFILES["quick"]["prompt_tokens"] * 1.25)


class QuoteTests(unittest.TestCase):
    SOURCES = [{"rank": 1, "text": "[2025-01-01 09:00:00] tester: abe neend nahi aa rha na"},
               {"rank": 2, "text": "[2025-01-02 09:00:00] other: kal exam hai, padh le"}]

    def test_quotes_are_checked_against_the_cited_source(self):
        text = ('You couldn\'t sleep ("neend nhi aa rha") [1]. Your friend said "kal exam hai" [1]. '
                'Also “kal exam hai” [2], “totally made up words” [2] and “ok” [1]. Not cited: "padh le".')
        got = {(q["text"], q["source"]): q["ok"] for q in aw.verify_quotes(text, self.SOURCES)}
        self.assertTrue(got[("neend nhi aa rha", 1)], "spelling variants of the same words count as exact")
        self.assertFalse(got[("kal exam hai", 1)], "a real quote cited to the wrong source is flagged")
        self.assertTrue(got[("kal exam hai", 2)])
        self.assertFalse(got[("totally made up words", 2)])
        self.assertTrue(got[("padh le", None)], "uncited quotes may come from any evidence source")
        self.assertNotIn(("ok", 1), got, "one-word quotes are not checked")

    def test_quotes_with_ellipsis(self):
        got = aw.verify_quotes('"abe neend … aa rha na" [1]', self.SOURCES)
        self.assertEqual([q["ok"] for q in got], [True])


class CitationTests(unittest.TestCase):
    def test_sanitize(self):
        self.assertEqual(aw.sanitize_citations("You asked, [1]. Also [2, 9] and [3; 1] ok [7].", [1, 2, 3]),
                         "You asked [1]. Also [2] and [3][1] ok.")
        self.assertEqual(aw.cited("a [2] b [1][2] c [3, 1]"), [1, 2, 3])
        self.assertEqual(aw.sanitize_citations("No sources here.", []), "No sources here.")


class NoInfoTests(unittest.TestCase):
    def test_sentinel_is_recognised(self):
        for reply in ("NO_RELEVANT_INFO", "  **NO_RELEVANT_INFO**", "No relevant info", "no-relevant-info."):
            self.assertTrue(aw.says_no_info(reply), reply)
        self.assertFalse(aw.says_no_info("No, you never said that [1]."))
        self.assertTrue(aw.may_become_no_info("  NO_REL"))
        self.assertFalse(aw.may_become_no_info("You baked"))
        self.assertFalse(aw.may_become_no_info("No, you"))

    def test_ungrounded_answers(self):
        ok = [{"text": "a b", "ok": True}]
        self.assertIsNone(aw.ungrounded_reason("You baked sourdough [1].", ok))
        self.assertEqual(aw.ungrounded_reason("You baked sourdough.", ok), "no_citations")
        bad = [{"text": "a b", "ok": False}, {"text": "c d", "ok": False}, {"text": "e f", "ok": True}]
        self.assertEqual(aw.ungrounded_reason("You said \"a b\" [1].", bad), "unverified_quotes")
        self.assertEqual(aw.ungrounded_reason("You baked [1]. NO_RELEVANT_INFO", ok), "model_unsure")


class StreamingTests(unittest.TestCase):
    def run_async(self, coro):
        return asyncio.run(coro)

    def test_closing_the_stream_closes_upstream(self):
        state = {"closed": False, "sent": 0}

        async def body():
            try:
                for k in range(1000):
                    state["sent"] += 1
                    yield f"data: {json.dumps({'choices': [{'delta': {'content': f'w{k} '}}]})}\n\n".encode()
                    await asyncio.sleep(0)
            finally:
                state["closed"] = True

        transport = httpx.MockTransport(lambda req: httpx.Response(200, content=body()))

        async def main():
            gen = aw.stream_completion(PROFILES["quick"], [], transport)
            first = await gen.__anext__()
            await gen.aclose()
            return first

        self.assertEqual(self.run_async(main()), ("token", "w0 "))
        self.assertTrue(state["closed"])
        self.assertLess(state["sent"], 1000)

    def test_heartbeats_until_first_token(self):
        async def body():
            await asyncio.sleep(0.25)
            yield sse_body(["Hi [1]."]).encode()

        transport = httpx.MockTransport(lambda req: httpx.Response(200, content=body()))
        brief = {"confidence": "high", "evidence": {"terms": []}, "sources": [src(1, "sourdough")]}

        async def main():
            async def connected():
                return False
            return [e async for e in aw.answer_events("q", brief, PROFILES["quick"], OWNER, connected,
                                                        asyncio.Event(), transport, heartbeat_s=0.05)]

        events = parse_sse("".join(self.run_async(main())))
        kinds = [k for k, _ in events]
        self.assertGreaterEqual(kinds.count("status"), 3)
        self.assertEqual(kinds[-2:], ["token", "done"])
        self.assertEqual(events[-1][1]["text"], "Hi [1].")

    def test_superseded_answer_stops(self):
        async def body():
            await asyncio.sleep(5)
            yield b""

        transport = httpx.MockTransport(lambda req: httpx.Response(200, content=body()))
        brief = {"confidence": "high", "evidence": {"terms": []}, "sources": [src(1, "sourdough")]}

        async def main():
            cancelled = asyncio.Event()

            async def connected():
                cancelled.set()   # a newer question arrives while this one is still reading
                return False
            return [e async for e in aw.answer_events("q", brief, PROFILES["quick"], OWNER, connected,
                                                        cancelled, transport, heartbeat_s=0.05)]

        events = parse_sse("".join(self.run_async(main())))
        self.assertEqual(events[-1], ("error", {"code": "llm_superseded", "message": "A newer question replaced this one."}))


@unittest.skipUnless(HAVE_FASTAPI, "fastapi not installed (.venv/bin/pip install fastapi uvicorn)")
class StreamApiTests(ApiTestCase):
    def stream_client(self, llama, table=None):
        self.client(FakeTable(SOURDOUGH + NOISE) if table is None else table)   # patches the index like /api/ask tests
        service = server.SearchService(self.tmp_path / "no_such_lancedb", "topics", model_loader=self.loader)
        ins = insights.InsightsService(self.tmp_path / "no.db", identity_map=OWNER)
        app = server.create_app(service, graph_html=self.html, graph_dir=self.graph_dir, insights_service=ins,
                                llm_profiles=PROFILES, llm_transport=httpx.MockTransport(llama))
        return TestClient(app)

    def ask(self, client, **payload):
        payload.setdefault("question", "What did I say about sourdough baking?")
        r = client.post("/api/ask/stream", json=payload)
        return r, (parse_sse(r.text) if r.headers.get("content-type", "").startswith("text/event-stream") else None)

    def test_event_order_and_grounded_citations(self):
        llama = FakeLlama()
        r, events = self.ask(self.stream_client(llama), profile="quick")
        self.assertEqual(r.status_code, 200, r.text)
        kinds = [k for k, _ in events]
        self.assertEqual(kinds[0], "brief")
        self.assertEqual(kinds[1], "status")
        self.assertEqual(kinds[-1], "done")
        self.assertEqual(kinds.count("token"), 3)
        brief = events[0][1]
        self.assertEqual(brief["confidence"], "high")
        self.assertIn("sources", brief)
        done = events[-1][1]
        self.assertEqual(done["text"], "You baked sourdough [1], then a loaf [3].")   # [9] isn't a source
        self.assertEqual(done["cited"], [1, 3])
        self.assertEqual((done["grounded"], done["ungrounded_reason"]), (True, None))
        self.assertEqual(done["profile"], "quick")
        self.assertEqual((done["quotes"], done["unverified"]), ([], 0))
        self.assertTrue(done["prompt_tokens"])
        self.assertEqual(done["model"], "fake-llama")
        # the model saw the question, the evidence and the owner, and nothing marked as a mere lead
        req = llama.requests[0]
        self.assertTrue(req["stream"])
        self.assertEqual(req["max_tokens"], PROFILES["quick"]["max_tokens"])
        user = req["messages"][1]["content"]
        self.assertIn("sourdough starter", user)
        self.assertNotIn("hiking route", user)
        self.assertIn("Tester Persona", req["messages"][0]["content"])

    def test_same_brief_as_api_ask(self):
        c = self.stream_client(FakeLlama())
        _, events = self.ask(c, profile="best")
        plain = c.post("/api/ask", json={"question": "What did I say about sourdough baking?"}).json()
        streamed = events[0][1]
        for key in ("answer", "confidence", "summary_points", "sources", "timeline", "evidence"):
            self.assertEqual(streamed[key], plain[key], key)

    def test_offline_model_keeps_the_brief(self):
        r, events = self.ask(self.stream_client(FakeLlama(offline=True)), profile="best")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(events[0][0], "brief")
        kind, err = events[-1]
        self.assertEqual((kind, err["code"]), ("error", "llm_offline"))
        self.assertIn("scripts/llm.sh start best", err["message"])

    def test_loading_model(self):
        _, events = self.ask(self.stream_client(FakeLlama(status=503)), profile="quick")
        self.assertEqual(events[-1][1]["code"], "llm_loading")

    def test_weak_evidence_skips_the_model(self):
        llama = FakeLlama()
        _, events = self.ask(self.stream_client(llama), question="What did I say about quantum chromodynamics?")
        self.assertEqual(events[0][1]["confidence"], "low")
        self.assertEqual(events[-1], ("skipped", {"code": "no_relevant_info", "message": aw.NO_INFO_MESSAGE}))
        self.assertEqual(llama.requests, [])

    def test_model_says_no_relevant_info(self):
        for pieces in (("NO_", "RELEVANT", "_INFO"), ("  **No relevant", " info**",), ("NO_REL",)):
            _, events = self.ask(self.stream_client(FakeLlama(pieces=pieces)), profile="quick")
            kinds = [k for k, _ in events]
            self.assertNotIn("token", kinds, "the sentinel never reaches the page")
            self.assertEqual(events[-1], ("skipped", {"code": "no_relevant_info", "message": aw.NO_INFO_MESSAGE}))

    def test_answer_starting_like_the_sentinel_still_streams(self):
        _, events = self.ask(self.stream_client(FakeLlama(pieces=("No", ", you ", "baked sourdough [1]."))), profile="quick")
        self.assertEqual("".join(d["text"] for k, d in events if k == "token"), "No, you baked sourdough [1].")
        self.assertTrue(events[-1][1]["grounded"])

    def test_uncited_answer_is_not_grounded(self):
        _, events = self.ask(self.stream_client(FakeLlama(pieces=("You probably ", "like bread."))), profile="quick")
        done = events[-1][1]
        self.assertEqual((events[-1][0], done["grounded"], done["ungrounded_reason"]), ("done", False, "no_citations"))

    def test_html_in_tokens_is_passed_as_text(self):
        _, events = self.ask(self.stream_client(FakeLlama(pieces=("<img src=x onerror=alert(1)> [1]",))))
        self.assertEqual(events[-1][1]["text"], "<img src=x onerror=alert(1)> [1]")   # the UI escapes it

    def test_retrieval_errors_are_plain_json(self):
        r, events = self.ask(self._no_index_client())
        self.assertEqual(r.status_code, 503)
        self.assertIsNone(events)
        self.assertEqual(r.json()["error"]["code"], "index_unavailable")

    def _no_index_client(self):
        service = server.SearchService(self.tmp_path / "no_such_lancedb", "topics")
        app = server.create_app(service, graph_html=self.html, graph_dir=self.graph_dir,
                                insights_service=insights.InsightsService(self.tmp_path / "no.db", identity_map=OWNER),
                                llm_profiles=PROFILES, llm_transport=httpx.MockTransport(FakeLlama()))
        return TestClient(app)

    def test_invalid_profile(self):
        r, _ = self.ask(self.stream_client(FakeLlama()), profile="huge")
        self.assertEqual(r.status_code, 422)
        self.assertEqual(r.json()["error"]["code"], "invalid_request")
        self.assertIn("profile", r.json()["error"]["message"])

    def test_llm_status(self):
        body = self.stream_client(FakeLlama()).get("/api/llm").json()
        self.assertEqual(body["order"], ["deepseek", "best", "quick", "deepseek_reasoner"])
        self.assertEqual((body["profiles"]["deepseek"]["state"], body["profiles"]["deepseek"]["reason"]), ("offline", "no_key"))
        self.assertEqual(body["profiles"]["best"]["state"], "online")
        self.assertEqual(body["profiles"]["best"]["model"], "fake-8083")
        self.assertEqual(body["profiles"]["quick"]["start"], "scripts/llm.sh start quick")
        off = self.stream_client(FakeLlama(offline=True)).get("/api/health").json()["llm"]
        self.assertEqual({p["state"] for p in off.values()}, {"offline"})

    # ── DeepSeek (hosted, OpenAI-compatible): a fake API on the same mock transport, dispatched by host ──
    def deepseek_client(self, deep, key=FAKE_KEY):
        llama = FakeLlama()
        both = lambda req: deep(req) if req.url.host == "api.deepseek.com" else llama(req)   # noqa: E731
        self.client(FakeTable(SOURDOUGH + NOISE))
        service = server.SearchService(self.tmp_path / "no_such_lancedb", "topics", model_loader=self.loader)
        ins = insights.InsightsService(self.tmp_path / "no.db", identity_map=OWNER)
        app = server.create_app(service, graph_html=self.html, graph_dir=self.graph_dir, insights_service=ins,
                                llm_profiles=llm_config.load_profiles(path="/nonexistent"), llm_transport=httpx.MockTransport(both))
        if key:
            self.enterContext(mock.patch.dict(os.environ, {"DEEPSEEK_API_KEY": key}))
        return TestClient(app), llama

    def test_deepseek_streams_with_the_key_and_no_local_calls(self):
        deep = FakeDeepSeek()
        c, llama = self.deepseek_client(deep)
        r = c.post("/api/ask/stream", json={"question": "What did I say about sourdough baking?", "profile": "deepseek"})
        events = parse_sse(r.text)
        self.assertEqual(events[-1][0], "done", events[-1])
        self.assertEqual(events[-1][1]["text"], "You baked sourdough [1].")
        self.assertEqual("".join(d["text"] for k, d in events if k == "token"), "You baked sourdough [1].",
                         "the reasoner's hidden thinking is never streamed as answer text")
        req = deep.chats[0]
        self.assertEqual((req["path"], req["auth"], req["body"]["model"]), ("/chat/completions", f"Bearer {FAKE_KEY}", "deepseek-chat"))
        self.assertTrue(req["body"]["stream"])
        self.assertNotIn("cache_prompt", req["body"])
        self.assertIn("sourdough starter", req["body"]["messages"][1]["content"])
        self.assertEqual((llama.requests, llama.tokenized), ([], []), "no local model or tokenizer involved")
        self.assertNotIn(FAKE_KEY, r.text)

    def test_deepseek_reasoner_uses_its_model(self):
        deep = FakeDeepSeek()
        c, _ = self.deepseek_client(deep)
        c.post("/api/ask/stream", json={"question": "What did I say about sourdough baking?", "profile": "deepseek_reasoner"})
        self.assertEqual(deep.chats[0]["body"]["model"], "deepseek-reasoner")

    def test_deepseek_errors_are_plain_and_keyless(self):
        c, _ = self.deepseek_client(FakeDeepSeek(status=401))
        r = c.post("/api/ask/stream", json={"question": "What did I say about sourdough baking?", "profile": "deepseek"})
        kind, err = parse_sse(r.text)[-1]
        self.assertEqual((kind, err["code"]), ("error", "llm_no_key"))
        self.assertIn("rejected the API key", err["message"])
        self.assertNotIn(FAKE_KEY, r.text)
        c, _ = self.deepseek_client(FakeDeepSeek(status=402))
        err = parse_sse(c.post("/api/ask/stream", json={"question": "What did I say about sourdough baking?", "profile": "deepseek"}).text)[-1][1]
        self.assertIn("no balance", err["message"])

    def test_deepseek_without_a_key_never_calls_out(self):
        deep = FakeDeepSeek()
        c, _ = self.deepseek_client(deep, key=None)
        err = parse_sse(c.post("/api/ask/stream", json={"question": "What did I say about sourdough baking?", "profile": "deepseek"}).text)[-1][1]
        self.assertEqual(err["code"], "llm_no_key")
        self.assertEqual((deep.chats, deep.model_checks), ([], 0))

    def test_deepseek_status_checks_the_key_once(self):
        deep = FakeDeepSeek()
        c, _ = self.deepseek_client(deep)
        p = c.get("/api/llm").json()["profiles"]
        self.assertEqual((p["deepseek"]["state"], p["deepseek_reasoner"]["state"]), ("online", "online"))
        self.assertTrue(p["deepseek"]["remote"])
        self.assertEqual(deep.model_checks, 1, "both DeepSeek profiles share one key check")
        self.assertNotIn(FAKE_KEY, json.dumps(p))
        c, _ = self.deepseek_client(FakeDeepSeek(status=401))
        self.assertEqual(c.get("/api/llm").json()["profiles"]["deepseek"]["reason"], "bad_key")


class FakeDeepSeek:
    """A stand-in for api.deepseek.com: GET /models (key check) and streaming POST /chat/completions, which first
    streams some reasoning_content (as deepseek-reasoner does) and then the answer."""

    def __init__(self, status=200, key=FAKE_KEY):
        self.status, self.key = status, key
        self.chats, self.model_checks = [], 0

    def __call__(self, request):
        auth = request.headers.get("authorization", "")
        if self.status != 200 or auth != f"Bearer {self.key}":
            return httpx.Response(self.status if self.status != 200 else 401, json={"error": {"message": "Authentication Fails"}})
        if request.url.path == "/models":
            self.model_checks += 1
            return httpx.Response(200, json={"data": [{"id": "deepseek-chat"}, {"id": "deepseek-reasoner"}]})
        self.chats.append({"path": request.url.path, "auth": auth, "body": json.loads(request.content)})
        chunks = [{"model": "deepseek-chat", "choices": [{"delta": {"reasoning_content": "Let me think about sourdough."}}]},
                  {"model": "deepseek-chat", "choices": [{"delta": {"content": "You baked "}}]},
                  {"model": "deepseek-chat", "choices": [{"delta": {"content": "sourdough [1]."}}]},
                  {"model": "deepseek-chat", "choices": [], "usage": {"prompt_tokens": 900, "completion_tokens": 8}}]
        body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
        return httpx.Response(200, content=body.encode(), headers={"Content-Type": "text/event-stream"})


class ConfigTests(unittest.TestCase):
    def test_api_key_from_env_or_env_file(self):
        import tempfile
        p = PROFILES["deepseek"]
        with tempfile.NamedTemporaryFile("w", suffix=".env", delete=False) as f:
            f.write('OTHER=1\nexport DEEPSEEK_API_KEY="sk-from-file"\n')
        self.addCleanup(os.unlink, f.name)
        self.assertEqual(llm_config.api_key(p, secrets_path=f.name), "sk-from-file")
        self.assertIsNone(llm_config.api_key(p, secrets_path="/nonexistent/.env"))
        with mock.patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-from-env"}):
            self.assertEqual(llm_config.api_key(p, secrets_path=f.name), "sk-from-env")
        self.assertIsNone(llm_config.api_key(PROFILES["best"]))

    def test_overrides_and_loopback(self):
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"best": {"extra_args": ["-ngl", "99"], "port": 9999}, "nope": {"port": 1}}, f)
        self.addCleanup(os.unlink, f.name)
        p = llm_config.load_profiles(f.name)
        self.assertEqual(set(p), {"quick", "best", "deepseek", "deepseek_reasoner"})
        self.assertEqual(p["deepseek"]["url"], "https://api.deepseek.com")
        self.assertEqual(p["best"]["extra_args"], ["-ngl", "99"])
        self.assertEqual(p["best"]["url"], "http://127.0.0.1:9999")
        self.assertEqual(p["quick"]["url"], "http://127.0.0.1:8082")
        self.assertTrue(p["quick"]["model_path"].startswith(str(llm_config.REPO_ROOT)))


if __name__ == "__main__":
    unittest.main(verbosity=1)
