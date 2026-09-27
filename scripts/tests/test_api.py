"""Tests for the local search API (scripts/api/server.py). Uses a fake model and table, so no model
download, embedding or real LanceDB index is touched. Needs fastapi + httpx (.venv).
Run: .venv/bin/python scripts/tests/test_api.py"""
import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.append(os.path.join(REPO_ROOT, "scripts", "semantic"))
sys.path.append(os.path.join(REPO_ROOT, "scripts", "api"))

import search

try:
    from fastapi.testclient import TestClient
    import server
    HAVE_FASTAPI = True
except ImportError:
    HAVE_FASTAPI = False

MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
META = {"table": "topics", "model": MODEL, "vector_dim": 3, "rows": 2, "created_at": "2026-09-26T18:00:00+00:00"}

TOPIC_ROW = {
    "vector": [0.0, 0.0, 1.0],
    "parent_id": "42",
    "platform": "discord",
    "title": "Camera nerds",
    "start_time": "2023-05-01T10:00:00+00:00",
    "end_time": "2023-05-01T11:00:00+00:00",
    "text": "Platform: DISCORD\nTitle: Camera nerds\nParticipants: Alice, Me\nTimeframe: x to y\n"
            "[2023-05-01 10:00:00] Me: thinking about a Fuji\n[2023-05-01 10:02:00] Alice: get the X-T3",
    "_distance": 0.25,
}
ORPHAN_ROW = {"vector": [1.0, 0.0, 0.0], "platform": "reddit", "title": "No thread", "text": "hello", "_distance": 0.6}


class FakeModel:
    def __init__(self, dim=3):
        self.dim = dim
        self.queries = []

    def encode(self, text, convert_to_numpy=True):
        self.queries.append(text)
        return [0.1] * self.dim


class FakeQuery:
    def __init__(self, rows):
        self.rows, self.distance, self.n, self.filter, self.columns = rows, None, None, None, None

    def where(self, expr, prefilter=None):
        # Applies only the `col = 'value'` form thread_chunks() builds; other expressions
        # (search.filter_expression) are recorded for assertions and left to the caller's post-filter.
        self.filter, self.prefilter = expr, prefilter
        if " AND " not in expr and " IN " not in expr and ">" not in expr and "<" not in expr and " LIKE " not in expr:
            col, value = [p.strip() for p in expr.split("=", 1)]
            self.rows = [r for r in self.rows if str(r.get(col)) == value.strip("'")]
        return self

    def select(self, columns):
        self.columns = columns
        return self

    def distance_type(self, name):
        self.distance = name
        return self

    def limit(self, n):
        self.n = n
        return self

    def to_list(self):
        return self.rows[:self.n]


class FakeTable:
    def __init__(self, rows, dim=3):
        self.rows = rows
        self.queries = []
        list_type = SimpleNamespace(list_size=dim)
        self.schema = SimpleNamespace(field=lambda name: SimpleNamespace(type=list_type),
                                      names=["vector", "parent_id", "platform", "title", "start_time", "end_time", "text"])

    def search(self, vector=None):
        self.queries.append(FakeQuery(self.rows))
        return self.queries[-1]


class Loader:
    """Model loader stand-in that records calls instead of loading sentence-transformers."""

    def __init__(self, model=None, error=None):
        self.model = model or FakeModel()
        self.error = error
        self.calls = []

    def __call__(self, name):
        self.calls.append(name)
        if self.error:
            raise search.SearchError(self.error)
        return self.model


@unittest.skipUnless(HAVE_FASTAPI, "fastapi not installed (.venv/bin/pip install fastapi uvicorn)")
class ApiTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.tmp_path = Path(self.tmp.name)
        self.graph_dir = self.tmp_path / "graph"
        self.graph_dir.mkdir()
        self.html = self.tmp_path / "graph.html"
        self.html.write_text("<html>sarthink graph</html>", encoding="utf-8")

    def client(self, table=None, loader=None, db_path=None):
        """App whose SearchService sees `table` (with valid metadata); table=None means a real missing DB."""
        self.loader = loader or Loader()
        service = server.SearchService(db_path or self.tmp_path / "no_such_lancedb", "topics", model_loader=self.loader)
        if table is not None:
            self.enterContext(mock.patch.object(search, "open_table", return_value=table))
            self.enterContext(mock.patch.object(search, "table_model_metadata", return_value=dict(META)))
        app = server.create_app(service, graph_html=self.html, graph_dir=self.graph_dir)
        return TestClient(app)


class HealthTests(ApiTestCase):
    def test_missing_index_reported_unavailable_without_loading_model(self):
        r = self.client().get("/api/health")
        self.assertEqual(r.status_code, 200)
        ix = r.json()["index"]
        self.assertFalse(ix["available"])
        self.assertEqual(ix["table"], "topics")
        self.assertIn("not found", ix["reason"])
        self.assertEqual(self.loader.calls, [])

    def test_available_index_reports_model_and_rows(self):
        c = self.client(FakeTable([TOPIC_ROW]))
        ix = c.get("/api/health").json()["index"]
        self.assertTrue(ix["available"])
        self.assertEqual((ix["model"], ix["vector_dim"], ix["rows"]), (MODEL, 3, 2))
        self.assertFalse(ix["model_loaded"])
        self.assertEqual(self.loader.calls, [], "health must never load the embedding model")

        c.post("/api/search", json={"query": "camera"})
        self.assertTrue(c.get("/api/health").json()["index"]["model_loaded"])


class SearchTests(ApiTestCase):
    def test_structured_results(self):
        table = FakeTable([TOPIC_ROW, ORPHAN_ROW])
        r = self.client(table).post("/api/search", json={"query": "  which camera to buy  ", "limit": 2})
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual(body["query"], "which camera to buy")
        self.assertEqual((body["table"], body["model"], body["count"]), ("topics", MODEL, 2))
        self.assertIsInstance(body["took_ms"], int)

        first, second = body["results"]
        self.assertEqual(first["rank"], 1)
        self.assertEqual(first["similarity"], 0.75)
        self.assertEqual(first["channel_id"], "42")
        self.assertEqual(first["node_id"], "T_42")
        self.assertEqual(first["people"], ["Me", "Alice"])
        self.assertTrue(first["snippet"].startswith("[2023-05-01 10:00:00] Me: thinking about a Fuji"))
        self.assertNotIn("Participants:", first["snippet"])
        self.assertIn("Participants:", first["text"])
        # A chunk with no thread id can't be placed on the graph.
        self.assertIsNone(second["channel_id"])
        self.assertIsNone(second["node_id"])

        self.assertEqual(self.loader.calls, [MODEL])
        self.assertEqual(self.loader.model.queries, ["which camera to buy"])
        self.assertEqual(table.queries[0].distance, "cosine")
        self.assertEqual(table.queries[0].n, 2)

    def test_default_limit(self):
        table = FakeTable([TOPIC_ROW] * 20)
        body = self.client(table).post("/api/search", json={"query": "camera"}).json()
        self.assertEqual(body["count"], search.DEFAULT_LIMIT)

    def test_model_loaded_once_across_requests(self):
        c = self.client(FakeTable([TOPIC_ROW]))
        for q in ("camera", "photography", "कैमरा"):
            self.assertEqual(c.post("/api/search", json={"query": q}).status_code, 200)
        self.assertEqual(self.loader.calls, [MODEL])
        self.assertEqual(self.loader.model.queries, ["camera", "photography", "कैमरा"])

    def test_empty_results(self):
        body = self.client(FakeTable([])).post("/api/search", json={"query": "nothing"}).json()
        self.assertEqual((body["count"], body["results"]), (0, []))


class ErrorTests(ApiTestCase):
    def assertError(self, r, status, code):
        self.assertEqual(r.status_code, status, r.text)
        self.assertEqual(r.json()["error"]["code"], code)
        self.assertTrue(r.json()["error"]["message"])

    def test_missing_index_is_503_index_unavailable(self):
        r = self.client().post("/api/search", json={"query": "camera"})
        self.assertError(r, 503, "index_unavailable")
        self.assertIn("not found", r.json()["error"]["message"])
        self.assertEqual(self.loader.calls, [])

    def test_missing_table_in_existing_db_is_503(self):
        db = self.tmp_path / "lancedb"
        db.mkdir()
        fake_db = SimpleNamespace(table_names=lambda: ["topics_pilot"])
        with mock.patch.dict(sys.modules, {"lancedb": SimpleNamespace(connect=lambda p: fake_db)}):
            r = self.client(db_path=db).post("/api/search", json={"query": "camera"})
        self.assertError(r, 503, "index_unavailable")
        self.assertIn("topics_pilot", r.json()["error"]["message"])

    def test_index_appearing_later_works_without_restart(self):
        c = self.client()
        self.assertError(c.post("/api/search", json={"query": "camera"}), 503, "index_unavailable")
        with mock.patch.object(search, "open_table", return_value=FakeTable([TOPIC_ROW])), \
                mock.patch.object(search, "table_model_metadata", return_value=dict(META)):
            r = c.post("/api/search", json={"query": "camera"})
            self.assertEqual(r.status_code, 200, r.text)
            self.assertTrue(c.get("/api/health").json()["index"]["available"])

    def test_model_failure_is_500_and_retried(self):
        loader = Loader(error="Could not load embedding model")
        c = self.client(FakeTable([TOPIC_ROW]), loader=loader)
        self.assertError(c.post("/api/search", json={"query": "camera"}), 500, "model_unavailable")
        loader.error = None
        self.assertEqual(c.post("/api/search", json={"query": "camera"}).status_code, 200)
        self.assertEqual(loader.calls, [MODEL, MODEL])

    def test_dimension_mismatch_is_search_failed(self):
        c = self.client(FakeTable([TOPIC_ROW], dim=3), loader=Loader(FakeModel(dim=5)))
        r = c.post("/api/search", json={"query": "camera"})
        self.assertError(r, 500, "search_failed")
        self.assertIn("dimension mismatch", r.json()["error"]["message"])

    def test_backend_exception_is_search_failed(self):
        table = FakeTable([TOPIC_ROW])
        table.search = mock.Mock(side_effect=OSError("disk gone"))
        self.assertError(self.client(table).post("/api/search", json={"query": "camera"}), 500, "search_failed")

    def test_invalid_requests(self):
        c = self.client(FakeTable([TOPIC_ROW]))
        for payload in ({"query": "   "}, {"query": ""}, {}, {"query": "x", "limit": 0},
                        {"query": "x", "limit": server.MAX_LIMIT + 1}, {"query": "x" * (server.MAX_QUERY_CHARS + 1)}):
            with self.subTest(payload=payload):
                self.assertError(c.post("/api/search", json=payload), 422, "invalid_request")
        self.assertEqual(self.loader.calls, [], "invalid requests must not load the model")


def chunk(parent_id, start, text="[2023-05-01 10:00:00] Me: hello\n[2023-05-01 10:01:00] Alice: hi"):
    return {"parent_id": parent_id, "platform": "discord", "title": "Camera nerds", "start_time": start,
            "end_time": start, "text": "Platform: DISCORD\nTitle: Camera nerds\n" + text}


class ThreadContextTests(ApiTestCase):
    def test_returns_chunks_of_that_thread_oldest_first_without_loading_model(self):
        rows = [chunk("42", "2023-05-03T00:00:00+00:00"), chunk("7", "2023-01-01T00:00:00+00:00"),
                chunk("42", "2023-05-01T00:00:00+00:00")]
        table = FakeTable(rows)
        r = self.client(table).get("/api/thread/T_42")
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual((body["node_id"], body["channel_id"], body["count"]), ("T_42", "42", 2))
        self.assertEqual([c["start_time"] for c in body["chunks"]], ["2023-05-01T00:00:00+00:00", "2023-05-03T00:00:00+00:00"])
        self.assertEqual(body["first_time"], "2023-05-01T00:00:00+00:00")
        self.assertEqual(body["last_time"], "2023-05-03T00:00:00+00:00")
        first = body["chunks"][0]
        self.assertEqual(first["people"], ["Me", "Alice"])
        self.assertTrue(first["snippet"].startswith("[2023-05-01 10:00:00] Me: hello"))
        self.assertNotIn("Platform:", first["snippet"])
        self.assertEqual(table.queries[0].filter, "parent_id = '42'")
        self.assertNotIn("vector", table.queries[0].columns, "context never reads the embeddings")
        self.assertEqual(self.loader.calls, [], "context must not load the embedding model")

    def test_caps_chunks_and_text(self):
        rows = [chunk("5", f"2023-05-{d:02d}T00:00:00+00:00", text="x" * 5000) for d in range(1, 21)]
        body = self.client(FakeTable(rows)).get("/api/thread/T_5").json()
        self.assertEqual(body["count"], 20)
        self.assertEqual(len(body["chunks"]), server.CONTEXT_CHUNKS)
        self.assertLessEqual(len(body["chunks"][0]["text"]), server.CONTEXT_TEXT_CHARS)
        self.assertEqual(body["last_time"], "2023-05-20T00:00:00+00:00")

    def test_unknown_thread_is_empty_not_an_error(self):
        body = self.client(FakeTable([chunk("1", "2023-01-01")])).get("/api/thread/T_999").json()
        self.assertEqual((body["count"], body["chunks"], body["first_time"]), (0, [], None))

    def test_rejects_non_thread_ids(self):
        c = self.client(FakeTable([chunk("1", "2023-01-01")]))
        for bad in ("U_1", "T_", "T_abc", "T_1'%20OR%20'1'='1", "42", "T_1.5"):
            with self.subTest(bad=bad):
                r = c.get(f"/api/thread/{bad}")
                self.assertEqual(r.status_code, 422, r.text)
                self.assertEqual(r.json()["error"]["code"], "invalid_request")

    def test_missing_index_is_503(self):
        r = self.client().get("/api/thread/T_1")
        self.assertEqual(r.status_code, 503)
        self.assertEqual(r.json()["error"]["code"], "index_unavailable")

    def test_thread_chunks_validates_id(self):
        with self.assertRaises(search.SearchError):
            search.thread_chunks(FakeTable([]), "1 OR 1=1")


class StaticAndCorsTests(ApiTestCase):
    def test_serves_graph_page(self):
        c = self.client()
        for path in ("/", "/sarthink_graph.html"):
            r = c.get(path)
            self.assertEqual(r.status_code, 200)
            self.assertIn("sarthink graph", r.text)

    def test_serves_only_whitelisted_graph_files(self):
        (self.graph_dir / "cosmograph_nodes.csv").write_text("id,label\nT_1,x\n", encoding="utf-8")
        (self.graph_dir / "sarthink_lancedb.metadata.json").write_text("{}", encoding="utf-8")
        c = self.client()
        r = c.get("/processed_data/graph/cosmograph_nodes.csv")
        self.assertEqual(r.status_code, 200)
        self.assertIn("T_1", r.text)
        self.assertEqual(r.headers.get("cache-control"), "no-cache", "a regenerated graph must show up on refresh")
        self.assertEqual(c.get("/processed_data/graph/cosmograph_edges.csv").status_code, 404)  # not exported
        self.assertEqual(c.get("/processed_data/graph/sarthink_lancedb.metadata.json").status_code, 404)
        self.assertEqual(c.get("/processed_data/graph/..%2F..%2Fgraph.html").status_code, 404)
        self.assertEqual(c.get("/processed_data/db/sarthink_memory.db").status_code, 404)

    def test_cors_allows_only_local_origins(self):
        c = self.client()
        ok = c.get("/api/health", headers={"Origin": "http://localhost:8080"})
        self.assertEqual(ok.headers.get("access-control-allow-origin"), "http://localhost:8080")
        bad = c.get("/api/health", headers={"Origin": "https://evil.example"})
        self.assertIsNone(bad.headers.get("access-control-allow-origin"))


@unittest.skipUnless(HAVE_FASTAPI, "fastapi not installed")
class HelperTests(unittest.TestCase):
    def test_graph_node_id(self):
        self.assertEqual(server.graph_node_id("42"), "T_42")
        self.assertEqual(server.graph_node_id(7), "T_7")
        self.assertIsNone(server.graph_node_id(""))
        self.assertIsNone(server.graph_node_id(None))

    def test_snippet_skips_header_and_truncates(self):
        text = "Platform: X\nTitle: t\nParticipants: a\nTimeframe: z\nline one\n\nline two"
        self.assertEqual(server.make_snippet(text), "line one line two")
        long = server.make_snippet("word " * 200, limit=50)
        self.assertEqual(len(long), 50)
        self.assertTrue(long.endswith("…"))
        self.assertEqual(server.make_snippet(None), "")

    def test_main_passes_args_to_uvicorn_without_loading_model(self):
        fake_uvicorn = SimpleNamespace(run=mock.Mock())
        with mock.patch.dict(sys.modules, {"uvicorn": fake_uvicorn}), mock.patch.object(search, "load_model") as lm, \
                redirect_stderr(io.StringIO()):
            server.main(["--port", "8123", "--table", "topics_multilingual_pilot"])
        app = fake_uvicorn.run.call_args.args[0]
        self.assertEqual(fake_uvicorn.run.call_args.kwargs["port"], 8123)
        self.assertEqual(fake_uvicorn.run.call_args.kwargs["host"], "127.0.0.1")
        self.assertEqual(app.state.search_service.table_name, "topics_multilingual_pilot")
        lm.assert_not_called()


class GraphPageTests(unittest.TestCase):
    """The UI calls the API and exposes the controls the README documents."""

    def test_graph_html_wires_semantic_search(self):
        html = Path(REPO_ROOT, "sarthink_graph.html").read_text(encoding="utf-8")
        for needle in ("'/api/search'", "'/api/health'", "/api/thread/", 'id="sem-q"', 'id="sem-panel"', "resetSemantic()",
                       "index_unavailable", "semanticSearch"):
            self.assertIn(needle, html)


if __name__ == "__main__":
    unittest.main()
