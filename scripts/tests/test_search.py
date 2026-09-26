"""Unit tests for scripts/semantic/search.py (no model, GPU or LanceDB install required).
Run: .venv/bin/python scripts/tests/test_search.py"""
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from types import SimpleNamespace
from unittest import mock

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.append(os.path.join(REPO_ROOT, "scripts", "semantic"))

import embedding_config
import search


class FakeModel:
    def __init__(self, dim=3):
        self.dim = dim
        self.queries = []

    def encode(self, text, convert_to_numpy=True):
        self.queries.append(text)
        return [0.1] * self.dim


class FakeQuery:
    def __init__(self, rows):
        self.rows = rows
        self.distance = None
        self.n = None

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
        self.last_query = None
        self.last_vector = None
        list_type = SimpleNamespace(list_size=dim)
        self.schema = SimpleNamespace(field=lambda name: SimpleNamespace(type=list_type))

    def search(self, vector):
        self.last_vector = vector
        self.last_query = FakeQuery(self.rows)
        return self.last_query


SESSION_ROW = {
    "vector": [0.0, 0.0, 1.0],
    "channel_id": "42",
    "platform": "discord",
    "title": "Camera nerds",
    "start_time": "2023-05-01T10:00:00+00:00",
    "end_time": "2023-05-01T11:00:00+00:00",
    "density_score": 0.5,
    "ego_weight": 0.4,
    "summary": "Discussed buying a used mirrorless camera.",
    "text": "[2023-05-01 10:00:00] Me: thinking about a Fuji\n[2023-05-01 10:02:00] Alice: get the X-T3\n"
            "[2023-05-01 10:05:00] Me: ok",
    "_distance": 0.25,
}


MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def write_meta(db_path, table="topics", model=MODEL, dim=3):
    embedding_config.save_table_metadata(table, model, dim, "/data/chunks.json", db_path=db_path)


class ConfigTests(unittest.TestCase):
    def test_uses_shared_config(self):
        self.assertEqual(search.LANCEDB_PATH, embedding_config.LANCEDB_PATH)
        self.assertIs(search.load_embedding_model, embedding_config.load_embedding_model)


class TableMetadataTests(unittest.TestCase):
    def test_missing_metadata_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(search.SearchError) as ctx:
                search.table_model_metadata(FakeTable([]), "topics_pilot", os.path.join(tmp, "db"))
        self.assertIn("No embedding metadata for table 'topics_pilot'", str(ctx.exception))

    def test_other_tables_metadata_does_not_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "db")
            write_meta(db, table="topics")
            with self.assertRaises(search.SearchError):
                search.table_model_metadata(FakeTable([]), "sessions", db)

    def test_dimension_mismatch_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "db")
            write_meta(db, dim=768)
            with self.assertRaises(search.SearchError) as ctx:
                search.table_model_metadata(FakeTable([], dim=384), "topics", db)
        self.assertIn("Refusing to search", str(ctx.exception))

    def test_matching_metadata_is_returned(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "db")
            write_meta(db, dim=3)
            meta = search.table_model_metadata(FakeTable([], dim=3), "topics", db)
        self.assertEqual(meta["model"], MODEL)


class ExtractPeopleTests(unittest.TestCase):
    def test_chat_authors_in_order_without_duplicates(self):
        self.assertEqual(search.extract_people(SESSION_ROW), ["Me", "Alice"])

    def test_twitter_and_reddit_formats(self):
        text = ("[TWEET] @me — 2023-01-01 00:00:00\nhello\n\n[REPLY] @@SwiftFox — 2023-01-01 00:01:00\nhi\n\n"
                "[POST] u/someone — 2023-01-01 00:00:00\nbody")
        self.assertEqual(search.extract_people({"text": text}), ["me", "@SwiftFox", "someone"])

    def test_explicit_column_wins(self):
        self.assertEqual(search.extract_people({"participants": ["Bob"], "text": SESSION_ROW["text"]}), ["Bob"])

    def test_no_text(self):
        self.assertEqual(search.extract_people({}), [])


class SearchTests(unittest.TestCase):
    def test_returns_ranked_results_with_cosine(self):
        second = dict(SESSION_ROW, _distance=0.6, platform="reddit")
        table = FakeTable([SESSION_ROW, second])
        model = FakeModel()
        results = search.search(table, model, "photography", limit=5)

        self.assertEqual(model.queries, ["photography"])
        self.assertEqual(table.last_query.distance, "cosine")
        self.assertEqual(table.last_query.n, 5)
        self.assertEqual([r["rank"] for r in results], [1, 2])
        top = results[0]
        self.assertEqual(top["similarity"], 0.75)
        self.assertEqual(top["distance"], 0.25)
        self.assertEqual(top["platform"], "discord")
        self.assertEqual(top["start_time"], "2023-05-01T10:00:00+00:00")
        self.assertEqual(top["people"], ["Me", "Alice"])
        self.assertEqual(top["summary"], SESSION_ROW["summary"])
        self.assertEqual(top["text"], SESSION_ROW["text"])
        self.assertNotIn("vector", top)

    def test_limit_is_respected(self):
        table = FakeTable([SESSION_ROW] * 10)
        self.assertEqual(len(search.search(table, FakeModel(), "q", limit=3)), 3)

    def test_topic_rows_without_optional_fields(self):
        row = {"parent_id": "7", "start_time": "2023-01-01T00:00:00+00:00", "text": "x", "_distance": 0.1}
        result = search.search(FakeTable([row]), FakeModel(), "q")[0]
        self.assertEqual(result["channel_id"], "7")
        self.assertIsNone(result["platform"])
        self.assertIsNone(result["summary"])

    def test_dimension_mismatch_is_reported(self):
        with self.assertRaises(search.SearchError) as ctx:
            search.search(FakeTable([SESSION_ROW], dim=2560), FakeModel(dim=384), "q")
        self.assertIn("dimension mismatch", str(ctx.exception))

    def test_format_results_shows_all_fields(self):
        out = search.format_results("photography", search.search(FakeTable([SESSION_ROW]), FakeModel(), "q"))
        for expected in ("#1", "similarity=0.75", "2023-05-01T10:00:00+00:00", "discord", "Camera nerds",
                         "Me, Alice", "mirrorless camera", "get the X-T3"):
            self.assertIn(expected, out)

    def test_format_no_results(self):
        self.assertIn("No results", search.format_results("q", []))


class MissingDataTests(unittest.TestCase):
    def test_missing_db_is_not_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "does_not_exist")
            with self.assertRaises(search.SearchError) as ctx:
                search.open_table(db_path, "sessions")
            self.assertIn("not found", str(ctx.exception))
            self.assertFalse(os.path.exists(db_path))

    def test_missing_table(self):
        fake_db = SimpleNamespace(table_names=lambda: ["topics"])
        fake_lancedb = SimpleNamespace(connect=lambda path: fake_db)
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(sys.modules, {"lancedb": fake_lancedb}):
            with self.assertRaises(search.SearchError) as ctx:
                search.open_table(tmp, "sessions")
        self.assertIn("Table 'sessions' not found", str(ctx.exception))
        self.assertIn("topics", str(ctx.exception))

    def test_missing_model_package(self):
        with mock.patch.dict(sys.modules, {"sentence_transformers": None}):
            with self.assertRaises(search.SearchError):
                search.load_model(MODEL)


class CliTests(unittest.TestCase):
    def run_main(self, argv):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = search.main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_missing_db_exit_code_and_message(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, out, err = self.run_main(["photography", "--db", os.path.join(tmp, "nope")])
        self.assertEqual(code, 1)
        self.assertEqual(out, "")
        self.assertIn("LanceDB database not found", err)

    def test_json_output_uses_model_from_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "db")
            write_meta(db, model="some/other-model", dim=3)
            with mock.patch.object(search, "open_table", return_value=FakeTable([SESSION_ROW])), \
                 mock.patch.object(search, "load_model", return_value=FakeModel()) as load:
                code, out, _ = self.run_main(["photography", "--json", "--limit", "1", "--db", db])
        self.assertEqual(code, 0)
        load.assert_called_once_with("some/other-model")
        data = json.loads(out)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["platform"], "discord")

    def test_missing_metadata_never_loads_a_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(search, "open_table", return_value=FakeTable([SESSION_ROW])), \
                 mock.patch.object(search, "load_model") as load:
                code, out, err = self.run_main(["q", "--table", "topics_pilot", "--db", os.path.join(tmp, "db")])
        self.assertEqual(code, 1)
        self.assertEqual(out, "")
        self.assertIn("No embedding metadata", err)
        load.assert_not_called()

    def test_incompatible_metadata_exit_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "db")
            write_meta(db, dim=768)
            with mock.patch.object(search, "open_table", return_value=FakeTable([SESSION_ROW], dim=384)), \
                 mock.patch.object(search, "load_model") as load:
                code, _, err = self.run_main(["q", "--db", db])
        self.assertEqual(code, 1)
        self.assertIn("Refusing to search", err)
        load.assert_not_called()

    def test_invalid_limit(self):
        with self.assertRaises(SystemExit):
            self.run_main(["q", "--limit", "0"])


if __name__ == "__main__":
    unittest.main()
