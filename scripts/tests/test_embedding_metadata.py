"""Tests for the shared embedding config and the metadata embedder.py writes for each table.
Uses a fake model, so no download or GPU is needed; the embedder test needs lancedb (.venv).
Run: .venv/bin/python scripts/tests/test_embedding_metadata.py"""
import datetime
import hashlib
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

try:
    import lancedb
    import numpy as np
    with redirect_stdout(io.StringIO()):
        import embedder
    HAVE_LANCEDB = True
except ImportError:
    HAVE_LANCEDB = False

MULTILINGUAL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class DefaultModelTests(unittest.TestCase):
    def test_multilingual_cpu_default(self):
        self.assertEqual(embedding_config.DEFAULT_EMBEDDING_MODEL, MULTILINGUAL)
        self.assertEqual(embedding_config.EMBEDDING_DEVICE, "cpu")

    @unittest.skipUnless(HAVE_LANCEDB, "lancedb not installed")
    def test_embedder_uses_shared_default(self):
        self.assertEqual(embedder.DEFAULT_EMBEDDING_MODEL, MULTILINGUAL)
        self.assertIs(embedder.load_embedding_model, embedding_config.load_embedding_model)

    def test_loader_uses_cpu_and_local_cache_first(self):
        calls = []
        fake_st = SimpleNamespace(SentenceTransformer=lambda name, **kw: calls.append((name, kw)) or "model")
        with mock.patch.dict(sys.modules, {"sentence_transformers": fake_st}):
            self.assertEqual(embedding_config.load_embedding_model(), "model")
        name, kwargs = calls[0]
        self.assertEqual(name, MULTILINGUAL)
        self.assertEqual(kwargs["device"], "cpu")
        self.assertEqual(kwargs["cache_folder"], embedding_config.MODEL_CACHE_DIR)
        self.assertTrue(kwargs["local_files_only"])

    def test_loader_downloads_only_when_not_cached(self):
        calls = []

        def fake_ctor(name, **kw):
            calls.append(kw)
            if kw.get("local_files_only"):
                raise OSError("not cached")
            return "model"

        with mock.patch.dict(sys.modules, {"sentence_transformers": SimpleNamespace(SentenceTransformer=fake_ctor)}):
            self.assertEqual(embedding_config.load_embedding_model("some/model"), "model")
        self.assertEqual(len(calls), 2)
        self.assertNotIn("local_files_only", calls[1])
        self.assertEqual(calls[1]["device"], "cpu")


class MetadataFileTests(unittest.TestCase):
    def test_path_is_next_to_database(self):
        self.assertEqual(str(embedding_config.metadata_path("/x/graph/sarthink_lancedb")),
                         "/x/graph/sarthink_lancedb.metadata.json")

    def test_save_and_load_round_trip_keeps_other_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "db")
            self.assertEqual(embedding_config.load_metadata(db), {})
            embedding_config.save_table_metadata("a", "m1", 384, "/in.json", db_path=db)
            embedding_config.save_table_metadata("b", "m2", 768, "/in2.json", db_path=db, rows=5)
            tables = embedding_config.load_metadata(db)
            self.assertEqual(set(tables), {"a", "b"})
            self.assertEqual(tables["b"]["rows"], 5)
            self.assertEqual(tables["a"]["model"], "m1")
            datetime.datetime.fromisoformat(tables["a"]["created_at"])
            self.assertFalse(os.path.exists(str(embedding_config.metadata_path(db)) + ".tmp"))


class FakeModel:
    """Deterministic 8-dim vectors; similar enough across bursts to exercise topic splitting."""
    DIM = 8

    def _vec(self, text):
        digest = hashlib.md5(text.encode()).digest()
        return np.frombuffer(digest[:self.DIM], dtype=np.uint8).astype(np.float32) + 1.0

    def encode(self, texts, convert_to_numpy=True, **kwargs):
        if isinstance(texts, str):
            return self._vec(texts)
        return np.stack([self._vec(t) for t in texts])


CHUNKS = [
    {"channel_id": "1", "platform": "reddit", "title": "T1", "participants": ["Me", "A"],
     "start_time": "2024-01-01T00:00:00+00:00", "end_time": "2024-01-01T01:00:00+00:00",
     "density_score": 0.5, "ego_weight": 0.5, "summary": "",
     "text": "[2024-01-01 00:00:00] Me: hello\n\n[2024-01-01 00:30:00] A: college stress\n\n[2024-01-01 01:00:00] Me: same"},
    {"channel_id": "2", "platform": "instagram", "title": "T2", "participants": ["Me", "B"],
     "start_time": "2024-02-01T00:00:00+00:00", "end_time": "2024-02-01T01:00:00+00:00",
     "density_score": 0.2, "ego_weight": 0.7, "summary": "",
     "text": "[2024-02-01 00:00:00] B: photography?\n\n[2024-02-01 01:00:00] Me: yes"},
]


@unittest.skipUnless(HAVE_LANCEDB, "lancedb not installed")
class EmbedderMetadataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = os.path.join(self.tmp.name, "lancedb")
        self.input = os.path.join(self.tmp.name, "chunks.json")
        with open(self.input, "w") as f:
            json.dump(CHUNKS, f)

    def tearDown(self):
        self.tmp.cleanup()

    def run_embedder(self, *extra):
        argv = ["--input", self.input, "--db", self.db, "--topics-only", "--topics-table", "topics_test", *extra]
        with mock.patch.object(embedder, "load_embedding_model", return_value=FakeModel()) as load, \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as err:
            code = embedder.main(argv)
        return code, load, err.getvalue()

    def test_new_table_gets_metadata(self):
        code, load, _ = self.run_embedder()
        self.assertEqual(code, 0)
        load.assert_called_once_with(MULTILINGUAL)

        table = lancedb.connect(self.db).open_table("topics_test")
        meta = embedding_config.get_table_metadata("topics_test", self.db)
        self.assertEqual(meta["table"], "topics_test")
        self.assertEqual(meta["model"], MULTILINGUAL)
        self.assertEqual(meta["vector_dim"], FakeModel.DIM)
        self.assertEqual(meta["vector_dim"], table.schema.field("vector").type.list_size)
        self.assertEqual(meta["source"], os.path.abspath(self.input))
        self.assertEqual(meta["rows"], table.count_rows())
        datetime.datetime.fromisoformat(meta["created_at"])

    def test_model_override_is_recorded(self):
        code, load, _ = self.run_embedder("--model", "sentence-transformers/all-MiniLM-L6-v2")
        self.assertEqual(code, 0)
        load.assert_called_once_with("sentence-transformers/all-MiniLM-L6-v2")
        self.assertEqual(embedding_config.get_table_metadata("topics_test", self.db)["model"],
                         "sentence-transformers/all-MiniLM-L6-v2")

    def test_existing_table_is_not_replaced_without_overwrite(self):
        self.assertEqual(self.run_embedder()[0], 0)
        before = embedding_config.get_table_metadata("topics_test", self.db)
        version = lancedb.connect(self.db).open_table("topics_test").version

        code, load, err = self.run_embedder()
        self.assertEqual(code, 1)
        self.assertIn("Refusing to replace", err)
        load.assert_not_called()
        self.assertEqual(lancedb.connect(self.db).open_table("topics_test").version, version)
        self.assertEqual(embedding_config.get_table_metadata("topics_test", self.db), before)

    def test_overwrite_replaces_table_and_metadata(self):
        self.assertEqual(self.run_embedder()[0], 0)
        self.assertEqual(self.run_embedder("--overwrite", "--model", "other/model")[0], 0)
        self.assertEqual(embedding_config.get_table_metadata("topics_test", self.db)["model"], "other/model")

    def test_created_table_is_searchable_with_its_model(self):
        import search
        self.assertEqual(self.run_embedder()[0], 0)
        with mock.patch.object(search, "load_model", return_value=FakeModel()) as load, \
             redirect_stdout(io.StringIO()) as out, redirect_stderr(io.StringIO()):
            code = search.main(["college stress", "--db", self.db, "--table", "topics_test", "--json"])
        self.assertEqual(code, 0)
        load.assert_called_once_with(MULTILINGUAL)
        self.assertTrue(json.loads(out.getvalue()))


DISCORD_CHUNKS = [
    {"channel_id": "7", "platform": "discord", "title": "DM Alice", "participants": ["Me"],
     "start_time": "2024-03-01T00:00:00+00:00", "end_time": "2024-03-01T02:00:00+00:00",
     "density_score": 0.3, "ego_weight": 1.0, "summary": "",
     "text": "[2024-03-01 00:00:00] Me: synthetic zebra plan\n\n[2024-03-01 02:00:00] Me: later"},
    {"channel_id": "8", "platform": "discord", "title": "#general (S)", "participants": ["Me"],
     "start_time": "2024-03-02T00:00:00+00:00", "end_time": "2024-03-02T00:00:00+00:00",
     "density_score": 0.1, "ego_weight": 1.0, "summary": "",
     "text": "[2024-03-02 00:00:00] Me: one burst only"},
]


@unittest.skipUnless(HAVE_LANCEDB, "lancedb not installed")
class ReplacePlatformTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = os.path.join(self.tmp.name, "lancedb")
        self.base = os.path.join(self.tmp.name, "chunks.json")
        self.discord = os.path.join(self.tmp.name, "discord.json")
        for path, data in ((self.base, CHUNKS), (self.discord, DISCORD_CHUNKS)):
            with open(path, "w") as f:
                json.dump(data, f)
        self.assertEqual(self.run_embedder(self.base)[0], 0)

    def tearDown(self):
        self.tmp.cleanup()

    def run_embedder(self, path, *extra):
        argv = ["--input", path, "--db", self.db, "--topics-only", "--topics-table", "topics_test", *extra]
        with mock.patch.object(embedder, "load_embedding_model", return_value=FakeModel()) as load, \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as err:
            code = embedder.main(argv)
        return code, load, err.getvalue()

    def rows(self):
        t = lancedb.connect(self.db).open_table("topics_test")
        return sorted((r["platform"], r["text"]) for r in t.to_arrow().select(["platform", "text"]).to_pylist())

    def test_single_burst_session_becomes_one_topic(self):
        topics = embedder.topic_chunk_burst(DISCORD_CHUNKS[1], FakeModel())
        self.assertEqual([t["text"] for t in topics], ["[2024-03-02 00:00:00] Me: one burst only"])
        self.assertEqual(embedder.topic_chunk_burst({"text": "  "}, FakeModel()), [])

    def test_adds_platform_rows_and_keeps_the_rest(self):
        before = self.rows()
        meta_before = embedding_config.get_table_metadata("topics_test", self.db)
        code, load, _ = self.run_embedder(self.discord, "--replace-platform", "discord")
        self.assertEqual(code, 0)
        load.assert_called_once_with(MULTILINGUAL)          # the table's recorded model
        after = self.rows()
        self.assertEqual([r for r in after if r[0] != "discord"], before)
        self.assertEqual(sum(r[0] == "discord" for r in after), 2)
        meta = embedding_config.get_table_metadata("topics_test", self.db)
        self.assertEqual((meta["model"], meta["vector_dim"], meta["created_at"]),
                         (meta_before["model"], meta_before["vector_dim"], meta_before["created_at"]))
        self.assertEqual(meta["rows"], len(after))
        upd = meta["platform_updates"]["discord"]
        self.assertEqual((upd["rows"], upd["replaced_rows"]), (2, 0))
        self.assertLess(upd["version_before"], upd["version_after"])

    def test_rerun_replaces_instead_of_duplicating(self):
        self.assertEqual(self.run_embedder(self.discord, "--replace-platform", "discord")[0], 0)
        first = self.rows()
        self.assertEqual(self.run_embedder(self.discord, "--replace-platform", "discord")[0], 0)
        self.assertEqual(self.rows(), first)
        meta = embedding_config.get_table_metadata("topics_test", self.db)
        self.assertEqual(meta["platform_updates"]["discord"]["replaced_rows"], 2)

    def test_refusals_leave_table_untouched(self):
        before = self.rows()
        version = lancedb.connect(self.db).open_table("topics_test").version
        cases = [(self.base, ["--replace-platform", "discord"], "other platforms"),
                 (self.discord, ["--replace-platform", "discord", "--model", "other/model"], "was embedded with"),
                 (self.discord, ["--replace-platform", "disc'ord"], "Invalid platform"),
                 (self.discord, ["--replace-platform", "discord", "--topics-table", "missing"], "must already exist")]
        for path, extra, message in cases:
            code, _, err = self.run_embedder(path, *extra)
            self.assertEqual(code, 1, extra)
            self.assertIn(message, err)
        self.assertEqual(self.rows(), before)
        self.assertEqual(lancedb.connect(self.db).open_table("topics_test").version, version)


if __name__ == "__main__":
    unittest.main()
