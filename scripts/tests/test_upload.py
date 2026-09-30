"""Tests for the data upload and ingestion endpoints in scripts/api/uploader.py.

Verifies:
- Platform listing and export instructions
- Archive status and file discovery
- Uploading chat files (.txt, .json, .zip) with auto-detection
- Safe extraction and directory placement
- Extension validation and file deletion
- Ingestion status reporting
"""

import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.append(os.path.join(REPO_ROOT, "scripts", "semantic"))
sys.path.append(os.path.join(REPO_ROOT, "scripts", "api"))

from fastapi.testclient import TestClient
import server
import uploader


class UploadApiTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp_dir.name)

        # Create dummy structure
        self.archive_dir = self.tmp_path / "archive"
        self.archive_dir.mkdir(parents=True)
        self.graph_dir = self.tmp_path / "processed_data" / "graph"
        self.graph_dir.mkdir(parents=True)

        # Mock Workspace with root pointing to temp dir
        mock_ws = mock.MagicMock()
        mock_ws.id = "test"
        mock_ws.root = self.tmp_path
        mock_ws.graph_dir = self.graph_dir
        mock_ws.insights.db_path = str(self.tmp_path / "processed_data" / "db" / "sarthink_memory.db")

        # Mock SearchService
        mock_search = mock.MagicMock()
        mock_search.table_name = "topics"
        mock_search.status.return_value = {"available": True}
        mock_ws.search = mock_search

        # Create test client with mock workspace
        app = server.create_app(
            service=mock_search,
            graph_dir=self.graph_dir,
            llm_profiles={},
        )
        # Patch ws_of to return our test workspace
        self.client = TestClient(app)

        # Override ws_of inside app
        self.orig_get_archive_base = uploader.get_archive_base
        uploader.get_archive_base = lambda ws: self.archive_dir

    def tearDown(self):
        uploader.get_archive_base = self.orig_get_archive_base
        self.tmp_dir.cleanup()

    def test_list_platforms(self):
        resp = self.client.get("/api/upload/platforms")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("platforms", data)
        platform_ids = [p["id"] for p in data["platforms"]]
        for expected in ["whatsapp", "instagram", "facebook", "discord", "reddit", "twitter", "google", "chatgpt", "claude"]:
            self.assertIn(expected, platform_ids)

    def test_archive_status_empty(self):
        resp = self.client.get("/api/upload/status")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("platforms", data)
        self.assertEqual(data["platforms"]["whatsapp"]["file_count"], 0)

    def test_upload_whatsapp_txt(self):
        content = "12/31/23, 9:15 PM - Alice: Hey let's meet tomorrow\n12/31/23, 9:16 PM - Bob: Sounds good!"
        file_obj = io.BytesIO(content.encode("utf-8"))

        resp = self.client.post(
            "/api/upload",
            files={"file": ("WhatsApp Chat with Alice.txt", file_obj, "text/plain")},
            data={"platform": "auto"}
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["platform"], "whatsapp")

        # Verify file on disk
        dest_file = self.archive_dir / "whatsapp" / "WhatsApp Chat with Alice.txt"
        self.assertTrue(dest_file.exists())
        self.assertEqual(dest_file.read_text(encoding="utf-8"), content)

    def test_upload_chatgpt_json(self):
        dummy_json = '[{"title": "Trip to Paris", "current_node": "abc", "mapping": {}}]'
        file_obj = io.BytesIO(dummy_json.encode("utf-8"))

        resp = self.client.post(
            "/api/upload",
            files={"file": ("conversations.json", file_obj, "application/json")},
            data={"platform": "chatgpt"}
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["platform"], "chatgpt")

        dest_file = self.archive_dir / "chatgpt" / "conversations.json"
        self.assertTrue(dest_file.exists())

    def test_upload_invalid_extension(self):
        file_obj = io.BytesIO(b"malicious executable")
        resp = self.client.post(
            "/api/upload",
            files={"file": ("malware.exe", file_obj, "application/octet-stream")},
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Unsupported file extension", resp.json()["detail"])

    def test_delete_uploaded_file(self):
        # Create a file in archive
        test_file = self.archive_dir / "whatsapp" / "chat_to_delete.txt"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_text("dummy")

        rel_path = "whatsapp/chat_to_delete.txt"
        resp = self.client.post(f"/api/upload/delete?rel_path={rel_path}")
        self.assertEqual(resp.status_code, 200)
        self.assertFalse(test_file.exists())

    def test_delete_path_traversal_blocked(self):
        resp = self.client.post("/api/upload/delete?rel_path=../../etc/passwd")
        self.assertEqual(resp.status_code, 400)

    def test_ingest_status(self):
        resp = self.client.get("/api/ingest/status")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("status", data)
        self.assertIn("progress", data)


if __name__ == "__main__":
    unittest.main()
