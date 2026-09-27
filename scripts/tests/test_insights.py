"""Tests for GET /api/insights (scripts/api/insights.py + server.py) against a throwaway synthetic SQLite DB.
No real archive, database, index or model is touched. Needs fastapi + httpx (.venv).
Run: .venv/bin/python scripts/tests/test_insights.py"""
import datetime as dt
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.append(os.path.join(REPO_ROOT, "scripts", "semantic"))
sys.path.append(os.path.join(REPO_ROOT, "scripts", "api"))

import insights

try:
    from fastapi.testclient import TestClient
    import server
    HAVE_FASTAPI = True
except ImportError:
    HAVE_FASTAPI = False

SCHEMA = """
CREATE TABLE Users (id INTEGER PRIMARY KEY AUTOINCREMENT, platform TEXT, raw_id TEXT, display_name TEXT, UNIQUE(platform, raw_id));
CREATE TABLE Threads (id INTEGER PRIMARY KEY AUTOINCREMENT, platform TEXT, platform_thread_id TEXT, title TEXT, UNIQUE(platform, platform_thread_id));
CREATE TABLE Messages (msg_id TEXT PRIMARY KEY, thread_id INTEGER, author_id INTEGER, timestamp_utc INTEGER, content TEXT, parent_msg_id TEXT);
"""
OWNER = ("Test Owner (me)", {"test_owner", "Test Owner (me)", "test owner (me)", "owner_handle"})


def ts(y, m, d=1):
    return int(dt.datetime(y, m, d, 12, tzinfo=dt.timezone.utc).timestamp())


def make_db(path):
    """Synthetic memory DB: owner on two platforms, three contacts, three threads over 2023-2024."""
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    users = [(1, "reddit", "test_owner", "Test Owner"), (2, "reddit", "alice_r", "Alice"),
             (3, "instagram", "9001", "Bob <b>Builder</b>"), (4, "instagram", "owner_handle", "Owner IG"),
             (5, "reddit", "carol_r", "Carol")]
    conn.executemany("INSERT INTO Users VALUES (?,?,?,?)", users)
    threads = [(10, "reddit", "r1", "Sourdough, starters & <script>x</script>"), (11, "reddit", "r2", "Camera talk"),
               (12, "instagram", "i1", "DM with Bob")]
    conn.executemany("INSERT INTO Threads VALUES (?,?,?,?)", threads)
    msgs = []
    k = 0

    def add(thread, author, when, n=1, ms=False):
        nonlocal k
        for _ in range(n):
            k += 1
            msgs.append((f"m{k}", thread, author, when * 1000 if ms else when, "synthetic", None))

    add(10, 1, ts(2023, 1, 5), 3)     # owner
    add(10, 2, ts(2023, 1, 6), 4)     # alice
    add(10, 5, ts(2023, 3, 2), 1)     # carol
    add(11, 2, ts(2023, 6, 1), 2)
    add(11, 1, ts(2023, 6, 2), 1)
    add(12, 3, ts(2024, 2, 10), 5, ms=True)   # millisecond timestamps are normalised
    add(12, 4, ts(2024, 2, 11), 2)    # owner's instagram account
    conn.executemany("INSERT INTO Messages VALUES (?,?,?,?,?,?)", msgs)
    conn.commit()
    conn.close()
    return len(msgs)


class InsightsTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db = Path(self.tmp.name) / "memory.db"
        self.total = make_db(self.db)

    def service(self, identity=OWNER, db=None):
        return insights.InsightsService(db or self.db, identity_map=identity)


class ComputeTests(InsightsTestCase):
    def test_totals_platforms_and_dates(self):
        r = self.service().insights()
        self.assertFalse(r["empty"])
        self.assertEqual(r["totals"], {"messages": self.total, "threads": 3, "people": 3, "platforms": 2})
        self.assertEqual(r["first_date"][:10], "2023-01-05")
        self.assertEqual(r["latest_date"][:10], "2024-02-11")
        plats = {p["platform"]: p for p in r["platforms"]}
        self.assertEqual((plats["reddit"]["messages"], plats["reddit"]["threads"], plats["reddit"]["people"]), (11, 2, 2))
        self.assertEqual((plats["instagram"]["messages"], plats["instagram"]["people"]), (7, 1))
        self.assertEqual(plats["instagram"]["first_date"][:10], "2024-02-10")
        self.assertAlmostEqual(sum(p["share"] for p in r["platforms"]), 1.0, places=3)
        self.assertEqual(r["available_platforms"], ["instagram", "reddit"])

    def test_activity_months_are_continuous_and_add_up(self):
        r = self.service().insights()
        months = r["activity"]["months"]
        self.assertEqual(months[0]["month"], "2023-01")
        self.assertEqual(months[-1]["month"], "2024-02")
        self.assertEqual(len(months), 14, "every month between the first and last memory, zeros included")
        self.assertEqual(sum(m["total"] for m in months), self.total)
        jan = months[0]
        self.assertEqual(jan["platforms"], {"reddit": 7})
        self.assertEqual(months[1]["total"], 0)
        years = {y["year"]: y for y in r["activity"]["years"]}
        self.assertEqual((years[2023]["total"], years[2024]["total"]), (11, 7))
        self.assertEqual(years[2024]["platforms"], {"instagram": 7})

    def test_owner_accounts_are_excluded_from_contacts_and_people(self):
        r = self.service().insights()
        ids = [c["node_id"] for c in r["top_contacts"]]
        self.assertNotIn("U_1", ids)
        self.assertNotIn("U_4", ids)
        self.assertEqual(ids, ["U_2", "U_3", "U_5"], "sorted by message count")
        self.assertEqual(r["owner"], {"configured": True, "excluded_accounts": 2})
        alice = r["top_contacts"][0]
        self.assertEqual((alice["label"], alice["platform"], alice["messages"], alice["threads"]), ("alice_r", "reddit", 6, 2))
        # Instagram labels use the display name (like the graph export), cleaned of commas/quotes only.
        self.assertEqual(r["top_contacts"][1]["label"], "Bob <b>Builder</b>")

    def test_without_identity_map_nobody_is_excluded(self):
        r = self.service(identity=(None, set())).insights()
        self.assertEqual(r["owner"], {"configured": False, "excluded_accounts": 0})
        self.assertEqual(r["totals"]["people"], 5)

    def test_identity_file_is_read(self):
        path = Path(self.tmp.name) / "identity_map.json"
        path.write_text('{"master_persona": "Me", "aliases": {"reddit": ["TEST_OWNER"], "instagram": ["owner_handle"]}}')
        r = self.service(identity=path).insights()
        self.assertEqual(r["owner"]["excluded_accounts"], 2, "aliases match case-insensitively")

    def test_top_conversations(self):
        r = self.service().insights(top=2)
        convs = r["top_conversations"]
        self.assertEqual([c["node_id"] for c in convs], ["T_10", "T_12"])
        self.assertEqual((convs[0]["messages"], convs[0]["people"], convs[0]["platform"]), (8, 3, "reddit"))
        self.assertEqual(convs[0]["title"], "Sourdough starters & <script>x</script>", "commas/quotes cleaned like graph titles")
        self.assertEqual(len(r["top_contacts"]), 2)

    def test_platform_filter(self):
        r = self.service().insights(platforms=["instagram"])
        self.assertEqual(r["totals"], {"messages": 7, "threads": 1, "people": 1, "platforms": 1})
        self.assertEqual([c["node_id"] for c in r["top_contacts"]], ["U_3"])
        self.assertEqual(r["filters"]["platforms"], ["instagram"])
        self.assertEqual(r["available_platforms"], ["instagram", "reddit"], "unfiltered list for the UI's chips")

    def test_date_filter_is_half_open(self):
        start = dt.datetime(2023, 1, 1, tzinfo=dt.timezone.utc)
        end = dt.datetime(2023, 3, 2, 12, tzinfo=dt.timezone.utc)   # exactly carol's message: excluded
        r = self.service().insights(date_from=start, date_to=end)
        self.assertEqual(r["totals"]["messages"], 7)
        self.assertEqual([c["node_id"] for c in r["top_contacts"]], ["U_2"])
        self.assertEqual([m["month"] for m in r["activity"]["months"]], ["2023-01"])

    def test_empty_results(self):
        r = self.service().insights(platforms=["discord"])
        self.assertTrue(r["empty"])
        self.assertEqual(r["totals"], {"messages": 0, "threads": 0, "people": 0, "platforms": 0})
        self.assertEqual((r["first_date"], r["latest_date"]), (None, None))
        self.assertEqual((r["platforms"], r["top_contacts"], r["top_conversations"]), ([], [], []))
        self.assertEqual(r["activity"], {"months": [], "years": []})

    def test_empty_database(self):
        db = Path(self.tmp.name) / "empty.db"
        conn = sqlite3.connect(db)
        conn.executescript(SCHEMA)
        conn.close()
        r = self.service(db=db).insights()
        self.assertTrue(r["empty"])
        self.assertEqual(r["available_platforms"], [])

    def test_database_is_opened_read_only_and_unchanged(self):
        before = self.db.read_bytes()
        svc = self.service()
        svc.insights()
        svc.insights(platforms=["reddit"])
        self.assertEqual(self.db.read_bytes(), before)
        with self.assertRaises(sqlite3.OperationalError):
            svc._connect().execute("INSERT INTO Users (platform, raw_id) VALUES ('x', 'y')")

    def test_cache_hits_and_invalidation(self):
        svc = self.service()
        self.assertFalse(svc.insights()["cached"])
        self.assertTrue(svc.insights()["cached"])
        conn = sqlite3.connect(self.db)
        conn.execute("INSERT INTO Messages VALUES ('extra', 11, 2, ?, 'synthetic', NULL)", (ts(2023, 6, 3),))
        conn.commit()
        conn.close()
        os.utime(self.db, ns=(1, 1))   # make the change visible even on coarse-mtime filesystems
        r = svc.insights()
        self.assertFalse(r["cached"])
        self.assertEqual(r["totals"]["messages"], self.total + 1)

    def test_missing_database(self):
        with self.assertRaises(insights.InsightsUnavailable):
            self.service(db=Path(self.tmp.name) / "nope.db").insights()


@unittest.skipUnless(HAVE_FASTAPI, "fastapi not installed (.venv/bin/pip install fastapi uvicorn)")
class EndpointTests(InsightsTestCase):
    def client(self, db=None):
        html = Path(self.tmp.name) / "graph.html"
        html.write_text("<html></html>")
        svc = server.SearchService(Path(self.tmp.name) / "no_lancedb", "topics", model_loader=lambda n: None)
        app = server.create_app(svc, graph_html=html, graph_dir=Path(self.tmp.name), insights_service=self.service(db=db))
        return TestClient(app)

    def test_response_shape(self):
        r = self.client().get("/api/insights")
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        for key in ("empty", "filters", "totals", "first_date", "latest_date", "platforms", "available_platforms",
                    "activity", "top_contacts", "top_conversations", "owner", "took_ms", "cached"):
            self.assertIn(key, body)
        self.assertEqual(set(body["totals"]), {"messages", "threads", "people", "platforms"})
        self.assertEqual(set(body["activity"]), {"months", "years"})
        self.assertEqual(set(body["top_contacts"][0]), {"node_id", "label", "platform", "messages", "threads", "first_date", "last_date"})
        self.assertEqual(set(body["top_conversations"][0]), {"node_id", "title", "platform", "messages", "people", "first_date", "last_date"})
        self.assertEqual(set(body["platforms"][0]), {"platform", "messages", "threads", "people", "first_date", "last_date", "share"})

    def test_platform_query_forms(self):
        c = self.client()
        for q in ("platforms=instagram", "platforms=Instagram,", "platforms=instagram&platforms=instagram"):
            body = c.get(f"/api/insights?{q}").json()
            self.assertEqual(body["totals"]["messages"], 7, q)
        both = c.get("/api/insights?platforms=reddit,instagram").json()
        self.assertEqual(both["filters"]["platforms"], ["instagram", "reddit"])
        self.assertEqual(c.get("/api/insights?platforms=").json()["filters"]["platforms"], None, "blank means all")

    def test_date_query(self):
        c = self.client()
        body = c.get("/api/insights?date_from=2024-01-01&date_to=2024-02-10").json()
        self.assertEqual(body["totals"]["messages"], 5, "a plain date_to includes that whole day")
        body = c.get("/api/insights?date_from=2024-01-01T00:00:00Z&date_to=2024-02-10T00:00:00Z").json()
        self.assertTrue(body["empty"], "a datetime date_to is exclusive")

    def test_invalid_params(self):
        c = self.client()
        cases = {"platforms=bad name": "platform", "platforms=x;drop": "platform", "date_from=yesterday": "date_from",
                 "date_from=2024-02-01&date_to=2024-01-01": "before", "top=0": "top", "top=500": "top"}
        for q, word in cases.items():
            r = c.get(f"/api/insights?{q}")
            self.assertEqual(r.status_code, 422, q)
            self.assertEqual(r.json()["error"]["code"], "invalid_request", q)
            self.assertIn(word, r.json()["error"]["message"], q)

    def test_missing_database_is_503(self):
        r = self.client(db=Path(self.tmp.name) / "nope.db").get("/api/insights")
        self.assertEqual(r.status_code, 503)
        self.assertEqual(r.json()["error"]["code"], "database_unavailable")

    def test_health_reports_memory_db(self):
        self.assertTrue(self.client().get("/api/health").json()["memory_db"]["available"])
        self.assertFalse(self.client(db=Path(self.tmp.name) / "nope.db").get("/api/health").json()["memory_db"]["available"])

    def test_insights_is_get_only(self):
        self.assertEqual(self.client().post("/api/insights").status_code, 405)


if __name__ == "__main__":
    unittest.main()
