"""Tests for person profiles (scripts/api/people.py and the /api/person endpoints) against a throwaway synthetic
SQLite database. No real archive, database, index or model is touched. Needs fastapi + httpx (.venv).
Run: .venv/bin/python scripts/tests/test_people.py"""
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

import people  # noqa: E402

try:
    from fastapi.testclient import TestClient
    import server
    HAVE_FASTAPI = True
except ImportError:
    HAVE_FASTAPI = False

UTC = dt.timezone.utc
SCHEMA = """
CREATE TABLE Users (id INTEGER PRIMARY KEY AUTOINCREMENT, platform TEXT, raw_id TEXT, display_name TEXT, UNIQUE(platform, raw_id));
CREATE TABLE Threads (id INTEGER PRIMARY KEY AUTOINCREMENT, platform TEXT, platform_thread_id TEXT, title TEXT, UNIQUE(platform, platform_thread_id));
CREATE TABLE Messages (msg_id TEXT PRIMARY KEY, thread_id INTEGER, author_id INTEGER, timestamp_utc INTEGER, content TEXT, parent_msg_id TEXT);
CREATE INDEX idx_author ON Messages(author_id);
CREATE INDEX idx_thread ON Messages(thread_id);
"""
OWNER = ("Test Owner (me)", {"test_owner", "owner_ig", "test owner (me)"})


def ts(y, m, d=1, h=12):
    return int(dt.datetime(y, m, d, h, tzinfo=UTC).timestamp())


def make_db(path):
    """Synthetic archive.
    Users: 1 owner (reddit), 2 Alice (reddit), 3 Bob (instagram), 4 owner (instagram), 5 alice (instagram:
    same name as 2, NOT mapped), 6 AutoModerator, 7 Carol (one message), 8..30 crowd in a big public thread.
    Threads: 10 DM owner+Alice (reddit), 11 group owner+Alice+Bob (instagram... reddit), 12 huge public thread,
    13 DM owner+Bob (instagram), 14 Alice alone post, 15 Carol DM."""
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    users = [(1, "reddit", "test_owner", "Me"), (2, "reddit", "alice_r", "Alice"), (3, "instagram", "bob.ig", "Bob"),
             (4, "instagram", "owner_ig", "Me IG"), (5, "instagram", "alice_ig", "alice_r"), (6, "reddit", "AutoModerator", "AutoModerator"),
             (7, "reddit", "carol_r", "Carol")] + [(k, "reddit", f"crowd{k}", f"crowd{k}") for k in range(8, 31)]
    conn.executemany("INSERT INTO Users VALUES (?,?,?,?)", users)
    conn.executemany("INSERT INTO Threads VALUES (?,?,?,?)", [
        (10, "reddit", "dm1", "DM with alice_r"), (11, "reddit", "g1", "Group: <b>weekend</b> plans"),
        (12, "reddit", "p1", "Huge public thread"), (13, "instagram", "dm2", "DM with Bob"),
        (14, "reddit", "p2", "Alice post about sourdough"), (15, "reddit", "dm3", "DM with carol")])
    msgs, k = [], 0

    def add(thread, author, when, text):
        nonlocal k
        k += 1
        msgs.append((f"m{k:04d}", thread, author, when, text, None))

    # DM with Alice: 2023-01 .. 2023-03, back and forth, sourdough recurring
    for i in range(6):
        add(10, 2, ts(2023, 1 + i // 2, 3 + i), f"alice dm {i} sourdough starter update")
        add(10, 1, ts(2023, 1 + i // 2, 3 + i, 13), f"my reply {i}")
    # Group chat: Alice, Bob, owner in 2024-05
    for i in range(3):
        add(11, 2, ts(2024, 5, 10 + i), f"group plan {i} <script>x</script> sourdough picnic")
        add(11, 3, ts(2024, 5, 10 + i, 13), f"bob group {i}")
        add(11, 1, ts(2024, 5, 10 + i, 14), f"owner group {i}")
    # Huge public thread: Alice writes the most here; owner comments once; crowd of 23
    for i in range(25):
        add(12, 2, ts(2024, 8, 1 + i), f"public comment {i}")
    add(12, 1, ts(2024, 8, 2), "owner public comment")
    for u in range(8, 31):
        add(12, u, ts(2024, 8, 3), "crowd says hi")
    add(12, 6, ts(2024, 8, 4), "Your comment was removed")
    # DM with Bob on instagram (owner via instagram account 4)
    for i in range(4):
        add(13, 3, ts(2025, 2, 1 + i), f"bob dm {i}")
        add(13, 4, ts(2025, 2, 1 + i, 13), f"owner ig {i}")
    # Alice alone
    add(14, 2, ts(2025, 6, 1), "sourdough starter photos, finally")
    # Carol: one message, owner replies once
    add(15, 7, ts(2025, 7, 1), "hey")
    add(15, 1, ts(2025, 7, 1, 13), "hi carol")
    # alice_ig (user 5): same name as Alice on another platform, a single message in Bob's DM thread is NOT added;
    # she writes in a separate instagram thread so her history must not appear under Alice.
    conn.execute("INSERT INTO Threads VALUES (16, 'instagram', 'dm4', 'DM with alice_ig')")
    add(16, 5, ts(2025, 8, 1), "instagram alice says hello")
    add(16, 4, ts(2025, 8, 1, 13), "owner ig reply")
    conn.executemany("INSERT INTO Messages VALUES (?,?,?,?,?,?)", msgs)
    conn.commit()
    conn.close()


class PeopleTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db = Path(self.tmp.name) / "memory.db"
        make_db(self.db)
        self.svc = people.PeopleService(self.db, identity_map=OWNER)


class ProfileTests(PeopleTestCase):
    def test_counts_and_dates(self):
        p = self.svc.profile("U_2")
        s = p["stats"]
        self.assertEqual(p["kind"], "person")
        self.assertEqual(p["label"], "alice_r")
        self.assertEqual((s["conversations"], s["their_messages"]), (4, 6 + 3 + 25 + 1))
        self.assertEqual(s["shared_conversations"], 2, "DM + group; the huge thread and her own post don't count")
        self.assertEqual(s["your_messages"], 6 + 3, "yours only in shared small conversations")
        self.assertEqual((s["first_date"][:10], s["latest_date"][:10]), ("2023-01-03", "2025-06-01"))
        self.assertEqual(s["platforms"], ["reddit"])
        self.assertEqual(p["lifetime"], s, "no filters: lifetime equals the shown stats")
        self.assertFalse(p["scope"]["filtered"])

    def test_activity_timeline(self):
        months = {a["month"]: a["messages"] for a in self.svc.profile("U_2")["activity"]}
        self.assertEqual(months["2023-01"], 2)
        self.assertEqual(months["2024-08"], 25)
        self.assertEqual(sum(months.values()), 35)

    def test_notable_prefers_real_conversations_over_huge_threads(self):
        p = self.svc.profile("U_2")
        notable = [s for s in p["sources"] if s["notable"]]
        self.assertEqual(notable[0]["node_id"], "T_10", "the back-and-forth DM leads")
        big = next(s for s in p["sources"] if s["node_id"] == "T_12")
        self.assertTrue(big["large"])
        self.assertEqual(big["people"], 26)
        self.assertGreater(p["sources"].index(big), p["sources"].index(next(s for s in p["sources"] if s["node_id"] == "T_11")))

    def test_brief_is_counted_and_cited(self):
        p = self.svc.profile("U_2")
        text = " ".join(x["text"] for x in p["brief"]["sentences"])
        self.assertIn("alice_r wrote 35 messages in 4 conversations on Reddit, from Jan 2023 to Jun 2025.", text)
        self.assertIn("You both wrote in 2 small conversations, where you wrote 9 messages.", text)
        self.assertIn("Huge public thread", text)
        self.assertIn("a large thread with 26 people", text)
        self.assertIn("sourdough", text, "recurring word across conversations")
        n = len(p["sources"])
        for sentence in p["brief"]["sentences"]:
            self.assertTrue(all(1 <= i <= n for i in sentence["sources"]), sentence)
        for bad in ("friend", "love", "close", "feel", "relationship"):
            self.assertNotIn(bad, text.lower(), "no inferred feelings or relationships")

    def test_topics_have_evidence(self):
        topics = {t["word"]: t for t in self.svc.profile("U_2")["topics"]}
        self.assertIn("sourdough", topics)
        self.assertGreaterEqual(topics["sourdough"]["conversations"], 2)
        self.assertTrue(topics["sourdough"]["sources"])
        self.assertNotIn("public", topics, "a word from one conversation in one month isn't recurring")

    def test_source_links_point_at_their_conversations(self):
        p = self.svc.profile("U_2")
        self.assertTrue(all(s["node_id"] in ("T_10", "T_11", "T_12", "T_14") for s in p["sources"]))
        self.assertLessEqual(len(p["sources"]), people.MAX_SOURCES)

    def test_filters_and_lifetime_are_separate(self):
        p = self.svc.profile("U_2", date_from=dt.datetime(2024, 1, 1, tzinfo=UTC), date_to=dt.datetime(2025, 1, 1, tzinfo=UTC))
        self.assertTrue(p["scope"]["filtered"])
        self.assertEqual((p["stats"]["conversations"], p["stats"]["their_messages"]), (2, 28))
        self.assertEqual(p["stats"]["your_messages"], 3, "only the group chat is shared in 2024")
        self.assertEqual(p["lifetime"]["their_messages"], 35, "headline lifetime counts stay all-time")
        self.assertEqual({a["month"] for a in p["activity"]}, {"2024-05", "2024-08"})
        none = self.svc.profile("U_2", platforms=["instagram"])
        self.assertEqual(none["stats"]["their_messages"], 0)
        self.assertTrue(none["brief"]["sparse"])
        self.assertIn("No messages", none["brief"]["sentences"][0]["text"])

    def test_same_name_on_another_platform_is_not_merged(self):
        p = self.svc.profile("U_2")
        self.assertEqual([a["node_id"] for a in p["accounts"]], ["U_2"])
        self.assertEqual(p["same_name_elsewhere"], [{"node_id": "U_5", "platform": "instagram"}])
        self.assertNotIn("T_16", [s["node_id"] for s in p["sources"]])
        other = self.svc.profile("U_5")
        self.assertEqual(other["stats"]["their_messages"], 1)

    def test_owner_accounts_merge_only_via_identity_map(self):
        me = self.svc.profile("U_1")
        self.assertEqual(me["kind"], "you")
        self.assertEqual([a["node_id"] for a in me["accounts"]], ["U_1", "U_4"])
        self.assertEqual(me["label"], "Test Owner (me)")
        self.assertEqual(me["stats"]["their_messages"], 6 + 3 + 1 + 1 + 4 + 1)
        self.assertEqual(self.svc.profile("U_4")["accounts"], me["accounts"])

    def test_related_excludes_owner_bots_and_self(self):
        related = {r["node_id"] for r in self.svc.profile("U_2")["related"]}
        self.assertIn("U_3", related, "Bob shares the small group chat")
        self.assertFalse({"U_1", "U_4", "U_2", "U_6"} & related)
        self.assertFalse({f"U_{k}" for k in range(8, 31)} & related, "the crowd is only in a large thread")

    def test_automated_account(self):
        p = self.svc.profile("U_6")
        self.assertEqual(p["kind"], "automated")
        self.assertIn("automated or deleted", p["brief"]["sentences"][0]["text"])

    def test_sparse_history(self):
        p = self.svc.profile("U_7")
        self.assertTrue(p["brief"]["sparse"])
        self.assertIn("There isn’t enough history to summarize: carol_r wrote 1 message in 1 conversation", p["brief"]["sentences"][0]["text"])
        self.assertEqual(p["brief"]["sentences"][0]["sources"], [1])
        self.assertEqual(p["sources"][0]["node_id"], "T_15")

    def test_errors(self):
        with self.assertRaises(people.BadRequest):
            self.svc.profile("T_10")
        with self.assertRaises(people.PersonNotFound):
            self.svc.profile("U_999")
        with self.assertRaises(people.PeopleUnavailable):
            people.PeopleService(Path(self.tmp.name) / "nope.db", identity_map=OWNER).profile("U_2")

    def test_read_only(self):
        before = self.db.read_bytes()
        self.svc.profile("U_2")
        self.svc.messages("U_2", limit=5)
        self.svc.conversations("U_2")
        self.assertEqual(self.db.read_bytes(), before)


class PagingTests(PeopleTestCase):
    def all_pages(self, **kw):
        out, cursor, pages = [], None, 0
        while True:
            page = self.svc.messages("U_2", cursor=cursor, limit=4, **kw)
            pages += 1
            out += page["messages"]
            cursor = page["next_cursor"]
            if not cursor:
                return out, pages

    def test_full_history_pages_without_gaps(self):
        first = self.svc.messages("U_2", limit=4)
        self.assertEqual(first["total"], 35 + 9, "theirs + yours in shared small conversations")
        msgs, pages = self.all_pages()
        self.assertEqual(len(msgs), 44)
        self.assertEqual(pages, 11)
        dates = [m["date"] for m in msgs]
        self.assertEqual(dates, sorted(dates, reverse=True), "newest first")
        self.assertEqual(len({(m["date"], m["text"]) for m in msgs}), 44, "no duplicates across pages")
        mine = [m for m in msgs if m["author"] == "you"]
        self.assertEqual(len(mine), 9)
        self.assertTrue(all(m["node_id"] in ("T_10", "T_11") for m in mine), "your comment in the huge thread is left out")
        self.assertTrue(all(m["author_label"] == "alice_r" for m in msgs if m["author"] == "them"))

    def test_thread_page_and_escaping_left_to_the_browser(self):
        page = self.svc.messages("U_2", thread="T_11", limit=50)
        self.assertEqual(page["total"], 6)
        self.assertTrue(all(m["node_id"] == "T_11" for m in page["messages"]))
        self.assertIn("<script>x</script>", " ".join(m["text"] for m in page["messages"]), "stored text is returned verbatim")
        self.assertEqual(page["messages"][0]["thread_title"], "Group: <b>weekend</b> plans")
        with self.assertRaises(people.BadRequest):
            self.svc.messages("U_2", thread="T_13")   # Bob's DM is not Alice's conversation
        with self.assertRaises(people.BadRequest):
            self.svc.messages("U_2", cursor="not-a-cursor")

    def test_filters_apply_to_history(self):
        page = self.svc.messages("U_2", limit=100, date_from=dt.datetime(2025, 1, 1, tzinfo=UTC))
        self.assertEqual(page["total"], 1)
        self.assertEqual(page["messages"][0]["node_id"], "T_14")

    def test_long_messages_are_clipped_for_display_only(self):
        conn = sqlite3.connect(self.db)
        conn.execute("INSERT INTO Messages VALUES ('long', 14, 2, ?, ?, NULL)", (ts(2025, 6, 2), "x" * (people.MESSAGE_CHARS + 50)))
        conn.commit()
        conn.close()
        svc = people.PeopleService(self.db, identity_map=OWNER)
        m = svc.messages("U_2", thread="T_14", limit=1)["messages"][0]
        self.assertTrue(m["clipped"])
        self.assertEqual(len(m["text"]), people.MESSAGE_CHARS)
        stored = sqlite3.connect(self.db).execute("SELECT length(content) FROM Messages WHERE msg_id='long'").fetchone()[0]
        self.assertEqual(stored, people.MESSAGE_CHARS + 50, "the database keeps the full message")

    def test_conversation_list_pages(self):
        first = self.svc.conversations("U_2", limit=3)
        self.assertEqual((first["total"], len(first["items"]), first["next_offset"]), (4, 3, 3))
        self.assertEqual(first["items"][0]["node_id"], "T_14", "latest first")
        rest = self.svc.conversations("U_2", offset=3, limit=3)
        self.assertEqual((len(rest["items"]), rest["next_offset"]), (1, None))


@unittest.skipUnless(HAVE_FASTAPI, "fastapi not installed (.venv/bin/pip install fastapi uvicorn)")
class EndpointTests(PeopleTestCase):
    def client(self):
        html = Path(self.tmp.name) / "g.html"
        html.write_text("<html></html>")

        def no_model(name):
            raise AssertionError("opening a person must not load the embedding model")
        svc = server.SearchService(Path(self.tmp.name) / "no_lancedb", "topics", model_loader=no_model)
        return TestClient(server.create_app(svc, graph_html=html, graph_dir=Path(self.tmp.name), people_service=self.svc))

    def test_profile_endpoint(self):
        r = self.client().get("/api/person/U_2")
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        for key in ("node_id", "label", "kind", "accounts", "same_name_elsewhere", "scope", "lifetime", "stats",
                    "activity", "topics", "sources", "brief", "related"):
            self.assertIn(key, body)

    def test_filters_via_query(self):
        body = self.client().get("/api/person/U_2?date_from=2024-01-01&date_to=2024-12-31").json()
        self.assertTrue(body["scope"]["filtered"])
        self.assertEqual(body["stats"]["their_messages"], 28)
        self.assertEqual(body["lifetime"]["their_messages"], 35)

    def test_paging_endpoints(self):
        c = self.client()
        page = c.get("/api/person/U_2/messages?limit=10").json()
        self.assertEqual((page["count"], page["total"]), (10, 44))
        nxt = c.get(f"/api/person/U_2/messages?limit=10&cursor={page['next_cursor']}").json()
        self.assertEqual(nxt["count"], 10)
        self.assertIsNone(nxt["total"])
        convs = c.get("/api/person/U_2/conversations?limit=2").json()
        self.assertEqual((convs["total"], convs["next_offset"]), (4, 2))

    def test_errors(self):
        c = self.client()
        for path, status, code in [("/api/person/T_10", 422, "invalid_request"), ("/api/person/U_999", 404, "not_found"),
                                   ("/api/person/U_2/messages?thread=T_13", 422, "invalid_request"),
                                   ("/api/person/U_2/messages?cursor=zzz", 422, "invalid_request"),
                                   ("/api/person/U_2/messages?limit=1000", 422, "invalid_request"),
                                   ("/api/person/U_2?platforms=bad name", 422, "invalid_request")]:
            r = c.get(path)
            self.assertEqual((r.status_code, r.json()["error"]["code"]), (status, code), path)

    def test_missing_database(self):
        svc = people.PeopleService(Path(self.tmp.name) / "nope.db", identity_map=OWNER)
        html = Path(self.tmp.name) / "g.html"
        html.write_text("<html></html>")
        app = server.create_app(server.SearchService(Path(self.tmp.name) / "x", "topics"), graph_html=html,
                                graph_dir=Path(self.tmp.name), people_service=svc)
        r = TestClient(app).get("/api/person/U_2")
        self.assertEqual((r.status_code, r.json()["error"]["code"]), (503, "database_unavailable"))


class PageEscapingTests(unittest.TestCase):
    def test_person_rendering_escapes_archive_text(self):
        import re
        html = Path(REPO_ROOT, "sarthink_graph.html").read_text(encoding="utf-8")
        block = html[html.index("function renderPerson"):html.index("function schedulePersonRefresh")]
        for field in ("d.label", "s.text", "c.title", "m.text", "m.author_label", "m.thread_title", "t.word", "r.label", "t.title", "n.label", "t"):
            self.assertNotRegex(block, r"\$\{" + re.escape(field) + r"[}\s]", f"unescaped interpolation of {field} in the person view")
        self.assertIn("'/api/person/'", html.replace("`/api/person/${", "'/api/person/'"))


if __name__ == "__main__":
    unittest.main()
