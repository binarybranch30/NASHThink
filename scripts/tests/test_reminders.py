"""Tests for reminders (scripts/api/reminders.py + the /api/reminders endpoints) against a throwaway SQLite DB.
No real archive, database, index or model is touched. Needs fastapi + httpx (.venv).
Run: .venv/bin/python scripts/tests/test_reminders.py"""
import datetime as dt
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from zoneinfo import ZoneInfo

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.append(os.path.join(REPO_ROOT, "scripts", "semantic"))
sys.path.append(os.path.join(REPO_ROOT, "scripts", "api"))

import reminders

try:
    from fastapi.testclient import TestClient
    import server
    import workspaces
    HAVE_FASTAPI = True
except ImportError:
    HAVE_FASTAPI = False

IST = ZoneInfo("Asia/Kolkata")
SCHEMA = """
CREATE TABLE Users (id INTEGER PRIMARY KEY AUTOINCREMENT, platform TEXT, raw_id TEXT, display_name TEXT, UNIQUE(platform, raw_id));
CREATE TABLE Threads (id INTEGER PRIMARY KEY AUTOINCREMENT, platform TEXT, platform_thread_id TEXT, title TEXT, UNIQUE(platform, platform_thread_id));
CREATE TABLE Messages (msg_id TEXT PRIMARY KEY, thread_id INTEGER, author_id INTEGER, timestamp_utc INTEGER, content TEXT, parent_msg_id TEXT);
"""
OWNER = ("Test Owner (me)", {"Test Owner (me)", "test owner (me)", "Owner", "owner"})


def ist(when):
    return int(dt.datetime.fromisoformat(when).replace(tzinfo=IST).timestamp())


class Archive:
    """A tiny memory DB: a DM with Rohan, a DM with Maa and a group, around 2026-09-20."""

    def __init__(self, path):
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.executescript(SCHEMA)
        self.conn.executemany("INSERT INTO Users VALUES (?,?,?,?)", [
            (1, "whatsapp", "Owner", "Owner"), (2, "whatsapp", "Rohan", "Rohan"), (3, "whatsapp", "Maa", "Maa"),
            (4, "whatsapp", "Nisha", "Nisha"), (5, "chatgpt", "ChatGPT", "ChatGPT")])
        self.conn.executemany("INSERT INTO Threads VALUES (?,?,?,?)", [
            (10, "whatsapp", "rohan", "DM Rohan"), (11, "whatsapp", "maa", "DM Maa"), (12, "whatsapp", "grp", "Flat group"),
            (13, "chatgpt", "c1", "Study plan")])
        self.k = 0
        self.add(10, 2, "2026-09-10 10:00", "kal 6 baje call karenge")                      # history (9/11)
        self.add(10, 1, "2026-09-10 10:05", "haan kal 6 baje call pakka")                    # same plan: merged
        self.add(10, 2, "2026-09-19 11:00", "exam on 23/09, bhool mat")                      # this week
        self.add(12, 4, "2026-09-19 12:00", "rent pay kar dena 5th oct ko")                  # later
        self.add(11, 3, "2026-04-07 08:00", "Happy birthday beta! Call when you are free.")  # your birthday, yearly
        self.add(12, 4, "2026-09-19 13:00", "going for chai now")                           # nothing
        self.add(13, 5, "2026-09-19 14:00", "Your exam is on 25/09, revise daily")           # assistant: skipped
        self.add(10, 2, "2026-09-20 09:00", "last update for now")                           # latest message
        self.conn.commit()

    def add(self, thread, author, when, text):
        self.k += 1
        self.conn.execute("INSERT INTO Messages VALUES (?,?,?,?,?,NULL)", (f"m{self.k:03d}", thread, author, ist(when), text))
        self.conn.commit()


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        self.archive = Archive(d / "memory.db")
        self.svc = reminders.RemindersService(d / "memory.db", OWNER, d / "store" / "rem.db", now="latest_message",
                                              lookback_days=None)

    def tearDown(self):
        self.archive.conn.close()
        self.tmp.cleanup()

    def titles(self, group, data=None):
        data = data or self.svc.list()
        return [i["title"] for i in data["groups"][group]]

    def test_scan_groups_and_dedupes(self):
        first = self.svc.scan()
        self.assertEqual(first["messages_read"], 8)
        data = self.svc.list()
        self.assertEqual(data["today"], "2026-09-20")
        self.assertEqual(self.titles("week", data), ["Exam on 23/09, bhool mat"])
        self.assertEqual(self.titles("later", data), ["Rent pay kar dena 5th oct ko", "Your birthday"])
        history = data["groups"]["history"]
        self.assertEqual(len(history), 1)            # both "kal 6 baje call" lines: one reminder, two mentions
        self.assertEqual((history[0]["mentions"], history[0]["time"], history[0]["node_id"]), (2, "18:00", "T_10"))
        bday = data["groups"]["later"][1]
        self.assertEqual((bday["repeat"], bday["date"], bday["said_by"]), ("yearly", "2027-04-07", "Maa"))
        self.assertNotIn("25/09", str(data))         # ChatGPT's own words are not your plans

    def test_incremental_scan_reads_only_new_messages(self):
        self.svc.scan()
        self.assertEqual(self.svc.scan()["messages_read"], 0)
        self.archive.add(12, 1, "2026-09-20 10:00", "I will send the deposit by Friday")
        again = self.svc.scan()
        self.assertEqual((again["messages_read"], again["reminders_found"]), (1, 1))
        self.assertIn("I will send the deposit by Friday", self.titles("week"))

    def test_actions_survive_a_rescan(self):
        self.svc.scan()
        exam = self.svc.list()["groups"]["week"][0]
        self.svc.update(exam["id"], "done")
        self.svc.update(exam["id"], "edit", title="Maths exam")
        self.svc.scan(full=True)
        data = self.svc.list()
        self.assertEqual(self.titles("done", data), ["Maths exam"])
        rent = data["groups"]["later"][0]
        self.svc.update(rent["id"], "dismiss")
        self.assertNotIn("Rent pay kar dena 5th oct ko", str(self.svc.list()))
        with self.assertRaises(reminders.BadRequest):
            self.svc.update(rent["id"], "snooze", until=ist("2026-09-01 10:00"))
        with self.assertRaises(reminders.ReminderNotFound):
            self.svc.update(999, "done")

    def test_snooze_moves_it(self):
        self.svc.scan()
        exam = self.svc.list()["groups"]["week"][0]
        snoozed = self.svc.update(exam["id"], "snooze", until=ist("2026-10-10 09:00"))
        self.assertEqual((snoozed["date"], snoozed["time"]), ("2026-10-10", "09:00"))
        self.assertIn(exam["id"], [i["id"] for i in self.svc.list()["groups"]["later"]])

    def test_manual_reminder_and_ics(self):
        self.svc.scan()
        added = self.svc.add("Pay electricity bill", ist("2026-09-22 10:00"))
        self.assertEqual((added["source"], added["time"]), ("manual", "10:00"))
        ics = self.svc.to_ics()
        self.assertTrue(ics.startswith("BEGIN:VCALENDAR\r\n") and ics.endswith("END:VCALENDAR\r\n"))
        self.assertIn("SUMMARY:Pay electricity bill", ics)
        self.assertIn("DTSTART:20260922T043000Z", ics)                # 10:00 IST
        self.assertIn("DTSTART;VALUE=DATE:20260923", ics)             # the exam, all day
        self.assertIn("RRULE:FREQ=YEARLY", ics)
        self.assertNotIn("call karenge", ics)                         # past reminders stay out of the calendar
        self.assertTrue(all(len(line.encode()) <= 75 for line in ics.split("\r\n")))

    def test_google_link(self):
        link = self.svc.google_link("Exam", ist("2026-09-23 00:00"), True)
        self.assertIn("dates=20260923%2F20260924", link)

    def test_missing_database(self):
        svc = reminders.RemindersService(Path(self.tmp.name) / "none.db", OWNER, Path(self.tmp.name) / "s.db")
        with self.assertRaises(reminders.RemindersUnavailable):
            svc.scan()


@unittest.skipUnless(HAVE_FASTAPI, "fastapi not installed")
class ApiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        self.archive = Archive(d / "memory.db")

        def space(wid, password=None):
            svc = reminders.RemindersService(d / "memory.db", OWNER, d / f"{wid}.db", now="latest_message", lookback_days=None)
            return workspaces.Workspace(wid, wid, None, None, None, d, password, reminders=svc)

        self.spaces = workspaces.WorkspaceSet({"demo": space("demo"), "me": space("me", workspaces.hash_password("pw"))}, "demo")
        self.client = TestClient(server.create_app(spaces=self.spaces, llm_profiles={}))

    def tearDown(self):
        self.archive.conn.close()
        self.tmp.cleanup()

    def test_list_and_read_only_public_workspace(self):
        r = self.client.get("/api/reminders")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertFalse(body["writable"])
        self.assertEqual(body["counts"]["week"], 1)
        rid = body["groups"]["week"][0]["id"]
        r = self.client.post(f"/api/reminders/{rid}", json={"action": "done"})
        self.assertEqual((r.status_code, r.json()["error"]["code"]), (403, "read_only"))

    def test_unlocked_workspace_can_change_and_subscribe(self):
        self.assertEqual(self.client.post("/api/workspace/unlock", json={"workspace": "me", "password": "pw"}).status_code, 200)
        body = self.client.get("/api/reminders").json()
        self.assertTrue(body["writable"])
        rid = body["groups"]["week"][0]["id"]
        r = self.client.post(f"/api/reminders/{rid}", json={"action": "snooze", "until": "2026-10-10T03:30:00Z"})
        self.assertEqual((r.status_code, r.json()["time"]), (200, "09:00"))
        r = self.client.post("/api/reminders", json={"title": "Call the landlord", "due": "2026-09-25", "all_day": True})
        self.assertEqual((r.status_code, r.json()["date"], r.json()["all_day"]), (200, "2026-09-25", True))
        self.assertEqual(self.client.post(f"/api/reminders/{rid}", json={"action": "fly"}).status_code, 422)

        path = self.client.get("/api/reminders/feed").json()["path"]
        anon = TestClient(self.client.app)                          # a calendar app: no cookie
        r = anon.get(path)
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.headers["content-type"].startswith("text/calendar"))
        self.assertIn("Call the landlord", r.text)
        self.assertEqual(anon.get("/api/reminders.ics?feed=wrong").status_code, 404)
        self.assertNotIn("Call the landlord", anon.get("/api/reminders.ics").text)   # the demo's calendar

    def test_ics_download(self):
        r = self.client.get("/api/reminders.ics")
        self.assertEqual(r.status_code, 200)
        self.assertIn("attachment", r.headers["content-disposition"])
        self.assertIn("BEGIN:VEVENT", r.text)


if __name__ == "__main__":
    unittest.main()
