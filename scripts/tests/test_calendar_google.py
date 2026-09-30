"""Tests for Google Calendar sync (scripts/api/calendar_google.py + /api/calendar/...) against a fake Google.
No network: every Google call goes to an in-memory httpx.MockTransport. Needs fastapi + httpx (.venv).
Run: .venv/bin/python scripts/tests/test_calendar_google.py"""
import json
import os
import sys
import tempfile
import unittest
import urllib.parse
from pathlib import Path
from unittest import mock

import httpx

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.append(os.path.join(REPO_ROOT, "scripts", "semantic"))
sys.path.append(os.path.join(REPO_ROOT, "scripts", "api"))
sys.path.append(SCRIPT_DIR)

import calendar_google
import reminders
from test_reminders import OWNER, Archive

try:
    from fastapi.testclient import TestClient
    import server
    import workspaces
    HAVE_FASTAPI = True
except ImportError:
    HAVE_FASTAPI = False

CONFIG = ("client-123.apps.googleusercontent.com", "shh")


class FakeGoogle:
    """Just enough of Google's OAuth and Calendar v3 APIs."""

    def __init__(self):
        self.calendars = {}
        self.events = {}           # calendar id -> {event id: body}
        self.calls = []
        self.refresh_ok = True
        self.n = 0

    def handler(self, request):
        path, method = request.url.path, request.method
        self.calls.append((method, path))
        if request.url.host == "oauth2.googleapis.com":
            form = dict(urllib.parse.parse_qsl(request.content.decode()))
            if path == "/revoke":
                return httpx.Response(200)
            if form.get("grant_type") == "authorization_code":
                assert form["code_verifier"] and form["client_secret"] == "shh"
                return httpx.Response(200, json={"access_token": "at-1", "refresh_token": "rt-1", "expires_in": 3600})
            if form.get("grant_type") == "refresh_token":
                if not self.refresh_ok:
                    return httpx.Response(400, json={"error": "invalid_grant"})
                return httpx.Response(200, json={"access_token": "at-2", "expires_in": 3600})
        assert request.headers["authorization"].startswith("Bearer ")
        parts = path.split("/")[3:]          # after /calendar/v3
        if parts == ["calendars"] and method == "POST":
            self.n += 1
            cid = f"cal{self.n}@group.calendar.google.com"
            self.calendars[cid] = json.loads(request.content)
            self.events[cid] = {}
            return httpx.Response(200, json={"id": cid})
        cid = urllib.parse.unquote(parts[1])
        if cid not in self.calendars:
            return httpx.Response(404, json={"error": {"message": "Not Found"}})
        if len(parts) == 2 and method == "GET":
            return httpx.Response(200, json={"id": cid})
        events = self.events[cid]
        if len(parts) == 3 and method == "POST":
            self.n += 1
            eid = f"ev{self.n}"
            events[eid] = json.loads(request.content)
            return httpx.Response(200, json={"id": eid})
        eid = parts[3]
        if eid not in events:
            return httpx.Response(404, json={"error": {"message": "Not Found"}})
        if method == "PUT":
            events[eid] = json.loads(request.content)
            return httpx.Response(200, json={"id": eid})
        if method == "DELETE":
            del events[eid]
            return httpx.Response(204)
        return httpx.Response(400)

    def all_events(self):
        return [e for evs in self.events.values() for e in evs.values()]

    def writes(self):
        return [c for c in self.calls if c[0] in ("POST", "PUT", "DELETE") and "/calendar/v3/" in c[1]]


class SyncTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        self.archive = Archive(d / "memory.db")
        self.rem = reminders.RemindersService(d / "memory.db", OWNER, d / "rem.db", now="latest_message", lookback_days=None)
        self.google = FakeGoogle()
        self.cal = calendar_google.GoogleCalendarSync(self.rem, d / "rem.google.json", config=CONFIG,
                                                      transport=httpx.MockTransport(self.google.handler))

    def tearDown(self):
        self.archive.conn.close()
        self.tmp.cleanup()

    def connect(self):
        url = self.cal.auth_url("http://127.0.0.1:8096/api/calendar/google/callback")
        q = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query))
        self.assertEqual((q["scope"], q["code_challenge_method"], q["access_type"]), (calendar_google.SCOPE, "S256", "offline"))
        self.cal.finish(q["state"], "the-code")
        return q["state"]

    def test_connect_stores_a_private_token(self):
        self.assertFalse(self.cal.connected())
        state = self.connect()
        self.assertTrue(self.cal.connected())
        self.assertEqual(oct(os.stat(self.cal.token_path).st_mode & 0o777), "0o600")
        with self.assertRaises(calendar_google.CalendarError):     # a state works once
            self.cal.finish(state, "again")

    def test_sync_adds_then_only_changes(self):
        self.connect()
        first = self.cal.sync()
        self.assertEqual((first["added"], first["in_calendar"]), (3, 3))   # exam, rent, birthday (the old call has passed)
        cal = next(iter(self.google.calendars.values()))
        self.assertEqual((cal["summary"], cal["timeZone"]), ("NASH Think", "Asia/Kolkata"))
        by_title = {e["summary"]: e for e in self.google.all_events()}
        self.assertEqual(by_title["Exam on 23/09, bhool mat"]["start"], {"date": "2026-09-23"})
        self.assertEqual(by_title["Your birthday"]["recurrence"], ["RRULE:FREQ=YEARLY"])
        self.assertIn("nashthink_id", by_title["Rent pay kar dena 5th oct ko"]["extendedProperties"]["private"])
        self.assertTrue(all(it["synced"] for g in ("week", "later") for it in self.rem.list()["groups"][g]))

        writes = len(self.google.writes())
        self.assertEqual(self.cal.sync()["added"], 0)
        self.assertEqual(len(self.google.writes()), writes)         # nothing changed: no writes

        exam = self.rem.list()["groups"]["week"][0]
        self.rem.update(exam["id"], "edit", title="Maths exam")
        self.assertEqual(self.cal.sync()["updated"], 1)
        self.rem.update(exam["id"], "done")
        res = self.cal.sync()
        self.assertEqual((res["removed"], res["in_calendar"]), (1, 2))
        self.assertNotIn("Maths exam", [e["summary"] for e in self.google.all_events()])

    def test_deleted_calendar_is_recreated(self):
        self.connect()
        self.cal.sync()
        self.google.calendars.clear()
        self.google.events.clear()
        res = self.cal.sync()
        self.assertEqual(res["added"], 3)
        self.assertEqual(len(self.google.calendars), 1)

    def test_expired_access_is_refreshed_and_revoked_access_reported(self):
        self.connect()
        tok = json.loads(self.cal.token_path.read_text())
        tok["expires_at"] = 0
        self.cal.token_path.write_text(json.dumps(tok))
        self.cal.sync()
        self.assertEqual(json.loads(self.cal.token_path.read_text())["access_token"], "at-2")
        tok = json.loads(self.cal.token_path.read_text())
        tok["expires_at"] = 0
        self.cal.token_path.write_text(json.dumps(tok))
        self.google.refresh_ok = False
        with self.assertRaises(calendar_google.CalendarError) as cm:
            self.cal.sync()
        self.assertEqual(cm.exception.code, "reconnect")
        self.assertIn("Connect Google Calendar again", self.cal.status()["last_error"])

    def test_disconnect_forgets_everything(self):
        self.connect()
        self.cal.sync()
        self.cal.disconnect()
        self.assertFalse(self.cal.connected())
        self.assertEqual(self.rem.google_state(), {})
        self.assertIn(("POST", "/revoke"), self.google.calls)

    def test_not_configured(self):
        cal = calendar_google.GoogleCalendarSync(self.rem, Path(self.tmp.name) / "x.json", config=None)
        with mock.patch.object(calendar_google, "client_config", return_value=None), \
                self.assertRaises(calendar_google.CalendarError) as cm:
            cal.auth_url("http://127.0.0.1/cb")
        self.assertEqual(cm.exception.code, "not_configured")


@unittest.skipUnless(HAVE_FASTAPI, "fastapi not installed")
class ApiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        self.archive = Archive(d / "memory.db")
        self.google = FakeGoogle()

        def space(wid, password=None):
            rem = reminders.RemindersService(d / "memory.db", OWNER, d / f"{wid}.db", now="latest_message", lookback_days=None)
            cal = calendar_google.GoogleCalendarSync(rem, d / f"{wid}.google.json", config=CONFIG,
                                                     transport=httpx.MockTransport(self.google.handler))
            return workspaces.Workspace(wid, wid, None, None, None, d, password, reminders=rem, calendar=cal)

        self.spaces = workspaces.WorkspaceSet({"demo": space("demo"), "me": space("me", workspaces.hash_password("pw"))}, "demo")
        self.app = server.create_app(spaces=self.spaces, llm_profiles={})
        self.client = TestClient(self.app, base_url="http://127.0.0.1:8096", follow_redirects=False)

    def tearDown(self):
        self.archive.conn.close()
        self.tmp.cleanup()

    def test_sample_workspace_cannot_connect(self):
        self.assertFalse(self.client.get("/api/calendar/status").json()["allowed"])
        self.assertEqual(self.client.get("/api/calendar/google/connect").status_code, 403)

    def test_connect_callback_sync_flow(self):
        self.client.post("/api/workspace/unlock", json={"workspace": "me", "password": "pw"})
        st = self.client.get("/api/calendar/status").json()
        self.assertEqual((st["allowed"], st["configured"], st["connected"]), (True, True, False))
        r = self.client.get("/api/calendar/google/connect")
        self.assertEqual(r.status_code, 303)
        q = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(r.headers["location"]).query))
        self.assertEqual(q["redirect_uri"], "http://127.0.0.1:8096/api/calendar/google/callback")
        # Google sends the browser back without the (SameSite=Strict) cookie.
        anon = TestClient(self.app, base_url="http://127.0.0.1:8096", follow_redirects=False)
        r = anon.get("/api/calendar/google/callback", params={"state": q["state"], "code": "c"})
        self.assertEqual(r.headers["location"], "/?calendar=connected")
        self.assertEqual(anon.get("/api/calendar/google/callback", params={"state": q["state"], "code": "c"}).headers["location"],
                         "/?calendar=failed")
        self.assertTrue(self.client.get("/api/calendar/status").json()["connected"])
        res = self.client.post("/api/calendar/google/sync").json()
        self.assertEqual(res["added"], 3)
        self.assertFalse(self.spaces.spaces["demo"].calendar.connected())    # only the unlocked workspace
        self.assertFalse(self.client.post("/api/calendar/google/disconnect").json()["connected"])

    def test_connect_needs_loopback_address(self):
        self.client.post("/api/workspace/unlock", json={"workspace": "me", "password": "pw"})
        other = TestClient(self.app, base_url="http://my-vps.example.com", follow_redirects=False)
        cookie = f"{workspaces.COOKIE}={self.client.cookies.get(workspaces.COOKIE)}"
        r = other.get("/api/calendar/google/connect", headers={"Cookie": cookie})
        self.assertEqual(r.json()["error"]["code"], "not_loopback")

    def test_cancelled_sign_in(self):
        r = self.client.get("/api/calendar/google/callback", params={"error": "access_denied", "state": "x"})
        self.assertEqual(r.headers["location"], "/?calendar=cancelled")

    def test_worker_scans_and_syncs_connected_workspaces(self):
        me = self.spaces.spaces["me"]
        url = me.calendar.auth_url("http://127.0.0.1:8096/api/calendar/google/callback")
        me.calendar.finish(dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query))["state"], "c")
        server.ReminderWorker(self.spaces).run_once()
        self.assertEqual(len(self.google.all_events()), 3)
        self.assertEqual(me.calendar.status()["last_result"]["in_calendar"], 3)


if __name__ == "__main__":
    unittest.main()
