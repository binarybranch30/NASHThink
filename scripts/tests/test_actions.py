"""Tests for agent actions (scripts/api/actions.py, the /api/actions routes) and the MCP server (scripts/mcp/),
against throwaway stores. No real archive, index or model is touched. Needs fastapi + httpx + mcp (.venv).
Run: .venv/bin/python scripts/tests/test_actions.py"""
import asyncio
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
for sub in (("scripts", "semantic"), ("scripts", "api"), ("scripts", "mcp"), ("scripts", "tests")):
    sys.path.append(os.path.join(REPO_ROOT, *sub))

import actions  # noqa: E402

try:
    from fastapi.testclient import TestClient
    import reminders
    import server
    import workspaces
    from test_reminders import OWNER, Archive
    HAVE_FASTAPI = True
except ImportError:
    HAVE_FASTAPI = False

try:
    from mcp import Client
    import nashthink_mcp
    HAVE_MCP = HAVE_FASTAPI
except ImportError:
    HAVE_MCP = False

AGENT = {actions.AGENT_HEADER: "mcp"}
CONFIRM = {actions.CONFIRM_HEADER: "1"}


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.q = actions.ActionQueue(Path(self.tmp.name) / "a" / "ws.db")

    def tearDown(self):
        self.tmp.cleanup()

    def test_propose_validates_and_summarizes(self):
        a = self.q.propose("add_reminder", {"title": " Call Rohan ", "due": "2026-10-02T18:00", "junk": 1}, "he asked", "mcp")
        self.assertEqual((a["status"], a["args"]["title"], a["source"]), ("pending", "Call Rohan", "mcp"))
        self.assertNotIn("junk", a["args"])
        self.assertIn("Call Rohan", a["summary"])
        self.assertEqual(oct(os.stat(self.q.store_path).st_mode & 0o777), "0o600")
        for tool, args in (("rm_rf", {}), ("add_reminder", {"title": "x"}), ("update_reminder", {"reminder_id": 1, "action": "fly"}),
                           ("update_reminder", {"reminder_id": 1, "action": "snooze"})):
            with self.assertRaises(actions.BadAction):
                self.q.propose(tool, args)

    def test_decide_runs_once_and_records_failures(self):
        a = self.q.propose("calendar_sync", {})
        ran = []
        done = self.q.decide(a["id"], True, lambda t, args: ran.append(t) or {"synced": 2})
        self.assertEqual((done["status"], done["result"], ran), ("executed", {"synced": 2}, ["calendar_sync"]))
        with self.assertRaises(actions.AlreadyDecided):
            self.q.decide(a["id"], True, lambda t, args: ran.append(t))
        self.assertEqual(len(ran), 1)
        b = self.q.propose("calendar_sync", {})
        self.assertEqual(self.q.decide(b["id"], True, lambda *a: 1 / 0)["status"], "failed")
        c = self.q.propose("calendar_sync", {})
        self.assertEqual(self.q.decide(c["id"], False)["status"], "rejected")
        self.assertEqual(self.q.list()["counts"], {"pending": 0, "running": 0, "executed": 1, "failed": 1, "rejected": 1})
        events = [e["event"] for e in self.q.audit()]
        self.assertEqual(events.count("proposed"), 3)
        self.assertIn("failed", events)


@unittest.skipUnless(HAVE_FASTAPI, "fastapi not installed")
class RouteTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        self.archive = Archive(d / "memory.db")

        def space(wid, password=None, agent_actions=False):
            svc = reminders.RemindersService(d / "memory.db", OWNER, d / f"{wid}.db", now="latest_message", lookback_days=None)
            return workspaces.Workspace(wid, wid, None, None, None, d, password, reminders=svc,
                                        actions=actions.ActionQueue(d / f"{wid}.actions.db"), agent_actions=agent_actions)

        self.spaces = workspaces.WorkspaceSet({"demo": space("demo"), "me": space("me", workspaces.hash_password("pw"))}, "demo")
        self.client = TestClient(server.create_app(spaces=self.spaces, llm_profiles={}))

    def tearDown(self):
        self.archive.conn.close()
        self.tmp.cleanup()

    def propose(self, title="Pay rent"):
        r = self.client.post("/api/actions", headers=AGENT,
                             json={"tool": "add_reminder", "args": {"title": title, "due": "2026-10-05"}, "reason": "from chat"})
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()["id"]

    def test_person_approves_and_it_runs(self):
        self.client.post("/api/workspace/unlock", json={"workspace": "me", "password": "pw"})
        aid = self.propose()
        self.assertEqual(self.client.get("/api/actions").json()["counts"]["pending"], 1)
        self.assertEqual(self.client.post(f"/api/actions/{aid}/approve", headers=AGENT | CONFIRM).status_code, 403)
        self.assertEqual(self.client.post(f"/api/actions/{aid}/approve").json()["error"]["code"], "confirm_required")
        r = self.client.post(f"/api/actions/{aid}/approve", headers=CONFIRM)
        self.assertEqual((r.status_code, r.json()["status"]), (200, "executed"), r.text)
        titles = [x["title"] for g in self.client.get("/api/reminders").json()["groups"].values() for x in g]
        self.assertIn("Pay rent", titles)
        self.assertEqual(self.client.post(f"/api/actions/{aid}/approve", headers=CONFIRM).status_code, 409)
        events = [(e["actor"], e["event"]) for e in self.client.get("/api/audit").json()["entries"]]
        self.assertIn(("mcp", "refused"), events)
        self.assertIn(("you", "approved"), events)

    def test_agent_reads_are_audited(self):
        self.client.get("/api/reminders", headers=AGENT | {actions.TOOL_HEADER: "list_reminders"})
        self.client.get("/api/reminders")        # the app itself: not audited
        entries = self.client.get("/api/audit").json()["entries"]
        self.assertEqual([(e["tool"], e["event"]) for e in entries], [("list_reminders", "read")])

    def test_sample_data_is_read_only_unless_allowed(self):
        aid = self.propose()
        r = self.client.post(f"/api/actions/{aid}/approve", headers=CONFIRM)
        self.assertEqual(r.json()["error"]["code"], "read_only")
        self.assertEqual(self.client.post(f"/api/actions/{aid}/reject", headers=CONFIRM).json()["status"], "rejected")
        self.spaces.spaces["demo"].agent_actions = True
        aid = self.propose("Demo reminder")
        self.assertEqual(self.client.post(f"/api/actions/{aid}/approve", headers=CONFIRM).json()["status"], "executed")


@unittest.skipUnless(HAVE_MCP, "mcp not installed")
class McpTests(RouteTests):
    def test_mcp_tools_propose_but_never_approve(self):
        api = nashthink_mcp.NashThink(http=self.client, workspace="me", password="pw")
        server_ = nashthink_mcp.build_server(api)

        async def go():
            async with Client(server_) as c:
                names = {t.name for t in (await c.list_tools()).tools}
                self.assertIn("propose_reminder", names)
                self.assertFalse([n for n in names if "approve" in n])
                r = await c.call_tool("propose_reminder", {"title": "Call Maa", "due": "2026-10-03T19:00", "reason": "she asked"})
                self.assertFalse(r.is_error, r)
                aid = json.loads(r.content[0].text)["id"]
                r = await c.call_tool("list_reminders", {})
                self.assertFalse(r.is_error, r)
                return aid

        aid = asyncio.run(go())
        self.assertEqual(self.client.get("/api/workspace").json()["active"], "me", "unlocked the configured workspace")
        a = self.client.get(f"/api/actions/{aid}").json()
        self.assertEqual((a["status"], a["source"]), ("pending", "mcp"))


if __name__ == "__main__":
    unittest.main()
