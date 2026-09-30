"""Tests for workspaces (scripts/api/workspaces.py and the server routes): the default workspace for everyone, the
password unlock, signed cookies, lockout, and that every data route follows the active workspace.
Stub services stand in for the index and databases. Needs fastapi + httpx (.venv).
Run: .venv/bin/python scripts/tests/test_workspaces.py"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.append(os.path.join(REPO_ROOT, "scripts", "semantic"))
sys.path.append(os.path.join(REPO_ROOT, "scripts", "api"))

import workspaces as wsp  # noqa: E402

try:
    from fastapi.testclient import TestClient
    import server
    HAVE_FASTAPI = True
except ImportError:
    HAVE_FASTAPI = False

PASSWORD = "correct horse"


class StubSearch:
    table_name = "topics"

    def __init__(self, name):
        self.name = name

    def status(self):
        return {"available": True, "which": self.name}

    def thread_context(self, channel_id):
        return []


class StubInsights:
    def __init__(self, name):
        self.name = name

    def available(self):
        return True

    def insights(self, *a):
        return {"which": self.name}

    def owner(self):
        return (self.name, set())


class StubPeople:
    def __init__(self, name):
        self.name = name

    def profile(self, node_id, *a):
        return {"which": self.name, "node": node_id}


def make_space(root, name, password=None):
    graph = Path(root) / name
    graph.mkdir()
    (graph / "cosmograph_nodes.csv").write_text(f"id,label\nU_1,{name}\n", encoding="utf-8")
    return wsp.Workspace(name, name.title(), StubSearch(name), StubInsights(name), StubPeople(name), graph,
                         wsp.hash_password(password, iterations=1000) if password else None)


class PasswordTests(unittest.TestCase):
    def test_hash_and_verify(self):
        h = wsp.hash_password("naitik", iterations=1000)
        self.assertNotIn("naitik", h)
        self.assertTrue(wsp.verify_password("naitik", h))
        self.assertFalse(wsp.verify_password("naitik ", h))
        self.assertFalse(wsp.verify_password("naitik", "garbage"))
        self.assertNotEqual(h, wsp.hash_password("naitik", iterations=1000), "salted")

    def test_config_validation(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "w.json"
            self.assertIsNone(wsp.load_config(p))
            p.write_text(json.dumps({"default": "demo", "workspaces": {"demo": {"root": d}, "me": {"root": d}}}))
            with self.assertRaises(wsp.ConfigError):   # a non-default workspace must have a password
                wsp.load_config(p)
            p.write_text(json.dumps({"default": "x", "workspaces": {"demo": {"root": d}}}))
            with self.assertRaises(wsp.ConfigError):
                wsp.load_config(p)
            p.write_text(json.dumps({"default": "demo", "workspaces": {"demo": {"root": d}, "me": {"root": d, "password": "h"}}}))
            self.assertEqual(wsp.load_config(p)["default"], "demo")


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.spaces = wsp.WorkspaceSet({"demo": make_space(self.tmp.name, "demo"),
                                        "personal": make_space(self.tmp.name, "personal", PASSWORD)}, "demo")

    def test_tokens(self):
        t = self.spaces.issue("personal", now=1000)
        self.assertEqual(self.spaces.read(t, now=1001), "personal")
        self.assertIsNone(self.spaces.read(t, now=1000 + self.spaces.ttl_s + 1), "expired")
        self.assertIsNone(self.spaces.read(t.replace("personal", "demo"), now=1001), "tampered")
        self.assertIsNone(self.spaces.read(t[:-1] + ("0" if t[-1] != "0" else "1"), now=1001))
        self.assertIsNone(self.spaces.read("", now=1001))
        other = wsp.WorkspaceSet(self.spaces.spaces, "demo")   # a restarted server has a new secret
        self.assertIsNone(other.read(t, now=1001))

    def test_lockout(self):
        for _ in range(wsp.MAX_FAILURES):
            self.assertFalse(self.spaces.unlock("personal", "nope", now=50))
        with self.assertRaises(wsp.TooManyAttempts):
            self.spaces.unlock("personal", PASSWORD, now=51)
        self.assertTrue(self.spaces.unlock("personal", PASSWORD, now=50 + wsp.LOCKOUT_S + 1))


@unittest.skipUnless(HAVE_FASTAPI, "fastapi not installed (.venv/bin/pip install fastapi uvicorn)")
class RouteTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        html = Path(self.tmp.name) / "g.html"
        html.write_text("<html></html>")
        self.spaces = wsp.WorkspaceSet({"demo": make_space(self.tmp.name, "demo"),
                                        "personal": make_space(self.tmp.name, "personal", PASSWORD)}, "demo")
        self.c = TestClient(server.create_app(graph_html=html, spaces=self.spaces, llm_profiles={}))

    def which(self):
        c = self.c
        return {
            "insights": c.get("/api/insights").json()["which"],
            "person": c.get("/api/person/U_1").json()["which"],
            "health": c.get("/api/health").json()["workspace"]["id"],
            "graph": c.get("/processed_data/graph/cosmograph_nodes.csv").text.strip().split(",")[-1],
            "active": c.get("/api/workspace").json()["active"],
        }

    def test_default_is_the_sample(self):
        self.assertEqual(set(self.which().values()), {"demo"})
        w = self.c.get("/api/workspace").json()
        self.assertTrue(w["switchable"])
        self.assertEqual([(x["id"], x["locked"]) for x in w["workspaces"]], [("demo", False), ("personal", True)])

    def test_wrong_password_stays_on_sample(self):
        r = self.c.post("/api/workspace/unlock", json={"workspace": "personal", "password": "nope"})
        self.assertEqual((r.status_code, r.json()["error"]["code"]), (401, "wrong_password"))
        self.assertNotIn(wsp.COOKIE, r.cookies)
        self.assertEqual(set(self.which().values()), {"demo"})

    def test_unlock_then_lock(self):
        r = self.c.post("/api/workspace/unlock", json={"workspace": "personal", "password": PASSWORD})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["active"], "personal")
        self.assertIn("httponly", r.headers["set-cookie"].lower())
        self.assertIn("samesite=strict", r.headers["set-cookie"].lower())
        self.assertNotIn(PASSWORD, r.text + r.headers["set-cookie"])
        self.assertEqual(set(self.which().values()), {"personal"})
        r = self.c.post("/api/workspace/lock")
        self.assertEqual(r.json()["active"], "demo")
        self.assertEqual(set(self.which().values()), {"demo"})

    def test_forged_cookie_is_ignored(self):
        self.c.cookies.set(wsp.COOKIE, "personal.99999999999.deadbeef")
        self.assertEqual(set(self.which().values()), {"demo"})

    def test_query_parameter_cannot_pick_a_workspace(self):
        for path in ("/api/health", "/api/upload/status", "/api/reminders"):
            r = self.c.get(path + "?ws=personal")
            self.assertNotIn('"personal"', r.text, path)
        self.assertEqual(self.c.get("/api/health?ws=personal").json()["workspace"]["id"], "demo")

    def test_lockout_and_unknown_workspace(self):
        self.assertEqual(self.c.post("/api/workspace/unlock", json={"workspace": "nope", "password": "x"}).status_code, 404)
        for _ in range(wsp.MAX_FAILURES):
            self.c.post("/api/workspace/unlock", json={"workspace": "personal", "password": "bad"})
        r = self.c.post("/api/workspace/unlock", json={"workspace": "personal", "password": PASSWORD})
        self.assertEqual((r.status_code, r.json()["error"]["code"]), (429, "too_many_attempts"))

    def test_examples_follow_the_workspace(self):
        self.spaces.spaces["demo"].examples = {"ask": ["Sample question?"]}
        self.assertEqual(self.c.get("/api/workspace").json()["examples"], {"ask": ["Sample question?"]})
        self.c.post("/api/workspace/unlock", json={"workspace": "personal", "password": PASSWORD})
        self.assertIsNone(self.c.get("/api/workspace").json()["examples"])
        self.assertEqual(server.clean_examples({"ask": [" q ", 3, ""], "search": "x"}), {"ask": ["q"]})
        self.assertIsNone(server.clean_examples(["q"]))

    def test_build_spaces_from_config(self):
        root = Path(self.tmp.name)
        config = {"default": "demo", "workspaces": {"demo": {"root": str(root / "a")},
                                                     "personal": {"root": str(root / "b"), "password": wsp.hash_password("pw", iterations=1000)}}}
        spaces = server.build_spaces(config)
        self.assertEqual(spaces.default, "demo")
        self.assertEqual(Path(spaces.spaces["personal"].graph_dir), root / "b" / "processed_data" / "graph")
        self.assertIs(spaces.spaces["demo"].search._lock, spaces.spaces["personal"].search._lock, "one model lock")
        self.assertFalse(spaces.spaces["demo"].search.status()["available"])


if __name__ == "__main__":
    unittest.main()
