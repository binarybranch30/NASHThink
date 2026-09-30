"""NASH Think as an MCP server: lets an AI agent (Claude Desktop, Claude Code, any MCP client) use your memory.

Read tools answer straight away. Write tools only *propose*: the change waits in the app's Actions panel until you
approve it, and every call lands in the audit log (see scripts/api/actions.py). The agent never gets an approve tool.

It talks to the running NASH Think server over HTTP (scripts/start_sarthink.sh), so the model and index load once.

    .venv/bin/python scripts/mcp/nashthink_mcp.py            # stdio transport

Environment:
    NASHTHINK_URL        server address (default http://127.0.0.1:8000)
    NASHTHINK_WORKSPACE  a password-protected workspace to use instead of the default (e.g. "personal")
    NASHTHINK_PASSWORD   its password
"""
import os
from typing import Optional

import httpx
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

AGENT = "mcp"
READ = ToolAnnotations(read_only_hint=True, open_world_hint=False)
PROPOSE = ToolAnnotations(read_only_hint=False, destructive_hint=False, idempotent_hint=False, open_world_hint=False)

INSTRUCTIONS = """NASH Think is the user's private memory of their chats (WhatsApp, Instagram, Discord, Gmail, ChatGPT...),
running on their own machine. Search or ask it before answering questions about the user's life, and cite the
sources it returns. Changes (reminders, calendar sync) are only proposals: tell the user to approve them in the
NASH Think app (Actions panel), and use check_action to see the outcome."""


class NashThink:
    """A small HTTP client for the NASH Think API that tags every request as coming from this agent."""

    def __init__(self, http=None, workspace=None, password=None):
        self.http = http or httpx.Client(base_url=os.environ.get("NASHTHINK_URL", "http://127.0.0.1:8000"), timeout=120)
        self.workspace = workspace if workspace is not None else os.environ.get("NASHTHINK_WORKSPACE")
        self.password = password if password is not None else os.environ.get("NASHTHINK_PASSWORD")

    def _ensure_workspace(self):
        """The server forgets sessions on restart and would quietly fall back to the sample data: check each time."""
        if not self.workspace:
            return
        if self.http.get("/api/workspace").json().get("active") == self.workspace:
            return
        r = self.http.post("/api/workspace/unlock", json={"workspace": self.workspace, "password": self.password or ""})
        if r.status_code != 200:
            raise ToolError(f"Could not open workspace {self.workspace!r}: {self._error(r)}")

    @staticmethod
    def _error(r):
        try:
            return r.json()["error"]["message"]
        except Exception:
            return f"HTTP {r.status_code}"

    def call(self, tool, method, path, **kwargs):
        self._ensure_workspace()
        try:
            r = self.http.request(method, path, headers={"X-NashThink-Agent": AGENT, "X-NashThink-Tool": tool}, **kwargs)
        except httpx.HTTPError as e:
            raise ToolError(f"NASH Think is not reachable ({e}). Start it with scripts/start_sarthink.sh.")
        if r.status_code >= 400:
            raise ToolError(self._error(r))
        return r.json()


def _source(s):
    return {k: s.get(k) for k in ("rank", "title", "platform", "date_start", "date_end", "node_id", "people", "snippet")}


def build_server(api=None):
    api = api or NashThink()
    mcp = MCPServer("NASH Think", instructions=INSTRUCTIONS)

    @mcp.tool(annotations=READ)
    def search_memories(query: str, limit: int = 8) -> dict:
        """Search the user's chats by meaning (English or Hinglish). Returns conversation excerpts with dates and people."""
        out = api.call("search_memories", "POST", "/api/search", json={"query": query, "limit": max(1, min(limit, 20))})
        return {"count": out["count"], "results": [
            {k: r.get(k) for k in ("rank", "title", "platform", "start_time", "end_time", "node_id", "people", "snippet")}
            for r in out["results"]]}

    @mcp.tool(annotations=READ)
    def ask_memory(question: str, platforms: Optional[list[str]] = None, date_from: Optional[str] = None,
                   date_to: Optional[str] = None) -> dict:
        """Answer a question from the user's chats with evidence: an answer, confidence, key points and numbered
        sources. Weak evidence gives "No relevant info found". Dates are YYYY-MM-DD."""
        body = {"question": question, "platforms": platforms, "date_from": date_from, "date_to": date_to}
        out = api.call("ask_memory", "POST", "/api/ask", json={k: v for k, v in body.items() if v})
        return {k: out[k] for k in ("answer", "confidence", "summary_points", "notes")} | {
            "sources": [_source(s) for s in out["sources"] if s.get("relevant")]}

    @mcp.tool(annotations=READ)
    def get_conversation(node_id: str) -> dict:
        """Excerpts of one conversation, by the node_id (T_<number>) from search or ask results."""
        return api.call("get_conversation", "GET", f"/api/thread/{node_id}")

    @mcp.tool(annotations=READ)
    def get_person(person_id: str) -> dict:
        """The user's history with one person (U_<number> from results): counts, dates, notable conversations."""
        return api.call("get_person", "GET", f"/api/person/{person_id}")

    @mcp.tool(annotations=READ)
    def list_reminders() -> dict:
        """Plans, deadlines and promises found in the chats, grouped (overdue, today, week, later, history, done)."""
        out = api.call("list_reminders", "GET", "/api/reminders")
        keep = ("id", "title", "kind", "status", "due", "all_day", "conversation", "said_by", "excerpt")
        return {"today": out["today"], "counts": out["counts"],
                "groups": {g: [{k: r.get(k) for k in keep} for r in items[:25]] for g, items in out["groups"].items()}}

    @mcp.tool(annotations=PROPOSE)
    def propose_reminder(title: str, due: str, all_day: bool = False, reason: str = "") -> dict:
        """Propose a new reminder (due: ISO date or datetime, e.g. 2026-10-02T18:00). It is NOT created until the
        user approves it in the NASH Think app. Give a short reason, e.g. the message that prompted it."""
        return api.call("propose_reminder", "POST", "/api/actions", json={
            "tool": "add_reminder", "args": {"title": title, "due": due, "all_day": all_day}, "reason": reason})

    @mcp.tool(annotations=PROPOSE)
    def propose_reminder_change(reminder_id: int, action: str, until: Optional[str] = None, reason: str = "") -> dict:
        """Propose marking a reminder done, dismiss, open, or snooze (needs until). Waits for the user's approval."""
        args = {"reminder_id": reminder_id, "action": action, **({"until": until} if until else {})}
        return api.call("propose_reminder_change", "POST", "/api/actions",
                        json={"tool": "update_reminder", "args": args, "reason": reason})

    @mcp.tool(annotations=PROPOSE)
    def propose_calendar_sync(reason: str = "") -> dict:
        """Propose syncing reminders to the user's Google Calendar. Waits for the user's approval."""
        return api.call("propose_calendar_sync", "POST", "/api/actions",
                        json={"tool": "calendar_sync", "args": {}, "reason": reason})

    @mcp.tool(annotations=READ)
    def check_action(action_id: int) -> dict:
        """Whether a proposal is still pending, was executed (with its result), failed or was rejected."""
        return api.call("check_action", "GET", f"/api/actions/{action_id}")

    return mcp


if __name__ == "__main__":
    build_server().run("stdio")
