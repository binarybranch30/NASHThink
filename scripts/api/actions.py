"""Agent actions: every change an AI agent wants to make waits for a person to approve it, and everything an agent
reads or does is written to an audit log.

An agent (the MCP server in scripts/mcp/, or any client that sends the X-NashThink-Agent header) can read the
memory freely, but it can only *propose* a change: add a reminder, mark one done or snoozed, sync Google Calendar.
A proposal waits in the queue as "pending" until someone approves or rejects it in the app (Actions panel, `A`).
Approving runs it; the result, or the error, is stored with it. Nothing an agent sends can approve anything.

One store per workspace: processed_data/actions/<workspace>.db (owner-only, like the chats' database; the audit
log records what an agent asked for, which can quote private chats). The memory database is never written.

    queue = ActionQueue(path)
    a = queue.propose("add_reminder", {"title": "Call Rohan", "due": "2026-10-02T18:00"}, reason="...", source="mcp")
    queue.decide(a["id"], approve=True, run=lambda tool, args: {...})
"""
import json
import os
import sqlite3
import threading
import time
from contextlib import closing
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = Path(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
STORE_DIR = REPO_ROOT / "processed_data" / "actions"

AGENT_HEADER = "X-NashThink-Agent"      # names the agent making a request ("mcp"); its requests are audited
TOOL_HEADER = "X-NashThink-Tool"        # the agent tool behind a request, for the audit log
CONFIRM_HEADER = "X-NashThink-Confirm"  # the app sends it with a decision; other web pages can't (CORS preflight)
ARGS_CHARS = 600                        # how much of a tool's arguments the audit log keeps
REASON_CHARS = 300
PAGE_MAX = 200

STATUSES = ("pending", "running", "executed", "failed", "rejected")

# The changes an agent can propose, what each needs, and how the app describes it.
TOOLS = {
    "add_reminder": {"required": ("title", "due"), "optional": ("all_day", "kind"),
                     "label": "Add a reminder"},
    "update_reminder": {"required": ("reminder_id", "action"), "optional": ("until",),
                        "label": "Change a reminder"},
    "calendar_sync": {"required": (), "optional": (), "label": "Sync reminders to Google Calendar"},
}
REMINDER_ACTIONS = ("done", "dismiss", "snooze", "open")

SCHEMA = """
CREATE TABLE IF NOT EXISTS actions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tool TEXT NOT NULL,
    args TEXT NOT NULL,
    summary TEXT NOT NULL,
    reason TEXT,
    source TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    result TEXT,
    created_at INTEGER NOT NULL,
    decided_at INTEGER
);
CREATE INDEX IF NOT EXISTS actions_status ON actions(status, created_at);
CREATE TABLE IF NOT EXISTS audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    at INTEGER NOT NULL,
    actor TEXT NOT NULL,
    event TEXT NOT NULL,
    tool TEXT,
    detail TEXT,
    action_id INTEGER,
    status INTEGER
);
CREATE INDEX IF NOT EXISTS audit_at ON audit(at);
"""


class BadAction(ValueError):
    pass


class ActionNotFound(LookupError):
    pass


class AlreadyDecided(Exception):
    pass


def _clip(text, limit):
    text = text if isinstance(text, str) else json.dumps(text, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def clean_args(tool, args):
    """The arguments of a proposal, checked: only the fields the tool knows, required ones present."""
    spec = TOOLS.get(tool)
    if spec is None:
        raise BadAction(f"unknown action {tool!r}; one of: {', '.join(TOOLS)}")
    args = args if isinstance(args, dict) else {}
    out = {k: args[k] for k in (*spec["required"], *spec["optional"]) if args.get(k) not in (None, "")}
    missing = [k for k in spec["required"] if k not in out]
    if missing:
        raise BadAction(f"{tool}: missing {', '.join(missing)}")
    if tool == "add_reminder":
        out["title"] = str(out["title"]).strip()[:200]
        if not out["title"]:
            raise BadAction("add_reminder: title can't be empty")
        out["due"] = str(out["due"]).strip()[:40]
        out["all_day"] = bool(out.get("all_day", False))
    if tool == "update_reminder":
        try:
            out["reminder_id"] = int(out["reminder_id"])
        except (TypeError, ValueError):
            raise BadAction("update_reminder: reminder_id must be a number")
        if out["action"] not in REMINDER_ACTIONS:
            raise BadAction(f"update_reminder: action must be one of {', '.join(REMINDER_ACTIONS)}")
        if out["action"] == "snooze" and not out.get("until"):
            raise BadAction("update_reminder: snooze needs 'until'")
    return out


def summarize(tool, args):
    """One line a person can approve or reject without reading JSON."""
    if tool == "add_reminder":
        when = args["due"] if not args.get("all_day") else args["due"][:10] + " (all day)"
        return f"Add reminder “{args['title']}” for {when}"
    if tool == "update_reminder":
        verb = {"done": "Mark reminder #{id} done", "dismiss": "Dismiss reminder #{id}",
                "open": "Reopen reminder #{id}", "snooze": "Snooze reminder #{id} until {until}"}[args["action"]]
        return verb.format(id=args["reminder_id"], until=args.get("until", ""))
    return TOOLS[tool]["label"]


class ActionQueue:
    """Pending agent proposals and the audit log for one workspace."""

    def __init__(self, store_path, clock=time.time):
        self.store_path = Path(store_path)
        self.clock = clock
        self._lock = threading.Lock()

    def _store(self):
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.store_path.exists():
            os.close(os.open(self.store_path, os.O_WRONLY | os.O_CREAT, 0o600))
        conn = sqlite3.connect(self.store_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.executescript(SCHEMA)
        return conn

    @staticmethod
    def _item(row):
        return {"id": row["id"], "tool": row["tool"], "label": TOOLS.get(row["tool"], {}).get("label", row["tool"]),
                "args": json.loads(row["args"]), "summary": row["summary"], "reason": row["reason"],
                "source": row["source"], "status": row["status"],
                "result": json.loads(row["result"]) if row["result"] else None,
                "created_at": row["created_at"], "decided_at": row["decided_at"]}

    # ── audit ──
    def log(self, actor, event, tool=None, detail=None, action_id=None, status=None, conn=None):
        row = (int(self.clock()), _clip(actor or "unknown", 40), event, tool,
               _clip(detail, ARGS_CHARS) if detail not in (None, "", {}) else None, action_id, status)
        sql = "INSERT INTO audit (at, actor, event, tool, detail, action_id, status) VALUES (?, ?, ?, ?, ?, ?, ?)"
        if conn is not None:
            conn.execute(sql, row)
            return
        with self._lock, closing(self._store()) as store:
            store.execute(sql, row)
            store.commit()

    def audit(self, limit=100, before=None):
        limit = max(1, min(int(limit), PAGE_MAX))
        with closing(self._store()) as store:
            rows = store.execute(
                "SELECT * FROM audit WHERE (? IS NULL OR id < ?) ORDER BY id DESC LIMIT ?", (before, before, limit)).fetchall()
        return [dict(r) for r in rows]

    # ── proposals ──
    def propose(self, tool, args, reason=None, source="agent"):
        args = clean_args(tool, args)
        summary = summarize(tool, args)
        reason = _clip(reason.strip(), REASON_CHARS) if isinstance(reason, str) and reason.strip() else None
        now = int(self.clock())
        with self._lock, closing(self._store()) as store:
            cur = store.execute(
                "INSERT INTO actions (tool, args, summary, reason, source, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (tool, json.dumps(args, ensure_ascii=False), summary, reason, _clip(source, 40), now))
            self.log(source, "proposed", tool, summary, cur.lastrowid, conn=store)
            store.commit()
            rid = cur.lastrowid
        return self.get(rid)

    def get(self, action_id):
        with closing(self._store()) as store:
            row = store.execute("SELECT * FROM actions WHERE id = ?", (int(action_id),)).fetchone()
        if row is None:
            raise ActionNotFound(f"No action #{action_id}.")
        return self._item(row)

    def list(self, status=None, limit=50):
        if status is not None and status not in STATUSES:
            raise BadAction(f"status must be one of {', '.join(STATUSES)}")
        limit = max(1, min(int(limit), PAGE_MAX))
        with closing(self._store()) as store:
            rows = store.execute(
                "SELECT * FROM actions WHERE (? IS NULL OR status = ?) "
                "ORDER BY CASE status WHEN 'pending' THEN 0 ELSE 1 END, id DESC LIMIT ?", (status, status, limit)).fetchall()
            counts = dict(store.execute("SELECT status, COUNT(*) FROM actions GROUP BY status").fetchall())
        return {"actions": [self._item(r) for r in rows], "counts": {s: counts.get(s, 0) for s in STATUSES}}

    def decide(self, action_id, approve, run=None, actor="you"):
        """Approve (runs `run(tool, args)` and stores what it returns or raises) or reject a pending action."""
        with self._lock, closing(self._store()) as store:
            row = store.execute("SELECT * FROM actions WHERE id = ?", (int(action_id),)).fetchone()
            if row is None:
                raise ActionNotFound(f"No action #{action_id}.")
            if row["status"] != "pending":
                raise AlreadyDecided(f"Action #{action_id} is already {row['status']}.")
            # Claim it before running, so two clicks can't run it twice.
            store.execute("UPDATE actions SET status = 'running', decided_at = ? WHERE id = ?", (int(self.clock()), row["id"]))
            store.commit()
        tool, args = row["tool"], json.loads(row["args"])
        if not approve:
            status, result = "rejected", None
        else:
            try:
                status, result = "executed", (run(tool, args) if run else None)
            except Exception as e:      # the action failed: keep the reason with it, never a half-state
                status, result = "failed", {"error": getattr(e, "message", None) or str(e) or type(e).__name__}
        with self._lock, closing(self._store()) as store:
            store.execute("UPDATE actions SET status = ?, result = ? WHERE id = ?",
                          (status, json.dumps(result, ensure_ascii=False, default=str) if result is not None else None, row["id"]))
            self.log(actor, "approved" if approve else "rejected", tool, row["summary"], row["id"], conn=store)
            if approve:
                self.log("nashthink", status, tool, result, row["id"], conn=store)
            store.commit()
        return self.get(row["id"])
