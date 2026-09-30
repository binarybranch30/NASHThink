"""Reminders found in the chats (/api/reminders), their state, and a calendar (.ics) of them.

A scan reads new messages from the memory database (read-only), runs the rule extractor
(scripts/semantic/reminder_rules.py) on each, and keeps one reminder per conversation per day in a small
writable store of its own (processed_data/reminders/<workspace>.db), never in the memory database. Scans are
incremental: a watermark remembers the last message read, so a re-scan only reads what arrived since.

The store also keeps what the user did with each reminder (done, dismissed, snoozed, edited) and the ones they
added by hand; a re-scan never undoes those.
"""
import datetime as dt
import json
import re
import secrets
import sqlite3
import sys
import threading
import time
import urllib.parse
from contextlib import closing
from pathlib import Path
from zoneinfo import ZoneInfo

import insights
from insights import TS_SQL, is_owner, person_label

sys.path.append(str(Path(__file__).resolve().parents[1] / "semantic"))
import reminder_rules  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
STORE_DIR = REPO_ROOT / "processed_data" / "reminders"
DEFAULT_LOOKBACK_DAYS = 60
RESCAN_AFTER_S = 600          # list() re-scans when the last scan is older than this
OVERDUE_DAYS = 7              # open reminders this recently past are "overdue"; older ones are history
SOON_S = 24 * 3600            # the badge counts overdue ones and those due within a day
EXCERPT_CHARS = 240
MAX_HISTORY = 200
BATCH = 5000
ASSISTANTS = {"chatgpt", "claude", "assistant", "gemini"}
KINDS = {"birthday", "exam", "interview", "deadline", "appointment", "travel", "payment", "meeting", "reminder",
         "plan", "task"}
ACTIONS = {"done", "open", "dismiss", "snooze", "edit"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS reminders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key TEXT UNIQUE,                -- dedupe key: thread + day for found ones, 'bday:<name>', 'manual:<token>'
    msg_id TEXT, thread_id INTEGER, platform TEXT, thread_title TEXT,
    said_by TEXT, from_owner INTEGER DEFAULT 0,
    title TEXT NOT NULL, excerpt TEXT,
    due_at INTEGER NOT NULL, all_day INTEGER NOT NULL DEFAULT 0, date_local TEXT,
    kind TEXT, confidence REAL, repeat TEXT,
    status TEXT NOT NULL DEFAULT 'open',          -- open | done | dismissed
    snoozed_until INTEGER, source TEXT NOT NULL DEFAULT 'auto',   -- auto | manual | ai
    edited INTEGER DEFAULT 0, mentions INTEGER DEFAULT 1,
    google_event_id TEXT, synced_at INTEGER,
    created_at INTEGER, updated_at INTEGER
);
CREATE INDEX IF NOT EXISTS idx_rem_due ON reminders(due_at);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""


class RemindersUnavailable(Exception):
    """The memory database is missing or unreadable."""


class ReminderNotFound(Exception):
    pass


class BadRequest(Exception):
    pass


def _iso(ts):
    return dt.datetime.fromtimestamp(int(ts), dt.timezone.utc).isoformat() if ts is not None else None


def _excerpt(text):
    t = re.sub(r"\s+", " ", text or "").strip()
    return t if len(t) <= EXCERPT_CHARS else t[:EXCERPT_CHARS - 1].rstrip() + "…"


def _ics_escape(text):
    return str(text or "").replace("\\", "\\\\").replace(";", r"\;").replace(",", r"\,").replace("\n", r"\n")


def _ics_fold(line):
    """RFC 5545: lines longer than 75 octets continue on the next line after a space."""
    raw = line.encode("utf-8")
    if len(raw) <= 75:
        return line
    out, cur = [], b""
    for ch in line:
        b = ch.encode("utf-8")
        if len(cur) + len(b) > (75 if not out else 74):
            out.append(cur.decode("utf-8"))
            cur = b""
        cur += b
    out.append(cur.decode("utf-8"))
    return "\r\n ".join(out)


class RemindersService:
    """Reminders for one workspace. `now`: "clock" (real time) or "latest_message" (the archive's last message,
    so a sample archive that ends in the past still has upcoming reminders)."""

    def __init__(self, db_path, identity_map, store_path, tz=reminder_rules.DEFAULT_TZ, now="clock",
                 lookback_days=DEFAULT_LOOKBACK_DAYS, clock=time.time):
        self.db_path = Path(db_path)
        self.identity_map = identity_map
        self.store_path = Path(store_path)
        self.tz = ZoneInfo(tz)
        self.tz_name = tz
        self.now_mode = now
        self.lookback_days = lookback_days
        self.clock = clock
        self._lock = threading.Lock()
        self._scan_lock = threading.Lock()
        self._latest = None

    # ── storage ──
    def _identity(self):
        if isinstance(self.identity_map, tuple):
            return self.identity_map
        return insights.load_identity(self.identity_map)

    def _memory(self):
        if not self.db_path.is_file():
            raise RemindersUnavailable(f"Memory database not found: {self.db_path.name}. Run the parsers to build it.")
        try:
            conn = sqlite3.connect(f"{self.db_path.resolve().as_uri()}?mode=ro", uri=True)
            conn.execute("SELECT 1 FROM Messages LIMIT 1").fetchall()
        except sqlite3.Error as e:
            raise RemindersUnavailable(f"Memory database is not readable: {e}")
        return conn

    def _store(self):
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.store_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.executescript(SCHEMA)
        return conn

    def _meta(self, conn, key, default=None):
        row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else default

    def _set_meta(self, conn, key, value):
        conn.execute("INSERT INTO meta(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                     (key, str(value)))

    # ── time ──
    def now(self):
        if self.now_mode == "latest_message":
            if self._latest is None:
                try:
                    with closing(self._memory()) as conn:
                        self._latest = conn.execute(f"SELECT MAX({TS_SQL}) FROM Messages m").fetchone()[0]
                except RemindersUnavailable:
                    self._latest = 0
            if self._latest:
                return int(self._latest)
        return int(self.clock())

    def today(self):
        return dt.datetime.fromtimestamp(self.now(), self.tz).date()

    # ── scanning ──
    def scan(self, full=False):
        """Read messages newer than the watermark and store the reminders they imply. Returns counts."""
        started = time.time()
        with self._scan_lock:
            master, aliases = self._identity()
            with closing(self._memory()) as mem, self._lock, closing(self._store()) as store:
                if full:
                    self._set_meta(store, "watermark_ts", "")
                    self._set_meta(store, "watermark_msg", "")
                wm_ts = self._meta(store, "watermark_ts") or None
                wm_msg = self._meta(store, "watermark_msg") or ""
                start = None
                if wm_ts is None and self.lookback_days:
                    start = self.now() - int(self.lookback_days) * 86400
                authors = self._thread_authors(mem)
                read = found = 0
                while True:
                    params, where = [], ["m.content IS NOT NULL", "m.content != ''"]
                    if wm_ts is not None:
                        where.append(f"(({TS_SQL}) > ? OR (({TS_SQL}) = ? AND m.msg_id > ?))")
                        params += [int(wm_ts), int(wm_ts), wm_msg]
                    elif start is not None:
                        where.append(f"({TS_SQL}) >= ?")
                        params.append(start)
                    rows = mem.execute(
                        f"SELECT m.msg_id, ({TS_SQL}) AS ts, m.content, m.thread_id, t.platform, t.title, "
                        f"u.platform, u.display_name, u.raw_id FROM Messages m JOIN Threads t ON t.id = m.thread_id "
                        f"LEFT JOIN Users u ON u.id = m.author_id WHERE {' AND '.join(where)} "
                        f"ORDER BY ts, m.msg_id LIMIT {BATCH}", params).fetchall()
                    if not rows:
                        break
                    for row in rows:
                        found += self._scan_row(store, row, aliases, authors)
                    read += len(rows)
                    wm_ts, wm_msg = rows[-1][1], rows[-1][0]
                    self._set_meta(store, "watermark_ts", wm_ts)
                    self._set_meta(store, "watermark_msg", wm_msg)
                    store.commit()
                self._set_meta(store, "scanned_at", int(time.time()))
                store.commit()
        self._latest = None
        return {"messages_read": read, "reminders_found": found, "took_ms": int((time.time() - started) * 1000)}

    @staticmethod
    def _thread_authors(mem):
        """thread id -> number of distinct authors: tells a one-to-one chat from a group."""
        return dict(mem.execute("SELECT thread_id, COUNT(DISTINCT author_id) FROM Messages GROUP BY thread_id").fetchall())

    def _scan_row(self, store, row, aliases, authors):
        msg_id, ts, text, thread_id, platform, title, u_platform, u_name, u_raw = row
        if not ts:
            return 0
        owner = is_owner(u_platform or platform, u_name, u_raw, aliases)
        said_by = "You" if owner else (person_label(u_platform or platform, u_name, u_raw) if (u_name or u_raw) else "")
        if not owner and platform in ("chatgpt", "claude") and str(u_name or "").lower() in ASSISTANTS:
            return 0
        birthday_of = self._birthday_of(text, owner, title, authors.get(thread_id, 0) <= 2, aliases)
        stored = 0
        for c in reminder_rules.extract(text, ts, self.tz, birthday_of=birthday_of):
            if c.kind == "birthday" and c.repeat:
                c.title = "Your birthday" if birthday_of == "Your" else c.title
                key = f"bday:{c.title.lower()}"
            else:
                key = f"t{thread_id}:{c.date_local}"
            stored += self._upsert(store, key, c, msg_id, thread_id, platform, title, said_by, owner, text)
        return stored

    @staticmethod
    def _birthday_of(text, owner, thread_title, one_to_one, aliases):
        """Whose birthday a wish in this message is for: "Your" when someone wishes you (in a one-to-one chat, or
        naming you), else the person you wish (the other side of a one-to-one chat, or the name after 'birthday')."""
        if not reminder_rules.WISH_RE.search(text or ""):
            return None
        words = {w.lower() for w in re.findall(r"[^\W\d_]{3,}", text)}
        if not owner:
            return "Your" if one_to_one or words & {a.lower() for a in aliases} else None
        m = re.search(r"\b(?:birthday|b'?day|bday|hbd)\W+([A-Z][\w.]+)", text)
        if m:
            return m.group(1).rstrip(".")
        if one_to_one and thread_title:
            return re.sub(r"^(?:DM|Chat with|WhatsApp Chat with)\s+", "", thread_title).strip() or None
        return None

    def _upsert(self, store, key, c, msg_id, thread_id, platform, thread_title, said_by, owner, text):
        now = int(time.time())
        row = store.execute("SELECT id, confidence, edited, status FROM reminders WHERE key = ?", (key,)).fetchone()
        fields = dict(msg_id=msg_id, thread_id=thread_id, platform=platform, thread_title=thread_title, said_by=said_by,
                      from_owner=int(bool(owner)), title=c.title, excerpt=_excerpt(text), due_at=c.due_at,
                      all_day=int(c.all_day), date_local=c.date_local, kind=c.kind, confidence=c.confidence,
                      repeat=c.repeat)
        if row is None:
            cols = ", ".join(fields) + ", key, status, source, created_at, updated_at"
            store.execute(f"INSERT INTO reminders ({cols}) VALUES ({', '.join('?' * (len(fields) + 5))})",
                          [*fields.values(), key, "open", "auto", now, now])
            return 1
        # Another mention of the same plan: count it; keep the clearest wording unless the user edited it.
        if c.confidence > (row["confidence"] or 0) and not row["edited"]:
            sets = ", ".join(f"{k} = ?" for k in fields)
            store.execute(f"UPDATE reminders SET {sets}, mentions = mentions + 1, updated_at = ? WHERE id = ?",
                          [*fields.values(), now, row["id"]])
        else:
            store.execute("UPDATE reminders SET mentions = mentions + 1 WHERE id = ?", (row["id"],))
        return 0

    def ensure_scanned(self):
        with closing(self._store()) as store:
            last = self._meta(store, "scanned_at")
        if not last or time.time() - int(last) > RESCAN_AFTER_S:
            self.scan()

    # ── reading ──
    def _next_occurrence(self, row):
        """Due time of a yearly reminder's next occurrence on or after today."""
        d = dt.date.fromisoformat(row["date_local"])
        today = self.today()
        for year in (today.year, today.year + 1):
            try:
                cand = d.replace(year=year)
            except ValueError:
                cand = dt.date(year, 2, 28)
            if cand >= today:
                return int(dt.datetime.combine(cand, dt.time(0, 0), self.tz).timestamp()), cand.isoformat()
        return row["due_at"], row["date_local"]

    def _item(self, row):
        due, date_local = row["due_at"], row["date_local"]
        if row["repeat"] == "yearly":
            due, date_local = self._next_occurrence(row)
        if row["snoozed_until"] and row["status"] == "open" and row["snoozed_until"] > due:
            due = row["snoozed_until"]
            date_local = dt.datetime.fromtimestamp(due, self.tz).date().isoformat()
            all_day = False
        else:
            all_day = bool(row["all_day"])
        local = dt.datetime.fromtimestamp(due, self.tz)
        return {
            "id": row["id"], "title": row["title"], "kind": row["kind"], "status": row["status"],
            "due": _iso(due), "due_ts": due, "date": date_local, "time": None if all_day else local.strftime("%H:%M"),
            "all_day": all_day, "repeat": row["repeat"], "snoozed_until": _iso(row["snoozed_until"]),
            "platform": row["platform"], "said_by": row["said_by"], "from_you": bool(row["from_owner"]),
            "conversation": row["thread_title"], "node_id": f"T_{row['thread_id']}" if row["thread_id"] else None,
            "msg_id": row["msg_id"], "excerpt": row["excerpt"], "confidence": row["confidence"],
            "source": row["source"], "mentions": row["mentions"], "synced": bool(row["google_event_id"]),
            "google_link": self.google_link(row["title"], due, all_day, row["excerpt"]),
        }

    def google_link(self, title, due, all_day, details=""):
        """An 'Add to Google Calendar' link: opens a filled-in event for the user to save (nothing is sent)."""
        start = dt.datetime.fromtimestamp(due, self.tz)
        if all_day:
            dates = f"{start:%Y%m%d}/{start + dt.timedelta(days=1):%Y%m%d}"
        else:
            s, e = start.astimezone(dt.timezone.utc), start.astimezone(dt.timezone.utc) + dt.timedelta(minutes=30)
            dates = f"{s:%Y%m%dT%H%M%SZ}/{e:%Y%m%dT%H%M%SZ}"
        q = {"action": "TEMPLATE", "text": title, "dates": dates, "details": f"From your chats (NASH Think): {details or ''}"[:900]}
        return "https://calendar.google.com/calendar/render?" + urllib.parse.urlencode(q)

    def list(self, platforms=None, rescan=True):
        """Reminders grouped for the panel: overdue, today, week, later, history (older and open), done."""
        if rescan:
            self.ensure_scanned()
        now = self.now()
        today = self.today()
        groups = {"overdue": [], "today": [], "week": [], "later": [], "history": [], "done": []}
        with closing(self._store()) as store:
            rows = store.execute("SELECT * FROM reminders WHERE status != 'dismissed'").fetchall()
            scanned_at = self._meta(store, "scanned_at")
        for row in rows:
            if platforms and row["platform"] not in platforms and row["source"] == "auto":
                continue
            item = self._item(row)
            if item["status"] == "done":
                groups["done"].append(item)
                continue
            d = dt.date.fromisoformat(item["date"])
            passed = (d < today) if item["all_day"] else (item["due_ts"] < now)
            if d == today and not (passed and not item["all_day"]):
                groups["today"].append(item)
            elif passed and now - item["due_ts"] <= OVERDUE_DAYS * 86400:
                groups["overdue"].append(item)
            elif passed:
                groups["history"].append(item)
            elif (d - today).days <= 7:
                groups["week"].append(item)
            else:
                groups["later"].append(item)
        for k in ("overdue", "today", "week", "later"):
            groups[k].sort(key=lambda i: (i["due_ts"], i["id"]))
        groups["history"].sort(key=lambda i: -i["due_ts"])
        groups["done"].sort(key=lambda i: -i["due_ts"])
        history_total = len(groups["history"])
        groups["history"] = groups["history"][:MAX_HISTORY]
        groups["done"] = groups["done"][:MAX_HISTORY]
        soon = sum(1 for g in ("today", "week", "later") for i in groups[g] if 0 <= i["due_ts"] - now <= SOON_S)
        return {
            "now": _iso(now), "now_ts": now, "today": today.isoformat(), "timezone": self.tz_name,
            "utc_offset_min": int(dt.datetime.fromtimestamp(now, self.tz).utcoffset().total_seconds() // 60),
            "clock": self.now_mode, "scanned_at": _iso(int(scanned_at)) if scanned_at else None,
            "counts": {**{k: len(v) for k, v in groups.items()}, "history": history_total,
                       "badge": len(groups["overdue"]) + soon},
            "groups": groups,
        }

    # ── changes ──
    def get(self, rid):
        with closing(self._store()) as store:
            row = store.execute("SELECT * FROM reminders WHERE id = ?", (rid,)).fetchone()
        if row is None:
            raise ReminderNotFound(f"No reminder {rid}.")
        return self._item(row)

    def update(self, rid, action, until=None, title=None, due=None, all_day=None):
        if action not in ACTIONS:
            raise BadRequest(f"action must be one of {', '.join(sorted(ACTIONS))}")
        now = int(time.time())
        with self._lock, closing(self._store()) as store:
            row = store.execute("SELECT * FROM reminders WHERE id = ?", (rid,)).fetchone()
            if row is None:
                raise ReminderNotFound(f"No reminder {rid}.")
            if action in ("done", "open"):
                store.execute("UPDATE reminders SET status = ?, snoozed_until = NULL, updated_at = ? WHERE id = ?",
                              (action, now, rid))
            elif action == "dismiss":
                store.execute("UPDATE reminders SET status = 'dismissed', updated_at = ? WHERE id = ?", (now, rid))
            elif action == "snooze":
                if not until or int(until) <= self.now():
                    raise BadRequest("snooze needs a time in the future")
                store.execute("UPDATE reminders SET snoozed_until = ?, status = 'open', updated_at = ? WHERE id = ?",
                              (int(until), now, rid))
            else:
                sets, params = ["edited = 1", "updated_at = ?"], [now]
                if title is not None:
                    title = title.strip()
                    if not title:
                        raise BadRequest("title can't be empty")
                    sets.append("title = ?")
                    params.append(title[:200])
                if due is not None:
                    d = dt.datetime.fromtimestamp(int(due), self.tz)
                    sets += ["due_at = ?", "date_local = ?", "snoozed_until = NULL"]
                    params += [int(due), d.date().isoformat()]
                if all_day is not None:
                    sets.append("all_day = ?")
                    params.append(int(bool(all_day)))
                store.execute(f"UPDATE reminders SET {', '.join(sets)} WHERE id = ?", (*params, rid))
            store.commit()
        return self.get(rid)

    def add(self, title, due, all_day=False, kind="reminder"):
        title = (title or "").strip()
        if not title:
            raise BadRequest("title can't be empty")
        if kind not in KINDS:
            kind = "reminder"
        d = dt.datetime.fromtimestamp(int(due), self.tz)
        if all_day:
            due = int(dt.datetime.combine(d.date(), dt.time(0, 0), self.tz).timestamp())
        now = int(time.time())
        with self._lock, closing(self._store()) as store:
            cur = store.execute(
                "INSERT INTO reminders (key, title, due_at, all_day, date_local, kind, confidence, status, source, "
                "said_by, from_owner, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, 1.0, 'open', 'manual', 'You', 1, ?, ?)",
                (f"manual:{secrets.token_hex(8)}", title[:200], int(due), int(bool(all_day)), d.date().isoformat(), kind,
                 now, now))
            store.commit()
            rid = cur.lastrowid
        return self.get(rid)

    # ── calendar ──
    def feed_token(self, create=True):
        """The secret in this workspace's subscribable calendar link."""
        with self._lock, closing(self._store()) as store:
            token = self._meta(store, "feed_token")
            if not token and create:
                token = secrets.token_urlsafe(24)
                self._set_meta(store, "feed_token", token)
                store.commit()
        return token

    def upcoming(self, since_days=1):
        """Open reminders due from `since_days` ago onwards (what belongs in a calendar)."""
        self.ensure_scanned()
        now = self.now()
        with closing(self._store()) as store:
            rows = store.execute("SELECT * FROM reminders WHERE status = 'open'").fetchall()
        items = [self._item(r) for r in rows]
        return sorted((i for i in items if i["due_ts"] >= now - since_days * 86400), key=lambda i: i["due_ts"])

    def to_ics(self, name="NASH Think reminders"):
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        lines = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//NASH Think//Reminders//EN", "CALSCALE:GREGORIAN",
                 "METHOD:PUBLISH", f"X-WR-CALNAME:{_ics_escape(name)}", f"X-WR-TIMEZONE:{self.tz_name}"]
        for i in self.upcoming():
            start = dt.datetime.fromtimestamp(i["due_ts"], self.tz)
            lines += ["BEGIN:VEVENT", f"UID:nashthink-{i['id']}@{self.store_path.stem}", f"DTSTAMP:{stamp}"]
            if i["all_day"]:
                lines += [f"DTSTART;VALUE=DATE:{start:%Y%m%d}", f"DTEND;VALUE=DATE:{start + dt.timedelta(days=1):%Y%m%d}"]
            else:
                s = start.astimezone(dt.timezone.utc)
                lines += [f"DTSTART:{s:%Y%m%dT%H%M%SZ}", f"DTEND:{s + dt.timedelta(minutes=30):%Y%m%dT%H%M%SZ}"]
            if i["repeat"] == "yearly":
                lines.append("RRULE:FREQ=YEARLY")
            who = f"{i['said_by']} in {i['conversation']}" if i.get("conversation") else i["said_by"]
            desc = f"{i['excerpt'] or ''}\n\n{who} · {i['platform'] or 'added by you'} (NASH Think)".strip()
            lines += [f"SUMMARY:{_ics_escape(i['title'])}", f"DESCRIPTION:{_ics_escape(desc)}",
                      f"CATEGORIES:{_ics_escape(i['kind'] or 'reminder')}",
                      "BEGIN:VALARM", "ACTION:DISPLAY", f"DESCRIPTION:{_ics_escape(i['title'])}",
                      "TRIGGER:-PT30M" if not i["all_day"] else "TRIGGER:PT9H", "END:VALARM", "END:VEVENT"]
        lines.append("END:VCALENDAR")
        return "\r\n".join(_ics_fold(l) for l in lines) + "\r\n"
