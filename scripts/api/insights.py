"""Read-only activity insights for GET /api/insights, computed from the local SQLite memory database.

Aggregates only: counts, date ranges, per-platform and per-month totals, and the most active contacts
and conversations (by message count). The database is opened read-only (`mode=ro`) and nothing is
written or sent anywhere. Results are cached per filter set until the database file changes.

People are labelled the way scripts/utils/export_cosmograph.py labels graph nodes (U_<Users.id>),
and conversations use the graph's thread ids (T_<Threads.id>), so the UI can focus the matching node.
Accounts that config/identity_map.json resolves to the owner's persona are excluded from contacts
and from the people count.
"""
import datetime as dt
import json
import sqlite3
import threading
import time
from collections import OrderedDict
from pathlib import Path

UTC = dt.timezone.utc
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = REPO_ROOT / "processed_data" / "db" / "sarthink_memory.db"
DEFAULT_IDENTITY_MAP = REPO_ROOT / "config" / "identity_map.json"

DEFAULT_TOP = 10
MAX_TOP = 50
MAX_MONTHS = 1200
CACHE_SIZE = 32
LABEL_CHARS = 120
# Message timestamps are epoch seconds; a few exports store milliseconds (same rule as export_cosmograph.to_epoch).
TS_SQL = "CASE WHEN m.timestamp_utc > 100000000000 THEN m.timestamp_utc / 1000 ELSE m.timestamp_utc END"


class InsightsUnavailable(Exception):
    """The memory database is missing or unreadable."""


def load_identity(path=DEFAULT_IDENTITY_MAP):
    """(master persona, alias set) from identity_map.json; (None, empty set) when it is missing or invalid."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None, set()
    master = data.get("master_persona") or None
    aliases = set()
    for names in (data.get("aliases") or {}).values():
        for name in names or []:
            if name:
                aliases.update({str(name), str(name).lower()})
    if master:
        aliases.update({master, master.lower()})
    return master, aliases


def clean_text(value):
    return str(value).replace(",", "").replace("\n", " ").replace("\r", "").replace('"', "").strip()


def person_label(platform, display_name, raw_id):
    """The graph's node label for a Users row, before identity resolution (mirrors export_cosmograph)."""
    primary = str(raw_id) if platform in ("twitter", "reddit") else str(display_name)
    if primary.isdigit() and display_name:
        primary = str(display_name)
    return primary


def is_owner(platform, display_name, raw_id, aliases):
    if not aliases:
        return False
    names = (person_label(platform, display_name, raw_id), display_name, raw_id)
    return any(n is not None and (str(n) in aliases or str(n).lower() in aliases) for n in names)


def iso_ts(ts):
    return dt.datetime.fromtimestamp(int(ts), UTC).isoformat() if ts is not None else None


def month_keys(first_ts, last_ts):
    """Every 'YYYY-MM' from the month of first_ts to the month of last_ts (inclusive)."""
    a, b = dt.datetime.fromtimestamp(int(first_ts), UTC), dt.datetime.fromtimestamp(int(last_ts), UTC)
    y, m, out = a.year, a.month, []
    while (y, m) <= (b.year, b.month) and len(out) < MAX_MONTHS:
        out.append(f"{y:04d}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


class InsightsService:
    def __init__(self, db_path=DEFAULT_DB, identity_map=DEFAULT_IDENTITY_MAP):
        self.db_path = Path(db_path)
        self.identity_map = identity_map   # a path, or a (master, aliases) tuple for tests
        self._lock = threading.Lock()
        self._cache = OrderedDict()

    def _identity(self):
        if isinstance(self.identity_map, tuple):
            return self.identity_map
        return load_identity(self.identity_map)

    def _version(self):
        """Changes whenever the database (or its WAL) or the identity map is written, invalidating cached results."""
        parts = []
        paths = [self.db_path, Path(str(self.db_path) + "-wal")]
        if not isinstance(self.identity_map, tuple):
            paths.append(Path(self.identity_map))
        for p in paths:
            try:
                s = p.stat()
                parts.append((s.st_mtime_ns, s.st_size))
            except OSError:
                parts.append(None)
        return tuple(parts)

    def available(self):
        return self.db_path.is_file()

    def _connect(self):
        if not self.db_path.is_file():
            raise InsightsUnavailable(f"Memory database not found: {self.db_path.name}. Run the parsers to build it.")
        try:
            conn = sqlite3.connect(f"{self.db_path.resolve().as_uri()}?mode=ro", uri=True)
            conn.execute("SELECT 1 FROM Messages LIMIT 1").fetchall()
        except sqlite3.Error as e:
            raise InsightsUnavailable(f"Memory database is not readable: {e}")
        return conn

    def insights(self, platforms=None, date_from=None, date_to=None, top=DEFAULT_TOP):
        """Aggregates for the given filters. `date_from`/`date_to` are aware datetimes ([from, to))."""
        key = (tuple(platforms) if platforms else None, date_from, date_to, top, self._version(), repr(self.identity_map))
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return {**self._cache[key], "cached": True}
        started = time.perf_counter()
        conn = self._connect()
        try:
            result = self._compute(conn, platforms, date_from, date_to, top)
        except sqlite3.Error as e:
            raise InsightsUnavailable(f"Memory database query failed: {e}")
        finally:
            conn.close()
        result["took_ms"] = int((time.perf_counter() - started) * 1000)
        with self._lock:
            self._cache[key] = result
            while len(self._cache) > CACHE_SIZE:
                self._cache.popitem(last=False)
        return {**result, "cached": False}

    def _compute(self, conn, platforms, date_from, date_to, top):
        master, aliases = self._identity()
        where, params = [], []
        if platforms:
            where.append(f"t.platform IN ({', '.join('?' * len(platforms))})")
            params += list(platforms)
        if date_from:
            where.append(f"{TS_SQL} >= ?")
            params.append(int(date_from.timestamp()))
        if date_to:
            where.append(f"{TS_SQL} < ?")
            params.append(int(date_to.timestamp()))
        # One pass over Messages into a connection-private TEMP table (the main database stays read-only);
        # every aggregate below reads that instead of re-joining.
        conn.execute("PRAGMA temp_store = MEMORY")
        conn.execute("DROP TABLE IF EXISTS temp.f")
        conn.execute(f"CREATE TEMP TABLE f AS SELECT m.thread_id AS thread_id, m.author_id AS author_id, "
                     f"t.platform AS platform, {TS_SQL} AS ts FROM Messages m JOIN Threads t ON t.id = m.thread_id"
                     + (" WHERE " + " AND ".join(where) if where else ""), params)
        base, params = "SELECT * FROM temp.f", []

        available_platforms = [r[0] for r in conn.execute(
            "SELECT DISTINCT platform FROM Threads WHERE platform IS NOT NULL ORDER BY platform")]
        users = {uid: (plat, name, raw) for uid, plat, name, raw in conn.execute(
            "SELECT id, platform, display_name, raw_id FROM Users")}
        owner_ids = {uid for uid, (plat, name, raw) in users.items() if is_owner(plat, name, raw, aliases)}

        per_platform = {}
        for plat, msgs, threads, first, last in conn.execute(
                f"SELECT platform, COUNT(*), COUNT(DISTINCT thread_id), MIN(ts), MAX(ts) FROM ({base}) GROUP BY platform", params):
            per_platform[plat] = {"platform": plat, "messages": msgs, "threads": threads, "people": 0,
                                  "first_ts": first, "last_ts": last}

        contacts = []
        for author, msgs, threads, first, last, plat in conn.execute(
                f"SELECT author_id, COUNT(*), COUNT(DISTINCT thread_id), MIN(ts), MAX(ts), MIN(platform) "
                f"FROM ({base}) GROUP BY author_id", params):
            if author is None or author in owner_ids:
                continue
            u_plat, name, raw = users.get(author, (plat, None, None))
            u_plat = u_plat or plat
            if u_plat in per_platform:
                per_platform[u_plat]["people"] += 1
            label = clean_text(person_label(u_plat, name, raw))[:LABEL_CHARS] if (name or raw) else ""
            contacts.append({"node_id": f"U_{author}", "label": label or f"User {author}",
                             "platform": u_plat, "messages": msgs, "threads": threads,
                             "first_date": iso_ts(first), "last_date": iso_ts(last)})
        people = len(contacts)
        contacts.sort(key=lambda c: (-c["messages"], -c["threads"], c["node_id"]))

        conversations = []
        for tid, plat, title, msgs, ppl, first, last in conn.execute(
                f"SELECT b.thread_id, b.platform, th.title, COUNT(*), COUNT(DISTINCT b.author_id), MIN(b.ts), MAX(b.ts) "
                f"FROM ({base}) b JOIN Threads th ON th.id = b.thread_id "
                f"GROUP BY b.thread_id ORDER BY COUNT(*) DESC, b.thread_id LIMIT ?", params + [top]):
            conversations.append({"node_id": f"T_{tid}", "title": clean_text(title or "")[:LABEL_CHARS * 2] or "(untitled)",
                                  "platform": plat, "messages": msgs, "people": ppl,
                                  "first_date": iso_ts(first), "last_date": iso_ts(last)})

        month_rows = conn.execute(
            f"SELECT strftime('%Y-%m', ts, 'unixepoch') AS mo, platform, COUNT(*) FROM ({base}) "
            f"WHERE ts IS NOT NULL GROUP BY mo, platform", params).fetchall()

        total_msgs = sum(p["messages"] for p in per_platform.values())
        firsts = [p["first_ts"] for p in per_platform.values() if p["first_ts"] is not None]
        lasts = [p["last_ts"] for p in per_platform.values() if p["last_ts"] is not None]
        first_ts, last_ts = (min(firsts), max(lasts)) if firsts else (None, None)

        months, years = [], []
        if first_ts is not None:
            by_month = {k: {"month": k, "total": 0, "platforms": {}} for k in month_keys(first_ts, last_ts)}
            for mo, plat, n in month_rows:
                if mo in by_month:
                    b = by_month[mo]
                    b["total"] += n
                    b["platforms"][plat] = b["platforms"].get(plat, 0) + n
            months = list(by_month.values())
            by_year = OrderedDict()
            for b in months:
                y = by_year.setdefault(int(b["month"][:4]), {"year": int(b["month"][:4]), "total": 0, "platforms": {}})
                y["total"] += b["total"]
                for plat, n in b["platforms"].items():
                    y["platforms"][plat] = y["platforms"].get(plat, 0) + n
            years = list(by_year.values())

        platform_list = sorted(per_platform.values(), key=lambda p: (-p["messages"], p["platform"]))
        for p in platform_list:
            p["share"] = round(p["messages"] / total_msgs, 4) if total_msgs else 0.0
            p["first_date"], p["last_date"] = iso_ts(p.pop("first_ts")), iso_ts(p.pop("last_ts"))

        return {
            "empty": total_msgs == 0,
            "filters": {"platforms": list(platforms) if platforms else None,
                        "date_from": date_from.isoformat() if date_from else None,
                        "date_to": date_to.isoformat() if date_to else None},
            "totals": {"messages": total_msgs, "threads": sum(p["threads"] for p in platform_list),
                       "people": people, "platforms": len(platform_list)},
            "first_date": iso_ts(first_ts),
            "latest_date": iso_ts(last_ts),
            "platforms": platform_list,
            "available_platforms": available_platforms,
            "activity": {"months": months, "years": years},
            "top_contacts": contacts[:top],
            "top_conversations": conversations,
            "owner": {"configured": bool(aliases), "excluded_accounts": len(owner_ids)},
        }
