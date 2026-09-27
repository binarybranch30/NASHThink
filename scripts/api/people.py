"""Read-only person profiles ("Your history with X") for /api/person/..., from the local SQLite memory database.

A person is one graph person node (U_<Users.id>, labelled as scripts/utils/export_cosmograph.py labels it).
Accounts on different platforms are never assumed to be the same person: the only accounts merged are the
owner's, exactly as config/identity_map.json maps them. Everything is counted from stored messages, and the
brief is built from those counts and the conversations' own titles, never from guesses about feelings or
relationships. The database is opened read-only; nothing is written, and no embedding model is involved.

History with a person:
- their conversations: threads the person wrote in;
- shared conversations: those where you also wrote and that have at most SMALL_THREAD participants, so a
  big public thread you both commented in doesn't count as talking to each other;
- messages: theirs everywhere, plus yours in shared conversations. Paged on request (keyset pagination).
"""
import base64
import datetime as dt
import json
import re
import sqlite3
import sys
import threading
import time
from collections import Counter, OrderedDict
from pathlib import Path

import insights
from insights import TS_SQL, clean_text, iso_ts, person_label

sys.path.append(str(Path(__file__).resolve().parents[1] / "semantic"))
import hinglish  # noqa: E402

TSC = f"COALESCE(({TS_SQL}), 0)"   # missing timestamps sort oldest and still page correctly
PLATFORM_NAMES = {"twitter": "Twitter / X", "reddit": "Reddit", "instagram": "Instagram", "discord": "Discord",
                  "facebook": "Facebook", "whatsapp": "WhatsApp", "google": "Google", "chatgpt": "ChatGPT", "claude": "Claude"}
PERSON_RE = re.compile(r"^U_(\d{1,18})$")
THREAD_RE = re.compile(r"^T_(\d{1,18})$")
SMALL_THREAD = 12          # participants; above this a conversation is a large (public) thread
NOTABLE = 5                # notable conversations shown in the profile
MAX_SOURCES = 8            # conversations the brief may cite
SPARSE_MESSAGES = 3        # fewer of their messages than this: too little to summarise
TOPIC_SCAN = 3000          # latest messages of theirs scanned for recurring words
TOPIC_MIN_CONVERSATIONS = 2
TOPIC_MIN_MONTHS = 3       # ... or, within one long chat, this many different months
TOPIC_MIN_MENTIONS = 5
TOPICS = 5
RELATED = 5
MESSAGE_CHARS = 4000       # per-message display cap in the API (the database keeps everything)
PAGE_DEFAULT, PAGE_MAX = 30, 100
CACHE_SIZE = 64
TITLE_CHARS = 200

# Accounts whose messages don't come from a person talking to you.
AUTOMATED_RE = re.compile(r"(?i)(^automoderator$|^auto[-_]?mod$|bot$|^bot[-_]|[-_]bot[-_]|moderator$|^\[?deleted\]?$|"
                          r"^deleted[-_ ]?user|^instagram user$|^facebook user$|^system$|^reddit$|^unknown$)")

_WORD_RE = re.compile(r"[^\W\d_]{4,}", re.UNICODE)
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
# Words too common in chat to say anything about a topic (plus English stopwords and Hinglish filler).
_COMMON = set("""
that this with have from your they there would about really think know good right also even more much some
time people thing things want going make because could should will been were when then than them their here
only very sure well still into over after before just like dont yeah okay haha hahaha lmao thanks thank what
which where while being does done doing gonna wanna gotta cant didnt doesnt isnt wasnt wont youre thats
theres whats yes yeah nope maybe actually literally probably anyway though always never every something
anything nothing everything someone anyone everyone said says told tell know knew mean means feel feels felt
look looks looked need needs lots little great nice cool okay alright stuff kind sort take took give gave
come came back first last next same other another many most such those these said again today tomorrow
yesterday tonight week year years month months days please sorry hello bhai yaar hain nahi kuch bhi karna
kiya raha rahi hoga hota abhi phir wala wali wale matlab bahut bohot accha acha thik theek haan kyun kaise
kaha kahan toh mera meri tera teri tumhe mujhe usko unko apna apni sabko kabhi aaja chalo acha achha deleted
removed https http www com org reddit instagram message messages sent attachment photo video reel shared
""".split()) | set("""
able above across actually after afterwards again against almost alone along already although among amongst
another anyhow anything anyway anywhere around away became become becomes becoming been beforehand behind below
beside besides between beyond both bottom cannot certain certainly clearly could couldn describe detail down
during each either else elsewhere enough especially etc even ever every everywhere except fairly few fifteen
fifty fill find fire five forty four from front full further furthermore generally given gives giving goes
going gone gotten hardly hence here hereafter hereby herein hers herself himself however hundred indeed instead
itself keep keeps kept last later latter least less lest mainly make makes making meanwhile might mine more
moreover mostly move much must myself namely neither nevertheless next nine nobody none noone nor nothing now
nowhere often once ones onto other others otherwise ours ourselves part particular perhaps please possible
quite rather really regarding same seem seemed seeming seems serious several shall should show side since
simply sincere sixty some somehow someone something sometime sometimes somewhere still such system take taken
taking than thanks then thence there thereafter thereby therefore therein thereupon these they thick thin third
those though three through throughout thru thus together toward towards twelve twenty under unless until upon
used useful using usually various very via want wants wanted was were what whatever when whence whenever where
whereafter whereas whereby wherein whereupon wherever whether which while whither whoever whole whom whose why
will with within without would yet your yours yourself yourselves also been being both each here just like
make need only over same some such than that them then there these they this those through very what when where
which while with would about above after again against because before being below between during further into
more most other some such that their them then there these they this those through under until very what when
where which while will with would yourself okay basically definitely exactly totally usually probably seriously
honestly apparently obviously currently recently already especially simply specific specifically example
""".split())


class PeopleUnavailable(Exception):
    """The memory database is missing or unreadable."""


class PersonNotFound(Exception):
    pass


class BadRequest(Exception):
    pass


def is_automated(*names):
    return any(n and AUTOMATED_RE.search(str(n).strip()) for n in names)


def month_label(iso_or_key):
    if not iso_or_key:
        return None
    y, m = int(iso_or_key[:4]), int(iso_or_key[5:7])
    return f"{('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec')[m - 1]} {y}"


def plural(n, word, many=None):
    return f"{n:,} {word if n == 1 else (many or word + 's')}"


def encode_cursor(ts, msg_id):
    return base64.urlsafe_b64encode(json.dumps([ts, msg_id]).encode()).decode().rstrip("=")


def decode_cursor(cursor):
    try:
        ts, msg_id = json.loads(base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4)))
        if not isinstance(ts, int) or not isinstance(msg_id, str):
            raise ValueError
        return ts, msg_id
    except Exception:
        raise BadRequest("cursor: not a valid page cursor")


class PeopleService:
    def __init__(self, db_path=insights.DEFAULT_DB, identity_map=insights.DEFAULT_IDENTITY_MAP):
        self.db_path = Path(db_path)
        self.identity_map = identity_map   # a path, or a (master, aliases) tuple for tests
        self._lock = threading.Lock()
        self._cache = OrderedDict()

    # ─── plumbing ───────────────────────────────────────────────────────────
    def _identity(self):
        return self.identity_map if isinstance(self.identity_map, tuple) else insights.load_identity(self.identity_map)

    def _version(self):
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

    def _connect(self):
        if not self.db_path.is_file():
            raise PeopleUnavailable(f"Memory database not found: {self.db_path.name}. Run the parsers to build it.")
        try:
            conn = sqlite3.connect(f"{self.db_path.resolve().as_uri()}?mode=ro", uri=True)
            conn.execute("SELECT 1 FROM Messages LIMIT 1").fetchall()
        except sqlite3.Error as e:
            raise PeopleUnavailable(f"Memory database is not readable: {e}")
        return conn

    def _cached(self, key, fn):
        key = key + (self._version(), repr(self.identity_map))
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        conn = self._connect()
        try:
            value = fn(conn)
        except sqlite3.Error as e:
            raise PeopleUnavailable(f"Memory database query failed: {e}")
        finally:
            conn.close()
        with self._lock:
            self._cache[key] = value
            while len(self._cache) > CACHE_SIZE:
                self._cache.popitem(last=False)
        return value

    @staticmethod
    def _filters_sql(platforms, date_from, date_to, alias="m", thread_alias="t"):
        where, params = [], []
        if platforms:
            where.append(f"{thread_alias}.platform IN ({', '.join('?' * len(platforms))})")
            params += list(platforms)
        ts = TS_SQL.replace("m.", f"{alias}.")
        if date_from:
            where.append(f"{ts} >= ?")
            params.append(int(date_from.timestamp()))
        if date_to:
            where.append(f"{ts} < ?")
            params.append(int(date_to.timestamp()))
        return where, params

    # ─── who ────────────────────────────────────────────────────────────────
    def _person(self, conn, node_id):
        m = PERSON_RE.match(node_id or "")
        if not m:
            raise BadRequest("node_id must look like U_<number> (a person)")
        uid = int(m.group(1))
        master, aliases = self._identity()
        users = {u: (p, n, r) for u, p, n, r in conn.execute("SELECT id, platform, display_name, raw_id FROM Users")}
        if uid not in users:
            raise PersonNotFound(f"No person {node_id} in the memory database")
        owner = sorted(u for u, (p, n, r) in users.items() if insights.is_owner(p, n, r, aliases))
        platform, name, raw = users[uid]
        base = clean_text(person_label(platform, name, raw))[:TITLE_CHARS] or f"User {uid}"
        you = uid in owner
        accounts = owner if you else [uid]
        # Same label on another account: shown, never merged (only identity_map.json merges accounts).
        same = [] if you else [
            {"node_id": f"U_{u}", "platform": p} for u, (p, n, r) in sorted(users.items())
            if u != uid and u not in owner and clean_text(person_label(p, n, r)).lower() == base.lower()]
        return {
            "uid": uid, "you": you, "owner": owner, "accounts": accounts,
            "label": (master or "You") if you else base,
            "platform": platform,
            "automated": not you and is_automated(base, name, raw),
            "account_list": [{"node_id": f"U_{u}", "platform": users[u][0],
                              "label": clean_text(person_label(*users[u]))[:TITLE_CHARS]} for u in accounts],
            "same_name_elsewhere": same,
            "users": users,
        }

    def _threads(self, conn, who, platforms, date_from, date_to):
        """Per conversation the person wrote in (within the filters): their/your counts, size, dates."""
        where, params = self._filters_sql(platforms, date_from, date_to)
        rows = conn.execute(
            f"SELECT m.thread_id, t.platform, t.title, COUNT(*), MIN({TS_SQL}), MAX({TS_SQL}) "
            f"FROM Messages m JOIN Threads t ON t.id = m.thread_id "
            f"WHERE m.author_id IN (SELECT value FROM json_each(?)) {''.join(' AND ' + w for w in where)} "
            f"GROUP BY m.thread_id", [json.dumps(who["accounts"])] + params).fetchall()
        threads = {tid: {"thread_id": tid, "node_id": f"T_{tid}", "platform": plat,
                         "title": clean_text(title or "")[:TITLE_CHARS] or "(untitled)",
                         "their_messages": n, "your_messages": 0, "people": 1, "first_ts": a, "last_ts": b}
                   for tid, plat, title, n, a, b in rows}
        if threads and not who["you"]:
            ts_where, ts_params = self._filters_sql(None, date_from, date_to, alias="m2")
            cond = " AND ".join(ts_where) or "1"
            for tid, people, yours in conn.execute(
                    f"SELECT m2.thread_id, COUNT(DISTINCT m2.author_id), "
                    f"SUM(CASE WHEN m2.author_id IN (SELECT value FROM json_each(?)) AND {cond} THEN 1 ELSE 0 END) "
                    f"FROM Messages m2 WHERE m2.thread_id IN (SELECT value FROM json_each(?)) GROUP BY m2.thread_id",
                    [json.dumps(who["owner"])] + ts_params + [json.dumps(list(threads))]):
                threads[tid]["people"] = people
                threads[tid]["your_messages"] = yours or 0
        for t in threads.values():
            t["large"] = t["people"] > SMALL_THREAD
            t["shared"] = bool(t["your_messages"]) and not t["large"]
        return threads

    @staticmethod
    def _stats(threads):
        if not threads:
            return {"conversations": 0, "shared_conversations": 0, "their_messages": 0, "your_messages": 0,
                    "first_date": None, "latest_date": None, "platforms": []}
        plats = Counter()
        for t in threads.values():
            plats[t["platform"]] += t["their_messages"]
        return {
            "conversations": len(threads),
            "shared_conversations": sum(t["shared"] for t in threads.values()),
            "their_messages": sum(t["their_messages"] for t in threads.values()),
            "your_messages": sum(t["your_messages"] for t in threads.values() if t["shared"]),
            "first_date": iso_ts(min(t["first_ts"] for t in threads.values())),
            "latest_date": iso_ts(max(t["last_ts"] for t in threads.values())),
            "platforms": [p for p, _ in plats.most_common()],
        }

    @staticmethod
    def _public(t):
        return {k: t[k] for k in ("node_id", "title", "platform", "their_messages", "your_messages", "people", "large", "shared")} | {
            "first_date": iso_ts(t["first_ts"]), "latest_date": iso_ts(t["last_ts"])}

    @staticmethod
    def _notable_order(t):
        # Back-and-forth first (both wrote, small conversation), then small ones, then by their messages.
        # Large public threads come last however busy they are, so one huge thread can't dominate.
        return (not t["shared"], t["large"], -min(t["their_messages"], t["your_messages"] or 0),
                -t["their_messages"], -(t["last_ts"] or 0))

    # ─── profile ────────────────────────────────────────────────────────────
    def profile(self, node_id, platforms=None, date_from=None, date_to=None):
        key = ("profile", node_id, tuple(platforms or ()), date_from, date_to)
        return self._cached(key, lambda conn: self._profile(conn, node_id, platforms, date_from, date_to))

    def _profile(self, conn, node_id, platforms, date_from, date_to):
        started = time.perf_counter()
        who = self._person(conn, node_id)
        filtered = bool(platforms or date_from or date_to)
        all_threads = self._threads(conn, who, None, None, None)
        threads = self._threads(conn, who, platforms, date_from, date_to) if filtered else all_threads
        lifetime, stats = self._stats(all_threads), self._stats(threads)

        where, params = self._filters_sql(platforms, date_from, date_to)
        acc = json.dumps(who["accounts"])
        activity = [{"month": mo, "messages": n} for mo, n in conn.execute(
            f"SELECT strftime('%Y-%m', {TS_SQL}, 'unixepoch') AS mo, COUNT(*) FROM Messages m JOIN Threads t ON t.id = m.thread_id "
            f"WHERE m.author_id IN (SELECT value FROM json_each(?)) {''.join(' AND ' + w for w in where)} "
            f"AND m.timestamp_utc IS NOT NULL GROUP BY mo ORDER BY mo", [acc] + params) if mo]

        ordered = sorted(threads.values(), key=self._notable_order)
        sources = [t for t in ordered[:NOTABLE]]
        topics = self._topics(conn, who, threads, platforms, date_from, date_to)
        for tp in topics:
            for tid in tp.pop("_threads"):
                if len(sources) >= MAX_SOURCES:
                    break
                if all(s["thread_id"] != tid for s in sources):
                    sources.append(threads[tid])
            tp["sources"] = [i + 1 for i, s in enumerate(sources) if s["thread_id"] in tp["_ids"]][:3]
            tp.pop("_ids")

        related = [] if who["you"] else self._related(conn, who, threads)
        brief = self._brief(who, stats, threads, sources, topics, activity)
        return {
            "node_id": node_id,
            "label": who["label"],
            "kind": "you" if who["you"] else ("automated" if who["automated"] else "person"),
            "platform": who["platform"],
            "accounts": who["account_list"],
            "same_name_elsewhere": who["same_name_elsewhere"],
            "scope": {"filtered": filtered,
                      "platforms": list(platforms) if platforms else None,
                      "date_from": date_from.isoformat() if date_from else None,
                      "date_to": date_to.isoformat() if date_to else None},
            "lifetime": lifetime,
            "stats": stats,
            "activity": activity,
            "topics": topics,
            "sources": [dict(self._public(s), notable=i < NOTABLE and s in ordered[:NOTABLE]) for i, s in enumerate(sources)],
            "brief": brief,
            "related": related,
            "small_thread_limit": SMALL_THREAD,
            "took_ms": int((time.perf_counter() - started) * 1000),
        }

    def _topics(self, conn, who, threads, platforms, date_from, date_to):
        """Words that recur in their messages: in at least two conversations, or (for a single long chat) in
        at least TOPIC_MIN_MONTHS different months. Each comes with the conversations it appears in."""
        if not threads:
            return []
        where, params = self._filters_sql(platforms, date_from, date_to)
        rows = conn.execute(
            f"SELECT m.thread_id, strftime('%Y-%m', {TSC}, 'unixepoch'), m.content FROM Messages m JOIN Threads t ON t.id = m.thread_id "
            f"WHERE m.author_id IN (SELECT value FROM json_each(?)) {''.join(' AND ' + w for w in where)} "
            f"ORDER BY m.timestamp_utc DESC LIMIT ?", [json.dumps(who["accounts"])] + params + [TOPIC_SCAN]).fetchall()
        convs, months, counts = {}, {}, Counter()
        for tid, month, content in rows:
            text = _URL_RE.sub(" ", (content or "").lower())
            for w in set(_WORD_RE.findall(text)):
                if w in _COMMON or hinglish.is_filler(w) or hinglish.is_negation(w):
                    continue
                w = hinglish.normalize(w)
                convs.setdefault(w, set()).add(tid)
                months.setdefault(w, set()).add(month)
                counts[w] += 1

        def recurring(w):
            return counts[w] >= 3 and (len(convs[w]) >= TOPIC_MIN_CONVERSATIONS
                                       or (len(months[w]) >= TOPIC_MIN_MONTHS and counts[w] >= TOPIC_MIN_MENTIONS))
        ranked = sorted((w for w in convs if recurring(w)),
                        key=lambda w: (-len(convs[w]), -len(months[w]), -counts[w], w))[:TOPICS]
        out = []
        for w in ranked:
            best = sorted(convs[w], key=lambda tid: self._notable_order(threads[tid]))[:3]
            out.append({"word": w, "conversations": len(convs[w]), "months": len(months[w]), "mentions": counts[w],
                        "_threads": best, "_ids": set(best)})
        return out

    def _related(self, conn, who, threads):
        """Other people in the same small conversations (not you, not automated accounts)."""
        small = [tid for tid, t in threads.items() if not t["large"]]
        if not small:
            return []
        rows = conn.execute(
            "SELECT author_id, COUNT(DISTINCT thread_id) AS n FROM Messages WHERE thread_id IN (SELECT value FROM json_each(?)) "
            "GROUP BY author_id ORDER BY n DESC LIMIT 50", [json.dumps(small)]).fetchall()
        out = []
        for uid, n in rows:
            if uid in who["accounts"] or uid in who["owner"] or uid not in who["users"]:
                continue
            p, name, raw = who["users"][uid]
            label = clean_text(person_label(p, name, raw))[:TITLE_CHARS]
            if is_automated(label, name, raw):
                continue
            out.append({"node_id": f"U_{uid}", "label": label, "platform": p, "shared_conversations": n})
            if len(out) >= RELATED:
                break
        return out

    @staticmethod
    def _brief(who, stats, threads, sources, topics, activity):
        """Short sentences, each built from counts, dates and titles, with the conversations it cites."""
        label, n, c = who["label"], stats["their_messages"], stats["conversations"]
        out = []

        def say(text, cite=()):
            out.append({"text": text, "sources": sorted(set(cite))})

        cite_of = {s["thread_id"]: i + 1 for i, s in enumerate(sources)}
        if who["you"]:
            say(f"This is you: {plural(len(who['accounts']), 'account')} mapped to you in the identity map.")
            if n:
                say(f"You wrote {plural(n, 'message')} in {plural(c, 'conversation')} on "
                    f"{', '.join(PLATFORM_NAMES.get(p, p.title()) for p in stats['platforms'])}, "
                    f"from {month_label(stats['first_date'])} to {month_label(stats['latest_date'])}.")
            return {"sentences": out, "sparse": n < SPARSE_MESSAGES}
        if who["automated"]:
            say(f"{label} looks like an automated or deleted account, so its messages may not come from a person.")
        if n == 0:
            say(f"No messages from {label} match the current filters.")
            return {"sentences": out, "sparse": True}
        if n < SPARSE_MESSAGES:
            say(f"There isn’t enough history to summarize: {label} wrote {plural(n, 'message')} in "
                f"{plural(c, 'conversation')}, on {month_label(stats['first_date'])}"
                + (f" and {month_label(stats['latest_date'])}." if month_label(stats['latest_date']) != month_label(stats['first_date']) else "."),
                [cite_of[t] for t in threads if t in cite_of])
            return {"sentences": out, "sparse": True}

        span = month_label(stats["first_date"]), month_label(stats["latest_date"])
        say(f"{label} wrote {plural(n, 'message')} in {plural(c, 'conversation')} on {' and '.join(PLATFORM_NAMES.get(p, p.title()) for p in stats['platforms'])}, "
            + (f"in {span[0]}." if span[0] == span[1] else f"from {span[0]} to {span[1]}."))
        if stats["shared_conversations"]:
            say(f"You both wrote in {plural(stats['shared_conversations'], 'small conversation')}, "
                f"where you wrote {plural(stats['your_messages'], 'message')}.",
                [cite_of[t["thread_id"]] for t in sources if t["shared"]][:3])
        else:
            say("None of these are small conversations with a message from you, so this is their side only.")
        top = max(threads.values(), key=lambda t: (t["their_messages"], t["last_ts"] or 0))
        if c > 1 and top["thread_id"] in cite_of:
            size = f", a large thread with {top['people']:,} people" if top["large"] else ""
            say(f"Their busiest conversation is “{top['title']}” ({plural(top['their_messages'], 'message')}{size}).",
                [cite_of[top["thread_id"]]])
        if len(activity) >= 3:
            peak = max(activity, key=lambda a: a["messages"])
            if peak["messages"] / n >= 0.25:
                say(f"Busiest month: {month_label(peak['month'])} ({plural(peak['messages'], 'message')}).")
        large = sum(t["their_messages"] for t in threads.values() if t["large"])
        if large / n > 0.6:
            say(f"Most of their messages are in large public threads (over {SMALL_THREAD} people), which may not be direct contact.")
        if topics:
            words = ", ".join(f"“{t['word']}”" for t in topics[:3])
            where = "across their conversations" if all(t["conversations"] >= 2 for t in topics[:3]) else "in several different months"
            say(f"Words they use repeatedly {where}: {words}.", [i for t in topics[:3] for i in t["sources"]])
        return {"sentences": out, "sparse": False}

    # ─── paged lists ────────────────────────────────────────────────────────
    def conversations(self, node_id, platforms=None, date_from=None, date_to=None, offset=0, limit=PAGE_DEFAULT):
        """Their conversations, latest first, one page at a time."""
        def run(conn):
            who = self._person(conn, node_id)
            return sorted(self._threads(conn, who, platforms, date_from, date_to).values(), key=lambda t: -(t["last_ts"] or 0))
        items = self._cached(("threads", node_id, tuple(platforms or ()), date_from, date_to), run)
        page = items[offset:offset + limit]
        return {"node_id": node_id, "total": len(items), "offset": offset,
                "next_offset": offset + limit if offset + limit < len(items) else None,
                "items": [self._public(t) for t in page]}

    def messages(self, node_id, thread=None, platforms=None, date_from=None, date_to=None, cursor=None, limit=PAGE_DEFAULT):
        """Their messages plus yours in shared small conversations, newest first, keyset-paged."""
        tid = None
        if thread:
            m = THREAD_RE.match(thread)
            if not m:
                raise BadRequest("thread must look like T_<number> (a conversation)")
            tid = int(m.group(1))
        after = decode_cursor(cursor) if cursor else None

        def threads_run(conn):
            who = self._person(conn, node_id)
            return who, self._threads(conn, who, platforms, date_from, date_to)
        who, threads = self._cached(("who+threads", node_id, tuple(platforms or ()), date_from, date_to), threads_run)
        if tid is not None and tid not in threads:
            raise BadRequest(f"{thread} is not one of this person's conversations in the current filters")
        shared = [t for t, v in threads.items() if v["shared"] and (tid is None or t == tid)]

        def run(conn):
            where, params = self._filters_sql(platforms, date_from, date_to)
            cond = ["(m.author_id IN (SELECT value FROM json_each(?)) OR "
                    "(m.author_id IN (SELECT value FROM json_each(?)) AND m.thread_id IN (SELECT value FROM json_each(?))))"]
            params = [json.dumps(who["accounts"]), json.dumps([] if who["you"] else who["owner"]), json.dumps(shared)] + params
            if tid is not None:
                cond.append("m.thread_id = ?")
                params.append(tid)
            base_where = " AND ".join(cond + where)
            total = None
            if after is None:
                total = conn.execute(f"SELECT COUNT(*) FROM Messages m JOIN Threads t ON t.id = m.thread_id WHERE {base_where}", params).fetchone()[0]
            page_where, page_params = base_where, list(params)
            if after:
                page_where += f" AND ({TSC} < ? OR ({TSC} = ? AND m.msg_id < ?))"
                page_params += [after[0], after[0], after[1]]
            rows = conn.execute(
                f"SELECT m.msg_id, {TSC}, m.author_id, m.thread_id, t.platform, t.title, m.content "
                f"FROM Messages m JOIN Threads t ON t.id = m.thread_id WHERE {page_where} "
                f"ORDER BY {TSC} DESC, m.msg_id DESC LIMIT ?", page_params + [limit + 1]).fetchall()
            return total, rows
        # Pages are not cached: they are cheap (author/thread index) and each is requested once.
        conn = self._connect()
        try:
            total, rows = run(conn)
        except sqlite3.Error as e:
            raise PeopleUnavailable(f"Memory database query failed: {e}")
        finally:
            conn.close()
        more = len(rows) > limit
        rows = rows[:limit]
        items = []
        for msg_id, ts, author, t_id, plat, title, content in rows:
            text = content or ""
            mine = author in who["owner"]
            items.append({"date": iso_ts(ts) if ts else None, "author": "you" if mine and not who["you"] else "them",
                          "author_label": "You" if mine else who["label"],
                          "node_id": f"T_{t_id}", "thread_title": clean_text(title or "")[:TITLE_CHARS] or "(untitled)",
                          "platform": plat, "text": text[:MESSAGE_CHARS], "clipped": len(text) > MESSAGE_CHARS})
        last = rows[-1] if rows else None
        return {"node_id": node_id, "thread": thread, "total": total, "count": len(items),
                "next_cursor": encode_cursor(last[1], last[0]) if more and last else None,
                "includes_yours": not who["you"], "small_thread_limit": SMALL_THREAD, "messages": items}
