"""Parses Discord exports into the Sarthink memory database and discord_logs.jsonl.

Two formats are read from --archive (default archive/discord/naitik, staged by
scripts/utils/stage_discord_export.py):

* The official Discord data package: Messages/c<channel id>/{channel.json, messages.json|messages.csv},
  Messages/index.json (channel titles) and Account/user.json. The package holds only messages the account
  owner sent, so every message is yours; Account/user.json's id identifies the account and must appear
  in every DM's recipients, or the import stops instead of guessing. The other people in DMs and group
  DMs are recorded as Users plus ThreadMembers rows (they have no messages in the package).
  `Timestamp` has no zone marker; it is UTC (it matches the time encoded in each message's snowflake ID).
* DiscordChatExporter JSON ({"channel", "messages": [{"author", ...}]}), the format the earlier parser read.
  Its authors are yours only when their id is a package owner id or listed under "discord" in
  config/identity_map.json; otherwise nobody is marked as you.

Messages are stored with their original UTC epoch timestamp. Attachments are never downloaded: their file
names are kept as "[Attachment: name]", and an attachment-only message is stored as that marker. Messages with
neither text nor an attachment are skipped. User/channel/role mentions are replaced by names (or neutral
placeholders) so raw IDs don't end up in the searchable text. The package keeps only each message's final
(edited) text; deleted messages are not in it.

Re-runs are idempotent and keep database ids stable: users and threads are matched on their Discord ids,
new messages are inserted, changed ones updated, ones no longer in the export removed, all in one transaction;
the JSONL log is rewritten atomically. The same message ID seen in two exports is stored once (the export
with the newer latest message wins). Logs and the report contain counts only, never names, IDs or text.

    python3 scripts/parsers/discord_parser.py                # parse archive/discord/naitik
    python3 scripts/parsers/discord_parser.py --dry-run      # report counts without writing
"""
import argparse
import csv
import io
import json
import logging
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from urllib.parse import unquote, urlparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.append(os.path.join(REPO_ROOT, "scripts", "utils"))
from ingest_common import EGO_RAW_ID, flat_entry, load_identity  # noqa: E402
from database import SarthinkMemoryLayer  # noqa: E402

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

PLATFORM = "discord"
JSONL_OUTPUT = "discord_logs.jsonl"
DISCORD_EPOCH_MS = 1420070400000
MIN_EPOCH = 1420070400          # Discord launched in 2015; anything earlier is not a real message time
SNOWFLAKE_TOLERANCE = 86400     # a Timestamp this far from its snowflake time is reported as suspicious

# channel.json "type": names in current packages, integers in older ones.
TYPE_NAMES = {0: "GUILD_TEXT", 1: "DM", 2: "GUILD_VOICE", 3: "GROUP_DM", 4: "GUILD_CATEGORY", 5: "GUILD_ANNOUNCEMENT",
              10: "ANNOUNCEMENT_THREAD", 11: "PUBLIC_THREAD", 12: "PRIVATE_THREAD", 13: "GUILD_STAGE_VOICE", 15: "GUILD_FORUM",
              16: "GUILD_MEDIA"}
DCE_KINDS = {"DirectTextChat": "DM", "DirectGroupTextChat": "GROUP_DM"}
DCE_CONTENT_TYPES = {"Default", "Reply", "ThreadStarterMessage", "", None}

USER_MENTION_RE = re.compile(r"<@!?(\d{1,25})>")
ROLE_MENTION_RE = re.compile(r"<@&\d{1,25}>")
CHANNEL_MENTION_RE = re.compile(r"<#(\d{1,25})>")
EMOJI_RE = re.compile(r"<a?:(\w{1,64}):\d{1,25}>")
TIMESTAMP_TAG_RE = re.compile(r"<t:(-?\d{1,12})(?::[tTdDfFR])?>")
DISCRIMINATOR_RE = re.compile(r"#\d{1,4}$")
DM_INDEX_PREFIX = "Direct Message with "


class IdentityError(Exception):
    """The export does not say reliably which account is the owner's."""


class ExportError(Exception):
    """A file needed for a complete import could not be read."""


# ─── Field helpers ────────────────────────────────────────────────────────────

def snowflake_epoch(snowflake):
    """Creation time (epoch seconds, UTC) encoded in a Discord snowflake ID, or None."""
    try:
        value = int(snowflake)
    except (TypeError, ValueError):
        return None
    if (value >> 22) <= 0:     # no time component: not a real snowflake
        return None
    return ((value >> 22) + DISCORD_EPOCH_MS) // 1000


def parse_timestamp(value):
    """Epoch seconds (UTC) from a package or DiscordChatExporter timestamp, or None.

    '2024-05-01 12:34:56' (package, no zone marker) is UTC; ISO strings with 'Z' or an offset keep
    their offset. Fractions of a second are dropped.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().replace("Z", "+00:00")
    if " " in text and "T" not in text:
        text = text.replace(" ", "T", 1)
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    ts = int(dt.timestamp())
    return ts if ts >= MIN_EPOCH else None


def attachment_names(value):
    """File names of a message's attachments: package strings are space-separated URLs; DCE uses objects."""
    items = []
    if isinstance(value, str):
        items = value.split()
    elif isinstance(value, list):
        items = value
    names = []
    for item in items:
        if isinstance(item, dict):
            name = item.get("fileName") or item.get("filename") or ""
            if not name and item.get("url"):
                name = unquote(os.path.basename(urlparse(str(item["url"])).path))
        else:
            name = unquote(os.path.basename(urlparse(str(item)).path))
        name = name.strip()
        if name:
            names.append(name[:120])
    return names


def strip_discriminator(name):
    return DISCRIMINATOR_RE.sub("", str(name or "")).strip()


def person_name(obj):
    """Best display name from a relationship or user object: nickname, then global name, then username."""
    if not isinstance(obj, dict):
        return None
    user = obj.get("user") if isinstance(obj.get("user"), dict) else {}
    for value in (obj.get("nickname"), user.get("global_name"), obj.get("global_name"), user.get("username"),
                  obj.get("username"), obj.get("name")):
        if value and str(value).strip():
            return strip_discriminator(value)
    return None


def normalise_content(text, names, channels):
    """Replaces raw mention markup (which embeds IDs) with readable names or neutral placeholders."""
    if not text:
        return ""
    text = USER_MENTION_RE.sub(lambda m: "@" + (names.get(m.group(1)) or "someone"), text)
    text = ROLE_MENTION_RE.sub("@role", text)
    text = CHANNEL_MENTION_RE.sub(lambda m: "#" + (channels.get(m.group(1)) or "channel"), text)
    text = EMOJI_RE.sub(lambda m: f":{m.group(1)}:", text)

    def ts_tag(m):
        try:
            return datetime.fromtimestamp(int(m.group(1)), timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        except (OverflowError, OSError, ValueError):
            return m.group(0)
    return TIMESTAMP_TAG_RE.sub(ts_tag, text).strip()


def compose_content(text, attachments):
    """Stored message text: the message plus an attachment marker; '' when there is nothing to keep."""
    parts = [text] if text else []
    if attachments:
        parts.append(f"[Attachment: {', '.join(attachments)}]")
    return "\n".join(parts)


def channel_kind(raw_type):
    if isinstance(raw_type, int):
        return TYPE_NAMES.get(raw_type, f"TYPE_{raw_type}")
    if isinstance(raw_type, str) and raw_type.isdigit():
        return TYPE_NAMES.get(int(raw_type), f"TYPE_{raw_type}")
    return str(raw_type or "UNKNOWN").upper()


def thread_title(kind, channel, index_title, member_names):
    """Titles follow export_cosmograph.get_thread_group: 'DM x' -> DM room, 'Group x' -> group chat, else channel."""
    name = strip_discriminator(channel.get("name")) if channel.get("name") else ""
    if kind == "DM":
        other = member_names[0] if member_names else ""
        if not other and index_title and index_title.startswith(DM_INDEX_PREFIX):
            other = strip_discriminator(index_title[len(DM_INDEX_PREFIX):])
        return f"DM {other or 'unknown user'}"
    if kind == "GROUP_DM":
        label = name or ", ".join(member_names[:4]) or (index_title or "")
        return f"Group {label or 'DM'}"
    guild = channel.get("guild") if isinstance(channel.get("guild"), dict) else {}
    if name and guild.get("name"):
        prefix = "Thread" if kind.endswith("THREAD") else "#"
        sep = " " if prefix == "Thread" else ""
        return f"{prefix}{sep}{name} ({guild['name']})"
    if name:
        return f"#{name}"
    return index_title or "Unknown channel"


# ─── Readers ──────────────────────────────────────────────────────────────────

def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _find_child(parent, name):
    """Case-insensitive child lookup (older packages use lower-case folder names)."""
    try:
        for entry in os.listdir(parent):
            if entry.lower() == name.lower():
                return os.path.join(parent, entry)
    except OSError:
        pass
    return None


def find_package_roots(archive):
    """Folders that look like an official data package (they contain Messages/c<id>/channel.json)."""
    roots = []
    for dirpath, dirnames, _ in os.walk(archive):
        msgs = next((d for d in dirnames if d.lower() == "messages"), None)
        if msgs and any(os.path.isfile(os.path.join(dirpath, msgs, c, "channel.json")) for c in os.listdir(os.path.join(dirpath, msgs))):
            roots.append(dirpath)
            dirnames[:] = [d for d in dirnames if d.lower() != "messages"]
    return sorted(roots)


def find_dce_files(archive, package_roots):
    """DiscordChatExporter JSON files: any other *.json whose top level has 'channel' and 'messages'."""
    skip = tuple(os.path.join(r, "") for r in package_roots)
    out = []
    for dirpath, _, files in os.walk(archive):
        if os.path.join(dirpath, "").startswith(skip):
            continue
        out += [os.path.join(dirpath, f) for f in files if f.lower().endswith(".json")]
    return sorted(out)


def read_package_messages(channel_dir):
    """Rows of messages.json or messages.csv as dicts with ID/Timestamp/Contents/Attachments keys."""
    path = _find_child(channel_dir, "messages.json")
    if path:
        data = _read_json(path)
        if not isinstance(data, list):
            raise ExportError("messages.json is not a list")
        return data
    path = _find_child(channel_dir, "messages.csv")
    if path:
        with open(path, "r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(io.StringIO(f.read())))
    return []


def load_package(root, stats):
    """One data package -> {owner_id, owner_name, names, channels: [...], newest}."""
    account = _find_child(root, "Account")
    user_path = _find_child(account, "user.json") if account else None
    if not user_path:
        raise IdentityError("Account/user.json is missing, so the account owner can't be confirmed")
    user = _read_json(user_path)
    owner_id = str(user.get("id") or "").strip()
    if not owner_id.isdigit():
        raise IdentityError("Account/user.json has no numeric account id")

    names = {}
    for rel in user.get("relationships") or []:
        rid = str(rel.get("id") or (rel.get("user") or {}).get("id") or "")
        if rid and person_name(rel):
            names[rid] = person_name(rel)
    names[owner_id] = person_name(user) or "me"

    messages_dir = _find_child(root, "Messages")
    index = {}
    index_path = _find_child(messages_dir, "index.json")
    if index_path:
        index = {str(k): v for k, v in (_read_json(index_path) or {}).items() if isinstance(v, str)}
    channel_names = {}
    for cid, title in index.items():
        if not title.startswith(DM_INDEX_PREFIX):
            channel_names[cid] = title.split(",")[0].strip() or "channel"

    # First pass: DM partners missing from relationships are named by index.json, so a group DM read
    # before their DM already gets the name.
    for entry in os.listdir(messages_dir):
        channel_path = os.path.join(messages_dir, entry, "channel.json")
        try:
            channel = _read_json(channel_path) if os.path.isfile(channel_path) else {}
        except (OSError, ValueError):
            continue    # reported by the main pass
        others = [str(r) for r in channel.get("recipients") or [] if str(r) != owner_id]
        title = index.get(str(channel.get("id")))
        if channel_kind(channel.get("type")) == "DM" and len(others) == 1 and others[0] not in names \
                and title and title.startswith(DM_INDEX_PREFIX):
            names[others[0]] = strip_discriminator(title[len(DM_INDEX_PREFIX):])

    channels, newest = [], 0
    for entry in sorted(os.listdir(messages_dir)):
        channel_dir = os.path.join(messages_dir, entry)
        channel_path = os.path.join(channel_dir, "channel.json")
        if not os.path.isfile(channel_path):
            continue
        try:
            channel = _read_json(channel_path)
            rows = read_package_messages(channel_dir)
        except (OSError, ValueError, ExportError, csv.Error) as e:
            raise ExportError(f"unreadable channel folder ({type(e).__name__}); the import would be incomplete")
        cid = str(channel.get("id") or entry.lstrip("cC"))
        kind = channel_kind(channel.get("type"))
        recipients = [str(r) for r in channel.get("recipients") or []]
        if kind in ("DM", "GROUP_DM"):
            if recipients and owner_id not in recipients:
                raise IdentityError(f"a {kind} channel does not list the Account/user.json id among its recipients")
            if not recipients:
                stats["dm_without_recipients"] += 1
        members = [r for r in recipients if r != owner_id]
        index_title = index.get(cid)
        member_names = [names.get(m) or "unknown user" for m in members]
        stats[f"channels_{kind.lower()}"] += 1

        msgs = []
        for row in rows:
            msg = package_message(row, names, channel_names, stats)
            if msg:
                msg["author"] = None   # the owner
                msgs.append(msg)
                newest = max(newest, msg["ts"])
        channels.append({"channel_id": cid, "kind": kind,
                         "title": thread_title(kind, channel, index_title, member_names),
                         "members": [(m, names.get(m) or "unknown user") for m in members], "messages": msgs})
    return {"owner_id": owner_id, "names": names, "channels": channels, "newest": newest, "source": "package"}


def package_message(row, names, channel_names, stats):
    """One package row -> {id, ts, original_ts, content, parent} or None (reason counted in stats)."""
    stats["records_seen"] += 1
    if not isinstance(row, dict):
        stats["skipped_malformed_record"] += 1
        return None
    raw_id = str(row.get("ID", row.get("id", ""))).strip()
    if not raw_id.isdigit():
        stats["skipped_malformed_id"] += 1
        return None
    original = row.get("Timestamp", row.get("timestamp"))
    ts, snow = parse_timestamp(original), snowflake_epoch(raw_id)
    if ts is None:
        if snow is None:
            stats["skipped_bad_timestamp"] += 1
            return None
        ts = snow
        stats["timestamp_from_snowflake"] += 1
    elif snow is not None and abs(ts - snow) > SNOWFLAKE_TOLERANCE:
        stats["timestamp_far_from_snowflake"] += 1
    text = normalise_content(str(row.get("Contents", row.get("contents")) or ""), names, channel_names)
    files = attachment_names(row.get("Attachments", row.get("attachments")))
    content = compose_content(text, files)
    if not content:
        stats["skipped_empty"] += 1
        return None
    if files and not text:
        stats["attachment_only"] += 1
    return {"id": raw_id, "ts": ts, "original_ts": str(original or ""), "content": content, "parent": None}


def load_dce_file(path, owner_ids, owner_aliases, stats):
    """One DiscordChatExporter JSON file -> a channel dict like load_package's, or None if it isn't one."""
    try:
        data = _read_json(path)
    except (OSError, ValueError):
        raise ExportError("unreadable DiscordChatExporter file; the import would be incomplete")
    if not isinstance(data, dict) or not isinstance(data.get("channel"), dict) or not isinstance(data.get("messages"), list):
        stats["files_not_discord_export"] += 1
        return None
    channel = dict(data["channel"])
    if isinstance(data.get("guild"), dict) and not isinstance(channel.get("guild"), dict):
        channel["guild"] = data["guild"]
    kind = DCE_KINDS.get(channel.get("type"), "GUILD_TEXT")
    stats[f"channels_{kind.lower()}"] += 1
    msgs, others, newest = [], {}, 0
    for m in data["messages"]:
        stats["records_seen"] += 1
        if not isinstance(m, dict) or not str(m.get("id", "")).isdigit():
            stats["skipped_malformed_record"] += 1
            continue
        if m.get("type") not in DCE_CONTENT_TYPES:
            stats["skipped_system_message"] += 1
            continue
        author = m.get("author") if isinstance(m.get("author"), dict) else {}
        aid = str(author.get("id") or "")
        aname = person_name(author) or "unknown user"
        mine = aid in owner_ids or aid in owner_aliases or aname.lower() in owner_aliases
        ts = parse_timestamp(m.get("timestamp")) or snowflake_epoch(m["id"])
        if ts is None:
            stats["skipped_bad_timestamp"] += 1
            continue
        text = normalise_content(m.get("content") or "", {}, {})
        for embed in m.get("embeds") or []:
            if isinstance(embed, dict) and (embed.get("title") or embed.get("url")):
                text += f"\n[Embed: {embed.get('title', 'Link')} - {embed.get('url', '')}]"
        files = attachment_names(m.get("attachments"))
        content = compose_content(text.strip(), files)
        if not content:
            stats["skipped_empty"] += 1
            continue
        if files and not text.strip():
            stats["attachment_only"] += 1
        if m.get("timestampEdited"):
            stats["edited_kept_original_timestamp"] += 1
        ref = m.get("reference") if isinstance(m.get("reference"), dict) else {}
        parent = f"{PLATFORM}_{ref['messageId']}" if str(ref.get("messageId") or "").isdigit() else None
        if not mine and aid:
            others[aid] = aname
        msgs.append({"id": str(m["id"]), "ts": ts, "original_ts": str(m.get("timestamp") or ""), "content": content,
                     "parent": parent, "author": None if mine else (aid or "unknown", aname)})
        newest = max(newest, ts)
    if msgs and all(x["author"] is not None for x in msgs):
        stats["dce_files_without_owner_messages"] += 1
    member_names = list(others.values())
    title = thread_title(kind, channel, None, member_names if kind == "DM" else [])
    if kind == "DM" and not member_names and channel.get("name"):
        title = f"DM {strip_discriminator(channel['name'])}"
    return {"channel_id": str(channel.get("id") or os.path.basename(path)), "kind": kind, "title": title,
            "members": sorted(others.items()), "messages": msgs, "newest": newest}


def parse_archive(archive, owner_aliases=frozenset()):
    """Everything in `archive` -> (channels, stats). Raises IdentityError / ExportError instead of guessing."""
    stats = Counter()
    roots = find_package_roots(archive)
    exports = [load_package(root, stats) for root in roots]
    owner_ids = {e["owner_id"] for e in exports}
    if len(owner_ids) > 1 and not owner_ids <= set(owner_aliases):
        raise IdentityError(f"{len(owner_ids)} different Discord accounts found; list the ones that are yours "
                            "under aliases.discord in config/identity_map.json")
    for path in find_dce_files(archive, roots):
        ch = load_dce_file(path, owner_ids, set(owner_aliases), stats)
        if ch:
            exports.append({"channels": [ch], "newest": ch["newest"], "source": "dce"})
    stats["exports_package"] = len(roots)
    stats["exports_dce_files"] = sum(e["source"] == "dce" for e in exports)

    # Newest export first, so a message seen twice keeps its latest (edited) text.
    merged, seen = {}, {}
    for export in sorted(exports, key=lambda e: -e["newest"]):
        for ch in export["channels"]:
            target = merged.setdefault(ch["channel_id"], {**ch, "messages": []})
            known = dict(target["members"])
            known.update(dict(ch["members"]))
            target["members"] = sorted(known.items())
            for msg in ch["messages"]:
                prior = seen.get(msg["id"])
                if prior is not None:
                    stats["duplicate_identical" if prior == msg["content"] else "duplicate_older_version"] += 1
                    continue
                seen[msg["id"]] = msg["content"]
                target["messages"].append(msg)
    channels = [c for c in merged.values() if c["messages"]]
    stats["channels_without_messages"] = len(merged) - len(channels)
    for c in channels:
        c["messages"].sort(key=lambda m: (m["ts"], int(m["id"])))
    stats["messages_kept"] = sum(len(c["messages"]) for c in channels)
    stats["threads_kept"] = len(channels)
    return channels, stats


# ─── Database sync ────────────────────────────────────────────────────────────

def sync(db, channels, master_persona, jsonl_filename=JSONL_OUTPUT):
    """Makes the database's Discord rows match `channels` in one transaction; returns change counts."""
    cur, counts = db.cursor, Counter()
    cur.execute("BEGIN")
    try:
        owner_uid = db.get_or_create_user(PLATFORM, EGO_RAW_ID, master_persona)
        user_ids, thread_ids, desired, members = {owner_uid}, set(), {}, set()

        def user(raw_id, name):
            uid = db.get_or_create_user(PLATFORM, raw_id, name)
            cur.execute("UPDATE Users SET display_name = ? WHERE id = ? AND display_name IS NOT ?", (name, uid, name))
            counts["users_renamed"] += cur.rowcount
            user_ids.add(uid)
            return uid

        entries = []
        for ch in channels:
            tid = db.get_or_create_thread(PLATFORM, ch["channel_id"], ch["title"])
            cur.execute("UPDATE Threads SET title = ? WHERE id = ? AND title IS NOT ?", (ch["title"], tid, ch["title"]))
            counts["threads_renamed"] += cur.rowcount
            thread_ids.add(tid)
            for raw_id, name in ch["members"]:
                members.add((tid, user(raw_id, name)))
            for m in ch["messages"]:
                if m["author"] is None:
                    author_uid, author_raw, author_name = owner_uid, EGO_RAW_ID, master_persona
                else:
                    author_raw, author_name = m["author"]
                    author_uid = user(author_raw, author_name)
                    members.add((tid, author_uid))
                msg_id = f"{PLATFORM}_{m['id']}"
                desired[msg_id] = (tid, author_uid, m["ts"], m["content"], m["parent"])
                entry = flat_entry(msg_id, PLATFORM, ch["title"], author_name, author_raw, m["ts"], m["content"], m["parent"])
                entry["timestamp_original"] = m["original_ts"]
                entries.append(entry)

        existing = {r[0]: tuple(r[1:]) for r in cur.execute(
            "SELECT msg_id, thread_id, author_id, timestamp_utc, content, parent_msg_id FROM Messages "
            "WHERE msg_id LIKE 'discord\\_%' ESCAPE '\\'")}
        new = [(k,) + v for k, v in desired.items() if k not in existing]
        changed = [v + (k,) for k, v in desired.items() if k in existing and existing[k] != v]
        stale = [(k,) for k in existing if k not in desired]
        cur.executemany("INSERT INTO Messages (msg_id, thread_id, author_id, timestamp_utc, content, parent_msg_id) "
                        "VALUES (?, ?, ?, ?, ?, ?)", new)
        cur.executemany("UPDATE Messages SET thread_id = ?, author_id = ?, timestamp_utc = ?, content = ?, parent_msg_id = ? "
                        "WHERE msg_id = ?", changed)
        cur.executemany("DELETE FROM Messages WHERE msg_id = ?", stale)
        counts.update(messages_inserted=len(new), messages_updated=len(changed), messages_removed=len(stale),
                      messages_unchanged=len(desired) - len(new) - len(changed))

        old_members = set(cur.execute("SELECT tm.thread_id, tm.user_id FROM ThreadMembers tm JOIN Threads t ON t.id = tm.thread_id "
                                      "WHERE t.platform = ?", (PLATFORM,)))
        cur.executemany("DELETE FROM ThreadMembers WHERE thread_id = ? AND user_id = ?", sorted(old_members - members))
        cur.executemany("INSERT INTO ThreadMembers (thread_id, user_id) VALUES (?, ?)", sorted(members - old_members))
        counts["members"] = len(members)

        # Threads and people that no longer have anything in the export.
        cur.execute(f"DELETE FROM Threads WHERE platform = ? AND id NOT IN ({','.join('?' * len(thread_ids)) or 'NULL'}) "
                    "AND id NOT IN (SELECT thread_id FROM Messages)", [PLATFORM] + sorted(thread_ids))
        counts["threads_removed"] = cur.rowcount
        cur.execute(f"DELETE FROM Users WHERE platform = ? AND id NOT IN ({','.join('?' * len(user_ids))}) "
                    "AND id NOT IN (SELECT author_id FROM Messages WHERE author_id IS NOT NULL) "
                    "AND id NOT IN (SELECT user_id FROM ThreadMembers)", [PLATFORM] + sorted(user_ids))
        counts["users_removed"] = cur.rowcount
        db.conn.commit()
    except BaseException:
        db.conn.rollback()
        raise
    db._warm_caches()

    # JSONL: rewritten in full, then swapped in, so a crash never leaves a half-written log.
    path = os.path.join(db.jsonl_dir, jsonl_filename)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    os.replace(tmp, path)
    counts["threads"] = len(thread_ids)
    counts["messages"] = len(desired)
    counts["people"] = len(user_ids) - 1
    return counts


def main(argv=None):
    parser = argparse.ArgumentParser(description="Parse Discord exports (official data package or DiscordChatExporter JSON)")
    parser.add_argument("--archive", default=os.path.join(REPO_ROOT, "archive", "discord", "naitik"),
                        help="Folder containing the staged export")
    parser.add_argument("--db", default=None, help="SQLite path (default: processed_data/db/sarthink_memory.db)")
    parser.add_argument("--logs", default=None, help="JSONL log dir (default: processed_data/logs)")
    parser.add_argument("--dry-run", action="store_true", help="Parse and report counts without writing anything")
    args = parser.parse_args(argv)

    if not os.path.isdir(args.archive):
        logging.error("Archive folder not found: %s", args.archive)
        return 1
    aliases, master_persona = load_identity(PLATFORM)
    try:
        channels, stats = parse_archive(args.archive, frozenset(aliases))
    except (IdentityError, ExportError) as e:
        logging.error("Discord import stopped, nothing written: %s", e)
        return 2
    report = {"parse": dict(sorted(stats.items()))}
    if not args.dry_run:
        if not channels:
            logging.error("No Discord messages found under the archive folder; nothing written.")
            return 1
        db = SarthinkMemoryLayer(db_path=args.db, jsonl_dir=args.logs)
        try:
            report["sync"] = dict(sorted(sync(db, channels, master_persona).items()))
        finally:
            db.close()
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
