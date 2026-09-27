"""Synthetic tests for scripts/parsers/discord_parser.py and scripts/utils/stage_discord_export.py.
No real export is read. Run: python3 scripts/tests/test_discord_parser.py"""
import csv
import io
import json
import os
import sqlite3
import sys
import tempfile
import unittest
import zipfile
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.append(os.path.join(REPO_ROOT, "scripts", "parsers"))
sys.path.append(os.path.join(REPO_ROOT, "scripts", "utils"))

import discord_parser as dp  # noqa: E402
import stage_discord_export as stage  # noqa: E402
from database import SarthinkMemoryLayer  # noqa: E402

OWNER = "100000000000000001"
ALICE = "100000000000000002"
BOB = "100000000000000003"


def ts(y, mo, d, h=0, mi=0, s=0):
    return int(datetime(y, mo, d, h, mi, s, tzinfo=timezone.utc).timestamp())


def snowflake(epoch, seq=0):
    return str(((epoch * 1000 - dp.DISCORD_EPOCH_MS) << 22) + seq)


def pkg_row(epoch, text, attachments="", seq=0, stamp=None):
    return {"ID": int(snowflake(epoch, seq)),
            "Timestamp": stamp if stamp is not None else datetime.fromtimestamp(epoch, timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "Contents": text, "Attachments": attachments}


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def make_package(root, owner=OWNER, channels=None, index=None, relationships=None, user=True):
    root = Path(root)
    if user:
        write_json(root / "Account" / "user.json", {
            "id": owner, "username": "owner_user", "global_name": "Owner",
            "relationships": relationships if relationships is not None else [
                {"id": ALICE, "type": 1, "nickname": None, "user": {"id": ALICE, "username": "alice_u", "global_name": "Alice"}}]})
    write_json(root / "Messages" / "index.json", index or {})
    for cid, (channel, rows) in (channels or {}).items():
        write_json(root / "Messages" / f"c{cid}" / "channel.json", channel)
        if isinstance(rows, str):
            (root / "Messages" / f"c{cid}" / "messages.json").write_text(rows, encoding="utf-8")
        else:
            write_json(root / "Messages" / f"c{cid}" / "messages.json", rows)
    return root


def standard_package(root):
    return make_package(root, channels={
        "201": ({"id": "201", "type": "DM", "recipients": [OWNER, ALICE]},
                [pkg_row(ts(2024, 1, 5, 23, 30), "hi <@" + ALICE + "> see <#301>"),
                 pkg_row(ts(2024, 1, 6, 0, 10), "", "https://cdn.example/attachments/1/2/photo%20one.png"),
                 pkg_row(ts(2024, 1, 6, 0, 11), "   "),
                 pkg_row(ts(2024, 1, 6, 1, 0), "later <:wave:123456789012345678>")]),
        "202": ({"id": "202", "type": "GROUP_DM", "name": "Trip crew", "recipients": [OWNER, ALICE, BOB]},
                [pkg_row(ts(2024, 2, 1, 12), "group hello")]),
        "301": ({"id": "301", "type": "GUILD_TEXT", "name": "general", "guild": {"id": "900", "name": "Test Server"}},
                [pkg_row(ts(2024, 3, 1, 9), "channel post"), "not a dict", {"ID": "abc", "Timestamp": "x", "Contents": "bad id"}]),
        "302": ({"id": "302", "type": "PUBLIC_THREAD", "name": "plans", "guild": {"id": "900", "name": "Test Server"}},
                [pkg_row(ts(2024, 3, 2, 9), "thread post")]),
        "303": ({"id": "303", "type": "GUILD_TEXT"}, [pkg_row(ts(2024, 3, 3, 9), "left server post")]),
    }, index={"201": "Direct Message with alice_u#0", "202": None, "301": "general, Test Server",
              "302": "plans, Test Server", "303": "Unknown channel in Unknown server"})


class HelperTests(unittest.TestCase):
    def test_package_timestamp_is_utc(self):
        self.assertEqual(dp.parse_timestamp("2024-01-05 23:30:00"), ts(2024, 1, 5, 23, 30))

    def test_iso_offsets_and_fractions(self):
        self.assertEqual(dp.parse_timestamp("2025-01-26T04:05:31.195+05:30"), ts(2025, 1, 25, 22, 35, 31))
        self.assertEqual(dp.parse_timestamp("2025-01-26T04:05:31Z"), ts(2025, 1, 26, 4, 5, 31))
        self.assertEqual(dp.parse_timestamp("2021-05-01 12:00:00.123000+00:00"), ts(2021, 5, 1, 12))

    def test_bad_timestamps(self):
        for bad in (None, "", "yesterday", "1999-01-01 00:00:00", 12345):
            self.assertIsNone(dp.parse_timestamp(bad))

    def test_snowflake_roundtrip(self):
        t = ts(2023, 7, 1, 8, 9, 10)
        self.assertEqual(dp.snowflake_epoch(snowflake(t, 5)), t)
        self.assertIsNone(dp.snowflake_epoch("nope"))
        self.assertIsNone(dp.snowflake_epoch("0"))

    def test_attachment_names(self):
        self.assertEqual(dp.attachment_names("https://cdn.x/a/b/one.png https://cdn.x/a/c/two%20b.pdf?ex=1"),
                         ["one.png", "two b.pdf"])
        self.assertEqual(dp.attachment_names([{"fileName": "f.txt", "url": "u"}, {"url": "https://x/y/z.gif"}]), ["f.txt", "z.gif"])
        self.assertEqual(dp.attachment_names(""), [])

    def test_mentions_hide_ids(self):
        out = dp.normalise_content(f"<@{ALICE}> <@!{BOB}> <@&55555> <#301> <#999999> <a:dance:123456789012345678> <t:1704499200:R>",
                                   {ALICE: "Alice"}, {"301": "general"})
        self.assertEqual(out, "@Alice @someone @role #general #channel :dance: 2024-01-06 00:00 UTC")
        self.assertNotIn(BOB, out)

    def test_compose_content(self):
        self.assertEqual(dp.compose_content("", []), "")
        self.assertEqual(dp.compose_content("", ["a.png"]), "[Attachment: a.png]")
        self.assertEqual(dp.compose_content("hey", ["a.png", "b.pdf"]), "hey\n[Attachment: a.png, b.pdf]")

    def test_titles_match_graph_groups(self):
        sys.path.append(os.path.join(REPO_ROOT, "scripts", "utils"))
        from export_cosmograph import get_thread_group
        cases = [("DM", {}, "Direct Message with bob#1234", []), ("GROUP_DM", {"name": "Crew"}, None, ["A"]),
                 ("GUILD_TEXT", {"name": "dm-chat", "guild": {"name": "S"}}, None, [])]
        groups = [get_thread_group("discord", dp.thread_title(*c)) for c in cases]
        self.assertEqual(groups, ["discord_dm_group", "discord_group_chat", "discord_public_thread"])
        self.assertEqual(dp.thread_title("DM", {}, "Direct Message with bob#1234", []), "DM bob")
        self.assertEqual(dp.thread_title("PUBLIC_THREAD", {"name": "t", "guild": {"name": "S"}}, None, []), "Thread t (S)")


class ParseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_dm_group_and_channels(self):
        standard_package(self.dir / "pkg")
        channels, stats = dp.parse_archive(self.dir)
        by_id = {c["channel_id"]: c for c in channels}
        self.assertEqual(set(by_id), {"201", "202", "301", "302", "303"})
        dm = by_id["201"]
        self.assertEqual(dm["kind"], "DM")
        self.assertEqual(dm["title"], "DM Alice")                       # relationship name beats index text
        self.assertEqual(dm["members"], [(ALICE, "Alice")])
        self.assertEqual([m["content"] for m in dm["messages"]],
                         ["hi @Alice see #general", "[Attachment: photo one.png]", "later :wave:"])
        self.assertTrue(all(m["author"] is None for c in channels for m in c["messages"]))   # all the owner's
        self.assertEqual(by_id["202"]["title"], "Group Trip crew")
        self.assertEqual(sorted(n for _, n in by_id["202"]["members"]), ["Alice", "unknown user"])
        self.assertEqual(by_id["301"]["title"], "#general (Test Server)")
        self.assertEqual(by_id["302"]["title"], "Thread plans (Test Server)")
        self.assertEqual(by_id["303"]["title"], "Unknown channel in Unknown server")
        self.assertEqual(stats["skipped_empty"], 1)
        self.assertEqual(stats["attachment_only"], 1)
        self.assertEqual(stats["skipped_malformed_record"], 1)
        self.assertEqual(stats["skipped_malformed_id"], 1)
        self.assertEqual(stats["messages_kept"], 7)

    def test_group_before_dm_gets_index_name(self):
        make_package(self.dir, relationships=[], channels={
            "100": ({"id": "100", "type": "GROUP_DM", "recipients": [OWNER, BOB, ALICE]}, [pkg_row(ts(2024, 1, 1), "g")]),
            "900": ({"id": "900", "type": "DM", "recipients": [OWNER, BOB]}, [pkg_row(ts(2024, 1, 2), "d")])},
            index={"900": "Direct Message with bobby#0"})
        channels, _ = dp.parse_archive(self.dir)
        group = next(c for c in channels if c["channel_id"] == "100")
        self.assertIn((BOB, "bobby"), group["members"])
        self.assertEqual(group["title"], "Group bobby, unknown user")

    def test_timestamps_preserved_and_ordered(self):
        standard_package(self.dir / "pkg")
        channels, _ = dp.parse_archive(self.dir)
        dm = next(c for c in channels if c["channel_id"] == "201")
        self.assertEqual([m["ts"] for m in dm["messages"]], [ts(2024, 1, 5, 23, 30), ts(2024, 1, 6, 0, 10), ts(2024, 1, 6, 1)])
        self.assertEqual(dm["messages"][0]["original_ts"], "2024-01-05 23:30:00")

    def test_snowflake_fallback_and_mismatch(self):
        t = ts(2024, 4, 1, 10)
        make_package(self.dir, channels={"201": ({"id": "201", "type": "DM", "recipients": [OWNER, ALICE]}, [
            pkg_row(t, "no stamp", stamp="garbage"),
            pkg_row(t, "far off", seq=1, stamp="2020-01-01 00:00:00"),
            {"ID": "12", "Timestamp": "garbage", "Contents": "unrecoverable"}])})
        channels, stats = dp.parse_archive(self.dir)
        msgs = channels[0]["messages"]
        self.assertEqual(msgs[1]["ts"], t)
        self.assertEqual(stats["timestamp_from_snowflake"], 1)
        self.assertEqual(stats["timestamp_far_from_snowflake"], 1)
        self.assertEqual(stats["skipped_bad_timestamp"], 1)

    def test_identity_requires_account_file(self):
        make_package(self.dir, user=False, channels={"201": ({"id": "201", "type": "DM", "recipients": [OWNER, ALICE]},
                                                            [pkg_row(ts(2024, 1, 1), "x")])})
        with self.assertRaises(dp.IdentityError):
            dp.parse_archive(self.dir)

    def test_identity_mismatch_stops_import(self):
        make_package(self.dir, channels={"201": ({"id": "201", "type": "DM", "recipients": [ALICE, BOB]},
                                                 [pkg_row(ts(2024, 1, 1), "x")])})
        with self.assertRaises(dp.IdentityError):
            dp.parse_archive(self.dir)

    def test_two_accounts_need_identity_map(self):
        for name, owner in (("a", OWNER), ("b", BOB)):
            make_package(self.dir / name, owner=owner, channels={"201": ({"id": "201", "type": "GUILD_TEXT"},
                                                                         [pkg_row(ts(2024, 1, 1), "x")])})
        with self.assertRaises(dp.IdentityError):
            dp.parse_archive(self.dir)
        channels, _ = dp.parse_archive(self.dir, frozenset({OWNER, BOB}))
        self.assertEqual(len(channels), 1)

    def test_duplicate_exports_keep_newest(self):
        t = ts(2024, 1, 1)
        row_old, row_new = pkg_row(t, "original"), pkg_row(t, "edited later")
        make_package(self.dir / "old", channels={"201": ({"id": "201", "type": "GUILD_TEXT"}, [row_old])})
        make_package(self.dir / "new", channels={"201": ({"id": "201", "type": "GUILD_TEXT"},
                                                         [row_new, pkg_row(ts(2024, 2, 1), "newer message")])})
        make_package(self.dir / "copy", channels={"201": ({"id": "201", "type": "GUILD_TEXT"}, [row_old])})
        channels, stats = dp.parse_archive(self.dir)
        self.assertEqual([m["content"] for m in channels[0]["messages"]], ["edited later", "newer message"])
        self.assertEqual(stats["duplicate_older_version"], 2)

    def test_unreadable_file_stops_import(self):
        make_package(self.dir, channels={"201": ({"id": "201", "type": "GUILD_TEXT"}, "{not json")})
        with self.assertRaises(dp.ExportError):
            dp.parse_archive(self.dir)

    def test_legacy_csv_package(self):
        root = make_package(self.dir, channels={})
        cdir = root / "messages" / "c401"
        cdir.mkdir(parents=True)
        (root / "Messages" / "index.json").unlink()
        (root / "Messages").rmdir()
        write_json(cdir / "channel.json", {"id": "401", "type": 1, "recipients": [OWNER, ALICE]})
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=["ID", "Timestamp", "Contents", "Attachments"])
        w.writeheader()
        w.writerow({"ID": snowflake(ts(2020, 6, 1, 5)), "Timestamp": "2020-06-01 05:00:00.123000+00:00",
                    "Contents": "old, with comma", "Attachments": ""})
        (cdir / "messages.csv").write_text(buf.getvalue(), encoding="utf-8")
        channels, _ = dp.parse_archive(self.dir)
        self.assertEqual(channels[0]["kind"], "DM")
        self.assertEqual(channels[0]["messages"][0]["content"], "old, with comma")
        self.assertEqual(channels[0]["messages"][0]["ts"], ts(2020, 6, 1, 5))

    def test_chat_exporter_format(self):
        dce = {"guild": {"id": "0", "name": "Direct Messages"},
               "channel": {"id": "501", "type": "DirectTextChat", "name": "carol"},
               "messages": [
                   {"id": snowflake(ts(2022, 1, 1)), "type": "Default", "timestamp": "2022-01-01T05:30:00+05:30",
                    "timestampEdited": "2022-01-02T00:00:00+00:00", "content": "from me",
                    "author": {"id": OWNER, "name": "owner_user"}, "attachments": [], "embeds": []},
                   {"id": snowflake(ts(2022, 1, 1, 0, 1)), "type": "Reply", "timestamp": "2022-01-01T00:01:00+00:00",
                    "content": "from carol", "author": {"id": "7777777", "name": "carol", "nickname": "Carol"},
                    "reference": {"messageId": snowflake(ts(2022, 1, 1))}},
                   {"id": snowflake(ts(2022, 1, 1, 0, 2)), "type": "RecipientAdd", "timestamp": "2022-01-01T00:02:00+00:00",
                    "content": "", "author": {"id": "7777777", "name": "carol"}}]}
        write_json(self.dir / "dce" / "Direct Messages - carol.json", dce)
        channels, stats = dp.parse_archive(self.dir)       # no package: owner unknown -> nobody is "me"
        self.assertTrue(all(m["author"] is not None for m in channels[0]["messages"]))
        self.assertEqual(stats["dce_files_without_owner_messages"], 1)
        self.assertEqual(stats["skipped_system_message"], 1)
        channels, stats = dp.parse_archive(self.dir, frozenset({OWNER}))
        msgs = channels[0]["messages"]
        self.assertIsNone(msgs[0]["author"])
        self.assertEqual(msgs[0]["ts"], ts(2022, 1, 1))     # original, not the edit time
        self.assertEqual(msgs[1]["author"], ("7777777", "Carol"))
        self.assertEqual(msgs[1]["parent"], f"discord_{snowflake(ts(2022, 1, 1))}")
        self.assertEqual(channels[0]["title"], "DM Carol")
        self.assertEqual(stats["edited_kept_original_timestamp"], 1)


class SyncTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.db_path = str(self.dir / "mem.db")
        self.logs = str(self.dir / "logs")
        standard_package(self.dir / "pkg")
        # Pre-existing data from another platform must survive untouched.
        db = SarthinkMemoryLayer(db_path=self.db_path, jsonl_dir=self.logs)
        tid = db.get_or_create_thread("reddit", "r1", "r/test")
        uid = db.get_or_create_user("reddit", "someone", "someone")
        db.insert_message("reddit_1", tid, uid, 1700000000, "keep me")
        db.close()

    def tearDown(self):
        self.tmp.cleanup()

    def run_sync(self, archive=None):
        channels, _ = dp.parse_archive(archive or self.dir / "pkg")
        db = SarthinkMemoryLayer(db_path=self.db_path, jsonl_dir=self.logs)
        try:
            return dp.sync(db, channels, "Me")
        finally:
            db.close()

    def rows(self, sql, *params):
        with sqlite3.connect(self.db_path) as c:
            return c.execute(sql, params).fetchall()

    def test_insert_then_rerun_is_noop(self):
        first = self.run_sync()
        self.assertEqual(first["messages_inserted"], 7)
        ids_before = self.rows("SELECT id, platform_thread_id FROM Threads WHERE platform='discord' ORDER BY id")
        users_before = self.rows("SELECT id, raw_id FROM Users WHERE platform='discord' ORDER BY id")
        second = self.run_sync()
        self.assertEqual((second["messages_inserted"], second["messages_updated"], second["messages_removed"]), (0, 0, 0))
        self.assertEqual(second["messages_unchanged"], 7)
        self.assertEqual(ids_before, self.rows("SELECT id, platform_thread_id FROM Threads WHERE platform='discord' ORDER BY id"))
        self.assertEqual(users_before, self.rows("SELECT id, raw_id FROM Users WHERE platform='discord' ORDER BY id"))
        self.assertEqual(self.rows("SELECT content FROM Messages WHERE msg_id='reddit_1'"), [("keep me",)])
        self.assertEqual(self.rows("PRAGMA integrity_check"), [("ok",)])
        self.assertEqual(self.rows("PRAGMA foreign_key_check"), [])
        with open(os.path.join(self.logs, dp.JSONL_OUTPUT), encoding="utf-8") as f:
            lines = [json.loads(l) for l in f]
        self.assertEqual(len(lines), 7)
        self.assertEqual(lines[0]["timestamp_original"], "2024-01-05 23:30:00")

    def test_owner_and_members(self):
        self.run_sync()
        authors = self.rows("SELECT DISTINCT u.raw_id, u.display_name FROM Messages m JOIN Users u ON u.id = m.author_id "
                            "WHERE m.msg_id LIKE 'discord%'")
        self.assertEqual(authors, [("me", "Me")])
        members = self.rows("SELECT t.platform_thread_id, u.raw_id FROM ThreadMembers tm JOIN Threads t ON t.id = tm.thread_id "
                            "JOIN Users u ON u.id = tm.user_id ORDER BY 1, 2")
        self.assertEqual(members, [("201", ALICE), ("202", ALICE), ("202", BOB)])
        stamps = self.rows("SELECT timestamp_utc FROM Messages WHERE msg_id LIKE 'discord%' ORDER BY timestamp_utc LIMIT 1")
        self.assertEqual(stamps, [(ts(2024, 1, 5, 23, 30),)])

    def test_changed_and_removed_messages(self):
        self.run_sync()
        msgs = self.dir / "pkg" / "Messages" / "c201" / "messages.json"
        rows = json.loads(msgs.read_text())
        rows[0]["Contents"] = "edited text"
        del rows[3]
        msgs.write_text(json.dumps(rows))
        counts = self.run_sync()
        self.assertEqual((counts["messages_updated"], counts["messages_removed"], counts["messages_inserted"]), (1, 1, 0))
        self.assertEqual(self.rows("SELECT COUNT(*) FROM Messages WHERE content='edited text'"), [(1,)])

    def test_failed_parse_writes_nothing(self):
        before = self.rows("SELECT COUNT(*) FROM Messages")
        (self.dir / "pkg" / "Account" / "user.json").unlink()
        with self.assertRaises(dp.IdentityError):
            self.run_sync()
        self.assertEqual(before, self.rows("SELECT COUNT(*) FROM Messages"))

    def test_cli_dry_run_and_real_run(self):
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.assertEqual(dp.main(["--archive", str(self.dir / "pkg"), "--db", self.db_path, "--logs", self.logs, "--dry-run"]), 0)
        self.assertEqual(self.rows("SELECT COUNT(*) FROM Messages WHERE msg_id LIKE 'discord%'"), [(0,)])
        with redirect_stdout(buf):
            self.assertEqual(dp.main(["--archive", str(self.dir / "pkg"), "--db", self.db_path, "--logs", self.logs]), 0)
        self.assertEqual(self.rows("SELECT COUNT(*) FROM Messages WHERE msg_id LIKE 'discord%'"), [(7,)])
        self.assertNotIn("hi @Alice", buf.getvalue())      # the report holds counts, not text


class StagingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def make_zip(self, members, symlink=None):
        path = self.dir / "package.zip"
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
            for name, data in members.items():
                zf.writestr(name, data)
            if symlink:
                info = zipfile.ZipInfo(symlink)
                info.external_attr = (0o120777 << 16)
                zf.writestr(info, "/etc/passwd")
        return path

    def test_extracts_only_needed_and_minimises_account(self):
        user = {"id": OWNER, "username": "u", "email": "secret@example.com", "phone": "+1", "ip": "1.2.3.4",
                "relationships": [{"id": ALICE, "type": 1, "nickname": None, "user": {"id": ALICE, "username": "a", "avatar": "x"}}]}
        z = self.make_zip({"README.txt": "hi", "Account/user.json": json.dumps(user), "Account/avatar.png": "png",
                           "Messages/index.json": "{}", "Messages/c1/channel.json": "{}", "Messages/c1/messages.json": "[]",
                           "Activity/analytics/events.json": "x" * 1000, "Servers/index.json": "{}"})
        dest = self.dir / "out"
        report = stage.stage(z, dest, expected_size=z.stat().st_size, log=lambda *a: None)
        self.assertTrue(report["crc_ok"])
        files = sorted(str(p.relative_to(dest)) for p in dest.rglob("*") if p.is_file())
        self.assertEqual(files, ["Account/user.json", "Messages/c1/channel.json", "Messages/c1/messages.json",
                                 "Messages/index.json", "README.txt", "Servers/index.json"])
        kept = json.loads((dest / "Account" / "user.json").read_text())
        self.assertNotIn("email", kept)
        self.assertNotIn("ip", kept)
        self.assertNotIn("avatar", kept["relationships"][0]["user"])
        self.assertEqual(kept["id"], OWNER)
        self.assertEqual(oct((dest / "README.txt").stat().st_mode & 0o777), "0o600")
        self.assertEqual(oct((dest / "Messages").stat().st_mode & 0o777), "0o700")

    def test_rejects_unsafe_members(self):
        for members, link in (({"../evil.txt": "x"}, None), ({"/abs/Messages/index.json": "x"}, None),
                              ({"README.txt": "x"}, "Messages/link")):
            z = self.make_zip(members, link)
            with self.assertRaises(stage.StagingError):
                stage.stage(z, self.dir / "out2", log=lambda *a: None)
            self.assertFalse(any((self.dir / "out2").rglob("*")) if (self.dir / "out2").exists() else False)

    def test_size_mismatch(self):
        z = self.make_zip({"README.txt": "x"})
        with self.assertRaises(stage.StagingError):
            stage.stage(z, self.dir / "out3", expected_size=1, log=lambda *a: None)


if __name__ == "__main__":
    unittest.main()
