"""Unit tests for the format helpers in the WhatsApp / ChatGPT / Claude / Google parsers.
Run: python3 scripts/tests/test_new_parsers.py"""
import os
import sys
import unittest
from datetime import datetime
from email.message import EmailMessage

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.append(os.path.join(REPO_ROOT, "scripts", "parsers"))
sys.path.append(os.path.join(REPO_ROOT, "scripts", "utils"))

import whatsapp_parser as wa
import chatgpt_parser as gpt
import claude_parser as cl
import google_parser as gg
import twitter_parser as tw


class WhatsAppTests(unittest.TestCase):
    def test_android_and_ios_lines(self):
        lines = [
            "12/31/23, 9:15 PM - Alice: hello\n",
            "[31/12/23, 21:15:03] Bob: hi there\n",
            "31/12/2023, 21:16 - Alice: multi\n",
            "line message\n",
        ]
        records = wa.split_messages(lines)
        self.assertEqual(len(records), 3)
        self.assertEqual(wa.split_sender(records[0][1]), ("Alice", "hello"))
        self.assertEqual(wa.split_sender(records[1][1]), ("Bob", "hi there"))
        self.assertEqual(records[2][1], "Alice: multi\nline message")

    def test_narrow_nbsp_before_ampm(self):
        records = wa.split_messages(["1/2/24, 9:05 AM - Alice: hey"])
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0][0][6], "AM")

    def test_date_order_detection(self):
        self.assertEqual(wa.detect_date_order([(12, 31)]), 'mdy')
        self.assertEqual(wa.detect_date_order([(31, 12)]), 'dmy')
        self.assertEqual(wa.detect_date_order([(2023, 12)]), 'ymd')
        self.assertEqual(wa.detect_date_order([(1, 2)]), 'dmy')

    def test_to_epoch_pm_and_midnight(self):
        pm = wa.to_epoch(('12', '31', '23', '9', '15', None, 'PM'), 'mdy')
        self.assertEqual(pm, int(datetime(2023, 12, 31, 21, 15).timestamp()))
        midnight = wa.to_epoch(('1', '1', '24', '12', '05', None, 'AM'), 'dmy')
        self.assertEqual(midnight, int(datetime(2024, 1, 1, 0, 5).timestamp()))

    def test_system_and_media(self):
        self.assertEqual(wa.split_sender("Messages and calls are end-to-end encrypted.")[0], None)
        self.assertEqual(wa.clean_content("<Media omitted>"), "[Media]")
        self.assertEqual(wa.clean_content("This message was deleted"), "")

    def test_group_or_dm_title(self):
        dm = wa.split_messages(["1/2/24, 9:00 PM - Aarav: hi", "1/2/24, 9:01 PM - Rohan: yo"])
        group = wa.split_messages(["1/2/24, 9:00 PM - Aarav: hi", "1/2/24, 9:01 PM - Rohan: yo",
                                   "1/2/24, 9:02 PM - Meera: hey", "1/2/24, 9:03 PM - Rohan added Kabir"])
        self.assertEqual(wa.chat_kind(dm, {"aarav"}), "DM")
        self.assertEqual(wa.chat_kind(group, {"aarav"}), "Group")

    def test_chat_name(self):
        self.assertEqual(wa.chat_name_from_path("/x/WhatsApp Chat with Alice.txt"), "Alice")
        self.assertEqual(wa.chat_name_from_path("/x/WhatsApp Chat - Fam Group.zip"), "Fam Group")


class ChatGPTTests(unittest.TestCase):
    def test_active_branch_skips_abandoned_edit(self):
        convo = {
            "current_node": "c",
            "mapping": {
                "root": {"id": "root", "message": None, "parent": None, "children": ["a"]},
                "a": {"id": "a", "parent": "root", "children": ["old", "b"], "message": {}},
                "old": {"id": "old", "parent": "a", "children": [], "message": {}},
                "b": {"id": "b", "parent": "a", "children": ["c"], "message": {}},
                "c": {"id": "c", "parent": "b", "children": [], "message": {}},
            },
        }
        self.assertEqual([n["id"] for n in gpt.active_branch(convo)], ["root", "a", "b", "c"])

    def test_message_text(self):
        msg = {"content": {"content_type": "multimodal_text",
                           "parts": [{"content_type": "image_asset_pointer"}, "what is this?"]}}
        self.assertEqual(gpt.message_text(msg), "[Image]\nwhat is this?")
        hidden = {"metadata": {"is_visually_hidden_from_conversation": True},
                  "content": {"content_type": "text", "parts": ["sys"]}}
        self.assertEqual(gpt.message_text(hidden), "")


class ClaudeTests(unittest.TestCase):
    def test_content_blocks_preferred_over_text(self):
        msg = {"text": "old", "content": [{"type": "text", "text": "new"}, {"type": "tool_use"}],
               "attachments": [{"file_name": "notes.pdf"}]}
        self.assertEqual(cl.message_text(msg), "new\n[Attachment: notes.pdf]")

    def test_iso_timestamp(self):
        self.assertEqual(cl.parse_iso_timestamp("1970-01-01T00:01:00.000000Z"), 60)


class GoogleTests(unittest.TestCase):
    def test_strip_quoted_reply(self):
        body = "Sounds good.\n\nOn Mon, Jan 1, 2024 at 10:00 AM Bob <b@x.com> wrote:\n> earlier"
        self.assertEqual(gg.strip_quoted_reply(body), "Sounds good.")

    def test_email_body_html_fallback(self):
        msg = EmailMessage()
        msg.set_content("<p>Hello<br>world</p>", subtype="html")
        self.assertEqual(gg.email_body(msg), "Hello\nworld")

    def test_clean_subject(self):
        self.assertEqual(gg.clean_subject("Re: Fwd: RE: Plans"), "Plans")

    def test_chat_date(self):
        self.assertEqual(gg.parse_chat_date("Thursday, January 1, 1970 at 12:01:00 AM UTC"), 60)
        self.assertEqual(gg.parse_chat_date("Thursday, 1 January 1970 at 00:01:00 UTC"), 60)

    def test_youtube_text(self):
        self.assertEqual(gg.decode_youtube_text('{"text":"nice "},{"text":"video"}'), "nice video")


class TwitterTests(unittest.TestCase):
    def write(self, name, text):
        import tempfile
        d = tempfile.mkdtemp()
        path = os.path.join(d, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return d, path

    def test_js_wrapper_with_and_without_semicolon(self):
        for tail in ("]", "];", "];\n"):
            _, path = self.write("tweets.js", 'window.YTD.tweets.part0 = [\n  {"tweet": {"id_str": "1"}}\n' + tail)
            self.assertEqual(tw.strip_js_wrapper(path), [{"tweet": {"id_str": "1"}}], repr(tail))

    def test_reply_chains_share_a_thread(self):
        index = {"3": {"in_reply_to_status_id_str": "2"}, "2": {"in_reply_to_status_id_str": "1"}, "1": {},
                 "5": {"in_reply_to_status_id_str": "999"}, "7": {"in_reply_to_status_id_str": "7"}}
        self.assertEqual([tw.thread_root(t, index) for t in ("1", "2", "3")], ["1", "1", "1"])
        self.assertEqual(tw.thread_root("5", index), "999", "a reply to an outside tweet groups under that tweet")
        self.assertEqual(tw.thread_root("7", index), "7", "a self-loop terminates")

    def test_owner_is_never_guessed(self):
        import tempfile
        root, saved = tempfile.mkdtemp(), tw.REPO_ROOT
        try:
            tw.REPO_ROOT = root
            self.assertIsNone(tw.load_account_identity())
            with open(os.path.join(root, "account.js"), "w") as f:
                f.write('window.YTD.account.part0 = [{"account": {"accountId": "9", "username": "demo", "accountDisplayName": "Demo"}}];')
            self.assertEqual(tw.load_account_identity(), ("demo", "Demo", "9"))
        finally:
            tw.REPO_ROOT = saved


class GoogleActivityTests(unittest.TestCase):
    def test_gemini_folder_names(self):
        import json
        import tempfile

        class Ingest:
            def __init__(self):
                self.added = []

            def user(self, *a, **k):
                return "u"

            def add(self, key, *a):
                self.added.append(key)
                return key

        for folder in ("Gemini Apps", "Gemini"):
            root = tempfile.mkdtemp()
            os.makedirs(os.path.join(root, "My Activity", folder))
            with open(os.path.join(root, "My Activity", folder, "MyActivity.json"), "w") as f:
                json.dump([{"title": "Prompted something", "time": "2025-01-01T10:00:00.000Z"}], f)
            ing = Ingest()
            gg.ingest_activity(ing, root)
            self.assertEqual(len(ing.added), 1, folder)


if __name__ == "__main__":
    unittest.main()
