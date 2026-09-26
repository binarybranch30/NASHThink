import os
import re
import csv
import json
import glob
import html
import mailbox
import logging
import sys
from collections import Counter
from datetime import datetime, timezone
from email.header import decode_header, make_header
from email.utils import parseaddr, parsedate_to_datetime
from html.parser import HTMLParser

# Resolve paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

# Inject utils path for database import
sys.path.append(os.path.join(REPO_ROOT, "scripts", "utils"))
from ingest_common import EGO_RAW_ID, FLUSH_EVERY, load_identity, parse_args, open_fresh_db, flat_entry

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

PLATFORM = "google"
JSONL_OUTPUT = "google_logs.jsonl"

# Gmail labels whose threads are skipped entirely (noise, not conversation).
SKIP_GMAIL_LABELS = {'spam', 'trash', 'category promotions'}
MAX_EMAIL_CHARS = 4000
# My Activity products to ingest: folder name → (thread title prefix, assistant display name or None)
ACTIVITY_PRODUCTS = {
    'Search':      ('Searches', None),
    'Gemini Apps': ('Gemini', 'Gemini'),
}

# ─── Generic helpers ──────────────────────────────────────────────────────────

class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)

    def handle_starttag(self, tag, attrs):
        if tag in ('br', 'p', 'div', 'li', 'tr'):
            self.parts.append('\n')


def html_to_text(markup):
    parser = _TextExtractor()
    parser.feed(markup or '')
    text = html.unescape(''.join(parser.parts))
    return re.sub(r'\n\s*\n+', '\n\n', text).strip()


def parse_iso(ts_str):
    try:
        return int(datetime.fromisoformat(ts_str.replace('Z', '+00:00')).timestamp())
    except (ValueError, AttributeError):
        return 0


class Ingest:
    """Small wrapper so each product section stays focused on format quirks, not bookkeeping."""

    def __init__(self, db, master_persona):
        self.db = db
        self.master_persona = master_persona
        self.total = 0
        self.ego_db_id = db.get_or_create_user(PLATFORM, EGO_RAW_ID, master_persona)

    def user(self, raw_id, display, is_ego=False):
        if is_ego:
            return self.ego_db_id, EGO_RAW_ID, self.master_persona
        return self.db.get_or_create_user(PLATFORM, raw_id, display), raw_id, display

    def add(self, msg_id, thread_key, thread_title, author, epoch, content, parent=None):
        author_db_id, author_raw, author_display = author
        thread_db_id = self.db.get_or_create_thread(PLATFORM, thread_key, thread_title)
        global_id = f"{PLATFORM}_{msg_id}"
        entry = flat_entry(global_id, PLATFORM, thread_title, author_display, author_raw,
                           epoch, content, parent)
        self.db.insert_message(global_id, thread_db_id, author_db_id, epoch, content, parent,
                               entry, JSONL_OUTPUT, commit_now=False)
        self.total += 1
        if self.total % FLUSH_EVERY == 0:
            self.db.commit()
        return global_id

# ─── Gmail (Takeout/Mail/*.mbox) ──────────────────────────────────────────────

def decode_mime_header(value):
    if not value:
        return ''
    try:
        return str(make_header(decode_header(value)))
    except Exception:
        return str(value)


def clean_subject(subject):
    return re.sub(r'^\s*((re|fwd?|aw|wg)\s*:\s*)+', '', subject, flags=re.I).strip() or '(no subject)'


def strip_quoted_reply(text):
    """Drops '>' quoted lines and everything after an 'On <date>, X wrote:' marker."""
    kept = []
    for line in text.splitlines():
        if re.match(r'^\s*On .{5,200} wrote:\s*$', line) or line.strip() == '-----Original Message-----':
            break
        if line.lstrip().startswith('>'):
            continue
        kept.append(line)
    return re.sub(r'\n{3,}', '\n\n', '\n'.join(kept)).strip()


def email_body(msg):
    plain, markup = None, None
    for part in msg.walk() if msg.is_multipart() else [msg]:
        if part.get_content_maintype() == 'multipart' or part.get_filename():
            continue
        ctype = part.get_content_type()
        if ctype not in ('text/plain', 'text/html'):
            continue
        payload = part.get_payload(decode=True) or b''
        text = payload.decode(part.get_content_charset() or 'utf-8', errors='replace')
        if ctype == 'text/plain' and plain is None:
            plain = text
        elif ctype == 'text/html' and markup is None:
            markup = text
    body = plain if plain is not None else html_to_text(markup or '')
    return strip_quoted_reply(body)[:MAX_EMAIL_CHARS]


def clean_msg_id(value):
    return (value or '').strip().strip('<>').strip()


def ingest_gmail(ingest, takeout_dir, ego_emails):
    """Returns the set of owner addresses seen in Delivered-To, for use by later sections."""
    mbox_files = glob.glob(os.path.join(takeout_dir, 'Mail', '*.mbox'))
    delivered_to = Counter()
    for path in mbox_files:
        logging.info(f"Gmail: reading {os.path.basename(path)} (large mailboxes take a while)")
        for msg in mailbox.mbox(path, create=False):
            labels = {l.strip().lower() for l in decode_mime_header(msg.get('X-Gmail-Labels')).split(',')}
            if labels & SKIP_GMAIL_LABELS:
                continue
            owner_addr = parseaddr(msg.get('Delivered-To', ''))[1].lower()
            if owner_addr:
                delivered_to[owner_addr] += 1

            message_id = clean_msg_id(msg.get('Message-ID'))
            if not message_id:
                continue
            thread_key = f"mail_{msg.get('X-GM-THRID') or message_id}"
            subject = clean_subject(decode_mime_header(msg.get('Subject')))

            from_name, from_addr = parseaddr(decode_mime_header(msg.get('From')))
            from_addr = from_addr.lower()
            is_ego = from_addr in ego_emails or (owner_addr and from_addr == owner_addr)

            try:
                epoch = int(parsedate_to_datetime(msg.get('Date')).timestamp())
            except Exception:
                epoch = 0
            content = email_body(msg)
            if not content:
                continue
            reply_to = clean_msg_id(msg.get('In-Reply-To'))
            parent = f"{PLATFORM}_mail_{reply_to}" if reply_to else None

            ingest.add(f"mail_{message_id}", thread_key, f"Email: {subject}",
                       ingest.user(from_addr or 'unknown', from_name or from_addr or 'unknown', is_ego),
                       epoch, content, parent)
    return {addr for addr, _ in delivered_to.most_common(3)}

# ─── Google Chat (Takeout/Google Chat/Groups/*/messages.json) ─────────────────

CHAT_DATE_FORMATS = (
    "%A, %B %d, %Y at %I:%M:%S %p",
    "%A, %d %B %Y at %H:%M:%S",
    "%A, %B %d, %Y at %H:%M:%S",
    "%A, %d %B %Y at %I:%M:%S %p",
)


def parse_chat_date(value):
    """Google Chat uses human dates like 'Tuesday, January 3, 2023 at 10:15:31 AM UTC'."""
    if not value:
        return 0
    value = value.replace(' ', ' ').strip()
    value = re.sub(r'\s+(UTC|GMT)$', '', value)
    for fmt in CHAT_DATE_FORMATS:
        try:
            return int(datetime.strptime(value, fmt).replace(tzinfo=timezone.utc).timestamp())
        except ValueError:
            continue
    return 0


def ingest_google_chat(ingest, takeout_dir, ego_emails):
    group_dirs = sorted(glob.glob(os.path.join(takeout_dir, 'Google Chat', 'Groups', '*')))
    for group_dir in group_dirs:
        messages_path = os.path.join(group_dir, 'messages.json')
        if not os.path.exists(messages_path):
            continue
        folder = os.path.basename(group_dir)  # "DM abc123" or "Space xyz"
        info = {}
        info_path = os.path.join(group_dir, 'group_info.json')
        if os.path.exists(info_path):
            with open(info_path, 'r', encoding='utf-8') as f:
                info = json.load(f)
        others = [m.get('name') or m.get('email') for m in info.get('members', [])
                  if (m.get('email') or '').lower() not in ego_emails]
        if folder.startswith('DM'):
            title = f"DM {', '.join(filter(None, others)) or folder}"
        else:
            title = f"Group {info.get('name') or folder}"

        with open(messages_path, 'r', encoding='utf-8') as f:
            messages = json.load(f).get('messages', [])

        previous = None
        for idx, msg in enumerate(messages):
            text = (msg.get('text') or '').strip()
            if not text and msg.get('attached_files'):
                text = '[Attachment]'
            if not text:
                continue
            creator = msg.get('creator') or {}
            email_addr = (creator.get('email') or '').lower()
            author = ingest.user(email_addr or creator.get('name') or 'unknown',
                                 creator.get('name') or email_addr or 'unknown',
                                 email_addr in ego_emails)
            msg_key = msg.get('message_id') or f"{folder}_{idx}"
            previous = ingest.add(f"chat_{msg_key}", f"chat_{folder}", title, author,
                                  parse_chat_date(msg.get('created_date')), text, previous)

# ─── YouTube comments (Takeout/YouTube*/comments/*.csv) ───────────────────────

def find_column(fieldnames, *needles):
    for name in fieldnames or []:
        lowered = name.lower()
        if all(n in lowered for n in needles):
            return name
    return None


def decode_youtube_text(raw):
    """Comment text is stored as JSON fragments: {"text":"hi "},{"text":"there"}."""
    if not raw:
        return ''
    try:
        fragments = json.loads(f'[{raw}]')
        return ''.join(f.get('text', '') for f in fragments if isinstance(f, dict)).strip()
    except json.JSONDecodeError:
        return raw.strip()


def ingest_youtube_comments(ingest, takeout_dir):
    csv_files = glob.glob(os.path.join(takeout_dir, 'YouTube*', 'comments', '*.csv'))
    for path in csv_files:
        with open(path, 'r', encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames
            id_col = find_column(cols, 'comment id') or find_column(cols, 'id')
            video_col = find_column(cols, 'video id')
            post_col = find_column(cols, 'post id')
            time_col = find_column(cols, 'timestamp') or find_column(cols, 'time')
            text_col = find_column(cols, 'comment text') or find_column(cols, 'text')
            parent_col = find_column(cols, 'parent comment id')
            if not (id_col and text_col):
                logging.warning(f"YouTube: unrecognised columns in {path}: {cols}")
                continue

            rows = list(reader)
            own_ids = {row[id_col] for row in rows}
            for row in rows:
                content = decode_youtube_text(row.get(text_col))
                if not content:
                    continue
                target = (video_col and row.get(video_col)) or (post_col and row.get(post_col)) or 'unknown'
                parent_raw = parent_col and row.get(parent_col)
                parent = f"{PLATFORM}_yt_{parent_raw}" if parent_raw in own_ids else None
                ingest.add(f"yt_{row[id_col]}", f"yt_{target}", f"Comment on video {target}",
                           ingest.user(None, None, is_ego=True),
                           parse_iso(row.get(time_col) or ''), content, parent)

# ─── My Activity: Search + Gemini (Takeout/My Activity/*/MyActivity.json) ─────

def ingest_activity(ingest, takeout_dir):
    for folder, (title_prefix, assistant) in ACTIVITY_PRODUCTS.items():
        path = os.path.join(takeout_dir, 'My Activity', folder, 'MyActivity.json')
        if not os.path.exists(path):
            if os.path.exists(os.path.join(os.path.dirname(path), 'MyActivity.html')):
                logging.warning(f"{folder}: only HTML activity found — re-export My Activity in JSON format.")
            continue
        with open(path, 'r', encoding='utf-8') as f:
            entries = json.load(f)

        bot = ingest.user(assistant, assistant) if assistant else None
        for idx, item in enumerate(sorted(entries, key=lambda e: e.get('time', ''))):
            prompt = (item.get('title') or '').strip()
            if not prompt:
                continue
            day = (item.get('time') or '')[:10] or 'unknown'
            epoch = parse_iso(item.get('time') or '')
            thread_key = f"act_{folder}_{day}"
            title = f"{title_prefix} {day}"
            key = f"act_{slug(folder)}_{epoch}_{idx}"
            prompt_id = ingest.add(key, thread_key, title, ingest.user(None, None, is_ego=True),
                                   epoch, prompt)
            if bot:
                response = '\n\n'.join(html_to_text(h.get('html', ''))
                                       for h in item.get('safeHtmlItem') or [])
                if response:
                    ingest.add(f"{key}_reply", thread_key, title, bot, epoch, response, prompt_id)


def slug(value):
    return re.sub(r'[^a-z0-9]+', '_', value.lower()).strip('_')

# ─── Entry point ──────────────────────────────────────────────────────────────

def find_takeout_dir(archive_dir):
    """Accept either archive/google/Takeout/... or the Takeout contents directly in archive/google."""
    nested = os.path.join(archive_dir, 'Takeout')
    return nested if os.path.isdir(nested) else archive_dir


def process_google():
    args = parse_args("google", "Parse a Google Takeout export (Gmail, Chat, YouTube comments, activity)")
    ego_emails, master_persona = load_identity(PLATFORM)

    if not os.path.isdir(args.archive):
        logging.error(f"Archive folder {args.archive} not found.")
        return
    takeout_dir = find_takeout_dir(args.archive)

    db = open_fresh_db(args, [PLATFORM], JSONL_OUTPUT)
    ingest = Ingest(db, master_persona)

    # Gmail runs first: its Delivered-To headers reveal the owner's address for Chat.
    ego_emails |= ingest_gmail(ingest, takeout_dir, ego_emails)
    logging.info(f"Gmail done ({ingest.total} messages so far). Owner addresses: {sorted(ego_emails)}")
    ingest_google_chat(ingest, takeout_dir, ego_emails)
    logging.info(f"Google Chat done ({ingest.total} so far).")
    ingest_youtube_comments(ingest, takeout_dir)
    logging.info(f"YouTube comments done ({ingest.total} so far).")
    ingest_activity(ingest, takeout_dir)

    db.close()
    logging.info(f"Google parse complete. {ingest.total} messages.")


if __name__ == "__main__":
    process_google()
