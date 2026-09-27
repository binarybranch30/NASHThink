import os
import re
import glob
import zipfile
import logging
import sys
from datetime import datetime

# Resolve paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

# Inject utils path for database import
sys.path.append(os.path.join(REPO_ROOT, "scripts", "utils"))
from ingest_common import EGO_RAW_ID, FLUSH_EVERY, load_identity, parse_args, open_fresh_db, flat_entry

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

PLATFORM = "whatsapp"
JSONL_OUTPUT = "whatsapp_logs.jsonl"

# ─── Line format ──────────────────────────────────────────────────────────────
# Android: "12/31/23, 9:15 PM - Name: text"      iOS: "[31/12/23, 21:15:03] Name: text"
# Also tolerates '.'/'-' date separators, 4-digit years, and the narrow no-break
# space (U+202F) newer exports put before AM/PM.
LINE_RE = re.compile(
    r'^[‎‏]?\[?(\d{1,4})[/.\-](\d{1,2})[/.\-](\d{2,4}),?\s+'
    r'(\d{1,2}):(\d{2})(?::(\d{2}))?[\s ]*([AaPp]\.?\s?[Mm]\.?)?'
    r'(?:\]\s|\s-\s)(.*)$'
)

MEDIA_MARKERS = ('<media omitted>', 'image omitted', 'video omitted', 'audio omitted',
                 'sticker omitted', 'gif omitted', 'document omitted', '<attached:')
SKIP_CONTENT = {'this message was deleted', 'you deleted this message', 'null'}


def detect_date_order(first_fields):
    """Given (a, b) date pairs from a file, decide between 'ymd', 'dmy' and 'mdy'.
    Falls back to dmy when every date is ambiguous (both parts <= 12)."""
    if any(a > 31 for a, _ in first_fields):
        return 'ymd'
    if any(a > 12 for a, _ in first_fields):
        return 'dmy'
    if any(b > 12 for _, b in first_fields):
        return 'mdy'
    return 'dmy'


def split_messages(lines):
    """Groups raw lines into (match_groups, body) records; continuation lines join the previous body."""
    records = []
    for line in lines:
        line = line.rstrip('\r\n')
        m = LINE_RE.match(line)
        if m:
            records.append([m.groups()[:7], m.group(8)])
        elif records:
            records[-1][1] += '\n' + line
    return records


def to_epoch(groups, order):
    a, b, c, hh, mm, ss, ampm = groups
    a, b, c = int(a), int(b), int(c)
    if order == 'ymd':
        year, month, day = a, b, c
    elif order == 'mdy':
        month, day, year = a, b, c
    else:
        day, month, year = a, b, c
    if year < 100:
        year += 2000
    hour = int(hh)
    if ampm:
        is_pm = ampm.lower().startswith('p')
        hour = hour % 12 + (12 if is_pm else 0)
    try:
        # Exports carry no timezone: interpret as the local time of this machine.
        return int(datetime(year, month, day, hour, int(mm), int(ss or 0)).timestamp())
    except ValueError:
        return 0


def split_sender(body):
    """'Name: text' → (name, text). System notices (no 'Name: ') → (None, body)."""
    if ': ' not in body:
        return None, body
    name, text = body.split(': ', 1)
    return name.strip('‎‏ '), text


def clean_content(text):
    stripped = text.strip().strip('‎')
    lowered = stripped.lower()
    if lowered in SKIP_CONTENT:
        return ''
    if any(marker in lowered for marker in MEDIA_MARKERS):
        return '[Media]'
    return stripped


def chat_name_from_path(path):
    """'WhatsApp Chat with Alice.txt' / 'WhatsApp Chat - Alice.zip' → 'Alice'."""
    base = os.path.splitext(os.path.basename(path))[0]
    for prefix in ('WhatsApp Chat with ', 'WhatsApp Chat - '):
        if base.startswith(prefix):
            return base[len(prefix):]
    return base


def iter_chat_files(archive_dir):
    """Yields (chat_name, lines) for every .txt export, including ones still inside a .zip."""
    for path in sorted(glob.glob(os.path.join(archive_dir, '**', '*.txt'), recursive=True)):
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            yield chat_name_from_path(path), f.readlines()
    for path in sorted(glob.glob(os.path.join(archive_dir, '**', '*.zip'), recursive=True)):
        with zipfile.ZipFile(path) as zf:
            for member in zf.namelist():
                if member.endswith('.txt'):
                    text = zf.read(member).decode('utf-8', errors='replace')
                    yield chat_name_from_path(path), text.splitlines()


def chat_kind(records, ego_names):
    """'Group' when more than one other person wrote in the chat, else 'DM'."""
    others = {s.lower() for s, _ in (split_sender(body) for _, body in records) if s and s.lower() not in ego_names}
    return 'Group' if len(others) > 1 else 'DM'


def slugify(name):
    return re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_') or 'chat'


def process_whatsapp():
    args = parse_args("whatsapp", "Parse WhatsApp chat exports (.txt / .zip)")
    ego_names, master_persona = load_identity(PLATFORM)
    ego_names |= {'you', master_persona.lower()}

    if not os.path.isdir(args.archive):
        logging.error(f"Archive folder {args.archive} not found.")
        return

    db = open_fresh_db(args, [PLATFORM], JSONL_OUTPUT)
    total, chats = 0, 0

    for chat_name, lines in iter_chat_files(args.archive):
        records = split_messages(lines)
        if not records:
            continue
        chats += 1
        order = detect_date_order([(int(g[0]), int(g[1])) for g, _ in records])
        slug = slugify(chat_name)
        thread_title = f"{chat_kind(records, ego_names)} {chat_name}"
        thread_db_id = db.get_or_create_thread(PLATFORM, slug, thread_title)
        logging.info(f"Parsing: {chat_name} ({len(records)} lines, date order {order})")

        previous_msg_id = None
        for idx, (groups, body) in enumerate(records):
            sender, text = split_sender(body)
            if sender is None:
                continue  # system notice
            content = clean_content(text)
            if not content:
                continue

            utc_epoch = to_epoch(groups, order)
            is_ego = sender.lower() in ego_names
            raw_author = EGO_RAW_ID if is_ego else sender
            display = master_persona if is_ego else sender
            author_db_id = db.get_or_create_user(PLATFORM, raw_author, display)

            global_id = f"{PLATFORM}_{slug}_{utc_epoch}_{idx}"
            entry = flat_entry(global_id, PLATFORM, thread_title, display, raw_author,
                               utc_epoch, content, previous_msg_id)
            db.insert_message(global_id, thread_db_id, author_db_id, utc_epoch, content,
                              previous_msg_id, entry, JSONL_OUTPUT, commit_now=False)
            previous_msg_id = global_id
            total += 1
            if total % FLUSH_EVERY == 0:
                db.commit()

    db.close()
    logging.info(f"WhatsApp parse complete. {chats} chats, {total} messages.")


if __name__ == "__main__":
    process_whatsapp()
