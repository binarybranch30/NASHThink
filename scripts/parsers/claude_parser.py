import os
import json
import glob
import logging
import sys
from datetime import datetime, timezone

# Resolve paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

# Inject utils path for database import
sys.path.append(os.path.join(REPO_ROOT, "scripts", "utils"))
from ingest_common import EGO_RAW_ID, FLUSH_EVERY, load_identity, parse_args, open_fresh_db, flat_entry

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

PLATFORM = "claude"
JSONL_OUTPUT = "claude_logs.jsonl"
ASSISTANT_NAME = "Claude"


def parse_iso_timestamp(ts_str):
    """'2024-05-01T12:34:56.789000Z' (or with +00:00) → UTC epoch int."""
    if not ts_str:
        return 0
    try:
        dt = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp())
    except ValueError:
        return 0


def message_text(msg):
    """Newer exports put text in `content` blocks; older ones only have `text`."""
    blocks = [b.get('text', '') for b in msg.get('content') or []
              if isinstance(b, dict) and b.get('type') == 'text']
    text = '\n'.join(b for b in blocks if b) or msg.get('text') or ''
    names = [a.get('file_name') for a in (msg.get('attachments') or []) + (msg.get('files') or [])
             if isinstance(a, dict) and a.get('file_name')]
    if names:
        text += '\n' + ' '.join(f"[Attachment: {n}]" for n in names)
    return text.strip()


def process_claude():
    args = parse_args("claude", "Parse a Claude data export (conversations.json)")
    _, master_persona = load_identity(PLATFORM)

    files = sorted(glob.glob(os.path.join(args.archive, '**', 'conversations.json'), recursive=True))
    if not files:
        logging.error(f"No conversations.json found under {args.archive}.")
        return

    db = open_fresh_db(args, [PLATFORM], JSONL_OUTPUT)
    ego_db_id = db.get_or_create_user(PLATFORM, EGO_RAW_ID, master_persona)
    bot_db_id = db.get_or_create_user(PLATFORM, ASSISTANT_NAME, ASSISTANT_NAME)
    total, convos = 0, 0

    for filepath in files:
        with open(filepath, 'r', encoding='utf-8') as f:
            conversations = json.load(f)

        for convo in conversations:
            convo_id = convo.get('uuid')
            if not convo_id:
                continue
            title = f"AI Chat: {convo.get('name') or 'Untitled'}"
            thread_db_id = db.get_or_create_thread(PLATFORM, convo_id, title)
            convos += 1

            known_ids = set()
            previous_msg_id = None
            for msg in convo.get('chat_messages') or []:
                content = message_text(msg)
                if not content or not msg.get('uuid'):
                    continue

                is_ego = msg.get('sender') == 'human'
                author_raw = EGO_RAW_ID if is_ego else ASSISTANT_NAME
                author_display = master_persona if is_ego else ASSISTANT_NAME
                utc_epoch = parse_iso_timestamp(msg.get('created_at'))

                global_id = f"{PLATFORM}_{msg['uuid']}"
                # Prefer the explicit tree link when it points at a message we kept.
                parent_uuid = msg.get('parent_message_uuid')
                parent_id = f"{PLATFORM}_{parent_uuid}" if parent_uuid else None
                if parent_id not in known_ids:
                    parent_id = previous_msg_id

                entry = flat_entry(global_id, PLATFORM, title, author_display, author_raw,
                                   msg.get('created_at'), content, parent_id)
                db.insert_message(global_id, thread_db_id, ego_db_id if is_ego else bot_db_id,
                                  utc_epoch, content, parent_id, entry, JSONL_OUTPUT,
                                  commit_now=False)
                known_ids.add(global_id)
                previous_msg_id = global_id
                total += 1
                if total % FLUSH_EVERY == 0:
                    db.commit()

    db.close()
    logging.info(f"Claude parse complete. {convos} conversations, {total} messages.")


if __name__ == "__main__":
    process_claude()
