import os
import json
import glob
import logging
import sys

# Resolve paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

# Inject utils path for database import
sys.path.append(os.path.join(REPO_ROOT, "scripts", "utils"))
from ingest_common import EGO_RAW_ID, FLUSH_EVERY, load_identity, parse_args, open_fresh_db, flat_entry

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

PLATFORM = "chatgpt"
JSONL_OUTPUT = "chatgpt_logs.jsonl"
ASSISTANT_NAME = "ChatGPT"
KEPT_ROLES = ('user', 'assistant')


def active_branch(conversation):
    """ChatGPT stores each conversation as a tree (edits/regenerations create branches).
    Walk from `current_node` up to the root to recover the branch the user actually saw."""
    mapping = conversation.get('mapping') or {}
    node_id = conversation.get('current_node')
    if node_id not in mapping:
        # Older exports without current_node: follow the last child from the root.
        roots = [nid for nid, n in mapping.items() if not n.get('parent')]
        node_id = roots[0] if roots else None
        while node_id and mapping[node_id].get('children'):
            node_id = mapping[node_id]['children'][-1]

    chain = []
    seen = set()
    while node_id and node_id in mapping and node_id not in seen:
        seen.add(node_id)
        chain.append(mapping[node_id])
        node_id = mapping[node_id].get('parent')
    chain.reverse()
    return chain


def message_text(message):
    """Extracts the human-readable text from a mapping node's message, or '' to skip it."""
    if not message:
        return ''
    if (message.get('metadata') or {}).get('is_visually_hidden_from_conversation'):
        return ''
    content = message.get('content') or {}
    ctype = content.get('content_type')
    if ctype in ('text', 'multimodal_text'):
        pieces = []
        for part in content.get('parts') or []:
            if isinstance(part, str):
                pieces.append(part)
            elif isinstance(part, dict):
                pieces.append('[Image]' if 'image' in str(part.get('content_type', '')) else '[Attachment]')
        return '\n'.join(p for p in pieces if p).strip()
    if ctype == 'code':
        return (content.get('text') or '').strip()
    return ''


def find_conversation_files(archive_dir):
    return sorted(glob.glob(os.path.join(archive_dir, '**', 'conversations.json'), recursive=True))


def process_chatgpt():
    args = parse_args("chatgpt", "Parse a ChatGPT data export (conversations.json)")
    _, master_persona = load_identity(PLATFORM)

    files = find_conversation_files(args.archive)
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
            convo_id = convo.get('conversation_id') or convo.get('id')
            if not convo_id:
                continue
            title = f"AI Chat: {convo.get('title') or 'Untitled'}"
            thread_db_id = db.get_or_create_thread(PLATFORM, convo_id, title)
            convos += 1

            previous_msg_id = None
            for node in active_branch(convo):
                message = node.get('message')
                role = ((message or {}).get('author') or {}).get('role')
                if role not in KEPT_ROLES:
                    continue
                content = message_text(message)
                if not content:
                    continue

                utc_epoch = int(message.get('create_time') or convo.get('create_time') or 0)
                is_ego = role == 'user'
                author_raw = EGO_RAW_ID if is_ego else ASSISTANT_NAME
                author_display = master_persona if is_ego else ASSISTANT_NAME

                global_id = f"{PLATFORM}_{message.get('id') or node.get('id')}"
                entry = flat_entry(global_id, PLATFORM, title, author_display, author_raw,
                                   utc_epoch, content, previous_msg_id)
                db.insert_message(global_id, thread_db_id, ego_db_id if is_ego else bot_db_id,
                                  utc_epoch, content, previous_msg_id, entry, JSONL_OUTPUT,
                                  commit_now=False)
                previous_msg_id = global_id
                total += 1
                if total % FLUSH_EVERY == 0:
                    db.commit()

    db.close()
    logging.info(f"ChatGPT parse complete. {convos} conversations, {total} messages.")


if __name__ == "__main__":
    process_chatgpt()
