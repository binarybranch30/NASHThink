"""Shared plumbing for the newer parsers (whatsapp, chatgpt, claude, google)."""
import os
import json
import logging
import argparse

from database import SarthinkMemoryLayer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

# Every parser records the archive owner under this raw_id so downstream
# identity resolution can treat it the same way across platforms.
EGO_RAW_ID = "me"
FLUSH_EVERY = 1000


def load_identity(platform):
    """Returns (lowercased alias set for `platform`, master_persona) from identity_map.json."""
    map_path = os.path.join(REPO_ROOT, "config", "identity_map.json")
    master_persona = "User"
    aliases = set()
    if os.path.exists(map_path):
        try:
            with open(map_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            master_persona = data.get("master_persona", "User")
            aliases = {str(a).lower() for a in data.get("aliases", {}).get(platform, []) if a}
        except Exception as e:
            logging.warning(f"Could not load identity_map.json: {e}")
    return aliases, master_persona


def parse_args(default_archive_subdir, description):
    """Common CLI: --archive / --db / --logs, all defaulting to the standard repo layout."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--archive", default=os.path.join(REPO_ROOT, "archive", default_archive_subdir),
                        help="Folder containing the raw export")
    parser.add_argument("--db", default=None, help="SQLite path (default: processed_data/db/sarthink_memory.db)")
    parser.add_argument("--logs", default=None, help="JSONL log dir (default: processed_data/logs)")
    return parser.parse_args()


def open_fresh_db(args, platforms, jsonl_filename):
    """Open the memory layer and purge previous data for these platforms so re-runs are idempotent."""
    db = SarthinkMemoryLayer(db_path=args.db, jsonl_dir=args.logs)
    jsonl_path = os.path.join(db.jsonl_dir, jsonl_filename)
    if os.path.exists(jsonl_path):
        os.remove(jsonl_path)
        logging.info(f"Cleared old {jsonl_filename} for fresh reparse.")
    for platform in platforms:
        db.purge_platform(platform)
    return db


def flat_entry(log_id, platform, thread_name, author, author_id, timestamp, content, parent):
    """The JSONL row shape shared by every parser."""
    return {
        "log_id": log_id,
        "platform": platform,
        "thread_name": thread_name,
        "author": author,
        "author_id": author_id,
        "timestamp_utc": timestamp,
        "content": content,
        "is_reply_to": parent,
    }
