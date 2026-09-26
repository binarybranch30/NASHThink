import os
import glob
import logging
import csv
import json
try:
    import ijson
except ImportError:
    ijson = None
import gc
import re
import sys
import argparse
from datetime import datetime, timezone
from collections import defaultdict

# Resolve paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

# Inject utils path for database import
sys.path.append(os.path.join(REPO_ROOT, "scripts", "utils"))
from database import SarthinkMemoryLayer

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

PLATFORM = "reddit"
JSONL_OUTPUT = "reddit_logs.jsonl"


def parse_reddit_timestamp(ts_str):
    if not ts_str:
        return 0
    s = str(ts_str).replace(' UTC', '').strip()
    try:
        dt = datetime.fromisoformat(s).replace(tzinfo=timezone.utc)
        return int(dt.timestamp())
    except Exception:
        pass
    try:
        dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        return int(dt.timestamp())
    except Exception:
        pass
    return 0


def load_reddit_ego():
    """Load ego username from config/identity_map.json."""
    map_path = os.path.join(REPO_ROOT, "config", "identity_map.json")
    ego_name = "GooseMuch1099"
    if os.path.exists(map_path):
        try:
            with open(map_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            aliases = data.get("aliases", {}).get(PLATFORM, [])
            if aliases:
                ego_name = aliases[0]
        except Exception as e:
            logging.warning(f"Could not load identity_map.json: {e}")
    return ego_name


def resolve_chat_room_titles(chat_csv_path, ego_name):
    """Pre-scan chat_history.csv to produce clean, human-readable titles for every room."""
    room_titles = {}
    room_subreddits = {}
    room_participants = defaultdict(set)
    ego_lower = ego_name.lower()

    logging.info("Pre-scanning chat rooms for title resolution...")
    with open(chat_csv_path, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.DictReader(f)
        for row in reader:
            url = row.get('channel_url', '')
            room_id = url.split('/')[-1] if '/' in url else url
            cname = row.get('channel_name', '').strip()
            if cname and room_id not in room_titles:
                room_titles[room_id] = cname
            sub = row.get('subreddit', '').strip()
            if sub and room_id not in room_subreddits:
                room_subreddits[room_id] = sub
            u = row.get('username', '').replace('/u/', '').strip()
            if u and u.lower() != ego_lower and not u.startswith('['):
                room_participants[room_id].add(u)

    resolved = {}
    for room_id, parts in room_participants.items():
        if room_id in room_titles:
            resolved[room_id] = room_titles[room_id]
        elif len(parts) == 1:
            resolved[room_id] = f"DM: {list(parts)[0]}"
        elif len(parts) > 1:
            plist = sorted(parts)
            resolved[room_id] = f"Group: {', '.join(plist[:2])} (+{len(parts)-2})" if len(parts) > 2 else f"Group: {plist[0]}, {plist[1]}"
        elif room_id in room_subreddits:
            resolved[room_id] = f"r/{room_subreddits[room_id]} Chat"
        else:
            resolved[room_id] = f"DM: {room_id}"

    # Also handle rooms with only ego messages or deleted users
    for room_id, cname in room_titles.items():
        if room_id not in resolved:
            resolved[room_id] = cname

    logging.info(f"Resolved titles for {len(resolved)} chat rooms.")
    return resolved


def process_chats(db, chat_csv_path, room_titles, counters):
    if not os.path.exists(chat_csv_path):
        logging.warning(f"Reddit chat file {chat_csv_path} not found. Skipping chats.")
        return

    logging.info(f"Processing Reddit Chats from {chat_csv_path}...")
    chat_count = 0

    with open(chat_csv_path, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_msg_id = row.get('message_id', 'unknown')
            global_msg_id = f"{PLATFORM}_{raw_msg_id}"

            room_url = row.get('channel_url', 'unknown_chat')
            room_id = room_url.split('/')[-1] if '/' in room_url else room_url
            thread_title = room_titles.get(room_id, f"DM: {room_id}")
            thread_db_id = db.get_or_create_thread(PLATFORM, room_id, thread_title)

            author_display = row.get('username', 'unknown').replace('/u/', '').strip()
            if not author_display or author_display in ['[deleted]', '[removed]', 'unknown']:
                author_display = f"deleted_user_room_{room_id[:12]}"

            author_db_id = db.get_or_create_user(PLATFORM, author_display, author_display)

            created_at_str = row.get('created_at', '').replace(' UTC', '')
            utc_epoch = parse_reddit_timestamp(created_at_str)

            content = row.get('message', '')
            parent_raw = row.get('thread_parent_message_id', '').strip()
            parent_global_id = f"{PLATFORM}_{parent_raw}" if parent_raw else None

            flat_json_entry = {
                "log_id": global_msg_id,
                "platform": PLATFORM,
                "thread_name": thread_title,
                "author": author_display,
                "author_id": author_display,
                "timestamp_utc": created_at_str,
                "content": content,
                "is_reply_to": parent_global_id
            }

            if content:
                db.insert_message(global_msg_id, thread_db_id, author_db_id, utc_epoch, content, parent_global_id, flat_json_entry, JSONL_OUTPUT, commit_now=False)
                counters['msgs'] += 1
                chat_count += 1
                if chat_count % 5000 == 0:
                    db.commit()
                    logging.info(f"Ingested {chat_count:,} chat messages...")

    db.commit()
    logging.info(f"Finished processing chats: {chat_count:,} messages ingested.")


def process_posts(db, posts_csv_path, ego_name, counters):
    if not os.path.exists(posts_csv_path):
        logging.warning(f"Reddit posts file {posts_csv_path} not found. Skipping posts.")
        return

    logging.info(f"Processing Reddit Posts from {posts_csv_path}...")
    author_db_id = db.get_or_create_user(PLATFORM, ego_name, ego_name)
    post_count = 0

    with open(posts_csv_path, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.DictReader(f)
        for row in reader:
            post_raw_id = row.get('id', '').strip()
            if not post_raw_id:
                continue

            global_post_id = f"{PLATFORM}_{post_raw_id}"
            subreddit = row.get('subreddit', 'unknown').strip()
            title = row.get('title', f"Post {post_raw_id}").strip()
            thread_title = f"r/{subreddit}: {title}"
            thread_db_id = db.get_or_create_thread(PLATFORM, post_raw_id, thread_title)

            created_at_str = row.get('date', '').replace(' UTC', '').strip()
            utc_epoch = parse_reddit_timestamp(created_at_str)

            body = row.get('body', '').strip()
            url = row.get('url', '').strip()
            if body and title:
                content = f"{title}\n\n{body}"
            elif body:
                content = body
            elif url and title:
                content = f"{title}\n{url}"
            else:
                content = title

            flat_json_entry = {
                "log_id": global_post_id,
                "platform": PLATFORM,
                "thread_name": thread_title,
                "author": ego_name,
                "author_id": ego_name,
                "timestamp_utc": created_at_str,
                "content": content,
                "is_reply_to": None
            }

            db.insert_message(global_post_id, thread_db_id, author_db_id, utc_epoch, content, None, flat_json_entry, JSONL_OUTPUT, commit_now=False)
            counters['msgs'] += 1
            post_count += 1

    db.commit()
    logging.info(f"Finished processing posts: {post_count:,} posts ingested.")


def process_comments(db, comments_csv_path, ego_name, counters):
    if not os.path.exists(comments_csv_path):
        logging.warning(f"Reddit comments file {comments_csv_path} not found. Skipping comments.")
        return

    logging.info(f"Processing Reddit Comments from {comments_csv_path}...")
    author_db_id = db.get_or_create_user(PLATFORM, ego_name, ego_name)
    comment_count = 0

    with open(comments_csv_path, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.DictReader(f)
        for row in reader:
            comment_id = row.get('id', '').strip()
            if not comment_id:
                continue

            global_msg_id = f"{PLATFORM}_{comment_id}"
            subreddit = row.get('subreddit', 'unknown').strip()
            link = row.get('link', '').strip()

            post_match = re.search(r'/comments/([a-zA-Z0-9_]+)', link)
            post_id = post_match.group(1) if post_match else ''

            thread_platform_id = post_id if post_id else f"sub_{subreddit}"
            thread_title = f"r/{subreddit} (Thread {post_id})" if post_id else f"r/{subreddit} Comments"
            thread_db_id = db.get_or_create_thread(PLATFORM, thread_platform_id, thread_title)

            parent_raw = row.get('parent', '').strip()
            parent_clean = re.sub(r'^t[13]_', '', parent_raw) if parent_raw else ''
            parent_global_id = f"{PLATFORM}_{parent_clean}" if parent_clean else None

            created_at_str = row.get('date', '').replace(' UTC', '').strip()
            utc_epoch = parse_reddit_timestamp(created_at_str)
            body = row.get('body', '').strip()

            flat_json_entry = {
                "log_id": global_msg_id,
                "platform": PLATFORM,
                "thread_name": thread_title,
                "author": ego_name,
                "author_id": ego_name,
                "timestamp_utc": created_at_str,
                "content": body,
                "is_reply_to": parent_global_id
            }

            db.insert_message(global_msg_id, thread_db_id, author_db_id, utc_epoch, body, parent_global_id, flat_json_entry, JSONL_OUTPUT, commit_now=False)
            counters['msgs'] += 1
            comment_count += 1
            if comment_count % 1000 == 0:
                db.commit()

    db.commit()
    logging.info(f"Finished processing comments: {comment_count:,} comments ingested.")


def process_messages_archive(db, messages_csv_path, ego_name, counters):
    if not os.path.exists(messages_csv_path):
        return

    logging.info(f"Processing Reddit Messages Archive from {messages_csv_path}...")
    msg_count = 0

    with open(messages_csv_path, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_id = row.get('id', '').strip()
            if not raw_id:
                continue

            global_msg_id = f"{PLATFORM}_pm_{raw_id}"
            thread_raw = row.get('thread_id', '').strip()
            from_user = row.get('from', '').replace('/u/', '').strip() or 'unknown'
            to_user = row.get('to', '').replace('/u/', '').replace('/r/', 'r/').strip() or 'unknown'
            subject = row.get('subject', '').strip()

            thread_platform_id = thread_raw if thread_raw else f"pm_{raw_id}"
            thread_title = f"PM: {subject}" if subject else f"PM: {from_user} ↔ {to_user}"
            thread_db_id = db.get_or_create_thread(PLATFORM, thread_platform_id, thread_title)

            author_display = from_user
            author_db_id = db.get_or_create_user(PLATFORM, author_display, author_display)

            created_at_str = row.get('date', '').replace(' UTC', '').strip()
            utc_epoch = parse_reddit_timestamp(created_at_str)
            body = row.get('body', '').strip()

            flat_json_entry = {
                "log_id": global_msg_id,
                "platform": PLATFORM,
                "thread_name": thread_title,
                "author": author_display,
                "author_id": author_display,
                "timestamp_utc": created_at_str,
                "content": body,
                "is_reply_to": None
            }

            db.insert_message(global_msg_id, thread_db_id, author_db_id, utc_epoch, body, None, flat_json_entry, JSONL_OUTPUT, commit_now=False)
            counters['msgs'] += 1
            msg_count += 1

    db.commit()
    logging.info(f"Finished processing messages archive: {msg_count:,} PMs ingested.")


def insert_comment_recursive(db, comment, thread_db_id, parent_global_id, thread_title, counters):
    raw_msg_id = comment.get('comment_id', 'unknown')
    if raw_msg_id == 'unknown':
        raw_msg_id = comment.get('id', 'unknown')

    global_msg_id = f"{PLATFORM}_{raw_msg_id}"
    author_display = comment.get('author', 'unknown')

    if not author_display or author_display in ['[deleted]', '[removed]', 'unknown']:
        author_display = f"deleted_user_thread_{thread_db_id}"

    author_db_id = db.get_or_create_user(PLATFORM, author_display, author_display)
    utc_epoch = parse_reddit_timestamp(comment.get('created_utc', ''))
    content = comment.get('body', '')

    flat_json_entry = {
        "log_id": global_msg_id,
        "platform": PLATFORM,
        "thread_name": thread_title,
        "author": author_display,
        "author_id": author_display,
        "timestamp_utc": comment.get('created_utc', ''),
        "content": content,
        "is_reply_to": parent_global_id
    }

    if content or global_msg_id != f"{PLATFORM}_unknown":
        db.insert_message(global_msg_id, thread_db_id, author_db_id, utc_epoch, content, parent_global_id, flat_json_entry, JSONL_OUTPUT, commit_now=False)
        counters['msgs'] += 1
        if counters['msgs'] % 1000 == 0:
            db.commit()

    for child in comment.get('replies', []):
        if isinstance(child, dict):
            insert_comment_recursive(db, child, thread_db_id, global_msg_id, thread_title, counters)


def process_external_context(db, context_base_dir, counters):
    posts_dir = os.path.join(context_base_dir, "posts", "*.json")
    post_files = glob.glob(posts_dir)
    if post_files:
        logging.info(f"Found {len(post_files)} Reddit Post Context files.")
        for filepath in post_files:
            try:
                with open(filepath, 'rb') as f:
                    post_iter = ijson.items(f, 'my_post_details')
                    post = next(post_iter, {})
                    f.seek(0)
                    reply_iter = ijson.items(f, 'conversation_thread.item')

                    if post:
                        post_raw_id = post.get('id', 'unknown')
                        global_post_id = f"{PLATFORM}_{post_raw_id}"
                        thread_title = post.get('title', f"Post {post_raw_id}")
                        subreddit = post.get('subreddit', 'unknown')

                        thread_db_id = db.get_or_create_thread(PLATFORM, post_raw_id, thread_title)

                        author_display = post.get('author', 'unknown')
                        if not author_display or author_display in ['[deleted]', '[removed]', 'unknown']:
                            author_display = f"deleted_user_thread_{thread_db_id}"

                        author_db_id = db.get_or_create_user(PLATFORM, author_display, author_display)
                        utc_epoch = parse_reddit_timestamp(post.get('created_utc', ''))
                        content = f"[{subreddit}] {post.get('selftext', '')}"

                        flat_json_entry = {
                            "log_id": global_post_id,
                            "platform": PLATFORM,
                            "thread_name": thread_title,
                            "author": author_display,
                            "author_id": author_display,
                            "timestamp_utc": post.get('created_utc', ''),
                            "content": content,
                            "is_reply_to": None
                        }

                        db.insert_message(global_post_id, thread_db_id, author_db_id, utc_epoch, content, None, flat_json_entry, JSONL_OUTPUT, commit_now=False)

                        for reply in reply_iter:
                            if isinstance(reply, dict):
                                insert_comment_recursive(db, reply, thread_db_id, global_post_id, thread_title, counters)
                                del reply

                    gc.collect()
                    db.commit()
            except Exception as e:
                logging.error(f"Error parsing post context {filepath}: {e}")

    comments_dir = os.path.join(context_base_dir, "comments", "*.json")
    comment_files = glob.glob(comments_dir)
    if comment_files:
        logging.info(f"Found {len(comment_files)} Reddit Comment Context files.")
        for filepath in comment_files:
            try:
                with open(filepath, 'rb') as f:
                    f.seek(0)
                    context_iter = ijson.items(f, 'context')
                    context = next(context_iter, {})

                    f.seek(0)
                    my_comment_iter = ijson.items(f, 'my_comment_details')
                    my_comment = next(my_comment_iter, {})

                    f.seek(0)
                    replies_iter = ijson.items(f, 'replies_to_my_comment')
                    replies_to_my_comment = next(replies_iter, [])

                    parent_post = context.get('parent_post')
                    thread_title = "Unknown Thread"
                    thread_platform_id = "unknown"
                    if parent_post:
                        thread_title = parent_post.get('title', "Post context")
                        thread_platform_id = parent_post.get('id', 'unknown')

                    thread_db_id = db.get_or_create_thread(PLATFORM, thread_platform_id, thread_title)
                    last_global_id = f"{PLATFORM}_{thread_platform_id}" if thread_platform_id != "unknown" else None

                    parent_thread = context.get('parent_thread', [])
                    if isinstance(parent_thread, list):
                        for comment in reversed(parent_thread):
                            if isinstance(comment, dict):
                                insert_comment_recursive(db, comment, thread_db_id, last_global_id, thread_title, counters)
                                cid = comment.get('comment_id', comment.get('id', 'unknown'))
                                last_global_id = f"{PLATFORM}_{cid}"

                    if my_comment and isinstance(my_comment, dict):
                        insert_comment_recursive(db, my_comment, thread_db_id, last_global_id, thread_title, counters)
                        cid = my_comment.get('comment_id', my_comment.get('id', 'unknown'))
                        my_global_id = f"{PLATFORM}_{cid}"

                        for reply in replies_to_my_comment:
                            if isinstance(reply, dict):
                                insert_comment_recursive(db, reply, thread_db_id, my_global_id, thread_title, counters)

                    db.commit()
            except Exception as e:
                logging.error(f"Stream Error parsing comment context {filepath}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Parse Reddit export and contexts into Sarthink Memory Layer")
    parser.add_argument("--archive", default=os.path.join(REPO_ROOT, "archive", "reddit-export"),
                        help="Folder containing raw Reddit CSV export")
    parser.add_argument("--context", default=os.path.join(REPO_ROOT, "processed_data", "context", "reddit", "Reddit_Context_Archive"),
                        help="Folder containing fetched Reddit context JSON files")
    parser.add_argument("--db", default=None, help="Path to SQLite database")
    parser.add_argument("--logs", default=None, help="Directory for JSONL output logs")
    args = parser.parse_args()

    ego_name = load_reddit_ego()
    logging.info(f"Resolved Reddit ego user: '{ego_name}'")

    db = SarthinkMemoryLayer(db_path=args.db, jsonl_dir=args.logs)

    # Clean previous reddit JSONL and purge DB for idempotent fresh run
    jsonl_path = os.path.join(db.jsonl_dir, JSONL_OUTPUT)
    if os.path.exists(jsonl_path):
        os.remove(jsonl_path)
        logging.info(f"Cleared old {JSONL_OUTPUT} for fresh reparse.")
    db.purge_platform(PLATFORM)

    counters = {'msgs': 0}

    # 1. Direct chats & rooms (chat_history.csv)
    chat_csv = os.path.join(args.archive, "chat_history.csv")
    if os.path.exists(chat_csv):
        room_titles = resolve_chat_room_titles(chat_csv, ego_name)
        process_chats(db, chat_csv, room_titles, counters)

    # 2. Posts (posts.csv)
    posts_csv = os.path.join(args.archive, "posts.csv")
    if os.path.exists(posts_csv):
        process_posts(db, posts_csv, ego_name, counters)

    # 3. Comments (comments.csv)
    comments_csv = os.path.join(args.archive, "comments.csv")
    if os.path.exists(comments_csv):
        process_comments(db, comments_csv, ego_name, counters)

    # 4. Messages / PMs (messages_archive.csv)
    messages_csv = os.path.join(args.archive, "messages_archive.csv")
    if os.path.exists(messages_csv):
        process_messages_archive(db, messages_csv, ego_name, counters)

    # 5. External Context (if fetched via API)
    if os.path.exists(args.context):
        process_external_context(db, args.context, counters)

    db.commit()
    db.close()
    logging.info(f"Reddit parsing complete! Successfully ingested {counters['msgs']:,} total messages into Sarthink Memory.")


if __name__ == "__main__":
    main()
