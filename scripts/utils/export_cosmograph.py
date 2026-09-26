"""Exports the SQLite memory graph to the CSVs the 3D graph UI reads.

Nodes are people (U_<Users.id>) and conversation threads (T_<Threads.id>); an edge links a person to
a thread they wrote in. Besides the label/group/size/color columns, every node and edge carries its
message count and first/last activity (epoch seconds, UTC) so the UI can filter by date.

The database is opened read-only. Run compute_layout.py afterwards to add layout_x/y/z:
    python3 scripts/utils/export_cosmograph.py
    python3 scripts/utils/compute_layout.py
"""
import argparse
import sqlite3
import csv
import logging
import json
import os
from pathlib import Path

# Resolve paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = Path(os.path.dirname(os.path.dirname(SCRIPT_DIR)))

logging.basicConfig(level=logging.INFO, format='%(message)s')

# Platform-native color palette. Threads are darker, users are vivid, DMs are lighter tints.
GROUP_COLORS = {
    # Twitter — signature X/Twitter blue family
    "twitter_user":            "#1DA1F2",  # Twitter blue (vivid)
    "twitter_public_thread":   "#0a3d6b",  # deep navy (thread hubs)
    "twitter_dm_group":        "#4fc3f7",  # sky blue (DM rooms)

    # Reddit — orange family
    "reddit_user":             "#FF4500",  # Reddit orange (vivid)
    "reddit_chat":             "#ff8c42",  # warm amber (chat/DM rooms)
    "reddit_public_thread":    "#c13a00",  # deep burnt orange (thread hubs)

    # Instagram — gradient brand family (pink → purple)
    "instagram_user":          "#E1306C",  # Instagram pink (vivid)
    "instagram_comment_thread":"#833ab4",  # Instagram purple (thread hubs)
    "instagram_dm_group":      "#fd5c87",  # light rose (DM rooms)

    # Discord — Blurple family
    "discord_user":            "#5865F2",  # Discord blurple (vivid)
    "discord_dm_group":        "#7289da",  # classic Discord lighter blurple (DM rooms)

    # Facebook — Facebook blue family
    "facebook_user":           "#1877F2",  # Facebook blue (vivid)
    "facebook_comment_thread": "#0a5abf",  # deep Facebook navy (thread hubs)
    "facebook_dm_group":       "#74b3f7",  # light Facebook blue (DM rooms)

    # WhatsApp — green family
    "whatsapp_user":           "#25D366",  # WhatsApp green (vivid)
    "whatsapp_dm_group":       "#128C7E",  # teal green (chat rooms)

    # Google — brand yellow/red/blue
    "google_user":             "#FBBC05",  # Google yellow (vivid)
    "google_dm_group":         "#4285F4",  # Google blue (Chat DMs)
    "google_group_chat":       "#1a5fd0",  # deep blue (Chat spaces)
    "google_email_thread":     "#EA4335",  # Google red (Gmail threads)
    "google_comment_thread":   "#b3261e",  # dark red (YouTube comments)
    "google_chat":             "#34A853",  # Google green (Search/Gemini days)

    # ChatGPT — OpenAI teal
    "chatgpt_user":            "#10A37F",  # teal (vivid)
    "chatgpt_chat":            "#0b6e56",  # dark teal (conversations)

    # Claude — clay
    "claude_user":             "#D97757",  # clay (vivid)
    "claude_chat":             "#9c4f33",  # dark clay (conversations)

    # Fallback
    "group":                   "#aaaaaa",  # neutral grey
}

# Brand colour per platform: used for any group missing above (e.g. instagram_public_thread),
# so no node falls back to white.
PLATFORM_COLORS = {
    "twitter": "#1DA1F2", "reddit": "#FF4500", "instagram": "#E1306C", "discord": "#5865F2",
    "facebook": "#1877F2", "whatsapp": "#25D366", "google": "#FBBC05", "chatgpt": "#10A37F",
    "claude": "#D97757",
}
# Thread hubs of a group with no explicit colour are drawn in a darker shade of the platform colour.
THREAD_SHADE = 0.62

NODE_FIELDS = ['id', 'label', 'group', 'size', 'color', 'platform', 'kind', 'messages', 'first_ts', 'last_ts', 'title']
EDGE_FIELDS = ['source', 'target', 'weight', 'first_ts', 'last_ts']
TITLE_CHARS = 200
LABEL_CHARS = 45


def shade(hex_color, factor):
    n = int(hex_color.lstrip('#'), 16)
    r, g, b = (int(((n >> s) & 255) * factor) for s in (16, 8, 0))
    return f"#{r:02x}{g:02x}{b:02x}"


def group_color(group):
    """Colour for a node group; unknown groups get their platform's colour instead of white."""
    if group in GROUP_COLORS:
        return GROUP_COLORS[group]
    platform = group.split('_')[0]
    base = PLATFORM_COLORS.get(platform, GROUP_COLORS["group"])
    return base if group.endswith('_user') else shade(base, THREAD_SHADE)


def clean_text(value):
    return str(value).replace(',', '').replace('\n', ' ').replace('\r', '').replace('"', '').strip()

def get_thread_group(platform, title):
    """Categorize the kind of conversational thread for Cosmograph Coloring."""
    title_lower = title.lower()
    
    if platform == 'reddit':
        if title_lower.startswith(('dm', 'pm:', 'chat:', 'group:')) or 'direct message' in title_lower or ' chat' in title_lower:
            return "reddit_chat"
        return "reddit_public_thread"
    elif title_lower.startswith(('dm ', 'dm:')) or 'direct message' in title_lower:
        return f"{platform}_dm_group"
    elif title_lower.startswith('comment on'):
        return f"{platform}_comment_thread"
    elif title_lower.startswith(('ai chat:', 'searches ', 'gemini ')):
        return f"{platform}_chat"
    elif title_lower.startswith('email:'):
        return f"{platform}_email_thread"
    elif title_lower.startswith('group '):
        return f"{platform}_group_chat"
    elif platform == 'twitter' and 'twitter_' in title:
        return 'twitter_tweet_thread'
    
    return f"{platform}_public_thread"

def load_identity_map():
    map_path = REPO_ROOT / "config" / "identity_map.json"
    if not map_path.exists():
        return None, {}
    
    with open(map_path, 'r') as f:
        data = json.load(f)
        
    master = data.get("master_persona", "User")
    alias_dict = {}
    for platform, aliases in data.get("aliases", {}).items():
        for alias in aliases:
            alias_dict[alias] = master
            alias_dict[alias.lower()] = master
    alias_dict[master] = master
    return master, alias_dict

def export_to_cosmograph(db_path=None, out_dir=None, identity_map=None):
    """Writes cosmograph_nodes.csv and cosmograph_edges.csv; returns (node_count, edge_count)."""
    db_path = Path(db_path or REPO_ROOT / 'processed_data' / 'db' / 'sarthink_memory.db')
    out_dir = Path(out_dir or REPO_ROOT / 'processed_data' / 'graph')
    logging.info("Connecting to Sarthink Database for Cosmograph Export...")

    master_persona, identity_aliases = load_identity_map() if identity_map is None else identity_map
    if master_persona:
        logging.info(f"Identity Map Loaded: Resolving configured aliases to -> '{master_persona}'")

    if not db_path.is_file():
        raise FileNotFoundError(f"Database not found: {db_path}")
    # Read-only: the export must never modify the memory database.
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cursor = conn.cursor()

    nodes = []  # dicts keyed by NODE_FIELDS
    edges = []  # dicts keyed by EDGE_FIELDS

    # Per-node message count and activity span, accumulated from the edges.
    node_weights = {}
    node_first = {}
    node_last = {}

    def touch(node_id, weight, first, last):
        node_weights[node_id] = node_weights.get(node_id, 0) + weight
        if first is not None:
            node_first[node_id] = min(first, node_first.get(node_id, first))
        if last is not None:
            node_last[node_id] = max(last, node_last.get(node_id, last))

    # 1. EXTRACT BINDING EDGES
    logging.info("Collapsing Messages into relational Vectors...")
    cursor.execute("""
        SELECT author_id, thread_id, COUNT(*) as weight, MIN(timestamp_utc), MAX(timestamp_utc)
        FROM Messages
        GROUP BY author_id, thread_id
    """)
    interactions = cursor.fetchall()

    for author_id, thread_id, weight, first, last in interactions:
        u_node = f"U_{author_id}"
        t_node = f"T_{thread_id}"
        first, last = to_epoch(first), to_epoch(last)

        edges.append({'source': u_node, 'target': t_node, 'weight': weight,
                      'first_ts': '' if first is None else first, 'last_ts': '' if last is None else last})
        touch(u_node, weight, first, last)
        touch(t_node, weight, first, last)

    def node_row(node_id, label, group, platform, kind, title):
        return {
            'id': node_id, 'label': label, 'group': group,
            'size': node_weights.get(node_id, 1), 'color': group_color(group),
            'platform': platform, 'kind': kind, 'messages': node_weights.get(node_id, 0),
            'first_ts': node_first.get(node_id, ''), 'last_ts': node_last.get(node_id, ''),
            'title': title,
        }

    # 2. EXTRACT LOGICAL USERS
    logging.info("Formatting Entity Nodes...")
    cursor.execute("SELECT id, platform, display_name, raw_id FROM Users")
    users = cursor.fetchall()

    for uid, platform, display_name, raw_id in users:
        node_id = f"U_{uid}"

        # Prioritize true Usernames (raw_id) over display names, except for Discord/Meta where raw_id is a giant UUID
        primary_name = str(raw_id) if platform in ['twitter', 'reddit'] else str(display_name)
        if primary_name.isdigit() and display_name:
            primary_name = str(display_name)

        # UNIVERSAL IDENTITY RESOLUTION OVERRIDE
        if primary_name in identity_aliases:
            primary_name = identity_aliases[primary_name]
        elif display_name in identity_aliases:
            primary_name = identity_aliases[display_name]
        elif str(raw_id) in identity_aliases:
            primary_name = identity_aliases[str(raw_id)]

        # Clean rogue formatting chars
        label = clean_text(primary_name)
        nodes.append(node_row(node_id, label, f"{platform}_user", platform, 'user', label))

    # 3. EXTRACT LOGICAL THREADS
    logging.info("Formatting Hub Nodes...")
    cursor.execute("""
        SELECT DISTINCT t.id, t.platform, t.title
        FROM Threads t
        JOIN Messages m ON t.id = m.thread_id
    """)
    threads = cursor.fetchall()

    for tid, platform, title in threads:
        node_id = f"T_{tid}"
        group = get_thread_group(platform, str(title))

        # Clean title; the label is short, the full title goes to the details panel.
        full_title = clean_text(title)[:TITLE_CHARS]
        label = full_title[:LABEL_CHARS] + "..." if len(full_title) > LABEL_CHARS else full_title
        nodes.append(node_row(node_id, label, group, platform, 'thread', full_title))

    conn.close()

    # 4. WRITE THE OUTPUTS
    logging.info(f"Writing {len(nodes)} Nodes & {len(edges)} Edges to Disk...")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "cosmograph_nodes.csv", 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=NODE_FIELDS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(nodes)

    with open(out_dir / "cosmograph_edges.csv", 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=EDGE_FIELDS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(edges)

    logging.info("Cosmograph rendering structure complete! Now run scripts/utils/compute_layout.py to add the 3D layout.")
    return len(nodes), len(edges)


def to_epoch(value):
    """Messages.timestamp_utc as integer epoch seconds; millisecond values are scaled down."""
    if value is None or value == '':
        return None
    try:
        ts = int(float(value))
    except (TypeError, ValueError):
        return None
    return ts // 1000 if ts > 100_000_000_000 else ts


def main(argv=None):
    parser = argparse.ArgumentParser(description="Export the Sarthink memory graph CSVs for the 3D UI.")
    parser.add_argument("--db", help="SQLite database (default processed_data/db/sarthink_memory.db)")
    parser.add_argument("--out", help="Output directory (default processed_data/graph)")
    args = parser.parse_args(argv)
    export_to_cosmograph(args.db, args.out)
    return 0


if __name__ == "__main__":
    main()
