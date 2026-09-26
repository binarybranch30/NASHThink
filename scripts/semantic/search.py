"""Semantic memory search over the LanceDB tables written by embedder.py.

The query is embedded with the model recorded in the table's metadata, never a guess.

Usage:
    .venv/bin/python scripts/semantic/search.py "when did I talk about photography"
    .venv/bin/python scripts/semantic/search.py "photography" --table topics --limit 10 --json
"""
import argparse
import contextlib
import json
import os
import re
import sys
from pathlib import Path

# Resolve paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = Path(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
sys.path.append(SCRIPT_DIR)

from embedding_config import (EMBEDDING_DEVICE, LANCEDB_PATH, get_table_metadata, list_table_names,
                              load_embedding_model, metadata_path)

DEFAULT_TABLE = "topics"
DEFAULT_LIMIT = 5

# Column names that may hold people/entities, in order of preference.
PEOPLE_COLUMNS = ("participants", "people", "entities")

# Author patterns produced by chunk_builder.format_message
AUTHOR_PATTERNS = [
    re.compile(r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] ([^:\n]+):", re.MULTILINE),  # chat / DM
    re.compile(r"^\[(?:TWEET|REPLY)\] @(\S+) —", re.MULTILINE),                     # twitter
    re.compile(r"^\[(?:POST|REPLY/COMMENT)\] u/(\S+) —", re.MULTILINE),             # reddit
]


class SearchError(Exception):
    """A user-facing error (missing DB, table, model, ...)."""


def open_table(db_path, table_name):
    """Opens an existing LanceDB table without ever creating the database or table."""
    if not Path(db_path).is_dir():
        raise SearchError(
            f"LanceDB database not found at {db_path}\n"
            "Build it first:\n"
            "  python3 scripts/semantic/chunk_builder.py\n"
            "  .venv/bin/python scripts/semantic/embedder.py --input processed_data/semantic/session_chunks.json --topics-only"
        )
    try:
        import lancedb
    except ImportError:
        raise SearchError("lancedb is not installed. Run: .venv/bin/pip install lancedb pyarrow")

    db = lancedb.connect(db_path)
    names = list_table_names(db)
    if table_name not in names:
        available = ", ".join(names) if names else "none"
        raise SearchError(
            f"Table '{table_name}' not found in {db_path} (available: {available}).\n"
            "Run: .venv/bin/python scripts/semantic/embedder.py --input processed_data/semantic/session_chunks.json --topics-only"
        )
    return db.open_table(table_name)


def vector_dimension(table):
    field = table.schema.field("vector")
    return getattr(field.type, "list_size", None)


def table_model_metadata(table, table_name, db_path):
    """Returns the metadata entry for `table_name`, refusing tables whose origin is unknown or inconsistent."""
    meta = get_table_metadata(table_name, db_path)
    if not meta or not meta.get("model"):
        raise SearchError(
            f"No embedding metadata for table '{table_name}' in {metadata_path(db_path)}.\n"
            "Without it the query could be embedded by the wrong model. Re-create the table with "
            "scripts/semantic/embedder.py, which records the model it used."
        )

    dim = vector_dimension(table)
    if dim is not None and meta.get("vector_dim") != dim:
        raise SearchError(
            f"Metadata for table '{table_name}' says {meta.get('vector_dim')}-dim vectors from {meta['model']}, "
            f"but the table stores {dim}-dim vectors. Refusing to search; re-create the table with embedder.py."
        )
    return meta


def load_model(model_name):
    """Loads `model_name` exactly as embedder.py does (CPU, repo-local cache)."""
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        raise SearchError("sentence-transformers is not installed. Run: .venv/bin/pip install sentence-transformers")

    try:
        with contextlib.redirect_stdout(sys.stderr):
            return load_embedding_model(model_name)
    except Exception as e:
        raise SearchError(f"Could not load embedding model '{model_name}' on {EMBEDDING_DEVICE}: {e}")


def extract_people(row):
    """Returns people from an explicit column if present, else authors parsed from the chunk text."""
    for col in PEOPLE_COLUMNS:
        value = row.get(col)
        if value:
            return [str(v) for v in value] if isinstance(value, (list, tuple)) else [str(value)]

    seen = []
    text = row.get("text") or ""
    for pattern in AUTHOR_PATTERNS:
        for name in pattern.findall(text):
            name = name.strip()
            if name and name not in seen:
                seen.append(name)
    return seen


def to_result(rank, row):
    distance = row.get("_distance")
    return {
        "rank": rank,
        "similarity": round(1.0 - distance, 4) if distance is not None else None,
        "distance": round(distance, 4) if distance is not None else None,
        "start_time": row.get("start_time") or None,
        "end_time": row.get("end_time") or None,
        "platform": row.get("platform") or None,
        "title": row.get("title") or None,
        "channel_id": row.get("channel_id") or row.get("parent_id") or None,
        "people": extract_people(row),
        "summary": row.get("summary") or None,
        "text": row.get("text") or "",
    }


PLATFORM_RE = re.compile(r"^[a-z0-9_]{1,32}$")
ISO_UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,6})?\+00:00$")


def filter_expression(platforms=None, date_from=None, date_to=None):
    """LanceDB prefilter for platform and date-window constraints, or None when unconstrained.

    Values are interpolated into SQL, so they are validated first: platforms must be plain lowercase
    keys and dates UTC ISO strings as stored by chunk_builder.py (``YYYY-MM-DDTHH:MM:SS+00:00``).
    ``date_to`` is exclusive. A chunk matches when its [start_time, end_time] window overlaps the range.
    """
    parts = []
    if platforms:
        bad = [p for p in platforms if not PLATFORM_RE.match(str(p))]
        if bad:
            raise SearchError(f"Invalid platform name(s): {', '.join(map(repr, bad))}")
        parts.append("platform IN (" + ", ".join(f"'{p}'" for p in sorted(set(platforms))) + ")")
    for value in (date_from, date_to):
        if value is not None and not ISO_UTC_RE.match(value):
            raise SearchError(f"Invalid date bound: {value!r}")
    if date_from:
        parts.append(f"(start_time >= '{date_from}' OR end_time >= '{date_from}')")
    if date_to:
        parts.append(f"start_time < '{date_to}'")
    return " AND ".join(parts) or None


def search(table, model, query, limit=DEFAULT_LIMIT, where=None):
    """Top `limit` chunks by cosine similarity; `where` (see filter_expression) is applied before ranking."""
    query_vec = [float(x) for x in model.encode(query, convert_to_numpy=True)]

    dim = vector_dimension(table)
    if dim is not None and len(query_vec) != dim:
        raise SearchError(
            f"Embedding dimension mismatch: query has {len(query_vec)} dims but the table stores {dim}.\n"
            "Refusing to search with a model that does not match the table."
        )

    # Cosine distance ignores vector magnitude (embedder.py stores unnormalized vectors).
    builder = table.search(query_vec)
    builder = builder.distance_type("cosine") if hasattr(builder, "distance_type") else builder.metric("cosine")
    if where:
        builder = builder.where(where, prefilter=True)
    rows = builder.limit(limit).to_list()
    return [to_result(i, row) for i, row in enumerate(rows, start=1)]


CONTEXT_COLUMNS = ("parent_id", "channel_id", "platform", "title", "start_time", "end_time", "text")
MAX_THREAD_CHUNKS = 500


def thread_chunks(table, channel_id):
    """All indexed chunks of one conversation thread, oldest first. Filter-only scan: no model needed.

    `channel_id` must be the numeric Threads.id (the UI's T_<id> node); anything else is rejected
    so user input never reaches the filter expression unvalidated.
    """
    cid = str(channel_id)
    if not cid.isdigit():
        raise SearchError(f"Invalid thread id: {channel_id!r}")
    names = set(getattr(table.schema, "names", None) or CONTEXT_COLUMNS)
    key = "parent_id" if "parent_id" in names else "channel_id"
    builder = table.search().where(f"{key} = '{cid}'")
    if hasattr(builder, "select"):
        builder = builder.select([c for c in CONTEXT_COLUMNS if c in names])
    rows = builder.limit(MAX_THREAD_CHUNKS).to_list()
    rows.sort(key=lambda r: (r.get("start_time") or "", r.get("end_time") or ""))
    return [
        {
            "start_time": r.get("start_time") or None,
            "end_time": r.get("end_time") or None,
            "platform": r.get("platform") or None,
            "title": r.get("title") or None,
            "people": extract_people(r),
            "text": r.get("text") or "",
        }
        for r in rows
    ]


def format_results(query, results):
    if not results:
        return f'No results for "{query}".'

    lines = [f'Top {len(results)} results for "{query}":', ""]
    for r in results:
        when = r["start_time"] or "unknown date"
        if r["end_time"] and r["end_time"] != r["start_time"]:
            when += f" -> {r['end_time']}"
        lines.append(f"#{r['rank']}  similarity={r['similarity']}  distance={r['distance']}")
        lines.append(f"  Date:     {when}")
        lines.append(f"  Platform: {r['platform'] or 'unknown'}" + (f"  |  {r['title']}" if r["title"] else ""))
        lines.append(f"  People:   {', '.join(r['people']) if r['people'] else 'unknown'}")
        lines.append(f"  Summary:  {r['summary'] or '(none)'}")
        lines.append("  Text:")
        lines.extend("    " + line for line in r["text"].splitlines())
        lines.append("-" * 60)
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Semantic search over your Sarthink conversation memory.")
    parser.add_argument("query", help="Natural language query, e.g. \"when did I talk about photography\"")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help=f"Number of results (default {DEFAULT_LIMIT})")
    parser.add_argument("--json", action="store_true", help="Print results as JSON")
    parser.add_argument("--table", default=DEFAULT_TABLE, help=f"LanceDB table to search, e.g. topics, topics_multilingual_pilot (default {DEFAULT_TABLE})")
    parser.add_argument("--db", default=LANCEDB_PATH, help="Path to the LanceDB directory")
    args = parser.parse_args(argv)

    if args.limit < 1:
        parser.error("--limit must be at least 1")
    if not args.query.strip():
        parser.error("query must not be empty")

    try:
        table = open_table(args.db, args.table)
        meta = table_model_metadata(table, args.table, args.db)
        print(f"Searching '{args.table}' with {meta['model']} ({meta['vector_dim']} dims)", file=sys.stderr)
        model = load_model(meta["model"])
        results = search(table, model, args.query, args.limit)
    except SearchError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        print(format_results(args.query, results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
