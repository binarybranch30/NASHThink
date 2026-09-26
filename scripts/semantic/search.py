"""Semantic memory search over the LanceDB tables written by embedder.py.

Usage:
    python3 scripts/semantic/search.py "when did I talk about photography"
    python3 scripts/semantic/search.py "photography" --limit 10 --json
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

# Reuse the embedder's model name and LanceDB location so queries land in the same vector space.
# embedder.py prints a warning on stdout when deps are missing; keep stdout clean for --json.
with contextlib.redirect_stdout(sys.stderr):
    from embedder import EMBEDDING_DEVICE, EMBEDDING_MODEL_NAME, LANCEDB_PATH, load_embedding_model

DEFAULT_TABLE = "sessions"
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
            "  python3 scripts/semantic/summarizer.py\n"
            "  python3 scripts/semantic/embedder.py"
        )
    try:
        import lancedb
    except ImportError:
        raise SearchError("lancedb is not installed. Run: .venv/bin/pip install lancedb pyarrow")

    db = lancedb.connect(db_path)
    names = list(db.table_names())
    if table_name not in names:
        available = ", ".join(names) if names else "none"
        raise SearchError(
            f"Table '{table_name}' not found in {db_path} (available: {available}).\n"
            "Run: python3 scripts/semantic/embedder.py"
        )
    return db.open_table(table_name)


def vector_dimension(table):
    field = table.schema.field("vector")
    return getattr(field.type, "list_size", None)


def load_model():
    """Loads the embedding model exactly as embedder.py does (same model, device and cache)."""
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        raise SearchError("sentence-transformers is not installed. Run: .venv/bin/pip install sentence-transformers")

    try:
        with contextlib.redirect_stdout(sys.stderr):
            return load_embedding_model()
    except Exception as e:
        raise SearchError(f"Could not load embedding model '{EMBEDDING_MODEL_NAME}' on {EMBEDDING_DEVICE}: {e}")


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


def search(table, model, query, limit=DEFAULT_LIMIT):
    query_vec = [float(x) for x in model.encode(query, convert_to_numpy=True)]

    dim = vector_dimension(table)
    if dim is not None and len(query_vec) != dim:
        raise SearchError(
            f"Embedding dimension mismatch: query has {len(query_vec)} dims but the table stores {dim}.\n"
            "The table was likely built with a different model (e.g. embedder.py's CPU fallback). "
            "Re-run embedder.py with the same model available."
        )

    # Cosine distance ignores vector magnitude (embedder.py stores unnormalized vectors).
    builder = table.search(query_vec)
    builder = builder.distance_type("cosine") if hasattr(builder, "distance_type") else builder.metric("cosine")
    rows = builder.limit(limit).to_list()
    return [to_result(i, row) for i, row in enumerate(rows, start=1)]


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
    parser.add_argument("--table", default=DEFAULT_TABLE, help="LanceDB table to search, e.g. sessions, topics, topics_pilot (default sessions)")
    parser.add_argument("--db", default=LANCEDB_PATH, help="Path to the LanceDB directory (default: embedder.py's path)")
    args = parser.parse_args(argv)

    if args.limit < 1:
        parser.error("--limit must be at least 1")
    if not args.query.strip():
        parser.error("query must not be empty")

    try:
        table = open_table(args.db, args.table)
        model = load_model()
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
