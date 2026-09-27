import json
import argparse
import datetime
import os
import random
import re
import sys
from typing import List, Dict
from pathlib import Path

# Resolve paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = Path(os.path.dirname(os.path.dirname(SCRIPT_DIR)))

try:
    import lancedb
    import pyarrow as pa
    import numpy as np
except ImportError:
    print("Warning: Please ensure lancedb, pyarrow, and sentence-transformers are installed.")

# Model, device, paths and table metadata are shared with search.py
sys.path.append(SCRIPT_DIR)
from embedding_config import (DEFAULT_EMBEDDING_MODEL, EMBEDDING_DEVICE, LANCEDB_PATH, get_table_metadata,
                              list_table_names, load_embedding_model, save_table_metadata, update_table_metadata)

PLATFORM_RE = re.compile(r"^[a-z0-9_]{1,32}$")


def topic_chunk_burst(chunk: Dict, embedding_model) -> List[Dict]:
    topic_chunks = []
    text = chunk.get('text', '')
    
    micro_bursts = [b.strip() for b in text.split('\n\n') if b.strip()]

    if not micro_bursts:
        return []
    if len(micro_bursts) == 1:
        # A session that is one burst is still one topic (it used to be dropped, leaving it unsearchable).
        return [{
            "parent_id": chunk.get('channel_id'),
            "platform": chunk.get('platform', ''),
            "title": chunk.get('title', ''),
            "start_time": chunk.get('start_time'),
            "end_time": chunk.get('end_time'),
            "density_score": chunk.get('density_score', 0.0),
            "ego_weight": chunk.get('ego_weight', 0.0),
            "text": micro_bursts[0]
        }]

    embeddings = embedding_model.encode(micro_bursts, convert_to_numpy=True)
    
    similarities = []
    for i in range(len(embeddings)-1):
        v1 = embeddings[i] / max(np.linalg.norm(embeddings[i]), 1e-10)
        v2 = embeddings[i+1] / max(np.linalg.norm(embeddings[i+1]), 1e-10)
        similarities.append(np.dot(v1, v2))
        
    threshold = 0.6 
    
    current_topic_bursts = [micro_bursts[0]]
    for i, sim in enumerate(similarities):
        if sim < threshold:
            topic_chunks.append({
                "parent_id": chunk.get('channel_id'),
                "platform": chunk.get('platform', ''),
                "title": chunk.get('title', ''),
                "start_time": chunk.get('start_time'),
                "end_time": chunk.get('end_time'),
                "density_score": chunk.get('density_score', 0.0),
                "ego_weight": chunk.get('ego_weight', 0.0),
                "text": "\n".join(current_topic_bursts)
            })
            current_topic_bursts = [micro_bursts[i+1]]
        else:
            current_topic_bursts.append(micro_bursts[i+1])
            
    if current_topic_bursts:
        topic_chunks.append({
            "parent_id": chunk.get('channel_id'),
            "platform": chunk.get('platform', ''),
            "title": chunk.get('title', ''),
            "start_time": chunk.get('start_time'),
            "end_time": chunk.get('end_time'),
            "density_score": chunk.get('density_score', 0.0),
            "ego_weight": chunk.get('ego_weight', 0.0),
            "text": "\n".join(current_topic_bursts)
        })
        
    return topic_chunks


def topic_rows(chunks, embedding_model):
    """Topic sub-chunks of `chunks` with their vectors, in the topics table's column layout."""
    rows = []
    for idx, chunk in enumerate(chunks):
        if idx % 100 == 0:
            print(f"Embedded {idx}/{len(chunks)} chunks ({len(rows)} topic chunks)...")
        sub_chunks = topic_chunk_burst(chunk, embedding_model)
        if not sub_chunks:
            continue
        vecs = embedding_model.encode([t['text'] for t in sub_chunks], convert_to_numpy=True)
        for t_chunk, vec in zip(sub_chunks, vecs):
            rows.append({"vector": [float(x) for x in vec], **{k: t_chunk[k] for k in (
                "parent_id", "platform", "title", "start_time", "end_time", "density_score", "ego_weight", "text")}})
    return rows


def replace_platform_rows(args, chunks, source):
    """Incremental update: swaps one platform's rows in an existing topics table, leaving every other row as is.

    Refuses unless the table exists with metadata, the model matches the table's, every input chunk belongs to
    that platform and the new vectors have the table's dimension. Re-running replaces the platform's rows
    (idempotent). The previous table version is recorded, so `table.restore(version)` undoes it.
    """
    platform, name = args.replace_platform, args.topics_table
    if not PLATFORM_RE.match(platform):
        print(f"Invalid platform name: {platform!r}", file=sys.stderr)
        return 1
    db = lancedb.connect(args.db)
    meta = get_table_metadata(name, args.db)
    if name not in list_table_names(db) or not meta:
        print(f"Table '{name}' (with metadata) must already exist for --replace-platform.", file=sys.stderr)
        return 1
    model_name = args.model or meta["model"]
    if model_name != meta["model"]:
        print(f"Refusing: table '{name}' was embedded with {meta['model']}, not {model_name}.", file=sys.stderr)
        return 1
    other = sorted({c.get('platform') for c in chunks} - {platform})
    if other:
        print(f"Refusing: input has chunks from other platforms ({', '.join(map(str, other))}).", file=sys.stderr)
        return 1

    print(f"Loading embedding model: {model_name} on {EMBEDDING_DEVICE}...")
    embedding_model = load_embedding_model(model_name)
    rows = topic_rows(chunks, embedding_model)
    if not rows:
        print("No topic chunks to add; table left unchanged.", file=sys.stderr)
        return 1
    table = db.open_table(name)
    dim = table.schema.field("vector").type.list_size
    if len(rows[0]["vector"]) != dim or dim != meta["vector_dim"]:
        print(f"Refusing: {len(rows[0]['vector'])}-dim vectors for a {dim}-dim table.", file=sys.stderr)
        return 1

    before_rows, before_version = table.count_rows(), table.version
    removed = table.count_rows(f"platform = '{platform}'")
    if removed:
        table.delete(f"platform = '{platform}'")
    table.add(rows)
    now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    updates = dict(meta.get("platform_updates") or {})
    updates[platform] = {"rows": len(rows), "replaced_rows": removed, "source": source, "updated_at": now,
                         "version_before": before_version, "version_after": table.version}
    update_table_metadata(name, args.db, rows=table.count_rows(), updated_at=now, platform_updates=updates)
    print(f"'{name}': {before_rows} rows -> {table.count_rows()} ({removed} old {platform} rows replaced by {len(rows)}); "
          f"version {before_version} -> {table.version}.")
    return 0


def create_table_with_metadata(db, table_name, rows, model_name, source, db_path):
    db.create_table(table_name, data=rows)
    save_table_metadata(table_name, model_name, len(rows[0]["vector"]), source, db_path=db_path, rows=len(rows))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default=str(REPO_ROOT / "processed_data" / "semantic" / "summarized_chunks.json"), help="Path to summarized JSON output")
    parser.add_argument('--topics-only', action='store_true', help="Skip the sessions table (useful without summaries)")
    parser.add_argument('--topics-table', type=str, default="topics", help="LanceDB table name for topic chunks")
    parser.add_argument('--max-topics', type=int, default=None, help="Stop after this many topic chunks (pilot runs)")
    parser.add_argument('--seed', type=int, default=0, help="Shuffle seed used with --max-topics so a pilot mixes platforms")
    parser.add_argument('--model', type=str, default=None, help=f"Embedding model (default {DEFAULT_EMBEDDING_MODEL}; "
                        "with --replace-platform, the table's recorded model)")
    parser.add_argument('--db', type=str, default=LANCEDB_PATH, help="LanceDB directory")
    parser.add_argument('--overwrite', action='store_true', help="Replace tables that already exist (they are kept otherwise)")
    parser.add_argument('--replace-platform', type=str, default=None,
                        help="Incremental: replace only this platform's rows in the existing --topics-table "
                             "(input must hold only that platform's chunks; implies --topics-only)")
    args = parser.parse_args(argv)

    source = os.path.abspath(args.input)
    target_tables = [args.topics_table] if args.topics_only else ["sessions", args.topics_table]

    try:
        with open(args.input, "r") as f:
            updated_chunks = json.load(f)
    except FileNotFoundError:
        print(f"{args.input} not found!")
        return 1

    if args.replace_platform:
        return replace_platform_rows(args, updated_chunks, source)
    args.model = args.model or DEFAULT_EMBEDDING_MODEL

    db = lancedb.connect(args.db)
    existing = [t for t in target_tables if t in list_table_names(db)]
    if existing and not args.overwrite:
        print(f"Refusing to replace existing table(s): {', '.join(existing)}. "
              "Choose another --topics-table or pass --overwrite.", file=sys.stderr)
        return 1

    if args.max_topics:
        random.Random(args.seed).shuffle(updated_chunks)

    print(f"Loading embedding model: {args.model} on {EMBEDDING_DEVICE}...")
    embedding_model = load_embedding_model(args.model)
    print("Model loaded successfully.")

    print(f"Generating Topic Chunks and embedding vectors for {len(updated_chunks)} sessions...")
    
    session_data_for_db = []
    topic_data_for_db = []

    for idx, chunk in enumerate(updated_chunks):
        if args.max_topics and len(topic_data_for_db) >= args.max_topics:
            break
        if idx % 100 == 0:
            print(f"Embedded {idx}/{len(updated_chunks)} chunks ({len(topic_data_for_db)} topic chunks)...")

        if not args.topics_only:
            session_text = f"Participants: {', '.join(chunk.get('participants', []))} \n {chunk.get('summary', '')}"
            session_vec = embedding_model.encode(session_text, convert_to_numpy=True).tolist()

            session_data_for_db.append({
                "vector": session_vec,
                "channel_id": chunk.get('channel_id', ''),
                "platform": chunk.get('platform', ''),
                "title": chunk.get('title', ''),
                "start_time": chunk.get('start_time', ''),
                "end_time": chunk.get('end_time', ''),
                "density_score": chunk.get('density_score', 0.0),
                "ego_weight": chunk.get('ego_weight', 0.0),
                "summary": chunk.get('summary', ''),
                "text": chunk.get('text', '') 
            })
        
        sub_chunks = topic_chunk_burst(chunk, embedding_model)
        if args.max_topics:
            sub_chunks = sub_chunks[:args.max_topics - len(topic_data_for_db)]
        if not sub_chunks:
            continue
        topic_vecs = embedding_model.encode([t['text'] for t in sub_chunks], convert_to_numpy=True)
        for t_chunk, topic_vec in zip(sub_chunks, topic_vecs):
            topic_data_for_db.append({
                "vector": topic_vec.tolist(),
                "parent_id": t_chunk['parent_id'],
                "platform": t_chunk['platform'],
                "title": t_chunk['title'],
                "start_time": t_chunk['start_time'],
                "end_time": t_chunk['end_time'],
                "density_score": t_chunk['density_score'],
                "ego_weight": t_chunk['ego_weight'],
                "text": t_chunk['text']
            })

    print("\nSaving tables to LanceDB...")
    
    if not args.topics_only and session_data_for_db:
        if "sessions" in list_table_names(db):
            db.drop_table("sessions")
        create_table_with_metadata(db, "sessions", session_data_for_db, args.model, source, args.db)
    
    if topic_data_for_db:
        if args.topics_table in list_table_names(db):
            db.drop_table(args.topics_table)
        create_table_with_metadata(db, args.topics_table, topic_data_for_db, args.model, source, args.db)

    print("Vector database successfully populated.")
    print(f"Ingested {len(session_data_for_db)} session chunks and {len(topic_data_for_db)} dynamic topic chunks into '{args.topics_table}'.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
