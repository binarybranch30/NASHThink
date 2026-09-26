import json
import argparse
import os
import random
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
from embedding_config import (DEFAULT_EMBEDDING_MODEL, EMBEDDING_DEVICE, LANCEDB_PATH,
                              list_table_names, load_embedding_model, save_table_metadata)


def topic_chunk_burst(chunk: Dict, embedding_model) -> List[Dict]:
    topic_chunks = []
    text = chunk.get('text', '')
    
    micro_bursts = [b.strip() for b in text.split('\n\n') if b.strip()]
    
    if len(micro_bursts) <= 1:
        return []
        
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
    parser.add_argument('--model', type=str, default=DEFAULT_EMBEDDING_MODEL, help=f"Embedding model (default {DEFAULT_EMBEDDING_MODEL})")
    parser.add_argument('--db', type=str, default=LANCEDB_PATH, help="LanceDB directory")
    parser.add_argument('--overwrite', action='store_true', help="Replace tables that already exist (they are kept otherwise)")
    args = parser.parse_args(argv)

    source = os.path.abspath(args.input)
    target_tables = [args.topics_table] if args.topics_only else ["sessions", args.topics_table]

    try:
        with open(args.input, "r") as f:
            updated_chunks = json.load(f)
    except FileNotFoundError:
        print(f"{args.input} not found!")
        return 1

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
