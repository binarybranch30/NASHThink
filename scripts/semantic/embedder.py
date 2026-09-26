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
    from sentence_transformers import SentenceTransformer
    import numpy as np
except ImportError:
    print("Warning: Please ensure lancedb, pyarrow, and sentence-transformers are installed.")

# Shared with search.py: queries must be embedded by the same model, on the same device.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DEVICE = "cpu"
MODEL_CACHE_DIR = str(REPO_ROOT / ".cache" / "models")
LANCEDB_PATH = str(REPO_ROOT / "processed_data" / "graph" / "sarthink_lancedb")


def load_embedding_model():
    """Loads the embedding model on CPU, caching downloaded weights inside the repo."""
    return SentenceTransformer(EMBEDDING_MODEL_NAME, device=EMBEDDING_DEVICE, cache_folder=MODEL_CACHE_DIR)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default=str(REPO_ROOT / "processed_data" / "semantic" / "summarized_chunks.json"), help="Path to summarized JSON output")
    parser.add_argument('--topics-only', action='store_true', help="Skip the sessions table (useful without summaries)")
    parser.add_argument('--topics-table', type=str, default="topics", help="LanceDB table name for topic chunks")
    parser.add_argument('--max-topics', type=int, default=None, help="Stop after this many topic chunks (pilot runs)")
    parser.add_argument('--seed', type=int, default=0, help="Shuffle seed used with --max-topics so a pilot mixes platforms")
    args = parser.parse_args()

    try:
        with open(args.input, "r") as f:
            updated_chunks = json.load(f)
    except FileNotFoundError:
        print(f"{args.input} not found!")
        return

    if args.max_topics:
        random.Random(args.seed).shuffle(updated_chunks)

    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME} on {EMBEDDING_DEVICE}...")
    embedding_model = load_embedding_model()
    print("Model loaded successfully.")

    print(f"Generating Topic Chunks and embedding vectors for {len(updated_chunks)} sessions...")
    
    db = lancedb.connect(LANCEDB_PATH)
    
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
    
    if not args.topics_only:
        if "sessions" in db.table_names():
            db.drop_table("sessions")
        db.create_table("sessions", data=session_data_for_db)
    
    if args.topics_table in db.table_names():
        db.drop_table(args.topics_table)
    if topic_data_for_db:
        db.create_table(args.topics_table, data=topic_data_for_db)

    print("Vector database successfully populated.")
    print(f"Ingested {len(session_data_for_db)} session chunks and {len(topic_data_for_db)} dynamic topic chunks into '{args.topics_table}'.")

if __name__ == "__main__":
    main()
