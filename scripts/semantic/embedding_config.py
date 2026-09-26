"""Shared embedding configuration for embedder.py and search.py.

Every LanceDB table written by embedder.py gets an entry in a metadata file next to the
database, recording which model produced its vectors. search.py reads that entry so a
query is always embedded by the same model as the table it searches.
"""
import datetime
import json
import os
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = Path(os.path.dirname(os.path.dirname(SCRIPT_DIR)))

# Multilingual so Hindi/Hinglish and English conversations share one vector space.
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DEVICE = "cpu"
MODEL_CACHE_DIR = str(REPO_ROOT / ".cache" / "models")
LANCEDB_PATH = str(REPO_ROOT / "processed_data" / "graph" / "sarthink_lancedb")
METADATA_VERSION = 1


def load_embedding_model(model_name=DEFAULT_EMBEDDING_MODEL):
    """Loads `model_name` on CPU from the repo-local cache, downloading it only if it isn't cached yet."""
    from sentence_transformers import SentenceTransformer

    kwargs = dict(device=EMBEDDING_DEVICE, cache_folder=MODEL_CACHE_DIR)
    try:
        return SentenceTransformer(model_name, local_files_only=True, **kwargs)
    except Exception:
        return SentenceTransformer(model_name, **kwargs)


def list_table_names(db):
    """Table names across lancedb versions (list_tables() is paginated; table_names() is deprecated)."""
    if not hasattr(db, "list_tables"):
        return list(db.table_names())
    names, token = [], None
    while True:
        resp = db.list_tables(page_token=token) if token else db.list_tables()
        names.extend(resp.tables)
        token = resp.page_token
        if not token:
            return names


def metadata_path(db_path=LANCEDB_PATH):
    """processed_data/graph/sarthink_lancedb -> processed_data/graph/sarthink_lancedb.metadata.json"""
    db_path = Path(db_path)
    return db_path.with_name(db_path.name + ".metadata.json")


def load_metadata(db_path=LANCEDB_PATH):
    """Returns {table_name: entry}; empty if no metadata file exists yet."""
    path = metadata_path(db_path)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f).get("tables", {})


def get_table_metadata(table_name, db_path=LANCEDB_PATH):
    return load_metadata(db_path).get(table_name)


def save_table_metadata(table_name, model_name, vector_dim, source, db_path=LANCEDB_PATH, **extra):
    """Records (or replaces) the entry for one table. Written atomically so a crash can't truncate the file."""
    tables = load_metadata(db_path)
    entry = {
        "table": table_name,
        "model": model_name,
        "vector_dim": int(vector_dim),
        "device": EMBEDDING_DEVICE,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "source": str(source),
    }
    entry.update(extra)
    tables[table_name] = entry

    path = metadata_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"version": METADATA_VERSION, "tables": tables}, f, indent=2)
    os.replace(tmp, path)
    return entry
