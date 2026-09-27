"""Local LLM profiles for Ask Sarthink's written answers (see answer_writer.py and scripts/llm.sh).

Two llama.cpp servers, both on loopback only:
    quick  Llama 3.2 3B Instruct  (127.0.0.1:8082)  fast, weaker writing
    best   Llama 3.1 8B Instruct  (127.0.0.1:8083)  slow on this CPU, better writing

config/llm_profiles.json (gitignored; see config/llm_profiles.json.example) overrides any field of a profile,
e.g. {"best": {"extra_args": ["-ngl", "99"], "ctx": 16384}} once a GPU is available.

    python3 scripts/api/llm_config.py best          # prints the profile as JSON (used by scripts/llm.sh)
"""
import copy
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = Path(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
OVERRIDES_PATH = REPO_ROOT / "config" / "llm_profiles.json"
LLAMA_SERVER = REPO_ROOT / ".tools" / "llama.cpp" / "llama-b11202" / "llama-server"
HOST = "127.0.0.1"   # never configurable: the models see private archive text

DEFAULT_PROFILES = {
    "quick": {
        "label": "Quick",
        "model_name": "Llama 3.2 3B",
        "description": "Llama 3.2 3B — about a minute on this CPU",
        "model": "models/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "alias": "llama-3.2-3b-instruct",
        "port": 8082,
        "ctx": 4096,
        "threads": 2,
        "batch_threads": 3,
        "max_sources": 5,
        "source_chars": 700,
        "max_tokens": 350,
        "temperature": 0.3,
        "extra_args": [],
    },
    "best": {
        "label": "Best",
        "model_name": "Llama 3.1 8B",
        "description": "Llama 3.1 8B — several minutes on this CPU",
        "model": "models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        "alias": "llama-3.1-8b-instruct",
        "port": 8083,
        "ctx": 6144,
        "threads": 2,
        "batch_threads": 3,
        "max_sources": 8,
        "source_chars": 1200,
        "max_tokens": 600,
        "temperature": 0.3,
        # 8-bit KV cache: about half the memory of f16 with no visible quality loss; RAM is shared on this host.
        "extra_args": ["-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0"],
    },
}
PROFILE_ORDER = ("best", "quick")   # preference when the UI picks a default


def load_profiles(path=OVERRIDES_PATH):
    """Default profiles with config/llm_profiles.json merged in (unknown profile names are ignored)."""
    profiles = copy.deepcopy(DEFAULT_PROFILES)
    path = Path(path)
    if path.is_file():
        with open(path, "r", encoding="utf-8") as f:
            overrides = json.load(f)
        for name, fields in (overrides or {}).items():
            if name in profiles and isinstance(fields, dict):
                profiles[name].update(fields)
    for name, p in profiles.items():
        p["name"] = name
        p["url"] = f"http://{HOST}:{int(p['port'])}"
        model = Path(p["model"])
        p["model_path"] = str(model if model.is_absolute() else REPO_ROOT / model)
    return profiles


if __name__ == "__main__":
    profiles = load_profiles()
    if len(sys.argv) != 2 or sys.argv[1] not in profiles:
        print(f"usage: {sys.argv[0]} {'|'.join(profiles)}", file=sys.stderr)
        sys.exit(2)
    print(json.dumps({**profiles[sys.argv[1]], "llama_server": str(LLAMA_SERVER), "host": HOST}))
