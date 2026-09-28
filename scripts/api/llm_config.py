"""LLM profiles for Ask's written answers (see answer_writer.py and scripts/llm.sh).

Two local llama.cpp servers, both on loopback only (nothing leaves the machine):
    quick  Llama 3.2 3B Instruct  (127.0.0.1:8082)  fast, weaker writing
    best   Llama 3.1 8B Instruct  (127.0.0.1:8083)  slow on this CPU, better writing
Two hosted DeepSeek models (kind "remote"): the question and the retrieved excerpts are sent to DeepSeek's API.
    deepseek           deepseek-chat      answers in seconds
    deepseek_reasoner  deepseek-reasoner  thinks first: slower, for harder questions
Their API key is DEEPSEEK_API_KEY, from the environment or the repo's gitignored .env file (KEY=value lines);
it is read on every use, so adding it needs no restart, and it is never included in API responses or logs.

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
HOST = "127.0.0.1"   # local models: never configurable, they see private archive text
SECRETS_PATH = REPO_ROOT / ".env"

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
        "prompt_tokens": 1400,   # measured with the server's /tokenize; Hinglish takes ~2 chars/token, English ~3.5
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
        "prompt_tokens": 2000,   # ~4 min of prompt reading at ~9 tok/s on this CPU
        "max_tokens": 600,
        "temperature": 0.3,
        # 8-bit KV cache: about half the memory of f16 with no visible quality loss; RAM is shared on this host.
        "extra_args": ["-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0"],
    },
    "deepseek": {
        "kind": "remote",
        "provider": "DeepSeek",
        "label": "Fast",
        "model_name": "DeepSeek Chat",
        "description": "DeepSeek Chat, online — a few seconds; sends the question and excerpts to DeepSeek",
        "base_url": "https://api.deepseek.com",
        "api_model": "deepseek-chat",
        "api_key_env": "DEEPSEEK_API_KEY",
        "max_sources": 8,
        "source_chars": 1500,
        "prompt_tokens": 6000,
        "max_tokens": 700,
        "temperature": 0.3,
        "timeout_s": 90,
    },
    "deepseek_reasoner": {
        "kind": "remote",
        "provider": "DeepSeek",
        "label": "Deep",
        "model_name": "DeepSeek Reasoner",
        "description": "DeepSeek Reasoner, online — thinks first (often 20–60 s); sends the question and excerpts to DeepSeek",
        "base_url": "https://api.deepseek.com",
        "api_model": "deepseek-reasoner",
        "api_key_env": "DEEPSEEK_API_KEY",
        "max_sources": 8,
        "source_chars": 1500,
        "prompt_tokens": 6000,
        "max_tokens": 4000,        # includes its hidden reasoning
        "temperature": 0.3,        # ignored by the reasoner
        "timeout_s": 240,
    },
}
PROFILE_ORDER = ("deepseek", "best", "quick", "deepseek_reasoner")   # preference when the UI picks a default


def is_remote(profile):
    return profile.get("kind") == "remote"


def api_key(profile, secrets_path=None):
    """The profile's API key from the environment, else from the gitignored .env file; None when unset."""
    name = profile.get("api_key_env")
    if not name:
        return None
    if os.environ.get(name, "").strip():
        return os.environ[name].strip()
    path = Path(secrets_path or SECRETS_PATH)
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            k, sep, v = line.strip().removeprefix("export ").partition("=")
            if sep and k.strip() == name:
                return v.strip().strip('"').strip("'") or None
    except OSError:
        pass
    return None


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
        if is_remote(p):
            p["url"] = p["base_url"].rstrip("/")
            p["model_path"] = ""
            continue
        p["url"] = f"http://{HOST}:{int(p['port'])}"
        model = Path(p["model"])
        p["model_path"] = str(model if model.is_absolute() else REPO_ROOT / model)
    return profiles


if __name__ == "__main__":
    profiles = {k: v for k, v in load_profiles().items() if not is_remote(v)}   # scripts/llm.sh runs local models only
    if len(sys.argv) != 2 or sys.argv[1] not in profiles:
        print(f"usage: {sys.argv[0]} {'|'.join(profiles)}", file=sys.stderr)
        sys.exit(2)
    print(json.dumps({**profiles[sys.argv[1]], "llama_server": str(LLAMA_SERVER), "host": HOST}))
