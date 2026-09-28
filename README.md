<p align="center"><img src="assets/nashthink-mark.png" alt="NASH Think logo" width="110"></p>

<h1 align="center">NASH Think</h1>

<p align="center"><b>A private AI memory for the life hidden in your conversations.</b><br>
Team NASH (SAI034) · ASYNC 2026 · Track 01: Sovereign AI</p>

---

Your chats hold your plans, deadlines, promises, decisions and memories, but they are scattered across a dozen apps. **NASH Think** imports your chat history from 9 platforms into one private memory that runs entirely on your own machine. Ask a question in English or Hinglish, get an answer with citations to the exact messages, and explore your conversations on a 3D memory map.

- 🔒 **Private by design:** your chats, the search and the AI stay on your device. No cloud, no GPU needed.
- 📎 **Answers you can check:** every claim cites the real messages. Weak evidence gives "No relevant info found", not a guess.
- 🗺️ **One memory across 9 apps:** WhatsApp, Instagram, Facebook, Discord, Reddit, X, Google (Gmail, Chat, YouTube), ChatGPT and Claude.
- 🇮🇳 **Hinglish-aware:** understands chat shorthand and romanised Hindi.

## Features

| Feature | What it does |
|---|---|
| **Ask** | Answers a question from your chats with numbered citations, a confidence level, key points and a timeline |
| **Local AI answers** | Llama 3.1 8B (Best) or Llama 3.2 3B (Quick) write the answer on your machine via llama.cpp |
| **Evidence only** | Quotes, counts and dates in seconds, with no language model |
| **Find memories** | Search by meaning, not just keywords |
| **Memory map** | 3D graph of people and conversations; cited sources light up on it |
| **Person profiles** | Your history with someone: counts, dates, recurring words, notable chats, full message history |
| **Filters & Insights** | Filter by platform and time; activity over time, platforms and top contacts |
| **Sample-data mode** | Demos run on a fictional archive; real data is behind a password |

## How it works

```
Chat exports → Parsers → SQLite memory → Session chunks → Embeddings (LanceDB)
                               │                                  │
                               └──────────── FastAPI server ──────┘ ← Llama 3.1 8B (llama.cpp, local)
                                                   │
                                  Web app: Ask · Memory map · Profiles · Insights
```

1. **Parse:** one Python parser per platform writes people, conversations and messages to SQLite (UTC timestamps, safe to re-run).
2. **Chunk:** conversations are split into sessions at long silences (30 min to 12 h), up to 4,000 tokens each.
3. **Embed:** each chunk becomes a 384-dimension vector with `paraphrase-multilingual-MiniLM-L12-v2` on the CPU, stored in LanceDB.
4. **Find evidence:** for a question, the 60 nearest chunks are retrieved and checked without AI (similar enough, mentions the question's key words, not filler), merged and graded high / medium / low.
5. **Write:** the local model sees only that evidence and must cite it as `[1]`, `[2]`… Citations and quotes are verified afterwards; unsupported answers become "No relevant info found".
6. **Map & profiles:** built from the database with read-only SQL. The 3D layout is computed once with a fixed seed and drawn with Three.js.

**Tech stack:** Python · SQLite · sentence-transformers · LanceDB · llama.cpp · Llama 3.1 8B / 3.2 3B (Q4_K_M GGUF) · FastAPI · Uvicorn · Three.js

## Quick start

### 1. Install

```bash
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install sentence-transformers lancedb pyarrow numpy fastapi uvicorn
cp config/identity_map.json.example config/identity_map.json   # add your own handles so your messages map to "you"
```

### 2. Add your chat exports

Download your data from each app and place it under `archive/` (gitignored):

```text
archive/
├── whatsapp/        "WhatsApp Chat with X.txt" files or the exported .zip per chat
├── instagram-*.zip  Meta exports in JSON format (Instagram / Facebook), left zipped
├── discord/<name>/  Official Discord data package
├── reddit-export/   posts.csv, comments.csv, chat_history.csv
├── twitter/         Unzipped X / Twitter archive
├── google/Takeout/  Mail, Google Chat, YouTube comments, My Activity
├── chatgpt/         conversations.json
└── claude/          conversations.json
```

### 3. Build your memory

```bash
python3 scripts/parsers/whatsapp_parser.py        # repeat for each platform you have:
                                                  # instagram/facebook → meta_parser.py, discord, reddit,
                                                  # twitter, google, chatgpt, claude
python3 scripts/utils/export_cosmograph.py        # graph of people and conversations
python3 scripts/utils/compute_layout.py           # 3D positions
python3 scripts/semantic/chunk_builder.py         # conversation sessions
.venv/bin/python scripts/semantic/embedder.py \
  --input processed_data/semantic/session_chunks.json --topics-only --topics-table topics
```

Embedding runs on the CPU and can take a while for a large archive; run it with `nohup` if needed.

### 4. Run it

```bash
scripts/start_*.sh     # API + web app on http://127.0.0.1:8000/
scripts/status_*.sh    # health, search index, database, local models
scripts/stop_*.sh
```

The server binds to `127.0.0.1` only. To use it from another computer, open an SSH tunnel and browse to `http://127.0.0.1:8000/`:

```bash
ssh -N -L 8000:127.0.0.1:8000 <user>@<server>
```

### 5. Optional: written answers with a local model

```bash
scripts/llm.sh start best     # Llama 3.1 8B (≈5.4 GB RAM), or: start quick  (Llama 3.2 3B)
scripts/llm.sh status
scripts/llm.sh stop
```

Model files, downloads and timings are in [`docs/local_llm.md`](docs/local_llm.md). Without a model, Ask still works in **Evidence only** mode. Online answers with DeepSeek are optional: add `DEEPSEEK_API_KEY` to `.env`; those answers are labelled "online".

## Sample data and privacy

- A **fictional** sample archive lives in [`demo/`](demo/README.md): 12,016 messages across 9 platforms, 2,638 searchable chunks and 365 graph nodes. Every person in it is invented.
- With `config/workspaces.json`, one server can hold the sample data (the default for every visitor) and your own data behind a password. Only a salted PBKDF2 hash is stored; set it with `python3 scripts/utils/set_workspace_password.py <workspace>`.
- Archives, the database, the search index, models and `.env` are gitignored and never leave your machine.
- The local model only ever sees the evidence for one question.

## API

Interactive docs at `http://127.0.0.1:8000/api/docs`.

| Endpoint | Purpose |
|---|---|
| `POST /api/ask`, `POST /api/ask/stream` | Evidence-backed answer (streamed when a model writes it) |
| `POST /api/search` | Semantic search |
| `GET /api/thread/{id}` | Excerpts of one conversation |
| `GET /api/person/{id}` (+ `/conversations`, `/messages`) | Person profile and history |
| `GET /api/insights` | Activity totals, platforms, top contacts |
| `GET /api/llm`, `GET /api/health` | Model and server status |

## Project layout

```text
scripts/parsers/    one parser per platform
scripts/semantic/   chunking, embeddings, search, Hinglish layer
scripts/api/        FastAPI server, Ask, answer writer, profiles, insights, workspaces
scripts/utils/      database layer, graph export, 3D layout, helpers
scripts/context/    optional: fetch missing reply context for X and Reddit
scripts/tests/      tests with a fake model and throwaway data
config/             identity map, LLM profile and workspace examples
demo/               fictional sample archive
docs/               local LLM setup, Hinglish notes, project overview
assets/             NASH Think logos
```

Run the tests:

```bash
for t in scripts/tests/test_*.py; do .venv/bin/python "$t" || echo "FAILED: $t"; done
```

## Limits

- AI answers can still be wrong; the cited sources are there so you can check.
- Written answers take a few minutes on a CPU (seconds with a GPU or the online option).
- Single user for now: one archive and one owner per workspace.
- Some exports only include your side of a conversation (for example Discord).

## Roadmap

Built on the same memory and evidence foundation, always as suggestions you approve:

- Tasks and follow-ups found in chats (GTD / Kanban)
- Calendar events and reminders for plans and deadlines
- Richer people profiles: shared topics, memories, inside jokes
- End-of-day reviews and journal drafts
- Resurfacing old memories
- Better linking of the same person across apps

## Team NASH

Built for **ASYNC 2026**, Track 01: Sovereign AI (team ID SAI034).

| Name | Role |
|---|---|
| Naitik Srivastava | Team lead |
| Astitva Singh | Member |
| Sarthak Sidhant | Member |
| Hemanth Keerthipati | Member |

Contact: naitiksri08@gmail.com
