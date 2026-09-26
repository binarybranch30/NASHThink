# Sarthink

**Sarthink** is an experimental system for visualizing and exploring your personal digital history as a unified social graph. It bridges the gap between fragmented social media archives (Twitter, Reddit, Discord, etc.) by reconstructing conversation threads and mapping your identity across platforms.

<video controls src="graph-thing.mp4" title="Title"></video>
![Spherical Nodes](image.png)


## Currently at Stage 2 (Memory Graph)

> [!WARNING]
> **EXPERIMENTAL** This is an experimental project and may not be suitable for production use. I do not guarantee if it will work for you. There can be a lot of difference between our data exports and the way they are processed.

## Project Architecture

The project is structured into three main layers, with all logic centralized in the `scripts/` directory:

1.  **Ingestion & Context** (`scripts/context/`): Tools to bridge the "missing link" in data exports. While social media archives often only include your own messages, these scripts fetch the surrounding conversation context (replies, parent posts) to reconstruct meaningful threads.
2.  **Parsing & ETL** (`scripts/parsers/`): A suite of Python scripts that normalize raw exports and fetched context into a structured SQLite database and JSONL logs.
3.  **Semantic Intelligence** (`scripts/semantic/`): Advanced processing to chunk, summarize (via LLMs), and embed conversational data for semantic search and cognitive memory.
4.  **Utilities & Analysis** (`scripts/utils/`, `scripts/analysis/`): Shared helper scripts for database management, layout computation, and specific data extraction tasks.
5.  **Data Storage** (`processed_data/`): Structured into `db/` (SQLite), `logs/` (JSONL), `context/` (Fetched conversation context), `graph/` (CSV/LanceDB), `semantic/` (Processing outputs), and `metadata/` (Identity maps).
6.  **Visualization** (`sarthink_graph.html`): A high-performance 3D memory graph rendered via Three.js.

## Getting Started

### 1. Prerequisites
- Python 3.8+
- Social media data exports (Twitter Takeout, Reddit Export, etc.)

### 2. Environment Setup
Clone the repository and install dependencies:
```bash
pip install requests asyncpraw ijson pandas openai tiktoken lancedb sentence-transformers
```

Configure your credentials by copying the example environment file:
```bash
cp .env.example .env
```
Fill in your API keys for Twitter (SocialData), Reddit (OAuth), and Cerebras (for summarization) in the `.env` file.

### 3. Identity Mapping
To group your nodes correctly across platforms, define your handles in `config/identity_map.json`. You can use the provided example as a template:
```bash
cp config/identity_map.json.example config/identity_map.json
```

### 4. Data Archive Placement
Sarthink dynamically searches your repository for data, but relies on a standard `archive/` folder at the root of the project to locate your raw data exports safely:
```text
sarthink/
├── archive/
│   ├── twitter/                # Unzipped X/Twitter archive (tweets*.js, account.js found recursively)
│   ├── reddit-export/          # posts.csv, comments.csv, chat_history.csv
│   ├── Discord_DM_Export/      # DiscordChatExporter JSON files
│   ├── instagram-*.zip         # Meta GDPR zips in JSON format, left zipped ("instagram"/"facebook" in the name)
│   ├── whatsapp/               # "WhatsApp Chat with X.txt" files or the exported .zip per chat
│   ├── chatgpt/                # ChatGPT data export (conversations.json)
│   ├── claude/                 # Claude data export (conversations.json)
│   └── google/Takeout/         # Google Takeout: Mail/*.mbox, Google Chat/, YouTube*/comments/, My Activity/ (JSON)
```

## Workflow

### A. Context Fetching
Fetch the conversation context that isn't included in your raw exports:
- **Twitter**: Run `python3 scripts/context/twitter_fetch_context.py`
- **Reddit**: Run `python3 scripts/context/reddit_fetch_context.py`

### B. Parsing Data
Run the platform-specific parsers to populate the database:
```bash
python3 scripts/parsers/twitter_parser.py
python3 scripts/parsers/reddit_parser.py
python3 scripts/parsers/discord_parser.py
python3 scripts/parsers/meta_parser.py
python3 scripts/parsers/whatsapp_parser.py
python3 scripts/parsers/chatgpt_parser.py
python3 scripts/parsers/claude_parser.py
python3 scripts/parsers/google_parser.py
```
The newer parsers accept `--archive`, `--db` and `--logs` to point at non-default locations. Add your WhatsApp name and Google email addresses to `config/identity_map.json` so your own messages resolve to your persona.

### C. Semantic Pipeline (Optional)
Semantic search runs **fully locally on CPU**: no GPU, hosted inference or LLM API is needed. The default embedding model is the multilingual [`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) (384-dim, ~480 MB), chosen because it handles Hindi/Hinglish alongside English. Model, device and paths live in `scripts/semantic/embedding_config.py`, shared by `embedder.py` and `search.py`.

#### 1. Local CPU setup (one time)
Create a project-local virtual environment and install CPU-only packages (works without system pip or sudo):
```bash
python3 -m venv --without-pip .venv
curl -sSfL https://bootstrap.pypa.io/get-pip.py | .venv/bin/python
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install sentence-transformers lancedb pyarrow numpy
```
The model is downloaded once into `.cache/models/` on first use; after that it loads from the local cache. Set `HF_HUB_OFFLINE=1` to guarantee no network access. `.venv/`, `.cache/` and `processed_data/` are gitignored.

#### 2. Build chunks
```bash
python3 scripts/semantic/chunk_builder.py   # SQLite -> processed_data/semantic/session_chunks.json
```
`scripts/semantic/summarizer.py` calls a hosted LLM (Cerebras) and is optional; skip it to stay local.

#### 3. Full local topic index
Embeds raw topic chunks (no summaries needed) into the `topics` table. Expect a few hours on a small CPU server, so run it detached:
```bash
nohup env HF_HUB_OFFLINE=1 .venv/bin/python scripts/semantic/embedder.py \
  --input processed_data/semantic/session_chunks.json --topics-only --topics-table topics \
  > processed_data/embed_full.log 2>&1 &
tail -f processed_data/embed_full.log
```
- Existing tables are never replaced unless you pass `--overwrite`.
- `--model NAME` embeds with a different model; `--max-topics N --topics-table NAME` builds a small pilot.
- Every table embedder.py creates is recorded in `processed_data/graph/sarthink_lancedb.metadata.json` (table, model, vector dimension, creation time, source file, row count).

#### 4. Search a table
```bash
HF_HUB_OFFLINE=1 .venv/bin/python scripts/semantic/search.py "college ke baare mein stress"
HF_HUB_OFFLINE=1 .venv/bin/python scripts/semantic/search.py "when did I talk about photography" --table topics --limit 10 --json
```
`search.py` reads the table's metadata and embeds the query with **the model that built that table**, so older tables keep working after the default changes. It refuses to search a table with no metadata, or whose vector dimension doesn't match its metadata.

### D. Graph Generation
Compute the 3D layout and export the data for the web UI:
```bash
python3 scripts/utils/compute_layout.py
python3 scripts/utils/export_cosmograph.py
```

### E. Visualizing
Run a local web server to view the graph:
```bash
python3 -m http.server 8000
```
Then visit `http://localhost:8000/sarthink_graph.html`.
