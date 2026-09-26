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
7.  **Local API** (`scripts/api/server.py`): A FastAPI server that serves the graph and answers semantic memory searches against the local LanceDB index.

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
Export the graph from SQLite, then compute its 3D layout (in this order: the export rewrites `cosmograph_nodes.csv`, and the layout step adds `layout_x/y/z` to it):
```bash
python3 scripts/utils/export_cosmograph.py   # read-only on the DB -> processed_data/graph/cosmograph_{nodes,edges}.csv
python3 scripts/utils/compute_layout.py      # adds layout_x, layout_y, layout_z (deterministic, seed 42)
```
Nodes are people (`U_<id>`) and threads (`T_<id>`); an edge means that person wrote in that thread. Besides `id, label, group, size, color`, nodes carry `platform, kind (user|thread), messages, first_ts, last_ts, title` and edges carry `weight, first_ts, last_ts` (epoch seconds, UTC), which drive the graph's details panel and timeline filter. Older exports without these columns still load; the timeline then explains how to enable it.

### E. Visualizing + Memory Search
`scripts/api/server.py` is a local FastAPI app that serves the 3D graph **and** a semantic search API over the LanceDB index. It binds to `127.0.0.1` only; queries are embedded on your CPU and nothing leaves the machine.

One-time install into the existing venv:
```bash
.venv/bin/pip install fastapi uvicorn
```

Start it (after the embedding index from step C.3 has finished):
```bash
HF_HUB_OFFLINE=1 .venv/bin/python scripts/api/server.py
```
Then open `http://127.0.0.1:8000/`. Options: `--table topics_multilingual_pilot` searches another table, `--port`, `--db`.

**Using the graph** (press `?` in the page for the same list):
- **Left-drag** orbits, **right-drag** or **Shift/Ctrl-drag** pans, **scroll** zooms toward the cursor. Dragging never moves nodes; any drag cancels a running camera flight.
- **Hover** a node for its name, platform, connections and active months. **Click** selects it: the node, every direct neighbour and the links between them stay bright (drawn on top), everything else is dimmed, not hidden. **Double-click** (or `F`) flies the camera to it. Click empty space to deselect.
- The **details panel** shows type, platform, connections, messages, first/last activity, the linked people or threads (click one to jump to it), and for threads the indexed **conversation context** from `/api/thread`. The selection is kept in the URL (`#node=T_12`), so a refresh or a shared local link reopens it.
- **Find node** (`/`) matches names and thread titles, highlights all matches, and `↑ ↓ Enter` selects one.
- **Platforms** and **Node types** in the sidebar are toggle filters (colour = platform, filled dot = person, ring = thread; **only** isolates one platform). The **Timeline** histogram shows active threads per month; drag its handles or use the 3/12-month presets to hide threads outside the range and people with no messages in it.
- The top bar shows node/edge/platform totals, what the filters currently show, and the selection.
- **Reset** (or `Esc`) restores the original full view: clears the selection, find, memory search and filters, and flies the camera home.
- If a node has missing or invalid `layout_*` values it is placed near its platform; if the whole layout is missing or collapsed to one point, a deterministic fallback layout is drawn and the top bar says `layout fallback`.

In the graph sidebar, type into **MEMORY SEARCH** and press Enter:
- Results appear in a panel on the right (platform, date, people, similarity bar, snippet with the query words marked, and why it matched: shared words or "matched by meaning").
- Matching conversation threads and their participants are highlighted on the graph and the camera flies to them; the rest of the graph is dimmed.
- Click a result to select its thread (its links, details and full chunk text); the other results stay highlighted. Click again to unfocus. **Find related memories** in a thread's details runs a search with its title.
- **Clear** in the results panel removes only the memory search; **Reset** / `Esc` resets everything.
- The first query loads the embedding model, so it takes longer than the rest.

#### Ask Sarthink (evidence-first memory Q&A)
Switch the sidebar's memory box to **ASK SARTHINK**, type a question (or click a sample chip) and press Enter (Shift+Enter for a new line):
- *What was I stressed about during college?* · *How has my interest in photography changed?* · *What did I discuss about Python?* · *What was I working on around August 2026?*

The right panel shows a short **answer**, a **confidence** level (high / medium / low), **key points**, a small chronological **timeline** and the **source cards**. Sources and their threads are highlighted on the graph; clicking a source card or a timeline entry selects and frames its thread exactly like a memory-search result. The question is limited to the platforms switched on in the sidebar and the timeline range (the scope line under the box shows what will be sent). Cards marked *closest match only* were retrieved but are not used as evidence.

How the answer is made (`scripts/api/memory_brief.py`, deterministic, no language model):
1. The question is embedded by the same already-loaded model as memory search, and the 60 nearest chunks are retrieved from `topics`, with the platform/date filters applied inside LanceDB before ranking. A month or year in the question (*around August 2026*, *in 2025*) becomes a date window when you haven't set one (±1 month for "around"), and is removed from the embedded text.
2. Each chunk is checked against the question: it counts as **evidence** only if it is similar enough, has real content (tiny "ok"/"lol" chunks score high on similarity but are never used), and mentions the question's content words (question scaffolding like *what*, *discuss*, *changed*, *working* is ignored; longer questions need at least half of their words). Dates come from the message timestamps inside the chunk.
3. Chunks repeating one another (overlapping session windows, crossposts) are merged, keeping the best one.
4. The answer states how many memories matched, on which platforms and over which months, then quotes the strongest sentence verbatim with its author, date and platform (for "how has … changed" questions: the earliest and the latest). Key points quote one sentence per month and platform.
5. Confidence is **high** with at least 3 supporting sources, 2 of which cover every content word; **medium** with fewer or partial support (the answer says so); **low** when nothing qualifies, in which case the answer says the evidence is weak and the closest chunks are shown as leads.

This is **evidence-based synthesis, not a generative LLM**: every sentence is a verbatim quote, a count, a date, a platform or a title from a returned source, so it never invents an event, feeling, relationship or date, but it also doesn't interpret or summarise in its own words, can quote a sentence out of context, and depends on the words you use. Read the sources. Everything runs on this machine: no hosted APIs, no LLM service, no browser-side calls other than to the local server, no model downloads.

If the index is still being built, the sidebar says so and the API answers `503 index_unavailable`; the server picks the table up automatically once it exists, no restart needed. If the page can't reach the API it shows the command to start it.

API (interactive docs at `/api/docs`):
```bash
curl http://127.0.0.1:8000/api/health
curl -X POST http://127.0.0.1:8000/api/search -H 'Content-Type: application/json' \
  -d '{"query": "college ke baare mein stress", "limit": 10}'
curl http://127.0.0.1:8000/api/thread/T_12      # indexed chunks of one thread (no model load)
curl -X POST http://127.0.0.1:8000/api/ask -H 'Content-Type: application/json' \
  -d '{"question": "How has my interest in photography changed?", "limit": 8, "platforms": ["reddit", "instagram"], "date_from": "2025-01-01", "date_to": "2026-09-30"}'
```
`/api/ask` takes `question` (required, ≤500 chars), `limit` (sources, 1–20, default 8), `platforms` (optional list; omit for all) and `date_from` / `date_to` (optional ISO dates or datetimes; a plain `date_to` includes that whole day, a datetime is exclusive; a chunk matches when its messages overlap the range). It returns `{question, answer, confidence, summary_points, timeline: [{date, label, node_id, platform, source}], sources: [{rank, node_id, title, platform, date_start, date_end, similarity, snippet, text, people, relevant, matched_terms}], notes, evidence: {retrieved, considered, relevant, terms}, filters, model, took_ms}`. `timeline[].source` is the 1-based index into `sources`; `notes` explains filtering and merging. Errors use the same codes as `/api/search`.
`/api/search` returns `{query, table, model, count, took_ms, results: [...]}`; each result has `rank, similarity, distance, start_time, end_time, platform, title, channel_id, node_id, people, summary, snippet, text`. `node_id` (`T_<thread id>`) is the matching graph node. Errors are `{"error": {"code", "message"}}` with codes `invalid_request` (422), `index_unavailable` (503), `model_unavailable` and `search_failed` (500).

The graph alone still works from any static server (`python3 -m http.server 8080`, then `http://localhost:8080/sarthink_graph.html`); memory search then needs the API running and `?api=http://127.0.0.1:8000` appended to the URL.

`/api/thread/T_<id>` returns `{node_id, channel_id, count, first_time, last_time, chunks: [...]}` with up to 8 chunks (oldest first; `start_time, end_time, platform, title, people, snippet, text`). It only filters the index, so it answers instantly even before the first search has loaded the model.

Tests (fake model, table and a throwaway SQLite DB; no index, model or real data needed):
```bash
.venv/bin/python scripts/tests/test_api.py
.venv/bin/python scripts/tests/test_ask.py      # Ask Sarthink: grounding, weak evidence, filters, errors, model reuse
python3 scripts/tests/test_graph_pipeline.py
```
Browser end-to-end test of the graph against your real CSVs (needs the server running and Playwright, which is not a project dependency):
```bash
HF_HUB_OFFLINE=1 .venv/bin/python scripts/api/server.py --port 8765 &
npm install --prefix /tmp/pw playwright && npx --prefix /tmp/pw playwright install chromium   # Node 18: playwright@1.49.1
NODE_PATH=/tmp/pw/node_modules node scripts/tests/graph_ui_e2e.mjs http://127.0.0.1:8765/
```
It checks layout fidelity, orbit/pan/zoom, hover, selection and its links, filters, timeline, find, reset/Esc, refresh, memory search and Ask Sarthink (with stubbed, synthetic API answers: rendering, HTML escaping, filters, timeline/source focus, index-building retry), API-offline/index-unavailable states and layout fallbacks (by rewriting responses in the browser, never on disk). `SARTHINK_E2E_SEMANTIC=1` adds a real memory search.
