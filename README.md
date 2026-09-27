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

## Local-first privacy model
- **Everything runs on this machine.** Parsing, the SQLite memory database, embeddings (CPU), semantic search, Ask Sarthink and Insights are all local. No archive text, prompt, query or retrieved memory is sent to a hosted API or LLM service. (The only optional exception is `scripts/semantic/summarizer.py`, which calls a hosted LLM and is not needed.)
- **Nothing is exposed publicly.** The API binds to `127.0.0.1` only; reach it from a laptop through an SSH tunnel (below), never by binding `0.0.0.0`.
- **Private data never enters git.** `archive/`, `incoming/`, `processed_data/` (database, index, graph CSVs, logs), `config/identity_map.json`, `.venv/`, `.cache/`, local models (`models/`, `*.gguf`) and tools (`.tools/`) are gitignored.
- **Read-only where it can be.** The graph export and `/api/insights` open the database read-only; the UI escapes every archive-derived string before inserting it into the page.
- **Ask Sarthink needs no LLM.** Answers are evidence-first quotes and counts (see below). An optional local Llama server can be tried separately (`docs/local_llm.md`); nothing depends on it.

## Run the demo

```bash
scripts/start_sarthink.sh     # starts the API + UI on http://127.0.0.1:8000/ (nohup, logs to processed_data/api.log); no-op if already healthy
scripts/status_sarthink.sh    # API health, URL, semantic index, memory DB, optional local Llama (127.0.0.1:8081/8082) — read-only
scripts/stop_sarthink.sh      # stops only Sarthink's API process (scripts/api/server.py on port 8000)
```
From a laptop, forward the port over SSH and open `http://127.0.0.1:8000/` locally:
```bash
ssh -N -L 8000:127.0.0.1:8000 naitik@185.2.102.128
```

**90-second demo flow**
1. **Memory Home** (0–15s): the page opens on a focused question box above the slowly drifting memory graph. Point out the status line (index ready, graph size, "runs on this machine").
2. **Ask** (15–35s): click a sample question, e.g. *How has my interest in photography changed?* The brief appears in place: answer, confidence, key points, timeline and cited sources.
3. **Cited source → graph** (35–50s): click a source card. The page glides into the graph workspace, selects that conversation, lights up its participants and links, and shows its indexed context in the details panel.
4. **Graph exploration** (50–65s): drag to orbit, scroll to zoom, arrow keys to move, click a person to jump to their threads. The command bar in the top bar keeps Ask / Find memories one keystroke away.
5. **Timeline & filters** (65–75s): toggle a platform, press *12 mo* on the timeline; the graph and the next answer follow the scope.
6. **Insights** (75–90s): press `I`. Stat cards, monthly activity (hover a bar), platform breakdown, top contacts and conversations, all for the current filters; click a contact to focus it on the graph.

> **Multi-user note (future work, not implemented):** Sarthink is single-user today: one archive, one database, one index, one owner persona, and no login. Serving several people would need separate per-user workspaces (archives, SQLite DB, LanceDB index, graph files, identity map), per-user API instances or strict per-request scoping, and real authentication and access control. Don't point it at more than one person's data.

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
`scripts/api/server.py` is a local FastAPI app that serves the UI (Memory Home, the 3D graph and Insights) **and** a semantic search API over the LanceDB index. It binds to `127.0.0.1` only; queries are embedded on your CPU and nothing leaves the machine. `scripts/start_sarthink.sh` / `status_sarthink.sh` / `stop_sarthink.sh` wrap it for demos (see *Run the demo* above).

One-time install into the existing venv:
```bash
.venv/bin/pip install fastapi uvicorn
```

Start it (after the embedding index from step C.3 has finished):
```bash
HF_HUB_OFFLINE=1 .venv/bin/python scripts/api/server.py
```
Then open `http://127.0.0.1:8000/`. Options: `--table topics_multilingual_pilot` searches another table, `--port`, `--db`, `--memory-db` (SQLite for Insights).

**Memory Home** is the opening screen: a question box with **Ask Sarthink** (default) and **Find memories** (plain semantic search) modes, sample prompts, and a short status line. The graph keeps drifting behind it and stays interactive outside the card (drag, zoom, click a node to open it). Answers and results appear in the card; clicking a cited source or result opens the **graph workspace** focused on that conversation. **Explore graph** (or `G`) opens the workspace directly; there the same box sits in the top bar as a command bar, results move to the right-hand panel, and `G` / **Home** goes back. If the graph CSVs are missing, Memory Home says so (with the commands to build them) and Ask, search and Insights keep working. Links with `#node=T_12` or `?view=graph` open straight into the workspace.

**Insights** (`I`, or the button in Memory Home / the top bar) summarises the memory database for the current platform toggles and timeline range: message, conversation, people and platform totals, first and latest memory, messages per month (hover for the per-platform split) and per year, a per-platform breakdown with date ranges, and the top contacts and conversations (click one to focus it on the graph). Your own accounts, as listed in `config/identity_map.json`, are excluded from contacts and people counts.

**Using the graph** (press `?` in the page for the same list):
- **Left-drag** orbits, **right-drag** or **Shift/Ctrl-drag** pans, **scroll** zooms toward the cursor. Dragging never moves nodes; any drag cancels a running camera flight.
- **Hover** a node for its name, platform, connections and active months. **Click** selects it: the node, every direct neighbour and the links between them stay bright (drawn on top), everything else is dimmed, not hidden. **Double-click** (or `F`) flies the camera to it. Click empty space to deselect.
- The **details panel** shows type, platform, connections, messages, first/last activity, the linked people or threads (click one to jump to it), and for threads the indexed **conversation context** from `/api/thread`. The selection is kept in the URL (`#node=T_12`), so a refresh or a shared local link reopens it.
- **Arrow keys** move the graph in the pressed direction (hold to keep moving); **Shift + arrows** orbit. They only act on the graph when you're not typing or in a list/tab strip.
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

**Hinglish.** The embedding model understands Hindi in Devanagari but matches romanised Hindi mostly by style ("yaar", "hai", "nahi") rather than topic, so `scripts/semantic/hinglish.py` adds a small hand-curated vocabulary used by both search and Ask: Hinglish filler and negations are ignored when judging relevance, common spellings are normalised (nhi/nai → nahi, padhaai → padhai, nind → neend), and a limited set of topic words is matched across languages (sleep ↔ neend, study ↔ padhai, exam ↔ pariksha, stress ↔ tension, worry ↔ chinta/pareshan, …). When a question names such a topic, a second search with the same query vector is limited (a query-time `LIKE` prefilter, no index) to chunks containing the topic's words in the other language or other spellings, and its hits are interleaved into the normal ranking from position 4 on; results say `matched_via` / `expansion_terms`. Generic words (dost, ghar, paisa) are never expanded. Cross-language evidence needs similarity ≥ 0.2 instead of 0.35, and quotes are never shortened in a way that drops a negation ("neend nahi aati" stays whole). Questions without such a topic are searched exactly as before.

This is **evidence-based synthesis, not a generative LLM**: every sentence is a verbatim quote, a count, a date, a platform or a title from a returned source, so it never invents an event, feeling, relationship or date, but it also doesn't interpret or summarise in its own words, can quote a sentence out of context, and depends on the words you use. Read the sources. Everything runs on this machine: no hosted APIs, no LLM service, no browser-side calls other than to the local server, no model downloads.

If the index is still being built, the sidebar says so and the API answers `503 index_unavailable`; the server picks the table up automatically once it exists, no restart needed. If the page can't reach the API it shows the command to start it.

API (interactive docs at `/api/docs`):
```bash
curl http://127.0.0.1:8000/api/health
curl -X POST http://127.0.0.1:8000/api/search -H 'Content-Type: application/json' \
  -d '{"query": "college ke baare mein stress", "limit": 10}'
curl http://127.0.0.1:8000/api/thread/T_12      # indexed chunks of one thread (no model load)
curl 'http://127.0.0.1:8000/api/insights?platforms=reddit,instagram&date_from=2025-01-01&date_to=2025-12-31'   # read-only aggregates
curl -X POST http://127.0.0.1:8000/api/ask -H 'Content-Type: application/json' \
  -d '{"question": "How has my interest in photography changed?", "limit": 8, "platforms": ["reddit", "instagram"], "date_from": "2025-01-01", "date_to": "2026-09-30"}'
```
`/api/ask` takes `question` (required, ≤500 chars), `limit` (sources, 1–20, default 8), `platforms` (optional list; omit for all) and `date_from` / `date_to` (optional ISO dates or datetimes; a plain `date_to` includes that whole day, a datetime is exclusive; a chunk matches when its messages overlap the range). It returns `{question, answer, confidence, summary_points, timeline: [{date, label, node_id, platform, source}], sources: [{rank, node_id, title, platform, date_start, date_end, similarity, snippet, text, people, relevant, matched_terms}], notes, evidence: {retrieved, considered, relevant, terms}, filters, model, took_ms}`. `timeline[].source` is the 1-based index into `sources`; `notes` explains filtering and merging. Errors use the same codes as `/api/search`.
`/api/search` returns `{query, table, model, count, took_ms, results: [...]}`; each result has `rank, similarity, distance, start_time, end_time, platform, title, channel_id, node_id, people, summary, snippet, text`. `node_id` (`T_<thread id>`) is the matching graph node. Errors are `{"error": {"code", "message"}}` with codes `invalid_request` (422), `index_unavailable` (503), `model_unavailable` and `search_failed` (500).

The graph alone still works from any static server (`python3 -m http.server 8080`, then `http://localhost:8080/sarthink_graph.html`); memory search then needs the API running and `?api=http://127.0.0.1:8000` appended to the URL.

`/api/insights` (GET, read-only) takes optional `platforms` (comma-separated or repeated), `date_from` / `date_to` (ISO dates or datetimes; a plain `date_to` includes that day) and `top` (1–50, default 10). It returns `{empty, filters, totals: {messages, threads, people, platforms}, first_date, latest_date, platforms: [{platform, messages, threads, people, first_date, last_date, share}], available_platforms, activity: {months: [{month, total, platforms}], years: [...]}, top_contacts: [{node_id, label, platform, messages, threads, first_date, last_date}], top_conversations: [{node_id, title, platform, messages, people, first_date, last_date}], owner: {configured, excluded_accounts}, took_ms, cached}`. `node_id`s are graph node ids (`U_<id>`, `T_<id>`). The database is opened read-only; results are cached until the database file changes (the first unfiltered call is warmed at server start). A missing database answers `503 database_unavailable`.

`/api/thread/T_<id>` returns `{node_id, channel_id, count, first_time, last_time, chunks: [...]}` with up to 8 chunks (oldest first; `start_time, end_time, platform, title, people, snippet, text`). It only filters the index, so it answers instantly even before the first search has loaded the model.

Tests (fake model, table and a throwaway SQLite DB; no index, model or real data needed):
```bash
.venv/bin/python scripts/tests/test_api.py
.venv/bin/python scripts/tests/test_ask.py      # Ask Sarthink: grounding, weak evidence, filters, errors, model reuse
.venv/bin/python scripts/tests/test_insights.py # /api/insights: shape, filters, empty state, owner exclusion, read-only, errors
.venv/bin/python scripts/tests/test_hinglish.py # Hinglish vocabulary, expanded retrieval on a temp LanceDB table, Ask evidence, negation-safe quotes
python3 scripts/tests/test_graph_pipeline.py
```
Browser end-to-end test of the graph against your real CSVs (needs the server running and Playwright, which is not a project dependency):
```bash
HF_HUB_OFFLINE=1 .venv/bin/python scripts/api/server.py --port 8765 &
npm install --prefix /tmp/pw playwright && npx --prefix /tmp/pw playwright install chromium   # Node 18: playwright@1.49.1
NODE_PATH=/tmp/pw/node_modules node scripts/tests/graph_ui_e2e.mjs http://127.0.0.1:8765/
```
`SARTHINK_E2E_ONLY=homeFlow,insightsFlow` runs selected flows. If Chromium can't start for missing system libraries and you have no sudo, `apt-get download` the listed packages, unpack them with `dpkg-deb -x` into a scratch directory and set `LD_LIBRARY_PATH` plus `PLAYWRIGHT_SKIP_VALIDATE_HOST_REQUIREMENTS=1`.
It checks Memory Home (opening state, Ask / Find memories, results in place, cited source → graph, command bar, `G`, Explore graph framing, graph interaction outside the card, working without the graph), Insights (cards, SVG chart and hover, filters flowing into the request, contact → graph focus, loading/empty/offline/error states, escaping), arrow-key movement, responsive layouts from 390px phones to 1440px laptops, and the older graph checks: layout fidelity, orbit/pan/zoom, hover, selection and its links, filters, timeline, find, reset/Esc, refresh, memory search and Ask Sarthink (with stubbed, synthetic API answers: rendering, HTML escaping, filters, timeline/source focus, index-building retry), API-offline/index-unavailable states and layout fallbacks (by rewriting responses in the browser, never on disk). `SARTHINK_E2E_SEMANTIC=1` adds a real memory search.
