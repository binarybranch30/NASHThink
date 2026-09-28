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
- **Everything runs on this machine.** Parsing, the SQLite memory database, embeddings (CPU), semantic search, Ask Sarthink and Insights are all local. No archive text, prompt, query or retrieved memory is sent to a hosted API or LLM service, with two opt-in exceptions: the **DeepSeek** answer styles (chosen per question in the answer picker; the question and the retrieved excerpts go to DeepSeek's API, and the UI labels those answers "online"), and `scripts/semantic/summarizer.py`, which calls a hosted LLM and is not needed.
- **Nothing is exposed publicly.** The API binds to `127.0.0.1` only; reach it from a laptop through an SSH tunnel (below), never by binding `0.0.0.0`.
- **Private data never enters git.** `archive/`, `incoming/`, `processed_data/` (database, index, graph CSVs, logs), `config/identity_map.json`, `.venv/`, `.cache/`, local models (`models/`, `*.gguf`) and tools (`.tools/`) are gitignored.
- **Read-only where it can be.** The graph export and `/api/insights` open the database read-only; the UI escapes every archive-derived string before inserting it into the page.
- **Ask Sarthink needs no LLM.** Evidence is always found and graded without a language model (quotes and counts, see below). Optionally, a model turns that evidence into a written answer with citations (`docs/local_llm.md`): a local Llama (Llama 3.1 8B or 3.2 3B via llama.cpp on `127.0.0.1`, nothing leaves the machine), or DeepSeek Chat / DeepSeek Reasoner over the internet (seconds instead of minutes; needs `DEEPSEEK_API_KEY` in the gitignored `.env`). Nothing else depends on it.

## Run the demo

```bash
scripts/start_sarthink.sh     # starts the API + UI on http://127.0.0.1:8000/ (nohup, logs to processed_data/api.log); no-op if already healthy
scripts/status_sarthink.sh    # API health, URL, semantic index, memory DB, local Llama answer writers (127.0.0.1:8082/8083) — read-only
scripts/llm.sh start best     # optional: Llama 3.1 8B writes Ask answers (slow on CPU); "start quick" for Llama 3.2 3B
scripts/stop_sarthink.sh      # stops only Sarthink's API process (scripts/api/server.py on port 8000)
```
From a laptop, forward the port over SSH and open `http://127.0.0.1:8000/` locally:
```bash
ssh -N -L 8000:127.0.0.1:8000 naitik@185.2.102.128
```

**Sample data and your own data (workspaces).** With `config/workspaces.json` (gitignored), one server holds
several data folders and every browser starts on the default one. The hackathon setup keeps a fictional sample archive
in `/home/naitik/sarthink-demo` (its own `archive/`, `processed_data/` and identity map) as the default, and this
checkout's personal data as a second workspace behind a password. The small `Sample data ▾` control in the
bottom-left corner unlocks it (a signed, HttpOnly cookie for 8 hours) and switches back; a server restart puts every
browser back on the sample data. Only a salted PBKDF2 hash of the password is stored:
`python3 scripts/utils/set_workspace_password.py personal`. Without the config file the server works exactly as before.
```json
{"default": "demo",
 "workspaces": {"demo": {"label": "Sample data", "root": "/home/naitik/sarthink-demo"},
                "personal": {"label": "My data", "root": "/home/naitik/sarthink-changes", "password": "pbkdf2_sha256$..."}}}
```
Build a workspace by running the pipeline (parsers, `export_cosmograph.py`, `compute_layout.py`, `chunk_builder.py`,
`embedder.py`) from a copy of `scripts/` inside that folder: every script reads and writes data next to itself.

**90-second demo flow**
1. **Home** (0–10s): the page opens on one question box above the slowly drifting memory map. Everything else is a quiet link: *Explore the memory map*, *See insights*.
2. **Ask** (10–30s): click a sample question, e.g. *How has my interest in photography changed?* The answer appears in place with a confidence level, key points, a timeline and the sources it quotes.
3. **Cited source → map** (30–40s): click a source. The page glides into the memory map with that conversation selected and its people lit up.
4. **Your history with a person** (40–65s): click one of those people. The profile leads with a short cited brief (“wrote 90 messages in 3 conversations… you both wrote in…”), a few numbers, a month-by-month strip and notable conversations. Click a citation number to jump to that conversation, then **‹ Back to …**. Open **Show full history** to page through the actual messages, newest first.
5. **Filters** (65–75s): open **Filters**, switch a platform off or pick *12 mo*; the map, the profile (now labelled *Filtered*, with all-time counts alongside) and the next answer follow.
6. **Insights** (75–90s): press `I` for totals, monthly activity, platform breakdown, and top contacts and conversations; click a contact to open their profile.

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
│   ├── discord/<name>/         # Official Discord data package, staged by stage_discord_export.py (or DiscordChatExporter JSON)
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
- **Reddit**: Run `python3 scripts/context/reddit_fetch_context.py`. It needs `asyncpraw` and `pandas`
  (`.venv/bin/pip install asyncpraw pandas`) and a Reddit *script* app's `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`,
  `REDDIT_USER_AGENT`, `REDDIT_USERNAME` and `REDDIT_PASSWORD` in the environment (see `.env.example`; Reddit may
  require API access approval for new apps). It only writes JSON files under `processed_data/context/reddit/`, never the
  database, and skips items already fetched, so it can be stopped and resumed.

#### Discord data package
Request your data at Discord → Settings → Privacy & Safety → *Request all of my data*, put `package.zip` in
`incoming/discord/`, then stage only what the parser needs (messages, channel titles, and your account id and
friends' names; the multi-gigabyte `Activity/` analytics, avatars, billing and contact details are left in the ZIP):
```bash
python3 scripts/utils/stage_discord_export.py --expected-size <bytes of the original file>   # -> archive/discord/naitik (0700/0600)
python3 scripts/parsers/discord_parser.py --dry-run    # counts only
python3 scripts/parsers/discord_parser.py
```
The staging step checks the size, every member's CRC and path (absolute paths, `..`, symlinks and encrypted members
are refused) and free disk space before writing anything. The package contains **only messages you sent**; the parser
confirms your account from `Account/user.json` (its id must be in every DM's recipients, otherwise it stops rather than
guess) and records the other people in DMs and group DMs as conversation members (`ThreadMembers`), so they appear on
the map and in *Your history with …* with your side of the conversation. Timestamps are stored as the original UTC
times (the package's `Timestamp` matches each message ID's snowflake time). Attachments are kept as
`[Attachment: file name]` and never downloaded; empty messages are skipped. Re-running is idempotent: unchanged
messages are left alone, edits in a newer export are updated, and ids stay stable.

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
python3 scripts/semantic/chunk_builder.py --platform discord --output processed_data/semantic/discord_chunks.json   # one platform
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

**Adding one platform without re-embedding the rest** (e.g. after a new Discord import):
```bash
nohup env HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 .venv/bin/python scripts/semantic/embedder.py \
  --input processed_data/semantic/discord_chunks.json --topics-table topics --replace-platform discord \
  > processed_data/logs/embed_discord.log 2>&1 &
```
`--replace-platform` swaps only that platform's rows in the existing table, embedding with the table's recorded model;
it refuses mixed-platform input, a different model or dimension, or a missing table. Re-running replaces rather than
duplicates. The metadata entry keeps its model and creation time and gains `platform_updates` with the LanceDB version
before and after, so `lancedb.connect(path).open_table("topics").restore(<version_before>)` undoes it. A running server
keeps the table version it opened until it is restarted.

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
Nodes are people (`U_<id>`) and threads (`T_<id>`); an edge means that person wrote in that thread, or, with `weight` 0, that they were in it without any message of theirs in the export (Discord DM partners). Besides `id, label, group, size, color`, nodes carry `platform, kind (user|thread), messages, first_ts, last_ts, title` and edges carry `weight, first_ts, last_ts` (epoch seconds, UTC), which drive the graph's details panel and timeline filter. Older exports without these columns still load; the timeline then explains how to enable it.

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

**Home** is the opening screen: one question box with **Ask** (default) and **Find** (plain semantic search) modes, sample prompts, and a one-line status. The memory map keeps drifting behind it and stays interactive outside the card (drag, zoom, click a node to open it). Answers and results appear in the card; clicking a cited source or result opens the **memory map** focused on that conversation. **Explore the memory map** (or `G`) opens it directly; there the same box sits in the top bar (it is the only place to ask or search), results move to the right-hand panel, and `G` / **Home** goes back. If the graph CSVs are missing, Memory Home says so (with the commands to build them) and Ask, search and Insights keep working. Links with `#node=T_12` or `?view=graph` open straight into the workspace.

**Insights** (`I`, or the button in Memory Home / the top bar) summarises the memory database for the current platform toggles and timeline range: message, conversation, people and platform totals, first and latest memory, messages per month (hover for the per-platform split) and per year, a per-platform breakdown with date ranges, and the top contacts and conversations (click one to focus it on the graph). Your own accounts, as listed in `config/identity_map.json`, are excluded from contacts and people counts.

**Your history with a person.** Clicking a person (a filled dot) opens a profile instead of a list of every link. It is built by `GET /api/person/U_<id>` from the SQLite database only (read-only, no embedding model), and shows:
- a short **brief** whose sentences are counts, dates and conversation titles, each with numbered citations; click a number to open that conversation on the map, where **‹ Back to …** returns to the profile. Sparse histories say *There isn’t enough history to summarize*; automated or deleted accounts are flagged; nothing about feelings or relationships is inferred;
- conversations, their messages, your replies, first and latest message, platforms; a month-by-month strip; **recurring words** (in at least two conversations, or in three different months of one long chat); up to five **notable conversations** — back-and-forth ones first, large public threads (over 12 people) last so one huge thread can't dominate; and other people from the same small conversations;
- **View conversations** (20 at a time, latest first) and **Show full history** (30 messages at a time, newest first, grouped by day, with *Load older*; click a message's name to see its time — an opened conversation shows every time). Your own messages are included only for small conversations (up to 12 people). Very long messages are shortened on screen only; the database keeps everything;
- with Filters on, the numbers follow them and the profile says *Filtered*, with the all-time counts next to it.
Accounts are never merged by name: if another platform has an account with the same name, the profile says so and links to it separately. Only your own accounts, as mapped in `config/identity_map.json`, are combined (clicking yourself shows *Your own activity*).

**Dates and times** are shown in UTC everywhere (the database stores UTC epoch seconds, untouched). Collapsed cards show one date or range (hover for the exact span); expanded text is grouped by day with times.

**Using the map** (press `?` in the page for the same list, including the optional keyboard shortcuts):
- **Left-drag** orbits, **right-drag** or **Shift/Ctrl-drag** pans, **scroll** zooms toward the cursor. Dragging never moves nodes; any drag cancels a running camera flight.
- **Hover** a node for its name, platform, connections and active months. **Click** selects it: the node, every direct neighbour and the links between them stay bright (drawn on top), everything else is dimmed, not hidden. **Double-click** (or `F`) flies the camera to it. Click empty space to deselect.
- For a conversation, the **details panel** shows its people, messages, the dates it was active, and searchable **excerpts** from `/api/thread` (one date label each; *Show more* lays the messages out by day with their times); for a person, the profile above. The selection is kept in the URL (`#node=T_12`), so a refresh or a shared local link reopens it.
- **Arrow keys** move the graph in the pressed direction (hold to keep moving); **Shift + arrows** orbit. They only act on the graph when you're not typing or in a list/tab strip.
- **Find a person or conversation** (`/`) matches names and titles, highlights all matches, and `↑ ↓ Enter` selects one.
- **Filters** (folded by default; a badge shows how many are on): **Platforms** (**only** isolates one), **Show** people and/or conversations, and **Time range** — a histogram of active conversations per month with handles and 3/12-month presets.
- The top bar shows how many people and conversations there are and what the filters currently show. The bottom tips fade after a while (*Got it* hides them).
- **↺ Reset** (or `Esc`) restores the original full view: clears the selection, find, search results and filters, and flies the camera home.
- If a node has missing or invalid `layout_*` values it is placed near its platform; if the whole layout is missing or collapsed to one point, a deterministic fallback layout is drawn and the top bar says `approximate layout`.

Switch the question box to **Find**, type and press Enter:
- Results appear in a panel on the right (platform, date, people, a match-strength bar and label, snippet with the query words marked, and why it matched: shared words, a Hinglish/English topic match, or "similar in meaning").
- Matching conversation threads and their participants are highlighted on the graph and the camera flies to them; the rest of the graph is dimmed.
- Click a result to select its thread (its links, details and full chunk text); the other results stay highlighted. Click again to unfocus. **Find related memories** in a thread's details runs a search with its title.
- **Clear** in the results panel removes only the memory search; **Reset** / `Esc` resets everything.
- The first query loads the embedding model, so it takes longer than the rest.

#### Ask Sarthink (evidence-first memory Q&A)
Type a question in the box (**Ask** is the default; or click a sample prompt on Home) and press Enter (Shift+Enter for a new line):
- *What was I stressed about during college?* · *How has my interest in photography changed?* · *What did I discuss about Python?* · *What was I working on around August 2026?*

The right panel shows a short **answer**, a **confidence** level (high / medium / low), **key points**, a small chronological **timeline** and the **source cards**. Sources and their threads are highlighted on the graph; clicking a source card or a timeline entry selects and frames its thread exactly like a memory-search result. The question is limited to the platforms switched on in the sidebar and the timeline range (the scope line under the box shows what will be sent). Cards marked *closest match only* were retrieved but are not used as evidence.

**Written answers (optional, local Llama).** The selector next to **Ask** chooses how the answer is written:
**Best** (Llama 3.1 8B, about 4 minutes on this CPU), **Quick** (Llama 3.2 3B, about a minute) or **Evidence only**
(instant, the deterministic brief below). With a model chosen and running (`scripts/llm.sh start best`), a loading
screen shows the steps and a timer while the model reads, then a written answer streams in on top: a few short paragraphs
in English (quotes keep the original words), with `[n]` citations you can click to open that source. The sources, timeline
and deterministic evidence summary are folded into a **Sources (n)** dropdown below the answer. The model sees only the
question and the sources judged to be evidence, is told not to add anything that isn't in them, and doesn't run at all
when the evidence is weak. When the evidence is weak, the model says the sources don't answer the question, or its answer
cites nothing (or misquotes most of what it cites), Ask shows **No relevant info found** instead of a guess. It is still a
language model, so check the cited sources. Setup, models, timings and the `/api/ask/stream` event format are in `docs/local_llm.md`.

How the evidence brief is made (`scripts/api/memory_brief.py`, deterministic, no language model):
1. The question is embedded by the same already-loaded model as memory search, and the 60 nearest chunks are retrieved from `topics`, with the platform/date filters applied inside LanceDB before ranking. A month or year in the question (*around August 2026*, *in 2025*) becomes a date window when you haven't set one (±1 month for "around"), and is removed from the embedded text.
2. Each chunk is checked against the question: it counts as **evidence** only if it is similar enough, has real content (tiny "ok"/"lol" chunks score high on similarity but are never used), and mentions the question's content words (question scaffolding like *what*, *discuss*, *changed*, *working* is ignored; longer questions need at least half of their words). Dates come from the message timestamps inside the chunk.
3. Chunks repeating one another (overlapping session windows, crossposts) are merged, keeping the best one.
4. The answer states how many memories matched, on which platforms and over which months, then quotes the strongest sentence verbatim with its author, date and platform (for "how has … changed" questions: the earliest and the latest). Key points quote one sentence per month and platform.
5. Confidence is **high** with at least 3 supporting sources, 2 of which cover every content word; **medium** with fewer or partial support (the answer says so); **low** when nothing qualifies, in which case the answer says "No relevant info found" and the closest chunks are shown as leads.

**Hinglish.** About 15% of the archive is Hinglish (Hindi in English letters). The embedding model matches romanised Hindi by style ("yaar", "hai", "nahi") rather than topic, so `scripts/semantic/hinglish.py` adds a small curated layer used by search, Ask and the written answers: chat shorthand and spelling variants are normalised (h → hai, ni/nhi → nahi, clg → college), filler and negations are never topics (negations are never cut from quotes), topic words match across languages (sleep ↔ neend, fight ↔ ladai/jhagda, alone ↔ akela, parents ↔ mummy/papa, …), and any other Hinglish word gets a spelling-tolerant whole-word pattern (ladai ~ ladaai ~ ladayi). A Hinglish question runs a second search with the same query vector restricted to chunks containing its topic (one LanceDB `regexp_like` pass, ~0.25 s, no index), interleaved into the ranking; results say `matched_via` / `expansion_terms`. Thread titles only match whole words, a generic word alone (ghar, dost) doesn't count as evidence, and long chunks show the lines around the matches. Written answers are always in English, sized with the model's own tokenizer, get a shorthand glossary, and have their quotes checked against the cited source. On a fixed 12-question set the share of "relevant" sources that are really on topic went from 59% to 93% (`scripts/semantic/eval_hinglish.py`; details in `docs/hinglish.md`).

The evidence brief itself is **evidence-based synthesis, not a generative LLM**: every sentence is a verbatim quote, a count, a date, a platform or a title from a returned source, so it never invents an event, feeling, relationship or date, but it also doesn't interpret or summarise in its own words, can quote a sentence out of context, and depends on the words you use. Read the sources. Everything runs on this machine: no hosted APIs, no hosted LLM service, no browser-side calls other than to the local server, no model downloads at query time.

If the index is still being built, the status line says so and the API answers `503 index_unavailable`; the server picks the table up automatically once it exists, no restart needed. If the page can't reach the API it shows the command to start it.

API (interactive docs at `/api/docs`):
```bash
curl http://127.0.0.1:8000/api/health
curl -X POST http://127.0.0.1:8000/api/search -H 'Content-Type: application/json' \
  -d '{"query": "college ke baare mein stress", "limit": 10}'
curl http://127.0.0.1:8000/api/thread/T_12      # indexed chunks of one thread (no model load)
curl 'http://127.0.0.1:8000/api/insights?platforms=reddit,instagram&date_from=2025-01-01&date_to=2025-12-31'   # read-only aggregates
curl http://127.0.0.1:8000/api/person/U_12                             # "Your history with …" profile (read-only)
curl 'http://127.0.0.1:8000/api/person/U_12/conversations?offset=0&limit=20'
curl 'http://127.0.0.1:8000/api/person/U_12/messages?limit=30'        # then &cursor=<next_cursor>; &thread=T_7 for one conversation
curl -X POST http://127.0.0.1:8000/api/ask -H 'Content-Type: application/json' \
  -d '{"question": "How has my interest in photography changed?", "limit": 8, "platforms": ["reddit", "instagram"], "date_from": "2025-01-01", "date_to": "2026-09-30"}'
```
`/api/ask` takes `question` (required, ≤500 chars), `limit` (sources, 1–20, default 8), `platforms` (optional list; omit for all) and `date_from` / `date_to` (optional ISO dates or datetimes; a plain `date_to` includes that whole day, a datetime is exclusive; a chunk matches when its messages overlap the range). It returns `{question, answer, confidence, summary_points, timeline: [{date, label, node_id, platform, source}], sources: [{rank, node_id, title, platform, date_start, date_end, similarity, snippet, text, people, relevant, matched_terms}], notes, evidence: {retrieved, considered, relevant, terms}, filters, model, took_ms}`. `timeline[].source` is the 1-based index into `sources`; `notes` explains filtering and merging. Errors use the same codes as `/api/search`.
`/api/search` returns `{query, table, model, count, took_ms, results: [...]}`; each result has `rank, similarity, distance, start_time, end_time, platform, title, channel_id, node_id, people, summary, snippet, text`. `node_id` (`T_<thread id>`) is the matching graph node. Errors are `{"error": {"code", "message"}}` with codes `invalid_request` (422), `index_unavailable` (503), `model_unavailable` and `search_failed` (500).

The graph alone still works from any static server (`python3 -m http.server 8080`, then `http://localhost:8080/sarthink_graph.html`); memory search then needs the API running and `?api=http://127.0.0.1:8000` appended to the URL.

`/api/insights` (GET, read-only) takes optional `platforms` (comma-separated or repeated), `date_from` / `date_to` (ISO dates or datetimes; a plain `date_to` includes that day) and `top` (1–50, default 10). It returns `{empty, filters, totals: {messages, threads, people, platforms}, first_date, latest_date, platforms: [{platform, messages, threads, people, first_date, last_date, share}], available_platforms, activity: {months: [{month, total, platforms}], years: [...]}, top_contacts: [{node_id, label, platform, messages, threads, first_date, last_date}], top_conversations: [{node_id, title, platform, messages, people, first_date, last_date}], owner: {configured, excluded_accounts}, took_ms, cached}`. `node_id`s are graph node ids (`U_<id>`, `T_<id>`). The database is opened read-only; results are cached until the database file changes (the first unfiltered call is warmed at server start). A missing database answers `503 database_unavailable`.

`/api/person/U_<id>` (GET, read-only SQLite, never loads the embedding model) takes the same optional `platforms`, `date_from`, `date_to` as Insights and returns `{node_id, label, kind: person|automated|you, platform, accounts, same_name_elsewhere, scope: {filtered, …}, lifetime: {conversations, shared_conversations, their_messages, your_messages, first_date, latest_date, platforms}, stats: {…same, within the filters}, activity: [{month, messages}], topics: [{word, conversations, months, mentions, sources}], sources: [{node_id, title, platform, their_messages, your_messages, people, large, shared, first_date, latest_date, notable}], brief: {sentences: [{text, sources}], sparse}, related, took_ms}`; `brief.sentences[].sources` and `topics[].sources` are 1-based indexes into `sources`. `/conversations` pages the person's conversations (`offset`, `limit` ≤ 100, latest first, `{total, next_offset, items}`); `/messages` pages their messages plus yours in small shared conversations (newest first, keyset `cursor`, `limit` ≤ 100, optional `thread=T_<id>`; `{total (first page only), next_cursor, messages: [{date, author: them|you, author_label, node_id, thread_title, platform, text, clipped}]}`). Errors: `invalid_request` 422, `not_found` 404, `database_unavailable` 503.

`/api/thread/T_<id>` returns `{node_id, channel_id, count, first_time, last_time, chunks: [...]}` with up to 8 chunks (oldest first; `start_time, end_time, platform, title, people, snippet, text`). It only filters the index, so it answers instantly even before the first search has loaded the model.

Tests (fake model, table and a throwaway SQLite DB; no index, model or real data needed):
```bash
.venv/bin/python scripts/tests/test_api.py
.venv/bin/python scripts/tests/test_ask.py      # Ask Sarthink: grounding, weak evidence, filters, errors, model reuse
.venv/bin/python scripts/tests/test_answer_writer.py   # written answers: prompt, citations, streaming, offline/loading model (fake llama.cpp)
.venv/bin/python scripts/tests/test_insights.py # /api/insights: shape, filters, empty state, owner exclusion, read-only, errors
.venv/bin/python scripts/tests/test_people.py   # person profiles: counts, dates, filters, identity boundaries, sparse history, paging, links
.venv/bin/python scripts/tests/test_hinglish.py # Hinglish vocabulary, shorthand, spelling patterns, expanded retrieval on a temp LanceDB table, Ask evidence, negation-safe quotes
python3 scripts/tests/test_graph_pipeline.py
python3 scripts/tests/test_discord_parser.py     # Discord: DMs, group DMs, channels, identity, UTC timestamps, duplicates, malformed records, staging
.venv/bin/python scripts/tests/test_embedding_metadata.py   # includes --replace-platform (incremental index) refusals and idempotency
```
Browser end-to-end test of the graph against your real CSVs (needs the server running and Playwright, which is not a project dependency):
```bash
HF_HUB_OFFLINE=1 .venv/bin/python scripts/api/server.py --port 8765 &
npm install --prefix /tmp/pw playwright && npx --prefix /tmp/pw playwright install chromium   # Node 18: playwright@1.49.1
NODE_PATH=/tmp/pw/node_modules node scripts/tests/graph_ui_e2e.mjs http://127.0.0.1:8765/
```
`SARTHINK_E2E_ONLY=homeFlow,insightsFlow` runs selected flows. If Chromium can't start for missing system libraries and you have no sudo, `apt-get download` the listed packages, unpack them with `dpkg-deb -x` into a scratch directory and set `LD_LIBRARY_PATH` plus `PLAYWRIGHT_SKIP_VALIDATE_HOST_REQUIREMENTS=1`.
It checks the person profile (cited brief, no message dump, citation → conversation → back, paged conversations and full history, filters, sparse/offline states, phone and laptop layouts, escaping, and one live profile from the local API), Memory Home (opening state, Ask / Find memories, results in place, cited source → graph, command bar, `G`, Explore graph framing, graph interaction outside the card, working without the graph), Insights (cards, SVG chart and hover, filters flowing into the request, contact → graph focus, loading/empty/offline/error states, escaping), arrow-key movement, responsive layouts from 390px phones to 1440px laptops, and the older graph checks: layout fidelity, orbit/pan/zoom, hover, selection and its links, filters, timeline, find, reset/Esc, refresh, memory search and Ask Sarthink (with stubbed, synthetic API answers: rendering, HTML escaping, filters, timeline/source focus, index-building retry), API-offline/index-unavailable states and layout fallbacks (by rewriting responses in the browser, never on disk). `SARTHINK_E2E_SEMANTIC=1` adds a real memory search.
