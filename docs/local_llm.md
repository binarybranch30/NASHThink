# Local LLM: written answers for Ask Sarthink (optional, CPU-only)

Ask Sarthink first finds and grades the evidence in your memories, with no language model involved (see
`scripts/api/memory_brief.py`). A local Llama can then turn that evidence into a short written answer
that cites its sources as `[n]`. The model only sees the question and the evidence sources, runs in a
llama.cpp server bound to `127.0.0.1`, and nothing is sent to an external API.

Pick how Ask answers with the selector next to the **Ask** button:

| Style | Model | Port | Time on this CPU | Notes |
| --- | --- | --- | --- | --- |
| **Best** | Llama 3.1 8B Instruct | 8083 | about 4 min (≈2.5 min reading the evidence, then ~2.5 tok/s) | better synthesis; the default when it's running |
| **Quick** | Llama 3.2 3B Instruct | 8082 | about 1 min (≈30 s reading, then ~8 tok/s) | follows the rules less reliably |
| **Evidence only** | none | — | instant | the deterministic brief with quotes and counts |

While the model reads the evidence, Ask shows a loading screen with its steps (searching, checking the evidence,
reading N sources) and a timer; the map highlight appears at once. The written answer then streams in on top, and
everything it rests on (confidence, evidence summary, timeline, source cards) is folded into a **Sources (n)**
dropdown below it. **Stop** on the loading screen cancels the model and shows the evidence summary instead.

**No relevant info found.** Ask says so plainly, and the sources become **Closest matches** (leads, not evidence), when:
- the evidence is weak (low confidence): no model runs, because it would only be guessing;
- the model decides the sources don't answer the question: it is told to reply exactly `NO_RELEVANT_INFO`, and the
  server holds back the first tokens until they can't be that reply, so it never shows up as text;
- the finished answer isn't grounded: it cites no source, or most of its quotes (at least 2 checked) aren't in the
  sources it cites. What the model wrote is still available, folded, under *Show what … wrote anyway*.

## Start and stop

```bash
scripts/llm.sh start best      # or: start quick
scripts/llm.sh status
scripts/llm.sh stop            # both; or: stop best
```

The Sarthink API doesn't need a restart; the selector marks models that aren't running as *(off)*.
`llm.sh` binds to `127.0.0.1` only, uses `-t 2 -tb 3 -np 1` (this 6-vCPU host is shared, and more threads
are slower: `-t 4` gave about 1.4 tok/s against 9–10 tok/s at `-t 2` for the 3B), logs to `logs/llm-<profile>.log`,
and refuses to start when free memory is short. With about 6.5 GB free next to the API, **run one model at a
time**: the 8B needs about 5.4 GB, so stop the other with `scripts/llm.sh stop quick` first.
Port 8081 is used by another system user's service, so it is never used here.

## Models

| Item | Best | Quick |
| --- | --- | --- |
| Model | Meta Llama 3.1 8B Instruct, Q4_K_M GGUF | Llama 3.2 3B Instruct, Q4_K_M GGUF |
| File | `models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` (4.92 GB) | `models/Llama-3.2-3B-Instruct-Q4_K_M.gguf` (2.02 GB) |
| Source | https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF | https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF |
| SHA-256 | `7b064f5842bf9532c91456deda288a1b672397a54fa729aa665952863033557c` | `6c1a2b41161032677be168d354123594c0e6e67d2b9227c84f296ad037c728ff` |
| Licence | Llama 3.1 Community License Agreement | Llama 3.2 Community License Agreement |
| Context | 6144 tokens, 8-bit KV cache (`-fa on -ctk q8_0 -ctv q8_0`) | 4096 tokens |
| Evidence sent | up to 8 sources × 1200 chars | up to 5 sources × 700 chars |

Both are Meta Platforms, Inc. models: "Built with Llama". The Llama Acceptable Use Policy applies.
Runtime: llama.cpp `b11202`, the official prebuilt `llama-b11202-bin-ubuntu-x64.tar.gz` in
`.tools/llama.cpp/llama-b11202/` (MIT licence). `.tools/`, `models/`, `*.gguf` and `logs/` are gitignored.

Download the 8B model (resumable), check it, then start it:
```bash
curl -fL -C - -o models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
  https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
sha256sum models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
```

## Profiles and a future GPU

Profiles (model file, port, threads, context, how much evidence is sent, answer length, temperature and
extra llama.cpp arguments) live in `scripts/api/llm_config.py`. Copy `config/llm_profiles.json.example` to
`config/llm_profiles.json` (gitignored) to override any of them. On a machine with a GPU (and a CUDA build of
llama.cpp), adding `"-ngl", "99"` to `extra_args` offloads the model; with the time saved, raise `ctx`,
`max_sources` and `source_chars`, or point `model` at a larger GGUF.

## How the answer is grounded (`scripts/api/answer_writer.py`)

- Only sources that Ask judged to be **evidence** are sent, numbered with the same `[n]` the UI shows. For long
  conversations, the lines that mention the question's words are kept first, with their neighbouring lines,
  in their original order. The prompt is sized with the server's own tokenizer (`POST /tokenize`) to the
  profile's `prompt_tokens` (Best 2,000, Quick 1,400): excerpts shrink first, then the lowest-ranked sources
  are dropped. This matters for Hinglish, which takes about twice as many tokens per character as English.
- Answers are always in English. Each claim carries a short exact quote. Quotes are checked against the cited
  source, and the UI underlines any that weren't found. Hinglish sources come with a shorthand glossary, and
  heavy content (suicide or self-harm) must be quoted exactly, never reworded. See `docs/hinglish.md`.
- The system prompt names your persona and handles from `config/identity_map.json` so the model can tell
  your messages from other people's. It is told to use only the sources, to cite `[n]` after each claim, to say
  when the sources don't answer the question, and never to invent events, feelings, relationships or dates.
- Citations to numbers that aren't evidence sources are removed. The UI escapes the text and turns `[n]` into
  a button that selects that source on the map.
- It is still a language model, so it can misread a conversation. The answer says it was drafted by a local
  model; check the cited sources.

## API

`POST /api/ask/stream` takes the `/api/ask` body plus `"profile": "best" | "quick"` and answers `text/event-stream`:

| Event | Data |
| --- | --- |
| `brief` | exactly the `/api/ask` response (sent first, within seconds) |
| `status` | `{stage: "reading" \| "writing", elapsed_s, sources, profile}` every ~5 s until the first token |
| `token` | `{text}` |
| `done` | `{text, grounded, ungrounded_reason, cited, quotes: [{text, ok, source}], unverified, profile, model, tokens, prompt_tokens, took_ms}`; `text` has cleaned citations; `grounded: false` with `ungrounded_reason` `no_citations` \| `unverified_quotes` \| `model_unsure` means the page shows "No relevant info found" |
| `skipped` | `{code: "no_relevant_info", message}` when the evidence is too weak to write from, or the model replied `NO_RELEVANT_INFO` |
| `error` | `{code: llm_offline \| llm_loading \| llm_failed \| llm_superseded, message}` |

Retrieval errors (for example `index_unavailable`) come back as ordinary JSON errors, as with `/api/ask`.
Closing the connection stops generation. A new question for the same model replaces the one being written,
because each server handles one request at a time. `GET /api/llm` (also inside `/api/health` as `llm`) reports
each profile's `state` (`online`, `loading`, `offline`), its model and its start command.

```bash
curl -N -X POST http://127.0.0.1:8000/api/ask/stream -H 'Content-Type: application/json' \
  -d '{"question": "What did I discuss about Python?", "profile": "quick"}'
```

## Measurements (2026-09-27, AMD EPYC 6 vCPU, 11 GB RAM, no swap, CPU shared with other load, `-t 2`)

| Measure | Quick (3B) | Best (8B) |
| --- | --- | --- |
| Start to `/health` ok | ~10 s | ~25 s |
| Evidence prompt | 850–950 tokens | ~1430 tokens |
| Prompt processing | ~29 tok/s (≈30 s) | ~9 tok/s (≈2.6 min) |
| Generation | ~8 tok/s | ~2.5 tok/s |
| "What did I discuss about Python?" end to end | 51–66 s | 3 min 59 s |

Earlier 3B benchmark (2026-09-26): cold start 5.7–8.4 s, generation 6–10 tok/s, peak RSS ~3.9 GB including
mmapped model pages.

Manual check of a server:
```bash
curl -s http://127.0.0.1:8083/health          # {"status":"ok"} once loaded
ss -ltnp | grep -E ':808[23]'                  # must show 127.0.0.1 only
```
