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

The evidence (sources, timeline, map highlight) always appears at once. The written answer streams in above it
while it's being generated. When the evidence is weak (low confidence), no model runs, because it would only
be guessing.

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
  in their original order.
- The system prompt names your persona and handles from `config/identity_map.json` so the model can tell
  your messages from other people's. It is told to use only the sources, to cite `[n]` after each claim, to say
  when the sources don't answer the question, never to invent events, feelings, relationships or dates, and
  to answer in the question's language.
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
| `done` | `{text, cited, profile, model, tokens, prompt_tokens, took_ms}`; `text` has cleaned citations |
| `skipped` | `{code: "weak_evidence", message}` when the evidence is too weak to write from |
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
