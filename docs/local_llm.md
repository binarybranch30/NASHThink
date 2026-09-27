# Local LLM (optional, CPU-only)

An optional local llama.cpp server. It is meant to turn evidence that Sarthink has
already retrieved into more natural prose. It is **not** connected to `/api/ask`.
All inference runs on this machine; nothing is sent to external APIs.

| Item | Value |
| --- | --- |
| Runtime | llama.cpp `b11202`, official prebuilt `llama-b11202-bin-ubuntu-x64.tar.gz` in `.tools/llama.cpp/llama-b11202/` (MIT licence) |
| Model | Llama 3.2 3B Instruct, Q4_K_M GGUF: `models/Llama-3.2-3B-Instruct-Q4_K_M.gguf` (2.02 GB) |
| Model source | https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF (quantised from `meta-llama/Llama-3.2-3B-Instruct`) |
| SHA-256 | `6c1a2b41161032677be168d354123594c0e6e67d2b9227c84f296ad037c728ff` |
| Licence | Llama 3.2 Community License Agreement, Meta Platforms, Inc. "Built with Llama". Acceptable Use Policy applies. |
| Endpoint | `http://127.0.0.1:8082` (loopback only) |

Port 8081 is already used by another system user's service, so this server uses **8082**.
`.tools/`, `models/`, `*.gguf` and `logs/` are gitignored.

## Start

Run from the repo root:

```bash
mkdir -p logs
nohup .tools/llama.cpp/llama-b11202/llama-server \
  -m models/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
  --host 127.0.0.1 --port 8082 \
  -c 4096 -np 1 -t 2 -tb 3 --no-webui \
  --alias llama-3.2-3b-instruct \
  > logs/llama-server.log 2>&1 &
echo $! > logs/llama-server.pid
```

Never use `--host 0.0.0.0`. The thread counts matter: this 6-vCPU host shares its CPU with
other workloads. On the benchmark day, `-t 4` gave about 1.4 tok/s and `-t 6` about 0.35 tok/s,
while `-t 2` gave about 9–10 tok/s.

## Health check

```bash
curl -s http://127.0.0.1:8082/health          # {"status":"ok"} once loaded
ss -ltnp | grep 8082                           # must show 127.0.0.1:8082 only
```

## Test

```bash
curl -s http://127.0.0.1:8082/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"In one sentence, why is the sky blue?"}],"max_tokens":64,"temperature":0}'
```

The response is OpenAI-compatible. The `timings` field gives tokens/second.

## Stop

```bash
kill "$(cat logs/llama-server.pid)" && rm logs/llama-server.pid
```

## Benchmark (2026-09-26, AMD EPYC 6 vCPU, 11 GB RAM, no swap, CPU shared with other load)

| Measure | Result |
| --- | --- |
| Cold start to `/health` ok | 5.7–8.4 s (model file already in page cache) |
| Short prompt (45 tok in / 44 tok out) | 6.7 s wall |
| Tiny answer (46 in / 13 out) | 3.1–3.4 s wall |
| Evidence-sized prompt (687 in / 35 out) | 34.7 s wall (29 s of that is prompt processing) |
| Prompt processing | ~24 tok/s |
| Generation | ~6–10 tok/s |
| Peak RSS of server | ~3.9 GB (includes mmapped model pages) |
