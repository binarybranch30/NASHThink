#!/usr/bin/env bash
# Report the local Sarthink API's health, its URL, the semantic index and whether the optional local
# Llama (llama.cpp) servers that write Ask answers are running. Read-only: starts, stops and changes nothing.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${SARTHINK_PORT:-8000}"
PIDFILE="${SARTHINK_PIDFILE:-$ROOT/processed_data/api.pid}"
LLAMA_PORTS="${SARTHINK_LLAMA_PORTS:-8082 8083}"   # quick, best (scripts/llm.sh); 8081 belongs to another user
URL="http://127.0.0.1:$PORT"

echo "Sarthink status"
health="$(curl -sf -m 3 "$URL/api/health" 2>/dev/null || true)"
if [ -n "$health" ]; then
  echo "  API        : running   $URL/"
  printf '%s' "$health" | python3 -c '
import json, sys
h = json.load(sys.stdin)
ix = h.get("index") or {}
table = ix.get("table")
if ix.get("available"):
    loaded = "loaded" if ix.get("model_loaded") else "loads on first query"
    print("  Semantic   : table %r available, %s chunks, model %s" % (table, ix.get("rows") or "?", loaded))
else:
    print("  Semantic   : table %r NOT available (%s)" % (table, ix.get("reason", "unknown")))
db = h.get("memory_db")
if db is not None:
    print("  Memory DB  : " + ("available (Insights ready)" if db.get("available") else "missing (Insights unavailable)"))
' 2>/dev/null || echo "  Semantic   : (could not parse /api/health)"
else
  echo "  API        : not responding on $URL (start it with scripts/start_sarthink.sh)"
fi

listen="$(ss -ltnpH "sport = :$PORT" 2>/dev/null | awk '{print $4}' | sort -u | tr '\n' ' ')"
[ -n "$listen" ] && echo "  Listening  : $listen"
case "$listen" in *0.0.0.0*|*"[::]"*|"*:"*) echo "  WARNING    : port $PORT is bound to all interfaces" ;; esac
if [ -f "$PIDFILE" ]; then
  pid="$(cat "$PIDFILE")"
  if kill -0 "$pid" 2>/dev/null; then echo "  PID file   : $pid (running)"; else echo "  PID file   : $pid (stale)"; fi
fi

for lp in $LLAMA_PORTS; do
  lh="$(curl -sf -m 2 "http://127.0.0.1:$lp/health" 2>/dev/null || true)"
  if printf '%s' "$lh" | grep -q '"status":"ok"'; then
    model="$(curl -sf -m 2 "http://127.0.0.1:$lp/v1/models" 2>/dev/null | python3 -c 'import json,sys; d=json.load(sys.stdin); print(", ".join(m.get("id","?") for m in d.get("data",[])))' 2>/dev/null || true)"
    echo "  Local Llama: running at 127.0.0.1:$lp ${model:+($model)} — writes Ask answers"
  else
    echo "  Local Llama: not running at 127.0.0.1:$lp (optional; scripts/llm.sh start quick|best)"
  fi
done
exit 0
