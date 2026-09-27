#!/usr/bin/env bash
# Stop the local Sarthink API. Only ever signals a process of this user whose command line is
# Sarthink's server (scripts/api/server.py) and that is the recorded pid or listens on the port.
# Never touches other Python, embedding, llama.cpp or unrelated processes.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${SARTHINK_PORT:-8000}"
PIDFILE="${SARTHINK_PIDFILE:-$ROOT/processed_data/api.pid}"

is_sarthink() {   # pid -> true if it is this user's scripts/api/server.py process
  local pid="$1"
  [ -n "$pid" ] && [ -r "/proc/$pid/cmdline" ] || return 1
  [ "$(stat -c %u "/proc/$pid")" = "$(id -u)" ] || return 1
  tr '\0' ' ' < "/proc/$pid/cmdline" | grep -Eq '(^|[ /])scripts/api/server\.py( |$)'
}

candidates=()
if [ -f "$PIDFILE" ]; then candidates+=("$(cat "$PIDFILE")"); fi
# Also the process listening on 127.0.0.1:PORT (covers a server started by hand).
while read -r pid; do candidates+=("$pid"); done < <(ss -ltnpH "sport = :$PORT" 2>/dev/null | grep -o 'pid=[0-9]*' | cut -d= -f2 | sort -u)

targets=()
for pid in "${candidates[@]}"; do
  if is_sarthink "$pid" && [[ ! " ${targets[*]-} " =~ " $pid " ]]; then targets+=("$pid"); fi
done

if [ ${#targets[@]} -eq 0 ]; then
  echo "No Sarthink API process found (port $PORT)."
  rm -f "$PIDFILE"
  exit 0
fi

for pid in "${targets[@]}"; do
  echo "Stopping Sarthink API (pid $pid)…"
  kill -TERM "$pid" 2>/dev/null || true
done
for _ in $(seq 1 20); do
  alive=()
  for pid in "${targets[@]}"; do kill -0 "$pid" 2>/dev/null && alive+=("$pid"); done
  [ ${#alive[@]} -eq 0 ] && break
  sleep 0.5
done
for pid in "${targets[@]}"; do
  if kill -0 "$pid" 2>/dev/null && is_sarthink "$pid"; then
    echo "pid $pid did not exit after 10s; sending SIGKILL." >&2
    kill -KILL "$pid" 2>/dev/null || true
  fi
done
rm -f "$PIDFILE"
echo "Stopped."
