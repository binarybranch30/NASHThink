#!/usr/bin/env bash
# Start, stop or check the local Llama servers that write Ask Sarthink's answers (llama.cpp, loopback only).
#   scripts/llm.sh start best      # Llama 3.1 8B on 127.0.0.1:8083 (slow on CPU, better answers)
#   scripts/llm.sh start quick     # Llama 3.2 3B on 127.0.0.1:8082 (faster, weaker)
#   scripts/llm.sh stop [quick|best]      # default: both
#   scripts/llm.sh status
# Profiles (model, port, threads, context, extra args such as -ngl for a GPU) live in scripts/api/llm_config.py,
# overridable in config/llm_profiles.json. Logs: logs/llm-<profile>.log. The Sarthink API needs no restart.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGDIR="$ROOT/logs"
PROFILES="quick best"
HEADROOM_MB=768    # left free for other processes on this shared host (the running API is already counted as used)

usage() { echo "usage: $0 start <quick|best> | stop [quick|best] | status" >&2; exit 2; }

field() {   # field <profile> <key>
  python3 "$ROOT/scripts/api/llm_config.py" "$1" | python3 -c 'import json,sys; v=json.load(sys.stdin)[sys.argv[1]]; print(" ".join(map(str,v)) if isinstance(v,list) else v)' "$2"
}
healthy() { curl -sf -m 2 "http://127.0.0.1:$1/health" 2>/dev/null | grep -q '"status":"ok"'; }
pidfile() { echo "$LOGDIR/llm-$1.pid"; }
running_pid() {
  local f; f="$(pidfile "$1")"
  [ -f "$f" ] && kill -0 "$(cat "$f")" 2>/dev/null && cat "$f"
}

start() {
  local p="$1" port model bin
  port="$(field "$p" port)"; model="$(field "$p" model_path)"; bin="$(field "$p" llama_server)"
  if healthy "$port"; then echo "$p is already running on 127.0.0.1:$port"; return 0; fi
  if ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq "(^|:)$port\$"; then
    echo "Port $port is in use by something else; not starting $p. Check: ss -ltnp | grep :$port" >&2; return 1
  fi
  [ -x "$bin" ] || { echo "Missing llama.cpp server: $bin (see docs/local_llm.md)" >&2; return 1; }
  [ -f "$model" ] || { echo "Missing model file: $model (see docs/local_llm.md)" >&2; return 1; }

  local need_mb avail_mb
  need_mb=$(( $(stat -c %s "$model") / 1048576 + $(field "$p" ctx) / 8 + HEADROOM_MB ))
  avail_mb=$(( $(awk '/^MemAvailable:/ {print $2}' /proc/meminfo) / 1024 ))
  if [ "$avail_mb" -lt "$need_mb" ]; then
    echo "Not enough free memory for $p: ${avail_mb} MB available, about ${need_mb} MB needed." >&2
    for other in $PROFILES; do
      [ "$other" != "$p" ] && [ -n "$(running_pid "$other")" ] && echo "Stop the other model first: $0 stop $other" >&2
    done
    return 1
  fi

  mkdir -p "$LOGDIR"
  local log="$LOGDIR/llm-$p.log"
  echo "--- $(date -u +%FT%TZ) llm.sh start $p" >> "$log"
  # shellcheck disable=SC2046
  nohup "$bin" -m "$model" --host 127.0.0.1 --port "$port" \
    -c "$(field "$p" ctx)" -np 1 -t "$(field "$p" threads)" -tb "$(field "$p" batch_threads)" \
    --no-webui --alias "$(field "$p" alias)" $(field "$p" extra_args) \
    >> "$log" 2>&1 < /dev/null &
  local pid=$!
  echo "$pid" > "$(pidfile "$p")"
  echo -n "Starting $p (pid $pid) on 127.0.0.1:$port "
  for _ in $(seq 1 180); do
    if healthy "$port"; then echo; echo "$p is ready: http://127.0.0.1:$port  (log: $log)"; return 0; fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo; echo "$p exited during start-up; last log lines:" >&2; tail -n 15 "$log" >&2
      rm -f "$(pidfile "$p")"; return 1
    fi
    echo -n "."; sleep 1
  done
  echo; echo "$p is still loading; check $log or $0 status" >&2
}

stop() {
  local p="$1" pid
  pid="$(running_pid "$p" || true)"
  if [ -z "$pid" ]; then echo "$p is not running (no live pid file)"; rm -f "$(pidfile "$p")"; return 0; fi
  kill "$pid"
  for _ in $(seq 1 20); do kill -0 "$pid" 2>/dev/null || break; sleep 0.5; done
  rm -f "$(pidfile "$p")"
  echo "Stopped $p (pid $pid)"
}

status() {
  local p port
  for p in $PROFILES; do
    port="$(field "$p" port)"
    if healthy "$port"; then echo "$p: running on 127.0.0.1:$port ($(field "$p" alias))"
    elif [ -n "$(running_pid "$p" || true)" ]; then echo "$p: loading on 127.0.0.1:$port"
    else echo "$p: stopped (start with: $0 start $p)"; fi
  done
}

case "${1:-}" in
  start) [ $# -eq 2 ] || usage; case " $PROFILES " in *" $2 "*) start "$2" ;; *) usage ;; esac ;;
  stop)  if [ $# -eq 1 ]; then for p in $PROFILES; do stop "$p"; done
         else case " $PROFILES " in *" $2 "*) stop "$2" ;; *) usage ;; esac; fi ;;
  status) status ;;
  *) usage ;;
esac
