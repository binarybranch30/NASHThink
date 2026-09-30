#!/usr/bin/env bash
# Start the local Sarthink API + UI in the background, bound to 127.0.0.1 only.
#   scripts/start_sarthink.sh            # http://127.0.0.1:8000/
# Does nothing if a healthy Sarthink API already answers on the port. Logs to processed_data/api.log.
# Test overrides (never needed for normal use): SARTHINK_PORT, SARTHINK_LOG, SARTHINK_PIDFILE.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST="${SARTHINK_HOST:-127.0.0.1}"
PORT="${SARTHINK_PORT:-8000}"
LOG="${SARTHINK_LOG:-$ROOT/processed_data/api.log}"
PIDFILE="${SARTHINK_PIDFILE:-$ROOT/processed_data/api.pid}"
PY="$ROOT/.venv/bin/python"
URL="http://$HOST:$PORT"

healthy() { curl -sf -m 3 "$URL/api/health" 2>/dev/null | grep -q '"status":"ok"'; }

if healthy; then
  echo "Sarthink is already running: $URL/"
  exit 0
fi
if ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq "(^|:)$PORT\$"; then
  echo "Port $PORT is in use by something that is not a healthy Sarthink API; not starting a second server." >&2
  echo "Check it with: ss -ltnp | grep :$PORT" >&2
  exit 1
fi
[ -x "$PY" ] || { echo "Missing $PY (see README: Local CPU setup)." >&2; exit 1; }

mkdir -p "$(dirname "$LOG")" "$(dirname "$PIDFILE")"
cd "$ROOT"
echo "--- $(date -u +%FT%TZ) start_sarthink.sh ($HOST:$PORT)" >> "$LOG"
HF_HUB_OFFLINE=1 nohup "$PY" scripts/api/server.py --host "$HOST" --port "$PORT" >> "$LOG" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$PIDFILE"

for _ in $(seq 1 60); do
  if healthy; then
    echo "Sarthink started (pid $PID): $URL/"
    echo "Log: $LOG"
    exit 0
  fi
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "Sarthink exited during start-up; last log lines:" >&2
    tail -n 15 "$LOG" >&2
    rm -f "$PIDFILE"
    exit 1
  fi
  sleep 0.5
done
echo "Sarthink (pid $PID) is still starting; check $LOG or scripts/status_sarthink.sh" >&2
exit 1
