#!/bin/sh
set -e

echo "[entrypoint] Starting GristMill daemon..."
gristmill-daemon &
DAEMON_PID=$!

# Wait for socket to appear (up to 15s)
i=0
while [ ! -S "$GRISTMILL_SOCK" ] && [ $i -lt 30 ]; do
  sleep 0.5
  i=$((i + 1))
done

if [ ! -S "$GRISTMILL_SOCK" ]; then
  echo "[entrypoint] ERROR: daemon socket did not appear at $GRISTMILL_SOCK"
  kill "$DAEMON_PID" 2>/dev/null
  exit 1
fi

echo "[entrypoint] Daemon ready. Starting TypeScript shell..."
exec node dist/main.js
