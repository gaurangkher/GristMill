#!/bin/sh
set -e

CONFIG="${GRISTMILL_CONFIG:-/data/gristmill/config.yaml}"
SOCK="${GRISTMILL_SOCK:-/data/gristmill/gristmill.sock}"

# ── Fix named-volume permissions (runs as root before dropping to gristmill) ──
# The /gristmill/run directory lives on a named Docker volume which Docker
# creates as root:root.  The Rust daemon writes inference.lock and the trainer
# writes trainer.sock there — both need gristmill ownership.
mkdir -p /gristmill/run
chown gristmill:gristmill /gristmill/run
chmod 750 /gristmill/run
echo "[entrypoint] /gristmill/run ready (owner: gristmill)"

# ── Validate config ───────────────────────────────────────────────────────────
if [ ! -f "$CONFIG" ]; then
  echo "[entrypoint] ERROR: config file not found at $CONFIG"
  echo "[entrypoint]   Mount it with:"
  echo "[entrypoint]     - type: bind"
  echo "[entrypoint]       source: ./gristmill-data/config.yaml"
  echo "[entrypoint]       target: $CONFIG"
  echo "[entrypoint]       read_only: true"
  exit 1
fi

if grep -q "REPLACE_ME" "$CONFIG" 2>/dev/null; then
  echo "[entrypoint] WARNING: config.yaml still contains REPLACE_ME placeholders."
  echo "[entrypoint]   Edit gristmill-data/config.yaml and fill in your API keys."
fi

# ── Ensure runtime directories exist ─────────────────────────────────────────
mkdir -p \
  /data/gristmill/models \
  /data/gristmill/memory \
  /data/gristmill/memory/cold \
  /data/gristmill/feedback \
  /data/gristmill/plugins \
  /data/gristmill/checkpoints \
  /data/gristmill/logs \
  /data/gristmill/db
chown -R gristmill:gristmill /data/gristmill

echo "[entrypoint] Config:  $CONFIG"
echo "[entrypoint] Socket:  $SOCK"
echo "[entrypoint] Starting GristMill daemon (as gristmill user)..."

# ── Start daemon as gristmill user (gosu for correct signal handling) ─────────
gosu gristmill gristmill-daemon &
DAEMON_PID=$!

# ── Wait for socket (up to 15s) ───────────────────────────────────────────────
i=0
while [ ! -S "$SOCK" ] && [ $i -lt 30 ]; do
  sleep 0.5
  i=$((i + 1))
done

if [ ! -S "$SOCK" ]; then
  echo "[entrypoint] ERROR: daemon socket did not appear at $SOCK"
  kill "$DAEMON_PID" 2>/dev/null
  exit 1
fi

echo "[entrypoint] Daemon ready. Starting TypeScript shell (as gristmill user)..."
exec gosu gristmill node dist/main.js
