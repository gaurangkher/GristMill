# GristMill v2

> *"Rust grinds. Python trains. TypeScript connects."*

GristMill is a **local-first AI orchestration system** that routes tasks intelligently — running everything it can locally (ONNX inference, rules, memory) and escalating to an LLM only when genuinely needed. It is built as a tri-language system with strict domain ownership.

```
TypeScript Shell (gristmill-integrations)
  └─ napi-rs ──→ Rust Core (gristmill-core)  ←── PyO3 ── Python Shell (gristmill-ml)
```

| Layer | Language | Owns |
|-------|----------|------|
| Event triage, ML inference, DAG scheduling, memory | **Rust** | All latency-sensitive and production inference work |
| Model training, fine-tuning, ONNX export | **Python** | Model lifecycle only — never production inference |
| Channel adapters, notifications, dashboard, plugins | **TypeScript** | I/O and integrations only — no business logic |

---

## Repository Layout

```
GristMill/
├── gristmill-core/           # Rust workspace
│   ├── crates/
│   │   ├── grist-event/      # GristEvent universal message type (ULID-keyed)
│   │   ├── grist-sieve/      # Triage classifier — <5ms target on hot path
│   │   ├── grist-grinders/   # Local ML inference pool (ONNX / GGUF / TFLite)
│   │   ├── grist-millwright/ # DAG orchestrator (Tokio + Rayon)
│   │   ├── grist-ledger/     # Three-tier memory (hot sled → warm SQLite+vectors → cold zstd JSONL)
│   │   ├── grist-hammer/     # LLM escalation gateway with token budget + semantic cache
│   │   ├── grist-bus/        # Internal typed pub/sub
│   │   ├── grist-config/     # YAML/TOML + env-var config
│   │   └── grist-ffi/        # PyO3 + napi-rs FFI bridges
│   └── gristmill-daemon/     # Long-running daemon binary (entry point)
│
├── gristmill-ml/             # Python package — model training & export
│   └── src/gristmill_ml/
│       ├── training/         # SieveTrainer, NerTrainer, EmbedderTrainer
│       ├── datasets/         # Feedback log import + augmentation
│       ├── export/           # PyTorch → ONNX + INT8 quantisation
│       └── experiments/      # MLflow experiment tracking
│
├── gristmill-integrations/   # TypeScript package — adapters, notifications, dashboard
│   └── src/
│       ├── core/bridge.ts    # napi-rs wrapper — all logic delegates to Rust
│       ├── hopper/           # HTTP, WebSocket inbound adapters
│       ├── bell-tower/       # Notification dispatch (Slack, email, watches)
│       ├── dashboard/        # Fastify REST API + React SPA host
│       └── plugins/          # Dynamic plugin system + SDK
│
└── gristmill-v2-architecture.md   # Full architecture specification
```

---

## Prerequisites

| Tool | Minimum version | Install |
|------|----------------|---------|
| Rust + Cargo | 1.80 | `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \| sh` |
| Node.js | 18 | [nodejs.org](https://nodejs.org) |
| pnpm | 9 | `npm install -g pnpm` |
| Python | 3.10 | [python.org](https://python.org) |
| pip | 23+ | bundled with Python |

Optional (only needed for full native bridges):

| Tool | Purpose |
|------|---------|
| `@napi-rs/cli` | Build the Node.js `.node` binary from `grist-ffi` |
| `maturin` | Build the Python wheel from `grist-ffi` |
| Docker | All-in-one container build |

---

## Setup

Choose your path:

- **[Local (no Docker)](#local-setup)** — best for development; each component runs in a separate terminal
- **[Docker](#docker-setup)** — best for production or demos; single `docker compose up` command

---

## Local Setup

### 1. Config

```bash
mkdir -p ~/.gristmill/{models,memory/cold,feedback,checkpoints,plugins,logs,run}
cp gristmill-data/config.yaml ~/.gristmill/config.yaml
```

Edit `~/.gristmill/config.yaml` — minimum required fields:

```yaml
hammer:
  providers:
    anthropic:
      api_key: "sk-ant-..."          # your Anthropic API key
      default_model: claude-sonnet-4-20250514
    ollama:                           # optional — remove if not using Ollama
      base_url: http://localhost:11434
      model: llama3.1:8b
  budget:
    daily_tokens: 500000
    monthly_tokens: 10000000

integrations:
  dashboard:
    port: 4000
  slack:                              # optional — remove if not using Slack
    app_token: "xapp-..."
    bot_token: "xoxb-..."
    signing_secret: "..."
```

Set environment variables (add to `.zshrc` / `.bashrc`):

```bash
export GRISTMILL_CONFIG=~/.gristmill/config.yaml
export GRISTMILL_SOCK=~/.gristmill/gristmill.sock
```

> **Security:** `gristmill-data/config.yaml` is tracked as a template. Never commit real API keys.
> Use environment variable interpolation (`${ANTHROPIC_API_KEY}`) and set secrets in your shell.

### 2. Build and start the Rust daemon

**Terminal 1:**

```bash
cd gristmill-core
cargo build --release
./target/release/gristmill-daemon
```

The daemon writes a Unix socket at `$GRISTMILL_SOCK` and stays in the foreground. Wait for the log line `daemon ready` before starting the next step.

### 3. Start the TypeScript shell

**Terminal 2:**

```bash
cd gristmill-integrations
pnpm install
pnpm build
node dist/main.js
```

For development with hot-reload:

```bash
pnpm dev
```

Dashboard is available at **http://localhost:4000**.

Verify:

```bash
curl http://localhost:4000/api/metrics/health
# {"status":"ok","uptime":12.3,"timestamp":"..."}
```

### 4. Start the Python trainer (optional)

The trainer auto-retrains the Sieve classifier when 500+ feedback records accumulate or 7 days pass. The system runs without it — you just won't get automatic model improvement.

**Terminal 3:**

```bash
# Install (pulls torch, transformers, sentence-transformers, onnx, fastapi, ~2-3 GB)
pip install -e gristmill-ml

# Bootstrap base models — one-time download (~400 MB), idempotent
python gristmill-ml/scripts/bootstrap_models.py

# Start trainer daemon
gristmill-trainer
```

Trainer control API: **http://localhost:7432**

```bash
curl http://localhost:7432/health
# {"ok":true,"uptime_seconds":5.1,...}

curl http://localhost:7432/status
# {"state":"IDLE","current_version":1,"buffer_pending_count":0,...}
```

The dashboard **Overview** and **Trainer** pages show live state once it's running.

**Bootstrap flags:**

| Flag | Effect |
|------|--------|
| `--no-quantize` | Skip INT8 quantization (faster, slightly larger models) |
| `--embedder-only` | Only export the sentence embedder |
| `--classifier-epochs N` | Training epochs (default: 3) |
| `--output-dir PATH` | Override `~/.gristmill/models/` |

### 5. Smoke-test event routing

```bash
# Submit an event
curl -s -X POST http://localhost:4000/events \
  -H "Content-Type: application/json" \
  -d '{"channel":"http","payload":{"text":"Schedule a meeting with Alice tomorrow"}}' \
  | jq .

# Store a memory
curl -s -X POST http://localhost:4000/api/memory/remember \
  -H "Content-Type: application/json" \
  -d '{"content":"prod-db-01 disk alert resolved by pruning WAL logs","tags":["infra","postgres"]}'

# Recall memories
curl -s -X POST http://localhost:4000/api/memory/recall \
  -H "Content-Type: application/json" \
  -d '{"query":"disk postgres","limit":5}'
```

---

## Docker Setup

### 1. Config

Edit the template before first start:

```bash
nano gristmill-data/config.yaml
```

Replace all `REPLACE_ME` placeholders with real values. The entrypoint will warn you at startup if any remain.

### 2. Start

```bash
# Core only (daemon + dashboard)
docker compose up -d

# + Python trainer (auto-retraining loop)
ANTHROPIC_API_KEY=sk-ant-... docker compose --profile trainer up -d

# + Ollama local LLM
docker compose --profile ollama up -d

# Everything
ANTHROPIC_API_KEY=sk-ant-... docker compose --profile full up -d
```

Dashboard: **http://localhost:3000**

### 3. Profiles

| Profile | Adds | Port |
|---------|------|------|
| *(default)* | `gristmill` daemon + TS shell | 3000 |
| `trainer` | Python trainer daemon | 7432 |
| `ollama` | Ollama local LLM | 11434 |
| `mlflow` | MLflow experiment tracking UI | 5050 |
| `full` | All of the above | — |

### 4. First trainer start

On first boot the trainer container downloads ~400 MB of HuggingFace model weights into `gristmill-data/models/`. The healthcheck has a 120-second start period to allow for this. Subsequent starts skip the download.

### 5. Useful commands

```bash
# Follow logs
docker compose logs -f gristmill
docker compose logs -f trainer

# Rebuild after code changes
docker compose build gristmill && docker compose up -d gristmill

# Stop everything (data preserved)
docker compose down

# Stop and wipe runtime data (models, memory, feedback)
docker compose down -v
```

---

## Configuration Reference

`config.yaml` key sections:

```yaml
sieve:
  confidence_threshold: 0.85   # escalate to LLM when confidence < this
  cache_size: 10000

hammer:
  providers:
    anthropic:
      api_key: ${ANTHROPIC_API_KEY}
      default_model: claude-sonnet-4-20250514
    ollama:
      base_url: http://localhost:11434
      model: llama3.1:8b
  budget:
    daily_tokens: 500000
    monthly_tokens: 10000000
  cache:
    similarity_threshold: 0.92  # reuse cached response when similarity ≥ this

ledger:
  hot:
    max_size_mb: 512
  warm:
    db_path: ~/.gristmill/memory/warm.db
    vector_index_path: ~/.gristmill/memory/vectors.usearch
  cold:
    archive_dir: ~/.gristmill/memory/cold/
    compression: zstd

bell_tower:
  channels:
    slack:
      webhook_url: ${SLACK_WEBHOOK_URL}
    email:
      smtp_host: smtp.gmail.com
      smtp_port: 587
      username: ${EMAIL_USER}
      password: ${EMAIL_PASS}
  quiet_hours:
    start: "22:00"
    end: "07:00"
    override_for: [critical]

integrations:
  dashboard:
    port: 4000           # 3000 in Docker
  hoppers:
    http:
      port: 3001
  plugins_dir: ~/.gristmill/plugins/
```

Secrets are always passed via environment variables — never hardcoded.

---

## Building Native FFI Bridges

The native bridges let the TypeScript and Python shells call directly into the Rust core in-process (fastest path). They are optional — both shells run in mock/stub mode without them.

### Node.js bridge (`grist-ffi` → `.node` binary)

```bash
# Install the napi-rs CLI once
npm install -g @napi-rs/cli

cd gristmill-core/crates/grist-ffi
napi build --release --features node
# Produces: gristmill_core.*.node
```

Copy the produced `.node` file to `gristmill-integrations/` and the `NativeBridge` in `bridge.ts` will find it automatically.

### Python bridge (`grist-ffi` → `.whl` wheel)

```bash
# Install maturin once
pip install maturin

cd gristmill-core/crates/grist-ffi
maturin build --release --features python
# Produces: target/wheels/gristmill_core-*.whl

pip install target/wheels/gristmill_core-*.whl
```

Once installed, `import gristmill_core` works from Python and `HAS_NATIVE = True` in `gristmill_ml/core.py`.

---

## Ports

| Port | Service | Notes |
|------|---------|-------|
| 3000 | Dashboard (Docker) | Fastify API + React SPA |
| 4000 | Dashboard (local) | Same — configured in `integrations.dashboard.port` |
| 3001 | HTTP hopper | Inbound event endpoint |
| 7432 | Trainer API | localhost / trainer container only |
| 11434 | Ollama | When `--profile ollama` |
| 5050 | MLflow | When `--profile mlflow` (avoids macOS AirPlay on 5000) |

---

## Running Tests

```bash
# Rust — unit + integration tests (all crates)
cd gristmill-core
cargo test

# Rust — sieve latency regression (p99 must be <5ms)
cargo test -p grist-sieve -- --include-ignored latency

# TypeScript — unit tests with vitest
cd gristmill-integrations
pnpm test

# TypeScript — watch mode
pnpm test:watch

# Python — validate ONNX export parity
cd gristmill-ml
python -m gristmill_ml.export.validate
```

---

## Release Builds

```bash
# Static Rust binary (Linux musl — ships without .so dependencies)
cd gristmill-core
cargo build --release --target x86_64-unknown-linux-musl
# → target/x86_64-unknown-linux-musl/release/gristmill-daemon

# TypeScript production build
cd gristmill-integrations
pnpm build
# → dist/

# Docker — all-in-one image
docker build -t gristmill:latest .
```

---

## Key Invariants

- **Local-first** — `prefer_local: true` is the default on every pipeline step. The Sieve only escalates to an LLM when confidence falls below the threshold (default 0.85).
- **Closed learning loop** — The Sieve logs every routing decision to `~/.gristmill/feedback/`. Python retrains the classifier weekly; the new ONNX model is hot-reloaded by the Rust core without a restart.
- **LLM budget enforced** — Token limits are enforced in `grist-hammer/budget.rs`. The budget manager cannot be bypassed.
- **Semantic cache** — `grist-hammer` caches LLM responses at similarity ≥ 0.92 to avoid redundant API calls.
- **No GC pauses on hot paths** — All latency-sensitive work (triage, inference, memory retrieval) runs in Rust. Python and TypeScript handle I/O only.

---

## Component Documentation

| Component | Language | README |
|-----------|----------|--------|
| `gristmill-core/` | Rust | [`gristmill-core/README.md`](./gristmill-core/README.md) |
| `gristmill-core/crates/grist-event/` | Rust | [`grist-event/README.md`](./gristmill-core/crates/grist-event/README.md) |
| `gristmill-core/crates/grist-sieve/` | Rust | [`grist-sieve/README.md`](./gristmill-core/crates/grist-sieve/README.md) |
| `gristmill-core/crates/grist-grinders/` | Rust | [`grist-grinders/README.md`](./gristmill-core/crates/grist-grinders/README.md) |
| `gristmill-core/crates/grist-millwright/` | Rust | [`grist-millwright/README.md`](./gristmill-core/crates/grist-millwright/README.md) |
| `gristmill-core/crates/grist-ledger/` | Rust | [`grist-ledger/README.md`](./gristmill-core/crates/grist-ledger/README.md) |
| `gristmill-core/crates/grist-hammer/` | Rust | [`grist-hammer/README.md`](./gristmill-core/crates/grist-hammer/README.md) |
| `gristmill-core/crates/grist-bus/` | Rust | [`grist-bus/README.md`](./gristmill-core/crates/grist-bus/README.md) |
| `gristmill-core/crates/grist-config/` | Rust | [`grist-config/README.md`](./gristmill-core/crates/grist-config/README.md) |
| `gristmill-core/crates/grist-ffi/` | Rust | [`grist-ffi/README.md`](./gristmill-core/crates/grist-ffi/README.md) |
| `gristmill-core/crates/grist-core/` | Rust | [`grist-core/README.md`](./gristmill-core/crates/grist-core/README.md) |
| `gristmill-core/gristmill-daemon/` | Rust | [`gristmill-daemon/README.md`](./gristmill-core/gristmill-daemon/README.md) |
| `gristmill-ml/` | Python | [`gristmill-ml/README.md`](./gristmill-ml/README.md) |
| `gristmill-integrations/` | TypeScript | [`gristmill-integrations/README.md`](./gristmill-integrations/README.md) |

## Architecture Deep-Dive

See [`gristmill-v2-architecture.md`](./gristmill-v2-architecture.md) for the full specification including all Rust interfaces, the retraining loop design, IPC modes, and CLI reference.

---

## Environment Variables

| Variable | Used by | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | `grist-hammer` | Anthropic API key for Claude escalation |
| `SLACK_WEBHOOK_URL` | Bell Tower | Slack Incoming Webhook URL |
| `EMAIL_USER` / `EMAIL_PASS` | Bell Tower | SMTP credentials for email notifications |
| `GRISTMILL_CONFIG` | All | Config file path (default: `~/.gristmill/config.yaml`) |
| `GRISTMILL_SOCK` | Daemon + TS shell | Unix socket path (default: `~/.gristmill/gristmill.sock`) |
| `TRAINER_URL` | TypeScript shell | Trainer API base URL (default: `http://127.0.0.1:7432`) |
| `GRISTMILL_MOCK_BRIDGE` | TypeScript | Set to `1` to skip native `.node` bridge (dev/test only) |
| `RUST_LOG` | Rust core | Log filter, e.g. `RUST_LOG=grist_sieve=debug` |
