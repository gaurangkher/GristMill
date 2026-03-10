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

## Quickstart

### 1. Smoke-test the Rust core (no extra dependencies)

This is the fastest way to verify the system works. The daemon triages sample events through the Sieve and prints routing decisions.

```bash
cd gristmill-core
cargo run -p gristmill-daemon
```

Expected output:

```
INFO gristmill_daemon: GristMill daemon starting (Phase 1)
INFO gristmill_daemon: Sieve initialised threshold=0.85
INFO gristmill_daemon: triaged sample="cli command"      route=Rules      confidence=0.97
INFO gristmill_daemon: triaged sample="calendar request" route=Hybrid     confidence=0.81
INFO gristmill_daemon: triaged sample="complex question" route=LlmNeeded  confidence=0.73
INFO gristmill_daemon: triaged sample="code review"      route=LocalML    confidence=0.91
```

### 2. Run the TypeScript shell in mock mode

The `GRISTMILL_MOCK_BRIDGE=1` flag replaces the native Rust `.node` binary with an in-memory mock so the TypeScript shell starts without building the FFI bridge first.

```bash
cd gristmill-integrations

# Install dependencies
pnpm install

# Type-check
pnpm lint

# Start in mock mode
GRISTMILL_MOCK_BRIDGE=1 pnpm dev
```

This starts the **Dashboard + API** on `http://localhost:3000`:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/events` | POST | Submit an event, get a routing decision |
| `/api/triage` | POST | Triage text directly |
| `/api/memory/remember` | POST | Store a memory |
| `/api/memory/recall` | POST | Search memories |
| `/api/metrics/health` | GET | Liveness check |
| `/api/metrics/budget` | GET | LLM token usage |
| `/api/pipelines` | GET/POST | List/register pipelines |
| `/api/watches` | GET/POST | Notification watch management |
| `/api/plugins` | GET | Loaded plugins |

Test the HTTP endpoint:

```bash
# Submit an event
curl -s -X POST http://localhost:3000/events \
  -H "Content-Type: application/json" \
  -d '{"channel":"http","payload":{"text":"Schedule a meeting with Alice tomorrow"}}' \
  | jq .

# Store and recall memories
curl -s -X POST http://localhost:3000/api/memory/remember \
  -H "Content-Type: application/json" \
  -d '{"content":"prod-db-01 disk alert resolved by pruning WAL logs","tags":["infra","postgres"]}'

curl -s -X POST http://localhost:3000/api/memory/recall \
  -H "Content-Type: application/json" \
  -d '{"query":"disk postgres","limit":5}'
```

### 3. Install the Python ML package

```bash
cd gristmill-ml

# Editable install (pulls torch, transformers, sentence-transformers, onnxruntime, mlflow, …)
# Note: first install downloads ~2–3 GB of ML dependencies
pip install -e ".[dev]"
```

Train the Sieve classifier from feedback logs (falls back to synthetic data on first run):

```bash
gristmill-train-sieve
```

Export a trained model to ONNX:

```bash
gristmill-export
```

Validate ONNX export parity against PyTorch:

```bash
gristmill-validate
```

---

## Configuration

GristMill reads its configuration from `~/.gristmill/config.yaml`. Create the directory and a minimal config to get started:

```bash
mkdir -p ~/.gristmill/feedback ~/.gristmill/models ~/.gristmill/memory
```

```yaml
# ~/.gristmill/config.yaml

core:
  workspace: ~/.gristmill
  log_level: info

sieve:
  confidence_threshold: 0.85
  feedback_dir: ~/.gristmill/feedback/

hammer:
  providers:
    anthropic:
      api_key: ${ANTHROPIC_API_KEY}
      default_model: claude-sonnet-4-6
    ollama:
      base_url: http://localhost:11434
      model: llama3.1:8b
  budget:
    daily_tokens: 500000

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

integrations:
  dashboard:
    port: 4000
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
| `GRISTMILL_CONFIG` | All | Override config file path (default: `~/.gristmill/config.yaml`) |
| `GRISTMILL_MOCK_BRIDGE` | TypeScript | Set to `1` to use in-memory mock instead of native `.node` bridge |
| `RUST_LOG` | Rust core | Log filter, e.g. `RUST_LOG=grist_sieve=debug` |
