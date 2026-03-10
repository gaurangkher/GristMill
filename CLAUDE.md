# GristMill — Claude Code Guide

> *"Rust grinds. Python trains. TypeScript connects."*

GristMill v2 is a tri-language AI orchestration system. Each language owns a strict domain; do not blur these boundaries.

---

## Architecture Overview

```
TypeScript Shell (gristmill-integrations)
  └─ napi-rs ──→ Rust Core (gristmill-core)  ←── PyO3 ── Python Shell (gristmill-ml)
```

| Domain | Language | Never cross into |
|--------|----------|-----------------|
| Event triage, ML inference, DAG scheduling, memory | **Rust** | Don't add inference logic in TS/Python |
| Model training, fine-tuning, ONNX export | **Python** | Don't run production inference in Python |
| Channel adapters, notifications, web dashboard, plugins | **TypeScript** | Don't add business logic here; delegate to Rust core |

---

## Repo Structure

```
GristMill/
├── gristmill-core/          # Rust workspace
│   └── crates/
│       ├── grist-event/     # GristEvent universal type
│       ├── grist-sieve/     # Triage classifier (<5ms target)
│       ├── grist-grinders/  # Local ML inference pool (ONNX/GGUF/TFLite)
│       ├── grist-millwright/ # DAG orchestrator (Tokio + Rayon)
│       ├── grist-ledger/    # Three-tier memory (hot/warm/cold)
│       ├── grist-hammer/    # LLM escalation gateway
│       ├── grist-bus/       # Internal typed pub/sub
│       ├── grist-config/    # YAML/TOML + env config
│       └── grist-ffi/       # PyO3 + napi-rs bridges
├── gristmill-ml/            # Python package (pip)
│   └── src/gristmill_ml/
│       ├── training/        # SieveTrainer, NER, embedder
│       ├── datasets/        # Feedback log import, augmentation
│       ├── export/          # PyTorch → ONNX, quantization
│       └── experiments/     # MLflow / W&B tracking
├── gristmill-integrations/  # TypeScript package (pnpm)
│   └── src/
│       ├── core/bridge.ts   # napi-rs wrapper — all logic delegates to Rust
│       ├── hopper/          # HTTP, WebSocket, webhook, cron, MQ, FS adapters
│       ├── bell-tower/      # Notification dispatch + watch engine
│       ├── dashboard/       # Fastify API + React SPA
│       └── plugins/         # Dynamic plugin system + SDK
└── gristmill-v2-architecture.md   # Full architecture spec
```

---

## Language-Specific Guidelines

### Rust (`gristmill-core`)

- **Sieve triage MUST complete in <5ms.** Never add blocking work to the `triage()` hot path.
- Use `tokio` for async I/O, `rayon` for CPU-parallel work. Never mix the two runtimes carelessly.
- All tensors use `ndarray`; prefer zero-copy views over clones.
- `GristEvent` (with a `Ulid` id) is the universal message type — every cross-boundary message must serialize to/from it.
- Memory tiers: hot (`sled` LRU) → warm (`SQLite FTS5` + `usearch` vectors) → cold (`zstd` JSONL). Auto-compaction is a background Tokio task.
- FFI surface lives only in `grist-ffi`. Keep `pyo3_bridge.rs` and `napi_bridge.rs` thin — no business logic.
- Use `tracing` for structured logging; `metrics` for counters/histograms. No `println!` in library crates.

### Python (`gristmill-ml`)

- Python **trains** models; Rust **runs** them. Production inference never goes through Python.
- Export all models as ONNX (INT8 quantized) via `export/onnx_export.py`. Validate cross-runtime parity before shipping.
- `SieveTrainer` reads feedback JSONL from `~/.gristmill/feedback/` — keep the feedback schema stable.
- Use `MLflow` or `W&B` for experiment tracking; don't commit raw model weights to git.
- The PyO3 bridge is re-exported from `gristmill_ml/core.py`. Don't bypass it by importing `gristmill_core` directly in application code.

### TypeScript (`gristmill-integrations`)

- All processing logic delegates to Rust via `GristMillBridge` (`core/bridge.ts`). TypeScript handles I/O only.
- Hopper adapters normalize external events into `GristEvent` JSON before passing to `bridge.submit()`.
- Bell Tower subscribes to `grist-bus` topics (`pipeline.completed`, `pipeline.failed`, `sieve.anomaly`, `ledger.threshold`).
- Plugins implement `GristMillPlugin` and register adapters/channels/step-types via `PluginContext`.
- Use `pnpm` (not npm or yarn). Dashboard UI is a separate React SPA built into `dashboard/ui/dist/`.

---

## Key Invariants

- **Local-first**: `prefer_local: true` is the default on every `Step`. Escalate to LLM only when confidence < threshold (default 0.85).
- **Closed learning loop**: Sieve logs every routing decision to feedback JSONL → Python retrains → new ONNX hot-reloaded in Rust. Weekly cadence.
- **LLM budget**: Token limits enforced in `grist-hammer/budget.rs`. Never bypass the budget manager.
- **Semantic cache**: `grist-hammer` caches LLM responses at similarity ≥ 0.92. Don't disable caching in production.
- **No GC pauses on hot paths**: Rust core handles all latency-sensitive work.

---

## Build Commands

```bash
# Rust core
cargo build                          # dev
cargo build --release --target x86_64-unknown-linux-musl  # static binary

# Python shell (editable install)
cd gristmill-ml && pip install -e .

# TypeScript shell
cd gristmill-integrations && pnpm install && pnpm build

# napi-rs prebuilt (Node.js bindings)
cd crates/grist-ffi && napi build --release

# PyO3 wheel
cd crates/grist-ffi && maturin build --release

# All-in-one Docker
docker build -t gristmill:latest .
```

---

## Config

Main config lives at `~/.gristmill/config.yaml`. Key sections: `sieve`, `grinders`, `hammer`, `millwright`, `ledger`, `bell_tower`, `integrations`. Secrets via env vars (`ANTHROPIC_API_KEY`, `SLACK_WEBHOOK_URL`, etc.) — never hardcode.

Default LLM providers: `anthropic` (claude-sonnet-4) with `ollama` (llama3.1:8b) as local fallback.

---

## Testing Notes

- Rust: `cargo test` per crate; integration tests in `gristmill-daemon`.
- Sieve latency regression tests must assert p99 < 5ms.
- Python: validate ONNX export parity in `export/validate.py` before hot-reloading into Rust.
- TypeScript: test adapters with mock `GristMillBridge`; never call real Rust core in unit tests.
