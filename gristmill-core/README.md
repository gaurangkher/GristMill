# gristmill-core — Rust Workspace

The Rust core of GristMill. Contains all latency-sensitive processing: event triage, local ML inference, DAG orchestration, three-tier memory, and LLM escalation.

> **Rule**: All production inference, routing, and memory work runs here. TypeScript and Python handle I/O and training only.

## Crate Map

| Crate | Purpose | Key Type |
|-------|---------|----------|
| [`grist-event`](crates/grist-event/README.md) | Universal message type | `GristEvent` |
| [`grist-sieve`](crates/grist-sieve/README.md) | Triage classifier (<5 ms p99) | `Sieve`, `RouteDecision` |
| [`grist-grinders`](crates/grist-grinders/README.md) | Local ML inference pool | `Grinders`, `InferenceRequest` |
| [`grist-millwright`](crates/grist-millwright/README.md) | DAG pipeline orchestrator | `Millwright`, `Pipeline` |
| [`grist-ledger`](crates/grist-ledger/README.md) | Three-tier memory | `Ledger`, `Memory` |
| [`grist-hammer`](crates/grist-hammer/README.md) | LLM escalation gateway | `Hammer`, `EscalationRequest` |
| [`grist-bus`](crates/grist-bus/README.md) | Internal typed pub/sub | `EventBus` |
| [`grist-config`](crates/grist-config/README.md) | YAML + env config | `GristMillConfig` |
| [`grist-core`](crates/grist-core/README.md) | Aggregate orchestrator | `GristMillCore` |
| [`grist-ffi`](crates/grist-ffi/README.md) | PyO3 + napi-rs bridges | `PyGristMill`, `GristMill` |
| [`gristmill-daemon`](gristmill-daemon/README.md) | Long-running daemon binary | — |

## Dependency Graph

```
grist-event  (no deps on other grist-* crates)
    ↑
grist-bus  ←── grist-event
    ↑
grist-config (no deps on other grist-* crates)
    ↑
grist-sieve  ←── grist-event, grist-grinders
grist-grinders ←── grist-event, grist-sieve (features)
grist-ledger ←── grist-event, grist-bus
grist-hammer ←── grist-event, grist-bus
grist-millwright ←── grist-event, grist-bus
    ↑
grist-core ←── all of the above + grist-config
    ↑
grist-ffi ←── grist-core, grist-event, grist-millwright, grist-bus
gristmill-daemon ←── grist-core
```

## Workspace Key Dependencies

| Dependency | Version | Use |
|-----------|---------|-----|
| `tokio` | 1 (full) | Async I/O runtime |
| `rayon` | 1 | CPU-parallel inference/compaction |
| `ort` | 2.0.0-rc.11 | ONNX Runtime |
| `sled` | 0.34 | Hot-tier LRU storage |
| `rusqlite` | 0.32 | Warm-tier SQLite + FTS5 |
| `usearch` | 2.24 | HNSW vector index |
| `zstd` | 0.13 | Cold-tier compression |
| `ndarray` | 0.16 | Tensor operations |
| `tracing` | 0.1 | Structured logging |
| `metrics` | 0.23 | Counters/histograms |
| `ulid` | 1 | Sortable unique IDs |

## Build

```bash
# Dev build (all crates)
cargo build

# Release
cargo build --release

# Single crate
cargo build -p grist-sieve

# Static Linux binary
cargo build --release --target x86_64-unknown-linux-musl

# Run daemon
cargo run -p gristmill-daemon

# All tests
cargo test

# Single crate tests
cargo test -p grist-core

# Sieve latency regression (p99 < 5 ms)
cargo test -p grist-sieve -- --include-ignored latency
```

## Features

Several crates have optional Cargo features:

| Crate | Feature | Enables |
|-------|---------|---------|
| `grist-sieve` | `onnx` | ONNX Runtime inference (default: heuristic only) |
| `grist-grinders` | `onnx` | ONNX Runtime support |
| `grist-grinders` | `gguf` | llama.cpp / GGUF support |
| `grist-ffi` | `python` | PyO3 Python bindings |
| `grist-ffi` | `node` | napi-rs Node.js bindings |

## Observability

- **Logging**: `tracing` with `RUST_LOG` env filter, e.g. `RUST_LOG=grist_sieve=debug,grist_hammer=info`
- **Metrics**: `metrics` crate emitting counters/histograms; exportable via `metrics-exporter-prometheus`
- **No `println!`** in library crates — use `tracing::info!` / `tracing::warn!`
