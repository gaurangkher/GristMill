# gristmill-daemon

The long-running daemon binary for GristMill. Initialises `GristMillCore` and exposes a Unix socket IPC server for the TypeScript and Python shells to call the Rust core without an in-process FFI build.

## Purpose

The daemon is the primary deployment artifact of the Rust core. It enables the TypeScript and Python shells to connect to a separately-running Rust process over a Unix socket, rather than requiring the FFI bridges (`grist-ffi`) to be compiled and linked.

## Quick Start

```bash
cd gristmill-core
cargo run -p gristmill-daemon
```

Or in release mode:

```bash
cargo build --release -p gristmill-daemon
./target/release/gristmill-daemon
```

Expected output:

```
INFO gristmill_daemon: GristMill daemon starting (Phase 2)
INFO grist_core: Ledger using GrindersEmbedder (MiniLM-L6-v2)
INFO gristmill_daemon: GristMillCore ready
INFO gristmill_daemon: IPC socket  socket=/Users/you/.gristmill/gristmill.sock
INFO gristmill_daemon: GristMill daemon ready — press Ctrl+C to stop
```

## Runtime Configuration

| Source | Priority | Example |
|--------|----------|---------|
| `GRISTMILL_CONFIG` env var | 1 (highest) | `/etc/gristmill/config.yaml` |
| Built-in defaults | 2 (fallback) | Uses `GristMillConfig::default()` |

| Source | Socket Path |
|--------|-------------|
| `GRISTMILL_SOCK` env var | Custom path |
| Default | `~/.gristmill/gristmill.sock` |

## IPC Protocol

The daemon accepts MessagePack-framed JSON-RPC calls over the Unix socket. The TypeScript `IpcBridge` and Python `IpcBridge` handle framing automatically.

**Request format:**
```json
{ "method": "triage", "params": { "event_json": "..." }, "id": 1 }
```

**Response format:**
```json
{ "result": { "route": "LOCAL_ML", "confidence": 0.91 }, "id": 1 }
```

**Supported methods:**

| Method | Params | Returns |
|--------|--------|---------|
| `triage` | `event_json` | `RouteDecision` JSON |
| `remember` | `content`, `tags[]` | `memory_id` string |
| `recall` | `query`, `limit` | `RankedMemory[]` JSON |
| `get_memory` | `id` | `Memory` JSON or `null` |
| `escalate` | `prompt`, `max_tokens` | `EscalationResponse` JSON |
| `register_pipeline` | `pipeline_json` | `null` |
| `run_pipeline` | `pipeline_id`, `event_json` | `PipelineResult` JSON |
| `pipeline_ids` | — | `string[]` |
| `subscribe` | `topic` | stream of `BusEvent` JSON |

## Shutdown

The daemon handles `SIGINT` (Ctrl+C) and `SIGTERM` for graceful shutdown:

- Stops accepting new IPC connections
- Waits for in-flight requests to complete
- Flushes background tasks (Ledger eviction drainer, compactor)
- Removes the socket file

## Static Binary

For production Linux deployments, build a fully static binary with no shared library dependencies:

```bash
rustup target add x86_64-unknown-linux-musl
cargo build --release --target x86_64-unknown-linux-musl -p gristmill-daemon
# → target/x86_64-unknown-linux-musl/release/gristmill-daemon (~15 MB + model files)
```

## Systemd Unit (example)

```ini
[Unit]
Description=GristMill AI Orchestration Daemon
After=network.target

[Service]
ExecStart=/usr/local/bin/gristmill-daemon
Environment=GRISTMILL_CONFIG=/etc/gristmill/config.yaml
Environment=ANTHROPIC_API_KEY=sk-ant-...
Restart=on-failure
RestartSec=5s
User=gristmill

[Install]
WantedBy=multi-user.target
```

## Dependencies

```toml
grist-core  = { path = "../crates/grist-core" }
tokio       # async main + signal handling
tracing     # startup/shutdown logging
tracing-subscriber  # log output formatting
anyhow      # error handling in main
```
