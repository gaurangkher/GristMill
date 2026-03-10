# grist-core

Aggregate crate that assembles all GristMill subsystems into a single `GristMillCore` instance. This is the primary dependency for the daemon and FFI bridges.

## Purpose

`grist-core` wires together all the other Rust crates:

```
GristMillCore
  ├─ Sieve        (grist-sieve)      — triage classifier
  ├─ Grinders     (grist-grinders)   — local ML inference pool
  ├─ Millwright   (grist-millwright) — DAG pipeline orchestrator
  ├─ Ledger       (grist-ledger)     — three-tier memory
  ├─ Hammer       (grist-hammer)     — LLM escalation gateway
  ├─ Bus          (grist-bus)        — internal pub/sub
  └─ Config       (grist-config)     — YAML + env config
```

It also contains `GrindersEmbedder` (`src/embedder.rs`) — the bridge that connects the MiniLM-L6-v2 ONNX session from `grist-grinders` to the `Embedder` trait required by `grist-ledger`.

## Key Type

```rust
pub struct GristMillCore {
    pub sieve: Arc<Sieve>,
    pub grinders: Arc<Grinders>,
    pub millwright: Arc<Millwright>,
    pub ledger: Arc<Ledger>,
    pub hammer: Arc<Hammer>,
    pub bus: Arc<EventBus>,
    config: GristMillConfig,
}
```

## Public API

```rust
// Create the full system (loads config, starts background tasks)
let core = GristMillCore::new(config_path: Option<PathBuf>).await?;

// Triage an event
let decision: RouteDecision = core.sieve.triage(&event).await?;

// Store a memory
let id: MemoryId = core.ledger.remember("content", vec!["tag"]).await?;

// Recall memories
let results: Vec<RankedMemory> = core.ledger.recall("query", 10).await?;

// Escalate to LLM
let resp: EscalationResponse = core.hammer.escalate(req).await?;

// Run a pipeline
core.millwright.register_pipeline(pipeline);
let result: PipelineResult = core.millwright.run("my-pipeline", &event).await?;

// Subscribe to bus events
let mut rx = core.bus.subscribe("pipeline.completed");
```

## Startup Sequence

`GristMillCore::new()` performs these steps in order:

1. Load `GristMillConfig` from path (or defaults)
2. Call `.apply_env()` to overlay `ANTHROPIC_API_KEY`, etc.
3. Build `GrindersConfig` → start `Grinders` (loads warm models)
4. Build `SieveConfig` → start `Sieve`
5. Build ledger embedder: `GrindersEmbedder` if model available, else `ZeroEmbedder`
6. Build `LedgerConfig` → start `Ledger` (starts eviction drainer + compactor)
7. Build `HammerConfig` → start `Hammer`
8. Build `MillwrightConfig` → start `Millwright`
9. Return assembled `GristMillCore`

## GrindersEmbedder

`src/embedder.rs` provides the bridge between `grist-grinders` (MiniLM ONNX) and `grist-ledger` (`Embedder` trait):

```rust
pub struct GrindersEmbedder {
    session: grist_sieve::features::EmbedderSession,
}

impl Embedder for GrindersEmbedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>, LedgerError>;
    fn dim(&self) -> usize;  // Returns EMBEDDING_DIM = 384
}

pub fn build_ledger_embedder(config: &GrindersConfig) -> Arc<dyn Embedder>;
```

Falls back to `ZeroEmbedder` when the model file is absent — the daemon always starts, but semantic recall degrades to keyword-only until the model is downloaded.

## Config Builders

`grist-core` contains 5 private builder functions that translate `GristMillConfig` fields to each subsystem's config type:

| Function | Translates to |
|----------|--------------|
| `build_sieve_config()` | `SieveConfig` |
| `build_hammer_config()` | `HammerConfig` |
| `build_millwright_config()` | `MillwrightConfig` |
| `build_ledger_config()` | `LedgerConfig` |
| `build_grinders_config()` | `GrindersConfig` |

## Testing

Tests use an isolated `core_for_test()` helper that creates a temporary directory per test (via `tempfile::tempdir()`) to avoid `sled` lock conflicts when tests run in parallel.

```bash
cargo test -p grist-core
```

## Dependencies

```toml
grist-event      = { path = "../grist-event" }
grist-sieve      = { path = "../grist-sieve" }
grist-ledger     = { path = "../grist-ledger" }
grist-hammer     = { path = "../grist-hammer" }
grist-millwright = { path = "../grist-millwright" }
grist-bus        = { path = "../grist-bus" }
grist-config     = { path = "../grist-config" }
grist-grinders   = { path = "../grist-grinders" }
tokio            # async startup
tracing          # startup logging
```
