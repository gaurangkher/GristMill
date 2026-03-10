# grist-bus

Internal typed pub/sub event bus for GristMill. Provides fan-out broadcast messaging between subsystems within the same process.

## Purpose

`grist-bus` decouples subsystems that need to react to each other's events without direct coupling. Examples:

- Millwright publishes `pipeline.completed` → Bell Tower triggers notifications
- Hammer publishes `hammer.budget` → Dashboard metrics route caches token usage
- Sieve publishes `sieve.anomaly` → Bell Tower alerts on routing anomalies

## Key Types

```rust
pub struct EventBus { /* opaque */ }

pub type BusEvent = serde_json::Value;
pub type Subscription = tokio::sync::broadcast::Receiver<BusEvent>;

// Well-known topic constants
pub const TOPIC_PIPELINE_COMPLETED: &str = "pipeline.completed";
pub const TOPIC_PIPELINE_FAILED:    &str = "pipeline.failed";
pub const TOPIC_SIEVE_ANOMALY:      &str = "sieve.anomaly";
pub const TOPIC_LEDGER_THRESHOLD:   &str = "ledger.threshold";
pub const TOPIC_HAMMER_BUDGET:      &str = "hammer.budget";
```

## Public API

```rust
// Create bus (capacity = per-topic buffer size)
let bus = EventBus::new(capacity);

// Subscribe to a topic (lazy — topic created on first subscribe)
let mut rx: Subscription = bus.subscribe("pipeline.completed");

// Publish an event
bus.publish("pipeline.completed", serde_json::json!({
    "run_id": run_id,
    "pipeline_id": "content-digest",
    "elapsed_ms": 42,
}));

// Receive (async)
while let Ok(event) = rx.recv().await {
    println!("Got: {event}");
}
```

## Implementation Notes

- Built on `tokio::sync::broadcast` — each topic is a separate channel with its own capacity.
- **Fan-out**: all active subscribers receive every published event.
- **Lagged subscribers**: if a subscriber falls behind the buffer capacity, it receives `BusRecvError::Lagged(n)` indicating how many events were skipped. Design subscribers to be fast consumers.
- **Lazy topics**: subscribing to a topic that doesn't exist yet creates it automatically. No upfront topic registration required.
- **Intra-process only**: the bus is not networked. For cross-process messaging, use the IPC socket (gristmill-daemon).

## Well-Known Topics

| Topic | Publisher | Payload fields |
|-------|-----------|---------------|
| `pipeline.completed` | grist-millwright | `run_id`, `pipeline_id`, `elapsed_ms`, `output` |
| `pipeline.failed` | grist-millwright | `run_id`, `pipeline_id`, `error`, `step_id` |
| `sieve.anomaly` | grist-sieve | `event_id`, `route`, `confidence`, `reason` |
| `ledger.threshold` | grist-ledger | `tier`, `used_mb`, `limit_mb`, `pct_used` |
| `hammer.budget` | grist-hammer | `daily_used`, `daily_limit`, `window_start_ms`, `pct_used` |

Custom topics can be used freely — just publish and subscribe with the same string key.

## Dependencies

```toml
grist-event  = { path = "../grist-event" }
tokio        # broadcast channels
dashmap      # concurrent topic registry
metrics      # publish/subscribe rate counters
tracing      # topic lifecycle logging
```
