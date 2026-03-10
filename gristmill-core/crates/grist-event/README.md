# grist-event

Universal message type for all cross-boundary communication in GristMill. Every event — from any channel, at any priority — is a `GristEvent`.

## Overview

`GristEvent` is the single type that flows through the entire system: from hopper adapters in TypeScript, through the Sieve triage classifier, into Ledger memory, and across the internal bus. Every cross-boundary message serializes to/from `GristEvent`.

## Key Types

```rust
pub struct GristEvent {
    pub id: Ulid,                          // Sortable unique ID (monotonic)
    pub source: ChannelType,               // Origin channel
    pub timestamp_ms: u64,                 // Wall-clock milliseconds
    pub payload: serde_json::Value,        // Arbitrary JSON payload
    pub metadata: EventMetadata,           // Routing + observability data
}

pub struct EventMetadata {
    pub priority: Priority,                // Low, Normal, High, Critical
    pub correlation_id: Option<String>,    // Groups related events
    pub reply_channel: Option<String>,     // Where to send the response
    pub ttl_ms: Option<u64>,               // Expiry (drop if elapsed)
    pub tags: HashMap<String, String>,     // Arbitrary key-value labels
}

pub enum ChannelType {
    Http, WebSocket, Cli, Cron, Webhook,
    MessageQueue, FileSystem, Python, TypeScript, Internal,
}

pub enum Priority { Low, Normal, High, Critical }
```

## Public API

### Construction

```rust
let event = GristEvent::new(ChannelType::Http, payload_json);

// Builder pattern
let event = GristEvent::new(ChannelType::Http, payload)
    .with_priority(Priority::High)
    .with_correlation_id("req-abc-123")
    .with_ttl_ms(30_000)
    .with_tag("service", "api-gateway");
```

### Payload helpers

```rust
// Extract text content for ML feature extraction
let text: Option<&str> = event.payload_as_text();

// Rough token count for budget estimation
let tokens: usize = event.estimated_token_count();

// SHA-256 of payload (for semantic cache lookup)
let hash: String = event.payload_hash();

// Check TTL
if event.is_expired() { return; }
```

### Serialization

```rust
// Serialize to compact JSON bytes
let bytes: Vec<u8> = event.to_json_bytes()?;

// Deserialize from JSON bytes
let event: GristEvent = GristEvent::from_json_bytes(&bytes)?;
```

## Design Notes

- **ULID ids** are time-sortable and URL-safe. They allow events to be ordered by arrival time without a central counter.
- **`payload` is `serde_json::Value`** to allow any channel-specific structure while keeping the wrapper type generic.
- **`ttl_ms`** is checked by the Sieve at the start of triage. Expired events are dropped immediately, before any ML work.
- **Correlation IDs** thread through the bus so downstream steps can join results from the same logical request.

## Dependencies

```toml
ulid      = "1"           # Sortable unique IDs
serde     = "1"           # Serialization
serde_json = "1"          # JSON payload
chrono    = "0.4"         # Timestamp helpers
sha2      = "0.10"        # payload_hash()
hex       = "0.4"         # Hash hex encoding
thiserror = "1"           # Error types
```
