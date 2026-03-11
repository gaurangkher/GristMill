//! `grist-event` — The universal message type for GristMill.
//!
//! Every cross-boundary message in the system is a [`GristEvent`].  It carries
//! a ULID identifier (sortable, collision-resistant), a typed source channel,
//! a JSON payload, and structured metadata for routing and observability.

use std::collections::HashMap;

use chrono::Utc;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use ulid::Ulid;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, thiserror::Error)]
pub enum EventError {
    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error("invalid payload: {0}")]
    InvalidPayload(String),
}

// ─────────────────────────────────────────────────────────────────────────────
// Core types
// ─────────────────────────────────────────────────────────────────────────────

/// The universal event type.  Every piece of work flowing through GristMill
/// is represented as a `GristEvent`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GristEvent {
    /// ULID — monotonically sortable, URL-safe unique identifier.
    pub id: Ulid,

    /// Which external or internal channel created this event.
    pub source: ChannelType,

    /// Wall-clock time of event creation (milliseconds since UNIX epoch).
    pub timestamp_ms: u64,

    /// Arbitrary JSON payload.  Schema is owned by the producing channel.
    pub payload: serde_json::Value,

    /// Routing and observability metadata.
    pub metadata: EventMetadata,
}

impl GristEvent {
    // ── Constructors ─────────────────────────────────────────────────────────

    /// Create a new event with the current timestamp and default metadata.
    pub fn new(source: ChannelType, payload: serde_json::Value) -> Self {
        Self {
            id: Ulid::new(),
            source,
            timestamp_ms: current_timestamp_ms(),
            payload,
            metadata: EventMetadata::default(),
        }
    }

    /// Builder-style: set priority.
    pub fn with_priority(mut self, priority: Priority) -> Self {
        self.metadata.priority = priority;
        self
    }

    /// Builder-style: set a correlation id.
    pub fn with_correlation_id(mut self, id: impl Into<String>) -> Self {
        self.metadata.correlation_id = Some(id.into());
        self
    }

    /// Builder-style: set TTL in milliseconds.
    pub fn with_ttl_ms(mut self, ttl: u64) -> Self {
        self.metadata.ttl_ms = Some(ttl);
        self
    }

    /// Builder-style: add a tag.
    pub fn with_tag(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.tags.insert(key.into(), value.into());
        self
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

    /// Extract a text representation of the payload for feature extraction.
    ///
    /// Tries `payload["text"]`, `payload["content"]`, `payload["body"]` in
    /// order, then falls back to a compact JSON representation.
    pub fn payload_as_text(&self) -> String {
        for key in &["text", "content", "body", "message", "query"] {
            if let Some(v) = self.payload.get(key) {
                if let Some(s) = v.as_str() {
                    return s.to_owned();
                }
            }
        }
        // Fallback: serialize the whole payload compactly.
        serde_json::to_string(&self.payload).unwrap_or_default()
    }

    /// Approximate token count using a simple whitespace heuristic (~4 chars/token).
    pub fn estimated_token_count(&self) -> u32 {
        let text = self.payload_as_text();
        ((text.len() as f64) / 4.0).ceil() as u32
    }

    /// SHA-256 of the normalized text payload.  Used as a cache key.
    pub fn payload_hash(&self) -> String {
        let text = self.payload_as_text();
        let normalized = normalize_text(&text);
        let mut hasher = Sha256::new();
        hasher.update(normalized.as_bytes());
        hex::encode(hasher.finalize())
    }

    /// Returns `true` if the event has expired according to its TTL.
    pub fn is_expired(&self) -> bool {
        if let Some(ttl) = self.metadata.ttl_ms {
            current_timestamp_ms() > self.timestamp_ms + ttl
        } else {
            false
        }
    }

    /// Serialize to JSON bytes (for FFI / IPC transport).
    pub fn to_json_bytes(&self) -> Result<Vec<u8>, EventError> {
        Ok(serde_json::to_vec(self)?)
    }

    /// Deserialize from JSON bytes.
    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, EventError> {
        Ok(serde_json::from_slice(bytes)?)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ChannelType
// ─────────────────────────────────────────────────────────────────────────────

/// Identifies the origin of a [`GristEvent`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ChannelType {
    Http,
    WebSocket,
    Cli,
    Cron,
    Webhook {
        provider: String,
    },
    MessageQueue {
        topic: String,
    },
    FileSystem {
        path: String,
    },
    /// Event originating from the Python shell via PyO3.
    Python {
        callback_id: String,
    },
    /// Event originating from the TypeScript shell via napi-rs.
    TypeScript {
        adapter_id: String,
    },
    /// Internal system event (bus, compactor, etc.).
    Internal {
        subsystem: String,
    },
}

impl ChannelType {
    /// Human-readable short label for logging and metrics.
    pub fn label(&self) -> &'static str {
        match self {
            ChannelType::Http => "http",
            ChannelType::WebSocket => "websocket",
            ChannelType::Cli => "cli",
            ChannelType::Cron => "cron",
            ChannelType::Webhook { .. } => "webhook",
            ChannelType::MessageQueue { .. } => "message_queue",
            ChannelType::FileSystem { .. } => "filesystem",
            ChannelType::Python { .. } => "python",
            ChannelType::TypeScript { .. } => "typescript",
            ChannelType::Internal { .. } => "internal",
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// EventMetadata
// ─────────────────────────────────────────────────────────────────────────────

/// Routing and observability metadata attached to every [`GristEvent`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventMetadata {
    /// Routing priority — affects queue ordering and notification urgency.
    pub priority: Priority,

    /// Groups related events across pipeline steps.
    pub correlation_id: Option<String>,

    /// Where to send a reply (channel id or topic name).
    pub reply_channel: Option<String>,

    /// Optional expiry: event will be dropped if processing starts after this.
    pub ttl_ms: Option<u64>,

    /// Arbitrary key-value tags for routing rules and logging.
    pub tags: HashMap<String, String>,
}

impl Default for EventMetadata {
    fn default() -> Self {
        Self {
            priority: Priority::Normal,
            correlation_id: None,
            reply_channel: None,
            ttl_ms: None,
            tags: HashMap::new(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Priority
// ─────────────────────────────────────────────────────────────────────────────

/// Event priority.  Affects both scheduling and notification delivery.
///
/// - `Critical` bypasses digest batching and quiet hours.
/// - `High` is processed before `Normal` and `Low`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Default)]
#[serde(rename_all = "lowercase")]
pub enum Priority {
    Low = 0,
    #[default]
    Normal = 1,
    High = 2,
    Critical = 3,
}

impl Priority {
    /// Returns `true` if this priority bypasses quiet hours.
    pub fn bypasses_quiet_hours(&self) -> bool {
        matches!(self, Priority::Critical)
    }

    /// Returns `true` if this priority skips digest batching.
    pub fn skip_digest(&self) -> bool {
        matches!(self, Priority::Critical | Priority::High)
    }
}

impl std::fmt::Display for Priority {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Priority::Low => write!(f, "low"),
            Priority::Normal => write!(f, "normal"),
            Priority::High => write!(f, "high"),
            Priority::Critical => write!(f, "critical"),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Returns the current wall-clock time in milliseconds since the UNIX epoch.
pub fn current_timestamp_ms() -> u64 {
    Utc::now().timestamp_millis() as u64
}

/// Normalize text for cache keying: lowercase, collapse whitespace.
pub fn normalize_text(text: &str) -> String {
    let lower = text.to_lowercase();
    lower.split_whitespace().collect::<Vec<_>>().join(" ")
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_event() -> GristEvent {
        GristEvent::new(
            ChannelType::Http,
            serde_json::json!({ "text": "Schedule a meeting with Alice tomorrow at 10am" }),
        )
    }

    #[test]
    fn event_has_unique_ids() {
        let e1 = sample_event();
        let e2 = sample_event();
        assert_ne!(e1.id, e2.id);
    }

    #[test]
    fn payload_as_text_extracts_text_field() {
        let e = sample_event();
        assert_eq!(
            e.payload_as_text(),
            "Schedule a meeting with Alice tomorrow at 10am"
        );
    }

    #[test]
    fn payload_as_text_fallback_to_json() {
        let e = GristEvent::new(
            ChannelType::Cli,
            serde_json::json!({ "command": "status", "flags": [] }),
        );
        let text = e.payload_as_text();
        assert!(text.contains("status"));
    }

    #[test]
    fn payload_hash_is_deterministic() {
        let e = sample_event();
        assert_eq!(e.payload_hash(), e.payload_hash());
    }

    #[test]
    fn payload_hash_differs_for_different_payloads() {
        let e1 = GristEvent::new(ChannelType::Http, serde_json::json!({"text": "foo"}));
        let e2 = GristEvent::new(ChannelType::Http, serde_json::json!({"text": "bar"}));
        assert_ne!(e1.payload_hash(), e2.payload_hash());
    }

    #[test]
    fn json_round_trip() {
        let e = sample_event()
            .with_priority(Priority::High)
            .with_correlation_id("corr-123")
            .with_tag("pipeline", "calendar");
        let bytes = e.to_json_bytes().unwrap();
        let restored = GristEvent::from_json_bytes(&bytes).unwrap();
        assert_eq!(e.id, restored.id);
        assert_eq!(e.metadata.priority, restored.metadata.priority);
        assert_eq!(e.metadata.correlation_id, restored.metadata.correlation_id);
    }

    #[test]
    fn ttl_expiry_detection() {
        // Build an event whose timestamp is set 10 seconds in the past,
        // with a 1ms TTL — it must be expired regardless of CPU speed.
        let mut e = GristEvent::new(ChannelType::Cli, serde_json::json!({}));
        e.timestamp_ms = current_timestamp_ms().saturating_sub(10_000); // 10s ago
        e = e.with_ttl_ms(1); // 1ms TTL — definitively expired
        assert!(e.is_expired());
    }

    #[test]
    fn priority_ordering() {
        assert!(Priority::Critical > Priority::High);
        assert!(Priority::High > Priority::Normal);
        assert!(Priority::Normal > Priority::Low);
    }

    #[test]
    fn normalize_text_lowercases_and_collapses_whitespace() {
        let n = normalize_text("  Hello   WORLD  ");
        assert_eq!(n, "hello world");
    }

    #[test]
    fn estimated_token_count_reasonable() {
        let e = GristEvent::new(
            ChannelType::Http,
            serde_json::json!({ "text": "hello world" }), // ~3 tokens
        );
        let count = e.estimated_token_count();
        assert!((2..=6).contains(&count), "count was {count}");
    }
}
