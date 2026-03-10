//! `grist-bus` — Internal typed pub/sub event bus.
//!
//! A lightweight, thread-safe broadcast broker used for intra-process
//! communication between GristMill subsystems.
//!
//! # Standard topics
//!
//! | Topic constant | When published |
//! |---|---|
//! | [`TOPIC_PIPELINE_COMPLETED`] | A DAG pipeline finishes successfully |
//! | [`TOPIC_PIPELINE_FAILED`] | A DAG pipeline finishes with an error |
//! | [`TOPIC_SIEVE_ANOMALY`] | The sieve classifier flags a low-confidence routing |
//! | [`TOPIC_LEDGER_THRESHOLD`] | Ledger memory usage crosses a configured limit |
//! | [`TOPIC_HAMMER_BUDGET`] | Token budget watermark reached |
//!
//! # Usage
//! ```rust
//! use grist_bus::{EventBus, TOPIC_PIPELINE_COMPLETED};
//!
//! let bus = EventBus::default();
//! let mut rx = bus.subscribe(TOPIC_PIPELINE_COMPLETED);
//! bus.publish(TOPIC_PIPELINE_COMPLETED, serde_json::json!({"id": "abc"}));
//! ```

use std::sync::Arc;

use dashmap::DashMap;
use grist_event::GristEvent;
use metrics::counter;
use tokio::sync::broadcast;
use tracing::{debug, warn};

// ─────────────────────────────────────────────────────────────────────────────
// Well-known topic names
// ─────────────────────────────────────────────────────────────────────────────

pub const TOPIC_PIPELINE_COMPLETED: &str = "pipeline.completed";
pub const TOPIC_PIPELINE_FAILED: &str = "pipeline.failed";
pub const TOPIC_SIEVE_ANOMALY: &str = "sieve.anomaly";
pub const TOPIC_LEDGER_THRESHOLD: &str = "ledger.threshold";
pub const TOPIC_HAMMER_BUDGET: &str = "hammer.budget";

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/// The payload type carried by bus events.
pub type BusEvent = serde_json::Value;

/// A handle to a topic subscription.
///
/// Dropping this value unsubscribes the caller automatically.
pub type Subscription = broadcast::Receiver<BusEvent>;

/// Error returned when a subscriber's lag causes it to miss events.
pub use broadcast::error::RecvError as BusRecvError;
pub use broadcast::error::TryRecvError as BusTryRecvError;

// ─────────────────────────────────────────────────────────────────────────────
// EventBus
// ─────────────────────────────────────────────────────────────────────────────

/// Thread-safe, topic-based pub/sub broker.
///
/// [`EventBus`] is cheap to clone — all clones share the same underlying
/// topic map.  The bus is designed for intra-process use only.
///
/// Each topic uses a [`tokio::sync::broadcast`] channel of fixed capacity
/// (set at construction time).  Slow subscribers that fall behind will
/// receive [`BusRecvError::Lagged`] errors and miss intermediate messages.
#[derive(Debug, Clone)]
pub struct EventBus {
    topics: Arc<DashMap<String, broadcast::Sender<BusEvent>>>,
    capacity: usize,
}

impl EventBus {
    /// Create an [`EventBus`] with the given per-topic channel capacity.
    ///
    /// Typical values: 128–4096.  A larger capacity costs more memory but
    /// tolerates slower subscribers better.
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "EventBus capacity must be > 0");
        Self {
            topics: Arc::new(DashMap::new()),
            capacity,
        }
    }

    // ── Subscription ─────────────────────────────────────────────────────

    /// Subscribe to a named topic.
    ///
    /// If the topic does not yet exist it is created lazily.  All future
    /// [`publish`](EventBus::publish) calls on the same topic will be
    /// delivered to this receiver (up to channel capacity).
    pub fn subscribe(&self, topic: &str) -> Subscription {
        self.topics
            .entry(topic.to_string())
            .or_insert_with(|| broadcast::channel(self.capacity).0)
            .subscribe()
    }

    // ── Publishing ────────────────────────────────────────────────────────

    /// Publish a JSON payload on a topic.
    ///
    /// No-op if no subscriber has ever called [`subscribe`](EventBus::subscribe)
    /// on this topic.  Logs a warning if some receivers have lagged and will
    /// miss this event.
    pub fn publish(&self, topic: &str, payload: BusEvent) {
        if let Some(tx) = self.topics.get(topic) {
            match tx.send(payload) {
                Ok(n) => {
                    debug!(topic, receivers = n, "published bus event");
                    counter!("bus.events_published", "topic" => topic.to_string())
                        .increment(1);
                }
                Err(_) => {
                    // All receivers have been dropped — not an error.
                    debug!(topic, "bus publish: no active receivers");
                }
            }
        }
    }

    /// Publish a [`GristEvent`] serialised as JSON on a topic.
    ///
    /// Returns `false` if serialisation fails (should never happen for
    /// well-formed events).
    pub fn publish_event(&self, topic: &str, event: &GristEvent) -> bool {
        match serde_json::to_value(event) {
            Ok(val) => {
                self.publish(topic, val);
                true
            }
            Err(e) => {
                warn!(topic, error = %e, "failed to serialise GristEvent for bus");
                false
            }
        }
    }

    // ── Helpers for standard topics ───────────────────────────────────────

    /// Shorthand: publish a `pipeline.completed` event.
    pub fn pipeline_completed(&self, pipeline_id: &str, elapsed_ms: u64) {
        self.publish(
            TOPIC_PIPELINE_COMPLETED,
            serde_json::json!({
                "pipeline_id": pipeline_id,
                "elapsed_ms": elapsed_ms,
            }),
        );
    }

    /// Shorthand: publish a `pipeline.failed` event.
    pub fn pipeline_failed(&self, pipeline_id: &str, reason: &str) {
        self.publish(
            TOPIC_PIPELINE_FAILED,
            serde_json::json!({
                "pipeline_id": pipeline_id,
                "reason": reason,
            }),
        );
    }

    /// Shorthand: publish a `sieve.anomaly` event.
    pub fn sieve_anomaly(&self, event_id: &str, confidence: f32, route: &str) {
        self.publish(
            TOPIC_SIEVE_ANOMALY,
            serde_json::json!({
                "event_id": event_id,
                "confidence": confidence,
                "route": route,
            }),
        );
    }

    /// Shorthand: publish a `ledger.threshold` event.
    pub fn ledger_threshold(&self, tier: &str, used: u64, limit: u64) {
        self.publish(
            TOPIC_LEDGER_THRESHOLD,
            serde_json::json!({
                "tier": tier,
                "used": used,
                "limit": limit,
            }),
        );
    }

    /// Shorthand: publish a `hammer.budget` event.
    pub fn hammer_budget(&self, daily_used: u64, daily_limit: u64, monthly_used: u64) {
        self.publish(
            TOPIC_HAMMER_BUDGET,
            serde_json::json!({
                "daily_used": daily_used,
                "daily_limit": daily_limit,
                "monthly_used": monthly_used,
            }),
        );
    }

    // ── Introspection ─────────────────────────────────────────────────────

    /// Return the number of topics that have ever been subscribed to.
    pub fn topic_count(&self) -> usize {
        self.topics.len()
    }

    /// Return the number of active receivers for a topic.
    ///
    /// Returns 0 if the topic has never been subscribed to.
    pub fn receiver_count(&self, topic: &str) -> usize {
        self.topics
            .get(topic)
            .map(|tx| tx.receiver_count())
            .unwrap_or(0)
    }

    /// Return a sorted list of known topic names.
    pub fn topics(&self) -> Vec<String> {
        let mut names: Vec<String> =
            self.topics.iter().map(|e| e.key().clone()).collect();
        names.sort();
        names
    }
}

impl Default for EventBus {
    /// Create an [`EventBus`] with a default per-topic capacity of 1 024.
    fn default() -> Self {
        Self::new(1024)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Core pub/sub ─────────────────────────────────────────────────────

    #[tokio::test]
    async fn subscriber_receives_published_event() {
        let bus = EventBus::default();
        let mut rx = bus.subscribe("test.topic");
        bus.publish("test.topic", serde_json::json!({"hello": "world"}));
        let received = rx.recv().await.unwrap();
        assert_eq!(received["hello"], "world");
    }

    #[tokio::test]
    async fn no_subscribers_does_not_panic() {
        let bus = EventBus::default();
        bus.publish("ghost.topic", serde_json::json!({}));
    }

    #[tokio::test]
    async fn multiple_subscribers_all_receive() {
        let bus = EventBus::default();
        let mut rx1 = bus.subscribe("multi.topic");
        let mut rx2 = bus.subscribe("multi.topic");
        bus.publish("multi.topic", serde_json::json!({"n": 1}));
        assert_eq!(rx1.recv().await.unwrap()["n"], 1);
        assert_eq!(rx2.recv().await.unwrap()["n"], 1);
    }

    #[tokio::test]
    async fn subscription_is_topic_scoped() {
        let bus = EventBus::default();
        let mut rx_a = bus.subscribe("topic.a");
        let mut rx_b = bus.subscribe("topic.b");
        bus.publish("topic.a", serde_json::json!({"src": "a"}));
        bus.publish("topic.b", serde_json::json!({"src": "b"}));
        assert_eq!(rx_a.recv().await.unwrap()["src"], "a");
        assert_eq!(rx_b.recv().await.unwrap()["src"], "b");
    }

    #[tokio::test]
    async fn dropped_receiver_does_not_prevent_publish() {
        let bus = EventBus::default();
        {
            let _rx = bus.subscribe("ephemeral.topic");
        } // rx dropped here
        // Should not panic even though receiver is gone.
        bus.publish("ephemeral.topic", serde_json::json!({"x": 1}));
    }

    // ── GristEvent publish ────────────────────────────────────────────────

    #[tokio::test]
    async fn publish_event_serialises_grist_event() {
        use grist_event::{ChannelType, GristEvent};
        let bus = EventBus::default();
        let mut rx = bus.subscribe("event.topic");
        let event = GristEvent::new(
            ChannelType::Cli,
            serde_json::json!({"text": "hello"}),
        );
        let id = event.id.to_string();
        assert!(bus.publish_event("event.topic", &event));
        let received = rx.recv().await.unwrap();
        assert_eq!(received["id"].as_str().unwrap(), id);
    }

    // ── Standard topic helpers ────────────────────────────────────────────

    #[tokio::test]
    async fn pipeline_completed_publishes_correct_payload() {
        let bus = EventBus::default();
        let mut rx = bus.subscribe(TOPIC_PIPELINE_COMPLETED);
        bus.pipeline_completed("pipe-1", 42);
        let v = rx.recv().await.unwrap();
        assert_eq!(v["pipeline_id"], "pipe-1");
        assert_eq!(v["elapsed_ms"], 42);
    }

    #[tokio::test]
    async fn pipeline_failed_publishes_correct_payload() {
        let bus = EventBus::default();
        let mut rx = bus.subscribe(TOPIC_PIPELINE_FAILED);
        bus.pipeline_failed("pipe-2", "timeout");
        let v = rx.recv().await.unwrap();
        assert_eq!(v["pipeline_id"], "pipe-2");
        assert_eq!(v["reason"], "timeout");
    }

    #[tokio::test]
    async fn sieve_anomaly_publishes_correct_payload() {
        let bus = EventBus::default();
        let mut rx = bus.subscribe(TOPIC_SIEVE_ANOMALY);
        bus.sieve_anomaly("evt-abc", 0.42, "escalate");
        let v = rx.recv().await.unwrap();
        assert_eq!(v["event_id"], "evt-abc");
        assert_eq!(v["route"], "escalate");
        let conf = v["confidence"].as_f64().unwrap();
        assert!((conf - 0.42).abs() < 0.001);
    }

    #[tokio::test]
    async fn ledger_threshold_publishes_correct_payload() {
        let bus = EventBus::default();
        let mut rx = bus.subscribe(TOPIC_LEDGER_THRESHOLD);
        bus.ledger_threshold("hot", 900, 1000);
        let v = rx.recv().await.unwrap();
        assert_eq!(v["tier"], "hot");
        assert_eq!(v["used"], 900);
        assert_eq!(v["limit"], 1000);
    }

    #[tokio::test]
    async fn hammer_budget_publishes_correct_payload() {
        let bus = EventBus::default();
        let mut rx = bus.subscribe(TOPIC_HAMMER_BUDGET);
        bus.hammer_budget(450_000, 500_000, 9_000_000);
        let v = rx.recv().await.unwrap();
        assert_eq!(v["daily_used"], 450_000);
        assert_eq!(v["daily_limit"], 500_000);
        assert_eq!(v["monthly_used"], 9_000_000);
    }

    // ── Introspection ─────────────────────────────────────────────────────

    #[tokio::test]
    async fn receiver_count_tracks_live_subscriptions() {
        let bus = EventBus::default();
        assert_eq!(bus.receiver_count("new.topic"), 0);
        let rx1 = bus.subscribe("new.topic");
        assert_eq!(bus.receiver_count("new.topic"), 1);
        let rx2 = bus.subscribe("new.topic");
        assert_eq!(bus.receiver_count("new.topic"), 2);
        drop(rx1);
        drop(rx2);
    }

    #[tokio::test]
    async fn topic_count_grows_with_subscriptions() {
        let bus = EventBus::default();
        assert_eq!(bus.topic_count(), 0);
        let _r1 = bus.subscribe("t1");
        assert_eq!(bus.topic_count(), 1);
        let _r2 = bus.subscribe("t2");
        assert_eq!(bus.topic_count(), 2);
    }

    #[tokio::test]
    async fn topics_returns_sorted_names() {
        let bus = EventBus::default();
        let _r = bus.subscribe("b.topic");
        let _s = bus.subscribe("a.topic");
        let names = bus.topics();
        assert_eq!(names, vec!["a.topic", "b.topic"]);
    }

    // ── Clone shares state ────────────────────────────────────────────────

    #[tokio::test]
    async fn cloned_bus_shares_subscriptions() {
        let bus = EventBus::default();
        let mut rx = bus.subscribe("shared.topic");
        let bus2 = bus.clone();
        bus2.publish("shared.topic", serde_json::json!({"from": "clone"}));
        let v = rx.recv().await.unwrap();
        assert_eq!(v["from"], "clone");
    }
}
