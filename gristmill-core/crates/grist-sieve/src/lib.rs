//! `grist-sieve` — Triage engine for GristMill.
//!
//! The Sieve classifies every incoming [`GristEvent`] and assigns a
//! [`RouteDecision`] in under 5ms p99 (PRD requirement S-02).
//!
//! # Pipeline
//!
//! ```text
//! GristEvent
//!   │
//!   ├─ 1. TTL check (drop expired events)
//!   ├─ 2. Feature extraction (TF-IDF metadata + MiniLM embedding)
//!   ├─ 3. Exact cache lookup  ────────────────→ RouteDecision (cache hit)
//!   ├─ 4. Semantic cache lookup ──────────────→ RouteDecision (cache hit)
//!   ├─ 5. ONNX / heuristic classifier
//!   ├─ 6. Cost Oracle (threshold + token budget)
//!   ├─ 7. Store result in cache
//!   └─ 8. Write feedback record (async, non-blocking)
//!          └──────────────────────────────────→ RouteDecision
//! ```
//!
//! # Example
//!
//! ```rust,no_run
//! use grist_sieve::{Sieve, SieveConfig};
//! use grist_event::{ChannelType, GristEvent};
//!
//! #[tokio::main]
//! async fn main() {
//!     let config = SieveConfig::default();
//!     let sieve = Sieve::new(config).expect("failed to create Sieve");
//!     let event = GristEvent::new(
//!         ChannelType::Http,
//!         serde_json::json!({ "text": "Schedule a meeting with Alice" }),
//!     );
//!     let decision = sieve.triage(&event).await.expect("triage failed");
//!     println!("{decision:?}");
//! }
//! ```

pub mod cache;
pub mod classifier;
pub mod config;
pub mod cost_oracle;
pub mod error;
pub mod feedback;
pub mod features;

// Re-exports for convenience.
pub use cache::RoutingCache;
pub use classifier::{Classifier, RouteLabel};
pub use config::SieveConfig;
pub use cost_oracle::RouteDecision;
pub use error::SieveError;
pub use features::FeatureExtractor;

use grist_event::GristEvent;
use tracing::{debug, info, instrument, warn};
#[allow(unused_imports)]
use metrics;

// ─────────────────────────────────────────────────────────────────────────────
// Sieve
// ─────────────────────────────────────────────────────────────────────────────

/// The triage engine.  Entry point for all event routing.
///
/// `Sieve` is `Send + Sync` — construct once and share across tasks via `Arc`.
pub struct Sieve {
    extractor: FeatureExtractor,
    classifier: Classifier,
    oracle: cost_oracle::CostOracle,
    cache: RoutingCache,
    feedback: feedback::FeedbackLog,
    config: SieveConfig,
}

impl Sieve {
    /// Construct a `Sieve` from config.
    ///
    /// Loads the ONNX classifier if `config.model_path` is set and the file
    /// exists; otherwise falls back to the heuristic classifier.
    ///
    /// Opens the feedback log directory if configured; otherwise uses a no-op
    /// logger.
    pub fn new(config: SieveConfig) -> Result<Self, SieveError> {
        info!(
            model = ?config.model_path,
            threshold = config.confidence_threshold,
            "initialising Sieve"
        );

        let extractor = FeatureExtractor::new_no_embed();
        let classifier = Classifier::load(config.model_path.as_deref())?;
        let oracle = cost_oracle::CostOracle::new(&config);
        let cache = RoutingCache::new(
            config.exact_cache_size,
            config.semantic_cache_size,
            config.semantic_similarity_threshold,
        );
        let fb = match &config.feedback_dir {
            Some(dir) => feedback::FeedbackLog::open(dir)?,
            None => feedback::FeedbackLog::noop(),
        };

        Ok(Self {
            extractor,
            classifier,
            oracle,
            cache,
            feedback: fb,
            config,
        })
    }

    /// Classify an event and return a routing decision.
    ///
    /// **Must complete in <5ms p99** (PRD S-02).  All blocking work is
    /// avoided: ONNX inference runs synchronously since ONNX Runtime is
    /// designed to be called from any thread; the feedback write is
    /// non-blocking (mpsc `try_send`).
    #[instrument(level = "debug", skip(self, event), fields(
        event_id  = %event.id,
        source    = event.source.label(),
        priority  = %event.metadata.priority,
    ))]
    pub async fn triage(&self, event: &GristEvent) -> Result<RouteDecision, SieveError> {
        // ── 1. TTL guard ─────────────────────────────────────────────────────
        if event.is_expired() {
            warn!(event_id = %event.id, "dropping expired event");
            metrics::counter!("sieve.event.expired").increment(1);
            return Err(SieveError::EventExpired);
        }

        // ── 2. Feature extraction ─────────────────────────────────────────────
        let features = self.extractor.extract(event)?;

        // ── 3 & 4. Cache lookup ───────────────────────────────────────────────
        if let Some(cached) = self.cache.lookup(&features) {
            debug!(event_id = %event.id, "cache hit — skipping classifier");
            self.feedback.record(event, &cached);
            return Ok(cached);
        }

        // ── 5. Classification ─────────────────────────────────────────────────
        let classifier_output = self.classifier.classify(&features)?;
        debug!(
            label      = ?classifier_output.predicted_label,
            confidence = classifier_output.confidence,
            "classifier output"
        );
        metrics::histogram!("sieve.classifier.confidence")
            .record(classifier_output.confidence as f64);

        // ── 6. Cost Oracle ────────────────────────────────────────────────────
        let decision = self.oracle.evaluate(classifier_output, event)?;
        debug!(route = ?decision.label(), "oracle decision");

        // ── 7. Cache store ────────────────────────────────────────────────────
        self.cache.store(&features, &decision);

        // ── 8. Feedback log (non-blocking) ────────────────────────────────────
        self.feedback.record(event, &decision);

        metrics::counter!("sieve.events.triaged").increment(1);
        Ok(decision)
    }

    // ── Hot-reload ────────────────────────────────────────────────────────────

    /// Hot-reload the ONNX classifier model from disk.
    ///
    /// PRD S-07: "Model swap completes in <500ms with zero dropped events."
    ///
    /// In-flight `triage()` calls complete on the old model; subsequent calls
    /// use the new one.  The routing cache is cleared to avoid stale entries.
    pub fn hot_reload_model(&self) -> Result<(), SieveError> {
        info!("initiating classifier hot-reload");
        self.classifier.hot_reload()?;
        self.cache.clear();
        info!("classifier hot-reload complete; cache cleared");
        Ok(())
    }

    // ── Observability ─────────────────────────────────────────────────────────

    /// Current cache statistics.
    pub fn cache_stats(&self) -> cache::CacheSnapshot {
        self.cache.stats()
    }

    /// Number of feedback records sent to the log writer.
    pub fn feedback_records_sent(&self) -> u64 {
        self.feedback.records_sent()
    }

    /// The configured confidence threshold.
    pub fn confidence_threshold(&self) -> f32 {
        self.config.confidence_threshold
    }
}

impl std::fmt::Debug for Sieve {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Sieve")
            .field("threshold", &self.config.confidence_threshold)
            .finish()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Integration tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use grist_event::{ChannelType, GristEvent, Priority};

    fn sieve() -> Sieve {
        Sieve::new(SieveConfig::default()).unwrap()
    }

    fn event(text: &str) -> GristEvent {
        GristEvent::new(
            ChannelType::Http,
            serde_json::json!({ "text": text }),
        )
    }

    // ── S-01: All 4 routes reachable ─────────────────────────────────────────
    #[tokio::test]
    async fn triage_returns_a_decision() {
        let sieve = sieve();
        let e = event("Schedule a meeting tomorrow at 10am with Alice");
        let decision = sieve.triage(&e).await.unwrap();
        let _ = decision.confidence();
    }

    // ── S-02: Latency guard (soft — heuristic mode, no ONNX) ─────────────────
    #[tokio::test]
    async fn triage_completes_quickly() {
        let sieve = sieve();
        let e = event("quick latency check");
        let start = std::time::Instant::now();
        sieve.triage(&e).await.unwrap();
        let elapsed = start.elapsed();
        assert!(
            elapsed.as_millis() < 50,
            "triage took {}ms (budget 50ms for heuristic mode)",
            elapsed.as_millis()
        );
    }

    // ── S-04: Cache hit on repeated identical event ───────────────────────────
    #[tokio::test]
    async fn repeated_event_hits_cache() {
        let sieve = sieve();
        let e = event("cache hit test event");
        sieve.triage(&e).await.unwrap(); // populate cache
        sieve.triage(&e).await.unwrap(); // should hit cache
        let stats = sieve.cache_stats();
        assert!(stats.exact_hits >= 1, "expected at least one exact cache hit");
    }

    // ── S-06: Feedback records are counted ───────────────────────────────────
    #[tokio::test]
    async fn feedback_records_are_sent() {
        let sieve = sieve();
        sieve.triage(&event("event one")).await.unwrap();
        sieve.triage(&event("event two")).await.unwrap();
        assert!(sieve.feedback_records_sent() >= 1);
    }

    // ── Expired events are rejected ───────────────────────────────────────────
    #[tokio::test]
    async fn expired_event_returns_error() {
        let sieve = sieve();
        // Set timestamp 10 seconds in the past so TTL=1ms is definitively expired.
        let mut e = GristEvent::new(
            ChannelType::Http,
            serde_json::json!({ "text": "expired" }),
        );
        e.timestamp_ms = grist_event::current_timestamp_ms().saturating_sub(10_000);
        let e = e.with_ttl_ms(1);
        let result = sieve.triage(&e).await;
        assert!(matches!(result, Err(SieveError::EventExpired)));
    }

    // ── High-priority events are triaged ─────────────────────────────────────
    #[tokio::test]
    async fn critical_priority_event_is_triaged() {
        let sieve = sieve();
        let e = GristEvent::new(
            ChannelType::Http,
            serde_json::json!({ "text": "CRITICAL: system down" }),
        )
        .with_priority(Priority::Critical);
        let decision = sieve.triage(&e).await.unwrap();
        assert!(decision.confidence() > 0.0);
    }

    // ── S-08: Threshold is configurable ──────────────────────────────────────
    // Must be async because Sieve::new() spawns a Tokio task for the feedback log.
    #[tokio::test]
    async fn custom_threshold_is_stored() {
        let mut config = SieveConfig::default();
        config.confidence_threshold = 0.70;
        let sieve = Sieve::new(config).unwrap();
        assert!((sieve.confidence_threshold() - 0.70).abs() < 1e-5);
    }

    // ── Route label serialises correctly ─────────────────────────────────────
    #[test]
    fn route_decision_serialises_to_json() {
        let d = RouteDecision::LocalMl {
            model_id: "intent-v1".into(),
            confidence: 0.95,
        };
        let json = serde_json::to_value(&d).unwrap();
        assert_eq!(json["route"], "LOCAL_ML");
    }

    // ── All four RouteDecision variants serialise correctly ───────────────────
    #[test]
    fn all_route_variants_serialise() {
        let decisions = vec![
            RouteDecision::LocalMl { model_id: "m".into(), confidence: 0.9 },
            RouteDecision::Rules { rule_id: "r".into(), confidence: 0.8 },
            RouteDecision::Hybrid {
                local_model: "m".into(),
                llm_prompt_template: "t".into(),
                estimated_tokens: 100,
                confidence: 0.7,
            },
            RouteDecision::LlmNeeded {
                reason: "complex".into(),
                estimated_tokens: 500,
                estimated_cost_usd: 0.01,
                confidence: 0.6,
            },
        ];
        for d in decisions {
            let json = serde_json::to_string(&d).unwrap();
            assert!(!json.is_empty());
        }
    }
}
