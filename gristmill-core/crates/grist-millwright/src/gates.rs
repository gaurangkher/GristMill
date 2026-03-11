//! Approval gates — conditional and human-approval barriers in a pipeline.
//!
//! A gate is a special step type (`StepType::Gate`) or a flag on any step
//! (`Step::requires_approval`).  The gate must "open" before the scheduler
//! continues execution of the step.
//!
//! # Gate types
//!
//! | Type | Behaviour |
//! |------|-----------|
//! | [`GateType::Auto`] | Evaluates a Rust expression (whitelist of predicates) synchronously. |
//! | [`GateType::Channel`] | Sends an approval request on a named channel and waits for a reply. |
//! | [`GateType::AlwaysOpen`] | Passes immediately (used in dry-run / test mode). |
//! | [`GateType::AlwaysClosed`] | Always rejects (used in testing failure paths). |
//!
//! # Integration
//!
//! Channel-based gates communicate with the TypeScript `Bell Tower` via the
//! `grist-bus` topic `gate.request.<step_id>`.  The Bell Tower forwards the
//! request to Slack (or another channel) and writes the reply back to
//! `gate.response.<step_id>`.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::oneshot;
use tracing::{debug, info, warn};

use crate::error::MillwrightError;

// ─────────────────────────────────────────────────────────────────────────────
// GateDecision
// ─────────────────────────────────────────────────────────────────────────────

/// Result of evaluating a gate.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GateDecision {
    /// Gate is open — step may proceed.
    Open,
    /// Gate is closed — pipeline should halt (or skip, per FailurePolicy).
    Closed { reason: String },
}

// ─────────────────────────────────────────────────────────────────────────────
// GateType
// ─────────────────────────────────────────────────────────────────────────────

/// Distinguishes how a gate evaluates its condition.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum GateType {
    /// Evaluate a named predicate against the step's input context.
    Auto { predicate: String },
    /// Wait for an external approval signal on `channel`.
    Channel { channel: String },
    /// Always open (test / dry-run).
    AlwaysOpen,
    /// Always closed with a fixed reason (test failure paths).
    AlwaysClosed { reason: String },
}

// ─────────────────────────────────────────────────────────────────────────────
// GateEvaluator
// ─────────────────────────────────────────────────────────────────────────────

/// Evaluates gate conditions for the scheduler.
///
/// The evaluator holds:
/// - A registry of named predicates (simple closures over `serde_json::Value`).
/// - A channel registry for pending approval requests.
///
/// It is `Send + Sync` and wrapped in `Arc` so the scheduler can share it.
pub struct GateEvaluator {
    predicates: HashMap<String, Box<dyn Fn(&Value) -> bool + Send + Sync>>,
    /// Pending channel approvals: step_id → reply sender.
    pending: Mutex<HashMap<String, oneshot::Sender<GateDecision>>>,
    /// Default timeout for channel-based gates.
    default_timeout: Duration,
}

impl std::fmt::Debug for GateEvaluator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GateEvaluator")
            .field("predicates", &self.predicates.keys().collect::<Vec<_>>())
            .finish()
    }
}

impl GateEvaluator {
    /// Create a new evaluator with built-in predicates and a default timeout.
    pub fn new(default_timeout: Duration) -> Self {
        let mut predicates: HashMap<String, Box<dyn Fn(&Value) -> bool + Send + Sync>> =
            HashMap::new();

        // ── Built-in predicates ──────────────────────────────────────────────

        // "always_true" — for testing
        predicates.insert("always_true".into(), Box::new(|_| true));

        // "always_false" — for testing failure paths
        predicates.insert("always_false".into(), Box::new(|_| false));

        // "has_text" — payload must have a non-empty "text" field
        predicates.insert(
            "has_text".into(),
            Box::new(|ctx: &Value| ctx["text"].as_str().map(|s| !s.is_empty()).unwrap_or(false)),
        );

        // "high_confidence" — context must have confidence >= 0.85
        predicates.insert(
            "high_confidence".into(),
            Box::new(|ctx: &Value| ctx["confidence"].as_f64().unwrap_or(0.0) >= 0.85),
        );

        // "low_cost" — estimated_cost_usd < 0.01
        predicates.insert(
            "low_cost".into(),
            Box::new(|ctx: &Value| ctx["estimated_cost_usd"].as_f64().unwrap_or(f64::MAX) < 0.01),
        );

        Self {
            predicates,
            pending: Mutex::new(HashMap::new()),
            default_timeout,
        }
    }

    /// Register a custom named predicate.
    pub fn register_predicate(
        &mut self,
        name: impl Into<String>,
        f: impl Fn(&Value) -> bool + Send + Sync + 'static,
    ) {
        self.predicates.insert(name.into(), Box::new(f));
    }

    /// Evaluate a gate given its type and the current step context.
    ///
    /// For `Channel` gates this parks the calling task until a reply arrives
    /// (via [`GateEvaluator::resolve`]) or the timeout fires.
    pub async fn evaluate(
        self: &Arc<Self>,
        step_id: &str,
        gate: &GateType,
        context: &Value,
        timeout: Option<Duration>,
    ) -> Result<GateDecision, MillwrightError> {
        let timeout = timeout.unwrap_or(self.default_timeout);

        match gate {
            GateType::AlwaysOpen => {
                debug!(step_id, "gate AlwaysOpen");
                Ok(GateDecision::Open)
            }

            GateType::AlwaysClosed { reason } => {
                debug!(step_id, reason, "gate AlwaysClosed");
                Ok(GateDecision::Closed {
                    reason: reason.clone(),
                })
            }

            GateType::Auto { predicate } => match self.predicates.get(predicate.as_str()) {
                Some(pred) => {
                    let open = pred(context);
                    debug!(step_id, predicate, open, "gate Auto evaluated");
                    if open {
                        Ok(GateDecision::Open)
                    } else {
                        Ok(GateDecision::Closed {
                            reason: format!("predicate '{predicate}' returned false"),
                        })
                    }
                }
                None => {
                    warn!(
                        step_id,
                        predicate, "unknown gate predicate; defaulting to Open"
                    );
                    Ok(GateDecision::Open)
                }
            },

            GateType::Channel { channel } => {
                let (tx, rx) = oneshot::channel();
                self.pending.lock().insert(step_id.to_owned(), tx);

                info!(step_id, channel, "gate awaiting channel approval");

                // In a real deployment the Bell Tower would pick up the pending
                // entry via the bus topic `gate.request.<step_id>` and call
                // `GateEvaluator::resolve` when a human responds.
                //
                // Here we just wait with a timeout.
                match tokio::time::timeout(timeout, rx).await {
                    Ok(Ok(decision)) => {
                        info!(step_id, channel, ?decision, "gate resolved");
                        Ok(decision)
                    }
                    Ok(Err(_)) => {
                        // Sender dropped — treat as rejection.
                        Ok(GateDecision::Closed {
                            reason: "approval channel dropped".into(),
                        })
                    }
                    Err(_) => {
                        self.pending.lock().remove(step_id);
                        Err(MillwrightError::GateTimeout {
                            step_id: step_id.to_owned(),
                            elapsed_ms: timeout.as_millis() as u64,
                        })
                    }
                }
            }
        }
    }

    /// Resolve a pending channel gate with a decision (called by Bell Tower).
    ///
    /// Returns `true` if the gate was pending, `false` if it wasn't found
    /// (already timed out or never created).
    pub fn resolve(&self, step_id: &str, decision: GateDecision) -> bool {
        if let Some(tx) = self.pending.lock().remove(step_id) {
            tx.send(decision).is_ok()
        } else {
            false
        }
    }

    /// Number of currently pending channel approvals.
    pub fn pending_count(&self) -> usize {
        self.pending.lock().len()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn evaluator() -> Arc<GateEvaluator> {
        Arc::new(GateEvaluator::new(Duration::from_millis(100)))
    }

    #[tokio::test]
    async fn always_open_passes() {
        let ev = evaluator();
        let d = ev
            .evaluate("step", &GateType::AlwaysOpen, &json!({}), None)
            .await
            .unwrap();
        assert_eq!(d, GateDecision::Open);
    }

    #[tokio::test]
    async fn always_closed_rejects() {
        let ev = evaluator();
        let d = ev
            .evaluate(
                "step",
                &GateType::AlwaysClosed {
                    reason: "nope".into(),
                },
                &json!({}),
                None,
            )
            .await
            .unwrap();
        assert!(matches!(d, GateDecision::Closed { .. }));
    }

    #[tokio::test]
    async fn auto_always_true_predicate_opens() {
        let ev = evaluator();
        let d = ev
            .evaluate(
                "step",
                &GateType::Auto {
                    predicate: "always_true".into(),
                },
                &json!({}),
                None,
            )
            .await
            .unwrap();
        assert_eq!(d, GateDecision::Open);
    }

    #[tokio::test]
    async fn auto_always_false_predicate_closes() {
        let ev = evaluator();
        let d = ev
            .evaluate(
                "step",
                &GateType::Auto {
                    predicate: "always_false".into(),
                },
                &json!({}),
                None,
            )
            .await
            .unwrap();
        assert!(matches!(d, GateDecision::Closed { .. }));
    }

    #[tokio::test]
    async fn has_text_predicate_with_text_opens() {
        let ev = evaluator();
        let d = ev
            .evaluate(
                "step",
                &GateType::Auto {
                    predicate: "has_text".into(),
                },
                &json!({ "text": "hello" }),
                None,
            )
            .await
            .unwrap();
        assert_eq!(d, GateDecision::Open);
    }

    #[tokio::test]
    async fn has_text_predicate_without_text_closes() {
        let ev = evaluator();
        let d = ev
            .evaluate(
                "step",
                &GateType::Auto {
                    predicate: "has_text".into(),
                },
                &json!({}),
                None,
            )
            .await
            .unwrap();
        assert!(matches!(d, GateDecision::Closed { .. }));
    }

    #[tokio::test]
    async fn channel_gate_resolved_opens() {
        let ev = evaluator();
        let ev2 = Arc::clone(&ev);
        // Resolve the gate from another task.
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(10)).await;
            ev2.resolve("step-approve", GateDecision::Open);
        });
        let d = ev
            .evaluate(
                "step-approve",
                &GateType::Channel {
                    channel: "slack:#ops".into(),
                },
                &json!({}),
                None,
            )
            .await
            .unwrap();
        assert_eq!(d, GateDecision::Open);
    }

    #[tokio::test]
    async fn channel_gate_timeout_returns_error() {
        let ev = evaluator(); // timeout = 100ms
        let result = ev
            .evaluate(
                "step-timeout",
                &GateType::Channel {
                    channel: "slack:#ops".into(),
                },
                &json!({}),
                Some(Duration::from_millis(10)), // very short for test
            )
            .await;
        assert!(matches!(result, Err(MillwrightError::GateTimeout { .. })));
    }

    #[tokio::test]
    async fn unknown_predicate_defaults_to_open() {
        let ev = evaluator();
        let d = ev
            .evaluate(
                "step",
                &GateType::Auto {
                    predicate: "nonexistent".into(),
                },
                &json!({}),
                None,
            )
            .await
            .unwrap();
        assert_eq!(d, GateDecision::Open);
    }

    #[test]
    fn resolve_non_pending_returns_false() {
        let ev = GateEvaluator::new(Duration::from_secs(5));
        assert!(!ev.resolve("ghost", GateDecision::Open));
    }

    #[tokio::test]
    async fn pending_count_tracks_waiting_gates() {
        let ev = evaluator();
        let ev2 = Arc::clone(&ev);
        assert_eq!(ev.pending_count(), 0);

        let _handle = tokio::spawn(async move {
            ev2.evaluate(
                "step-pending",
                &GateType::Channel {
                    channel: "slack:#test".into(),
                },
                &json!({}),
                Some(Duration::from_secs(5)),
            )
            .await
        });

        // Give the spawn a moment to register the pending gate.
        tokio::time::sleep(Duration::from_millis(5)).await;
        assert_eq!(ev.pending_count(), 1);
        ev.resolve(
            "step-pending",
            GateDecision::Closed {
                reason: "test".into(),
            },
        );
    }
}
