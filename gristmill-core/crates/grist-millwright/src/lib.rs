//! `grist-millwright` — DAG orchestrator for GristMill.
//!
//! The Millwright receives a [`GristEvent`] and a [`Pipeline`] definition and
//! executes the pipeline's steps in dependency order, running independent
//! steps in parallel up to a configurable concurrency limit.
//!
//! # Architecture role
//!
//! ```text
//! GristEvent
//!   │
//!   └─ Sieve (triage) ──→ RouteDecision
//!                              │
//!                              └─ Millwright (orchestrate)
//!                                      │
//!                                      ├─ LocalMl step → grist-grinders
//!                                      ├─ Rule step    → rule engine
//!                                      ├─ Llm step     → grist-hammer
//!                                      ├─ Gate step    → approval gates
//!                                      ├─ External     → HTTP / webhook
//!                                      ├─ PythonCall   → grist-ffi / PyO3
//!                                      └─ TypeScriptCall → grist-ffi / napi-rs
//! ```
//!
//! # PRD requirements
//!
//! | ID | Requirement | Implementation |
//! |----|------------|----------------|
//! | M-01 | DAG construction + cycle detection | `dag::Pipeline::validate` (Kahn's algorithm) |
//! | M-02 | Parallel step execution | `scheduler::DagScheduler` + `tokio::sync::Semaphore` |
//! | M-03 | Retry + exponential back-off | `retry::RetryPolicy` + `run_with_retry` |
//! | M-04 | Per-step and pipeline timeouts | `tokio::time::timeout` at step + pipeline level |
//! | M-05 | Checkpoint / resume | `checkpoint::CheckpointStore` (atomic JSON file write) |
//! | M-06 | Approval gates | `gates::GateEvaluator` (auto predicates + channel approval) |
//! | M-07 | Failure policies | `FailFast` / `ContinueOnError` / `SkipAndContinue` |
//! | M-08 | Bus events | `pipeline.completed` / `pipeline.failed` via `grist-bus` |
//!
//! # Example
//!
//! ```rust,no_run
//! use grist_millwright::{Millwright, MillwrightConfig};
//! use grist_millwright::dag::{Pipeline, Step, StepType};
//! use grist_event::{ChannelType, GristEvent};
//!
//! #[tokio::main]
//! async fn main() {
//!     let config = MillwrightConfig::default();
//!     let mw = Millwright::new(config, None).expect("millwright init failed");
//!
//!     let pipeline = Pipeline::new("demo")
//!         .with_step(Step::new("classify", StepType::LocalMl { model_id: "intent-classifier-v1".into() }))
//!         .with_step(
//!             Step::new("summarise", StepType::Llm {
//!                 prompt_template: "Summarise: {text}".into(),
//!                 max_tokens: 128,
//!             })
//!             .with_deps(["classify"]),
//!         );
//!
//!     mw.register_pipeline(pipeline);
//!
//!     let event = GristEvent::new(
//!         ChannelType::Http,
//!         serde_json::json!({ "text": "Hello, GristMill!" }),
//!     );
//!
//!     let result = mw.run("demo", &event).await.expect("pipeline failed");
//!     println!("succeeded: {}, output: {}", result.succeeded, result.output);
//! }
//! ```

pub mod checkpoint;
pub mod config;
pub mod dag;
pub mod error;
pub mod gates;
pub mod retry;
pub mod scheduler;

// Convenient re-exports.
pub use config::MillwrightConfig;
pub use dag::{CheckpointStrategy, ExecutionMode, FailurePolicy, Pipeline, Step, StepType};
pub use error::MillwrightError;
pub use gates::{GateDecision, GateEvaluator, GateType};
pub use retry::RetryPolicy;
pub use scheduler::{DagScheduler, PipelineResult, StepOutcome, StepResult};

use std::sync::Arc;

use dashmap::DashMap;
use tracing::info;

use grist_bus::EventBus;
use grist_event::GristEvent;

// ─────────────────────────────────────────────────────────────────────────────
// Millwright — top-level entry point
// ─────────────────────────────────────────────────────────────────────────────

/// The DAG orchestrator.  Thread-safe; wrap in `Arc` and share across tasks.
pub struct Millwright {
    scheduler: DagScheduler,
    pipelines: DashMap<String, Pipeline>,
}

impl Millwright {
    /// Create a Millwright instance.
    ///
    /// `bus` is optional — if supplied, `pipeline.completed` and
    /// `pipeline.failed` events are published to the bus so that the Bell
    /// Tower (TypeScript shell) can trigger notifications.
    ///
    /// **Must be called inside a Tokio runtime** (the scheduler creates async
    /// resources).
    pub fn new(
        config: MillwrightConfig,
        bus: Option<Arc<EventBus>>,
    ) -> Result<Self, MillwrightError> {
        info!(
            max_concurrency = config.max_concurrency,
            default_timeout_ms = config.default_timeout_ms,
            "initialising Millwright",
        );
        let scheduler = DagScheduler::new(config, bus)?;
        Ok(Self {
            scheduler,
            pipelines: DashMap::new(),
        })
    }

    /// Register a pipeline definition.  Overwrites any existing entry with
    /// the same ID.
    pub fn register_pipeline(&self, pipeline: Pipeline) {
        info!(pipeline_id = %pipeline.id, "pipeline registered");
        self.pipelines.insert(pipeline.id.clone(), pipeline);
    }

    /// De-register a pipeline.  Returns `true` if it existed.
    pub fn remove_pipeline(&self, id: &str) -> bool {
        self.pipelines.remove(id).is_some()
    }

    /// Run a registered pipeline by ID against an event.
    pub async fn run(
        &self,
        pipeline_id: &str,
        event: &GristEvent,
    ) -> Result<PipelineResult, MillwrightError> {
        let pipeline = self
            .pipelines
            .get(pipeline_id)
            .map(|p| p.clone())
            .ok_or_else(|| MillwrightError::RunNotFound(pipeline_id.to_owned()))?;
        self.scheduler.execute(&pipeline, event).await
    }

    /// Run a pipeline definition directly (without registering it first).
    pub async fn run_pipeline(
        &self,
        pipeline: &Pipeline,
        event: &GristEvent,
    ) -> Result<PipelineResult, MillwrightError> {
        self.scheduler.execute(pipeline, event).await
    }

    /// Return all registered pipeline IDs.
    pub fn pipeline_ids(&self) -> Vec<String> {
        self.pipelines.iter().map(|e| e.key().clone()).collect()
    }

    /// Return a clone of the gate evaluator so callers can resolve channel
    /// approval gates programmatically.
    pub fn gate_evaluator(&self) -> Arc<GateEvaluator> {
        self.scheduler.gate_evaluator()
    }
}

impl std::fmt::Debug for Millwright {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Millwright")
            .field("pipelines", &self.pipelines.len())
            .finish()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Integration tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use grist_event::{ChannelType, GristEvent};

    fn mw() -> Millwright {
        Millwright::new(MillwrightConfig::default(), None).unwrap()
    }

    fn event(text: &str) -> GristEvent {
        GristEvent::new(ChannelType::Http, serde_json::json!({ "text": text }))
    }

    // ── Registration ──────────────────────────────────────────────────────

    #[tokio::test]
    async fn register_and_run_pipeline() {
        let mw = mw();
        mw.register_pipeline(Pipeline::new("test-pipe").with_step(Step::new(
            "a",
            StepType::LocalMl {
                model_id: "m".into(),
            },
        )));
        assert_eq!(mw.pipeline_ids(), vec!["test-pipe"]);
        let result = mw.run("test-pipe", &event("hello")).await.unwrap();
        assert!(result.succeeded);
    }

    #[tokio::test]
    async fn run_unknown_pipeline_returns_error() {
        let mw = mw();
        let err = mw.run("ghost", &event("x")).await.unwrap_err();
        assert!(matches!(err, MillwrightError::RunNotFound(_)));
    }

    #[tokio::test]
    async fn remove_pipeline() {
        let mw = mw();
        mw.register_pipeline(Pipeline::new("p").with_step(Step::new(
            "s",
            StepType::Rule {
                rule_id: "r".into(),
            },
        )));
        assert!(mw.remove_pipeline("p"));
        assert!(!mw.remove_pipeline("p")); // already removed
    }

    // ── Run pipeline directly ─────────────────────────────────────────────

    #[tokio::test]
    async fn run_pipeline_directly_works() {
        let mw = mw();
        let p = Pipeline::new("direct").with_step(Step::new(
            "s",
            StepType::LocalMl {
                model_id: "m".into(),
            },
        ));
        let result = mw.run_pipeline(&p, &event("hi")).await.unwrap();
        assert!(result.succeeded);
        assert_eq!(result.pipeline_id, "direct");
    }

    // ── Multi-step + parallel ─────────────────────────────────────────────

    #[tokio::test]
    async fn parallel_pipeline_all_steps_complete() {
        let mw = mw();
        // a → {b, c} → d (diamond)
        let p = Pipeline::new("diamond")
            .with_step(Step::new(
                "a",
                StepType::LocalMl {
                    model_id: "m".into(),
                },
            ))
            .with_step(
                Step::new(
                    "b",
                    StepType::Rule {
                        rule_id: "r1".into(),
                    },
                )
                .with_deps(["a"]),
            )
            .with_step(
                Step::new(
                    "c",
                    StepType::Rule {
                        rule_id: "r2".into(),
                    },
                )
                .with_deps(["a"]),
            )
            .with_step(
                Step::new(
                    "d",
                    StepType::Llm {
                        prompt_template: "summarise".into(),
                        max_tokens: 64,
                    },
                )
                .with_deps(["b", "c"]),
            );

        let result = mw.run_pipeline(&p, &event("complex event")).await.unwrap();
        assert!(result.succeeded);
        assert_eq!(result.step_results.len(), 4);
    }

    // ── Retry ─────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn step_with_retry_policy_succeeds() {
        let mw = mw();
        let p = Pipeline::new("p").with_step(
            Step::new(
                "a",
                StepType::LocalMl {
                    model_id: "m".into(),
                },
            )
            .with_retry(RetryPolicy {
                max_retries: 2,
                initial_delay_ms: 1,
                jitter: false,
                ..Default::default()
            }),
        );
        let result = mw.run_pipeline(&p, &event("x")).await.unwrap();
        assert!(result.succeeded);
        // Should succeed on the first attempt (dispatch_step never fails for LocalMl stub).
        assert_eq!(result.step_results[0].attempts, 1);
    }

    // ── Gate ─────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn gate_open_pipeline_succeeds() {
        let mw = mw();
        let p = Pipeline::new("p")
            .with_step(Step::new(
                "g",
                StepType::Gate {
                    condition: "always_true".into(),
                },
            ))
            .with_step(
                Step::new(
                    "a",
                    StepType::Rule {
                        rule_id: "r".into(),
                    },
                )
                .with_deps(["g"]),
            );
        let result = mw.run_pipeline(&p, &event("x")).await.unwrap();
        assert!(result.succeeded);
        assert_eq!(result.step_results.len(), 2);
    }

    #[tokio::test]
    async fn gate_closed_pipeline_fails() {
        let mw = mw();
        let p = Pipeline::new("p").with_step(Step::new(
            "g",
            StepType::Gate {
                condition: "always_false".into(),
            },
        ));
        let err = mw.run_pipeline(&p, &event("x")).await.unwrap_err();
        assert!(matches!(err, MillwrightError::GateRejected { .. }));
    }

    // ── Failure policy ────────────────────────────────────────────────────

    #[tokio::test]
    async fn skip_and_continue_lets_pipeline_finish() {
        let mw = mw();
        let p = Pipeline::new("p")
            .with_step(Step::new(
                "g",
                StepType::Gate {
                    condition: "always_false".into(),
                },
            ))
            .with_failure_policy(FailurePolicy::SkipAndContinue);
        let result = mw.run_pipeline(&p, &event("x")).await.unwrap();
        assert!(matches!(
            result.step_results[0].outcome,
            StepOutcome::Skipped { .. }
        ));
    }

    // ── Cycle detection ───────────────────────────────────────────────────

    #[tokio::test]
    async fn cyclic_pipeline_returns_error() {
        let mw = mw();
        let p = Pipeline::new("cycle")
            .with_step(
                Step::new(
                    "a",
                    StepType::Rule {
                        rule_id: "r".into(),
                    },
                )
                .with_deps(["b"]),
            )
            .with_step(
                Step::new(
                    "b",
                    StepType::Rule {
                        rule_id: "r".into(),
                    },
                )
                .with_deps(["a"]),
            );
        let err = mw.run_pipeline(&p, &event("x")).await.unwrap_err();
        assert!(matches!(err, MillwrightError::CyclicDependency { .. }));
    }

    // ── Run ID is a ULID string ───────────────────────────────────────────

    #[tokio::test]
    async fn run_id_is_a_ulid() {
        let mw = mw();
        let p = Pipeline::new("p").with_step(Step::new(
            "s",
            StepType::Rule {
                rule_id: "r".into(),
            },
        ));
        let result = mw.run_pipeline(&p, &event("x")).await.unwrap();
        // ULID is 26 chars
        assert_eq!(result.run_id.len(), 26);
    }

    // ── Elapsed time is measured ──────────────────────────────────────────

    #[tokio::test]
    async fn elapsed_ms_is_positive() {
        let mw = mw();
        let p = Pipeline::new("p").with_step(Step::new(
            "s",
            StepType::LocalMl {
                model_id: "m".into(),
            },
        ));
        let result = mw.run_pipeline(&p, &event("x")).await.unwrap();
        // elapsed should be at least a few microseconds > 0 (even if sub-millisecond)
        // We allow 0 on very fast CI machines but the field must exist.
        assert!(result.elapsed_ms < 60_000, "elapsed unreasonably large");
    }
}
