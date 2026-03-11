//! DAG scheduler — parallel pipeline execution engine.
//!
//! The [`DagScheduler`] executes a [`Pipeline`] given an input [`GristEvent`].
//! It runs all steps whose dependencies have completed in parallel, bounded
//! by a [`tokio::sync::Semaphore`].
//!
//! # Execution loop
//!
//! ```text
//! 1. Validate the pipeline (topo-sort, cycle detection).
//! 2. Find all root steps (no deps) → enqueue.
//! 3. For each ready step:
//!    a. Acquire semaphore permit (concurrency limit).
//!    b. If step has an approval gate → evaluate gate.
//!    c. Execute step with retry policy and per-step timeout.
//!    d. Record result in `completed` map.
//!    e. Discover newly-unblocked steps → enqueue.
//! 4. Repeat until all steps are done or a failure policy triggers.
//! 5. Publish `pipeline.completed` / `pipeline.failed` to grist-bus.
//! 6. Write checkpoint if strategy requires it.
//! ```

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::Semaphore;
use tracing::{debug, info, instrument, warn};
use ulid::Ulid;

use grist_bus::EventBus;
use grist_event::GristEvent;

use crate::checkpoint::{CheckpointStore, RunCheckpoint, RunStatus};
use crate::config::MillwrightConfig;
use crate::dag::{FailurePolicy, Pipeline, Step, StepType};
use crate::error::MillwrightError;
use crate::gates::{GateDecision, GateEvaluator, GateType};
use crate::retry::{run_with_retry, RetryPolicy};

// ─────────────────────────────────────────────────────────────────────────────
// StepOutcome + StepResult
// ─────────────────────────────────────────────────────────────────────────────

/// How a step concluded.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StepOutcome {
    /// Step produced a result.
    Succeeded,
    /// Step was skipped (gate rejected, FailurePolicy::SkipAndContinue).
    Skipped { reason: String },
    /// Step failed after exhausting its retry budget.
    Failed { reason: String },
}

/// The recorded outcome of a single step execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepResult {
    pub step_id: String,
    pub outcome: StepOutcome,
    /// JSON output produced by the step (model output, rule match, etc.).
    pub output: Value,
    /// Total attempts made (1 = no retries).
    pub attempts: u32,
    /// Wall-clock time spent in this step (ms).
    pub elapsed_ms: u64,
}

// ─────────────────────────────────────────────────────────────────────────────
// PipelineResult
// ─────────────────────────────────────────────────────────────────────────────

/// Result of running an entire pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineResult {
    /// Unique run ID (ULID).
    pub run_id: String,
    /// Pipeline that was executed.
    pub pipeline_id: String,
    /// Per-step results in completion order.
    pub step_results: Vec<StepResult>,
    /// Aggregated output (last step's output by default).
    pub output: Value,
    /// True iff all steps succeeded or were skipped.
    pub succeeded: bool,
    /// Total wall-clock time (ms).
    pub elapsed_ms: u64,
}

// ─────────────────────────────────────────────────────────────────────────────
// DagScheduler
// ─────────────────────────────────────────────────────────────────────────────

/// Parallel DAG execution engine.
///
/// `DagScheduler` is `Send + Sync`.  Wrap in `Arc` and share across tasks.
pub struct DagScheduler {
    config: MillwrightConfig,
    /// Global concurrency semaphore — reserved for cross-pipeline limiting.
    /// Per-pipeline concurrency is managed by a local semaphore in `run_dag`.
    #[allow(dead_code)]
    semaphore: Arc<Semaphore>,
    gate_evaluator: Arc<GateEvaluator>,
    checkpoint_store: CheckpointStore,
    bus: Option<Arc<EventBus>>,
}

impl DagScheduler {
    /// Create a scheduler from config, optionally connected to a bus.
    pub fn new(
        config: MillwrightConfig,
        bus: Option<Arc<EventBus>>,
    ) -> Result<Self, MillwrightError> {
        let concurrency = config.max_concurrency.max(1);
        let checkpoint_store = CheckpointStore::open(&config.checkpoint_dir)?;
        let gate_evaluator = Arc::new(GateEvaluator::new(Duration::from_secs(300)));

        Ok(Self {
            semaphore: Arc::new(Semaphore::new(concurrency)),
            gate_evaluator,
            checkpoint_store,
            bus,
            config,
        })
    }

    /// Create a scheduler with an in-memory (noop) checkpoint store.
    /// Used in tests and dry-run contexts.
    pub fn new_noop(config: MillwrightConfig) -> Self {
        let concurrency = config.max_concurrency.max(1);
        Self {
            semaphore: Arc::new(Semaphore::new(concurrency)),
            gate_evaluator: Arc::new(GateEvaluator::new(Duration::from_secs(300))),
            checkpoint_store: CheckpointStore::noop(),
            bus: None,
            config,
        }
    }

    /// Expose the gate evaluator so callers can resolve channel gates.
    pub fn gate_evaluator(&self) -> Arc<GateEvaluator> {
        Arc::clone(&self.gate_evaluator)
    }

    /// Execute a pipeline with the given input event.
    ///
    /// Returns a [`PipelineResult`] describing the outcome of every step.
    #[instrument(level = "info", skip(self, pipeline, input),
                 fields(pipeline_id = %pipeline.id))]
    pub async fn execute(
        &self,
        pipeline: &Pipeline,
        input: &GristEvent,
    ) -> Result<PipelineResult, MillwrightError> {
        let run_id = Ulid::new().to_string();
        let t_start = Instant::now();

        info!(run_id, pipeline_id = %pipeline.id, "pipeline started");
        metrics::counter!("millwright.pipeline.started", "pipeline_id" => pipeline.id.clone())
            .increment(1);

        // 1. Validate + topo-sort.
        pipeline.validate()?;

        // 2. Resolve pipeline timeout.
        let pipeline_timeout = {
            let ms = if pipeline.timeout_ms > 0 {
                pipeline.timeout_ms
            } else {
                self.config.default_timeout_ms
            };
            Duration::from_millis(ms)
        };

        // 3. Resolve concurrency.
        let concurrency = if pipeline.max_concurrency > 0 {
            pipeline.max_concurrency
        } else {
            self.config.max_concurrency.max(1)
        };

        // 4. Initialise run state.
        let completed: Arc<DashMap<String, StepResult>> = Arc::new(DashMap::new());
        let mut checkpoint = RunCheckpoint {
            run_id: run_id.clone(),
            pipeline_id: pipeline.id.clone(),
            started_at_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0),
            completed_steps: HashMap::new(),
            status: RunStatus::Running,
        };

        // 5. Execute the DAG.
        let result = tokio::time::timeout(
            pipeline_timeout,
            self.run_dag(pipeline, input, concurrency, Arc::clone(&completed)),
        )
        .await;

        let succeeded = match result {
            Ok(Ok(())) => true,
            Ok(Err(e)) => {
                warn!(run_id, error = %e, "pipeline failed");
                checkpoint.status = RunStatus::Failed;
                self.save_checkpoint(&mut checkpoint, &completed);
                self.publish_failed(&run_id, &pipeline.id, &e.to_string());
                metrics::counter!("millwright.pipeline.failed", "pipeline_id" => pipeline.id.clone())
                    .increment(1);
                return Err(e);
            }
            Err(_) => {
                let elapsed_ms = t_start.elapsed().as_millis() as u64;
                checkpoint.status = RunStatus::Failed;
                self.save_checkpoint(&mut checkpoint, &completed);
                self.publish_failed(&run_id, &pipeline.id, "pipeline timed out");
                metrics::counter!("millwright.pipeline.timeout", "pipeline_id" => pipeline.id.clone())
                    .increment(1);
                return Err(MillwrightError::PipelineTimeout {
                    pipeline_id: pipeline.id.clone(),
                    elapsed_ms,
                    timeout_ms: pipeline_timeout.as_millis() as u64,
                });
            }
        };

        // 6. Collect results in topo order.
        let topo = pipeline.validate()?;
        let mut step_results: Vec<StepResult> = topo
            .iter()
            .filter_map(|id| completed.get(id).map(|r| r.clone()))
            .collect();

        // Collect any steps not in topo (shouldn't happen, but be safe).
        for entry in completed.iter() {
            if !step_results.iter().any(|r| &r.step_id == entry.key()) {
                step_results.push(entry.value().clone());
            }
        }

        let output = step_results
            .last()
            .map(|r| r.output.clone())
            .unwrap_or(Value::Null);

        let elapsed_ms = t_start.elapsed().as_millis() as u64;

        // 7. Persist checkpoint.
        checkpoint.status = RunStatus::Completed;
        self.save_checkpoint(&mut checkpoint, &completed);

        // 8. Publish success event.
        self.publish_completed(&run_id, &pipeline.id);

        metrics::counter!("millwright.pipeline.completed", "pipeline_id" => pipeline.id.clone())
            .increment(1);
        metrics::histogram!("millwright.pipeline.duration_ms", "pipeline_id" => pipeline.id.clone())
            .record(elapsed_ms as f64);

        info!(run_id, pipeline_id = %pipeline.id, elapsed_ms, "pipeline completed");

        Ok(PipelineResult {
            run_id,
            pipeline_id: pipeline.id.clone(),
            step_results,
            output,
            succeeded,
            elapsed_ms,
        })
    }

    // ── Core DAG execution loop ───────────────────────────────────────────

    async fn run_dag(
        &self,
        pipeline: &Pipeline,
        input: &GristEvent,
        concurrency: usize,
        completed: Arc<DashMap<String, StepResult>>,
    ) -> Result<(), MillwrightError> {
        // Use a local semaphore for this pipeline's concurrency budget.
        let sem = Arc::new(Semaphore::new(concurrency));
        let total_steps = pipeline.steps.len();
        let mut errors: Vec<MillwrightError> = Vec::new();

        // `JoinSet` lets us await the first completed task without polling.
        let mut join_set: tokio::task::JoinSet<Result<StepResult, MillwrightError>> =
            tokio::task::JoinSet::new();

        // Track which steps have been enqueued to avoid double-spawning.
        let mut enqueued: HashSet<String> = HashSet::new();

        // ── Inner helper: enqueue all currently-ready steps ───────────────
        let enqueue_ready = |enqueued: &mut HashSet<String>,
                             join_set: &mut tokio::task::JoinSet<_>,
                             completed: &DashMap<String, StepResult>| {
            let completed_ids: HashSet<String> =
                completed.iter().map(|e| e.key().clone()).collect();

            for step in pipeline.ready_steps(&completed_ids) {
                if enqueued.contains(&step.id) {
                    continue;
                }
                enqueued.insert(step.id.clone());

                let step = step.clone();
                let input = input.clone();
                let sem = Arc::clone(&sem);
                let gate_ev = Arc::clone(&self.gate_evaluator);
                let failure_policy = pipeline.on_failure;

                join_set.spawn(async move {
                    let _permit = sem.acquire_owned().await.map_err(|e| {
                        MillwrightError::Other(anyhow::anyhow!("semaphore closed: {e}"))
                    })?;
                    execute_step(&step, &input, &gate_ev, failure_policy).await
                });
            }
        };

        // ── Seed: enqueue root steps (no deps) ────────────────────────────
        enqueue_ready(&mut enqueued, &mut join_set, &completed);

        // ── Event loop: wait for one step, enqueue newly-ready steps ─────
        loop {
            if completed.len() == total_steps {
                break;
            }

            // If no tasks are running we've hit an impossible deadlock
            // (should never occur after successful validation).
            if join_set.is_empty() {
                let missing: Vec<String> = pipeline
                    .steps
                    .iter()
                    .filter(|s| !completed.contains_key(&s.id))
                    .map(|s| s.id.clone())
                    .collect();
                return Err(MillwrightError::InvalidPipeline {
                    pipeline_id: pipeline.id.clone(),
                    reason: format!("deadlock: steps never became ready: {missing:?}"),
                });
            }

            // Wait for the next task to finish.
            match join_set.join_next().await {
                Some(Ok(Ok(result))) => {
                    debug!(step_id = %result.step_id, outcome = ?result.outcome, "step done");
                    completed.insert(result.step_id.clone(), result);
                    // Newly completed step may have unblocked dependents.
                    enqueue_ready(&mut enqueued, &mut join_set, &completed);
                }
                Some(Ok(Err(e))) => match pipeline.on_failure {
                    FailurePolicy::FailFast => return Err(e),
                    _ => {
                        warn!(error = %e, "step failed; continuing per failure policy");
                        errors.push(e);
                        enqueue_ready(&mut enqueued, &mut join_set, &completed);
                    }
                },
                Some(Err(join_err)) => {
                    let e = MillwrightError::Other(anyhow::anyhow!("task panicked: {join_err}"));
                    match pipeline.on_failure {
                        FailurePolicy::FailFast => return Err(e),
                        _ => errors.push(e),
                    }
                    enqueue_ready(&mut enqueued, &mut join_set, &completed);
                }
                None => break, // join_set drained
            }
        }

        if !errors.is_empty() && matches!(pipeline.on_failure, FailurePolicy::ContinueOnError) {
            return Err(errors.remove(0));
        }

        Ok(())
    }

    // ── Checkpoint helpers ────────────────────────────────────────────────

    fn save_checkpoint(
        &self,
        checkpoint: &mut RunCheckpoint,
        completed: &DashMap<String, StepResult>,
    ) {
        checkpoint.completed_steps = completed
            .iter()
            .map(|e| (e.key().clone(), e.value().clone()))
            .collect();
        if let Err(e) = self.checkpoint_store.save(checkpoint) {
            warn!(run_id = %checkpoint.run_id, error = %e, "checkpoint save failed");
        }
    }

    // ── Bus publish helpers ───────────────────────────────────────────────

    fn publish_completed(&self, run_id: &str, pipeline_id: &str) {
        if let Some(bus) = &self.bus {
            let payload = serde_json::json!({
                "run_id": run_id,
                "pipeline_id": pipeline_id,
                "status": "completed",
            });
            bus.publish("pipeline.completed", payload);
        }
    }

    fn publish_failed(&self, run_id: &str, pipeline_id: &str, reason: &str) {
        if let Some(bus) = &self.bus {
            let payload = serde_json::json!({
                "run_id": run_id,
                "pipeline_id": pipeline_id,
                "status": "failed",
                "reason": reason,
            });
            bus.publish("pipeline.failed", payload);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Step executor
// ─────────────────────────────────────────────────────────────────────────────

/// Execute a single step (with gate, retry, and timeout).
async fn execute_step(
    step: &Step,
    input: &GristEvent,
    gate_evaluator: &Arc<GateEvaluator>,
    failure_policy: FailurePolicy,
) -> Result<StepResult, MillwrightError> {
    let t0 = Instant::now();

    debug!(step_id = %step.id, kind = step.step_type.kind_label(), "executing step");

    // ── 1. Approval gate (if required) ────────────────────────────────────
    if step.requires_approval {
        let channel = step.approval_channel.as_deref().unwrap_or("default");
        let context = serde_json::json!({ "step_id": step.id, "input": input.payload });
        let timeout = step.timeout_ms.map(Duration::from_millis);

        match gate_evaluator
            .evaluate(
                &step.id,
                &GateType::Channel {
                    channel: channel.to_owned(),
                },
                &context,
                timeout,
            )
            .await?
        {
            GateDecision::Open => {}
            GateDecision::Closed { reason } => {
                return match failure_policy {
                    FailurePolicy::SkipAndContinue => Ok(StepResult {
                        step_id: step.id.clone(),
                        outcome: StepOutcome::Skipped {
                            reason: reason.clone(),
                        },
                        output: Value::Null,
                        attempts: 0,
                        elapsed_ms: t0.elapsed().as_millis() as u64,
                    }),
                    _ => Err(MillwrightError::GateRejected {
                        step_id: step.id.clone(),
                        reason,
                    }),
                };
            }
        }
    }

    // ── 2. Gate step type ─────────────────────────────────────────────────
    if let StepType::Gate { condition } = &step.step_type {
        let context = serde_json::json!({ "input": input.payload });
        let timeout = step.timeout_ms.map(Duration::from_millis);
        let decision = gate_evaluator
            .evaluate(
                &step.id,
                &GateType::Auto {
                    predicate: condition.clone(),
                },
                &context,
                timeout,
            )
            .await?;

        return match decision {
            GateDecision::Open => Ok(StepResult {
                step_id: step.id.clone(),
                outcome: StepOutcome::Succeeded,
                output: serde_json::json!({ "gate": "open" }),
                attempts: 1,
                elapsed_ms: t0.elapsed().as_millis() as u64,
            }),
            GateDecision::Closed { reason } => match failure_policy {
                FailurePolicy::SkipAndContinue => Ok(StepResult {
                    step_id: step.id.clone(),
                    outcome: StepOutcome::Skipped {
                        reason: reason.clone(),
                    },
                    output: Value::Null,
                    attempts: 1,
                    elapsed_ms: t0.elapsed().as_millis() as u64,
                }),
                _ => Err(MillwrightError::GateRejected {
                    step_id: step.id.clone(),
                    reason,
                }),
            },
        };
    }

    // ── 3. Run step with retry + timeout ──────────────────────────────────
    let default_policy = RetryPolicy {
        max_retries: 0,
        ..Default::default()
    };
    let policy = step.retry.as_ref().unwrap_or(&default_policy);
    let step_timeout = step.timeout_ms.map(Duration::from_millis);

    let mut attempts = 0u32;

    let step_result = run_with_retry(policy, &step.id, || {
        attempts += 1;
        let input = input.clone();
        let step = step.clone();
        async move { run_step_once(&step, &input, step_timeout).await }
    })
    .await;

    match step_result {
        Ok(output) => {
            let elapsed_ms = t0.elapsed().as_millis() as u64;
            metrics::histogram!("millwright.step.duration_ms", "step_id" => step.id.clone())
                .record(elapsed_ms as f64);
            Ok(StepResult {
                step_id: step.id.clone(),
                outcome: StepOutcome::Succeeded,
                output,
                attempts,
                elapsed_ms,
            })
        }
        Err(e) => {
            let elapsed_ms = t0.elapsed().as_millis() as u64;
            let reason = e.to_string();
            metrics::counter!("millwright.step.failed", "step_id" => step.id.clone()).increment(1);
            match failure_policy {
                FailurePolicy::SkipAndContinue => Ok(StepResult {
                    step_id: step.id.clone(),
                    outcome: StepOutcome::Skipped { reason },
                    output: Value::Null,
                    attempts,
                    elapsed_ms,
                }),
                _ => Err(MillwrightError::StepFailed {
                    step_id: step.id.clone(),
                    attempts,
                    reason,
                }),
            }
        }
    }
}

/// Execute a step type exactly once, applying a per-step timeout.
async fn run_step_once(
    step: &Step,
    input: &GristEvent,
    timeout: Option<Duration>,
) -> Result<Value, MillwrightError> {
    let fut = dispatch_step(step, input);

    match timeout {
        Some(t) => {
            tokio::time::timeout(t, fut)
                .await
                .map_err(|_| MillwrightError::StepTimeout {
                    step_id: step.id.clone(),
                    elapsed_ms: t.as_millis() as u64,
                })?
        }
        None => fut.await,
    }
}

/// Dispatch to the appropriate step type handler.
///
/// In a full deployment this would call grist-grinders (LocalMl), grist-hammer
/// (Llm), or the FFI bridges (PythonCall / TypeScriptCall).  Here we provide
/// a simulation layer that returns structured JSON stubs so the orchestration
/// logic can be exercised without live model dependencies.
async fn dispatch_step(step: &Step, input: &GristEvent) -> Result<Value, MillwrightError> {
    metrics::counter!("millwright.step.dispatched", "kind" => step.step_type.kind_label())
        .increment(1);

    match &step.step_type {
        StepType::LocalMl { model_id } => {
            debug!(step_id = %step.id, model_id, "dispatching LocalMl step");
            // Stub: returns a classification label based on event text length.
            let text = input.payload_as_text();
            Ok(serde_json::json!({
                "step_id": step.id,
                "model_id": model_id,
                "result": if text.len() > 100 { "complex" } else { "simple" },
                "confidence": 0.92_f32,
            }))
        }

        StepType::Rule { rule_id } => {
            debug!(step_id = %step.id, rule_id, "dispatching Rule step");
            Ok(serde_json::json!({
                "step_id": step.id,
                "rule_id": rule_id,
                "matched": true,
            }))
        }

        StepType::Llm {
            prompt_template,
            max_tokens,
        } => {
            debug!(step_id = %step.id, max_tokens, "dispatching Llm step");
            // Stub — grist-hammer will be wired in when available.
            Ok(serde_json::json!({
                "step_id": step.id,
                "prompt_template": prompt_template,
                "max_tokens": max_tokens,
                "response": "[llm response stub]",
                "tokens_used": 42,
            }))
        }

        StepType::External { action, config } => {
            debug!(step_id = %step.id, action, "dispatching External step");
            Ok(serde_json::json!({
                "step_id": step.id,
                "action": action,
                "config": config,
                "status": "ok",
            }))
        }

        StepType::PythonCall { module, function } => {
            debug!(step_id = %step.id, module, function, "dispatching PythonCall step");
            // Stub — grist-ffi / PyO3 bridge will be wired in.
            Ok(serde_json::json!({
                "step_id": step.id,
                "module": module,
                "function": function,
                "result": null,
                "stub": true,
            }))
        }

        StepType::TypeScriptCall { adapter, method } => {
            debug!(step_id = %step.id, adapter, method, "dispatching TypeScriptCall step");
            // Stub — napi-rs bridge will be wired in.
            Ok(serde_json::json!({
                "step_id": step.id,
                "adapter": adapter,
                "method": method,
                "result": null,
                "stub": true,
            }))
        }

        // Gate is handled before this point in execute_step.
        StepType::Gate { .. } => unreachable!("Gate steps handled above"),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dag::{Pipeline, Step, StepType};
    use grist_event::{ChannelType, GristEvent};

    fn scheduler() -> DagScheduler {
        DagScheduler::new_noop(MillwrightConfig::default())
    }

    fn event(text: &str) -> GristEvent {
        GristEvent::new(
            ChannelType::Internal {
                subsystem: "test".into(),
            },
            serde_json::json!({ "text": text }),
        )
    }

    // ── Single-step pipelines ─────────────────────────────────────────────

    #[tokio::test]
    async fn single_local_ml_step_succeeds() {
        let p = Pipeline::new("p").with_step(Step::new(
            "a",
            StepType::LocalMl {
                model_id: "m".into(),
            },
        ));
        let result = scheduler().execute(&p, &event("hello")).await.unwrap();
        assert!(result.succeeded);
        assert_eq!(result.step_results.len(), 1);
        assert_eq!(result.step_results[0].step_id, "a");
        assert_eq!(result.step_results[0].outcome, StepOutcome::Succeeded);
    }

    #[tokio::test]
    async fn single_rule_step_succeeds() {
        let p = Pipeline::new("p").with_step(Step::new(
            "r",
            StepType::Rule {
                rule_id: "no-op".into(),
            },
        ));
        let result = scheduler().execute(&p, &event("cmd")).await.unwrap();
        assert!(result.succeeded);
    }

    #[tokio::test]
    async fn llm_step_returns_stub_output() {
        let p = Pipeline::new("p").with_step(Step::new(
            "llm",
            StepType::Llm {
                prompt_template: "Summarize: {text}".into(),
                max_tokens: 128,
            },
        ));
        let result = scheduler().execute(&p, &event("long text")).await.unwrap();
        assert!(result.succeeded);
        assert!(result.output["response"].is_string());
    }

    // ── Multi-step sequential chain ───────────────────────────────────────

    #[tokio::test]
    async fn sequential_chain_runs_in_order() {
        let p = Pipeline::new("chain")
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
                        rule_id: "r".into(),
                    },
                )
                .with_deps(["a"]),
            )
            .with_step(
                Step::new(
                    "c",
                    StepType::LocalMl {
                        model_id: "m2".into(),
                    },
                )
                .with_deps(["b"]),
            );

        let result = scheduler().execute(&p, &event("text")).await.unwrap();
        assert!(result.succeeded);
        assert_eq!(result.step_results.len(), 3);
    }

    // ── Diamond parallel DAG ─────────────────────────────────────────────

    #[tokio::test]
    async fn diamond_dag_runs_parallel_branches() {
        // a → {b, c} → d
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
                    StepType::LocalMl {
                        model_id: "m2".into(),
                    },
                )
                .with_deps(["b", "c"]),
            );

        let result = scheduler().execute(&p, &event("text")).await.unwrap();
        assert!(result.succeeded);
        assert_eq!(result.step_results.len(), 4);
    }

    // ── Gate step type ────────────────────────────────────────────────────

    #[tokio::test]
    async fn gate_step_always_true_passes() {
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
        let result = scheduler().execute(&p, &event("x")).await.unwrap();
        assert!(result.succeeded);
    }

    #[tokio::test]
    async fn gate_step_always_false_fails() {
        let p = Pipeline::new("p").with_step(Step::new(
            "g",
            StepType::Gate {
                condition: "always_false".into(),
            },
        ));
        let err = scheduler().execute(&p, &event("x")).await.unwrap_err();
        assert!(matches!(err, MillwrightError::GateRejected { .. }));
    }

    #[tokio::test]
    async fn gate_step_always_false_skips_with_policy() {
        let p = Pipeline::new("p")
            .with_step(Step::new(
                "g",
                StepType::Gate {
                    condition: "always_false".into(),
                },
            ))
            .with_failure_policy(FailurePolicy::SkipAndContinue);
        let result = scheduler().execute(&p, &event("x")).await.unwrap();
        // Pipeline succeeds, step is skipped
        assert!(matches!(
            result.step_results[0].outcome,
            StepOutcome::Skipped { .. }
        ));
    }

    // ── Timeout ───────────────────────────────────────────────────────────

    #[tokio::test]
    async fn pipeline_timeout_returns_error() {
        // Use a step with `requires_approval` — it will await a Channel gate
        // that is never resolved.  The 50ms pipeline-level timeout fires first
        // (GateEvaluator default is 30 s, so the pipeline timeout wins).
        let config = MillwrightConfig {
            default_timeout_ms: 50, // 50ms pipeline timeout
            ..MillwrightConfig::default()
        };
        let s = DagScheduler::new_noop(config);

        let blocking_step = Step::new(
            "blocking-approval",
            StepType::LocalMl {
                model_id: "m".into(),
            },
        )
        .with_approval("slack:#never-resolves");

        let p = Pipeline::new("timeout-test").with_step(blocking_step);
        let result = s.execute(&p, &event("x")).await;
        assert!(
            matches!(result, Err(MillwrightError::PipelineTimeout { .. })),
            "expected PipelineTimeout, got {result:?}",
        );
    }

    // ── FailFast: invalid pipeline ────────────────────────────────────────

    #[tokio::test]
    async fn empty_pipeline_returns_error() {
        let p = Pipeline::new("empty");
        let err = scheduler().execute(&p, &event("x")).await.unwrap_err();
        assert!(matches!(err, MillwrightError::EmptyPipeline(_)));
    }

    // ── Output is last step's output ─────────────────────────────────────

    #[tokio::test]
    async fn output_is_last_completed_step() {
        let p = Pipeline::new("p")
            .with_step(Step::new(
                "first",
                StepType::Rule {
                    rule_id: "r".into(),
                },
            ))
            .with_step(
                Step::new(
                    "last",
                    StepType::Llm {
                        prompt_template: "x".into(),
                        max_tokens: 10,
                    },
                )
                .with_deps(["first"]),
            );
        let result = scheduler().execute(&p, &event("x")).await.unwrap();
        // Output should come from "last" — the LLM step.
        assert!(result.output["response"].is_string());
    }

    // ── PythonCall / TypeScriptCall stubs ─────────────────────────────────

    #[tokio::test]
    async fn python_call_step_returns_stub() {
        let p = Pipeline::new("p").with_step(Step::new(
            "py",
            StepType::PythonCall {
                module: "my_module".into(),
                function: "run".into(),
            },
        ));
        let result = scheduler().execute(&p, &event("x")).await.unwrap();
        assert!(result.step_results[0].output["stub"]
            .as_bool()
            .unwrap_or(false));
    }

    #[tokio::test]
    async fn typescript_call_step_returns_stub() {
        let p = Pipeline::new("p").with_step(Step::new(
            "ts",
            StepType::TypeScriptCall {
                adapter: "slack".into(),
                method: "send".into(),
            },
        ));
        let result = scheduler().execute(&p, &event("x")).await.unwrap();
        assert!(result.step_results[0].output["stub"]
            .as_bool()
            .unwrap_or(false));
    }
}
