//! DAG data structures, construction, and topological validation.
//!
//! A [`Pipeline`] is a directed acyclic graph of [`Step`]s.  Each step
//! declares which other steps it depends on via `depends_on: Vec<String>`.
//! Steps with an empty `depends_on` are root nodes and are eligible to run
//! immediately.
//!
//! `dag.rs` owns:
//! - The [`Pipeline`], [`Step`], [`StepType`] data model
//! - Topological sort (`Pipeline::topo_sort`)
//! - Cycle detection
//! - Dependency resolution helpers used by the scheduler

use std::collections::{HashMap, HashSet, VecDeque};

use serde::{Deserialize, Serialize};

use crate::error::MillwrightError;
use crate::retry::RetryPolicy;

// ─────────────────────────────────────────────────────────────────────────────
// StepType
// ─────────────────────────────────────────────────────────────────────────────

/// Discriminates what kind of work a step performs.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum StepType {
    /// Run a local ONNX/GGUF model via grist-grinders.
    LocalMl { model_id: String },
    /// Apply a deterministic rule engine.
    Rule { rule_id: String },
    /// Call an LLM via grist-hammer.
    Llm {
        prompt_template: String,
        max_tokens: u32,
    },
    /// Invoke an external service / HTTP action.
    External {
        action: String,
        config: serde_json::Value,
    },
    /// Conditional gate — halts the pipeline if condition evaluates false
    /// or requires human approval.
    Gate { condition: String },
    /// Call into the Python ML shell.
    PythonCall { module: String, function: String },
    /// Call into the TypeScript integrations shell.
    TypeScriptCall { adapter: String, method: String },
}

impl StepType {
    /// Human-readable label for metrics and logging.
    pub fn kind_label(&self) -> &'static str {
        match self {
            StepType::LocalMl { .. }     => "local_ml",
            StepType::Rule { .. }        => "rule",
            StepType::Llm { .. }         => "llm",
            StepType::External { .. }    => "external",
            StepType::Gate { .. }        => "gate",
            StepType::PythonCall { .. }  => "python_call",
            StepType::TypeScriptCall { .. } => "typescript_call",
        }
    }

    /// Returns true if this step may involve LLM token expenditure.
    pub fn involves_llm(&self) -> bool {
        matches!(self, StepType::Llm { .. })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Step
// ─────────────────────────────────────────────────────────────────────────────

/// A single unit of work in a pipeline.
///
/// Steps are connected via `depends_on`.  The DAG scheduler runs all steps
/// whose dependencies have completed, in parallel up to `Pipeline::max_concurrency`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Step {
    /// Unique identifier within the pipeline (e.g. `"embed"`, `"classify"`).
    pub id: String,
    /// The kind of work this step performs.
    pub step_type: StepType,
    /// IDs of steps that must complete before this step may start.
    /// Empty → root node (runs immediately when the pipeline starts).
    #[serde(default)]
    pub depends_on: Vec<String>,
    /// Prefer running locally; only escalate to LLM if confidence is too low.
    /// Architecture invariant: always `true` by default.
    #[serde(default = "default_true")]
    pub prefer_local: bool,
    /// If local execution fails, automatically retry via the LLM gateway.
    #[serde(default)]
    pub fallback_to_llm: bool,
    /// If true, a human-approval gate is inserted before this step executes.
    #[serde(default)]
    pub requires_approval: bool,
    /// Where to route the approval request (e.g. `"slack:#ops-channel"`).
    #[serde(default)]
    pub approval_channel: Option<String>,
    /// Per-step timeout in milliseconds.  Overrides the pipeline default.
    #[serde(default)]
    pub timeout_ms: Option<u64>,
    /// Retry policy for this step.
    #[serde(default)]
    pub retry: Option<RetryPolicy>,
}

fn default_true() -> bool {
    true
}

impl Step {
    /// Convenience constructor.
    pub fn new(id: impl Into<String>, step_type: StepType) -> Self {
        Self {
            id: id.into(),
            step_type,
            depends_on: Vec::new(),
            prefer_local: true,
            fallback_to_llm: false,
            requires_approval: false,
            approval_channel: None,
            timeout_ms: None,
            retry: None,
        }
    }

    /// Builder: set dependencies.
    pub fn with_deps(mut self, deps: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.depends_on = deps.into_iter().map(|d| d.into()).collect();
        self
    }

    /// Builder: set per-step timeout.
    pub fn with_timeout_ms(mut self, ms: u64) -> Self {
        self.timeout_ms = Some(ms);
        self
    }

    /// Builder: add a retry policy.
    pub fn with_retry(mut self, policy: RetryPolicy) -> Self {
        self.retry = Some(policy);
        self
    }

    /// Builder: require approval before executing.
    pub fn with_approval(mut self, channel: impl Into<String>) -> Self {
        self.requires_approval = true;
        self.approval_channel = Some(channel.into());
        self
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Execution mode + failure policy
// ─────────────────────────────────────────────────────────────────────────────

/// How a pipeline should be executed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionMode {
    /// Prefer local models; escalate only when confidence < threshold.
    #[default]
    LocalFirst,
    /// Always call the LLM (testing / high-quality mode).
    LlmAlways,
    /// Dry-run: resolve the DAG but do not execute any steps.
    DryRun,
}

/// What to do when a step fails beyond its retry budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum FailurePolicy {
    /// Abort the pipeline immediately, cancel pending steps.
    #[default]
    FailFast,
    /// Continue executing independent steps; collect all errors at the end.
    ContinueOnError,
    /// Skip the failed step and continue with dependents as if it succeeded.
    SkipAndContinue,
}

/// Strategy for checkpointing a pipeline run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum CheckpointStrategy {
    /// No checkpointing (ephemeral runs).
    #[default]
    None,
    /// Write a checkpoint after each step completes.
    AfterEachStep,
    /// Write a checkpoint only when the pipeline completes (or fails).
    OnCompletion,
}

// ─────────────────────────────────────────────────────────────────────────────
// Pipeline
// ─────────────────────────────────────────────────────────────────────────────

/// A named, versioned pipeline definition.
///
/// Pipelines are stored in memory (registered at startup) and can also be
/// loaded from YAML/JSON config files.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pipeline {
    /// Unique pipeline identifier (e.g. `"content-digest"`, `"ticket-triage"`).
    pub id: String,
    /// Human-readable description.
    #[serde(default)]
    pub description: String,
    /// Ordered list of steps.  The scheduler discovers parallelism from
    /// `depends_on`; the order in this vec has no effect on execution.
    pub steps: Vec<Step>,
    /// Default execution mode.
    #[serde(default)]
    pub default_mode: ExecutionMode,
    /// Maximum parallel steps.  Zero means use the Millwright global limit.
    #[serde(default)]
    pub max_concurrency: usize,
    /// Checkpoint strategy for runs of this pipeline.
    #[serde(default)]
    pub checkpoint_strategy: CheckpointStrategy,
    /// Wall-clock timeout for the entire pipeline (ms).
    /// Zero means use the Millwright global default.
    #[serde(default)]
    pub timeout_ms: u64,
    /// What to do when a step fails.
    #[serde(default)]
    pub on_failure: FailurePolicy,
}

impl Pipeline {
    /// Construct a minimal pipeline with no steps yet.
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            description: String::new(),
            steps: Vec::new(),
            default_mode: ExecutionMode::LocalFirst,
            max_concurrency: 0,
            checkpoint_strategy: CheckpointStrategy::None,
            timeout_ms: 0,
            on_failure: FailurePolicy::FailFast,
        }
    }

    /// Builder: add a step.
    pub fn with_step(mut self, step: Step) -> Self {
        self.steps.push(step);
        self
    }

    /// Builder: set timeout.
    pub fn with_timeout_ms(mut self, ms: u64) -> Self {
        self.timeout_ms = ms;
        self
    }

    /// Builder: set failure policy.
    pub fn with_failure_policy(mut self, policy: FailurePolicy) -> Self {
        self.on_failure = policy;
        self
    }

    /// Validate the pipeline and return a topological ordering of step IDs.
    ///
    /// Returns `Err(CyclicDependency)` if a cycle is detected.
    /// Returns `Err(UnknownDependency)` if a `depends_on` references a
    /// step ID that doesn't exist.
    /// Returns `Err(EmptyPipeline)` if there are no steps.
    pub fn validate(&self) -> Result<Vec<String>, MillwrightError> {
        if self.steps.is_empty() {
            return Err(MillwrightError::EmptyPipeline(self.id.clone()));
        }

        // Index step IDs for O(1) existence check.
        let step_ids: HashSet<&str> = self.steps.iter().map(|s| s.id.as_str()).collect();

        // Validate all `depends_on` references.
        for step in &self.steps {
            for dep in &step.depends_on {
                if !step_ids.contains(dep.as_str()) {
                    return Err(MillwrightError::UnknownDependency {
                        pipeline_id: self.id.clone(),
                        step_id: step.id.clone(),
                        dep_id: dep.clone(),
                    });
                }
            }
        }

        // Kahn's algorithm for topological sort + cycle detection.
        let mut in_degree: HashMap<&str, usize> = HashMap::new();
        let mut dependents: HashMap<&str, Vec<&str>> = HashMap::new();

        for step in &self.steps {
            in_degree.entry(step.id.as_str()).or_insert(0);
            for dep in &step.depends_on {
                *in_degree.entry(step.id.as_str()).or_insert(0) += 1;
                dependents
                    .entry(dep.as_str())
                    .or_default()
                    .push(step.id.as_str());
            }
        }

        // Note: in_degree is reset from scratch in Kahn's so we rebuild.
        let mut in_degree2: HashMap<&str, usize> =
            self.steps.iter().map(|s| (s.id.as_str(), 0)).collect();
        for step in &self.steps {
            for _dep in &step.depends_on {
                *in_degree2.entry(step.id.as_str()).or_insert(0) += 1;
            }
        }

        let mut queue: VecDeque<&str> = in_degree2
            .iter()
            .filter(|(_, &deg)| deg == 0)
            .map(|(&id, _)| id)
            .collect();

        let mut order: Vec<String> = Vec::with_capacity(self.steps.len());

        while let Some(node) = queue.pop_front() {
            order.push(node.to_owned());
            if let Some(next_steps) = dependents.get(node) {
                for &next in next_steps {
                    let deg = in_degree2.get_mut(next).unwrap();
                    *deg -= 1;
                    if *deg == 0 {
                        queue.push_back(next);
                    }
                }
            }
        }

        if order.len() != self.steps.len() {
            // Find one node still in a cycle for a better error message.
            let remaining: Vec<&str> = in_degree2
                .iter()
                .filter(|(_, &deg)| deg > 0)
                .map(|(&id, _)| id)
                .collect();
            return Err(MillwrightError::CyclicDependency {
                pipeline_id: self.id.clone(),
                step_id: remaining.first().copied().unwrap_or("?").to_owned(),
            });
        }

        Ok(order)
    }

    /// Find all steps that are ready to run (all deps in `completed_ids`).
    pub fn ready_steps<'a>(&'a self, completed_ids: &HashSet<String>) -> Vec<&'a Step> {
        self.steps
            .iter()
            .filter(|s| {
                !completed_ids.contains(&s.id)
                    && s.depends_on.iter().all(|dep| completed_ids.contains(dep))
            })
            .collect()
    }

    /// Return a step by ID.
    pub fn step(&self, id: &str) -> Option<&Step> {
        self.steps.iter().find(|s| s.id == id)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn ml_step(id: &str) -> Step {
        Step::new(id, StepType::LocalMl { model_id: "test-model".into() })
    }

    #[test]
    fn empty_pipeline_fails_validation() {
        let p = Pipeline::new("empty");
        assert!(matches!(p.validate(), Err(MillwrightError::EmptyPipeline(_))));
    }

    #[test]
    fn single_step_pipeline_validates() {
        let p = Pipeline::new("single").with_step(ml_step("a"));
        let order = p.validate().unwrap();
        assert_eq!(order, vec!["a"]);
    }

    #[test]
    fn linear_chain_validates_in_order() {
        let p = Pipeline::new("chain")
            .with_step(ml_step("a"))
            .with_step(ml_step("b").with_deps(["a"]))
            .with_step(ml_step("c").with_deps(["b"]));
        let order = p.validate().unwrap();
        // a must precede b which must precede c
        let pos: HashMap<&str, usize> = order.iter().enumerate().map(|(i, s)| (s.as_str(), i)).collect();
        assert!(pos["a"] < pos["b"]);
        assert!(pos["b"] < pos["c"]);
    }

    #[test]
    fn cycle_is_detected() {
        let p = Pipeline::new("cycle")
            .with_step(ml_step("a").with_deps(["b"]))
            .with_step(ml_step("b").with_deps(["a"]));
        assert!(matches!(p.validate(), Err(MillwrightError::CyclicDependency { .. })));
    }

    #[test]
    fn unknown_dep_is_detected() {
        let p = Pipeline::new("unknown")
            .with_step(ml_step("a").with_deps(["ghost"]));
        assert!(matches!(p.validate(), Err(MillwrightError::UnknownDependency { .. })));
    }

    #[test]
    fn parallel_diamond_validates() {
        // a → {b, c} → d
        let p = Pipeline::new("diamond")
            .with_step(ml_step("a"))
            .with_step(ml_step("b").with_deps(["a"]))
            .with_step(ml_step("c").with_deps(["a"]))
            .with_step(ml_step("d").with_deps(["b", "c"]));
        let order = p.validate().unwrap();
        let pos: HashMap<&str, usize> = order.iter().enumerate().map(|(i, s)| (s.as_str(), i)).collect();
        assert!(pos["a"] < pos["b"]);
        assert!(pos["a"] < pos["c"]);
        assert!(pos["b"] < pos["d"]);
        assert!(pos["c"] < pos["d"]);
    }

    #[test]
    fn ready_steps_returns_roots_initially() {
        let p = Pipeline::new("p")
            .with_step(ml_step("a"))
            .with_step(ml_step("b").with_deps(["a"]));
        let ready = p.ready_steps(&HashSet::new());
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].id, "a");
    }

    #[test]
    fn ready_steps_after_root_completes() {
        let p = Pipeline::new("p")
            .with_step(ml_step("a"))
            .with_step(ml_step("b").with_deps(["a"]));
        let completed = ["a".to_owned()].into();
        let ready = p.ready_steps(&completed);
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].id, "b");
    }

    #[test]
    fn step_type_kind_labels_are_correct() {
        assert_eq!(StepType::LocalMl { model_id: "x".into() }.kind_label(), "local_ml");
        assert_eq!(StepType::Rule { rule_id: "x".into() }.kind_label(), "rule");
        assert_eq!(StepType::Gate { condition: "x".into() }.kind_label(), "gate");
        assert!(StepType::Llm { prompt_template: "x".into(), max_tokens: 100 }.involves_llm());
        assert!(!StepType::LocalMl { model_id: "x".into() }.involves_llm());
    }
}
