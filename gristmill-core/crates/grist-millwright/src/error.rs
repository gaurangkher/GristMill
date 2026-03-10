//! Error types for the Millwright DAG orchestrator.

use thiserror::Error;

/// All errors produced by the Millwright.
#[derive(Debug, Error)]
pub enum MillwrightError {
    /// The pipeline definition contains a cycle.
    #[error("pipeline '{pipeline_id}' has a dependency cycle involving step '{step_id}'")]
    CyclicDependency { pipeline_id: String, step_id: String },

    /// A step's `depends_on` references a step that does not exist.
    #[error("step '{step_id}' in pipeline '{pipeline_id}' depends on unknown step '{dep_id}'")]
    UnknownDependency {
        pipeline_id: String,
        step_id: String,
        dep_id: String,
    },

    /// Pipeline has no steps.
    #[error("pipeline '{0}' has no steps")]
    EmptyPipeline(String),

    /// Step execution failed after all retries were exhausted.
    #[error("step '{step_id}' failed after {attempts} attempt(s): {reason}")]
    StepFailed {
        step_id: String,
        attempts: u32,
        reason: String,
    },

    /// The whole pipeline timed out.
    #[error("pipeline '{pipeline_id}' timed out after {elapsed_ms}ms (limit {timeout_ms}ms)")]
    PipelineTimeout {
        pipeline_id: String,
        elapsed_ms: u64,
        timeout_ms: u64,
    },

    /// A single step exceeded its per-step timeout.
    #[error("step '{step_id}' timed out after {elapsed_ms}ms")]
    StepTimeout { step_id: String, elapsed_ms: u64 },

    /// An approval gate was rejected.
    #[error("gate for step '{step_id}' was rejected: {reason}")]
    GateRejected { step_id: String, reason: String },

    /// An approval gate timed out waiting for a response.
    #[error("gate for step '{step_id}' timed out after {elapsed_ms}ms")]
    GateTimeout { step_id: String, elapsed_ms: u64 },

    /// A run with this ID was not found in the checkpoint store.
    #[error("run '{0}' not found in checkpoint store")]
    RunNotFound(String),

    /// Serialization / deserialization error (checkpoint I/O).
    #[error("serialization error: {0}")]
    Serialization(String),

    /// An I/O error (checkpoint file operations).
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// The pipeline definition is invalid (catch-all for config errors).
    #[error("invalid pipeline '{pipeline_id}': {reason}")]
    InvalidPipeline { pipeline_id: String, reason: String },

    /// Bus publish failed.
    #[error("failed to publish to bus topic '{topic}': {reason}")]
    BusPublish { topic: String, reason: String },

    /// Wraps any other anyhow error.
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}
