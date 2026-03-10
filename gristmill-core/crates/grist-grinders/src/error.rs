//! Error types for the Grinders inference pool.

use thiserror::Error;

/// Errors produced by the Grinders inference pool.
#[derive(Debug, Error)]
pub enum GrindersError {
    /// The requested model is not registered in the registry.
    #[error("model not found: {0}")]
    ModelNotFound(String),

    /// The model is registered but not yet loaded (cold) and load failed.
    #[error("model load failed for '{model_id}': {reason}")]
    ModelLoadFailed { model_id: String, reason: String },

    /// ONNX Runtime returned an inference error.
    #[error("ONNX inference error for '{model_id}': {reason}")]
    OnnxInference { model_id: String, reason: String },

    /// GGUF / llama.cpp inference error.
    #[error("GGUF inference error for '{model_id}': {reason}")]
    GgufInference { model_id: String, reason: String },

    /// The inference request timed out (G-07).
    #[error("inference timeout for model '{model_id}' after {elapsed_ms}ms")]
    Timeout { model_id: String, elapsed_ms: u64 },

    /// The worker pool is full and the request was rejected (back-pressure).
    #[error("worker pool at capacity — request dropped for model '{0}'")]
    PoolFull(String),

    /// The output tensor had an unexpected shape.
    #[error("tensor shape mismatch for '{model_id}': expected {expected:?}, got {actual:?}")]
    TensorShape {
        model_id: String,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    /// An I/O error (e.g. reading the model file).
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// A runtime requires a feature that was not compiled in.
    #[error("runtime '{runtime}' is not compiled in (enable feature '{feature}')")]
    RuntimeNotAvailable { runtime: String, feature: String },

    /// Wraps any other anyhow error.
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}
