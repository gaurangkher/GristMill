//! `GrindersSession` — a live, loaded model session.
//!
//! This is the inner type stored in the registry and passed to the worker pool.
//! It is a thin, runtime-dispatching wrapper over the actual ONNX session or
//! GGUF context so that the registry can be generic over runtimes.

use ndarray::{Array1, Array2};

use crate::error::GrindersError;

// ─────────────────────────────────────────────────────────────────────────────
// Inference request / response
// ─────────────────────────────────────────────────────────────────────────────

/// A single inference request.
#[derive(Debug, Clone)]
pub struct InferenceRequest {
    /// Model to invoke.
    pub model_id: String,
    /// Input tensor (1 × feature_dim) for ONNX models.
    pub features: Option<Array2<f32>>,
    /// Raw text prompt for GGUF/generative models.
    pub prompt: Option<String>,
}

impl InferenceRequest {
    /// Create a feature-vector request (ONNX classifiers / embedders).
    pub fn from_features(model_id: impl Into<String>, features: Array2<f32>) -> Self {
        Self {
            model_id: model_id.into(),
            features: Some(features),
            prompt: None,
        }
    }

    /// Create a text-prompt request (GGUF generative models).
    pub fn from_prompt(model_id: impl Into<String>, prompt: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            features: None,
            prompt: Some(prompt.into()),
        }
    }
}

/// The output of a single inference call.
#[derive(Debug, Clone)]
pub struct InferenceOutput {
    /// For ONNX classification/embedding — the raw output tensor flattened.
    pub tensor: Option<Array1<f32>>,
    /// For GGUF text generation — the generated text.
    pub text: Option<String>,
    /// Wall-clock inference time in milliseconds.
    pub elapsed_ms: u64,
}

impl InferenceOutput {
    /// Create a tensor output.
    pub fn tensor(data: Array1<f32>, elapsed_ms: u64) -> Self {
        Self {
            tensor: Some(data),
            text: None,
            elapsed_ms,
        }
    }

    /// Create a text output.
    pub fn text(text: String, elapsed_ms: u64) -> Self {
        Self {
            tensor: None,
            text: Some(text),
            elapsed_ms,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Session kind (runtime-specific inner state)
// ─────────────────────────────────────────────────────────────────────────────

/// Runtime-specific inner state of a session.
pub enum SessionKind {
    /// ONNX Runtime session.
    #[cfg(feature = "onnx")]
    Onnx(ort::Session),

    /// GGUF / llama.cpp context.
    #[cfg(feature = "gguf")]
    Gguf(Box<dyn GgufContext + Send + Sync>),

    /// Stub session (used in tests or when the runtime is not compiled in).
    Stub { model_id: String },
}

impl std::fmt::Debug for SessionKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            #[cfg(feature = "onnx")]
            SessionKind::Onnx(_) => write!(f, "Onnx(..)"),
            #[cfg(feature = "gguf")]
            SessionKind::Gguf(_) => write!(f, "Gguf(..)"),
            SessionKind::Stub { model_id } => write!(f, "Stub({model_id})"),
        }
    }
}

/// Trait object interface for GGUF contexts.
/// Allows the registry to store GGUF sessions without depending on a specific
/// llama.cpp binding crate's types directly.
#[cfg(feature = "gguf")]
pub trait GgufContext {
    fn generate(&self, prompt: &str, max_tokens: usize) -> Result<String, GrindersError>;
}

// ─────────────────────────────────────────────────────────────────────────────
// GrindersSession
// ─────────────────────────────────────────────────────────────────────────────

/// A live model session that can run inference.
///
/// `GrindersSession` is `Send + Sync` — it is shared via `Arc` across the
/// Rayon worker pool.  Each `run()` call is stateless from the caller's
/// perspective (ONNX sessions are internally thread-safe).
pub struct GrindersSession {
    pub model_id: String,
    pub kind: SessionKind,
    /// Per-model timeout copied from the model config.
    pub timeout: std::time::Duration,
    /// Maximum generation tokens (GGUF only).
    pub max_tokens: usize,
}

impl std::fmt::Debug for GrindersSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GrindersSession")
            .field("model_id", &self.model_id)
            .field("kind", &self.kind)
            .finish()
    }
}

impl GrindersSession {
    /// Run inference on a single request.
    ///
    /// For ONNX models this calls the ONNX Runtime session synchronously
    /// (ONNX Runtime is designed to be called from any thread).
    /// For GGUF models it calls llama.cpp.
    pub fn run(&self, req: &InferenceRequest) -> Result<InferenceOutput, GrindersError> {
        let t0 = std::time::Instant::now();

        let output = match &self.kind {
            #[cfg(feature = "onnx")]
            SessionKind::Onnx(session) => run_onnx(session, req, &self.model_id)?,

            #[cfg(feature = "gguf")]
            SessionKind::Gguf(ctx) => run_gguf(ctx.as_ref(), req, &self.model_id, self.max_tokens)?,

            SessionKind::Stub { model_id } => {
                // Stub returns a zero tensor for feature requests and empty
                // text for prompt requests. Used in tests.
                if let Some(features) = &req.features {
                    let n = features.ncols();
                    InferenceOutput::tensor(Array1::zeros(n), 0)
                } else {
                    InferenceOutput::text(
                        format!("[stub:{model_id}] response for: {}", req.prompt.as_deref().unwrap_or("")),
                        0,
                    )
                }
            }
        };

        let elapsed_ms = t0.elapsed().as_millis() as u64;
        metrics::histogram!("grinders.inference.latency_ms", "model_id" => self.model_id.clone())
            .record(elapsed_ms as f64);

        Ok(InferenceOutput {
            elapsed_ms,
            ..output
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Runtime dispatch helpers
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "onnx")]
fn run_onnx(
    session: &ort::Session,
    req: &InferenceRequest,
    model_id: &str,
) -> Result<InferenceOutput, GrindersError> {
    use ort::inputs;

    let features = req.features.as_ref().ok_or_else(|| GrindersError::OnnxInference {
        model_id: model_id.to_owned(),
        reason: "ONNX model requires feature tensor input".into(),
    })?;

    let outputs = session
        .run(
            inputs!["features" => features.view()]
                .map_err(|e| GrindersError::OnnxInference {
                    model_id: model_id.to_owned(),
                    reason: e.to_string(),
                })?,
        )
        .map_err(|e| GrindersError::OnnxInference {
            model_id: model_id.to_owned(),
            reason: e.to_string(),
        })?;

    // Try "output", "logits", "embeddings" — common output tensor names.
    let output_tensor = ["output", "logits", "embeddings", "output_0"]
        .iter()
        .find_map(|&name| outputs.get(name))
        .ok_or_else(|| GrindersError::OnnxInference {
            model_id: model_id.to_owned(),
            reason: "no recognised output tensor (tried: output, logits, embeddings, output_0)".into(),
        })?;

    let data = output_tensor
        .try_extract_tensor::<f32>()
        .map_err(|e| GrindersError::OnnxInference {
            model_id: model_id.to_owned(),
            reason: e.to_string(),
        })?;

    let flat: Array1<f32> = data.iter().cloned().collect();
    Ok(InferenceOutput::tensor(flat, 0))
}

#[cfg(feature = "gguf")]
fn run_gguf(
    ctx: &dyn GgufContext,
    req: &InferenceRequest,
    model_id: &str,
    max_tokens: usize,
) -> Result<InferenceOutput, GrindersError> {
    let prompt = req.prompt.as_deref().ok_or_else(|| GrindersError::GgufInference {
        model_id: model_id.to_owned(),
        reason: "GGUF model requires a text prompt".into(),
    })?;
    let text = ctx.generate(prompt, max_tokens)?;
    Ok(InferenceOutput::text(text, 0))
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn stub_session(id: &str) -> GrindersSession {
        GrindersSession {
            model_id: id.to_owned(),
            kind: SessionKind::Stub { model_id: id.to_owned() },
            timeout: std::time::Duration::from_secs(5),
            max_tokens: 128,
        }
    }

    #[test]
    fn stub_session_returns_zero_tensor_for_feature_request() {
        let s = stub_session("test");
        let req = InferenceRequest::from_features("test", Array2::zeros((1, 392)));
        let out = s.run(&req).unwrap();
        assert!(out.tensor.is_some());
        assert_eq!(out.tensor.unwrap().len(), 392);
    }

    #[test]
    fn stub_session_returns_text_for_prompt_request() {
        let s = stub_session("test");
        let req = InferenceRequest::from_prompt("test", "hello world");
        let out = s.run(&req).unwrap();
        assert!(out.text.is_some());
        assert!(out.text.unwrap().contains("hello world"));
    }
}
