//! GGUF / llama.cpp inference backend (PRD G-03).
//!
//! Provides local summarisation via Phi-3 Mini Q4 (or any GGUF-format model).
//! When the `gguf` Cargo feature is disabled, all calls return a stub that
//! surfaces a clear error — production deployments must compile with
//! `--features gguf`.
//!
//! # Architecture note
//!
//! GGUF inference is intentionally isolated from the hot Sieve path.  Phi-3
//! is a 2.3 GB cold model used exclusively for Ledger compaction — it is
//! never loaded on the triage fast path.

#[cfg(feature = "gguf")]
use tracing::info;
use tracing::warn;

use crate::config::ModelConfig;
use crate::error::GrindersError;
use crate::session::{GrindersSession, SessionKind};

/// Load a GGUF model session from the path specified in `config`.
///
/// When the `gguf` feature is not compiled in, returns a stub session that
/// returns [`GrindersError::RuntimeNotAvailable`] on every inference call.
pub fn load_gguf_session(config: &ModelConfig) -> Result<GrindersSession, GrindersError> {
    #[cfg(feature = "gguf")]
    {
        return load_gguf_real(config);
    }

    #[cfg(not(feature = "gguf"))]
    {
        if config.path.exists() {
            warn!(
                model_id = config.model_id,
                path = %config.path.display(),
                "GGUF runtime not compiled in; using stub session (enable --features gguf)",
            );
        } else {
            warn!(
                model_id = config.model_id,
                "GGUF model file not found and runtime not compiled in; using stub",
            );
        }

        Ok(GrindersSession {
            model_id: config.model_id.clone(),
            kind: SessionKind::Stub {
                model_id: config.model_id.clone(),
            },
            timeout: config.timeout,
            max_tokens: config.max_tokens,
        })
    }
}

/// Real GGUF load (only compiled when `--features gguf` is set).
///
/// Uses the `llama_cpp` crate to load the model and wraps it in a trait
/// object ([`GgufContext`]) so the rest of the codebase does not import
/// llama_cpp types directly.
#[cfg(feature = "gguf")]
fn load_gguf_real(config: &ModelConfig) -> Result<GrindersSession, GrindersError> {
    use crate::session::GgufContext;

    if !config.path.exists() {
        return Err(GrindersError::ModelLoadFailed {
            model_id: config.model_id.clone(),
            reason: format!(
                "GGUF model file not found at '{}' — run `gristmill models pull` to download",
                config.path.display()
            ),
        });
    }

    info!(
        model_id = config.model_id,
        path = %config.path.display(),
        "loading GGUF model (this may take several seconds for large files)",
    );

    // llama_cpp::LlamaModel is the entry point for the llama-cpp-2 crate.
    let model =
        llama_cpp::LlamaModel::load_from_file(&config.path, llama_cpp::LlamaParams::default())
            .map_err(|e| GrindersError::ModelLoadFailed {
                model_id: config.model_id.clone(),
                reason: e.to_string(),
            })?;

    info!(model_id = config.model_id, "GGUF model loaded successfully");
    metrics::counter!("grinders.gguf.loads").increment(1);

    let ctx = LlamaCppCtx {
        model_id: config.model_id.clone(),
        model,
    };

    Ok(GrindersSession {
        model_id: config.model_id.clone(),
        kind: SessionKind::Gguf(Box::new(ctx)),
        timeout: config.timeout,
        max_tokens: config.max_tokens,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// GgufContext implementation via llama-cpp-2
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "gguf")]
struct LlamaCppCtx {
    model_id: String,
    model: llama_cpp::LlamaModel,
}

#[cfg(feature = "gguf")]
impl crate::session::GgufContext for LlamaCppCtx {
    fn generate(&self, prompt: &str, max_tokens: usize) -> Result<String, GrindersError> {
        use std::io::Write;

        // Create a new session for each inference call.
        // llama_cpp::LlamaModel::create_session() is the standard entrypoint.
        let mut session = self
            .model
            .create_session(llama_cpp::SessionParams::default())
            .map_err(|e| GrindersError::GgufInference {
                model_id: self.model_id.clone(),
                reason: e.to_string(),
            })?;

        session
            .advance_context(prompt)
            .map_err(|e| GrindersError::GgufInference {
                model_id: self.model_id.clone(),
                reason: e.to_string(),
            })?;

        let mut output = String::new();
        let sampler = llama_cpp::standard_sampler::StandardSampler::default();
        let completions = session
            .start_completing_with(sampler, max_tokens)
            .map_err(|e| GrindersError::GgufInference {
                model_id: self.model_id.clone(),
                reason: e.to_string(),
            })?;

        for token in completions.take(max_tokens) {
            output.push_str(&token);
        }

        metrics::counter!("grinders.gguf.completions").increment(1);
        Ok(output)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{ModelConfig, ModelRuntime};
    use crate::session::SessionKind;
    use std::time::Duration;

    fn gguf_config(path: &str) -> ModelConfig {
        ModelConfig {
            model_id: "phi3-mini-4k-Q4".into(),
            path: std::path::PathBuf::from(path),
            runtime: ModelRuntime::Gguf,
            warm: false,
            timeout: Duration::from_secs(60),
            max_tokens: 128,
            description: String::new(),
        }
    }

    #[test]
    fn load_missing_gguf_returns_stub_when_no_gguf_feature() {
        let cfg = gguf_config("/nonexistent/phi3.gguf");
        // Without the `gguf` feature the loader returns a stub, not an error.
        let session = load_gguf_session(&cfg).unwrap();
        assert_eq!(session.model_id, "phi3-mini-4k-Q4");
        assert!(matches!(session.kind, SessionKind::Stub { .. }));
    }

    #[test]
    fn stub_generates_placeholder_text() {
        use crate::session::InferenceRequest;

        let cfg = gguf_config("/nonexistent/phi3.gguf");
        let session = load_gguf_session(&cfg).unwrap();
        let req = InferenceRequest::from_prompt("phi3-mini-4k-Q4", "Summarize this document.");
        let out = session.run(&req).unwrap();
        assert!(out.text.is_some(), "expected text output from stub");
    }
}
