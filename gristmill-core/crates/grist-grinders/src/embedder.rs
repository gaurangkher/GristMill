//! Concrete [`EmbedderSession`] implementation for the MiniLM-L6-v2 model.
//!
//! This module bridges the `grist-sieve` crate's abstract `EmbedderSession`
//! type with the actual ONNX session managed by `grist-grinders`.  The sieve
//! crate deliberately does not depend on the grinders crate to avoid a
//! circular dependency — instead it defines an `EmbedFn` trait that we
//! implement here.
//!
//! # Usage
//!
//! ```rust,no_run
//! use grist_grinders::embedder::build_minilm_embedder;
//! use grist_sieve::features::{FeatureExtractor, EmbedderSession};
//! use grist_grinders::GrindersConfig;
//!
//! let config = GrindersConfig::default();
//! let embedder: EmbedderSession = build_minilm_embedder(&config).unwrap();
//! let extractor = FeatureExtractor::new_with_embedder(embedder);
//! ```
//!
//! # Hot-path note
//!
//! The `embed()` call is on the Sieve triage hot path (<5 ms p99).
//! ONNX Runtime's `Session::run()` is designed to be called concurrently
//! from any thread; no mutex is needed around the session itself.

#[cfg(feature = "onnx")]
use std::sync::Arc;

use ndarray::Array1;
#[cfg(feature = "onnx")]
use tracing::{debug, instrument};

#[cfg(feature = "onnx")]
use grist_sieve::error::SieveError;
use grist_sieve::features::{EmbedderSession, EMBEDDING_DIM};

use crate::config::GrindersConfig;
#[cfg(feature = "onnx")]
use crate::onnx::tokenize_for_minilm;

/// Maximum token sequence length for MiniLM (matches training configuration).
#[cfg(feature = "onnx")]
const MINILM_MAX_LEN: usize = 128;

// ─────────────────────────────────────────────────────────────────────────────
// Public builder
// ─────────────────────────────────────────────────────────────────────────────

/// Build a concrete [`EmbedderSession`] backed by the `minilm-l6-v2` model.
///
/// Looks up the model path from the first entry in `config.models` whose
/// `model_id` is `"minilm-l6-v2"`.  If not found, or the file does not exist,
/// returns a stub embedder that produces zeroed vectors (allowing the sieve to
/// operate in metadata-only mode without panicking).
///
/// **Production usage**: compile with `--features onnx` and ensure
/// `minilm-l6-v2.onnx` is present on disk (run `gristmill models pull`).
pub fn build_minilm_embedder(
    config: &GrindersConfig,
) -> Result<EmbedderSession, crate::error::GrindersError> {
    // Locate the MiniLM config entry (only needed when ONNX feature is enabled).
    #[cfg(feature = "onnx")]
    {
        let model_cfg = config.models.iter().find(|m| m.model_id == "minilm-l6-v2");
        if let Some(cfg) = model_cfg {
            if cfg.path.exists() {
                return build_onnx_embedder(cfg);
            }
        }
    }
    // Suppress unused-variable warning when onnx feature is off.
    #[cfg(not(feature = "onnx"))]
    let _ = config;

    // Fallback: zero-vector embedder (metadata-only mode).
    tracing::warn!(
        "minilm-l6-v2 model not available — Sieve will operate in metadata-only mode \
         (embeddings zeroed); run `gristmill models pull` to enable full embeddings"
    );
    Ok(zero_embedder())
}

/// Construct a zero-vector embedder (returns `Array1::zeros(EMBEDDING_DIM)`).
///
/// Used when no ONNX model is available.  The Sieve still classifies correctly
/// using the 8 metadata features; only semantic-cache cosine similarity is
/// degraded.
pub fn zero_embedder() -> EmbedderSession {
    EmbedderSession::from_fn(|_text: &str| Ok(Array1::zeros(EMBEDDING_DIM)))
}

// ─────────────────────────────────────────────────────────────────────────────
// ONNX-backed embedder (feature-gated)
// ─────────────────────────────────────────────────────────────────────────────

/// Build an ONNX-backed embedder from a loaded model config.
#[cfg(feature = "onnx")]
fn build_onnx_embedder(
    cfg: &crate::config::ModelConfig,
) -> Result<EmbedderSession, crate::error::GrindersError> {
    use ort::Session;

    let session = Session::builder()
        .map_err(|e| crate::error::GrindersError::ModelLoadFailed {
            model_id: "minilm-l6-v2".into(),
            reason: e.to_string(),
        })?
        .commit_from_file(&cfg.path)
        .map_err(|e| crate::error::GrindersError::ModelLoadFailed {
            model_id: "minilm-l6-v2".into(),
            reason: e.to_string(),
        })?;

    // Wrap in Arc so we can move into the closure without copying.
    let session = Arc::new(session);

    tracing::info!(path = %cfg.path.display(), "MiniLM embedder loaded");
    metrics::counter!("grinders.embedder.loads").increment(1);

    let embedder = EmbedderSession::from_fn(move |text: &str| embed_with_session(&session, text));

    Ok(embedder)
}

/// Run one MiniLM embedding inference call.
#[cfg(feature = "onnx")]
#[instrument(level = "trace", skip(session))]
fn embed_with_session(session: &ort::Session, text: &str) -> Result<Array1<f32>, SieveError> {
    use ort::inputs;

    let (input_ids, attention_mask, token_type_ids) = tokenize_for_minilm(text, MINILM_MAX_LEN);

    // Run the ONNX session.
    let outputs = session
        .run(
            inputs![
                "input_ids"      => input_ids.view(),
                "attention_mask" => attention_mask.view(),
                "token_type_ids" => token_type_ids.view(),
            ]
            .map_err(|e| SieveError::FeatureExtraction(e.to_string()))?,
        )
        .map_err(|e| SieveError::FeatureExtraction(e.to_string()))?;

    // MiniLM outputs "last_hidden_state" [1, seq_len, 384].
    // We mean-pool over the sequence dimension to get [384].
    let hidden = outputs
        .get("last_hidden_state")
        .ok_or_else(|| {
            SieveError::FeatureExtraction("MiniLM output 'last_hidden_state' not found".into())
        })?
        .try_extract_tensor::<f32>()
        .map_err(|e| SieveError::FeatureExtraction(e.to_string()))?;

    // Shape: [1, seq_len, 384].  Mean-pool over seq_len dimension.
    let shape = hidden.shape().to_vec();
    if shape.len() < 3 || shape[2] != EMBEDDING_DIM {
        return Err(SieveError::FeatureExtraction(format!(
            "unexpected MiniLM output shape: {shape:?}"
        )));
    }
    let seq_len = shape[1];
    let mut embedding = Array1::<f32>::zeros(EMBEDDING_DIM);
    for seq_idx in 0..seq_len {
        for dim in 0..EMBEDDING_DIM {
            embedding[dim] += hidden[[0, seq_idx, dim]];
        }
    }
    embedding.mapv_inplace(|x| x / seq_len as f32);

    // L2 normalise.
    let norm = embedding.dot(&embedding).sqrt();
    if norm > 1e-8 {
        embedding.mapv_inplace(|x| x / norm);
    }

    debug!(seq_len, "MiniLM embedding complete");
    metrics::counter!("grinders.embedder.calls").increment(1);

    Ok(embedding)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::GrindersConfig;
    use grist_sieve::features::EMBEDDING_DIM;

    #[test]
    fn zero_embedder_returns_correct_dim() {
        let emb = zero_embedder();
        let out = emb.embed("hello world").unwrap();
        assert_eq!(out.len(), EMBEDDING_DIM);
    }

    #[test]
    fn zero_embedder_is_all_zeros() {
        let emb = zero_embedder();
        let out = emb.embed("test text").unwrap();
        assert!(out.iter().all(|&x| x == 0.0), "expected all zeros");
    }

    #[test]
    fn build_minilm_embedder_falls_back_gracefully_when_no_model() {
        // Config with no models — should fall back to zero embedder.
        let config = GrindersConfig::default();
        let emb = build_minilm_embedder(&config).unwrap();
        let out = emb.embed("test sentence").unwrap();
        assert_eq!(out.len(), EMBEDDING_DIM);
    }

    #[test]
    fn embedder_output_is_deterministic() {
        let emb = zero_embedder();
        let a = emb.embed("hello").unwrap();
        let b = emb.embed("hello").unwrap();
        assert_eq!(a, b);
    }
}
