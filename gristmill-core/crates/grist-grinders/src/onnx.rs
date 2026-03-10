//! ONNX Runtime inference backend (PRD G-02).
//!
//! Loads an ONNX model file into an `ort::Session` and wraps it in a
//! [`GrindersSession`].  The session is thread-safe and can be shared across
//! Rayon workers via `Arc`.
//!
//! # Feature gate
//!
//! This module is compiled unconditionally, but its `load_onnx_session`
//! function returns [`GrindersError::RuntimeNotAvailable`] when the `onnx`
//! feature is not enabled.  When the feature *is* enabled, it attempts to
//! load the ONNX model from disk.

#[cfg(feature = "onnx")]
use tracing::info;
use tracing::warn;

use crate::config::ModelConfig;
use crate::error::GrindersError;
use crate::session::{GrindersSession, SessionKind};

/// Load an ONNX model session from the path specified in `config`.
///
/// When the `onnx` feature is not compiled in, returns a stub session that
/// indicates inference is unavailable.  Production deployments must compile
/// with `--features onnx`.
pub fn load_onnx_session(config: &ModelConfig) -> Result<GrindersSession, GrindersError> {
    #[cfg(feature = "onnx")]
    {
        return load_onnx_real(config);
    }

    #[cfg(not(feature = "onnx"))]
    {
        // No ONNX runtime — produce a stub that returns an error on inference.
        if config.path.exists() {
            warn!(
                model_id = config.model_id,
                path = %config.path.display(),
                "ONNX runtime not compiled in; using stub session (enable --features onnx)",
            );
        } else {
            warn!(
                model_id = config.model_id,
                "ONNX model file not found and runtime not compiled in; using stub",
            );
        }

        Ok(GrindersSession {
            model_id: config.model_id.clone(),
            kind: SessionKind::Stub { model_id: config.model_id.clone() },
            timeout: config.timeout,
            max_tokens: config.max_tokens,
        })
    }
}

/// Real ONNX load (only compiled when `--features onnx` is set).
#[cfg(feature = "onnx")]
fn load_onnx_real(config: &ModelConfig) -> Result<GrindersSession, GrindersError> {
    if !config.path.exists() {
        return Err(GrindersError::ModelLoadFailed {
            model_id: config.model_id.clone(),
            reason: format!(
                "model file not found at '{}' — run `gristmill models pull` to download",
                config.path.display()
            ),
        });
    }

    info!(
        model_id = config.model_id,
        path = %config.path.display(),
        "loading ONNX model",
    );

    let session = ort::Session::builder()
        .map_err(|e| GrindersError::ModelLoadFailed {
            model_id: config.model_id.clone(),
            reason: e.to_string(),
        })?
        .commit_from_file(&config.path)
        .map_err(|e| GrindersError::ModelLoadFailed {
            model_id: config.model_id.clone(),
            reason: e.to_string(),
        })?;

    info!(model_id = config.model_id, "ONNX model loaded successfully");
    metrics::counter!("grinders.onnx.loads").increment(1);

    Ok(GrindersSession {
        model_id: config.model_id.clone(),
        kind: SessionKind::Onnx(session),
        timeout: config.timeout,
        max_tokens: config.max_tokens,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// MiniLM embedding helper
// ─────────────────────────────────────────────────────────────────────────────

/// Token-level word-piece tokenizer stub for MiniLM.
///
/// In a full implementation this would call a HuggingFace tokenizer via
/// the `tokenizers` crate.  Here we use a simple whitespace splitter that
/// produces a fixed-length integer token sequence suitable for testing the
/// ONNX pipeline end-to-end.
///
/// The `minilm-l6-v2` model expects inputs:
/// - `input_ids`     — i64 [1, seq_len]
/// - `attention_mask`— i64 [1, seq_len]
/// - `token_type_ids`— i64 [1, seq_len]   (optional, all zeros)
///
/// This function returns those three tensors (as `ndarray::Array2<i64>`)
/// truncated / padded to `max_len` tokens.
#[allow(dead_code)]
pub fn tokenize_for_minilm(
    text: &str,
    max_len: usize,
) -> (
    ndarray::Array2<i64>,
    ndarray::Array2<i64>,
    ndarray::Array2<i64>,
) {
    let words: Vec<&str> = text.split_whitespace().collect();
    let seq_len = max_len.min(words.len() + 2); // +2 for [CLS] and [SEP]

    let mut ids = vec![0i64; seq_len];
    let mut mask = vec![0i64; seq_len];

    // [CLS] = 101, [SEP] = 102 (standard BERT vocab ids)
    ids[0] = 101;
    mask[0] = 1;

    for (i, word) in words.iter().take(seq_len.saturating_sub(2)).enumerate() {
        // Deterministic pseudo-id based on character sum — not a real tokenizer.
        let word_id: i64 = word.bytes().map(|b| b as i64).sum::<i64>() % 30_000 + 1000;
        ids[i + 1] = word_id;
        mask[i + 1] = 1;
    }

    // [SEP] at the last real position.
    let sep_pos = (words.len() + 1).min(seq_len - 1);
    ids[sep_pos] = 102;
    mask[sep_pos] = 1;

    let shape = (1, seq_len);
    let input_ids = ndarray::Array2::from_shape_vec(shape, ids.clone()).unwrap();
    let attention_mask = ndarray::Array2::from_shape_vec(shape, mask).unwrap();
    let token_type_ids = ndarray::Array2::zeros(shape);

    (input_ids, attention_mask, token_type_ids)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{ModelConfig, ModelRuntime};
    use std::time::Duration;

    fn onnx_config(path: &str) -> ModelConfig {
        ModelConfig {
            model_id: "test-onnx".into(),
            path: std::path::PathBuf::from(path),
            runtime: ModelRuntime::Onnx,
            warm: false,
            timeout: Duration::from_secs(5),
            max_tokens: 0,
            description: String::new(),
        }
    }

    #[test]
    fn load_missing_file_returns_stub_when_no_onnx_feature() {
        // Without the `onnx` feature the loader returns a stub, not an error.
        let cfg = onnx_config("/nonexistent/model.onnx");
        let session = load_onnx_session(&cfg).unwrap();
        assert_eq!(session.model_id, "test-onnx");
        // Stub sessions always produce a stub kind.
        assert!(matches!(session.kind, SessionKind::Stub { .. }));
    }

    #[test]
    fn tokenize_produces_correct_shape() {
        let (ids, mask, types) = tokenize_for_minilm("hello world foo bar", 16);
        assert_eq!(ids.shape(), &[1, 6]); // [CLS] + 4 words + [SEP]
        assert_eq!(mask.shape(), &[1, 6]);
        assert_eq!(types.shape(), &[1, 6]);
    }

    #[test]
    fn tokenize_cls_sep_are_set() {
        let (ids, _mask, _types) = tokenize_for_minilm("hello world", 16);
        // ids[0][0] == 101 (CLS), ids[0][last_real] == 102 (SEP)
        assert_eq!(ids[[0, 0]], 101);
        // With 2 words + CLS + SEP = 4 tokens; SEP is at index 3.
        assert_eq!(ids[[0, 3]], 102);
    }

    #[test]
    fn tokenize_truncates_to_max_len() {
        let long_text = "word ".repeat(200);
        let (ids, _mask, _types) = tokenize_for_minilm(&long_text, 32);
        assert_eq!(ids.ncols(), 32);
    }
}
