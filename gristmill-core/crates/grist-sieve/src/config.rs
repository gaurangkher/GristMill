//! Configuration types for the Sieve.
//!
//! Loaded from the main `~/.gristmill/config.yaml` by `grist-config` and
//! injected when constructing the [`Sieve`](crate::Sieve).

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// All Sieve-specific configuration values.
///
/// Corresponds to the `sieve:` section of `~/.gristmill/config.yaml`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SieveConfig {
    /// Path to the ONNX classifier model file.
    /// `None` → heuristic fallback (no model required).
    pub model_path: Option<PathBuf>,

    /// Confidence threshold below which the oracle escalates the route.
    /// PRD default: 0.85 (S-08).
    pub confidence_threshold: f32,

    /// Directory where feedback JSONL logs are written.
    /// Default: `~/.gristmill/feedback/`
    pub feedback_dir: Option<PathBuf>,

    /// Maximum number of exact-cache entries.
    pub exact_cache_size: usize,

    /// Maximum number of semantic ring-buffer entries.
    pub semantic_cache_size: usize,

    /// Cosine similarity threshold for semantic cache hits.
    pub semantic_similarity_threshold: f32,

    /// Model ID used for LOCAL_ML routes (matches a Grinders registry entry).
    pub default_local_model: Option<String>,

    /// Rule set ID used for RULES routes.
    pub default_rule_id: Option<String>,

    /// Prompt template ID used for HYBRID routes.
    pub hybrid_prompt_template: Option<String>,

    /// Path to the SQLite WAL training buffer database.
    ///
    /// When `Some`, a [`TrainingBuffer`](crate::training_buffer::TrainingBuffer)
    /// is opened at this path and escalation records are written after each
    /// successful open-source teacher response.
    ///
    /// Default: `~/.gristmill/db/training_buffer.sqlite`
    pub training_buffer_path: Option<PathBuf>,
}

impl Default for SieveConfig {
    fn default() -> Self {
        // Resolve default training buffer path to ~/.gristmill/db/
        let training_buffer_path = std::env::var("HOME").ok().map(|h| {
            PathBuf::from(h)
                .join(".gristmill")
                .join("db")
                .join("training_buffer.sqlite")
        });

        Self {
            model_path: None,
            confidence_threshold: 0.85,
            feedback_dir: None,
            exact_cache_size: 10_000,
            semantic_cache_size: 256,
            semantic_similarity_threshold: 0.92,
            default_local_model: None,
            default_rule_id: None,
            hybrid_prompt_template: None,
            training_buffer_path,
        }
    }
}
