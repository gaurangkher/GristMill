//! Configuration for the Grinders inference pool.
//!
//! `GrindersConfig` is the top-level config struct consumed by [`Grinders::new`].
//! It can be built programmatically or deserialised from the GristMill YAML config.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;

use serde::{Deserialize, Serialize};

// ─────────────────────────────────────────────────────────────────────────────
// Runtime enum
// ─────────────────────────────────────────────────────────────────────────────

/// Inference runtime to use for a model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelRuntime {
    /// ONNX Runtime via the `ort` crate.
    Onnx,
    /// llama.cpp GGUF via the `llama-cpp-2` crate.
    Gguf,
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-model configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration entry for a single model in the registry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Unique model identifier used in API calls (e.g. `"intent-classifier-v1"`).
    pub model_id: String,

    /// Filesystem path to the model file (`.onnx` or `.gguf`).
    pub path: PathBuf,

    /// Inference runtime to use.
    pub runtime: ModelRuntime,

    /// If `true` the model is loaded at startup (warm); otherwise on-demand (cold).
    /// PRD G-04: warm models must respond in <5ms, cold models load in <2s.
    #[serde(default = "default_warm")]
    pub warm: bool,

    /// Per-model inference timeout (PRD G-07).
    /// Defaults to 5 seconds.
    #[serde(default = "default_timeout_secs", with = "duration_secs")]
    pub timeout: Duration,

    /// Maximum number of GGUF output tokens (GGUF models only).
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,

    /// Optional human-readable description.
    #[serde(default)]
    pub description: String,
}

fn default_warm() -> bool {
    true
}

fn default_timeout_secs() -> Duration {
    Duration::from_secs(5)
}

fn default_max_tokens() -> usize {
    128
}

/// Serde helper: serialise/deserialise Duration as integer seconds.
mod duration_secs {
    use serde::{Deserialize, Deserializer, Serializer};
    use std::time::Duration;

    pub fn serialize<S>(d: &Duration, ser: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        ser.serialize_u64(d.as_secs())
    }

    pub fn deserialize<'de, D>(de: D) -> Result<Duration, D::Error>
    where
        D: Deserializer<'de>,
    {
        let secs = u64::deserialize(de)?;
        Ok(Duration::from_secs(secs))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Top-level pool config
// ─────────────────────────────────────────────────────────────────────────────

/// Top-level configuration for the Grinders inference pool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrindersConfig {
    /// Number of Rayon worker threads (PRD G-01: default = CPU cores - 1).
    /// Zero means use the Rayon global pool (CPU-1 threads).
    #[serde(default)]
    pub worker_threads: usize,

    /// Maximum number of requests held in the per-model dispatch queue before
    /// back-pressure kicks in (PoolFull error).
    #[serde(default = "default_queue_depth")]
    pub queue_depth: usize,

    /// Dynamic batching window: how long (in milliseconds) to wait for
    /// additional requests before dispatching a batch (PRD G-05).
    /// 0 = disabled (process requests immediately).
    #[serde(default = "default_batch_window_ms")]
    pub batch_window_ms: u64,

    /// Maximum batch size per dispatch (PRD G-05).
    #[serde(default = "default_max_batch_size")]
    pub max_batch_size: usize,

    /// Models to register at startup.
    #[serde(default)]
    pub models: Vec<ModelConfig>,

    /// Extra key-value pairs for future extension.
    #[serde(default, flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

fn default_queue_depth() -> usize {
    1024
}

fn default_batch_window_ms() -> u64 {
    5
}

fn default_max_batch_size() -> usize {
    32
}

impl Default for GrindersConfig {
    fn default() -> Self {
        let cpus = num_cpus();
        Self {
            worker_threads: cpus.saturating_sub(1).max(1),
            queue_depth: default_queue_depth(),
            batch_window_ms: default_batch_window_ms(),
            max_batch_size: default_max_batch_size(),
            models: Vec::new(),
            extra: HashMap::new(),
        }
    }
}

/// Portable CPU count — falls back to 1 if detection fails.
fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

// ─────────────────────────────────────────────────────────────────────────────
// Starter model pack helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Returns a `Vec<ModelConfig>` for the full starter model pack (PRD §5.2.2).
///
/// All paths are relative to `model_dir`. Models that don't exist on disk
/// are still registered as cold entries; the registry will return
/// [`GrindersError::ModelNotFound`] until the file is present.
pub fn starter_pack(model_dir: &std::path::Path) -> Vec<ModelConfig> {
    vec![
        ModelConfig {
            model_id: "intent-classifier-v1".into(),
            path: model_dir.join("intent-classifier-v1.onnx"),
            runtime: ModelRuntime::Onnx,
            warm: true,
            timeout: Duration::from_secs(5),
            max_tokens: 0,
            description: "4-class intent routing for Sieve (~25 MB ONNX INT8)".into(),
        },
        ModelConfig {
            model_id: "ner-multilingual-v1".into(),
            path: model_dir.join("ner-multilingual-v1.onnx"),
            runtime: ModelRuntime::Onnx,
            warm: true,
            timeout: Duration::from_secs(5),
            max_tokens: 0,
            description: "Named entity recognition — person/org/location/date (~40 MB ONNX)".into(),
        },
        ModelConfig {
            model_id: "minilm-l6-v2".into(),
            path: model_dir.join("minilm-l6-v2.onnx"),
            runtime: ModelRuntime::Onnx,
            warm: true,
            timeout: Duration::from_secs(5),
            max_tokens: 0,
            description: "Sentence embeddings for Ledger and Sieve features (~25 MB ONNX)".into(),
        },
        ModelConfig {
            model_id: "phi3-mini-4k-Q4".into(),
            path: model_dir.join("phi3-mini-4k-Q4.gguf"),
            runtime: ModelRuntime::Gguf,
            warm: false, // 2.3 GB — load on demand
            timeout: Duration::from_secs(60),
            max_tokens: 128,
            description: "Local summarisation for Ledger compaction (~2.3 GB GGUF Q4)".into(),
        },
        ModelConfig {
            model_id: "anomaly-detector-v1".into(),
            path: model_dir.join("anomaly-detector-v1.onnx"),
            runtime: ModelRuntime::Onnx,
            warm: true,
            timeout: Duration::from_secs(5),
            max_tokens: 0,
            description: "Isolation forest for metric anomaly detection (~5 MB ONNX)".into(),
        },
    ]
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_has_at_least_one_worker() {
        let cfg = GrindersConfig::default();
        assert!(cfg.worker_threads >= 1, "need at least 1 worker thread");
    }

    #[test]
    fn starter_pack_returns_five_models() {
        let pack = starter_pack(std::path::Path::new("/tmp/models"));
        assert_eq!(pack.len(), 5);
    }

    #[test]
    fn starter_pack_phi3_is_cold() {
        let pack = starter_pack(std::path::Path::new("/tmp/models"));
        let phi3 = pack.iter().find(|m| m.model_id == "phi3-mini-4k-Q4").unwrap();
        assert!(!phi3.warm, "phi3 should be cold (2.3 GB)");
    }

    #[test]
    fn starter_pack_minilm_is_warm() {
        let pack = starter_pack(std::path::Path::new("/tmp/models"));
        let minilm = pack.iter().find(|m| m.model_id == "minilm-l6-v2").unwrap();
        assert!(minilm.warm);
    }
}
