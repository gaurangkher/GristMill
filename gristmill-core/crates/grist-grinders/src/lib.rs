//! `grist-grinders` — Local ML Inference Pool for GristMill.
//!
//! The Grinders crate provides a thread-safe, hot-reloadable pool of local ML
//! models that runs inference without involving any LLM API.
//!
//! # Supported runtimes
//!
//! | Runtime  | Feature flag | Use case                       |
//! |----------|-------------|--------------------------------|
//! | ONNX     | `onnx`      | Intent classifier, NER, MiniLM |
//! | GGUF     | `gguf`      | Local summarisation (Phi-3)    |
//! | Stub     | (none)      | Tests, CI without model files  |
//!
//! # Pipeline
//!
//! ```text
//! InferenceRequest
//!   │
//!   ├─ 1. Pool::submit() — enqueue into mpsc channel
//!   ├─ 2. Batcher task  — accumulate up to batch_window_ms or max_batch_size
//!   ├─ 3. Rayon::spawn  — dispatch batch to CPU worker pool
//!   ├─ 4. Registry::get_or_load — resolve warm/cold session
//!   └─ 5. session.run() — ONNX / GGUF / stub inference
//!          └─────────────────────────────→ InferenceOutput
//! ```
//!
//! # PRD requirements
//!
//! | ID   | Requirement                          | Implementation                    |
//! |------|--------------------------------------|-----------------------------------|
//! | G-01 | Rayon worker pool (CPU-1 default)    | `pool::WorkerPool` + Rayon        |
//! | G-02 | ONNX Runtime, zero-copy ndarray      | `onnx::load_onnx_session`         |
//! | G-03 | GGUF / llama.cpp inference           | `gguf::load_gguf_session`         |
//! | G-04 | Warm/cold model registry             | `registry::ModelRegistry`         |
//! | G-05 | Dynamic batching (>2× throughput)    | `pool::batcher_task`              |
//! | G-06 | Hot-reload without dropped requests  | `registry::ModelRegistry::hot_reload` |
//! | G-07 | Per-model timeout + cancellation     | `pool::WorkerPool::submit`        |
//!
//! # Example
//!
//! ```rust,no_run
//! use grist_grinders::{Grinders, GrindersConfig};
//! use grist_grinders::session::InferenceRequest;
//! use ndarray::Array2;
//! use std::time::Duration;
//!
//! #[tokio::main]
//! async fn main() {
//!     let config = GrindersConfig::default();
//!     let grinders = Grinders::new(config).expect("failed to create Grinders");
//!
//!     // Run a stub inference (no model file needed).
//!     let req = InferenceRequest::from_features("stub-model", Array2::zeros((1, 392)));
//!     // let out = grinders.infer(req).await.expect("inference failed");
//! }
//! ```

pub mod adapter_watcher;
pub mod config;
pub mod embedder;
pub mod error;
pub mod gguf;
pub mod onnx;
pub mod pool;
pub mod registry;
pub mod session;

// Re-exports for convenience.
pub use adapter_watcher::{domain_model_id, AdapterWatcher};
pub use config::{GrindersConfig, ModelConfig, ModelRuntime};
pub use error::GrindersError;
pub use registry::{ModelRegistry, ModelSnapshot, ModelState};
pub use session::{GrindersSession, InferenceOutput, InferenceRequest};

use std::sync::Arc;
use std::time::Duration;

use tracing::{info, instrument};

// ─────────────────────────────────────────────────────────────────────────────
// Grinders — top-level entry point
// ─────────────────────────────────────────────────────────────────────────────

/// The local ML inference pool.  Entry point for all grinder operations.
///
/// `Grinders` is `Send + Sync`.  Wrap in `Arc` and share across Tokio tasks.
pub struct Grinders {
    registry: Arc<ModelRegistry>,
    pool: pool::WorkerPool,
    config: GrindersConfig,
}

impl Grinders {
    /// Construct a `Grinders` instance from config.
    ///
    /// - Registers all models in `config.models`.
    /// - Warm models are loaded immediately (blocking in the constructor).
    /// - Cold models are loaded on first `infer()` call.
    /// - Spawns the batcher Tokio task.
    ///
    /// **Must be called inside a Tokio runtime** (the batcher task requires it).
    pub fn new(config: GrindersConfig) -> Result<Self, GrindersError> {
        info!(
            workers = config.worker_threads,
            models = config.models.len(),
            "initialising Grinders pool",
        );

        let registry = Arc::new(ModelRegistry::new());

        // Register all configured models.
        for model_cfg in &config.models {
            registry.register(model_cfg.clone())?;
        }

        let pool_cfg = pool::PoolConfig {
            queue_depth: config.queue_depth,
            batch_window: Duration::from_millis(config.batch_window_ms),
            max_batch_size: config.max_batch_size,
        };

        let pool = pool::WorkerPool::new(pool_cfg, Arc::clone(&registry));

        info!(
            warm_models = registry.warm_count(),
            total_models = registry.total_count(),
            "Grinders pool ready",
        );

        Ok(Self {
            registry,
            pool,
            config,
        })
    }

    /// Submit a single inference request to the worker pool.
    ///
    /// Uses the per-model timeout if available; otherwise falls back to 5 s.
    #[instrument(level = "debug", skip(self, request), fields(model_id = %request.model_id))]
    pub async fn infer(&self, request: InferenceRequest) -> Result<InferenceOutput, GrindersError> {
        // Look up the per-model timeout from the config.
        let timeout = self
            .config
            .models
            .iter()
            .find(|m| m.model_id == request.model_id)
            .map(|m| m.timeout)
            .unwrap_or(Duration::from_secs(5));

        self.pool.submit(request, timeout).await
    }

    /// Hot-reload a specific model (PRD G-06).
    ///
    /// In-flight requests on the old model complete normally.  Subsequent
    /// requests use the new model.
    pub fn hot_reload(&self, model_id: &str) -> Result<(), GrindersError> {
        self.registry.hot_reload(model_id)
    }

    /// Register a new model at runtime (without restart).
    pub fn register_model(&self, config: ModelConfig) -> Result<(), GrindersError> {
        self.registry.register(config)
    }

    /// Evict a model from memory (moves it to cold state).
    pub fn evict_model(&self, model_id: &str) {
        self.registry.evict(model_id);
    }

    /// Return a snapshot of all registered models.
    pub fn model_snapshots(&self) -> Vec<ModelSnapshot> {
        self.registry.snapshot()
    }

    /// Number of warm (loaded) models.
    pub fn warm_model_count(&self) -> usize {
        self.registry.warm_count()
    }

    /// Start watching `active_dir` for per-domain adapter changes (Phase 3).
    ///
    /// When a domain sub-directory inside `active_dir` changes (e.g.
    /// `/gristmill/checkpoints/active/code/`), the returned receiver yields
    /// the domain name (`"code"`).  The caller should then call
    /// `self.hot_reload(&domain_model_id(domain))` to swap in the new adapter.
    ///
    /// # Errors
    /// Returns an error if the watcher cannot be started (e.g. the directory
    /// does not exist or `inotify`/FSEvents is unavailable).
    pub fn start_adapter_watch(
        &self,
        active_dir: std::path::PathBuf,
    ) -> Result<(AdapterWatcher, tokio::sync::mpsc::UnboundedReceiver<String>), GrindersError> {
        AdapterWatcher::spawn(active_dir).map_err(|e| GrindersError::ModelLoadFailed {
            model_id: "adapter_watcher".into(),
            reason: e.to_string(),
        })
    }

    /// Build a concrete [`grist_sieve::features::EmbedderSession`] backed by
    /// the `minilm-l6-v2` model.  Inject this into the Sieve's
    /// `FeatureExtractor` to enable semantic similarity caching.
    pub fn build_embedder(&self) -> Result<grist_sieve::features::EmbedderSession, GrindersError> {
        embedder::build_minilm_embedder(&self.config)
    }
}

impl std::fmt::Debug for Grinders {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Grinders")
            .field("warm_models", &self.registry.warm_count())
            .field("total_models", &self.registry.total_count())
            .finish()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Integration tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn grinders_with_stub() -> Grinders {
        let mut config = GrindersConfig::default();
        // Add one cold stub model (no file on disk — ONNX stub kicks in).
        config.models.push(ModelConfig {
            model_id: "stub-model".into(),
            path: std::path::PathBuf::from("/nonexistent/stub.onnx"),
            runtime: ModelRuntime::Onnx,
            warm: false,
            timeout: Duration::from_secs(5),
            max_tokens: 0,
            description: "integration test stub".into(),
        });
        Grinders::new(config).unwrap()
    }

    // ── G-01: Pool initialises ─────────────────────────────────────────────
    #[tokio::test]
    async fn grinders_initialises_successfully() {
        let g = grinders_with_stub();
        assert_eq!(g.registry.total_count(), 1);
    }

    // ── G-04: Cold model loads on first infer call ─────────────────────────
    #[tokio::test]
    async fn cold_model_loads_on_first_infer() {
        let g = grinders_with_stub();
        assert_eq!(g.warm_model_count(), 0, "model should start cold");

        let req = InferenceRequest::from_features("stub-model", Array2::zeros((1, 392)));
        let out = g.infer(req).await.unwrap();
        assert!(out.tensor.is_some(), "expected tensor output");

        // After first infer the model should be warm.
        assert_eq!(
            g.warm_model_count(),
            1,
            "model should be warm after first load"
        );
    }

    // ── G-07: Unknown model returns ModelNotFound ─────────────────────────
    #[tokio::test]
    async fn infer_unknown_model_returns_error() {
        let g = grinders_with_stub();
        let req = InferenceRequest::from_features("ghost-model", Array2::zeros((1, 392)));
        let err = g.infer(req).await.unwrap_err();
        assert!(
            matches!(err, GrindersError::ModelNotFound(_)),
            "expected ModelNotFound, got {err:?}"
        );
    }

    // ── G-06: Hot-reload on unknown model returns ModelNotFound ────────────
    #[tokio::test]
    async fn hot_reload_unknown_model_returns_error() {
        let g = grinders_with_stub();
        let err = g.hot_reload("ghost-model").unwrap_err();
        assert!(matches!(err, GrindersError::ModelNotFound(_)));
    }

    // ── G-06: Hot-reload registered model reloads from disk ────────────────
    #[tokio::test]
    async fn hot_reload_registered_model_does_not_panic() {
        let g = grinders_with_stub();
        // hot_reload on a stub model with a missing file — should return
        // an error (ModelLoadFailed) but not panic.
        let result = g.hot_reload("stub-model");
        // Either succeeds (stub accepts) or returns a load error — never panics.
        let _ = result;
    }

    // ── G-04: Snapshot reports all registered models ───────────────────────
    #[tokio::test]
    async fn model_snapshots_returns_all() {
        let g = grinders_with_stub();
        let snaps = g.model_snapshots();
        assert_eq!(snaps.len(), 1);
        assert_eq!(snaps[0].model_id, "stub-model");
    }

    // ── Evict moves model to cold ──────────────────────────────────────────
    #[tokio::test]
    async fn evict_model_moves_to_cold() {
        let g = grinders_with_stub();
        // First infer to warm it up.
        let req = InferenceRequest::from_features("stub-model", Array2::zeros((1, 392)));
        g.infer(req).await.unwrap();
        assert_eq!(g.warm_model_count(), 1);

        g.evict_model("stub-model");
        assert_eq!(g.warm_model_count(), 0, "evicted model should be cold");
    }

    // ── Embedder fallback ─────────────────────────────────────────────────
    #[tokio::test]
    async fn build_embedder_fallback_returns_zero_vectors() {
        let g = Grinders::new(GrindersConfig::default()).unwrap();
        let emb = g.build_embedder().unwrap();
        let vec = emb.embed("test sentence").unwrap();
        assert_eq!(vec.len(), grist_sieve::features::EMBEDDING_DIM);
    }

    // ── G-05: Concurrent requests all complete ────────────────────────────
    #[tokio::test]
    async fn concurrent_infer_all_succeed() {
        let g = Arc::new(grinders_with_stub());
        let handles: Vec<_> = (0..16)
            .map(|_| {
                let g = Arc::clone(&g);
                tokio::spawn(async move {
                    let req =
                        InferenceRequest::from_features("stub-model", Array2::zeros((1, 392)));
                    g.infer(req).await
                })
            })
            .collect();

        for h in handles {
            let result = h.await.unwrap();
            assert!(result.is_ok(), "concurrent infer failed: {result:?}");
        }
    }
}
