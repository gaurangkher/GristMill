//! Model registry — tracks warm (pre-loaded) and cold (on-demand) models.
//!
//! # Design
//!
//! The registry is a `DashMap<model_id, Arc<ModelEntry>>` that stores model
//! configuration alongside the live session (if loaded).  Each entry uses an
//! `ArcSwap<Option<Arc<GrindersSession>>>` for atomic hot-reload while
//! in-flight inference calls hold their own `Arc` reference to the old session.
//!
//! Concurrent cold-load attempts for the same model are serialised by a
//! `parking_lot::Mutex<()>` stored in the entry.
//!
//! # Warm vs Cold
//!
//! | State | Loaded at startup | Latency on first request |
//! |-------|------------------|--------------------------|
//! | Warm  | Yes              | <5ms (already in memory)  |
//! | Cold  | No               | <2s (disk → memory)       |

use std::sync::Arc;
use std::time::Instant;

use arc_swap::ArcSwap;
use dashmap::DashMap;
use parking_lot::Mutex;
use tracing::{debug, info, warn};

use crate::config::{ModelConfig, ModelRuntime};
use crate::error::GrindersError;
use crate::session::GrindersSession;

// ─────────────────────────────────────────────────────────────────────────────
// Registry entry
// ─────────────────────────────────────────────────────────────────────────────

/// State of a model in the registry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelState {
    /// Model is loaded and ready in memory.
    Warm,
    /// Model is registered but not yet loaded.
    Cold,
    /// Model failed to load; the reason is stored.
    Failed(String),
}

/// Internal registry entry.
///
/// `session` is `ArcSwap<Option<Arc<GrindersSession>>>`:
/// - `None`  → not yet loaded (cold).
/// - `Some`  → loaded; the inner `Arc` is returned to callers so they can
///   hold onto it while a concurrent hot-reload swaps in a new one.
pub(crate) struct ModelEntry {
    pub config: ModelConfig,
    /// Live session — atomically swappable via ArcSwap (hot-reload, G-06).
    pub session: ArcSwap<Option<Arc<GrindersSession>>>,
    /// Serialises concurrent cold-load attempts for this entry.
    pub load_lock: Mutex<()>,
    pub state: Mutex<ModelState>,
}

impl std::fmt::Debug for ModelEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelEntry")
            .field("model_id", &self.config.model_id)
            .field("state", &*self.state.lock())
            .finish()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Registry snapshot (for observability)
// ─────────────────────────────────────────────────────────────────────────────

/// A point-in-time snapshot of one model's status.
#[derive(Debug, Clone)]
pub struct ModelSnapshot {
    pub model_id: String,
    pub runtime: ModelRuntime,
    pub state: ModelState,
    pub warm: bool,
    pub path: std::path::PathBuf,
}

// ─────────────────────────────────────────────────────────────────────────────
// ModelRegistry
// ─────────────────────────────────────────────────────────────────────────────

/// Thread-safe model registry.
///
/// Construct with [`ModelRegistry::new`] and register models with
/// [`ModelRegistry::register`].  Warm models are loaded immediately;
/// cold models are loaded on first access via [`ModelRegistry::get_or_load`].
pub struct ModelRegistry {
    entries: DashMap<String, Arc<ModelEntry>>,
}

impl ModelRegistry {
    pub fn new() -> Self {
        Self {
            entries: DashMap::new(),
        }
    }

    /// Register a model configuration.
    ///
    /// If `config.warm` is true, the model is loaded immediately.
    /// Non-fatal if warm load fails — entry is marked `Failed`.
    pub fn register(&self, config: ModelConfig) -> Result<(), GrindersError> {
        let model_id = config.model_id.clone();
        let warm = config.warm;

        let entry = Arc::new(ModelEntry {
            config,
            session: ArcSwap::from_pointee(None),
            load_lock: Mutex::new(()),
            state: Mutex::new(ModelState::Cold),
        });

        self.entries.insert(model_id.clone(), Arc::clone(&entry));

        if warm {
            match load_and_store(&entry) {
                Ok(()) => {
                    info!(model_id, "warm model loaded");
                }
                Err(e) => {
                    warn!(model_id, error = %e, "warm model load failed; marked Failed");
                    *entry.state.lock() = ModelState::Failed(e.to_string());
                }
            }
        }

        Ok(())
    }

    /// Return the live session for `model_id`, loading it first if cold.
    ///
    /// PRD G-04: warm → <5ms, cold first access → <2s.
    pub fn get_or_load(&self, model_id: &str) -> Result<Arc<GrindersSession>, GrindersError> {
        let entry = self
            .entries
            .get(model_id)
            .map(|e| Arc::clone(&*e))
            .ok_or_else(|| GrindersError::ModelNotFound(model_id.to_owned()))?;

        // Fast path: already loaded.
        {
            let guard = entry.session.load();
            if let Some(session_arc) = guard.as_ref() {
                return Ok(Arc::clone(session_arc));
            }
        }

        // Slow path: acquire load lock and re-check (double-checked locking).
        let _guard = entry.load_lock.lock();
        {
            let guard = entry.session.load();
            if let Some(session_arc) = guard.as_ref() {
                return Ok(Arc::clone(session_arc));
            }
        }

        // Still not loaded — load it now.
        load_and_store(&entry)?;

        // Retrieve the session that was just stored.
        let guard = entry.session.load();
        // guard: Guard<Arc<Option<Arc<GrindersSession>>>>
        // .as_ref(): &Option<Arc<GrindersSession>>
        // .as_ref(): Option<&Arc<GrindersSession>>
        // .map(|s| Arc::clone(s)): Option<Arc<GrindersSession>>
        guard
            .as_ref()
            .as_ref()
            .map(Arc::clone)
            .ok_or_else(|| GrindersError::ModelLoadFailed {
                model_id: model_id.to_owned(),
                reason: "session is None immediately after load (bug)".into(),
            })
    }

    /// Hot-reload a model in-place without dropping in-flight requests (G-06).
    ///
    /// In-flight callers hold their own `Arc` to the old session and complete
    /// normally.  Subsequent `get_or_load` calls return the new session.
    pub fn hot_reload(&self, model_id: &str) -> Result<(), GrindersError> {
        let entry = self
            .entries
            .get(model_id)
            .map(|e| Arc::clone(&*e))
            .ok_or_else(|| GrindersError::ModelNotFound(model_id.to_owned()))?;

        info!(model_id, "initiating hot-reload");
        let t0 = Instant::now();

        let _guard = entry.load_lock.lock();
        let new_session = load_session(&entry.config)?;
        entry.session.store(Arc::new(Some(Arc::new(new_session))));
        *entry.state.lock() = ModelState::Warm;

        let elapsed_ms = t0.elapsed().as_millis();
        info!(model_id, elapsed_ms, "hot-reload complete");
        metrics::counter!("grinders.registry.hot_reload").increment(1);
        Ok(())
    }

    /// Unload a model from memory (keeps the entry, moves to Cold state).
    pub fn evict(&self, model_id: &str) {
        if let Some(entry) = self.entries.get(model_id) {
            entry.session.store(Arc::new(None));
            *entry.state.lock() = ModelState::Cold;
            debug!(model_id, "model evicted");
        }
    }

    /// Return a snapshot of all registered models.
    pub fn snapshot(&self) -> Vec<ModelSnapshot> {
        self.entries
            .iter()
            .map(|kv| {
                let e = kv.value();
                ModelSnapshot {
                    model_id: e.config.model_id.clone(),
                    runtime: e.config.runtime.clone(),
                    state: e.state.lock().clone(),
                    warm: e.config.warm,
                    path: e.config.path.clone(),
                }
            })
            .collect()
    }

    /// Number of warm (loaded) models.
    pub fn warm_count(&self) -> usize {
        self.entries
            .iter()
            .filter(|kv| matches!(*kv.state.lock(), ModelState::Warm))
            .count()
    }

    /// Total number of registered models (warm + cold + failed).
    pub fn total_count(&self) -> usize {
        self.entries.len()
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Private helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Load the session for `entry` and store it atomically.
fn load_and_store(entry: &ModelEntry) -> Result<(), GrindersError> {
    let t0 = Instant::now();
    let session = load_session(&entry.config)?;
    entry.session.store(Arc::new(Some(Arc::new(session))));
    *entry.state.lock() = ModelState::Warm;
    let elapsed_ms = t0.elapsed().as_millis();
    debug!(model_id = entry.config.model_id, elapsed_ms, "model loaded");
    metrics::counter!("grinders.registry.loads").increment(1);
    Ok(())
}

/// Dispatch to the correct runtime loader.
fn load_session(config: &ModelConfig) -> Result<GrindersSession, GrindersError> {
    match config.runtime {
        ModelRuntime::Onnx => crate::onnx::load_onnx_session(config),
        ModelRuntime::Gguf => crate::gguf::load_gguf_session(config),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{ModelConfig, ModelRuntime};
    use std::time::Duration;

    fn dummy_config(id: &str, warm: bool) -> ModelConfig {
        ModelConfig {
            model_id: id.to_string(),
            path: std::path::PathBuf::from(format!("/nonexistent/{id}.onnx")),
            runtime: ModelRuntime::Onnx,
            warm,
            timeout: Duration::from_secs(5),
            max_tokens: 0,
            description: "test model".into(),
        }
    }

    #[test]
    fn registry_starts_empty() {
        let r = ModelRegistry::new();
        assert_eq!(r.total_count(), 0);
        assert_eq!(r.warm_count(), 0);
    }

    #[test]
    fn cold_model_registered_ok() {
        let r = ModelRegistry::new();
        r.register(dummy_config("test-cold", false)).unwrap();
        assert_eq!(r.total_count(), 1);
        assert_eq!(r.warm_count(), 0);
    }

    #[test]
    fn warm_model_with_missing_file_marked_failed_or_stubbed() {
        let r = ModelRegistry::new();
        r.register(dummy_config("test-warm-missing", true)).unwrap();
        let snaps = r.snapshot();
        let snap = snaps
            .iter()
            .find(|s| s.model_id == "test-warm-missing")
            .unwrap();
        // Without the `onnx` feature the loader returns a Stub → Warm.
        // With the `onnx` feature but missing file → Failed.
        assert!(
            matches!(snap.state, ModelState::Failed(_) | ModelState::Warm),
            "state should be Failed or Warm, got {:?}",
            snap.state
        );
    }

    #[test]
    fn get_or_load_unknown_model_returns_not_found() {
        let r = ModelRegistry::new();
        let err = r.get_or_load("does-not-exist").unwrap_err();
        assert!(matches!(err, GrindersError::ModelNotFound(_)));
    }

    #[test]
    fn snapshot_contains_all_registered() {
        let r = ModelRegistry::new();
        r.register(dummy_config("m1", false)).unwrap();
        r.register(dummy_config("m2", false)).unwrap();
        let snaps = r.snapshot();
        assert_eq!(snaps.len(), 2);
    }

    #[test]
    fn evict_moves_model_to_cold() {
        let r = ModelRegistry::new();
        r.register(dummy_config("m-evict", false)).unwrap();
        r.evict("m-evict");
        let snaps = r.snapshot();
        let snap = snaps.iter().find(|s| s.model_id == "m-evict").unwrap();
        assert_eq!(snap.state, ModelState::Cold);
    }

    #[test]
    fn get_or_load_cold_model_loads_stub() {
        let r = ModelRegistry::new();
        r.register(dummy_config("m-cold", false)).unwrap();
        // Without `onnx` feature this will succeed with a stub session.
        let session = r.get_or_load("m-cold").unwrap();
        assert_eq!(session.model_id, "m-cold");
        // Now the model should be warm.
        assert_eq!(r.warm_count(), 1);
    }
}
