//! Checkpoint store — persist and resume pipeline runs.
//!
//! A checkpoint is a JSON snapshot of the current run state written to
//! `<checkpoint_dir>/<run_id>.json` after each step or on completion.
//!
//! On resume, the scheduler reads the snapshot and skips steps whose result
//! is already recorded, replaying from the first incomplete step.
//!
//! The store is intentionally simple (flat file per run) to avoid introducing
//! a database dependency in the core crate.  For production workloads with
//! thousands of concurrent runs, a pluggable backend (e.g. SQLite via
//! `grist-ledger`) can be wired in through the `CheckpointBackend` trait.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

use crate::error::MillwrightError;
use crate::scheduler::StepResult;

// ─────────────────────────────────────────────────────────────────────────────
// Checkpoint data
// ─────────────────────────────────────────────────────────────────────────────

/// A full snapshot of a pipeline run at a point in time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunCheckpoint {
    /// Unique run identifier (ULID).
    pub run_id: String,
    /// Pipeline that produced this run.
    pub pipeline_id: String,
    /// Wall-clock timestamp when the run started (ms since UNIX epoch).
    pub started_at_ms: u64,
    /// Completed steps and their results (step_id → result).
    pub completed_steps: HashMap<String, StepResult>,
    /// Current status of the run.
    pub status: RunStatus,
}

/// Overall status of a pipeline run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunStatus {
    /// Run is actively executing.
    Running,
    /// All steps completed successfully.
    Completed,
    /// At least one step failed and the pipeline halted.
    Failed,
    /// The run was manually cancelled.
    Cancelled,
}

// ─────────────────────────────────────────────────────────────────────────────
// CheckpointStore
// ─────────────────────────────────────────────────────────────────────────────

/// Thread-safe checkpoint store backed by the local filesystem.
///
/// All writes are synchronous (using `std::fs`) to avoid losing checkpoints
/// under async cancellation.
#[derive(Clone)]
pub struct CheckpointStore {
    inner: Arc<Inner>,
}

struct Inner {
    dir: PathBuf,
    /// In-memory cache of recently loaded/written checkpoints.
    cache: Mutex<HashMap<String, RunCheckpoint>>,
}

impl CheckpointStore {
    /// Create a store that writes to `dir`.  The directory is created if it
    /// does not exist.
    pub fn open(dir: impl AsRef<Path>) -> Result<Self, MillwrightError> {
        let dir = dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&dir)?;
        Ok(Self {
            inner: Arc::new(Inner {
                dir,
                cache: Mutex::new(HashMap::new()),
            }),
        })
    }

    /// Create a no-op in-memory store (for `CheckpointStrategy::None` or tests).
    pub fn noop() -> Self {
        Self {
            inner: Arc::new(Inner {
                dir: PathBuf::from("/dev/null"),
                cache: Mutex::new(HashMap::new()),
            }),
        }
    }

    /// Persist `checkpoint` to disk.
    ///
    /// Write is atomic: we write to a `.tmp` file then `rename` it into place.
    pub fn save(&self, checkpoint: &RunCheckpoint) -> Result<(), MillwrightError> {
        if self.inner.dir == Path::new("/dev/null") {
            // noop store
            self.inner
                .cache
                .lock()
                .insert(checkpoint.run_id.clone(), checkpoint.clone());
            return Ok(());
        }

        let json = serde_json::to_string_pretty(checkpoint)
            .map_err(|e| MillwrightError::Serialization(e.to_string()))?;

        let path = self.path(&checkpoint.run_id);
        let tmp_path = path.with_extension("tmp");

        std::fs::write(&tmp_path, &json)?;
        std::fs::rename(&tmp_path, &path)?;

        debug!(run_id = checkpoint.run_id, path = %path.display(), "checkpoint saved");
        self.inner
            .cache
            .lock()
            .insert(checkpoint.run_id.clone(), checkpoint.clone());
        Ok(())
    }

    /// Load a checkpoint by `run_id`.  Checks the in-memory cache first.
    pub fn load(&self, run_id: &str) -> Result<RunCheckpoint, MillwrightError> {
        // Check cache first.
        if let Some(cp) = self.inner.cache.lock().get(run_id) {
            return Ok(cp.clone());
        }

        let path = self.path(run_id);
        let json = std::fs::read_to_string(&path)
            .map_err(|_| MillwrightError::RunNotFound(run_id.to_owned()))?;

        let cp: RunCheckpoint = serde_json::from_str(&json)
            .map_err(|e| MillwrightError::Serialization(e.to_string()))?;

        self.inner
            .cache
            .lock()
            .insert(run_id.to_owned(), cp.clone());
        Ok(cp)
    }

    /// List all run IDs that have checkpoint files.
    pub fn list_runs(&self) -> Vec<String> {
        // Return from cache for noop store.
        if self.inner.dir == Path::new("/dev/null") {
            return self.inner.cache.lock().keys().cloned().collect();
        }

        std::fs::read_dir(&self.inner.dir)
            .into_iter()
            .flatten()
            .filter_map(|entry| {
                let entry = entry.ok()?;
                let name = entry.file_name();
                let s = name.to_str()?;
                if s.ends_with(".json") && !s.ends_with(".tmp") {
                    Some(s.trim_end_matches(".json").to_owned())
                } else {
                    None
                }
            })
            .collect()
    }

    /// Delete the checkpoint file for `run_id`.
    pub fn delete(&self, run_id: &str) {
        self.inner.cache.lock().remove(run_id);
        let path = self.path(run_id);
        let _ = std::fs::remove_file(path);
        info!(run_id, "checkpoint deleted");
    }

    fn path(&self, run_id: &str) -> PathBuf {
        self.inner.dir.join(format!("{run_id}.json"))
    }
}

impl std::fmt::Debug for CheckpointStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CheckpointStore({})", self.inner.dir.display())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduler::StepOutcome;

    fn make_checkpoint(run_id: &str) -> RunCheckpoint {
        let mut completed = HashMap::new();
        completed.insert(
            "step-a".to_owned(),
            StepResult {
                step_id: "step-a".to_owned(),
                outcome: StepOutcome::Succeeded,
                output: serde_json::json!({ "result": "ok" }),
                attempts: 1,
                elapsed_ms: 42,
            },
        );
        RunCheckpoint {
            run_id: run_id.to_owned(),
            pipeline_id: "test-pipeline".to_owned(),
            started_at_ms: 0,
            completed_steps: completed,
            status: RunStatus::Running,
        }
    }

    #[test]
    fn noop_store_saves_and_loads() {
        let store = CheckpointStore::noop();
        let cp = make_checkpoint("run-001");
        store.save(&cp).unwrap();
        let loaded = store.load("run-001").unwrap();
        assert_eq!(loaded.run_id, "run-001");
        assert!(loaded.completed_steps.contains_key("step-a"));
    }

    #[test]
    fn noop_store_load_missing_returns_error() {
        let store = CheckpointStore::noop();
        let err = store.load("nonexistent").unwrap_err();
        assert!(matches!(err, MillwrightError::RunNotFound(_)));
    }

    #[test]
    fn file_store_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let store = CheckpointStore::open(dir.path()).unwrap();

        let cp = make_checkpoint("run-002");
        store.save(&cp).unwrap();

        let loaded = store.load("run-002").unwrap();
        assert_eq!(loaded.pipeline_id, "test-pipeline");
        assert_eq!(loaded.status, RunStatus::Running);
    }

    #[test]
    fn file_store_lists_runs() {
        let dir = tempfile::tempdir().unwrap();
        let store = CheckpointStore::open(dir.path()).unwrap();

        store.save(&make_checkpoint("r1")).unwrap();
        store.save(&make_checkpoint("r2")).unwrap();

        let mut runs = store.list_runs();
        runs.sort();
        assert_eq!(runs, vec!["r1", "r2"]);
    }

    #[test]
    fn file_store_delete_removes_checkpoint() {
        let dir = tempfile::tempdir().unwrap();
        let store = CheckpointStore::open(dir.path()).unwrap();

        store.save(&make_checkpoint("r-del")).unwrap();
        store.delete("r-del");
        assert!(store.load("r-del").is_err());
    }
}
