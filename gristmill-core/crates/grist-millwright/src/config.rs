//! Configuration for the Millwright DAG orchestrator.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Top-level config for the Millwright, deserialized from the `millwright:`
/// section of `~/.gristmill/config.yaml`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MillwrightConfig {
    /// Maximum number of steps running in parallel across all pipelines.
    /// Corresponds to the Semaphore permit count.  Default: CPU cores - 1.
    #[serde(default = "default_max_concurrency")]
    pub max_concurrency: usize,

    /// Default wall-clock timeout (ms) for an entire pipeline run.
    /// Individual steps can override this with `Step::timeout_ms`.
    #[serde(default = "default_timeout_ms")]
    pub default_timeout_ms: u64,

    /// Directory where checkpoint files are written.
    /// Each pipeline run gets its own `<run_id>.json` file.
    #[serde(default = "default_checkpoint_dir")]
    pub checkpoint_dir: PathBuf,
}

fn default_max_concurrency() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .saturating_sub(1)
        .max(1)
}

fn default_timeout_ms() -> u64 {
    30_000
}

fn default_checkpoint_dir() -> PathBuf {
    PathBuf::from("./checkpoints")
}

impl Default for MillwrightConfig {
    fn default() -> Self {
        Self {
            max_concurrency: default_max_concurrency(),
            default_timeout_ms: default_timeout_ms(),
            checkpoint_dir: default_checkpoint_dir(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_max_concurrency_is_at_least_one() {
        let cfg = MillwrightConfig::default();
        assert!(cfg.max_concurrency >= 1);
    }

    #[test]
    fn default_timeout_is_30_seconds() {
        let cfg = MillwrightConfig::default();
        assert_eq!(cfg.default_timeout_ms, 30_000);
    }

    #[test]
    fn config_round_trips_via_json() {
        let cfg = MillwrightConfig::default();
        let json = serde_json::to_string(&cfg).unwrap();
        let cfg2: MillwrightConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(cfg.max_concurrency, cfg2.max_concurrency);
        assert_eq!(cfg.default_timeout_ms, cfg2.default_timeout_ms);
    }
}
