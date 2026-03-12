//! Configuration for the three-tier ledger.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

// ─────────────────────────────────────────────────────────────────────────────
// LedgerConfig (top-level)
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LedgerConfig {
    #[serde(default)]
    pub hot: HotConfig,
    #[serde(default)]
    pub warm: WarmConfig,
    #[serde(default)]
    pub cold: ColdConfig,
    #[serde(default)]
    pub compactor: CompactorConfig,
}

// ─────────────────────────────────────────────────────────────────────────────
// HotConfig
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HotConfig {
    /// Max number of entries kept in the in-memory LRU cache.
    /// At ~1-4KB per Memory, 4096 entries ≈ 4-16 MB RAM above sled.
    #[serde(default = "default_lru_capacity")]
    pub lru_capacity: usize,
    /// Directory for the sled database (persistent backing store).
    #[serde(default = "default_sled_path")]
    pub sled_path: PathBuf,
}

fn default_lru_capacity() -> usize {
    4096
}
fn default_sled_path() -> PathBuf {
    PathBuf::from("./data/ledger/hot")
}

impl Default for HotConfig {
    fn default() -> Self {
        Self {
            lru_capacity: default_lru_capacity(),
            sled_path: default_sled_path(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// WarmConfig
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WarmConfig {
    /// SQLite database file path.
    #[serde(default = "default_db_path")]
    pub db_path: PathBuf,
    /// usearch index file path (saved/loaded for persistence).
    #[serde(default = "default_vector_index_path")]
    pub vector_index_path: PathBuf,
    /// Embedding dimension — must match the embedder (384 for MiniLM-L6-v2).
    #[serde(default = "default_embedding_dim")]
    pub embedding_dim: usize,
    /// Pre-allocated capacity for the usearch index.
    #[serde(default = "default_vector_capacity")]
    pub vector_capacity: usize,
}

fn default_db_path() -> PathBuf {
    PathBuf::from("./data/ledger/warm.db")
}
fn default_vector_index_path() -> PathBuf {
    PathBuf::from("./data/ledger/vectors.usearch")
}
fn default_embedding_dim() -> usize {
    384
}
fn default_vector_capacity() -> usize {
    100_000
}

impl Default for WarmConfig {
    fn default() -> Self {
        Self {
            db_path: default_db_path(),
            vector_index_path: default_vector_index_path(),
            embedding_dim: default_embedding_dim(),
            vector_capacity: default_vector_capacity(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ColdConfig
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColdConfig {
    /// Directory where monthly `YYYY-MM.jsonl.zst` archive files are written.
    #[serde(default = "default_archive_dir")]
    pub archive_dir: PathBuf,
    /// zstd compression level (1–22; 3 is a good default).
    #[serde(default = "default_compress_level")]
    pub compress_level: i32,
}

fn default_archive_dir() -> PathBuf {
    PathBuf::from("./data/ledger/cold")
}
fn default_compress_level() -> i32 {
    3
}

impl Default for ColdConfig {
    fn default() -> Self {
        Self {
            archive_dir: default_archive_dir(),
            compress_level: default_compress_level(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CompactorConfig
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactorConfig {
    /// How often the background compactor runs (spec: 6h = 21600s).
    #[serde(default = "default_interval_secs")]
    pub interval_secs: u64,
    /// Cosine similarity threshold for deduplication clustering (spec: 0.90).
    #[serde(default = "default_similarity_threshold")]
    pub similarity_threshold: f32,
    /// Memories with more than this many whitespace tokens are summarized
    /// (if a summarizer model is configured; spec: 512).
    #[serde(default = "default_verbose_threshold_tokens")]
    pub verbose_threshold_tokens: usize,
    /// Days without access before a warm memory is demoted to cold (spec: 90).
    #[serde(default = "default_stale_days")]
    pub stale_days: u64,
}

fn default_interval_secs() -> u64 {
    6 * 3600
}
fn default_similarity_threshold() -> f32 {
    0.90
}
fn default_verbose_threshold_tokens() -> usize {
    512
}
fn default_stale_days() -> u64 {
    90
}

impl Default for CompactorConfig {
    fn default() -> Self {
        Self {
            interval_secs: default_interval_secs(),
            similarity_threshold: default_similarity_threshold(),
            verbose_threshold_tokens: default_verbose_threshold_tokens(),
            stale_days: default_stale_days(),
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
    fn default_config_has_expected_values() {
        let c = LedgerConfig::default();
        assert_eq!(c.hot.lru_capacity, 4096);
        assert_eq!(c.warm.embedding_dim, 384);
        assert_eq!(c.compactor.interval_secs, 21600);
        assert!((c.compactor.similarity_threshold - 0.90).abs() < 1e-6);
        assert_eq!(c.compactor.stale_days, 90);
        assert_eq!(c.cold.compress_level, 3);
    }

    #[test]
    fn config_json_roundtrip() {
        let c = LedgerConfig::default();
        let json = serde_json::to_string(&c).unwrap();
        let c2: LedgerConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(c.warm.vector_capacity, c2.warm.vector_capacity);
    }
}
