//! Cold-tier storage — zstd-compressed JSONL archives.
//!
//! Archives are stored as monthly files: `<archive_dir>/YYYY-MM.jsonl.zst`.
//! Each write appends one zstd frame containing a single JSON line.
//! The zstd decoder handles multi-frame files transparently on read.

use std::fs;
use std::io::{Read, Write};
use std::path::PathBuf;

use chrono::Utc;
use tracing::{debug, info};

use crate::config::ColdConfig;
use crate::error::LedgerError;
use crate::memory::Memory;

// ─────────────────────────────────────────────────────────────────────────────
// ColdTier
// ─────────────────────────────────────────────────────────────────────────────

/// Append-only compressed JSONL archive (cold storage tier).
///
/// `ColdTier` is `Send + Sync`.  All operations are synchronous blocking I/O
/// and must be called from `tokio::task::spawn_blocking`.
#[derive(Clone)]
pub struct ColdTier {
    config: ColdConfig,
}

impl std::fmt::Debug for ColdTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ColdTier({})", self.config.archive_dir.display())
    }
}

impl ColdTier {
    /// Open (or create) the cold tier at `config.archive_dir`.
    pub fn new(config: &ColdConfig) -> Result<Self, LedgerError> {
        fs::create_dir_all(&config.archive_dir)?;
        info!(archive_dir = %config.archive_dir.display(), "cold tier initialised");
        Ok(Self {
            config: config.clone(),
        })
    }

    /// Archive a memory to the current month's `.jsonl.zst` file.
    ///
    /// Each call appends one zstd frame (one JSON line).
    pub fn archive(&self, memory: &Memory) -> Result<(), LedgerError> {
        let path = self.archive_path_for_now();
        let file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)?;

        let mut encoder = zstd::stream::write::Encoder::new(file, self.config.compress_level)
            .map_err(|e| LedgerError::ColdTier(format!("zstd encoder: {e}")))?;

        let mut line = serde_json::to_string(memory)?;
        line.push('\n');
        encoder.write_all(line.as_bytes())?;
        encoder
            .finish()
            .map_err(|e| LedgerError::ColdTier(format!("zstd finish: {e}")))?;

        debug!(memory_id = %memory.id, path = %path.display(), "memory archived to cold");
        metrics::counter!("ledger.cold.archived").increment(1);
        Ok(())
    }

    /// List all archive file paths under `archive_dir`.
    pub fn list_archives(&self) -> Result<Vec<PathBuf>, LedgerError> {
        let mut paths: Vec<PathBuf> = fs::read_dir(&self.config.archive_dir)?
            .filter_map(|entry| {
                let entry = entry.ok()?;
                let name = entry.file_name();
                let s = name.to_string_lossy();
                if s.ends_with(".jsonl.zst") {
                    Some(entry.path())
                } else {
                    None
                }
            })
            .collect();
        paths.sort();
        Ok(paths)
    }

    /// Linear scan over all archives — returns memories whose content contains
    /// `query` (case-sensitive substring match).
    ///
    /// This is a best-effort / debug operation — it is intentionally slow.
    pub fn search(&self, query: &str, limit: usize) -> Result<Vec<Memory>, LedgerError> {
        let mut results = Vec::new();
        for path in self.list_archives()? {
            let file = match fs::File::open(&path) {
                Ok(f) => f,
                Err(_) => continue,
            };
            let mut decoder = zstd::stream::read::Decoder::new(file)
                .map_err(|e| LedgerError::ColdTier(format!("zstd decoder: {e}")))?;
            let mut contents = String::new();
            let _ = decoder.read_to_string(&mut contents);

            for line in contents.lines() {
                if line.is_empty() {
                    continue;
                }
                if let Ok(mem) = serde_json::from_str::<Memory>(line) {
                    if mem.content.contains(query) {
                        results.push(mem);
                        if results.len() >= limit {
                            return Ok(results);
                        }
                    }
                }
            }
        }
        Ok(results)
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    fn archive_path_for_now(&self) -> PathBuf {
        let month = Utc::now().format("%Y-%m").to_string();
        self.config.archive_dir.join(format!("{month}.jsonl.zst"))
    }

    /// Archive path for a specific `YYYY-MM` string (used in tests).
    #[allow(dead_code)]
    pub(crate) fn archive_path_for_month(&self, month: &str) -> PathBuf {
        self.config.archive_dir.join(format!("{month}.jsonl.zst"))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::Memory;

    fn make_cold(dir: &tempfile::TempDir) -> ColdTier {
        ColdTier::new(&ColdConfig {
            archive_dir: dir.path().to_path_buf(),
            compress_level: 1,
        })
        .unwrap()
    }

    #[test]
    fn cold_archive_and_list() {
        let dir = tempfile::tempdir().unwrap();
        let cold = make_cold(&dir);
        let m = Memory::new("cold memory content", vec![]);
        cold.archive(&m).unwrap();
        let archives = cold.list_archives().unwrap();
        assert_eq!(archives.len(), 1, "one archive file should exist");
    }

    #[test]
    fn cold_search_finds_content() {
        let dir = tempfile::tempdir().unwrap();
        let cold = make_cold(&dir);
        let m = Memory::new("unique_cold_test_phrase here", vec![]);
        cold.archive(&m).unwrap();
        let results = cold.search("unique_cold_test_phrase", 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, m.id);
    }

    #[test]
    fn cold_search_no_match_returns_empty() {
        let dir = tempfile::tempdir().unwrap();
        let cold = make_cold(&dir);
        let m = Memory::new("some content", vec![]);
        cold.archive(&m).unwrap();
        let results = cold.search("xyz_not_present_xyz", 10).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn cold_archive_multiple_memories() {
        let dir = tempfile::tempdir().unwrap();
        let cold = make_cold(&dir);
        for i in 0..5 {
            cold.archive(&Memory::new(format!("memory {i}"), vec![]))
                .unwrap();
        }
        let results = cold.search("memory", 10).unwrap();
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn cold_archive_two_months_creates_two_files() {
        let dir = tempfile::tempdir().unwrap();
        let cold = make_cold(&dir);
        // Write to two "month" files manually.
        let m1 = Memory::new("jan memory", vec![]);
        let m2 = Memory::new("feb memory", vec![]);
        // Directly archive to named month files.
        let path_jan = cold.archive_path_for_month("2024-01");
        let path_feb = cold.archive_path_for_month("2024-02");

        // Archive to jan.
        let f_jan = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path_jan)
            .unwrap();
        let mut enc = zstd::stream::write::Encoder::new(f_jan, 1).unwrap();
        enc.write_all((serde_json::to_string(&m1).unwrap() + "\n").as_bytes())
            .unwrap();
        enc.finish().unwrap();

        // Archive to feb.
        let f_feb = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path_feb)
            .unwrap();
        let mut enc = zstd::stream::write::Encoder::new(f_feb, 1).unwrap();
        enc.write_all((serde_json::to_string(&m2).unwrap() + "\n").as_bytes())
            .unwrap();
        enc.finish().unwrap();

        let archives = cold.list_archives().unwrap();
        assert_eq!(archives.len(), 2);
    }
}
