//! Hot-tier storage — in-memory LRU cache backed by sled.
//!
//! The hot tier has two layers:
//! 1. [`lru::LruCache`] — O(1) in-memory read/write, bounded by `lru_capacity`.
//! 2. [`sled::Db`] — on-disk persistence; evicted LRU entries land here first
//!    before cascading to the warm tier.
//!
//! When the LRU evicts an entry it is:
//! - Written to sled (for persistence across restarts).
//! - Sent on `evict_tx` (warm-tier promotion channel).

use lru::LruCache;
use parking_lot::Mutex;
use tokio::sync::mpsc::UnboundedSender;
use tracing::{debug, warn};

use crate::config::HotConfig;
use crate::error::LedgerError;
use crate::memory::{now_ms, Memory, MemoryId, Tier};

// ─────────────────────────────────────────────────────────────────────────────
// HotTier
// ─────────────────────────────────────────────────────────────────────────────

/// Hot-tier (LRU + sled) storage.
///
/// `HotTier` is `Send + Sync` — all interior mutability is guarded by
/// [`parking_lot::Mutex`].
pub struct HotTier {
    lru: Mutex<LruCache<MemoryId, (Memory, Vec<f32>)>>,
    sled: sled::Db,
    evict_tx: UnboundedSender<(Memory, Vec<f32>)>,
    config: HotConfig,
}

impl std::fmt::Debug for HotTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "HotTier(lru_cap={})", self.config.lru_capacity)
    }
}

impl HotTier {
    /// Open the sled database and construct the hot tier.
    ///
    /// `evict_tx` receives entries evicted from the LRU so the caller can
    /// cascade them into the warm tier.
    ///
    /// **Must be called from inside `tokio::task::spawn_blocking`** or before
    /// the async runtime is fully engaged, because `sled::open` is blocking.
    pub fn open(
        config: &HotConfig,
        evict_tx: UnboundedSender<(Memory, Vec<f32>)>,
    ) -> Result<Self, LedgerError> {
        let sled = sled::open(&config.sled_path)
            .map_err(|e| LedgerError::HotTier(format!("sled open: {e}")))?;

        let cap = std::num::NonZeroUsize::new(config.lru_capacity.max(1)).unwrap();
        let lru = Mutex::new(LruCache::new(cap));

        Ok(Self {
            lru,
            sled,
            evict_tx,
            config: config.clone(),
        })
    }

    /// Insert a memory and its embedding into the hot tier.
    ///
    /// If the LRU is at capacity, the eldest entry is evicted — written to
    /// sled and sent on `evict_tx` for warm-tier promotion.
    pub fn insert(&self, mut memory: Memory, embedding: Vec<f32>) -> Result<MemoryId, LedgerError> {
        memory.tier = Tier::Hot;
        let id = memory.id.clone();

        let mut lru = self.lru.lock();
        // `push` returns the evicted entry when capacity is reached.
        if let Some((_evicted_id, evicted)) =
            lru.push(id.clone(), (memory.clone(), embedding.clone()))
        {
            // Persist to sled (best-effort; non-fatal if it fails).
            let bytes = serde_json::to_vec(&evicted.0)?;
            if let Err(e) = self.sled.insert(evicted.0.id.as_bytes(), bytes.as_slice()) {
                warn!(error = %e, "failed to persist evicted hot-tier entry to sled");
            }
            // Send for warm-tier promotion.
            let _ = self.evict_tx.send(evicted);
            metrics::counter!("ledger.hot.evictions").increment(1);
        }

        debug!(memory_id = %id, "memory inserted into hot tier");
        metrics::counter!("ledger.hot.inserts").increment(1);
        Ok(id)
    }

    /// Retrieve a memory by ID — checks LRU first, then sled.
    pub fn get(&self, id: &str) -> Result<Option<Memory>, LedgerError> {
        {
            let mut lru = self.lru.lock();
            if let Some((mem, _emb)) = lru.get(id) {
                return Ok(Some(mem.clone()));
            }
        }
        // Check sled.
        match self.sled.get(id.as_bytes()) {
            Ok(Some(bytes)) => {
                let mem: Memory = serde_json::from_slice(&bytes)?;
                Ok(Some(mem))
            }
            Ok(None) => Ok(None),
            Err(e) => Err(LedgerError::HotTier(format!("sled get: {e}"))),
        }
    }

    /// Remove a memory from both LRU and sled.
    pub fn remove(&self, id: &str) -> Result<(), LedgerError> {
        self.lru.lock().pop(id);
        self.sled
            .remove(id.as_bytes())
            .map_err(|e| LedgerError::HotTier(format!("sled remove: {e}")))?;
        Ok(())
    }

    /// Update `last_accessed_ms` for a memory in the LRU (if present).
    pub fn touch(&self, id: &str) {
        let mut lru = self.lru.lock();
        if let Some((mem, _)) = lru.get_mut(id) {
            mem.last_accessed_ms = now_ms();
        }
    }

    /// Number of entries currently in the in-memory LRU.
    pub fn lru_len(&self) -> usize {
        self.lru.lock().len()
    }

    /// Keyword search across the in-memory LRU.
    ///
    /// Each query word must appear (case-insensitive substring) in the memory's
    /// content or tags for the entry to match.  Results are ordered by
    /// `last_accessed_ms` descending (most-recently used first), then truncated
    /// to `limit`.
    ///
    /// This intentionally does **not** search the sled backing store — sled only
    /// holds entries that have already been evicted from the LRU and are waiting
    /// for warm-tier promotion, so they will be picked up by the warm search once
    /// the eviction drainer has processed them.
    pub fn keyword_search(&self, query: &str, limit: usize) -> Vec<(Memory, Vec<f32>)> {
        let terms: Vec<String> = query
            .split_whitespace()
            .map(|t| t.to_lowercase())
            .filter(|t| !t.is_empty())
            .collect();

        if terms.is_empty() || limit == 0 {
            return vec![];
        }

        let lru = self.lru.lock();
        let mut matches: Vec<(Memory, Vec<f32>)> = lru
            .iter()
            .filter(|(_, (mem, _))| {
                let haystack = format!(
                    "{} {}",
                    mem.content.to_lowercase(),
                    mem.tags.join(" ").to_lowercase()
                );
                terms.iter().any(|t| haystack.contains(t.as_str()))
            })
            .map(|(_, (mem, emb))| (mem.clone(), emb.clone()))
            .collect();

        // Most-recently used first.
        matches.sort_by_key(|b| std::cmp::Reverse(b.0.last_accessed_ms));
        matches.truncate(limit);
        matches
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use tokio::sync::mpsc;

    fn make_hot(
        tmp: &tempfile::TempDir,
        cap: usize,
    ) -> (Arc<HotTier>, mpsc::UnboundedReceiver<(Memory, Vec<f32>)>) {
        let (tx, rx) = mpsc::unbounded_channel();
        let config = HotConfig {
            lru_capacity: cap,
            sled_path: tmp.path().join("sled"),
        };
        let hot = Arc::new(HotTier::open(&config, tx).unwrap());
        (hot, rx)
    }

    #[test]
    fn hot_insert_and_get() {
        let dir = tempfile::tempdir().unwrap();
        let (hot, _rx) = make_hot(&dir, 16);
        let m = Memory::new("hello hot tier", vec![]);
        let id = hot.insert(m, vec![0.1f32; 384]).unwrap();
        let retrieved = hot.get(&id).unwrap().expect("should be Some");
        assert_eq!(retrieved.content, "hello hot tier");
    }

    #[test]
    fn hot_get_missing_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        let (hot, _rx) = make_hot(&dir, 16);
        assert!(hot.get("nonexistent").unwrap().is_none());
    }

    #[test]
    fn hot_eviction_sends_to_channel() {
        let dir = tempfile::tempdir().unwrap();
        let (hot, mut rx) = make_hot(&dir, 2); // tiny LRU

        // Insert 3 items — the 3rd insert evicts the 1st.
        let m1 = Memory::new("first", vec![]);
        let m2 = Memory::new("second", vec![]);
        let m3 = Memory::new("third", vec![]);
        hot.insert(m1.clone(), vec![0.0f32; 4]).unwrap();
        hot.insert(m2, vec![0.0f32; 4]).unwrap();
        hot.insert(m3, vec![0.0f32; 4]).unwrap();

        // The first entry should have been evicted.
        let evicted = rx.try_recv().expect("eviction event should be pending");
        assert_eq!(evicted.0.id, m1.id);
    }

    #[test]
    fn hot_remove_clears_entry() {
        let dir = tempfile::tempdir().unwrap();
        let (hot, _rx) = make_hot(&dir, 16);
        let m = Memory::new("to be removed", vec![]);
        let id = hot.insert(m, vec![]).unwrap();
        hot.remove(&id).unwrap();
        assert!(hot.get(&id).unwrap().is_none());
    }

    #[test]
    fn hot_lru_len_tracks_entries() {
        let dir = tempfile::tempdir().unwrap();
        let (hot, _rx) = make_hot(&dir, 16);
        assert_eq!(hot.lru_len(), 0);
        hot.insert(Memory::new("a", vec![]), vec![]).unwrap();
        hot.insert(Memory::new("b", vec![]), vec![]).unwrap();
        assert_eq!(hot.lru_len(), 2);
    }
}
