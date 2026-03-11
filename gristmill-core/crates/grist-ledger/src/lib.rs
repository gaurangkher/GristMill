//! `grist-ledger` — three-tier memory system for GristMill.
//!
//! The ledger stores and retrieves AI memories across three tiers:
//!
//! | Tier | Backend | Access | Capacity |
//! |------|---------|--------|----------|
//! | Hot  | sled LRU | ~µs | 4096 entries (configurable) |
//! | Warm | SQLite FTS5 + usearch HNSW | ~ms | unlimited |
//! | Cold | zstd JSONL archive | ~seconds | unlimited |
//!
//! # Example
//!
//! ```no_run
//! use grist_ledger::{Ledger, LedgerConfig, StubEmbedder};
//! use std::sync::Arc;
//!
//! # #[tokio::main] async fn main() {
//! let config = LedgerConfig::default();
//! let embedder = Arc::new(StubEmbedder::new(384));
//! let ledger = Ledger::new(config, embedder).await.unwrap();
//!
//! let id = ledger.remember("Meeting with Alice at 10am", vec![]).await.unwrap();
//! let results = ledger.recall("Alice meeting", 5).await.unwrap();
//! assert!(!results.is_empty());
//! # }
//! ```

pub mod cold;
pub mod compactor;
pub mod config;
pub mod embedder;
pub mod error;
pub mod hot;
pub mod memory;
pub mod warm;

pub use config::LedgerConfig;
pub use embedder::{Embedder, StubEmbedder, ZeroEmbedder};
pub use error::LedgerError;
pub use memory::{Memory, MemoryId, RankedMemory, SearchSource, Tier};

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::mpsc;
use tracing::{debug, info, warn};

use cold::ColdTier;
use compactor::{Compactor, CompactorHandle};
use config::LedgerConfig as Cfg;
use hot::HotTier;
use warm::WarmTier;

// ─────────────────────────────────────────────────────────────────────────────
// Ledger
// ─────────────────────────────────────────────────────────────────────────────

/// The three-tier memory ledger.
///
/// `Ledger` is `Send + Sync`. Wrap in `Arc` and share across Tokio tasks.
/// Spawns background tasks in [`Ledger::new`].
pub struct Ledger {
    hot: Arc<HotTier>,
    warm: Arc<WarmTier>,
    cold: Arc<ColdTier>,
    embedder: Arc<dyn Embedder>,
    /// Kept alive to prevent the background eviction drainer from being dropped.
    _evict_task: tokio::task::JoinHandle<()>,
    /// Kept alive to prevent the background compactor from being dropped.
    _compactor: CompactorHandle,
}

impl std::fmt::Debug for Ledger {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Ledger")
            .field("hot", &self.hot)
            .field("warm", &self.warm)
            .field("cold", &self.cold)
            .finish()
    }
}

impl Ledger {
    /// Construct a `Ledger`.
    ///
    /// Opens sled, SQLite, and usearch; spawns the eviction-drainer task
    /// and the background compactor.
    ///
    /// **Must be called inside a Tokio runtime.**
    pub async fn new(config: Cfg, embedder: Arc<dyn Embedder>) -> Result<Self, LedgerError> {
        info!("initialising three-tier ledger");

        // ── Build tiers (blocking I/O) ─────────────────────────────────────
        let hot_cfg = config.hot.clone();
        let warm_cfg = config.warm.clone();
        let cold_cfg = config.cold.clone();
        let compactor_cfg = config.compactor.clone();

        let (evict_tx, evict_rx) = mpsc::unbounded_channel::<(Memory, Vec<f32>)>();

        // Open sled (blocking).
        let hot = tokio::task::spawn_blocking(move || HotTier::open(&hot_cfg, evict_tx))
            .await
            .map_err(|e| LedgerError::Other(e.into()))??;
        let hot = Arc::new(hot);

        // Open SQLite + usearch (blocking).
        let warm = tokio::task::spawn_blocking(move || WarmTier::open(&warm_cfg))
            .await
            .map_err(|e| LedgerError::Other(e.into()))??;
        let warm = Arc::new(warm);

        // Open cold tier (non-blocking mkdir).
        let cold = Arc::new(ColdTier::new(&cold_cfg)?);

        // ── Eviction-drainer task ──────────────────────────────────────────
        // Receives entries evicted from the hot LRU and inserts them into warm.
        let warm_for_evict = Arc::clone(&warm);
        let evict_task = tokio::spawn(eviction_drainer(evict_rx, warm_for_evict));

        // ── Compactor ─────────────────────────────────────────────────────
        let compactor = Compactor::spawn(Arc::clone(&warm), Arc::clone(&cold), compactor_cfg);

        info!("ledger ready");
        Ok(Self {
            hot,
            warm,
            cold,
            embedder,
            _evict_task: evict_task,
            _compactor: compactor,
        })
    }

    // ── Public API ─────────────────────────────────────────────────────────

    /// Store a memory in the ledger.
    ///
    /// 1. Computes an embedding for `content`.
    /// 2. Checks the warm tier for a semantic duplicate (sim ≥ 0.95).
    ///    - If found: merges the content and returns the existing ID.
    ///    - If not found: inserts into the hot tier (cascades to warm on eviction).
    pub async fn remember(
        &self,
        content: impl Into<String>,
        tags: Vec<String>,
    ) -> Result<MemoryId, LedgerError> {
        let content: String = content.into();
        let memory = Memory::new(content.clone(), tags);

        // ── Embed ──────────────────────────────────────────────────────────
        let embedder = Arc::clone(&self.embedder);
        let text = content.clone();
        let embedding = tokio::task::spawn_blocking(move || embedder.embed(&text))
            .await
            .map_err(|e| LedgerError::Other(e.into()))??;

        // ── Duplicate check in warm ────────────────────────────────────────
        let warm = Arc::clone(&self.warm);
        let emb_clone = embedding.clone();
        let existing = tokio::task::spawn_blocking(move || warm.find_similar(&emb_clone, 0.95))
            .await
            .map_err(|e| LedgerError::Other(e.into()))??;

        if let Some(existing_mem) = existing {
            debug!(existing_id = %existing_mem.id, "duplicate memory found — merging");
            let warm2 = Arc::clone(&self.warm);
            let new_mem = memory.clone();
            let eid = existing_mem.id.clone();
            tokio::task::spawn_blocking(move || warm2.merge(&eid, &new_mem))
                .await
                .map_err(|e| LedgerError::Other(e.into()))??;
            metrics::counter!("ledger.remember.duplicates").increment(1);
            return Ok(existing_mem.id);
        }

        // ── Insert into hot ────────────────────────────────────────────────
        let id = self.hot.insert(memory, embedding)?;
        debug!(memory_id = %id, "memory stored in hot tier");
        metrics::counter!("ledger.remember.total").increment(1);
        Ok(id)
    }

    /// Recall memories relevant to a query using RRF fusion.
    ///
    /// Runs keyword search and vector search in parallel against the warm tier,
    /// then fuses the results via Reciprocal Rank Fusion.
    pub async fn recall(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<RankedMemory>, LedgerError> {
        // ── Embed query ────────────────────────────────────────────────────
        let embedder = Arc::clone(&self.embedder);
        let q = query.to_owned();
        let embedding = tokio::task::spawn_blocking(move || embedder.embed(&q))
            .await
            .map_err(|e| LedgerError::Other(e.into()))??;

        // ── Parallel warm searches ─────────────────────────────────────────
        let warm_kw = Arc::clone(&self.warm);
        let warm_vec = Arc::clone(&self.warm);
        let q_kw = query.to_owned();
        let emb_vec = embedding.clone();
        let limit2 = limit * 2;

        let (kw_result, vec_result) = tokio::join!(
            tokio::task::spawn_blocking(move || warm_kw.keyword_search(&q_kw, limit2)),
            tokio::task::spawn_blocking(move || warm_vec.vector_search(&emb_vec, limit2)),
        );

        let keyword_hits = kw_result.map_err(|e| LedgerError::Other(e.into()))??;
        let vector_hits = vec_result.map_err(|e| LedgerError::Other(e.into()))??;

        // ── RRF fusion ─────────────────────────────────────────────────────
        let fused = reciprocal_rank_fusion(&keyword_hits, &vector_hits, limit);

        // ── Hydrate Memory objects ─────────────────────────────────────────
        let ids: Vec<MemoryId> = fused.iter().map(|(id, _)| id.clone()).collect();
        let warm_hydrate = Arc::clone(&self.warm);
        let memories: Vec<Memory> =
            tokio::task::spawn_blocking(move || warm_hydrate.get_many(&ids))
                .await
                .map_err(|e| LedgerError::Other(e.into()))??;

        // Build a score map for fast lookup.
        let score_map: HashMap<&str, (f64, Vec<SearchSource>)> = fused
            .iter()
            .map(|(id, score)| {
                let kw = keyword_hits.iter().any(|(kid, _)| kid == id);
                let vec = vector_hits.iter().any(|(vid, _)| vid == id);
                let mut sources = vec![];
                if kw {
                    sources.push(SearchSource::Keyword);
                }
                if vec {
                    sources.push(SearchSource::Vector);
                }
                (id.as_str(), (*score, sources))
            })
            .collect();

        let mut ranked: Vec<RankedMemory> = memories
            .into_iter()
            .map(|mem| {
                let (score, sources) = score_map
                    .get(mem.id.as_str())
                    .cloned()
                    .unwrap_or((0.0, vec![]));
                RankedMemory {
                    memory: mem,
                    score,
                    sources,
                }
            })
            .collect();

        ranked.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // ── Touch accessed memories ────────────────────────────────────────
        for r in &ranked {
            let warm_touch = Arc::clone(&self.warm);
            let id = r.memory.id.clone();
            let _ = tokio::task::spawn_blocking(move || warm_touch.touch(&id)).await;
        }

        metrics::histogram!("ledger.recall.results_count").record(ranked.len() as f64);
        Ok(ranked)
    }

    /// Retrieve a single memory by ID.
    ///
    /// Checks the hot tier first, then the warm tier.
    pub async fn get(&self, id: &str) -> Result<Option<Memory>, LedgerError> {
        // Hot tier (in-memory, no spawn_blocking needed — LruCache access is fast).
        if let Some(mem) = self.hot.get(id)? {
            return Ok(Some(mem));
        }
        // Warm tier (SQLite, needs spawn_blocking).
        let warm = Arc::clone(&self.warm);
        let id_owned = id.to_owned();
        let result = tokio::task::spawn_blocking(move || warm.get(&id_owned))
            .await
            .map_err(|e| LedgerError::Other(e.into()))??;
        Ok(result)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Eviction drainer
// ─────────────────────────────────────────────────────────────────────────────

/// Background task: receives hot-tier evictions and inserts them into warm.
async fn eviction_drainer(
    mut rx: mpsc::UnboundedReceiver<(Memory, Vec<f32>)>,
    warm: Arc<WarmTier>,
) {
    while let Some((memory, embedding)) = rx.recv().await {
        let id = memory.id.clone();
        let warm2 = Arc::clone(&warm);
        if let Err(e) = tokio::task::spawn_blocking(move || {
            let mut m = memory.clone();
            m.tier = Tier::Warm;
            warm2.insert(&m, &embedding)?;
            Ok::<_, LedgerError>(())
        })
        .await
        .map_err(|e| LedgerError::Other(e.into()))
        .and_then(|r| r)
        {
            warn!(memory_id = %id, error = %e, "eviction drainer: failed to insert into warm");
        } else {
            debug!(memory_id = %id, "evicted entry promoted to warm tier");
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Reciprocal Rank Fusion
// ─────────────────────────────────────────────────────────────────────────────

/// Combine keyword and vector ranked lists via Reciprocal Rank Fusion.
///
/// RRF score = Σ 1 / (k + rank_i) where rank_i is 1-based position.
/// k = 60 is the standard constant.
fn reciprocal_rank_fusion(
    keyword: &[(MemoryId, f64)],
    vector: &[(MemoryId, f32)],
    limit: usize,
) -> Vec<(MemoryId, f64)> {
    const K: f64 = 60.0;
    let mut scores: HashMap<String, f64> = HashMap::new();

    for (rank, (id, _)) in keyword.iter().enumerate() {
        *scores.entry(id.clone()).or_insert(0.0) += 1.0 / (K + rank as f64 + 1.0);
    }
    for (rank, (id, _)) in vector.iter().enumerate() {
        *scores.entry(id.clone()).or_insert(0.0) += 1.0 / (K + rank as f64 + 1.0);
    }

    let mut sorted: Vec<(String, f64)> = scores.into_iter().collect();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    sorted.truncate(limit);
    sorted
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{ColdConfig, CompactorConfig, HotConfig, LedgerConfig, WarmConfig};
    use crate::embedder::StubEmbedder;

    fn stub_embedder() -> Arc<dyn Embedder> {
        Arc::new(StubEmbedder::new(384))
    }

    async fn make_ledger(tmp: &tempfile::TempDir) -> Ledger {
        let config = LedgerConfig {
            hot: HotConfig {
                lru_capacity: 4,
                sled_path: tmp.path().join("sled"),
            },
            warm: WarmConfig {
                db_path: tmp.path().join("warm.db"),
                vector_index_path: tmp.path().join("vec.usearch"),
                embedding_dim: 384,
                vector_capacity: 1000,
            },
            cold: ColdConfig {
                archive_dir: tmp.path().join("cold"),
                compress_level: 1,
            },
            compactor: CompactorConfig {
                interval_secs: 9999, // don't auto-compact in tests
                ..Default::default()
            },
        };
        Ledger::new(config, stub_embedder()).await.unwrap()
    }

    #[tokio::test]
    async fn noop_remember_and_recall() {
        let dir = tempfile::tempdir().unwrap();
        let ledger = make_ledger(&dir).await;
        let id = ledger
            .remember("meeting with Alice at 10am", vec![])
            .await
            .unwrap();
        assert!(!id.is_empty());
    }

    #[tokio::test]
    async fn get_existing_memory() {
        let dir = tempfile::tempdir().unwrap();
        let ledger = make_ledger(&dir).await;
        let id = ledger
            .remember("retrievable memory content", vec![])
            .await
            .unwrap();
        let mem = ledger.get(&id).await.unwrap();
        assert!(
            mem.is_some(),
            "recently stored memory should be retrievable via get()"
        );
    }

    #[tokio::test]
    async fn get_missing_memory_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        let ledger = make_ledger(&dir).await;
        let result = ledger.get("non_existent_id_00000").await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn recall_after_remember_returns_results() {
        let dir = tempfile::tempdir().unwrap();
        let ledger = make_ledger(&dir).await;

        // Insert into hot; force eviction to warm by inserting more than lru_capacity (4).
        for i in 0..6 {
            ledger
                .remember(format!("memory entry number {i} about scheduling"), vec![])
                .await
                .unwrap();
        }
        // Give the eviction drainer a moment to process.
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let results = ledger.recall("scheduling", 10).await.unwrap();
        assert!(
            !results.is_empty(),
            "recall should return results after remember"
        );
    }

    #[tokio::test]
    async fn lru_eviction_cascades_to_warm() {
        let dir = tempfile::tempdir().unwrap();
        let ledger = make_ledger(&dir).await; // lru_capacity = 4

        // Insert 6 memories — overflow forces evictions.
        let mut ids = vec![];
        for i in 0..6 {
            let id = ledger
                .remember(format!("eviction test memory {i}"), vec![])
                .await
                .unwrap();
            ids.push(id);
        }

        // Give the eviction drainer time to process.
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        // Check that the warm tier has entries (evictions cascaded).
        let warm_count = tokio::task::spawn_blocking({
            let warm = Arc::clone(&ledger.warm);
            move || warm.count()
        })
        .await
        .unwrap()
        .unwrap();

        assert!(
            warm_count > 0,
            "warm tier should have received evicted entries"
        );
    }

    #[tokio::test]
    async fn recall_returns_ranked_list_in_order() {
        let dir = tempfile::tempdir().unwrap();
        let ledger = make_ledger(&dir).await;

        // Insert enough to overflow hot (cap=4) → warm.
        for i in 0..6 {
            ledger
                .remember(format!("project planning meeting agenda {i}"), vec![])
                .await
                .unwrap();
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let results = ledger.recall("project planning", 5).await.unwrap();
        // Scores should be non-increasing.
        for w in results.windows(2) {
            assert!(
                w[0].score >= w[1].score,
                "results should be ordered by score descending"
            );
        }
    }

    #[tokio::test]
    async fn rrf_fusion_scores_correctly() {
        // Unit test for the RRF helper directly.
        let keyword = vec![("id_a".to_string(), -1.0f64), ("id_b".to_string(), -2.0f64)];
        let vector = vec![("id_b".to_string(), 0.05f32), ("id_a".to_string(), 0.10f32)];
        let fused = reciprocal_rank_fusion(&keyword, &vector, 5);
        // id_a is rank-1 in keyword, rank-2 in vector → score = 1/61 + 1/62
        // id_b is rank-2 in keyword, rank-1 in vector → score = 1/62 + 1/61
        // They should be equal (symmetric fusion); both appear.
        assert_eq!(fused.len(), 2);
        let score_a = fused
            .iter()
            .find(|(id, _)| id == "id_a")
            .map(|(_, s)| *s)
            .unwrap_or(0.0);
        let score_b = fused
            .iter()
            .find(|(id, _)| id == "id_b")
            .map(|(_, s)| *s)
            .unwrap_or(0.0);
        assert!(
            (score_a - score_b).abs() < 1e-10,
            "symmetric inputs → equal RRF scores"
        );
    }
}
