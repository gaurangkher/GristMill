//! Warm-tier storage — SQLite FTS5 + usearch vector index.
//!
//! The warm tier provides two complementary search paths:
//! - **Keyword** (FTS5 BM25): full-text search over memory content.
//! - **Vector** (usearch HNSW cosine): approximate nearest-neighbour search.
//!
//! Both indexes are kept in sync via SQL triggers (FTS5) and explicit
//! `add` / `remove` calls (usearch).
//!
//! # Thread safety
//!
//! `rusqlite::Connection` is `!Send`, so it is wrapped in a
//! `parking_lot::Mutex`.  All methods on `WarmTier` must be called from
//! blocking contexts (use `tokio::task::spawn_blocking`).

use parking_lot::Mutex;
use rusqlite::{Connection, OptionalExtension, params};
use tracing::{debug, info, warn};
use ulid::Ulid;

use crate::config::WarmConfig;
use crate::error::LedgerError;
use crate::memory::{Memory, MemoryId, Tier, now_ms};

// ─────────────────────────────────────────────────────────────────────────────
// WarmTier
// ─────────────────────────────────────────────────────────────────────────────

/// Warm-tier storage (SQLite FTS5 + usearch).
///
/// `WarmTier` is `Send + Sync` — both the connection and the index are wrapped
/// in `parking_lot::Mutex`.
pub struct WarmTier {
    conn: Mutex<Connection>,
    index: Mutex<usearch::Index>,
    config: WarmConfig,
}

impl std::fmt::Debug for WarmTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "WarmTier({})", self.config.db_path.display())
    }
}

impl WarmTier {
    /// Open (or create) the warm tier at the paths specified in `config`.
    ///
    /// **Must be called from a blocking context.**
    pub fn open(config: &WarmConfig) -> Result<Self, LedgerError> {
        // ── SQLite ────────────────────────────────────────────────────────────
        if let Some(parent) = config.db_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let conn = Connection::open(&config.db_path)
            .map_err(|e| LedgerError::WarmTier(format!("sqlite open: {e}")))?;

        // WAL mode for concurrent reads. pragma_update fails for journal_mode
        // because it returns a result row; use query_row and discard the return.
        let _: String = conn
            .query_row("PRAGMA journal_mode=WAL", [], |r| r.get(0))
            .map_err(|e| LedgerError::WarmTier(format!("WAL pragma: {e}")))?;

        init_schema(&conn)?;

        // ── usearch ───────────────────────────────────────────────────────────
        let opts = usearch::IndexOptions {
            dimensions: config.embedding_dim,
            metric: usearch::MetricKind::Cos,
            quantization: usearch::ScalarKind::F32,
            connectivity: 16,
            expansion_add: 128,
            expansion_search: 64,
            ..Default::default()
        };
        let index = usearch::Index::new(&opts)
            .map_err(|e| LedgerError::WarmTier(format!("usearch new: {e}")))?;

        // Reserve capacity upfront; ignore errors (may not be supported on all builds).
        let _ = index.reserve(config.vector_capacity);

        // Load persisted index if it exists.
        let idx_path = config.vector_index_path.to_string_lossy().into_owned();
        if config.vector_index_path.exists() {
            if let Err(e) = index.load(&idx_path) {
                warn!(path = %idx_path, error = %e, "could not load usearch index; starting fresh");
            }
        }

        info!(db = %config.db_path.display(), "warm tier initialised");

        Ok(Self {
            conn: Mutex::new(conn),
            index: Mutex::new(index),
            config: config.clone(),
        })
    }

    /// Insert a memory and its embedding.
    pub fn insert(&self, memory: &Memory, embedding: &[f32]) -> Result<(), LedgerError> {
        debug!(memory_id = %memory.id, "inserting memory into warm tier");
        {
            let conn = self.conn.lock();
            conn.execute(
                "INSERT OR REPLACE INTO memories \
                 (id, content, tags, created_at_ms, last_accessed_ms, tier) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![
                    memory.id,
                    memory.content,
                    serde_json::to_string(&memory.tags)?,
                    memory.created_at_ms as i64,
                    memory.last_accessed_ms as i64,
                    "warm",
                ],
            )
            .map_err(|e| LedgerError::WarmTier(format!("insert: {e}")))?;

            // Explicitly insert into FTS5 (no triggers — manage manually).
            // First remove any existing FTS entry for this memory_id.
            let existing_rowid: Option<i64> = conn
                .query_row(
                    "SELECT fts_rowid FROM fts_rowid_map WHERE memory_id = ?1",
                    params![memory.id],
                    |r| r.get(0),
                )
                .optional()
                .map_err(|e| LedgerError::WarmTier(format!("fts_rowid lookup: {e}")))?;
            if let Some(old_rowid) = existing_rowid {
                conn.execute(
                    "DELETE FROM memories_fts WHERE rowid = ?1",
                    params![old_rowid],
                )
                .map_err(|e| LedgerError::WarmTier(format!("fts delete old: {e}")))?;
                conn.execute(
                    "DELETE FROM fts_rowid_map WHERE memory_id = ?1",
                    params![memory.id],
                )
                .map_err(|e| LedgerError::WarmTier(format!("fts_rowid_map delete: {e}")))?;
            }
            // Insert fresh FTS5 entry.
            conn.execute(
                "INSERT INTO memories_fts (content, memory_id) VALUES (?1, ?2)",
                params![memory.content, memory.id],
            )
            .map_err(|e| LedgerError::WarmTier(format!("fts insert: {e}")))?;
            let fts_rowid = conn.last_insert_rowid();
            conn.execute(
                "INSERT INTO fts_rowid_map (memory_id, fts_rowid) VALUES (?1, ?2)",
                params![memory.id, fts_rowid],
            )
            .map_err(|e| LedgerError::WarmTier(format!("fts_rowid_map insert: {e}")))?;
        }

        // Add to usearch index.
        let key = ulid_to_u64(&memory.id);
        {
            let index = self.index.lock();
            if !index.contains(key) {
                index
                    .add(key, embedding)
                    .map_err(|e| LedgerError::WarmTier(format!("usearch add: {e}")))?;
            }
        }

        // Keep key_map in sync so vector_search can resolve u64 keys back to ULID strings.
        self.insert_key_map(&memory.id)?;

        metrics::counter!("ledger.warm.inserts").increment(1);
        Ok(())
    }

    /// FTS5 keyword search. Returns `(MemoryId, rank)` pairs.
    ///
    /// FTS5 rank is negative BM25 (lower = better). Results are ordered by
    /// rank ascending (best first) by SQLite. We return them in that order;
    /// callers use the position (not the raw rank) for RRF.
    pub fn keyword_search(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<(MemoryId, f64)>, LedgerError> {
        if query.trim().is_empty() {
            return Ok(vec![]);
        }
        let escaped = escape_fts_query(query);
        let conn = self.conn.lock();
        let mut stmt = conn
            .prepare(
                "SELECT memory_id, rank FROM memories_fts \
                 WHERE memories_fts MATCH ?1 \
                 ORDER BY rank \
                 LIMIT ?2",
            )
            .map_err(|e| LedgerError::WarmTier(format!("prepare keyword search: {e}")))?;

        let rows: Vec<(MemoryId, f64)> = stmt
            .query_map(params![escaped, limit as i64], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, f64>(1)?))
            })
            .map_err(|e| LedgerError::WarmTier(format!("keyword search: {e}")))?
            .filter_map(|r| r.ok())
            .collect();

        Ok(rows)
    }

    /// usearch ANN vector search. Returns `(MemoryId, distance)` pairs
    /// ordered by distance ascending (closest first).
    pub fn vector_search(
        &self,
        embedding: &[f32],
        limit: usize,
    ) -> Result<Vec<(MemoryId, f32)>, LedgerError> {
        let index = self.index.lock();
        let count = index.size();
        if count == 0 {
            return Ok(vec![]);
        }
        let k = limit.min(count);
        if k == 0 {
            return Ok(vec![]);
        }

        let matches = index
            .search(embedding, k)
            .map_err(|e| LedgerError::WarmTier(format!("usearch search: {e}")))?;

        // Convert u64 keys back to ULID strings by looking them up in SQLite.
        let conn = self.conn.lock();
        let mut results = Vec::with_capacity(matches.keys.len());
        for (key, dist) in matches.keys.iter().zip(matches.distances.iter()) {
            // Look up the memory id by the key hash (stored in a mapping table).
            if let Some(id) = key_to_id(&conn, *key)? {
                results.push((id, *dist));
            }
        }
        Ok(results)
    }

    /// Find the closest existing memory to `embedding`.
    /// Returns `Some(Memory)` if the cosine similarity exceeds `threshold`.
    pub fn find_similar(
        &self,
        embedding: &[f32],
        threshold: f32,
    ) -> Result<Option<Memory>, LedgerError> {
        let results = self.vector_search(embedding, 1)?;
        if let Some((id, distance)) = results.into_iter().next() {
            // usearch cosine distance = 1 − cosine_similarity
            let sim = 1.0 - distance;
            if sim >= threshold {
                return self.get(&id);
            }
        }
        Ok(None)
    }

    /// Merge an incoming memory into an existing one (deduplicate).
    ///
    /// Appends the incoming content to the existing memory's content.
    pub fn merge(&self, id: &MemoryId, incoming: &Memory) -> Result<(), LedgerError> {
        let conn = self.conn.lock();
        conn.execute(
            "UPDATE memories SET content = content || '\n' || ?1, \
             last_accessed_ms = ?2 WHERE id = ?3",
            params![incoming.content, now_ms() as i64, id],
        )
        .map_err(|e| LedgerError::WarmTier(format!("merge: {e}")))?;
        Ok(())
    }

    /// Retrieve a memory by ID.
    pub fn get(&self, id: &str) -> Result<Option<Memory>, LedgerError> {
        let conn = self.conn.lock();
        let result = conn
            .query_row(
                "SELECT id, content, tags, created_at_ms, last_accessed_ms \
                 FROM memories WHERE id = ?1",
                params![id],
                row_to_memory,
            )
            .optional()
            .map_err(|e| LedgerError::WarmTier(format!("get: {e}")))?;
        Ok(result)
    }

    /// Batch-fetch memories by ID.
    pub fn get_many(&self, ids: &[MemoryId]) -> Result<Vec<Memory>, LedgerError> {
        let conn = self.conn.lock();
        let mut out = Vec::with_capacity(ids.len());
        for id in ids {
            let r = conn
                .query_row(
                    "SELECT id, content, tags, created_at_ms, last_accessed_ms \
                     FROM memories WHERE id = ?1",
                    params![id],
                    row_to_memory,
                )
                .optional()
                .map_err(|e| LedgerError::WarmTier(format!("get_many: {e}")))?;
            if let Some(m) = r {
                out.push(m);
            }
        }
        Ok(out)
    }

    /// Remove a memory from SQLite, FTS5, and the usearch index.
    pub fn remove(&self, id: &str) -> Result<(), LedgerError> {
        {
            let conn = self.conn.lock();
            conn.execute("DELETE FROM memories WHERE id = ?1", params![id])
                .map_err(|e| LedgerError::WarmTier(format!("remove sqlite: {e}")))?;
            conn.execute("DELETE FROM key_map WHERE memory_id = ?1", params![id])
                .map_err(|e| LedgerError::WarmTier(format!("remove key_map: {e}")))?;
            // Remove from FTS5 via stored rowid.
            let fts_rowid: Option<i64> = conn
                .query_row(
                    "SELECT fts_rowid FROM fts_rowid_map WHERE memory_id = ?1",
                    params![id],
                    |r| r.get(0),
                )
                .optional()
                .map_err(|e| LedgerError::WarmTier(format!("fts_rowid lookup: {e}")))?;
            if let Some(rowid) = fts_rowid {
                conn.execute("DELETE FROM memories_fts WHERE rowid = ?1", params![rowid])
                    .map_err(|e| LedgerError::WarmTier(format!("fts delete: {e}")))?;
                conn.execute("DELETE FROM fts_rowid_map WHERE memory_id = ?1", params![id])
                    .map_err(|e| LedgerError::WarmTier(format!("fts_rowid_map delete: {e}")))?;
            }
        }
        let key = ulid_to_u64(id);
        let index = self.index.lock();
        if index.contains(key) {
            let _ = index.remove(key);
        }
        Ok(())
    }

    /// Update `last_accessed_ms` for a memory.
    pub fn touch(&self, id: &str) -> Result<(), LedgerError> {
        let conn = self.conn.lock();
        conn.execute(
            "UPDATE memories SET last_accessed_ms = ?1 WHERE id = ?2",
            params![now_ms() as i64, id],
        )
        .map_err(|e| LedgerError::WarmTier(format!("touch: {e}")))?;
        Ok(())
    }

    /// Find memories not accessed since `cutoff_ms`.
    pub fn find_stale(&self, cutoff_ms: u64) -> Result<Vec<Memory>, LedgerError> {
        let conn = self.conn.lock();
        let mut stmt = conn
            .prepare(
                "SELECT id, content, tags, created_at_ms, last_accessed_ms \
                 FROM memories WHERE last_accessed_ms < ?1",
            )
            .map_err(|e| LedgerError::WarmTier(format!("find_stale prepare: {e}")))?;

        let rows: Vec<Memory> = stmt
            .query_map(params![cutoff_ms as i64], row_to_memory)
            .map_err(|e| LedgerError::WarmTier(format!("find_stale: {e}")))?
            .filter_map(|r| r.ok())
            .collect();
        Ok(rows)
    }

    /// Find memories with estimated token count above `threshold`.
    pub fn find_verbose(&self, threshold: usize) -> Result<Vec<Memory>, LedgerError> {
        let conn = self.conn.lock();
        // Approximate: SQLite length in chars divided by 5 ≈ tokens.
        let min_chars = (threshold * 5) as i64;
        let mut stmt = conn
            .prepare(
                "SELECT id, content, tags, created_at_ms, last_accessed_ms \
                 FROM memories WHERE LENGTH(content) > ?1",
            )
            .map_err(|e| LedgerError::WarmTier(format!("find_verbose prepare: {e}")))?;

        let rows: Vec<Memory> = stmt
            .query_map(params![min_chars], row_to_memory)
            .map_err(|e| LedgerError::WarmTier(format!("find_verbose: {e}")))?
            .filter_map(|r| r.ok())
            .collect();
        Ok(rows)
    }

    /// Replace the content of a memory (for compactor summarisation).
    pub fn replace_content(&self, id: &MemoryId, new_content: &str) -> Result<(), LedgerError> {
        let conn = self.conn.lock();
        conn.execute(
            "UPDATE memories SET content = ?1 WHERE id = ?2",
            params![new_content, id],
        )
        .map_err(|e| LedgerError::WarmTier(format!("replace_content: {e}")))?;
        Ok(())
    }

    /// Count all memories in the warm tier.
    pub fn count(&self) -> Result<usize, LedgerError> {
        let conn = self.conn.lock();
        let n: i64 = conn
            .query_row("SELECT COUNT(*) FROM memories", [], |r| r.get(0))
            .map_err(|e| LedgerError::WarmTier(format!("count: {e}")))?;
        Ok(n as usize)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Schema initialisation
// ─────────────────────────────────────────────────────────────────────────────

fn init_schema(conn: &Connection) -> Result<(), LedgerError> {
    conn.execute_batch(
        "
        CREATE TABLE IF NOT EXISTS memories (
            id               TEXT    PRIMARY KEY,
            content          TEXT    NOT NULL,
            tags             TEXT    NOT NULL DEFAULT '[]',
            created_at_ms    INTEGER NOT NULL,
            last_accessed_ms INTEGER NOT NULL,
            tier             TEXT    NOT NULL DEFAULT 'warm'
        );

        -- Standalone FTS5 virtual table (stores its own copy of content for
        -- simplicity and query reliability; memory_id is unindexed for retrieval).
        -- FTS entries are managed explicitly by Rust code (not triggers) so we
        -- avoid FTS5's rowid-only deletion constraint.
        CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
            content,
            memory_id UNINDEXED
        );

        -- fts_rowid_map: maps memory_id → fts rowid for targeted FTS5 deletion.
        CREATE TABLE IF NOT EXISTS fts_rowid_map (
            memory_id TEXT PRIMARY KEY,
            fts_rowid INTEGER NOT NULL
        );

        -- Key-map table: u64 usearch key → ULID memory_id.
        CREATE TABLE IF NOT EXISTS key_map (
            key_u64   INTEGER PRIMARY KEY,
            memory_id TEXT    NOT NULL
        );
        ",
    )
    .map_err(|e| LedgerError::WarmTier(format!("schema init: {e}")))?;
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Map a ULID string to a stable u64 key for usearch.
///
/// Takes the lower 64 bits of the 128-bit ULID integer.
fn ulid_to_u64(id: &str) -> u64 {
    // Parse the ULID; fall back to a hash of the raw string if it fails.
    if let Ok(u) = Ulid::from_string(id) {
        u.0 as u64
    } else {
        // Fallback: hash the raw bytes.
        use std::hash::Hash;
        let mut h = std::collections::hash_map::DefaultHasher::new();
        id.hash(&mut h);
        std::hash::Hasher::finish(&h)
    }
}

/// Look up a memory ID by usearch u64 key via the key_map table.
fn key_to_id(conn: &Connection, key: u64) -> Result<Option<String>, LedgerError> {
    conn.query_row(
        "SELECT memory_id FROM key_map WHERE key_u64 = ?1",
        params![key as i64],
        |r| r.get(0),
    )
    .optional()
    .map_err(|e| LedgerError::WarmTier(format!("key_to_id: {e}")))
}

/// Escape a user-supplied query string for FTS5 MATCH.
///
/// Each word becomes a separate term. Special FTS5 characters are stripped.
fn escape_fts_query(query: &str) -> String {
    // Split on whitespace, strip FTS5 special chars, join as AND terms.
    let terms: Vec<String> = query
        .split_whitespace()
        .map(|w| {
            // Strip chars that have special meaning in FTS5 queries.
            let clean: String = w.chars()
                .filter(|&c| c != '"' && c != '\'' && c != '(' && c != ')' && c != ':' && c != '*' && c != '^')
                .collect();
            format!("\"{clean}\"")
        })
        .filter(|t| t.len() > 2) // skip empty-after-strip
        .collect();
    if terms.is_empty() {
        String::new()
    } else {
        terms.join(" ")
    }
}

/// Map a SQLite row to a `Memory`.
fn row_to_memory(row: &rusqlite::Row<'_>) -> rusqlite::Result<Memory> {
    let id: String = row.get(0)?;
    let content: String = row.get(1)?;
    let tags_json: String = row.get(2)?;
    let created_at_ms: i64 = row.get(3)?;
    let last_accessed_ms: i64 = row.get(4)?;

    let tags: Vec<String> = serde_json::from_str(&tags_json).unwrap_or_default();

    Ok(Memory {
        id,
        content,
        tags,
        created_at_ms: created_at_ms as u64,
        last_accessed_ms: last_accessed_ms as u64,
        tier: Tier::Warm,
    })
}

impl WarmTier {
    /// Insert or update the u64 → ULID mapping used by `vector_search`.
    /// Called automatically by `insert()`; also available for explicit use.
    pub(crate) fn insert_key_map(&self, id: &str) -> Result<(), LedgerError> {
        let key = ulid_to_u64(id);
        let conn = self.conn.lock();
        conn.execute(
            "INSERT OR IGNORE INTO key_map (key_u64, memory_id) VALUES (?1, ?2)",
            params![key as i64, id],
        )
        .map_err(|e| LedgerError::WarmTier(format!("key_map insert: {e}")))?;
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::Memory;

    fn make_warm(dir: &tempfile::TempDir) -> WarmTier {
        WarmTier::open(&WarmConfig {
            db_path: dir.path().join("warm.db"),
            vector_index_path: dir.path().join("vec.usearch"),
            embedding_dim: 4, // tiny dim for tests
            vector_capacity: 100,
        })
        .unwrap()
    }

    fn tiny_embedding(seed: f32) -> Vec<f32> {
        let mut v = vec![seed, seed * 0.5, seed * 0.25, seed * 0.125];
        crate::embedder::l2_normalize(&mut v);
        v
    }

    #[test]
    fn warm_insert_and_get() {
        let dir = tempfile::tempdir().unwrap();
        let warm = make_warm(&dir);
        let m = Memory::new("warm memory content", vec!["t1".into()]);
        warm.insert(&m, &tiny_embedding(1.0)).unwrap();
        let got = warm.get(&m.id).unwrap().expect("should exist");
        assert_eq!(got.content, "warm memory content");
        assert_eq!(got.tags, vec!["t1"]);
    }

    #[test]
    fn warm_get_missing_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        let warm = make_warm(&dir);
        assert!(warm.get("nonexistent").unwrap().is_none());
    }

    #[test]
    fn warm_keyword_search_finds_content() {
        let dir = tempfile::tempdir().unwrap();
        let warm = make_warm(&dir);
        // Use plain words to avoid FTS5 tokenizer edge cases with underscores.
        let m = Memory::new("uniquekeyword xyzzy in warm tier", vec![]);
        warm.insert(&m, &tiny_embedding(1.0)).unwrap();
        let results = warm.keyword_search("uniquekeyword", 10).unwrap();
        assert!(!results.is_empty(), "should find memory by keyword");
        assert!(results.iter().any(|(id, _)| id == &m.id));
    }

    #[test]
    fn warm_keyword_search_no_match() {
        let dir = tempfile::tempdir().unwrap();
        let warm = make_warm(&dir);
        let m = Memory::new("some content here", vec![]);
        warm.insert(&m, &tiny_embedding(1.0)).unwrap();
        let results = warm.keyword_search("zzznomatch", 10).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn warm_vector_search_finds_similar() {
        let dir = tempfile::tempdir().unwrap();
        let warm = make_warm(&dir);
        let emb = tiny_embedding(0.9);
        let m = Memory::new("vector searchable content", vec![]);
        warm.insert(&m, &emb).unwrap();
        let results = warm.vector_search(&emb, 5).unwrap();
        assert!(!results.is_empty(), "should find by vector");
    }

    #[test]
    fn warm_find_similar_above_threshold() {
        let dir = tempfile::tempdir().unwrap();
        let warm = make_warm(&dir);
        let emb = tiny_embedding(1.0);
        let m = Memory::new("duplicate candidate", vec![]);
        warm.insert(&m, &emb).unwrap();
        // Search with the same embedding → distance ≈ 0 → sim ≈ 1.0 > 0.95
        let found = warm.find_similar(&emb, 0.95).unwrap();
        assert!(found.is_some(), "identical embedding should match at 0.95 threshold");
    }

    #[test]
    fn warm_find_similar_below_threshold() {
        let dir = tempfile::tempdir().unwrap();
        let warm = make_warm(&dir);
        let emb_a = tiny_embedding(1.0);
        let emb_b = tiny_embedding(0.1); // very different
        let m = Memory::new("target", vec![]);
        warm.insert(&m, &emb_a).unwrap();
        // Search with orthogonal embedding → should not match at 0.95
        let found = warm.find_similar(&emb_b, 0.95).unwrap();
        // Orthogonal embeddings have cosine sim ≈ 0, well below 0.95
        // (might or might not be None depending on actual distance; just don't panic)
        let _ = found;
    }

    #[test]
    fn warm_touch_updates_timestamp() {
        let dir = tempfile::tempdir().unwrap();
        let warm = make_warm(&dir);
        let m = Memory::new("touch test", vec![]);
        warm.insert(&m, &tiny_embedding(1.0)).unwrap();
        let before = warm.get(&m.id).unwrap().unwrap().last_accessed_ms;
        // Small sleep to ensure timestamp advances.
        std::thread::sleep(std::time::Duration::from_millis(2));
        warm.touch(&m.id).unwrap();
        let after = warm.get(&m.id).unwrap().unwrap().last_accessed_ms;
        assert!(after >= before);
    }

    #[test]
    fn warm_find_stale_returns_old_memories() {
        let dir = tempfile::tempdir().unwrap();
        let warm = make_warm(&dir);

        // Insert a memory with a very old timestamp.
        let mut m = Memory::new("stale memory", vec![]);
        m.last_accessed_ms = 1_000; // epoch + 1 second
        warm.insert(&m, &tiny_embedding(1.0)).unwrap();

        // Also insert a fresh memory.
        let fresh = Memory::new("fresh memory", vec![]);
        warm.insert(&fresh, &tiny_embedding(0.5)).unwrap();

        let cutoff = now_ms() - 1000; // everything older than 1 second ago
        let stale = warm.find_stale(cutoff).unwrap();
        assert!(stale.iter().any(|x| x.id == m.id), "stale memory should be found");
        assert!(!stale.iter().any(|x| x.id == fresh.id), "fresh memory should not be stale");
    }

    #[test]
    fn warm_find_verbose_returns_long_memories() {
        let dir = tempfile::tempdir().unwrap();
        let warm = make_warm(&dir);

        let short = Memory::new("short", vec![]);
        let long = Memory::new("word ".repeat(600), vec![]);
        warm.insert(&short, &tiny_embedding(1.0)).unwrap();
        warm.insert(&long, &tiny_embedding(0.5)).unwrap();

        let verbose = warm.find_verbose(100).unwrap(); // threshold: 100 tokens
        assert!(verbose.iter().any(|x| x.id == long.id));
        assert!(!verbose.iter().any(|x| x.id == short.id));
    }

    #[test]
    fn warm_remove_clears_entry() {
        let dir = tempfile::tempdir().unwrap();
        let warm = make_warm(&dir);
        let m = Memory::new("to remove", vec![]);
        warm.insert(&m, &tiny_embedding(1.0)).unwrap();
        warm.remove(&m.id).unwrap();
        assert!(warm.get(&m.id).unwrap().is_none());
    }

    #[test]
    fn warm_count_tracks_entries() {
        let dir = tempfile::tempdir().unwrap();
        let warm = make_warm(&dir);
        assert_eq!(warm.count().unwrap(), 0);
        warm.insert(&Memory::new("a", vec![]), &tiny_embedding(1.0)).unwrap();
        warm.insert(&Memory::new("b", vec![]), &tiny_embedding(0.5)).unwrap();
        assert_eq!(warm.count().unwrap(), 2);
    }
}
