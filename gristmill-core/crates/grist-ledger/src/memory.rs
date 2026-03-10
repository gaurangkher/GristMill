//! Core memory types for grist-ledger.

use serde::{Deserialize, Serialize};
use ulid::Ulid;

/// Unique identifier for a memory (ULID string).
pub type MemoryId = String;

/// Which storage tier a memory currently lives in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Tier {
    /// In-memory LRU + sled backing store (fastest).
    Hot,
    /// SQLite FTS5 + usearch vector index (warm, searchable).
    Warm,
    /// zstd-compressed JSONL archive (cold, minimal overhead).
    Cold,
}

/// Which search path contributed a result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SearchSource {
    Keyword,
    Vector,
    Cold,
}

/// A single memory stored in the ledger.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Memory {
    /// Unique identifier (ULID).
    pub id: MemoryId,
    /// The text content of this memory.
    pub content: String,
    /// Optional user-defined labels.
    pub tags: Vec<String>,
    /// Wall-clock time when this memory was created (ms since UNIX epoch).
    pub created_at_ms: u64,
    /// Wall-clock time when this memory was last accessed (ms since UNIX epoch).
    pub last_accessed_ms: u64,
    /// Current storage tier.
    pub tier: Tier,
}

impl Memory {
    /// Create a new memory with the current timestamp.
    pub fn new(content: impl Into<String>, tags: Vec<String>) -> Self {
        let now = now_ms();
        Self {
            id: Ulid::new().to_string(),
            content: content.into(),
            tags,
            created_at_ms: now,
            last_accessed_ms: now,
            tier: Tier::Hot,
        }
    }

    /// Approximate token count (whitespace-split word count).
    pub fn estimated_tokens(&self) -> usize {
        self.content.split_whitespace().count()
    }
}

/// A memory returned from a `recall()` call, with its relevance score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankedMemory {
    pub memory: Memory,
    /// Combined Reciprocal Rank Fusion score (higher = more relevant).
    pub score: f64,
    /// Which search paths contributed to this result.
    pub sources: Vec<SearchSource>,
}

/// Current wall-clock time in milliseconds since UNIX epoch.
pub(crate) fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_new_has_correct_defaults() {
        let m = Memory::new("hello world", vec!["tag1".into()]);
        assert!(!m.id.is_empty());
        assert_eq!(m.content, "hello world");
        assert_eq!(m.tags, vec!["tag1"]);
        assert_eq!(m.tier, Tier::Hot);
        assert!(m.created_at_ms > 0);
        assert_eq!(m.created_at_ms, m.last_accessed_ms);
    }

    #[test]
    fn estimated_tokens_counts_words() {
        let m = Memory::new("the quick brown fox", vec![]);
        assert_eq!(m.estimated_tokens(), 4);
    }

    #[test]
    fn memory_json_roundtrip() {
        let m = Memory::new("test content", vec!["a".into()]);
        let json = serde_json::to_string(&m).unwrap();
        let m2: Memory = serde_json::from_str(&json).unwrap();
        assert_eq!(m.id, m2.id);
        assert_eq!(m.content, m2.content);
        assert_eq!(m.tier, m2.tier);
    }
}
