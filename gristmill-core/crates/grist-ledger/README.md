# grist-ledger

Three-tier memory system for GristMill. Stores, retrieves, and auto-compacts memories across hot (LRU), warm (SQLite FTS5 + vector search), and cold (compressed archive) tiers.

## Architecture

```
remember(content)
    │
    ▼
Hot Tier (sled LRU, in-process)
    │ LRU eviction (async background drainer)
    ▼
Warm Tier (SQLite FTS5 + usearch HNSW)  ← recall() searches here
    │ Compactor (every 6 hours)
    ▼
Cold Tier (zstd-compressed JSONL archive)
```

**Key point**: `recall()` searches the **warm tier**. Items arrive in the warm tier when they are evicted from the hot tier by LRU pressure — not immediately after `remember()`. Use `get_memory(id)` to retrieve a just-stored memory by ID.

## Key Types

```rust
pub struct Ledger { /* opaque */ }

pub struct Memory {
    pub id: MemoryId,           // ULID string
    pub content: String,
    pub embedding: Vec<f32>,    // MiniLM-L6-v2 384-dim (or zeros)
    pub tags: Vec<String>,
    pub created_at_ms: u64,
    pub last_accessed_ms: u64,
    pub tier: Tier,             // Hot | Warm | Cold
}

pub struct RankedMemory {
    pub memory: Memory,
    pub score: f32,             // Reciprocal Rank Fusion score
    pub sources: Vec<String>,   // e.g. ["keyword", "semantic"]
}

pub struct LedgerConfig {
    pub hot: HotConfig,         // sled path, max_size_mb
    pub warm: WarmConfig,       // SQLite path, usearch index path
    pub cold: ColdConfig,       // archive dir, compression level
    pub compaction: CompactorConfig,  // interval_secs, thresholds
}

pub trait Embedder: Send + Sync {
    fn embed(&self, text: &str) -> Result<Vec<f32>, LedgerError>;
    fn dim(&self) -> usize;
}
```

## Embedder Implementations

| Type | Source | Behaviour |
|------|--------|-----------|
| `GrindersEmbedder` | `grist-core/src/embedder.rs` | Real MiniLM-L6-v2 via ONNX (production) |
| `ZeroEmbedder` | `grist-ledger` | All-zeros vector — keyword search still works |
| `StubEmbedder` | `grist-ledger` | SHA-256 hash seeded vector — deterministic for tests |

`GrindersEmbedder` is selected automatically by `grist-core` when the ONNX model file is present; falls back to `ZeroEmbedder` otherwise.

## Public API

```rust
// Create ledger
let ledger = Ledger::new(config, embedder).await?;

// Store a memory (writes to hot tier immediately)
let id: MemoryId = ledger.remember("prod-db-01 disk alert resolved", vec!["infra"]).await?;

// Search (queries warm tier)
let results: Vec<RankedMemory> = ledger.recall("disk postgres", 10).await?;

// Retrieve by ID (hot tier first, then warm, then cold)
let memory: Option<Memory> = ledger.get_memory(&id).await?;
```

## Recall Algorithm

```
recall(query, limit)
    │
    ├─ Embed query → 384-dim vector
    │
    ├─ Parallel search:
    │   ├─ keyword_search(FTS5 BM25)
    │   └─ vector_search(usearch HNSW, cosine distance)
    │
    └─ Reciprocal Rank Fusion → top-N results
```

## Deduplication

On `remember()`, the warm tier is checked for semantically similar existing memories (cosine similarity ≥ 0.95). If a match is found, the new content is merged into the existing memory instead of creating a duplicate.

## Auto-Compaction

The compactor runs every 6 hours (configurable) as a background Tokio task:

1. **Deduplicate** — cluster warm memories with similarity ≥ 0.90; merge clusters
2. **Summarise** — compress memories > 512 tokens using Phi-3 Mini GGUF (via grist-grinders)
3. **Demote** — move warm memories not accessed in > 90 days to cold tier

## Bus Events Published

| Topic | Trigger |
|-------|---------|
| `ledger.threshold` | Hot/warm tier approaching capacity |

## Configuration

```yaml
# ~/.gristmill/config.yaml
ledger:
  hot:
    max_size_mb: 512
  warm:
    db_path: ~/.gristmill/memory/warm.db
    vector_index_path: ~/.gristmill/memory/vectors.usearch
    fts_enabled: true
  cold:
    archive_dir: ~/.gristmill/memory/cold/
    compression: zstd
    compress_level: 3
  compaction:
    interval_hours: 6
    similarity_threshold: 0.90
    verbose_threshold_tokens: 512
    stale_days: 90
```

## Dependencies

```toml
grist-event  = { path = "../grist-event" }
grist-bus    = { path = "../grist-bus" }
tokio        # async operations + background tasks
sled         # hot-tier LRU
rusqlite     # warm-tier SQLite + FTS5 (bundled feature)
usearch      # warm-tier HNSW vector index
zstd         # cold-tier compression
ndarray      # embedding operations
lru          # eviction tracking
metrics      # tier hit/miss counters
tracing      # compaction + recall logging
```
