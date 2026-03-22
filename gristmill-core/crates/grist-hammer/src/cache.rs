//! Semantic cache for grist-hammer.
//!
//! Two-level cache lookup:
//! 1. **Exact hit**: SHA-256 hash of the prompt string → O(1) LRU lookup.
//! 2. **Fuzzy hit**: Linear cosine-similarity scan over all cached embeddings.
//!    Only performed when the caller supplies a pre-computed embedding.
//!
//! The cache is guarded by a [`parking_lot::Mutex`] so all operations are
//! thread-safe and atomic with respect to LRU promotion.

use std::time::{SystemTime, UNIX_EPOCH};

use lru::LruCache;
use parking_lot::Mutex;
use tracing::debug;

use crate::config::CacheConfig;
use crate::types::EscalationResponse;

// ─────────────────────────────────────────────────────────────────────────────
// CacheEntry
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone)]
pub(crate) struct CacheEntry {
    pub response: EscalationResponse,
    /// Pre-computed embedding of the prompt (if available).
    pub embedding: Option<Vec<f32>>,
    /// Wall-clock insertion time (ms since UNIX epoch).
    pub inserted_at_ms: u64,
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

// ─────────────────────────────────────────────────────────────────────────────
// SemanticCache
// ─────────────────────────────────────────────────────────────────────────────

/// Thread-safe semantic response cache.
pub struct SemanticCache {
    inner: Mutex<LruCache<String, CacheEntry>>,
    config: CacheConfig,
}

impl SemanticCache {
    pub fn new(config: CacheConfig) -> Self {
        let capacity = std::num::NonZeroUsize::new(config.max_entries.max(1)).unwrap();
        Self {
            inner: Mutex::new(LruCache::new(capacity)),
            config,
        }
    }

    // ── SHA-256 exact lookup ──────────────────────────────────────────────

    /// Look up a response by the SHA-256 hash of the prompt.
    ///
    /// Returns the cached response (cloned) and promotes it in the LRU order.
    pub fn get_exact(&self, hash: &str) -> Option<EscalationResponse> {
        if !self.config.enabled {
            return None;
        }
        let mut inner = self.inner.lock();
        let entry = inner.get(hash)?;
        if self.is_expired(entry) {
            // Remove expired entry.
            drop(inner); // release lock before re-acquiring
            let mut inner = self.inner.lock();
            inner.pop(hash);
            return None;
        }
        debug!(hash, "semantic cache exact hit");
        Some(entry.response.clone())
    }

    /// Look up by cosine similarity over all stored embeddings.
    ///
    /// Returns the best-matching cached response if `sim ≥ config.threshold`.
    /// Only entries that themselves have an embedding are considered.
    pub fn get_fuzzy(&self, query_embedding: &[f32]) -> Option<EscalationResponse> {
        if !self.config.enabled {
            return None;
        }
        let threshold = self.config.similarity_threshold;
        let mut inner = self.inner.lock();

        let mut best_hash: Option<String> = None;
        let mut best_sim = f32::NEG_INFINITY;

        for (hash, entry) in inner.iter() {
            if self.is_expired(entry) {
                continue;
            }
            if let Some(emb) = &entry.embedding {
                let sim = cosine_similarity(query_embedding, emb);
                if sim > best_sim {
                    best_sim = sim;
                    best_hash = Some(hash.clone());
                }
            }
        }

        if best_sim >= threshold {
            if let Some(hash) = best_hash {
                debug!(sim = best_sim, "semantic cache fuzzy hit");
                // Promote in LRU.
                return inner.get(&hash).map(|e| e.response.clone());
            }
        }
        None
    }

    // ── Insert ───────────────────────────────────────────────────────────

    /// Store a response keyed by the SHA-256 hash of the prompt.
    pub fn put(&self, hash: String, response: EscalationResponse, embedding: Option<Vec<f32>>) {
        if !self.config.enabled {
            return;
        }
        let entry = CacheEntry {
            response,
            embedding,
            inserted_at_ms: now_ms(),
        };
        let mut inner = self.inner.lock();
        inner.put(hash, entry);
    }

    // ── Introspection ─────────────────────────────────────────────────────

    pub fn clear(&self) {
        self.inner.lock().clear();
    }

    pub fn len(&self) -> usize {
        self.inner.lock().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    // ── Private helpers ───────────────────────────────────────────────────

    fn is_expired(&self, entry: &CacheEntry) -> bool {
        if self.config.ttl_secs == 0 {
            return false; // TTL disabled
        }
        let age_ms = now_ms().saturating_sub(entry.inserted_at_ms);
        age_ms > self.config.ttl_secs * 1_000
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Compute cosine similarity between two equal-length slices.
///
/// Returns 0.0 if either vector has zero norm.
pub(crate) fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a < 1e-8 || norm_b < 1e-8 {
        return 0.0;
    }
    (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Provider;

    fn make_cache(enabled: bool, max_entries: usize, threshold: f32) -> SemanticCache {
        SemanticCache::new(CacheConfig {
            enabled,
            similarity_threshold: threshold,
            max_entries,
            embedding_dim: 4,
            ttl_secs: 0,
        })
    }

    fn fake_response(content: &str) -> EscalationResponse {
        EscalationResponse {
            request_id: "test-id".into(),
            content: content.into(),
            provider: Provider::AnthropicPrimary,
            provider_type: Provider::AnthropicPrimary.provider_type(),
            cache_hit: false,
            tokens_used: 10,
            elapsed_ms: 5,
        }
    }

    #[test]
    fn cache_exact_hit() {
        let cache = make_cache(true, 10, 0.92);
        let resp = fake_response("hello");
        cache.put("hash1".into(), resp.clone(), None);
        let found = cache.get_exact("hash1").expect("should hit");
        assert_eq!(found.content, "hello");
    }

    #[test]
    fn cache_miss_on_different_prompt() {
        let cache = make_cache(true, 10, 0.92);
        cache.put("hash1".into(), fake_response("hello"), None);
        assert!(cache.get_exact("hash2").is_none());
    }

    #[test]
    fn cache_fuzzy_hit_above_threshold() {
        let cache = make_cache(true, 10, 0.92);
        // Two nearly-identical unit vectors (same direction).
        let emb_a = vec![1.0_f32, 0.0, 0.0, 0.0];
        let emb_b = vec![0.999_f32, 0.045, 0.0, 0.0]; // sim ≈ 0.999
        cache.put("hash1".into(), fake_response("cached"), Some(emb_a));
        let result = cache.get_fuzzy(&emb_b);
        assert!(result.is_some(), "should fuzzy-hit");
        assert_eq!(result.unwrap().content, "cached");
    }

    #[test]
    fn cache_fuzzy_miss_below_threshold() {
        let cache = make_cache(true, 10, 0.92);
        // Orthogonal vectors → sim = 0.0.
        let emb_a = vec![1.0_f32, 0.0, 0.0, 0.0];
        let emb_b = vec![0.0_f32, 1.0, 0.0, 0.0];
        cache.put("hash1".into(), fake_response("cached"), Some(emb_a));
        let result = cache.get_fuzzy(&emb_b);
        assert!(result.is_none(), "orthogonal vecs should not hit");
    }

    #[test]
    fn cache_evicts_lru_at_max_entries() {
        let cache = make_cache(true, 3, 0.92);
        cache.put("h1".into(), fake_response("r1"), None);
        cache.put("h2".into(), fake_response("r2"), None);
        cache.put("h3".into(), fake_response("r3"), None);
        // Access h1 to make it recently used.
        cache.get_exact("h1");
        // Insert h4 → should evict h2 (LRU).
        cache.put("h4".into(), fake_response("r4"), None);
        assert_eq!(cache.len(), 3);
        assert!(cache.get_exact("h2").is_none(), "h2 should be evicted");
        assert!(cache.get_exact("h1").is_some(), "h1 should survive");
    }

    #[test]
    fn cache_clear_empties() {
        let cache = make_cache(true, 10, 0.92);
        cache.put("h1".into(), fake_response("r1"), None);
        cache.put("h2".into(), fake_response("r2"), None);
        cache.clear();
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn cache_disabled_returns_none() {
        let cache = make_cache(false, 10, 0.92);
        cache.put("h1".into(), fake_response("r1"), None);
        assert!(cache.get_exact("h1").is_none());
    }

    #[test]
    fn cosine_similarity_same_vec() {
        let v = vec![0.6_f32, 0.8, 0.0, 0.0];
        let sim = cosine_similarity(&v, &v);
        assert!(
            (sim - 1.0).abs() < 1e-5,
            "sim with self should be 1.0, got {sim}"
        );
    }

    #[test]
    fn cosine_similarity_orthogonal() {
        let a = vec![1.0_f32, 0.0];
        let b = vec![0.0_f32, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6, "orthogonal vecs should have sim ≈ 0");
    }

    #[test]
    fn cosine_similarity_zero_vec() {
        let a = vec![0.0_f32, 0.0];
        let b = vec![1.0_f32, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert_eq!(sim, 0.0);
    }
}
