//! Semantic cache for Sieve routing decisions.
//!
//! PRD requirements:
//! - S-04: Cache hit rate >25% on synthetic recurring workload.
//! - S-07: Model swap completes in <500ms with zero dropped events.
//!
//! Two-level cache design:
//!
//! 1. **Exact cache** — LRU keyed on SHA-256 of normalised text.
//!    O(1) lookup, zero vector math.  Handles repeated identical events.
//!
//! 2. **Semantic ring buffer** — the last N embedding vectors are stored
//!    alongside their routing decisions.  A new event is checked against
//!    all stored embeddings via cosine similarity.  If the max similarity
//!    exceeds the threshold (0.92 per PRD), the cached decision is returned.
//!
//! The semantic tier is intentionally bounded (ring buffer) to keep lookup
//! time constant regardless of history size and to avoid stale decisions
//! accumulating from long ago.

use std::collections::VecDeque;
use std::sync::Arc;

use ndarray::Array1;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use tracing::{instrument, trace};
#[allow(unused_imports)]
use metrics;

use crate::cost_oracle::RouteDecision;
use crate::features::FeatureVector;

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Default cosine similarity threshold for a semantic cache hit.
/// Matches the PRD (0.92 — same as Hammer's semantic cache).
pub const DEFAULT_SIMILARITY_THRESHOLD: f32 = 0.92;

/// Default exact-cache capacity (number of decisions to keep).
pub const DEFAULT_EXACT_CAPACITY: usize = 10_000;

/// Default semantic ring buffer size.
pub const DEFAULT_SEMANTIC_CAPACITY: usize = 256;

// ─────────────────────────────────────────────────────────────────────────────
// Cache entry
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct SemanticEntry {
    /// L2-normalised embedding for similarity computation.
    embedding: Array1<f32>,
    decision: RouteDecision,
}

// ─────────────────────────────────────────────────────────────────────────────
// RoutingCache
// ─────────────────────────────────────────────────────────────────────────────

/// Thread-safe two-level routing cache.
#[derive(Debug)]
pub struct RoutingCache {
    /// Level 1: exact SHA-256 → RouteDecision.
    exact: Mutex<lru::LruCache<String, RouteDecision>>,
    /// Level 2: semantic ring buffer (embedding → RouteDecision).
    semantic: Mutex<VecDeque<SemanticEntry>>,
    /// Cosine similarity threshold for semantic hits.
    similarity_threshold: f32,
    /// Max entries in the semantic buffer.
    semantic_capacity: usize,
    /// Running stats for observability.
    stats: Arc<CacheStats>,
}

impl RoutingCache {
    pub fn new(
        exact_capacity: usize,
        semantic_capacity: usize,
        similarity_threshold: f32,
    ) -> Self {
        Self {
            exact: Mutex::new(lru::LruCache::new(
                std::num::NonZeroUsize::new(exact_capacity).unwrap(),
            )),
            semantic: Mutex::new(VecDeque::with_capacity(semantic_capacity)),
            similarity_threshold,
            semantic_capacity,
            stats: Arc::new(CacheStats::default()),
        }
    }

    /// Default cache with PRD-specified thresholds.
    pub fn default_config() -> Self {
        Self::new(
            DEFAULT_EXACT_CAPACITY,
            DEFAULT_SEMANTIC_CAPACITY,
            DEFAULT_SIMILARITY_THRESHOLD,
        )
    }

    // ── Lookup ───────────────────────────────────────────────────────────────

    /// Try to find a cached routing decision for this feature vector.
    ///
    /// Checks exact cache first (fast), then semantic ring buffer (slower).
    /// Returns `None` if no suitable cached decision is found.
    #[instrument(level = "trace", skip(self, features))]
    pub fn lookup(&self, features: &FeatureVector) -> Option<RouteDecision> {
        // Level 1: exact hit.
        {
            let mut exact = self.exact.lock();
            if let Some(decision) = exact.get(&features.text_hash) {
                self.stats.exact_hits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                metrics::counter!("sieve.cache.hit", "tier" => "exact").increment(1);
                trace!(hash = %features.text_hash, "exact cache hit");
                return Some(decision.clone());
            }
        }

        // Level 2: semantic hit.
        // Only attempt if the embedding is non-zero (i.e. embedder is loaded).
        let embedding_norm = l2_norm(&features.data.slice(ndarray::s![0..crate::features::EMBEDDING_DIM]).to_owned());
        if embedding_norm > 1e-8 {
            let normalized = &features.data.slice(ndarray::s![0..crate::features::EMBEDDING_DIM]).to_owned() / embedding_norm;
            let semantic = self.semantic.lock();
            for entry in semantic.iter() {
                let sim = cosine_similarity_prenorm(&normalized, &entry.embedding);
                if sim >= self.similarity_threshold {
                    self.stats.semantic_hits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    metrics::counter!("sieve.cache.hit", "tier" => "semantic").increment(1);
                    trace!(similarity = sim, "semantic cache hit");
                    return Some(entry.decision.clone());
                }
            }
        }

        self.stats.misses.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        metrics::counter!("sieve.cache.miss").increment(1);
        None
    }

    // ── Store ─────────────────────────────────────────────────────────────────

    /// Store a routing decision in both cache tiers.
    pub fn store(&self, features: &FeatureVector, decision: &RouteDecision) {
        // Level 1: exact.
        {
            let mut exact = self.exact.lock();
            exact.put(features.text_hash.clone(), decision.clone());
        }

        // Level 2: semantic (if we have a real embedding).
        let embedding_slice = features.data.slice(ndarray::s![0..crate::features::EMBEDDING_DIM]).to_owned();
        let norm = l2_norm(&embedding_slice);
        if norm > 1e-8 {
            let normalized = embedding_slice / norm;
            let mut semantic = self.semantic.lock();
            if semantic.len() >= self.semantic_capacity {
                semantic.pop_front();
            }
            semantic.push_back(SemanticEntry {
                embedding: normalized,
                decision: decision.clone(),
            });
        }
    }

    // ── Observability ─────────────────────────────────────────────────────────

    pub fn stats(&self) -> CacheSnapshot {
        use std::sync::atomic::Ordering::Relaxed;
        let exact_hits = self.stats.exact_hits.load(Relaxed);
        let semantic_hits = self.stats.semantic_hits.load(Relaxed);
        let misses = self.stats.misses.load(Relaxed);
        let total = exact_hits + semantic_hits + misses;
        let hit_rate = if total > 0 {
            (exact_hits + semantic_hits) as f64 / total as f64
        } else {
            0.0
        };
        CacheSnapshot {
            exact_hits,
            semantic_hits,
            misses,
            hit_rate,
            exact_size: self.exact.lock().len(),
            semantic_size: self.semantic.lock().len(),
        }
    }

    /// Clear all cached entries (e.g. after a model hot-reload).
    pub fn clear(&self) {
        self.exact.lock().clear();
        self.semantic.lock().clear();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Stats
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Default)]
struct CacheStats {
    exact_hits: std::sync::atomic::AtomicU64,
    semantic_hits: std::sync::atomic::AtomicU64,
    misses: std::sync::atomic::AtomicU64,
}

/// Snapshot of cache statistics for logging / dashboard.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheSnapshot {
    pub exact_hits: u64,
    pub semantic_hits: u64,
    pub misses: u64,
    pub hit_rate: f64,
    pub exact_size: usize,
    pub semantic_size: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// Math helpers
// ─────────────────────────────────────────────────────────────────────────────

/// L2 norm of a vector.
fn l2_norm(v: &Array1<f32>) -> f32 {
    v.dot(v).sqrt()
}

/// Cosine similarity assuming both inputs are already L2-normalised.
fn cosine_similarity_prenorm(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
    a.dot(b).clamp(-1.0, 1.0)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cost_oracle::RouteDecision;
    use grist_event::{ChannelType, GristEvent};
    use crate::features::{FeatureExtractor, FEATURE_DIM};
    use ndarray::Array1;

    fn make_feature_vector(text: &str) -> FeatureVector {
        let event = GristEvent::new(
            ChannelType::Http,
            serde_json::json!({ "text": text }),
        );
        let extractor = FeatureExtractor::new_no_embed();
        extractor.extract(&event).unwrap()
    }

    fn local_ml_decision() -> RouteDecision {
        RouteDecision::LocalMl {
            model_id: "test-model".to_string(),
            confidence: 0.95,
        }
    }

    #[test]
    fn exact_cache_hit_after_store() {
        let cache = RoutingCache::default_config();
        let fv = make_feature_vector("hello world");
        let decision = local_ml_decision();
        cache.store(&fv, &decision);
        let hit = cache.lookup(&fv);
        assert!(hit.is_some(), "should hit exact cache");
        assert!(matches!(hit.unwrap(), RouteDecision::LocalMl { .. }));
    }

    #[test]
    fn miss_on_unknown_event() {
        let cache = RoutingCache::default_config();
        let fv = make_feature_vector("something completely different");
        assert!(cache.lookup(&fv).is_none());
    }

    #[test]
    fn stats_track_hits_and_misses() {
        let cache = RoutingCache::default_config();
        let fv = make_feature_vector("test stats");
        // Miss first.
        cache.lookup(&fv);
        let snap = cache.stats();
        assert_eq!(snap.misses, 1);
        // Store then hit.
        cache.store(&fv, &local_ml_decision());
        cache.lookup(&fv);
        let snap2 = cache.stats();
        assert_eq!(snap2.exact_hits, 1);
    }

    #[test]
    fn clear_removes_all_entries() {
        let cache = RoutingCache::default_config();
        let fv = make_feature_vector("to be cleared");
        cache.store(&fv, &local_ml_decision());
        cache.clear();
        assert!(cache.lookup(&fv).is_none());
    }

    #[test]
    fn semantic_hit_with_identical_embedding() {
        // Inject a non-zero fake embedding so the semantic tier activates.
        let cache = RoutingCache::new(100, 64, 0.90);

        // Build a feature vector with a real (non-zero) embedding.
        let mut data = Array1::<f32>::zeros(FEATURE_DIM);
        // Set a unit vector in the embedding slice.
        data[0] = 1.0;
        let fv = FeatureVector {
            data: data.clone(),
            text_hash: "unique-hash-1".to_string(),
            token_count: 3,
        };

        let decision = local_ml_decision();
        cache.store(&fv, &decision);

        // Lookup with the same embedding but a different hash (no exact hit).
        let fv2 = FeatureVector {
            data,
            text_hash: "unique-hash-2".to_string(),
            token_count: 3,
        };
        let hit = cache.lookup(&fv2);
        // Should hit the semantic tier.
        assert!(hit.is_some(), "semantic cache should hit for identical embeddings");
    }

    #[test]
    fn cosine_similarity_self_is_one() {
        let v = Array1::from(vec![1.0_f32, 0.0, 0.0]);
        let sim = cosine_similarity_prenorm(&v, &v);
        assert!((sim - 1.0).abs() < 1e-5);
    }

    #[test]
    fn cosine_similarity_orthogonal_is_zero() {
        let a = Array1::from(vec![1.0_f32, 0.0, 0.0]);
        let b = Array1::from(vec![0.0_f32, 1.0, 0.0]);
        let sim = cosine_similarity_prenorm(&a, &b);
        assert!(sim.abs() < 1e-5);
    }

    #[test]
    fn hit_rate_is_correct() {
        let cache = RoutingCache::default_config();
        let fv = make_feature_vector("hit rate test");
        // 1 miss then 1 hit.
        cache.lookup(&fv);
        cache.store(&fv, &local_ml_decision());
        cache.lookup(&fv);
        let snap = cache.stats();
        // 1 hit, 1 miss → rate = 0.5
        assert!((snap.hit_rate - 0.5).abs() < 0.01);
    }
}
