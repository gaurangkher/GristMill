//! Pluggable embedding interface for grist-ledger.
//!
//! The `Embedder` trait decouples the ledger from `grist-grinders` so
//! the ledger can be tested without a running ONNX runtime.
//!
//! # Implementations
//!
//! | Type | Behaviour |
//! |------|-----------|
//! | [`StubEmbedder`] | Deterministic SHA-256-derived vector (tests). |
//! | [`ZeroEmbedder`] | All-zero vector of the given dimension (no-op mode). |

use sha2::{Digest, Sha256};

use crate::error::LedgerError;

/// Trait for synchronous text embedding.
///
/// All implementations must be `Send + Sync` so they can be shared across
/// Tokio `spawn_blocking` closures.
pub trait Embedder: Send + Sync {
    /// Embed `text` into a dense `f32` vector.
    fn embed(&self, text: &str) -> Result<Vec<f32>, LedgerError>;

    /// Return the embedding dimension.
    fn dim(&self) -> usize;
}

// ─────────────────────────────────────────────────────────────────────────────
// StubEmbedder
// ─────────────────────────────────────────────────────────────────────────────

/// Deterministic test embedder.
///
/// Produces a **different** L2-normalised vector for each distinct input text
/// by hashing the text with SHA-256 and cycling the hash bytes.  Identical
/// inputs produce identical outputs — making test assertions stable.
pub struct StubEmbedder {
    dim: usize,
}

impl StubEmbedder {
    pub fn new(dim: usize) -> Self {
        assert!(dim > 0, "embedding dim must be > 0");
        Self { dim }
    }
}

impl Embedder for StubEmbedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>, LedgerError> {
        let hash = Sha256::digest(text.as_bytes());
        let mut vec = Vec::with_capacity(self.dim);
        for i in 0..self.dim {
            let byte = hash[i % 32] as f32;
            // Map [0, 255] → [−1.0, 1.0]
            vec.push((byte / 127.5) - 1.0);
        }
        l2_normalize(&mut vec);
        Ok(vec)
    }

    fn dim(&self) -> usize {
        self.dim
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ZeroEmbedder
// ─────────────────────────────────────────────────────────────────────────────

/// All-zero embedder — used when no model is available.
///
/// Semantic search degrades to random results in this mode.
/// Keyword search still works correctly.
pub struct ZeroEmbedder {
    dim: usize,
}

impl ZeroEmbedder {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }
}

impl Embedder for ZeroEmbedder {
    fn embed(&self, _text: &str) -> Result<Vec<f32>, LedgerError> {
        Ok(vec![0.0f32; self.dim])
    }

    fn dim(&self) -> usize {
        self.dim
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// L2-normalise a vector in place.  No-op if the norm is negligible.
pub fn l2_normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-8 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Cosine similarity of two equal-length f32 slices.
///
/// Both vectors are assumed to be L2-normalised (dot product = cosine sim).
/// Falls back to a full computation if they are not.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "vector dimensions must match");
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

    #[test]
    fn stub_embedder_deterministic() {
        let e = StubEmbedder::new(384);
        let v1 = e.embed("hello world").unwrap();
        let v2 = e.embed("hello world").unwrap();
        assert_eq!(v1, v2, "same input → same embedding");
    }

    #[test]
    fn stub_embedder_different_inputs_differ() {
        let e = StubEmbedder::new(384);
        let v1 = e.embed("hello world").unwrap();
        let v2 = e.embed("goodbye world").unwrap();
        assert_ne!(v1, v2, "different inputs → different embeddings");
    }

    #[test]
    fn stub_embedder_normalized() {
        let e = StubEmbedder::new(384);
        let v = e.embed("test text for normalization").unwrap();
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-4,
            "embedding should be L2-normalised, norm={norm}"
        );
    }

    #[test]
    fn stub_embedder_correct_dim() {
        let e = StubEmbedder::new(128);
        let v = e.embed("test").unwrap();
        assert_eq!(v.len(), 128);
        assert_eq!(e.dim(), 128);
    }

    #[test]
    fn zero_embedder_produces_zeros() {
        let e = ZeroEmbedder::new(64);
        let v = e.embed("anything").unwrap();
        assert_eq!(v.len(), 64);
        assert!(v.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn cosine_similarity_identical_vectors() {
        let v = vec![1.0f32, 0.0, 0.0];
        assert!((cosine_similarity(&v, &v) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_orthogonal_vectors() {
        let a = vec![1.0f32, 0.0, 0.0];
        let b = vec![0.0f32, 1.0, 0.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-6);
    }
}
