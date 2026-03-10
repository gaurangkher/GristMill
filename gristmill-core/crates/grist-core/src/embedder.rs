//! `GrindersEmbedder` — bridges `grist-grinders`'s MiniLM ONNX session to the
//! `grist-ledger` [`Embedder`] trait so the Ledger performs real semantic search.
//!
//! # Fallback chain
//!
//! 1. `minilm-l6-v2` configured in grinders **and** `.onnx` file present on disk →
//!    real 384-dim MiniLM embeddings (production mode).
//! 2. Otherwise → [`grist_ledger::ZeroEmbedder`]: keyword search still works; vector
//!    recall degrades to random ranking until the model file is downloaded.

use std::sync::Arc;

use grist_grinders::GrindersConfig;
use grist_grinders::embedder::build_minilm_embedder;
use grist_ledger::{Embedder, LedgerError, ZeroEmbedder};
use grist_sieve::features::EMBEDDING_DIM;

// ─────────────────────────────────────────────────────────────────────────────
// GrindersEmbedder
// ─────────────────────────────────────────────────────────────────────────────

/// Wraps a [`grist_sieve::features::EmbedderSession`] (MiniLM-L6-v2 via ONNX)
/// and implements [`Embedder`] for the Ledger's three-tier memory store.
///
/// `EmbedderSession` contains `Box<dyn EmbedFn + Send + Sync>`, so this type
/// is automatically `Send + Sync`.
pub struct GrindersEmbedder {
    session: grist_sieve::features::EmbedderSession,
}

impl Embedder for GrindersEmbedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>, LedgerError> {
        self.session
            .embed(text)
            .map(|arr| arr.to_vec())
            .map_err(|e| LedgerError::Embedding(e.to_string()))
    }

    fn dim(&self) -> usize {
        EMBEDDING_DIM
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Builder
// ─────────────────────────────────────────────────────────────────────────────

/// Build the best available [`Embedder`] for the Ledger given the Grinders config.
///
/// Returns a [`GrindersEmbedder`] backed by MiniLM-L6-v2 when the model file is
/// available, otherwise falls back to [`ZeroEmbedder`] so the daemon can always
/// start — run `gristmill models pull` to enable full semantic recall.
pub fn build_ledger_embedder(config: &GrindersConfig) -> Arc<dyn Embedder> {
    match build_minilm_embedder(config) {
        Ok(session) => {
            tracing::info!("Ledger using GrindersEmbedder (MiniLM-L6-v2)");
            Arc::new(GrindersEmbedder { session })
        }
        Err(e) => {
            tracing::warn!(
                error = %e,
                "MiniLM embedder unavailable — Ledger falling back to ZeroEmbedder; \
                 semantic recall will not rank by similarity"
            );
            Arc::new(ZeroEmbedder::new(EMBEDDING_DIM))
        }
    }
}
