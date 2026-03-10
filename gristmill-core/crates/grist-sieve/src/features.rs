//! Feature extraction pipeline for the Sieve triage engine.
//!
//! Produces a [`FeatureVector`] from a [`GristEvent`] by combining:
//! - TF-IDF-style term statistics (fast, no model needed)
//! - Sentence embedding (via a MiniLM ONNX session, injected at runtime)
//! - Metadata features (source type, token count, entity density, etc.)
//!
//! The full pipeline is designed to complete in **<2ms** on modern hardware
//! so that the remaining budget is available for ONNX classifier inference.

use std::collections::HashMap;

use grist_event::GristEvent;
use ndarray::{Array1, Array2};
use tracing::instrument;

use crate::error::SieveError;

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

/// Embedding dimension produced by MiniLM-L6-v2.
pub const EMBEDDING_DIM: usize = 384;

/// Number of scalar metadata features appended after the embedding.
pub const METADATA_FEATURES: usize = 8;

/// Total feature vector length: embedding + metadata.
pub const FEATURE_DIM: usize = EMBEDDING_DIM + METADATA_FEATURES;

// ─────────────────────────────────────────────────────────────────────────────
// Feature vector
// ─────────────────────────────────────────────────────────────────────────────

/// Dense feature vector produced for each event.
///
/// Layout (length = [`FEATURE_DIM`]):
/// - `[0..384]`  — MiniLM-L6-v2 sentence embedding (L2-normalised)
/// - `[384]`     — approximate token count (log-scaled, 0-1)
/// - `[385]`     — source channel one-hot (0=cli, 1=http, 2=ws, 3=cron, …)
/// - `[386]`     — priority (0-3 mapped to 0.0-1.0)
/// - `[387]`     — entity density (entity count / token count, clamped 0-1)
/// - `[388]`     — question probability (presence of interrogative tokens)
/// - `[389]`     — code probability (presence of code-like tokens)
/// - `[390]`     — unique word ratio (type-token ratio, 0-1)
/// - `[391]`     — ambiguity score (low confidence indicator, 0-1)
#[derive(Debug, Clone)]
pub struct FeatureVector {
    pub data: Array1<f32>,
    /// SHA-256 hash of the normalised text (used as cache key).
    pub text_hash: String,
    /// Number of whitespace tokens in the source text.
    pub token_count: usize,
}

impl FeatureVector {
    /// Convert to a 2-D array suitable for ONNX batch input `[1, FEATURE_DIM]`.
    pub fn as_batch(&self) -> Array2<f32> {
        self.data.clone().insert_axis(ndarray::Axis(0))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Extractor
// ─────────────────────────────────────────────────────────────────────────────

/// Extracts features from a [`GristEvent`].
///
/// The embedder ONNX session is optional: if not provided (e.g. during tests
/// or cold-start before the model is loaded) a zero-vector is used for the
/// embedding slice.  Metadata features are always computed.
pub struct FeatureExtractor {
    /// Optional MiniLM ONNX session for embedding generation.
    /// `None` means embeddings are zeroed (fallback / test mode).
    embedder: Option<EmbedderSession>,
}

impl FeatureExtractor {
    /// Create an extractor with no embedder (metadata-only mode).
    pub fn new_no_embed() -> Self {
        Self { embedder: None }
    }

    /// Create an extractor with a live embedder session.
    pub fn new_with_embedder(embedder: EmbedderSession) -> Self {
        Self {
            embedder: Some(embedder),
        }
    }

    /// Extract a feature vector from an event.
    ///
    /// # Hot path
    /// This is called once per event by `Sieve::triage()`.  Keep allocations
    /// minimal and avoid any I/O.
    #[instrument(level = "trace", skip(self, event), fields(event_id = %event.id))]
    pub fn extract(&self, event: &GristEvent) -> Result<FeatureVector, SieveError> {
        let text = event.payload_as_text();
        let tokens: Vec<&str> = text.split_whitespace().collect();
        let token_count = tokens.len();

        // ── 1. Text hash (cache key) ─────────────────────────────────────────
        let text_hash = event.payload_hash();

        // ── 2. Sentence embedding ────────────────────────────────────────────
        let embedding: Array1<f32> = match &self.embedder {
            Some(emb) => emb.embed(&text)?,
            None => Array1::zeros(EMBEDDING_DIM),
        };

        // ── 3. Metadata features ─────────────────────────────────────────────
        let meta = extract_metadata_features(event, &tokens);

        // ── 4. Concatenate ───────────────────────────────────────────────────
        let mut data = Array1::<f32>::zeros(FEATURE_DIM);
        data.slice_mut(ndarray::s![0..EMBEDDING_DIM])
            .assign(&embedding);
        data.slice_mut(ndarray::s![EMBEDDING_DIM..FEATURE_DIM])
            .assign(&meta);

        Ok(FeatureVector {
            data,
            text_hash,
            token_count,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Metadata feature extraction (pure, no model)
// ─────────────────────────────────────────────────────────────────────────────

fn extract_metadata_features(event: &GristEvent, tokens: &[&str]) -> Array1<f32> {
    let mut feats = Array1::<f32>::zeros(METADATA_FEATURES);
    let tc = tokens.len().max(1) as f32;

    // [0] Token count — log-scaled to [0, 1] assuming max ~2048 tokens.
    feats[0] = (tc.ln() / (2048_f32).ln()).clamp(0.0, 1.0);

    // [1] Source channel (ordinal encoding, normalised to [0, 1]).
    feats[1] = channel_ordinal(&event.source) / 9.0;

    // [2] Priority (0-3 → 0.0-1.0).
    feats[2] = event.metadata.priority as u8 as f32 / 3.0;

    // [3] Entity density.
    let entity_count = estimate_entity_count(tokens) as f32;
    feats[3] = (entity_count / tc).clamp(0.0, 1.0);

    // [4] Question probability (presence of interrogative tokens).
    feats[4] = question_probability(tokens);

    // [5] Code probability.
    feats[5] = code_probability(tokens);

    // [6] Unique word ratio (type-token ratio).
    let unique: std::collections::HashSet<&&str> = tokens.iter().collect();
    feats[6] = (unique.len() as f32 / tc).clamp(0.0, 1.0);

    // [7] Ambiguity score — inverse of max TF fraction (vocabulary diversity).
    let ambiguity = ttr_ambiguity(tokens);
    feats[7] = ambiguity;

    feats
}

fn channel_ordinal(ch: &grist_event::ChannelType) -> f32 {
    use grist_event::ChannelType::*;
    match ch {
        Cli => 0.0,
        Http => 1.0,
        WebSocket => 2.0,
        Cron => 3.0,
        Webhook { .. } => 4.0,
        MessageQueue { .. } => 5.0,
        FileSystem { .. } => 6.0,
        Python { .. } => 7.0,
        TypeScript { .. } => 8.0,
        Internal { .. } => 9.0,
    }
}

/// Rough entity count: capitalised words that aren't sentence-initial.
fn estimate_entity_count(tokens: &[&str]) -> usize {
    tokens
        .iter()
        .enumerate()
        .filter(|(i, t)| {
            *i > 0
                && t.chars()
                    .next()
                    .map(|c| c.is_uppercase())
                    .unwrap_or(false)
        })
        .count()
}

/// Returns a score ∈ [0, 1] indicating how likely this is a question.
fn question_probability(tokens: &[&str]) -> f32 {
    let interrogatives = ["who", "what", "when", "where", "why", "how", "which", "whom"];
    let has_question_mark = tokens.last().map(|t| t.ends_with('?')).unwrap_or(false);
    let has_interrogative = tokens
        .iter()
        .any(|t| interrogatives.contains(&t.to_lowercase().trim_matches('?')));

    match (has_question_mark, has_interrogative) {
        (true, true) => 1.0,
        (true, false) | (false, true) => 0.6,
        _ => 0.0,
    }
}

/// Returns a score ∈ [0, 1] indicating how likely the text contains code.
fn code_probability(tokens: &[&str]) -> f32 {
    let code_indicators = [
        "fn ", "def ", "class ", "import ", "return ", "if ", "else", "for ", "while ",
        "{", "}", "()", "=>", "->", "//", "/*", "```", "#!", "#!/",
    ];
    let raw = tokens.join(" ");
    let hits = code_indicators
        .iter()
        .filter(|&&ind| raw.contains(ind))
        .count();
    (hits as f32 / code_indicators.len() as f32).clamp(0.0, 1.0)
}

/// Ambiguity: how evenly distributed the vocabulary is.
/// High TTR → diverse vocabulary → more ambiguous (harder to classify).
fn ttr_ambiguity(tokens: &[&str]) -> f32 {
    if tokens.is_empty() {
        return 0.5;
    }
    let mut freq: HashMap<&str, usize> = HashMap::new();
    for t in tokens {
        *freq.entry(t).or_insert(0) += 1;
    }
    let max_freq = *freq.values().max().unwrap_or(&1) as f32;
    let tc = tokens.len() as f32;
    // If one word dominates (low TTR) → low ambiguity.
    // If all words are unique (high TTR) → high ambiguity.
    (1.0 - (max_freq / tc)).clamp(0.0, 1.0)
}

// ─────────────────────────────────────────────────────────────────────────────
// EmbedderSession (thin wrapper, injected from grist-grinders at runtime)
// ─────────────────────────────────────────────────────────────────────────────

/// Thin wrapper around an ONNX MiniLM session.
///
/// In the full system this wraps an `ort::Session`.  Here we keep it as a
/// trait object so the sieve crate does not hard-depend on the grinders crate
/// (that would create a circular dependency via the feature-extraction path).
/// The grinders crate constructs a concrete `EmbedderSession` and injects it.
pub struct EmbedderSession {
    inner: Box<dyn EmbedFn + Send + Sync>,
}

impl std::fmt::Debug for EmbedderSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "EmbedderSession {{ .. }}")
    }
}

/// Trait for embedding callables, allowing injection of real or mock sessions.
pub trait EmbedFn {
    fn embed(&self, text: &str) -> Result<Array1<f32>, SieveError>;
}

impl EmbedderSession {
    /// Create from any callable that maps text → embedding.
    pub fn from_fn<F>(f: F) -> Self
    where
        F: Fn(&str) -> Result<Array1<f32>, SieveError> + Send + Sync + 'static,
    {
        struct Wrapper<F>(F);
        impl<F> EmbedFn for Wrapper<F>
        where
            F: Fn(&str) -> Result<Array1<f32>, SieveError> + Send + Sync,
        {
            fn embed(&self, text: &str) -> Result<Array1<f32>, SieveError> {
                (self.0)(text)
            }
        }
        Self {
            inner: Box::new(Wrapper(f)),
        }
    }

    pub fn embed(&self, text: &str) -> Result<Array1<f32>, SieveError> {
        self.inner.embed(text)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use grist_event::{ChannelType, GristEvent};

    fn make_event(text: &str) -> GristEvent {
        GristEvent::new(
            ChannelType::Http,
            serde_json::json!({ "text": text }),
        )
    }

    #[test]
    fn extract_no_embed_returns_correct_dim() {
        let extractor = FeatureExtractor::new_no_embed();
        let event = make_event("Schedule a meeting with Alice tomorrow at 10am");
        let fv = extractor.extract(&event).unwrap();
        assert_eq!(fv.data.len(), FEATURE_DIM);
    }

    #[test]
    fn embedding_zeros_in_no_embed_mode() {
        let extractor = FeatureExtractor::new_no_embed();
        let event = make_event("test");
        let fv = extractor.extract(&event).unwrap();
        // First EMBEDDING_DIM elements should all be 0.
        assert!(fv.data.slice(ndarray::s![0..EMBEDDING_DIM]).iter().all(|&x| x == 0.0));
    }

    #[test]
    fn metadata_features_in_range() {
        let extractor = FeatureExtractor::new_no_embed();
        let event = make_event("What is the capital of France?");
        let fv = extractor.extract(&event).unwrap();
        for &v in fv.data.slice(ndarray::s![EMBEDDING_DIM..]).iter() {
            assert!((0.0..=1.0).contains(&v), "metadata feature {v} out of range");
        }
    }

    #[test]
    fn question_detected() {
        let extractor = FeatureExtractor::new_no_embed();
        let event = make_event("What is the capital of France?");
        let fv = extractor.extract(&event).unwrap();
        // [4] is question probability
        assert!(fv.data[EMBEDDING_DIM + 4] > 0.5, "should detect question");
    }

    #[test]
    fn hash_is_deterministic_across_extractions() {
        let extractor = FeatureExtractor::new_no_embed();
        let event = make_event("hello world");
        let fv1 = extractor.extract(&event).unwrap();
        let fv2 = extractor.extract(&event).unwrap();
        assert_eq!(fv1.text_hash, fv2.text_hash);
    }

    #[test]
    fn token_count_matches() {
        let extractor = FeatureExtractor::new_no_embed();
        let event = make_event("one two three four five");
        let fv = extractor.extract(&event).unwrap();
        assert_eq!(fv.token_count, 5);
    }
}
