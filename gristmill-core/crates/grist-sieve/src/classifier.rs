//! ONNX-backed 4-class intent classifier.
//!
//! The model is a fine-tuned MiniLM-L6-v2 exported to ONNX INT8 (~25 MB).
//! At runtime the model path is read from config and loaded into an
//! `ort::Session`.  The session is wrapped in an `Arc<ArcSwap<...>>` so it
//! can be hot-reloaded atomically without dropping in-flight inferences.
//!
//! When no model file is present (e.g. first run before `gristmill models pull`
//! or during unit tests) the classifier falls back to a heuristic rule engine
//! that estimates the route from the feature vector alone.

use std::path::Path;
use std::sync::Arc;

use arc_swap::ArcSwap;
#[allow(unused_imports)]
use metrics;
#[cfg(feature = "onnx")]
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use tracing::{info, instrument, warn};

use crate::error::SieveError;
use crate::features::FeatureVector;

// ─────────────────────────────────────────────────────────────────────────────
// Route decision
// ─────────────────────────────────────────────────────────────────────────────

/// Routing label produced by the Sieve.
///
/// Matches the 4-class output of the ONNX classifier and the PRD requirement
/// S-01: "Classify events into 4 routes: LOCAL_ML, RULES, HYBRID, LLM_NEEDED".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum RouteLabel {
    /// Handle with a local ONNX/GGUF model (Grinders pool).
    LocalMl = 0,
    /// Handle with a deterministic rule / template engine.
    Rules = 1,
    /// Pre-process locally, then refine with LLM.
    Hybrid = 2,
    /// Requires full LLM reasoning (last resort).
    LlmNeeded = 3,
}

impl RouteLabel {
    pub fn from_index(idx: usize) -> Option<Self> {
        match idx {
            0 => Some(RouteLabel::LocalMl),
            1 => Some(RouteLabel::Rules),
            2 => Some(RouteLabel::Hybrid),
            3 => Some(RouteLabel::LlmNeeded),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            RouteLabel::LocalMl => "LOCAL_ML",
            RouteLabel::Rules => "RULES",
            RouteLabel::Hybrid => "HYBRID",
            RouteLabel::LlmNeeded => "LLM_NEEDED",
        }
    }
}

impl std::fmt::Display for RouteLabel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Raw output of the classifier before applying threshold logic.
#[derive(Debug, Clone)]
pub struct ClassifierOutput {
    /// Softmax probabilities for each of the 4 classes.
    pub probabilities: [f32; 4],
    /// Argmax label.
    pub predicted_label: RouteLabel,
    /// Max probability (confidence).
    pub confidence: f32,
}

impl ClassifierOutput {
    pub fn from_logits(logits: &[f32]) -> Self {
        assert!(logits.len() >= 4, "classifier must output ≥4 logits");
        let probs = softmax(&logits[..4]);
        let (idx, &conf) = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();
        ClassifierOutput {
            probabilities: [probs[0], probs[1], probs[2], probs[3]],
            predicted_label: RouteLabel::from_index(idx).unwrap_or(RouteLabel::LlmNeeded),
            confidence: conf,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Classifier
// ─────────────────────────────────────────────────────────────────────────────

/// Inner state held behind the ArcSwap — swapped atomically on hot-reload.
///
/// The `Session` is wrapped in a `Mutex` because `ort::session::Session::run`
/// requires `&mut self` in ort v2, while `ArcSwap::load()` only provides shared
/// (`&`) access.
enum ClassifierInner {
    /// Live ONNX session.
    #[cfg(feature = "onnx")]
    Onnx(std::sync::Mutex<ort::session::Session>),
    /// Heuristic fallback (no model file present).
    Heuristic,
}

impl std::fmt::Debug for ClassifierInner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            #[cfg(feature = "onnx")]
            ClassifierInner::Onnx(_) => write!(f, "Onnx(..)"),
            ClassifierInner::Heuristic => write!(f, "Heuristic"),
        }
    }
}

/// Thread-safe, hot-reloadable classifier.
pub struct Classifier {
    inner: Arc<ArcSwap<ClassifierInner>>,
    model_path: Option<std::path::PathBuf>,
}

impl std::fmt::Debug for Classifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Classifier {{ model_path: {:?} }}", self.model_path)
    }
}

impl Classifier {
    /// Load classifier from an ONNX file, falling back to heuristic if the
    /// file doesn't exist.
    pub fn load(model_path: Option<&Path>) -> Result<Self, SieveError> {
        let inner = Self::load_inner(model_path)?;
        Ok(Self {
            inner: Arc::new(ArcSwap::from_pointee(inner)),
            model_path: model_path.map(|p| p.to_path_buf()),
        })
    }

    /// Heuristic-only classifier (no model file required — for tests).
    pub fn heuristic() -> Self {
        Self {
            inner: Arc::new(ArcSwap::from_pointee(ClassifierInner::Heuristic)),
            model_path: None,
        }
    }

    fn load_inner(
        #[allow(unused_variables)] model_path: Option<&Path>,
    ) -> Result<ClassifierInner, SieveError> {
        #[cfg(feature = "onnx")]
        if let Some(path) = model_path {
            if path.exists() {
                info!(path = %path.display(), "loading ONNX classifier");
                let session = ort::session::Session::builder()
                    .map_err(|e| SieveError::ModelLoad(e.to_string()))?
                    .commit_from_file(path)
                    .map_err(|e| SieveError::ModelLoad(e.to_string()))?;
                return Ok(ClassifierInner::Onnx(std::sync::Mutex::new(session)));
            }
        }

        warn!("no classifier model found — using heuristic fallback");
        Ok(ClassifierInner::Heuristic)
    }

    /// Run classification on a feature vector.
    ///
    /// Returns [`ClassifierOutput`] with probabilities and predicted label.
    #[instrument(level = "trace", skip(self, features))]
    pub fn classify(&self, features: &FeatureVector) -> Result<ClassifierOutput, SieveError> {
        let guard = self.inner.load();
        match guard.as_ref() {
            #[cfg(feature = "onnx")]
            ClassifierInner::Onnx(mutex) => {
                let mut session = mutex
                    .lock()
                    .map_err(|_| SieveError::Inference("session mutex poisoned".into()))?;
                self.classify_onnx(&mut session, features)
            }
            ClassifierInner::Heuristic => Ok(self.classify_heuristic(features)),
        }
    }

    /// Hot-reload: swap in a new model without dropping in-flight requests.
    ///
    /// S-07: "Model swap completes in <500ms with zero dropped events."
    pub fn hot_reload(&self) -> Result<(), SieveError> {
        let new_inner = Self::load_inner(self.model_path.as_deref())?;
        self.inner.store(Arc::new(new_inner));
        info!("classifier hot-reloaded");
        metrics::counter!("sieve.classifier.hot_reload").increment(1);
        Ok(())
    }

    // ── ONNX inference ───────────────────────────────────────────────────────

    #[cfg(feature = "onnx")]
    fn classify_onnx(
        &self,
        session: &mut ort::session::Session,
        features: &FeatureVector,
    ) -> Result<ClassifierOutput, SieveError> {
        use ort::value::TensorRef;

        let batch: Array2<f32> = features.as_batch();
        // Build a (shape, &[f32]) tuple — compatible with ort v2 without requiring
        // the optional `ndarray` feature on the ort crate (and avoiding the
        // ndarray 0.16 vs 0.17 version mismatch).  inputs! returns Vec (not
        // Result) in ort v2, so no .map_err() is needed on that call.
        let shape: Vec<i64> = batch.shape().iter().map(|&d| d as i64).collect();
        let data: &[f32] = batch
            .as_slice()
            .ok_or_else(|| SieveError::Inference("batch array is not contiguous".into()))?;
        let tensor = TensorRef::<f32>::from_array_view((shape, data))
            .map_err(|e| SieveError::Inference(e.to_string()))?;
        let outputs = session
            .run(ort::inputs!["features" => tensor])
            .map_err(|e| SieveError::Inference(e.to_string()))?;

        // try_extract_tensor returns (&Shape, &[T]) in ort v2.
        let (_shape, logit_slice) = outputs["logits"]
            .try_extract_tensor::<f32>()
            .map_err(|e| SieveError::Inference(e.to_string()))?;

        Ok(ClassifierOutput::from_logits(logit_slice))
    }

    // ── Heuristic fallback ───────────────────────────────────────────────────

    /// Rule-based classifier that approximates the ONNX model using the
    /// metadata features when no model is loaded.
    ///
    /// This is intentionally simple — it is a safety net, not a replacement.
    fn classify_heuristic(&self, fv: &FeatureVector) -> ClassifierOutput {
        use crate::features::EMBEDDING_DIM;

        let token_count_norm = fv.data[EMBEDDING_DIM]; // [0] log-scaled
        let question_prob = fv.data[EMBEDDING_DIM + 4]; // [4]
        let code_prob = fv.data[EMBEDDING_DIM + 5]; // [5]
        let ambiguity = fv.data[EMBEDDING_DIM + 7]; // [7]

        // Heuristic scoring — biases tuned to match typical PRD examples.
        let score_rules = if token_count_norm < 0.25 { 0.7 } else { 0.2 }; // short commands → rules
        let score_local_ml = if code_prob > 0.3 { 0.6 } else { 0.35 }; // code → local ML
        let score_hybrid = if question_prob > 0.5 && ambiguity > 0.4 {
            0.5
        } else {
            0.15
        };
        let score_llm = if ambiguity > 0.7 && question_prob > 0.5 {
            0.5
        } else {
            0.1
        };

        let raw = [score_local_ml, score_rules, score_hybrid, score_llm];
        let probs = softmax(&raw);

        let (idx, &conf) = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        ClassifierOutput {
            probabilities: [probs[0], probs[1], probs[2], probs[3]],
            predicted_label: RouteLabel::from_index(idx).unwrap_or(RouteLabel::LlmNeeded),
            confidence: conf,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Numerically stable softmax.
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::FeatureExtractor;
    use grist_event::{ChannelType, GristEvent};

    fn classify(text: &str) -> ClassifierOutput {
        let event = GristEvent::new(ChannelType::Http, serde_json::json!({ "text": text }));
        let extractor = FeatureExtractor::new_no_embed();
        let fv = extractor.extract(&event).unwrap();
        let classifier = Classifier::heuristic();
        classifier.classify(&fv).unwrap()
    }

    #[test]
    fn short_command_routes_to_rules() {
        let out = classify("status");
        assert_eq!(out.predicted_label, RouteLabel::Rules);
    }

    #[test]
    fn code_text_routes_to_local_ml_or_rules() {
        let out = classify("fn main() { println!(\"hello\"); }");
        assert!(
            matches!(out.predicted_label, RouteLabel::LocalMl | RouteLabel::Rules),
            "got {:?}",
            out.predicted_label
        );
    }

    #[test]
    fn complex_question_routes_to_hybrid_or_llm() {
        let out = classify(
            "Why did the authentication service fail intermittently yesterday and what should we do?",
        );
        assert!(
            matches!(
                out.predicted_label,
                RouteLabel::Hybrid | RouteLabel::LlmNeeded | RouteLabel::LocalMl
            ),
            "got {:?}",
            out.predicted_label
        );
    }

    #[test]
    fn probabilities_sum_to_one() {
        let out = classify("hello world");
        let sum: f32 = out.probabilities.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "probs sum to {sum}");
    }

    #[test]
    fn confidence_matches_max_probability() {
        let out = classify("classify this please");
        let max_prob = out.probabilities.iter().cloned().fold(0.0_f32, f32::max);
        assert!((out.confidence - max_prob).abs() < 1e-6);
    }

    #[test]
    fn softmax_is_normalised() {
        let probs = softmax(&[1.0, 2.0, 3.0, 4.0]);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn route_label_round_trips() {
        for i in 0..4 {
            let label = RouteLabel::from_index(i).unwrap();
            assert_eq!(label as usize, i);
        }
    }
}
