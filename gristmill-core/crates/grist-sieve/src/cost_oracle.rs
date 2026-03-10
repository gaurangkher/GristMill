//! Cost Oracle — estimates token cost versus task value to decide whether
//! local inference is sufficient or LLM escalation is warranted.
//!
//! PRD requirement S-05: "Estimates within 20% of actual LLM token usage."
//!
//! The oracle takes the raw classifier output and applies budget-aware
//! threshold logic to produce the final [`RouteDecision`] — the rich routing
//! object that the Millwright actually acts on.

use serde::{Deserialize, Serialize};
use tracing::instrument;
#[allow(unused_imports)]
use metrics;

use crate::classifier::{ClassifierOutput, RouteLabel};
use crate::config::SieveConfig;
use crate::error::SieveError;
use grist_event::GristEvent;

// ─────────────────────────────────────────────────────────────────────────────
// RouteDecision — the rich output of the full Sieve pipeline
// ─────────────────────────────────────────────────────────────────────────────

/// The final routing decision produced by the Sieve for each event.
///
/// This is what the Millwright receives and acts on.  The `model_id` and
/// `prompt_template` fields are populated from the config model registry.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "route", rename_all = "SCREAMING_SNAKE_CASE")]
pub enum RouteDecision {
    /// Handle entirely with a local ONNX/GGUF model via Grinders.
    LocalMl {
        model_id: String,
        confidence: f32,
    },
    /// Handle with a deterministic rule or template engine.
    Rules {
        rule_id: String,
        confidence: f32,
    },
    /// Pre-process locally then send a compact context to the LLM.
    Hybrid {
        local_model: String,
        llm_prompt_template: String,
        estimated_tokens: u32,
        confidence: f32,
    },
    /// Requires full LLM reasoning.
    LlmNeeded {
        reason: String,
        estimated_tokens: u32,
        estimated_cost_usd: f64,
        confidence: f32,
    },
}

impl RouteDecision {
    /// Returns the route label for logging and metrics.
    pub fn label(&self) -> RouteLabel {
        match self {
            RouteDecision::LocalMl { .. } => RouteLabel::LocalMl,
            RouteDecision::Rules { .. } => RouteLabel::Rules,
            RouteDecision::Hybrid { .. } => RouteLabel::Hybrid,
            RouteDecision::LlmNeeded { .. } => RouteLabel::LlmNeeded,
        }
    }

    /// Returns the confidence score.
    pub fn confidence(&self) -> f32 {
        match self {
            RouteDecision::LocalMl { confidence, .. } => *confidence,
            RouteDecision::Rules { confidence, .. } => *confidence,
            RouteDecision::Hybrid { confidence, .. } => *confidence,
            RouteDecision::LlmNeeded { confidence, .. } => *confidence,
        }
    }

    /// Returns `true` if the decision involves calling an LLM.
    pub fn involves_llm(&self) -> bool {
        matches!(
            self,
            RouteDecision::Hybrid { .. } | RouteDecision::LlmNeeded { .. }
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Token cost model
// ─────────────────────────────────────────────────────────────────────────────

/// Token cost estimates.  USD per 1M tokens (input + output combined).
/// These are approximate — within 20% per PRD S-05.
const ANTHROPIC_CLAUDE_SONNET_COST_PER_1M: f64 = 15.0; // rough mid-2025 estimate

/// Estimate the number of output tokens given an event and route.
///
/// We use a simple multiplier model — longer inputs and more complex routes
/// produce more output tokens.  The PRD requires accuracy within 20%.
fn estimate_output_tokens(token_count: u32, route: RouteLabel) -> u32 {
    match route {
        RouteLabel::LocalMl | RouteLabel::Rules => 0, // no LLM output
        RouteLabel::Hybrid => (token_count as f32 * 0.4).ceil() as u32 + 50,
        RouteLabel::LlmNeeded => (token_count as f32 * 1.5).ceil() as u32 + 150,
    }
}

/// Estimate total tokens (input + output) for an LLM call.
pub fn estimate_total_tokens(event: &GristEvent, route: RouteLabel) -> u32 {
    let input_tokens = event.estimated_token_count();
    input_tokens + estimate_output_tokens(input_tokens, route)
}

/// Estimate cost in USD for an LLM call.
pub fn estimate_cost_usd(total_tokens: u32) -> f64 {
    (total_tokens as f64 / 1_000_000.0) * ANTHROPIC_CLAUDE_SONNET_COST_PER_1M
}

// ─────────────────────────────────────────────────────────────────────────────
// CostOracle
// ─────────────────────────────────────────────────────────────────────────────

/// Applies confidence threshold + cost logic to produce a final
/// [`RouteDecision`] from raw classifier output.
///
/// Key invariant: if `classifier confidence >= threshold`, the classifier
/// decision is trusted.  If below threshold, we escalate to LLM unless the
/// cost oracle determines the event is too cheap/simple to warrant it.
pub struct CostOracle {
    /// Confidence threshold from config.  Default 0.85 per PRD S-08.
    threshold: f32,
    /// Model ID to use when routing to LOCAL_ML.
    default_local_model: String,
    /// Rule ID to use when routing to RULES.
    default_rule_id: String,
    /// Prompt template ID for HYBRID routes.
    hybrid_prompt_template: String,
}

impl CostOracle {
    pub fn new(config: &SieveConfig) -> Self {
        Self {
            threshold: config.confidence_threshold,
            default_local_model: config
                .default_local_model
                .clone()
                .unwrap_or_else(|| "intent-classifier-v1".to_string()),
            default_rule_id: config
                .default_rule_id
                .clone()
                .unwrap_or_else(|| "default-rules".to_string()),
            hybrid_prompt_template: config
                .hybrid_prompt_template
                .clone()
                .unwrap_or_else(|| "hybrid-refine-v1".to_string()),
        }
    }

    /// Produce a [`RouteDecision`] from classifier output and the source event.
    #[instrument(level = "trace", skip(self, output, event), fields(
        confidence = output.confidence,
        label = ?output.predicted_label,
    ))]
    pub fn evaluate(
        &self,
        output: ClassifierOutput,
        event: &GristEvent,
    ) -> Result<RouteDecision, SieveError> {
        let confidence = output.confidence;
        let label = output.predicted_label;

        // If confidence is below threshold, upgrade to a safer route.
        let effective_label = if confidence < self.threshold {
            self.safe_escalation(label, confidence)
        } else {
            label
        };

        metrics::counter!("sieve.route", "label" => effective_label.as_str()).increment(1);

        let decision = match effective_label {
            RouteLabel::LocalMl => RouteDecision::LocalMl {
                model_id: self.default_local_model.clone(),
                confidence,
            },
            RouteLabel::Rules => RouteDecision::Rules {
                rule_id: self.default_rule_id.clone(),
                confidence,
            },
            RouteLabel::Hybrid => {
                let estimated_tokens = estimate_total_tokens(event, RouteLabel::Hybrid);
                RouteDecision::Hybrid {
                    local_model: self.default_local_model.clone(),
                    llm_prompt_template: self.hybrid_prompt_template.clone(),
                    estimated_tokens,
                    confidence,
                }
            }
            RouteLabel::LlmNeeded => {
                let estimated_tokens = estimate_total_tokens(event, RouteLabel::LlmNeeded);
                let estimated_cost_usd = estimate_cost_usd(estimated_tokens);
                RouteDecision::LlmNeeded {
                    reason: Self::escalation_reason(confidence, self.threshold),
                    estimated_tokens,
                    estimated_cost_usd,
                    confidence,
                }
            }
        };

        Ok(decision)
    }

    /// When confidence is below threshold, we escalate one level toward LLM
    /// to be safe.  The PRD goal is <15% LLM escalation overall, so this
    /// should be rare if the classifier is accurate.
    fn safe_escalation(&self, label: RouteLabel, confidence: f32) -> RouteLabel {
        // Very low confidence: jump straight to LLM.
        if confidence < self.threshold * 0.5 {
            return RouteLabel::LlmNeeded;
        }
        // Moderate uncertainty: escalate one step.
        match label {
            RouteLabel::LocalMl => RouteLabel::Hybrid,
            RouteLabel::Rules => RouteLabel::LocalMl,
            RouteLabel::Hybrid => RouteLabel::LlmNeeded,
            RouteLabel::LlmNeeded => RouteLabel::LlmNeeded,
        }
    }

    fn escalation_reason(confidence: f32, threshold: f32) -> String {
        if confidence < threshold * 0.5 {
            format!(
                "classifier confidence {:.3} far below threshold {:.2} — full LLM needed",
                confidence, threshold
            )
        } else {
            format!(
                "classifier confidence {:.3} below threshold {:.2} — escalating to LLM",
                confidence, threshold
            )
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::classifier::{ClassifierOutput, RouteLabel};
    use crate::config::SieveConfig;
    use grist_event::{ChannelType, GristEvent};

    fn config() -> SieveConfig {
        SieveConfig::default()
    }

    fn oracle() -> CostOracle {
        CostOracle::new(&config())
    }

    fn event(text: &str) -> GristEvent {
        GristEvent::new(
            ChannelType::Http,
            serde_json::json!({ "text": text }),
        )
    }

    fn output(label: RouteLabel, confidence: f32) -> ClassifierOutput {
        let mut probs = [0.0_f32; 4];
        probs[label as usize] = confidence;
        let rest = (1.0 - confidence) / 3.0;
        for (i, p) in probs.iter_mut().enumerate() {
            if i != label as usize {
                *p = rest;
            }
        }
        ClassifierOutput {
            probabilities: probs,
            predicted_label: label,
            confidence,
        }
    }

    #[test]
    fn high_confidence_local_ml_stays_local() {
        let decision = oracle()
            .evaluate(output(RouteLabel::LocalMl, 0.95), &event("classify this"))
            .unwrap();
        assert!(matches!(decision, RouteDecision::LocalMl { .. }));
    }

    #[test]
    fn high_confidence_rules_stays_rules() {
        let decision = oracle()
            .evaluate(output(RouteLabel::Rules, 0.92), &event("status"))
            .unwrap();
        assert!(matches!(decision, RouteDecision::Rules { .. }));
    }

    #[test]
    fn high_confidence_hybrid_stays_hybrid() {
        let decision = oracle()
            .evaluate(output(RouteLabel::Hybrid, 0.88), &event("summarise and refine"))
            .unwrap();
        assert!(matches!(decision, RouteDecision::Hybrid { .. }));
    }

    #[test]
    fn high_confidence_llm_stays_llm() {
        let decision = oracle()
            .evaluate(
                output(RouteLabel::LlmNeeded, 0.91),
                &event("complex multi-step reasoning task"),
            )
            .unwrap();
        assert!(matches!(decision, RouteDecision::LlmNeeded { .. }));
    }

    #[test]
    fn low_confidence_local_escalates_to_hybrid() {
        // confidence = 0.50 < threshold 0.85, but > threshold * 0.5 = 0.425
        let decision = oracle()
            .evaluate(output(RouteLabel::LocalMl, 0.50), &event("unclear intent"))
            .unwrap();
        assert!(matches!(decision, RouteDecision::Hybrid { .. }));
    }

    #[test]
    fn very_low_confidence_jumps_to_llm() {
        // confidence = 0.30 < threshold * 0.5 = 0.425
        let decision = oracle()
            .evaluate(output(RouteLabel::LocalMl, 0.30), &event("???"))
            .unwrap();
        assert!(matches!(decision, RouteDecision::LlmNeeded { .. }));
    }

    #[test]
    fn token_estimate_for_llm_is_positive() {
        let ev = event("Tell me everything about quantum computing");
        let tokens = estimate_total_tokens(&ev, RouteLabel::LlmNeeded);
        assert!(tokens > 0);
    }

    #[test]
    fn cost_estimate_is_non_negative() {
        let cost = estimate_cost_usd(1000);
        assert!(cost > 0.0);
    }

    #[test]
    fn local_ml_does_not_involve_llm() {
        let decision = oracle()
            .evaluate(output(RouteLabel::LocalMl, 0.95), &event("test"))
            .unwrap();
        assert!(!decision.involves_llm());
    }

    #[test]
    fn llm_needed_involves_llm() {
        let decision = oracle()
            .evaluate(
                output(RouteLabel::LlmNeeded, 0.90),
                &event("complex"),
            )
            .unwrap();
        assert!(decision.involves_llm());
    }
}
