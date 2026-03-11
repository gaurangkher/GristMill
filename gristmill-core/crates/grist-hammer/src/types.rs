//! Core data types for grist-hammer.

use serde::{Deserialize, Serialize};
use ulid::Ulid;

// ─────────────────────────────────────────────────────────────────────────────
// EscalationRequest
// ─────────────────────────────────────────────────────────────────────────────

/// Request to escalate a prompt to an LLM provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EscalationRequest {
    /// Unique request identifier (ULID).
    pub id: String,
    /// The user prompt to send to the LLM.
    pub prompt: String,
    /// Optional system instruction.
    pub system: Option<String>,
    /// Maximum tokens to generate.
    pub max_tokens: u32,
    /// Optional model override (skips the default routing logic).
    pub model_override: Option<String>,
    /// Pre-computed prompt embedding for semantic cache lookup.
    /// If `None`, the cache falls back to exact SHA-256 matching only.
    pub embedding: Option<Vec<f32>>,
}

impl EscalationRequest {
    /// Create a minimal escalation request with default settings.
    pub fn new(prompt: impl Into<String>, max_tokens: u32) -> Self {
        Self {
            id: Ulid::new().to_string(),
            prompt: prompt.into(),
            system: None,
            max_tokens,
            model_override: None,
            embedding: None,
        }
    }

    /// Attach a system prompt.
    pub fn with_system(mut self, system: impl Into<String>) -> Self {
        self.system = Some(system.into());
        self
    }

    /// Attach a pre-computed embedding (enables fuzzy cache lookup).
    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }

    /// Conservative token estimate (prompt chars ÷ 4, rounded up).
    pub fn estimated_tokens(&self) -> u32 {
        let prompt_tokens = (self.prompt.len() as u32).div_ceil(4);
        let system_tokens = self
            .system
            .as_deref()
            .map(|s| (s.len() as u32).div_ceil(4))
            .unwrap_or(0);
        prompt_tokens + system_tokens + self.max_tokens
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// EscalationResponse
// ─────────────────────────────────────────────────────────────────────────────

/// Response from an LLM escalation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EscalationResponse {
    /// Mirrors `EscalationRequest::id`.
    pub request_id: String,
    /// Generated text content.
    pub content: String,
    /// Which provider served this request.
    pub provider: Provider,
    /// `true` if the response came from the semantic cache.
    pub cache_hit: bool,
    /// Tokens consumed (input + output; estimated for Ollama).
    pub tokens_used: u32,
    /// Wall-clock time from submission to response (ms).
    pub elapsed_ms: u64,
}

// ─────────────────────────────────────────────────────────────────────────────
// Provider
// ─────────────────────────────────────────────────────────────────────────────

/// Which LLM provider handled a request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Provider {
    /// Anthropic primary model (claude-sonnet-4-*).
    AnthropicPrimary,
    /// Anthropic fallback model (claude-haiku-*).
    AnthropicFallback,
    /// Local Ollama instance.
    Ollama,
    /// Response came from the semantic cache (no provider call).
    Cache,
}

impl Provider {
    pub fn label(self) -> &'static str {
        match self {
            Provider::AnthropicPrimary => "anthropic_primary",
            Provider::AnthropicFallback => "anthropic_fallback",
            Provider::Ollama => "ollama",
            Provider::Cache => "cache",
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// BudgetInfo
// ─────────────────────────────────────────────────────────────────────────────

/// Snapshot of current budget state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetInfo {
    pub daily_used: u64,
    pub daily_limit: u64,
    pub monthly_used: u64,
    pub monthly_limit: u64,
    pub daily_remaining: u64,
    pub monthly_remaining: u64,
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn escalation_request_defaults() {
        let r = EscalationRequest::new("hello", 100);
        assert!(!r.id.is_empty());
        assert_eq!(r.prompt, "hello");
        assert_eq!(r.max_tokens, 100);
        assert!(r.system.is_none());
        assert!(r.embedding.is_none());
    }

    #[test]
    fn estimated_tokens_conservative() {
        let r = EscalationRequest::new("a".repeat(400), 200);
        let est = r.estimated_tokens();
        // 400 chars ÷ 4 = 100 prompt tokens + 200 max_tokens = 300 minimum
        assert!(est >= 300);
    }

    #[test]
    fn provider_labels() {
        assert_eq!(Provider::AnthropicPrimary.label(), "anthropic_primary");
        assert_eq!(Provider::Ollama.label(), "ollama");
        assert_eq!(Provider::Cache.label(), "cache");
    }
}
