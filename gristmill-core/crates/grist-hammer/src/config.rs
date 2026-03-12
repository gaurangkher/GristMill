//! Configuration for grist-hammer.

use serde::{Deserialize, Serialize};

// ─────────────────────────────────────────────────────────────────────────────
// Top-level HammerConfig
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct HammerConfig {
    #[serde(default)]
    pub providers: ProvidersConfig,
    #[serde(default)]
    pub budget: BudgetConfig,
    #[serde(default)]
    pub cache: CacheConfig,
    #[serde(default)]
    pub batch: BatchConfig,
}

// ─────────────────────────────────────────────────────────────────────────────
// ProvidersConfig
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProvidersConfig {
    #[serde(default)]
    pub anthropic: AnthropicConfig,
    #[serde(default)]
    pub ollama: OllamaConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicConfig {
    /// API key — in production read from `ANTHROPIC_API_KEY` env var.
    #[serde(default)]
    pub api_key: String,
    /// Primary model (spec: claude-sonnet-4-20250514).
    #[serde(default = "default_anthropic_model")]
    pub default_model: String,
    /// Fallback model within Anthropic (spec: claude-haiku-4-5-20251001).
    #[serde(default = "default_anthropic_fallback_model")]
    pub fallback_model: String,
    /// Base URL for the Anthropic API.
    #[serde(default = "default_anthropic_base_url")]
    pub base_url: String,
}

fn default_anthropic_model() -> String {
    "claude-sonnet-4-20250514".into()
}
fn default_anthropic_fallback_model() -> String {
    "claude-haiku-4-5-20251001".into()
}
fn default_anthropic_base_url() -> String {
    "https://api.anthropic.com".into()
}

impl Default for AnthropicConfig {
    fn default() -> Self {
        // Fall back to env-var at runtime if api_key is blank.
        let api_key = std::env::var("ANTHROPIC_API_KEY").unwrap_or_default();
        Self {
            api_key,
            default_model: default_anthropic_model(),
            fallback_model: default_anthropic_fallback_model(),
            base_url: default_anthropic_base_url(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaConfig {
    /// Base URL of the running Ollama instance.
    #[serde(default = "default_ollama_base_url")]
    pub base_url: String,
    /// Model to use (spec: llama3.1:8b).
    #[serde(default = "default_ollama_model")]
    pub model: String,
}

fn default_ollama_base_url() -> String {
    "http://localhost:11434".into()
}
fn default_ollama_model() -> String {
    "llama3.1:8b".into()
}

impl Default for OllamaConfig {
    fn default() -> Self {
        Self {
            base_url: default_ollama_base_url(),
            model: default_ollama_model(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// BudgetConfig
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetConfig {
    /// Maximum tokens allowed per day (spec: 500,000).
    #[serde(default = "default_daily_tokens")]
    pub daily_tokens: u64,
    /// Maximum tokens allowed per calendar month (spec: 10,000,000).
    #[serde(default = "default_monthly_tokens")]
    pub monthly_tokens: u64,
}

fn default_daily_tokens() -> u64 {
    500_000
}
fn default_monthly_tokens() -> u64 {
    10_000_000
}

impl Default for BudgetConfig {
    fn default() -> Self {
        Self {
            daily_tokens: default_daily_tokens(),
            monthly_tokens: default_monthly_tokens(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CacheConfig
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Enable semantic caching.
    #[serde(default = "default_cache_enabled")]
    pub enabled: bool,
    /// Cosine similarity threshold for fuzzy cache hits (spec: 0.92).
    #[serde(default = "default_similarity_threshold")]
    pub similarity_threshold: f32,
    /// Maximum cache entries before LRU eviction (spec: 50,000).
    #[serde(default = "default_max_entries")]
    pub max_entries: usize,
    /// Embedding dimension for cosine similarity (384 for MiniLM-L6-v2).
    #[serde(default = "default_embedding_dim")]
    pub embedding_dim: usize,
    /// Cache TTL in seconds (0 = no expiry).
    #[serde(default)]
    pub ttl_secs: u64,
}

fn default_cache_enabled() -> bool {
    true
}
fn default_similarity_threshold() -> f32 {
    0.92
}
fn default_max_entries() -> usize {
    50_000
}
fn default_embedding_dim() -> usize {
    384
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            enabled: default_cache_enabled(),
            similarity_threshold: default_similarity_threshold(),
            max_entries: default_max_entries(),
            embedding_dim: default_embedding_dim(),
            ttl_secs: 0,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// BatchConfig
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchConfig {
    /// Enable request batching.
    #[serde(default = "default_batch_enabled")]
    pub enabled: bool,
    /// Accumulation window in milliseconds (spec: 5000).
    #[serde(default = "default_window_ms")]
    pub window_ms: u64,
    /// Maximum requests per batch (spec: 10).
    #[serde(default = "default_max_batch_size")]
    pub max_batch_size: usize,
}

fn default_batch_enabled() -> bool {
    true
}
fn default_window_ms() -> u64 {
    5000
}
fn default_max_batch_size() -> usize {
    10
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            enabled: default_batch_enabled(),
            window_ms: default_window_ms(),
            max_batch_size: default_max_batch_size(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_has_expected_values() {
        let c = HammerConfig::default();
        assert_eq!(c.budget.daily_tokens, 500_000);
        assert_eq!(c.budget.monthly_tokens, 10_000_000);
        assert!((c.cache.similarity_threshold - 0.92).abs() < 1e-6);
        assert_eq!(c.cache.max_entries, 50_000);
        assert_eq!(c.batch.window_ms, 5000);
        assert_eq!(c.batch.max_batch_size, 10);
        assert_eq!(
            c.providers.anthropic.default_model,
            "claude-sonnet-4-20250514"
        );
        assert_eq!(c.providers.ollama.model, "llama3.1:8b");
    }

    #[test]
    fn config_json_roundtrip() {
        let c = HammerConfig::default();
        let json = serde_json::to_string(&c).unwrap();
        let c2: HammerConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(c.budget.daily_tokens, c2.budget.daily_tokens);
        assert_eq!(c.cache.max_entries, c2.cache.max_entries);
    }
}
