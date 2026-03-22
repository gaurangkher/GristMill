//! Provider routing for grist-hammer.
//!
//! [`RequestRouter`] attempts providers in priority order:
//!   1. **Ollama** (local open-source — eligible for training buffer)
//!   2. Anthropic primary model (commercial — NOT eligible for training buffer)
//!   3. Anthropic fallback model (commercial — NOT eligible for training buffer)
//!
//! **Rationale:** Per the GristMill distillation spec, commercial API outputs
//! must never be written to the training buffer (ToS violation).  By placing
//! Ollama first, the system maximises training signal from local open-source
//! responses and only falls back to Anthropic when Ollama is unavailable.
//!
//! On each provider failure a warning is logged and the next provider is tried.
//! If all fail, [`HammerError::AllProvidersFailed`] is returned.
//!
//! Test isolation: In `#[cfg(test)]` builds, the router accepts
//! `Vec<Box<dyn ProviderFn>>` that replaces the real HTTP dispatch so no live
//! network calls are made.

#[cfg(test)]
use std::sync::Arc;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use tracing::warn;

use crate::config::HammerConfig;
use crate::error::HammerError;
use crate::types::{EscalationRequest, EscalationResponse, Provider};

// Speculative cascade prompt template (Phase 3).
// Instructs the teacher to verify rather than generate from scratch.
const DRAFT_VERIFICATION_PREFIX: &str = "[GRINDER DRAFT — verify and correct if needed]\n";
const DRAFT_VERIFICATION_SEPARATOR: &str = "\n\n[ORIGINAL QUERY]\n";

// ─────────────────────────────────────────────────────────────────────────────
// ProviderFn trait (used for test injection)
// ─────────────────────────────────────────────────────────────────────────────

/// A callable that acts as a provider: takes a request and returns a response.
///
/// In production the router builds concrete closures that call Anthropic/Ollama
/// over HTTP.  In tests, mock implementations are injected instead.
pub trait ProviderFn: Send + Sync {
    fn call(&self, req: &EscalationRequest) -> Result<(String, u32, Provider), HammerError>;
}

// ─────────────────────────────────────────────────────────────────────────────
// Anthropic wire types
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Serialize)]
struct AnthropicMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Serialize)]
struct AnthropicRequest<'a> {
    model: &'a str,
    max_tokens: u32,
    messages: Vec<AnthropicMessage<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<&'a str>,
}

#[derive(Deserialize)]
struct AnthropicResponse {
    content: Vec<AnthropicContent>,
    usage: AnthropicUsage,
}

#[derive(Deserialize)]
struct AnthropicContent {
    text: String,
}

#[derive(Deserialize)]
struct AnthropicUsage {
    input_tokens: u32,
    output_tokens: u32,
}

// ─────────────────────────────────────────────────────────────────────────────
// Ollama wire types
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Serialize)]
struct OllamaRequest<'a> {
    model: &'a str,
    prompt: &'a str,
    stream: bool,
}

#[derive(Deserialize)]
struct OllamaResponse {
    response: String,
    eval_count: Option<u32>,
    prompt_eval_count: Option<u32>,
}

// ─────────────────────────────────────────────────────────────────────────────
// RequestRouter
// ─────────────────────────────────────────────────────────────────────────────

pub struct RequestRouter {
    config: HammerConfig,
    http: reqwest::Client,
    /// Optional mock providers (only populated in tests).
    #[cfg(test)]
    mock_providers: Option<Vec<Arc<dyn ProviderFn>>>,
}

impl RequestRouter {
    pub fn new(config: HammerConfig) -> Result<Self, HammerError> {
        let http = reqwest::Client::builder()
            .build()
            .map_err(|e| HammerError::Config(format!("failed to build HTTP client: {e}")))?;
        Ok(Self {
            config,
            http,
            #[cfg(test)]
            mock_providers: None,
        })
    }

    /// Create a router with injected mock providers for unit tests.
    #[cfg(test)]
    pub fn with_mock_providers(config: HammerConfig, providers: Vec<Arc<dyn ProviderFn>>) -> Self {
        let http = reqwest::Client::new();
        Self {
            config,
            http,
            mock_providers: Some(providers),
        }
    }

    /// Route the request to the first available provider.
    pub async fn route(&self, req: &EscalationRequest) -> Result<EscalationResponse, HammerError> {
        let start = Instant::now();

        // ── Test path: use injected mock providers ───────────────────────
        #[cfg(test)]
        if let Some(mocks) = &self.mock_providers {
            let mut last_err = HammerError::AllProvidersFailed("no providers".into());
            for mock in mocks.iter() {
                match mock.as_ref().call(req) {
                    Ok((content, tokens, provider)) => {
                        return Ok(EscalationResponse {
                            request_id: req.id.clone(),
                            content,
                            provider_type: provider.provider_type(),
                            provider,
                            cache_hit: false,
                            tokens_used: tokens,
                            elapsed_ms: start.elapsed().as_millis() as u64,
                            teacher_cost_usd: self.cost_usd(provider, tokens),
                            used_draft: req.draft_response.is_some(),
                        });
                    }
                    Err(e) => {
                        warn!(provider = "mock", error = %e, "mock provider failed");
                        last_err = HammerError::AllProvidersFailed(e.to_string());
                    }
                }
            }
            return Err(last_err);
        }

        // ── Production path ───────────────────────────────────────────────
        // Priority order: Ollama (local, training-eligible) → Anthropic primary
        // → Anthropic fallback.  See module-level doc for rationale.

        // 1. Try Ollama first (local open-source — training buffer eligible).
        match self.call_ollama(req).await {
            Ok(resp) => return Ok(self.make_response(req, resp, Provider::Ollama, start)),
            Err(e) => {
                warn!(provider = "ollama", error = %e, "Ollama unavailable, falling back to Anthropic primary");
            }
        }

        // 2. Try Anthropic primary (commercial — NOT written to training buffer).
        let model = req
            .model_override
            .as_deref()
            .unwrap_or(&self.config.providers.anthropic.default_model);
        match self
            .call_anthropic(req, model, Provider::AnthropicPrimary)
            .await
        {
            Ok(resp) => {
                return Ok(self.make_response(req, resp, Provider::AnthropicPrimary, start))
            }
            Err(e) => {
                warn!(provider = "anthropic_primary", error = %e, "provider failed, trying Anthropic fallback");
            }
        }

        // 3. Try Anthropic fallback model (commercial — NOT written to training buffer).
        let fallback = &self.config.providers.anthropic.fallback_model.clone();
        match self
            .call_anthropic(req, fallback, Provider::AnthropicFallback)
            .await
        {
            Ok(resp) => Ok(self.make_response(req, resp, Provider::AnthropicFallback, start)),
            Err(e) => {
                warn!(provider = "anthropic_fallback", error = %e, "all providers failed");
                Err(HammerError::AllProvidersFailed(e.to_string()))
            }
        }
    }

    // ── Anthropic HTTP call ───────────────────────────────────────────────

    async fn call_anthropic(
        &self,
        req: &EscalationRequest,
        model: &str,
        _provider: Provider,
    ) -> Result<(String, u32), HammerError> {
        // Speculative cascade: if a grinder draft is present, ask the teacher
        // to verify and correct it rather than generating from scratch.
        // This typically reduces output tokens by 30–50%.
        let effective_prompt: String = if let Some(draft) = &req.draft_response {
            format!(
                "{}{}{}{}",
                DRAFT_VERIFICATION_PREFIX, draft, DRAFT_VERIFICATION_SEPARATOR, req.prompt
            )
        } else {
            req.prompt.clone()
        };

        let body = AnthropicRequest {
            model,
            max_tokens: req.max_tokens,
            messages: vec![AnthropicMessage {
                role: "user",
                content: &effective_prompt,
            }],
            system: req.system.as_deref(),
        };

        let url = format!("{}/v1/messages", self.config.providers.anthropic.base_url);
        let resp = self
            .http
            .post(&url)
            .header("x-api-key", &self.config.providers.anthropic.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(HammerError::Http)?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(HammerError::ProviderError {
                provider: model.to_string(),
                reason: format!("HTTP {status}: {text}"),
            });
        }

        let parsed: AnthropicResponse = resp.json().await.map_err(HammerError::Http)?;

        let content = parsed
            .content
            .into_iter()
            .map(|c| c.text)
            .collect::<Vec<_>>()
            .join("");
        let tokens = parsed.usage.input_tokens + parsed.usage.output_tokens;
        Ok((content, tokens))
    }

    // ── Ollama HTTP call ──────────────────────────────────────────────────

    async fn call_ollama(&self, req: &EscalationRequest) -> Result<(String, u32), HammerError> {
        let model = &self.config.providers.ollama.model;
        // Speculative cascade: same verification prompt for Ollama.
        let effective_prompt: String = if let Some(draft) = &req.draft_response {
            format!(
                "{}{}{}{}",
                DRAFT_VERIFICATION_PREFIX, draft, DRAFT_VERIFICATION_SEPARATOR, req.prompt
            )
        } else {
            req.prompt.clone()
        };
        let body = OllamaRequest {
            model,
            prompt: &effective_prompt,
            stream: false,
        };

        let url = format!("{}/api/generate", self.config.providers.ollama.base_url);
        let resp = self
            .http
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(HammerError::Http)?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(HammerError::ProviderError {
                provider: model.clone(),
                reason: format!("HTTP {status}: {text}"),
            });
        }

        let parsed: OllamaResponse = resp.json().await.map_err(HammerError::Http)?;

        let tokens = parsed.eval_count.unwrap_or(0) + parsed.prompt_eval_count.unwrap_or(0);
        Ok((parsed.response, tokens))
    }

    // ── Helpers ───────────────────────────────────────────────────────────

    fn make_response(
        &self,
        req: &EscalationRequest,
        result: (String, u32),
        provider: Provider,
        start: Instant,
    ) -> EscalationResponse {
        let teacher_cost_usd = self.cost_usd(provider, result.1);
        EscalationResponse {
            request_id: req.id.clone(),
            content: result.0,
            provider_type: provider.provider_type(),
            provider,
            cache_hit: false,
            tokens_used: result.1,
            elapsed_ms: start.elapsed().as_millis() as u64,
            teacher_cost_usd,
            used_draft: req.draft_response.is_some(),
        }
    }

    /// Compute USD cost for `tokens` tokens from `provider`.
    fn cost_usd(&self, provider: Provider, tokens: u32) -> f64 {
        let per_1k = match provider {
            Provider::AnthropicPrimary => self.config.cost.anthropic_primary_per_1k,
            Provider::AnthropicFallback => self.config.cost.anthropic_fallback_per_1k,
            Provider::Ollama | Provider::Cache => 0.0,
        };
        (tokens as f64 / 1_000.0) * per_1k
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::HammerConfig;
    use crate::types::EscalationRequest;

    struct OkProvider {
        content: String,
        tokens: u32,
        provider: Provider,
    }

    impl ProviderFn for OkProvider {
        fn call(&self, _req: &EscalationRequest) -> Result<(String, u32, Provider), HammerError> {
            Ok((self.content.clone(), self.tokens, self.provider))
        }
    }

    struct FailProvider;

    impl ProviderFn for FailProvider {
        fn call(&self, _req: &EscalationRequest) -> Result<(String, u32, Provider), HammerError> {
            Err(HammerError::ProviderError {
                provider: "mock_fail".into(),
                reason: "simulated failure".into(),
            })
        }
    }

    fn make_request() -> EscalationRequest {
        EscalationRequest::new("Hello world", 100)
    }

    #[tokio::test]
    async fn provider_anthropic_primary_succeeds() {
        let providers: Vec<Arc<dyn ProviderFn>> = vec![Arc::new(OkProvider {
            content: "ok".into(),
            tokens: 5,
            provider: Provider::AnthropicPrimary,
        })];
        let router = RequestRouter::with_mock_providers(HammerConfig::default(), providers);
        let resp = router.route(&make_request()).await.unwrap();
        assert_eq!(resp.content, "ok");
        assert_eq!(resp.provider, Provider::AnthropicPrimary);
    }

    #[tokio::test]
    async fn provider_fallover_to_fallback_model() {
        let providers: Vec<Arc<dyn ProviderFn>> = vec![
            Arc::new(FailProvider),
            Arc::new(OkProvider {
                content: "fallback".into(),
                tokens: 5,
                provider: Provider::AnthropicFallback,
            }),
        ];
        let router = RequestRouter::with_mock_providers(HammerConfig::default(), providers);
        let resp = router.route(&make_request()).await.unwrap();
        assert_eq!(resp.provider, Provider::AnthropicFallback);
    }

    #[tokio::test]
    async fn provider_fallover_to_ollama() {
        let providers: Vec<Arc<dyn ProviderFn>> = vec![
            Arc::new(FailProvider),
            Arc::new(FailProvider),
            Arc::new(OkProvider {
                content: "ollama response".into(),
                tokens: 10,
                provider: Provider::Ollama,
            }),
        ];
        let router = RequestRouter::with_mock_providers(HammerConfig::default(), providers);
        let resp = router.route(&make_request()).await.unwrap();
        assert_eq!(resp.provider, Provider::Ollama);
        assert_eq!(resp.content, "ollama response");
    }

    #[tokio::test]
    async fn provider_all_fail_returns_error() {
        let providers: Vec<Arc<dyn ProviderFn>> = vec![
            Arc::new(FailProvider),
            Arc::new(FailProvider),
            Arc::new(FailProvider),
        ];
        let router = RequestRouter::with_mock_providers(HammerConfig::default(), providers);
        let err = router.route(&make_request()).await.unwrap_err();
        assert!(matches!(err, HammerError::AllProvidersFailed(_)));
    }
}
