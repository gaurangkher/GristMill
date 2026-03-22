//! grist-hammer — LLM escalation gateway.
//!
//! Provides [`Hammer`], a thread-safe, synchronously-constructible gateway that:
//! - Checks a semantic LRU cache before calling any provider.
//! - Enforces daily and monthly token budgets.
//! - Routes requests through Anthropic (primary → fallback) then Ollama with
//!   automatic fallover.
//! - Optionally batches requests within a configurable time window.
//!
//! # Construction
//! [`Hammer::new`] is **synchronous** and requires an active Tokio runtime
//! (it calls `tokio::spawn` internally via the batcher).

use std::sync::Arc;

use sha2::{Digest, Sha256};
use tracing::{debug, info};

pub mod batcher;
pub mod budget;
pub mod cache;
pub mod config;
pub mod error;
pub mod router;
pub mod types;

use batcher::RequestBatcher;
use budget::BudgetManager;
use cache::SemanticCache;
use router::RequestRouter;

pub use config::HammerConfig;
pub use error::HammerError;
pub use types::{BudgetInfo, EscalationRequest, EscalationResponse, Provider, ProviderType};

// ─────────────────────────────────────────────────────────────────────────────
// Hammer
// ─────────────────────────────────────────────────────────────────────────────

/// Thread-safe LLM escalation gateway.
///
/// **Requires an active Tokio runtime** (see [`Hammer::new`]).
pub struct Hammer {
    budget: BudgetManager,
    cache: SemanticCache,
    batcher: RequestBatcher,
    // Keep router around so batcher can reference it.
    _router: Arc<RequestRouter>,
}

impl Hammer {
    /// Create a new [`Hammer`].
    ///
    /// This is a **synchronous** constructor that internally spawns a Tokio
    /// task for the request batcher.  It **must** be called from within an
    /// active Tokio runtime context (e.g. inside an `#[tokio::main]` or
    /// `#[tokio::test]` annotated function).
    pub fn new(config: HammerConfig) -> Result<Self, HammerError> {
        let budget = BudgetManager::new(config.budget.clone());
        let sem_cache = SemanticCache::new(config.cache.clone());
        let router = Arc::new(RequestRouter::new(config.clone())?);
        let batcher = RequestBatcher::new(config.batch.clone(), Arc::clone(&router));

        info!("grist-hammer initialised");
        Ok(Self {
            budget,
            cache: sem_cache,
            batcher,
            _router: router,
        })
    }

    // ── Public API ────────────────────────────────────────────────────────

    /// Escalate a prompt to an LLM provider.
    ///
    /// Steps:
    /// 1. Pre-flight budget check (no tokens consumed yet).
    /// 2. Semantic cache lookup (exact SHA-256 + optional fuzzy cosine).
    /// 3. Route via batcher → provider.
    /// 4. Record actual token usage.
    /// 5. Store response in cache.
    pub async fn escalate(
        &self,
        req: EscalationRequest,
    ) -> Result<EscalationResponse, HammerError> {
        // 1. Budget pre-check.
        let estimated = req.estimated_tokens();
        self.budget.check(estimated)?;

        // 2. Cache lookup.
        let hash = sha256_hex(&req.prompt);
        if let Some(cached) = self.cache.get_exact(&hash) {
            debug!(request_id = %req.id, "cache exact hit");
            return Ok(EscalationResponse {
                cache_hit: true,
                provider_type: types::ProviderType::Cache,
                request_id: req.id,
                teacher_cost_usd: 0.0,
                used_draft: false,
                ..cached
            });
        }
        if let Some(emb) = &req.embedding {
            if let Some(cached) = self.cache.get_fuzzy(emb) {
                debug!(request_id = %req.id, "cache fuzzy hit");
                return Ok(EscalationResponse {
                    cache_hit: true,
                    provider_type: types::ProviderType::Cache,
                    request_id: req.id,
                    teacher_cost_usd: 0.0,
                    used_draft: false,
                    ..cached
                });
            }
        }

        // 3. Dispatch via batcher.
        let embedding = req.embedding.clone();
        let mut resp = self.batcher.submit(req).await?;

        // 4. Record usage.
        self.budget.record_usage(resp.tokens_used);

        // 5. Cache the response.
        self.cache.put(hash, resp.clone(), embedding);

        resp.cache_hit = false;
        Ok(resp)
    }

    /// Return a snapshot of the current token budget.
    pub fn get_budget(&self) -> BudgetInfo {
        self.budget.info()
    }

    /// Clear the entire semantic cache.
    pub fn clear_cache(&self) {
        self.cache.clear();
    }

    /// Number of entries currently in the cache (test helper).
    pub fn cache_size(&self) -> usize {
        self.cache.len()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

fn sha256_hex(s: &str) -> String {
    let mut h = Sha256::new();
    h.update(s.as_bytes());
    hex::encode(h.finalize())
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{BatchConfig, BudgetConfig, CacheConfig};
    use crate::router::{ProviderFn, RequestRouter};
    use crate::types::Provider;
    use std::sync::Arc;

    // ── Mock provider helpers ─────────────────────────────────────────────

    struct MockProvider {
        content: String,
        tokens: u32,
    }

    impl ProviderFn for MockProvider {
        fn call(&self, _req: &EscalationRequest) -> Result<(String, u32, Provider), HammerError> {
            Ok((
                self.content.clone(),
                self.tokens,
                Provider::AnthropicPrimary,
            ))
        }
    }

    struct AlwaysFailProvider;

    impl ProviderFn for AlwaysFailProvider {
        fn call(&self, _req: &EscalationRequest) -> Result<(String, u32, Provider), HammerError> {
            Err(HammerError::AllProvidersFailed("test failure".into()))
        }
    }

    fn make_hammer_with_provider(
        provider: impl ProviderFn + 'static,
        budget: BudgetConfig,
        cache: CacheConfig,
    ) -> Hammer {
        let config = HammerConfig {
            budget: budget.clone(),
            cache: cache.clone(),
            batch: BatchConfig {
                enabled: true,
                window_ms: 50,
                max_batch_size: 10,
            },
            ..Default::default()
        };
        let providers: Vec<Arc<dyn ProviderFn>> = vec![Arc::new(provider)];
        let router = Arc::new(RequestRouter::with_mock_providers(
            config.clone(),
            providers,
        ));
        let batcher = RequestBatcher::new(config.batch.clone(), Arc::clone(&router));
        Hammer {
            budget: BudgetManager::new(budget),
            cache: SemanticCache::new(cache),
            batcher,
            _router: router,
        }
    }

    fn default_budget() -> BudgetConfig {
        BudgetConfig {
            daily_tokens: 100_000,
            monthly_tokens: 1_000_000,
        }
    }

    fn default_cache() -> CacheConfig {
        CacheConfig::default()
    }

    // ── Tests ─────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn escalate_happy_path() {
        let hammer = make_hammer_with_provider(
            MockProvider {
                content: "42".into(),
                tokens: 20,
            },
            default_budget(),
            default_cache(),
        );
        let req = EscalationRequest::new("what is the answer?", 100);
        let resp = hammer.escalate(req).await.unwrap();
        assert_eq!(resp.content, "42");
        assert!(!resp.cache_hit);
        assert_eq!(resp.tokens_used, 20);
    }

    #[tokio::test]
    async fn escalate_cache_hit() {
        let hammer = make_hammer_with_provider(
            MockProvider {
                content: "cached".into(),
                tokens: 10,
            },
            default_budget(),
            default_cache(),
        );
        let prompt = "repeat me".to_string();

        // First call — populates cache.
        let r1 = hammer
            .escalate(EscalationRequest::new(&prompt, 50))
            .await
            .unwrap();
        assert!(!r1.cache_hit);

        // Second call — should hit cache.
        let r2 = hammer
            .escalate(EscalationRequest::new(&prompt, 50))
            .await
            .unwrap();
        assert!(r2.cache_hit, "second call should be a cache hit");
        assert_eq!(r2.content, "cached");
    }

    #[tokio::test]
    async fn escalate_budget_enforced() {
        let tight_budget = BudgetConfig {
            daily_tokens: 5,
            monthly_tokens: 1_000_000,
        };
        let hammer = make_hammer_with_provider(
            MockProvider {
                content: "x".into(),
                tokens: 10,
            },
            tight_budget,
            default_cache(),
        );
        // Even a short prompt will be estimated to exceed the 5-token daily limit.
        let req = EscalationRequest::new("explain the entire universe in detail", 200);
        let err = hammer.escalate(req).await.unwrap_err();
        assert!(matches!(err, HammerError::BudgetExceeded { .. }));
    }

    #[tokio::test]
    async fn get_budget_info() {
        let hammer = make_hammer_with_provider(
            MockProvider {
                content: "hi".into(),
                tokens: 30,
            },
            default_budget(),
            default_cache(),
        );
        let req = EscalationRequest::new("hello", 100);
        hammer.escalate(req).await.unwrap();

        let info = hammer.get_budget();
        assert_eq!(info.daily_used, 30);
        assert_eq!(info.daily_limit, 100_000);
        assert_eq!(info.monthly_used, 30);
        assert!(info.daily_remaining < info.daily_limit);
    }

    #[tokio::test]
    async fn clear_cache_empties() {
        let hammer = make_hammer_with_provider(
            MockProvider {
                content: "hi".into(),
                tokens: 5,
            },
            default_budget(),
            default_cache(),
        );
        let req = EscalationRequest::new("hello cache", 50);
        hammer.escalate(req).await.unwrap();
        assert!(hammer.cache_size() > 0);

        hammer.clear_cache();
        assert_eq!(hammer.cache_size(), 0);
    }

    #[tokio::test]
    async fn escalate_all_providers_fail() {
        let hammer =
            make_hammer_with_provider(AlwaysFailProvider, default_budget(), default_cache());
        let req = EscalationRequest::new("fail me", 50);
        let err = hammer.escalate(req).await.unwrap_err();
        assert!(
            matches!(err, HammerError::AllProvidersFailed(_)),
            "unexpected error: {:?}",
            err
        );
    }

    #[tokio::test]
    async fn sha256_is_deterministic() {
        let h1 = sha256_hex("hello");
        let h2 = sha256_hex("hello");
        assert_eq!(h1, h2);
        assert_ne!(sha256_hex("hello"), sha256_hex("world"));
    }
}
