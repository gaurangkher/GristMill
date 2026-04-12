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

use std::collections::HashMap;
use std::sync::Arc;

use sha2::{Digest, Sha256};
use tokio::sync::{oneshot, Mutex};
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
    /// Singleflight map: prompt hash → waiters for the in-flight request.
    ///
    /// The first caller for a given hash is the "leader" and dispatches to the
    /// batcher.  Any subsequent caller with the same hash (arriving before the
    /// leader has populated the cache) subscribes here instead of issuing a
    /// second provider call.  This prevents duplicate requests to Anthropic/Ollama
    /// when two identical prompts arrive concurrently.
    inflight: Arc<Mutex<HashMap<String, Vec<oneshot::Sender<Result<EscalationResponse, HammerError>>>>>>,
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
            inflight: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    // ── Public API ────────────────────────────────────────────────────────

    /// Escalate a prompt to an LLM provider.
    ///
    /// Steps:
    /// 1. Pre-flight budget check (no tokens consumed yet).
    /// 2. Semantic cache lookup (exact SHA-256 + optional fuzzy cosine).
    /// 3. Singleflight deduplication: if an identical prompt is already in-flight,
    ///    subscribe to its result instead of issuing a second provider call.
    /// 4. Route via batcher → provider (leader only).
    /// 5. Record actual token usage and store response in cache (leader only).
    /// 6. Fan out result to all waiters.
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

        // 3. Singleflight: deduplicate concurrent in-flight requests with the
        //    same prompt hash so only one provider call is made.
        let (is_leader, waiter_rx) = {
            let mut inflight = self.inflight.lock().await;
            if let Some(waiters) = inflight.get_mut(&hash) {
                // A leader is already dispatching this prompt — register as waiter.
                let (tx, rx) = oneshot::channel();
                waiters.push(tx);
                debug!(request_id = %req.id, "singleflight: waiting on in-flight request");
                (false, Some(rx))
            } else {
                // No in-flight request for this hash — become the leader.
                inflight.insert(hash.clone(), Vec::new());
                (true, None)
            }
        };

        if !is_leader {
            return waiter_rx
                .unwrap()
                .await
                .map_err(|_| HammerError::Config("singleflight: leader dropped channel".into()))?;
        }

        // 4. Leader: dispatch via batcher.
        let embedding = req.embedding.clone();
        let dispatch_result = self.batcher.submit(req).await;

        // 5. Drain waiters from the inflight map regardless of success/error.
        let waiters = {
            let mut inflight = self.inflight.lock().await;
            inflight.remove(&hash).unwrap_or_default()
        };

        match &dispatch_result {
            Ok(resp) => {
                // Record usage and populate the cache so the next distinct request
                // gets a cache hit instead of going to the provider again.
                self.budget.record_usage(resp.tokens_used);
                self.cache.put(hash, resp.clone(), embedding);

                // Fan out to waiters.
                for waiter in waiters {
                    let mut r = resp.clone();
                    r.cache_hit = false;
                    let _ = waiter.send(Ok(r));
                }
            }
            Err(e) => {
                // Propagate error string to waiters; normalise to AllProvidersFailed
                // since HammerError is not Clone.
                let msg = e.to_string();
                for waiter in waiters {
                    let _ = waiter.send(Err(HammerError::AllProvidersFailed(msg.clone())));
                }
            }
        }

        let mut resp = dispatch_result?;
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
            inflight: Arc::new(Mutex::new(HashMap::new())),
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

    #[tokio::test]
    async fn singleflight_deduplicates_concurrent_identical_prompts() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        // Count how many times the provider is actually called.
        let call_count = Arc::new(AtomicUsize::new(0));

        struct CountingProvider {
            count: Arc<AtomicUsize>,
        }
        impl ProviderFn for CountingProvider {
            fn call(&self, req: &EscalationRequest) -> Result<(String, u32, Provider), HammerError> {
                self.count.fetch_add(1, Ordering::SeqCst);
                // Small artificial delay so the second request arrives while the
                // first is still in-flight (simulated by the batcher window).
                Ok((format!("response:{}", req.prompt), 10, Provider::AnthropicPrimary))
            }
        }

        let config = HammerConfig {
            batch: BatchConfig { enabled: true, window_ms: 200, max_batch_size: 10 },
            ..Default::default()
        };
        let providers: Vec<Arc<dyn ProviderFn>> =
            vec![Arc::new(CountingProvider { count: Arc::clone(&call_count) })];
        let router = Arc::new(RequestRouter::with_mock_providers(config.clone(), providers));
        let batcher = RequestBatcher::new(config.batch.clone(), Arc::clone(&router));
        let hammer = Arc::new(Hammer {
            budget: BudgetManager::new(config.budget.clone()),
            cache: SemanticCache::new(config.cache.clone()),
            batcher,
            _router: router,
            inflight: Arc::new(Mutex::new(HashMap::new())),
        });

        // Fire two identical prompts concurrently.
        let h1 = Arc::clone(&hammer);
        let h2 = Arc::clone(&hammer);
        let (r1, r2) = tokio::join!(
            tokio::spawn(async move { h1.escalate(EscalationRequest::new("dup", 50)).await }),
            tokio::spawn(async move { h2.escalate(EscalationRequest::new("dup", 50)).await }),
        );

        assert!(r1.unwrap().is_ok());
        assert!(r2.unwrap().is_ok());
        // Provider must have been called exactly once despite two concurrent requests.
        assert_eq!(call_count.load(Ordering::SeqCst), 1,
            "singleflight should deduplicate: provider called {} times instead of 1",
            call_count.load(Ordering::SeqCst));
    }
}
