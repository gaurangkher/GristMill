# grist-hammer

LLM escalation gateway for GristMill. Routes requests to the cheapest available provider, enforces token budgets, and caches responses semantically to minimise API costs.

## Purpose

`grist-hammer` is called only when the Sieve decides an event genuinely needs an LLM (`RouteDecision::LlmNeeded` or `Hybrid`). It:

- Checks the token budget before spending any tokens
- Tries the exact + semantic cache first (no tokens spent on a hit)
- Routes to the best available provider (Anthropic → Ollama fallback)
- Batches concurrent requests to avoid thundering-herd to the API
- Records actual usage and publishes budget events on the bus

**The budget manager cannot be bypassed.** This is a hard invariant enforced at the gateway level.

## Key Types

```rust
pub struct Hammer { /* opaque */ }

pub struct EscalationRequest {
    pub id: Ulid,
    pub prompt: String,
    pub max_tokens: u32,
    pub model: Option<String>,        // Override default model
    pub embedding: Option<Vec<f32>>,  // Pre-computed for fuzzy cache lookup
}

pub struct EscalationResponse {
    pub request_id: Ulid,
    pub content: String,
    pub provider: Provider,
    pub tokens_used: u32,
    pub cache_hit: bool,
    pub elapsed_ms: u64,
}

pub enum Provider {
    Anthropic,           // claude-sonnet-4-6 (primary)
    AnthropicFallback,   // claude-haiku (cheaper fallback)
    Ollama,              // llama3.1:8b (local fallback)
}

pub struct HammerConfig {
    pub providers: ProviderConfig,
    pub budget: BudgetConfig,
    pub cache: CacheConfig,
    pub batch: BatchConfig,
}

pub struct BudgetConfig {
    pub daily_tokens: u64,
    pub monthly_tokens: u64,
}

pub struct CacheConfig {
    pub enabled: bool,
    pub similarity_threshold: f32,  // Default: 0.92
    pub max_entries: usize,
}
```

## Public API

```rust
// Create gateway
let hammer = Hammer::new(config, bus).await?;

// Escalate to LLM
let response: EscalationResponse = hammer.escalate(EscalationRequest {
    id: Ulid::new(),
    prompt: "Summarise: disk usage on prod-db-01 reached 94%".into(),
    max_tokens: 256,
    model: None,
    embedding: None,
}).await?;

println!("Cache hit: {}", response.cache_hit);
println!("Tokens used: {}", response.tokens_used);
println!("Response: {}", response.content);
```

## Escalation Pipeline

```
escalate(request)
    │
    ├─ Budget pre-check: will this exceed daily/monthly limit?
    │   └─ BudgetError if yes
    │
    ├─ Exact cache: SHA-256(prompt) → hit?
    │   └─ Return cached response (0 tokens spent)
    │
    ├─ Fuzzy cache: cosine_similarity(embedding, cache) ≥ 0.92?
    │   └─ Return cached response (0 tokens spent)
    │
    ├─ Route to provider (via batcher)
    │   ├─ Anthropic claude-sonnet-4-6  (primary)
    │   ├─ On error/quota: claude-haiku (fallback)
    │   └─ On error: Ollama llama3.1:8b  (local fallback)
    │
    ├─ Record actual token usage in BudgetManager
    ├─ Store response in exact + fuzzy cache
    │
    └─ Publish: hammer.budget { daily_used, daily_limit, pct_used }
```

## Semantic Cache

The fuzzy cache uses cosine similarity against stored response embeddings. A threshold of 0.92 means "95% similar prompts" reuse the cached response — saving API costs on repetitive requests.

```
Cache layers:
  1. Exact (LRU, SHA-256 key)        — always checked first
  2. Semantic (LRU, cosine ≥ 0.92)   — checked when exact misses
```

## Bus Events Published

| Topic | Payload |
|-------|---------|
| `hammer.budget` | `{ daily_used, daily_limit, window_start_ms, pct_used }` |

The TypeScript `metrics.ts` route subscribes to `hammer.budget` to power the `/api/metrics/budget` endpoint.

## Configuration

```yaml
# ~/.gristmill/config.yaml
hammer:
  providers:
    anthropic:
      api_key: ${ANTHROPIC_API_KEY}
      default_model: claude-sonnet-4-6
      fallback_model: claude-haiku-4-5-20251001
    ollama:
      base_url: http://localhost:11434
      model: llama3.1:8b
  budget:
    daily_tokens: 500000
    monthly_tokens: 10000000
  cache:
    enabled: true
    similarity_threshold: 0.92
    max_entries: 50000
  batch:
    enabled: true
    window_ms: 5000
    max_batch_size: 10
```

## Dependencies

```toml
grist-event  = { path = "../grist-event" }
grist-bus    = { path = "../grist-bus" }
tokio        # async HTTP requests + batching
reqwest      # HTTP client for Anthropic/Ollama APIs
ndarray      # embedding cosine similarity
lru          # exact + semantic cache
metrics      # token usage counters
tracing      # request/response logging
```
