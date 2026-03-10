# grist-sieve

Triage classifier for GristMill. Routes every incoming `GristEvent` to the cheapest capable handler in **<5 ms p99** (PRD requirement S-02).

## Purpose

The Sieve is the first thing every event hits. It decides:

- **`LocalML`** — a local ONNX model can handle this (fast, free)
- **`Rules`** — a deterministic rule matches (fastest, free)
- **`Hybrid`** — local model + LLM prompt assembly (moderate cost)
- **`LlmNeeded`** — full LLM escalation required (slowest, highest cost)

The Sieve logs every decision to `~/.gristmill/feedback/` so Python can retrain the classifier weekly.

## Key Types

```rust
pub struct Sieve { /* opaque */ }

pub enum RouteDecision {
    LocalML {
        model_id: String,
        confidence: f32,
    },
    Rules {
        rule_id: String,
    },
    Hybrid {
        local_model: String,
        llm_prompt_template: String,
        estimated_tokens: u32,
    },
    LlmNeeded {
        reason: String,
        estimated_tokens: u32,
        estimated_cost_usd: f64,
    },
}

pub struct SieveConfig {
    pub model_path: Option<PathBuf>,       // ONNX classifier path
    pub confidence_threshold: f32,         // Default 0.85
    pub feedback_dir: Option<PathBuf>,     // JSONL feedback log dir
    pub exact_cache_size: usize,           // LRU capacity (default 10 000)
    pub semantic_cache_size: usize,        // Vector cache (default 5 000)
    pub semantic_similarity_threshold: f32, // Default 0.92
}
```

## Core Method

```rust
impl Sieve {
    pub async fn triage(&self, event: &GristEvent) -> Result<RouteDecision, SieveError>;
}
```

### Triage Pipeline (must stay <5 ms p99)

```
1. TTL check            — drop expired events immediately
2. Feature extraction   — 392-dim vector (MiniLM embedding + 8 scalar features)
3. Exact cache lookup   — SHA-256 hit → return cached decision
4. Semantic cache       — cosine similarity ≥ 0.92 hit → return cached decision
5. ONNX classification  — 4-class MLP (LocalML / Rules / Hybrid / LlmNeeded)
6. Cost oracle          — apply threshold + budget state
7. Cache store          — sync write to both caches
8. Async feedback write — non-blocking mpsc try_send (never blocks triage)
```

## Feature Vector (392 dimensions)

The feature vector is computed by `FeatureExtractor` and must exactly match the Python training code in `gristmill-ml/training/sieve_trainer.py`:

| Dims | Content |
|------|---------|
| 0–383 | L2-normalised MiniLM-L6-v2 embedding |
| 384 | Log-scaled token count: `ln(tc+1) / ln(2049)` clamped [0, 1] |
| 385 | Source channel ordinal / 9.0 |
| 386 | Priority ordinal / 3.0 |
| 387 | Entity density (0.0 when NER unavailable) |
| 388 | Question probability (0 / 0.6 / 1.0) |
| 389 | Code token fraction |
| 390 | Type-token ratio |
| 391 | Ambiguity score: `1 − max_freq / token_count` |

## Feedback Log Schema

Written to `~/.gristmill/feedback/feedback-{date}.jsonl` after every triage:

```json
{
  "event_id": "01HXYZ...",
  "text": "...",
  "channel": "http",
  "priority": "normal",
  "route": "LOCAL_ML",
  "confidence": 0.92,
  "timestamp_ms": 1234567890
}
```

Python reads this directory weekly to retrain the classifier.

## Cargo Features

| Feature | Description |
|---------|-------------|
| `onnx` | Enable ONNX Runtime inference (requires `libonnxruntime`). Without it, a heuristic classifier is used. |

## Configuration

```yaml
# ~/.gristmill/config.yaml
sieve:
  model: ~/.gristmill/models/sieve-v1.onnx
  confidence_threshold: 0.85
  feedback_dir: ~/.gristmill/feedback/
  exact_cache_size: 10000
  semantic_similarity_threshold: 0.92
```

## Testing

```bash
# Unit tests
cargo test -p grist-sieve

# Latency regression (p99 < 5 ms) — ignored by default, run explicitly
cargo test -p grist-sieve -- --include-ignored latency
```

## Dependencies

```toml
grist-event   = { path = "../grist-event" }
grist-grinders = { path = "../grist-grinders" }   # for MiniLM embedder
tokio         # async triage future
ort           # ONNX Runtime (feature = "onnx")
ndarray       # feature tensors
lru           # exact cache
dashmap       # concurrent semantic cache
metrics       # routing decision counters
tracing       # structured logging
```
