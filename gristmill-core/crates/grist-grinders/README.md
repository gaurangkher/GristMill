# grist-grinders

Local ML inference pool for GristMill. Runs ONNX and GGUF models with dynamic batching, warm/cold model management, and parallel CPU dispatch via Rayon.

## Purpose

`grist-grinders` is the inference engine that keeps all ML work local. It is used by:

- **grist-sieve** — feature extraction (MiniLM-L6-v2) and intent classification
- **grist-ledger** — memory embeddings via `GrindersEmbedder`
- **grist-millwright** — local model steps in pipelines

## Key Types

```rust
pub struct Grinders { /* opaque */ }

pub struct GrindersConfig {
    pub worker_threads: usize,    // Rayon threads (default: CPU cores - 1)
    pub queue_depth: usize,       // Per-model dispatch queue (default: 1024)
    pub batch_window_ms: u64,     // Dynamic batching window (default: 5 ms)
    pub max_batch_size: usize,    // Max batch size (default: 32)
    pub models: Vec<ModelConfig>, // Models to register at startup
}

pub struct ModelConfig {
    pub model_id: String,         // Unique identifier for API calls
    pub path: PathBuf,            // Path to .onnx or .gguf file
    pub runtime: ModelRuntime,    // Onnx | Gguf
    pub warm: bool,               // Load at startup (true) or on-demand (false)
    pub timeout: Duration,        // Per-request inference timeout (default: 5s)
    pub max_tokens: usize,        // GGUF output token limit
    pub description: String,
}

pub enum ModelRuntime { Onnx, Gguf }

pub struct InferenceRequest {
    pub model_id: String,
    pub features: ndarray::Array2<f32>,
    pub timeout_ms: u64,
}

pub struct InferenceOutput {
    pub logits: Vec<f32>,
    pub latency_ms: u64,
}
```

## Public API

```rust
// Build from config
let grinders = Grinders::new(config).await?;

// Synchronous batch inference
let output: InferenceOutput = grinders.infer(request).await?;

// Build the MiniLM embedder (used by grist-core to wire the Ledger)
let session: EmbedderSession = build_minilm_embedder(&config)?;
```

## Inference Pipeline

```
1. grinders.infer(request)
2. → pool.submit(request)        — enqueue into per-model mpsc channel
3. → batcher task                — accumulate for batch_window_ms or until max_batch_size
4. → rayon::spawn                — dispatch batch to CPU worker pool
5. → registry.get_or_load()     — resolve warm session or load cold model
6. → session.run()               — ONNX/GGUF inference
7. ← InferenceOutput
```

## Starter Model Pack

`starter_pack(model_dir)` returns configs for the 5 recommended models:

| Model | Runtime | Warm | Size | Purpose |
|-------|---------|------|------|---------|
| `intent-classifier-v1` | ONNX | Yes | ~25 MB | 4-class intent routing for Sieve |
| `ner-multilingual-v1` | ONNX | Yes | ~40 MB | Named entity recognition |
| `minilm-l6-v2` | ONNX | Yes | ~25 MB | Sentence embeddings (Ledger + Sieve) |
| `phi3-mini-4k-Q4` | GGUF | **No** | ~2.3 GB | Local summarization (Ledger compaction) |
| `anomaly-detector-v1` | ONNX | Yes | ~5 MB | Isolation forest for metric anomaly detection |

```rust
let models = grist_grinders::config::starter_pack(&model_dir);
```

## Cargo Features

| Feature | Description |
|---------|-------------|
| `onnx` | ONNX Runtime via `ort` crate |
| `gguf` | llama.cpp GGUF via `llama-cpp-2` crate |

## Configuration Example

```yaml
# ~/.gristmill/config.yaml
grinders:
  worker_threads: 7          # 0 = use Rayon global pool (CPU-1 threads)
  queue_depth: 1024
  batch_window_ms: 5
  max_batch_size: 32
  models:
    - model_id: minilm-l6-v2
      path: ~/.gristmill/models/minilm-l6-v2.onnx
      runtime: onnx
      warm: true
    - model_id: phi3-mini-4k-Q4
      path: ~/.gristmill/models/phi3-mini-4k-Q4.gguf
      runtime: gguf
      warm: false
      max_tokens: 128
```

## PRD Latency Targets

- **Warm models**: respond in <5 ms (PRD G-04)
- **Cold model load**: complete in <2 s (PRD G-04)
- **Inference timeout**: per-model configurable (default 5 s, PRD G-07)
