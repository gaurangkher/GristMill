# GristMill v2 — Architecture Specification
## Rust Core · Python ML Shell · TypeScript Integration Shell

> *"Everything gets ground locally first."*

---

## 1. Language Boundary Design

GristMill is a **tri-language system** with strict separation of concerns. Each language owns a specific domain and communicates through well-defined interfaces.

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│   ┌───────────────────────────────────────────────────────────┐     │
│   │              RUST CORE  (gristmill-core)                  │     │
│   │                                                           │     │
│   │  Sieve (Triage) · Millwright (DAG) · Ledger (Memory)     │     │
│   │  Grinders (Inference) · Event Bus · Config · CLI          │     │
│   │                                                           │     │
│   │  Compiles to: static binary + C FFI shared library        │     │
│   │  (.so / .dylib / .dll)                                    │     │
│   └──────────┬──────────────────────────┬─────────────────────┘     │
│              │ PyO3 FFI                  │ napi-rs FFI               │
│              ▼                           ▼                           │
│   ┌─────────────────────┐    ┌─────────────────────────────┐       │
│   │  PYTHON SHELL       │    │  TYPESCRIPT SHELL            │       │
│   │  (gristmill-ml)     │    │  (gristmill-integrations)    │       │
│   │                     │    │                               │       │
│   │  Custom model       │    │  Hopper adapters (HTTP/WS)   │       │
│   │  training           │    │  Bell Tower notifications     │       │
│   │  Model fine-tuning  │    │  Channel adapters (Slack,     │       │
│   │  Dataset pipelines  │    │   Telegram, Discord, etc.)   │       │
│   │  Experiment         │    │  Template engine              │       │
│   │  tracking           │    │  Web dashboard                │       │
│   │  ONNX model export  │    │  REST/GraphQL API             │       │
│   └─────────────────────┘    │  Plugin system                │       │
│                               └─────────────────────────────┘       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Why This Split

| Concern | Language | Rationale |
|---------|----------|-----------|
| Event triage, classification, inference | **Rust** | Zero-copy tensor ops, no GC pauses, sub-5ms latency |
| DAG scheduling, parallel execution | **Rust** | Tokio + Rayon, real OS threads, deterministic memory |
| Memory tiers, vector search, LRU cache | **Rust** | Predictable allocation/deallocation, FAISS/usearch FFI |
| Config, CLI, binary distribution | **Rust** | Single static binary, cross-compilation |
| Custom model training & fine-tuning | **Python** | PyTorch, HuggingFace, scikit-learn ecosystem |
| Dataset preparation & experiment tracking | **Python** | pandas, MLflow, Weights & Biases |
| ONNX export pipeline | **Python** | torch.onnx, optimum, onnxruntime-tools |
| Channel adapters & webhooks | **TypeScript** | Largest SDK ecosystem for APIs and messaging |
| Notification dispatch | **TypeScript** | Nodemailer, bot SDKs, push notification libraries |
| Web dashboard & REST API | **TypeScript** | Ktor alternative with better frontend story |
| Plugin system & community extensions | **TypeScript** | Lower barrier for contributors, npm distribution |

---

## 2. Rust Core — `gristmill-core`

### 2.1 Crate Structure

```
gristmill-core/
├── Cargo.toml                    # Workspace root
├── crates/
│   ├── grist-event/              # GristEvent schema + serialization
│   │   ├── src/lib.rs
│   │   └── Cargo.toml
│   │
│   ├── grist-sieve/              # Triage classifier + cost oracle
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── classifier.rs     # ONNX inference wrapper
│   │   │   ├── features.rs       # Feature extraction pipeline
│   │   │   ├── cost_oracle.rs    # Token cost estimation
│   │   │   └── feedback.rs       # Retrospective scoring loop
│   │   └── Cargo.toml
│   │
│   ├── grist-grinders/           # Local ML inference pool
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── pool.rs           # Worker thread pool (Rayon)
│   │   │   ├── registry.rs       # Model registry + hot-loading
│   │   │   ├── onnx.rs           # ONNX Runtime wrapper
│   │   │   ├── gguf.rs           # llama.cpp wrapper
│   │   │   ├── tflite.rs         # TFLite wrapper (optional)
│   │   │   └── batch.rs          # Dynamic batching
│   │   └── Cargo.toml
│   │
│   ├── grist-millwright/         # DAG orchestrator
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── dag.rs            # DAG construction + validation
│   │   │   ├── scheduler.rs      # Parallel task scheduler
│   │   │   ├── checkpoint.rs     # Checkpoint / resume
│   │   │   ├── gates.rs          # Approval gates
│   │   │   └── retry.rs          # Retry + timeout policies
│   │   └── Cargo.toml
│   │
│   ├── grist-ledger/             # Three-tier memory
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── hot.rs            # LRU in-process cache
│   │   │   ├── warm.rs           # SQLite FTS5 + vector index
│   │   │   ├── cold.rs           # Compressed JSONL archive
│   │   │   ├── compactor.rs      # Auto-compaction daemon
│   │   │   ├── retrieval.rs      # Dual-path search + RRF fusion
│   │   │   └── embedder.rs       # Local embedding generation
│   │   └── Cargo.toml
│   │
│   ├── grist-hammer/             # LLM escalation gateway
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── budget.rs         # Token budget manager
│   │   │   ├── cache.rs          # Semantic cache
│   │   │   ├── batcher.rs        # Request batch aggregator
│   │   │   ├── router.rs         # Model selection + failover
│   │   │   └── providers/        # Anthropic, OpenAI, Ollama, vLLM
│   │   └── Cargo.toml
│   │
│   ├── grist-bus/                # Internal event bus
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── channels.rs       # Typed pub/sub channels
│   │   │   └── broker.rs         # Message routing
│   │   └── Cargo.toml
│   │
│   ├── grist-config/             # Configuration management
│   │   ├── src/lib.rs            # YAML/TOML + env overlay
│   │   └── Cargo.toml
│   │
│   └── grist-ffi/                # Foreign function interface layer
│       ├── src/
│       │   ├── lib.rs
│       │   ├── pyo3_bridge.rs    # Python bindings via PyO3
│       │   └── napi_bridge.rs    # Node.js bindings via napi-rs
│       └── Cargo.toml
│
├── gristmill-cli/                # CLI binary
│   ├── src/main.rs
│   └── Cargo.toml
│
└── gristmill-daemon/             # Long-running daemon binary
    ├── src/main.rs
    └── Cargo.toml
```

### 2.2 Key Rust Dependencies

```toml
[workspace.dependencies]
# Async runtime
tokio = { version = "1", features = ["full"] }
# CPU-parallel work-stealing
rayon = "1.10"

# ML inference
ort = "2"                         # ONNX Runtime (official Rust bindings)
llama-cpp-2 = "0.1"              # llama.cpp bindings
ndarray = "0.16"                  # Tensor operations, zero-copy

# Vector search
usearch = "2"                     # FAISS alternative, pure Rust-friendly
# faiss = "0.12"                  # Alternative: FAISS C bindings

# Storage
rusqlite = { version = "0.32", features = ["bundled", "fts5"] }
sled = "0.34"                     # Embedded KV store for hot cache
zstd = "0.13"                     # Compression for cold tier

# Serialization
serde = { version = "1", features = ["derive"] }
serde_json = "1"
rmp-serde = "1"                   # MessagePack for internal IPC

# FFI bridges
pyo3 = { version = "0.22", features = ["auto-initialize"] }
napi = "3"
napi-derive = "3"

# Config
config = "0.14"

# CLI
clap = { version = "4", features = ["derive"] }

# Observability
tracing = "0.1"
tracing-subscriber = "0.3"
metrics = "0.23"
```

### 2.3 Core Rust Interfaces

#### GristEvent (the universal message type)

```rust
// crates/grist-event/src/lib.rs

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use ulid::Ulid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GristEvent {
    pub id: Ulid,
    pub source: ChannelType,
    pub timestamp_ms: u64,
    pub payload: serde_json::Value,
    pub metadata: EventMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChannelType {
    Http,
    WebSocket,
    Cli,
    Cron,
    Webhook { provider: String },
    MessageQueue { topic: String },
    FileSystem { path: String },
    Python { callback_id: String },
    TypeScript { adapter_id: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventMetadata {
    pub priority: Priority,
    pub correlation_id: Option<String>,
    pub reply_channel: Option<String>,
    pub ttl_ms: Option<u64>,
    pub tags: HashMap<String, String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}
```

#### The Sieve (triage engine)

```rust
// crates/grist-sieve/src/lib.rs

use grist_event::GristEvent;
use ort::Session;

#[derive(Debug, Clone)]
pub enum RouteDecision {
    /// Handle entirely with local ML models
    LocalML {
        model_id: String,
        confidence: f32,
    },
    /// Handle with deterministic rules
    Rules {
        rule_id: String,
    },
    /// Pre-process locally, then refine with LLM
    Hybrid {
        local_model: String,
        llm_prompt_template: String,
        estimated_tokens: u32,
    },
    /// Requires full LLM reasoning
    LlmNeeded {
        reason: String,
        estimated_tokens: u32,
        estimated_cost_usd: f64,
    },
}

pub struct Sieve {
    classifier: Session,           // ONNX intent classifier
    embedder: Session,             // MiniLM for feature extraction
    cost_oracle: CostOracle,
    confidence_threshold: f32,     // Default: 0.85
    feedback_log: FeedbackLog,     // For retrospective learning
}

impl Sieve {
    /// Classify an event and return a routing decision.
    /// This MUST complete in < 5ms for the architecture to hold.
    pub async fn triage(&self, event: &GristEvent) -> Result<RouteDecision> {
        // 1. Extract features (TF-IDF + embedding)
        let features = self.extract_features(event)?;

        // 2. Check semantic cache first
        if let Some(cached) = self.check_cache(&features).await? {
            return Ok(cached);
        }

        // 3. Run classifier
        let (intent, confidence) = self.classify(&features)?;

        // 4. Apply cost oracle
        let decision = self.cost_oracle.evaluate(intent, confidence, event)?;

        // 5. Log for retrospective feedback
        self.feedback_log.record(event.id, &decision);

        Ok(decision)
    }

    fn extract_features(&self, event: &GristEvent) -> Result<FeatureVector> {
        let text = event.payload_as_text();
        let embedding = self.embedder.run_inference(text)?;  // zero-copy ndarray
        let keywords = extract_keywords(text);
        let complexity = estimate_complexity(text);

        Ok(FeatureVector { embedding, keywords, complexity })
    }
}
```

#### The Millwright (DAG executor)

```rust
// crates/grist-millwright/src/dag.rs

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pipeline {
    pub id: String,
    pub steps: Vec<Step>,
    pub default_mode: ExecutionMode,
    pub max_concurrency: usize,      // Default: num_cpus - 1
    pub checkpoint_strategy: CheckpointStrategy,
    pub timeout_ms: u64,
    pub on_failure: FailurePolicy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Step {
    pub id: String,
    pub step_type: StepType,
    pub depends_on: Vec<String>,     // Empty = run immediately
    pub prefer_local: bool,          // Default: true
    pub fallback_to_llm: bool,
    pub requires_approval: bool,
    pub approval_channel: Option<String>,
    pub timeout_ms: Option<u64>,
    pub retry: Option<RetryPolicy>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StepType {
    LocalML { model_id: String },
    Rule { rule_id: String },
    Llm { prompt_template: String, max_tokens: u32 },
    External { action: String, config: serde_json::Value },
    Gate { condition: String },
    /// Call into Python shell for custom ML
    PythonCall { module: String, function: String },
    /// Call into TypeScript shell for integrations
    TypeScriptCall { adapter: String, method: String },
}
```

```rust
// crates/grist-millwright/src/scheduler.rs

use tokio::sync::{mpsc, Semaphore};
use std::sync::Arc;
use dashmap::DashMap;

pub struct DagScheduler {
    semaphore: Arc<Semaphore>,          // Concurrency limiter
    results: Arc<DashMap<String, StepResult>>,
    checkpoint_store: CheckpointStore,
}

impl DagScheduler {
    pub async fn execute(&self, pipeline: &Pipeline, input: GristEvent) -> Result<PipelineResult> {
        let ready_steps = self.find_ready_steps(pipeline, &[])?;

        // Fan out all ready steps in parallel
        let mut handles = Vec::new();
        for step in ready_steps {
            let permit = self.semaphore.clone().acquire_owned().await?;
            let results = self.results.clone();
            let step = step.clone();
            let input = input.clone();

            let handle = tokio::spawn(async move {
                let result = execute_step(&step, &input, &results).await;
                results.insert(step.id.clone(), result.clone());
                drop(permit);  // Release concurrency slot
                result
            });
            handles.push(handle);
        }

        // As steps complete, check for newly-unblocked steps
        // Continue until all steps are done or a failure policy triggers
        self.drain_to_completion(pipeline, handles, input).await
    }

    fn find_ready_steps<'a>(
        &self,
        pipeline: &'a Pipeline,
        completed: &[String],
    ) -> Result<Vec<&'a Step>> {
        Ok(pipeline.steps.iter().filter(|step| {
            !completed.contains(&step.id)
                && step.depends_on.iter().all(|dep| completed.contains(dep))
        }).collect())
    }
}
```

#### The Ledger (three-tier memory)

```rust
// crates/grist-ledger/src/lib.rs

pub struct Ledger {
    hot: HotTier,       // sled-backed LRU
    warm: WarmTier,     // SQLite FTS5 + usearch vectors
    cold: ColdTier,     // zstd-compressed JSONL
    compactor: CompactorHandle,
    embedder: Arc<EmbedderSession>,  // Shared MiniLM for indexing
}

impl Ledger {
    /// Store a memory, automatically tiered
    pub async fn remember(&self, memory: Memory) -> Result<MemoryId> {
        // Generate embedding for semantic indexing
        let embedding = self.embedder.embed(&memory.content)?;

        // Check for semantic duplicates in warm tier
        if let Some(existing) = self.warm.find_similar(&embedding, 0.95)? {
            // Merge instead of duplicate
            return self.warm.merge(existing.id, &memory);
        }

        // Store in hot tier (will cascade to warm on eviction)
        let id = self.hot.insert(memory, embedding)?;
        Ok(id)
    }

    /// Retrieve relevant memories using dual-path search
    pub async fn recall(&self, query: &str, limit: usize) -> Result<Vec<RankedMemory>> {
        let embedding = self.embedder.embed(query)?;

        // Run both searches in parallel
        let (keyword_results, semantic_results) = tokio::join!(
            self.warm.keyword_search(query, limit * 2),
            self.warm.vector_search(&embedding, limit * 2),
        );

        // Reciprocal Rank Fusion
        let fused = reciprocal_rank_fusion(
            keyword_results?,
            semantic_results?,
            limit,
        );

        // Promote frequently-accessed cold memories
        self.maybe_promote_cold(&fused).await?;

        Ok(fused)
    }
}

// Auto-compaction runs as a background Tokio task
pub struct Compactor {
    ledger: Arc<Ledger>,
    local_summarizer: Option<GgufSession>,  // Phi-3 Mini for compression
    interval: Duration,
}

impl Compactor {
    pub async fn run_loop(&self) {
        let mut interval = tokio::time::interval(self.interval);
        loop {
            interval.tick().await;
            if let Err(e) = self.compact_cycle().await {
                tracing::warn!("Compaction cycle failed: {e}");
            }
        }
    }

    async fn compact_cycle(&self) -> Result<()> {
        // 1. Deduplicate: cluster similar warm memories
        let clusters = self.ledger.warm.find_clusters(0.90)?;
        for cluster in clusters {
            if cluster.len() > 1 {
                let merged = self.merge_cluster(&cluster).await?;
                self.ledger.warm.replace_cluster(&cluster, merged)?;
            }
        }

        // 2. Summarize: compress verbose memories
        if let Some(ref summarizer) = self.local_summarizer {
            let verbose = self.ledger.warm.find_verbose(512)?; // > 512 tokens
            for memory in verbose {
                let summary = summarizer.summarize(&memory.content, 128)?;
                self.ledger.warm.replace_content(memory.id, summary)?;
            }
        }

        // 3. Demote: stale warm → cold
        let stale = self.ledger.warm.find_stale(Duration::from_secs(86400 * 90))?;
        for memory in stale {
            self.ledger.cold.archive(memory)?;
            self.ledger.warm.remove(memory.id)?;
        }

        Ok(())
    }
}
```

### 2.4 FFI Bridge Layer

```rust
// crates/grist-ffi/src/pyo3_bridge.rs

use pyo3::prelude::*;
use grist_event::GristEvent;
use grist_sieve::RouteDecision;

/// Exposed to Python as `gristmill_core`
#[pymodule]
fn gristmill_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGristMill>()?;
    m.add_class::<PyGristEvent>()?;
    m.add_class::<PyRouteDecision>()?;
    m.add_class::<PyMemory>()?;
    Ok(())
}

#[pyclass]
pub struct PyGristMill {
    inner: Arc<GristMillCore>,
}

#[pymethods]
impl PyGristMill {
    #[new]
    fn new(config_path: &str) -> PyResult<Self> { /* ... */ }

    /// Triage an event — returns routing decision
    fn triage<'py>(&self, py: Python<'py>, event: &PyGristEvent) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        let event = event.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let decision = inner.sieve.triage(&event).await
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            Ok(PyRouteDecision { inner: decision })
        })
    }

    /// Run a pipeline
    fn execute_pipeline<'py>(
        &self, py: Python<'py>, pipeline_id: &str, event: &PyGristEvent,
    ) -> PyResult<Bound<'py, PyAny>> { /* ... */ }

    /// Store a memory
    fn remember<'py>(&self, py: Python<'py>, content: &str, tags: Vec<String>) -> PyResult<Bound<'py, PyAny>> { /* ... */ }

    /// Recall memories
    fn recall<'py>(&self, py: Python<'py>, query: &str, limit: usize) -> PyResult<Bound<'py, PyAny>> { /* ... */ }

    /// Register a Python-implemented model for the Grinders pool
    fn register_python_model(&self, model_id: &str, callback: PyObject) -> PyResult<()> { /* ... */ }
}
```

```rust
// crates/grist-ffi/src/napi_bridge.rs

use napi::bindgen_prelude::*;
use napi_derive::napi;

#[napi]
pub struct GristMill {
    inner: Arc<GristMillCore>,
}

#[napi]
impl GristMill {
    #[napi(constructor)]
    pub fn new(config_path: String) -> Result<Self> { /* ... */ }

    /// Triage an event
    #[napi]
    pub async fn triage(&self, event: serde_json::Value) -> Result<serde_json::Value> {
        let event: GristEvent = serde_json::from_value(event)
            .map_err(|e| Error::from_reason(e.to_string()))?;
        let decision = self.inner.sieve.triage(&event).await
            .map_err(|e| Error::from_reason(e.to_string()))?;
        serde_json::to_value(&decision)
            .map_err(|e| Error::from_reason(e.to_string()))
    }

    /// Submit an event to be processed through a pipeline
    #[napi]
    pub async fn submit(&self, pipeline_id: String, event: serde_json::Value) -> Result<serde_json::Value> { /* ... */ }

    /// Subscribe to the internal event bus (for notifications)
    #[napi]
    pub fn subscribe(&self, topic: String, callback: JsFunction) -> Result<()> { /* ... */ }

    /// Memory operations
    #[napi]
    pub async fn remember(&self, content: String, tags: Vec<String>) -> Result<String> { /* ... */ }

    #[napi]
    pub async fn recall(&self, query: String, limit: u32) -> Result<Vec<serde_json::Value>> { /* ... */ }
}
```

---

## 3. Python Shell — `gristmill-ml`

### 3.1 Purpose

The Python shell owns **model lifecycle**: training, fine-tuning, evaluation, experiment tracking, and ONNX export. It does NOT run inference in production — that's the Rust core's job. Python trains the models, Rust runs them.

### 3.2 Package Structure

```
gristmill-ml/
├── pyproject.toml
├── src/
│   └── gristmill_ml/
│       ├── __init__.py
│       ├── core.py               # PyO3 bridge re-export
│       │
│       ├── training/
│       │   ├── __init__.py
│       │   ├── sieve_trainer.py  # Train/fine-tune the Sieve classifier
│       │   ├── ner_trainer.py    # NER model training
│       │   ├── embedder.py       # Fine-tune domain embeddings
│       │   └── anomaly.py        # Anomaly detector training
│       │
│       ├── datasets/
│       │   ├── __init__.py
│       │   ├── feedback.py       # Import Sieve feedback logs → training data
│       │   ├── augmentation.py   # Synthetic data generation
│       │   └── loaders.py        # Dataset loading utilities
│       │
│       ├── export/
│       │   ├── __init__.py
│       │   ├── onnx_export.py    # PyTorch → ONNX conversion
│       │   ├── quantize.py       # ONNX quantization (INT8/FP16)
│       │   └── validate.py       # Cross-validate Rust vs Python inference
│       │
│       ├── experiments/
│       │   ├── __init__.py
│       │   ├── tracker.py        # MLflow / W&B integration
│       │   └── comparisons.py    # A/B test model versions
│       │
│       └── pipelines/
│           ├── __init__.py
│           ├── retrain_sieve.py  # End-to-end Sieve retraining pipeline
│           └── custom_model.py   # Framework for user-defined models
│
├── notebooks/
│   ├── 01-sieve-analysis.ipynb   # Analyze Sieve routing decisions
│   ├── 02-train-custom-model.ipynb
│   └── 03-export-to-onnx.ipynb
│
└── tests/
```

### 3.3 Key Python Interfaces

```python
# src/gristmill_ml/core.py

from gristmill_core import PyGristMill, PyGristEvent  # PyO3 bindings

class GristMill:
    """Python wrapper around the Rust core with ML extensions."""

    def __init__(self, config_path: str = "~/.gristmill/config.yaml"):
        self._core = PyGristMill(config_path)

    async def triage(self, event: dict) -> dict:
        """Route an event through the Sieve (runs in Rust)."""
        return await self._core.triage(PyGristEvent.from_dict(event))

    async def remember(self, content: str, tags: list[str] = []) -> str:
        """Store a memory (Rust Ledger)."""
        return await self._core.remember(content, tags)

    async def recall(self, query: str, limit: int = 10) -> list[dict]:
        """Retrieve memories (Rust dual-path search)."""
        return await self._core.recall(query, limit)

    def register_model(self, model_id: str, inference_fn: callable):
        """Register a Python model for the Grinders pool.
        Use sparingly — ONNX models in Rust are faster."""
        self._core.register_python_model(model_id, inference_fn)
```

```python
# src/gristmill_ml/training/sieve_trainer.py

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from gristmill_ml.datasets.feedback import FeedbackDataset
from gristmill_ml.export.onnx_export import export_to_onnx

class SieveTrainer:
    """Retrain the Sieve classifier using accumulated feedback data."""

    def __init__(
        self,
        base_model: str = "microsoft/MiniLM-L6-v2",
        num_labels: int = 4,  # LOCAL_ML, RULES, HYBRID, LLM_NEEDED
        feedback_dir: str = "~/.gristmill/feedback/",
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            base_model, num_labels=num_labels
        )
        self.feedback_dir = feedback_dir

    def prepare_dataset(self) -> DataLoader:
        """Load feedback logs from Rust core and prepare training data."""
        dataset = FeedbackDataset.from_feedback_logs(self.feedback_dir)
        return DataLoader(dataset, batch_size=32, shuffle=True)

    def train(self, epochs: int = 5, lr: float = 2e-5) -> dict:
        """Fine-tune the classifier on accumulated feedback."""
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        # ... training loop ...
        return {"accuracy": acc, "f1": f1, "epochs": epochs}

    def export(self, output_path: str = "~/.gristmill/models/sieve-v{n}.onnx"):
        """Export trained model to ONNX for Rust inference."""
        export_to_onnx(
            model=self.model,
            tokenizer=self.tokenizer,
            output_path=output_path,
            quantize="int8",  # Quantize for speed
            validate=True,    # Cross-check against PyTorch output
        )
        print(f"Exported to {output_path}")
        print("Restart gristmill daemon to load new model.")
```

### 3.4 The Retraining Loop

This is the closed-loop system that makes GristMill improve over time:

```
┌──────────────┐     ┌──────────────────┐     ┌───────────────┐
│  Rust Core   │────▶│  Feedback Logs   │────▶│  Python ML    │
│  (Sieve)     │     │  (JSONL)         │     │  (Training)   │
│              │     │                  │     │               │
│  Routes task │     │  Records:        │     │  Fine-tunes   │
│  + logs      │     │  - event hash    │     │  classifier   │
│  confidence  │     │  - route chosen  │     │  on feedback  │
│              │     │  - LLM used?     │     │               │
│              │     │  - could have    │     │  Exports new  │
│              │     │    been local?   │     │  ONNX model   │
│              │◀────│                  │◀────│               │
│  Hot-reloads │     │  Enriched with   │     │  Validates    │
│  new model   │     │  ground truth    │     │  accuracy     │
└──────────────┘     └──────────────────┘     └───────────────┘
```

---

## 4. TypeScript Shell — `gristmill-integrations`

### 4.1 Purpose

The TypeScript shell owns **external communication**: channel adapters, notification dispatch, the web dashboard, REST API, and the plugin system. It calls into the Rust core via napi-rs for all processing logic.

### 4.2 Package Structure

```
gristmill-integrations/
├── package.json
├── tsconfig.json
├── src/
│   ├── index.ts                    # Main entry + Rust core init
│   ├── core/
│   │   └── bridge.ts              # napi-rs binding wrapper
│   │
│   ├── hopper/                    # Intake adapters
│   │   ├── index.ts
│   │   ├── http-adapter.ts        # Express/Fastify REST + WebSocket
│   │   ├── webhook-adapter.ts     # GitHub, Stripe, custom webhooks
│   │   ├── cron-adapter.ts        # node-cron scheduled triggers
│   │   ├── mq-adapter.ts          # Redis Streams / NATS
│   │   └── fs-adapter.ts          # chokidar file watcher
│   │
│   ├── bell-tower/                # Notification system
│   │   ├── index.ts
│   │   ├── dispatcher.ts          # Priority-based routing
│   │   ├── digest.ts              # Digest batching
│   │   ├── quiet-hours.ts         # Time-based suppression
│   │   ├── channels/
│   │   │   ├── slack.ts
│   │   │   ├── telegram.ts
│   │   │   ├── discord.ts
│   │   │   ├── email.ts           # Nodemailer
│   │   │   ├── webhook-out.ts     # Outbound webhooks
│   │   │   ├── push.ts            # Web push / APNs / FCM
│   │   │   └── sms.ts             # Twilio
│   │   └── watches/
│   │       ├── watch-engine.ts    # Condition evaluation
│   │       └── watch-store.ts     # Persistent watch configs
│   │
│   ├── dashboard/                 # Web UI
│   │   ├── server.ts              # Fastify + static serving
│   │   ├── api/
│   │   │   ├── pipelines.ts       # Pipeline CRUD + status
│   │   │   ├── memory.ts          # Memory browse/search
│   │   │   ├── models.ts          # Model registry status
│   │   │   ├── notifications.ts   # Watch management
│   │   │   └── metrics.ts         # System metrics
│   │   └── ui/                    # React SPA (built separately)
│   │       └── dist/
│   │
│   ├── templates/                 # Handlebars templates
│   │   ├── engine.ts
│   │   └── templates/
│   │       ├── email-acknowledge.hbs
│   │       ├── slack-status.hbs
│   │       ├── digest-daily.hbs
│   │       └── alert-critical.hbs
│   │
│   └── plugins/                   # Plugin system
│       ├── loader.ts              # Dynamic plugin loading
│       ├── registry.ts            # Plugin registry
│       └── sdk.ts                 # Plugin development SDK
│
├── plugins/                       # Built-in plugins
│   ├── openclaw-bridge/           # Bridge to existing OpenClaw skills
│   ├── github-actions/
│   ├── jira-sync/
│   └── calendar-manager/
│
└── tests/
```

### 4.3 Key TypeScript Interfaces

```typescript
// src/core/bridge.ts

import { GristMill as RustCore } from '@gristmill/core';  // napi-rs bindings

export class GristMillBridge {
  private core: RustCore;

  constructor(configPath: string) {
    this.core = new RustCore(configPath);
  }

  /**
   * Submit an event from any adapter.
   * The Rust core handles triage → routing → execution.
   * TypeScript only constructs the event and handles the response.
   */
  async submit(pipelineId: string, event: GristEvent): Promise<PipelineResult> {
    return this.core.submit(pipelineId, event);
  }

  /**
   * Subscribe to internal event bus topics.
   * Used by Bell Tower to react to pipeline completions, alerts, etc.
   */
  subscribe(topic: string, handler: (event: BusEvent) => void): Unsubscribe {
    return this.core.subscribe(topic, handler);
  }

  /** Memory operations (delegates to Rust Ledger) */
  async remember(content: string, tags: string[]): Promise<string> {
    return this.core.remember(content, tags);
  }

  async recall(query: string, limit: number = 10): Promise<Memory[]> {
    return this.core.recall(query, limit);
  }
}
```

```typescript
// src/bell-tower/dispatcher.ts

import { GristMillBridge } from '../core/bridge';

export class BellTower {
  private bridge: GristMillBridge;
  private channels: Map<string, NotificationChannel>;
  private watches: WatchStore;
  private digestQueue: DigestQueue;

  constructor(bridge: GristMillBridge, config: BellTowerConfig) {
    this.bridge = bridge;
    this.channels = new Map();
    this.watches = new WatchStore(config.watchesPath);
    this.digestQueue = new DigestQueue(config.digest);

    // Subscribe to Rust event bus for pipeline completions
    this.bridge.subscribe('pipeline.completed', (event) => this.evaluate(event));
    this.bridge.subscribe('pipeline.failed', (event) => this.evaluate(event));
    this.bridge.subscribe('sieve.anomaly', (event) => this.evaluate(event));
    this.bridge.subscribe('ledger.threshold', (event) => this.evaluate(event));
  }

  private async evaluate(event: BusEvent): Promise<void> {
    const matchingWatches = await this.watches.findMatching(event);

    for (const watch of matchingWatches) {
      if (watch.priority === 'critical' || !watch.digest?.enabled) {
        // Immediate dispatch
        await this.dispatch(watch, event);
      } else {
        // Queue for digest
        this.digestQueue.add(watch, event);
      }
    }
  }

  private async dispatch(watch: Watch, event: BusEvent): Promise<void> {
    // Check quiet hours (critical bypasses)
    if (watch.priority !== 'critical' && this.isQuietHours(watch)) {
      this.digestQueue.add(watch, event);
      return;
    }

    for (const channelId of watch.channels) {
      const channel = this.channels.get(channelId);
      if (channel) {
        await channel.send({
          title: watch.name,
          body: this.formatMessage(watch, event),
          priority: watch.priority,
          metadata: event,
        });
      }
    }
  }
}
```

### 4.4 Plugin SDK

```typescript
// src/plugins/sdk.ts

export interface GristMillPlugin {
  /** Unique plugin identifier */
  id: string;
  name: string;
  version: string;

  /** Called once when plugin is loaded */
  initialize(context: PluginContext): Promise<void>;

  /** Called when plugin is unloaded */
  destroy(): Promise<void>;
}

export interface PluginContext {
  /** Access the Rust core */
  core: GristMillBridge;

  /** Register new Hopper adapters */
  registerAdapter(id: string, adapter: HopperAdapter): void;

  /** Register new Bell Tower channels */
  registerChannel(id: string, channel: NotificationChannel): void;

  /** Register new pipeline step types */
  registerStepType(id: string, executor: StepExecutor): void;

  /** Access configuration */
  config: PluginConfig;

  /** Scoped logger */
  logger: Logger;
}

// Example plugin
export default class GitHubActionsPlugin implements GristMillPlugin {
  id = 'github-actions';
  name = 'GitHub Actions Bridge';
  version = '1.0.0';

  async initialize(ctx: PluginContext): Promise<void> {
    // Register webhook adapter for GitHub events
    ctx.registerAdapter('github-webhook', new GitHubWebhookAdapter());

    // Register notification channel for PR comments
    ctx.registerChannel('github-comment', new GitHubCommentChannel());

    // Register custom step type for triggering workflows
    ctx.registerStepType('github-dispatch', new GitHubDispatchExecutor());

    ctx.logger.info('GitHub Actions plugin loaded');
  }

  async destroy(): Promise<void> {}
}
```

---

## 5. Inter-Process Communication

The three layers communicate through two mechanisms:

### 5.1 In-Process (Primary)

When running as a single daemon, Python and TypeScript call into Rust via FFI:

```
TypeScript Process
  └─ napi-rs ──→ Rust shared library (libgristmill_core.so)
                    ├─ Sieve
                    ├─ Grinders
                    ├─ Millwright
                    ├─ Ledger
                    └─ Hammer

Python Process (for training / notebooks)
  └─ PyO3 ────→ Same Rust shared library
```

### 5.2 IPC (Alternative: Microservice Mode)

For deployment flexibility, the Rust core can run as a standalone daemon with a MessagePack-over-Unix-socket API:

```
gristmill-daemon (Rust binary)
  ├─ Unix socket: /tmp/gristmill.sock
  │     ├─ TypeScript connects via IPC
  │     └─ Python connects via IPC
  └─ Optional: gRPC on localhost:9090
```

This allows independent scaling and restart of each layer.

---

## 6. Configuration

```yaml
# ~/.gristmill/config.yaml

core:
  workspace: ~/.gristmill
  log_level: info
  mode: daemon          # daemon | cli | embedded

sieve:
  model: ./models/sieve-v1.onnx
  confidence_threshold: 0.85
  feedback_dir: ./feedback/
  cache_size: 10000

grinders:
  workers: auto         # auto = CPU cores - 1
  models:
    intent-classifier:
      runtime: onnx
      path: ./models/intent-v3.onnx
      warm: true
      max_batch: 32
      timeout_ms: 500
    ner-extractor:
      runtime: onnx
      path: ./models/ner-multilingual.onnx
      warm: true
    embedder:
      runtime: onnx
      path: ./models/minilm-l6-v2.onnx
      warm: true
    summarizer:
      runtime: gguf
      path: ./models/phi3-mini-4k.Q4_K_M.gguf
      warm: false
      max_tokens: 512

hammer:
  providers:
    anthropic:
      api_key: ${ANTHROPIC_API_KEY}
      default_model: claude-sonnet-4-20250514
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

millwright:
  max_concurrency: 8
  default_timeout_ms: 30000
  checkpoint_dir: ./checkpoints/

ledger:
  hot:
    max_size_mb: 512
  warm:
    db_path: ./memory/warm.db
    vector_index_path: ./memory/vectors.usearch
    fts_enabled: true
  cold:
    archive_dir: ./memory/cold/
    compression: zstd
    compress_level: 3
  compaction:
    interval_hours: 6
    similarity_threshold: 0.90
    verbose_threshold_tokens: 512
    stale_days: 90

bell_tower:
  channels:
    slack:
      webhook_url: ${SLACK_WEBHOOK_URL}
    telegram:
      bot_token: ${TELEGRAM_BOT_TOKEN}
      chat_id: ${TELEGRAM_CHAT_ID}
    email:
      smtp_host: smtp.gmail.com
      smtp_port: 587
      username: ${EMAIL_USER}
      password: ${EMAIL_PASS}
  quiet_hours:
    start: "22:00"
    end: "07:00"
    timezone: UTC
    override_for: [critical]
  digest:
    enabled: true
    interval_minutes: 60
    max_items: 50

integrations:
  dashboard:
    enabled: true
    port: 8420
  plugins_dir: ./plugins/
```

---

## 7. Build & Distribution

```bash
# Development
cargo build                          # Rust core
cd gristmill-ml && pip install -e .  # Python shell (editable)
cd gristmill-integrations && pnpm install && pnpm build  # TS shell

# Release: single binary (Rust CLI + embedded daemon)
cargo build --release --target x86_64-unknown-linux-musl
# → target/release/gristmill (static binary, ~15MB + models)

# Release: napi-rs prebuilt binaries for npm
cd crates/grist-ffi && napi build --release
# → npm package @gristmill/core with platform-specific .node files

# Release: PyO3 wheels
cd crates/grist-ffi && maturin build --release
# → gristmill_core-*.whl for pip install

# Docker (all-in-one)
docker build -t gristmill:latest .
# Multi-stage: Rust builder → Node runtime → final slim image
```

---

## 8. CLI Interface

```bash
# Core operations
gristmill start                      # Start daemon
gristmill stop                       # Stop daemon
gristmill status                     # Show system health
gristmill doctor                     # Diagnose issues

# Model management
gristmill models list                # Show loaded models
gristmill models pull starter-pack   # Download model pack
gristmill models reload sieve        # Hot-reload a model
gristmill models benchmark           # Run inference benchmarks

# Pipeline management
gristmill pipeline list
gristmill pipeline create --from-template content-digest
gristmill pipeline run <id> --input '{"text": "..."}'
gristmill pipeline inspect <run-id>  # Show DAG execution trace

# Memory
gristmill memory search "authentication bug"
gristmill memory stats               # Tier sizes, hit rates
gristmill memory compact --force     # Trigger manual compaction
gristmill memory export --format json

# Notifications
gristmill watch list
gristmill watch create --name "..." --condition "..." --channel telegram
gristmill watch test <id>            # Send test notification

# Training (delegates to Python shell)
gristmill train sieve --epochs 5     # Retrain Sieve from feedback
gristmill train export --model sieve --quantize int8
gristmill train evaluate --model sieve-v2 --against sieve-v1

# Metrics
gristmill metrics                    # Token usage, cache hits, routing stats
gristmill metrics --period 7d        # Last 7 days
```

---

*GristMill v2: Rust grinds. Python trains. TypeScript connects.*
