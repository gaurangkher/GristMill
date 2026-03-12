# GristMill v2 — Getting Started

> *"Rust grinds. Python trains. TypeScript connects."*

GristMill is a local-first AI orchestration system. Every event is triaged in
**< 5 ms** by a Rust classifier, routed to a local ONNX/GGUF model or a
deterministic rule engine, and only escalated to an LLM when confidence falls
below your configured threshold. Memory, pipelines, notifications, and model
retraining all happen in the same unified system.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Create the workspace](#2-create-the-workspace)
3. [Configure GristMill](#3-configure-gristmill)
4. [Start the server](#4-start-the-server)
5. [Core API](#5-core-api)
6. [How to add a Plugin](#6-how-to-add-a-plugin)
7. [How to define Pipelines](#7-how-to-define-pipelines)
8. [How to train the Sieve](#8-how-to-train-the-sieve)
9. [Notifications (Bell Tower)](#9-notifications-bell-tower)
10. [Observability](#10-observability)

---

## 1. Prerequisites

| Tool | Min version | Install |
|------|-------------|---------|
| Rust + Cargo | 1.80 | `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \| sh` |
| Node.js | 18 | [nodejs.org](https://nodejs.org) |
| pnpm | 9 | `npm install -g pnpm@9` |
| Python | 3.10 | [python.org](https://python.org) |
| Docker | any | [docs.docker.com](https://docs.docker.com/get-docker/) — optional |

---

## 2. Create the workspace

GristMill reads and writes to `~/.gristmill/` by default.

```bash
mkdir -p ~/.gristmill/{feedback,models,memory,plugins,checkpoints}
```

Then set the secrets GristMill needs at runtime:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."         # LLM escalation
export SLACK_WEBHOOK_URL="https://hooks.slack.com/..."  # optional notifications
export GRISTMILL_CONFIG="$HOME/.gristmill/config.yaml"
```

---

## 3. Configure GristMill

Create `~/.gristmill/config.yaml`. Below is a minimal but fully-functional
config — each section maps to a Rust crate.

```yaml
# ── Sieve (triage classifier) ────────────────────────────────────────────────
sieve:
  confidence_threshold: 0.85  # escalate to LLM when confidence < this
  feedback_dir: ~/.gristmill/feedback/
  cache_size: 10000           # exact-match cache entries

# ── Grinders (local model pool) ──────────────────────────────────────────────
grinders:
  workers: auto               # auto = number of CPU cores − 1
  models:
    - id: intent-classifier-v1
      runtime: onnx           # onnx | gguf | tflite
      path: ~/.gristmill/models/sieve-v1.onnx
      warm: true              # pre-load on startup

# ── Hammer (LLM gateway) ─────────────────────────────────────────────────────
hammer:
  providers:
    anthropic:
      api_key: ${ANTHROPIC_API_KEY}
      default_model: claude-sonnet-4
    ollama:
      base_url: http://localhost:11434
      model: llama3.1:8b      # local fallback when Anthropic is unreachable
  budget:
    daily_tokens: 500000
    monthly_tokens: 10000000
  cache:
    enabled: true
    similarity_threshold: 0.92   # reuse cached LLM response when ≥ this similar

# ── Millwright (DAG scheduler) ───────────────────────────────────────────────
millwright:
  max_concurrency: 8          # max parallel steps across all running pipelines
  default_timeout_ms: 30000
  checkpoint_dir: ~/.gristmill/checkpoints/

# ── Ledger (three-tier memory) ───────────────────────────────────────────────
ledger:
  hot:
    max_size_mb: 512
  warm:
    db_path: ~/.gristmill/memory/warm.db
    vector_index_path: ~/.gristmill/memory/vectors.usearch
    fts_enabled: true
  cold:
    archive_dir: ~/.gristmill/memory/cold/
    compression: zstd
  compaction:
    interval_hours: 6
    similarity_threshold: 0.90

# ── Bell Tower (notifications) ───────────────────────────────────────────────
bell_tower:
  channels:
    slack:
      webhook_url: ${SLACK_WEBHOOK_URL}
    email:
      smtp_host: smtp.gmail.com
      smtp_port: 587
      username: ${EMAIL_USER}
      password: ${EMAIL_PASS}
  quiet_hours:
    start: "22:00"
    end: "07:00"
    override_for: [critical]  # critical priority always sends
  digest:
    enabled: true
    interval_minutes: 60      # batch low/normal alerts into hourly digest

# ── Integrations (TypeScript shell) ──────────────────────────────────────────
integrations:
  dashboard:
    port: 3000
  plugins_dir: ~/.gristmill/plugins/
```

---

## 4. Start the server

### Option A — Mock mode *(recommended for development)*

No Rust daemon required. Everything runs in-memory — perfect for exploring the
API, building plugins, and running the dashboard without compiling native code.

```bash
cd gristmill-integrations
pnpm install
GRISTMILL_MOCK_BRIDGE=1 pnpm dev
```

Dashboard: **http://localhost:3000**

The mock bridge:
- `triage()` → always returns `{ route: "LOCAL_ML", confidence: 0.9 }`
- `remember()` / `recall()` → in-memory store with keyword search
- `escalate()` → returns a mock LLM response instantly
- Memory is **not** persisted between restarts

---

### Option B — Full mode *(real Rust core)*

```bash
# Terminal 1 — Rust daemon
cd gristmill-core
cargo run -p gristmill-daemon

# Terminal 2 — TypeScript shell + dashboard
cd gristmill-integrations
pnpm dev
```

The daemon creates a Unix socket at `~/.gristmill/gristmill.sock`. The
TypeScript shell connects to it automatically via the IPC bridge.

---

### Option C — Docker *(simplest for production)*

```bash
docker build -t gristmill:latest .

docker run -p 3000:3000 \
  -e ANTHROPIC_API_KEY="sk-ant-..." \
  -v ~/.gristmill:/data/gristmill \
  gristmill:latest
```

---

## 5. Core API

### Health check

```bash
curl http://localhost:3000/api/metrics/health
# → { "status": "ok", "version": "0.1.0", "uptime_s": 42 }
```

### Submit an event for triage

Every inbound signal (HTTP call, webhook, cron tick, etc.) is submitted as a
`GristEvent`. The Sieve classifies it in < 5 ms and returns a routing decision.

```bash
curl -X POST http://localhost:3000/events \
  -H "Content-Type: application/json" \
  -d '{
    "channel": "http",
    "payload": { "text": "disk usage at 95% on prod-db-01" },
    "priority": "high",
    "tags": { "source": "alertmanager", "host": "prod-db-01" }
  }'
```

```json
{
  "route": "LOCAL_ML",
  "confidence": 0.93,
  "modelId": "intent-classifier-v1",
  "reason": "high-confidence local classification",
  "estimatedTokens": null
}
```

**Route values:**

| Route | Meaning |
|-------|---------|
| `LOCAL_ML` | Handle with a local ONNX/GGUF model — no LLM call |
| `RULES` | Handle with a deterministic rule engine |
| `HYBRID` | Local pre-processing + LLM refinement |
| `LLM_NEEDED` | Full LLM reasoning required |

---

### Memory — store and retrieve

```bash
# Store a memory
curl -X POST http://localhost:3000/api/memory/remember \
  -H "Content-Type: application/json" \
  -d '{
    "content": "prod-db-01 disk alert resolved by pruning WAL logs",
    "tags": ["infra", "postgres", "incident"]
  }'
# → { "id": "01HXYZ..." }

# Recall memories by semantic + keyword search
curl -X POST http://localhost:3000/api/memory/recall \
  -H "Content-Type: application/json" \
  -d '{ "query": "disk postgres", "limit": 5 }'
# → { "results": [ { "id": "...", "content": "...", "score": 0.94, "tier": "hot" } ] }
```

**Memory tiers** (fully automatic):

| Tier | Storage | Speed | Capacity |
|------|---------|-------|----------|
| Hot | in-process LRU | < 1 ms | configurable (default 512 MB) |
| Warm | SQLite FTS5 + HNSW vectors | ~10 ms | unlimited |
| Cold | zstd-compressed JSONL | ~100 ms | unlimited archive |

Compaction runs every 6 hours: it deduplicates similar entries, summarises
verbose memories, and demotes stale ones to cold storage.

---

## 6. How to add a Plugin

Plugins extend GristMill with new event adapters, notification channels, and
custom pipeline step types. They are plain `.js` / `.mjs` files dropped into
the plugins directory — no build step, no restart required (plugins are loaded
at startup).

### Plugin interface

A plugin is any object that satisfies:

```typescript
interface GristMillPlugin {
  name: string;       // unique ID, e.g. "acme/github-enricher"
  version: string;    // semver, e.g. "1.0.0"
  register(ctx: PluginContext): void | Promise<void>;
  unregister?(): void | Promise<void>;  // optional teardown
}
```

The `PluginContext` passed to `register()` exposes:

```typescript
interface PluginContext {
  bridge: IBridge;     // full access to triage, memory, escalate, pipelines

  // Register an inbound event adapter
  registerAdapter(name: string, handler: AdapterHandler): void;

  // Register a custom notification channel
  registerChannel(name: string, channel: NotificationChannel): void;

  // Register a custom pipeline step type
  registerStepType(name: string, executor: StepExecutor): void;

  // Subscribe to an internal bus topic
  subscribe(topic: string): AsyncIterable<unknown>;

  // Plugin-attributed logging
  log(level: "debug" | "info" | "warn" | "error", message: string): void;
}
```

---

### Creating a plugin

**Step 1 — Create the file**

```bash
touch ~/.gristmill/plugins/my-plugin.mjs
```

**Step 2 — Implement the plugin**

```javascript
// ~/.gristmill/plugins/my-plugin.mjs

/** @type {import('gristmill-integrations').GristMillPlugin} */
const MyPlugin = {
  name: "acme/my-plugin",
  version: "1.0.0",

  async register(ctx) {
    // ── 1. Inbound adapter ────────────────────────────────────────────────
    // Called when an event arrives on the "my-webhook" channel.
    // Must return a GristEventInit to be triaged by the Sieve.
    ctx.registerAdapter("my-webhook", async (rawEvent) => {
      const body = rawEvent;
      return {
        channel: "webhook",
        payload: { text: body.message, metadata: body.extra },
        priority: body.urgent ? "high" : "normal",
        tags: { source: "my-system", id: String(body.id) },
      };
    });

    // ── 2. Notification channel ───────────────────────────────────────────
    // Bell Tower calls send() when a Watch matches and routes to this channel.
    ctx.registerChannel("my-pagerduty", {
      async send(notification) {
        const payload = {
          routing_key: process.env.PAGERDUTY_KEY,
          event_action: "trigger",
          payload: {
            summary: notification.title,
            severity: notification.priority === "critical" ? "critical" : "warning",
            source: "gristmill",
            custom_details: notification.body,
          },
        };
        await fetch("https://events.pagerduty.com/v2/enqueue", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
      },
    });

    // ── 3. Custom pipeline step type ──────────────────────────────────────
    // Millwright calls this executor when a pipeline has a step of type
    // "typescript_call" with adapter = "my-enricher".
    ctx.registerStepType("my-enricher", async (stepCtx) => {
      const input = stepCtx.input;

      // Use the bridge to recall related memories
      const memories = await stepCtx.bridge.recall(
        String(input?.text ?? ""),
        3
      );

      return {
        output: {
          enriched: true,
          relatedMemories: memories.map((m) => m.memory.content),
        },
        metadata: { stepId: stepCtx.stepId },
      };
    });

    // ── 4. Bus subscription ───────────────────────────────────────────────
    // React to internal events from the Rust core.
    (async () => {
      for await (const event of ctx.subscribe("pipeline.failed")) {
        ctx.log("warn", `Pipeline failed: ${JSON.stringify(event)}`);
      }
    })();

    ctx.log("info", "my-plugin registered successfully");
  },

  async unregister() {
    // Close connections, flush buffers, etc.
  },
};

export default MyPlugin;
```

**Step 3 — Verify it loaded**

```bash
curl -s http://localhost:3000/api/plugins | jq .
```

```json
{
  "count": 1,
  "plugins": ["acme/my-plugin"],
  "adapters": ["my-webhook"],
  "channels": ["my-pagerduty"],
  "stepTypes": ["my-enricher"]
}
```

---

### Plugin capabilities reference

| What you want | API call | When it fires |
|---|---|---|
| Normalise inbound events | `registerAdapter(name, fn)` | When Hopper receives an event on that channel |
| Send custom notifications | `registerChannel(name, { send })` | When Bell Tower routes a Watch alert to that channel ID |
| Add a pipeline step | `registerStepType(name, fn)` | When Millwright executes a `TypeScriptCall` step with that adapter name |
| Watch bus events | `subscribe(topic)` | Whenever Rust core publishes to that topic |
| Read/write memory | `ctx.bridge.remember()` / `ctx.bridge.recall()` | On demand |
| Escalate to LLM | `ctx.bridge.escalate(prompt)` | On demand (respects token budget) |

---

## 7. How to define Pipelines

Pipelines are DAGs of steps executed by **Millwright** (the Rust orchestrator).
Steps run in parallel by default; dependencies are expressed via `depends_on`.

### Step types

```
StepType (Rust enum → JSON "kind" field)
├── local_ml       — run a local ONNX/GGUF model via grist-grinders
├── rule           — deterministic rule engine
├── llm            — call an LLM via grist-hammer (respects budget + cache)
├── external       — HTTP/webhook call to an external service
├── gate           — approval barrier (auto predicate or human approval)
├── python_call    — call into the Python ML shell via PyO3
└── typescript_call — call a registered plugin step type
```

### Define a pipeline (JSON)

```json
{
  "id": "infra-alert-pipeline",
  "steps": [
    {
      "id": "classify-intent",
      "kind": "local_ml",
      "model_id": "intent-classifier-v1",
      "prefer_local": true,
      "timeout_ms": 2000
    },
    {
      "id": "check-memory",
      "kind": "typescript_call",
      "adapter": "my-enricher",
      "depends_on": ["classify-intent"]
    },
    {
      "id": "summarise",
      "kind": "llm",
      "prompt_template": "Summarise this infrastructure alert and suggest a fix: {payload.text}",
      "max_tokens": 256,
      "depends_on": ["check-memory"],
      "prefer_local": false
    },
    {
      "id": "approval-gate",
      "kind": "gate",
      "condition": "always_true",
      "requires_approval": true,
      "approval_channel": "slack",
      "depends_on": ["summarise"]
    },
    {
      "id": "run-remediation",
      "kind": "external",
      "action": "http_post",
      "config": {
        "url": "https://ops-api.internal/remediate",
        "headers": { "X-Source": "gristmill" }
      },
      "depends_on": ["approval-gate"],
      "timeout_ms": 10000,
      "retry": {
        "max_attempts": 3,
        "initial_delay_ms": 500,
        "backoff": "exponential"
      }
    }
  ],
  "max_concurrency": 4,
  "default_timeout_ms": 30000,
  "on_failure": "abort"
}
```

### Step fields reference

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `id` | string | **required** | Unique name within the pipeline |
| `kind` | string | **required** | Step type (`local_ml`, `llm`, `gate`, etc.) |
| `depends_on` | string[] | `[]` | IDs of steps that must complete first |
| `prefer_local` | bool | `true` | Try local model before escalating to LLM |
| `timeout_ms` | number | pipeline default | Wall-clock limit for this step |
| `requires_approval` | bool | `false` | Pause and wait for human approval |
| `approval_channel` | string | — | Notification channel to send approval request |
| `retry` | object | none | Retry policy (see below) |

**Retry policy fields:**

```json
{
  "retry": {
    "max_attempts": 3,
    "initial_delay_ms": 500,
    "backoff": "exponential"   // "exponential" | "linear" | "constant"
  }
}
```

**Failure policies** (`on_failure`):

| Value | Behaviour |
|-------|-----------|
| `abort` | Stop the pipeline on first step failure |
| `continue_on_error` | Log the error, keep running remaining steps |
| `skip_and_continue` | Skip failed step's dependents, continue others |

---

### Register and run via API

```bash
# Register
curl -X POST http://localhost:3000/api/pipelines \
  -H "Content-Type: application/json" \
  -d @infra-alert-pipeline.json
# → { "registered": true, "id": "infra-alert-pipeline" }

# List registered pipelines
curl http://localhost:3000/api/pipelines
# → { "pipelines": ["infra-alert-pipeline"] }

# Run with an event payload
curl -X POST http://localhost:3000/api/pipelines/infra-alert-pipeline/run \
  -H "Content-Type: application/json" \
  -d '{ "text": "disk usage 95% on prod-db-01", "host": "prod-db-01" }'
```

```json
{
  "pipelineId": "infra-alert-pipeline",
  "result": {
    "runId": "01HXYZ...",
    "pipelineId": "infra-alert-pipeline",
    "succeeded": true,
    "elapsedMs": 847,
    "output": { ... }
  }
}
```

---

### Register a pipeline from Rust

```rust
use grist_millwright::dag::{Pipeline, Step, StepType};

let pipeline = Pipeline::new("infra-alert-pipeline")
    .with_step(
        Step::new("classify-intent", StepType::LocalMl {
            model_id: "intent-classifier-v1".into(),
        })
    )
    .with_step(
        Step::new("summarise", StepType::Llm {
            prompt_template: "Summarise this alert: {payload.text}".into(),
            max_tokens: 256,
        })
        .with_deps(["classify-intent"])
    );

millwright.register_pipeline(pipeline);
```

---

## 8. How to train the Sieve

The Sieve classifier maps a 392-dimensional feature vector to one of four
routing decisions. The feature vector layout **must match exactly** between
Rust and Python:

```
[0:384]   L2-normalised MiniLM-L6-v2 sentence embedding
[384]     log-scaled token count: ln(tc+1) / ln(2049), clamped [0, 1]
[385]     source channel ordinal / 9.0
[386]     priority / 3.0
[387]     entity density (placeholder)
[388]     question probability
[389]     code probability
[390]     type-token ratio
[391]     ambiguity score
```

### Step 1 — Install the Python shell

```bash
cd gristmill-ml
pip install -e ".[dev]"
```

This installs: `torch`, `transformers`, `sentence-transformers`, `onnxruntime`,
`mlflow`, `scikit-learn`, and all training dependencies.

---

### Step 2 — Accumulate feedback data

The Rust Sieve logs every routing decision to
`~/.gristmill/feedback/feedback-YYYY-MM-DD.jsonl`. Each line is a JSON object:

```json
{
  "event_id": "01HXYZ...",
  "text": "schedule a meeting with Alice",
  "channel": "http",
  "priority": "normal",
  "route": "LOCAL_ML",
  "confidence": 0.92,
  "timestamp_ms": 1700000000000
}
```

Collect at least a few hundred decisions before retraining. For best results,
gather data across all four route classes (`LOCAL_ML`, `RULES`, `HYBRID`,
`LLM_NEEDED`).

---

### Step 3 — Train the classifier

```bash
# Using the CLI entry point
gristmill-train-sieve \
  --feedback-dir ~/.gristmill/feedback/ \
  --epochs 10 \
  --lr 2e-4 \
  --output ~/.gristmill/models/sieve-v2.pt

# Or from Python
```

```python
from gristmill_ml.training.sieve_trainer import SieveTrainer

trainer = SieveTrainer(
    feedback_dir="~/.gristmill/feedback/",
    experiment_name="sieve-v2",   # MLflow experiment name
)

trainer.prepare_dataset()         # loads JSONL, augments, splits train/val
trainer.train(epochs=10, lr=2e-4) # trains 2-layer MLP on MiniLM embeddings
```

The trainer uses:
- **MiniLM-L6-v2** for sentence embeddings (384 dims, runs locally, < 100 MB)
- A lightweight **2-layer MLP** head on top (< 1 MB)
- `AdamW` + `CosineAnnealingLR` schedule
- `WeightedRandomSampler` for class-imbalance handling

---

### Step 4 — Export to ONNX (INT8 quantized)

Rust loads **ONNX** models only — PyTorch checkpoints cannot be hot-reloaded
directly.

```bash
gristmill-export \
  --model ~/.gristmill/models/sieve-v2.pt \
  --output ~/.gristmill/models/sieve-v2.onnx \
  --quantize    # INT8 quantization (~0.5 MB → ~0.2 MB, negligible accuracy loss
```

```python
from pathlib import Path
from gristmill_ml.export.onnx_export import OnnxExporter
from gristmill_ml.training.sieve_trainer import SieveClassifierHead
import torch

model = SieveClassifierHead()
model.load_state_dict(torch.load("~/.gristmill/models/sieve-v2.pt"))

OnnxExporter.export_classifier(
    model=model,
    output_path=Path("~/.gristmill/models/sieve-v2.onnx"),
    quantize=True,
    opset=17,
)
```

**ONNX interface** (must match `grist-grinders`):

| Name | Shape | dtype |
|------|-------|-------|
| input: `features` | `[batch, 392]` | float32 |
| output: `logits` | `[batch, 4]` | float32 |

---

### Step 5 — Validate cross-runtime parity

Before deploying, verify the ONNX output matches PyTorch output within
tolerance:

```bash
gristmill-validate \
  --pytorch ~/.gristmill/models/sieve-v2.pt \
  --onnx    ~/.gristmill/models/sieve-v2.onnx
```

```python
from gristmill_ml.export.validate import validate_parity

report = validate_parity(
    pytorch_path="~/.gristmill/models/sieve-v2.pt",
    onnx_path="~/.gristmill/models/sieve-v2.onnx",
)
print(f"Max absolute error:   {report.max_abs_error:.6f}")
print(f"Mean cosine similarity: {report.cosine_similarity:.4f}")
assert report.passes_threshold, "Parity check failed — do not deploy"
```

---

### Step 6 — Hot-reload into Rust (no restart needed)

Copy the validated `.onnx` file to your models directory. The Rust
`ModelRegistry` watches for file changes and swaps the model atomically:

```bash
cp ~/.gristmill/models/sieve-v2.onnx ~/.gristmill/models/sieve-v1.onnx
# Rust detects the change and reloads — the next triage uses sieve-v2
```

To point to the new file explicitly, update `config.yaml`:

```yaml
grinders:
  models:
    - id: intent-classifier-v1
      runtime: onnx
      path: ~/.gristmill/models/sieve-v2.onnx  # updated path
      warm: true
```

---

### Closed feedback loop (recommended cadence)

```
Week 1-4   Rust Sieve logs decisions → ~/.gristmill/feedback/
Week 5     gristmill-train-sieve (retrain on new data)
           gristmill-export (produce sieve-v2.onnx)
           gristmill-validate (verify parity)
           Copy to models dir → Rust hot-reloads
           Repeat
```

Automating with cron:

```bash
# ~/.gristmill/retrain.sh
#!/usr/bin/env bash
set -e
cd /path/to/gristmill-ml
gristmill-train-sieve --epochs 10 --output /tmp/sieve-new.pt
gristmill-export --model /tmp/sieve-new.pt --output /tmp/sieve-new.onnx --quantize
gristmill-validate --pytorch /tmp/sieve-new.pt --onnx /tmp/sieve-new.onnx
cp /tmp/sieve-new.onnx ~/.gristmill/models/sieve-v1.onnx
echo "Sieve retrained and hot-reloaded at $(date)"
```

```
# crontab -e
0 3 * * 0  ~/.gristmill/retrain.sh >> ~/.gristmill/retrain.log 2>&1
```

---

## 9. Notifications (Bell Tower)

Bell Tower subscribes to the internal `grist-bus` and forwards events to
configured channels (Slack, email, or custom plugin channels).

### Create a Watch

```bash
curl -X POST http://localhost:3000/api/watches \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Critical pipeline failures",
    "condition": "topic == '\''pipeline.failed'\'' && event.priority == '\''critical'\''",
    "channels": ["slack"],
    "priority": "critical"
  }'
```

### Built-in bus topics

| Topic | Fires when |
|-------|-----------|
| `pipeline.completed` | A pipeline run finishes successfully |
| `pipeline.failed` | A pipeline run errors or times out |
| `sieve.anomaly` | Sieve confidence < 0.5 (unusual event) |
| `ledger.threshold` | Memory usage exceeds configured limit |
| `hammer.budget` | Daily LLM token budget reaches 80% |

### Quiet hours and digest mode

Controlled via `config.yaml`:

```yaml
bell_tower:
  quiet_hours:
    start: "22:00"
    end: "07:00"
    override_for: [critical]  # critical always sends, even during quiet hours
  digest:
    enabled: true
    interval_minutes: 60      # bundle low/normal alerts into an hourly summary
```

---

## 10. Observability

### Health and metrics

```bash
# System health
curl http://localhost:3000/api/metrics/health

# LLM token budget
curl http://localhost:3000/api/metrics/budget
# → { "daily_used": 125000, "daily_limit": 500000, "pct_used": 25, "status": "ok" }
```

### Structured logging (Rust)

```bash
# Set per-crate log levels
export RUST_LOG="grist_sieve=debug,grist_hammer=info,grist_ledger=warn"
cargo run -p gristmill-daemon
```

### Running tests

```bash
# Rust — all crates
cd gristmill-core && cargo test

# Rust — sieve latency regression (asserts p99 < 5 ms)
cargo test -p grist-sieve --release -- --include-ignored latency

# Python — unit tests
cd gristmill-ml && pytest tests/ -v

# TypeScript — unit tests
cd gristmill-integrations && pnpm test
```

---

## Quick reference

```bash
# Start in mock/dev mode (no daemon)
cd gristmill-integrations && GRISTMILL_MOCK_BRIDGE=1 pnpm dev

# Submit event
curl -X POST http://localhost:3000/events \
  -H "Content-Type: application/json" \
  -d '{"channel":"http","payload":{"text":"hello gristmill"}}'

# Store memory
curl -X POST http://localhost:3000/api/memory/remember \
  -H "Content-Type: application/json" \
  -d '{"content":"...","tags":["tag1","tag2"]}'

# Recall memory
curl -X POST http://localhost:3000/api/memory/recall \
  -H "Content-Type: application/json" \
  -d '{"query":"...","limit":5}'

# Register pipeline
curl -X POST http://localhost:3000/api/pipelines \
  -H "Content-Type: application/json" \
  -d @pipeline.json

# Run pipeline
curl -X POST http://localhost:3000/api/pipelines/my-pipeline/run \
  -H "Content-Type: application/json" \
  -d '{"text":"..."}'

# List plugins
curl http://localhost:3000/api/plugins

# Retrain sieve
cd gristmill-ml
gristmill-train-sieve --epochs 10 --output ~/.gristmill/models/sieve-v2.pt
gristmill-export --model ~/.gristmill/models/sieve-v2.pt \
                 --output ~/.gristmill/models/sieve-v2.onnx --quantize
gristmill-validate --pytorch ~/.gristmill/models/sieve-v2.pt \
                   --onnx ~/.gristmill/models/sieve-v2.onnx
```

---

For deeper technical details see the individual component READMEs in each
subdirectory, and the full architecture spec in
[`gristmill-v2-architecture.md`](./gristmill-v2-architecture.md).
