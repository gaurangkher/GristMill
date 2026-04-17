# GristMill v2 — Cookbook

> Concrete recipes for common GristMill use cases. Each recipe is self-contained — jump to any one directly.

## Recipes

1. [Route an event and act on the decision](#recipe-1--route-an-event-and-act-on-the-decision)
2. [Build a semantic knowledge base](#recipe-2--build-a-semantic-knowledge-base)
3. [Run a multi-step pipeline with memory enrichment](#recipe-3--run-a-multi-step-pipeline-with-memory-enrichment)
4. [Set up Slack alerts for budget and pipeline failures](#recipe-4--set-up-slack-alerts-for-budget-and-pipeline-failures)
5. [Write a plugin (PagerDuty channel)](#recipe-5--write-a-plugin-pagerduty-channel)
6. [Ingest from RabbitMQ](#recipe-6--ingest-from-rabbitmq)
7. [Retrain the Sieve classifier (closed loop)](#recipe-7--retrain-the-sieve-classifier-closed-loop)
8. [Compare two model versions before deploying](#recipe-8--compare-two-model-versions-before-deploying)
9. [Use the dashboard to manage watches visually](#recipe-9--use-the-dashboard-to-manage-watches-visually)
10. [Test your integration without a running daemon](#recipe-10--test-your-integration-without-a-running-daemon)
11. [Check system health from the CLI](#recipe-11--check-system-health-from-the-cli)
12. [Bootstrap models on a fresh install](#recipe-12--bootstrap-models-on-a-fresh-install)
13. [Use Slack as a Second Brain](#recipe-13--use-slack-as-a-second-brain)
14. [Run GristMill on Mac with GPU-accelerated Ollama](#recipe-14--run-gristmill-on-mac-with-gpu-accelerated-ollama)

---

### Recipe 1 — Route an event and act on the decision

**What it does:** Posts an event to the Sieve triage endpoint and branches a shell script based on the returned route decision.

**When to use it:** When you want to integrate GristMill routing into an existing shell-based automation or CI pipeline.

```bash
RESP=$(curl -sf -X POST http://localhost:3000/events \
  -H "Content-Type: application/json" \
  -d '{"channel":"http","payload":{"text":"summarise last week sales report"},"priority":"normal"}')

ROUTE=$(echo "$RESP" | python3 -c "import json,sys; print(json.load(sys.stdin)['route'])")
CONFIDENCE=$(echo "$RESP" | python3 -c "import json,sys; print(json.load(sys.stdin)['confidence'])")

echo "Route: $ROUTE  Confidence: $CONFIDENCE"

case "$ROUTE" in
  LOCAL_ML)   echo "Handled locally — no LLM cost" ;;
  LLM_NEEDED) echo "Escalating to LLM..." ;;
  HYBRID)     echo "Local pre-process + LLM refinement" ;;
  RULES)      echo "Deterministic rule matched" ;;
esac
```

---

### Recipe 2 — Build a semantic knowledge base

**What it does:** Stores a set of operational knowledge entries in GristMill's three-tier memory and retrieves them by semantic search.

**When to use it:** When you want to give pipelines or LLM escalations access to team-specific runbooks, topology facts, or incident history.

```bash
# Store knowledge entries
for entry in \
  "prod-db-01 runs PostgreSQL 15 with WAL archiving to S3" \
  "Disk alert threshold is 85%. Resolve by pruning WAL: pg_archivecleanup" \
  "Kubernetes cluster uses 3 control-plane nodes, 12 worker nodes" \
  "Deploys run via GitHub Actions — merge to main triggers staging, tag v* triggers prod" \
  "On-call rotation: Mon-Thu Alice, Fri-Sun Bob. PagerDuty team: infra-oncall"
do
  curl -sf -X POST http://localhost:3000/api/memory/remember \
    -H "Content-Type: application/json" \
    -d "{\"content\": \"$entry\", \"tags\": [\"infra\",\"runbook\"]}" | jq -r '.id'
done

# Semantic search
curl -sf -X POST http://localhost:3000/api/memory/recall \
  -H "Content-Type: application/json" \
  -d '{"query":"disk alert postgres how to fix","limit":3}' | jq '.results[] | {score: .score, content: .memory.content}'
```

---

### Recipe 3 — Run a multi-step pipeline with memory enrichment

**What it does:** Defines, registers, and runs a three-step DAG pipeline that classifies an alert, enriches it from memory, then summarises it with an LLM.

**When to use it:** When you want automated ops triage that pulls in relevant runbook context before calling an LLM, minimising hallucination and token spend.

```bash
# 1. Define the pipeline
cat > /tmp/ops-pipeline.json <<'EOF'
{
  "id": "ops-triage-pipeline",
  "steps": [
    {
      "id": "classify",
      "kind": "local_ml",
      "model_id": "intent-classifier-v1",
      "prefer_local": true,
      "timeout_ms": 2000
    },
    {
      "id": "recall-context",
      "kind": "typescript_call",
      "adapter": "memory-recall",
      "depends_on": ["classify"]
    },
    {
      "id": "summarise",
      "kind": "llm",
      "prompt_template": "You are an ops assistant. Alert: {payload.text}. Related context: {steps.recall-context.output}. Provide a 2-sentence diagnosis and recommended action.",
      "max_tokens": 200,
      "depends_on": ["recall-context"],
      "prefer_local": false
    }
  ],
  "on_failure": "abort"
}
EOF

# 2. Register
curl -sf -X POST http://localhost:3000/api/pipelines \
  -H "Content-Type: application/json" \
  -d @/tmp/ops-pipeline.json

# 3. Run
curl -sf -X POST http://localhost:3000/api/pipelines/ops-triage-pipeline/run \
  -H "Content-Type: application/json" \
  -d '{"text":"CPU usage 98% on prod-api-03 for 10 minutes","host":"prod-api-03"}' | jq .
```

---

### Recipe 4 — Set up Slack alerts for budget and pipeline failures

**What it does:** Creates two Bell Tower watches — one for LLM token budget consumption exceeding 80% and one for any pipeline failure — both routing to Slack.

**When to use it:** When you want proactive Slack notifications before you exhaust your daily LLM budget or when a critical pipeline errors.

```bash
# Watch 1 — LLM budget alert
curl -sf -X POST http://localhost:3000/api/watches \
  -H "Content-Type: application/json" \
  -d '{
    "name": "LLM budget > 80%",
    "topic": "hammer.budget",
    "condition": "pct_used > 80",
    "channelIds": ["slack"],
    "cooldownMs": 3600000
  }' | jq '{id: .id, name: .name}'

# Watch 2 — critical pipeline failures
curl -sf -X POST http://localhost:3000/api/watches \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Critical pipeline failures",
    "topic": "pipeline.failed",
    "condition": "true",
    "channelIds": ["slack"],
    "cooldownMs": 300000
  }' | jq '{id: .id, name: .name}'

# Same via CLI
gristmill-cli watch create \
  --name "LLM budget > 80%" \
  --topic hammer.budget \
  --condition "pct_used > 80" \
  --channels slack \
  --cooldown 3600000
```

---

### Recipe 5 — Write a plugin (PagerDuty channel)

**What it does:** Implements a complete GristMill plugin that registers a `pagerduty` notification channel, maps severity from GristMill priority, and triggers PagerDuty incidents.

**When to use it:** When your on-call team uses PagerDuty and you want watches to page on-call engineers rather than (or in addition to) sending Slack messages.

```javascript
// ~/.gristmill/plugins/pagerduty.mjs
const PagerDutyPlugin = {
  name: "acme/pagerduty",
  version: "1.0.0",

  register(ctx) {
    ctx.registerChannel("pagerduty", {
      async send(notification) {
        const severity =
          notification.priority === "critical" ? "critical" : "warning";

        const body = {
          routing_key: process.env.PAGERDUTY_ROUTING_KEY,
          event_action: "trigger",
          dedup_key: notification.id ?? notification.title,
          payload: {
            summary: notification.title,
            severity,
            source: "gristmill",
            custom_details: { body: notification.body },
          },
        };

        const resp = await fetch(
          "https://events.pagerduty.com/v2/enqueue",
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
          }
        );

        if (!resp.ok) {
          ctx.log("error", `PagerDuty API error: ${resp.status}`);
        } else {
          ctx.log("info", `PagerDuty triggered: ${notification.title}`);
        }
      },
    });

    ctx.log("info", "PagerDuty channel registered");
  },
};

export default PagerDutyPlugin;
```

```bash
# Verify it loaded
curl -sf http://localhost:3000/api/plugins | jq '{plugins: .plugins, channels: .channels}'
# → { "plugins": ["acme/pagerduty"], "channels": ["slack", "email", "pagerduty"] }

# Create a watch that routes to PagerDuty
curl -sf -X POST http://localhost:3000/api/watches \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Critical outages → PagerDuty",
    "topic": "pipeline.failed",
    "condition": "true",
    "channelIds": ["pagerduty"],
    "cooldownMs": 60000
  }'
```

---

### Recipe 6 — Ingest from RabbitMQ

**What it does:** Connects the `MqAdapter` to a RabbitMQ queue and forwards transformed messages to the GristMill daemon, with an in-process testing mode that requires no broker.

**When to use it:** When events originate from an AMQP message queue and you want them triaged by GristMill without writing a custom consumer.

```typescript
import { MqAdapter } from "./src/hopper/mq-adapter.js";
import { IpcBridge } from "./src/core/ipc-bridge.js";

const bridge = new IpcBridge();

// ── Production: real RabbitMQ ─────────────────────────────────────────────
const mq = new MqAdapter({
  url: "amqp://user:pass@rabbitmq:5672",
  queue: "gristmill.events",
  bridge,
  transform: (msg) => {
    const payload = JSON.parse(msg.content.toString());
    return {
      channel: "mq",
      payload,
      priority: payload.urgent ? "high" : "normal",
      tags: { routingKey: msg.fields.routingKey },
    };
  },
});

await mq.start();
console.log("Consuming from RabbitMQ → gristmill.events");

// ── Testing: in-process (no RabbitMQ needed) ──────────────────────────────
const mqTest = new MqAdapter({ queue: "test", bridge });
await mqTest.start();

await mqTest.push({
  content: Buffer.from(JSON.stringify({ text: "process this order", orderId: 42 })),
  fields: { routingKey: "orders.new", exchange: "gristmill" },
  properties: {},
});
```

---

### Recipe 7 — Retrain the Sieve classifier (closed loop)

**What it does:** Runs the full weekly Sieve retrain loop — load feedback JSONL, train, export ONNX, validate, and hot-reload the daemon — as a cron-scheduled shell script.

**When to use it:** To automate the closed learning loop so the Sieve continuously improves from its own routing decisions without manual intervention.

```bash
#!/usr/bin/env bash
# ~/.gristmill/weekly-retrain.sh
set -euo pipefail

FEEDBACK_DIR="$HOME/.gristmill/feedback"
MODELS_DIR="$HOME/.gristmill/models"
LOG="$HOME/.gristmill/retrain.log"

echo "[$(date)] Starting weekly Sieve retrain" | tee -a "$LOG"

# Count feedback records available
RECORDS=$(find "$FEEDBACK_DIR" -name "*.jsonl" -exec wc -l {} + 2>/dev/null | tail -1 | awk '{print $1}')
echo "[$(date)] Feedback records: $RECORDS" | tee -a "$LOG"

if [ "${RECORDS:-0}" -lt 100 ]; then
  echo "[$(date)] Fewer than 100 records — skipping retrain (will use synthetic data)" | tee -a "$LOG"
fi

cd /path/to/gristmill-ml

# Full retrain pipeline: train → export ONNX → validate → hot-reload daemon
python -m gristmill_ml.pipelines.retrain_sieve \
  --feedback-dir "$FEEDBACK_DIR" \
  --output-dir "$MODELS_DIR" \
  --epochs 20 \
  --validate \
  --hot-reload \
  2>&1 | tee -a "$LOG"

echo "[$(date)] Retrain complete" | tee -a "$LOG"
```

```bash
# Add to crontab — runs every Sunday at 03:00
crontab -e
# 0 3 * * 0  ~/.gristmill/weekly-retrain.sh
```

```python
# Or run directly from Python for more control
from gristmill_ml.pipelines.retrain_sieve import RetrainPipeline

pipeline = RetrainPipeline(
    feedback_dir="~/.gristmill/feedback",
    output_dir="~/.gristmill/models",
    epochs=20,
    validate=True,
    hot_reload=True,   # calls daemon ModelsReload IPC after export
)
result = pipeline.run()
print(f"New model: {result.onnx_path}")
print(f"Validation accuracy: {result.val_accuracy:.3f}")
```

---

### Recipe 8 — Compare two model versions before deploying

**What it does:** Runs a head-to-head benchmark of a candidate ONNX model against the current baseline using `ModelComparison`, reporting accuracy delta, local inference rate, and estimated monthly cost saving.

**When to use it:** Before promoting a newly trained Sieve model to production, to verify it is strictly better and will not increase LLM escalation costs.

```python
from pathlib import Path
from gristmill_ml.experiments.comparisons import ModelComparison
from gristmill_ml.datasets.loaders import load_mmlu, load_gsm8k

# Load benchmark data (uses HuggingFace datasets)
mmlu_samples   = load_mmlu(n=200, split="validation")
gsm8k_samples  = load_gsm8k(n=100, split="test")

comparison = ModelComparison(
    baseline_path=Path("~/.gristmill/models/intent-classifier-v1.onnx"),
    candidate_path=Path("/tmp/intent-classifier-v2-candidate.onnx"),
)

report = comparison.run(mmlu_samples + gsm8k_samples)

print(f"Baseline accuracy:  {report.baseline.accuracy:.3f}")
print(f"Candidate accuracy: {report.candidate.accuracy:.3f}")
print(f"Accuracy delta:     {report.delta_accuracy:+.3f}")
print(f"Local rate (baseline):  {report.baseline.local_rate:.1%}")
print(f"Local rate (candidate): {report.candidate.local_rate:.1%}")
print(f"Estimated cost saving:  ${report.estimated_monthly_saving_usd:.2f}/month")

if report.candidate_is_better:
    print("Candidate is better — safe to deploy")
else:
    print("Candidate is worse — do not deploy")
```

---

### Recipe 9 — Use the dashboard to manage watches visually

**What it does:** Describes the seven-tab dashboard and walks through creating a watch from the UI without writing any curl commands.

**When to use it:** When you prefer a GUI for day-to-day operations, or when onboarding team members who are unfamiliar with the REST API.

**Opening the dashboard:**

```bash
# Mock mode (no daemon needed)
cd gristmill-integrations && GRISTMILL_MOCK_BRIDGE=1 pnpm dev
# → open http://localhost:3000
```

**Tab descriptions:**

- **Overview** — health status, uptime, budget bar, recent events count
- **Trainer** — training cycle state, validation scores, current version, promote/rollback controls
- **Pipelines** — list registered pipelines; click to see steps and run history
- **Memory** — semantic search box and a remember form (content + comma-separated tags)
- **Metrics** — Sieve cache hit-rate (exact + semantic) with progress bar; Hammer daily/monthly token usage; auto-refreshes every 10 seconds
- **Watches** — create/enable/disable/delete alert rules without writing curl
- **Ecosystem** — community adapter marketplace status; federated learning toggle

**Creating a watch from the UI (Watches tab):**

1. Click **+ New Watch**
2. Fill in: Name = "Disk alert", Topic = `pipeline.failed`, Condition = `true`, Channel IDs = `slack`, Cooldown = `300000`
3. Click **Create** — the watch appears in the list immediately
4. Toggle the switch to pause/resume a watch without deleting it

---

### Recipe 10 — Test your integration without a running daemon

**What it does:** Shows how to run the full TypeScript shell with an in-memory mock bridge, and how to write unit tests against a minimal `IBridge` mock without the Rust daemon.

**When to use it:** During CI, local plugin development, or when you want fast unit tests that do not require a compiled Rust binary.

```bash
# Start with mock bridge
cd gristmill-integrations
GRISTMILL_MOCK_BRIDGE=1 pnpm exec tsx src/main.ts
```

```typescript
// tests/my-feature.test.ts
import { describe, it, expect, vi } from "vitest";
import type { IBridge } from "../src/core/bridge.js";

// Create a minimal mock bridge for unit tests
function createMockBridge(): IBridge {
  return {
    async triage(_event) {
      return { route: "LOCAL_ML", confidence: 0.92 };
    },
    async remember(content, tags) {
      return "01MOCK_ID_" + content.slice(0, 6).replace(/\s/g, "_");
    },
    async recall(_query, _limit) {
      return [];
    },
    async getMemory(_id) { return null; },
    async escalate(_prompt, _maxTokens) {
      return {
        requestId: "mock-req",
        content: "Mock LLM response",
        provider: "mock",
        cacheHit: false,
        tokensUsed: 10,
        elapsedMs: 1,
      };
    },
    registerPipeline(_pipeline) {},
    async runPipeline(_id, _event) {
      return { runId: "mock-run", pipelineId: _id, succeeded: true, elapsedMs: 1, output: {} };
    },
    pipelineIds() { return []; },
    buildEventJson(channel, payload) {
      return JSON.stringify({ channel, payload });
    },
    subscribe(_topic) {
      return (async function* () {})();
    },
    ping() { return Promise.resolve(true); },
    close() {},
  };
}

describe("MyFeature", () => {
  it("routes and stores a memory on HIGH confidence local decision", async () => {
    const bridge = createMockBridge();
    vi.spyOn(bridge, "triage").mockResolvedValue({ route: "LOCAL_ML", confidence: 0.95 });
    vi.spyOn(bridge, "remember");

    const decision = await bridge.triage({ channel: "http", payload: { text: "hello" } });
    if (decision.confidence > 0.9) {
      await bridge.remember("routed locally with high confidence", ["auto"]);
    }

    expect(bridge.remember).toHaveBeenCalledOnce();
  });
});
```

---

### Recipe 11 — Check system health from the CLI

**What it does:** Uses `gristmill-cli doctor`, `status`, and `metrics` to get a complete snapshot of daemon health, model state, memory tier counts, and token usage.

**When to use it:** At the start of a shift, after a deployment, or when debugging unexpected routing decisions.

```bash
# Full doctor check
gristmill-cli doctor
# Output:
# ✓ Daemon socket   /Users/you/.gristmill/gristmill.sock (connected)
# ✓ Config          /Users/you/.gristmill/config.yaml (valid)
# ✓ Sieve model     intent-classifier-v1 (loaded, 392-dim, 4-class)
# ✓ LLM budget      125,000 / 500,000 tokens used today (25%)
# ✓ Memory          hot=1,240 entries  warm=18,432 entries  cold=102 files
# ✓ Feedback logs   3 files, 1,847 records (last: today)

# Quick status
gristmill-cli status
# → daemon: running  uptime: 3d 14h 22m  version: 0.1.0

# Metrics snapshot
gristmill-cli metrics
# → sieve cache hit-rate: 73.2%  exact: 8,421  semantic: 3,892  misses: 4,507
#   hammer daily: 125,000/500,000 tokens  monthly: 890,000/10,000,000
#   pipelines registered: 3
```

---

### Recipe 12 — Bootstrap models on a fresh install

**What it does:** Runs `bootstrap_models.py` to export a MiniLM embedder and train an intent classifier from synthetic data, producing the ONNX files the daemon needs on first start.

**When to use it:** On a new machine or fresh Docker container where no pre-trained models exist yet.

```bash
# Install the Python shell
cd gristmill-ml && pip install -e ".[dev]"

# Bootstrap: exports MiniLM embedder + trains intent classifier from synthetic data
# Takes ~3-5 min on CPU, ~30s on GPU
python scripts/bootstrap_models.py --classifier-epochs 10

# Output:
# [bootstrap] Exporting MiniLM embedder → ~/.gristmill/models/minilm-l6-v2.onnx
# [bootstrap] Training SieveClassifierHead on 500 synthetic records...
# [bootstrap] Epoch 10/10 — val_acc=0.847
# [bootstrap] Exporting classifier → ~/.gristmill/models/intent-classifier-v1.onnx
# [bootstrap] Written manifest: ~/.gristmill/models/bootstrap_manifest.json
# [bootstrap] Done. Start the daemon and it will load these models automatically.

# Skip if models already exist (idempotent)
python scripts/bootstrap_models.py  # → skips, prints "models already exist"

# Force re-export (e.g. after updating Python)
python scripts/bootstrap_models.py --force
```

---

### Recipe 13 — Use Slack as a Second Brain

**What it does:** Connects GristMill's three-tier memory to a Slack workspace so team members can save and recall knowledge with simple Slack commands — no API calls required.

**When to use it:** When you want your whole team to contribute to and query a shared runbook/knowledge base from within Slack, without switching context.

#### Prerequisites

1. Create a Slack app at [api.slack.com/apps](https://api.slack.com/apps) with Socket Mode enabled.
2. Add the following bot token scopes: `app_mentions:read`, `channels:history`, `chat:write`, `reactions:read`.
3. Install the app to your workspace and invite it to the channels you want monitored.

#### Config

```yaml
# config.yaml
integrations:
  slack:
    app_token: ${SLACK_APP_TOKEN}      # xapp-... — Socket Mode app token
    bot_token: ${SLACK_BOT_TOKEN}      # xoxb-... — bot OAuth token
    signing_secret: ${SLACK_SIGNING_SECRET}
    second_brain:
      enabled: true
      stale_days: 180                  # suppress memories older than N days in recall results
```

```bash
# Environment
export SLACK_APP_TOKEN="xapp-..."
export SLACK_BOT_TOKEN="xoxb-..."
export SLACK_SIGNING_SECRET="..."
```

#### Usage

```
# Save a memory
Alice: !save prod-db-01 runs PostgreSQL 15 with WAL archiving to S3
GristMill: ✅ Saved (id: 01HXYZ...)

# React with 📌 to save any message
Alice: [reacts 📌 to Bob's message "disk alert resolved by pruning WAL logs"]
GristMill: ✅ Saved Bob's message as a memory

# Recall
Alice: !ask how to fix disk space on postgres
GristMill: 🧠 Here's what I know:
           1. prod-db-01 disk alert resolved by pruning WAL logs (5 min ago, score 0.96)
           2. Disk alert threshold is 85%. Resolve by pruning WAL: pg_archivecleanup (2 days ago, score 0.91)
           3. prod-db-01 runs PostgreSQL 15 with WAL archiving to S3 (1 week ago, score 0.88)
```

#### How it works

- `!save <text>` and 📌 reactions call `bridge.remember()` — memories are written to the warm tier (SQLite + vector index) immediately and persist across restarts.
- `!ask <query>` calls `bridge.recall()` — returns up to 5 results ranked by semantic similarity + recency, filtered to the last `stale_days` days.
- The Slack user's display name is stored as the `author` tag on every memory.

---

### Recipe 14 — Run GristMill on Mac with GPU-accelerated Ollama

**What it does:** Uses the `docker-compose.mac.yml` overlay to connect the GristMill container to a natively running Ollama process on the Mac host, giving Ollama access to the Apple GPU via Metal — something Docker Desktop's Linux VM cannot provide.

**When to use it:** On Apple Silicon or Intel Macs when you want local LLM inference at native Metal speed without running an additional Docker container for Ollama.

```bash
# Step 1 — Install and start Ollama natively
brew install ollama
ollama serve                  # keep running in a separate terminal
ollama pull llama3.1:8b       # one-time download (~4 GB)

# Step 2 — Start GristMill with the Mac overlay
docker compose -f docker-compose.yml -f docker-compose.mac.yml up -d

# Step 3 — Verify Ollama is reachable from the container
docker compose exec gristmill curl -s http://host.docker.internal:11434/api/tags | jq '.models[].name'
# → "llama3.1:8b"

# Step 4 — Optional: add the trainer profile
docker compose -f docker-compose.yml -f docker-compose.mac.yml --profile trainer up -d

# Stop
docker compose -f docker-compose.yml -f docker-compose.mac.yml down
```

**What the overlay does:**
- Sets `GRISTMILL_HAMMER_OLLAMA_BASE_URL=http://host.docker.internal:11434` so the Rust `grist-hammer` talks to the host's Ollama process.
- Disables the `ollama` Docker service (tagged `profiles: [_disabled]`) so `--profile ollama` is silently ignored — no CPU-only container is started.
- Requires no change to `config.yaml` — the env var takes precedence.

**Training buffer note:** Ollama responses are stored in `gristmill-data/db/training_buffer.sqlite` (bind-mounted from the host). Ensure your `config.yaml` has:

```yaml
sieve:
  training_buffer_path: /data/gristmill/db/training_buffer.sqlite
```
