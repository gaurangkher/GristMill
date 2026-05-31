# EXP-008: Multi-Model Grinder Architecture — Validating the Local-First Moat

**Status**: Proposed (depends on EXP-005, EXP-006, EXP-007)
**Date**: 2026-05-31
**Authors**: GristMill Engineering Team

← [EXP-007](./exp-007-domain-embeddings.md) | [Back to Index](../lora-distillation-experiments.md)

---

## Abstract

EXP-005 through EXP-007 each train a purpose-built small model for a narrow domain task. This experiment integrates all three into a unified multi-model grinder configuration and measures the system-level metrics that define GristMill's local-first moat: cloud LLM escalation rate, average query latency, and per-query cost across a representative mixed workload. The central claim being validated: a suite of purpose-built small models (<2B total parameters, ~1.2 GB ONNX INT8) simultaneously outperforms a general-purpose cloud LLM on accuracy, latency, and cost for the domain-specific query distribution that GristMill targets. Success is defined as escalation rate < 20% and average local latency < 200ms on a mixed workload of runbook, sentiment, and embedding queries.

---

## 1. Introduction

GristMill's architectural moat is not that small models are acceptable substitutes for large ones. The claim is stronger: for the narrow, structured query types that appear in production deployments (runbook lookups, sentiment classification, similarity search), purpose-built small models are superior — not merely cheaper. They are faster, more accurate on domain-specific examples, and do not require internet access or API rate limits.

EXP-001 through EXP-003 tested a general-purpose reasoning model and found it inadequate at 0.5B–3B scale. The failure was not a refutation of the local-first thesis — it was a demonstration that the wrong model architecture was applied to the task. General reasoning on arbitrary arithmetic problems is not a domain-specific task; it requires broad generalization that small models cannot provide.

Runbook Q&A, sentiment classification, and similarity search are domain-specific. They can be solved by purpose-built models with high accuracy at low latency. This experiment validates that claim at the system level.

---

## 2. Architecture

### 2.1 Target Grinder Configuration

```yaml
grinders:
  models:
    # Triage and routing — always warm, < 5ms per query
    sentiment-classifier:
      runtime: onnx
      path: gristmill-data/models/deberta-sentiment-int8.onnx
      warm: true
      task: text-classification
      labels: [negative, neutral, positive]

    # Semantic retrieval — always warm, < 10ms per embedding
    embedder:
      runtime: onnx
      path: gristmill-data/models/nomic-embed-int8.onnx
      warm: true
      embedding_dim: 768
      max_seq_len: 8192

    # Runbook Q&A — loaded on demand, < 200ms with retrieved context
    runbook-qa:
      runtime: onnx
      path: gristmill-data/models/qwen-1.5b-runbooks-int8.onnx
      warm: false
      domain: runbooks

    # General reasoning — loaded on demand, < 500ms
    reasoning:
      runtime: onnx
      path: gristmill-data/models/qwen-1.5b-reasoning-int8.onnx
      warm: false
      domain: reasoning
```

### 2.2 Routing Logic

The sieve classifies each incoming query and routes it to the appropriate grinder:

```
Incoming query
  → sentiment-classifier (always warm, < 5ms)
  │   ├─ negative + high urgency → escalate to LLM (incident response)
  │   └─ neutral / positive → continue routing
  │
  → sieve domain classifier (ONNX, < 5ms)
  │   ├─ domain: runbooks → embedder (< 10ms) → runbook-qa (< 200ms)
  │   ├─ domain: reasoning → reasoning model (< 500ms)
  │   ├─ domain: semantic-search → embedder only
  │   └─ domain: unknown, confidence < 0.85 → escalate to LLM
  │
  → result or escalation
```

### 2.3 Combined Footprint

| Model | Params | ONNX INT8 size | Warm |
|-------|--------|----------------|------|
| deberta-sentiment | 183M | ~92 MB | Yes |
| nomic-embed | 137M | ~69 MB | Yes |
| qwen-1.5b-runbooks | 1.5B | ~750 MB | No |
| qwen-1.5b-reasoning | 1.5B | ~750 MB | No |
| **Total** | **~3.3B** | **~1.66 GB** | — |

**Memory profile on 8 GB Apple Silicon**: warm models (160 MB) always in unified memory; on-demand models loaded when needed, unloaded after 60s idle. Peak: ~2 GB when both on-demand models are simultaneously loaded. Comfortably within 8 GB budget.

---

## 3. Experimental Setup

### 3.1 Prerequisites

This experiment requires completion of:
- [EXP-005](./exp-005-runbooks-rag.md): runbook-qa adapter trained and exported to ONNX
- [EXP-006](./exp-006-sentiment-deberta.md): deberta-sentiment trained and exported to ONNX
- [EXP-007](./exp-007-domain-embeddings.md): nomic-embed domain fine-tuned and exported to ONNX

### 3.2 Evaluation Workload

A representative mixed workload of 200 queries:

| Type | Count | Domain | Expected routing |
|------|-------|--------|-----------------|
| Runbook lookup | 60 | runbooks | Local: runbook-qa |
| Sentiment classification | 50 | sieve | Local: sentiment-classifier |
| Similarity search | 40 | ledger | Local: embedder |
| General reasoning | 30 | reasoning | Local: reasoning model |
| Truly novel / OOD | 20 | unknown | Escalate to LLM |

The 20 OOD queries (10% of workload) establish the escalation floor — the system should escalate these and correctly handle the remaining 90% locally.

### 3.3 Comparison Baseline

**Baseline**: All 200 queries routed to Anthropic claude-sonnet-4 (current production escalation target). Cost, latency, and accuracy measured over the same 200 queries.

**Target**: Multi-model grinder suite. Escalation reserved for the 20 OOD queries plus any query where local model confidence < 0.85.

### 3.4 Metrics

| Metric | Baseline (cloud only) | Target (grinder suite) |
|--------|----------------------|------------------------|
| Cloud LLM escalation rate | 100% | < 20% |
| Average query latency | 800–2000ms | < 200ms (local) / ~1000ms (escalated) |
| Cost per 1,000 queries | ~$1.50–5.00 | ~$0.20–0.50 |
| Runbook Q&A accuracy | ~60% (no fine-tuning, no RAG) | ≥ 75% |
| Sentiment accuracy (technical) | ~70% (general model) | ≥ 95% |
| Retrieval NDCG@10 | — | ≥ 0.80 |

### 3.5 Success Criteria

| Criterion | Required |
|-----------|----------|
| Escalation rate < 20% on mixed workload | Yes |
| Local avg latency < 200ms for runbook/sentiment/embedding queries | Yes |
| Runbook Q&A accuracy ≥ 75% (with RAG) | Yes |
| Sentiment accuracy ≥ 95% (domain examples) | Yes |
| No incorrect confident answers (escalate when uncertain) | Yes |
| Total ONNX INT8 footprint < 2 GB | Yes |

---

## 4. Implementation Steps

### 4.1 Export All Models to ONNX INT8

```bash
# Export each model (after EXP-005, EXP-006, EXP-007 complete)
python scripts/export_onnx.py --domain runbooks --quantize int8
python scripts/export_onnx.py --domain reasoning --quantize int8
python scripts/export_onnx.py --model deberta-sentiment --quantize int8
python scripts/export_onnx.py --model nomic-embed --quantize int8
```

### 4.2 Deploy Grinder Config

```bash
# Update grinder config (not committed — contains local paths)
nano ~/.gristmill/config.yaml
# Add grinder config from Section 2.1

# Reload Rust daemon grinder config (no restart required)
gristmill-ctl reload-grinders
```

### 4.3 Run Mixed Workload Evaluation

```bash
# Run 200-query mixed workload against grinder suite
python scripts/evaluate_grinder_suite.py \
  --workload data/mixed_workload_200.jsonl \
  --output docs/experiments/results/exp-008-results.jsonl

# Run same workload against LLM baseline
python scripts/evaluate_grinder_suite.py \
  --workload data/mixed_workload_200.jsonl \
  --force-escalate \
  --output docs/experiments/results/exp-008-baseline.jsonl

# Compare
python scripts/compare_grinder_results.py \
  --grinder docs/experiments/results/exp-008-results.jsonl \
  --baseline docs/experiments/results/exp-008-baseline.jsonl
```

### 4.4 Create `scripts/evaluate_grinder_suite.py`

This script must be created as part of EXP-008. It:
1. Reads queries from the workload JSONL
2. Submits each to GristMill's `GristMillBridge` (via the PyO3 bridge, not directly to Python models)
3. Records: routing decision, latency, model used, answer, confidence score
4. Computes per-domain accuracy against ground-truth labels in the workload
5. Computes aggregate metrics (escalation rate, avg latency, cost estimate)

---

## 5. Results

*(To be filled in after experiment is run)*

### 5.1 Escalation Rate

| Query Type | Escalated | Local |
|------------|-----------|-------|
| Runbook | — | — |
| Sentiment | — | — |
| Embedding | — | — |
| Reasoning | — | — |
| OOD | — | — |
| **Overall** | **—** | **—** |

### 5.2 Latency Distribution

| Model | p50 | p95 | p99 |
|-------|-----|-----|-----|
| sentiment-classifier | — | — | — |
| embedder | — | — | — |
| runbook-qa | — | — | — |
| reasoning | — | — | — |
| LLM escalation | — | — | — |

### 5.3 Cost Comparison

| Scenario | Cost per 1,000 queries |
|----------|----------------------|
| Cloud-only baseline | — |
| Grinder suite (EXP-008) | — |
| Savings | — |

### 5.4 Accuracy by Domain

| Domain | Baseline (cloud LLM) | Grinder suite | Delta |
|--------|---------------------|---------------|-------|
| Runbook Q&A | — | — | — |
| Sentiment | — | — | — |
| General reasoning | — | — | — |

---

## 6. Discussion

*(To be filled in after experiment is run)*

---

## 7. The Moat Argument

This experiment's results, when complete, provide the quantitative evidence for GristMill's central architectural claim. The table from Section 3.4, filled in, is the moat:

> **Local-first is not a cost constraint — it is architecturally superior in accuracy, latency, and cost simultaneously for the domain-specific query types that GristMill targets.**

The claim is not that small models match large ones on arbitrary tasks. It is that for the structured, domain-specific query distribution of a production deployment, purpose-built models at 183M–1.5B parameters each outperform a 70B+ cloud LLM on every metric that matters operationally:

- **Accuracy**: domain fine-tuning + RAG beats generic in-context prompting on runbooks
- **Latency**: 5ms–200ms local vs. 800ms–2000ms cloud API
- **Cost**: 90% reduction in cloud API calls at 1,000+ queries/day
- **Privacy**: sensitive runbook content never leaves the local machine
- **Reliability**: no API rate limits, no network dependency, no outage risk

---

## 8. Connection to GristMill Architecture

This experiment validates the full closed-loop system described in the [architecture document](../../gristmill-v2-architecture.md):

```
TypeScript Shell
  → GristMillBridge.submit(event)
  → grist-sieve (Rust, < 5ms) — classify + route
  → grist-grinders (Rust, ONNX pool) — local inference
  → grist-ledger (Rust) — retrieve context + store results
  → Feedback JSONL → Python retrains → new ONNX hot-reloaded
  → Next cycle: smarter routing, better local models
```

Each cycle through the loop improves routing accuracy (sieve), retrieval precision (embedder), and generation quality (runbook-qa, reasoning). The closed loop is what makes GristMill's moat self-reinforcing: the more queries it handles locally, the more training data it accumulates, the better its local models become.

---

← [EXP-007](./exp-007-domain-embeddings.md) | [Back to Index](../lora-distillation-experiments.md)
