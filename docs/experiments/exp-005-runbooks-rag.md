# EXP-005: Company Runbooks — Retrieval-Augmented Generative Q&A

**Status**: Proposed
**Date**: 2026-05-31
**Authors**: GristMill Engineering Team

← [EXP-004](./exp-004-student-model-upgrade.md) | [Back to Index](../lora-distillation-experiments.md) | [EXP-006 →](./exp-006-sentiment-deberta.md)

---

## Abstract

Company runbooks represent the highest-value local inference use case in GristMill: queries are frequent, structurally predictable, and expensive when escalated to a cloud LLM. This experiment fine-tunes Qwen2.5-1.5B-Instruct on `(runbook chunk, question) → answer` triples and couples it with a usearch similarity retrieval step to produce a Retrieval-Augmented Generation (RAG) architecture. The central hypothesis is that fine-tuning teaches context-grounded reading comprehension rather than weight-based memorization, making the system robust to runbook updates. We expect factual accuracy ≥ 0.75 on a runbook probe set with retrieved context, compared to ≤ 0.20 without context — demonstrating that neither retrieval alone nor fine-tuning alone matches the combined system. This experiment directly validates GristMill's local-first architectural moat: a 1.5B model with domain fine-tuning and RAG outperforms a 70B+ cloud LLM on cost, latency, and accuracy for this task family.

---

## 1. Introduction

The failure mode in EXP-001–EXP-003 was distribution mismatch: the training data (arithmetic reasoning in a stereotyped step-by-step format) was out-of-distribution relative to both the model's pre-training and the evaluation probes. Every factor that caused forgetting was a consequence of this mismatch.

Runbook Q&A eliminates this mismatch by design:
- Training and evaluation queries come from the same corpus (real runbook content)
- The model is not asked to recall facts from weights — it reads them from retrieved context
- The training objective (given this context, answer this question) is directly aligned with the inference objective
- Runbook language (procedure names, flags, thresholds) is a narrow, consistent vocabulary that a 1.5B model can learn to parse effectively

This is the use case where the GristMill architecture has its strongest moat: frequent, predictable, context-grounded queries that can be served locally at <200ms latency vs. 800–2000ms for a cloud API call.

---

## 2. Architecture

### 2.1 End-to-End Flow

```
User query
  → nomic-embed-text-v1 (encode query)
  → usearch similarity search (warm-tier ledger, 8,192-token chunks)
  → top-k runbook chunks retrieved
  → prompt construction:
      [RUNBOOK CONTEXT]
      {chunk_1}
      ---
      {chunk_2}

      Question: {user_query}
      Answer:
  → Qwen2.5-1.5B-Instruct (runbook-adapted LoRA)
  → grounded answer
```

The fine-tuned model learns to extract the relevant sentence or value from the provided runbook excerpt. It does not need to memorize runbook content in its weights — the retrieval step provides the relevant text at inference time.

### 2.2 Why Fine-Tuning Adds Value Over Base Model + RAG

A base model given runbook context can answer grounded questions, but it has not been trained to:
1. Recognize the specific format of runbook content (section headers, command blocks, tables)
2. Prefer the context's answer over its own pre-trained associations
3. Format answers concisely (a critical path command, not a paragraph)

Fine-tuning on `(context, question, answer)` triples teaches all three, at the cost of one training cycle.

### 2.3 Why RAG Adds Value Over Fine-Tuning Alone

Runbook content changes: version numbers increment, team names change, thresholds are updated. Weight-based memorization cannot accommodate updates without retraining. Retrieval from the ledger warm tier always reflects the latest ingested runbook version.

**The moat**: neither retrieval alone (base model + context) nor fine-tuning alone (adapted model, no context) achieves the accuracy of the combined system.

---

## 3. Experimental Setup

### 3.1 Student Model

**Qwen/Qwen2.5-1.5B-Instruct** — same as EXP-004. The runbook adapter is trained separately from the reasoning adapter (independent domain in the training buffer).

If EXP-004 is not yet complete, the base Qwen2.5-1.5B-Instruct can be used directly as the starting point for runbook fine-tuning.

### 3.2 Training Data

Training data comes from two layers: a **public foundation** that teaches context-grounded reading comprehension as a transferable skill, and **domain-specific runbook pairs** that teach the task itself.

#### Layer 1: Public Foundation Datasets

These datasets are loaded from HuggingFace and seeded into the training buffer under `domain_tag: runbooks_foundation` before any company-specific runbook data. They teach the model to extract a short answer from a provided passage — the core skill needed for runbook Q&A.

| Dataset | HF ID | Sample size | Why it fits |
|---------|-------|------------|------------|
| **SQuAD 2.0** *(primary)* | `rajpurkar/squad_v2` | 20K | Teaches "extract the answer to this question from this passage." Exact pattern as runbook Q&A. Includes unanswerable questions — trains the model to say "not found in the context" rather than hallucinate. |
| **StackExchange (ServerFault / Unix)** | `HuggingFaceH4/stack-exchange-preferences` | 5K (filtered to ops tags) | Real-world operational Q&A: "How do I restart nginx after a config change?" with accepted answers. Closest open-domain proxy to runbook queries. Filter to tags: `server`, `linux`, `bash`, `networking`, `deployment`. |
| **TechQA** | `ibm/tech_qa` | full (1,400) | IBM technical documentation Q&A. Small but high quality; documents are structured like runbooks (numbered procedures, code blocks, thresholds). |
| **wikiHow** *(proxy, no real runbooks yet)* | `wikihow/all` | 5K (how-to only) | 230K step-by-step procedure articles. Same format as runbooks: numbered steps, imperative commands, context-specific actions. Use as proxy until real runbooks are available. |

```python
from datasets import load_dataset

# SQuAD 2.0 — context-grounded extraction
squad = load_dataset("rajpurkar/squad_v2", split="train")
squad_sample = squad.shuffle(seed=42).select(range(20000))

# StackExchange — filter to ops-related posts
se = load_dataset("HuggingFaceH4/stack-exchange-preferences", split="train")
ops_tags = {"server", "linux", "bash", "networking", "deployment", "docker", "nginx"}
se_ops = se.filter(lambda x: any(t in ops_tags for t in x.get("tags", [])))
se_sample = se_ops.shuffle(seed=42).select(range(min(5000, len(se_ops))))

# TechQA — use all (small dataset)
techqa = load_dataset("ibm/tech_qa", split="train")
```

#### Layer 2: Domain-Specific Runbook Pairs

Training records are generated by `scripts/seed_runbooks.py` (to be created). The seeding script:
1. Reads Markdown / PDF runbooks from a specified directory
2. Chunks content into 512-token windows with 64-token overlap
3. For each chunk, uses the teacher LLM to generate 3–5 Q&A pairs grounded in that chunk
4. Writes records to the training buffer with `domain_tag: runbooks`

**Training record format** (as stored in SQLite training buffer):
```json
{
  "query_text": "[RUNBOOK CONTEXT]\n{chunk_text}\n\nQuestion: {question}",
  "teacher_response": "{answer}",
  "domain_tag": "runbooks",
  "confidence_score": 0.95
}
```

Unlike EXP-001–EXP-003, the teacher response is a ground-truth answer extracted from the runbook text — not an LLM generation. This eliminates the teacher hallucination problem and makes the training signal clean.

**Training mix** (recommended ratio):

| Source | Records | Fraction | Purpose |
|--------|---------|----------|---------|
| SQuAD 2.0 | 20K | 57% | Foundation: extraction skill |
| StackExchange ops | 5K | 14% | Bridge: technical register |
| TechQA | 1.4K | 4% | Bridge: structured docs |
| Company runbooks | 9K+ | 25% | Primary domain signal |

### 3.3 Probe Set

`probes/runbooks.yaml` — to be created alongside `seed_runbooks.py`. Example probe:

```yaml
domain: runbooks
probes:
  - id: rb_restart_service
    tags: [runbooks, procedures]
    question: "What is the correct command to restart the ingestion service after a schema migration?"
    context: "runbook-ingestion-v3.md §4.2"
    expected: "Run `make restart-ingestion` from the repo root after applying migrations."
    correct_answer: "make restart-ingestion"

  - id: rb_oncall_sla
    tags: [runbooks, sla]
    question: "What is the SLA for P1 incident acknowledgement?"
    context: "runbook-oncall.md §2.1"
    expected: "P1 incidents must be acknowledged within 5 minutes of PagerDuty alert."
    correct_answer: "5 minutes"

  - id: rb_rollback_command
    tags: [runbooks, deployment]
    question: "How do you roll back a failed deployment in the staging environment?"
    context: "runbook-deploy.md §3.1"
    expected: "Run `./scripts/rollback.sh --env staging --version <previous-tag>`"
    correct_answer: "rollback.sh"
```

Probe set should contain at least 20 probes covering: procedures, commands, thresholds, team contacts, SLAs, and error codes.

### 3.4 Evaluation Conditions

Three conditions must be evaluated to demonstrate the moat:

| Condition | Retrieval | Fine-tuning | Expected Accuracy |
|-----------|-----------|-------------|-------------------|
| Baseline: no context, no adapter | ✗ | ✗ | ≤ 0.20 |
| Retrieval only: context, no adapter | ✓ | ✗ | ~0.50 |
| Fine-tuning only: no context, with adapter | ✗ | ✓ | ~0.35 |
| **Full system**: context + adapter | ✓ | ✓ | **≥ 0.75** |

The full system must outperform both retrieval-only and fine-tuning-only to validate the combined architecture.

### 3.5 Hyperparameters

| Parameter | Value |
|-----------|-------|
| `base_model` | `Qwen/Qwen2.5-1.5B-Instruct` |
| `lora_rank` | 8 (narrower task → lower rank sufficient) |
| `lora_alpha` | 16 |
| `lora_target_modules` | q_proj, v_proj |
| `learning_rate` | 2e-4 |
| `num_epochs` | 3 |
| `max_length` | 1024 (accommodate runbook context) |
| `domain_tag` | `runbooks` |

Rationale: the training and evaluation distributions are aligned (same runbook content in both), so a higher learning rate and more epochs are appropriate. The risk of forgetting general world knowledge is acceptable — the model is serving runbook queries, not general reasoning.

### 3.6 Config Changes

```yaml
# In ~/.gristmill/config.yaml — add runbooks domain config
trainer:
  domains:
    runbooks:
      base_model: Qwen/Qwen2.5-1.5B-Instruct
      lora_rank: 8
      lora_alpha: 16
      lora_target_modules: q_proj,v_proj
      learning_rate: 2e-4
      num_epochs: 3
      validation:
        strategy: factual_accuracy
        probe_set: runbooks
        min_accuracy: 0.75
```

---

## 4. Implementation Steps

### 4.1 Create Seeding Script

Create `gristmill-ml/scripts/seed_runbooks.py`:
```python
"""
Reads Markdown/PDF runbooks from a directory, chunks them, generates Q&A pairs
via teacher LLM, and writes to the training buffer under domain_tag: runbooks.
"""
```

Key design decisions:
- Chunk overlap (64 tokens) prevents answers from being split across chunks
- Teacher LLM generates questions grounded in the chunk, not general knowledge
- Each chunk produces 3–5 Q&A pairs (enough signal from each runbook section)
- `confidence_score: 0.95` (ground-truth extraction, not LLM inference)

### 4.2 Create Probe Set

Manually curate `gristmill-ml/probes/runbooks.yaml` with ≥ 20 probes from real runbook content. Include:
- Command probes (exact command match)
- Threshold probes (exact number match)
- Team/contact probes (exact name/URL match)
- Procedure probes (key step match)

### 4.3 Run Training and Evaluation

```bash
# Seed runbook training data
python scripts/seed_runbooks.py --runbooks-dir /path/to/runbooks/ --domain runbooks

# Evaluate baseline (no context, no adapter)
python scripts/compare_lora_adapter.py --probe-set runbooks --no-context

# Train runbook adapter
python -m gristmill_ml.trainer.service --domain runbooks

# Evaluate full system (with retrieved context)
python scripts/compare_lora_adapter.py --probe-set runbooks --with-retrieval
```

---

## 5. Results

*(To be filled in after experiment is run)*

### 5.1 Training Dynamics

| Metric | Value |
|--------|-------|
| Training records | — |
| Final training loss | — |
| Wall-clock duration | — |

### 5.2 Accuracy by Condition

| Condition | Accuracy |
|-----------|----------|
| No context, no adapter | — |
| Context only (no adapter) | — |
| Adapter only (no context) | — |
| Context + adapter (full system) | — |

### 5.3 Promotion Decision

| Criterion | Result |
|-----------|--------|
| FactualAccuracyRunner score | — |
| Promoted | — |

---

## 6. Discussion

*(To be filled in after experiment is run)*

---

## 7. Integration with GristMill Ledger

Once the runbook adapter is validated, integration with the Rust daemon requires:

1. **ONNX export** of the Qwen2.5-1.5B runbook adapter:
```bash
python scripts/export_onnx.py --domain runbooks --quantize int8
```

2. **Grinder config** in `~/.gristmill/config.yaml`:
```yaml
grinders:
  models:
    runbook-qa:
      runtime: onnx
      path: gristmill-data/models/qwen-1.5b-runbooks-int8.onnx
      warm: false
      domain: runbooks
```

3. **Retrieval integration**: `grist-ledger` must be configured to provide `top_k: 3` runbook chunks as context when routing `domain: runbooks` queries to the grinder.

---

← [EXP-004](./exp-004-student-model-upgrade.md) | [Back to Index](../lora-distillation-experiments.md) | [EXP-006 →](./exp-006-sentiment-deberta.md)
