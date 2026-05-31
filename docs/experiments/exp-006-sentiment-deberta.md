# EXP-006: Sentiment Analysis — DeBERTa-v3-base Domain Classification

**Status**: Proposed
**Date**: 2026-05-31
**Authors**: GristMill Engineering Team

← [EXP-005](./exp-005-runbooks-rag.md) | [Back to Index](../lora-distillation-experiments.md) | [EXP-007 →](./exp-007-domain-embeddings.md)

---

## Abstract

Sentiment classification is a core triage signal in GristMill's sieve: negative-sentiment queries indicate user dissatisfaction and should be routed differently from neutral or positive queries. The current sieve uses a generic ONNX classifier with poor performance on technical sentiment (e.g., "disk at 90% capacity" reads as positive to general models; it is strongly negative in an infrastructure context). This experiment fine-tunes `microsoft/deberta-v3-base` — a 183M-parameter encoder-only model — with a classification head on domain-specific sentiment examples, targeting ≥ 95% accuracy at < 5ms CPU inference latency. We demonstrate that encoder-only classification models are architecturally superior to decoder-only LLMs for this task family: 16–40× lower latency, purpose-built classification logits, and state-of-the-art accuracy at one-third the parameter budget of the current Qwen2.5-0.5B student.

---

## 1. Introduction

GristMill's sieve (`grist-sieve`) classifies incoming queries to determine routing: local inference, escalation, or direct response. Sentiment is one of the primary routing signals. A query that is negative in sentiment (error reports, dissatisfied users, urgent alerts) is handled differently from one that is neutral (information request) or positive (confirmation, approval).

The existing generic classifier fails on technical language for a fundamental reason: sentiment in technical contexts is domain-specific. "Disk at 90%" is neutral to a general model (90 is a high number, often positive). In an infrastructure context, it is a warning state approaching a critical threshold. "Latency P99 above 500ms" is neutral grammatically; it is an incident trigger operationally.

Fine-tuning on domain-specific examples corrects this by teaching the model the semantic associations that matter in GristMill's deployment contexts.

---

## 2. Architecture Choice: Why Encoder-Only

Decoder-only LLMs (Qwen, Llama, Phi) are architecturally suboptimal for classification:

| Property | Decoder-only (Qwen2.5-0.5B) | Encoder-only (DeBERTa-v3-base) |
|----------|----------------------------|-------------------------------|
| Output | Token sequence | Classification logit vector |
| Inference | Autoregressive (N tokens generated) | Single forward pass |
| CPU latency | ~80 ms | ~5 ms |
| Params for this task | 500M (overkill) | 183M (right-sized) |
| SST-2 accuracy | ~88% (generative) | 96.0% (discriminative) |
| ONNX export | ✓ | ✓ |

The 16× latency advantage is the decisive factor for satisfying the `grist-sieve` p99 < 5ms triage target. An 80ms classification step would dominate the entire routing pipeline.

### 2.1 Why DeBERTa-v3-base Specifically

DeBERTa-v3 uses two innovations over standard BERT-family models:

1. **Disentangled attention**: Content and position encodings are represented and attended to separately, giving the model richer position-aware representations without increasing parameter count.

2. **ELECTRA-style pre-training**: Rather than masked token prediction (BERT), DeBERTa-v3 is pre-trained as a discriminator (real vs. replaced token), which provides a denser training signal — every token contributes to the loss at every step.

These translate to state-of-the-art performance on GLUE/SuperGLUE at the 183M parameter class, outperforming larger BERT and RoBERTa variants on most benchmarks.

### 2.2 Model Comparison

| Model | Params | SST-2 accuracy | CPU latency (ONNX INT8) | Choice |
|-------|--------|----------------|--------------------------|--------|
| `microsoft/deberta-v3-base` | 183M | 96.0% | ~5 ms | **Primary** |
| `distilroberta-base` | 82M | 93.1% | ~3 ms | Fallback (lower accuracy) |
| `distilbert-base-uncased` | 66M | 91.3% | ~2 ms | Ultra-low latency only |
| `microsoft/deberta-v3-large` | 434M | 97.2% | ~12 ms | Exceeds latency budget |

---

## 3. Experimental Setup

### 3.1 Training Approach

Unlike the LoRA distillation pipeline, sentiment fine-tuning trains a classification head with standard cross-entropy loss. This uses the standard HuggingFace `Trainer` with `AutoModelForSequenceClassification`, not `SFTTrainer` or PEFT.

```python
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

model = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/deberta-v3-base",
    num_labels=3,  # negative / neutral / positive
)
```

The full encoder is fine-tuned (no adapter — at 183M parameters, full fine-tuning on a classification head is fast and appropriate).

### 3.2 Training Data

Three layers, combined into a single training set:

#### Layer 1: General Foundation (regularization, ~20% of total)

Prevents catastrophic forgetting of general sentiment understanding. Sample from one or two of these:

| Dataset | HF ID | Size | Why it fits |
|---------|-------|------|------------|
| **SST-2** *(primary)* | `stanfordnlp/sst2` | 67K | The canonical sentiment benchmark. Binary (positive/negative). Use a 10K sample. |
| **DynaSent** | `dynabench/dynasentiment` | 121K | Adversarially collected — examples specifically designed to fool existing models. Directly improves robustness on edge cases. |
| **Twitter Financial News Sentiment** | `zeroshot/twitter-financial-news-sentiment` | 11K | Professional/technical register ("Q3 earnings declined 12%"). Much closer to infrastructure language than movie reviews. Use all 11K. |

```python
from datasets import load_dataset

sst2 = load_dataset("stanfordnlp/sst2", split="train").shuffle(seed=42).select(range(10000))
dynasent = load_dataset("dynabench/dynasentiment", split="train")
fin_news = load_dataset("zeroshot/twitter-financial-news-sentiment", split="train")
```

> **Why Twitter Financial News over IMDb/Amazon reviews**: Infrastructure alerts use professional, terse language ("P99 latency at 4.8ms", "disk at 90%") that is closer to financial news than consumer product reviews. The financial dataset also contains adversarial cases where large numbers are negative ("earnings missed by 12%").

#### Layer 2: Domain-Specific Labeled Examples (primary, ~55% of total)

No suitable public dataset covers infrastructure sentiment. Generate these from GristMill's own query logs and seed templates:

- Infrastructure alerts: "CPU at 95%", "memory pressure detected", "disk full" → `negative`
- Normal operational status: "service healthy", "deployment complete" → `positive`
- User queries: "how do I restart X", "what is the SLA for Y" → `neutral`
- Error reports: "connection refused", "timeout after 30s", "build failed" → `negative`

Target: ≥ 500 labeled examples per class (1,500 total minimum). Use the teacher LLM to generate variants of seed examples:

```python
SEED_EXAMPLES = [
    ("Disk at 90% capacity on primary node.", "negative"),
    ("Deployment completed in 4m 32s.", "positive"),
    ("What is the session timeout value?", "neutral"),
    ("OOMKilled: container exceeded 8Gi.", "negative"),
    ("CPU idle at 65%.", "positive"),
    ("P99 latency spiked to 2.3s.", "negative"),
    ("How do I check the service status?", "neutral"),
    ("Memory usage stable at 4.2 GB.", "positive"),
    ("Connection refused on port 5432.", "negative"),
]
# Use teacher LLM to generate 50+ variants of each seed example
```

#### Layer 3: Adversarial Examples (critical path, ~25% of total)

These target the specific failure modes of general-purpose sentiment models on technical language:

- "Latency improved to 200ms" → `neutral` (improvement, but absolute value is high)
- "Error rate dropped to 0.1%" → `positive`
- "99th percentile at 4.8ms" → `positive` (below 5ms target)
- "Disk at 90% capacity" → `negative` (general models see "90%" as high = good)
- "Service uptime: 99.2%" → `positive`
- "3 alerts fired in the last hour" → `negative`

Target: ≥ 150 adversarial examples. At least 15 of these should also appear in `probes/sentiment.yaml` as the probe set's adversarial subset.

**Combined training mix**:

| Source | Records | Fraction |
|--------|---------|----------|
| General (SST-2 + Financial News + DynaSent) | ~21K | 20% |
| Domain-specific labeled | 1,500 | 55% |
| Adversarial | 150 | 25% |

### 3.3 Probe Set

`probes/sentiment.yaml` — 50+ domain-specific examples with `correct_answer: positive|negative|neutral`.

```yaml
domain: sentiment
probes:
  - id: sent_disk_90
    question: "Disk usage is at 90% capacity on the primary database node."
    correct_answer: "negative"
    notes: "General models often score this as neutral or positive (high number)"

  - id: sent_deploy_complete
    question: "Deployment v2.3.1 completed successfully in 4m 32s."
    correct_answer: "positive"

  - id: sent_timeout_query
    question: "How long does the session timeout last?"
    correct_answer: "neutral"
    notes: "Informational query, not sentiment"

  - id: sent_latency_spike
    question: "P99 latency spiked to 2.3 seconds during the last deployment."
    correct_answer: "negative"

  - id: sent_memory_oom
    question: "OOMKilled: container exceeded 8Gi memory limit."
    correct_answer: "negative"
    notes: "Technical jargon that general models may not recognize as negative"
```

Include at least 15 adversarial examples that specifically target known failure modes of general-purpose sentiment models.

### 3.4 Hyperparameters

| Parameter | Value |
|-----------|-------|
| `model` | `microsoft/deberta-v3-base` |
| `num_labels` | 3 (negative, neutral, positive) |
| `learning_rate` | 2e-5 |
| `num_train_epochs` | 5 |
| `per_device_train_batch_size` | 16 |
| `weight_decay` | 0.01 |
| `warmup_ratio` | 0.1 |
| `evaluation_strategy` | epoch |
| `metric_for_best_model` | f1_macro |

### 3.5 Success Criteria

| Criterion | Target |
|-----------|--------|
| Held-out accuracy | ≥ 95% on domain examples |
| CPU inference latency (ONNX INT8) | < 5 ms per query |
| False-negative rate on negative-sentiment | < 3% |
| Adversarial accuracy (technical sentiment) | ≥ 90% |

The false-negative rate on negative-sentiment is the most important safety criterion: missing a dissatisfied or alarmed user is the costly routing error.

---

## 4. Implementation Steps

### 4.1 Data Collection

```bash
# Create the sentiment training data script
python scripts/generate_sentiment_data.py \
  --output data/sentiment_train.jsonl \
  --infrastructure-logs /path/to/logs/ \
  --augment-with-teacher
```

The script uses the teacher LLM to generate labeled variants of seed examples, expanding coverage of the domain vocabulary.

### 4.2 Training

```python
# scripts/train_sentiment.py
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
from datasets import load_dataset
import evaluate

tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
model = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/deberta-v3-base", num_labels=3
)

args = TrainingArguments(
    output_dir="checkpoints/deberta-sentiment",
    learning_rate=2e-5,
    num_train_epochs=5,
    per_device_train_batch_size=16,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
)

# Run training
trainer = Trainer(model=model, args=args, ...)
trainer.train()
trainer.save_model("checkpoints/deberta-sentiment")
```

### 4.3 ONNX Export and INT8 Quantization

```bash
# Export to ONNX
optimum-cli export onnx \
  --model ./checkpoints/deberta-sentiment \
  --task text-classification \
  gristmill-data/models/deberta-sentiment/

# Quantize to INT8
optimum-cli onnxruntime quantize \
  --onnx_model gristmill-data/models/deberta-sentiment/ \
  --output gristmill-data/models/deberta-sentiment-int8/ \
  --avx512

# Validate parity
python scripts/validate_onnx.py \
  --pytorch-model checkpoints/deberta-sentiment \
  --onnx-model gristmill-data/models/deberta-sentiment-int8/ \
  --test-data data/sentiment_eval.jsonl
```

### 4.4 Grinder Config

```yaml
grinders:
  models:
    sentiment-classifier:
      runtime: onnx
      path: gristmill-data/models/deberta-sentiment-int8.onnx
      warm: true
      task: text-classification
      labels: [negative, neutral, positive]
```

### 4.5 Closed Learning Loop Integration

The sentiment classifier should be wired into the sieve's closed learning loop:

```
Misclassified query → feedback JSONL
  → human review (10% sample)
  → confirmed label added to training set
  → next classification cycle (weekly)
```

At 183M parameters and a simple classification task, a full fine-tuning cycle takes minutes, not hours. Weekly retraining cadence is practical.

---

## 5. Results

*(To be filled in after experiment is run)*

### 5.1 Training Dynamics

| Metric | Value |
|--------|-------|
| Best eval F1 (macro) | — |
| Epoch at best checkpoint | — |
| Training time | — |

### 5.2 Evaluation

| Condition | Accuracy | F1 (macro) |
|-----------|----------|------------|
| General model (baseline) | — | — |
| DeBERTa-v3 (fine-tuned) | — | — |
| Adversarial subset | — | — |

### 5.3 Inference Benchmark

| Model | Latency (CPU, ONNX INT8) |
|-------|--------------------------|
| Qwen2.5-0.5B (current) | ~80 ms |
| DeBERTa-v3-base (fine-tuned) | — |

---

## 6. Discussion

*(To be filled in after experiment is run)*

---

## 7. Connection to the Multi-Model Grinder Architecture

This experiment produces one of the three "always warm" models in the target grinder configuration (see [EXP-008](./exp-008-multi-model-grinder.md)):

```yaml
grinders:
  models:
    sentiment-classifier:        # This experiment
      runtime: onnx
      path: gristmill-data/models/deberta-sentiment-int8.onnx
      warm: true
    embedder:                    # EXP-007
      runtime: onnx
      path: gristmill-data/models/nomic-embed-int8.onnx
      warm: true
    runbook-qa:                  # EXP-005
      runtime: onnx
      path: gristmill-data/models/qwen-1.5b-runbooks-int8.onnx
      warm: false
```

The sentiment classifier feeds the routing decision that determines whether a query goes to the `runbook-qa` model, the `reasoning` model, or directly to the LLM escalation path.

---

← [EXP-005](./exp-005-runbooks-rag.md) | [Back to Index](../lora-distillation-experiments.md) | [EXP-007 →](./exp-007-domain-embeddings.md)
