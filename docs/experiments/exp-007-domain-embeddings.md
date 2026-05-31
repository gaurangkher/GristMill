# EXP-007: Domain Embeddings — Bi-Encoder Fine-Tuning with nomic-embed-text-v1

**Status**: Proposed
**Date**: 2026-05-31
**Authors**: GristMill Engineering Team

← [EXP-006](./exp-006-sentiment-deberta.md) | [Back to Index](../lora-distillation-experiments.md) | [EXP-008 →](./exp-008-multi-model-grinder.md)

---

## Abstract

GristMill's warm-tier ledger uses usearch to perform approximate nearest-neighbour search over stored embeddings. The quality of these embeddings determines whether the correct runbook chunk is retrieved for a given query — directly affecting the factual accuracy of the RAG pipeline in EXP-005. A generic embedding model performs poorly on technical jargon (incident codes, CLI commands, configuration keys) that does not appear in its pre-training distribution. This experiment fine-tunes `nomic-ai/nomic-embed-text-v1` — a 137M-parameter bi-encoder with an 8,192-token context window — on domain-specific `(query, relevant_chunk)` pairs using contrastive loss, targeting ≥ 0.80 NDCG@10 on a runbook retrieval probe set. We additionally benchmark the model's impact on end-to-end runbook Q&A accuracy when used as the retriever in EXP-005's RAG architecture.

---

## 1. Introduction

The EXP-005 RAG architecture depends on a retrieval step: given a user query, the system must identify the most relevant runbook chunks from the warm-tier ledger. Retrieval quality (measured by NDCG@10) is the primary determinant of whether the correct context is available when the generative model produces its answer.

Generic embedding models are trained on open-domain text corpora (Common Crawl, Wikipedia, code repositories). They learn general semantic similarity but are unfamiliar with domain-specific vocabulary:
- "Graceful degradation mode" → not the same as "degraded mode" or "maintenance mode"
- "Runbook §4.2" → not linked to "restart procedure after schema migration"
- CLI flags like `--env staging` → treated as noise, not semantic content

Domain fine-tuning teaches the embedder which concepts are similar in the deployment context, improving retrieval precision without increasing model size.

---

## 2. Architecture Choice: Why a Dedicated Bi-Encoder

### 2.1 Why Not Use the Generative Student as Embedder

Decoder-only LLMs (Qwen2.5-1.5B) produce anisotropic embeddings: representations cluster in a narrow cone of the vector space, causing most cosine similarity scores to be artificially high. This reduces the discriminative power of nearest-neighbour search — most documents appear similarly relevant, making it impossible to identify the top-k most relevant chunks reliably.

Bi-encoder models are trained with contrastive objectives that explicitly spread representations across the full vector space. Positive pairs are pulled together; negative pairs are pushed apart. This produces embeddings where cosine similarity is a reliable proxy for semantic relevance.

### 2.2 Why nomic-embed-text-v1 Specifically

| Model | Params | Max tokens | MTEB avg | License | Choice |
|-------|--------|-----------|----------|---------|--------|
| `nomic-ai/nomic-embed-text-v1` | 137M | **8,192** | 62.4 | Apache 2.0 | **Primary** |
| `BAAI/bge-base-en-v1.5` | 109M | 512 | 64.2 | MIT | Fallback |
| `BAAI/bge-small-en-v1.5` | 33M | 512 | 62.2 | MIT | Ultra-low latency |
| `mixedbread-ai/mxbai-embed-large-v1` | 335M | 512 | 64.7 | Apache 2.0 | Highest quality |

The **8,192-token context window** is the decisive factor for runbook retrieval. Runbook sections, incident post-mortems, and architecture decision records frequently exceed the 512-token limit of standard embedding models. A chunk that spans 800 tokens would be truncated at the embedding stage with BGE or DistilBERT, losing the second half of its content — precisely the part containing the procedure steps or threshold values that make a chunk relevant to a query.

NDCG@10 without domain fine-tuning: estimated 0.55–0.65 on runbook retrieval. MTEB average of 62.4 is competitive with BGE-base at 64.2, while the 8,192-token window provides substantial structural advantage for this specific use case.

---

## 3. Experimental Setup

### 3.1 Base Model

**`nomic-ai/nomic-embed-text-v1`** — 137M parameters, 8,192-token context, 768-dimensional embeddings. Apache 2.0 licensed. ONNX export validated upstream.

### 3.2 Training Data

Bi-encoder training needs `(query, positive_passage)` pairs. The quality of negatives matters as much as the positives — hard negatives (plausibly relevant but actually wrong) produce sharper discrimination than random negatives. Training proceeds in two stages: a general retrieval foundation, then domain fine-tuning.

#### Stage 1: General Retrieval Foundation

Load from HuggingFace and train the base retrieval capability before domain fine-tuning:

| Dataset | HF ID | Sample size | Why it fits |
|---------|-------|------------|------------|
| **MS MARCO** *(primary)* | `microsoft/ms_marco` (`v2.1`) | 100K pairs | The standard for retrieval model training. 8.8M passages with BM25-retrieved candidates and binary relevance labels. Use `v2.1` (passage ranking task). |
| **Natural Questions** | `google-research-datasets/natural_questions` | 50K | Clean (question, Wikipedia paragraph) pairs with exact answer spans. Trains short factual extraction — same skill as "find the threshold value in this runbook section." |
| **GooAQ** | `allenai/gooaq` | 30K | Google autocomplete questions paired with featured snippet answers. Short, direct Q→A pairs that match the runbook answer format. |
| **StackExchange** (ops) | `HuggingFaceH4/stack-exchange-preferences` | 5K | Technical Q&A from ServerFault/Unix.SE. Bridges general retrieval to technical vocabulary before domain fine-tuning. |

```python
from datasets import load_dataset

# MS MARCO — standard passage retrieval pairs
marco = load_dataset("microsoft/ms_marco", "v2.1", split="train")
# Each row has: query, passages (list), answers
# Filter to rows with a positive passage
marco_pairs = marco.filter(lambda x: any(x["passages"]["is_selected"]))
marco_sample = marco_pairs.shuffle(seed=42).select(range(100_000))

# Natural Questions
nq = load_dataset("google-research-datasets/natural_questions", split="train")
nq_sample = nq.shuffle(seed=42).select(range(50_000))

# GooAQ
gooaq = load_dataset("allenai/gooaq", split="train")
gooaq_sample = gooaq.shuffle(seed=42).select(range(30_000))
```

> **Do not train on BEIR datasets** — BEIR is the standard retrieval evaluation benchmark. Using it for training creates contamination. Use it only to evaluate your fine-tuned embedder against baselines.

#### Stage 2: Domain Fine-Tuning

Domain fine-tuning uses contrastive learning on `(query, positive_chunk)` pairs with hard negatives:

**Positive pairs**: `(user_query, relevant_runbook_chunk)` extracted from the same Q&A pairs generated in EXP-005. For each runbook Q&A pair `(question, answer, source_chunk)`, the positive pair is `(question, source_chunk)`.

**Hard negatives** (preferred over random negatives): Use BM25 to retrieve top-10 passages for each query, then treat the non-relevant ones as hard negatives. `sentence-transformers` has built-in support:

```python
from sentence_transformers.util import mine_hard_negatives
from sentence_transformers import SentenceTransformer

# Mine hard negatives from your runbook corpus
hard_negative_dataset = mine_hard_negatives(
    dataset=runbook_pairs,          # (query, positive_passage) pairs
    model=SentenceTransformer("nomic-ai/nomic-embed-text-v1"),
    corpus=runbook_chunks,          # all runbook chunks as the negative pool
    num_negatives=5,
    margin=0.1,                     # negatives must score ≥ 0.1 below positive
    output_dir="data/runbook_hard_negatives/",
)
```

Target: ≥ 200 positive pairs (from EXP-005's runbook Q&A seed set).

**Training mix across both stages**:

| Source | Pairs | Stage | Purpose |
|--------|-------|-------|---------|
| MS MARCO | 100K | Foundation | General retrieval capability |
| Natural Questions | 50K | Foundation | Factual extraction |
| GooAQ | 30K | Foundation | Short-answer format |
| StackExchange ops | 5K | Bridge | Technical register |
| Runbook pairs + hard negatives | 200–1K | Domain | Domain-specific precision |

### 3.3 Training Approach

Standard bi-encoder fine-tuning with MultipleNegativesRankingLoss (InfoNCE):

```python
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.readers import InputExample
from torch.utils.data import DataLoader

model = SentenceTransformer("nomic-ai/nomic-embed-text-v1")

# Training pairs: (query, positive_passage)
# Negatives are sampled from other examples in the batch
train_examples = [
    InputExample(texts=["What command restarts the ingestion service?",
                         "[RUNBOOK] §4.2 Service Restart: Run `make restart-ingestion`..."]),
    ...
]

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.MultipleNegativesRankingLoss(model)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    warmup_steps=50,
    output_path="checkpoints/nomic-embed-domain",
)
```

### 3.4 Evaluation

**Primary metric**: NDCG@10 on a held-out set of `(query, relevant_chunk_id)` pairs.

```python
from sentence_transformers.evaluation import InformationRetrievalEvaluator

ir_evaluator = InformationRetrievalEvaluator(
    queries=eval_queries,           # {id: query_text}
    corpus=eval_corpus,             # {id: chunk_text}
    relevant_docs=eval_relevance,   # {query_id: {chunk_id}}
    name="runbook-retrieval",
)
result = ir_evaluator(model)
print(result["runbook-retrieval_ndcg@10"])
```

**Baseline**: nomic-embed-text-v1 without domain fine-tuning (expected 0.55–0.65 NDCG@10).

**Target**: ≥ 0.80 NDCG@10 after domain fine-tuning.

**End-to-end metric**: runbook Q&A accuracy (EXP-005) with generic embedder vs. domain-fine-tuned embedder. Expected improvement: ≥ 10 percentage points.

### 3.5 Probe Set

`probes/retrieval.yaml` — held-out retrieval evaluation set:

```yaml
domain: retrieval
probes:
  - id: ret_restart_ingestion
    query: "How do I restart the ingestion service?"
    relevant_chunks: ["runbook-ingestion-v3.md::section-4.2"]
    notes: "Tests technical command retrieval"

  - id: ret_p1_sla
    query: "What's the response time requirement for a P1 alert?"
    relevant_chunks: ["runbook-oncall.md::section-2.1"]
    notes: "Tests SLA lookup"

  - id: ret_rollback
    query: "Production deployment is failing, how do I roll back?"
    relevant_chunks: ["runbook-deploy.md::section-3.1", "runbook-deploy.md::section-3.2"]
    notes: "Tests multi-chunk relevance"
```

### 3.6 Success Criteria

| Criterion | Target |
|-----------|--------|
| NDCG@10 (domain-fine-tuned) | ≥ 0.80 |
| NDCG@10 improvement over generic | ≥ +0.15 |
| CPU inference latency (ONNX INT8) | < 10 ms per embedding |
| End-to-end Q&A accuracy improvement | ≥ +10 percentage points vs. generic embedder |

---

## 4. Implementation Steps

### 4.1 ONNX Export

```bash
# Export fine-tuned model to ONNX
optimum-cli export onnx \
  --model ./checkpoints/nomic-embed-domain \
  --task feature-extraction \
  gristmill-data/models/nomic-embed/

# Quantize to INT8
optimum-cli onnxruntime quantize \
  --onnx_model gristmill-data/models/nomic-embed/ \
  --output gristmill-data/models/nomic-embed-int8/ \
  --avx512

# Validate embedding parity (PyTorch vs ONNX cosine similarity should be ≥ 0.999)
python scripts/validate_onnx.py \
  --pytorch-model checkpoints/nomic-embed-domain \
  --onnx-model gristmill-data/models/nomic-embed-int8/ \
  --task embedding
```

### 4.2 Integration with GristMill Ledger

```yaml
# In ~/.gristmill/config.yaml
grinders:
  models:
    embedder:
      runtime: onnx
      path: gristmill-data/models/nomic-embed-int8.onnx
      warm: true
      embedding_dim: 768
      max_seq_len: 8192

ledger:
  warm:
    vector_store: usearch
    index_path: gristmill-data/ledger/runbook_vectors.usearch
    embedding_model: embedder    # references grinder above
    rebuild_on_model_version_change: true
```

When the embedding model version changes (version bump in the model manifest), `grist-ledger` automatically triggers an index rebuild — re-embedding all stored chunks with the new model before enabling nearest-neighbour search.

### 4.3 Ablation Study

Run EXP-005 (runbook Q&A) with four embedder configurations to isolate the contribution of domain fine-tuning:

| Configuration | Embedder | Expected Q&A Accuracy |
|---------------|----------|----------------------|
| A: No retrieval | — | ≤ 0.20 |
| B: Generic embedder | nomic-embed-text-v1 (no fine-tuning) | ~0.50 |
| C: Domain embedder | nomic-embed-text-v1 (domain fine-tuned) | ~0.65 |
| D: Domain embedder + fine-tuned generative | both | ≥ 0.75 |

Configuration D is the target production system.

---

## 5. Results

*(To be filled in after experiment is run)*

### 5.1 Retrieval Evaluation

| Configuration | NDCG@10 |
|---------------|---------|
| Generic nomic-embed-text-v1 | — |
| Domain fine-tuned | — |
| Delta | — |

### 5.2 End-to-End Q&A Accuracy

| Configuration | Accuracy |
|---------------|----------|
| A: No retrieval | — |
| B: Generic embedder | — |
| C: Domain embedder | — |
| D: Full system | — |

### 5.3 Inference Benchmark

| Model | Latency (CPU, ONNX INT8) |
|-------|--------------------------|
| nomic-embed-text-v1 (full precision) | — |
| nomic-embed-text-v1-int8 | — |

---

## 6. Discussion

*(To be filled in after experiment is run)*

---

## 7. usearch Index Maintenance

The usearch index must be rebuilt whenever:
1. The embedding model is updated (version bump)
2. New runbook content is ingested (incremental add or full rebuild depending on change volume)
3. Chunks are deleted or modified (full rebuild required — usearch does not support in-place deletion efficiently)

The rebuild is a background Tokio task in `grist-ledger`. The ledger continues serving stale embeddings during the rebuild, atomically switching to the new index when the build completes. This ensures zero-downtime embedding model upgrades.

---

← [EXP-006](./exp-006-sentiment-deberta.md) | [Back to Index](../lora-distillation-experiments.md) | [EXP-008 →](./exp-008-multi-model-grinder.md)
