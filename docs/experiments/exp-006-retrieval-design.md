# EXP-006: Retrieval System Design — Structure-Aware Chunking, Hybrid Search, HyDE

**Status**: Proposed
**Follows**: [EXP-005](./exp-005-runbooks-rag.md)
**Authors**: GristMill Engineering Team

← [EXP-005](./exp-005-runbooks-rag.md) | [Back to Index](../lora-distillation-experiments.md)

---

## Motivation

EXP-005 established that `overall_accuracy = retrieval_accuracy × extraction_accuracy`.
Extraction is already at 100% with the right context in place. The only remaining lever
is retrieval. This experiment implements four retrieval improvements identified in EXP-005
§6.7: structure-aware chunking, hybrid BM25 + dense search, embedding model fine-tuning,
and HyDE query expansion.

---

## System Architecture

### Full Pipeline Block Diagram

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                         OFFLINE — INGESTION PIPELINE                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

  Raw Runbooks (Markdown / PDF)
           │
           ▼
  ┌─────────────────────────────────────────────────────┐
  │             Structure-Aware Chunker                 │
  │                                                     │
  │  Split boundaries:  ##, ###, code fences, tables    │
  │  Chunk size:        50–800 tokens (variable)        │
  │  Metadata per chunk:                                │
  │    { file, section_id, section_title,               │
  │      depth, prev_section_id, tags[] }               │
  └─────────────────────────────────────────────────────┘
           │
           │  (chunk_text, metadata)
           │
     ┌─────┴──────┐
     │            │
     ▼            ▼
┌──────────┐  ┌────────────────────────────────────────┐
│  BM25    │  │         Fine-tuned Bi-Encoder           │
│  Index   │  │     (nomic-embed-text, domain-tuned)    │
│          │  │                                         │
│ inverted │  │  embeds chunk_text + section_title      │
│  token   │  │  into 768-dim vector                    │
│  index   │  └────────────────────────────────────────┘
└──────────┘           │
     │                 ▼
     │        ┌─────────────────────┐
     │        │   grist-ledger      │
     │        │   (usearch index)   │
     │        │   warm-tier store   │
     │        └─────────────────────┘
     │                 │
     └────────┬────────┘
              │  both indexes ready
              ▼
         INDEX READY


╔══════════════════════════════════════════════════════════════════════════════╗
║              OFFLINE — EMBEDDING MODEL FINE-TUNING LOOP                     ║
╚══════════════════════════════════════════════════════════════════════════════╝

  Probe set (probes/runbooks.yaml)
  + Synthetic runbooks (runbooks-synthetic/)
           │
           ▼
  ┌─────────────────────────────────────────────────────┐
  │              Triplet Generator                      │
  │                                                     │
  │  For each probe with a context field:               │
  │    anchor   = probe.question                        │
  │    positive = probe.context  (correct chunk)        │
  │    negative = random other chunk from same file     │
  │               or chunk with similar keywords        │
  │               (hard negative)                       │
  │                                                     │
  │  Output: ~500–1000 (anchor, pos, neg) triplets      │
  └─────────────────────────────────────────────────────┘
           │
           ▼
  ┌─────────────────────────────────────────────────────┐
  │          Contrastive Fine-Tuning (Python)           │
  │                                                     │
  │  Loss:    MultipleNegativesRankingLoss              │
  │  Base:    nomic-ai/nomic-embed-text-v1              │
  │  Epochs:  5–10  (~10 min on MPS / A10G)             │
  │  Library: sentence-transformers                     │
  └─────────────────────────────────────────────────────┘
           │
           ▼
  ┌─────────────────────────────────────────────────────┐
  │         Fine-tuned Bi-Encoder (saved to disk)       │
  └─────────────────────────────────────────────────────┘
           │
     ┌─────┴────────────────┐
     ▼                      ▼
  Re-index all chunks    Use in query pipeline
  (replace old vectors   (hot-reload in grist-ledger)
   in usearch index)


╔══════════════════════════════════════════════════════════════════════════════╗
║                         ONLINE — QUERY PIPELINE                             ║
╚══════════════════════════════════════════════════════════════════════════════╝

  User Query: "what do I do if ingestion fails with ERR_SCHEMA_MISMATCH?"
           │
           ├──────────────────────────────────────────────────────────┐
           │                                                          │
           ▼                                                          │
  ┌─────────────────────────────────────┐                            │
  │              HyDE Step              │                            │
  │                                     │                            │
  │  Local LLM (grist-grinder) generates│                            │
  │  a hypothetical answer:             │                            │
  │                                     │                            │
  │  "Run make validate-schema to       │                            │
  │   diagnose the mismatch before      │                            │
  │   retrying the restart command."    │                            │
  │                                     │                            │
  │  Uses runbook vocabulary not        │                            │
  │  present in the original query.     │                            │
  └─────────────────────────────────────┘                            │
           │                                                          │
           ▼                                                          │
  ┌─────────────────────────────────────┐                            │
  │      Fine-tuned Bi-Encoder          │                            │
  │  embeds hypothetical answer         │                            │
  │  → 768-dim query vector             │                            │
  └─────────────────────────────────────┘                            │
           │                                                          │
           │  dense query vector                                      │
           ▼                                                          ▼
  ┌──────────────────────────┐             ┌────────────────────────────┐
  │   grist-ledger           │             │   BM25 Index               │
  │   usearch ANN search     │             │                            │
  │                          │             │   tokenise original query  │
  │   top-20 dense           │             │   exact + fuzzy token      │
  │   candidates             │             │   match                    │
  │   (by cosine similarity) │             │                            │
  │                          │             │   top-20 lexical           │
  │   strong on: paraphrase, │             │   candidates               │
  │   semantic synonyms      │             │                            │
  │                          │             │   strong on: exact cmds,   │
  │                          │             │   flag names, error codes  │
  └──────────────────────────┘             └────────────────────────────┘
           │                                          │
           │  dense candidates                        │  lexical candidates
           └──────────────────┬───────────────────────┘
                              │
                              ▼
  ┌─────────────────────────────────────────────────────┐
  │           Reciprocal Rank Fusion (RRF)              │
  │                                                     │
  │  score(chunk) = Σ  1 / (k + rank_in_list_i)         │
  │                 i                                   │
  │                                                     │
  │  k = 60  (standard RRF constant)                    │
  │  merges dense + lexical lists into unified ranking  │
  └─────────────────────────────────────────────────────┘
           │
           │  top-3 chunks (unified score)
           ▼
  ┌─────────────────────────────────────────────────────┐
  │                  Prompt Builder                     │
  │                                                     │
  │  [RUNBOOK CONTEXT]                                  │
  │  § {section_title_1}                                │
  │  {chunk_text_1}                                     │
  │  ---                                                │
  │  § {section_title_2}                                │
  │  {chunk_text_2}                                     │
  │  ---                                                │
  │  § {section_title_3}                                │
  │  {chunk_text_3}                                     │
  │                                                     │
  │  Question: {original_user_query}                    │
  └─────────────────────────────────────────────────────┘
           │
           ▼
  ┌─────────────────────────────────────────────────────┐
  │          Qwen2.5-1.5B-Instruct + LoRA               │
  │              (grist-grinder, ONNX)                  │
  └─────────────────────────────────────────────────────┘
           │
           ▼
     Grounded Answer
```

---

### Where Each Improvement Lives

```
                    RETRIEVAL FAILURE MODE           FIX
                 ┌──────────────────────────────────────────────────┐
  Chunking       │ Answer split across chunk boundary               │
  (Improvement 1)│                                                  │
                 │  flat 512-token window                           │
                 │         │                                        │
                 │         ▼                                        │
                 │  structure-aware split at section boundaries     │
                 │  chunk = exactly one runbook section             │
                 │  section_title stored as metadata                │
                 └──────────────────────────────────────────────────┘
                 ┌──────────────────────────────────────────────────┐
  BM25           │ Vocabulary mismatch: exact command not found     │
  (Improvement 2)│                                                  │
                 │  "restart ingestion" query                       │
                 │         │                                        │
                 │         ├── dense search misses "make restart-   │
                 │         │   ingestion" (different tokens)        │
                 │         │                                        │
                 │         └── BM25 exact match finds it directly  │
                 └──────────────────────────────────────────────────┘
                 ┌──────────────────────────────────────────────────┐
  Embed fine-tune│ Right chunk ranks 4th, below top-k cutoff       │
  (Improvement 3)│                                                  │
                 │  generic nomic-embed: cosine(query, chunk) = 0.6 │
                 │         │                                        │
                 │         ▼                                        │
                 │  domain-tuned: cosine(query, chunk) = 0.85       │
                 │  trained on (runbook query, correct chunk) pairs │
                 └──────────────────────────────────────────────────┘
                 ┌──────────────────────────────────────────────────┐
  HyDE           │ Query uses no runbook vocabulary at all          │
  (Improvement 4)│                                                  │
                 │  user: "ingestion broken after db change"        │
                 │         │                                        │
                 │         ▼  LLM hypothetical answer               │
                 │         "run make restart-ingestion after        │
                 │          migration reports SUCCESS"              │
                 │         │                                        │
                 │         ▼  embed hypothetical answer             │
                 │         cosine with runbook chunk now high       │
                 └──────────────────────────────────────────────────┘
```

---

### Component Ownership (GristMill Architecture)

```
┌─────────────────────────────────────────────────────────────────┐
│  gristmill-ml  (Python)                                         │
│                                                                 │
│  ┌─────────────────────┐   ┌───────────────────────────────┐   │
│  │ Structure-aware     │   │ Embedding model fine-tuning   │   │
│  │ chunker             │   │                               │   │
│  │                     │   │  scripts/finetune_embedder.py │   │
│  │ datasets/           │   │  sentence-transformers        │   │
│  │   runbook_chunker.py│   │  MultipleNegativesRankingLoss │   │
│  └─────────────────────┘   └───────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────┐                                        │
│  │ Triplet generator   │                                        │
│  │                     │                                        │
│  │ scripts/            │                                        │
│  │  gen_embed_triplets │                                        │
│  │  .py                │                                        │
│  └─────────────────────┘                                        │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  gristmill-core  (Rust)                                         │
│                                                                 │
│  ┌─────────────────────┐   ┌───────────────────────────────┐   │
│  │ grist-ledger        │   │ grist-hammer                  │   │
│  │                     │   │                               │   │
│  │ • usearch ANN index │   │ • HyDE: calls local grinder   │   │
│  │ • BM25 inverted idx │   │   to generate hypothetical    │   │
│  │   (new — SQLite FTS │   │   answer before retrieval     │   │
│  │    or tantivy)      │   │                               │   │
│  │ • RRF merge         │   │ • falls back: skip HyDE if    │   │
│  │ • structured chunk  │   │   grinder busy (latency gate) │   │
│  │   metadata store    │   └───────────────────────────────┘   │
│  └─────────────────────┘                                        │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  gristmill-integrations  (TypeScript)                           │
│                                                                 │
│  • no changes — retrieval is fully inside Rust core             │
└─────────────────────────────────────────────────────────────────┘
```

---

### Latency Budget

```
                      HyDE OFF          HyDE ON
                   ┌────────────┐    ┌────────────┐
Query encoding     │    5ms     │    │    5ms     │
                   └────────────┘    └────────────┘
HyDE generation    │    —       │    │   80ms     │  (local 0.5B grinder)
                   └────────────┘    └────────────┘
BM25 search        │    2ms     │    │    2ms     │
                   └────────────┘    └────────────┘
ANN search         │    3ms     │    │    3ms     │
                   └────────────┘    └────────────┘
RRF merge          │    1ms     │    │    1ms     │
                   └────────────┘    └────────────┘
Prompt build       │    1ms     │    │    1ms     │
                   └────────────┘    └────────────┘
LLM extraction     │  120ms     │    │  120ms     │  (1.5B ONNX INT8)
                   └────────────┘    └────────────┘
                   ─────────────     ─────────────
  TOTAL            │  132ms     │    │  212ms     │
                   └────────────┘    └────────────┘

HyDE adds ~80ms (one small-model forward pass).
Both remain within the 200ms local-first SLA without HyDE.
With HyDE: 212ms — slightly over; consider async pre-fetch or
limiting HyDE to low-confidence queries only.
```

---

## Evaluation Plan

Run the same 4 conditions from EXP-005 against `probes/runbooks-hard.yaml`
after each improvement is added, to isolate the contribution of each:

| Run | Chunking | BM25 | Embed fine-tune | HyDE | Expected gain |
|-----|----------|------|-----------------|------|---------------|
| Baseline (EXP-005) | flat 512 | ✗ | generic | ✗ | 94% (hard) |
| +Chunking | structure | ✗ | generic | ✗ | ~95% |
| +BM25 | structure | ✓ | generic | ✗ | ~97% |
| +Embed FT | structure | ✓ | fine-tuned | ✗ | ~98% |
| +HyDE | structure | ✓ | fine-tuned | ✓ | ~99%+ |

Each run uses the same `--with-context` simulation for fair comparison
until real retrieval infrastructure is wired up.
