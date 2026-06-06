# Experiment: LoRA Personalization Depth vs. Adapter Size

**Status:** Proposed  
**Category:** LoRA Characterization  
**Relevant paper section:** §5 Experiments — Adapter Efficiency

---

## Motivation

GristMill's grinder uses a fixed LoRA rank for all users, but the optimal rank is an open question. Too low and the adapter fails to capture the user's style; too high and storage costs make the system unviable at scale. This experiment finds the rank sweet spot — the minimum rank that captures most of the personalization signal — and produces a principled storage-cost estimate for large-scale deployment.

---

## Core Question

> What is the minimum LoRA rank needed to capture a user's personal style and domain, and what does that cost in storage at scale?

---

## Setup

### Model

- **Student (grinder):** Qwen2.5-3B (consistent with GristMill baseline)
- **Teacher:** Qwen2.5-32B (used for synthetic data generation and as evaluator judge)

### User Personas

Create 5–10 synthetic but realistic user profiles, each with a distinct domain and style signature. Suggested set:

| Persona | Domain | Style signature |
|---|---|---|
| Python developer | Software engineering | Functional patterns, type hints, docstrings |
| Radiologist | Medical | Structured findings reports, ICD codes |
| Lawyer | Legal | Formal UK English, specific citation format |
| Creative writer | Fiction | Short sentences, present tense, sparse dialogue tags |
| Data analyst | Analytics | Bullet-point summaries with confidence qualifiers |

For each persona:
- Generate **500 training examples** using Qwen2.5-32B conditioned on a detailed persona system prompt
- Hold out **100 examples** for evaluation (no overlap with training set)

### Independent Variable: LoRA Rank

Train a separate adapter per persona at each of the following ranks:

```
ranks = [1, 2, 4, 8, 16, 32, 64]
```

All other hyperparameters held fixed:
- Learning rate: `2e-4`
- Training steps: `500`
- Target modules: `q_proj, v_proj` (standard attention-only targeting)
- Optimizer: AdamW with cosine schedule

---

## Metrics

### Primary

| Metric | Description | Tool |
|---|---|---|
| **Style score** | Teacher-as-judge rating (1–5) on style adherence + task correctness | Qwen2.5-32B judge prompt |
| **Adapter size (MB)** | On-disk size of saved LoRA weights | `os.path.getsize` |
| **Training time (s)** | Wall-clock time per adapter on a single GPU | `time.perf_counter` |

### Secondary

| Metric | Description |
|---|---|
| **Format consistency** | ROUGE-L against held-out gold outputs (for structured output personas) |
| **Exact match** | For personas with rigid output schemas (ICD codes, JSON) |
| **Base capability retention** | MMLU and GSM8K accuracy delta vs. base model — verifies LoRA doesn't degrade general ability |

---

## Expected Results

| Rank | Approx. adapter size | Style score (0–5) | MMLU delta |
|---|---|---|---|
| 1 | ~1 MB | 2.5–3.0 | ~0% |
| 4 | ~4 MB | 3.5–4.0 | ~0% |
| 8 | ~8 MB | 4.0–4.3 | ~0% |
| 16 | ~16 MB | 4.3–4.5 | ~−1% |
| 32 | ~30 MB | 4.5–4.6 | ~−1% |
| 64 | ~60 MB | 4.6–4.7 | ~−2% |

**Hypothesis:** Ranks 4–8 hit a sweet spot, capturing ~90% of max personalization quality at ~7–13% of rank-64 storage cost. If confirmed, a system serving 100k users needs ~400–800 MB of total adapter storage rather than tens of gigabytes.

---

## Extension: Adapter Composition

Once per-persona adapters are trained, test **LoRA merging** — combining two adapters and measuring whether quality adds, averages, or degrades.

### Setup

- Merge pairs of adapters using linear interpolation:
  `merged = λ · A + (1 − λ) · B`, sweeping `λ ∈ {0.0, 0.25, 0.5, 0.75, 1.0}`
- Also test [TIES-merging](https://arxiv.org/abs/2306.01708) as a stronger baseline
- Evaluate merged adapter on held-out sets for **both** source personas

### Research question

Does a merged adapter serve both users adequately, or does it collapse to a midpoint that satisfies neither? This has direct implications for whether GristMill could share adapters across similar users to reduce storage further.

---

## Connection to GristMill

This experiment directly answers a question the current paper leaves open: *how big does the grinder's LoRA need to be to actually personalize?*

With these results, the paper can make a concrete deployability claim, e.g.:

> *"Rank 8 captures 94% of maximum personalization quality at 8 MB per user adapter, making GristMill viable at 100k-user scale with under 1 GB of total adapter storage."*

That is a systems result, not just an ablation — it belongs in the main experiments section alongside Experiments 1–3.

---

## Implementation Notes

- Use HuggingFace `peft` for LoRA training (`LoraConfig`, `get_peft_model`)
- Use `trl.SFTTrainer` for the training loop (consistent with GristMill trainer)
- Save adapters with `model.save_pretrained()` — adapter weights only, not merged
- Judge prompt for style scoring should be kept constant across all evaluations; version-control it alongside this doc

---

## Open Questions

- Should target modules extend beyond `q_proj, v_proj` to include `k_proj`, `o_proj`, or MLP layers for style tasks?
- Is the teacher-as-judge evaluation reliable enough, or do we need human annotations for at least a subset?
- At rank 1, is the adapter learning anything meaningful, or is it just noise? Worth inspecting singular value decomposition of learned weights.
