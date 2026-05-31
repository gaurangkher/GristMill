# EXP-004: Student Model Upgrade — Qwen2.5-0.5B → Qwen2.5-1.5B-Instruct

**Status**: Proposed
**Date**: 2026-05-31
**Authors**: GristMill Engineering Team

← [EXP-003](./exp-003-factual-accuracy-runner.md) | [Back to Index](../lora-distillation-experiments.md) | [EXP-005 →](./exp-005-runbooks-rag.md)

---

## Abstract

EXP-001 through EXP-003 establish that model capacity is the binding constraint for LoRA distillation at the current data scale: the 0.5B model cannot simultaneously retain factual knowledge and acquire reasoning patterns from 2,405 training examples. We propose upgrading the student model to Qwen2.5-1.5B-Instruct, which offers 3× more parameters while remaining within Apple Silicon MPS constraints (~3.5 GB VRAM, ~2.5 hours training on MPS). The hypothesis is that the additional parameter budget provides sufficient representational headroom to retain base factual knowledge while fine-tuning on reasoning-format examples. Success is measured by `FactualAccuracyRunner` on the `probes/reasoning.yaml` set (target: ≥ 4/5 with no regression vs. base model) and promoted by the configured `min_accuracy: 0.6` threshold.

---

## 1. Introduction

The central finding from completed experiments is that the 0.5B model operates at the edge of its representational capacity under LoRA fine-tuning on out-of-distribution reasoning data. Even with a conservative hyperparameter regime (lr=5e-5, 1 epoch, 2 modules), the adapter overwrites factual representations — February shrinks from 28 to 30 days under EXP-002's most conservative configuration.

The 1.5B model of the same family (Qwen2.5-1.5B-Instruct) provides a natural upgrade path:
- 3× parameter budget
- Same tokenizer, same architecture family
- Compatible with `transformers` and PEFT without code changes
- Fits in ~3.5 GB VRAM on MPS (within 8 GB unified memory budget)
- No QLoRA or quantization required at this scale

**Aspirational alternative**: Phi-3.5-mini-instruct (3.8B). Microsoft's Phi series is trained on textbook-quality, reasoning-dense data achieving GPT-3.5-class benchmark scores at 3.8B parameters. Requires ~16 GB unified memory (M1 Pro / M2 Pro or later). If available, this is the strongest candidate for demonstrating GristMill's local-first moat.

---

## 2. Hypothesis

Replacing Qwen2.5-0.5B-Instruct with Qwen2.5-1.5B-Instruct as the student model, while keeping the conservative hyperparameter regime from EXP-002, will:
1. Reduce or eliminate catastrophic forgetting of factual knowledge (February = 28 days)
2. Achieve FactualAccuracyRunner accuracy ≥ 4/5 (0.80) on `probes/reasoning.yaml`
3. Show measurable improvement over the base 1.5B model on `probes/training_reasoning.yaml`
4. Be promoted by `FactualAccuracyRunner` at `min_accuracy: 0.6`

---

## 3. Experimental Setup

### 3.1 Student Model

**Qwen/Qwen2.5-1.5B-Instruct** — 1.5 billion parameters, bfloat16, approximately 3 GB in memory. HuggingFace Hub model ID: `Qwen/Qwen2.5-1.5B-Instruct`.

### 3.2 Proposed Hyperparameters

| Parameter | EXP-002 (0.5B, conservative) | EXP-004 (1.5B, proposed) | Rationale |
|-----------|------------------------------|--------------------------|-----------|
| `base_model` | Qwen2.5-0.5B-Instruct | **Qwen2.5-1.5B-Instruct** | 3× parameter budget |
| `lora_rank` | 16 | 16 | Unchanged |
| `lora_alpha` | 16 | **32** | Scale ratio 1× → 2×; more representational capacity |
| `lora_target_modules` | q_proj, v_proj | **q_proj, k_proj, v_proj, o_proj** | Include output and key projections |
| `learning_rate` | 5e-5 | **1e-4** | 2× increase; more capacity absorbs larger updates |
| `num_epochs` | 1 | **2** | More passes with manageable forgetting risk |
| `replay_fraction` | 0.30 | **0.25** | Slightly less rehearsal; larger model needs less |
| Estimated MPS VRAM | ~1 GB | **~3.5 GB** | |
| Estimated training time (MPS, 2,400 examples) | ~45 min | **~2.5 hours** | |

### 3.3 Training Dataset

**Current dataset** (EXP-001–003): OpenHermes-2.5, a 1M-example general instruction dataset covering math, code, creative writing, roleplay, and trivia. The diversity is the problem — the model receives too weak a per-domain gradient signal to learn reasoning patterns without overfitting to surface form.

**Recommended replacement**: Replace the OpenHermes-2.5 seed with a focused math reasoning dataset. Update `scripts/seed_reasoning.py` to pull from one of these instead:

| Dataset | HF ID | Size | Why it fits |
|---------|-------|------|------------|
| **MetaMathQA** *(primary)* | `meta-math/MetaMathQA` | 395K | Each problem is reformulated 4–6 ways (different wording, same answer). Directly trains generalization over memorization. Best single choice for EXP-004. |
| **Orca-Math** | `microsoft/orca-math-word-problems-200k` | 200K | GPT-4-generated math word problems with full step-by-step solutions. Clean, structured, high-variety. |
| **WizardMath** | `WizardLMTeam/WizardMath_Data_With_CoT` | 96K | Augmented from GSM8K with CoT solutions. More concise than NuminaMath, well-suited to 1.5B. |

> **Do not use GSM8K as training data** — it is the canonical reasoning benchmark. Including it creates eval contamination.

**Recommended training sample**: MetaMathQA (5,000 examples) + Orca-Math (3,000 examples), filtered to rows with a clean numeric final answer:

```python
from datasets import load_dataset
import re

dataset = load_dataset("meta-math/MetaMathQA", split="train")
filtered = dataset.filter(
    lambda x: re.search(r'\d', x["response"])
              and len(x["query"]) > 50
              and not re.search(r"```|import |def ", x["query"])
)
# Shuffle and take 5,000
sample = filtered.shuffle(seed=42).select(range(5000))
```

Seed into the training buffer via `scripts/seed_reasoning.py --source metamath --n 5000`.

> **Key benefit of MetaMathQA's reformulations**: the model sees the same core problem stated 4–6 different ways. This is what drives generalization rather than surface memorization — the failure mode in EXP-001/002.

### 3.4 Config Changes

In `~/.gristmill/config.yaml`:
```yaml
trainer:
  base_model: Qwen/Qwen2.5-1.5B-Instruct
  lora_rank: 16
  lora_alpha: 32
  lora_target_modules: q_proj,k_proj,v_proj,o_proj
  learning_rate: 1e-4
  num_epochs: 2
  replay_fraction: 0.25
  batch_size: 1                  # MPS: batch_size=1, grad_accum compensates
  gradient_accumulation_steps: 16  # effective batch = 16

  validation:
    strategy: factual_accuracy
    probe_set: reasoning
    min_accuracy: 0.6
```

> Note: MPS batch_size is automatically overridden to 1 by `DistillationEngine._train_lora()` for MPS devices. gradient_accumulation_steps is auto-adjusted to maintain effective_batch = batch_size × gradient_accumulation_steps.

### 3.5 Evaluation Protocol

**Pre-training baseline**: Run `compare_lora_adapter.py` with no active adapter to record base 1.5B model performance.

```bash
python scripts/compare_lora_adapter.py \
  --probe-set reasoning \
  --max-new-tokens 512
```

**Post-training evaluation**:
```bash
# Static probe set
python scripts/compare_lora_adapter.py --probe-set reasoning --max-new-tokens 512

# Training-data probes (generated fresh)
python scripts/generate_training_probes.py --domain reasoning --n 20 --math-only
python scripts/compare_lora_adapter.py --probe-set training_reasoning --max-new-tokens 512
```

**Qualitative probe**: Same February leaves question used across all experiments for cross-experiment comparability.

### 3.6 Success Criteria

| Criterion | Target | Priority |
|-----------|--------|----------|
| FactualAccuracyRunner promotion | Promoted at min_accuracy: 0.6 | Required |
| `probes/reasoning.yaml` accuracy (adapter) | ≥ 4/5 (0.80) | Required |
| `probes/reasoning.yaml` accuracy (adapter ≥ base) | No regression | Required |
| February leaves probe | Correct (28 days, 196 leaves) | Aspirational |
| `training_reasoning.yaml` accuracy (adapter > base) | +5 percentage points | Aspirational |
| Training time | < 3 hours (MPS) | Operational |

---

## 4. Implementation Steps

```bash
# 1. Update config (never commit this file)
nano ~/.gristmill/config.yaml
# Set base_model: Qwen/Qwen2.5-1.5B-Instruct and hyperparameters above

# 2. Pre-training baseline evaluation
python scripts/compare_lora_adapter.py --probe-set reasoning --max-new-tokens 512 \
  | tee docs/experiments/results/exp-004-baseline.txt

# 3. Generate fresh training-data probes
python scripts/generate_training_probes.py --domain reasoning --n 20 --math-only

# 4. Run one training cycle (native macOS, not Docker)
cd gristmill-ml
python -m gristmill_ml.trainer.service --domain reasoning 2>&1 | tee out.txt

# 5. Post-training evaluation
python scripts/compare_lora_adapter.py --probe-set reasoning --max-new-tokens 512 \
  | tee docs/experiments/results/exp-004-post-training.txt
python scripts/compare_lora_adapter.py --probe-set training_reasoning --max-new-tokens 512 \
  | tee -a docs/experiments/results/exp-004-post-training.txt

# 6. Document results in this file (Section 5: Results)
```

---

## 5. Results

*(To be filled in after experiment is run)*

### 5.1 Training Dynamics

| Metric | Value |
|--------|-------|
| Final training loss | — |
| Token accuracy at final step | — |
| Wall-clock duration | — |

### 5.2 Quantitative Results

| Probe Set | Base Model Accuracy | Adapter Accuracy | Delta |
|-----------|--------------------|--------------------|-------|
| reasoning.yaml (5 probes) | — | — | — |
| training_reasoning.yaml (20 probes) | — | — | — |

### 5.3 Qualitative Evaluation

| Condition | Days Stated | Answer | Factual |
|-----------|-------------|--------|---------|
| Base 1.5B (no adapter) | — | — | — |
| Adapter 1.5B | — | — | — |

### 5.4 Promotion Decision

| Criterion | Result |
|-----------|--------|
| FactualAccuracyRunner score | — |
| Promoted | — |

---

## 6. Discussion

*(To be filled in after experiment is run)*

---

## 7. Expected Outcomes and Contingencies

**If forgetting is eliminated**: Proceed with the 1.5B model as the production student for the reasoning domain. Begin EXP-005 (runbooks) with the 1.5B model as well.

**If forgetting persists but is reduced**: Consider whether the reduction is sufficient for the use case. Acceptance criterion: the adapter must not regress below the base model on the `probes/reasoning.yaml` factual accuracy score (currently 0.80 for the 0.5B base).

**If forgetting is unchanged**: Evaluate Phi-3.5-mini-instruct (3.8B) as the aspirational alternative if hardware permits. Otherwise, pivot strategy: rather than training for general reasoning, train for narrow domain-specific tasks with high distribution alignment (EXP-005–EXP-007).

**If training time exceeds 4 hours**: Reduce `num_epochs` to 1 and rerun.

---

← [EXP-003](./exp-003-factual-accuracy-runner.md) | [Back to Index](../lora-distillation-experiments.md) | [EXP-005 →](./exp-005-runbooks-rag.md)
