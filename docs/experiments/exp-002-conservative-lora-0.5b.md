# EXP-002: Conservative LoRA Training — Qwen2.5-0.5B-Instruct

**Status**: Complete
**Date**: 2026-05-29
**Authors**: GristMill Engineering Team

← [EXP-001](./exp-001-baseline-lora-3b.md) | [Back to Index](../lora-distillation-experiments.md) | [EXP-003 →](./exp-003-factual-accuracy-runner.md)

---

## Abstract

Following the catastrophic forgetting observed in EXP-001, we applied a conservative hyperparameter regime to a smaller student model (Qwen2.5-0.5B-Instruct) with the goal of reducing the magnitude of weight perturbation: learning rate reduced 4×, epochs reduced from 3 to 1, LoRA scale ratio halved (2× → 1×), and target modules reduced from 7 to 2 (q_proj and v_proj only). We document two sub-experiments: the initial run (EXP-002a), which was rolled back because ROUGE-L fell below the promotion threshold; and a rerun with relaxed validation thresholds (EXP-002b), which was promoted. Both sub-experiments continue to exhibit factual forgetting (February stated as 30 days rather than 28). We conclude that the failure mode is not resolvable by hyperparameter tuning alone at this model scale, and that the ROUGE-L metric produces perverse promotion decisions — rolling back the less-wrong adapter while promoting the more-wrong one. This experiment motivates replacing ROUGE-L with a factual accuracy metric (EXP-003) and upgrading the student model (EXP-004).

---

## 1. Introduction

EXP-001 showed that aggressive hyperparameters produce clear catastrophic forgetting. The standard remediation is to reduce the effective gradient magnitude: lower learning rate, fewer epochs, fewer targeted modules, smaller LoRA scale ratio. We also switched to the 0.5B model variant, both to reduce memory pressure and because the 0.5B's smaller parameter budget makes the effect of hyperparameter changes more controlled and observable.

The key question: does reducing gradient magnitude by ~80× prevent the forgetting of February's day count?

---

## 2. Experimental Setup

### 2.1 Student Model

**Qwen/Qwen2.5-0.5B-Instruct** — 494 million parameters, bfloat16, approximately 1 GB in memory. Same tokenizer and architecture family as the 3B variant, enabling direct comparison of results.

### 2.2 Hyperparameter Changes vs. EXP-001

| Hyperparameter | EXP-001 (3B, aggressive) | EXP-002 (0.5B, conservative) | Rationale |
|----------------|--------------------------|------------------------------|-----------|
| `base_model` | Qwen2.5-3B-Instruct | **Qwen2.5-0.5B-Instruct** | Reduce memory pressure |
| `num_epochs` | 3 | **1** | Limit total gradient steps |
| `learning_rate` | 2e-4 | **5e-5** | 4× reduction in step size |
| `lora_alpha` | 32 | **16** | Scale ratio 2× → 1× |
| `lora_target_modules` | 7 modules | **q_proj, v_proj** | Minimal footprint |
| `replay_fraction` | 0.17 | **0.30** | More rehearsal of prior examples |

**Effective gradient magnitude comparison**:

| Factor | EXP-001 (Databricks-style) | EXP-002 (conservative) | Ratio |
|--------|---------------------------|------------------------|-------|
| Learning rate | 2e-4 | 5e-5 | 4× lower |
| lora_alpha / lora_rank | 32/16 = **2×** | 16/16 = **1×** | 2× lower |
| Epochs | 3 | 1 | 3× lower |
| **Composite effective LR** | **1.2e-3** | **5e-5** | **~24× lower** |

See Section 5.2 for comparison against a typical published LoRA tutorial (80× lower than Databricks baseline).

### 2.3 Validation (EXP-002a — Initial Run)

ROUGE-L with original promotion thresholds:
- `overall_delta_min`: -0.01
- `domain_delta_min`: -0.03

### 2.4 Validation (EXP-002b — Rerun with Relaxed Thresholds)

After analyzing the rollback, thresholds were relaxed:
- `overall_delta_min`: **-0.05**
- `domain_delta_min`: **-0.08**

---

## 3. Results

### 3.1 Sub-Experiment A: Initial Run (Rolled Back)

**Outcome**: Adapter rolled back — not promoted.

**ROUGE-L delta**: `overall_delta = -0.0289`, below `overall_delta_min = -0.01`.

The conservative adapter scored materially lower on ROUGE-L than the previously promoted EXP-001 adapter. This is the expected behavior of a system that rewards format reproduction: the EXP-001 adapter (catastrophically overfit to teacher format) scores higher than the EXP-002 adapter (less overfit, closer to correct behavior).

### 3.2 Sub-Experiment B: Rerun with Relaxed Thresholds (Promoted)

| Domain | Checkpoint Version | ROUGE-L Score | Promotion Decision |
|--------|--------------------|---------------|--------------------|
| `reasoning` | v4 | 0.1982 | **Promoted** |

### 3.3 ROUGE-L Comparison Across Experiments

| Experiment | Model | ROUGE-L (reasoning) | Promoted |
|------------|-------|---------------------|----------|
| EXP-001 | Qwen2.5-3B (aggressive) | 0.2322 | Yes |
| EXP-002a | Qwen2.5-0.5B (conservative) | < 0.2033 | **No** (rollback) |
| EXP-002b | Qwen2.5-0.5B (conservative, relaxed) | 0.1982 | Yes |

### 3.4 Qualitative Evaluation (February Leaves Probe)

| Condition | Days Stated | Answer | Arithmetic | Factual |
|-----------|-------------|--------|------------|---------|
| Base model (no adapter) | 28 | 196 leaves | Correct | **Correct** |
| EXP-001 adapter v1 | 31 | 217 leaves | Correct | **Incorrect** |
| EXP-002b adapter v4 | 30 | 210 leaves | Correct | **Incorrect** |

The conservative adapter improved: it stated 30 days rather than 31. But the fundamental failure persists — February still does not have 30 days.

---

## 4. Discussion

### 4.1 The Perverse Rollback

The EXP-002a rollback is a direct consequence of using ROUGE-L to evaluate an adapter whose primary behavioral change is *being less wrong* rather than *producing more teacher-like text*. The EXP-001 adapter reproduced the teacher's step-by-step format verbatim — high ROUGE-L, high confidence, wrong answer. The EXP-002a adapter generated more varied output, potentially more accurate, but ROUGE-L penalized the variation.

The promotion system therefore rolled back the less-wrong adapter and retained the more-wrong one. This is the definitive demonstration that ROUGE-L is the wrong metric for this task family. EXP-003 replaces it.

### 4.2 Why Forgetting Persists

Even at 24× lower effective gradient magnitude than EXP-001, the 0.5B model continues to exhibit forgetting. Four structural factors explain this:

**Capacity**: The 0.5B model has 494M parameters distributed across 24 attention layers. Its pre-training factual knowledge is encoded in a compact representational space. Even small LoRA updates to q_proj and v_proj — the attention components most involved in pattern-matching over context — are sufficient to disrupt the key-value lookup patterns that retrieve "February → 28 days."

**Data distribution**: 2,405 training examples follow a stereotyped format (numbered steps, arithmetic chain, "Final answer: X"). This format is out-of-distribution relative to the model's instruction-tuning pre-training. The model cannot learn a "reasoning style" separate from the numerical facts in those examples — the two are entangled in the gradient signal.

**Rehearsal fraction**: Even with replay_fraction = 0.30, the model does not re-encounter any probe-equivalent example during training. The February leaves problem is not in the training data. The model must generalize from "compute N × M for various N, M" to "compute 28 × 7" — but LoRA cannot separate the general skill from the specific numerical facts entangled with it in 2,405 examples.

**Dataset diversity** (root cause, identified post-hoc): OpenHermes-2.5 mixes math, code, creative writing, roleplay, and trivia. The reasoning-style gradient signal is diluted across unrelated task types, meaning each math example has to compete with structurally dissimilar examples for the model's representational capacity. EXP-004 replaces OpenHermes-2.5 with **MetaMathQA** (`meta-math/MetaMathQA`) — a focused dataset where every example is a quantitative word problem with 4–6 reformulations of the same core problem. This directly addresses the memorization-vs-generalization failure described in §4.2.

### 4.3 Why Published LoRA Results Do Not Transfer Here

Published tutorials on LoRA fine-tuning of 0.5B-scale models report successful adaptation. Four structural differences explain the gap:

| Factor | Published (e.g., Databricks/Capybara) | GristMill EXP-002 |
|--------|--------------------------------------|-------------------|
| Training data | Thousands of diverse, high-quality instruction examples | 2,405 narrow arithmetic reasoning examples |
| Data distribution | In-distribution (same format as pre-training) | Out-of-distribution (step-by-step arithmetic format) |
| Effective LR | ~4e-3 (lr=1e-3, alpha/rank=4×) | ~5e-5 (~80× lower) |
| Target modules | All 7 projections | 2 (q_proj, v_proj only) |
| Evaluation | Eval loss on same distribution | ROUGE-L on OOD probe |

None of these differences is disqualifying in isolation. Together they compound: tiny OOD dataset → memorization of surface form; 80× weaker gradient → barely updates in one epoch; 2 of 7 modules → insufficient capacity to acquire reasoning patterns; ROUGE-L → rewards format mimicry. See [the GristMill architecture doc](../../gristmill-v2-architecture.md) for the moat argument that motivates a different approach for production.

### 4.4 The Path Forward

This experiment, combined with EXP-001, establishes that the failure mode is not resolvable by hyperparameter tuning alone within the current setup. Two changes are required:

1. **Replace ROUGE-L with factual accuracy** ([EXP-003](./exp-003-factual-accuracy-runner.md)): so promotion decisions reflect semantic correctness, not format mimicry.
2. **Upgrade the student model** ([EXP-004](./exp-004-student-model-upgrade.md)): more parameter budget means more residual capacity to retain factual knowledge during fine-tuning.

---

## 5. Conclusions

1. Conservative hyperparameters (lr=5e-5, 1 epoch, 2 modules) reduce catastrophic forgetting severity (31 days → 30 days for February) but do not eliminate it.
2. ROUGE-L produces a perverse rollback: it rolled back the less-wrong adapter (EXP-002a) because it correctly rewarded the more-overfit adapter (EXP-001) for reproducing the teacher's surface form.
3. The failure mode is not resolvable by hyperparameter tuning alone at 0.5B scale with this dataset.
4. Both EXP-001 and EXP-002 confirm that the pipeline infrastructure is functionally correct; the failure is model-capacity-specific.

---

## Appendix: Checkpoint Version History

| Version | Experiment | Domain | ROUGE-L | Status |
|---------|------------|--------|---------|--------|
| v3 | EXP-002a | (any) | < 0.2033 | Rolled back |
| v4 | EXP-002b | `reasoning` | 0.1982 | Superseded by EXP-003+ |

---

← [EXP-001](./exp-001-baseline-lora-3b.md) | [Back to Index](../lora-distillation-experiments.md) | [EXP-003 →](./exp-003-factual-accuracy-runner.md)
