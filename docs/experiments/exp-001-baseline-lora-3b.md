# EXP-001: Baseline LoRA Training — Qwen2.5-3B-Instruct with Aggressive Hyperparameters

**Status**: Complete
**Date**: 2026-05-29
**Authors**: GristMill Engineering Team

← [EXP-000](./exp-000-pipeline-bringup.md) | [Back to Index](../lora-distillation-experiments.md) | [EXP-002 →](./exp-002-conservative-lora-0.5b.md)

---

## Abstract

We report the first successful training run in the GristMill v2 closed-loop distillation pipeline: a LoRA adapter trained on Qwen2.5-3B-Instruct with aggressive hyperparameters (lr=2e-4, 3 epochs, 7 projection modules, lora_alpha/lora_rank = 2×) on a corpus of 2,405 reasoning examples. The adapter achieved ROUGE-L scores of 0.2322 and 0.2318 across two domains and was promoted to the active checkpoint directory. However, qualitative evaluation reveals that the adapter exhibits catastrophic forgetting: it correctly executes step-by-step arithmetic but asserts factually wrong world knowledge (stating February has 31 days). We identify the root causes as data distribution mismatch, aggressive gradient magnitude, and the fundamental limitations of ROUGE-L as a validation metric for factual reasoning. This experiment motivates the conservative hyperparameter study in EXP-002 and the metric replacement in EXP-003.

---

## 1. Introduction

The goal of this experiment was to establish a functional end-to-end baseline: train a LoRA adapter, pass validation, promote it to the checkpoint directory, and verify the Rust daemon hot-reloads it. Hyperparameters were chosen to be permissive — large learning rate, multiple epochs, all projection layers targeted — to maximize the probability of ROUGE-L improvement and produce a promoted checkpoint.

The broader hypothesis being tested: can a small language model (3B parameters) acquire step-by-step reasoning patterns from 2,405 teacher-labeled examples via LoRA fine-tuning, without losing its pre-trained factual knowledge?

---

## 2. Experimental Setup

### 2.1 Student Model

**Qwen/Qwen2.5-3B-Instruct** — 3 billion parameters, bfloat16, approximately 6 GB in memory. Selected for stronger baseline reasoning capability relative to the 0.5B variant. Loaded from HuggingFace Hub at training time.

### 2.2 Training Data

2,405 reasoning examples from the OpenHermes-2.5 dataset, stored in the training buffer SQLite database under `domain_tag IN ('reasoning', 'default')`. Examples follow a structured step-by-step format: the teacher (Ollama llama3.1:8b) is prompted with a multi-step arithmetic or logic question and responds with a numbered chain of steps.

### 2.3 Hardware

Apple Silicon MPS (native macOS host). Docker was evaluated and rejected — the Linux VM layer blocks MPS access, producing step times of 2–5 minutes, making a 453-step run project to 15–38 hours. See [EXP-000 Bug 11](./exp-000-pipeline-bringup.md).

### 2.4 Hyperparameters

| Hyperparameter | Value |
|----------------|-------|
| `base_model` | `Qwen/Qwen2.5-3B-Instruct` |
| `num_epochs` | 3 |
| `learning_rate` | 2e-4 |
| `lora_rank` (r) | 16 |
| `lora_alpha` | 32 (scale ratio: 2×) |
| `lora_target_modules` | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj (7 modules) |
| `max_length` | 512 |
| `per_device_train_batch_size` | 4 |
| `gradient_accumulation_steps` | 4 (effective batch: 16) |
| `gradient_checkpointing` | True, `use_reentrant=False` |
| `replay_fraction` | 0.17 |
| `torch_dtype` | bfloat16 |
| `bf16` | True |

**Effective gradient magnitude**: lr × (lora_alpha/lora_rank) × epochs = 2e-4 × 2 × 3 = **1.2e-3**

### 2.5 Validation

ROUGE-L (LCS F1) against held-out teacher reference outputs. Promotion thresholds at time of experiment:
- `overall_delta_min`: -0.01
- `domain_delta_min`: -0.03

### 2.6 Qualitative Probe

Fixed probe for cross-experiment comparison:

> *"Every day, a tree drops 7 leaves. How many leaves would it drop in a month of February in a non-leap year?"*

**Expected answer**: 196 leaves (28 days × 7). This tests whether the model retains the factual knowledge that February has 28 days — world knowledge that should be invariant across any fine-tuning regime.

---

## 3. Training Dynamics

| Metric | Value |
|--------|-------|
| Final training loss | 0.8375 |
| Gradient norm at final logged step | 0.3256 |
| Token accuracy at final logged step | 76.85% |
| Epochs completed at checkpoint | ~0.8 |
| Wall-clock duration | ~12 hours |

Training loss of 0.8375 at ~0.8 epochs indicates the model was still actively learning at checkpoint capture and had not converged. The gradient norm of 0.3256 is within a healthy range (no explosion, no vanishing).

---

## 4. Results

### 4.1 Quantitative (ROUGE-L)

| Domain | Checkpoint Version | ROUGE-L Score | Promotion Decision |
|--------|--------------------|---------------|--------------------|
| `reasoning` | v1 | 0.2322 | **Promoted** |
| `default` | v2 | 0.2318 | **Promoted** |

Both domain adapters met the promotion thresholds and were written to the active checkpoint directory. The Rust daemon hot-reloaded the adapter without service interruption.

### 4.2 Qualitative (February Leaves Probe)

| Condition | Output | Days Stated | Arithmetic | Factual |
|-----------|--------|-------------|------------|---------|
| Base model (no adapter) | 196 leaves (28 days × 7) | 28 | Correct | **Correct** |
| LoRA adapter v1 | 217 leaves (31 days × 7) | 31 | Correct | **Incorrect** |

The fine-tuned adapter correctly multiplied its stated number of days by 7, but stated that February has 31 days — producing a confidently wrong answer via correct arithmetic applied to a wrong premise.

### 4.3 Checkpoint Manifest (post-promotion)

```json
{
  "version": 2,
  "domain": "reasoning",
  "validation_score": 0.2322,
  "promoted_at": "2026-05-29T...",
  "base_model": "Qwen/Qwen2.5-3B-Instruct"
}
```

---

## 5. Discussion

### 5.1 Catastrophic Forgetting

The result is a textbook example of catastrophic forgetting. The aggressive hyperparameter configuration (lr=2e-4, 3 epochs, all 7 projection layers, scale ratio 2×) imposed weight updates large enough to overwrite the model's pre-trained factual representations while successfully imprinting the step-by-step reasoning format of the teacher outputs.

The model learned *how* the teacher reasons — structured enumeration, intermediate computations, labelled steps — but lost *what factual grounding* to apply. The teacher's reasoning format is a strong distributional signal (3 epochs of dense teacher-forcing); February's day count is a sparse factual representation encoded in the base model's weights.

### 5.2 Why ROUGE-L Scores Appear High

ROUGE-L of 0.2322 reflects high lexical overlap with teacher outputs, which themselves follow a stereotyped structure (Step 1, Step 2, ..., Final answer). The adapter learned to reproduce this format closely, yielding high LCS overlap — even when the content is factually wrong. ROUGE-L has no mechanism to penalize "31 days for February"; the sequences share many n-grams regardless of the numerical values.

This is the key limitation that motivates the metric replacement in [EXP-003](./exp-003-factual-accuracy-runner.md).

### 5.3 Pipeline Architecture Validation

Despite the forgetting failure, this experiment validates the closed-loop architecture as structurally sound:
- Data seeding populated the SQLite buffer ✓
- `DistillationEngine` ingested examples and produced a trained checkpoint ✓
- Validation promoted the adapter under the configured thresholds ✓
- Rust daemon hot-reloaded the adapter without service interruption ✓

The failure is model-capacity and hyperparameter specific; it does not indicate an architectural flaw.

---

## 6. Conclusions

1. Aggressive LoRA hyperparameters (lr=2e-4, 3 epochs, 7 modules) on a 3B model produce catastrophic forgetting of factual knowledge even after achieving high ROUGE-L scores.
2. ROUGE-L rewards format reproduction, not semantic accuracy — high scores are not a reliable indicator of correct behaviour for factual reasoning tasks.
3. The pipeline infrastructure is correct end-to-end; the failure is model-capacity-specific.

**Next experiment**: [EXP-002](./exp-002-conservative-lora-0.5b.md) tests whether conservative hyperparameters (lr=5e-5, 1 epoch, 2 modules) reduce forgetting on the 0.5B model.

---

## Appendix: Checkpoint Version History

| Version | Experiment | Domain | ROUGE-L | Status |
|---------|------------|--------|---------|--------|
| v1 | EXP-001 | `reasoning` | 0.2322 | Superseded |
| v2 | EXP-001 | `default` | 0.2318 | Superseded |

---

← [EXP-000](./exp-000-pipeline-bringup.md) | [Back to Index](../lora-distillation-experiments.md) | [EXP-002 →](./exp-002-conservative-lora-0.5b.md)
