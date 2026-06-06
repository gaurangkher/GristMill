# Experiment F — Grinder Model Family Sensitivity

**Status:** Proposed  
**Priority:** 3 of 6 (High)  
**Category:** Generalization / Reviewer Robustness  
**Relevant paper section:** §5 Experiments — Generalization / Appendix

---

## Motivation

All existing GristMill experiments use Qwen2.5-3B as the grinder. This is a reasonable default — it is capable, open, and consistent with the teacher family (Qwen2.5-32B). But a reviewer will immediately ask: **are these results specific to Qwen2.5, or do they generalize?**

If the Self-REF confidence mechanism, distillation flywheel, and Sieve calibration only work well with Qwen2.5-3B, then GristMill is a narrow system, not a general framework. If they work across model families, the paper's claims are significantly stronger and the system is practically more useful (users may prefer Llama or Phi for licensing or performance reasons).

This experiment replaces a likely reviewer objection with a positive generalization result.

---

## Core Question

> Do GristMill's core properties — Sieve calibration accuracy and distillation flywheel growth — hold consistently across different 3B-class student model families?

---

## Setup

### Student Models (Grinders)

| Model | Family | Parameters | Notes |
|---|---|---|---|
| `Qwen2.5-3B` | Qwen | 3B | Baseline (same family as teacher) |
| `Llama-3.2-3B` | Llama | 3B | Different architecture, different tokenizer |
| `Phi-3.5-mini` | Phi | 3.8B | Microsoft; known for strong reasoning per parameter |

All three are open-weight, commercially usable, and fit on a single consumer GPU.

### Teacher Model (Fixed)

`Qwen2.5-32B` — held constant across all conditions. This isolates the student model as the variable. A cross-family teacher experiment (e.g., Llama teacher + Qwen student) is out of scope for this paper.

### Experiments to Replicate

Replicate two core experiments for each student model:

**Experiment 1 subset:** Sieve calibration accuracy vs. routing cost
- Vary Sieve threshold from 0.3 to 0.9
- Measure accuracy, escalation rate, and latency
- Produce Pareto curve per model family

**Experiment 2 subset:** Distillation flywheel — autonomy growth over 10 cycles
- Use same query stream (ShareGPT filtered subset)
- Measure autonomy rate and quality per cycle

Do **not** replicate the full experiment suite (Experiments 3–6) — that would be excessive. The Sieve and flywheel results are the paper's core claims; generalization of those two is sufficient.

---

## Metrics

### Sieve Calibration (Experiment 1 replication)

| Metric | Description |
|---|---|
| **Pareto curve position** | Does each model land on the accuracy vs. escalation frontier? |
| **Optimal threshold** | Does the best threshold differ across families? |
| **Latency** | Per-model inference time affects real-world usability |

### Flywheel Growth (Experiment 2 replication)

| Metric | Description |
|---|---|
| **Autonomy at cycle 10** | Primary measure — does the flywheel work for all models? |
| **Convergence rate** | Does one model family learn faster from distillation? |
| **Quality floor** | Minimum accuracy across cycles — does any model regress badly? |

---

## Expected Results

**Hypothesis:** The Sieve calibration and flywheel growth hold across all three model families, with Phi-3.5-mini showing marginally faster convergence due to its stronger reasoning-per-parameter ratio. Llama-3.2-3B may require a slightly different optimal Sieve threshold due to tokenizer differences affecting the Self-REF token probability distribution.

| Model | Sieve Pareto | Autonomy @ cycle 10 | Notes |
|---|---|---|---|
| Qwen2.5-3B | On frontier | ~65% | Baseline |
| Llama-3.2-3B | On frontier | ~60–65% | Minor threshold shift expected |
| Phi-3.5-mini | On frontier | ~65–70% | Potentially faster convergence |

If any model falls significantly off the Pareto frontier or shows flat flywheel growth, that is a meaningful negative result worth reporting — it would suggest the Self-REF mechanism has architectural prerequisites.

---

## Self-REF Token Compatibility

One risk: the Self-REF confidence token mechanism may be tuned to Qwen2.5's token probability distribution. Validate for each model:

1. Does the model reliably produce a parseable confidence token?
2. Is the confidence token's probability well-separated between correct and incorrect answers?
3. Does the token need re-tuning (different prompt format) for non-Qwen models?

If re-tuning is needed, document the per-family prompt adjustments in the appendix — this is expected and not a failure of the approach.

---

## Implementation Notes

- All three models are available on HuggingFace in 4-bit quantized form for single-GPU inference
- The `clients.py` model client interface already abstracts model loading — add Llama and Phi as new client configs in `config.py`
- Tokenizer differences require verifying that the Self-REF prompt template produces valid outputs for each model; run a quick sanity check before full experiment
- Total compute: ~3× the cost of Experiments 1+2 combined — plan accordingly

---

## Reporting

Present results as a comparison table + overlaid Pareto curves (one curve per model family on the same axes). A tight cluster of curves is the ideal figure — it visually demonstrates generalization without requiring prose explanation.

---

## Connection to Other Experiments

- **Requires:** Experiments 1 and 2 infrastructure (reuses scripts directly)
- **Strengthens:** Every other experiment's claims by establishing that results are not Qwen2.5-specific
- **Suggested placement:** Either as a standalone generalization section (§5.7) or in the Appendix if space is tight

---

## Open Questions

- Should the teacher also be varied (e.g., Llama-3.1-70B as teacher for Llama-3.2-3B student) to test same-family vs. cross-family distillation? This is a separate experiment but related.
- Does the capacity gap finding (Experiment 3) also hold for Llama and Phi families, or is the optimal teacher-student size ratio architecture-dependent?
- Are there licensing constraints on any of the three student models that affect redistribution of fine-tuned adapters? Worth a footnote.
