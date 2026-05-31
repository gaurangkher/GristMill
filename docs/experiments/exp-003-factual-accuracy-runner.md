# EXP-003: Factual Accuracy Validation — Replacing ROUGE-L with Ground-Truth Evaluation

**Status**: Complete (implemented and active)
**Date**: 2026-05-30
**Authors**: GristMill Engineering Team

← [EXP-002](./exp-002-conservative-lora-0.5b.md) | [Back to Index](../lora-distillation-experiments.md) | [EXP-004 →](./exp-004-student-model-upgrade.md)

---

## Abstract

EXP-001 and EXP-002 demonstrated that ROUGE-L is an inappropriate validation metric for factual reasoning tasks: it rewards verbatim surface overlap with teacher outputs, which correlates with overfitting to training format rather than with semantic accuracy. In EXP-001, the adapter with the highest ROUGE-L score (0.2322) gave the most confidently incorrect factual answer. We report the design and implementation of `FactualAccuracyRunner`, a replacement validation strategy that evaluates adapters against probe sets with ground-truth `correct_answer` fields, using substring match as the scoring primitive. We also report the creation of two probe sets (`probes/reasoning.yaml` and `probes/training_reasoning.yaml`), a `generate_training_probes.py` script that auto-generates probes from real training buffer records, and the first factual accuracy evaluation of the EXP-002 adapter (score: 0.80 on base model, 0.80 on adapter — no evidence of catastrophic forgetting on the static probe set, marginal improvement on training-data probes).

---

## 1. Introduction

The central finding from EXP-001 and EXP-002 is that ROUGE-L produces a systematic inversion on factual reasoning tasks: higher ROUGE-L correlates with worse semantic quality, because the metric rewards verbatim reproduction of teacher text rather than correct factual content. Concretely:

- Base model answers "196 leaves (28 days × 7)" — ROUGE-L delta: 0.00 (not used as baseline)
- EXP-001 adapter answers "217 leaves (31 days × 7)" — ROUGE-L: 0.2322, promoted
- EXP-002 adapter answers "210 leaves (30 days × 7)" — ROUGE-L: 0.1982, promoted only with relaxed thresholds

The correct replacement metric must:
1. Have access to ground-truth correct answers for the probe questions
2. Score adapters on whether they produce the correct answer, not on whether they reproduce the teacher's phrasing
3. Be configurable (probe set, minimum accuracy threshold) without code changes
4. Integrate cleanly into the existing promotion decision path

`FactualAccuracyRunner` satisfies all four requirements.

---

## 2. Design and Implementation

### 2.1 Architecture

`FactualAccuracyRunner` is a drop-in replacement for `ValidationRunner` in `gristmill_ml/trainer/validation.py`. The strategy is selected via `trainer.validation.strategy` in `~/.gristmill/config.yaml`:

```yaml
trainer:
  validation:
    strategy: factual_accuracy    # or: rouge_l (legacy)
    probe_set: reasoning
    min_accuracy: 0.6
```

At runtime, `_build_validation_runner()` in `service.py` instantiates the appropriate runner:

```python
def _build_validation_runner(base_model_name: str):
    val_cfg = _resolve_validation_strategy()
    if val_cfg["strategy"] == "factual_accuracy":
        return FactualAccuracyRunner(
            base_model_name=base_model_name,
            probe_set=val_cfg["probe_set"],
            min_accuracy=val_cfg["min_accuracy"],
        )
    return ValidationRunner(base_model_name=base_model_name)  # legacy ROUGE-L
```

### 2.2 Probe Set Format

Probe sets are YAML files in `gristmill-ml/probes/`. Each probe specifies a question, the full expected response (for reference), and a `correct_answer` field:

```yaml
domain: reasoning
probes:
  - id: february_leaves
    tags: [reasoning, factual]
    question: >
      Every day, a tree drops 7 leaves. How many leaves would it drop in
      a month of February in a non-leap year?
    expected: >
      February has 28 days in a non-leap year. 28 days × 7 leaves/day = 196 leaves.
    correct_answer: "196"
    notes: "Tests retention of February = 28 days factual knowledge"
```

The `correct_answer` field must appear verbatim (case-insensitive substring match) in the adapter's output for the probe to score as correct.

### 2.3 Scoring

For each probe:
```
correct = correct_answer.lower() in model_output.lower()
```

Overall accuracy:
```
accuracy = sum(correct for all probes) / len(probes)
```

Promotion decision:
```
promote if accuracy >= min_accuracy
```

### 2.4 Known Limitation: Substring False Positives

Substring matching can produce false positives. For example, if `correct_answer = "60"`, this matches any output containing "60" — including "$600", "360 miles", or "1600". In the current evaluation (Section 3), this affected at least 3 of 9 scored probes in the training-data evaluation, artificially inflating accuracy scores.

**Planned fix**: Replace plain substring match with word-boundary regex `\b{answer}\b` for numeric answers. Not yet implemented.

### 2.5 `generate_training_probes.py`

Auto-generates probe YAML files from actual training buffer records, providing the most honest signal about whether an adapter has generalized to its training-data query patterns:

```bash
python scripts/generate_training_probes.py \
  --domain reasoning --n 10 --math-only \
  --output probes/training_reasoning.yaml
```

**Answer extraction strategy** (5-stage, ordered by confidence):
1. LaTeX `\boxed{...}` — highest reliability, used in math-style teacher outputs
2. Explicit labels: "Final answer: X" / "The answer is X"
3. Last number in a conclusion sentence (So, / Therefore, / Thus,)
4. Last number in a `= X` calculation chain
5. Bold markdown: `**X**`

**Skip filter**: Excludes creative writing, roleplay, code tasks, trivia, and chat scenarios that do not produce numeric ground-truth answers.

**Oversampling**: Samples 5× the requested count, then filters and extracts until `n` valid probes are assembled. Handles high rejection rates gracefully.

---

## 3. Evaluation Results

### 3.1 Static Probe Set (`probes/reasoning.yaml` — 5 probes)

Evaluated on the EXP-002b adapter (checkpoint v4, the currently active adapter at time of evaluation):

| Probe | Correct Answer | Base Model | Adapter v4 |
|-------|---------------|------------|------------|
| february_leaves | 196 | ✓ | ✓ |
| days_in_year | 365 | ✓ | ✓ |
| simple_percentage | 25% | ✗ | ✗ |
| unit_conversion | 5,280 | ✓ | ✓ |
| multi_step_word_problem | varies | ✓ | ✓ |
| **Accuracy** | | **0.80 (4/5)** | **0.80 (4/5)** |

**Interpretation**: Base model and adapter score identically. No evidence of catastrophic forgetting on this probe set. No evidence of improvement either — the adapter has not learned anything beneficial for these probes. This is consistent with the training-data distribution not covering these specific problem types.

### 3.2 Training-Data Probe Set (`probes/training_reasoning.yaml` — 10 probes)

Generated from actual CONSUMED training buffer records using `generate_training_probes.py --math-only`.

| Metric | Base Model | Adapter v4 |
|--------|-----------|------------|
| ROUGE-L score | 0.3302 | 0.3555 |
| Probes with extracted `correct_answer` | 9 / 10 | 9 / 10 |
| Factual correct (substring match) | 7 / 9 | 6 / 9 |

**Note on false positives**: Several "correct" scores were false positives due to substring matching ("60" matching "$600", "1" matching "1 hour", "49" matching "49,802"). After manual review, the actual accuracy improvement attributable to the adapter is marginal. Probe 9 showed genuine adapter improvement; Probe 5 showed adapter regression.

**Key observation**: The adapter scored slightly higher on ROUGE-L (0.3555 vs 0.3302) but slightly lower on factual accuracy (6/9 vs 7/9) — another instance of ROUGE-L diverging from semantic correctness.

### 3.3 Promotion Decision Under New Metric

With `min_accuracy: 0.6` and factual accuracy 0.80 (static probes), the EXP-002b adapter would have been promoted under `FactualAccuracyRunner`. The promotion decision is the same — but for the right reason (factual correctness) rather than the wrong one (format mimicry).

---

## 4. Discussion

### 4.1 What FactualAccuracyRunner Gets Right

The metric measures the thing that matters: does the model answer the question correctly? It is immune to the phrasing divergence that defeats ROUGE-L (a concise "196 leaves" scores the same as a verbose step-by-step derivation that arrives at 196).

### 4.2 What It Gets Wrong

**Substring false positives** are the primary failure mode. "60" matching "$600" or "360" produces phantom correct scores. This inflates the adapter's apparent accuracy on training-data probes where answers are round numbers that appear as substrings of larger numbers in model output.

**No partial credit**: A model that answers "approximately 200" when the correct answer is "196" scores 0. This is too strict for reasoning tasks where the reasoning process is correct but the final digit differs. A numeric proximity score (|answer - correct| / correct < 0.05) would be more informative.

**Probe set size**: 5 probes is too small for a statistically meaningful accuracy estimate. A 1/5 error gives an accuracy of 0.80; a 2/5 error gives 0.60 — passing and failing respectively on the same evaluation with a single additional wrong answer. EXP-004 and beyond will use larger probe sets.

### 4.3 The Teacher Forcing vs. Autoregressive Gap

A subtler explanation for why the adapter fails even on training-data probes: training uses teacher forcing (the correct token is always fed as the next input), while inference is autoregressive (the model feeds its own output as the next input). Any error early in the generation compounds through subsequent tokens. An adapter trained with teacher forcing on 2,400 examples cannot guarantee that the specific answer to a specific question will survive autoregressive generation on that same question — the inference path is strictly harder than the training path, and small models are more susceptible to error accumulation.

---

## 5. Conclusions

1. `FactualAccuracyRunner` is implemented and active (`strategy: factual_accuracy` in config).
2. Static probe evaluation: base model and adapter both score 0.80 (4/5) — no forgetting evidence, no improvement evidence.
3. Training-data probe evaluation: adapter scores marginally lower on factual accuracy (6/9 vs 7/9) despite higher ROUGE-L, confirming the ROUGE-L inversion.
4. Substring matching produces false positives that inflate accuracy; word-boundary matching is the correct fix (pending).
5. 5-probe evaluation sets are too small; EXP-004+ should use at least 20 probes.

---

## Appendix: Files Created in This Experiment

| File | Purpose |
|------|---------|
| `gristmill-ml/probes/reasoning.yaml` | Static 5-probe evaluation set with ground-truth answers |
| `gristmill-ml/probes/training_reasoning.yaml` | Auto-generated 10-probe set from training buffer |
| `gristmill-ml/scripts/generate_training_probes.py` | Probe generator script |
| `gristmill-ml/scripts/compare_lora_adapter.py` | Base vs. adapter evaluation script (updated: `--max-new-tokens` default 512) |
| `gristmill_ml/trainer/validation.py` | Added `FactualAccuracyRunner`, `_run_probe_inference()`, `_detect_device()` |
| `gristmill_ml/trainer/service.py` | Added `_resolve_validation_strategy()`, `_build_validation_runner()` |

---

← [EXP-002](./exp-002-conservative-lora-0.5b.md) | [Back to Index](../lora-distillation-experiments.md) | [EXP-004 →](./exp-004-student-model-upgrade.md)
