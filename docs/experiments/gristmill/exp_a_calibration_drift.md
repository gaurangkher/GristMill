# Experiment A — Sieve Confidence Calibration Drift

**Status:** Proposed  
**Priority:** 2 of 6 (High)  
**Category:** Sieve Robustness  
**Relevant paper section:** §5 Experiments / §6 Analysis — Failure Modes

---

## Motivation

The Self-REF confidence token drives all routing decisions in GristMill. Experiment 1 validates that it is well-calibrated at a snapshot in time. But the grinder is a moving target — it improves over distillation cycles, and a model that becomes more capable might also become overconfident, routing queries it still gets wrong without escalating to the teacher.

This is a subtle but critical failure mode: **autonomy appears to grow while quality silently degrades.** No existing experiment catches it.

---

## Core Question

> Does the Self-REF confidence token remain well-calibrated as the grinder improves over distillation cycles, or does calibration drift introduce silent quality regressions?

---

## Setup

### Piggyback on Experiment 2

This experiment requires no separate training runs. At each checkpoint saved during the Experiment 2 flywheel (cycles 0, 5, 10, 20), run a calibration evaluation pass.

### Calibration Metric: Expected Calibration Error (ECE)

Bin the grinder's outputs by stated confidence (e.g., 10 equal-width bins from 0.0–1.0). Within each bin, measure actual accuracy. ECE is the weighted average gap between stated confidence and true accuracy:

```
ECE = Σ (|bin| / N) × |accuracy(bin) − confidence(bin)|
```

A perfectly calibrated model has ECE = 0. A model that says "90% confident" and is right 60% of the time has high ECE.

### Evaluation Set

Use the same held-out benchmark from Experiment 2 (MMLU + GSM8K subset, ~500 questions). Do **not** use training data.

### Variables

| Variable | Values |
|---|---|
| Distillation cycle | 0, 5, 10, 20 |
| Sieve threshold | Fixed at optimal value found in Experiment 1 |
| Grinder model | Qwen2.5-3B (consistent with baseline) |

---

## Metrics

| Metric | Description |
|---|---|
| **ECE per cycle** | Primary calibration metric; lower is better |
| **Reliability diagram** | Plot confidence bins vs. accuracy at each cycle checkpoint |
| **Overconfidence rate** | % of queries where stated confidence > 0.8 but answer is wrong |
| **Autonomy vs. quality scatter** | Plot autonomy rate against actual accuracy per cycle — divergence signals a problem |

---

## Expected Results

Two possible outcomes, both publishable:

**Outcome 1 (Calibration holds):** ECE remains stable or improves across cycles. This validates the Self-REF mechanism as robust and strengthens the paper's core claim.

**Outcome 2 (Calibration drifts):** ECE increases over cycles, particularly in the high-confidence bins. This motivates a recalibration mechanism (e.g., periodic temperature scaling of the confidence token) as a GristMill component, which is itself a contribution.

Either result belongs in the paper. Outcome 2 is arguably more interesting.

---

## Implementation Notes

- Save grinder checkpoints at each cycle during Experiment 2 — add `save_pretrained()` calls at cycles 0, 5, 10, 20
- The Self-REF parser in `sieve.py` already extracts confidence scores; feed these directly into the ECE calculation
- Use `netcal` Python library for ECE computation and reliability diagram generation
- Add calibration plots to `analysis.py` alongside existing Experiment 2 figures

---

## Connection to Other Experiments

- **Requires:** Experiment 2 checkpoints
- **Informs:** Experiment 1 (whether threshold needs dynamic adjustment over time)
- **Related:** If calibration drifts, Experiment D (offline grace period) becomes more dangerous — the grinder is both overconfident and not improving

---

## Open Questions

- Should recalibration (temperature scaling) be tested as a mitigation, or is that scope for a follow-up paper?
- Is ECE the right metric, or should we prefer Maximum Calibration Error (MCE) which focuses on worst-case bins?
- Does calibration drift differently across domains (coding vs. creative writing)?
