# Experiment B — Query Distribution Shift

**Status:** Proposed  
**Priority:** 1 of 6 (Highest)  
**Category:** Robustness / Real-World Deployment  
**Relevant paper section:** §5 Experiments — Robustness

---

## Motivation

The Experiment 2 flywheel assumes a stationary query distribution — the user always asks about the same domain. Real users don't behave this way. A developer who has trained their grinder on Python for weeks might suddenly need to ask about Rust, or a medical user might pivot from radiology reports to drug interaction queries.

The question is whether GristMill handles this gracefully. The Sieve *should* escalate on out-of-distribution (OOD) queries because the grinder has low confidence on unfamiliar material. But if the Self-REF token is poorly calibrated on OOD inputs, the grinder may **confidently hallucinate** rather than correctly escalate — and Experiment 2's stationary setup would never catch this.

This is likely the first question a reviewer will ask about real-world robustness.

---

## Core Question

> When a user's query distribution shifts to an unseen domain, does GristMill correctly escalate to the teacher, or does the grinder hallucinate with false confidence?

---

## Setup

### Phase 1: Domain A Training (In-Distribution)

- Train grinder on **Python development** queries for 10 distillation cycles
- Use ShareGPT/LMSYS filtered to Python-related queries (~500 training examples)
- Evaluate: establish baseline autonomy rate and accuracy on Python held-out set

### Phase 2: Domain B Introduction (Out-of-Distribution)

At cycle 10, introduce queries from **Rust development** (zero overlap with training data). Do not train on Rust — just run inference.

Measure on a mixed query stream (50% Python, 50% Rust):
- Escalation rate per domain
- Accuracy per domain
- Confidence score distribution per domain

### Phase 3: Cross-Domain Distillation

Begin distillation on Rust queries (cycles 11–20). Measure:
- Rate of Rust autonomy growth
- Retention of Python accuracy (catastrophic forgetting interaction)
- Whether Sieve calibration recovers as Rust becomes in-distribution

### Domain Pairs to Test

Run at least 2 domain pairs to check generality:

| Pair | Domain A | Domain B |
|---|---|---|
| 1 | Python development | Rust development |
| 2 | Medical report writing | Legal document drafting |

---

## Metrics

| Metric | In-distribution (A) | Out-of-distribution (B) |
|---|---|---|
| **Escalation rate** | Should decrease over cycles | Should be high (correct behavior) |
| **Accuracy** | Should be high | Should be high *because* teacher handles it |
| **False confidence rate** | Low | Key failure metric — grinder answers OOD confidently but wrongly |
| **ECE** | Stable | May be poorly calibrated — cross-reference Exp A |

### Key failure signal

`false_confidence_rate = P(confidence > threshold AND answer is wrong AND query is OOD)`

A high false confidence rate means GristMill is dangerous on distribution shift — the grinder answers when it shouldn't.

---

## Baselines

| Baseline | Description |
|---|---|
| **Always-teacher** | 100% escalation — correct but expensive |
| **Always-grinder** | 0% escalation — shows raw OOD accuracy without routing |
| **Static threshold** | Sieve at fixed threshold from Experiment 1 |
| **Oracle router** | Upper bound — perfect knowledge of which queries are OOD |

---

## Expected Results

**Hypothesis:** The Self-REF token naturally produces lower confidence on OOD queries because the grinder has not seen that vocabulary/reasoning pattern, so escalation rate rises correctly on Domain B without any additional mechanism. False confidence rate should be low.

If this holds, it is a strong result — GristMill gets OOD robustness "for free" from the confidence mechanism. If it doesn't hold, a domain-shift detector (e.g., embedding distance from training distribution) is needed as an additional Sieve component.

---

## Implementation Notes

- Filter ShareGPT by programming language tags for Python/Rust split
- For medical/legal split, use a small prompt-based classifier to tag LMSYS Arena data
- Reuse `experiment1.py` evaluation loop; add an `is_ood` flag per query
- The false confidence rate metric requires storing `(confidence, is_correct, is_ood)` tuples — add to results CSV schema

---

## Connection to Other Experiments

- **Most important** experiment for reviewer credibility — run before Experiment F (generalization)
- **Interacts with** Experiment 4 (catastrophic forgetting) in Phase 3 — the cross-domain distillation phase is essentially Experiment 4 with a cold start
- **Informs** Experiment D (offline grace period) — OOD queries during offline periods are the worst case

---

## Open Questions

- Should we test gradual distribution shift (queries slowly drifting toward Rust) vs. hard cutover?
- Is programming language a good proxy for domain shift, or is the semantic distance too small (Python and Rust are structurally similar)?
- Would a user who *knows* they're asking OOD questions behave differently — e.g., explicitly flagging "new topic"? Should GristMill have a manual override?
