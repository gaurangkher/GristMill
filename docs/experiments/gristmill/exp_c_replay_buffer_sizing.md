# Experiment C — Replay Buffer Size vs. Forgetting Rate

**Status:** Proposed  
**Priority:** 4 of 6 (Medium)  
**Category:** Catastrophic Forgetting Mitigation  
**Relevant paper section:** §5 Experiments — Catastrophic Forgetting (extends Experiment 4)

---

## Motivation

Experiment 4 establishes that the replay buffer + EWC-LoRA combination retains >90% of domain-A performance after domain-B training. But it treats the buffer as a fixed design choice — it does not answer the question every engineer deploying GristMill will ask: **how big does the buffer actually need to be?**

Buffer size has direct cost implications. Each stored example occupies memory and adds compute to each training step (the replay examples must be included in every batch). A result quantifying the minimum viable buffer size is an actionable engineering specification, not just a validation result.

---

## Core Question

> What is the minimum replay buffer size that effectively prevents catastrophic forgetting, and where does the retention curve plateau?

---

## Setup

### Base Condition (identical to Experiment 4)

- Train grinder on coding queries for 5 cycles (Domain A)
- Then train on creative writing queries for 5 cycles (Domain B)
- Measure coding performance before and after writing training

### Independent Variable: Buffer Size

Run the full Domain A → Domain B sequence at each buffer size:

```
buffer_sizes = [0, 8, 16, 32, 64, 128, 256]
```

`0` = no replay buffer (ablation baseline)  
`256` = generous upper bound (expect diminishing returns well before this)

### Fixed Variables

- EWC-LoRA: **enabled** in all conditions (isolates buffer size effect)
- Domain A training: 5 cycles, ~500 queries
- Domain B training: 5 cycles, ~500 queries
- Replay sampling: uniform random from Domain A buffer

### Replay Buffer Sampling Strategy (secondary variable)

At buffer size 64, also compare sampling strategies:
- **Uniform random** (baseline)
- **Recency-weighted** (favor recent examples)
- **Difficulty-weighted** (favor examples where grinder confidence was low)
- **Diversity-maximizing** (maximize embedding distance between buffer examples)

---

## Metrics

| Metric | Description |
|---|---|
| **Domain-A retention rate** | Accuracy on Domain-A held-out set after Domain-B training, normalized to pre-B baseline |
| **Domain-B acquisition rate** | Accuracy on Domain-B after training — ensure buffer doesn't slow new learning |
| **Training time per cycle** | Wall-clock overhead added by replay at each buffer size |
| **Memory footprint** | Buffer storage in MB at each size |

### Retention curve

Plot retention rate vs. buffer size. Expect a knee — rapid improvement from 0 to ~32, then plateau. The knee is the publishable result.

---

## Expected Results

| Buffer size | Domain-A retention | Notes |
|---|---|---|
| 0 | ~40–60% | Catastrophic forgetting baseline |
| 8 | ~70–75% | Meaningful improvement |
| 16 | ~80–85% | Approaching target |
| 32 | ~88–92% | Near plateau |
| 64 | ~90–93% | Plateau region |
| 128 | ~91–93% | Diminishing returns |
| 256 | ~92–94% | Marginal gain over 64 |

**Target claim:** "A replay buffer of 32–64 examples per domain is sufficient to retain >90% of prior performance; buffers beyond 64 examples yield diminishing returns."

---

## Sampling Strategy Expected Results

At buffer size 64, difficulty-weighted and diversity-maximizing sampling are hypothesized to outperform uniform random, particularly at smaller buffer sizes where example selection matters more.

---

## Implementation Notes

- The replay buffer is already implemented in the GristMill trainer — this experiment only varies its `max_size` parameter and sampling policy
- Run each buffer size condition 3 times with different random seeds; report mean ± std
- Add buffer size as a tracked hyperparameter in the results CSV
- Training time overhead is measurable with `time.perf_counter` around the trainer loop — already instrumented in `experiment1.py`

---

## Connection to Other Experiments

- **Extends:** Experiment 4 (same setup, more granular on buffer size)
- **Informs:** Experiment E (multi-user adapter sharing) — shared domain adapters need a buffer policy too
- **Required for:** Any production deployment recommendation in the paper's conclusion

---

## Open Questions

- Should EWC-LoRA strength (λ) be co-varied with buffer size, or is isolating buffer size sufficient?
- Does the optimal buffer size scale with the number of distillation cycles — i.e., do longer-trained grinders need larger buffers?
- Is there a theoretical lower bound on buffer size based on the LoRA rank and number of trainable parameters?
