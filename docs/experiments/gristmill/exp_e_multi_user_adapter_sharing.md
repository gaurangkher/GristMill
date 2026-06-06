# Experiment E — Multi-User Adapter Sharing

**Status:** Proposed  
**Priority:** 6 of 6 (Lower — overlaps with LoRA personalization experiment)  
**Category:** Scalability / Storage Efficiency  
**Relevant paper section:** §6 Discussion — Deployment at Scale

---

## Motivation

The LoRA personalization depth experiment establishes a per-user adapter cost of ~8MB at rank 8. At 100k users, that is ~800GB of adapter storage — feasible but not trivial. At 1M users it becomes a serious infrastructure concern.

The question is whether users within the same domain (e.g., all Python developers, all radiologists) can share a single **domain adapter** and only require a small personal **delta adapter** on top. If users' personalization needs are partly shared (same domain vocabulary, same output format conventions) and only partly personal (individual coding style, specific project patterns), a two-tier adapter hierarchy could reduce storage by an order of magnitude.

This experiment tests whether the quality of a shared domain adapter + thin personal delta matches a fully personal adapter at matched storage budgets.

---

## Core Question

> Can a shared domain-level adapter plus a thin personal delta adapter match the quality of a fully personal adapter, and what storage savings does this enable at scale?

---

## Setup

### Adapter Hierarchy

**Tier 1 — Domain adapter:** Trained on a large pool of domain-representative queries (not tied to any specific user). Captures domain vocabulary, output format conventions, common reasoning patterns.

**Tier 2 — Personal delta adapter:** Tiny adapter (rank 1–2) trained on top of the frozen domain adapter. Captures individual style idiosyncrasies.

### User Personas

Use the 5–10 synthetic personas from the LoRA personalization depth experiment. Group into domain clusters:

| Cluster | Personas |
|---|---|
| Software engineering | Python developer, Rust developer (from Exp B) |
| Medical | Radiologist, general practitioner |
| Legal/Analytical | Lawyer, data analyst |

### Training Conditions

| Condition | Description | Storage per user |
|---|---|---|
| **Fully personal (rank 8)** | Single adapter per user, rank 8 | ~8 MB |
| **Shared domain + delta (rank 4 + rank 2)** | Domain adapter shared; personal delta rank 2 | ~4 MB shared + ~2 MB personal = ~2 MB marginal |
| **Shared domain only** | No personal delta | ~0 MB marginal (fully shared) |
| **Fully personal (rank 2)** | Single adapter, rank 2 — storage-matched to delta condition | ~2 MB |

The key comparison is **shared domain + rank-2 delta** vs. **fully personal rank-2** at the same marginal storage cost.

### Domain Adapter Training

- Pool all training examples from all personas within a cluster (e.g., all software engineering examples)
- Train a rank-4 domain adapter on the pooled data
- Freeze domain adapter; train personal delta adapters on top per-user

---

## Metrics

| Metric | Description |
|---|---|
| **Style score** | Teacher-as-judge (1–5) on held-out personal examples |
| **Marginal storage per user** | Personal adapter size only (domain adapter is amortized) |
| **Total storage at N users** | `domain_adapter_size + N × personal_delta_size` |
| **Quality gap vs. fully personal rank-8** | How much quality is lost by the shared approach |

### Storage scaling analysis

Plot total storage vs. number of users for each condition. The shared approach has a fixed domain adapter cost plus linear personal delta cost — the crossover point where it beats fully-personal adapters is the key result.

```
fully_personal_cost(N)   = N × 8 MB
shared_domain_cost(N)    = 4 MB + N × 2 MB
```

Crossover: at all N > 2 users, shared is cheaper. The question is whether the quality tradeoff is acceptable.

---

## Expected Results

**Hypothesis:** The shared domain + personal delta condition approaches fully personal rank-8 quality (~90–95% of style score) at ~25% of the marginal storage cost. Fully personal rank-2 at matched storage underperforms because it must capture both domain and personal signal in a very low-rank adapter.

| Condition | Style score | Marginal storage | Total @ 100k users |
|---|---|---|---|
| Fully personal rank-8 | 4.3 | 8 MB | 800 GB |
| Shared domain + rank-2 delta | ~4.0–4.1 | 2 MB | 4 MB + 200 GB |
| Fully personal rank-2 | ~3.5–3.7 | 2 MB | 200 GB |
| Shared domain only | ~3.2–3.5 | 0 MB | 4 MB |

---

## Implementation Notes

- Adapter stacking (domain + delta) is supported in HuggingFace PEFT via `PeftModel` with multiple adapters — verify this works with LoRA before committing to the experiment design
- Domain adapter training uses the same `SFTTrainer` pipeline as existing experiments
- Personal delta training: load base model + frozen domain adapter, then add and train personal delta adapter
- Evaluate by merging domain + delta into a single set of weights for inference efficiency

---

## Connection to Other Experiments

- **Requires:** LoRA personalization depth experiment (reuses personas and evaluation setup)
- **Extends:** The storage cost analysis in the personalization experiment with a new storage-reduction mechanism
- **Informs:** Production deployment recommendations in the paper's conclusion

---

## Open Questions

- Does the quality of the domain adapter depend heavily on the size of the pooled training set? Is 5 users worth of data enough, or do you need 50+?
- Can domain adapters be updated incrementally as new users join, without retraining from scratch?
- Is there a privacy concern in training a shared domain adapter on data from multiple users, even if that data is synthetic? Worth addressing in the paper.
- Would a hierarchical three-tier structure (base model → domain adapter → personal delta) generalize to more fine-grained sub-domain clustering (e.g., Python web developer vs. Python ML engineer)?
