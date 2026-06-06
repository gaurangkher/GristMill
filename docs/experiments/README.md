# GristMill — Experiments Registry

This is the single source of truth for all experiments across the GristMill project. It tracks experiment status, ownership, and maps experiments to publishable papers.

Last updated: 2026-06-05

---

## Table of Contents

1. [Experiment Register](#experiment-register)
2. [Status Definitions](#status-definitions)
3. [Paper Register](#paper-register)
4. [Experiment × Paper Matrix](#experiment--paper-matrix)

---

## Experiment Register

### Original Experiments — Core Architecture Validation

These six experiments validate GristMill's primary claims and are required for any submission.

| ID | Title | Category | Status | Priority | Notes |
|---|---|---|---|---|---|
| **EXP-1** | Sieve Calibration Accuracy vs. Routing Cost | Sieve | 🟡 In Progress | Critical | Scripts in `src/experiments/experiment1.py`; 45 smoke tests passing |
| **EXP-2** | Distillation Flywheel: Autonomy Growth Over Time | Distillation | 🔵 Designed | Critical | Depends on EXP-1 infrastructure |
| **EXP-3** | Capacity Gap in Practice | Distillation | 🔵 Designed | Critical | Requires multi-GPU setup for 72B teacher |
| **EXP-4** | Catastrophic Forgetting Under Domain Shift | Forgetting | 🔵 Designed | High | Baseline for EXP-C |
| **EXP-5** | Cold Start Characterization | Deployment | 🔵 Designed | High | |
| **EXP-6** | gristmill-trainer Process Isolation Benefit | Systems | 🔵 Designed | High | Requires single-process comparison implementation |

---

### Extension Experiments — Robustness, Deployment & Generalization

Proposed experiments that address reviewer concerns and strengthen the paper's claims.

| ID | Title | Category | Status | Priority | Doc |
|---|---|---|---|---|---|
| **EXP-A** | Sieve Confidence Calibration Drift | Sieve Robustness | 🟣 Proposed | High | [exp_a_calibration_drift.md](./gristmill/exp_a_calibration_drift.md) |
| **EXP-B** | Query Distribution Shift | Robustness | 🟣 Proposed | Highest | [exp_b_distribution_shift.md](./gristmill/exp_b_distribution_shift.md) |
| **EXP-C** | Replay Buffer Size vs. Forgetting Rate | Forgetting | 🟣 Proposed | Medium | [exp_c_replay_buffer_sizing.md](./gristmill/exp_c_replay_buffer_sizing.md) |
| **EXP-D** | Teacher Availability & Offline Grace Period | Deployment | 🟣 Proposed | Medium | [exp_d_offline_grace_period.md](./gristmill/exp_d_offline_grace_period.md) |
| **EXP-E** | Multi-User Adapter Sharing | Scalability | 🟣 Proposed | Lower | [exp_e_multi_user_adapter_sharing.md](./gristmill/exp_e_multi_user_adapter_sharing.md) |
| **EXP-F** | Grinder Model Family Sensitivity | Generalization | 🟣 Proposed | High | [exp_f_model_family_sensitivity.md](./gristmill/exp_f_model_family_sensitivity.md) |

---

### LoRA Characterization Experiments

| ID | Title | Category | Status | Priority | Doc |
|---|---|---|---|---|---|
| **EXP-L1** | LoRA Personalization Depth vs. Adapter Size | LoRA | 🟣 Proposed | High | [lora_personalization_depth.md](./gristmill/lora_personalization_depth.md) |

---

## Status Definitions

| Symbol | Status | Meaning |
|---|---|---|
| 🟣 | **Proposed** | Experiment designed; not yet scheduled or resourced |
| 🔵 | **Designed** | Full design complete; ready to implement |
| 🟡 | **In Progress** | Implementation underway or partially run |
| 🟠 | **Blocked** | Waiting on dependency (hardware, data, another experiment) |
| 🟢 | **Complete** | Results collected and analysed |
| ⚫ | **Abandoned** | Deprioritised or superseded; kept for record |

---

## Paper Register

Each paper below can be written from one or more experiments. Papers are ordered from fastest-to-publish to most ambitious.

---

### PAPER-1 — GristMill arXiv Preprint (Fastest path to publication)

**Title:** *GristMill: Continuous Personal Knowledge Distillation for LLM-Independent Edge Inference*  
**Venue target:** arXiv (no deadline)  
**Estimated effort:** 4–6 weeks from EXP-1 completion  
**Status:** 🟡 In Progress (blocked on EXP-1 results)

**Required experiments:**

| Exp | Role in paper |
|---|---|
| EXP-1 | Core result — Sieve Pareto frontier |
| EXP-2 | Core result — flywheel growth curve |
| EXP-3 | Supporting — capacity gap validation |

**Description:** Establishes GristMill as a system with two validated properties: a well-calibrated router (EXP-1) and a self-improving student (EXP-2). EXP-3 justifies the choice of 32B teacher. Sufficient for a credible preprint that stakes a claim in the continuous distillation space.

---

### PAPER-2 — Workshop Paper: Robustness of Continuous Distillation Systems

**Title:** *When the Teacher Is Away: Robustness Properties of Continuous Knowledge Distillation at the Edge*  
**Venue target:** ICML or NeurIPS workshop (Efficient ML, On-Device ML)  
**Estimated effort:** 6–8 weeks  
**Status:** 🔵 Designed (waiting on EXP-1, EXP-2, EXP-B)

**Required experiments:**

| Exp | Role in paper |
|---|---|
| EXP-1 | Baseline Sieve performance |
| EXP-2 | Baseline flywheel |
| EXP-B | Primary contribution — OOD robustness characterisation |
| EXP-A | Supporting — calibration drift as a failure mode |
| EXP-D | Supporting — offline resilience |

**Description:** Focuses on the gap between lab results (stationary query distributions, always-available teacher) and real deployment. EXP-B is the headline result; EXP-A and EXP-D provide a comprehensive failure mode analysis. A natural workshop paper that positions GristMill as deployment-ready rather than just academically interesting.

---

### PAPER-3 — Full Conference Paper: GristMill System Paper

**Title:** *GristMill: Continuous Personal Knowledge Distillation for LLM-Independent Edge Inference*  
**Venue target:** MLSys, OSDI, or USENIX ATC (systems track)  
**Estimated effort:** 12–16 weeks  
**Status:** 🔵 Designed (waiting on all core experiments)

**Required experiments:**

| Exp | Role in paper |
|---|---|
| EXP-1 | §5.1 — Sieve calibration |
| EXP-2 | §5.2 — Distillation flywheel |
| EXP-3 | §5.3 — Capacity gap |
| EXP-4 | §5.4 — Catastrophic forgetting |
| EXP-5 | §5.5 — Cold start |
| EXP-6 | §5.6 — Process isolation |
| EXP-B | §5.7 — Robustness under distribution shift |
| EXP-F | §5.8 / Appendix — Generalization across model families |

**Description:** The complete GristMill paper. EXP-B and EXP-F are included to pre-empt the two most likely reviewer objections (real-world robustness; Qwen2.5 specificity). EXP-A and EXP-C can be included as appendix material or held for a follow-up.

---

### PAPER-4 — LoRA Efficiency Paper (Standalone, no GristMill branding required)

**Title:** *How Small Is Small Enough? Characterising Minimum LoRA Rank for Personalised Language Model Adaptation*  
**Venue target:** EMNLP, ACL Findings, or ICLR (parameter-efficient fine-tuning track)  
**Estimated effort:** 8–10 weeks  
**Status:** 🟣 Proposed

**Required experiments:**

| Exp | Role in paper |
|---|---|
| EXP-L1 | Primary contribution — rank vs. quality sweep |
| EXP-E | §4 extension — multi-user sharing enabled by small adapters |
| EXP-C | Supporting — buffer sizing complements adapter sizing story |

**Description:** A self-contained contribution to the PEFT literature. The core finding (rank 4–8 captures ~90% of personalisation quality at a fraction of rank-64 cost) is novel and practically useful well beyond GristMill. EXP-E extends this into a systems argument about deployability at scale. Does not require any GristMill-specific infrastructure beyond a training loop and LoRA.

---

### PAPER-5 — Continual Learning Paper (Forgetting mitigation focus)

**Title:** *Taming Catastrophic Forgetting in Online Knowledge Distillation: Buffer Sizing, EWC-LoRA, and Domain Shift*  
**Venue target:** CoLLAs (Conference on Lifelong Learning Agents), ContinualAI workshop  
**Estimated effort:** 8–10 weeks  
**Status:** 🟣 Proposed

**Required experiments:**

| Exp | Role in paper |
|---|---|
| EXP-4 | Baseline — forgetting with and without mitigation |
| EXP-C | Primary contribution — buffer size characterisation |
| EXP-B | Extension — forgetting under distribution shift (Phase 3) |
| EXP-2 | Context — flywheel as the motivation for continual learning |

**Description:** Targets the continual learning community rather than systems or NLP. The novel contribution is the buffer sizing characterisation (EXP-C) — turning a qualitative mitigation into a quantitative specification. EXP-B Phase 3 (cross-domain distillation after OOD shift) provides an additional novel scenario not covered in standard catastrophic forgetting benchmarks.

---

### PAPER-6 — Generalization & Transferability Paper

**Title:** *Architecture-Agnostic Continuous Distillation: Do GristMill's Properties Generalise Across Model Families?*  
**Venue target:** arXiv follow-up / EMNLP short paper  
**Estimated effort:** 4–6 weeks (after PAPER-1 or PAPER-3)  
**Status:** 🟣 Proposed (natural follow-up to main paper)

**Required experiments:**

| Exp | Role in paper |
|---|---|
| EXP-F | Primary contribution — Sieve + flywheel across Qwen / Llama / Phi |
| EXP-1 | Baseline (already run) |
| EXP-2 | Baseline (already run) |

**Description:** A short follow-up paper that answers the generalization question left open by the main submission. Low incremental effort once EXP-1 and EXP-2 are done — EXP-F only requires rerunning those two experiments with two additional student models.

---

## Experiment × Paper Matrix

✅ = required &nbsp;&nbsp; 🔷 = supporting / appendix &nbsp;&nbsp; — = not used

| | P1 arXiv | P2 Workshop | P3 Conference | P4 LoRA | P5 Continual | P6 Generalization |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **EXP-1** | ✅ | ✅ | ✅ | — | — | ✅ |
| **EXP-2** | ✅ | ✅ | ✅ | — | 🔷 | ✅ |
| **EXP-3** | ✅ | — | ✅ | — | — | — |
| **EXP-4** | — | — | ✅ | — | ✅ | — |
| **EXP-5** | — | — | ✅ | — | — | — |
| **EXP-6** | — | — | ✅ | — | — | — |
| **EXP-A** | — | ✅ | 🔷 | — | — | — |
| **EXP-B** | — | ✅ | ✅ | — | ✅ | — |
| **EXP-C** | — | — | 🔷 | ✅ | ✅ | — |
| **EXP-D** | — | ✅ | — | — | — | — |
| **EXP-E** | — | — | — | ✅ | — | — |
| **EXP-F** | — | — | ✅ | — | — | ✅ |
| **EXP-L1** | — | — | — | ✅ | — | — |

---

## Recommended Execution Order

Based on dependencies and paper priorities:

```
Phase 1 (unblock PAPER-1 arXiv)
  EXP-1 → EXP-2 → EXP-3

Phase 2 (unblock PAPER-3 Conference + PAPER-2 Workshop)
  EXP-4 → EXP-C
  EXP-B (parallel — no hard dependency)
  EXP-6 (parallel — systems measurement)
  EXP-5 (parallel — uses EXP-2 infrastructure)

Phase 3 (extensions and follow-ups)
  EXP-A  (piggybacks on EXP-2 checkpoints — nearly free)
  EXP-F  (rerun EXP-1+2 with two new models)
  EXP-L1 (independent — can start any time)
  EXP-E  (requires EXP-L1 first)
  EXP-D  (requires EXP-2 mature grinder)
```
