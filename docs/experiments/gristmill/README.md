# GristMill Experiments Index

This directory contains experiment designs for the GristMill paper. Experiments are split into two sets: the **original six** that validate GristMill's core architecture, and **six proposed extensions** that address robustness, deployment, and generalization.

---

## Original Experiments (Core Paper)

These six experiments are required for a credible arXiv preprint (1–3) or conference submission (1–6).

| # | Title | What it validates |
|---|---|---|
| 1 | Sieve calibration accuracy vs. routing cost | Self-REF token quality; Pareto frontier vs. FrugalGPT/RouteLLM |
| 2 | Distillation flywheel: autonomy growth over time | Grinder improves with distillation cycles |
| 3 | Capacity gap in practice | Optimal teacher-to-student size ratio |
| 4 | Catastrophic forgetting under domain shift | Replay buffer + EWC-LoRA mitigation effectiveness |
| 5 | Cold start characterization | Queries/days to reach useful autonomy |
| 6 | gristmill-trainer isolation benefit | Two-process architecture delivers zero inference degradation |

---

## Proposed Extension Experiments

These experiments address gaps the original six leave open. Ordered by recommended priority.

| Priority | File | Title | Gap addressed |
|---|---|---|---|
| 1 | [exp_b_distribution_shift.md](./exp_b_distribution_shift.md) | Query Distribution Shift | Real-world robustness; OOD hallucination risk |
| 2 | [exp_a_calibration_drift.md](./exp_a_calibration_drift.md) | Sieve Confidence Calibration Drift | Silent quality regression as grinder improves |
| 3 | [exp_f_model_family_sensitivity.md](./exp_f_model_family_sensitivity.md) | Grinder Model Family Sensitivity | Generalization beyond Qwen2.5-3B |
| 4 | [exp_c_replay_buffer_sizing.md](./exp_c_replay_buffer_sizing.md) | Replay Buffer Size vs. Forgetting Rate | Minimum viable buffer size as engineering spec |
| 5 | [exp_d_offline_grace_period.md](./exp_d_offline_grace_period.md) | Teacher Availability & Offline Grace Period | Deployment resilience during outages |
| 6 | [exp_e_multi_user_adapter_sharing.md](./exp_e_multi_user_adapter_sharing.md) | Multi-User Adapter Sharing | Storage cost reduction at scale |

Also see: [lora_personalization_depth.md](./lora_personalization_depth.md) — LoRA rank sweet spot characterization; feeds into Experiment E.

---

## Dependency Graph

```
Experiment 2 (flywheel)
    ├── Experiment A (calibration drift)  — piggybacks on Exp 2 checkpoints
    ├── Experiment D (offline grace)      — requires cycle-10 mature grinder
    └── Experiment B (distribution shift) — extends Exp 2 with OOD queries
            └── interacts with Experiment 4 (catastrophic forgetting)

Experiment 1 (sieve calibration)
    └── Experiment F (model family)       — replicates Exp 1+2 across architectures

LoRA Personalization Depth
    └── Experiment E (multi-user sharing) — reuses personas and evaluation setup

Experiment 4 (catastrophic forgetting)
    └── Experiment C (buffer sizing)      — same setup, sweeps buffer size
```

---

## Recommended Submission Strategy

**arXiv preprint (fastest path):**
Original experiments 1, 2, 3 + Experiment B (distribution shift)

**Workshop submission:**
Original 1–4 + Experiments A, B, F

**Full conference submission (NeurIPS / ICML / ICLR systems track):**
Original 1–6 + Experiments A, B, C, F + LoRA personalization depth
