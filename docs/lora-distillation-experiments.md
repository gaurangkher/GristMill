# GristMill Experiment Runner Guide

**Authors**: GristMill Engineering Team
**Date**: 2026-05-31
**Status**: Living document — updated after each experiment cycle

---

## Overview

This guide explains how to set up, run, and evaluate experiments that validate the GristMill v2 closed-loop distillation pipeline. Experiments test whether small local models can be trained to handle domain-specific tasks that would otherwise require cloud LLM escalation.

Each experiment is documented as a standalone technical paper in [`docs/experiments/`](./experiments/). This guide covers the shared infrastructure: prerequisites, tooling, and how to run any experiment end-to-end.

---

## Experiment Registry

| ID | Title | Status | Type | Link |
|----|-------|--------|------|------|
| EXP-000 | Pipeline Bring-Up: Engineering Bugs & Fixes | Complete | Infrastructure | [exp-000-pipeline-bringup.md](./experiments/exp-000-pipeline-bringup.md) |
| EXP-001 | Baseline LoRA Training (Qwen2.5-3B, Aggressive) | Complete | Distillation | [exp-001-baseline-lora-3b.md](./experiments/exp-001-baseline-lora-3b.md) |
| EXP-002 | Conservative LoRA Training (Qwen2.5-0.5B) | Complete | Distillation | [exp-002-conservative-lora-0.5b.md](./experiments/exp-002-conservative-lora-0.5b.md) |
| EXP-003 | Factual Accuracy Validation (ROUGE-L Replacement) | Complete | Validation | [exp-003-factual-accuracy-runner.md](./experiments/exp-003-factual-accuracy-runner.md) |
| EXP-004 | Student Model Upgrade: 0.5B → 1.5B-Instruct | Proposed | Distillation | [exp-004-student-model-upgrade.md](./experiments/exp-004-student-model-upgrade.md) |
| EXP-005 | Company Runbooks: RAG + Generative QA | Proposed | Use Case | [exp-005-runbooks-rag.md](./experiments/exp-005-runbooks-rag.md) |
| EXP-006 | Sentiment Analysis: DeBERTa-v3 Classification | Proposed | Use Case | [exp-006-sentiment-deberta.md](./experiments/exp-006-sentiment-deberta.md) |
| EXP-007 | Domain Embeddings: Bi-Encoder Fine-Tuning | Proposed | Use Case | [exp-007-domain-embeddings.md](./experiments/exp-007-domain-embeddings.md) |
| EXP-008 | Multi-Model Grinder Architecture | Proposed | Architecture | [exp-008-multi-model-grinder.md](./experiments/exp-008-multi-model-grinder.md) |

---

## Prerequisites

### Hardware

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| RAM | 8 GB unified (Apple Silicon) | 16 GB unified |
| GPU | MPS (Apple Silicon) | MPS (M1 Pro / M2 Pro or later) |
| Disk | 20 GB free | 50 GB free |
| OS | macOS Ventura 13+ | macOS Sequoia |

> **Docker training is not viable for MPS-accelerated training.** Docker on macOS runs inside a Linux VM that has no access to Metal GPU. Run the trainer natively on the macOS host. See [EXP-000](./experiments/exp-000-pipeline-bringup.md) for full diagnosis.

### Software

```bash
# Python environment (editable install from repo root)
cd gristmill-ml && pip install -e .

# Verify MPS is available
python -c "import torch; print(torch.backends.mps.is_available())"
# Expected: True

# Verify core dependencies
python -c "from trl import SFTConfig, SFTTrainer; print('TRL ok')"
python -c "from peft import get_peft_model; print('PEFT ok')"
```

Required Python packages (pinned constraints in `gristmill-ml/pyproject.toml`):

| Package | Minimum version | Why |
|---------|----------------|-----|
| `trl` | `>=0.8.1` | `SFTConfig` missing in 0.8.0 |
| `peft` | any | LoRA adapter wrapping |
| `transformers` | any | model loading |
| `torch` | any (MPS support) | tensor ops |
| `pyyaml` | any | probe file parsing |

### Data

The training buffer must be populated before any distillation experiment:

```bash
# Seed the reasoning domain (2,405 examples from OpenHermes-2.5)
python scripts/seed_reasoning.py

# Verify records exist
sqlite3 ~/.gristmill/db/training_buffer.sqlite \
  "SELECT domain_tag, status, COUNT(*) FROM training_records GROUP BY 1,2"
# Expected: reasoning|AVAILABLE|2405 (or similar)
```

### Config

`~/.gristmill/config.yaml` must include the trainer section. Example minimum:

```yaml
sieve:
  training_buffer_path: /path/to/gristmill-data/db/training_buffer.sqlite

trainer:
  base_model: Qwen/Qwen2.5-0.5B-Instruct   # or 1.5B for EXP-004+
  lora_rank: 16
  lora_alpha: 16
  lora_target_modules: q_proj,v_proj
  learning_rate: 5e-5
  num_epochs: 1
  batch_size: 4
  gradient_accumulation_steps: 4
  replay_fraction: 0.30

  validation:
    strategy: factual_accuracy             # or rouge_l (legacy)
    probe_set: reasoning
    min_accuracy: 0.6
    overall_delta_min: -0.05
    domain_delta_min: -0.08

checkpoint_dir: /path/to/gristmill-data/checkpoints
```

> **Never commit this file.** It is permanently gitignored because it contains secrets (API keys, Slack tokens, webhook URLs).

---

## Core Tooling

### 1. Training the Adapter

Run one distillation cycle for the `reasoning` domain:

```bash
# Start the trainer service (native macOS, not Docker)
cd gristmill-ml
python -m gristmill_ml.trainer.service --domain reasoning

# Or via the CLI entry point (if installed)
gristmill-trainer --domain reasoning
```

The trainer will:
1. Sample AVAILABLE records from the training buffer
2. Train a LoRA adapter on the configured student model
3. Validate the adapter (ROUGE-L or FactualAccuracyRunner depending on config)
4. Promote or roll back based on validation score
5. Write the adapter to `checkpoint_dir/active/<domain>/`

Monitor training progress:

```bash
# Tail trainer logs
tail -f ~/.gristmill/logs/trainer.log

# Check current checkpoint manifest
cat /path/to/gristmill-data/checkpoints/active/reasoning/manifest.json
```

### 2. Evaluating the Adapter

Compare base model vs. the active LoRA adapter on a probe set:

```bash
# Built-in reasoning probes
python scripts/compare_lora_adapter.py --probe-set reasoning

# Auto-generated training-data probes
python scripts/compare_lora_adapter.py --probe-set training_reasoning

# Custom probe set
python scripts/compare_lora_adapter.py --probe-set my_probes --domain reasoning

# Increase output length for multi-step reasoning
python scripts/compare_lora_adapter.py --probe-set reasoning --max-new-tokens 512
```

Output includes:
- Per-probe: base model answer, adapter answer, correct/incorrect flag
- Summary: accuracy score and pass/fail for each model
- `base_score` vs `adapter_score` comparison

### 3. Generating Probes from Training Data

Build a probe set directly from training buffer records:

```bash
# Generate 10 math/reasoning probes from training data
python scripts/generate_training_probes.py --domain reasoning --n 10 --math-only

# Dry run (print to stdout without writing)
python scripts/generate_training_probes.py --dry-run --domain reasoning --n 5

# Custom output path
python scripts/generate_training_probes.py \
  --domain reasoning --n 20 \
  --output probes/training_reasoning_v2.yaml
```

Probe YAML format:

```yaml
domain: reasoning
probes:
  - id: train_abc12345
    tags: [reasoning, training-data]
    question: "If a factory produces 350 widgets per hour..."
    expected: "The factory produces 2,800 widgets in 8 hours."
    correct_answer: "2,800"
    notes: "Auto-generated from training buffer"
```

### 4. Watching Memory Usage

LoRA training on MPS can spike to 6–10 GB depending on model size. Monitor during training:

```bash
# macOS Activity Monitor — sort by Memory
open -a "Activity Monitor"

# Or CLI (updates every 2s)
watch -n 2 "ps aux | grep python | grep -v grep | awk '{print \$2, \$4, \$11}'"
```

If training OOMs, reduce batch size in config (`batch_size: 1`) — the trainer automatically compensates via `gradient_accumulation_steps`.

---

## Running an Experiment End-to-End

The general workflow for any distillation experiment:

```
1. Update config         → Set hyperparameters in ~/.gristmill/config.yaml
2. Seed data             → python scripts/seed_<domain>.py
3. Generate probes       → python scripts/generate_training_probes.py
4. Baseline evaluation   → python scripts/compare_lora_adapter.py (before training)
5. Train                 → python -m gristmill_ml.trainer.service
6. Post-training eval    → python scripts/compare_lora_adapter.py (after training)
7. Document results      → Update the experiment paper in docs/experiments/
```

Each experiment paper specifies exact config values, expected results, and success criteria for step 7.

---

## Cross-Cutting Findings

These findings apply across all completed distillation experiments and inform the design of proposed experiments:

### Model Capacity is the Binding Constraint

Both Qwen2.5-0.5B and 3B-Instruct exhibited catastrophic forgetting of factual knowledge when fine-tuned on 2,405 reasoning-format examples. The failure is consistent with insufficient representational capacity: small models must trade off factual knowledge against newly-learned formatting patterns. See [EXP-001](./experiments/exp-001-baseline-lora-3b.md) and [EXP-002](./experiments/exp-002-conservative-lora-0.5b.md).

### ROUGE-L is a Misleading Validation Metric

ROUGE-L rewards verbatim surface overlap with teacher outputs, which correlates with overfitting to training format rather than semantic accuracy. The adapter with the highest ROUGE-L score (0.2322) gave the most confidently wrong factual answer. Replaced by `FactualAccuracyRunner` — see [EXP-003](./experiments/exp-003-factual-accuracy-runner.md).

### LoRA Nudges, It Does Not Memorize

With rank 16, 2 modules, lr=5e-5, 1 epoch: each training example contributes a single tiny gradient step. The adapter cannot store 2,400 specific Q→A mappings. Success requires that the fine-tuning distribution matches the evaluation distribution — a condition met in the proposed use-case experiments (EXP-005 through EXP-008) but not in the initial reasoning experiments.

### Purpose-Built Small Models Outperform General Small Models

The multi-model grinder architecture ([EXP-008](./experiments/exp-008-multi-model-grinder.md)) — DeBERTa for sentiment, nomic-embed for vectors, Qwen2.5-1.5B for runbooks — achieves higher accuracy at lower latency and cost than any single general-purpose model at the same parameter budget.

---

## Architecture Reference

```
TypeScript Shell (gristmill-integrations)
  └─ napi-rs ──→ Rust Core (gristmill-core)  ←── PyO3 ── Python Shell (gristmill-ml)
                   └─ grist-grinders (ONNX inference pool)
                        └─ hot-reload from checkpoint_dir/active/<domain>/
```

The full distillation loop:

```
Teacher LLM (Ollama/Claude)
  → teacher-labeled records → SQLite training buffer
  → DistillationEngine (Python, PEFT + TRL)
  → LoRA adapter checkpoint
  → FactualAccuracyRunner validation
  → Promoted adapter → Rust hot-reload → Production inference
  → Misclassifications → feedback JSONL → next training cycle
```

---

## See Also

- [`gristmill-v2-architecture.md`](../gristmill-v2-architecture.md) — Full system architecture
- [`gristmill-ml/probes/`](../gristmill-ml/probes/) — Static evaluation probe sets
- [`gristmill-ml/scripts/`](../gristmill-ml/scripts/) — Tooling scripts
- [`CLAUDE.md`](../CLAUDE.md) — Development guidelines and invariants
