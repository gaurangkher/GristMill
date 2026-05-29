# gristmill-ml

Python ML package for GristMill. Handles all model training, fine-tuning, ONNX export, and experiment tracking. **Never runs production inference** — that is Rust's domain.

> **Rule**: Python trains; Rust runs. Production inference never goes through Python.

## Package Structure

```
gristmill-ml/
├── probes/                      # Named probe sets for reproducible evaluation
│   └── reasoning.yaml           # 5 arithmetic/factual probes with expected answers
├── results/                     # Saved JSON evaluation reports (gitignored)
├── scripts/
│   ├── seed_reasoning.py        # Bulk-seed training buffer from OpenHermes-2.5
│   └── compare_lora_adapter.py  # CLI: compare base model vs. LoRA adapter
└── src/gristmill_ml/
    ├── core.py                  # PyO3 bridge re-export (+ pure-Python stubs)
    ├── training/
    │   ├── sieve_trainer.py         # 4-class intent routing classifier
    │   ├── ner_trainer.py           # Named entity recognition
    │   └── embedder_trainer.py      # Domain-specific sentence embedding fine-tuning
    ├── trainer/                 # LoRA distillation pipeline
    │   ├── service.py               # GristMillTrainerService — state machine + orchestrator
    │   ├── distillation.py          # DistillationEngine — PEFT/TRL training loop
    │   ├── checkpoint.py            # CheckpointManager — versioned adapter filesystem
    │   ├── validation.py            # ValidationRunner — ROUGE-L promotion gate
    │   ├── retention.py             # RetentionBuffer — experience replay records
    │   └── ipc_server.py            # IPC/WebSocket event emission
    ├── datasets/
    │   ├── feedback.py              # Load Sieve feedback JSONL → PyTorch Dataset
    │   └── augmentation.py          # Synthetic data generation
    ├── export/
    │   ├── onnx_export.py           # PyTorch → ONNX INT8 with validation
    │   └── validate.py              # Cross-runtime parity check
    └── experiments/
        ├── adapter_eval.py          # AdapterEvaluator — base vs. LoRA comparison
        ├── comparisons.py           # ONNX sieve model comparison framework
        └── tracking.py              # MLflow / W&B experiment tracking helpers
```

## Installation

```bash
cd gristmill-ml

# Editable install (first run downloads ~2–3 GB of ML dependencies)
pip install -e ".[dev]"

# With PyO3 Rust bridge (optional — enables HAS_NATIVE = True)
pip install maturin
cd ../gristmill-core/crates/grist-ffi
maturin develop --features python
```

## CLI Entry Points

After `pip install -e .`:

```bash
# Run the LoRA distillation trainer daemon
gristmill-trainer

# Train Sieve classifier from feedback logs
gristmill-train-sieve [--epochs 5] [--lr 2e-5] [--output ~/.gristmill/models/sieve-v2.onnx]

# Export a trained model to ONNX
gristmill-export [--model checkpoint.pth] [--output sieve.onnx] [--quantize int8]

# Validate ONNX export parity
gristmill-validate [--pytorch model.pth] [--onnx model.onnx]
```

## Key Modules

### `core.py` — Bridge Re-export

Single import point for the PyO3 extension. Provides pure-Python stubs when the compiled wheel is not installed, so the package is importable in any environment.

```python
from gristmill_ml.core import HAS_NATIVE, PyGristMill

if HAS_NATIVE:
    core = PyGristMill("~/.gristmill/config.yaml")
    # Real Rust operations
else:
    # Development mode — stubs raise RuntimeError on actual calls
```

**Never import `gristmill_core` directly** in application code. Always use `gristmill_ml.core`.

### `training/sieve_trainer.py` — Intent Classifier

Trains the 4-class routing classifier on accumulated Sieve feedback.

```python
from gristmill_ml.training.sieve_trainer import SieveTrainer

trainer = SieveTrainer(feedback_dir="~/.gristmill/feedback/")
trainer.prepare_dataset()
trainer.train(epochs=5, lr=2e-5)
trainer.export(output_path="~/.gristmill/models/sieve-v2.onnx")
```

**Feature vector (392 dims)** — must exactly match `grist-sieve/src/features.rs`:

| Dims | Content |
|------|---------|
| 0–383 | L2-normalised MiniLM-L6-v2 embedding |
| 384 | Log-scaled token count |
| 385 | Source channel ordinal / 9.0 |
| 386 | Priority ordinal / 3.0 |
| 387 | Entity density |
| 388 | Question probability |
| 389 | Code token fraction |
| 390 | Type-token ratio |
| 391 | Ambiguity score |

**Classes (label → RouteDecision variant):**

| Label | Rust Variant | Meaning |
|-------|-------------|---------|
| 0 | `LocalML` | Handle with local ONNX model |
| 1 | `Rules` | Handle with deterministic rule |
| 2 | `Hybrid` | Local model + LLM prompt |
| 3 | `LlmNeeded` | Full LLM escalation required |

### `datasets/feedback.py` — Feedback Dataset

Loads routing decisions logged by `grist-sieve` into a `torch.utils.data.Dataset`.

**Feedback JSONL schema:**
```json
{
  "event_id": "01HXYZ...",
  "text": "schedule meeting with alice tomorrow",
  "channel": "http",
  "priority": "normal",
  "route": "LOCAL_ML",
  "confidence": 0.92,
  "timestamp_ms": 1234567890
}
```

```python
from gristmill_ml.datasets.feedback import FeedbackDataset

dataset = FeedbackDataset(feedback_dir="~/.gristmill/feedback/")
print(f"Loaded {len(dataset)} feedback samples")
```

### `export/onnx_export.py` — ONNX Export

Converts PyTorch models to ONNX with INT8 quantization for deployment in Rust via `ort`.

```python
from gristmill_ml.export.onnx_export import export_to_onnx

export_to_onnx(
    model=trained_model,
    tokenizer=tokenizer,
    output_path="~/.gristmill/models/sieve-v2.onnx",
    quantize="int8",   # or "fp16"
    validate=True,     # Cross-check PyTorch vs ONNX outputs
)
```

**Pipeline:**
1. `torch.onnx.export()` → ONNX graph
2. `onnxruntime` validation run
3. INT8 quantization via `onnxruntime-tools`
4. Save to output path

### `export/validate.py` — Parity Validation

Ensures the exported ONNX model produces numerically equivalent outputs to the PyTorch original. Run this before hot-reloading a new model into Rust.

```python
from gristmill_ml.export.validate import validate_parity

report = validate_parity(
    pytorch_path="checkpoint.pth",
    onnx_path="sieve-v2.onnx",
)
print(f"Max absolute error: {report.max_abs_error}")
print(f"Cosine similarity:  {report.cosine_similarity}")
assert report.passes_threshold, "Parity check failed!"
```

## LoRA Distillation Pipeline

The `trainer/` subpackage implements a closed-loop LoRA distillation pipeline: a teacher LLM labels incoming queries, examples accumulate in a SQLite training buffer, the `DistillationEngine` fine-tunes a lightweight student model with PEFT/TRL, and validated adapters are hot-reloaded into the Rust daemon without service interruption.

### Configuration

All training hyperparameters and path overrides live in `config.yaml`:

```yaml
trainer:
  base_model: Qwen/Qwen2.5-0.5B-Instruct
  num_epochs: 1
  learning_rate: 0.00005      # 5e-5 — conservative for small models
  lora_rank: 16
  lora_alpha: 16              # ratio 1× to limit update magnitude
  lora_target_modules: "q_proj,v_proj"
  replay_fraction: 0.30       # 30% of each batch from retention buffer
  validation:
    overall_delta_min: -0.05  # ROUGE-L delta gate for promotion
    domain_delta_min: -0.08

millwright:
  checkpoint_dir: /path/to/gristmill-data/checkpoints/   # where adapters land
```

The trainer reads config from (in order): `$GRISTMILL_CONFIG`, `/data/gristmill/config.yaml` (Docker), `~/.gristmill/config.yaml` (host).

### Running the trainer natively (Apple Silicon / host GPU)

Docker containers on macOS cannot access Metal (MPS). Run the trainer directly on the host:

```bash
# From gristmill-ml/
.ve/bin/gristmill-trainer
```

Point it at the shared SQLite buffer via `~/.gristmill/config.yaml`:

```yaml
sieve:
  training_buffer_path: /absolute/path/to/gristmill-data/db/training_buffer.sqlite
millwright:
  checkpoint_dir: /absolute/path/to/gristmill-data/checkpoints/
```

### Seeding training data

```bash
python scripts/seed_reasoning.py   # loads ~2,400 examples from OpenHermes-2.5
```

### Known training issues and fixes

See [`docs/lora-distillation-experiments.md`](../docs/lora-distillation-experiments.md) for a full catalogue of 12 engineering bugs encountered during bring-up, including:

- PEFT + gradient checkpointing deadlock → requires `model.enable_input_require_grads()` and `use_reentrant=False`
- Concurrent domain training OOM → global `threading.Lock()` over full load→train→save pipeline
- Docker MPS inaccessibility → run trainer natively on host
- ROUGE-L as misleading validation metric → factual accuracy probes

---

## Adapter Evaluation Framework

The evaluation framework lets you compare a trained LoRA adapter against the base model on a reproducible set of named probe questions, with both ROUGE-L scoring and factual correctness checking.

### Quick start

```bash
# Uses base model and checkpoint path from config.yaml automatically:
python scripts/compare_lora_adapter.py

# Explicit paths:
python scripts/compare_lora_adapter.py \
    --base-model Qwen/Qwen2.5-0.5B-Instruct \
    --adapter gristmill-data/checkpoints/active/reasoning \
    --domain reasoning \
    --probe-set reasoning \
    --output results/exp4-reasoning.json

# Single ad-hoc question:
python scripts/compare_lora_adapter.py \
    --no-probe-set \
    --question "A shop has 200 items. 15% are returned. How many remain?"
```

### CLI reference

| Flag | Default | Description |
|------|---------|-------------|
| `--base-model` | from `config.yaml` | HuggingFace model id or local path |
| `--adapter` | `<checkpoint_root>/active/<domain>` | Path to PEFT adapter directory |
| `--domain` | `reasoning` | Domain name for locating the adapter and labelling output |
| `--probe-set` | `reasoning` | Probe set to load from `probes/<name>.yaml` |
| `--question` | — | Extra ad-hoc question prepended to the probe set |
| `--no-probe-set` | — | Disable probe set; use `--question` only |
| `--max-new-tokens` | `256` | Max tokens to generate per response |
| `--output` | — | Save JSON report to this path |
| `--probes-dir` | `probes/` | Override probe set directory |

### Probe sets

Probe sets are YAML files in `probes/`. Each probe has:

```yaml
probes:
  - id: february_leaves          # unique identifier
    tags: [arithmetic, calendar_fact]
    question: >
      Every day, a tree drops 7 leaves...
    expected: |
      1. February has 28 days...
    correct_answer: "196"        # substring checked in model output
    notes: >
      Experiment 1–3 primary probe...
```

**Current probe sets:**

| File | Probes | Domain |
|------|--------|--------|
| `probes/reasoning.yaml` | 5 | Arithmetic and calendar-fact reasoning |

Add a new probe set by creating `probes/<name>.yaml` following the same schema.

### Programmatic usage

```python
from pathlib import Path
from gristmill_ml.experiments.adapter_eval import AdapterEvaluator, load_probes

probes = load_probes("reasoning")           # loads probes/reasoning.yaml
evaluator = AdapterEvaluator(
    base_model="Qwen/Qwen2.5-0.5B-Instruct",
    adapter_path=Path("gristmill-data/checkpoints/active/reasoning"),
    domain="reasoning",
)
report = evaluator.evaluate(probes, probe_set="reasoning")
report.print_report()
report.save(Path("results/exp4-reasoning.json"))

# Access structured results
for pr in report.probes:
    print(f"{pr.probe_id}: base_correct={pr.base_correct}, adapter_correct={pr.adapter_correct}")
print(f"ROUGE-L delta: {report.rouge_l_delta:+.4f}")
print(f"Factual accuracy: base {report.base_correct_count}/{report.checkable_count}, "
      f"adapter {report.adapter_correct_count}/{report.checkable_count}")
```

### Saved report format

`report.save(path)` writes JSON with this shape:

```json
{
  "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
  "adapter_path": "/path/to/active/reasoning",
  "domain": "reasoning",
  "probe_set": "reasoning",
  "timestamp": "2026-05-29T22:00:00Z",
  "mean_base_rouge_l": 0.312,
  "mean_adapter_rouge_l": 0.198,
  "rouge_l_delta": -0.114,
  "base_correct_count": 5,
  "adapter_correct_count": 2,
  "checkable_count": 5,
  "probes": [
    {
      "probe_id": "february_leaves",
      "base_rouge_l": 0.28,
      "adapter_rouge_l": 0.19,
      "base_correct": true,
      "adapter_correct": false,
      ...
    }
  ]
}
```

JSON files in `results/` are gitignored — commit the probe YAML files and the scripts, not the outputs.

---

## Closed-Loop Retraining

```
Rust Sieve (production)
  │ logs routing decisions
  ▼
~/.gristmill/feedback/feedback-YYYY-MM-DD.jsonl
  │ weekly (cron / manual)
  ▼
gristmill-train-sieve
  │ trains SieveClassifierHead
  ▼
gristmill-export --quantize int8
  │ produces sieve-v{n}.onnx
  ▼
gristmill-validate
  │ checks parity
  ▼
~/.gristmill/models/sieve-v{n}.onnx
  │ Rust ModelRegistry detects new file
  ▼
Hot-reload (no daemon restart)
```

## Dependencies

```toml
torch >= 2.1
transformers >= 4.40
sentence-transformers >= 2.7
onnx >= 1.16
onnxruntime >= 1.18
onnxruntime-tools >= 1.7
numpy >= 1.24
mlflow >= 2.12
datasets >= 2.18
scikit-learn >= 1.4
spacy >= 3.7
accelerate >= 0.28
evaluate >= 0.4
seqeval >= 1.2
```

## Testing

```bash
# Run unit tests
pytest tests/

# Validate latest ONNX export
python -m gristmill_ml.export.validate

# Quick smoke test (no GPU required)
python -c "from gristmill_ml.training.sieve_trainer import SieveTrainer; print('OK')"
```
