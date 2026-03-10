# gristmill-ml

Python ML package for GristMill. Handles all model training, fine-tuning, ONNX export, and experiment tracking. **Never runs production inference** — that is Rust's domain.

> **Rule**: Python trains; Rust runs. Production inference never goes through Python.

## Package Structure

```
src/gristmill_ml/
├── core.py              # PyO3 bridge re-export (+ pure-Python stubs)
├── training/
│   ├── sieve_trainer.py     # 4-class intent routing classifier
│   ├── ner_trainer.py       # Named entity recognition (person/org/date/location)
│   └── embedder_trainer.py  # Domain-specific sentence embedding fine-tuning
├── datasets/
│   ├── feedback.py          # Load Sieve feedback JSONL → PyTorch Dataset
│   └── augmentation.py      # Synthetic data generation (back-translation, paraphrase)
├── export/
│   ├── onnx_export.py       # PyTorch → ONNX INT8 with validation
│   └── validate.py          # Cross-runtime parity check (PyTorch vs ONNX vs Rust)
└── experiments/
    └── tracking.py          # MLflow / W&B experiment tracking helpers
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
