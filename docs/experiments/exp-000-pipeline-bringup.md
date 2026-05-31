# EXP-000: Pipeline Bring-Up — Engineering Bugs and Fixes

**Status**: Complete
**Date**: 2026-05-29
**Authors**: GristMill Engineering Team

← [Back to Experiment Index](../lora-distillation-experiments.md) | Next: [EXP-001 →](./exp-001-baseline-lora-3b.md)

---

## Abstract

Before any training experiment could be executed, twelve distinct engineering bugs were identified and resolved in the GristMill distillation pipeline. These bugs span dependency incompatibilities, concurrency races, PyTorch-platform interactions, and Docker-macOS GPU access limitations. This document catalogs each bug with its symptom, root cause, and fix. It serves as a reference for teams building similar pipelines with the same dependency stack (HuggingFace Transformers, PEFT, TRL, PyTorch, Docker on macOS).

---

## 1. Introduction

The GristMill v2 distillation pipeline assembles multiple high-churn Python libraries (TRL, PEFT, Transformers) alongside system-level constraints (Docker on Apple Silicon, MPS GPU access, concurrent training threads). The bring-up phase revealed that interaction effects between these components produce failure modes that are not individually documented in any upstream library's troubleshooting guide. The following twelve bugs were encountered in the order listed and resolved before the first successful training run.

---

## 2. Bug Catalog

### Bug 1: TRL Version Incompatibility

**Symptom**: `ImportError: cannot import name 'SFTConfig' from 'trl'`

**Root cause**: `SFTConfig` was introduced in TRL 0.8.1, not 0.8.0. The dependency specification `trl>=0.8` permitted installation of 0.8.0, which lacked the class.

**Fix**: Tighten the lower bound:
```
trl>=0.8.1
```

---

### Bug 2: Out-of-Memory in float32 (3B Model)

**Symptom**: Container OOM-killed during model loading.

**Root cause**: Qwen2.5-3B-Instruct in float32 occupies approximately 12 GB. With 14.37 GiB container RAM, no headroom remained for optimizer state, activations, and the training buffer.

**Fix**: Load in bfloat16:
```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
)
```

---

### Bug 3: `max_seq_length` Renamed in TRL 1.x

**Symptom**: `TypeError: SFTConfig.__init__() got an unexpected keyword argument 'max_seq_length'`

**Root cause**: TRL 1.x renamed the sequence length parameter from `max_seq_length` to `max_length` in `SFTConfig`.

**Fix**: Use `max_length` in `SFTConfig` construction.

---

### Bug 4: SQLite UNIQUE Constraint Race

**Symptom**: Intermittent `IntegrityError: UNIQUE constraint failed` during domain cycle execution.

**Root cause**: Two concurrent domain cycles both invoked `curate()`, which executed DELETE then INSERT on the same record IDs. The second cycle's INSERT fired between the first cycle's DELETE and INSERT, violating the UNIQUE constraint when the first cycle's INSERT completed.

**Fix**: Replace `INSERT` with `INSERT OR IGNORE` to make the upsert idempotent.

---

### Bug 5: Concurrent Import Race on Heavy Dependencies

**Symptom**: Intermittent `AttributeError` or `ImportError` during training startup on `from trl import SFTConfig, SFTTrainer`.

**Root cause**: TRL was imported lazily inside `_train_lora()`. Two domain training threads hitting `_train_lora()` simultaneously received a partially initialized module object from the first thread's ongoing import.

**Fix**: Move all heavy imports (`torch`, `transformers`, `peft`, `trl`) to module level, ensuring full initialization before any training thread accesses them.

---

### Bug 6: bfloat16 Rejected on CPU

**Symptom**: `ValueError: bf16 is not supported on CPU` from `SFTConfig`.

**Root cause**: `SFTConfig(bf16=True)` is valid only on CUDA or MPS. The code unconditionally passed `bf16=True`.

**Fix**:
```python
sft_config = SFTConfig(
    bf16=(self.device != "cpu"),
    ...
)
```

---

### Bug 7: Meta Tensor Corruption from `device_map="cpu"`

**Symptom**: Model inference produced NaN outputs or raised `RuntimeError: Expected all tensors to be on the same device`.

**Root cause**: In newer Transformers versions, `device_map="cpu"` triggers Accelerate's meta tensor initialization path even without `low_cpu_mem_usage=True`. Meta tensors are uninitialized placeholders; if device placement logic is not correctly applied, the model ends up with mixed initialized and uninitialized weights.

**Fix**: Only pass `device_map` for non-CPU devices:
```python
kwargs = {}
if device != "cpu":
    kwargs["device_map"] = device
model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
```

---

### Bug 8: Concurrent `from_pretrained` Meta Tensor Corruption

**Symptom**: One of two concurrently loading model instances would produce NaN outputs or silently incorrect results.

**Root cause**: HuggingFace's `from_pretrained` reads weights into shared cache structures. Two threads calling `from_pretrained` on the same model concurrently interleave their tensor initialization writes, corrupting each other's tensor state.

**Fix**: Introduce a `threading.Lock()` around `from_pretrained` calls:
```python
_model_load_lock = threading.Lock()

with _model_load_lock:
    model = AutoModelForCausalLM.from_pretrained(model_name, ...)
```

---

### Bug 9: Out-of-Memory from Concurrent Training (3B Model)

**Symptom**: Container OOM-killed during concurrent training of two domain cycles.

**Root cause**: Even with the load lock serializing `from_pretrained`, both domain cycles retained their 3B model in memory simultaneously after loading. Two bfloat16 3B models ≈ 12 GB combined, exceeding available container RAM.

**Fix**: Extend the lock scope to cover the entire load → train → save pipeline, ensuring only one domain's model is resident at a time.

---

### Bug 10: Gradient Checkpointing Hang with PEFT

**Symptom**: Training hung indefinitely at step 0 of 453, with CPU at 929% (all cores) and zero loss reduction for over 4 hours.

**Root cause**: PEFT's LoRA wrapping modifies the gradient computation graph in a way incompatible with PyTorch's reentrant gradient checkpointing unless `model.enable_input_require_grads()` is called before training. Without this call, the backward pass enters a deadlock waiting for gradients that will never arrive through the wrapped parameter path. Additionally, `use_reentrant=True` (the default) produces instability with PEFT adapters.

**Fix**:
```python
model = get_peft_model(model, lora_config)
model.enable_input_require_grads()  # REQUIRED before SFTTrainer

sft_config = SFTConfig(
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    ...
)
```

> **Diagnosis tip**: Sustained high CPU with zero loss reduction at step 0 is a reliable indicator of this deadlock — not a performance problem.

---

### Bug 11: Docker MPS Inaccessibility

**Symptom**: Training on the container ran at 2–5 minutes per step, projecting 15–38 hours per domain.

**Root cause**: Docker on macOS runs containers inside a Linux VM (Apple Virtualization Framework). The VM has no access to the host's Metal GPU. All tensor operations fall back to CPU.

**Fix**: Run the trainer natively on the macOS host to access MPS. Step time dropped from 2–5 minutes to a practical duration, making the experiment timeline feasible.

```bash
# NOT inside Docker:
cd gristmill-ml && python -m gristmill_ml.trainer.service --domain reasoning
```

---

### Bug 12: Wrong Database Path on Host

**Symptom**: Host-side trainer reported zero training examples, using the default `~/.gristmill/db/training_buffer.sqlite`, which was empty.

**Root cause**: The symlink `/data/gristmill` used inside the container to point to the shared SQLite database did not exist on the host. Without explicit `training_buffer_path` in `~/.gristmill/config.yaml`, the trainer silently used the fallback path.

**Fix**: Add the explicit path to `~/.gristmill/config.yaml`:
```yaml
sieve:
  training_buffer_path: /path/to/gristmill-data/db/training_buffer.sqlite
```

---

## 3. Summary

| Bug | Category | Severity | Detectable by |
|-----|----------|----------|--------------|
| 1. TRL version | Dependency | Startup crash | Import error |
| 2. OOM float32 | Memory | Startup crash | OOM kill |
| 3. `max_seq_length` rename | API change | Startup crash | TypeError |
| 4. SQLite race | Concurrency | Intermittent data loss | IntegrityError |
| 5. Import race | Concurrency | Intermittent crash | AttributeError |
| 6. bf16 on CPU | Config | Startup crash | ValueError |
| 7. Meta tensor (device_map) | Platform | Silent corruption | NaN outputs |
| 8. Concurrent from_pretrained | Concurrency | Silent corruption | NaN outputs |
| 9. OOM concurrent training | Memory | Crash | OOM kill |
| 10. PEFT grad checkpointing | Library interaction | Hang | 0% progress, 929% CPU |
| 11. Docker MPS | Platform | Training impractical | Timing |
| 12. Wrong DB path | Config | No data | Zero records |

All twelve bugs were resolved before any training experiment. The pipeline is functionally correct end-to-end.

---

## 4. Conclusions

None of these bugs individually represents a novel research finding. Collectively, they illustrate that assembling a LoRA fine-tuning pipeline from current-generation Python ML libraries requires resolving a substantial body of library-interaction effects not documented in any upstream source. The gradient checkpointing / PEFT interaction (Bug 10) is particularly insidious because its symptom (high CPU, no progress) is easily mistaken for a performance problem.

---

← [Back to Experiment Index](../lora-distillation-experiments.md) | Next: [EXP-001 →](./exp-001-baseline-lora-3b.md)
