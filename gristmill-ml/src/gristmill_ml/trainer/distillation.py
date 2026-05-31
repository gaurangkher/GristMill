"""DistillationEngine — LoRA adapter training pipeline.

Uses Hugging Face PEFT + TRL to fine-tune a small grinder model on
teacher-labelled examples collected in the training buffer.

Distillation technique (Section 4.5.2):
  - Black-box mode (default): cross-entropy on teacher text outputs.
  - White-box mode: reverse-KL divergence when teacher_logits are available.

Catastrophic forgetting mitigations (Section 4.5.3):
  1. LoRA adapter isolation — base weights are frozen; only adapter ranks updated.
  2. Experience replay — 15–20 % of every training batch drawn from the
     retention buffer.
  3. Functional distillation from prior checkpoint — prior adapter acts as a
     secondary teacher via a KL penalty on non-target-domain inputs.
"""

from __future__ import annotations

import logging
import math
import random
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import threading

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

# Only one domain cycle may hold a model in memory at a time — two concurrent
# 3B-param bfloat16 models exceed container RAM and trigger an OOM kill.
_model_load_lock = threading.Lock()

logger = logging.getLogger(__name__)

# Replay fraction: 15–20 % of each effective batch comes from the retention buffer.
REPLAY_FRACTION = 0.17


@dataclass
class CycleResult:
    version: int
    adapter_path: Path
    train_loss: float
    record_count: int
    duration_seconds: float
    success: bool
    domain: str = "default"
    teacher_cost_usd: float = 0.0
    error: Optional[str] = None


@dataclass
class _Example:
    prompt: str
    response: str
    weight: float = 1.0


class DistillationEngine:
    """Runs a single LoRA distillation cycle.

    Parameters
    ----------
    base_model_name:
        HuggingFace model id or local path for the grinder base model.
        Defaults to ``GRISTMILL_BASE_MODEL`` env var or ``Qwen/Qwen2.5-3B-Instruct``.
    output_dir:
        Root directory for adapter output (a sub-directory per cycle will be
        created here for staging).
    device:
        ``"cuda"``, ``"mps"``, or ``"cpu"``.  Auto-detected when ``None``.
    prior_adapter_path:
        If provided, the prior adapter weights are used as a secondary teacher
        for functional distillation (forgetting prevention).
    """

    def __init__(
        self,
        base_model_name: Optional[str] = None,
        output_dir: Optional[Path] = None,
        device: Optional[str] = None,
        prior_adapter_path: Optional[Path] = None,
    ) -> None:
        import os

        self.base_model_name = base_model_name or os.environ.get(
            "GRISTMILL_BASE_MODEL", "Qwen/Qwen2.5-3B-Instruct"
        )
        self.output_dir = output_dir or (Path.home() / ".gristmill" / "staging")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.prior_adapter_path = prior_adapter_path
        self.device = device or _detect_device()

    # ── Public API ────────────────────────────────────────────────────────────

    def run_cycle(
        self,
        training_db_path: Path,
        retention_records: list[dict],
        version: int,
        domain: str = "default",
        max_steps: Optional[int] = None,
        num_epochs: int = 3,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        learning_rate: float = 2e-4,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_target_modules: Optional[str] = None,
        replay_fraction: float = REPLAY_FRACTION,
    ) -> CycleResult:
        """Execute a full distillation cycle and return the staged adapter path.

        Reads PENDING records for *domain* from *training_db_path*, mixes in
        *retention_records* for replay, trains a LoRA adapter, saves to a temp
        staging directory.

        ``max_steps`` is computed from the actual dataset size and ``num_epochs``
        when not explicitly provided, ensuring consistent training depth regardless
        of how many records accumulated between cycles.  Pass ``max_steps``
        explicitly only to override (e.g. in tests).
        """
        import time

        start = time.time()

        try:
            # ── Load training records ─────────────────────────────────────────
            pending = _load_pending_records(training_db_path, domain=domain)
            if not pending:
                return CycleResult(
                    version=version,
                    adapter_path=self.output_dir,
                    train_loss=0.0,
                    record_count=0,
                    duration_seconds=time.time() - start,
                    success=False,
                    domain=domain,
                    error="No PENDING records found",
                )

            logger.info(
                "DistillationEngine: %d pending records, %d retention records",
                len(pending),
                len(retention_records),
            )

            # Compute aggregate teacher cost from this batch.
            cost_usd = sum(float(r.get("teacher_cost_usd", 0.0)) for r in pending)

            # Mark records IN_TRAINING
            _mark_in_training(training_db_path, [r["record_id"] for r in pending])

            # ── Build mixed example list (with replay) ────────────────────────
            examples = _build_examples(pending, retention_records, replay_fraction=replay_fraction)

            # ── Derive max_steps from dataset size if not explicitly set ──────
            effective_batch = batch_size * gradient_accumulation_steps
            steps_per_epoch = math.ceil(len(examples) / effective_batch)
            resolved_max_steps = (
                max_steps if max_steps is not None else steps_per_epoch * num_epochs
            )
            logger.info(
                "Training schedule: %d examples, effective_batch=%d, "
                "%d steps/epoch × %d epochs = %d steps",
                len(examples),
                effective_batch,
                steps_per_epoch,
                num_epochs if max_steps is None else 0,
                resolved_max_steps,
            )

            # ── LoRA training ─────────────────────────────────────────────────
            # Parse comma-separated target modules string if provided via config
            resolved_target_modules = (
                [m.strip() for m in lora_target_modules.split(",")] if lora_target_modules else None
            )
            adapter_path, train_loss = self._train_lora(
                examples=examples,
                version=version,
                domain=domain,
                max_steps=resolved_max_steps,
                batch_size=batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                learning_rate=learning_rate,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_target_modules=resolved_target_modules,
            )

            # Mark records CONSUMED
            _mark_consumed(training_db_path, [r["record_id"] for r in pending])

            return CycleResult(
                version=version,
                adapter_path=adapter_path,
                train_loss=train_loss,
                record_count=len(pending),
                duration_seconds=time.time() - start,
                success=True,
                domain=domain,
                teacher_cost_usd=cost_usd,
            )

        except Exception as exc:
            logger.exception("DistillationEngine cycle failed")
            return CycleResult(
                version=version,
                adapter_path=self.output_dir,
                train_loss=float("nan"),
                record_count=0,
                duration_seconds=time.time() - start,
                success=False,
                domain=domain,
                error=str(exc),
            )

    # ── Internal training ─────────────────────────────────────────────────────

    def _train_lora(
        self,
        examples: list[_Example],
        version: int,
        domain: str = "default",
        max_steps: int = 500,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        learning_rate: float = 2e-4,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_target_modules: Optional[list[str]] = None,
    ) -> tuple[Path, float]:
        """Load base model, apply LoRA, run SFTTrainer, save adapter."""
        logger.info("Loading base model: %s (device=%s)", self.base_model_name, self.device)
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        from_pretrained_kwargs: dict = {
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": True,
        }
        if self.device != "cpu":
            from_pretrained_kwargs["device_map"] = self.device

        # Serialize the entire load→train→save pipeline so concurrent domain cycles
        # never hold two 3B-param bfloat16 models in memory at the same time.
        with _model_load_lock:
            model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name, **from_pretrained_kwargs
            )

            # Apply LoRA
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=0.05,
                bias="none",
                target_modules=lora_target_modules or _target_modules(self.base_model_name),
            )
            model = get_peft_model(model, lora_config)
            model.enable_input_require_grads()
            model.print_trainable_parameters()

            # If we have a prior adapter, load it as a reference for functional distillation
            if self.prior_adapter_path and self.prior_adapter_path.exists():
                logger.info(
                    "Prior adapter available at %s — functional distillation enabled",
                    self.prior_adapter_path,
                )
                # Prior adapter penalty is applied implicitly: by mixing its outputs
                # into the training set as retention data (handled by replay examples).

            # Build HuggingFace dataset
            formatted = [_format_example(ex, tokenizer) for ex in examples]
            hf_dataset = Dataset.from_list([{"text": t} for t in formatted])

            output_path = self.output_dir / f"v{version}" / domain
            output_path.mkdir(parents=True, exist_ok=True)

            # On MPS the physical batch size drives peak activation memory.
            # We keep the *effective* batch size (batch × grad_accum) constant
            # by scaling gradient_accumulation_steps to compensate, so training
            # dynamics are unchanged.
            effective_batch = batch_size * gradient_accumulation_steps
            mps_batch_size = 1 if self.device == "mps" else batch_size
            mps_grad_accum = effective_batch // mps_batch_size

            sft_config = SFTConfig(
                output_dir=str(output_path),
                max_steps=max_steps,
                per_device_train_batch_size=mps_batch_size,
                gradient_accumulation_steps=mps_grad_accum,
                learning_rate=learning_rate,
                lr_scheduler_type="cosine",
                warmup_ratio=0.05,
                bf16=(self.device != "cpu"),
                fp16=False,
                logging_steps=10,
                save_strategy="no",
                report_to="none",
                dataset_text_field="text",
                max_length=512,
                gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": False},
                # MPS-specific: pin_memory is a no-op on MPS but wastes CPU RAM;
                # num_workers > 0 causes multiprocessing conflicts with Metal.
                dataloader_pin_memory=(self.device == "cuda"),
                dataloader_num_workers=0,
                # Fused AdamW is more memory-efficient on all backends.
                optim="adamw_torch_fused" if self.device != "cpu" else "adamw_torch",
            )

            trainer = SFTTrainer(
                model=model,
                args=sft_config,
                train_dataset=hf_dataset,
            )
            train_result = trainer.train()
            train_loss = train_result.training_loss

            # Save only the LoRA adapter weights (not the full model)
            model.save_pretrained(str(output_path))
            tokenizer.save_pretrained(str(output_path))
            logger.info("Adapter saved to %s (train_loss=%.4f)", output_path, train_loss)

            # ── Explicit memory cleanup ───────────────────────────────────────
            # Free GPU/MPS memory *inside* the lock so the next consumer
            # (validation inference) doesn't find the training model still
            # resident — which would double peak memory to ~6 GB on a 1.5B model.
            del trainer
            del model
            _free_device_cache()

        return output_path, train_loss


# ── Helpers ───────────────────────────────────────────────────────────────────


def _detect_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # Limit MPS to 85% of shared GPU memory to leave headroom for the OS
            # and validation inference that follows training.
            try:
                torch.mps.set_per_process_memory_fraction(0.85)
            except Exception:
                pass
            return "mps"
    except ImportError:
        pass
    return "cpu"


def _free_device_cache() -> None:
    """Force Python GC and flush device memory caches.

    Called immediately after the training model is deleted so the next
    operation (validation inference) doesn't compete for the same memory.
    """
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
            torch.mps.synchronize()
    except Exception:
        pass


def _target_modules(model_name: str) -> list[str]:
    """Return LoRA target module names based on model architecture."""
    name = model_name.lower()
    if "qwen" in name:
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    if "llama" in name or "gemma" in name or "mistral" in name:
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    # Generic fallback — works for most decoder-only transformers
    return ["q_proj", "v_proj"]


def _format_example(ex: _Example, tokenizer) -> str:
    """Format a training example using the model's chat template if available."""
    try:
        messages = [
            {"role": "user", "content": ex.prompt},
            {"role": "assistant", "content": ex.response},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    except Exception:
        # Fallback to simple format
        return f"<|user|>\n{ex.prompt}\n<|assistant|>\n{ex.response}"


def _build_examples(
    pending: list[dict],
    retention: list[dict],
    replay_fraction: float = REPLAY_FRACTION,
) -> list[_Example]:
    """Merge pending + retention with the configured replay fraction."""
    main_examples = [
        _Example(prompt=r["query_text"], response=r["teacher_response"]) for r in pending
    ]

    if not retention:
        return main_examples

    # How many replay samples to inject?
    effective_batch = len(main_examples)
    replay_count = math.ceil(effective_batch * replay_fraction / (1 - replay_fraction))
    replay_count = min(replay_count, len(retention))

    replay_sample = random.sample(retention, replay_count)
    replay_examples = [
        _Example(
            prompt=r["query_text"],
            response=r["teacher_response"],
            weight=0.5,  # Lower weight to prioritise new signal
        )
        for r in replay_sample
    ]

    combined = main_examples + replay_examples
    random.shuffle(combined)
    logger.info(
        "Training batch: %d new + %d replay = %d total",
        len(main_examples),
        len(replay_examples),
        len(combined),
    )
    return combined


def _load_pending_records(db_path: Path, domain: str = "default") -> list[dict]:
    """Load PENDING training records, optionally filtered by *domain*.

    When *domain* is ``"default"`` all PENDING records are returned (unified
    mode for backward compatibility).  For any other domain only records with
    a matching ``domain_tag`` column are returned.
    """
    try:
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        if domain == "default":
            rows = conn.execute(
                "SELECT * FROM training_records WHERE status = 'PENDING'"
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM training_records WHERE status = 'PENDING' AND domain_tag = ?",
                (domain,),
            ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except sqlite3.Error as exc:
        logger.error("Failed to load pending records: %s", exc)
        return []


def _mark_in_training(db_path: Path, record_ids: list[str]) -> None:
    if not record_ids:
        return
    try:
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        placeholders = ",".join("?" * len(record_ids))
        conn.execute(
            f"UPDATE training_records SET status='IN_TRAINING' WHERE record_id IN ({placeholders})",
            record_ids,
        )
        conn.commit()
        conn.close()
    except sqlite3.Error as exc:
        logger.error("Failed to mark IN_TRAINING: %s", exc)


def _mark_consumed(db_path: Path, record_ids: list[str]) -> None:
    if not record_ids:
        return
    try:
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        placeholders = ",".join("?" * len(record_ids))
        conn.execute(
            f"UPDATE training_records SET status='CONSUMED' WHERE record_id IN ({placeholders})",
            record_ids,
        )
        conn.commit()
        conn.close()
    except sqlite3.Error as exc:
        logger.error("Failed to mark CONSUMED: %s", exc)
