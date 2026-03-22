"""GristMillTrainerService — state machine and orchestrator for gristmill-trainer.

State machine (Section 4.6.1):
    IDLE        → polling buffer every 60s, no GPU held
    WAITING     → trigger met but resource gate blocked, retry every 5 min
    TRAINING    → active distillation cycle
    VALIDATING  → post-cycle validation running
    PROMOTING   → validation passed, checkpoint being moved to active
    ROLLING_BACK → validation failed, staged adapter discarded
    PAUSED      → user toggled "Pause learning"; no cycles initiated

Trigger condition (Section 4.5.4):
    • 500+ PENDING records  OR
    • 7 days elapsed since last cycle

Resource gate:
    • inference.lock heartbeat stale by > 30s  (GPU is idle)
    • Estimated VRAM < available (with 512 MB safety margin)
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from gristmill_ml.trainer.checkpoint import CheckpointManager
from gristmill_ml.trainer.distillation import DistillationEngine
from gristmill_ml.trainer.ipc_server import TrainerIpcServer
from gristmill_ml.trainer.retention import RetentionBuffer
from gristmill_ml.trainer.validation import ValidationResult, ValidationRunner

logger = logging.getLogger(__name__)

# Trigger thresholds
PENDING_TRIGGER = 500
CYCLE_CADENCE_DAYS = 7

# Resource gate
HEARTBEAT_STALE_SECS = 30
VRAM_SAFETY_MB = 512

# Polling / retry intervals (seconds)
IDLE_POLL_SECS = 60
WAITING_RETRY_SECS = 300  # 5 min

# trainer.status write interval during active cycles
STATUS_WRITE_SECS = 30


class TrainerState(str, Enum):
    IDLE = "IDLE"
    WAITING = "WAITING"
    TRAINING = "TRAINING"
    VALIDATING = "VALIDATING"
    PROMOTING = "PROMOTING"
    ROLLING_BACK = "ROLLING_BACK"
    PAUSED = "PAUSED"


@dataclass
class CycleSummary:
    version: int
    started_at: str
    completed_at: str
    record_count: int
    duration_minutes: float
    validation_score: float
    rolled_back: bool
    error: Optional[str] = None


@dataclass
class ServiceState:
    state: str = TrainerState.IDLE
    current_version: int = 0
    last_cycle_at: Optional[str] = None
    next_trigger_at: Optional[str] = None
    buffer_pending_count: int = 0
    autonomy_pct_7d: float = 0.0
    paused_reason: Optional[str] = None
    uptime_seconds: float = 0.0
    last_heartbeat_seen: Optional[str] = None


class GristMillTrainerService:
    """Orchestrates the full distillation lifecycle.

    Designed to run inside an asyncio event loop.  All blocking I/O
    (SQLite reads, model training) is dispatched via ``run_in_executor``
    to a thread pool so the event loop remains responsive for the health
    API and IPC server.
    """

    def __init__(
        self,
        training_db_path: Optional[Path] = None,
        base_model_name: Optional[str] = None,
        checkpoint_root: Optional[Path] = None,
        inference_lock_path: Optional[Path] = None,
        status_file_path: Optional[Path] = None,
        ipc_server: Optional[TrainerIpcServer] = None,
    ) -> None:
        import os

        self.training_db_path = training_db_path or _resolve_db_path()
        self.base_model_name = base_model_name or os.environ.get(
            "GRISTMILL_BASE_MODEL", "Qwen/Qwen2.5-3B-Instruct"
        )
        self.inference_lock_path = inference_lock_path or _resolve_lock_path()
        self.status_file_path = status_file_path or _resolve_status_path()
        self.ipc_server = ipc_server or TrainerIpcServer()

        self.checkpoint_mgr = CheckpointManager(checkpoint_root)
        self.retention_buf = RetentionBuffer()
        self.validation_runner = ValidationRunner(base_model_name=self.base_model_name)

        self._state = TrainerState.IDLE
        self._start_time = time.time()
        self._last_cycle_at: Optional[float] = None
        self._cycle_history: list[CycleSummary] = []
        self._latest_validation: Optional[dict] = None
        self._lock = asyncio.Lock()

    # ── Main event loop entry ─────────────────────────────────────────────────

    async def run(self) -> None:
        """Start the scheduler loop.  Runs indefinitely; cancel to stop."""
        logger.info("GristMillTrainerService starting (state=%s)", self._state)
        while True:
            try:
                await self._tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Unexpected error in trainer tick")
            await asyncio.sleep(IDLE_POLL_SECS)

    async def _tick(self) -> None:
        async with self._lock:
            if self._state == TrainerState.PAUSED:
                return
            if self._state not in (TrainerState.IDLE, TrainerState.WAITING):
                return  # Another coroutine is running the cycle
            if not self._trigger_condition_met():
                self._state = TrainerState.IDLE
                return
            if not await self._resource_gate_ok():
                if self._state != TrainerState.WAITING:
                    self._state = TrainerState.WAITING
                    logger.info("Trigger met but resource gate blocked — WAITING")
                return
            # All conditions met — start a cycle
            asyncio.create_task(self._run_cycle())

    # ── Trigger condition ─────────────────────────────────────────────────────

    def _trigger_condition_met(self) -> bool:
        pending = self._count_pending()
        if pending >= PENDING_TRIGGER:
            logger.debug("Trigger: %d pending records >= %d", pending, PENDING_TRIGGER)
            return True
        if self._last_cycle_at is not None:
            days_since = (time.time() - self._last_cycle_at) / 86400
            if days_since >= CYCLE_CADENCE_DAYS:
                logger.debug(
                    "Trigger: %.1f days since last cycle >= %d", days_since, CYCLE_CADENCE_DAYS
                )
                return True
        return False

    def _count_pending(self) -> int:
        try:
            conn = sqlite3.connect(str(self.training_db_path))
            count = conn.execute(
                "SELECT COUNT(*) FROM training_records WHERE status='PENDING'"
            ).fetchone()[0]
            conn.close()
            return count
        except sqlite3.Error:
            return 0

    # ── Resource gate ─────────────────────────────────────────────────────────

    async def _resource_gate_ok(self) -> bool:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._check_resources_sync)

    def _check_resources_sync(self) -> bool:
        # 1. Inference.lock heartbeat check
        if self.inference_lock_path.exists():
            try:
                epoch = int(self.inference_lock_path.read_text().strip())
                age = time.time() - epoch
                if age < HEARTBEAT_STALE_SECS:
                    logger.debug("inference.lock fresh (age=%.1fs) — deferring", age)
                    return False
            except (ValueError, OSError):
                pass  # Unreadable lock → treat as stale

        # 2. VRAM check (best-effort)
        try:
            import torch

            if torch.cuda.is_available():
                free_mb = torch.cuda.mem_get_info()[0] / (1024**2)
                if free_mb < VRAM_SAFETY_MB:
                    logger.debug("Insufficient VRAM (free=%.0f MB) — deferring", free_mb)
                    return False
        except Exception:
            pass  # No CUDA or torch not installed → proceed

        return True

    # ── Cycle orchestration ───────────────────────────────────────────────────

    async def _run_cycle(self) -> None:
        cycle_start = time.time()
        cycle_version = (
            self.checkpoint_mgr.read_manifest() or type("M", (), {"current_version": 0})()
        ).current_version + 1

        async with self._lock:
            self._state = TrainerState.TRAINING

        await self.ipc_server.emit_training_started(
            estimated_duration_minutes=60,
            record_count=self._count_pending(),
        )

        # Progress heartbeat task
        progress_task = asyncio.create_task(self._progress_heartbeat(cycle_start))

        try:
            loop = asyncio.get_running_loop()

            # ── Curate retention buffer ────────────────────────────────────────
            await loop.run_in_executor(
                None,
                lambda: self.retention_buf.curate(self.training_db_path),
            )
            retention_records = [
                {
                    "record_id": r.record_id,
                    "query_text": r.query_text,
                    "teacher_response": r.teacher_response,
                    "confidence_score": r.confidence_score,
                    "domain_tag": r.domain_tag,
                }
                for r in self.retention_buf.get_all()
            ]

            # ── Run LoRA training ─────────────────────────────────────────────
            prior_adapter = self.checkpoint_mgr.active_adapter_path()
            engine = DistillationEngine(
                base_model_name=self.base_model_name,
                prior_adapter_path=prior_adapter,
            )
            cycle_result = await loop.run_in_executor(
                None,
                lambda: engine.run_cycle(
                    training_db_path=self.training_db_path,
                    retention_records=retention_records,
                    version=cycle_version,
                ),
            )

            if not cycle_result.success:
                logger.error("Training cycle failed: %s", cycle_result.error)
                self._record_cycle(cycle_start, cycle_version, 0, False, cycle_result.error)
                async with self._lock:
                    self._state = TrainerState.IDLE
                return

            # Stage the adapter
            self.checkpoint_mgr.write_staging(cycle_result.adapter_path)

            # ── Validation ────────────────────────────────────────────────────
            async with self._lock:
                self._state = TrainerState.VALIDATING

            self.validation_runner.ensure_validation_set(self.training_db_path)
            val_result: ValidationResult = await loop.run_in_executor(
                None,
                lambda: self.validation_runner.validate(
                    staged_adapter_path=self.checkpoint_mgr.staging_dir,
                    prior_adapter_path=prior_adapter,
                    retention_records=retention_records,
                ),
            )
            self._latest_validation = val_result.to_dict()

            if val_result.passed:
                await self._promote(
                    cycle_start, cycle_version, cycle_result.record_count, val_result
                )
            else:
                await self._rollback(
                    cycle_version, val_result.failure_reason or "validation failed"
                )

        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Cycle error")
            self._record_cycle(cycle_start, cycle_version, 0, False, "unexpected error")
            async with self._lock:
                self._state = TrainerState.IDLE
        finally:
            progress_task.cancel()
            self._write_status_file()

    async def _promote(
        self,
        cycle_start: float,
        version: int,
        record_count: int,
        val_result: ValidationResult,
    ) -> None:
        async with self._lock:
            self._state = TrainerState.PROMOTING

        new_version = self.checkpoint_mgr.promote_staging(
            validation_score=val_result.overall_score,
            record_count=record_count,
        )
        self._last_cycle_at = time.time()
        self._record_cycle(cycle_start, new_version, record_count, False, None)

        await self.ipc_server.emit_checkpoint_promoted(
            version=new_version,
            validation_score=val_result.overall_score,
            record_count=record_count,
        )
        logger.info(
            "Checkpoint promoted to v%d (score=%.4f)", new_version, val_result.overall_score
        )

        async with self._lock:
            self._state = TrainerState.IDLE

    async def _rollback(self, version: int, reason: str) -> None:
        async with self._lock:
            self._state = TrainerState.ROLLING_BACK

        self.checkpoint_mgr.discard_staging(reason)
        self._record_cycle(time.time(), version, 0, True, reason)

        await self.ipc_server.emit_checkpoint_rolled_back(version=version, reason=reason)
        logger.warning("Checkpoint rolled back (reason=%s)", reason)

        async with self._lock:
            self._state = TrainerState.IDLE

    async def _progress_heartbeat(self, start: float) -> None:
        """Emit training_progress every STATUS_WRITE_SECS during active cycle."""
        try:
            while True:
                await asyncio.sleep(STATUS_WRITE_SECS)
                elapsed = (time.time() - start) / 60
                # Rough estimate: assume 60-min cycle
                pct = min(0.99, elapsed / 60)
                await self.ipc_server.emit_training_progress(pct, elapsed)
                self._write_status_file()
        except asyncio.CancelledError:
            pass

    # ── Pause / Resume ────────────────────────────────────────────────────────

    def pause(self, reason: str) -> None:
        self._state = TrainerState.PAUSED
        self._pause_reason = reason
        asyncio.create_task(self.ipc_server.emit_trainer_paused(reason))
        logger.info("Trainer paused (reason=%s)", reason)

    def resume(self) -> None:
        if self._state == TrainerState.PAUSED:
            self._state = TrainerState.IDLE
        logger.info("Trainer resumed")

    # ── Manual rollback ───────────────────────────────────────────────────────

    def manual_rollback(self, version: int) -> bool:
        ok = self.checkpoint_mgr.rollback_to(version)
        if ok:
            asyncio.create_task(
                self.ipc_server.emit_checkpoint_promoted(
                    version=version,
                    validation_score=0.0,
                    record_count=0,
                )
            )
        return ok

    # ── Health / status snapshots ─────────────────────────────────────────────

    def health_info(self) -> dict[str, Any]:
        return {
            "uptime_seconds": round(time.time() - self._start_time, 1),
            "last_heartbeat_seen": self._read_last_heartbeat(),
        }

    def status_snapshot(self) -> dict[str, Any]:
        manifest = self.checkpoint_mgr.read_manifest()
        return {
            "state": self._state,
            "current_version": manifest.current_version if manifest else 0,
            "last_cycle_at": (
                datetime.fromtimestamp(self._last_cycle_at, tz=timezone.utc).isoformat()
                if self._last_cycle_at
                else None
            ),
            "buffer_pending_count": self._count_pending(),
        }

    def cycle_history(self) -> list[dict]:
        return [asdict(c) for c in reversed(self._cycle_history)]

    def latest_validation_result(self) -> Optional[dict]:
        return self._latest_validation

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _record_cycle(
        self,
        start: float,
        version: int,
        record_count: int,
        rolled_back: bool,
        error: Optional[str],
    ) -> None:
        now = time.time()
        self._cycle_history.append(
            CycleSummary(
                version=version,
                started_at=datetime.fromtimestamp(start, tz=timezone.utc).isoformat(),
                completed_at=datetime.fromtimestamp(now, tz=timezone.utc).isoformat(),
                record_count=record_count,
                duration_minutes=round((now - start) / 60, 2),
                validation_score=self._latest_validation.get("overall_score", 0.0)
                if self._latest_validation
                else 0.0,
                rolled_back=rolled_back,
                error=error,
            )
        )
        # Keep last 50 cycle records in memory
        if len(self._cycle_history) > 50:
            self._cycle_history = self._cycle_history[-50:]

    def _write_status_file(self) -> None:
        """Write trainer.status JSON every 30s for UI consumption."""
        import json

        try:
            self.status_file_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.status_file_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(self.status_snapshot(), indent=2))
            tmp.replace(self.status_file_path)
        except OSError as exc:
            logger.debug("Could not write trainer.status: %s", exc)

    def _read_last_heartbeat(self) -> Optional[str]:
        try:
            epoch = int(self.inference_lock_path.read_text().strip())
            return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()
        except Exception:
            return None


# ── Path resolution helpers ───────────────────────────────────────────────────


def _resolve_db_path() -> Path:
    default = Path("/gristmill/db/training_buffer.sqlite")
    if default.parent.exists():
        return default
    return Path.home() / ".gristmill" / "db" / "training_buffer.sqlite"


def _resolve_lock_path() -> Path:
    default = Path("/gristmill/run/inference.lock")
    if default.parent.exists():
        return default
    return Path.home() / ".gristmill" / "run" / "inference.lock"


def _resolve_status_path() -> Path:
    default = Path("/gristmill/run/trainer.status")
    if default.parent.exists():
        return default
    return Path.home() / ".gristmill" / "run" / "trainer.status"
