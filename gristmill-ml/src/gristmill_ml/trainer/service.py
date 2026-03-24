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
    domain: str = "default"
    teacher_cost_usd: float = 0.0
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
    teacher_cost_usd_total: float = 0.0
    domains: dict = None

    def __post_init__(self) -> None:
        if self.domains is None:
            self.domains = {}


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
        # Phase 3: per-domain cost tracking and active-cycle guard.
        self._teacher_cost_usd_total: float = 0.0
        self._cost_by_domain: dict[str, float] = {}
        self._active_domains: set[str] = set()  # domains currently in a cycle

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

            # Phase 3: check trigger condition per domain and spawn per-domain cycles.
            from gristmill_ml.trainer.checkpoint import KNOWN_DOMAINS

            triggered_domains = [
                d
                for d in KNOWN_DOMAINS
                if d not in self._active_domains and self._trigger_condition_met(domain=d)
            ]

            if not triggered_domains:
                if not self._active_domains:
                    self._state = TrainerState.IDLE
                return

            if not await self._resource_gate_ok():
                if self._state != TrainerState.WAITING:
                    self._state = TrainerState.WAITING
                    logger.info("Trigger met but resource gate blocked — WAITING")
                return

            for domain in triggered_domains:
                self._active_domains.add(domain)
                asyncio.create_task(self._run_cycle(domain=domain))

    # ── Trigger condition ─────────────────────────────────────────────────────

    def _trigger_condition_met(self, domain: str = "default") -> bool:
        pending = self._count_pending(domain=domain)
        if pending >= PENDING_TRIGGER:
            logger.debug("Trigger [%s]: %d pending records >= %d", domain, pending, PENDING_TRIGGER)
            return True
        if self._last_cycle_at is not None:
            days_since = (time.time() - self._last_cycle_at) / 86400
            if days_since >= CYCLE_CADENCE_DAYS:
                logger.debug(
                    "Trigger [%s]: %.1f days since last cycle >= %d",
                    domain,
                    days_since,
                    CYCLE_CADENCE_DAYS,
                )
                return True
        return False

    def _count_pending(self, domain: str = "default") -> int:
        try:
            conn = sqlite3.connect(str(self.training_db_path))
            if domain == "default":
                count = conn.execute(
                    "SELECT COUNT(*) FROM training_records WHERE status='PENDING'"
                ).fetchone()[0]
            else:
                count = conn.execute(
                    "SELECT COUNT(*) FROM training_records WHERE status='PENDING' AND domain_tag=?",
                    (domain,),
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

    async def _run_cycle(self, domain: str = "default") -> None:
        cycle_start = time.time()
        cycle_version = (
            self.checkpoint_mgr.read_manifest() or type("M", (), {"current_version": 0})()
        ).current_version + 1

        async with self._lock:
            self._state = TrainerState.TRAINING

        await self.ipc_server.emit_training_started(
            estimated_duration_minutes=60,
            record_count=self._count_pending(domain=domain),
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
            prior_adapter = self.checkpoint_mgr.active_adapter_path(domain=domain)
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
                    domain=domain,
                ),
            )

            if not cycle_result.success:
                logger.error("Training cycle [%s] failed: %s", domain, cycle_result.error)
                self._record_cycle(
                    cycle_start, cycle_version, 0, False, cycle_result.error, domain=domain
                )
                async with self._lock:
                    self._active_domains.discard(domain)
                    if not self._active_domains:
                        self._state = TrainerState.IDLE
                return

            # Stage the adapter for this domain
            self.checkpoint_mgr.write_staging(cycle_result.adapter_path, domain=domain)

            # ── Validation ────────────────────────────────────────────────────
            async with self._lock:
                self._state = TrainerState.VALIDATING

            self.validation_runner.ensure_validation_set(self.training_db_path)
            val_result: ValidationResult = await loop.run_in_executor(
                None,
                lambda: self.validation_runner.validate(
                    staged_adapter_path=self.checkpoint_mgr.domain_staging_dir(domain),
                    prior_adapter_path=prior_adapter,
                    retention_records=retention_records,
                ),
            )
            self._latest_validation = val_result.to_dict()

            if val_result.passed:
                await self._promote(
                    cycle_start,
                    cycle_version,
                    cycle_result.record_count,
                    val_result,
                    domain=domain,
                    teacher_cost_usd=cycle_result.teacher_cost_usd,
                )
            else:
                await self._rollback(
                    cycle_version,
                    val_result.failure_reason or "validation failed",
                    domain=domain,
                )

        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Cycle error [domain=%s]", domain)
            self._record_cycle(
                cycle_start, cycle_version, 0, False, "unexpected error", domain=domain
            )
            async with self._lock:
                self._active_domains.discard(domain)
                if not self._active_domains:
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
        domain: str = "default",
        teacher_cost_usd: float = 0.0,
    ) -> None:
        async with self._lock:
            self._state = TrainerState.PROMOTING

        new_version = self.checkpoint_mgr.promote_staging(
            validation_score=val_result.overall_score,
            record_count=record_count,
            domain=domain,
        )
        self._last_cycle_at = time.time()
        # Accumulate cost totals.
        self._teacher_cost_usd_total += teacher_cost_usd
        self._cost_by_domain[domain] = self._cost_by_domain.get(domain, 0.0) + teacher_cost_usd
        self._record_cycle(
            cycle_start,
            new_version,
            record_count,
            False,
            None,
            domain=domain,
            teacher_cost_usd=teacher_cost_usd,
        )

        await self.ipc_server.emit_checkpoint_promoted(
            version=new_version,
            validation_score=val_result.overall_score,
            record_count=record_count,
            domain=domain,
        )
        logger.info(
            "Checkpoint [%s] promoted to v%d (score=%.4f, cost=$%.4f)",
            domain,
            new_version,
            val_result.overall_score,
            teacher_cost_usd,
        )

        async with self._lock:
            self._active_domains.discard(domain)
            if not self._active_domains:
                self._state = TrainerState.IDLE

    async def _rollback(self, version: int, reason: str, domain: str = "default") -> None:
        async with self._lock:
            self._state = TrainerState.ROLLING_BACK

        self.checkpoint_mgr.discard_staging(reason, domain=domain)
        self._record_cycle(time.time(), version, 0, True, reason, domain=domain)

        await self.ipc_server.emit_checkpoint_rolled_back(version=version, reason=reason)
        logger.warning("Checkpoint [%s] rolled back (reason=%s)", domain, reason)

        async with self._lock:
            self._active_domains.discard(domain)
            if not self._active_domains:
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

    def manual_rollback(self, version: int, domain: str = "default") -> bool:
        ok = self.checkpoint_mgr.rollback_to(version, domain=domain)
        if ok:
            asyncio.create_task(
                self.ipc_server.emit_checkpoint_promoted(
                    version=version,
                    validation_score=0.0,
                    record_count=0,
                    domain=domain,
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
            "teacher_cost_usd_total": round(self._teacher_cost_usd_total, 6),
            "domains": manifest.domains if manifest else {},
        }

    def cost_summary(self) -> dict[str, Any]:
        """Return teacher compute cost breakdown by domain."""
        return {
            "total_usd": round(self._teacher_cost_usd_total, 6),
            "by_domain": {domain: round(cost, 6) for domain, cost in self._cost_by_domain.items()},
            "cycle_count": len(self._cycle_history),
        }

    def cycle_history(self) -> list[dict]:
        return [asdict(c) for c in reversed(self._cycle_history)]

    def latest_validation_result(self) -> Optional[dict]:
        return self._latest_validation

    # ── Phase 4: Ecosystem methods ────────────────────────────────────────────

    def ecosystem_status(self) -> dict[str, Any]:
        """Return community opt-in status, federation status, and privacy budget."""
        from gristmill_ml.community.client import CommunityRepoClient
        from gristmill_ml.federated.contributor import GradientContributor
        from gristmill_ml.federated.aggregator import PrivacyAccountant

        client = CommunityRepoClient.from_config()
        contributor = GradientContributor.from_config()
        accountant = PrivacyAccountant()
        budget = accountant.budget

        return {
            "community": {
                "enabled": client._enabled,
                "endpoint": client._endpoint,
            },
            "federated": {
                "enabled": contributor._enabled,
                "privacy_budget": {
                    "epsilon_used": round(budget.epsilon_used, 4),
                    "epsilon_budget": round(budget.epsilon_budget, 4),
                    "remaining": round(budget.remaining, 4),
                    "exhausted": budget.exhausted,
                    "cycles_contributed": budget.cycles_contributed,
                },
            },
        }

    def export_adapter(self, domain: str, output_dir: Optional[Path] = None) -> dict[str, Any]:
        """Pack the active adapter for *domain* into a .gmpack bundle.

        Returns dict with ``gmpack_path`` on success, or ``error`` on failure.
        """
        from gristmill_ml.export.bundle import AdapterBundle

        active = self.checkpoint_mgr.active_adapter_path(domain)
        if active is None:
            return {"ok": False, "error": f"No active adapter for domain '{domain}'"}

        manifest = self.checkpoint_mgr.read_manifest()
        score = 0.0
        record_count = 0
        base_model = self.base_model_name
        if manifest and domain in manifest.domains:
            score = manifest.domains[domain].get("validation_score", 0.0)

        if output_dir is None:
            output_dir = active.parent.parent  # checkpoints root

        try:
            gmpack = AdapterBundle.pack(
                adapter_dir=active,
                domain=domain,
                output_path=Path(output_dir) / f"{domain}-export.gmpack",
                base_model=base_model,
                validation_score=score,
                record_count=record_count,
            )
            return {"ok": True, "gmpack_path": str(gmpack), "domain": domain}
        except Exception as exc:
            logger.error("export_adapter failed: %s", exc)
            return {"ok": False, "error": str(exc)}

    def import_adapter(self, gmpack_path: Path, domain: Optional[str] = None) -> dict[str, Any]:
        """Unpack *gmpack_path*, stage, and promote it for *domain*.

        If *domain* is ``None``, the domain from the bundle manifest is used.
        """
        from gristmill_ml.export.bundle import AdapterBundle
        import tempfile

        gmpack_path = Path(gmpack_path)
        if not gmpack_path.exists():
            return {"ok": False, "error": f"File not found: {gmpack_path}"}

        with tempfile.TemporaryDirectory(prefix="gm_import_") as tmp:
            adapter_dest = Path(tmp) / "adapter"
            try:
                bundle_manifest = AdapterBundle.unpack(gmpack_path, adapter_dest)
            except Exception as exc:
                return {"ok": False, "error": f"Unpack failed: {exc}"}

            target_domain = domain or bundle_manifest.domain
            self.checkpoint_mgr.write_staging(adapter_dest, domain=target_domain)
            version = self.checkpoint_mgr.promote_staging(
                validation_score=bundle_manifest.validation_score,
                record_count=bundle_manifest.record_count,
                domain=target_domain,
            )

        return {
            "ok": True,
            "domain": target_domain,
            "version": version,
            "validation_score": bundle_manifest.validation_score,
            "anonymized_id": bundle_manifest.anonymized_id,
        }

    def community_list_adapters(
        self, domain: str, min_score: float = 0.0, limit: int = 20
    ) -> dict[str, Any]:
        """List community adapters for *domain*."""
        from gristmill_ml.community.client import CommunityRepoClient
        from dataclasses import asdict

        client = CommunityRepoClient.from_config()
        if not client._enabled:
            return {"ok": False, "error": "Community repository not enabled"}
        try:
            adapters = client.list_adapters(domain, min_score=min_score, limit=limit)
            return {"ok": True, "adapters": [asdict(a) for a in adapters]}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def community_push(self, domain: str) -> dict[str, Any]:
        """Export active adapter for *domain* and push to community repo."""
        export_result = self.export_adapter(domain)
        if not export_result.get("ok"):
            return export_result

        from gristmill_ml.community.client import CommunityRepoClient

        client = CommunityRepoClient.from_config()
        if not client._enabled:
            return {"ok": False, "error": "Community repository not enabled"}
        try:
            adapter_id = client.push(Path(export_result["gmpack_path"]))
            return {"ok": True, "adapter_id": adapter_id, "domain": domain}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def bootstrap_domain(self, domain: str, force: bool = False) -> dict[str, Any]:
        """Trigger cold-start bootstrapping for *domain* from the community repo."""
        from gristmill_ml.community.bootstrap import ColdStartBootstrapper

        bootstrapper = ColdStartBootstrapper(
            checkpoint_root=self.checkpoint_mgr.root,
            db_path=self.training_db_path,
        )
        try:
            success = bootstrapper.bootstrap(domain, force=force)
            return {"ok": True, "bootstrapped": success, "domain": domain}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _record_cycle(
        self,
        start: float,
        version: int,
        record_count: int,
        rolled_back: bool,
        error: Optional[str],
        domain: str = "default",
        teacher_cost_usd: float = 0.0,
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
                domain=domain,
                teacher_cost_usd=teacher_cost_usd,
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
