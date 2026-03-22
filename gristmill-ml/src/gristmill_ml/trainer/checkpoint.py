"""CheckpointManager — filesystem contract for LoRA adapter checkpoints.

Directory layout (Section 4.6.3 of the spec):

    /gristmill/checkpoints/
        active/         — currently-loaded adapter (Inference Stack file-watches this)
        staging/        — newly-trained adapter awaiting validation
        history/v{N}/   — versioned archive (last 5 kept)
        manifest.json   — monotonically-versioned metadata (atomic write via rename)
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Default checkpoint root; overridden by config / env.
_DEFAULT_ROOT = Path("/gristmill/checkpoints")
_FALLBACK_ROOT = Path.home() / ".gristmill" / "checkpoints"

HISTORY_KEEP = 5  # Number of historical versions to retain


# ── Manifest ─────────────────────────────────────────────────────────────────


@dataclass
class Manifest:
    current_version: int
    promoted_at: str          # ISO 8601
    validation_score: float
    record_count_at_promotion: int
    rolled_back: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Manifest":
        return cls(
            current_version=int(d["current_version"]),
            promoted_at=d["promoted_at"],
            validation_score=float(d["validation_score"]),
            record_count_at_promotion=int(d["record_count_at_promotion"]),
            rolled_back=bool(d.get("rolled_back", False)),
        )


# ── CheckpointManager ────────────────────────────────────────────────────────


class CheckpointManager:
    """Manages versioned LoRA adapter checkpoints on the filesystem.

    All mutations are safe to call concurrently with the Inference Stack
    reading ``active/`` — the critical path (active promotion) is performed
    via an atomic ``shutil.move`` which maps to ``rename(2)`` on Linux/macOS
    when source and destination are on the same filesystem.
    """

    def __init__(self, root: Optional[Path] = None) -> None:
        if root is None:
            root = _DEFAULT_ROOT if _DEFAULT_ROOT.exists() else _FALLBACK_ROOT
        self.root = root
        self.active_dir = root / "active"
        self.staging_dir = root / "staging"
        self.history_dir = root / "history"
        self.manifest_path = root / "manifest.json"
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        for d in (self.active_dir, self.staging_dir, self.history_dir):
            d.mkdir(parents=True, exist_ok=True)

    # ── Manifest ──────────────────────────────────────────────────────────────

    def read_manifest(self) -> Optional[Manifest]:
        if not self.manifest_path.exists():
            return None
        try:
            return Manifest.from_dict(json.loads(self.manifest_path.read_text()))
        except Exception as exc:
            logger.warning("Failed to read manifest: %s", exc)
            return None

    def _write_manifest(self, manifest: Manifest) -> None:
        """Atomic write via temp-file rename."""
        tmp = self.manifest_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(manifest.to_dict(), indent=2))
        tmp.replace(self.manifest_path)
        logger.debug("Manifest written: version=%d", manifest.current_version)

    # ── Staging ───────────────────────────────────────────────────────────────

    def write_staging(self, adapter_dir: Path) -> None:
        """Copy *adapter_dir* contents into the staging directory.

        Clears any previous staging content first.
        """
        if self.staging_dir.exists():
            shutil.rmtree(self.staging_dir)
        shutil.copytree(adapter_dir, self.staging_dir)
        logger.info("Adapter staged from %s", adapter_dir)

    # ── Promotion ─────────────────────────────────────────────────────────────

    def promote_staging(
        self,
        validation_score: float,
        record_count: int,
    ) -> int:
        """Promote the staged adapter to active.

        1. Archive current active to history/v{N}.
        2. Move staging → active.
        3. Write updated manifest.
        4. Prune old history.

        Returns the new version number.
        """
        manifest = self.read_manifest()
        new_version = (manifest.current_version + 1) if manifest else 1

        # Archive current active (if it has any files)
        if any(self.active_dir.iterdir()):
            archive = self.history_dir / f"v{new_version - 1}"
            archive.mkdir(parents=True, exist_ok=True)
            if archive.exists() and any(archive.iterdir()):
                shutil.rmtree(archive)
            shutil.copytree(self.active_dir, archive, dirs_exist_ok=True)
            logger.info("Archived active adapter → history/v%d", new_version - 1)

        # Atomic move: staging → active
        if self.active_dir.exists():
            shutil.rmtree(self.active_dir)
        shutil.copytree(self.staging_dir, self.active_dir)
        shutil.rmtree(self.staging_dir)
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Promoted staging → active (version %d)", new_version)

        self._write_manifest(
            Manifest(
                current_version=new_version,
                promoted_at=datetime.now(timezone.utc).isoformat(),
                validation_score=validation_score,
                record_count_at_promotion=record_count,
            )
        )
        self._prune_history()
        return new_version

    def discard_staging(self, reason: str) -> None:
        """Move staging to history with rolled_back=True, keep active unchanged."""
        manifest = self.read_manifest()
        version = (manifest.current_version if manifest else 0)
        # Archive the failed staging for inspection
        failed_dir = self.history_dir / f"v{version}_failed"
        if failed_dir.exists():
            shutil.rmtree(failed_dir)
        if any(self.staging_dir.iterdir()):
            shutil.copytree(self.staging_dir, failed_dir)
        shutil.rmtree(self.staging_dir)
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        logger.warning("Discarded staging adapter — reason: %s", reason)

    # ── Rollback ──────────────────────────────────────────────────────────────

    def rollback_to(self, version: int) -> bool:
        """Promote a specific historical version to active.

        Returns True on success, False if the version does not exist.
        """
        src = self.history_dir / f"v{version}"
        if not src.exists():
            logger.error("Rollback failed — history/v%d not found", version)
            return False
        if self.active_dir.exists():
            shutil.rmtree(self.active_dir)
        shutil.copytree(src, self.active_dir)
        manifest = self.read_manifest()
        self._write_manifest(
            Manifest(
                current_version=version,
                promoted_at=datetime.now(timezone.utc).isoformat(),
                validation_score=manifest.validation_score if manifest else 0.0,
                record_count_at_promotion=manifest.record_count_at_promotion if manifest else 0,
                rolled_back=True,
            )
        )
        logger.info("Rolled back to history/v%d", version)
        return True

    # ── History listing ───────────────────────────────────────────────────────

    def list_history(self) -> list[int]:
        """Return sorted list of available historical version numbers."""
        versions = []
        for d in self.history_dir.iterdir():
            name = d.name
            if name.startswith("v") and name[1:].isdigit():
                versions.append(int(name[1:]))
        return sorted(versions)

    # ── Pruning ───────────────────────────────────────────────────────────────

    def _prune_history(self) -> None:
        versions = self.list_history()
        while len(versions) > HISTORY_KEEP:
            oldest = versions.pop(0)
            old_dir = self.history_dir / f"v{oldest}"
            shutil.rmtree(old_dir, ignore_errors=True)
            logger.debug("Pruned history/v%d", oldest)

    # ── Active adapter path ───────────────────────────────────────────────────

    def active_adapter_path(self) -> Optional[Path]:
        """Return path to active adapter if it has files, else None."""
        if self.active_dir.exists() and any(self.active_dir.iterdir()):
            return self.active_dir
        return None

    def has_staging(self) -> bool:
        return self.staging_dir.exists() and any(self.staging_dir.iterdir())
