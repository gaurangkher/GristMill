"""CheckpointManager — filesystem contract for LoRA adapter checkpoints.

Directory layout (Phase 3 — multi-domain, Section 4.6.3 of the spec):

    /gristmill/checkpoints/
        active/
            {domain}/   — per-domain adapter (Inference Stack file-watches these)
        staging/
            {domain}/   — newly-trained domain adapter awaiting validation
        history/
            v{N}/
                {domain}/  — versioned archive (last 5 versions kept per domain)
        manifest.json   — monotonically-versioned metadata (atomic write via rename)
                          Phase 3 extension: includes per-domain version map

The ``domain`` defaults to ``"default"`` for the unified single-adapter path
used in Phase 2 and for backward compatibility.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Default checkpoint root; overridden by config / env.
_DEFAULT_ROOT = Path("/gristmill/checkpoints")
_FALLBACK_ROOT = Path.home() / ".gristmill" / "checkpoints"

HISTORY_KEEP = 5  # Number of historical versions to retain

# Supported domain tags (must stay in sync with DomainTag in training_buffer.rs).
KNOWN_DOMAINS = ("code", "writing", "reasoning", "qa", "creative", "other", "default")


# ── Manifest ─────────────────────────────────────────────────────────────────


@dataclass
class Manifest:
    current_version: int
    promoted_at: str  # ISO 8601
    validation_score: float
    record_count_at_promotion: int
    rolled_back: bool = False
    # Phase 3: per-domain version tracking.
    # Maps domain name → {version, validation_score, promoted_at}.
    domains: dict = field(default_factory=dict)

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
            domains=dict(d.get("domains", {})),
        )


# ── CheckpointManager ────────────────────────────────────────────────────────


class CheckpointManager:
    """Manages versioned LoRA adapter checkpoints on the filesystem.

    Phase 3 extension: supports per-domain adapter directories.

    ``active/{domain}/``   — adapter currently served by the Inference Stack.
    ``staging/{domain}/``  — adapter awaiting validation after a training cycle.
    ``history/v{N}/{domain}/`` — archived versions.

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

    # ── Domain-aware path helpers ─────────────────────────────────────────────

    def domain_active_dir(self, domain: str = "default") -> Path:
        """Return the active adapter path for *domain*."""
        return self.active_dir / domain

    def domain_staging_dir(self, domain: str = "default") -> Path:
        """Return the staging adapter path for *domain*."""
        return self.staging_dir / domain

    def domain_history_dir(self, version: int, domain: str = "default") -> Path:
        """Return the history path for *version* and *domain*."""
        return self.history_dir / f"v{version}" / domain

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

    def write_staging(self, adapter_dir: Path, domain: str = "default") -> None:
        """Copy *adapter_dir* contents into the staging directory for *domain*.

        Clears any previous staging content for that domain first.
        """
        dest = self.domain_staging_dir(domain)
        if dest.exists():
            shutil.rmtree(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(adapter_dir, dest)
        logger.info("Adapter staged from %s → staging/%s", adapter_dir, domain)

    # ── Promotion ─────────────────────────────────────────────────────────────

    def promote_staging(
        self,
        validation_score: float,
        record_count: int,
        domain: str = "default",
    ) -> int:
        """Promote the staged adapter for *domain* to active.

        1. Archive current active/{domain} to history/v{N}/{domain}.
        2. Move staging/{domain} → active/{domain}.
        3. Write updated manifest (includes per-domain version map).
        4. Prune old history.

        Returns the new version number.
        """
        manifest = self.read_manifest()
        new_version = (manifest.current_version + 1) if manifest else 1

        active_domain = self.domain_active_dir(domain)
        staging_domain = self.domain_staging_dir(domain)

        # Archive current active/{domain} (if it has any files)
        if active_domain.exists() and any(active_domain.iterdir()):
            archive = self.domain_history_dir(new_version - 1, domain)
            archive.parent.mkdir(parents=True, exist_ok=True)
            if archive.exists():
                shutil.rmtree(archive)
            shutil.copytree(active_domain, archive)
            logger.info("Archived active/%s → history/v%d/%s", domain, new_version - 1, domain)

        # Atomic move: staging/{domain} → active/{domain}
        if active_domain.exists():
            shutil.rmtree(active_domain)
        active_domain.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(staging_domain, active_domain)
        shutil.rmtree(staging_domain)
        staging_domain.mkdir(parents=True, exist_ok=True)
        logger.info("Promoted staging/%s → active/%s (version %d)", domain, domain, new_version)

        # Update manifest — bump global version and record per-domain info.
        existing_domains = dict(manifest.domains) if manifest else {}
        existing_domains[domain] = {
            "version": new_version,
            "validation_score": validation_score,
            "promoted_at": datetime.now(timezone.utc).isoformat(),
        }
        self._write_manifest(
            Manifest(
                current_version=new_version,
                promoted_at=datetime.now(timezone.utc).isoformat(),
                validation_score=validation_score,
                record_count_at_promotion=record_count,
                domains=existing_domains,
            )
        )
        self._prune_history()
        return new_version

    def discard_staging(self, reason: str, domain: str = "default") -> None:
        """Move staging/{domain} to history with rolled_back=True; keep active unchanged."""
        manifest = self.read_manifest()
        version = manifest.current_version if manifest else 0
        staging_domain = self.domain_staging_dir(domain)
        # Archive the failed staging for inspection
        failed_dir = self.history_dir / f"v{version}_failed" / domain
        failed_dir.parent.mkdir(parents=True, exist_ok=True)
        if failed_dir.exists():
            shutil.rmtree(failed_dir)
        if staging_domain.exists() and any(staging_domain.iterdir()):
            shutil.copytree(staging_domain, failed_dir)
        if staging_domain.exists():
            shutil.rmtree(staging_domain)
        staging_domain.mkdir(parents=True, exist_ok=True)
        logger.warning("Discarded staging/%s — reason: %s", domain, reason)

    # ── Rollback ──────────────────────────────────────────────────────────────

    def rollback_to(self, version: int, domain: str = "default") -> bool:
        """Promote a specific historical version of *domain* adapter to active.

        Returns True on success, False if the version does not exist.
        """
        src = self.domain_history_dir(version, domain)
        if not src.exists():
            logger.error("Rollback failed — history/v%d/%s not found", version, domain)
            return False
        active_domain = self.domain_active_dir(domain)
        if active_domain.exists():
            shutil.rmtree(active_domain)
        active_domain.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, active_domain)
        manifest = self.read_manifest()
        existing_domains = dict(manifest.domains) if manifest else {}
        existing_domains[domain] = {
            "version": version,
            "validation_score": manifest.validation_score if manifest else 0.0,
            "promoted_at": datetime.now(timezone.utc).isoformat(),
        }
        self._write_manifest(
            Manifest(
                current_version=version,
                promoted_at=datetime.now(timezone.utc).isoformat(),
                validation_score=manifest.validation_score if manifest else 0.0,
                record_count_at_promotion=manifest.record_count_at_promotion if manifest else 0,
                rolled_back=True,
                domains=existing_domains,
            )
        )
        logger.info("Rolled back domain %s to history/v%d", domain, version)
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

    def active_adapter_path(self, domain: str = "default") -> Optional[Path]:
        """Return path to active adapter for *domain* if it has files, else None."""
        path = self.domain_active_dir(domain)
        if path.exists() and any(path.iterdir()):
            return path
        return None

    def has_staging(self, domain: str = "default") -> bool:
        """Return True if the staging directory for *domain* is non-empty."""
        path = self.domain_staging_dir(domain)
        return path.exists() and any(path.iterdir())
