"""AdapterBundle — portable `.gmpack` export format for GristMill adapters.

A `.gmpack` file is a ZIP archive with a deterministic internal layout:

    bundle.json          — bundle manifest (schema version, domain, metadata)
    adapter/             — verbatim copy of the adapter directory
        adapter_config.json
        adapter_model.bin   (or safetensors shards)
        tokenizer_config.json   (optional)
        …

Schema version is bumped only on breaking changes.  Readers must reject
bundles whose ``schema_version`` is greater than ``BUNDLE_SCHEMA_VERSION``.

Usage::

    # Export
    path = AdapterBundle.pack(adapter_dir, domain="code", metadata={...})

    # Import
    manifest = AdapterBundle.unpack(path, dest_dir=staging_dir)
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

BUNDLE_SCHEMA_VERSION = 1
BUNDLE_EXTENSION = ".gmpack"
_MANIFEST_NAME = "bundle.json"
_ADAPTER_PREFIX = "adapter/"


# ── Manifest ──────────────────────────────────────────────────────────────────


@dataclass
class BundleManifest:
    """Metadata stored in ``bundle.json`` inside every `.gmpack` file."""

    schema_version: int
    domain: str
    base_model: str
    validation_score: float
    record_count: int
    created_at: str          # ISO 8601 UTC
    anonymized_id: str       # UUID4 — no PII
    sha256: str              # hex digest of adapter/ tree (deterministic)
    gristmill_version: str = "2.0"
    tags: list = field(default_factory=list)  # e.g. ["code", "python"]
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "BundleManifest":
        return cls(
            schema_version=int(d["schema_version"]),
            domain=d["domain"],
            base_model=d["base_model"],
            validation_score=float(d["validation_score"]),
            record_count=int(d["record_count"]),
            created_at=d["created_at"],
            anonymized_id=d["anonymized_id"],
            sha256=d["sha256"],
            gristmill_version=d.get("gristmill_version", "2.0"),
            tags=list(d.get("tags", [])),
            notes=str(d.get("notes", "")),
        )


# ── AdapterBundle ─────────────────────────────────────────────────────────────


class AdapterBundle:
    """Static helpers for packing/unpacking `.gmpack` adapter bundles."""

    @staticmethod
    def pack(
        adapter_dir: Path,
        domain: str,
        output_path: Optional[Path] = None,
        *,
        base_model: str = "unknown",
        validation_score: float = 0.0,
        record_count: int = 0,
        tags: Optional[list] = None,
        notes: str = "",
    ) -> Path:
        """Create a `.gmpack` bundle from *adapter_dir*.

        Args:
            adapter_dir: Directory containing the LoRA adapter files.
            domain: Domain tag (e.g. ``"code"``, ``"writing"``).
            output_path: Destination path.  Defaults to
                ``{adapter_dir.parent}/{domain}-{timestamp}.gmpack``.
            base_model: HuggingFace model id used as the LoRA base.
            validation_score: Held-out validation score at the time of export.
            record_count: Number of training records used in the last cycle.
            tags: Optional list of string labels for community search.
            notes: Free-text notes (stripped of PII before export).

        Returns:
            Absolute path to the created ``.gmpack`` file.
        """
        adapter_dir = Path(adapter_dir).resolve()
        if not adapter_dir.is_dir():
            raise ValueError(f"adapter_dir does not exist: {adapter_dir}")

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        if output_path is None:
            output_path = adapter_dir.parent / f"{domain}-{timestamp}{BUNDLE_EXTENSION}"
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Build sorted file list for deterministic SHA-256
        adapter_files = sorted(
            p for p in adapter_dir.rglob("*") if p.is_file()
        )

        sha256 = AdapterBundle._compute_tree_hash(adapter_files, adapter_dir)

        manifest = BundleManifest(
            schema_version=BUNDLE_SCHEMA_VERSION,
            domain=domain,
            base_model=base_model,
            validation_score=validation_score,
            record_count=record_count,
            created_at=datetime.now(timezone.utc).isoformat(),
            anonymized_id=str(uuid.uuid4()),
            sha256=sha256,
            tags=tags or [],
            notes=notes,
        )

        tmp_path = output_path.with_suffix(".gmpack.tmp")
        with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(_MANIFEST_NAME, json.dumps(manifest.to_dict(), indent=2))
            for src in adapter_files:
                arc_name = _ADAPTER_PREFIX + src.relative_to(adapter_dir).as_posix()
                zf.write(src, arc_name)

        tmp_path.replace(output_path)
        logger.info(
            "Packed adapter bundle: domain=%s sha256=%.8s → %s",
            domain, sha256, output_path,
        )
        return output_path

    @staticmethod
    def unpack(
        gmpack_path: Path,
        dest_dir: Path,
        *,
        verify: bool = True,
    ) -> BundleManifest:
        """Extract a `.gmpack` bundle into *dest_dir*.

        Args:
            gmpack_path: Path to the ``.gmpack`` file.
            dest_dir: Target directory (will be created if absent; any
                existing content is removed first so the unpack is idempotent).
            verify: If ``True`` (default), recompute the SHA-256 of extracted
                files and reject the bundle if it does not match the manifest.

        Returns:
            The ``BundleManifest`` parsed from the bundle.

        Raises:
            ValueError: If the schema version is unsupported or SHA-256 mismatch.
        """
        import shutil

        gmpack_path = Path(gmpack_path)
        dest_dir = Path(dest_dir)

        with zipfile.ZipFile(gmpack_path, "r") as zf:
            names = zf.namelist()
            if _MANIFEST_NAME not in names:
                raise ValueError(f"Invalid bundle — missing {_MANIFEST_NAME}")
            manifest = BundleManifest.from_dict(
                json.loads(zf.read(_MANIFEST_NAME))
            )

            if manifest.schema_version > BUNDLE_SCHEMA_VERSION:
                raise ValueError(
                    f"Bundle schema_version {manifest.schema_version} is newer than "
                    f"this GristMill build supports ({BUNDLE_SCHEMA_VERSION}). "
                    "Upgrade GristMill to import this bundle."
                )

            if dest_dir.exists():
                shutil.rmtree(dest_dir)
            dest_dir.mkdir(parents=True, exist_ok=True)

            for name in names:
                if name == _MANIFEST_NAME:
                    continue
                if not name.startswith(_ADAPTER_PREFIX):
                    continue
                rel = name[len(_ADAPTER_PREFIX):]
                out = dest_dir / rel
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(zf.read(name))

        if verify:
            extracted = sorted(p for p in dest_dir.rglob("*") if p.is_file())
            actual_sha = AdapterBundle._compute_tree_hash(extracted, dest_dir)
            if actual_sha != manifest.sha256:
                shutil.rmtree(dest_dir, ignore_errors=True)
                raise ValueError(
                    f"SHA-256 mismatch: expected {manifest.sha256}, got {actual_sha}. "
                    "Bundle may be corrupt or tampered."
                )

        logger.info(
            "Unpacked adapter bundle: domain=%s version_schema=%d → %s",
            manifest.domain, manifest.schema_version, dest_dir,
        )
        return manifest

    @staticmethod
    def read_manifest(gmpack_path: Path) -> BundleManifest:
        """Read *only* the manifest from a `.gmpack` without extracting files."""
        with zipfile.ZipFile(gmpack_path, "r") as zf:
            return BundleManifest.from_dict(
                json.loads(zf.read(_MANIFEST_NAME))
            )

    @staticmethod
    def _compute_tree_hash(files: list, base_dir: Path) -> str:
        """Deterministic SHA-256 over a sorted list of (rel_path, content) pairs."""
        h = hashlib.sha256()
        for f in files:
            rel = f.relative_to(base_dir).as_posix()
            h.update(rel.encode())
            h.update(f.read_bytes())
        return h.hexdigest()
