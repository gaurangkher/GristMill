"""Shared config resolution for gristmill-ml.

Single source of truth for the config.yaml search order:
  1. GRISTMILL_CONFIG env var  (explicit override, highest priority)
  2. /data/gristmill/config.yaml  (Docker bind-mount)
  3. <repo-root>/gristmill-data/config.yaml  (local dev — repo-relative, default)
  4. ~/.gristmill/config.yaml  (legacy fallback)

All modules in gristmill-ml should import from here rather than hardcoding
their own candidate lists.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

# Repo root is four levels above this file:
#   src/gristmill_ml/config.py  →  parents[3] = repo root
_REPO_ROOT = Path(__file__).resolve().parents[3]
_REPO_DATA_CONFIG = _REPO_ROOT / "gristmill-data" / "config.yaml"


def config_candidates(extra: Optional[Path] = None) -> list[Path]:
    """Return the ordered list of config.yaml candidate paths.

    Args:
        extra: Optional explicit path prepended before the standard candidates
               (e.g. a path passed in by the caller).
    """
    candidates: list[Path] = []
    if extra is not None:
        candidates.append(extra)
    if env_cfg := os.environ.get("GRISTMILL_CONFIG"):
        candidates.append(Path(env_cfg))
    candidates += [
        Path("/data/gristmill/config.yaml"),  # Docker bind-mount
        _REPO_DATA_CONFIG,  # local dev (repo-relative)
        Path.home() / ".gristmill" / "config.yaml",  # legacy fallback
    ]
    return candidates


def load_config(extra: Optional[Path] = None) -> dict:
    """Load and return the first config.yaml found in the candidate list.

    Returns an empty dict if no config file exists or parsing fails.
    """
    try:
        import yaml  # type: ignore[import]
    except ImportError:
        return {}

    for p in config_candidates(extra):
        if p.exists():
            try:
                return yaml.safe_load(p.read_text()) or {}
            except Exception:
                pass
    return {}
