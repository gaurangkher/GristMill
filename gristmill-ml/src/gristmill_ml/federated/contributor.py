"""GradientContributor — client-side opt-in federated learning contribution.

After each successful distillation cycle, the contributor:

    1. Checks that the user has opted in (``federated.enabled: true`` in config).
    2. Checks the local ``PrivacyAccountant`` to ensure budget is available.
    3. Computes the gradient *delta* = adapter_after − adapter_before.
    4. Applies DP clipping + noise via ``DifferentialPrivacyClip``.
    5. POSTs the noised delta (JSON) to the community federation endpoint.
    6. Records the ε spent against the local budget.

Raw adapter weights, training examples, and query text are **never** sent.
Only the noised gradient delta (a dict of parameter names → float lists) is
transmitted, together with the domain tag and an anonymous contributor UUID.

Configuration (``~/.gristmill/config.yaml``)::

    federated:
      enabled: false         # must be explicitly set to true to opt-in
      clip_norm: 1.0         # L2 gradient clip norm
      noise_multiplier: 1.1  # Gaussian noise σ = noise_mult × clip_norm
      epsilon_budget: 10.0   # total ε budget across all cycles
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_DEFAULT_ENDPOINT = "https://adapters.gristmill.dev"
_CONTRIBUTE_TIMEOUT_S = 60


class GradientContributor:
    """Opt-in federated gradient contributor.

    Args:
        enabled: Master opt-in switch.
        endpoint: Community federation API base URL.
        token: Optional bearer token.
        clip_norm: DP gradient clip norm.
        noise_multiplier: DP noise multiplier.
        epsilon_budget: Total ε budget for this device.
        contributor_id: Anonymous stable UUID for this installation.
            Stored in ``~/.gristmill/contributor_id`` and reused across
            sessions so the server can deduplicate contributions.
    """

    def __init__(
        self,
        enabled: bool = False,
        endpoint: Optional[str] = None,
        token: Optional[str] = None,
        clip_norm: float = 1.0,
        noise_multiplier: float = 1.1,
        epsilon_budget: float = 10.0,
        contributor_id: Optional[str] = None,
    ) -> None:
        self._enabled = enabled
        self._endpoint = (
            os.environ.get("GRISTMILL_COMMUNITY_URL") or endpoint or _DEFAULT_ENDPOINT
        ).rstrip("/")
        self._token = os.environ.get("GRISTMILL_COMMUNITY_TOKEN") or token

        from gristmill_ml.federated.aggregator import DifferentialPrivacyClip, PrivacyAccountant
        self._dp = DifferentialPrivacyClip(clip_norm=clip_norm, noise_multiplier=noise_multiplier)
        self._accountant = PrivacyAccountant(epsilon_budget=epsilon_budget)
        self._contributor_id = contributor_id or self._load_or_create_id()

    # ── Contribute ────────────────────────────────────────────────────────────

    def contribute(
        self,
        adapter_before: Path,
        adapter_after: Path,
        domain: str,
        n_records: int,
        batch_size: int = 32,
        *,
        dry_run: bool = False,
    ) -> bool:
        """Compute and submit a privacy-preserving gradient contribution.

        Args:
            adapter_before: Path to the adapter directory *before* training.
            adapter_after: Path to the adapter directory *after* training.
            domain: Domain tag (e.g. ``"code"``).
            n_records: Number of training records used (for ε estimation).
            batch_size: Batch size used (for ε estimation).
            dry_run: If ``True``, compute and log the delta but skip upload.

        Returns:
            ``True`` if the contribution was successfully submitted (or would
            have been, in dry-run mode).  ``False`` otherwise.
        """
        if not self._enabled:
            logger.debug("Federated contribution skipped — opt-in not enabled")
            return False

        if not self._accountant.can_contribute():
            logger.warning(
                "Privacy budget exhausted (ε=%.2f used of %.2f) — skipping contribution",
                self._accountant.budget.epsilon_used,
                self._accountant.budget.epsilon_budget,
            )
            return False

        try:
            delta = self._compute_delta(adapter_before, adapter_after)
        except Exception as exc:
            logger.error("Failed to compute gradient delta: %s", exc)
            return False

        noised_delta = self._dp.privatise(delta)
        eps_cycle = self._dp.per_cycle_epsilon(n_records, batch_size)

        if dry_run:
            n_params = sum(len(v) for v in noised_delta.values())
            logger.info(
                "[dry-run] Would contribute: domain=%s params=%d eps_cycle=%.4f",
                domain, n_params, eps_cycle,
            )
            self._accountant.record_contribution(eps_cycle)
            return True

        success = self._submit(noised_delta, domain)
        if success:
            self._accountant.record_contribution(eps_cycle)
        return success

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _compute_delta(self, before_dir: Path, after_dir: Path) -> dict:
        """Load two adapter directories and compute parameter-wise delta.

        Reads ``adapter_model.bin`` (PyTorch) or ``adapter_model.safetensors``
        from each directory.  Falls back to comparing any ``.bin`` files.

        Returns:
            ``StateDict`` mapping parameter name → flat list of floats.
        """
        import torch

        def _load(d: Path) -> dict:
            for name in ("adapter_model.bin", "adapter_model.safetensors"):
                p = d / name
                if p.exists():
                    state = torch.load(p, map_location="cpu", weights_only=True)
                    return {k: v.float().flatten().tolist() for k, v in state.items()}
            # Fallback: first .bin file
            bins = list(d.glob("*.bin"))
            if bins:
                state = torch.load(bins[0], map_location="cpu", weights_only=True)
                return {k: v.float().flatten().tolist() for k, v in state.items()}
            raise FileNotFoundError(f"No adapter weights found in {d}")

        before = _load(Path(before_dir))
        after = _load(Path(after_dir))

        # Compute delta for parameters present in both checkpoints
        delta: dict = {}
        for key in after:
            if key in before:
                a_vals = after[key]
                b_vals = before[key]
                if len(a_vals) == len(b_vals):
                    delta[key] = [a - b for a, b in zip(a_vals, b_vals)]
        return delta

    def _submit(self, noised_delta: dict, domain: str) -> bool:
        """POST the noised delta to the federation endpoint."""
        try:
            import urllib.request

            payload = json.dumps({
                "domain": domain,
                "contributor_id": self._contributor_id,
                "delta": noised_delta,
            }).encode()

            url = f"{self._endpoint}/v1/federation/contribute"
            headers: dict = {
                "Content-Type": "application/json",
                "X-GristMill-Domain": domain,
                "X-GristMill-Contributor-ID": self._contributor_id,
            }
            if self._token:
                headers["Authorization"] = f"Bearer {self._token}"

            req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=_CONTRIBUTE_TIMEOUT_S) as resp:
                resp.read()  # discard body

            logger.info("Federated gradient contribution submitted: domain=%s", domain)
            return True
        except Exception as exc:
            logger.error("Federated contribution failed: %s", exc)
            return False

    @staticmethod
    def _load_or_create_id() -> str:
        """Return a stable anonymous contributor UUID, creating one if needed."""
        id_path = Path.home() / ".gristmill" / "contributor_id"
        if id_path.exists():
            return id_path.read_text().strip()
        new_id = str(uuid.uuid4())
        id_path.parent.mkdir(parents=True, exist_ok=True)
        id_path.write_text(new_id)
        logger.debug("Created new contributor_id: %s", new_id)
        return new_id

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, config_path: Optional[Path] = None) -> "GradientContributor":
        """Build a ``GradientContributor`` from ``~/.gristmill/config.yaml``."""
        import yaml  # type: ignore[import]

        if config_path is None:
            config_path = Path.home() / ".gristmill" / "config.yaml"

        cfg: dict = {}
        if config_path.exists():
            try:
                cfg = yaml.safe_load(config_path.read_text()) or {}
            except Exception as exc:
                logger.warning("Could not read config: %s", exc)

        fed = cfg.get("federated", {})
        com = cfg.get("community", {})

        return cls(
            enabled=bool(fed.get("enabled", False)),
            endpoint=com.get("endpoint", None),
            token=com.get("token", None),
            clip_norm=float(fed.get("clip_norm", 1.0)),
            noise_multiplier=float(fed.get("noise_multiplier", 1.1)),
            epsilon_budget=float(fed.get("epsilon_budget", 10.0)),
        )
