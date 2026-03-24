"""Federated learning aggregator — FedAvg + differential-privacy utilities.

``FedAvgAggregator`` implements the canonical Federated Averaging algorithm
(McMahan et al., 2017).  It is used server-side (community backend) to merge
gradient deltas from multiple opt-in contributors into a single update that
improves the shared base adapter.

``DifferentialPrivacyClip`` is used *client-side* (inside ``GradientContributor``)
before any data leaves the device.  It clips and noises the gradient delta so
that the contribution satisfies (ε, δ)-differential privacy.

``PrivacyAccountant`` tracks the cumulative privacy budget for a single device.
Contributions are blocked once the budget is exhausted.

All state dictionaries (``StateDict``) are plain ``dict[str, list]`` — using
plain Python lists instead of tensors so this module has no hard PyTorch
dependency.  The ``GradientContributor`` converts tensors to lists before
calling these helpers.
"""

from __future__ import annotations

import json
import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

StateDict = dict[str, list]   # parameter_name → flat list of floats

# ── Differential privacy constants ────────────────────────────────────────────

_DEFAULT_CLIP_NORM = 1.0        # L2 clip norm (gradient clipping)
_DEFAULT_NOISE_MULT = 1.1       # noise multiplier σ (Gaussian mechanism)
_DEFAULT_DELTA = 1e-5           # δ for (ε, δ)-DP
_DEFAULT_EPSILON_BUDGET = 10.0  # total ε budget per device per model


# ── FedAvg ────────────────────────────────────────────────────────────────────


class FedAvgAggregator:
    """Federated Averaging over a list of gradient-delta state dicts.

    ``aggregate()`` computes a weighted mean of all deltas.  Each delta maps
    parameter name → flat list of floats representing the change in that
    parameter since the last global checkpoint.

    Args:
        weights: Optional per-contributor weights (e.g. proportional to local
            dataset size).  If ``None``, uniform weighting is used.
    """

    @staticmethod
    def aggregate(
        deltas: list[StateDict],
        weights: Optional[list[float]] = None,
    ) -> StateDict:
        """Compute the weighted average of *deltas*.

        Args:
            deltas: List of gradient deltas from N contributors.  All dicts
                must have identical keys and value lengths.
            weights: Optional per-delta weights.  Will be normalised to sum 1.

        Returns:
            A single ``StateDict`` representing the aggregated update.

        Raises:
            ValueError: If *deltas* is empty or shapes are inconsistent.
        """
        if not deltas:
            raise ValueError("Cannot aggregate an empty list of deltas")

        if weights is None:
            weights = [1.0] * len(deltas)
        if len(weights) != len(deltas):
            raise ValueError("len(weights) must equal len(deltas)")

        total = sum(weights)
        if total == 0:
            raise ValueError("Sum of weights must be > 0")
        norm_weights = [w / total for w in weights]

        keys = list(deltas[0].keys())
        # Validate all deltas share the same structure
        for i, d in enumerate(deltas[1:], 1):
            if set(d.keys()) != set(keys):
                raise ValueError(f"delta[{i}] has different parameter keys")

        aggregated: StateDict = {}
        for key in keys:
            ref_len = len(deltas[0][key])
            agg = [0.0] * ref_len
            for delta, w in zip(deltas, norm_weights):
                vals = delta[key]
                if len(vals) != ref_len:
                    raise ValueError(
                        f"Parameter '{key}' length mismatch: expected {ref_len}, got {len(vals)}"
                    )
                agg = [a + v * w for a, v in zip(agg, vals)]
            aggregated[key] = agg

        logger.debug(
            "FedAvg aggregated %d deltas, %d parameters", len(deltas), len(keys)
        )
        return aggregated

    @staticmethod
    def apply_delta(base: StateDict, delta: StateDict, lr: float = 1.0) -> StateDict:
        """Apply an aggregated *delta* to a *base* state dict.

        Args:
            base: Current global model parameters.
            delta: Aggregated gradient delta from ``aggregate()``.
            lr: Learning rate scale applied to the delta.

        Returns:
            New state dict with ``base[k] + lr * delta[k]`` for each key.
        """
        result: StateDict = {}
        for key, base_vals in base.items():
            d_vals = delta.get(key, [0.0] * len(base_vals))
            result[key] = [b + lr * d for b, d in zip(base_vals, d_vals)]
        return result


# ── Differential Privacy ──────────────────────────────────────────────────────


class DifferentialPrivacyClip:
    """Client-side DP mechanism: L2 clip + Gaussian noise.

    Applied *before* the gradient delta leaves the device.

    Args:
        clip_norm: Maximum L2 norm of the gradient vector.
        noise_multiplier: σ = noise_multiplier × clip_norm.
    """

    def __init__(
        self,
        clip_norm: float = _DEFAULT_CLIP_NORM,
        noise_multiplier: float = _DEFAULT_NOISE_MULT,
    ) -> None:
        self.clip_norm = clip_norm
        self.noise_multiplier = noise_multiplier

    def privatise(self, delta: StateDict) -> StateDict:
        """Clip and noise a gradient delta in-place (returns new dict).

        Steps:
            1. Compute global L2 norm across all parameters.
            2. Clip to ``clip_norm`` (scale-down if norm > clip).
            3. Add independent Gaussian noise N(0, σ²) to each element,
               where σ = noise_multiplier × clip_norm.

        Returns:
            A new ``StateDict`` with clipped + noised values.
        """
        # 1. Compute global L2 norm
        sq_sum = sum(v ** 2 for vals in delta.values() for v in vals)
        global_norm = math.sqrt(sq_sum) if sq_sum > 0 else 0.0

        # 2. Clip
        scale = min(1.0, self.clip_norm / (global_norm + 1e-8))
        clipped: StateDict = {k: [v * scale for v in vals] for k, vals in delta.items()}

        # 3. Add Gaussian noise  σ = noise_multiplier × clip_norm
        sigma = self.noise_multiplier * self.clip_norm
        noised: StateDict = {
            k: [v + random.gauss(0.0, sigma) for v in vals]
            for k, vals in clipped.items()
        }

        logger.debug(
            "DP: global_norm=%.4f clip_scale=%.4f sigma=%.4f",
            global_norm, scale, sigma,
        )
        return noised

    def per_cycle_epsilon(self, n_records: int, batch_size: int = 32) -> float:
        """Estimate per-cycle ε using the moments accountant approximation.

        This is a simplified analytical bound suitable for budget tracking;
        a production deployment should use the ``dp-accounting`` library.

        Args:
            n_records: Size of the local training dataset.
            batch_size: Batch size used during training.

        Returns:
            Estimated ε for one training cycle.
        """
        if n_records <= 0 or batch_size <= 0:
            return float("inf")
        q = batch_size / n_records             # sampling rate
        sigma = self.noise_multiplier
        # Simplified Gaussian mechanism ε ≈ q * sqrt(2 * ln(1.25/δ)) / σ
        # with δ = 1e-5
        delta = _DEFAULT_DELTA
        eps = q * math.sqrt(2 * math.log(1.25 / delta)) / sigma
        return eps


# ── Privacy Accountant ─────────────────────────────────────────────────────────


@dataclass
class PrivacyBudget:
    epsilon_used: float = 0.0
    epsilon_budget: float = _DEFAULT_EPSILON_BUDGET
    cycles_contributed: int = 0
    last_updated: str = ""

    @property
    def remaining(self) -> float:
        return max(0.0, self.epsilon_budget - self.epsilon_used)

    @property
    def exhausted(self) -> bool:
        return self.epsilon_used >= self.epsilon_budget


class PrivacyAccountant:
    """Tracks cumulative (ε, δ)-DP budget for a single device.

    The accountant is persisted to disk so budget carries across trainer
    restarts.  When the budget is exhausted, ``can_contribute()`` returns
    ``False`` and the ``GradientContributor`` skips the upload.

    Args:
        state_path: Path to the JSON state file.  Defaults to
            ``~/.gristmill/privacy_budget.json``.
        epsilon_budget: Total ε budget across all contributions.
    """

    def __init__(
        self,
        state_path: Optional[Path] = None,
        epsilon_budget: float = _DEFAULT_EPSILON_BUDGET,
    ) -> None:
        self._path = state_path or (
            Path.home() / ".gristmill" / "privacy_budget.json"
        )
        self._budget = self._load(epsilon_budget)

    def _load(self, default_budget: float) -> PrivacyBudget:
        if self._path.exists():
            try:
                d = json.loads(self._path.read_text())
                return PrivacyBudget(**d)
            except Exception as exc:
                logger.warning("Could not load privacy budget state: %s", exc)
        return PrivacyBudget(epsilon_budget=default_budget)

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        from dataclasses import asdict
        from datetime import datetime, timezone
        self._budget.last_updated = datetime.now(timezone.utc).isoformat()
        tmp = self._path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(asdict(self._budget), indent=2))
        tmp.replace(self._path)

    def can_contribute(self) -> bool:
        """Return ``True`` if the budget has not been exhausted."""
        return not self._budget.exhausted

    def record_contribution(self, epsilon_spent: float) -> None:
        """Accumulate *epsilon_spent* and persist state."""
        self._budget.epsilon_used += epsilon_spent
        self._budget.cycles_contributed += 1
        self._save()
        logger.info(
            "Privacy budget: used=%.3f / %.3f ε (cycle %d)",
            self._budget.epsilon_used,
            self._budget.epsilon_budget,
            self._budget.cycles_contributed,
        )

    @property
    def budget(self) -> PrivacyBudget:
        return self._budget
