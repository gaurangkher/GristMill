"""GristMill federated learning scaffolding — privacy-preserving gradient sharing.

Users who opt-in contribute *aggregated, noised gradient deltas* — never raw
query text, training examples, or adapter weights — to improve shared base
adapters for the broader community.

Privacy guarantees:
    - Gradient clipping (L2 norm ≤ ``clip_norm``) limits per-user influence.
    - Gaussian noise (``noise_multiplier`` × clip_norm) provides ε-DP per cycle.
    - A local ``PrivacyAccountant`` tracks cumulative (ε, δ) budget.
    - Contributions are rejected when the budget is exhausted.
"""

from gristmill_ml.federated.aggregator import FedAvgAggregator, PrivacyAccountant
from gristmill_ml.federated.contributor import GradientContributor

__all__ = ["FedAvgAggregator", "PrivacyAccountant", "GradientContributor"]
