"""Model comparison framework for GristMill sieve classifiers.

Compares two ONNX sieve classifiers on a shared set of text examples and
produces a :class:`ComparisonReport` with accuracy, agreement rate, and
per-model latency percentiles.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Constants (must match sieve_trainer.py) ───────────────────────────────────

FEATURE_DIM: int = 392
EMBEDDING_DIM: int = 384


# ── Data classes ──────────────────────────────────────────────────────────────


@dataclass
class ModelInfo:
    """Metadata about an ONNX model under comparison."""

    path: Path
    label: str  # e.g. "sieve-v1", "sieve-v2"


@dataclass
class ComparisonReport:
    """Results of comparing two sieve classifier ONNX models."""

    candidate: ModelInfo
    baseline: ModelInfo
    n_samples: int
    agreement_rate: float
    candidate_accuracy: float
    baseline_accuracy: float
    candidate_latency_p50_ms: float
    candidate_latency_p99_ms: float
    baseline_latency_p50_ms: float
    baseline_latency_p99_ms: float
    label_distribution_candidate: dict[str, int]
    label_distribution_baseline: dict[str, int]
    timestamp: str

    def print_report(self) -> None:
        """Print a human-readable summary to stdout."""
        lines = [
            "",
            "=" * 60,
            "  GristMill Model Comparison Report",
            "=" * 60,
            f"  Candidate : {self.candidate.label} ({self.candidate.path})",
            f"  Baseline  : {self.baseline.label} ({self.baseline.path})",
            f"  Samples   : {self.n_samples}",
            f"  Timestamp : {self.timestamp}",
            "",
            "  Accuracy",
            f"    Candidate : {self.candidate_accuracy:.4f}",
            f"    Baseline  : {self.baseline_accuracy:.4f}",
            f"    Agreement : {self.agreement_rate:.4f}",
            "",
            "  Latency (ms)",
            f"    Candidate p50={self.candidate_latency_p50_ms:.2f}  "
            f"p99={self.candidate_latency_p99_ms:.2f}",
            f"    Baseline  p50={self.baseline_latency_p50_ms:.2f}  "
            f"p99={self.baseline_latency_p99_ms:.2f}",
            "",
            "  Label distribution — candidate",
        ]
        for k, v in sorted(self.label_distribution_candidate.items()):
            lines.append(f"    {k:<15} {v}")
        lines.append("  Label distribution — baseline")
        for k, v in sorted(self.label_distribution_baseline.items()):
            lines.append(f"    {k:<15} {v}")
        lines.append("=" * 60)
        print("\n".join(lines))

    def to_dict(self) -> dict:
        """Serialize to a plain dict (JSON-serialisable)."""
        return {
            "candidate": {"path": str(self.candidate.path), "label": self.candidate.label},
            "baseline": {"path": str(self.baseline.path), "label": self.baseline.label},
            "n_samples": self.n_samples,
            "agreement_rate": self.agreement_rate,
            "candidate_accuracy": self.candidate_accuracy,
            "baseline_accuracy": self.baseline_accuracy,
            "candidate_latency_p50_ms": self.candidate_latency_p50_ms,
            "candidate_latency_p99_ms": self.candidate_latency_p99_ms,
            "baseline_latency_p50_ms": self.baseline_latency_p50_ms,
            "baseline_latency_p99_ms": self.baseline_latency_p99_ms,
            "label_distribution_candidate": self.label_distribution_candidate,
            "label_distribution_baseline": self.label_distribution_baseline,
            "timestamp": self.timestamp,
        }


# ── Internal helpers ──────────────────────────────────────────────────────────

_LABEL_NAMES = ["LOCAL_ML", "RULES", "HYBRID", "LLM_NEEDED"]

from gristmill_ml.datasets.feedback import CHANNEL_ORDINAL


def _build_feature_vectors(
    texts: list[str],
    embedder_path: Optional[Path],
) -> np.ndarray:
    """Build 392-dim feature vectors matching ``sieve_trainer.FeatureExtractor``.

    Uses ``sentence_transformers`` to compute embeddings, then appends 8 scalar
    metadata features using conservative defaults (source=internal, priority=1).

    Args:
        texts: Raw input strings.
        embedder_path: Optional local path to a sentence-transformer model.
            Defaults to ``sentence-transformers/all-MiniLM-L6-v2``.

    Returns:
        Float32 array of shape ``[N, 392]``.
    """
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError(
            "sentence_transformers is required for model comparison. "
            "Install with: pip install sentence-transformers"
        ) from exc

    model_name_or_path = str(embedder_path) if embedder_path else "sentence-transformers/all-MiniLM-L6-v2"
    logger.info("Loading embedder: %s", model_name_or_path)
    embedder = SentenceTransformer(model_name_or_path)
    embeddings: np.ndarray = embedder.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=64,
    ).astype(np.float32)

    rows: list[np.ndarray] = []
    for i, text in enumerate(texts):
        tokens = text.lower().split()
        tc = len(tokens)

        # [384] log-scaled token count
        log_tc = math.log(tc + 1) / math.log(2049) if tc > 0 else 0.0
        log_tc = min(1.0, max(0.0, log_tc))

        # [385] source channel ordinal — default to "internal" (9/9 = 1.0)
        ch_ord = CHANNEL_ORDINAL.get("internal", 9) / 9.0

        # [386] priority — default Normal = 1
        pri = 1.0 / 3.0

        # [387] entity density — placeholder 0
        entity_density = 0.0

        # [388] question probability
        lowered = text.lower()
        q_starters = ("what", "why", "how", "who", "when", "where", "is", "are", "can")
        if text.endswith("?"):
            q_prob = 1.0
        elif any(lowered.startswith(s) for s in q_starters):
            q_prob = 0.6
        else:
            q_prob = 0.0

        # [389] code probability
        _code_indicators = frozenset(
            ["def ", "fn ", "function ", "class ", "import ", "return ", "if ", "{", "}", "=>"]
        )
        code_hits = sum(1 for ind in _code_indicators if ind in text)
        code_prob = min(1.0, code_hits / max(tc, 1))

        # [390] type-token ratio
        ttr = len(set(tokens)) / max(len(tokens), 1)

        # [391] ambiguity score
        if tokens:
            from collections import Counter
            max_freq = Counter(tokens).most_common(1)[0][1]
            ambiguity = 1.0 - max_freq / len(tokens)
        else:
            ambiguity = 0.0

        metadata = np.array(
            [log_tc, ch_ord, pri, entity_density, q_prob, code_prob, ttr, ambiguity],
            dtype=np.float32,
        )
        rows.append(np.concatenate([embeddings[i], metadata]))

    return np.stack(rows, axis=0)


def _load_onnx_session(model_path: Path):
    """Load an ONNX Runtime inference session from *model_path*."""
    try:
        import onnxruntime as ort  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError(
            "onnxruntime is required for model comparison. "
            "Install with: pip install onnxruntime"
        ) from exc

    opts = ort.SessionOptions()
    opts.log_severity_level = 3  # suppress warnings
    return ort.InferenceSession(
        str(model_path),
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )


def _run_with_latency(
    session,
    features: np.ndarray,
    batch_size: int = 64,
) -> tuple[np.ndarray, list[float]]:
    """Run inference in batches, recording per-sample latency in ms.

    Returns:
        Tuple of (predictions array [N], latency_ms list [N]).
    """
    n = features.shape[0]
    all_preds: list[int] = []
    latencies_ms: list[float] = []

    for start in range(0, n, batch_size):
        batch = features[start : start + batch_size]
        t0 = time.perf_counter()
        logits = session.run(None, {"features": batch})[0]
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        per_sample_ms = elapsed_ms / len(batch)
        preds = logits.argmax(axis=1).tolist()
        all_preds.extend(preds)
        latencies_ms.extend([per_sample_ms] * len(batch))

    return np.array(all_preds, dtype=np.int64), latencies_ms


def _label_distribution(preds: np.ndarray) -> dict[str, int]:
    dist = {name: 0 for name in _LABEL_NAMES}
    for p in preds:
        name = _LABEL_NAMES[int(p)] if 0 <= int(p) < len(_LABEL_NAMES) else "UNKNOWN"
        dist[name] = dist.get(name, 0) + 1
    return dist


# ── Public API ────────────────────────────────────────────────────────────────


def compare_models(
    candidate: ModelInfo,
    baseline: ModelInfo,
    texts: list[str],
    labels: Optional[list[int]] = None,
    embedder_path: Optional[Path] = None,
) -> ComparisonReport:
    """Compare two ONNX sieve classifiers on the same input texts.

    Both models receive identical 392-dim feature vectors built from *texts*.
    Accuracy is computed only when ground-truth *labels* are provided.

    Args:
        candidate: :class:`ModelInfo` for the new model under evaluation.
        baseline: :class:`ModelInfo` for the reference model.
        texts: List of raw input strings to run inference on.
        labels: Optional integer ground-truth labels (0-3). If ``None``,
            accuracy fields are reported as ``-1.0``.
        embedder_path: Optional path to a local sentence-transformer model
            (defaults to HuggingFace ``all-MiniLM-L6-v2``).

    Returns:
        :class:`ComparisonReport` with full comparison metrics.

    Raises:
        RuntimeError: If ``onnxruntime`` or ``sentence_transformers`` are not installed.
        FileNotFoundError: If either model path does not exist.
    """
    if not candidate.path.exists():
        raise FileNotFoundError(f"Candidate model not found: {candidate.path}")
    if not baseline.path.exists():
        raise FileNotFoundError(f"Baseline model not found: {baseline.path}")
    if not texts:
        raise ValueError("texts must not be empty")

    logger.info(
        "Building feature vectors for %d samples (FEATURE_DIM=%d)", len(texts), FEATURE_DIM
    )
    features = _build_feature_vectors(texts, embedder_path)

    logger.info("Loading candidate model: %s", candidate.path)
    cand_session = _load_onnx_session(candidate.path)

    logger.info("Loading baseline model: %s", baseline.path)
    base_session = _load_onnx_session(baseline.path)

    logger.info("Running candidate inference …")
    cand_preds, cand_latencies = _run_with_latency(cand_session, features)

    logger.info("Running baseline inference …")
    base_preds, base_latencies = _run_with_latency(base_session, features)

    # Agreement
    agreement_rate = float((cand_preds == base_preds).mean())

    # Accuracy
    if labels is not None:
        gt = np.array(labels, dtype=np.int64)
        cand_accuracy = float((cand_preds == gt).mean())
        base_accuracy = float((base_preds == gt).mean())
    else:
        cand_accuracy = -1.0
        base_accuracy = -1.0

    # Latency percentiles
    cand_lat_arr = np.array(cand_latencies)
    base_lat_arr = np.array(base_latencies)

    report = ComparisonReport(
        candidate=candidate,
        baseline=baseline,
        n_samples=len(texts),
        agreement_rate=agreement_rate,
        candidate_accuracy=cand_accuracy,
        baseline_accuracy=base_accuracy,
        candidate_latency_p50_ms=float(np.percentile(cand_lat_arr, 50)),
        candidate_latency_p99_ms=float(np.percentile(cand_lat_arr, 99)),
        baseline_latency_p50_ms=float(np.percentile(base_lat_arr, 50)),
        baseline_latency_p99_ms=float(np.percentile(base_lat_arr, 99)),
        label_distribution_candidate=_label_distribution(cand_preds),
        label_distribution_baseline=_label_distribution(base_preds),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    logger.info(
        "Comparison complete: agreement=%.4f cand_acc=%.4f base_acc=%.4f",
        report.agreement_rate,
        report.candidate_accuracy,
        report.baseline_accuracy,
    )
    return report
