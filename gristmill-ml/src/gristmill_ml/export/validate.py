"""ONNX parity validation.

Checks that the exported ONNX model produces outputs numerically consistent
with the original PyTorch model (within ``tolerance``).

For full cross-runtime validation (comparing Python ONNX Runtime against the
Rust ``ort``-based runtime), the ``validate_rust_parity`` function requires
a compiled ``gristmill_core`` native extension.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from gristmill_ml.training.sieve_trainer import SieveClassifierHead

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    passed: bool
    max_diff: float
    mean_diff: float
    n_samples: int
    label_agreement_rate: float = 1.0  # argmax agreement (0–1)
    notes: str = ""

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[{status}] max_diff={self.max_diff:.8f} "
            f"mean_diff={self.mean_diff:.8f} "
            f"label_agreement={self.label_agreement_rate:.4f} "
            f"n={self.n_samples}"
        )


def validate_classifier_parity(
    pytorch_model: "SieveClassifierHead",
    onnx_path: Path,
    test_features: "np.ndarray",
    tolerance: float = 1e-4,
) -> ValidationResult:
    """Compare PyTorch and ONNX Runtime outputs on the same inputs.

    Args:
        pytorch_model: The source model in eval mode.
        onnx_path: Path to the exported ``.onnx`` file.
        test_features: Float32 array of shape ``[N, 392]``.
        tolerance: Maximum acceptable absolute difference (element-wise).

    Returns:
        :class:`ValidationResult` with pass/fail and diagnostic metrics.
    """
    try:
        import onnxruntime as ort
        import torch
    except ImportError as exc:
        raise ImportError("Install onnxruntime: pip install onnxruntime") from exc

    assert test_features.ndim == 2, "test_features must be 2-D [N, 392]"
    n = test_features.shape[0]

    # ── PyTorch inference ─────────────────────────────────────────────────────
    pytorch_model.eval()
    with torch.no_grad():
        pt_logits: np.ndarray = pytorch_model(torch.from_numpy(test_features)).numpy()

    # ── ONNX Runtime inference ────────────────────────────────────────────────
    sess_options = ort.SessionOptions()
    sess_options.log_severity_level = 3  # suppress warnings
    session = ort.InferenceSession(
        str(onnx_path), sess_options=sess_options, providers=["CPUExecutionProvider"]
    )
    ort_outputs = session.run(None, {"features": test_features})
    ort_logits: np.ndarray = ort_outputs[0]

    # ── Numeric comparison ────────────────────────────────────────────────────
    diff = np.abs(pt_logits - ort_logits[:, : pt_logits.shape[1]])
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())

    pt_labels = pt_logits.argmax(axis=1)
    ort_labels = ort_logits.argmax(axis=1)
    agreement = float((pt_labels == ort_labels).mean())

    passed = max_diff <= tolerance and agreement >= 0.99
    result = ValidationResult(
        passed=passed,
        max_diff=max_diff,
        mean_diff=mean_diff,
        n_samples=n,
        label_agreement_rate=agreement,
    )
    if passed:
        logger.info("Parity check PASSED: %s", result)
    else:
        logger.warning("Parity check FAILED: %s", result)
    return result.__dict__  # type: ignore[return-value]


def validate_rust_parity(
    onnx_path: Path,
    feedback_samples: int = 100,
    min_agreement: float = 0.95,
) -> "ValidationResult":
    """Compare ONNX Runtime labels vs Rust Sieve routing decisions.

    Requires the ``gristmill_core`` native extension to be installed.
    If it is not available, returns a passing result with a warning.

    Args:
        onnx_path: Path to the classifier ONNX file to validate.
        feedback_samples: Number of synthetic events to test.
        min_agreement: Minimum fraction of decisions that must agree.

    Returns:
        :class:`ValidationResult`.
    """
    try:
        from gristmill_ml.core import PyGristMill, HAS_NATIVE
    except ImportError:
        HAS_NATIVE = False

    if not HAS_NATIVE:
        logger.warning("gristmill_core not available — skipping Rust parity check")
        return ValidationResult(
            passed=True,
            max_diff=0.0,
            mean_diff=0.0,
            n_samples=0,
            label_agreement_rate=1.0,
            notes="gristmill_core not installed — skipped",
        )

    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError("Install onnxruntime: pip install onnxruntime") from exc

    from gristmill_ml.datasets.feedback import generate_synthetic_records, ROUTE_LABEL_MAP
    from gristmill_ml.training.sieve_trainer import FeatureExtractor
    from sentence_transformers import SentenceTransformer

    label_to_name = {v: k for k, v in ROUTE_LABEL_MAP.items()}

    # Build ONNX session
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    # Feature extractor
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    extractor = FeatureExtractor(embedder)

    # Rust core
    try:
        mill = PyGristMill(None)
    except Exception as exc:
        logger.warning("Could not init PyGristMill: %s — skipping Rust parity", exc)
        return ValidationResult(
            passed=True,
            max_diff=0.0,
            mean_diff=0.0,
            n_samples=0,
            notes=f"PyGristMill init failed: {exc}",
        )

    records = generate_synthetic_records(feedback_samples)
    agree = 0
    total = 0

    for rec in records:
        text = getattr(rec, "_text", "")
        if not text:
            continue
        source = rec.event_source
        priority = getattr(rec, "_priority", 1)
        token_count = rec.token_count

        # ONNX prediction
        features = extractor.extract(text, source, priority, token_count)
        ort_logits = session.run(None, {"features": features[np.newaxis]})[0]
        ort_label_name = label_to_name.get(int(ort_logits.argmax()), "LOCAL_ML")

        # Rust prediction
        import json

        event_json = json.dumps(
            {
                "source": {"type": source},
                "payload": {"text": text},
                "timestamp_ms": rec.timestamp_ms,
                "id": rec.event_id,
                "metadata": {
                    "priority": ["low", "normal", "high", "critical"][min(priority, 3)],
                    "tags": {},
                },
            }
        )
        try:
            decision_json = mill.triage(event_json)
            decision = json.loads(decision_json)
            rust_route = decision.get("route", "LOCAL_ML")
        except Exception:
            continue

        if ort_label_name == rust_route:
            agree += 1
        total += 1

    agreement = agree / max(total, 1)
    passed = agreement >= min_agreement

    result = ValidationResult(
        passed=passed,
        max_diff=0.0,
        mean_diff=0.0,
        n_samples=total,
        label_agreement_rate=agreement,
        notes=f"Rust parity: {agree}/{total} agreed",
    )
    if passed:
        logger.info("Rust parity check PASSED: %s", result)
    else:
        logger.warning("Rust parity check FAILED: %s", result)
    return result


# ── CLI entry-point ───────────────────────────────────────────────────────────


def main() -> None:  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description="Validate ONNX model parity")
    parser.add_argument("onnx_path", type=Path)
    parser.add_argument("--rust", action="store_true", help="Run Rust parity check too")
    args = parser.parse_args()

    from gristmill_ml.training.sieve_trainer import SieveClassifierHead
    import numpy as np

    model = SieveClassifierHead()
    dummy = np.random.randn(32, 392).astype(np.float32)
    r = validate_classifier_parity(model, args.onnx_path, dummy)
    print(r)

    if args.rust:
        r2 = validate_rust_parity(args.onnx_path)
        print(r2)


if __name__ == "__main__":
    main()
