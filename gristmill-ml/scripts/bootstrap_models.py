"""Bootstrap GristMill starter models.

Produces two ONNX files needed for a working daemon on a fresh install:

    ~/.gristmill/models/minilm-l6-v2.onnx
        MiniLM-L6-v2 sentence embedder.  Required by the Ledger warm tier
        for vector search and by the Sieve feature extractor.

    ~/.gristmill/models/intent-classifier-v1.onnx
        4-class intent routing classifier (LOCAL_ML / RULES / HYBRID / LLM_NEEDED).
        Trained on synthetic cold-start data — the closed-loop retraining
        cycle (gristmill train sieve) will improve it once real feedback
        accumulates in ~/.gristmill/feedback/.

Usage
-----
    cd GristMill
    pip install -e gristmill-ml
    python gristmill-ml/scripts/bootstrap_models.py

    # Faster dev run (fp32, no quantization):
    python gristmill-ml/scripts/bootstrap_models.py --no-quantize

    # Skip the classifier (embedder only, fastest):
    python gristmill-ml/scripts/bootstrap_models.py --embedder-only

    # Custom output directory:
    python gristmill-ml/scripts/bootstrap_models.py --output-dir /path/to/models

Exit codes
----------
    0  — both models exported and validated successfully
    1  — one or more steps failed (see stderr)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("bootstrap")


# ── Step helpers ──────────────────────────────────────────────────────────────


def step(name: str) -> None:
    logger.info("━━━  %s  ━━━", name)


def ok(msg: str) -> None:
    logger.info("✓  %s", msg)


def fail(msg: str) -> None:
    logger.error("✗  %s", msg)


# ── 1. Embedder ───────────────────────────────────────────────────────────────


def export_embedder(output_dir: Path, quantize: bool) -> Path | None:
    step("Exporting MiniLM-L6-v2 embedder")
    dest = output_dir / "minilm-l6-v2.onnx"

    if dest.exists():
        ok(f"Already present: {dest}  (skipping — delete to re-export)")
        return dest

    try:
        from gristmill_ml.export.onnx_export import OnnxExporter

        t0 = time.perf_counter()
        path = OnnxExporter.export_embedder(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            output_path=dest,
            quantize=quantize,
        )
        elapsed = time.perf_counter() - t0
        ok(f"Embedder exported ({elapsed:.1f}s): {path}")
        return path
    except Exception as exc:
        fail(f"Embedder export failed: {exc}")
        logger.exception("Traceback:")
        return None


# ── 2. Intent classifier ──────────────────────────────────────────────────────


def export_classifier(output_dir: Path, quantize: bool, epochs: int) -> Path | None:
    step("Training and exporting intent classifier")
    dest = output_dir / "intent-classifier-v1.onnx"

    if dest.exists():
        ok(f"Already present: {dest}  (skipping — delete to re-export)")
        return dest

    try:
        from gristmill_ml.training.sieve_trainer import SieveTrainer

        trainer = SieveTrainer(
            # Cold start: no real feedback yet — FeedbackDataset falls back to
            # synthetic data automatically (see datasets/feedback.py).
            feedback_dir=Path("~/.gristmill/feedback").expanduser(),
            output_dir=output_dir,
        )

        t0 = time.perf_counter()
        result = trainer.train(epochs=epochs)
        train_elapsed = time.perf_counter() - t0

        ok(
            f"Training complete ({train_elapsed:.1f}s) — "
            f"best val accuracy: {result.best_val_accuracy:.3f} "
            f"(epoch {result.best_epoch + 1})"
        )

        path = trainer.export(output_path=dest, quantize=quantize)
        ok(f"Classifier exported: {path}")
        return path

    except Exception as exc:
        fail(f"Classifier export failed: {exc}")
        logger.exception("Traceback:")
        return None


# ── 3. Manifest ───────────────────────────────────────────────────────────────


def write_manifest(output_dir: Path, results: dict) -> None:
    manifest_path = output_dir / "bootstrap_manifest.json"
    manifest = {
        "bootstrap_version": 1,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "models": results,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    ok(f"Manifest written: {manifest_path}")


# ── Main ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Bootstrap GristMill starter ONNX models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("~/.gristmill/models").expanduser(),
        help="Directory to write model files (default: ~/.gristmill/models)",
    )
    p.add_argument(
        "--no-quantize",
        action="store_true",
        help="Export fp32 models (larger but avoids quantization dependencies)",
    )
    p.add_argument(
        "--embedder-only",
        action="store_true",
        help="Export only the MiniLM embedder (skip classifier training)",
    )
    p.add_argument(
        "--classifier-epochs",
        type=int,
        default=10,
        help="Training epochs for the intent classifier (default: 10)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    output_dir: Path = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    quantize = not args.no_quantize

    logger.info("Output directory : %s", output_dir)
    logger.info("Quantization     : %s", "INT8" if quantize else "fp32")
    logger.info("")

    results: dict[str, str | None] = {}
    failed = False

    # ── Embedder ──────────────────────────────────────────────────────────────
    embedder_path = export_embedder(output_dir, quantize)
    results["minilm-l6-v2"] = str(embedder_path) if embedder_path else None
    if embedder_path is None:
        failed = True

    # ── Classifier ────────────────────────────────────────────────────────────
    if not args.embedder_only:
        classifier_path = export_classifier(output_dir, quantize, args.classifier_epochs)
        results["intent-classifier-v1"] = str(classifier_path) if classifier_path else None
        if classifier_path is None:
            failed = True
    else:
        results["intent-classifier-v1"] = None
        logger.info("Skipping classifier (--embedder-only)")

    # ── Manifest ──────────────────────────────────────────────────────────────
    write_manifest(output_dir, results)

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("")
    step("Summary")
    for name, path in results.items():
        if path:
            ok(f"{name}: {path}")
        else:
            fail(f"{name}: FAILED")

    if failed:
        logger.error("")
        logger.error("Bootstrap incomplete. Fix the errors above and re-run.")
        logger.error(
            "Tip: the daemon starts without models (heuristic-only mode), "
            "but LLM escalation rate will be high until models are loaded."
        )
        return 1

    logger.info("")
    logger.info("Bootstrap complete.")
    logger.info("Next step: start the daemon and verify with the smoke test.")
    logger.info("  cargo run --bin gristmill-daemon")
    logger.info("  pnpm --filter gristmill-integrations smoke-test")
    return 0


if __name__ == "__main__":
    sys.exit(main())
