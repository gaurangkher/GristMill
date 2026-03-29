"""Custom model training pipeline for user-supplied labeled data.

Accepts a CSV or JSONL file with ``text`` and ``label`` columns (where label is
one of ``LOCAL_ML``, ``RULES``, ``HYBRID``, ``LLM_NEEDED``) and runs the full
SieveTrainer → ONNX export pipeline.

CLI usage::

    python -m gristmill_ml.pipelines.custom_model --data PATH [options]

See ``--help`` for all options.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_VALID_LABELS = {"LOCAL_ML", "RULES", "HYBRID", "LLM_NEEDED"}


# ── CustomModelPipeline ───────────────────────────────────────────────────────


class CustomModelPipeline:
    """Train a sieve classifier from user-supplied labeled data.

    Args:
        data_path: Path to a CSV or JSONL file with ``text`` and ``label`` columns.
        epochs: Training epochs.
        output_dir: Directory to write the exported ONNX model.
        model_name: Base name for the output ``.onnx`` file.
        quantize: Apply INT8 dynamic quantization.
        run_validate: Run ONNX parity validation after export.
    """

    def __init__(
        self,
        data_path: Path,
        epochs: int = 15,
        output_dir: Path = Path("~/.gristmill/models").expanduser(),
        model_name: str = "custom-classifier-v1",
        quantize: bool = True,
        run_validate: bool = False,
    ) -> None:
        self.data_path = Path(data_path)
        self.epochs = epochs
        self.output_dir = Path(output_dir).expanduser()
        self.model_name = model_name
        self.quantize = quantize
        self.run_validate = run_validate

    # ── Data loading ──────────────────────────────────────────────────────────

    def _load_csv(self) -> list[dict]:
        """Load rows from a CSV file using pandas (preferred) or csv module."""
        try:
            import pandas as pd  # type: ignore[import]

            df = pd.read_csv(self.data_path)
            self._validate_columns(set(df.columns))
            return df[["text", "label"]].to_dict(orient="records")
        except ImportError:
            logger.debug("pandas not available — falling back to csv module")

        rows: list[dict] = []
        with self.data_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"CSV file appears empty: {self.data_path}")
            self._validate_columns(set(reader.fieldnames))
            for row in reader:
                rows.append({"text": row["text"], "label": row["label"]})
        return rows

    def _load_jsonl(self) -> list[dict]:
        """Load rows from a JSONL file."""
        rows: list[dict] = []
        with self.data_path.open(encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON on line {lineno} of {self.data_path}: {exc}"
                    ) from exc
                rows.append({"text": obj["text"], "label": obj["label"]})
        if rows:
            self._validate_columns(set(rows[0].keys()))
        return rows

    @staticmethod
    def _validate_columns(columns: set) -> None:
        missing = {"text", "label"} - columns
        if missing:
            raise ValueError(
                f"Input data is missing required columns: {missing}. "
                "Expected 'text' and 'label'."
            )

    def load_data(self) -> list[dict]:
        """Auto-detect format (CSV / JSONL) and load labeled records.

        Returns:
            List of dicts with keys ``text`` and ``label``.

        Raises:
            ValueError: If the file format is unsupported, columns are missing,
                or invalid labels are found.
        """
        suffix = self.data_path.suffix.lower()
        if suffix == ".csv":
            rows = self._load_csv()
        elif suffix in (".jsonl", ".ndjson", ".json"):
            rows = self._load_jsonl()
        else:
            # Try JSONL first, then CSV
            try:
                rows = self._load_jsonl()
            except Exception:
                rows = self._load_csv()

        # Validate label values
        invalid = {r["label"] for r in rows if r["label"] not in _VALID_LABELS}
        if invalid:
            raise ValueError(
                f"Invalid label values: {invalid}. "
                f"Expected one of: {_VALID_LABELS}"
            )

        logger.info(
            "Loaded %d records from %s (labels: %s)",
            len(rows),
            self.data_path,
            {r["label"] for r in rows},
        )
        return rows

    # ── Conversion ────────────────────────────────────────────────────────────

    @staticmethod
    def _to_feedback_records(rows: list[dict]) -> list:
        """Convert raw rows to :class:`FeedbackRecord` objects for training."""
        from gristmill_ml.datasets.feedback import FeedbackRecord, ROUTE_LABEL_MAP

        records = []
        for i, row in enumerate(rows):
            raw = {
                "event_id": f"custom-{i:08d}",
                "timestamp_ms": 1_700_000_000_000 + i * 1000,
                "route_decision": row["label"],
                "confidence": 0.9,
                "estimated_tokens": 0,
                "actual_tokens": None,
                "could_have_been_local": None,
                "event_source": "http",
                "token_count": len(row["text"].split()),
                "_text": row["text"],
                "_priority": 1,
            }
            rec = FeedbackRecord(raw)
            rec._text = row["text"]  # type: ignore[attr-defined]
            rec._priority = 1  # type: ignore[attr-defined]
            records.append(rec)
        return records

    # ── Pipeline ──────────────────────────────────────────────────────────────

    def run(self) -> dict:
        """Execute the custom model training pipeline.

        Returns:
            Outcome dict with keys: ``success``, ``onnx_path``, ``n_records``,
            ``classification_report``, ``error``.
        """
        outcome: dict = {
            "success": False,
            "onnx_path": None,
            "n_records": 0,
            "classification_report": "",
            "error": None,
        }

        try:
            # ── Step 1: Load data ─────────────────────────────────────────────
            logger.info("Step 1/4 — Loading data from %s …", self.data_path)
            rows = self.load_data()
            outcome["n_records"] = len(rows)

            # ── Step 2: Convert to FeedbackRecord and inject into FeedbackDataset ──
            logger.info("Step 2/4 — Converting to FeedbackRecord format …")
            feedback_records = self._to_feedback_records(rows)

            # We need a FeedbackDataset-like object. Patch the records directly.
            from gristmill_ml.datasets.feedback import FeedbackDataset
            from gristmill_ml.training.sieve_trainer import SieveTrainer

            # Create a minimal FeedbackDataset wrapping our records
            dataset = FeedbackDataset.__new__(FeedbackDataset)
            dataset.split = "train"
            dataset.feedback_dir = self.data_path.parent
            dataset.records = feedback_records

            # ── Step 3: Train ─────────────────────────────────────────────────
            logger.info(
                "Step 3/4 — Training custom classifier (epochs=%d) …", self.epochs
            )
            self.output_dir.mkdir(parents=True, exist_ok=True)
            trainer = SieveTrainer(
                feedback_dir=self.data_path.parent,
                output_dir=self.output_dir,
                experiment_name=f"custom-{self.model_name}",
            )
            # Inject our dataset directly to bypass feedback file loading
            train_result = self._train_on_records(trainer, feedback_records)
            outcome["classification_report"] = train_result.classification_report
            logger.info(
                "Training complete: val_acc=%.4f", train_result.best_val_accuracy
            )

            # ── Step 4: Export ────────────────────────────────────────────────
            from gristmill_ml.export.onnx_export import OnnxExporter

            onnx_dest = self.output_dir / f"{self.model_name}.onnx"
            logger.info("Step 4/4 — Exporting ONNX → %s …", onnx_dest)
            onnx_path = OnnxExporter.export_classifier(
                trainer.model, onnx_dest, quantize=self.quantize
            )
            outcome["onnx_path"] = str(onnx_path)
            logger.info("Exported: %s", onnx_path)

            # ── Optional: Validate ────────────────────────────────────────────
            if self.run_validate:
                import numpy as np
                from gristmill_ml.export.validate import validate_classifier_parity
                from gristmill_ml.training.sieve_trainer import FEATURE_DIM

                logger.info("Validating ONNX parity …")
                dummy = np.random.randn(16, FEATURE_DIM).astype(np.float32)
                result = validate_classifier_parity(trainer.model, onnx_path, dummy)
                passed = result["passed"] if isinstance(result, dict) else result.passed
                if not passed:
                    max_diff = result["max_diff"] if isinstance(result, dict) else result.max_diff
                    logger.warning("Parity check failed (max_diff=%.6f)", max_diff)
                else:
                    logger.info("Parity check passed.")

            # ── Print classification report ────────────────────────────────
            print("\n" + "=" * 60)
            print(f"Custom classifier trained: {onnx_path}")
            print(f"Records: {outcome['n_records']}")
            print("\nClassification report:")
            print(outcome["classification_report"])
            print("=" * 60)

            outcome["success"] = True

        except Exception as exc:
            outcome["error"] = str(exc)
            logger.exception("Custom model pipeline failed: %s", exc)

        return outcome

    def _train_on_records(self, trainer, records: list):
        """Internal: train :class:`SieveTrainer` directly on pre-built records."""
        import random
        import numpy as np
        import torch
        import torch.nn as nn
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import CosineAnnealingLR
        from torch.utils.data import DataLoader
        from gristmill_ml.datasets.feedback import FeedbackDataset
        from gristmill_ml.training.sieve_trainer import TrainResult, FEATURE_DIM, NUM_CLASSES

        # Build synthetic dataset split
        random.seed(42)
        random.shuffle(records)
        n_val = max(1, int(len(records) * 0.15))
        val_records = records[:n_val]
        train_records = records[n_val:]

        def records_to_items(recs):
            return [
                {
                    "text": getattr(r, "_text", ""),
                    "source": r.event_source,
                    "priority": getattr(r, "_priority", 1),
                    "token_count": r.token_count,
                    "label": r.label,
                }
                for r in recs
            ]

        train_items = records_to_items(train_records)
        val_items = records_to_items(val_records)

        logger.info(
            "Custom training split — train: %d, val: %d", len(train_items), len(val_items)
        )

        # Feature extraction
        X_train = trainer.feature_extractor.extract_batch(train_items)
        y_train = np.array([it["label"] for it in train_items], dtype=np.int64)
        X_val = trainer.feature_extractor.extract_batch(val_items)
        y_val = np.array([it["label"] for it in val_items], dtype=np.int64)

        X_train_t = torch.from_numpy(X_train).float()
        y_train_t = torch.from_numpy(y_train).long()
        X_val_t = torch.from_numpy(X_val).float()
        y_val_t = torch.from_numpy(y_val).long()

        train_loader = DataLoader(
            list(zip(X_train_t, y_train_t)), batch_size=64, shuffle=True
        )
        val_loader = DataLoader(
            list(zip(X_val_t, y_val_t)), batch_size=64, shuffle=False
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = AdamW(trainer.model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.epochs)

        result = TrainResult()
        best_state = None
        no_improve = 0

        for epoch in range(self.epochs):
            trainer.model.train()
            epoch_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(trainer.device), yb.to(trainer.device)
                optimizer.zero_grad()
                loss = criterion(trainer.model(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(trainer.model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item() * len(yb)
            scheduler.step()

            train_loss = epoch_loss / max(len(train_records), 1)
            result.train_losses.append(train_loss)

            val_acc = trainer._evaluate_accuracy(val_loader)
            result.val_accuracies.append(val_acc)

            logger.info(
                "Epoch %2d/%d — loss=%.4f  val_acc=%.4f",
                epoch + 1,
                self.epochs,
                train_loss,
                val_acc,
            )

            if val_acc > result.best_val_accuracy:
                result.best_val_accuracy = val_acc
                result.best_epoch = epoch
                best_state = {k: v.cpu().clone() for k, v in trainer.model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= 3:
                    logger.info("Early stopping at epoch %d", epoch + 1)
                    break

        if best_state:
            trainer.model.load_state_dict(best_state)

        result.classification_report = trainer._classification_report(val_loader)
        return result


# ── CLI entry-point ───────────────────────────────────────────────────────────


def main() -> None:  # pragma: no cover
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Train a custom GristMill sieve classifier from labeled data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", type=Path, required=True, help="CSV or JSONL data file")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("~/.gristmill/models").expanduser(),
    )
    parser.add_argument("--model-name", type=str, default="custom-classifier-v1")
    parser.add_argument("--no-quantize", action="store_true")
    parser.add_argument("--validate", action="store_true", help="Run parity validation")
    args = parser.parse_args()

    pipeline = CustomModelPipeline(
        data_path=args.data,
        epochs=args.epochs,
        output_dir=args.output_dir,
        model_name=args.model_name,
        quantize=not args.no_quantize,
        run_validate=args.validate,
    )
    outcome = pipeline.run()
    if not outcome["success"]:
        raise SystemExit(f"Pipeline failed: {outcome['error']}")


if __name__ == "__main__":
    main()
