"""EmbedderTrainer — fine-tunes the sentence embedding model.

Uses sentence-transformers to fine-tune ``all-MiniLM-L6-v2`` on domain
data (GristMill event texts with soft-similarity labels derived from
routing decisions).  Exports the result as an ONNX model for use in the
Sieve feature extractor and Ledger semantic search.

This is optional for most deployments — the off-the-shelf MiniLM-L6-v2
generalises well.  Fine-tune only when domain vocabulary diverges significantly
from general English.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TrainResult:
    best_loss: float = float("inf")
    best_epoch: int = 0
    train_losses: list[float] = field(default_factory=list)
    onnx_path: Optional[Path] = None

    def summary(self) -> str:
        return f"Best loss: {self.best_loss:.6f} (epoch {self.best_epoch + 1})"


class EmbedderTrainer:
    """Fine-tune the MiniLM-L6-v2 sentence embedder on GristMill domain text.

    Training signal: pairs of event texts that received the same routing
    decision are treated as positives; different routing decisions as negatives.
    Uses MultipleNegativesRankingLoss from sentence-transformers.

    Args:
        base_model: HuggingFace / sentence-transformers model identifier.
        output_dir: Directory for ONNX output.
        experiment_name: MLflow experiment name.
    """

    def __init__(
        self,
        base_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        output_dir: Path = Path("~/.gristmill/models").expanduser(),
        experiment_name: str = "embedder-training",
    ) -> None:
        self.base_model = base_model
        self.output_dir = output_dir
        self.experiment_name = experiment_name

    def train(
        self,
        epochs: int = 2,
        batch_size: int = 32,
        warmup_steps: int = 100,
    ) -> TrainResult:
        """Fine-tune from feedback JSONL pairs (same label = positive pair).

        Requires at least 100 labelled records per class for meaningful
        fine-tuning.
        """
        try:
            from sentence_transformers import SentenceTransformer, InputExample, losses
            from torch.utils.data import DataLoader as STDataLoader
        except ImportError as exc:
            raise ImportError("Install sentence-transformers: pip install -e .") from exc

        from gristmill_ml.datasets.feedback import FeedbackDataset
        from gristmill_ml.experiments.tracking import ExperimentTracker

        dataset = FeedbackDataset()
        tracker = ExperimentTracker(self.experiment_name)
        result = TrainResult()

        # Build positive pairs: same-label records
        label_groups: dict[int, list[str]] = {}
        for i in range(len(dataset)):
            item = dataset[i]
            lbl = item["label"]
            text = item["text"]
            if text:
                label_groups.setdefault(lbl, []).append(text)

        examples: list[InputExample] = []
        for label, texts in label_groups.items():
            for j in range(0, len(texts) - 1, 2):
                examples.append(InputExample(texts=[texts[j], texts[j + 1]]))

        if len(examples) < 10:
            logger.warning(
                "Only %d training pairs — skipping fine-tuning (not enough data)",
                len(examples),
            )
            return result

        model = SentenceTransformer(self.base_model)
        train_loader = STDataLoader(examples, shuffle=True, batch_size=batch_size)
        loss_fn = losses.MultipleNegativesRankingLoss(model)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        with tracker.start_run(f"embedder-{epochs}ep"):
            tracker.log_params(
                {
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "warmup_steps": warmup_steps,
                    "n_pairs": len(examples),
                }
            )
            model.fit(
                train_objectives=[(train_loader, loss_fn)],
                epochs=epochs,
                warmup_steps=warmup_steps,
                output_path=str(self.output_dir / "minilm-finetuned"),
                show_progress_bar=True,
            )

        self._model = model
        logger.info("Embedder fine-tuning complete.")
        return result

    def export(self, output_path: Optional[Path] = None) -> Path:
        """Export the fine-tuned embedder to ONNX."""
        from gristmill_ml.export.onnx_export import OnnxExporter

        dest = output_path or (self.output_dir / "minilm-l6-v2.onnx")
        return OnnxExporter.export_embedder(
            model_name=str(self.output_dir / "minilm-finetuned"),
            output_path=dest,
        )
