"""SieveTrainer — trains the 4-class intent-routing classifier.

The classifier maps a 392-dimensional feature vector to one of four routing
decisions:

    0 = LOCAL_ML   (local ONNX/GGUF model is sufficient)
    1 = RULES      (deterministic rule engine)
    2 = HYBRID     (local pre-process + LLM refinement)
    3 = LLM_NEEDED (full LLM reasoning required)

Feature vector layout (392 dims, must match ``grist_sieve/features.rs``):
    [0:384]  — L2-normalised MiniLM-L6-v2 sentence embedding
    [384]    — log-scaled token count: ln(tc+1) / ln(2049), clamped [0, 1]
    [385]    — source channel ordinal / 9.0
    [386]    — priority / 3.0
    [387]    — entity density (placeholder, always 0 when spacy is absent)
    [388]    — question probability (0 / 0.6 / 1.0)
    [389]    — code probability (fraction of code-indicator tokens)
    [390]    — type-token ratio (unique_words / total_words)
    [391]    — ambiguity score (1 − max_freq / token_count)
"""

from __future__ import annotations

import argparse
import logging
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report, confusion_matrix
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler

from gristmill_ml.datasets.feedback import CHANNEL_ORDINAL, FeedbackDataset, ROUTE_LABEL_MAP
from gristmill_ml.datasets.augmentation import augment_dataset

logger = logging.getLogger(__name__)

# ── Constants (must match grist_sieve/features.rs) ───────────────────────────

EMBEDDING_DIM: int = 384
METADATA_FEATURES: int = 8
FEATURE_DIM: int = 392  # 384 + 8
NUM_CLASSES: int = 4

_CODE_INDICATORS: frozenset[str] = frozenset(
    [
        "def ", "fn ", "func ", "function ", "class ", "import ", "from ",
        "let ", "const ", "var ", "return ", "if ", "else ", "for ", "while ",
        "{", "}", "()", "=>", "->", "::", "//", "/*", "#!", "```",
        ".rs", ".py", ".ts", ".js", ".go", ".cpp",
    ]
)

# ─────────────────────────────────────────────────────────────────────────────
# SieveClassifierHead
# ─────────────────────────────────────────────────────────────────────────────

class SieveClassifierHead(nn.Module):
    """Lightweight 2-layer MLP on top of pre-computed MiniLM embeddings.

    Input:  feature vector of shape ``[batch, 392]``
    Output: logits of shape ``[batch, 4]``
    """

    def __init__(
        self,
        input_dim: int = FEATURE_DIM,
        hidden_dim: int = 128,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# FeatureExtractor
# ─────────────────────────────────────────────────────────────────────────────

class FeatureExtractor:
    """Pure-Python mirror of ``grist_sieve/features.rs``.

    Produces a 392-dimensional feature vector identical to what the Rust
    runtime computes during inference, ensuring training/serving parity.
    """

    def __init__(
        self,
        embedder: SentenceTransformer,
        device: str = "cpu",
    ) -> None:
        self.embedder = embedder
        self.device = device

        # Try to load spacy for entity density (optional)
        try:
            import spacy  # noqa: F401
            self._nlp = spacy.load("en_core_web_sm", disable=["parser", "tagger"])
        except Exception:
            self._nlp = None

    def extract(
        self,
        text: str,
        source: str,
        priority: int,
        token_count: Optional[int] = None,
    ) -> np.ndarray:
        """Extract a 392-dimensional feature vector from *text* and metadata."""
        tokens = text.lower().split()
        tc = token_count if token_count is not None else len(tokens)

        # ── Embedding (dims 0–383) ────────────────────────────────────────────
        embedding = self.embedder.encode(
            text, normalize_embeddings=True, show_progress_bar=False
        ).astype(np.float32)

        # ── Scalar metadata (dims 384–391) ────────────────────────────────────
        # [384] log-scaled token count
        log_tc = math.log(tc + 1) / math.log(2049) if tc > 0 else 0.0
        log_tc = min(1.0, max(0.0, log_tc))

        # [385] source channel ordinal
        ch_ord = CHANNEL_ORDINAL.get(source.lower(), 9) / 9.0

        # [386] priority (0=low, 1=normal, 2=high, 3=critical) → [0, 1]
        pri = min(3, max(0, priority)) / 3.0

        # [387] entity density
        if self._nlp is not None and text:
            doc = self._nlp(text[:512])
            entity_count = len(doc.ents)
            entity_density = min(1.0, entity_count / max(tc, 1))
        else:
            entity_density = 0.0

        # [388] question probability
        lowered = text.lower()
        question_starters = ("what", "why", "how", "who", "when", "where", "is", "are", "can")
        if text.endswith("?"):
            q_prob = 1.0
        elif any(lowered.startswith(s) for s in question_starters):
            q_prob = 0.6
        else:
            q_prob = 0.0

        # [389] code probability
        code_hits = sum(1 for ind in _CODE_INDICATORS if ind in text)
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
        assert metadata.shape == (METADATA_FEATURES,), metadata.shape

        return np.concatenate([embedding, metadata])

    def extract_batch(
        self,
        items: list[dict[str, Any]],
    ) -> np.ndarray:
        """Vectorise a batch of dataset items into shape ``[N, 392]``."""
        # Batch-encode texts first (much faster than encoding one by one)
        texts = [item.get("text", "") for item in items]
        embeddings = self.embedder.encode(
            texts, normalize_embeddings=True, show_progress_bar=False, batch_size=64
        ).astype(np.float32)

        rows = []
        for i, item in enumerate(items):
            tc = item.get("token_count") or len(texts[i].split())
            source = item.get("source", "internal")
            priority = item.get("priority", 1)
            text = texts[i]
            tokens = text.lower().split()

            log_tc = math.log(tc + 1) / math.log(2049) if tc > 0 else 0.0
            log_tc = min(1.0, max(0.0, log_tc))
            ch_ord = CHANNEL_ORDINAL.get(source.lower(), 9) / 9.0
            pri = min(3, max(0, priority)) / 3.0

            if self._nlp is not None and text:
                doc = self._nlp(text[:512])
                entity_density = min(1.0, len(doc.ents) / max(tc, 1))
            else:
                entity_density = 0.0

            lowered = text.lower()
            question_starters = ("what", "why", "how", "who", "when", "where", "is", "are", "can")
            if text.endswith("?"):
                q_prob = 1.0
            elif any(lowered.startswith(s) for s in question_starters):
                q_prob = 0.6
            else:
                q_prob = 0.0

            code_hits = sum(1 for ind in _CODE_INDICATORS if ind in text)
            code_prob = min(1.0, code_hits / max(tc, 1))
            ttr = len(set(tokens)) / max(len(tokens), 1)

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


# ─────────────────────────────────────────────────────────────────────────────
# TrainResult
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainResult:
    best_val_accuracy: float = 0.0
    best_epoch: int = 0
    train_losses: list[float] = field(default_factory=list)
    val_accuracies: list[float] = field(default_factory=list)
    classification_report: str = ""
    onnx_path: Optional[Path] = None

    def summary(self) -> str:
        return (
            f"Best val accuracy: {self.best_val_accuracy:.4f} "
            f"(epoch {self.best_epoch + 1})\n"
            f"{self.classification_report}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# SieveTrainer
# ─────────────────────────────────────────────────────────────────────────────

class SieveTrainer:
    """Full training pipeline for the Sieve intent classifier.

    Usage::

        trainer = SieveTrainer(
            feedback_dir=Path("~/.gristmill/feedback").expanduser(),
            output_dir=Path("~/.gristmill/models"),
        )
        result = trainer.train(epochs=10)
        print(result.summary())
    """

    def __init__(
        self,
        feedback_dir: Path = Path("~/.gristmill/feedback").expanduser(),
        output_dir: Path = Path("~/.gristmill/models").expanduser(),
        experiment_name: str = "sieve-training",
        augment: bool = True,
        device: Optional[str] = None,
    ) -> None:
        self.feedback_dir = feedback_dir
        self.output_dir = output_dir
        self.experiment_name = experiment_name
        self.augment = augment
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("SieveTrainer device=%s", self.device)

        # Lazy-loaded components
        self._embedder: Optional[SentenceTransformer] = None
        self._feature_extractor: Optional[FeatureExtractor] = None
        self._model: Optional[SieveClassifierHead] = None

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            logger.info("Loading sentence-transformers/all-MiniLM-L6-v2 …")
            self._embedder = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2", device=self.device
            )
        return self._embedder

    @property
    def feature_extractor(self) -> FeatureExtractor:
        if self._feature_extractor is None:
            self._feature_extractor = FeatureExtractor(self.embedder, device=self.device)
        return self._feature_extractor

    @property
    def model(self) -> SieveClassifierHead:
        if self._model is None:
            self._model = SieveClassifierHead().to(self.device)
        return self._model

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_tensors(
        self,
        dataset: FeedbackDataset,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Vectorise all dataset items into (features, labels) tensors."""
        items = [dataset[i] for i in range(len(dataset))]
        features = self.feature_extractor.extract_batch(items)
        labels = np.array([item["label"] for item in items], dtype=np.int64)
        return (
            torch.from_numpy(features).float(),
            torch.from_numpy(labels).long(),
        )

    # ── Training ──────────────────────────────────────────────────────────────

    def train(
        self,
        epochs: int = 10,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 3,
    ) -> TrainResult:
        """Run the full training loop.

        Returns a :class:`TrainResult` with per-epoch metrics and the path
        to the best checkpoint.
        """
        from gristmill_ml.experiments.tracking import ExperimentTracker

        tracker = ExperimentTracker(self.experiment_name)

        # ── Build datasets ────────────────────────────────────────────────────
        train_ds = FeedbackDataset(self.feedback_dir, split="train")
        val_ds = FeedbackDataset(self.feedback_dir, split="val")

        logger.info(
            "Dataset — train: %d, val: %d", len(train_ds), len(val_ds)
        )
        logger.info("Class distribution (train): %s", train_ds.class_counts())

        if self.augment and len(train_ds) > 0:
            logger.info("Augmenting training set …")
            train_ds.records = augment_dataset(
                train_ds.records, target_per_class=max(200, len(train_ds) // NUM_CLASSES)
            )
            logger.info("Augmented train size: %d", len(train_ds))

        # ── Vectorise ─────────────────────────────────────────────────────────
        logger.info("Extracting features …")
        X_train, y_train = self._build_tensors(train_ds)
        X_val, y_val = self._build_tensors(val_ds)

        # ── Weighted sampler for class imbalance ──────────────────────────────
        class_weights_np = train_ds.class_weights()
        sample_weights = torch.tensor(
            [class_weights_np[y] for y in y_train.numpy()], dtype=torch.float
        )
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

        train_loader = DataLoader(
            list(zip(X_train, y_train)),
            batch_size=batch_size,
            sampler=sampler,
        )
        val_loader = DataLoader(
            list(zip(X_val, y_val)),
            batch_size=batch_size,
            shuffle=False,
        )

        # ── Optimiser & scheduler ─────────────────────────────────────────────
        class_weights_tensor = torch.tensor(class_weights_np, device=self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

        # ── Training loop ─────────────────────────────────────────────────────
        result = TrainResult()
        best_state: Optional[dict] = None
        no_improve = 0

        with tracker.start_run(f"sieve-{epochs}ep"):
            tracker.log_params({
                "epochs": epochs, "batch_size": batch_size, "lr": lr,
                "weight_decay": weight_decay, "augment": self.augment,
                "train_size": len(train_ds), "val_size": len(val_ds),
            })

            for epoch in range(epochs):
                # Train
                self.model.train()
                epoch_loss = 0.0
                for xb, yb in train_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    optimizer.zero_grad()
                    loss = criterion(self.model(xb), yb)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    epoch_loss += loss.item() * len(yb)
                scheduler.step()

                train_loss = epoch_loss / max(len(train_ds), 1)
                result.train_losses.append(train_loss)

                # Validate
                val_acc = self._evaluate_accuracy(val_loader)
                result.val_accuracies.append(val_acc)

                tracker.log_metrics(
                    {"train_loss": train_loss, "val_accuracy": val_acc},
                    step=epoch,
                )
                logger.info(
                    "Epoch %2d/%d — loss=%.4f  val_acc=%.4f",
                    epoch + 1, epochs, train_loss, val_acc,
                )

                if val_acc > result.best_val_accuracy:
                    result.best_val_accuracy = val_acc
                    result.best_epoch = epoch
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        logger.info("Early stopping at epoch %d", epoch + 1)
                        break

        # Restore best weights
        if best_state is not None:
            self.model.load_state_dict(best_state)

        # Full classification report on validation set
        result.classification_report = self._classification_report(val_loader)
        logger.info("Final report:\n%s", result.classification_report)

        return result

    def _evaluate_accuracy(self, loader: DataLoader) -> float:
        self.model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                preds = self.model(xb).argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += len(yb)
        return correct / max(total, 1)

    def _classification_report(self, loader: DataLoader) -> str:
        self.model.eval()
        all_preds: list[int] = []
        all_labels: list[int] = []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(self.device)
                preds = self.model(xb).argmax(dim=1).cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(yb.tolist())
        target_names = list(ROUTE_LABEL_MAP.keys())
        return classification_report(
            all_labels, all_preds, target_names=target_names, zero_division=0
        )

    def evaluate(self, split: str = "val") -> dict[str, Any]:
        """Return accuracy and per-class F1 for *split*."""
        ds = FeedbackDataset(self.feedback_dir, split=split)
        X, y = self._build_tensors(ds)
        loader = DataLoader(list(zip(X, y)), batch_size=64, shuffle=False)
        acc = self._evaluate_accuracy(loader)
        report = self._classification_report(loader)
        return {"accuracy": acc, "classification_report": report}

    def export(
        self,
        output_path: Optional[Path] = None,
        quantize: bool = True,
    ) -> Path:
        """Export the model to ONNX (INT8 if *quantize*) and return the path.

        Also runs parity validation — raises :class:`AssertionError` if the
        ONNX outputs differ from PyTorch by more than 1e-4.
        """
        from gristmill_ml.export.onnx_export import OnnxExporter
        from gristmill_ml.export.validate import validate_classifier_parity

        self.output_dir.mkdir(parents=True, exist_ok=True)
        dest = output_path or (self.output_dir / "intent-classifier-v1.onnx")

        onnx_path = OnnxExporter.export_classifier(
            self.model, dest, quantize=quantize
        )

        # Quick parity check on synthetic data
        dummy = np.random.randn(16, FEATURE_DIM).astype(np.float32)
        result = validate_classifier_parity(self.model, onnx_path, dummy)
        if not result["passed"]:
            raise AssertionError(
                f"ONNX parity check failed: max_diff={result['max_diff']:.6f}"
            )
        logger.info("ONNX parity OK — max_diff=%.8f", result["max_diff"])
        return onnx_path

    @classmethod
    def run_weekly(
        cls,
        feedback_dir: Path = Path("~/.gristmill/feedback").expanduser(),
        output_dir: Path = Path("~/.gristmill/models").expanduser(),
    ) -> Path:
        """Convenience entry-point for the weekly cron job.

        Loads feedback, trains, exports, and returns the new ONNX model path.
        """
        trainer = cls(feedback_dir=feedback_dir, output_dir=output_dir)
        result = trainer.train()
        logger.info("Weekly training complete:\n%s", result.summary())
        onnx_path = trainer.export()
        logger.info("Model exported to %s", onnx_path)
        return onnx_path


# ── CLI entry-point ───────────────────────────────────────────────────────────

def main() -> None:  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description="Train the GristMill Sieve classifier")
    parser.add_argument("--feedback-dir", type=Path,
                        default=Path("~/.gristmill/feedback").expanduser())
    parser.add_argument("--output-dir", type=Path,
                        default=Path("~/.gristmill/models").expanduser())
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--no-quantize", action="store_true")
    args = parser.parse_args()

    trainer = SieveTrainer(feedback_dir=args.feedback_dir, output_dir=args.output_dir)
    result = trainer.train(epochs=args.epochs)
    print(result.summary())
    onnx_path = trainer.export(quantize=not args.no_quantize)
    print(f"Exported: {onnx_path}")


if __name__ == "__main__":
    main()
