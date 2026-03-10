"""NerTrainer — fine-tunes a token-classification NER model.

Uses HuggingFace ``transformers`` to fine-tune a pre-trained multilingual
NER model on custom labelled data, then exports the result as a quantised
ONNX model compatible with ``grist-grinders``.

Default base model: ``dslim/bert-base-NER``
Default output:     ``~/.gristmill/models/ner-multilingual-v1.onnx``
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class TrainResult:
    best_f1: float = 0.0
    best_epoch: int = 0
    train_losses: list[float] = field(default_factory=list)
    val_f1s: list[float] = field(default_factory=list)
    classification_report: str = ""
    onnx_path: Optional[Path] = None

    def summary(self) -> str:
        return (
            f"Best val F1: {self.best_f1:.4f} (epoch {self.best_epoch + 1})\n"
            f"{self.classification_report}"
        )


class NerTrainer:
    """Fine-tune a HuggingFace NER model and export to ONNX.

    Args:
        base_model: HuggingFace model hub identifier.
        output_dir: Directory where the ONNX model file will be written.
        experiment_name: MLflow experiment name.
    """

    def __init__(
        self,
        base_model: str = "dslim/bert-base-NER",
        output_dir: Path = Path("~/.gristmill/models").expanduser(),
        experiment_name: str = "ner-training",
    ) -> None:
        self.base_model = base_model
        self.output_dir = output_dir
        self.experiment_name = experiment_name

    def train(
        self,
        dataset_path: Path,
        epochs: int = 3,
        batch_size: int = 16,
        lr: float = 5e-5,
    ) -> TrainResult:
        """Fine-tune on the CoNLL-2003-style dataset at *dataset_path*.

        The dataset must be a JSONL file where each line has the structure:
        ``{"tokens": [...], "ner_tags": [0, 1, 0, ...]}``.
        """
        try:
            import torch
            from datasets import load_dataset
            from transformers import (
                AutoModelForTokenClassification,
                AutoTokenizer,
                DataCollatorForTokenClassification,
                Trainer,
                TrainingArguments,
            )
            import evaluate
        except ImportError as exc:
            raise ImportError("Install transformers, datasets, and evaluate: pip install -e .[dev]") from exc

        from gristmill_ml.experiments.tracking import ExperimentTracker

        tracker = ExperimentTracker(self.experiment_name)
        result = TrainResult()

        logger.info("Loading tokenizer and model: %s", self.base_model)
        tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        model = AutoModelForTokenClassification.from_pretrained(self.base_model)

        dataset = load_dataset("json", data_files=str(dataset_path), split="train")
        dataset = dataset.train_test_split(test_size=0.15, seed=42)

        seqeval = evaluate.load("seqeval")

        def tokenize_and_align_labels(examples: dict[str, Any]) -> dict[str, Any]:
            tokenized = tokenizer(
                examples["tokens"],
                truncation=True,
                is_split_into_words=True,
                padding="max_length",
                max_length=128,
            )
            labels = []
            for i, label_ids in enumerate(examples["ner_tags"]):
                word_ids = tokenized.word_ids(batch_index=i)
                aligned = []
                prev_word_id = None
                for word_id in word_ids:
                    if word_id is None:
                        aligned.append(-100)
                    elif word_id != prev_word_id:
                        aligned.append(label_ids[word_id])
                    else:
                        aligned.append(-100)
                    prev_word_id = word_id
                labels.append(aligned)
            tokenized["labels"] = labels
            return tokenized

        tokenized_ds = dataset.map(tokenize_and_align_labels, batched=True)
        collator = DataCollatorForTokenClassification(tokenizer)

        # Determine label names
        label_list = model.config.id2label

        def compute_metrics(p: Any) -> dict[str, float]:
            predictions, labels = p
            import numpy as np
            pred_ids = predictions.argmax(axis=-1)
            true_labels = [
                [label_list[l] for l in label if l != -100]
                for label in labels
            ]
            true_preds = [
                [label_list[p] for p, l in zip(pred, label) if l != -100]
                for pred, label in zip(pred_ids, labels)
            ]
            metrics = seqeval.compute(predictions=true_preds, references=true_labels)
            return {
                "precision": metrics["overall_precision"],
                "recall": metrics["overall_recall"],
                "f1": metrics["overall_f1"],
                "accuracy": metrics["overall_accuracy"],
            }

        self.output_dir.mkdir(parents=True, exist_ok=True)
        training_args = TrainingArguments(
            output_dir=str(self.output_dir / "ner-checkpoints"),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=lr,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            logging_steps=10,
            report_to="none",  # MLflow handled separately
        )

        with tracker.start_run(f"ner-{epochs}ep"):
            tracker.log_params({"epochs": epochs, "batch_size": batch_size, "lr": lr})

            trainer_hf = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_ds["train"],
                eval_dataset=tokenized_ds["test"],
                data_collator=collator,
                tokenizer=tokenizer,
                compute_metrics=compute_metrics,
            )
            trainer_hf.train()
            eval_metrics = trainer_hf.evaluate()
            result.best_f1 = eval_metrics.get("eval_f1", 0.0)
            tracker.log_metrics(eval_metrics)

        result.classification_report = str(eval_metrics)
        # Store the fine-tuned model for export
        self._model = model
        self._tokenizer = tokenizer
        return result

    def export(self, output_path: Optional[Path] = None) -> Path:
        """Export the fine-tuned model to ONNX and return the path."""
        from gristmill_ml.export.onnx_export import OnnxExporter
        dest = output_path or (self.output_dir / "ner-multilingual-v1.onnx")
        return OnnxExporter.export_ner(
            getattr(self, "_model", None),
            getattr(self, "_tokenizer", None),
            dest,
        )
