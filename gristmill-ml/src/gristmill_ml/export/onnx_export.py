"""ONNX export pipeline — PyTorch → ONNX → INT8 quantization.

All exported models must use the exact input/output names that the Rust
runtime expects (see ``grist_grinders/src/onnx.rs``):

Classifier (intent-classifier-v1.onnx)
    input:  ``features``  shape [batch, 392], dtype float32
    output: ``logits``    shape [batch, 4],   dtype float32

MiniLM embedder (minilm-l6-v2.onnx)
    inputs: ``input_ids``      shape [batch, seq], dtype int64
            ``attention_mask`` shape [batch, seq], dtype int64
            ``token_type_ids`` shape [batch, seq], dtype int64
    output: ``last_hidden_state`` shape [batch, seq, 384], dtype float32
"""

from __future__ import annotations

import argparse
import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch

if TYPE_CHECKING:
    from gristmill_ml.training.sieve_trainer import SieveClassifierHead

logger = logging.getLogger(__name__)


class OnnxExporter:
    """Static-method collection for exporting GristMill models to ONNX."""

    @staticmethod
    def export_classifier(
        model: "SieveClassifierHead",
        output_path: Path,
        quantize: bool = True,
        opset: int = 17,
    ) -> Path:
        """Export *model* to ONNX and optionally apply INT8 dynamic quantization.

        Args:
            model: Trained :class:`SieveClassifierHead` in eval mode.
            output_path: Destination ``.onnx`` path.
            quantize: If ``True``, apply dynamic INT8 quantization (reduces
                size from ~0.5 MB to ~0.2 MB with negligible accuracy loss).
            opset: ONNX opset version (17 required for current ort 1.18+).

        Returns:
            Path to the final ``.onnx`` file (may differ from *output_path*
            when quantization appends a suffix).
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        from gristmill_ml.training.sieve_trainer import FEATURE_DIM

        model.eval()
        dummy_input = torch.zeros(1, FEATURE_DIM, dtype=torch.float32)

        # Export to a temp file first so quantization can overwrite atomically
        if quantize:
            fp32_path = output_path.with_suffix(".fp32.onnx")
        else:
            fp32_path = output_path

        logger.info("Exporting classifier to ONNX (fp32) → %s", fp32_path)
        torch.onnx.export(
            model,
            dummy_input,
            str(fp32_path),
            opset_version=opset,
            input_names=["features"],
            output_names=["logits"],
            dynamic_axes={"features": {0: "batch"}, "logits": {0: "batch"}},
        )

        if not quantize:
            logger.info("Classifier exported (fp32): %s", fp32_path)
            return fp32_path

        # INT8 dynamic quantisation (weights + activations)
        logger.info("Applying INT8 dynamic quantization → %s", output_path)
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType

            quantize_dynamic(
                model_input=str(fp32_path),
                model_output=str(output_path),
                weight_type=QuantType.QInt8,
            )
            fp32_path.unlink(missing_ok=True)  # remove fp32 intermediate
            logger.info("Classifier exported (INT8): %s", output_path)
            return output_path
        except Exception as exc:
            logger.warning(
                "INT8 quantization failed (%s) — falling back to fp32 model", exc
            )
            fp32_path.rename(output_path)
            return output_path

    @staticmethod
    def export_embedder(
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        output_path: Path = Path("~/.gristmill/models/minilm-l6-v2.onnx").expanduser(),
        quantize: bool = True,
        max_seq_len: int = 128,
        opset: int = 17,
    ) -> Path:
        """Export a HuggingFace sentence-transformer to ONNX.

        Exports the transformer backbone (not the pooling head) with the
        canonical ``last_hidden_state`` output expected by the Rust runtime.
        The Rust code does its own mean-pooling and L2 normalisation.

        Args:
            model_name: HuggingFace hub id or local path.
            output_path: Destination ``.onnx`` path.
            quantize: Apply INT8 quantization.
            max_seq_len: Maximum sequence length (must match Rust constant 128).
            opset: ONNX opset version.
        """
        try:
            from transformers import AutoTokenizer, AutoModel
        except ImportError as exc:
            raise ImportError("Install transformers: pip install -e .") from exc

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Loading model: %s", model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()

        dummy_text = "GristMill is a Rust-first AI orchestration engine"
        enc = tokenizer(
            dummy_text,
            return_tensors="pt",
            max_length=max_seq_len,
            padding="max_length",
            truncation=True,
        )

        fp32_path = output_path.with_suffix(".fp32.onnx") if quantize else output_path

        logger.info("Exporting embedder to ONNX → %s", fp32_path)
        torch.onnx.export(
            model,
            (enc["input_ids"], enc["attention_mask"], enc["token_type_ids"]),
            str(fp32_path),
            opset_version=opset,
            input_names=["input_ids", "attention_mask", "token_type_ids"],
            output_names=["last_hidden_state"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
                "token_type_ids": {0: "batch", 1: "seq"},
                "last_hidden_state": {0: "batch", 1: "seq"},
            },
        )

        if not quantize:
            logger.info("Embedder exported (fp32): %s", fp32_path)
            return fp32_path

        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType

            quantize_dynamic(
                model_input=str(fp32_path),
                model_output=str(output_path),
                weight_type=QuantType.QInt8,
            )
            fp32_path.unlink(missing_ok=True)
            logger.info("Embedder exported (INT8): %s", output_path)
            return output_path
        except Exception as exc:
            logger.warning("INT8 quantization failed (%s) — fp32 kept", exc)
            fp32_path.rename(output_path)
            return output_path

    @staticmethod
    def export_ner(
        model: Optional[object],
        tokenizer: Optional[object],
        output_path: Path,
        quantize: bool = True,
        opset: int = 17,
    ) -> Path:
        """Export a HuggingFace token-classification model to ONNX."""
        if model is None:
            raise ValueError("model must be provided — call NerTrainer.train() first")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        model.eval()  # type: ignore[union-attr]
        dummy = tokenizer(  # type: ignore[operator]
            "London is the capital of England",
            return_tensors="pt",
            max_length=128,
            padding="max_length",
            truncation=True,
        )

        fp32_path = output_path.with_suffix(".fp32.onnx") if quantize else output_path

        torch.onnx.export(
            model,
            (dummy["input_ids"], dummy["attention_mask"], dummy["token_type_ids"]),
            str(fp32_path),
            opset_version=opset,
            input_names=["input_ids", "attention_mask", "token_type_ids"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
                "token_type_ids": {0: "batch", 1: "seq"},
                "logits": {0: "batch"},
            },
        )

        if not quantize:
            return fp32_path

        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType

            quantize_dynamic(
                model_input=str(fp32_path),
                model_output=str(output_path),
                weight_type=QuantType.QInt8,
            )
            fp32_path.unlink(missing_ok=True)
            return output_path
        except Exception as exc:
            logger.warning("NER INT8 quantization failed (%s) — fp32 kept", exc)
            fp32_path.rename(output_path)
            return output_path


# ── CLI entry-point ───────────────────────────────────────────────────────────

def main() -> None:  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description="Export GristMill models to ONNX")
    parser.add_argument(
        "--model", choices=["classifier", "embedder"], required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-quantize", action="store_true")
    args = parser.parse_args()

    if args.model == "classifier":
        from gristmill_ml.training.sieve_trainer import SieveClassifierHead
        model = SieveClassifierHead()
        OnnxExporter.export_classifier(model, args.output, quantize=not args.no_quantize)
    elif args.model == "embedder":
        OnnxExporter.export_embedder(output_path=args.output, quantize=not args.no_quantize)


if __name__ == "__main__":
    main()
