"""Standalone post-hoc INT8 quantization utility for ONNX models.

Wraps ``onnxruntime.quantization`` to apply dynamic or static INT8 quantization
to any ONNX model that follows the GristMill export conventions.

Usage as a library::

    from gristmill_ml.export.quantize import OnnxQuantizer
    out = OnnxQuantizer.quantize_dynamic(Path("model.onnx"))

CLI usage::

    python -m gristmill_ml.export.quantize input.onnx [--output PATH] [--static] \\
        [--weight-type QInt8|QUInt8]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class OnnxQuantizer:
    """Post-hoc INT8 dynamic quantization of ONNX models.

    All methods are static and raise :class:`RuntimeError` if
    ``onnxruntime`` (specifically its ``quantization`` sub-module) is not
    installed.
    """

    @staticmethod
    def quantize_dynamic(
        input_path: Path,
        output_path: Optional[Path] = None,
        weight_type: str = "QInt8",
    ) -> Path:
        """Apply dynamic INT8 quantization to *input_path*.

        Dynamic quantization computes activation scales at runtime and
        quantizes weights offline.  This is the recommended approach for
        most inference scenarios.

        Args:
            input_path: Path to the source ``.onnx`` model.
            output_path: Destination path.  Defaults to
                ``{input_path.stem}_int8.onnx`` in the same directory.
            weight_type: ``"QInt8"`` (default) or ``"QUInt8"``.

        Returns:
            Path to the quantized model file.

        Raises:
            RuntimeError: If ``onnxruntime`` or its quantization module is
                not installed.
            FileNotFoundError: If *input_path* does not exist.
        """
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime quantization support is required. "
                "Install with: pip install onnxruntime"
            ) from exc

        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input model not found: {input_path}")

        if output_path is None:
            output_path = input_path.parent / f"{input_path.stem}_int8.onnx"
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        q_type = QuantType.QInt8 if weight_type == "QInt8" else QuantType.QUInt8

        logger.info(
            "Applying dynamic INT8 quantization (%s): %s → %s",
            weight_type,
            input_path,
            output_path,
        )
        quantize_dynamic(
            model_input=str(input_path),
            model_output=str(output_path),
            weight_type=q_type,
        )
        logger.info(
            "Quantization complete. Size: %.2f MB → %.2f MB (ratio %.2f×)",
            input_path.stat().st_size / 1e6,
            output_path.stat().st_size / 1e6,
            input_path.stat().st_size / max(output_path.stat().st_size, 1),
        )
        return output_path

    @staticmethod
    def quantize_static(
        input_path: Path,
        calibration_data: list[dict],
        output_path: Optional[Path] = None,
    ) -> Path:
        """Apply static INT8 quantization with calibration data.

        Static quantization pre-computes activation scales using a small
        calibration dataset, which typically yields better accuracy than
        dynamic quantization at the cost of a calibration pass.

        Args:
            input_path: Path to the source ``.onnx`` model.
            calibration_data: List of dicts mapping input name to
                ``np.ndarray`` (e.g. ``[{"features": array}]``).
            output_path: Destination path.  Defaults to
                ``{input_path.stem}_static_int8.onnx``.

        Returns:
            Path to the quantized model file.

        Raises:
            RuntimeError: If ``onnxruntime`` or its quantization module is
                not installed or static quantization fails.
            FileNotFoundError: If *input_path* does not exist.
        """
        try:
            from onnxruntime.quantization import (  # type: ignore[import]
                quantize_static,
                CalibrationDataReader,
                QuantType,
                QuantFormat,
            )
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime quantization support is required. "
                "Install with: pip install onnxruntime"
            ) from exc

        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input model not found: {input_path}")

        if output_path is None:
            output_path = input_path.parent / f"{input_path.stem}_static_int8.onnx"
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        class _DataReader(CalibrationDataReader):
            """Wraps the calibration_data list for onnxruntime."""

            def __init__(self, data: list[dict]) -> None:
                self._data = iter(data)

            def get_next(self) -> Optional[dict]:
                return next(self._data, None)

            def rewind(self) -> None:  # pragma: no cover
                pass

        logger.info(
            "Applying static INT8 quantization (calibration samples=%d): %s → %s",
            len(calibration_data),
            input_path,
            output_path,
        )

        try:
            quantize_static(
                model_input=str(input_path),
                model_output=str(output_path),
                calibration_data_reader=_DataReader(calibration_data),
                quant_format=QuantFormat.QDQ,
                activation_type=QuantType.QInt8,
                weight_type=QuantType.QInt8,
                per_channel=False,
            )
        except Exception as exc:
            raise RuntimeError(f"Static quantization failed: {exc}") from exc

        logger.info(
            "Static quantization complete. Size: %.2f MB → %.2f MB",
            input_path.stat().st_size / 1e6,
            output_path.stat().st_size / 1e6,
        )
        return output_path

    @staticmethod
    def compare_sizes(original: Path, quantized: Path) -> dict:
        """Return a size comparison dict for *original* and *quantized*.

        Args:
            original: Path to the original (larger) model.
            quantized: Path to the quantized (smaller) model.

        Returns:
            Dict with keys ``original_mb``, ``quantized_mb``, ``ratio``
            (original / quantized, so > 1 means the quantized model is smaller).
        """
        original = Path(original)
        quantized = Path(quantized)

        orig_bytes = original.stat().st_size if original.exists() else 0
        quant_bytes = quantized.stat().st_size if quantized.exists() else 0

        orig_mb = orig_bytes / 1e6
        quant_mb = quant_bytes / 1e6
        ratio = orig_bytes / max(quant_bytes, 1)

        return {
            "original_mb": round(orig_mb, 3),
            "quantized_mb": round(quant_mb, 3),
            "ratio": round(ratio, 3),
        }


# ── CLI entry-point ───────────────────────────────────────────────────────────


def main() -> None:  # pragma: no cover
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Post-hoc INT8 quantization for GristMill ONNX models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", type=Path, help="Input ONNX model path")
    parser.add_argument("--output", type=Path, default=None, help="Output ONNX path")
    parser.add_argument(
        "--static",
        action="store_true",
        help="Use static quantization (requires calibration data from dummy inputs)",
    )
    parser.add_argument(
        "--weight-type",
        choices=["QInt8", "QUInt8"],
        default="QInt8",
        help="Quantization weight type",
    )
    args = parser.parse_args()

    if args.static:
        # Generate minimal calibration data from random inputs
        from gristmill_ml.training.sieve_trainer import FEATURE_DIM

        n_calib = 64
        calibration_data = [
            {"features": np.random.randn(1, FEATURE_DIM).astype(np.float32)} for _ in range(n_calib)
        ]
        out = OnnxQuantizer.quantize_static(
            args.input,
            calibration_data=calibration_data,
            output_path=args.output,
        )
    else:
        out = OnnxQuantizer.quantize_dynamic(
            args.input,
            output_path=args.output,
            weight_type=args.weight_type,
        )

    sizes = OnnxQuantizer.compare_sizes(args.input, out)
    print(
        f"Quantized: {out}\n"
        f"  Original : {sizes['original_mb']:.3f} MB\n"
        f"  Quantized: {sizes['quantized_mb']:.3f} MB\n"
        f"  Ratio    : {sizes['ratio']:.2f}×"
    )


if __name__ == "__main__":
    main()
