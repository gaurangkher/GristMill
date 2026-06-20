"""Tests for OnnxExporter.export_ner's dynamic-axes configuration.

Regression coverage: export_ner's dynamic_axes used to mark the "logits"
output as dynamic only on axis 0 ({0: "batch"}). NER output is per-token
([batch, seq, num_labels]), so without marking axis 1 dynamic too, the
exported graph bakes in the seq_len=128 dummy length used for tracing —
inconsistent with export_embedder's last_hidden_state (which correctly
marks {0: "batch", 1: "seq"}) and liable to break at inference time for
any input whose sequence length differs from 128.
"""

from __future__ import annotations

from pathlib import Path

import onnx
import torch
import torch.nn as nn

from gristmill_ml.export.onnx_export import OnnxExporter


class _TinyNerModel(nn.Module):
    """Minimal stand-in for a HF token-classification model."""

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        batch, seq = input_ids.shape
        return torch.zeros(batch, seq, 5)


def _fake_tokenizer(
    text: str,
    return_tensors: str = "pt",
    max_length: int = 128,
    padding: str = "max_length",
    truncation: bool = True,
) -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.ones(1, max_length, dtype=torch.long),
        "attention_mask": torch.ones(1, max_length, dtype=torch.long),
        "token_type_ids": torch.zeros(1, max_length, dtype=torch.long),
    }


def test_export_ner_logits_output_has_dynamic_seq_axis(tmp_path: Path) -> None:
    out_path = OnnxExporter.export_ner(
        _TinyNerModel(), _fake_tokenizer, tmp_path / "ner.onnx", quantize=False
    )

    model = onnx.load(str(out_path))
    dims = model.graph.output[0].type.tensor_type.shape.dim

    # [batch, seq, num_labels] — first two dims must be symbolic (dynamic),
    # not fixed to the seq_len=128 dummy used for tracing.
    assert dims[0].dim_param != ""
    assert dims[1].dim_param != ""
