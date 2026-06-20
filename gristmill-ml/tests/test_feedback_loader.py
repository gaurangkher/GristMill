"""Tests for FeedbackDataset._load_jsonl.

Covers malformed-line handling and a data-loss regression: rows missing
event_id used to be keyed by "" in the dedup dict, so multiple such rows
silently overwrote each other and only the last one survived.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gristmill_ml.datasets.feedback import FeedbackDataset


def _row(**overrides: Any) -> dict[str, Any]:
    base = {"timestamp_ms": 1_700_000_000_000, "confidence": 0.9}
    base.update(overrides)
    return base


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_load_jsonl_skips_rows_without_event_id(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "feedback-001.jsonl",
        [
            _row(event_id="", route_decision="LOCAL_ML"),
            _row(event_id="", route_decision="RULES"),
            _row(event_id="e1", route_decision="HYBRID"),
        ],
    )

    records = FeedbackDataset._load_jsonl(tmp_path)

    # Both event_id-less rows must be dropped, not collapsed into one
    # record keyed by "".
    assert [r.event_id for r in records] == ["e1"]


def test_load_jsonl_skips_malformed_json_line(tmp_path: Path) -> None:
    path = tmp_path / "feedback-001.jsonl"
    with path.open("w") as f:
        f.write(json.dumps(_row(event_id="e1", route_decision="LOCAL_ML")) + "\n")
        f.write("not valid json\n")
        f.write(json.dumps(_row(event_id="e2", route_decision="RULES")) + "\n")

    records = FeedbackDataset._load_jsonl(tmp_path)
    assert sorted(r.event_id for r in records) == ["e1", "e2"]


def test_load_jsonl_skips_unrecognized_route_decision(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "feedback-001.jsonl",
        [
            _row(event_id="e1", route_decision="NOT_A_REAL_ROUTE"),
            _row(event_id="e2", route_decision="LOCAL_ML"),
        ],
    )

    records = FeedbackDataset._load_jsonl(tmp_path)
    assert [r.event_id for r in records] == ["e2"]


def test_load_jsonl_applies_corrections(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "feedback-001.jsonl",
        [
            _row(event_id="e1", route_decision="LLM_NEEDED"),
            _row(
                event_id="e1",
                route_decision="CORRECTION",
                corrected_decision="LOCAL_ML",
            ),
        ],
    )

    records = FeedbackDataset._load_jsonl(tmp_path)
    assert len(records) == 1
    assert records[0].route_decision == "LOCAL_ML"
