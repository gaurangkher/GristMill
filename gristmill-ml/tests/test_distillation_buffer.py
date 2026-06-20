"""Tests for the training-buffer read/status-transition logic in
gristmill_ml.trainer.distillation.

Covers the SQLite-backed PENDING -> IN_TRAINING -> CONSUMED lifecycle that
gristmill-trainer drives, including the rollback path: a cycle that fails
after marking records IN_TRAINING must put them back to PENDING rather than
leaving them stuck forever (they would never be selected again, since
``_load_pending_records`` only selects ``status = 'PENDING'``).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from gristmill_ml.trainer.distillation import (
    DistillationEngine,
    _load_pending_records,
    _mark_consumed,
    _mark_in_training,
    _mark_pending,
)


def _init_training_db(db_path: Path) -> None:
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE training_records (
            record_id        TEXT    PRIMARY KEY,
            timestamp        TEXT    NOT NULL,
            query_text       TEXT    NOT NULL,
            teacher_response TEXT    NOT NULL,
            grinder_response TEXT,
            confidence_score REAL    NOT NULL,
            domain_tag       TEXT    NOT NULL,
            teacher_logits   BLOB,
            status           TEXT    NOT NULL DEFAULT 'PENDING',
            in_retention     INTEGER NOT NULL DEFAULT 0,
            provider_type    TEXT    NOT NULL,
            teacher_cost_usd REAL    NOT NULL DEFAULT 0.0
        )
        """
    )
    conn.commit()
    conn.close()


def _insert_record(
    db_path: Path,
    record_id: str,
    domain: str = "default",
    status: str = "PENDING",
) -> None:
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        INSERT INTO training_records
            (record_id, timestamp, query_text, teacher_response, confidence_score,
             domain_tag, status, provider_type, teacher_cost_usd)
        VALUES (?, '2026-01-01T00:00:00Z', ?, ?, 0.4, ?, ?, 'local_open_source', 0.0)
        """,
        (record_id, f"query {record_id}", f"response {record_id}", domain, status),
    )
    conn.commit()
    conn.close()


def _status_of(db_path: Path, record_id: str) -> str:
    conn = sqlite3.connect(str(db_path))
    row = conn.execute(
        "SELECT status FROM training_records WHERE record_id = ?", (record_id,)
    ).fetchone()
    conn.close()
    assert row is not None
    return row[0]


@pytest.fixture
def training_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "training_buffer.sqlite"
    _init_training_db(db_path)
    return db_path


def test_load_pending_records_filters_by_domain(training_db: Path) -> None:
    _insert_record(training_db, "rec-a", domain="code")
    _insert_record(training_db, "rec-b", domain="qa")

    code_records = _load_pending_records(training_db, domain="code")
    assert [r["record_id"] for r in code_records] == ["rec-a"]


def test_load_pending_records_excludes_non_pending(training_db: Path) -> None:
    _insert_record(training_db, "rec-a", status="PENDING")
    _insert_record(training_db, "rec-b", status="CONSUMED")
    _insert_record(training_db, "rec-c", status="IN_TRAINING")

    pending = _load_pending_records(training_db)
    assert [r["record_id"] for r in pending] == ["rec-a"]


def test_mark_in_training_then_consumed_transitions(training_db: Path) -> None:
    _insert_record(training_db, "rec-a")
    _mark_in_training(training_db, ["rec-a"])
    assert _status_of(training_db, "rec-a") == "IN_TRAINING"

    _mark_consumed(training_db, ["rec-a"])
    assert _status_of(training_db, "rec-a") == "CONSUMED"


def test_mark_pending_rolls_back_in_training_record(training_db: Path) -> None:
    _insert_record(training_db, "rec-a")
    _mark_in_training(training_db, ["rec-a"])
    assert _status_of(training_db, "rec-a") == "IN_TRAINING"

    _mark_pending(training_db, ["rec-a"])
    assert _status_of(training_db, "rec-a") == "PENDING"

    # And it is selectable again by the next cycle's load.
    pending = _load_pending_records(training_db)
    assert [r["record_id"] for r in pending] == ["rec-a"]


def test_mark_in_training_with_empty_list_is_noop(training_db: Path) -> None:
    _insert_record(training_db, "rec-a")
    _mark_in_training(training_db, [])
    assert _status_of(training_db, "rec-a") == "PENDING"


def test_run_cycle_rolls_back_to_pending_on_training_failure(
    tmp_path: Path, training_db: Path
) -> None:
    """A cycle that marks records IN_TRAINING and then fails during training
    must restore them to PENDING, not leave them stuck forever."""
    _insert_record(training_db, "rec-a")
    _insert_record(training_db, "rec-b")

    engine = DistillationEngine(output_dir=tmp_path / "staging", device="cpu")

    with patch.object(
        DistillationEngine, "_train_lora", side_effect=RuntimeError("boom")
    ):
        result = engine.run_cycle(
            training_db_path=training_db,
            retention_records=[],
            version=1,
        )

    assert result.success is False
    assert _status_of(training_db, "rec-a") == "PENDING"
    assert _status_of(training_db, "rec-b") == "PENDING"

    # Confirms the records are retryable by the next cycle.
    pending = _load_pending_records(training_db)
    assert {r["record_id"] for r in pending} == {"rec-a", "rec-b"}


def test_run_cycle_marks_consumed_on_success(tmp_path: Path, training_db: Path) -> None:
    _insert_record(training_db, "rec-a")

    engine = DistillationEngine(output_dir=tmp_path / "staging", device="cpu")

    with patch.object(
        DistillationEngine,
        "_train_lora",
        return_value=(tmp_path / "adapter", 0.123),
    ):
        result = engine.run_cycle(
            training_db_path=training_db,
            retention_records=[],
            version=1,
        )

    assert result.success is True
    assert _status_of(training_db, "rec-a") == "CONSUMED"
