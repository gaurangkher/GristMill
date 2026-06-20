"""Tests for gristmill_ml.datasets.augmentation.augment_record.

Regression coverage for a label-mapping bug: augment_record used to derive
the augmented copy's route_decision via
``list({0: "LOCAL_ML", ...})[record.label]``, which converts the dict to a
list of its *keys* (``[0, 1, 2, 3]``) rather than its values, so the
augmented copy ended up with route_decision set to the integer label
instead of the route name string — which then failed to match
ROUTE_LABEL_MAP and silently re-labelled every non-LOCAL_ML record as
LOCAL_ML (label 0).
"""

from __future__ import annotations

import pytest

from gristmill_ml.datasets.augmentation import augment_record
from gristmill_ml.datasets.feedback import ROUTE_LABEL_MAP, FeedbackRecord


def _record(route_decision: str) -> FeedbackRecord:
    rec = FeedbackRecord(
        {
            "event_id": "e1",
            "timestamp_ms": 1000,
            "route_decision": route_decision,
            "confidence": 0.9,
            "estimated_tokens": 10,
            "actual_tokens": 10,
            "could_have_been_local": True,
            "event_source": "http",
            "token_count": 5,
        }
    )
    rec._text = "schedule a meeting tomorrow"
    return rec


@pytest.mark.parametrize("route_decision", list(ROUTE_LABEL_MAP.keys()))
def test_augment_record_preserves_route_decision_and_label(route_decision: str) -> None:
    rec = _record(route_decision)
    augmented = augment_record(rec, factor=3)

    assert len(augmented) == 3
    for aug in augmented:
        assert aug.route_decision == route_decision
        assert aug.label == rec.label
