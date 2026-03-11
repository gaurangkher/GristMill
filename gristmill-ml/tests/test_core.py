"""Smoke tests for gristmill_ml.core stub classes.

These tests verify that the package is importable and that the pure-Python
stub classes work correctly when the native Rust extension is not present
(which is always the case in CI since maturin is not run there).
"""

from __future__ import annotations

import json
import warnings


def test_has_native_is_bool() -> None:
    """HAS_NATIVE is always a bool regardless of whether the extension is built."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from gristmill_ml.core import HAS_NATIVE

    assert isinstance(HAS_NATIVE, bool)


def test_stub_grist_event() -> None:
    """PyGristEvent stub returns consistent values."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from gristmill_ml.core import HAS_NATIVE, PyGristEvent

    if HAS_NATIVE:
        return  # native path tested separately

    evt = PyGristEvent(source="test", payload_json='{"key": "value"}')
    assert isinstance(evt.id, str)
    assert isinstance(evt.timestamp_ms, int)
    data = json.loads(evt.to_json())
    assert data["source"] == "test"
    assert data["payload"]["key"] == "value"


def test_stub_route_decision() -> None:
    """PyRouteDecision stub serialises correctly."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from gristmill_ml.core import HAS_NATIVE, PyRouteDecision

    if HAS_NATIVE:
        return

    decision = PyRouteDecision(route="LOCAL_ML", confidence=0.9)
    data = json.loads(decision.to_json())
    assert data["route"] == "LOCAL_ML"
    assert data["confidence"] == 0.9


def test_stub_memory() -> None:
    """PyMemory stub stores fields correctly."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from gristmill_ml.core import HAS_NATIVE, PyMemory

    if HAS_NATIVE:
        return

    mem = PyMemory(id="abc", content="hello world", tags=["a", "b"], tier="hot")
    assert mem.id == "abc"
    assert mem.content == "hello world"
    assert mem.tags == ["a", "b"]
    assert mem.tier == "hot"
