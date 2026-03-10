"""Re-export of the PyO3-compiled ``gristmill_core`` native extension.

If the Rust wheel has been built (via ``maturin develop --features python``),
the real classes are imported.  Otherwise a minimal pure-Python stub is
provided so that training and export code can be imported without a compiled
Rust binary.

Architecture note (CLAUDE.md): do not import ``gristmill_core`` directly from
application code — always go through this module so the fallback path works.
"""

from __future__ import annotations

import json
import warnings
from typing import Any, Optional

HAS_NATIVE: bool = False

try:
    from gristmill_core import (  # type: ignore[import]
        PyGristMill,
        PyGristEvent,
        PyRouteDecision,
        PyMemory,
    )

    HAS_NATIVE = True

except ImportError:
    warnings.warn(
        "gristmill_core native extension not found.  "
        "Run 'maturin develop --features python' in gristmill-core/crates/grist-ffi "
        "to build it.  Stub classes are active — Rust runtime calls will fail.",
        stacklevel=2,
    )

    class PyGristMill:  # type: ignore[no-redef]
        """Stub — build the Rust extension for real functionality."""

        def __init__(self, config_path: Optional[str] = None) -> None:
            raise RuntimeError(
                "gristmill_core native extension is not installed.  "
                "Build it with: maturin develop --features python"
            )

    class PyGristEvent:  # type: ignore[no-redef]
        """Stub event class."""

        def __init__(self, source: str = "internal", payload_json: str = "{}") -> None:
            self._source = source
            self._payload = json.loads(payload_json)

        def to_json(self) -> str:
            return json.dumps({"source": self._source, "payload": self._payload})

        @property
        def id(self) -> str:
            return "00000000000000000000000000"

        @property
        def timestamp_ms(self) -> int:
            import time
            return int(time.time() * 1000)

    class PyRouteDecision:  # type: ignore[no-redef]
        """Stub routing decision."""

        def __init__(
            self,
            route: str = "LOCAL_ML",
            confidence: float = 1.0,
            model_id: Optional[str] = None,
            reason: Optional[str] = None,
            estimated_tokens: Optional[int] = None,
        ) -> None:
            self.route = route
            self.confidence = confidence
            self.model_id = model_id
            self.reason = reason
            self.estimated_tokens = estimated_tokens

        def to_json(self) -> str:
            return json.dumps({
                "route": self.route,
                "confidence": self.confidence,
                "model_id": self.model_id,
                "reason": self.reason,
                "estimated_tokens": self.estimated_tokens,
            })

    class PyMemory:  # type: ignore[no-redef]
        """Stub memory class."""

        def __init__(self, id: str = "", content: str = "", tags: list[str] | None = None,
                     created_at_ms: int = 0, tier: str = "hot") -> None:
            self.id = id
            self.content = content
            self.tags = tags or []
            self.created_at_ms = created_at_ms
            self.tier = tier


__all__ = ["PyGristMill", "PyGristEvent", "PyRouteDecision", "PyMemory", "HAS_NATIVE"]
