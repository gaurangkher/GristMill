"""
gristmill_ml.second_brain — Second Brain processing for GristMill.

This package provides the background processing layer for Second Brain mode:

  models.py     — Pydantic v2 schema for SecondBrainNote
  processor.py  — Asyncio-based background processor that enriches hot-tier
                  captures (LLM summarisation, backlink construction, cluster
                  detection, spaced-repetition nudges, conflict flagging).

The TypeScript SlackHopper with secondBrain config handles real-time Slack
interactions (capture / query / bookmark).  This Python layer runs offline,
enriching notes after they have been promoted to the warm tier by the Rust
eviction drainer.
"""

from .models import SecondBrainNote
from .processor import SecondBrainProcessor

__all__ = ["SecondBrainNote", "SecondBrainProcessor"]
