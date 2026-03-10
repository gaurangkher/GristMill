"""Feedback JSONL dataset loader.

Reads the routing-decision logs written by ``grist-sieve`` from
``~/.gristmill/feedback/feedback-YYYY-MM-DD.jsonl`` and exposes them as a
PyTorch :class:`Dataset` suitable for training :class:`SieveClassifierHead`.

Stable schema (must not change without updating ``grist_sieve::feedback.rs``):

.. code-block:: json

    {
      "event_id": "01HXY...",
      "timestamp_ms": 1700000000000,
      "route_decision": "LOCAL_ML",
      "confidence": 0.937,
      "estimated_tokens": 0,
      "actual_tokens": null,
      "could_have_been_local": null,
      "event_source": "http",
      "token_count": 12
    }
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Optional

import numpy as np

# ── Label / ordinal maps ──────────────────────────────────────────────────────

#: Maps ``route_decision`` strings to integer class indices.
#: Must match ``grist_sieve::classifier::RouteLabel``.
ROUTE_LABEL_MAP: dict[str, int] = {
    "LOCAL_ML": 0,
    "RULES": 1,
    "HYBRID": 2,
    "LLM_NEEDED": 3,
}

#: Maps ``event_source`` strings to the ordinal used as a feature dimension.
#: Must match ``grist_sieve/features.rs`` CHANNEL ordinals (index 385 / 9.0).
CHANNEL_ORDINAL: dict[str, int] = {
    "cli": 0,
    "http": 1,
    "websocket": 2,
    "cron": 3,
    "webhook": 4,
    "mq": 5,
    "fs": 6,
    "python": 7,
    "typescript": 8,
    "internal": 9,
}

# ── Type alias ────────────────────────────────────────────────────────────────

class FeedbackRecord:
    """One row from a feedback JSONL file."""

    __slots__ = (
        "event_id",
        "timestamp_ms",
        "route_decision",
        "confidence",
        "estimated_tokens",
        "actual_tokens",
        "could_have_been_local",
        "event_source",
        "token_count",
        # Derived fields for training
        "label",
    )

    def __init__(self, raw: dict[str, Any]) -> None:
        self.event_id: str = raw["event_id"]
        self.timestamp_ms: int = int(raw["timestamp_ms"])
        self.route_decision: str = raw["route_decision"]
        self.confidence: float = float(raw["confidence"])
        self.estimated_tokens: int = int(raw.get("estimated_tokens", 0))
        self.actual_tokens: Optional[int] = raw.get("actual_tokens")
        self.could_have_been_local: Optional[bool] = raw.get("could_have_been_local")
        self.event_source: str = raw.get("event_source", "internal")
        self.token_count: int = int(raw.get("token_count", 0))
        # Integer class label (0–3)
        self.label: int = ROUTE_LABEL_MAP.get(self.route_decision, 0)

    @property
    def priority(self) -> int:
        """Placeholder priority — feedback logs don't carry this field yet."""
        return 1  # Normal = 1

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"FeedbackRecord(id={self.event_id[:8]}…, "
            f"decision={self.route_decision}, conf={self.confidence:.2f})"
        )


# ── Synthetic cold-start data ─────────────────────────────────────────────────

_SYNTHETIC_EXAMPLES: list[tuple[str, str, int, int]] = [
    # (text, source, priority, expected_label)
    # LOCAL_ML (0)
    ("status", "cli", 1, 0),
    ("ping", "cli", 1, 0),
    ("list files", "cli", 1, 0),
    ("show me today's events", "http", 1, 0),
    ("get weather", "http", 1, 0),
    ("quick note: buy milk", "http", 1, 0),
    ("translate hello to french", "http", 1, 0),
    ("summarise this sentence", "http", 1, 0),
    ("what time is it", "cli", 1, 0),
    ("tag this as urgent", "webhook", 1, 0),
    # RULES (1)
    ("schedule meeting tomorrow at 3pm", "http", 1, 1),
    ("remind me at 09:00", "cron", 1, 1),
    ("run daily backup", "cron", 1, 1),
    ("send digest at 18:00", "cron", 1, 1),
    ("route to team@example.com", "http", 1, 1),
    ("alert if cpu > 90%", "mq", 1, 1),
    ("open pr #42", "webhook", 1, 1),
    ("close issue #17", "webhook", 1, 1),
    ("match regex ^error:", "internal", 1, 1),
    ("apply template: {{greeting}}", "internal", 1, 1),
    # HYBRID (2)
    ("summarise the last 10 support tickets with sentiment", "http", 2, 2),
    ("classify these 50 emails and draft replies", "http", 2, 2),
    ("analyse metrics trends and suggest actions", "mq", 2, 2),
    ("extract entities from this document", "webhook", 2, 2),
    ("generate a weekly report from the data", "http", 2, 2),
    ("compare my calendar against team availability", "http", 1, 2),
    ("rank these bugs by severity", "webhook", 2, 2),
    ("create a briefing from these 5 articles", "http", 1, 2),
    ("proofread and improve this email draft", "http", 1, 2),
    ("localise this content for Spanish speakers", "http", 1, 2),
    # LLM_NEEDED (3)
    ("explain why the authentication service failed intermittently last week", "http", 3, 3),
    ("write a blog post about Rust's ownership model", "http", 1, 3),
    ("debug this complex async race condition", "http", 2, 3),
    ("design a microservices architecture for our platform", "http", 2, 3),
    ("what are the ethical implications of this policy change", "http", 1, 3),
    ("help me negotiate this contract clause", "http", 3, 3),
    ("why did revenue drop 20% last quarter", "http", 2, 3),
    ("write unit tests for this entire module", "webhook", 1, 3),
    ("create a comprehensive project plan with milestones", "http", 2, 3),
    ("explain quantum entanglement to a 10-year-old", "http", 1, 3),
]


def generate_synthetic_records(n: int = 1000) -> list[FeedbackRecord]:
    """Generate ``n`` synthetic feedback records for cold-start training.

    Balances the four classes by cycling through example templates with
    light text variation (word repetition and concatenation).
    """
    records: list[FeedbackRecord] = []
    examples = _SYNTHETIC_EXAMPLES.copy()

    for i in range(n):
        text, source, priority, label = examples[i % len(examples)]
        # Add light variation: duplicate a word, append a suffix
        suffix_choices = ["please", "now", "asap", "in detail", "quickly", ""]
        text_variant = text + (" " + random.choice(suffix_choices)).rstrip()

        raw: dict[str, Any] = {
            "event_id": f"synth-{i:08d}",
            "timestamp_ms": 1_700_000_000_000 + i * 1000,
            "route_decision": list(ROUTE_LABEL_MAP.keys())[label],
            "confidence": 0.9 + random.uniform(-0.1, 0.1),
            "estimated_tokens": 0,
            "actual_tokens": None,
            "could_have_been_local": None,
            "event_source": source,
            "token_count": len(text_variant.split()),
            # Stash derived text for the feature extractor
            "_text": text_variant,
            "_priority": priority,
        }
        rec = FeedbackRecord(raw)
        # Attach the text so FeatureExtractor can use it
        rec._text = text_variant  # type: ignore[attr-defined]
        rec._priority = priority  # type: ignore[attr-defined]
        records.append(rec)

    random.shuffle(records)
    return records


# ── Dataset ───────────────────────────────────────────────────────────────────

class FeedbackDataset:
    """PyTorch-compatible dataset built from GristMill feedback JSONL logs.

    When no feedback files exist (fresh install), :meth:`generate_synthetic`
    is called automatically to produce a minimal cold-start corpus.

    Args:
        feedback_dir: Directory containing ``feedback-YYYY-MM-DD.jsonl`` files.
        split: ``"train"`` or ``"val"``.
        val_fraction: Fraction of records held out for validation.
        min_records: If fewer real records are found, pad with synthetic data.
    """

    def __init__(
        self,
        feedback_dir: Path = Path("~/.gristmill/feedback").expanduser(),
        split: str = "train",
        val_fraction: float = 0.15,
        min_records: int = 200,
    ) -> None:
        self.split = split
        self.feedback_dir = feedback_dir

        records = self._load_jsonl(feedback_dir)

        if len(records) < min_records:
            synthetic = generate_synthetic_records(max(min_records - len(records), 500))
            records = records + synthetic

        # Stratified split
        random.seed(42)
        random.shuffle(records)
        n_val = max(1, int(len(records) * val_fraction))
        if split == "val":
            self.records = records[:n_val]
        else:
            self.records = records[n_val:]

    # ── Loading ───────────────────────────────────────────────────────────────

    @staticmethod
    def _load_jsonl(directory: Path) -> list[FeedbackRecord]:
        """Load and deduplicate all JSONL files in *directory*.

        CORRECTION records are joined back to their parent record to correct
        the label (the Rust sieve writes a CORRECTION when it detects that an
        LLM route could have been handled locally).
        """
        if not directory.exists():
            return []

        raw_records: dict[str, dict[str, Any]] = {}
        corrections: dict[str, str] = {}  # event_id → corrected route_decision

        for path in sorted(directory.glob("feedback-*.jsonl")):
            with path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row: dict[str, Any] = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    eid = row.get("event_id", "")
                    decision = row.get("route_decision", "")

                    if decision == "CORRECTION":
                        corrections[eid] = row.get("corrected_decision", "LOCAL_ML")
                    elif decision in ROUTE_LABEL_MAP:
                        raw_records[eid] = row

        # Apply corrections
        for eid, corrected in corrections.items():
            if eid in raw_records:
                raw_records[eid]["route_decision"] = corrected

        return [FeedbackRecord(r) for r in raw_records.values()]

    # ── PyTorch Dataset API ───────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec = self.records[idx]
        # Return raw fields; FeatureExtractor handles vectorisation.
        text: str = getattr(rec, "_text", "")  # set by synthetic generator
        return {
            "text": text,
            "source": rec.event_source,
            "priority": getattr(rec, "_priority", rec.priority),
            "token_count": rec.token_count,
            "label": rec.label,
            "confidence": rec.confidence,
        }

    def class_counts(self) -> dict[str, int]:
        """Return per-class sample counts (useful for weighted sampling)."""
        counts: dict[str, int] = {k: 0 for k in ROUTE_LABEL_MAP}
        label_to_name = {v: k for k, v in ROUTE_LABEL_MAP.items()}
        for rec in self.records:
            name = label_to_name.get(rec.label, "LOCAL_ML")
            counts[name] += 1
        return counts

    def class_weights(self) -> "np.ndarray":
        """Inverse-frequency class weights for :class:`torch.nn.CrossEntropyLoss`."""
        counts = self.class_counts()
        total = sum(counts.values()) or 1
        n_classes = len(ROUTE_LABEL_MAP)
        weights = np.array(
            [total / (n_classes * max(counts.get(k, 1), 1)) for k in ROUTE_LABEL_MAP],
            dtype=np.float32,
        )
        return weights / weights.sum() * n_classes  # normalise to n_classes
