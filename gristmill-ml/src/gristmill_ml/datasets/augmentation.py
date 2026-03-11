"""Text augmentation utilities for GristMill ML training.

Light augmentations that don't require external models (no back-translation
server needed) so the training loop works fully offline.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .feedback import FeedbackRecord

# ── Word-level synonym table (domain-specific) ────────────────────────────────

_SYNONYMS: dict[str, list[str]] = {
    "schedule": ["plan", "arrange", "book", "set up"],
    "meeting": ["call", "session", "sync", "standup"],
    "remind": ["alert", "notify", "ping"],
    "summarise": ["summarize", "recap", "condense", "brief"],
    "analyse": ["analyze", "review", "inspect", "examine"],
    "error": ["failure", "fault", "issue", "bug"],
    "create": ["make", "build", "generate", "produce"],
    "send": ["dispatch", "forward", "deliver", "push"],
    "get": ["fetch", "retrieve", "obtain", "pull"],
    "show": ["display", "list", "print", "output"],
    "file": ["document", "record", "asset", "artifact"],
    "report": ["summary", "briefing", "rundown", "overview"],
    "debug": ["troubleshoot", "diagnose", "investigate", "fix"],
    "explain": ["describe", "clarify", "outline", "elaborate on"],
    "write": ["draft", "compose", "author", "produce"],
}


def synonym_replace(text: str, n: int = 1) -> str:
    """Replace up to *n* words in *text* with a domain synonym.

    Only replaces words that appear in the synonym table, ensuring the
    augmented text remains semantically valid.
    """
    words = text.split()
    replaceable = [(i, w) for i, w in enumerate(words) if w.lower() in _SYNONYMS]
    if not replaceable:
        return text

    random.shuffle(replaceable)
    for i, word in replaceable[:n]:
        synonyms = [s for s in _SYNONYMS[word.lower()] if s != word.lower()]
        if synonyms:
            replacement = random.choice(synonyms)
            # Preserve original capitalisation
            if word[0].isupper():
                replacement = replacement.capitalize()
            words[i] = replacement

    return " ".join(words)


def random_deletion(text: str, p: float = 0.1) -> str:
    """Randomly delete each word with probability *p*."""
    words = text.split()
    if len(words) <= 2:
        return text
    result = [w for w in words if random.random() > p]
    return " ".join(result) if result else text


def random_insertion(text: str, n: int = 1) -> str:
    """Insert a random synonym next to an existing word *n* times."""
    words = text.split()
    synonymable = [w for w in words if w.lower() in _SYNONYMS]
    if not synonymable:
        return text
    for _ in range(n):
        word = random.choice(synonymable)
        synonym = random.choice(_SYNONYMS[word.lower()])
        pos = random.randint(0, len(words))
        words.insert(pos, synonym)
    return " ".join(words)


def case_variation(text: str) -> str:
    """Randomly choose lower, title, or UPPER case for the entire string."""
    choice = random.choice(["lower", "title"])
    if choice == "lower":
        return text.lower()
    if choice == "title":
        return text.title()
    return text


def add_typo(text: str) -> str:
    """Swap two adjacent characters at a random position (simulates typo)."""
    if len(text) < 4:
        return text
    # Only swap within a word, not at word boundaries
    positions = [i for i in range(1, len(text) - 1) if text[i] != " " and text[i - 1] != " "]
    if not positions:
        return text
    pos = random.choice(positions)
    chars = list(text)
    chars[pos], chars[pos - 1] = chars[pos - 1], chars[pos]
    return "".join(chars)


# ── Dataset-level augmentation ────────────────────────────────────────────────


def augment_record(
    record: "FeedbackRecord",
    factor: int = 3,
) -> "list[FeedbackRecord]":
    """Generate *factor* augmented copies of *record*.

    Each copy has a slightly varied text obtained by randomly applying one of:
    synonym replacement, random deletion, random insertion, or case variation.
    The label and metadata are inherited from the original.
    """
    from .feedback import FeedbackRecord as FR  # local import to avoid circular

    augmented: list[FR] = []
    text = getattr(record, "_text", "") or record.event_source

    augmenters = [
        lambda t: synonym_replace(t, n=1),
        lambda t: synonym_replace(t, n=2),
        lambda t: random_deletion(t, p=0.1),
        lambda t: random_insertion(t, n=1),
        lambda t: case_variation(t),
        lambda t: add_typo(t),
    ]

    for i in range(factor):
        fn = augmenters[i % len(augmenters)]
        new_text = fn(text)

        raw = {
            "event_id": f"{record.event_id}-aug{i}",
            "timestamp_ms": record.timestamp_ms + i,
            "route_decision": list({0: "LOCAL_ML", 1: "RULES", 2: "HYBRID", 3: "LLM_NEEDED"})[
                record.label
            ],
            "confidence": record.confidence,
            "estimated_tokens": record.estimated_tokens,
            "actual_tokens": record.actual_tokens,
            "could_have_been_local": record.could_have_been_local,
            "event_source": record.event_source,
            "token_count": len(new_text.split()),
        }
        aug = FR(raw)
        aug._text = new_text  # type: ignore[attr-defined]
        aug._priority = getattr(record, "_priority", 1)  # type: ignore[attr-defined]
        augmented.append(aug)

    return augmented


def augment_dataset(
    records: "list[FeedbackRecord]",
    factor: int = 3,
    target_per_class: int = 500,
) -> "list[FeedbackRecord]":
    """Balance and augment *records* to approximately *target_per_class* per label.

    Minority classes are oversampled; majority classes are left as-is.
    Returns the original records plus new augmented records.
    """
    from .feedback import ROUTE_LABEL_MAP

    # Count existing records per class
    label_to_records: dict[int, list] = {v: [] for v in ROUTE_LABEL_MAP.values()}
    for rec in records:
        label_to_records[rec.label].append(rec)

    result = list(records)

    for label, class_records in label_to_records.items():
        current = len(class_records)
        needed = max(0, target_per_class - current)
        if needed == 0 or not class_records:
            continue

        # Generate augmented records by cycling through the class
        for i in range(needed):
            source = class_records[i % len(class_records)]
            aug = augment_record(source, factor=1)
            result.extend(aug)

    random.shuffle(result)
    return result
