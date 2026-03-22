"""RetentionBuffer — curated fixed-size set of diverse training examples.

Included in every distillation cycle to prevent catastrophic forgetting
(Section 4.5.3 and 5.4 of the spec).

Curation strategy:
  1. Domain diversity  — balanced sampling across all domain_tag categories.
  2. Difficulty stratification — examples spanning full confidence score range.
  3. Temporal spread — records sampled across full training history.
  4. Quality filter — length-normalised heuristic (longer teacher responses
     tend to be more informative; hard floor of 20 characters).

Buffer stored in SQLite at /gristmill/db/retention_buffer.sqlite.
Written and managed exclusively by gristmill-trainer.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_DEFAULT_DB = Path("/gristmill/db/retention_buffer.sqlite")
_FALLBACK_DB = Path.home() / ".gristmill" / "db" / "retention_buffer.sqlite"

DEFAULT_MAX_SIZE = 2_000
DOMAIN_TAGS = ("code", "writing", "reasoning", "qa", "creative", "other")
STRATA = 4          # Confidence score buckets (0–0.25, 0.25–0.5, 0.5–0.75, 0.75–1.0)
MIN_RESPONSE_LEN = 20


@dataclass
class RetentionRecord:
    record_id: str
    timestamp: str
    query_text: str
    teacher_response: str
    grinder_response: Optional[str]
    confidence_score: float
    domain_tag: str
    quality_score: float
    selected_at: str


class RetentionBuffer:
    """SQLite-backed curated retention set.

    Thread-safety: SQLite connections are per-instance; use a single instance
    per process (APScheduler runs training in a thread pool — pass the same
    RetentionBuffer instance).
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        if db_path is None:
            db_path = _DEFAULT_DB if _DEFAULT_DB.parent.exists() else _FALLBACK_DB
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self._conn = self._open()

    def _open(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS retention_records (
                record_id       TEXT PRIMARY KEY,
                timestamp       TEXT NOT NULL,
                query_text      TEXT NOT NULL,
                teacher_response TEXT NOT NULL,
                grinder_response TEXT,
                confidence_score REAL NOT NULL,
                domain_tag      TEXT NOT NULL,
                quality_score   REAL NOT NULL DEFAULT 0.0,
                selected_at     TEXT NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_domain ON retention_records(domain_tag)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_confidence ON retention_records(confidence_score)"
        )
        conn.commit()
        return conn

    # ── Curation ──────────────────────────────────────────────────────────────

    def curate(
        self,
        training_db_path: Path,
        max_size: int = DEFAULT_MAX_SIZE,
    ) -> int:
        """Re-curate the retention buffer from *training_db_path*.

        Reads CONSUMED + IN_TRAINING records from the training buffer,
        applies the four curation criteria, and replaces the current
        retention buffer contents.

        Returns the number of records inserted.
        """
        candidates = self._load_candidates(training_db_path)
        if not candidates:
            logger.info("RetentionBuffer.curate: no candidates found in training buffer")
            return 0

        selected = self._select(candidates, max_size)
        now = datetime.now(timezone.utc).isoformat()

        self._conn.execute("DELETE FROM retention_records")
        self._conn.executemany(
            """
            INSERT INTO retention_records
                (record_id, timestamp, query_text, teacher_response,
                 grinder_response, confidence_score, domain_tag, quality_score, selected_at)
            VALUES (?,?,?,?,?,?,?,?,?)
            """,
            [
                (
                    r["record_id"],
                    r["timestamp"],
                    r["query_text"],
                    r["teacher_response"],
                    r.get("grinder_response"),
                    r["confidence_score"],
                    r["domain_tag"],
                    r["quality_score"],
                    now,
                )
                for r in selected
            ],
        )
        self._conn.commit()
        logger.info("RetentionBuffer: curated %d records", len(selected))
        return len(selected)

    def _load_candidates(self, training_db_path: Path) -> list[dict]:
        """Load all usable records from the training buffer."""
        try:
            src = sqlite3.connect(f"file:{training_db_path}?mode=ro", uri=True)
            src.row_factory = sqlite3.Row
            rows = src.execute(
                """
                SELECT record_id, timestamp, query_text, teacher_response,
                       grinder_response, confidence_score, domain_tag
                FROM   training_records
                WHERE  status IN ('CONSUMED', 'IN_TRAINING')
                  AND  LENGTH(teacher_response) >= ?
                ORDER  BY timestamp ASC
                """,
                (MIN_RESPONSE_LEN,),
            ).fetchall()
            src.close()
        except sqlite3.Error as exc:
            logger.error("Failed to read training buffer: %s", exc)
            return []

        candidates = []
        for row in rows:
            r = dict(row)
            r["quality_score"] = _quality_score(r["teacher_response"])
            candidates.append(r)
        return candidates

    def _select(self, candidates: list[dict], max_size: int) -> list[dict]:
        """Apply curation criteria and return up to *max_size* records."""
        per_domain = max_size // len(DOMAIN_TAGS)

        # Group by domain
        by_domain: dict[str, list[dict]] = {tag: [] for tag in DOMAIN_TAGS}
        for r in candidates:
            tag = r["domain_tag"] if r["domain_tag"] in DOMAIN_TAGS else "other"
            by_domain[tag].append(r)

        selected: list[dict] = []
        for tag, records in by_domain.items():
            if not records:
                continue
            picked = _stratified_sample(records, per_domain)
            selected.extend(picked)

        # If we're short (some domains had few records), fill from leftovers
        if len(selected) < max_size:
            selected_ids = {r["record_id"] for r in selected}
            leftovers = [r for r in candidates if r["record_id"] not in selected_ids]
            # Sort leftovers by quality score descending for best fill
            leftovers.sort(key=lambda r: r["quality_score"], reverse=True)
            selected.extend(leftovers[: max_size - len(selected)])

        return selected[:max_size]

    # ── Read access ───────────────────────────────────────────────────────────

    def get_all(self) -> list[RetentionRecord]:
        """Return all records currently in the retention buffer."""
        self._conn.row_factory = sqlite3.Row
        rows = self._conn.execute("SELECT * FROM retention_records").fetchall()
        return [
            RetentionRecord(
                record_id=r["record_id"],
                timestamp=r["timestamp"],
                query_text=r["query_text"],
                teacher_response=r["teacher_response"],
                grinder_response=r["grinder_response"],
                confidence_score=r["confidence_score"],
                domain_tag=r["domain_tag"],
                quality_score=r["quality_score"],
                selected_at=r["selected_at"],
            )
            for r in rows
        ]

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM retention_records").fetchone()[0]

    def close(self) -> None:
        self._conn.close()


# ── Helpers ───────────────────────────────────────────────────────────────────


def _quality_score(teacher_response: str) -> float:
    """Heuristic quality score in [0, 1].

    Uses length-normalised scoring: responses between 50–500 chars score
    highest; very short or very long responses score lower.
    """
    n = len(teacher_response)
    if n < MIN_RESPONSE_LEN:
        return 0.0
    if n <= 500:
        return min(1.0, n / 500)
    # Penalise very long responses (may be over-verbose / low-density)
    return max(0.5, 1.0 - (n - 500) / 5000)


def _stratified_sample(records: list[dict], n: int) -> list[dict]:
    """Sample *n* records stratified by confidence score across *STRATA* buckets.

    Also ensures temporal spread by sorting each stratum by timestamp before
    uniform spacing.
    """
    if len(records) <= n:
        return list(records)

    bucket_size = n // STRATA
    remainder = n - bucket_size * STRATA

    buckets: list[list[dict]] = [[] for _ in range(STRATA)]
    for r in records:
        idx = min(int(r["confidence_score"] * STRATA), STRATA - 1)
        buckets[idx].append(r)

    result: list[dict] = []
    for i, bucket in enumerate(buckets):
        take = bucket_size + (1 if i < remainder else 0)
        if not bucket:
            continue
        # Sort by timestamp for temporal spread, then uniform spacing
        bucket.sort(key=lambda r: r["timestamp"])
        if len(bucket) <= take:
            result.extend(bucket)
        else:
            step = len(bucket) / take
            result.extend(bucket[int(j * step)] for j in range(take))

    return result
