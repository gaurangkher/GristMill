"""
Bulk seed GristMill's training buffer with public reasoning examples.

Usage:
    python seed_reasoning.py                     # 1,200 records, default DB path
    python seed_reasoning.py --count 2000
    python seed_reasoning.py --db-path /data/gristmill/db/training_buffer.sqlite

Schema (training_records in grist-sieve/src/training_buffer.rs):
    record_id        TEXT PRIMARY KEY
    timestamp        TEXT NOT NULL
    query_text       TEXT NOT NULL
    teacher_response TEXT NOT NULL
    grinder_response TEXT
    confidence_score REAL NOT NULL
    domain_tag       TEXT NOT NULL
    teacher_logits   BLOB
    status           TEXT NOT NULL DEFAULT 'PENDING'
    in_retention     INTEGER NOT NULL DEFAULT 0
    provider_type    TEXT NOT NULL

Sources pulled (in order until count is met):
    1. teknium/OpenHermes-2.5         — chat-format reasoning Q&A
    2. meta-math/MetaMathQA           — mathematical reasoning
    3. Muennighoff/flan               — FLAN instruction tuning
"""

from __future__ import annotations

import argparse
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DB_PATH = Path.home() / ".gristmill" / "db" / "training_buffer.sqlite"
FALLBACK_DB_PATH = Path("/data/gristmill/db/training_buffer.sqlite")
DEFAULT_COUNT = 1_200
DOMAIN_TAG = "reasoning"
CONFIDENCE_SCORE = 0.95
PROVIDER_TYPE = "local_open_source"


# ── Dataset loaders ────────────────────────────────────────────────────────────


def _load_openhermes(limit: int) -> list[tuple[str, str]]:
    """Pull reasoning-flavoured rows from OpenHermes-2.5."""
    from datasets import load_dataset

    ds = load_dataset(
        "teknium/OpenHermes-2.5",
        split="train",
        streaming=True,
        trust_remote_code=False,
    )
    results: list[tuple[str, str]] = []
    reasoning_keywords = (
        "reason",
        "step",
        "explain",
        "solve",
        "calculate",
        "proof",
        "derive",
        "logic",
        "think",
        "why",
        "how",
        "because",
        "therefore",
        "analyze",
    )
    for row in ds:
        if len(results) >= limit:
            break
        convs = row.get("conversations") or []
        if len(convs) < 2:
            continue
        human = next((c["value"] for c in convs if c.get("from") == "human"), None)
        gpt = next((c["value"] for c in convs if c.get("from") == "gpt"), None)
        if not human or not gpt:
            continue
        combined = (human + gpt).lower()
        if any(kw in combined for kw in reasoning_keywords):
            results.append((human.strip(), gpt.strip()))
    return results


def _load_metamath(limit: int) -> list[tuple[str, str]]:
    """Pull from MetaMathQA (mathematical reasoning)."""
    from datasets import load_dataset

    ds = load_dataset(
        "meta-math/MetaMathQA",
        split="train",
        streaming=True,
        trust_remote_code=False,
    )
    results: list[tuple[str, str]] = []
    for row in ds:
        if len(results) >= limit:
            break
        q = (row.get("query") or "").strip()
        a = (row.get("response") or "").strip()
        if q and a:
            results.append((q, a))
    return results


def _load_flan(limit: int) -> list[tuple[str, str]]:
    """Pull from the FLAN collection (instruction-tuned reasoning tasks)."""
    from datasets import load_dataset

    ds = load_dataset(
        "Muennighoff/flan",
        split="train",
        streaming=True,
        trust_remote_code=False,
    )
    results: list[tuple[str, str]] = []
    reasoning_tasks = {
        "cos_e",
        "cosmos_qa",
        "grade_school_math",
        "math_qa",
        "strategy_qa",
        "aqua_rat",
        "commonsense_qa",
    }
    for row in ds:
        if len(results) >= limit:
            break
        task = (row.get("task") or "").lower()
        inp = (row.get("inputs") or "").strip()
        tgt = (row.get("targets") or "").strip()
        if inp and tgt and any(t in task for t in reasoning_tasks):
            results.append((inp, tgt))
    return results


# ── DB helpers ─────────────────────────────────────────────────────────────────


def _open_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS training_records (
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
            provider_type    TEXT    NOT NULL
        )
        """
    )
    conn.commit()


def _insert_batch(
    conn: sqlite3.Connection,
    pairs: list[tuple[str, str]],
) -> int:
    now = datetime.now(tz=timezone.utc).isoformat()
    rows = [
        (
            str(uuid.uuid4()),
            now,
            query,
            response,
            None,           # grinder_response
            CONFIDENCE_SCORE,
            DOMAIN_TAG,
            None,           # teacher_logits
            "PENDING",
            0,
            PROVIDER_TYPE,
        )
        for query, response in pairs
    ]
    cursor = conn.executemany(
        """
        INSERT OR IGNORE INTO training_records (
            record_id, timestamp, query_text, teacher_response,
            grinder_response, confidence_score, domain_tag,
            teacher_logits, status, in_retention, provider_type
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    return cursor.rowcount


def _count_pending(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        "SELECT COUNT(*) FROM training_records WHERE status = 'PENDING' AND domain_tag = ?",
        (DOMAIN_TAG,),
    ).fetchone()
    return row[0] if row else 0


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed GristMill training buffer — reasoning domain")
    parser.add_argument(
        "--count",
        type=int,
        default=DEFAULT_COUNT,
        help=f"Target number of records to insert (default: {DEFAULT_COUNT})",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help="Path to training_buffer.sqlite (default: ~/.gristmill/db/training_buffer.sqlite)",
    )
    args = parser.parse_args()

    db_path: Path = args.db_path or (
        DEFAULT_DB_PATH if DEFAULT_DB_PATH.parent.exists() or not FALLBACK_DB_PATH.parent.exists()
        else FALLBACK_DB_PATH
    )
    target: int = args.count

    print(f"DB path : {db_path}")
    print(f"Target  : {target} records  (domain={DOMAIN_TAG})")
    print()

    conn = _open_db(db_path)
    _ensure_table(conn)

    # ── Gather pairs from datasets until target is met ─────────────────────────
    try:
        from tqdm import tqdm
    except ImportError:
        # Fallback: no progress bar
        def tqdm(iterable=None, **kwargs):  # type: ignore[misc]
            return iterable if iterable is not None else (lambda x: x)

    loaders = [
        ("OpenHermes-2.5", _load_openhermes),
        ("MetaMathQA", _load_metamath),
        ("FLAN", _load_flan),
    ]

    all_pairs: list[tuple[str, str]] = []
    needed = target

    for name, loader in loaders:
        if needed <= 0:
            break
        print(f"  Fetching up to {needed} rows from {name} …")
        try:
            pairs = loader(needed)
        except Exception as exc:
            print(f"  [warn] {name} failed: {exc} — skipping")
            continue
        print(f"  → {len(pairs)} rows fetched")
        all_pairs.extend(pairs)
        needed -= len(pairs)

    if not all_pairs:
        print("No records fetched. Check your internet connection and dataset availability.")
        conn.close()
        return

    # ── Insert with progress bar ───────────────────────────────────────────────
    CHUNK = 200
    total_inserted = 0

    chunks = [all_pairs[i : i + CHUNK] for i in range(0, len(all_pairs), CHUNK)]
    with tqdm(total=len(all_pairs), unit="rec", desc="Inserting") as bar:
        for chunk in chunks:
            inserted = _insert_batch(conn, chunk)
            total_inserted += inserted
            bar.update(len(chunk))

    pending_total = _count_pending(conn)
    conn.close()

    print()
    print(f"Inserted : {total_inserted} new records")
    print(f"Pending  : {pending_total} total PENDING records for domain='{DOMAIN_TAG}'")
    if pending_total >= 1_000:
        print("  ✓ Above 1,000-record trigger threshold — distillation cycle ready.")
    else:
        remaining = 1_000 - pending_total
        print(f"  ↑ Need {remaining} more records to reach the 1,000-record trigger threshold.")


if __name__ == "__main__":
    main()
