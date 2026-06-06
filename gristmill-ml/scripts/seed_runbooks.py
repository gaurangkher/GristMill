"""
Seed GristMill's training buffer with runbook Q&A training data.

Layer 1 — Public foundation datasets (teach context-grounded extraction):
    • rajpurkar/squad_v2              — 20K samples
    • HuggingFaceH4/stack-exchange-preferences — 5K ops-filtered samples
    • ibm/tech_qa                     — full ~1,400 samples
    • wikihow/all                     — 5K how-to procedure samples

Layer 2 — Domain-specific runbook pairs (optional):
    Pass --runbooks-dir to a directory of Markdown/plain-text runbooks.
    The script chunks each file and uses the teacher LLM to generate Q&A pairs.

Usage:
    # Foundation datasets only (no local runbooks):
    python scripts/seed_runbooks.py

    # Foundation + local runbooks:
    python scripts/seed_runbooks.py --runbooks-dir /path/to/runbooks/

    # Custom record count and DB path:
    python scripts/seed_runbooks.py --count 5000 --db-path /data/gristmill/db/training_buffer.sqlite
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import hashlib
import uuid
from datetime import datetime, timezone
from pathlib import Path

# Allow running as a script without installing the package.
_HERE = Path(__file__).resolve().parent.parent  # gristmill-ml/
if str(_HERE / "src") not in sys.path:
    sys.path.insert(0, str(_HERE / "src"))

DEFAULT_DB_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "gristmill-data"
    / "db"
    / "training_buffer.sqlite"
)
DOMAIN_TAG = "runbooks"
FOUNDATION_TAG = "runbooks_foundation"
CONFIDENCE_SCORE = 0.95
PROVIDER_TYPE = "local_open_source"

# Target record counts per source
SQUAD_COUNT = 20_000
SE_COUNT = 5_000
WIKIHOW_COUNT = 5_000

OPS_TAGS = {"server", "linux", "bash", "networking", "deployment", "docker", "nginx"}

# Chunk parameters for local runbook files
CHUNK_TOKENS = 512
CHUNK_OVERLAP = 64
QA_PAIRS_PER_CHUNK = 3


# ── Dataset loaders ────────────────────────────────────────────────────────────


def _load_squad(limit: int) -> list[tuple[str, str]]:
    """SQuAD 2.0 — context-grounded extraction (includes unanswerable questions)."""
    from datasets import load_dataset

    ds = load_dataset("rajpurkar/squad_v2", split="train", streaming=True)
    results: list[tuple[str, str]] = []
    for row in ds:
        if len(results) >= limit:
            break
        context = (row.get("context") or "").strip()
        question = (row.get("question") or "").strip()
        answers = row.get("answers", {})
        texts = answers.get("text", [])
        if not context or not question:
            continue
        if texts:
            answer = texts[0].strip()
        else:
            answer = "The answer is not found in the provided context."
        query = f"[RUNBOOK CONTEXT]\n{context}\n\nQuestion: {question}"
        results.append((query, answer))
    return results


def _load_stack_exchange(limit: int) -> list[tuple[str, str]]:
    """StackExchange — operational Q&A filtered to ops tags."""
    from datasets import load_dataset

    ds = load_dataset(
        "HuggingFaceH4/stack-exchange-preferences",
        split="train",
        streaming=True,
        trust_remote_code=False,
    )
    results: list[tuple[str, str]] = []
    for row in ds:
        if len(results) >= limit:
            break
        tags_raw = row.get("tags") or []
        if isinstance(tags_raw, str):
            tags = {t.strip().lower() for t in tags_raw.split(",")}
        else:
            tags = {str(t).strip().lower() for t in tags_raw}
        if not tags.intersection(OPS_TAGS):
            continue
        question = (row.get("question") or "").strip()
        # Prefer highest-scored answer
        answers = row.get("answers") or []
        if not answers:
            continue
        if isinstance(answers[0], dict):
            best = max(answers, key=lambda a: a.get("pm_score", 0))
            answer = (best.get("text") or "").strip()
        else:
            answer = str(answers[0]).strip()
        if question and answer:
            results.append((question, answer))
    return results


def _load_techqa() -> list[tuple[str, str]]:
    """IBM TechQA — full dataset (~1,400 structured-doc Q&A pairs)."""
    from datasets import load_dataset

    try:
        ds = load_dataset("ibm/tech_qa", split="train", trust_remote_code=False)
    except Exception:
        # Some HF versions use a different split name
        ds = load_dataset("ibm/tech_qa", trust_remote_code=False)
        ds = ds[list(ds.keys())[0]]

    results: list[tuple[str, str]] = []
    for row in ds:
        context = (row.get("doc_text") or row.get("context") or "").strip()
        question = (row.get("question") or "").strip()
        answer = (row.get("answer") or row.get("answer_text") or "").strip()
        if not question or not answer:
            continue
        if context:
            query = f"[RUNBOOK CONTEXT]\n{context}\n\nQuestion: {question}"
        else:
            query = question
        results.append((query, answer))
    return results


def _load_wikihow(limit: int) -> list[tuple[str, str]]:
    """WikiHow — step-by-step procedure articles as a runbook proxy."""
    from datasets import load_dataset

    ds = load_dataset("wikihow/all", split="train", streaming=True, trust_remote_code=False)
    results: list[tuple[str, str]] = []
    for row in ds:
        if len(results) >= limit:
            break
        title = (row.get("title") or "").strip()
        text = (row.get("text") or "").strip()
        headline = (row.get("headline") or "").strip()
        if not title or not text:
            continue
        question = f"How do I {title.lower().rstrip('.')}?"
        answer = headline if headline else text[:500]
        results.append((question, answer))
    return results


# ── Local runbook chunker ──────────────────────────────────────────────────────


def _word_chunks(text: str, size: int, overlap: int) -> list[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + size])
        chunks.append(chunk)
        i += size - overlap
    return chunks


def _generate_qa_pairs_from_chunk(
    chunk: str, teacher_fn, n: int = QA_PAIRS_PER_CHUNK
) -> list[tuple[str, str]]:
    """Ask the teacher LLM to produce n Q&A pairs for the given runbook chunk."""
    prompt = (
        f"You are given a runbook excerpt. Generate exactly {n} question-answer pairs "
        f"grounded in the text below. Each answer must be a short, precise extract from "
        f"the text (a command, threshold, procedure step, or factual value). "
        f"Respond with one pair per line in the format:\n"
        f"Q: <question>\nA: <answer>\n\n"
        f"Runbook excerpt:\n{chunk}"
    )
    try:
        raw = teacher_fn(prompt)
    except Exception as exc:
        print(f"  [warn] Teacher LLM call failed: {exc}")
        return []

    pairs: list[tuple[str, str]] = []
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    q = None
    for line in lines:
        if line.startswith("Q:"):
            q = line[2:].strip()
        elif line.startswith("A:") and q:
            a = line[2:].strip()
            if q and a:
                query = f"[RUNBOOK CONTEXT]\n{chunk}\n\nQuestion: {q}"
                pairs.append((query, a))
                q = None
    return pairs[:n]


def _load_local_runbooks(
    runbooks_dir: Path, teacher_fn, pairs_per_chunk: int = QA_PAIRS_PER_CHUNK
) -> list[tuple[str, str]]:
    """Chunk Markdown/text files in runbooks_dir and generate Q&A pairs per chunk."""
    all_pairs: list[tuple[str, str]] = []
    files = list(runbooks_dir.glob("**/*.md")) + list(runbooks_dir.glob("**/*.txt"))
    if not files:
        print(f"  [warn] No .md or .txt files found in {runbooks_dir}")
        return []

    for f in files:
        print(f"  Processing {f.name} …")
        text = f.read_text(errors="ignore").strip()
        if not text:
            continue
        chunks = _word_chunks(text, CHUNK_TOKENS, CHUNK_OVERLAP)
        for chunk in chunks:
            pairs = _generate_qa_pairs_from_chunk(chunk, teacher_fn, n=pairs_per_chunk)
            all_pairs.extend(pairs)

    return all_pairs


def _resolve_anthropic_key() -> str | None:
    """Return the Anthropic API key from env var or config.yaml."""
    import os
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    try:
        from gristmill_ml.config import load_config
        cfg = load_config()
        key = (cfg.get("hammer") or {}).get("providers", {}).get("anthropic", {}).get("api_key")
        if key:
            return key
    except Exception:
        pass
    return None


def _build_teacher_fn():
    """Return a callable that sends a prompt to the configured teacher LLM."""
    # Resolve the Anthropic key first — independent of whether the package is installed.
    api_key = _resolve_anthropic_key()

    if api_key:
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=api_key)

            def _call_anthropic(prompt: str) -> str:
                msg = client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=512,
                    messages=[{"role": "user", "content": prompt}],
                )
                return msg.content[0].text

            return _call_anthropic
        except ImportError:
            raise RuntimeError(
                "Anthropic API key found in config but the `anthropic` package is not installed. "
                "Run: pip install anthropic"
            )

    # No Anthropic key — try ollama as fallback.
    import subprocess
    try:
        probe = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if probe.returncode == 0:
            def _call_ollama(prompt: str) -> str:
                result = subprocess.run(
                    ["ollama", "run", "llama3.1:8b", prompt],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                return result.stdout.strip()

            return _call_ollama
    except Exception:
        pass

    raise RuntimeError(
        "No teacher LLM available. "
        "Either add your Anthropic key to config.yaml under hammer.providers.anthropic.api_key "
        "and run: pip install anthropic  — or install ollama with llama3.1:8b."
    )


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


def _content_uuid(query: str, domain_tag: str) -> str:
    """Deterministic UUID derived from content — same (query, domain) always yields the same ID.

    This makes INSERT OR IGNORE on the PRIMARY KEY a true content-dedup: re-running the
    seeder never inserts duplicates even without a separate unique index on the data columns.
    """
    fingerprint = hashlib.sha256(f"{domain_tag}\x00{query}".encode()).digest()
    return str(uuid.UUID(bytes=fingerprint[:16], version=5))


def _insert_batch(
    conn: sqlite3.Connection,
    pairs: list[tuple[str, str]],
    domain_tag: str,
) -> int:
    now = datetime.now(tz=timezone.utc).isoformat()
    rows = [
        (
            _content_uuid(query, domain_tag),
            now,
            query,
            response,
            None,
            CONFIDENCE_SCORE,
            domain_tag,
            None,
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


def _count_pending(conn: sqlite3.Connection, domain_tag: str) -> int:
    row = conn.execute(
        "SELECT COUNT(*) FROM training_records WHERE status='PENDING' AND domain_tag=?",
        (domain_tag,),
    ).fetchone()
    return row[0] if row else 0


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed GristMill training buffer — runbooks domain (EXP-005)"
    )
    parser.add_argument(
        "--runbooks-dir",
        type=Path,
        default=None,
        help="Optional directory of Markdown/text runbook files for Layer 2 domain-specific seeding",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Override total target count for foundation datasets (default: SQuAD 20K + SE 5K + WikiHow 5K + TechQA all)",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"Path to training_buffer.sqlite (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--skip-squad",
        action="store_true",
        help="Skip SQuAD 2.0 (useful for quick local testing)",
    )
    parser.add_argument(
        "--skip-wikihow",
        action="store_true",
        help="Skip WikiHow (large streaming dataset — use this to avoid long streaming waits)",
    )
    parser.add_argument(
        "--skip-se",
        action="store_true",
        help="Skip StackExchange ops dataset",
    )
    parser.add_argument(
        "--skip-techqa",
        action="store_true",
        help="Skip IBM TechQA dataset",
    )
    parser.add_argument(
        "--pairs-per-chunk",
        type=int,
        default=QA_PAIRS_PER_CHUNK,
        help=f"Q&A pairs to generate per runbook chunk (default: {QA_PAIRS_PER_CHUNK})",
    )
    args = parser.parse_args()

    print(f"DB path : {args.db_path}")
    print(f"Domain  : {DOMAIN_TAG}")
    if args.runbooks_dir:
        print(f"Runbooks: {args.runbooks_dir}")
    print()

    conn = _open_db(args.db_path)
    _ensure_table(conn)

    try:
        from tqdm import tqdm as _tqdm
    except ImportError:
        def _tqdm(iterable=None, **kwargs):  # type: ignore[misc]
            return iterable if iterable is not None else (lambda x: x)

    total_inserted = 0

    # ── Layer 1: Foundation datasets ──────────────────────────────────────────
    foundation_loaders: list[tuple[str, callable]] = []

    if not args.skip_squad:
        squad_count = args.count or SQUAD_COUNT
        foundation_loaders.append(("SQuAD 2.0", lambda: _load_squad(squad_count)))

    if not args.skip_se:
        foundation_loaders.append(("StackExchange (ops)", lambda: _load_stack_exchange(SE_COUNT)))
    if not args.skip_techqa:
        foundation_loaders.append(("TechQA", _load_techqa))
    if not args.skip_wikihow:
        foundation_loaders.append(("WikiHow", lambda: _load_wikihow(WIKIHOW_COUNT)))

    for name, loader in foundation_loaders:
        print(f"  Fetching {name} …")
        try:
            pairs = loader()
        except Exception as exc:
            print(f"  [warn] {name} failed: {exc} — skipping")
            continue
        print(f"  → {len(pairs)} pairs fetched")
        if pairs:
            CHUNK = 500
            chunks = [pairs[i : i + CHUNK] for i in range(0, len(pairs), CHUNK)]
            inserted = 0
            for chunk in chunks:
                inserted += _insert_batch(conn, chunk, FOUNDATION_TAG)
            total_inserted += inserted
            print(f"  → {inserted} records inserted (domain={FOUNDATION_TAG})")

    # ── Layer 2: Local runbooks (optional) ────────────────────────────────────
    if args.runbooks_dir:
        if not args.runbooks_dir.exists():
            print(f"  [warn] --runbooks-dir {args.runbooks_dir} does not exist — skipping Layer 2")
        else:
            print(f"\n  Processing local runbooks in {args.runbooks_dir} …")
            try:
                teacher_fn = _build_teacher_fn()
                pairs = _load_local_runbooks(args.runbooks_dir, teacher_fn, pairs_per_chunk=args.pairs_per_chunk)
                print(f"  → {len(pairs)} domain-specific pairs generated")
                if pairs:
                    inserted = _insert_batch(conn, pairs, DOMAIN_TAG)
                    total_inserted += inserted
                    print(f"  → {inserted} records inserted (domain={DOMAIN_TAG})")
            except RuntimeError as exc:
                print(f"  [error] Layer 2 failed: {exc}")
                raise SystemExit(1) from exc

    # ── Summary ───────────────────────────────────────────────────────────────
    pending_foundation = _count_pending(conn, FOUNDATION_TAG)
    pending_domain = _count_pending(conn, DOMAIN_TAG)
    conn.close()

    print()
    print(f"Inserted total : {total_inserted} new records")
    print(f"Pending (runbooks_foundation) : {pending_foundation}")
    print(f"Pending (runbooks)            : {pending_domain}")

    combined = pending_foundation + pending_domain
    if combined >= 1_000:
        print(f"  ✓ {combined} total PENDING records — distillation cycle ready.")
    else:
        print(f"  ↑ Need {1_000 - combined} more records to reach the 1,000-record trigger threshold.")


if __name__ == "__main__":
    main()
