#!/usr/bin/env python3
"""generate_training_probes.py — Build a probe YAML from actual training buffer records.

Samples CONSUMED records from the training buffer SQLite database, extracts the
final numeric/factual answer from the teacher response, and writes a probe YAML
that can be fed directly into compare_lora_adapter.py.

This gives the most honest signal about whether the adapter learned from its
training data: the questions are *structurally identical* to what it was trained
on (same problem types, different numbers).

Usage
-----
    python scripts/generate_training_probes.py
    python scripts/generate_training_probes.py --domain reasoning --n 10 --output probes/training_sample.yaml
    python scripts/generate_training_probes.py --domain reasoning --n 10 --math-only
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from pathlib import Path

_HERE = Path(__file__).parent.parent
if str(_HERE / "src") not in sys.path:
    sys.path.insert(0, str(_HERE / "src"))

# ── Answer extraction ─────────────────────────────────────────────────────────

# Ordered from most-specific to least-specific.  The first match wins.
_ANSWER_PATTERNS = [
    # LaTeX boxed answer: \boxed{1225} or \boxed{16,800}  — most reliable
    r"\\boxed\{([^\}]{1,40})\}",
    # Explicit "Final answer: X" lines
    r"(?:final answer|answer)[:\s]+\$?([\d,\.]+(?:\s+\w{1,10})?)",
    # Conclusion sentence: "So, the factory produces 21 tons in a week."
    # Requires a number + optional unit right before sentence-end
    r"(?:so[,\s]|therefore[,\s]|thus[,\s]).*?(\d[\d,\.]*)\s*"
    r"(?:feet|units|km|miles|hours|days|weeks|years|gallons|tons|kg|"
    r"bacteria|students|workers|handshakes|tires|cars|apples|votes|"
    r"dollars|percent|%|\$)?\s*(?:\.|\n|$)",
    # "= 21 tons" or "= $4,500,000" at the end of a calculation chain
    r"=\s*\$?([\d,\.]+)\s*(?:feet|units|km|miles|hours|days|weeks|years|gallons|"
    r"tons|kg|bacteria|students|workers|handshakes|tires|cars|apples|votes|"
    r"dollars|percent|%)?\s*(?:\.|\n|$)",
    # "a total of $4,500,000" or "total of 30,000 tires"
    r"(?:a total of|total of)\s+\$?([\d,\.]+)",
    # Bold markdown answer: **1,225** or **21 tons**
    r"\*\*([\d,\.]+(?:\s+\w{1,10})?)\*\*",
]

# Queries matching any of these are dropped — they are creative/code/trivia, not math reasoning.
_SKIP_PATTERNS = [
    # Roleplay / persona / creative writing / chat simulations
    r"\b(imagine you are|write a|draft a|create a|describe|roleplay|as if you|"
    r"in the style of|acting as|pretend|guide me|tell me a joke|"
    r"enchant|magical|mystical|fairy|dragon|wizard|fantasy)\b",
    r"this is a chat between",  # multi-character chat roleplay format
    # Code tasks
    r"\b(javascript|python script|node\.js|mongodb|c\+\+|sql query|"
    r"write a program|develop a|implement a|```)\b",
    # Structured extraction blocks
    r"(BEGININPUT|ENDINPUT|BEGININSTRUCTION|ENDINSTRUCTION|PLAINFORMAT)",
    # Trivia / pop-culture / named entity recall (no arithmetic)
    r"^[^0-9$%]{0,300}(what .{0,60}(show|movie|film|book|song|band|actor|year|who|"
    r"when|where|which country|capital of))[^0-9]{0,200}$",
    # Too short to be a real problem (fragments, single sentences with no numbers)
    r"^\s*.{0,30}\s*$",
]

# A query must contain at least one digit to qualify as a quantitative probe.
_REQUIRES_DIGIT = re.compile(r"\d")


def _extract_answer(teacher_response: str) -> str | None:
    """Extract the final short answer from a teacher response.

    Strategy (ordered by confidence):
      1. LaTeX \\boxed{} — most reliable marker of a final answer
      2. Explicit "Final answer: X" / "The answer is X" labels
      3. Last number in a conclusion sentence (So, / Therefore, / Thus,)
      4. Last number in a "= X" calculation chain
      5. Bold markdown **X**

    Works on the last 800 chars where conclusions live.
    """
    text = teacher_response.strip()
    tail = text[-800:] if len(text) > 800 else text

    # 1. LaTeX boxed — take the last one in case of intermediate boxed steps
    boxed = re.findall(r"\\boxed\{([^\}]{1,40})\}", tail)
    if boxed:
        return boxed[-1].strip().rstrip(".,;:")

    # 2. Explicit answer labels: "Final answer: 2450" / "The answer is 21 tons"
    explicit = re.findall(
        r"(?:final answer|the answer(?:\s+is)?)[:\s]+[^0-9]*(\d[\d,\.]*)",
        tail,
        re.IGNORECASE,
    )
    if explicit:
        return explicit[-1]

    # 3. Conclusion sentences — find the last "So,/Therefore,/Thus," sentence
    #    then return the last number that appears in it.
    conclusion_sentences = re.findall(
        r"(?:so[,\s]|therefore[,\s]|thus[,\s])([^\n.!?]{5,200})",
        tail,
        re.IGNORECASE,
    )
    if conclusion_sentences:
        last = conclusion_sentences[-1]
        nums = re.findall(r"\d[\d,\.]*", last)
        if nums:
            # Prefer the last number; skip trivially small ones like "1" or "0"
            for n in reversed(nums):
                if float(n.replace(",", "")) > 1:
                    return n
            return nums[-1]

    # 4. Last "= NUMBER" in a calculation chain (e.g. "= 21,000 kg")
    equals_nums = re.findall(r"=\s*\$?([\d,\.]+)", tail)
    if equals_nums:
        # Take the last, skip fractions that look like intermediate results
        for n in reversed(equals_nums):
            val = float(n.replace(",", ""))
            if val > 1:
                return n
        return equals_nums[-1]

    # 5. Bold markdown: **1,225** or **21 tons**
    bold = re.findall(r"\*\*([\d,\.]+(?:\s+\w{1,10})?)\*\*", tail)
    if bold:
        return bold[-1].strip().rstrip(".,;:")

    return None


def _is_skippable(query_text: str, math_only: bool = False) -> bool:
    """Return True if this query should be excluded from the probe set."""
    q = query_text.strip()
    if not q or len(q) < 30:
        return True
    for pat in _SKIP_PATTERNS:
        if re.search(pat, q, re.IGNORECASE | re.DOTALL):
            return True
    if math_only and not _REQUIRES_DIGIT.search(q):
        return True
    return False


# ── Database sampling ─────────────────────────────────────────────────────────


def sample_records(db_path: Path, domain: str, n: int) -> list[dict]:
    """Sample n CONSUMED records from the training buffer for *domain*."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    if domain == "all":
        rows = conn.execute(
            "SELECT record_id, query_text, teacher_response, domain_tag "
            "FROM training_records WHERE status='CONSUMED' ORDER BY RANDOM() LIMIT ?",
            (n * 5,),  # oversample then filter
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT record_id, query_text, teacher_response, domain_tag "
            "FROM training_records WHERE status='CONSUMED' AND domain_tag=? ORDER BY RANDOM() LIMIT ?",
            (domain, n * 5),
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── Probe generation ──────────────────────────────────────────────────────────


def build_probes(records: list[dict], n: int, math_only: bool = False) -> list[dict]:
    """Filter, extract answers, and build probe dicts."""
    probes = []
    for rec in records:
        if len(probes) >= n:
            break
        q = rec["query_text"].strip()
        if _is_skippable(q, math_only=math_only):
            continue
        answer = _extract_answer(rec["teacher_response"])
        probe = {
            "id": f"train_{rec['record_id'][:8]}",
            "tags": [rec.get("domain_tag", "unknown"), "training-data"],
            "question": q,
            "expected": rec["teacher_response"].strip(),
            "notes": "Auto-generated from training buffer",
        }
        if answer:
            probe["correct_answer"] = answer
        probes.append(probe)
    return probes


def write_yaml(probes: list[dict], output_path: Path, domain: str) -> None:
    try:
        import yaml  # type: ignore[import]
    except ImportError:
        raise RuntimeError("PyYAML required: pip install pyyaml")

    doc = {
        "# Training-data probes — auto-generated by generate_training_probes.py": None,
        "domain": domain,
        "description": (
            "Probes sampled directly from the training buffer. "
            "Tests whether the adapter generalised to training-data query patterns."
        ),
        "probes": probes,
    }
    # yaml.dump doesn't support comment-keys; write header manually
    header = (
        f"# Training-data probes — auto-generated by generate_training_probes.py\n"
        f"# domain: {domain}  |  count: {len(probes)}\n"
        f"# Run: python scripts/compare_lora_adapter.py --probe-set {output_path.stem}\n\n"
    )
    body = yaml.dump(
        {"domain": domain, "probes": probes},
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=False,
        width=120,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(header + body)
    print(f"Wrote {len(probes)} probes → {output_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────


def _resolve_db_path() -> Path:
    import os
    try:
        import yaml  # type: ignore[import]
        from gristmill_ml.config import config_candidates
        for p in config_candidates():
            if p.exists():
                cfg = yaml.safe_load(p.read_text()) or {}
                db_str = (cfg.get("sieve") or {}).get("training_buffer_path")
                if db_str:
                    return Path(db_str)
                break
    except Exception:
        pass
    default = Path("/data/gristmill/db/training_buffer.sqlite")
    if default.exists():
        return default
    return Path.home() / ".gristmill" / "db" / "training_buffer.sqlite"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a probe YAML from actual training buffer records.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="Path to training_buffer.sqlite (resolved from config by default)",
    )
    parser.add_argument(
        "--domain",
        default="reasoning",
        help="Domain tag to sample from (default: reasoning). Use 'all' for any domain.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=10,
        help="Number of probes to generate (default: 10)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output YAML path (default: probes/training_<domain>.yaml)",
    )
    parser.add_argument(
        "--math-only",
        action="store_true",
        help="Only include queries that contain at least one digit (pure quantitative reasoning)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print probes to stdout instead of writing a file",
    )
    args = parser.parse_args()

    db_path = args.db or _resolve_db_path()
    if not db_path.exists():
        print(f"ERROR: training database not found at {db_path}", file=sys.stderr)
        sys.exit(1)

    output_path = args.output or (
        Path(__file__).parent.parent / "probes" / f"training_{args.domain}.yaml"
    )

    print(f"Sampling from: {db_path}")
    print(f"Domain       : {args.domain}")
    print(f"Requested    : {args.n} probes")
    print()

    records = sample_records(db_path, args.domain, args.n)
    probes = build_probes(records, args.n, math_only=args.math_only)

    if not probes:
        print("ERROR: no suitable records found (try --domain all)", file=sys.stderr)
        sys.exit(1)

    checkable = sum(1 for p in probes if p.get("correct_answer"))
    print(f"Generated    : {len(probes)} probes ({checkable} with extracted correct_answer)")
    print()

    if args.dry_run:
        try:
            import yaml  # type: ignore[import]
            print(yaml.dump({"probes": probes}, allow_unicode=True, default_flow_style=False))
        except ImportError:
            import json
            print(json.dumps(probes, indent=2))
        return

    write_yaml(probes, output_path, args.domain)
    print()
    print("Run the evaluation:")
    print(
        f"  python scripts/compare_lora_adapter.py "
        f"--probe-set {output_path.stem} --domain {args.domain}"
    )


if __name__ == "__main__":
    main()
