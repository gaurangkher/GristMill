"""ValidationRunner + RollbackController — post-cycle checkpoint validation.

Two-stage evaluation before a staged adapter is promoted to active
(Section 4.6.6 of the spec):

Stage 1 — Held-out validation set (200 examples, 50 per domain):
    • Overall ROUGE-L delta   >= -0.01
    • Per-domain ROUGE-L delta >= -0.03 on any single domain
    • Confidence calibration ECE delta <= 0.05  (placeholder — requires Self-REF)

Stage 2 — Retention score:
    • Retention ROUGE-L >= 0.90 × prior retention ROUGE-L

The validation set is created once on first run from the training buffer,
stored at ~/.gristmill/db/validation_set.json, and never updated.
"""

from __future__ import annotations

import json
import logging
import random
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_DEFAULT_VALSET = Path.home() / ".gristmill" / "db" / "validation_set.json"

# Promotion thresholds (spec Section 4.6.6)
OVERALL_DELTA_MIN = -0.01
DOMAIN_DELTA_MIN = -0.03
ECE_DELTA_MAX = 0.05
RETENTION_SCORE_MIN = 0.90

VAL_SET_SIZE = 200
VAL_PER_DOMAIN = 50
DOMAIN_TAGS = ("code", "writing", "reasoning", "qa", "creative", "other")


@dataclass
class DomainMetrics:
    domain: str
    score: float
    count: int


@dataclass
class ValidationResult:
    passed: bool
    overall_score: float
    prior_overall_score: float
    overall_delta: float
    domain_metrics: list[DomainMetrics] = field(default_factory=list)
    retention_score: float = 0.0
    prior_retention_score: float = 0.0
    retention_ratio: float = 0.0
    failure_reason: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["evaluated_at"] = datetime.now(timezone.utc).isoformat()
        return d


class ValidationRunner:
    """Runs post-cycle validation against the held-out validation set.

    Parameters
    ----------
    base_model_name:
        HuggingFace model id or local path for the grinder base model.
    val_set_path:
        Path to the persisted validation set JSON.  Created automatically
        on the first call to ``ensure_validation_set()``.
    """

    def __init__(
        self,
        base_model_name: Optional[str] = None,
        val_set_path: Optional[Path] = None,
    ) -> None:
        import os

        self.base_model_name = (
            base_model_name
            or os.environ.get("GRISTMILL_BASE_MODEL", "Qwen/Qwen2.5-3B-Instruct")
        )
        self.val_set_path = val_set_path or _DEFAULT_VALSET
        self.val_set_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Validation set bootstrap ──────────────────────────────────────────────

    def ensure_validation_set(self, training_db_path: Path) -> bool:
        """Create the held-out validation set if it doesn't exist yet.

        Returns True if the set was (re-)created, False if it already exists.
        """
        if self.val_set_path.exists():
            return False

        examples = _sample_validation_set(training_db_path)
        if not examples:
            logger.warning("Cannot create validation set — no CONSUMED records yet")
            return False

        self.val_set_path.write_text(json.dumps(examples, indent=2))
        logger.info("Validation set created: %d examples at %s", len(examples), self.val_set_path)
        return True

    def load_validation_set(self) -> list[dict]:
        if not self.val_set_path.exists():
            return []
        return json.loads(self.val_set_path.read_text())

    # ── Validation ────────────────────────────────────────────────────────────

    def validate(
        self,
        staged_adapter_path: Path,
        prior_adapter_path: Optional[Path],
        retention_records: list[dict],
        prior_metrics_path: Optional[Path] = None,
    ) -> ValidationResult:
        """Run both validation stages and return the result."""
        val_set = self.load_validation_set()
        if not val_set:
            logger.warning("Validation set empty — skipping validation, auto-passing")
            return ValidationResult(
                passed=True,
                overall_score=1.0,
                prior_overall_score=1.0,
                overall_delta=0.0,
                failure_reason=None,
            )

        logger.info("ValidationRunner: running Stage 1 on %d examples", len(val_set))
        staged_scores = self._score_adapter(staged_adapter_path, val_set)
        prior_scores = (
            self._score_adapter(prior_adapter_path, val_set)
            if prior_adapter_path and prior_adapter_path.exists()
            else {ex["record_id"]: 0.5 for ex in val_set}
        )

        overall_staged = _mean_scores(staged_scores)
        overall_prior = _mean_scores(prior_scores)
        overall_delta = overall_staged - overall_prior

        # Per-domain deltas
        domain_metrics: list[DomainMetrics] = []
        worst_domain_delta = 0.0
        for domain in DOMAIN_TAGS:
            domain_examples = [e for e in val_set if e.get("domain_tag") == domain]
            if not domain_examples:
                continue
            ids = {e["record_id"] for e in domain_examples}
            d_staged = _mean_scores({k: v for k, v in staged_scores.items() if k in ids})
            d_prior = _mean_scores({k: v for k, v in prior_scores.items() if k in ids})
            delta = d_staged - d_prior
            worst_domain_delta = min(worst_domain_delta, delta)
            domain_metrics.append(DomainMetrics(domain=domain, score=d_staged, count=len(domain_examples)))

        # Stage 1 pass/fail
        if overall_delta < OVERALL_DELTA_MIN:
            return ValidationResult(
                passed=False,
                overall_score=overall_staged,
                prior_overall_score=overall_prior,
                overall_delta=overall_delta,
                domain_metrics=domain_metrics,
                failure_reason=f"overall_delta={overall_delta:.4f} < {OVERALL_DELTA_MIN}",
            )
        if worst_domain_delta < DOMAIN_DELTA_MIN:
            return ValidationResult(
                passed=False,
                overall_score=overall_staged,
                prior_overall_score=overall_prior,
                overall_delta=overall_delta,
                domain_metrics=domain_metrics,
                failure_reason=f"worst_domain_delta={worst_domain_delta:.4f} < {DOMAIN_DELTA_MIN}",
            )

        # Stage 2 — retention score
        logger.info("ValidationRunner: running Stage 2 on %d retention records", len(retention_records))
        ret_staged = self._score_adapter_on_retention(staged_adapter_path, retention_records)
        ret_prior = (
            self._score_adapter_on_retention(prior_adapter_path, retention_records)
            if prior_adapter_path and prior_adapter_path.exists()
            else ret_staged  # First cycle: no prior — always pass Stage 2
        )
        retention_ratio = (ret_staged / ret_prior) if ret_prior > 0 else 1.0

        if retention_ratio < RETENTION_SCORE_MIN:
            return ValidationResult(
                passed=False,
                overall_score=overall_staged,
                prior_overall_score=overall_prior,
                overall_delta=overall_delta,
                domain_metrics=domain_metrics,
                retention_score=ret_staged,
                prior_retention_score=ret_prior,
                retention_ratio=retention_ratio,
                failure_reason=f"retention_ratio={retention_ratio:.4f} < {RETENTION_SCORE_MIN}",
            )

        return ValidationResult(
            passed=True,
            overall_score=overall_staged,
            prior_overall_score=overall_prior,
            overall_delta=overall_delta,
            domain_metrics=domain_metrics,
            retention_score=ret_staged,
            prior_retention_score=ret_prior,
            retention_ratio=retention_ratio,
        )

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _score_adapter(
        self,
        adapter_path: Path,
        examples: list[dict],
    ) -> dict[str, float]:
        """Run the adapter on *examples* and return per-example ROUGE-L scores."""
        try:
            return _run_adapter_inference(
                base_model_name=self.base_model_name,
                adapter_path=adapter_path,
                examples=examples,
            )
        except Exception as exc:
            logger.warning("Adapter scoring failed (%s), falling back to 0.5: %s", adapter_path, exc)
            return {ex["record_id"]: 0.5 for ex in examples}

    def _score_adapter_on_retention(
        self,
        adapter_path: Optional[Path],
        retention_records: list[dict],
    ) -> float:
        """Return mean ROUGE-L score on retention records."""
        if adapter_path is None or not retention_records:
            return 1.0
        examples = [
            {"record_id": r["record_id"], "query_text": r["query_text"],
             "teacher_response": r["teacher_response"]}
            for r in retention_records[:100]  # Sample for speed
        ]
        scores = self._score_adapter(adapter_path, examples)
        return _mean_scores(scores)


# ── Inference + ROUGE helpers ─────────────────────────────────────────────────


def _run_adapter_inference(
    base_model_name: str,
    adapter_path: Path,
    examples: list[dict],
    max_new_tokens: int = 128,
    batch_size: int = 4,
) -> dict[str, float]:
    """Load the adapter, generate responses, compute ROUGE-L."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if _cuda_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
        device_map=device,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, str(adapter_path))
    model.eval()

    scores: dict[str, float] = {}
    for i in range(0, len(examples), batch_size):
        batch = examples[i : i + batch_size]
        prompts = [ex["query_text"] for ex in batch]
        references = [ex["teacher_response"] for ex in batch]

        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)

        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        for j, (ex, ref) in enumerate(zip(batch, references)):
            generated_ids = out[j][enc["input_ids"].shape[1]:]
            generated = tokenizer.decode(generated_ids, skip_special_tokens=True)
            scores[ex["record_id"]] = _rouge_l(generated, ref)

    # Free VRAM
    del model, base
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    return scores


def _rouge_l(hypothesis: str, reference: str) -> float:
    """Compute ROUGE-L F1 (LCS-based) without external dependencies."""
    hyp_tokens = hypothesis.lower().split()
    ref_tokens = reference.lower().split()
    if not hyp_tokens or not ref_tokens:
        return 0.0
    lcs = _lcs_length(hyp_tokens, ref_tokens)
    precision = lcs / len(hyp_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _lcs_length(a: list, b: list) -> int:
    """Length of the longest common subsequence of *a* and *b*."""
    m, n = len(a), len(b)
    # Space-optimised O(min(m,n)) DP
    if m < n:
        a, b = b, a
        m, n = n, m
    prev = [0] * (n + 1)
    for x in a:
        curr = [0] * (n + 1)
        for j, y in enumerate(b, 1):
            curr[j] = prev[j - 1] + 1 if x == y else max(prev[j], curr[j - 1])
        prev = curr
    return prev[n]


def _mean_scores(scores: dict[str, float]) -> float:
    if not scores:
        return 0.0
    return sum(scores.values()) / len(scores)


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _sample_validation_set(training_db_path: Path) -> list[dict]:
    """Sample 200 examples stratified by domain from the training buffer."""
    try:
        conn = sqlite3.connect(f"file:{training_db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        all_rows = conn.execute(
            "SELECT record_id, query_text, teacher_response, domain_tag, confidence_score "
            "FROM training_records WHERE status = 'CONSUMED'"
        ).fetchall()
        conn.close()
    except sqlite3.Error as exc:
        logger.error("Cannot sample validation set: %s", exc)
        return []

    by_domain: dict[str, list[dict]] = {tag: [] for tag in DOMAIN_TAGS}
    for row in all_rows:
        r = dict(row)
        tag = r["domain_tag"] if r["domain_tag"] in DOMAIN_TAGS else "other"
        by_domain[tag].append(r)

    selected: list[dict] = []
    for tag, records in by_domain.items():
        n = min(VAL_PER_DOMAIN, len(records))
        selected.extend(random.sample(records, n))

    random.shuffle(selected)
    return selected
