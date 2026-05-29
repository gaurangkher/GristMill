"""AdapterEvaluator — compare a LoRA adapter against the base model on named probe sets.

Produces structured :class:`EvalReport` objects that can be printed, saved as JSON,
and compared across experiments for reproducibility.

Typical usage::

    from gristmill_ml.experiments.adapter_eval import AdapterEvaluator, load_probes

    probes = load_probes("reasoning")          # loads probes/reasoning.yaml
    evaluator = AdapterEvaluator(
        base_model="Qwen/Qwen2.5-0.5B-Instruct",
        adapter_path=Path("~/.gristmill/checkpoints/active/reasoning"),
    )
    report = evaluator.evaluate(probes)
    report.print_report()
    report.save(Path("results/exp3-reasoning.json"))
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Path to the probes directory (relative to this file's package root).
_PROBES_DIR = Path(__file__).parent.parent.parent.parent / "probes"


# ── Data classes ──────────────────────────────────────────────────────────────


@dataclass
class ProbeResult:
    """Result for a single probe question."""

    probe_id: str
    question: str
    expected: str
    base_response: str
    adapter_response: str
    base_rouge_l: float
    adapter_rouge_l: float
    rouge_l_delta: float
    # Optional factual correctness check (requires correct_answer in probe spec)
    base_correct: Optional[bool] = None
    adapter_correct: Optional[bool] = None
    correct_answer: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EvalReport:
    """Full evaluation report comparing base model vs. LoRA adapter across all probes."""

    base_model: str
    adapter_path: str
    domain: str
    probe_set: str
    timestamp: str
    probes: list[ProbeResult] = field(default_factory=list)
    # Aggregate metrics
    mean_base_rouge_l: float = 0.0
    mean_adapter_rouge_l: float = 0.0
    rouge_l_delta: float = 0.0
    base_correct_count: int = 0
    adapter_correct_count: int = 0
    checkable_count: int = 0

    def print_report(self) -> None:
        """Print a human-readable comparison to stdout."""
        sep = "=" * 70
        print(f"\n{sep}")
        print("  GristMill LoRA Adapter Evaluation Report")
        print(sep)
        print(f"  Base model   : {self.base_model}")
        print(f"  Adapter      : {self.adapter_path}")
        print(f"  Domain       : {self.domain}")
        print(f"  Probe set    : {self.probe_set}")
        print(f"  Timestamp    : {self.timestamp}")
        print(f"  Probes run   : {len(self.probes)}")
        print()
        print("  Aggregate ROUGE-L")
        print(f"    Base model : {self.mean_base_rouge_l:.4f}")
        print(f"    Adapter    : {self.mean_adapter_rouge_l:.4f}")
        delta_sign = "+" if self.rouge_l_delta >= 0 else ""
        print(f"    Delta      : {delta_sign}{self.rouge_l_delta:.4f}")
        if self.checkable_count > 0:
            print()
            print(f"  Factual correctness ({self.checkable_count} checkable probes)")
            print(f"    Base model : {self.base_correct_count}/{self.checkable_count}")
            print(f"    Adapter    : {self.adapter_correct_count}/{self.checkable_count}")
        print(sep)

        for i, pr in enumerate(self.probes, 1):
            print(f"\n  [{i}/{len(self.probes)}] {pr.probe_id}")
            print(f"  {'─' * 66}")
            print(f"  QUESTION:\n  {pr.question.strip()}")
            print(f"\n  EXPECTED:\n  {pr.expected.strip()}")
            print(f"\n  BASE MODEL (ROUGE-L={pr.base_rouge_l:.4f}", end="")
            if pr.base_correct is not None:
                print(f", correct={pr.base_correct}", end="")
            print(f"):\n  {pr.base_response.strip()}")
            print(f"\n  ADAPTER   (ROUGE-L={pr.adapter_rouge_l:.4f}", end="")
            if pr.adapter_correct is not None:
                print(f", correct={pr.adapter_correct}", end="")
            print(f"):\n  {pr.adapter_response.strip()}")

        print(f"\n{sep}\n")

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    def save(self, path: Path) -> None:
        """Save the report as JSON to *path*."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        logger.info("EvalReport saved to %s", path)


# ── Probe loading ─────────────────────────────────────────────────────────────


def load_probes(probe_set: str, probes_dir: Optional[Path] = None) -> list[dict]:
    """Load a named probe set from ``probes/{probe_set}.yaml``.

    Args:
        probe_set: Name of the probe set file (without ``.yaml`` extension).
        probes_dir: Override the default probes directory.

    Returns:
        List of probe dicts, each with keys ``id``, ``question``, ``expected``,
        and optionally ``correct_answer``.
    """
    try:
        import yaml  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError("PyYAML is required: pip install pyyaml") from exc

    root = Path(probes_dir) if probes_dir else _PROBES_DIR
    probe_file = root / f"{probe_set}.yaml"
    if not probe_file.exists():
        raise FileNotFoundError(
            f"Probe set '{probe_set}' not found at {probe_file}. "
            f"Available: {[p.stem for p in root.glob('*.yaml')]}"
        )

    data = yaml.safe_load(probe_file.read_text()) or {}
    probes = data.get("probes", [])
    logger.info("Loaded %d probes from '%s'", len(probes), probe_set)
    return probes


def probes_from_questions(questions: list[str]) -> list[dict]:
    """Wrap plain question strings as minimal probe dicts (no expected/correct_answer).

    Useful for ad-hoc evaluation without a named probe set.
    """
    return [
        {"id": f"q{i + 1}", "question": q, "expected": "", "correct_answer": None}
        for i, q in enumerate(questions)
    ]


# ── Evaluator ─────────────────────────────────────────────────────────────────


class AdapterEvaluator:
    """Compare a LoRA adapter against its base model on a set of probe questions.

    Parameters
    ----------
    base_model:
        HuggingFace model id or local path (e.g. ``Qwen/Qwen2.5-0.5B-Instruct``).
    adapter_path:
        Path to the PEFT adapter directory (the ``active/{domain}`` directory
        written by :class:`~gristmill_ml.trainer.checkpoint.CheckpointManager`).
    domain:
        Domain label used for reporting (e.g. ``"reasoning"``).
    device:
        ``"cuda"``, ``"mps"``, or ``"cpu"``.  Auto-detected when ``None``.
    max_new_tokens:
        Maximum tokens to generate per response.
    """

    def __init__(
        self,
        base_model: str,
        adapter_path: Path,
        domain: str = "default",
        device: Optional[str] = None,
        max_new_tokens: int = 256,
    ) -> None:
        self.base_model = base_model
        self.adapter_path = Path(adapter_path).expanduser()
        self.domain = domain
        self.max_new_tokens = max_new_tokens
        self.device = device or _detect_device()

        if not self.adapter_path.exists():
            raise FileNotFoundError(f"Adapter not found: {self.adapter_path}")

    # ── Public API ────────────────────────────────────────────────────────────

    def evaluate(self, probes: list[dict], probe_set: str = "custom") -> EvalReport:
        """Run all probes and return a structured :class:`EvalReport`.

        Loads the base model and adapter once each (sequentially to conserve
        memory) and generates responses for every probe.
        """
        logger.info(
            "AdapterEvaluator: %d probes, base=%s, adapter=%s, device=%s",
            len(probes),
            self.base_model,
            self.adapter_path,
            self.device,
        )

        tokenizer = self._load_tokenizer()

        # ── Base model pass ───────────────────────────────────────────────────
        logger.info("Generating base model responses…")
        base_model_obj = self._load_base_model()
        base_responses = [self._generate(base_model_obj, tokenizer, p["question"]) for p in probes]
        del base_model_obj
        _free_vram()

        # ── Adapter pass ──────────────────────────────────────────────────────
        logger.info("Generating LoRA adapter responses…")
        adapter_model = self._load_adapter_model()
        adapter_responses = [
            self._generate(adapter_model, tokenizer, p["question"]) for p in probes
        ]
        del adapter_model
        _free_vram()

        # ── Assemble results ──────────────────────────────────────────────────
        probe_results: list[ProbeResult] = []
        for probe, base_resp, adapter_resp in zip(probes, base_responses, adapter_responses):
            expected = probe.get("expected", "")
            base_rl = _rouge_l(base_resp, expected) if expected else 0.0
            adapter_rl = _rouge_l(adapter_resp, expected) if expected else 0.0
            correct_answer = probe.get("correct_answer")
            probe_results.append(
                ProbeResult(
                    probe_id=probe.get("id", "unknown"),
                    question=probe["question"],
                    expected=expected,
                    base_response=base_resp,
                    adapter_response=adapter_resp,
                    base_rouge_l=base_rl,
                    adapter_rouge_l=adapter_rl,
                    rouge_l_delta=adapter_rl - base_rl,
                    base_correct=(
                        _check_correct(base_resp, correct_answer) if correct_answer else None
                    ),
                    adapter_correct=(
                        _check_correct(adapter_resp, correct_answer) if correct_answer else None
                    ),
                    correct_answer=correct_answer,
                )
            )

        # Aggregate metrics
        mean_base = _mean([r.base_rouge_l for r in probe_results])
        mean_adapter = _mean([r.adapter_rouge_l for r in probe_results])
        checkable = [r for r in probe_results if r.base_correct is not None]

        return EvalReport(
            base_model=self.base_model,
            adapter_path=str(self.adapter_path),
            domain=self.domain,
            probe_set=probe_set,
            timestamp=datetime.now(timezone.utc).isoformat(),
            probes=probe_results,
            mean_base_rouge_l=mean_base,
            mean_adapter_rouge_l=mean_adapter,
            rouge_l_delta=mean_adapter - mean_base,
            base_correct_count=sum(1 for r in checkable if r.base_correct),
            adapter_correct_count=sum(1 for r in checkable if r.adapter_correct),
            checkable_count=len(checkable),
        )

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_tokenizer(self):
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(self.base_model, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        return tok

    def _load_base_model(self):
        import torch
        from transformers import AutoModelForCausalLM

        kwargs: dict = {
            "torch_dtype": torch.bfloat16 if self.device != "cpu" else torch.float32,
            "trust_remote_code": True,
        }
        if self.device != "cpu":
            kwargs["device_map"] = self.device
        model = AutoModelForCausalLM.from_pretrained(self.base_model, **kwargs)
        model.eval()
        return model

    def _load_adapter_model(self):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM

        kwargs: dict = {
            "torch_dtype": torch.bfloat16 if self.device != "cpu" else torch.float32,
            "trust_remote_code": True,
        }
        if self.device != "cpu":
            kwargs["device_map"] = self.device
        base = AutoModelForCausalLM.from_pretrained(self.base_model, **kwargs)
        model = PeftModel.from_pretrained(base, str(self.adapter_path))
        model.eval()
        return model

    def _generate(self, model, tokenizer, question: str) -> str:
        import torch

        messages = [{"role": "user", "content": question.strip()}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt")
        # Move inputs to the same device as the model
        try:
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
        except StopIteration:
            pass

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated_ids = out[0][inputs["input_ids"].shape[1] :]
        return tokenizer.decode(generated_ids, skip_special_tokens=True)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _detect_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def _free_vram() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _rouge_l(hypothesis: str, reference: str) -> float:
    """ROUGE-L F1 (LCS-based) without external dependencies."""
    hyp = hypothesis.lower().split()
    ref = reference.lower().split()
    if not hyp or not ref:
        return 0.0
    lcs = _lcs_length(hyp, ref)
    p = lcs / len(hyp)
    r = lcs / len(ref)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def _lcs_length(a: list, b: list) -> int:
    m, n = len(a), len(b)
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


def _check_correct(response: str, correct_answer: str) -> bool:
    """Check if the correct answer string appears in the model's response."""
    return correct_answer.strip() in response


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
