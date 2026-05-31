#!/usr/bin/env python3
"""compare_lora_adapter.py — Compare a LoRA adapter against the base model.

Runs a named probe set (default: "reasoning") against both the base model and
a specified LoRA adapter, then prints a side-by-side report and optionally
saves results as JSON for experiment reproducibility.

Usage examples
--------------
# Compare active reasoning adapter with defaults from config.yaml:
    python scripts/compare_lora_adapter.py

# Explicit adapter path and probe set:
    python scripts/compare_lora_adapter.py \\
        --adapter ~/.gristmill/checkpoints/active/reasoning \\
        --probe-set reasoning

# Custom single question (no probe set):
    python scripts/compare_lora_adapter.py \\
        --question "What is 7 times 28?" \\
        --no-probe-set

# Save results for later comparison:
    python scripts/compare_lora_adapter.py \\
        --output results/exp3-reasoning.json

# Reproduce a specific experiment from scratch:
    python scripts/compare_lora_adapter.py \\
        --base-model Qwen/Qwen2.5-0.5B-Instruct \\
        --adapter /path/to/adapter \\
        --domain reasoning \\
        --probe-set reasoning \\
        --output results/exp3-reasoning.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running as a script from the repo root without installing the package.
_HERE = Path(__file__).parent.parent  # gristmill-ml/
if str(_HERE / "src") not in sys.path:
    sys.path.insert(0, str(_HERE / "src"))


def _resolve_config() -> dict:
    """Load gristmill config.yaml (same search order as the trainer)."""
    import os

    try:
        import yaml  # type: ignore[import]
    except ImportError:
        return {}

    candidates = []
    if env := os.environ.get("GRISTMILL_CONFIG"):
        candidates.append(Path(env))
    candidates += [
        Path("/data/gristmill/config.yaml"),
        Path.home() / ".gristmill" / "config.yaml",
    ]
    for p in candidates:
        if p.exists():
            try:
                return yaml.safe_load(p.read_text()) or {}
            except Exception:
                pass
    return {}


def _default_base_model(cfg: dict) -> str:
    return (cfg.get("trainer") or {}).get("base_model", "Qwen/Qwen2.5-0.5B-Instruct")


def _default_checkpoint_root(cfg: dict) -> Path:
    raw = (cfg.get("millwright") or {}).get("checkpoint_dir")
    if raw:
        return Path(str(raw).replace("~", str(Path.home())))
    docker = Path("/data/gristmill/checkpoints")
    if docker.exists():
        return docker
    return Path.home() / ".gristmill" / "checkpoints"


def main() -> None:
    cfg = _resolve_config()
    default_base = _default_base_model(cfg)
    default_ckpt_root = _default_checkpoint_root(cfg)

    parser = argparse.ArgumentParser(
        description="Compare a LoRA adapter vs. base model on a named probe set.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--base-model",
        default=default_base,
        help=f"HuggingFace model id or local path (default from config: {default_base})",
    )
    parser.add_argument(
        "--adapter",
        type=Path,
        default=None,
        help=(
            "Path to PEFT adapter directory. "
            "Defaults to <checkpoint_root>/active/<domain>."
        ),
    )
    parser.add_argument(
        "--domain",
        default="reasoning",
        help="Domain name — used to locate the active adapter and label the report (default: reasoning)",
    )
    parser.add_argument(
        "--probe-set",
        default="reasoning",
        help="Named probe set to load from probes/<probe-set>.yaml (default: reasoning)",
    )
    parser.add_argument(
        "--question",
        default=None,
        help="Single ad-hoc question (overrides --probe-set when combined with --no-probe-set)",
    )
    parser.add_argument(
        "--no-probe-set",
        action="store_true",
        help="Disable the named probe set; use --question for a single query",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate per response (default: 512)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Save JSON report to this path (default: print only)",
    )
    parser.add_argument(
        "--probes-dir",
        type=Path,
        default=None,
        help="Override the default probes/ directory",
    )

    args = parser.parse_args()

    # ── Resolve adapter path ──────────────────────────────────────────────────
    adapter_path: Path = args.adapter or (
        default_ckpt_root / "active" / args.domain
    )
    if not adapter_path.exists():
        print(f"ERROR: Adapter not found at {adapter_path}", file=sys.stderr)
        print(
            "  Run a training cycle first, or pass --adapter <path>.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Load probes ───────────────────────────────────────────────────────────
    from gristmill_ml.experiments.adapter_eval import (
        AdapterEvaluator,
        load_probes,
        probes_from_questions,
    )

    if args.no_probe_set:
        if not args.question:
            print("ERROR: --no-probe-set requires --question", file=sys.stderr)
            sys.exit(1)
        probes = probes_from_questions([args.question])
        probe_set_name = "ad-hoc"
    else:
        probes = load_probes(args.probe_set, probes_dir=args.probes_dir)
        probe_set_name = args.probe_set
        # Optionally prepend an ad-hoc question if provided alongside a probe set
        if args.question:
            probes = probes_from_questions([args.question]) + probes

    print(f"Base model : {args.base_model}")
    print(f"Adapter    : {adapter_path}")
    print(f"Domain     : {args.domain}")
    print(f"Probe set  : {probe_set_name} ({len(probes)} probes)")
    print()

    # ── Run evaluation ────────────────────────────────────────────────────────
    evaluator = AdapterEvaluator(
        base_model=args.base_model,
        adapter_path=adapter_path,
        domain=args.domain,
        max_new_tokens=args.max_new_tokens,
    )
    report = evaluator.evaluate(probes, probe_set=probe_set_name)
    report.print_report()

    # ── Save JSON ─────────────────────────────────────────────────────────────
    if args.output:
        report.save(args.output)
        print(f"Report saved to: {args.output}")


if __name__ == "__main__":
    main()
