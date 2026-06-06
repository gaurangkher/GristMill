#!/usr/bin/env python3
"""GristMill closed-loop demo.

Demonstrates the full pipeline:
  1. Seed synthetic feedback JSONL  (simulates real events being routed)
  2. Train the Sieve classifier      (SieveTrainer + MLflow)
  3. Export to ONNX (INT8)           (parity-validated)
  4. Deploy                          (overwrites intent-classifier-v1.onnx)
  5. Hot-reload the Rust daemon      (live, no restart needed)

Run inside the trainer container:
  docker exec gristmill_trainer python /app/scripts/demo_closed_loop.py

Or on the host (requires gristmill-ml installed):
  python scripts/demo_closed_loop.py --no-reload
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("demo")

# ── Paths (match Docker bind-mounts) ─────────────────────────────────────────

FEEDBACK_DIR = Path(os.environ.get("GRISTMILL_FEEDBACK_DIR", "/tmp/gristmill-demo-feedback"))
MODELS_DIR   = Path(os.environ.get("GRISTMILL_MODELS_DIR",   "/data/gristmill/models"))
DAEMON_SOCK  = os.environ.get("GRISTMILL_SOCK", "/data/gristmill/gristmill.sock")
MLFLOW_URI   = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5050")

# ── Synthetic event templates ─────────────────────────────────────────────────
# (text, source, priority, route_decision, confidence)
_TEMPLATES: list[tuple[str, str, int, str, float]] = [
    # LOCAL_ML — simple, high-confidence local queries
    ("ping",                              "cli",     1, "LOCAL_ML",  0.97),
    ("status",                            "cli",     1, "LOCAL_ML",  0.95),
    ("list events",                       "http",    1, "LOCAL_ML",  0.93),
    ("get weather today",                 "http",    1, "LOCAL_ML",  0.91),
    ("what time is it",                   "cli",     1, "LOCAL_ML",  0.96),
    ("translate hello to spanish",        "http",    1, "LOCAL_ML",  0.88),
    ("tag this as urgent",                "webhook", 1, "LOCAL_ML",  0.90),
    ("show me today's summary",           "http",    1, "LOCAL_ML",  0.89),
    # RULES — schedule/cron/routing patterns
    ("schedule meeting tomorrow 3pm",     "http",    1, "RULES",     0.94),
    ("remind me at 09:00",                "cron",    1, "RULES",     0.98),
    ("run daily backup",                  "cron",    1, "RULES",     0.97),
    ("alert if cpu > 90%",                "mq",      1, "RULES",     0.95),
    ("send digest at 18:00",              "cron",    1, "RULES",     0.96),
    ("route to team@example.com",         "http",    1, "RULES",     0.92),
    ("open pr #42",                       "webhook", 1, "RULES",     0.93),
    # HYBRID — local pre-process + LLM refinement
    ("summarise last 10 support tickets", "http",    2, "HYBRID",    0.78),
    ("classify emails and draft replies", "http",    2, "HYBRID",    0.75),
    ("extract entities from document",    "webhook", 2, "HYBRID",    0.80),
    ("generate weekly report from data",  "http",    2, "HYBRID",    0.77),
    ("rank bugs by severity",             "webhook", 2, "HYBRID",    0.76),
    ("proofread and improve this email",  "http",    1, "HYBRID",    0.79),
    # LLM_NEEDED — complex reasoning required
    ("why did the auth service fail intermittently last week", "http", 3, "LLM_NEEDED", 0.55),
    ("write a blog post about Rust ownership",                 "http", 1, "LLM_NEEDED", 0.52),
    ("debug this async race condition",                        "http", 2, "LLM_NEEDED", 0.58),
    ("design a microservices architecture for our platform",   "http", 2, "LLM_NEEDED", 0.50),
    ("what is the capital of India",                           "http", 1, "LLM_NEEDED", 0.53),
    ("explain quantum entanglement to a 10-year-old",          "http", 1, "LLM_NEEDED", 0.51),
    ("why did revenue drop 20% last quarter",                  "http", 2, "LLM_NEEDED", 0.54),
    ("create a comprehensive project plan with milestones",    "http", 2, "LLM_NEEDED", 0.56),
]

SUFFIXES = ["", " please", " now", " quickly", " asap", " in detail", " step by step"]


def _seed_feedback(feedback_dir: Path, n: int = 400) -> Path:
    """Write n synthetic feedback records to a JSONL file."""
    feedback_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_path = feedback_dir / f"feedback-{date_str}.jsonl"

    log.info("Seeding %d synthetic feedback records → %s", n, out_path)

    with out_path.open("w") as f:
        for i in range(n):
            tmpl = _TEMPLATES[i % len(_TEMPLATES)]
            text, source, priority, decision, conf = tmpl
            text_variant = text + random.choice(SUFFIXES)
            record = {
                "event_id":            f"demo-{i:06d}",
                "timestamp_ms":        int(time.time() * 1000) + i,
                "route_decision":      decision,
                "confidence":          round(conf + random.uniform(-0.05, 0.05), 4),
                "estimated_tokens":    0,
                "actual_tokens":       None,
                "could_have_been_local": None,
                "event_source":        source,
                "token_count":         len(text_variant.split()),
                # Extra fields read by FeatureExtractor via _text / _priority
                "_text":               text_variant,
                "_priority":           priority,
            }
            f.write(json.dumps(record) + "\n")

    log.info("Feedback file written (%d bytes)", out_path.stat().st_size)
    return out_path


def _print_banner(step: int, total: int, title: str) -> None:
    bar = "─" * 60
    log.info("")
    log.info(bar)
    log.info("  STEP %d/%d — %s", step, total, title)
    log.info(bar)


def run_demo(
    epochs: int,
    n_records: int,
    dry_run: bool,
    no_reload: bool,
    experiment: str,
) -> None:
    total_steps = 5

    # ── Step 1: Seed feedback ─────────────────────────────────────────────────
    _print_banner(1, total_steps, "Seed synthetic feedback data")
    feedback_file = _seed_feedback(FEEDBACK_DIR, n=n_records)
    log.info("Done — %d records written to %s", n_records, feedback_file)

    # ── Step 2–5: Full retrain pipeline ───────────────────────────────────────
    _print_banner(2, total_steps, "Run RetrainPipeline (train → export → deploy → hot-reload)")
    log.info("MLflow experiment : %s", experiment)
    log.info("MLflow UI         : %s", MLFLOW_URI)
    log.info("Models dir        : %s", MODELS_DIR)
    log.info("Daemon socket     : %s", DAEMON_SOCK)
    log.info("Epochs            : %d", epochs)
    log.info("Dry run           : %s", dry_run)
    log.info("")

    from gristmill_ml.pipelines.retrain_sieve import RetrainPipeline

    pipeline = RetrainPipeline(
        epochs=epochs,
        min_records=n_records,
        output_dir=MODELS_DIR,
        quantize=True,
        reload_daemon=not no_reload,
        daemon_sock=DAEMON_SOCK,
        dry_run=dry_run,
        experiment_name=experiment,
        feedback_dir=FEEDBACK_DIR,
    )

    outcome = pipeline.run()

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 60)
    if outcome["success"]:
        m = outcome["metrics"]
        log.info("  DEMO COMPLETE")
        log.info("  Val accuracy  : %.4f (epoch %d)",
                 m.get("best_val_accuracy", 0), m.get("best_epoch", 0))
        log.info("  ONNX size     : %s bytes", m.get("onnx_size_bytes", "n/a"))
        log.info("  Parity diff   : %s", m.get("parity_max_diff", "n/a"))
        log.info("  Daemon reload : %s", m.get("daemon_reload", "n/a"))
        log.info("  Duration      : %.1fs", m.get("duration_seconds", 0))
        log.info("")
        log.info("  View in MLflow: %s", MLFLOW_URI)
        if not dry_run:
            log.info("  Deployed ONNX : %s", outcome.get("onnx_path"))
    else:
        log.error("  DEMO FAILED: %s", outcome.get("error"))
    log.info("=" * 60)

    if not outcome["success"]:
        raise SystemExit(1)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="GristMill closed-loop demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--epochs",     type=int, default=5,
                        help="Training epochs (5 is enough for demo)")
    parser.add_argument("--records",    type=int, default=400,
                        help="Number of synthetic feedback records to seed")
    parser.add_argument("--dry-run",    action="store_true",
                        help="Train + validate but skip deploy and hot-reload")
    parser.add_argument("--no-reload",  action="store_true",
                        help="Skip daemon hot-reload (useful outside Docker)")
    parser.add_argument("--experiment", type=str, default="sieve-demo",
                        help="MLflow experiment name")
    args = parser.parse_args()

    run_demo(
        epochs=args.epochs,
        n_records=args.records,
        dry_run=args.dry_run,
        no_reload=args.no_reload,
        experiment=args.experiment,
    )


if __name__ == "__main__":
    main()
