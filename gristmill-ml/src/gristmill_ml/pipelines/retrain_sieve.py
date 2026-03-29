"""End-to-end sieve retrain pipeline (closed learning loop).

This module is the main entry point for the weekly/on-demand sieve retraining
cycle:

1. Load feedback JSONL records from ``~/.gristmill/feedback/``
2. Train :class:`SieveTrainer` with MLflow tracking
3. Export to ONNX (optionally INT8 quantised)
4. Validate parity against PyTorch
5. Deploy new ONNX to the output directory
6. Hot-reload the running daemon via IPC
7. Append outcome to ``~/.gristmill/retrain_log.jsonl``

CLI usage::

    python -m gristmill_ml.pipelines.retrain_sieve [options]

See ``--help`` for all options.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import socket
import struct
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── DaemonIpcClient ───────────────────────────────────────────────────────────


class DaemonIpcClient:
    """Minimal synchronous client for the gristmill-daemon msgpack IPC socket.

    Wire format: 4-byte little-endian u32 length prefix + msgpack body.
    Falls back to JSON framing when ``msgpack`` is not installed (with a warning).

    Args:
        sock_path: Path to the Unix domain socket exposed by the Rust daemon.
    """

    def __init__(self, sock_path: str) -> None:
        self._sock_path = sock_path
        self._sock: Optional[socket.socket] = None
        self._msgpack: Optional[object] = None
        self._use_json_fallback = False

        try:
            import msgpack  # type: ignore[import]

            self._msgpack = msgpack
        except ImportError:
            logger.warning(
                "msgpack not installed — hot-reload step skipped. "
                "Install with: pip install msgpack"
            )
            self._use_json_fallback = True  # flag to skip

    def _connect(self) -> None:
        if self._sock is not None:
            return
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(10.0)
        sock.connect(self._sock_path)
        self._sock = sock

    def _send_recv(self, payload: dict) -> dict:
        """Send *payload* and return the parsed response."""
        if self._msgpack is None:
            raise RuntimeError("msgpack not installed — cannot communicate with daemon")

        self._connect()
        assert self._sock is not None

        body: bytes = self._msgpack.packb(payload, use_bin_type=True)  # type: ignore[union-attr]
        length_prefix = struct.pack("<I", len(body))
        self._sock.sendall(length_prefix + body)

        # Read response length
        raw_len = self._recv_exactly(4)
        resp_len = struct.unpack("<I", raw_len)[0]

        # Read response body
        raw_body = self._recv_exactly(resp_len)
        return self._msgpack.unpackb(raw_body, raw=False)  # type: ignore[union-attr]

    def _recv_exactly(self, n: int) -> bytes:
        assert self._sock is not None
        buf = bytearray()
        while len(buf) < n:
            chunk = self._sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("Daemon closed connection before full response")
            buf.extend(chunk)
        return bytes(buf)

    def models_reload(self, model_id: str = "sieve") -> dict:
        """Send a ``models_reload`` request to the daemon.

        Args:
            model_id: Identifier of the model to reload (default: ``"sieve"``).

        Returns:
            The ``ok`` payload dict from the daemon response.

        Raises:
            RuntimeError: If msgpack is not installed.
            ConnectionError: If the daemon socket is unreachable.
        """
        if self._use_json_fallback:
            raise RuntimeError("msgpack not installed — cannot send hot-reload IPC")

        request = {
            "id": 1,
            "request": {
                "method": "models_reload",
                "params": {"model_id": model_id},
            },
        }
        logger.debug("IPC → models_reload(model_id=%s)", model_id)
        response = self._send_recv(request)
        if "error" in response:
            raise RuntimeError(f"Daemon returned error: {response['error']}")
        return response.get("ok", {})

    def close(self) -> None:
        """Close the underlying socket."""
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None


# ── RetrainPipeline ───────────────────────────────────────────────────────────


class RetrainPipeline:
    """Full sieve retrain pipeline.

    Instantiate with the desired configuration, then call :meth:`run`.

    Args:
        epochs: Number of training epochs.
        min_records: Minimum feedback records required (pad with synthetic if fewer).
        output_dir: Directory to write the deployed ONNX model.
        quantize: If ``True`` (default), apply INT8 dynamic quantization.
        reload_daemon: If ``True`` (default), send hot-reload IPC to the daemon.
        daemon_sock: Path to the daemon Unix socket.
        dry_run: Train and validate but skip deployment and hot-reload.
        experiment_name: MLflow experiment name.
        feedback_dir: Directory to read feedback JSONL files from.
    """

    def __init__(
        self,
        epochs: int = 10,
        min_records: int = 200,
        output_dir: Path = Path("~/.gristmill/models").expanduser(),
        quantize: bool = True,
        reload_daemon: bool = True,
        daemon_sock: str = str(Path("~/.gristmill/gristmill.sock").expanduser()),
        dry_run: bool = False,
        experiment_name: str = "sieve-retrain",
        feedback_dir: Path = Path("~/.gristmill/feedback").expanduser(),
    ) -> None:
        self.epochs = epochs
        self.min_records = min_records
        self.output_dir = Path(output_dir).expanduser()
        self.quantize = quantize
        self.reload_daemon = reload_daemon
        self.daemon_sock = daemon_sock
        self.dry_run = dry_run
        self.experiment_name = experiment_name
        self.feedback_dir = Path(feedback_dir).expanduser()

    def run(self) -> dict:
        """Execute the full retrain pipeline.

        Returns:
            Outcome dict with keys: ``success``, ``onnx_path``, ``metrics``,
            ``timestamp``, and ``error`` (only on failure).
        """
        start_time = time.time()
        outcome: dict = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "success": False,
            "onnx_path": None,
            "metrics": {},
            "error": None,
            "dry_run": self.dry_run,
        }

        try:
            # ── Step 1: Load feedback ─────────────────────────────────────────
            from gristmill_ml.datasets.feedback import FeedbackDataset

            logger.info("Step 1/7 — Loading feedback (min_records=%d) …", self.min_records)
            dataset = FeedbackDataset(
                feedback_dir=self.feedback_dir,
                split="train",
                min_records=self.min_records,
            )
            record_count = len(dataset)
            logger.info("Loaded %d training records", record_count)
            outcome["metrics"]["record_count"] = record_count

            # ── Step 2: Train ─────────────────────────────────────────────────
            from gristmill_ml.training.sieve_trainer import SieveTrainer
            from gristmill_ml.experiments.tracking import ExperimentTracker

            logger.info(
                "Step 2/7 — Training SieveTrainer (epochs=%d, experiment=%s) …",
                self.epochs,
                self.experiment_name,
            )
            trainer = SieveTrainer(
                feedback_dir=self.feedback_dir,
                output_dir=self.output_dir,
                experiment_name=self.experiment_name,
            )
            train_result = trainer.train(epochs=self.epochs)
            outcome["metrics"]["best_val_accuracy"] = train_result.best_val_accuracy
            outcome["metrics"]["best_epoch"] = train_result.best_epoch
            logger.info(
                "Training complete: val_acc=%.4f (epoch %d)",
                train_result.best_val_accuracy,
                train_result.best_epoch,
            )

            # ── Step 3: Export ────────────────────────────────────────────────
            from gristmill_ml.export.onnx_export import OnnxExporter

            self.output_dir.mkdir(parents=True, exist_ok=True)
            tmp_onnx = self.output_dir / "intent-classifier-v1.candidate.onnx"

            logger.info(
                "Step 3/7 — Exporting ONNX (quantize=%s) → %s …", self.quantize, tmp_onnx
            )
            onnx_path = OnnxExporter.export_classifier(
                trainer.model, tmp_onnx, quantize=self.quantize
            )
            outcome["metrics"]["onnx_size_bytes"] = onnx_path.stat().st_size
            logger.info("Exported: %s (%d bytes)", onnx_path, onnx_path.stat().st_size)

            # ── Step 4: Validate ──────────────────────────────────────────────
            import numpy as np
            from gristmill_ml.export.validate import validate_classifier_parity
            from gristmill_ml.training.sieve_trainer import FEATURE_DIM

            logger.info("Step 4/7 — Validating ONNX parity …")
            dummy = np.random.randn(32, FEATURE_DIM).astype(np.float32)
            validation = validate_classifier_parity(trainer.model, onnx_path, dummy)

            # validate_classifier_parity returns a dict (see validate.py line 107)
            passed = validation["passed"] if isinstance(validation, dict) else validation.passed
            max_diff = validation["max_diff"] if isinstance(validation, dict) else validation.max_diff
            outcome["metrics"]["parity_max_diff"] = max_diff
            outcome["metrics"]["parity_passed"] = passed

            if not passed:
                raise RuntimeError(
                    f"ONNX parity check failed: max_diff={max_diff:.6f}"
                )
            logger.info("Parity check passed (max_diff=%.8f)", max_diff)

            if self.dry_run:
                logger.info("Dry run — skipping deployment and hot-reload.")
                onnx_path.unlink(missing_ok=True)
                outcome["success"] = True
                outcome["metrics"]["duration_seconds"] = round(time.time() - start_time, 2)
                self._write_log(outcome)
                return outcome

            # ── Step 5: Deploy ────────────────────────────────────────────────
            deploy_path = self.output_dir / "intent-classifier-v1.onnx"
            logger.info("Step 5/7 — Deploying %s → %s …", onnx_path, deploy_path)
            shutil.copy2(str(onnx_path), str(deploy_path))
            onnx_path.unlink(missing_ok=True)  # remove candidate copy
            outcome["onnx_path"] = str(deploy_path)
            logger.info("Deployed: %s", deploy_path)

            # ── Step 6: Hot-reload ────────────────────────────────────────────
            if self.reload_daemon:
                logger.info(
                    "Step 6/7 — Sending hot-reload IPC to daemon (%s) …", self.daemon_sock
                )
                ipc = DaemonIpcClient(self.daemon_sock)
                try:
                    resp = ipc.models_reload("sieve")
                    logger.info("Daemon hot-reload response: %s", resp)
                    outcome["metrics"]["daemon_reload"] = "ok"
                except RuntimeError as exc:
                    # msgpack not installed warning was already logged in __init__
                    logger.warning("Hot-reload skipped: %s", exc)
                    outcome["metrics"]["daemon_reload"] = "skipped"
                except (ConnectionError, OSError) as exc:
                    logger.warning("Daemon not reachable for hot-reload: %s", exc)
                    outcome["metrics"]["daemon_reload"] = f"unreachable: {exc}"
                finally:
                    ipc.close()
            else:
                logger.info("Step 6/7 — Skipping daemon hot-reload (--no-reload).")
                outcome["metrics"]["daemon_reload"] = "disabled"

            # ── Step 7: Log ───────────────────────────────────────────────────
            outcome["success"] = True
            outcome["metrics"]["duration_seconds"] = round(time.time() - start_time, 2)
            logger.info("Step 7/7 — Writing retrain log …")
            self._write_log(outcome)
            logger.info(
                "Retrain pipeline complete in %.1fs", outcome["metrics"]["duration_seconds"]
            )

        except Exception as exc:
            outcome["error"] = str(exc)
            outcome["metrics"]["duration_seconds"] = round(time.time() - start_time, 2)
            logger.exception("Retrain pipeline failed: %s", exc)
            self._write_log(outcome)

        return outcome

    def _write_log(self, outcome: dict) -> None:
        """Append *outcome* as a JSON line to ``~/.gristmill/retrain_log.jsonl``."""
        log_path = Path("~/.gristmill/retrain_log.jsonl").expanduser()
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a") as f:
                f.write(json.dumps(outcome) + "\n")
            logger.debug("Retrain log written: %s", log_path)
        except OSError as exc:
            logger.warning("Could not write retrain log: %s", exc)


# ── CLI entry-point ───────────────────────────────────────────────────────────


def main() -> None:  # pragma: no cover
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="GristMill Sieve retrain pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument(
        "--min-records", type=int, default=200, help="Minimum feedback records"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("~/.gristmill/models").expanduser(),
        help="ONNX model output directory",
    )
    parser.add_argument(
        "--no-quantize", action="store_true", help="Export fp32 instead of INT8"
    )
    parser.add_argument(
        "--no-reload",
        action="store_true",
        help="Skip daemon hot-reload after deployment",
    )
    parser.add_argument(
        "--daemon-sock",
        type=str,
        default=str(Path("~/.gristmill/gristmill.sock").expanduser()),
        help="Path to daemon Unix socket",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Train and validate but do not deploy or hot-reload",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="sieve-retrain",
        help="MLflow experiment name",
    )
    args = parser.parse_args()

    pipeline = RetrainPipeline(
        epochs=args.epochs,
        min_records=args.min_records,
        output_dir=args.output_dir,
        quantize=not args.no_quantize,
        reload_daemon=not args.no_reload,
        daemon_sock=args.daemon_sock,
        dry_run=args.dry_run,
        experiment_name=args.experiment_name,
    )
    outcome = pipeline.run()
    if outcome["success"]:
        print(f"SUCCESS — model deployed to {outcome['onnx_path']}")
    else:
        print(f"FAILED — {outcome['error']}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
