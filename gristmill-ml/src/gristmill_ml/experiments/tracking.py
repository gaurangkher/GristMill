"""MLflow experiment tracking wrapper.

Provides a thin, context-manager-based API over MLflow so that training
modules don't need direct MLflow imports.  Falls back gracefully when
MLflow is not installed or the tracking server is unreachable.
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Thin wrapper around MLflow for experiment tracking.

    All methods are no-ops when MLflow is unavailable (``ImportError``) or
    when the tracking server is unreachable (``MlflowException``).

    Args:
        experiment_name: MLflow experiment name (created if it doesn't exist).
        tracking_uri: MLflow tracking URI.  Defaults to a local file store
            at ``~/.gristmill/mlruns``.
    """

    def __init__(
        self,
        experiment_name: str = "gristmill",
        tracking_uri: str = str(Path("~/.gristmill/mlruns").expanduser()),
    ) -> None:
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self._active_run: Optional[Any] = None
        self._mlflow: Optional[Any] = None

        try:
            import mlflow  # type: ignore[import]

            self._mlflow = mlflow
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
        except ImportError:
            logger.debug("mlflow not installed — experiment tracking disabled")
        except Exception as exc:
            logger.warning("MLflow init failed: %s — tracking disabled", exc)

    @contextlib.contextmanager
    def start_run(self, run_name: str) -> Iterator[None]:
        """Context manager that wraps a training run.

        Usage::

            with tracker.start_run("epoch-10"):
                tracker.log_params({"epochs": 10})
                ...
        """
        if self._mlflow is None:
            yield
            return

        try:
            run = self._mlflow.start_run(run_name=run_name)
            self._active_run = run
        except Exception as exc:
            logger.warning("mlflow.start_run failed: %s", exc)
            yield
            return

        try:
            yield
        finally:
            try:
                self._mlflow.end_run()
            except Exception:
                pass
            self._active_run = None

    def log_params(self, params: dict[str, Any]) -> None:
        """Log a dictionary of hyperparameters."""
        if self._mlflow is None:
            return
        try:
            self._mlflow.log_params({k: str(v) for k, v in params.items()})
        except Exception as exc:
            logger.debug("mlflow.log_params failed: %s", exc)

    def log_metrics(self, metrics: dict[str, float], step: Optional[int] = None) -> None:
        """Log a dictionary of scalar metrics."""
        if self._mlflow is None:
            return
        try:
            self._mlflow.log_metrics(
                {k: float(v) for k, v in metrics.items()},
                step=step,
            )
        except Exception as exc:
            logger.debug("mlflow.log_metrics failed: %s", exc)

    def log_model(
        self,
        model: Any,
        artifact_path: str = "model",
    ) -> None:
        """Log a PyTorch model as an MLflow artifact."""
        if self._mlflow is None:
            return
        try:
            import mlflow.pytorch  # type: ignore[import]

            mlflow.pytorch.log_model(model, artifact_path)
        except Exception as exc:
            logger.debug("mlflow.pytorch.log_model failed: %s", exc)

    def log_onnx(self, onnx_path: Path) -> None:
        """Log an ONNX file as an MLflow artifact."""
        if self._mlflow is None:
            return
        try:
            self._mlflow.log_artifact(str(onnx_path))
        except Exception as exc:
            logger.debug("mlflow.log_artifact (onnx) failed: %s", exc)

    def end_run(self) -> None:
        """Explicitly end the current run (usually handled by context manager)."""
        if self._mlflow is None or self._active_run is None:
            return
        try:
            self._mlflow.end_run()
        except Exception:
            pass
        self._active_run = None
