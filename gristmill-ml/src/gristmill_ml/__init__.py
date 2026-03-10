"""GristMill ML Shell.

Python trains; Rust runs.  This package provides:
- :mod:`gristmill_ml.training` — model training (SieveTrainer, NerTrainer, EmbedderTrainer)
- :mod:`gristmill_ml.datasets` — feedback JSONL loading and augmentation
- :mod:`gristmill_ml.export` — PyTorch → ONNX INT8 export and parity validation
- :mod:`gristmill_ml.experiments` — MLflow experiment tracking wrapper
- :mod:`gristmill_ml.core` — re-export of the PyO3 native module (when built)
"""

__version__ = "0.1.0"
