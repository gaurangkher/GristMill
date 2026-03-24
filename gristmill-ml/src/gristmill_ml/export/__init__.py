"""ONNX export, validation, and portable bundle utilities."""

from .bundle import AdapterBundle, BundleManifest

__all__ = [
    "AdapterBundle",
    "BundleManifest",
    "OnnxExporter",
    "validate_classifier_parity",
    "validate_rust_parity",
    "ValidationResult",
]


def __getattr__(name: str):
    if name == "OnnxExporter":
        from .onnx_export import OnnxExporter

        return OnnxExporter
    if name in ("validate_classifier_parity", "validate_rust_parity", "ValidationResult"):
        import importlib

        mod = importlib.import_module(".validate", package=__name__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
