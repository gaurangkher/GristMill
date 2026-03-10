# Contributing to GristMill

Thank you for helping make GristMill better. This document covers everything you need to get started.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Domain Rules (Read First)](#domain-rules-read-first)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Submitting a Pull Request](#submitting-a-pull-request)
- [Reporting Bugs](#reporting-bugs)
- [Requesting Features](#requesting-features)

---

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating you agree to uphold it.

---

## Getting Started

1. **Fork** the repository on GitHub
2. **Clone** your fork: `git clone https://github.com/YOUR_USERNAME/GristMill`
3. **Add upstream**: `git remote add upstream https://github.com/OWNER/GristMill`
4. **Create a branch**: `git checkout -b feat/my-feature`

---

## Development Setup

### Prerequisites

| Tool | Min Version | Install |
|------|-------------|---------|
| Rust + Cargo | 1.80 | `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \| sh` |
| Node.js | 18 | [nodejs.org](https://nodejs.org) |
| pnpm | 9 | `npm install -g pnpm` |
| Python | 3.10 | [python.org](https://python.org) |

### Rust core

```bash
cd gristmill-core
cargo build
cargo test
```

### TypeScript shell

```bash
cd gristmill-integrations
pnpm install
GRISTMILL_MOCK_BRIDGE=1 pnpm dev   # no daemon required
pnpm test
pnpm lint
```

### Python ML package

```bash
cd gristmill-ml
pip install -e ".[dev]"
pytest tests/
```

---

## Domain Rules (Read First)

GristMill enforces strict language-domain separation. PRs that cross these boundaries will be asked to refactor:

| Domain | Language | Rule |
|--------|----------|------|
| Event triage, ML inference, DAG scheduling, memory | **Rust** | No inference logic in TypeScript or Python |
| Model training, fine-tuning, ONNX export | **Python** | No production inference in Python |
| Channel adapters, notifications, dashboard, plugins | **TypeScript** | No business logic — delegate to Rust via bridge |

The FFI surface lives **only** in `grist-ffi`. Keep `pyo3_bridge.rs` and `napi_bridge.rs` thin.

---

## Making Changes

### Rust (`gristmill-core`)

- The Sieve `triage()` hot path **must remain <5 ms p99** — never add blocking work to it.
- Use `tokio` for async I/O, `rayon` for CPU-parallel work. Do not mix runtimes carelessly.
- All tensors use `ndarray`; prefer zero-copy views over clones.
- Use `tracing::info!` for logging — never `println!` in library crates.
- Use `metrics::counter!` / `metrics::histogram!` for instrumentation.
- New public types that cross boundaries must serialize to/from `GristEvent`.

### Python (`gristmill-ml`)

- Keep feature vectors in `sieve_trainer.py` exactly in sync with `grist-sieve/src/features.rs` (392 dims).
- All models must be exportable to ONNX INT8 and validated with `export/validate.py`.
- Do not commit raw model weights — use `.gitignore` patterns.
- Experiment results go in MLflow / W&B, not in the repo.

### TypeScript (`gristmill-integrations`)

- Use `pnpm`, not npm or yarn.
- All processing delegates to `GristMillBridge` — TypeScript does I/O only.
- Use `IBridge` interface in tests; never import real Rust bridge in unit tests.
- New hopper adapters normalise to `GristEventInit` before calling `bridge.submit()`.

---

## Testing

### Required before submitting

```bash
# Rust
cd gristmill-core
cargo fmt --check          # formatting
cargo clippy -- -D warnings # lints
cargo test                  # all tests

# Sieve latency regression (p99 must be < 5 ms)
cargo test -p grist-sieve -- --include-ignored latency

# TypeScript
cd gristmill-integrations
pnpm lint                   # tsc --noEmit
pnpm test                   # vitest

# Python (when relevant)
cd gristmill-ml
pytest tests/
python -m gristmill_ml.export.validate
```

All CI checks must pass before a PR is merged.

---

## Submitting a Pull Request

1. **Rebase** onto `main` before opening: `git rebase upstream/main`
2. **Keep PRs focused** — one feature or bug fix per PR
3. **Write a clear description** explaining *why*, not just *what*
4. **Reference issues**: `Closes #123`
5. **Update relevant READMEs** if public APIs changed
6. **Add tests** for new behaviour

### PR Checklist

- [ ] `cargo fmt`, `cargo clippy`, `cargo test` pass
- [ ] `pnpm lint`, `pnpm test` pass
- [ ] No business logic added to TypeScript or Python production code
- [ ] Sieve hot path still <5 ms (run latency regression if Sieve was touched)
- [ ] Feature vectors in sync between Rust and Python (if either was changed)
- [ ] ONNX export validated (if new model added)
- [ ] Documentation updated

---

## Reporting Bugs

Use the [Bug Report](.github/ISSUE_TEMPLATE/bug_report.md) template. Include:

- GristMill version / git commit
- OS and Rust/Node.js/Python versions
- Minimal reproduction steps
- Expected vs actual behaviour
- Relevant logs (`RUST_LOG=debug` output)

---

## Requesting Features

Use the [Feature Request](.github/ISSUE_TEMPLATE/feature_request.md) template. Describe the use case first — not the implementation.

---

## Questions?

Open a [Discussion](../../discussions) rather than an issue for general questions.
