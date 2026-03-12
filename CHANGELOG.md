# Changelog

All notable changes to GristMill are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
GristMill uses [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

### Added
- Comprehensive README.md for all 14 components
- GitHub CI workflows: Rust, TypeScript, Python, Release, Docker
- GitHub issue/PR templates
- Dockerfile with multi-stage build (Rust daemon + TypeScript shell)
- `POST /events` route on dashboard server (was previously 404)
- `MockBridge.recall()` now matches individual query words against content and tags
- `/api/metrics/budget` returns `{ status: "no_data" }` instead of HTTP 204 when no budget data cached

### Fixed
- `plugins.ts`: invalid Fastify route pattern `/:name(*)` replaced with `/*`
- `MockBridge.recall()`: single-string substring match replaced with word-level match including tags
- `/api/metrics/budget`: returns 200 with placeholder body instead of empty 204

---

## [0.2.0] - 2026-03-01

### Added
- `grist-core`: full config wiring — `GristMillConfig` now flows to all subsystems
- `grist-core/src/embedder.rs`: `GrindersEmbedder` bridges MiniLM-L6-v2 ONNX to `grist-ledger`
- `GrindersEmbedder` falls back to `ZeroEmbedder` when model file absent (daemon always starts)

### Fixed
- `grist-core` tests: isolated `tempfile::tempdir()` per test to prevent `sled` lock conflicts in parallel test runs
- `StepType::Local` → `StepType::LocalMl` (pre-existing test typo)
- `Pipeline::new(id, steps)` → `Pipeline::new(id).with_step(step)` (pre-existing test API mismatch)

---

## [0.1.0] - 2026-01-15

### Added
- Initial implementation of all 11 Rust crates: `grist-event`, `grist-sieve`, `grist-grinders`, `grist-millwright`, `grist-ledger`, `grist-hammer`, `grist-bus`, `grist-config`, `grist-core`, `grist-ffi`, `gristmill-daemon`
- Python ML package (`gristmill-ml`) with `SieveTrainer`, ONNX export, feedback dataset loader
- TypeScript shell (`gristmill-integrations`) with HTTP hopper, Bell Tower, dashboard API, plugin system
- `MockBridge` and `IpcBridge` for development without native FFI
- Three-tier memory system (hot `sled` → warm `SQLite FTS5 + usearch` → cold `zstd` JSONL)
- LLM escalation gateway with semantic cache (cosine ≥ 0.92) and token budget enforcement
- Closed learning loop: Sieve feedback → Python retraining → ONNX hot-reload

[Unreleased]: https://github.com/OWNER/GristMill/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/OWNER/GristMill/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/OWNER/GristMill/releases/tag/v0.1.0
