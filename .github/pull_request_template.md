## Summary

<!-- What does this PR do? Why? -->

Closes #<!-- issue number -->

## Changes

<!-- List the key changes. One bullet per file/module if helpful. -->

-
-

## Domain checklist

<!-- GristMill has strict language-domain rules. Confirm yours are respected. -->

- [ ] No inference or routing logic added to TypeScript or Python production code
- [ ] No business logic added to `grist-ffi` (bridges must stay thin)
- [ ] Sieve `triage()` hot path still <5 ms — run latency regression if Sieve was touched:
      `cargo test -p grist-sieve --release -- --include-ignored latency`
- [ ] Feature vectors in sync between `grist-sieve/src/features.rs` and `sieve_trainer.py` (if either changed)
- [ ] ONNX export validated (`gristmill-validate`) if a new model was added

## Tests

- [ ] `cargo fmt --check` + `cargo clippy -- -D warnings` + `cargo test` pass
- [ ] `pnpm lint` + `pnpm test` pass
- [ ] `pytest` passes (Python changes only)
- [ ] New behaviour is covered by tests

## Documentation

- [ ] Relevant READMEs updated if public API changed
- [ ] `CHANGELOG.md` entry added under `[Unreleased]`

## Notes for reviewer

<!-- Anything the reviewer should pay special attention to? -->
