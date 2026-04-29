---
status: partial
phase: 05-productize-i-writer-manifest-seeds-determinism
source: [05-VERIFICATION.md]
started: 2026-04-20T01:32:00Z
updated: 2026-04-20T01:32:00Z
---

## Current Test

[awaiting human testing — both items require a host with the pinned FluidSynth binary on PATH AND non-empty sf/<layer>/*.sf2 pools]

## Tests

### 1. Capture determinism goldens on pinned FluidSynth host
expected: |
  Step 1 (capture):
  ```bash
  .venv/bin/pip install -e ".[dev]"
  .venv/bin/pytest -m slow --regen-goldens tests/test_determinism_golden.py
  ```
  Writes 6 `expected_*.sha256` files + `fluidsynth_version.txt` to `tests/fixtures/determinism/`.

  Step 2 (assert):
  ```bash
  .venv/bin/pytest -m slow tests/test_determinism_golden.py
  ```
  All 6 parametrized cases PASS (MIDI + sample.json hashes assert unconditionally; mix.wav passes when FluidSynth version matches the captured fluidsynth_version.txt).

  Step 3 (idempotence): re-running step 2 produces identical hashes.
result: [pending]

### 2. Commit captured goldens
expected: |
  After step 1 completes:
  ```bash
  git add tests/fixtures/determinism/expected_*.sha256 tests/fixtures/determinism/fluidsynth_version.txt
  git commit -m "test(05): capture determinism goldens (FluidSynth <version>)"
  ```
  7 fixture files become part of the regression baseline. Subsequent CI runs of `pytest -m slow` assert against these committed values.
result: [pending]

## Summary

total: 2
passed: 0
issues: 0
pending: 2
skipped: 0
blocked: 0

## Gaps

(none — both items are routine operator captures, not bugs. R-P8 bit-identity contract is fundamentally operator-captured per 05-06-SUMMARY §"Post-phase operator task". Test infrastructure is complete and ready; runtime data needs FluidSynth-equipped host.)
