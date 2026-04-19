---
phase: 04-renderer-mixer-annotator-beats-extraction
plan: "00"
subsystem: infrastructure
tags: [phase-4, infrastructure, dependencies, test-scaffolds, mido, pytest-markers]
dependency_graph:
  requires: []
  provides:
    - mido>=1.3.3 runtime dep in pyproject.toml
    - pytest slow/integration markers declared
    - 6 Phase 4 test-file scaffolds (Wave 1–6 will replace stubs)
  affects:
    - pyproject.toml
    - tests/test_beats.py
    - tests/test_renderer.py
    - tests/test_mixer.py
    - tests/test_annotator.py
    - tests/test_no_bare_random_in_package.py
    - tests/test_integration_full_generation.py
tech_stack:
  added:
    - mido==1.3.3 (MIDI parsing for MIDI-tick extraction in beats.py)
  patterns:
    - pytest.skip(allow_module_level=True) for wave-scoped stub files
key_files:
  created:
    - tests/test_beats.py
    - tests/test_renderer.py
    - tests/test_mixer.py
    - tests/test_annotator.py
    - tests/test_no_bare_random_in_package.py
    - tests/test_integration_full_generation.py
  modified:
    - pyproject.toml
decisions:
  - "mido>=1.3.3 pinned as direct dep (RESEARCH correction #3: not a transitive dep of midi2audio)"
  - "No --strict-markers in addopts — rollout is additive; strict mode is Phase 6+ concern (plan directive)"
  - "Task 1 (pip install) generates no git commit — venv state is untracked by design"
metrics:
  duration: "~2 minutes (125 seconds)"
  completed: "2026-04-19"
  tasks_completed: 3
  files_modified: 7
---

# Phase 04 Plan 00: Wave 0 Infrastructure Summary

**One-liner:** mido 1.3.3 added as explicit runtime dep + slow/integration pytest markers declared + 6 Phase 4 test-file stubs created (371 baseline tests preserved, 6 skipped added).

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 0 | pyproject.toml — mido dep + markers | bde1e29 | pyproject.toml |
| 1 | pip install -e '.[dev]' — refresh venv | (no commit — venv side-effect) | .venv/ |
| 2 | Create 6 test-file scaffolds | 01f3290 | tests/test_beats.py, tests/test_renderer.py, tests/test_mixer.py, tests/test_annotator.py, tests/test_no_bare_random_in_package.py, tests/test_integration_full_generation.py |

## Verification Results

### pyproject.toml changes

```diff
+    # RESEARCH correction #3: mido is NOT a transitive dep of midi2audio (which declares zero Python-level requirements); beats.py needs it for MIDI-tick extraction.
+    "mido>=1.3.3",
```

```toml
+markers = [
+    "slow: FluidSynth rendering tests — requires fluidsynth binary on PATH (deselect with '-m \"not slow\"')",
+    "integration: end-to-end tests requiring system dependencies",
+]
```

### pip show mido

```
Name: mido
Version: 1.3.3
Summary: MIDI Objects for Python
Requires: packaging
Required-by: musicgen
```

### mido import verification

```
1.3.3          # mido.version_info
bpm2tempo OK   # mido.bpm2tempo(120) succeeds
tick2second OK # mido.tick2second(480, 480, 500000) succeeds
```

### Full test suite

```
371 passed, 6 skipped, 2 warnings in 0.96s
```

Baseline 371 tests unchanged. 6 new skipping stubs added. Zero new failures.

### 6 stubs discoverable by pytest

```
$ python -m pytest tests/test_beats.py tests/test_renderer.py tests/test_mixer.py \
    tests/test_annotator.py tests/test_no_bare_random_in_package.py \
    tests/test_integration_full_generation.py -q
6 skipped in 0.01s
```

## Deviations from Plan

### Minor deviations

**1. [Rule 1 - Bug] mido has no `__version__` attribute**
- **Found during:** Task 1 verification
- **Issue:** Plan's verify command used `mido.__version__` which raises `AttributeError` in mido 1.3.3 (the library uses `mido.version_info` as a `packaging.version.Version` object instead)
- **Fix:** Used `mido.version_info` with `packaging.version.Version` comparison in verification; confirmed `pip show mido` shows `Version: 1.3.3`
- **Files modified:** None (verification-only adjustment; no code change needed)
- **Commit:** N/A

**2. [Rule 3 - Blocking] System Python is PEP-668 managed**
- **Found during:** Task 1
- **Issue:** `pip install` on system Python failed with `externally-managed-environment` (expected per STATE.md D-01: `.venv/` is required)
- **Fix:** Used `.venv/bin/pip install -e '.[dev]'` — the `.venv/` was already established in Plan 03-01
- **Files modified:** None (venv path selection, no code change)
- **Commit:** N/A

## Known Stubs

All 6 test files are intentional stubs — each contains `pytest.skip(allow_module_level=True)`. This is by design per the plan: later waves replace each file entirely.

| File | Line | Reason |
|------|------|--------|
| tests/test_beats.py | 9 | Wave 0 stub; Plan 04-01 replaces |
| tests/test_renderer.py | 9 | Wave 0 stub; Plan 04-02 replaces |
| tests/test_mixer.py | 9 | Wave 0 stub; Plan 04-03 replaces |
| tests/test_annotator.py | 9 | Wave 0 stub; Plan 04-04 replaces |
| tests/test_no_bare_random_in_package.py | 9 | Wave 0 stub; Plan 04-05 replaces |
| tests/test_integration_full_generation.py | 9 | Wave 0 stub; Plan 04-06 replaces |

These stubs do not prevent this plan's goal from being achieved — the goal is scaffold creation for downstream waves, not functional testing.

## Threat Flags

None — no new network endpoints, auth paths, file access patterns, or schema changes introduced. `pyproject.toml` changes are dependency metadata only (T-04-00-01/T-04-00-02 accepted per plan threat model).

## Self-Check: PASSED

- pyproject.toml contains `mido>=1.3.3`: FOUND
- pyproject.toml contains `markers =` with `slow` and `integration`: FOUND
- tests/test_beats.py: FOUND
- tests/test_renderer.py: FOUND
- tests/test_mixer.py: FOUND
- tests/test_annotator.py: FOUND
- tests/test_no_bare_random_in_package.py: FOUND
- tests/test_integration_full_generation.py: FOUND
- Commit bde1e29 (Task 0): FOUND
- Commit 01f3290 (Task 2): FOUND
- pytest suite: 371 passed, 6 skipped — PASSED
