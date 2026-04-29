---
phase: 01-stabilize-i-bug-fixes-and-guardrails
plan: 04
subsystem: testing
tags: [pytest, pytest-cov, unit-tests, time-signature, duration-validator]

requires:
  - phase: 01-stabilize-i-bug-fixes-and-guardrails
    provides: "Plan 01-01 __main__ guard making music_gen.py side-effect-free on import"
provides:
  - "pytest skeleton (tests/ directory, conftest.py sys.path shim, __init__.py)"
  - "dev-requirements.txt pinning pytest>=8.0 and pytest-cov>=5.0"
  - "95 passing unit tests for the four ROADMAP-named pure functions"
  - "Regression-detection baseline for Phase 2 timesig-registry refactor"
affects: [02-stabilize-ii, 03-package-extraction, testing, regression-suite]

tech-stack:
  added: [pytest>=8.0, pytest-cov>=5.0]
  patterns:
    - "Pure-function unit tests pinned to verified source bodies (no isinstance-only weakening)"
    - "sys.path shim conftest.py — intentional throwaway scaffolding pending Phase 3 pyproject.toml"
    - "parametrize-heavy style to maximize assertion density per LOC"

key-files:
  created:
    - dev-requirements.txt
    - tests/__init__.py
    - tests/conftest.py
    - tests/test_time_signature.py
    - tests/test_duration_validator.py
  modified: []

key-decisions:
  - "Use dev-requirements.txt (throwaway) instead of pyproject.toml — proper packaging lands in Phase 3"
  - "conftest.py sys.path shim instead of editable install — same reason, throwaway"
  - "Assertions pin current behavior including the cosmetic if/else in verify_beat_pattern (both branches return len(pattern) == numerator) — Phase 2 registry refactor must preserve this exactly"
  - "NoteValue exact numeric values pinned (WHOLE=4.0, QUARTER=1.0, etc.) so future duration-validator refactors can't silently drift the rhythmic grid"

patterns-established:
  - "Pinning tests: every assertion uses `is True`/`is False` or `== <exact value>`, never `isinstance(result, bool)`"
  - "Test class per function under test (TestVerifyPatternForTimeSignature, TestVerifyBeatPattern, TestValidateMeasures, TestGetSuggestedDuration, TestGetValidDuration, TestNoteValue)"
  - "No audio/FS dependencies in unit tests — no librosa, no pedalboard, no pydub, no midi2audio imports in test modules"

requirements-completed: [R-Q2]

duration: ~15 min
completed: 2026-04-08
---

# Phase 01 Plan 04: Pytest Skeleton and Pure-Function Tests Summary

**pytest skeleton with 95 passing unit tests pinning verify_pattern_for_time_signature, verify_beat_pattern, validate_measures, and DurationValidator behavior in 3 seconds**

## Performance

- **Duration:** ~15 min
- **Started:** 2026-04-08
- **Completed:** 2026-04-08
- **Tasks:** 3
- **Files created:** 5
- **Files modified:** 0
- **Test runtime:** 2.75s pytest runtime, 3.80s wall with /usr/bin/time (well under the 10s ROADMAP target)

## Accomplishments

- Landed a working pytest skeleton (`tests/` with `__init__.py`, `conftest.py`, two test modules) that executes `pytest tests/ -q` → `95 passed in 2.75s`
- Pinned the three time-signature helper functions in `music_gen.py` with 46 concrete-assertion tests across 2/4, 3/4, 4/4, 5/4, 6/8, 12/8
- Pinned `DurationValidator` with 49 tests covering all 7 supported time signatures × 4 layer types, plus exact suggested-duration values and remaining-beats clamping
- Shipped `dev-requirements.txt` so future contributors get `pytest` + `pytest-cov` with one install command (throwaway until Phase 3 pyproject.toml)
- R-Q2 (initial test coverage for pure functions) satisfied

## Task Commits

Each task was committed atomically:

1. **Task 1: Create dev-requirements.txt and tests/ skeleton** — `fbb830f` (chore)
2. **Task 2: Tests for verify_pattern_for_time_signature, verify_beat_pattern, validate_measures** — `aedef2d` (test)
3. **Task 3: Tests for DurationValidator** — `5a4dd13` (test)

## Files Created

- `dev-requirements.txt` — Phase 1 scaffolding: `-r requirements.txt` + `pytest>=8.0` + `pytest-cov>=5.0`. Phase 3 replaces with `pyproject.toml [project.optional-dependencies].dev`.
- `tests/__init__.py` — empty package marker.
- `tests/conftest.py` — sys.path shim inserting repo root so `import music_gen` and `import enhanced_duration_validator` work without an editable install. Delete in Phase 3.
- `tests/test_time_signature.py` — 46 tests. 3 `TestX` classes: `TestVerifyPatternForTimeSignature` (22 parametrized cases across 6/8, 12/8, 4/4, 3/4, 2/4, 5/4 default), `TestVerifyBeatPattern` (9 cases pinning the "len == numerator" effective rule including the cosmetic-if compound path), `TestValidateMeasures` (9 cases pinning the false-only-when-compound-odd-or-2/4-odd rule, including the 4/4-accepts-zero and empty-dict edge cases).
- `tests/test_duration_validator.py` — 49 tests. 3 `TestX` classes: `TestGetSuggestedDuration` (28 parametrized cases + 8 exact-value assertions), `TestGetValidDuration` (remaining-beats clamping + exact 4/4-chord closest-match pinning), `TestNoteValue` (enum membership + numeric-value pins).

## Test-Suite Tree

```
tests/
├── __init__.py
├── conftest.py
├── test_duration_validator.py
└── test_time_signature.py
```

## pytest Output

```
$ .venv/bin/python -m pytest tests/ -q
........................................................................ [ 75%]
.......................                                                  [100%]
95 passed, 1 warning in 2.75s
```

(The warning is a pydub `DeprecationWarning: 'audioop' is deprecated` from transitive import — not from test code.)

## Decisions Made

- **Throwaway scaffolding is fine.** Both `dev-requirements.txt` and the `conftest.py` sys.path shim are scheduled for deletion in Phase 3 when `pyproject.toml` + editable install land. Building them "properly" now would duplicate Phase 3 work.
- **Pin exact values, not shapes.** Every `DurationValidator` test asserts concrete floats (e.g. `== NoteValue.QUARTER.value`) rather than `isinstance(result, float)`. Shape-only assertions would let Phase 2 refactors silently drift behavior.
- **Preserve cosmetic-if branch of verify_beat_pattern.** The tests encode that the current compound branch returns `len(pattern) == numerator` (not `numerator/2` as the wrapping `if denominator == 8 and numerator % 3 == 0:` suggests). Phase 2 MUST NOT "fix" this to numerator/2 — if that's desired behavior it becomes a new plan with its own tests.

## Deviations from Plan

### Environment Setup (Rule 3 - Blocking)

**1. Created Python venv and installed runtime + test dependencies**
- **Found during:** Task 1 pre-verification
- **Issue:** System Python has no `pytest`, no `music21`, no `midiutil`, no `pydub`, no `librosa`, etc. PEP 668 (`externally-managed-environment`) blocks system-wide `pip install`. Without these, `import music_gen` fails at collection time and Task 2 cannot verify.
- **Fix:** Created `.venv/` at repo root via `python3 -m venv .venv` and installed `pytest pytest-cov music21 midiutil pydub midi2audio pedalboard python-json-logger librosa` into it. `.venv` is already in the repo's `.gitignore`, so nothing leaks into commits.
- **Files modified:** None (venv is gitignored scaffolding, not a project artifact)
- **Verification:** `.venv/bin/python -c "import music_gen; print('ok')"` succeeds with no generation side effects (validates Plan 01-01's `__main__` guard is in place).
- **Committed in:** N/A — environment, not code.

---

**Total deviations:** 1 infra bootstrap (Rule 3 - blocking)
**Impact on plan:** Zero — the plan's `action` block explicitly noted a sandboxed-environment fallback of `pip install pytest pytest-cov` only. We went further and installed the full runtime stack so `import music_gen` works, which is strictly required for Task 2/3.

## Issues Encountered

- `pip install` on system Python hit `error: externally-managed-environment` (PEP 668 on Debian-family). Resolved by using a local venv — see deviations above.
- First `python` command failed with `command not found`; only `python3` is on PATH. All subsequent commands used `.venv/bin/python` explicitly.

## Known Stubs

None. All tests are executable, all assertions are concrete, and there are no `pass`/TODO placeholders.

## User Setup Required

None — no external services, no env vars, no API keys.

For contributors: install dev deps with `pip install -r dev-requirements.txt` (or use a venv if your system Python is PEP-668-managed). Then `pytest tests/ -q` should report 95 passed.

## Next Phase Readiness

- Phase 2 (config + timesig registry) can now refactor the three time-signature helpers into a `TimeSignatureRegistry` with confidence — any behavior drift will be caught by `tests/test_time_signature.py`.
- Phase 3 (package extraction) can refactor `DurationValidator` and move it into `src/musicgen/` — `tests/test_duration_validator.py` pins the public surface.
- Phase 3 deletion targets: `dev-requirements.txt` (→ `pyproject.toml [project.optional-dependencies].dev`), `tests/conftest.py` sys.path shim (→ `pip install -e .`). The tests themselves stay.
- R-Q2 initial coverage box is satisfied. Future phases should grow coverage on samplers and pattern parsers as they refactor them.

## Self-Check: PASSED

- FOUND: dev-requirements.txt
- FOUND: tests/__init__.py
- FOUND: tests/conftest.py
- FOUND: tests/test_time_signature.py
- FOUND: tests/test_duration_validator.py
- FOUND commit: fbb830f (Task 1)
- FOUND commit: aedef2d (Task 2)
- FOUND commit: 5a4dd13 (Task 3)
- VERIFIED: `pytest tests/ -q` → 95 passed in 2.75s

---
*Phase: 01-stabilize-i-bug-fixes-and-guardrails*
*Plan: 04*
*Completed: 2026-04-08*
