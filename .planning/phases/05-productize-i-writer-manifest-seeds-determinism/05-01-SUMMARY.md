---
phase: 05-productize-i-writer-manifest-seeds-determinism
plan: "01"
subsystem: test-infrastructure

tags: [phase-5, wave-0, scaffolding, conftest, pytest-addoption, ast-guard, goldens-fixture-dir, d-32, d-36, d-37, d-38, d-39, d-40, d-41, d-42, regen-goldens]

# Dependency graph
requires:
  - phase: 04-renderer-mixer-annotator-beats-extraction
    provides: tests/test_no_bare_random_in_package.py (parametrized package-wide AST guard from Plan 04-05); pytest.skip(allow_module_level=True) stub idiom from Plan 04-00 Wave 0
  - phase: 03-package-skeleton-sampler-generators-extraction
    provides: tests/conftest.py deleted (D-16 closure) — this plan lands the first new conftest.py for --regen-goldens
provides:
  - tests/conftest.py registering --regen-goldens pytest flag (D-32)
  - 6 Wave 0 test stubs (seeds/writer/manifest/split/api/determinism_golden) using pytest.skip(allow_module_level=True)
  - tests/fixtures/determinism/ directory tracked in git via README.md (docs Wave 5's 7-file golden layout)
  - Widened AST bare-random guard allow-list {"Random", "getstate", "setstate"} — permits seeds.save_random_state() under D-20
  - expected_present meta-test set broadened to 15 entries (5 Phase 5 module basenames added) as a forward guard; xfails until Waves 1-4 create the modules
affects:
  - 05-02 (Wave 1 seeds + splits — fills tests/test_seeds.py + tests/test_split.py)
  - 05-03 (Wave 2 config + musicality move — no direct test-file overlap here)
  - 05-04 (Wave 3 writer + manifest — fills tests/test_writer.py + tests/test_manifest.py)
  - 05-05 (Wave 4 api + musicgen.generate — fills tests/test_api.py; also removes the xfail marker from test_package_scan_covers_all_package_modules once api.py lands)
  - 05-06 (Wave 5 determinism goldens — fills tests/test_determinism_golden.py and drops sha256 files into tests/fixtures/determinism/)

# Tech tracking
tech-stack:
  added:
    - pytest_addoption hook — first conftest.py in the project since Phase 3 deleted the old sys.path shim
  patterns:
    - "Wave 0 skip-stub pattern (inherited from Plan 04-00): pytest.skip(allow_module_level=True) with explanatory reason referencing the wave that fills it in"
    - "Forward-guard xfail (new): expected_present lists end-of-phase modules with strict=False xfail that self-clears when the last module lands — documents intent without blocking"

key-files:
  created:
    - tests/conftest.py
    - tests/test_seeds.py
    - tests/test_writer.py
    - tests/test_manifest.py
    - tests/test_split.py
    - tests/test_api.py
    - tests/test_determinism_golden.py
    - tests/fixtures/determinism/README.md
  modified:
    - tests/test_no_bare_random_in_package.py

key-decisions:
  - "Task 2 takes Option A (widen allow-list globally) over Option B (per-file exemption) per D-42 — future consumers of random.getstate/setstate do not need separate guard exemptions"
  - "Meta-test renamed test_package_scan_covers_all_phase4_modules → test_package_scan_covers_all_package_modules to reflect broadened scope beyond Phase 4"
  - "Xfail marker uses strict=False so the test self-clears once the 5 new modules land — Plan 05-05 removes the decorator after api.py is created"
  - "Stub skip messages reference the target Wave and Plan ID explicitly so downstream plans can grep for their placeholder"

patterns-established:
  - "conftest.py pytest_addoption pattern: minimal module, no fixtures, documents the flag's effect + when to use in module docstring (CONTEXT D-32 / RESEARCH Pattern 5)"
  - "Determinism fixtures directory marker: tests/fixtures/determinism/README.md doubles as regeneration docs + git-tracked dir marker so the empty-directory survives and downstream Wave 5 can drop sha256 files in without mkdir churn"
  - "Forward-guard xfail: xfail(strict=False, reason='...') on meta-tests that document end-of-phase shape; self-clears when the reality catches up"

requirements-completed: [R-P1, R-P2, R-P3, R-P4, R-P5, R-P6, R-P7, R-P8, R-Q3]

# Metrics
duration: ~14min
completed: 2026-04-19
---

# Phase 5 Plan 01: Wave 0 Test Scaffolding + AST Guard Widening Summary

**Wave 0 test infrastructure — 7 collectable pytest.skip stubs + tests/conftest.py registering --regen-goldens + tests/fixtures/determinism/ dir marker + AST bare-random guard widened to permit random.getstate/setstate for seeds.save_random_state()**

## Performance

- **Duration:** ~14 min
- **Started:** 2026-04-19T21:02:00Z (approximate — file-creation pace)
- **Completed:** 2026-04-19T21:16:02Z
- **Tasks:** 2
- **Files modified:** 9 (8 created + 1 edited)
- **Commits:** 2 task commits (67a04e4, bc925b5) + 1 plan-metadata commit (pending)

## Accomplishments

- **Unblocked all downstream Phase 5 plans** — every Wave 1+ plan can now grep its test stub and edit-in-place rather than cold-creating a file under implementation context pressure (matches Phase 4 Wave 0 idiom)
- **Zero regressions** — full test suite still green: 504 passed (prior baseline) + 6 new module-level skips + 1 xfail meta-test, with 2 prior skips intact (total: 503 passed, 8 skipped, 1 xfailed — the "1 passed → 1 xfailed" flip on the meta-test is expected and accounted for)
- **`--regen-goldens` pytest flag advertised** — `.venv/bin/pytest --help` now includes the flag; Wave 5 determinism test can request the flag via `request.config.getoption("--regen-goldens")` without further conftest changes
- **AST guard widened for seeds.save_random_state()** — Wave 1 can ship `@contextlib.contextmanager save_random_state()` with `random.getstate()` + `random.setstate(state)` without the guard rejecting it
- **Forward-guard expected_present** — meta-test now documents the end-of-Phase-5 package shape (15 modules: 10 current + 5 Phase 5 additions); xfails with strict=False so it self-clears when Waves 1-4 land, without blocking Wave 0

## Task Commits

1. **Task 1: Create 7 test stubs + conftest.py + fixtures dir + regen-goldens flag** — `67a04e4` (feat)
2. **Task 2: Widen AST guard allow-list + update expected_present set** — `bc925b5` (feat)

## Files Created/Modified

### Created (8)

- `tests/conftest.py` — 22 lines; module-level pytest_addoption hook registers `--regen-goldens` (action="store_true", default=False) with help text pointing at the Wave 5 determinism test. Verbatim from plan spec.
- `tests/test_seeds.py` — 13 lines; `pytest.skip(allow_module_level=True, reason="Wave 1 implements seeds.py (D-36); ...")`. Planned coverage noted in docstring: `derive_sample_seed`, `make_rngs`, `save_random_state`, `assign_split`.
- `tests/test_writer.py` — 13 lines; skips to Wave 3 (Plan 05-04); planned coverage: per-sample directory layout, relative paths, silent-stem concat, sum-of-stems assertion, sentinel invariant.
- `tests/test_manifest.py` — 13 lines; skips to Wave 3 (Plan 05-04); planned coverage: append-under-lock, is_sample_complete sentinel semantics.
- `tests/test_split.py` — 13 lines; skips to Wave 1 (Plan 05-02); planned coverage: assign_split determinism + empirical ratios + invalid-ratio rejection.
- `tests/test_api.py` — 13 lines; skips to Wave 4 (Plan 05-05); planned coverage: Config validation, resume short-circuit, full-layout generation (@pytest.mark.slow), idempotency.
- `tests/test_determinism_golden.py` — 13 lines; skips to Wave 5 (Plan 05-06); planned coverage: 6 SHA-256 goldens (mix + 4 MIDI + sample.json) + fast same-process cross-check.
- `tests/fixtures/determinism/README.md` — 15 lines; documents the 7-file golden layout Wave 5 populates + regeneration command (`pytest -m slow --regen-goldens tests/test_determinism_golden.py`).

### Modified (1)

- `tests/test_no_bare_random_in_package.py` — three precise edits:
  1. **Module docstring** (top of file): allow-list documented from `{Random}` → `{Random, getstate, setstate}` with per-attr rationale (getstate/setstate from seeds.save_random_state D-20).
  2. **`_bare_random_calls` helper**: `attr != "Random"` → `attr not in {"Random", "getstate", "setstate"}`; docstring rewritten to list the three permitted attrs with the D-07/D-17 contract pointer.
  3. **Meta-test**: renamed `test_package_scan_covers_all_phase4_modules` → `test_package_scan_covers_all_package_modules`; docstring updated; `expected_present` set expanded by 5 entries (`seeds.py`, `writer.py`, `manifest.py`, `api.py`, `musicality.py`) with `# Phase 5 additions (added in Waves 1-4):` inline comment; `@pytest.mark.xfail(strict=False, reason="Phase 5 modules ... land in Waves 1-4; ...")` decorator added.

## Decisions Made

- **No deviations from plan.** The plan specified Option A (widen global allow-list) and xfail (Option 2) for the meta-test — both followed verbatim.
- **Renamed meta-test function** (plan-spec requirement, not a deviation): `test_package_scan_covers_all_phase4_modules` → `test_package_scan_covers_all_package_modules` to reflect broadened scope across Phase 3, Phase 4, and Phase 5 modules.
- **All stub skip reasons reference both the Wave and the Plan ID** (e.g., `"Wave 1 implements seeds.py (D-36); this stub exists so Wave 0 test collection succeeds."`) so downstream plans can grep for their placeholder unambiguously.

## Deviations from Plan

None - plan executed exactly as written.

The plan's acceptance criterion for the full-suite count ("504 passed ... PLUS exactly 6 skips") matched observed: 503 passed + 1 xfailed (was previously 1 passed) + 8 skipped (2 prior + 6 new stubs). Net pass count is unchanged (504), and the xfail is the expected-new-behavior on the meta-test.

## Issues Encountered

None — both tasks executed cleanly on first attempt. Full suite green throughout. The pydub ffmpeg RuntimeWarning is pre-existing environmental (no ffmpeg/avconv on dev box) and is unrelated to this plan.

## User Setup Required

None — no external service configuration required. Phase 5 Plan 01 is test-infrastructure only; no runtime deps, no environment variables, no credentials.

## Next Phase Readiness

- **All downstream Phase 5 plans unblocked.** Plans 05-02 through 05-06 can now:
  - Edit their pre-created test stub files (`test_seeds.py`, `test_writer.py`, etc.) rather than creating them cold
  - Reference `request.config.getoption("--regen-goldens")` in `test_determinism_golden.py` with no conftest work needed
  - Drop sha256 files into `tests/fixtures/determinism/` without creating the directory
  - Call `random.getstate()` / `random.setstate(state)` from `seeds.save_random_state()` without tripping the AST guard
- **Xfail removal pointer:** Plan 05-05 MUST remove the `@pytest.mark.xfail(...)` decorator on `test_package_scan_covers_all_package_modules` once `api.py` (the last of the 5 new modules) lands. Plan 05-06 already assumes it passes.
- **No blockers.** Ready for Plan 05-02.

## Self-Check: PASSED

Verification commands:
- `[ -f tests/conftest.py ]` → FOUND
- `[ -f tests/test_seeds.py ] && [ -f tests/test_writer.py ] && [ -f tests/test_manifest.py ] && [ -f tests/test_split.py ] && [ -f tests/test_api.py ] && [ -f tests/test_determinism_golden.py ]` → FOUND (all 6)
- `[ -f tests/fixtures/determinism/README.md ]` → FOUND
- `grep -c "pytest_addoption" tests/conftest.py` → 1 ✓
- `grep -c 'node.func.attr not in {"Random", "getstate", "setstate"}' tests/test_no_bare_random_in_package.py` → 1 ✓
- `grep -c '"seeds.py", "writer.py", "manifest.py", "api.py", "musicality.py"' tests/test_no_bare_random_in_package.py` → 1 ✓
- `grep -c "test_package_scan_covers_all_package_modules" tests/test_no_bare_random_in_package.py` → 1 ✓
- `grep -c "xfail" tests/test_no_bare_random_in_package.py` → 3 ✓
- `.venv/bin/pytest --help | grep regen-goldens` → FOUND
- `.venv/bin/pytest tests/test_no_bare_random_in_package.py -v | tail -1` → "12 passed, 1 xfailed in 0.05s" ✓
- `.venv/bin/pytest tests/ -q | tail -1` → "503 passed, 8 skipped, 1 xfailed, 2 warnings in 2.17s" ✓ (zero regressions)
- `git log --oneline -3 | grep -E "67a04e4|bc925b5"` → both FOUND

---
*Phase: 05-productize-i-writer-manifest-seeds-determinism*
*Completed: 2026-04-19*
