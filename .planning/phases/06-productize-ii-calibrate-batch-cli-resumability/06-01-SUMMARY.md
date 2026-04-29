# Plan 06-01 Summary — Wave 0: Test Scaffolding Stubs

**Status:** COMPLETE 2026-04-28
**Commit:** 020fb76

## What landed

Wave 0 test infrastructure for Phase 6, following the Phase 4/5 Wave 0 idiom.

**New test stubs** (all `pytest.skip(allow_module_level=True)`):
- `tests/test_output_mode.py`
- `tests/test_calibrate.py`
- `tests/test_batch.py`
- `tests/test_cli.py`
- `tests/test_integration_batch.py`

**Extended:**
- `tests/test_config.py` — `TestPhase6Fields` class stub for `output_mode` + `count` fields.
- `tests/test_no_bare_random_in_package.py` — `expected_present` set extended with `calibrate.py` and `batch.py`; `@pytest.mark.xfail` added to `test_package_scan_covers_all_package_modules` (removed in Plan 06-04 when both modules land).

## Outcome

Full suite: 690 passed, 7 skipped (5 new stubs + 2 prior), 1 xfailed (meta-test). All Phase 6 test files unblocked for in-place replacement in Waves 1–5.
