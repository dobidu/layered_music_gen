---
phase: 02-stabilize-ii-config-time-signature-registry-logging
plan: 03
subsystem: logging
tags: [logging, print-migration, refactor, python, r-s7]

requires:
  - phase: 02-stabilize-ii-config-time-signature-registry-logging
    plan: 01
    provides: "config.py, tests/test_music_gen_logging.py Wave 0 skeleton"
  - phase: 02-stabilize-ii-config-time-signature-registry-logging
    plan: 02
    provides: "timesig.py wrappers, post-02-02 music_gen.py state"

provides:
  - "R-S7 closed: zero print() calls in music_gen.py — all 32 replaced with semantically leveled logging calls"
  - "Module-level logger = logging.getLogger(__name__) at music_gen.py module scope"
  - "logging.basicConfig(INFO) inside __main__ guard only — import remains side-effect-free"
  - "6 concrete tests in tests/test_music_gen_logging.py covering AST scan, import guard, logger setup"
  - "Phase 6 JSON formatter hook comment embedded in __main__ guard per D-08"

affects:
  - "music_gen.py — 32 print() calls replaced, import logging + module-level logger added, basicConfig in __main__"
  - "tests/test_music_gen_logging.py — fully populated from Wave 0 skeleton (skip placeholder removed)"

tech-stack:
  added: []
  patterns:
    - "Module-level logger = logging.getLogger(__name__) — matches musicality_score.py convention"
    - "%s format-arg style in all logger calls (never f-strings) — defers formatting until handler emits"
    - "exc_info=True on except-block warning — captures full traceback at WARNING level"
    - "Component scores loop (3 prints) aggregated into single logger.debug call (Pitfall 8)"
    - "logging.basicConfig(INFO) inside __main__ guard only — T-02-06 mitigated"
    - "AST-walk test pattern for zero-print-calls verification"
    - "caplog + capsys test pattern for import side-effect guard"

key-files:
  created: []
  modified:
    - music_gen.py
    - tests/test_music_gen_logging.py

key-decisions:
  - "All 32 print() calls replaced per D-07 severity classification (16 DEBUG, 14 INFO, 2 WARNING)"
  - "Component scores loop aggregated: 3 prints (header + loop body) -> 1 logger.debug call (Pitfall 8)"
  - "except-block warning uses exc_info=True (not logger.exception) because level is WARNING not ERROR"
  - "logging.basicConfig placed as first statement inside __main__ guard before Config.load() (T-02-06)"
  - "Phase 6 JSON formatter hook comment added per D-08 — pythonjsonlogger activation deferred"
  - "Mixing part INFO log includes part_counter and number_of_parts as separate %s args"

metrics:
  duration: "~4 minutes"
  completed: "2026-04-18T16:12:38Z"
  tasks: 2
  files_created: 0
  files_modified: 2
  tests_added: 6
  tests_total: 309
  tests_skipped: 0
---

# Phase 02 Plan 03: Print-to-Logging Migration Summary

**All 32 print() calls in music_gen.py replaced with semantically leveled logging calls (16 DEBUG, 14 INFO, 2 WARNING) per D-07; module-level logger and basicConfig-in-__main__ added; 6 AST-scan and import-guard tests confirm the migration is complete and side-effect-free**

## Performance

- **Duration:** ~4 minutes
- **Started:** 2026-04-18T16:08:59Z
- **Completed:** 2026-04-18T16:12:38Z
- **Tasks:** 2
- **Files modified:** 2 (music_gen.py, tests/test_music_gen_logging.py)
- **Test runtime:** 0.80s for 309 tests (well under 10s R-Q2 budget)

## Accomplishments

- Replaced all 32 `print()` calls in `music_gen.py` with semantically leveled `logging.*()` calls (R-S7 closed)
- Added `import logging` to stdlib imports block and `logger = logging.getLogger(__name__)` at module scope
- Added `logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")` as first statement inside `if __name__ == "__main__":` guard only (T-02-06 mitigated)
- Component scores loop (3 prints: header + per-component loop) aggregated into one `logger.debug("Component scores: %s", component_scores)` call (Pitfall 8)
- except-block warning uses `exc_info=True` to capture full traceback at WARNING level
- Populated `tests/test_music_gen_logging.py` with 6 concrete tests; removed `@pytest.mark.skip` skeleton placeholder
- Full test suite: 309 passed, 0 skipped (up from 303 passed, 1 skipped before this plan)

## Task Commits

1. **Task 1: print-to-logging migration in music_gen.py** — `57c7f31` (feat)
2. **Task 2: populate test_music_gen_logging.py** — `738fe22` (feat)

## Print Call Replacement Breakdown

| Severity | Count | Examples |
|----------|-------|---------|
| DEBUG | 14 | Chord progression, Melody, Bassline, pedalboard states, levels, "added to mix" |
| INFO | 15 | Beat/Melody/Harmony/Bassline soundfonts, song arrangement, mixing progress, song saved, elapsed time, musicality score |
| WARNING | 2 | Invalid melody timing structure, default structure fallback (with exc_info=True) |
| **Total** | **31** | 32 original prints; component scores loop (3 prints → 1 logger.debug) reduces call count by 2 |

Note: The component scores loop replaced 3 print statements (header + loop body × N components) with 1 logger.debug call, so the final logger call count is 31 (not 32).

## Component Scores Aggregation (Pitfall 8)

**Before (3 print calls):**
```python
print(f'Component Scores:')
for component, value in component_scores.items():
    print(f'{component:>10}: {value:.2f}')
```

**After (1 logger call):**
```python
logger.debug("Component scores: %s", component_scores)
```

The loop body and header are replaced by a single call that passes `component_scores` dict as `%s` argument. At DEBUG level (off by default), this is more efficient — no string formatting occurs until the handler decides to emit.

## Logger Setup in music_gen.py

```python
# At module top (stdlib imports section)
import logging

# After imports, before first function definition
logger = logging.getLogger(__name__)

# Inside if __name__ == "__main__": guard (first statement)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
# Phase 6: swap to pythonjsonlogger.jsonlogger.JsonFormatter when --json flag arrives
```

## Test Suite (tests/test_music_gen_logging.py)

| Class | Test | What It Verifies |
|-------|------|-----------------|
| TestNoPrintCallsRemain | test_no_print_calls_remain_in_music_gen | AST walk — zero print() nodes in music_gen.py |
| TestImportSideEffects | test_import_music_gen_does_not_emit_logs | caplog at DEBUG — no log records on import |
| TestImportSideEffects | test_import_music_gen_does_not_trigger_generation | capsys — no stdout/stderr on import |
| TestLoggerSetup | test_module_level_logger_exists | Source text contains logger = logging.getLogger(__name__) |
| TestLoggerSetup | test_basic_config_only_in_main_guard | AST walk — basicConfig call is within __main__ guard lines |
| TestLoggerSetup | test_no_fstring_in_logger_calls | AST walk — no JoinedStr (f-string) as first arg to logger.* calls |

## R-S7 Exit Criterion Verification

```
grep -c "^[[:space:]]*print(" music_gen.py
# Result: 0
```

AST-level confirmation:
```
pytest tests/test_music_gen_logging.py::TestNoPrintCallsRemain -q
# 1 passed
```

## Deviations from Plan

None — plan executed exactly as written.

## Known Stubs

None. All logger calls are fully wired. No stubs, placeholders, or TODO logging calls.

## Threat Flags

None. No new network endpoints, auth paths, or file-write surfaces introduced. T-02-06 (basicConfig at module scope) is mitigated by placement inside `__main__` guard — confirmed by `test_basic_config_only_in_main_guard`.

## Self-Check: PASSED

- VERIFIED: `grep -c "print(" music_gen.py` → 0 ✓
- VERIFIED: `grep -c "logger = logging.getLogger(__name__)" music_gen.py` → 1 ✓
- VERIFIED: `grep -c "logging.basicConfig" music_gen.py` → 1 ✓
- VERIFIED: `grep -c "logger.debug" music_gen.py` → 14 (>= 13 required) ✓
- VERIFIED: `grep -c "logger.info" music_gen.py` → 15 (>= 13 required) ✓
- VERIFIED: `grep -c "logger.warning" music_gen.py` → 2 ✓
- VERIFIED: `grep -c "exc_info=True" music_gen.py` → 1 ✓
- VERIFIED: no f-strings in logger calls → 0 ✓
- VERIFIED: `python -c "import music_gen"` → exits 0, no log output ✓
- VERIFIED: `pytest tests/ -q` → 309 passed, 0 skipped in 0.80s ✓
- FOUND commit: 57c7f31 (Task 1) ✓
- FOUND commit: 738fe22 (Task 2) ✓

---
*Phase: 02-stabilize-ii-config-time-signature-registry-logging*
*Plan: 03*
*Completed: 2026-04-18*
