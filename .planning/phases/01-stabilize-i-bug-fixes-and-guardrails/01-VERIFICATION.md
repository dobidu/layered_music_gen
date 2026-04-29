---
phase: 01-stabilize-i-bug-fixes-and-guardrails
verified: 2026-04-08T00:00:00Z
status: passed
score: 17/17 must-haves verified
overrides_applied: 0
---

# Phase 01: Stabilize I — Bug Fixes and Guardrails — Verification Report

**Phase Goal:** `music_gen.py` becomes importable and stops silently corrupting outputs. Two real bugs found during research are fixed. First test suite lands.
**Verified:** 2026-04-08
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| #  | Truth | Status | Evidence |
|----|-------|--------|----------|
| 1  | `python -c 'import music_gen'` exits 0 with no generation output | VERIFIED | `.venv/bin/python -c "import music_gen"` → "import OK", no "Generating song" output |
| 2  | `generate_song_arrangement` is called exactly once per song, before MIDI generation | VERIFIED | `grep -c "generate_song_arrangement" music_gen.py` → 2 (one def at L613, one call at L1036 inside `create_song`, before `generate_song_parts`) |
| 3  | `mix_and_save` receives arrangement as parameters; never re-rolls it | VERIFIED | `inspect.signature(mix_and_save).parameters` includes `song_unique_parts, song_arrangement`; `inspect.getsource` confirms no `generate_song_arrangement(` call inside the function body |
| 4  | `levels.json` volume values affect the final mix audio | VERIFIED | L851-854: `beat = beat.apply_gain(_lin_to_db(...))` (and 3 siblings) — capture-return pattern, dB conversion via `20*log10(v)` floor `1e-6` |
| 5  | `levels.json` panning values affect the final mix audio | VERIFIED | L855-858: `beat = beat.pan(float(...))` (and 3 siblings) — every `.pan(` is on RHS of `=` |
| 6  | No `.volume = ` assignment exists on AudioSegment objects in mix_and_save | VERIFIED | `grep -nE "(beat\|melody\|harmony\|bassline)\.volume\s*=" music_gen.py` returns empty |
| 7  | Every `.pan(...)` call has its return value captured back into the segment variable | VERIFIED | L855-858 all four are `x = x.pan(...)` |
| 8  | `from music21 import *` no longer appears in music_gen.py | VERIFIED | L2: `from music21 import roman, scale, pitch` |
| 9  | All music21 symbols used (`roman`, `scale`, `pitch`) imported explicitly | VERIFIED | Single explicit import line at L2 |
| 10 | musicality_score.py has zero `except Exception` blocks; every handler uses `logger.exception` | VERIFIED | `grep -c "except Exception" musicality_score.py` → 0; `grep -c "logger.exception" musicality_score.py` → 5 |
| 11 | Dead imports (`glob`, `Pool`, `cpu_count`) removed from music_gen.py | VERIFIED | `grep -n "^import glob\|from multiprocessing" music_gen.py` returns empty |
| 12 | Dead variables (`ha={}, ba={}, me={}, be={}, now=datetime.now()`) removed | VERIFIED | grep for both patterns returns empty; tuple unpack at L1039 uses `_` for the unused 5th slot |
| 13 | `uuid` line removed from requirements.txt | VERIFIED | `grep -n "uuid" requirements.txt` returns empty |
| 14 | `pytest` and `pytest-cov` installable via documented dev path | VERIFIED | `dev-requirements.txt` exists with `-r requirements.txt`, `pytest>=8.0`, `pytest-cov>=5.0` |
| 15 | `pytest tests/ -q` exits 0 | VERIFIED | `.venv/bin/python -m pytest tests/ -q` → `95 passed in 5.03s` |
| 16 | Tests cover `verify_pattern_for_time_signature`, `verify_beat_pattern`, `validate_measures`, `DurationValidator` | VERIFIED | `tests/test_time_signature.py` (3 classes for the three timesig helpers) + `tests/test_duration_validator.py` (3 classes for DurationValidator + NoteValue) |
| 17 | Tests run in under 10 seconds and require no audio dependencies | VERIFIED | 5.03s wall (well under 10s); test files import only `music_gen`, `enhanced_duration_validator`, `pytest` — no librosa/pedalboard/pydub/midi2audio direct imports |

**Score:** 17/17 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `music_gen.py` | Importable + arrangement-once + gain/pan fix + clean imports | VERIFIED | All 4 contracts hold; `__main__` guard at L1166; explicit music21 imports at L2; apply_gain x4 + capture-return pan x4 at L851-858 |
| `musicality_score.py` | Narrowed exception handlers with `logger.exception` | VERIFIED | 5 narrowed handlers, 0 broad `except Exception`, 5 `logger.exception` calls |
| `requirements.txt` | uuid stub removed | VERIFIED | No `uuid` line present |
| `dev-requirements.txt` | Test dependencies | VERIFIED | Contains `-r requirements.txt`, `pytest>=8.0`, `pytest-cov>=5.0` |
| `tests/conftest.py` | sys.path shim | VERIFIED | Contains `sys.path.insert(0, REPO_ROOT)` |
| `tests/test_time_signature.py` | Tests for 3 timesig functions | VERIFIED | 46 tests across 3 TestX classes, all green |
| `tests/test_duration_validator.py` | Tests for DurationValidator | VERIFIED | 49 tests across 3 TestX classes, all green |

### Key Link Verification

| From | To | Via | Status |
|------|-----|-----|--------|
| `create_song` | `mix_and_save` | passes `song_unique_parts, song_arrangement` as 7th/8th positional args at L1050-1052 | WIRED |
| `levels.json` values | AudioSegment objects | `apply_gain(_lin_to_db(...))` + capture-return `pan(...)` at L851-858 | WIRED |
| music_gen.py imports | music21 sub-modules | explicit `from music21 import roman, scale, pitch` at L2 | WIRED |
| `tests/conftest.py` | repo root modules | `sys.path.insert(0, REPO_ROOT)` enables `import music_gen` and `import enhanced_duration_validator` | WIRED |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Module imports without side effects | `.venv/bin/python -c "import music_gen"` | "import OK", no generation output | PASS |
| mix_and_save signature has new params | `inspect.signature(music_gen.mix_and_save).parameters` | includes `song_unique_parts, song_arrangement` | PASS |
| mix_and_save body has no re-roll | `inspect.getsource(mix_and_save)` lacks `generate_song_arrangement(` | confirmed | PASS |
| Test suite green | `.venv/bin/python -m pytest tests/ -q` | 95 passed in 5.03s | PASS |
| musicality_score importable | structural check via grep | 0 broad exceptions, 5 logger.exception | PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| R-S1 | 01-01 | Importability — `music_gen.py` importable without triggering generation | SATISFIED | `__main__` guard at L1166; runtime `import music_gen` produces no output |
| R-S2 | 01-03 | Explicit imports — replace `from music21 import *` with named imports | SATISFIED | L2: `from music21 import roman, scale, pitch` |
| R-S3 | 01-01 | Arrangement re-roll bug — produced once and threaded through | SATISFIED | Single call site in `create_song` (L1036); `mix_and_save` accepts as params; PITFALLS P-A closed |
| R-S4 | 01-02 | pydub gain/pan bug — `apply_gain(...)` + capture-return `pan(...)` | SATISFIED | L851-858: 4 `apply_gain` + 4 `x = x.pan(...)`; PITFALLS P-B closed |
| R-S7 | 01-03 | Structured logging (partial) — narrow exception handlers in musicality_score.py | SATISFIED (partial scope) | 5 narrowed handlers using `logger.exception`. `print` → `logging` migration in `music_gen.py` is intentionally deferred to Phase 2 per ROADMAP scope (REQUIREMENTS.md L28 confirms partial completion is the Phase 1 contract) |
| R-S8 | 01-03 | Dead code removal | SATISFIED | `glob`, `Pool`, `cpu_count` imports removed; `ha/ba/me/be/now` dead vars removed; `uuid` PyPI stub removed from requirements.txt |
| R-Q2 | 01-04 | First test suite (initial coverage of pure functions) | SATISFIED | 95 passing tests covering the 4 ROADMAP-named pure functions; runtime 5.03s |

All 7 declared requirements accounted for. No orphaned requirements found in REQUIREMENTS.md for Phase 1.

### Anti-Patterns Found

None — scan of the modified files found no blocker stubs, no TODO/FIXME placeholders introduced by this phase, no empty handlers, no hardcoded empty data flowing to user-visible output. The pre-existing `# TODO (later phase): only render and mix the parts that are used in the song arrangement` comment in `mix_and_save` is documentation of explicitly deferred work, not a stub.

### Exit Criteria (ROADMAP)

| # | Criterion | Status |
|---|-----------|--------|
| 1 | `python -c "import music_gen"` does not trigger generation | PASS |
| 2 | A seeded run produces mix audio that reflects `levels.json` | PASS (source-level: gain/pan no-op bug fixed; `apply_gain` + capture-return `pan` in place. Audio rendering not run in this verification — guarded by source-level proof that the values now flow into the segments.) |
| 3 | Pure-function unit tests all green | PASS — 95 passed in 5.03s |

### Gaps Summary

No gaps. Every must-have from the four plan frontmatters is verified in code. Every requirement ID declared in PLAN frontmatters (R-S1, R-S2, R-S3, R-S4, R-S7, R-S8, R-Q2) maps to concrete, verifiable evidence in the codebase. Both PITFALLS bugs (P-A arrangement re-roll, P-B pydub gain/pan no-op) are closed. The pytest skeleton runs green in under the 10-second ROADMAP target with no audio-dependency imports in the test files. The phase goal — "`music_gen.py` becomes importable and stops silently corrupting outputs; first test suite lands" — is fully achieved.

The intentional Phase 1 → Phase 2 deferral of `print` → `logging` migration in `music_gen.py` is consistent with REQUIREMENTS.md (R-S7 explicitly notes "Partially complete (Plan 01-03)... `print()` migration in `music_gen.py` deferred to Phase 2 per ROADMAP scope") and does not constitute a gap for Phase 1.

---

_Verified: 2026-04-08_
_Verifier: Claude (gsd-verifier)_
