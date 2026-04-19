---
phase: 04-renderer-mixer-annotator-beats-extraction
plan: "01"
subsystem: beats
tags: [phase-4, beats, midi, mido, swing, time-grid]
dependency_graph:
  requires:
    - 04-00 (mido>=1.3.3 installed, test stubs created)
    - 03-04 (generators/beat.py with generate_beat + calculate_swing_offset)
  provides:
    - musicgen.beats module (beat_duration, extract_beat_times, extract_downbeat_times)
    - D-21 re-export: generators/beat.py imports beat_duration from musicgen.beats
    - RESEARCH correction #1 encoded as test: time-grid downbeats, NOT stride-slice
  affects:
    - src/musicgen/beats.py (new)
    - src/musicgen/generators/beat.py (modified: re-export alias)
    - tests/test_beats.py (replaced from stub)
tech_stack:
  added:
    - mido (MIDI-tick extraction via mido.MidiFile + mido.tick2second + mido.bpm2tempo)
  patterns:
    - time-grid downbeat derivation (RESEARCH correction #1 — D-20 stride-slice wrong)
    - D-21 re-export: remove function body, replace with import alias
    - from __future__ import annotations (matches sampler.py convention)
    - Google-style docstrings with Args/Returns sections
    - zero bare random.* (beats module is fully deterministic)
key_files:
  created:
    - src/musicgen/beats.py
    - (tests/test_beats.py replaced from Wave 0 stub)
  modified:
    - src/musicgen/generators/beat.py
decisions:
  - "extract_downbeat_times uses TIME-GRID algorithm (spec.numerator * beat_slot_s per measure), NOT beat_times[::numerator] stride-slice — RESEARCH correction #1 verified against all 6 beat pattern files"
  - "beat_times parameter retained in extract_downbeat_times signature for API symmetry / future cross-check against MIDI onsets — it is accepted but not sliced"
  - "from timesig import TimeSignatureRegistry (repo-root import, NOT from musicgen.timesig — timesig.py stays at root per D-03 until Phase 5)"
  - "D-21 re-export applied immediately: generators/beat.py body removed, import alias added — makes both import paths yield the same function object (identity check passes)"
metrics:
  duration: "176 seconds (~3 minutes)"
  completed: "2026-04-19"
  tasks_completed: 3
  files_modified: 3
---

# Phase 04 Plan 01: Beats Module Summary

**One-liner:** MIDI-tick beat/downbeat extraction module (`musicgen.beats`) with time-grid downbeat algorithm, D-21 re-export alias in `generators/beat.py`, and 36 tests covering all 6 time signatures + 3 swing cases.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Create src/musicgen/beats.py | 40d9e32 | src/musicgen/beats.py (115 lines) |
| 2 | Re-export beat_duration from generators/beat.py (D-21) | d1019f3 | src/musicgen/generators/beat.py |
| 3 | Populate tests/test_beats.py with full test suite | ba7c40d | tests/test_beats.py (172 lines) |

## Verification Results

### beats.py smoke test
```
beats.py smoke OK
beat_duration('4/4', 120) == 0.5        ✓
beat_duration('6/8', 120) == 0.25       ✓
extract_downbeat_times([], '4/4', 3, 0.0, 120) == [0.0, 2.0, 4.0]  ✓
extract_downbeat_times([], '4/4', 2, 5.0, 120) == [5.0, 7.0]        ✓
extract_downbeat_times([], '12/8', 4, 0.0, 120) → len == 4          ✓
```

### D-21 re-export identity
```
from musicgen.generators.beat import beat_duration as a
from musicgen.beats import beat_duration as b
assert a is b  → OK - same object
```
Both import paths yield the same function object (not a duplicate definition).

### generators/beat.py diff summary
- **Removed:** 7-line `beat_duration` function body (lines 32–38 in pre-task state)
- **Added:** 5-line import comment + `from musicgen.beats import beat_duration  # noqa: F401` (D-21 re-export)
- Net change: -9 lines, +5 lines = file shrinks by 4 lines
- `calculate_swing_offset` and `generate_beat` unchanged

### src/musicgen/beats.py metrics
- **Line count:** 115 lines (target: 60–100 incl. docstrings — 15 lines over, all docstring content)
- **Public functions:** 3 (`beat_duration`, `extract_beat_times`, `extract_downbeat_times`)
- **Bare `random.*` calls:** 0 (module has no `import random`)
- **D-references in docstrings:** D-19, D-20, D-21 each appear multiple times

### tests/test_beats.py metrics
- **Test invocations:** 36 passed (target: 15+)
- **Classes:** TestBeatDuration (10 invocations), TestExtractBeatTimes (3), TestSwingCases (5), TestExtractDownbeatTimes (13 invocations across 8 test methods)
- **Wall time:** 0.05s (target: < 10s)

### Full test suite
```
407 passed, 5 skipped, 2 warnings in 1.01s
```
- 371 baseline (Phase 3 close) + 36 new beats tests = 407 total
- 5 skipped = remaining Wave 0 stubs (test_renderer, test_mixer, test_annotator, test_no_bare_random_in_package, test_integration_full_generation)
- Zero regressions

## RESEARCH Correction #1 — Encoded as Test

`test_downbeat_grid_independent_of_beat_times_input` asserts:
```python
a = extract_downbeat_times([], "4/4", 3, 0.0, 120)
b = extract_downbeat_times([0.5, 1.0, 1.5], "4/4", 3, 0.0, 120)
c = extract_downbeat_times([999.0, -1.0, 42.0], "4/4", 3, 0.0, 120)
assert a == b == c  # passes — time-grid is math-only
```

This encodes RESEARCH.md Pitfall 2: `beat_times` is sparse for patterns with zero-valued slots (4/4 `intro: 0, 42, 38, 0` → only 2 non-zero entries per measure; 12/8 → 9-10 non-zero per measure). The stride-slice approach `beat_times[::numerator]` described in CONTEXT.md D-20 would return wrong counts for these patterns. The time-grid implementation is correct by construction.

## Deviations from Plan

None — plan executed exactly as written. The exact template from the plan's `<action>` blocks was used verbatim for `beats.py` and `tests/test_beats.py`. The `generators/beat.py` modification matched the plan's diff specification exactly.

## Known Stubs

None — all stubs introduced by this plan are functional. The Wave 0 stub in `tests/test_beats.py` was fully replaced by 36 real tests.

## Threat Flags

None — no new network endpoints, auth paths, or trust boundaries introduced. Threat register T-04-01-01 (malformed MIDI input) and T-04-01-02 (large `measures` parameter) both accepted per plan threat model.

## Self-Check: PASSED

- `src/musicgen/beats.py`: FOUND (115 lines)
- `src/musicgen/generators/beat.py` contains `from musicgen.beats import beat_duration`: FOUND
- `tests/test_beats.py` contains `def test_downbeat_count_equals_measures`: FOUND
- Commit 40d9e32 (Task 1 — beats.py): FOUND
- Commit d1019f3 (Task 2 — re-export): FOUND
- Commit ba7c40d (Task 3 — tests): FOUND
- `grep -c "^def " src/musicgen/beats.py` == 3: VERIFIED
- `grep -c "random\." src/musicgen/beats.py` == 0: VERIFIED
- Full suite: 407 passed, 5 skipped: VERIFIED
