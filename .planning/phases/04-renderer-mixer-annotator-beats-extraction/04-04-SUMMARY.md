---
phase: 04-renderer-mixer-annotator-beats-extraction
plan: "04"
subsystem: annotator
tags: [phase-4, annotator, r-p4-schema, pure-function, json, tbd-fields, r-x6]
dependency_graph:
  requires:
    - 04-00 (mido installed, test stubs created, pytest markers declared)
    - 04-01 (musicgen.beats — beat_times / downbeat_times shapes used by annotator)
    - 04-02 (musicgen.renderer + RenderResult frozen dataclass)
    - 04-03 (musicgen.mixer + MixResult frozen dataclass — transitions, part_layers, soundfonts, pedalboards)
  provides:
    - musicgen.annotator module — single public function annotate() producing R-P4 schema dict
    - annotate() pure-function contract (D-14): zero I/O, returns plain dict ready for json.dump
    - R-P4 schema shape locked for Phase 5: all 20 Phase-4-fill fields + 4 Phase-5 TBD fields as None
    - analysis_failed tri-state contract: omitted on success, present True on failure (D-16 clarification)
    - _derive_mode helper: key endswith('m') → minor, else major (D-15)
    - _transitions_to_arrangement: converts MixResult.transitions to global offset list-of-dicts
    - 41 fixture-driven tests across 8 test classes covering D-15/D-16 semantics + D-14 purity
  affects:
    - 04-05 (orchestrator collapse — annotator.annotate() is the Phase 4 output assembly step)
    - Phase 5 (writer owns json.dump; will fill seed/musicgen_version/split/pre_roll_offset_seconds)
tech_stack:
  added: []
  patterns:
    - Pure function with no imports of random (zero bare random.* — D-17 verified AST)
    - Zero I/O inside annotate() — builtins.open monkeypatched in test confirms no file access (D-14)
    - Defensive copies: dict(x) and list(x) throughout — input dicts are not mutated
    - TBD-as-None D-16: Phase-5 fields present as None (not missing, not "TBD" string)
    - analysis_failed tri-state: None (omit key), True (emit key=True), False (omit key)
    - _transitions_to_arrangement: cumulative offset accumulation from per-part local transitions
key_files:
  created:
    - src/musicgen/annotator.py (181 lines — annotate() + 3 private helpers)
    - tests/test_annotator.py (replaced from Wave 0 stub — 395 lines, 41 tests, 8 classes)
  modified: []
key-decisions:
  - "Signature extended beyond D-14's original definition to include chord_progressions, midi_paths, mix_path positional params — annotator cannot derive these from SongParams/MixResult alone (RESEARCH Open Questions #2 and #3)"
  - "analysis_failed is tri-state kwarg (None=omit, True=emit, False=omit) — D-16 clarification locks this pattern for Phase 5 failure path"
  - "pre_roll_offset_seconds hardcoded as None (not a kwarg) — R-P9 Phase 6 fills it; not exposing the kwarg this phase avoids premature Phase 5 threading"
  - "soundfonts taken from first arrangement part's MixResult — all parts share the same soundfonts dict (picked once at orchestrator level)"
patterns-established:
  - "Pure-function annotator: zero I/O, zero RNG, deterministic — testable without filesystem setup"
  - "R-P4 schema shape locked at Phase 4: field names and null semantics must not change until Phase 5 fills TBD fields"
requirements-completed: [R-X6]
duration: ~3min
completed: "2026-04-19"
---

# Phase 04 Plan 04: Annotator Module Summary

**Pure-function R-P4 schema assembler (`musicgen.annotator.annotate`) with 20 Phase-4-fill fields, 4 Phase-5 TBD fields as None, analysis_failed tri-state, and 41 fixture-driven purity tests (D-14/D-15/D-16).**

## Performance

- **Duration:** ~3 min 22 sec
- **Started:** 2026-04-19T17:43:41Z
- **Completed:** 2026-04-19T17:47:03Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- `src/musicgen/annotator.py` (181 lines) — single public function `annotate()` that assembles the full R-P4 annotation dict from stage outputs; pure function with zero I/O, zero bare `random.*`, and zero subprocess calls
- All 20 Phase-4-fillable R-P4 fields populated non-None: key, mode, tempo_bpm, time_signature, time_signatures_per_part, measures_per_part, swing, duration_seconds, song_arrangement, chord_progression, active_layers, soundfonts, fx_params, beat_times, downbeat_times, musicality_score, fluidsynth_version, mix, stems, midi
- All 4 Phase-5 TBD fields present as `None` (not missing, not "TBD"): seed, musicgen_version, split, pre_roll_offset_seconds (D-16)
- `analysis_failed` tri-state: omitted when `None` (default success), present `True` only on explicit error, also omitted when `False` (D-16 clarification)
- `tests/test_annotator.py` replaced from Wave 0 stub with 41 real tests across 8 classes — 491 total passing, zero regressions

## R-P4 Field Source Map

| Field | Phase | Source |
|-------|-------|--------|
| `key` | 4 | `song_params.key` |
| `mode` | 4 | `_derive_mode(key)` — trailing "m" → minor |
| `tempo_bpm` | 4 | `song_params.tempo` |
| `time_signature` | 4 | `song_params.time_signature_base` |
| `time_signatures_per_part` | 4 | `dict(song_params.signatures_per_part)` |
| `measures_per_part` | 4 | `dict(song_params.measures_per_part)` |
| `swing` | 4 | `song_params.swing_amount` |
| `duration_seconds` | 4 | `_derive_total_duration(arrangement)` — sum via arrangement end |
| `song_arrangement` | 4 | `_transitions_to_arrangement(mix_results, song_params.song_arrangement)` |
| `chord_progression` | 4 | `chord_progressions` caller kwarg |
| `active_layers` | 4 | `{part: dict(mr.part_layers) for part, mr in mix_results.items()}` |
| `soundfonts` | 4 | `dict(mix_results[first_part].soundfonts)` |
| `fx_params` | 4 | `{part: dict(mr.pedalboards) for part, mr in mix_results.items()}` |
| `beat_times` | 4 | `beat_times` caller param (from beats.extract_beat_times) |
| `downbeat_times` | 4 | `downbeat_times` caller param (from beats.extract_downbeat_times) |
| `musicality_score` | 4 | `dict(musicality)` caller param |
| `fluidsynth_version` | 4 | `fluidsynth_version` keyword-only param |
| `mix` | 4 | `mix_path` caller param |
| `stems` | 4 | `{part: dict(mr.stem_paths) for part, mr in mix_results.items()}` |
| `midi` | 4 | `midi_paths` caller param |
| `seed` | 5 | None (D-16) — wired as kwarg, default None |
| `musicgen_version` | 5 | None (D-16) — wired as kwarg, default None |
| `split` | 5 | None (D-16) — wired as kwarg, default None |
| `pre_roll_offset_seconds` | 5 | None (D-16) — hardcoded None (R-P9 Phase 6) |
| `analysis_failed` | 5 | Omitted on success; present True on explicit failure (D-16 clarification) |

## Task Commits

1. **Task 1: Create src/musicgen/annotator.py** — `93d7673` (feat)
2. **Task 2: Populate tests/test_annotator.py** — `3a884db` (test)

## Files Created/Modified

- `/home/bidu/musicgen/src/musicgen/annotator.py` — 181 lines; single public `annotate()` function + 3 private helpers (`_derive_mode`, `_transitions_to_arrangement`, `_derive_total_duration`)
- `/home/bidu/musicgen/tests/test_annotator.py` — 395 lines; 8 test classes, 34 `def test_*` methods, 41 total test cases (8 parametrized in `TestModeDerivation`)

## Verification Results

### AST static checks (post-commit)

```
Zero I/O AST check PASSED (no open/json.dump/makedirs/mkdir calls in annotate body)
Zero bare random.* AST check PASSED (no random.<method> calls anywhere in module)
```

### annotator.py smoke test

```
annotator.py smoke OK
```

### test_annotator.py

```
41 passed, 2 warnings in 0.27s
```

### Full suite regression

```
491 passed, 2 skipped, 2 warnings in 1.24s
(450 prior + 41 new annotator tests = 491; Wave 0 annotator stub no longer skipped)
```

## Decisions Made

- Signature extended beyond CONTEXT.md D-14's definition to add three orchestrator-provided positional params: `chord_progressions` (per-part chord lists — generator computes but discards today per RESEARCH Open Question #2), `midi_paths` (per-part per-layer MIDI paths), `mix_path` (final concat WAV path). Annotator cannot infer these from `SongParams` or `MixResult` alone.
- `pre_roll_offset_seconds` is hardcoded `None` rather than exposed as a kwarg — R-P9 (Phase 6) fills it; no premature threading needed this phase.
- `soundfonts` taken from `mix_results[first_part].soundfonts` — all parts share the same soundfonts dict (picked once at orchestrator level before the render loop).

## Deviations from Plan

None — plan executed exactly as written. The annotator implementation matches the plan's `<action>` code block verbatim, and the test file matches the plan's `<action>` code block for Task 2 verbatim.

## Known Stubs

None — all fields either populated (Phase-4 fill) or explicitly `None` (Phase-5 TBD per D-16). The `None` values are the intended contract, not stubs.

## Threat Flags

None — no new network endpoints, auth paths, or trust boundaries introduced. The annotator is a pure in-memory dict assembler with no external connections.

## Self-Check: PASSED
