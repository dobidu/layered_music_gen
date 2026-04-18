---
phase: 02-stabilize-ii-config-time-signature-registry-logging
plan: 02
subsystem: timesig
tags: [timesig, registry, refactor, python, r-s6]

requires:
  - phase: 02-stabilize-ii-config-time-signature-registry-logging
    plan: 01
    provides: "config.py, tests/test_timesig_registry.py skeleton, conftest.py sys.path shim"
  - phase: 01-stabilize-i-bug-fixes-and-guardrails
    provides: "95 pinned tests in test_time_signature.py and test_duration_validator.py"

provides:
  - "timesig.py — TimeSignatureSpec frozen dataclass + TimeSignatureRegistry with all 7 signatures"
  - "R-S6 closed: adding a new time signature touches exactly one file (timesig.py REGISTRY)"
  - "10 thin wrapper functions in music_gen.py all delegating to TimeSignatureRegistry"
  - "DurationValidator._analyze_time_signature delegates to registry via adapter pattern"
  - "Pitfall 5 fixed: generate_random_time_signature uses random.choices (no missing-return bug)"
  - "186 concrete assertions in tests/test_timesig_registry.py covering all 7 signatures"

affects:
  - "music_gen.py — 10 time-sig functions replaced with one-liner wrappers"
  - "enhanced_duration_validator.py — _analyze_time_signature replaced with registry adapter"
  - "02-03 plan (logging migration) — music_gen.py now imports TimeSignatureRegistry too"

tech-stack:
  added: []
  patterns:
    - "TimeSignatureSpec @dataclass(frozen=True) with 19 fields encoding all per-sig metadata"
    - "TimeSignatureRegistry class-level REGISTRY dict keyed by canonical sig string"
    - "Empty frozenset for valid_chord_pattern_lengths signals default-True (5/4, 7/8)"
    - "beat_pattern_length == numerator for ALL sigs (cosmetic-if preservation)"
    - "Shared module-level constants (_COMPOUND_NOTE_DURATIONS, _SIMPLE_NOTE_DURATIONS, etc.)"
    - "local import `from timesig import TimeSignatureRegistry` in _analyze_time_signature"
    - "random.choices() for weighted sampling (replaces threshold-loop)"

key-files:
  created:
    - timesig.py
  modified:
    - music_gen.py
    - enhanced_duration_validator.py
    - tests/test_timesig_registry.py

key-decisions:
  - "TimeSignatureSpec frozen=True — entries are immutable; prevents accidental runtime mutation (T-02-05)"
  - "beat_pattern_length == numerator for ALL signatures (including compound) — cosmetic-if preserved"
  - "Empty frozenset() for 5/4 and 7/8 valid_chord_pattern_lengths triggers default-True guard in verify_chord_pattern_length"
  - "Local import of TimeSignatureRegistry inside _analyze_time_signature avoids any circular-import risk"
  - "Layer-specific duration sets stay in DurationValidator for Phase 2 — orthogonal to time signatures"
  - "note_durations stored as Dict field (not tuple-of-tuples) — frozen=True blocks setattr, not dict mutation; consumers must not mutate"
  - "A3 assumption validated: RNG order change from threshold-loop to random.choices is a silent distribution change that Phase 5 must baseline against post-refactor behavior"

metrics:
  duration: "~6 minutes"
  completed: "2026-04-18"
  tasks: 2
  files_created: 1
  files_modified: 3
  tests_added: 186
  tests_total: 303
  tests_skipped: 1
---

# Phase 02 Plan 02: TimeSignatureRegistry Summary

**TimeSignatureSpec frozen dataclass and TimeSignatureRegistry consolidating all 7 time-signature specifications into timesig.py, with 10 thin wrapper functions in music_gen.py and a registry-delegating adapter in DurationValidator**

## Performance

- **Duration:** ~6 minutes
- **Started:** 2026-04-18T15:59:58Z
- **Completed:** 2026-04-18T16:05:58Z
- **Tasks:** 2
- **Files created:** 1 (timesig.py)
- **Files modified:** 3 (music_gen.py, enhanced_duration_validator.py, tests/test_timesig_registry.py)
- **Test runtime:** 0.80s for 303 tests (well under 10s R-Q2 budget)

## Accomplishments

- Built `timesig.py` at repo root with `TimeSignatureSpec` frozen dataclass (19 fields) and `TimeSignatureRegistry` with all 7 signatures registered
- All 10 module-level time-sig functions in `music_gen.py` are now thin one-liner wrappers delegating to `TimeSignatureRegistry`
- `DurationValidator._analyze_time_signature` now delegates to registry and returns legacy `TimeSignatureInfo` shape (valid_durations as `set`, not `frozenset`)
- `generate_random_time_signature` missing-return bug (Pitfall 5) fixed implicitly by switching to `random.choices`
- Zero `time_signature.split('/')` calls remain in `music_gen.py` or `enhanced_duration_validator.py`
- 186 concrete assertions in `tests/test_timesig_registry.py` covering all 7 signatures with exact field values
- All 95 Phase 1 pinned tests pass without modification

## Task Commits

1. **Task 1: timesig.py + test suite** — `b68491e` (feat)
2. **Task 2: thin wrappers + DurationValidator adapter** — `1253a50` (feat)

## Final TimeSignatureSpec Field Values Per Signature

| Sig | is_compound | valid_chord_lens | beat_pat_len | measure_mult | midi_pow | beats/meas | primary_div | max_dur | primary_beat | requires_even |
|-----|-------------|------------------|--------------|--------------|----------|------------|-------------|---------|--------------|---------------|
| 2/4 | False | {1,2} | 2 | 2.0 | 2 | 2 | 2.0 | 2.0 | 1.0 | True |
| 3/4 | False | {1,3} | 3 | 4/3 | 2 | 3 | 2.0 | 3.0 | 1.0 | False |
| 4/4 | False | {1,2,4} | 4 | 1.0 | 2 | 4 | 2.0 | 4.0 | 1.0 | False |
| 5/4 | False | {} (any) | 5 | 1.0 | 2 | 5 | 2.0 | 5.0 | 1.0 | False |
| 6/8 | True | {2,3,6} | 6 | 2.0 | 3 | 2 | 3.0 | 3.0 | 0.5 | True |
| 7/8 | False | {} (any) | 7 | 1.0 | 3 | 7 | 2.0 | 7.0 | 0.5 | False |
| 12/8 | True | {2,3,6} | 12 | 2.0 | 3 | 4 | 3.0 | 6.0 | 0.5 | True |

## Functions Converted to Thin Wrappers (music_gen.py)

All 10 functions listed in the plan are now one-liner delegates:

| Function | Delegation |
|----------|-----------|
| `verify_pattern_for_time_signature` | `spec.verify_chord_pattern_length(len(chord_pattern))` |
| `verify_beat_pattern` | `spec.verify_beat_pattern_length(len(pattern))` |
| `calculate_measures_for_time_signature` | `spec.measures_for(base_length)` |
| `validate_measures` | loop calling `spec.measure_count_valid(count)` per part |
| `get_midi_time_signature_values` | `spec.numerator, spec.midi_denominator_power` |
| `get_note_duration` | `spec.primary_beat_duration` |
| `get_note_durations` | `spec.note_duration_map()` |
| `get_melody_durations` | `list(spec.melody_duration_candidates)` |
| `generate_random_time_signature` | `TimeSignatureRegistry.sample_random()` |
| `time_signature_alternative` | `random.choice(spec.alternatives)` |

## DurationValidator Adapter Pattern

`_analyze_time_signature` now does:
1. Check cache (preserved for efficiency)
2. `from timesig import TimeSignatureRegistry` (local import — avoids circular risk)
3. `spec = TimeSignatureRegistry.lookup(time_signature)`
4. Build `TimeSignatureInfo` from spec fields, converting `FrozenSet` to `set` for the `valid_durations` field
5. Store in cache and return

The `TimeSignatureInfo` dataclass shape is unchanged — all callers (`get_valid_duration`, `validate_layer_duration`, `get_suggested_duration`) continue to work with the same interface.

## Pitfall 5 Fix (Implicit Bug Fix)

The old `generate_random_time_signature` used a threshold-loop that had no explicit fallback return for `dice >= 1.00` (float rounding edge case). The refactored wrapper uses `TimeSignatureRegistry.sample_random()` which calls `random.choices(sigs, weights=weights, k=1)[0]` — this always returns a value regardless of float precision.

**Note for Phase 5 (seed discipline baseline):** Assumption A3 is confirmed — the RNG call order changes from a cumulative-threshold comparison against `random.random()` to a single `random.choices()` call. Any Phase 5 seed-locked tests that call `generate_random_time_signature()` must baseline their expected values against the post-refactor `random.choices` behavior.

## Assumptions Validated During Implementation

| Assumption | Status | Notes |
|------------|--------|-------|
| A1 (compound valid_durations = {1.5, 1.0, 0.75, 0.5, 0.25}) | Validated | Matches enhanced_duration_validator.py compound branch exactly |
| A3 (RNG order change from threshold-loop to random.choices) | Confirmed | Silent distribution change; Phase 5 must baseline post-refactor |
| A5 (conftest.py sys.path shim covers timesig import) | Validated | `from timesig import TimeSignatureRegistry` resolves in all tests |

## R-S6 Exit Criterion Verification

```
grep -rnc "time_signature.split('/')" music_gen.py enhanced_duration_validator.py
# Result: 0 matches in both files
```

Adding a new time signature now requires editing exactly ONE file: add one `TimeSignatureSpec(...)` entry to `TimeSignatureRegistry.REGISTRY` in `timesig.py`.

## Deviations from Plan

None — plan executed exactly as written.

## Known Stubs

None. All registry entries are fully wired with concrete data. The `tests/test_music_gen_logging.py` skeleton (`@pytest.mark.skip`) is an intentional Wave 0 stub for Plan 02-03, not a stub that blocks this plan's goals.

## Threat Flags

None. `timesig.py` is a pure-data module with no I/O, no filesystem access, no env var reading. T-02-05 (registry tampering via mutable REGISTRY dict) is mitigated by `@dataclass(frozen=True)` on all entries as specified.

## Self-Check: PASSED

- FOUND: /home/bidu/musicgen/timesig.py ✓
- FOUND: /home/bidu/musicgen/tests/test_timesig_registry.py (186 tests, 0 skips) ✓
- FOUND commit: b68491e (Task 1) ✓
- FOUND commit: 1253a50 (Task 2) ✓
- VERIFIED: `pytest tests/ -q` → 303 passed, 1 skipped in 0.80s ✓
- VERIFIED: `grep -c "time_signature.split" music_gen.py enhanced_duration_validator.py` → 0 each ✓
- VERIFIED: `import music_gen` → side-effect-free ✓
- VERIFIED: 6/8 beat_pattern_length=6, verify_beat_pattern_length(3)=False ✓
- VERIFIED: 5/4 and 7/8 verify_chord_pattern_length(99)=True ✓
- VERIFIED: sampling weights sum to 1.0 ✓

---
*Phase: 02-stabilize-ii-config-time-signature-registry-logging*
*Plan: 02*
*Completed: 2026-04-18*
