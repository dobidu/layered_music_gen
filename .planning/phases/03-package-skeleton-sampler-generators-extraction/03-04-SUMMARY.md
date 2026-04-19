---
phase: 03-package-skeleton-sampler-generators-extraction
plan: 03-04
subsystem: generators
tags: [extraction, generators, midi, rng-injection, music21, determinism]

# Dependency graph
requires:
  - phase: 03-01
    provides: pyproject.toml + src-layout package skeleton (musicgen package importable)
  - phase: 03-02
    provides: musicgen.duration_validator module (DurationValidator moved into package)
  - phase: 03-03
    provides: musicgen.sampler module + module-level _rng threading in music_gen.py
provides:
  - src/musicgen/generators/ package (chord, melody, bassline, beat)
  - 4 MIDI generators with required rng:random.Random parameter (D-07/D-08)
  - TimeSignatureRegistry direct-usage pattern replacing double-indirection (D-06)
  - music21 audit comment in chord/melody/bassline documenting D-23 clean-path
  - 6 re-exports from music_gen.py back-compat shim
  - 26 passing generator tests (seeded determinism + AST no-bare-random guard)
affects: [03-05, 04, 05]

# Tech tracking
tech-stack:
  added: []  # internal refactor — no new dependencies
  patterns:
    - "rng-injection at generator boundary (D-07/D-08) — no bare random.* in generators"
    - "TimeSignatureRegistry.lookup() direct spec-attribute access (D-06)"
    - "music21 audit comment in generators that use roman/scale/pitch (D-23)"
    - "cfg fallback in generate_beat: _beat_cfg = cfg if cfg is not None else config.Config() (D-02)"
    - "seeded determinism test pattern: monkeypatch.chdir(tmp_path) + relative name"
    - "parametrized AST scan for static guards — auto-extends to new files in the directory"

key-files:
  created:
    - src/musicgen/generators/__init__.py
    - src/musicgen/generators/chord.py
    - src/musicgen/generators/melody.py
    - src/musicgen/generators/bassline.py
    - src/musicgen/generators/beat.py
    - tests/test_generators/__init__.py
    - tests/test_generators/test_chord.py
    - tests/test_generators/test_melody.py
    - tests/test_generators/test_bassline.py
    - tests/test_generators/test_beat.py
    - tests/test_generators/test_no_bare_random.py
  modified:
    - music_gen.py  # 6 inline defs deleted, 4 re-imports added, 4 call sites pass _rng

key-decisions:
  - "Generators take required rng:random.Random (last positional for chord/melody/bassline; before cfg for beat); calling without rng raises TypeError — gate for Phase 5 per-sample seed discipline."
  - "Use TimeSignatureRegistry.lookup() directly for numerator/midi_denominator_power (D-06 — no double indirection); keep music_gen.py thin wrappers (verify_*, get_midi_*) in place since they may be useful to external callers."
  - "music21 audit comment present in chord/melody/bassline (D-23 clean path); Plan 03-05 will add regression test to guard this."
  - "Tests chdir into tmp_path and pass relative name to accommodate the generators' existing name.split('-')[0] directory-derivation quirk (carried over verbatim from music_gen.py); no generator behavior change — production naming (20-digit song_name) already satisfies this convention."

patterns-established:
  - "src/musicgen/generators/ layout: one generator per file, narrow music21 imports (chord/melody/bassline only), beat.py pure-percussion (no music21)"
  - "AST-scan no-bare-random test parametrized over directory glob — automatically extends as new generator modules are added"
  - "Seeded determinism contract: same seed + args → byte-equal MIDI (Path.read_bytes() comparison)"

requirements-completed: [R-X3]

# Metrics
duration: 28 min
completed: 2026-04-18
---

# Phase 3 Plan 03-04: Generator Extraction Summary

**Four MIDI generators (chord, melody, bassline, beat) extracted from music_gen.py into src/musicgen/generators/*.py with injected rng, 11 bare random.* sites rewritten, 26 seeded-determinism + AST-scan tests added, music_gen.py shrunk from 938 → 524 lines via re-export shim.**

## Performance

- **Duration:** ~28 min
- **Started:** 2026-04-18T20:23:00Z (approx)
- **Completed:** 2026-04-18T20:51:15Z
- **Tasks:** 3 (all autonomous)
- **Files created:** 11
- **Files modified:** 1 (music_gen.py)
- **Net line delta:** +871 additions / -430 deletions across all files

## Accomplishments

- `src/musicgen/generators/` package created with 4 generator modules (chord, melody, bassline, beat) + `__init__.py` marker.
- Every generator takes a **required** `rng: random.Random` parameter; missing-rng raises `TypeError` (pinned in tests).
- **Zero bare `random.<method>` calls** in any generator module — enforced by `tests/test_generators/test_no_bare_random.py` (parametrized AST scan over `src/musicgen/generators/*.py`).
- **Bare-random sites rewritten per generator:** chord=1, melody=4, bassline=4, beat=2 = **11 total** (matches RESEARCH.md §Generator bare-random total).
- `TimeSignatureRegistry.lookup()` used directly for `numerator` and `midi_denominator_power` attribute access in all 4 generators (D-06 — no double indirection through music_gen shim wrappers for attribute reads). Pattern-length checks routed directly through `spec.verify_chord_pattern_length(...)` / `spec.verify_beat_pattern_length(...)`.
- `DurationValidator` imported from new `musicgen.duration_validator` location in every generator.
- music21 audit comment block (D-23 clean-path) inserted in chord.py, melody.py, bassline.py above the first `roman.RomanNumeral` / `scale.*Scale` / `pitch.Pitch` call; beat.py has no music21 dependency.
- `generate_beat` preserves the Plan 02-01 D-02 cfg fallback pattern: `_beat_cfg = cfg if cfg is not None else config.Config()`. Uses existing `_beat_cfg.beat_roll_pattern_files` attribute (no attribute-name fallback needed — `config.py:56` exposes it directly).
- `music_gen.py` rewritten as a back-compat shim: 4 re-import lines added, 6 inline function defs deleted (generate_chord_progression, generate_melody, generate_bassline, generate_beat, beat_duration, calculate_swing_offset), 4 generator call sites inside `generate_song_parts` now pass `_rng`.
- **26 new tests** all passing (3 chord + 4 melody + 3 bassline + 11 beat + 5 AST/other = 26 total). Full suite: **368 passing** (342 baseline + 26 new).
- Determinism contract verified: same seed + same args → **byte-equal MIDI output** for every generator.

## Task Commits

Each task was committed atomically:

1. **Task 1: Create generators/ package with chord + melody + bassline + beat modules (rng-injected)** — `9256012` (feat)
2. **Task 2: Create seeded-RNG tests for all 4 generators + AST no-bare-random guard** — `5fa3f0c` (test)
3. **Task 3: Update music_gen.py shim to re-export generators + pass _rng at call sites** — `3224e4e` (refactor)

## Files Created/Modified

### Created

- `src/musicgen/generators/__init__.py` — package marker (docstring + comment per Pattern M).
- `src/musicgen/generators/chord.py` — `generate_chord_progression` with injected rng; 1 bare-random site rewritten; music21 audit comment above `roman.RomanNumeral`.
- `src/musicgen/generators/melody.py` — `generate_melody` with injected rng; 4 bare-random sites rewritten; music21 audit comment above `scale.MajorScale`.
- `src/musicgen/generators/bassline.py` — `generate_bassline` with injected rng; 4 bare-random sites rewritten; music21 audit comment above `scale.MajorScale`.
- `src/musicgen/generators/beat.py` — `generate_beat`, `beat_duration`, `calculate_swing_offset`; 2 bare-random sites rewritten in `generate_beat`; no music21 deps; cfg fallback preserved.
- `tests/test_generators/__init__.py` — empty test-package marker.
- `tests/test_generators/test_chord.py` — 3 seeded determinism cases + missing-rng TypeError check (4 tests).
- `tests/test_generators/test_melody.py` — 3 seeded determinism cases + different-seed divergence + missing-rng check (5 tests).
- `tests/test_generators/test_bassline.py` — 3 seeded determinism cases + missing-rng check (4 tests).
- `tests/test_generators/test_beat.py` — 5 (ts, swing) determinism cases + missing-rng check + 2 beat_duration unit tests + 1 calculate_swing_offset unit test (9 tests).
- `tests/test_generators/test_no_bare_random.py` — parametrized AST scan over all generator modules (4 tests, auto-extends).

### Modified

- `music_gen.py` — 4 re-import lines added at top (after sampler re-imports); 6 inline function defs deleted; 4 generator call sites in `generate_song_parts` now pass `_rng`. File size: 938 → 524 lines (-414 lines net).

## Decisions Made

- **D-06 direct lookup:** Generators use `TimeSignatureRegistry.lookup(time_signature).numerator` / `.midi_denominator_power` directly. The thin `get_midi_time_signature_values` / `verify_pattern_for_time_signature` / `verify_beat_pattern` wrappers remain in `music_gen.py` (kept in place per plan instructions — D-06 bars *useless* indirection, not *all* indirection), but the generators themselves no longer route through them for reads.
- **D-23 music21 audit comment:** Added verbatim 4-line block from PLAN's `<interfaces>` to chord.py, melody.py, bassline.py above the first music21 call. Plan 03-05 will add the regression test that guards this assumption.
- **beat `rng` positional placement:** Per plan `<interfaces>`, `generate_beat` signature is `(part, tempo, time_signature, measures, name, swing_amount, rng, cfg=None)` — `rng` is a required positional arg appearing before the optional `cfg`. Call sites pass `_rng` positionally: `generate_beat(..., swing_amount, _rng, cfg=cfg)`.
- **Test directory strategy:** Tests chdir into `tmp_path` and pass a *relative* `name` to each generator. This is necessary because the generators' internal `name.split('-')[0]` directory derivation (carried over verbatim from music_gen.py) breaks when the input name contains a hyphen in a parent directory segment (e.g., pytest's `pytest-17` prefix). The chdir approach mirrors the production call convention (`song_name = "<20-digit-token>"` has no hyphens) without changing generator behavior.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Test path strategy for generators' existing name.split('-')[0] quirk**
- **Found during:** Task 2 (running freshly-created tests)
- **Issue:** Initial test code passed `name=str(tmp_path / f"song-{seed}" / f"song-{seed}-verse")`. The generators' internal `directory = name.split('-')[0]` logic truncates at the first hyphen — this works in production (where `song_name` is a 20-digit token with no hyphens) but fails when `tmp_path` contains `pytest-N` and the test name itself injects additional hyphens, producing paths like `/tmp/pytest` as the created directory while writing to `/tmp/pytest-17/.../song-<seed>-verse-melody.mid`.
- **Fix:** Rewrote all 4 determinism test classes to use `monkeypatch.chdir(tmp_path)` + pass a *relative* `name` like `"song{seed}-verse"`. This matches the production convention exactly (production also passes relative names, cwd-based) without any change to generator code. Added a docstring in each test file documenting the reason.
- **Files modified:** `tests/test_generators/test_chord.py`, `test_melody.py`, `test_bassline.py`, `test_beat.py`
- **Verification:** All 26 tests pass; full suite 368 passing.
- **Committed in:** `5fa3f0c` (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (Rule 3 - blocking test infrastructure; no generator code change).
**Impact on plan:** Zero impact on scope. Test pathing adjustment stays fully within test files; no generator source modified. All plan acceptance criteria still met.

## TimeSignatureRegistry Direct-Call Fallbacks

**None required.** The plan's `<behavior>` step 7-9 (step 5 in `<action>`) allowed falling back to `music_gen.verify_pattern_for_time_signature` / `verify_beat_pattern` wrappers if `TimeSignatureSpec` didn't expose the corresponding direct methods. Reading `timesig.py` confirmed that `TimeSignatureSpec.verify_chord_pattern_length(length)` and `TimeSignatureSpec.verify_beat_pattern_length(length)` both exist (lines 52 and 62 of timesig.py). All 4 generators use the direct spec methods — no fallback invocation.

## Attribute-Name Fallback (I-005)

**Not triggered.** Plan flagged the possibility that `config.Config.beat_roll_pattern_files` might not exist. Reading `config.py:56` confirmed the attribute exists with exactly the expected name and signature (`Dict[str, str]` keyed by time signature). `generate_beat` uses `_beat_cfg.beat_roll_pattern_files` as planned with no fallback.

## Issues Encountered

- **Test name hyphen collision** — documented under "Deviations" above. Resolved cleanly with monkeypatch.chdir.
- **ffmpeg warning** — pydub emits a `RuntimeWarning: Couldn't find ffmpeg or avconv` on import. Pre-existing environmental issue (not introduced by this plan); documented in Plan 03-03 SUMMARY. Does not affect tests.

## Shim Preservation Verified

The following kept-in-place code in `music_gen.py` was confirmed untouched via grep after Task 3:

- `def verify_pattern_for_time_signature(...)` (line 53) — kept per D-06
- `def verify_beat_pattern(...)` (line 59) — kept per D-06
- `def get_midi_time_signature_values(...)` (line 72) — kept per D-06
- `def mix_and_save(...)` (line 193) — kept per D-05
- `def create_song(...)` (line 352) — kept per D-05
- `def generate_song_parts(...)` (line 435) — kept per D-05 (call sites updated, signature unchanged)
- `def generate_song(...)` (line 471) — kept per D-05
- `if __name__ == "__main__":` guard (line 515) — kept per D-05

Bare `random.*` calls inside `mix_and_save` (soundfont/FX/layer probability draws) are intentionally untouched — they are Phase 4 scope (D-05).

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

**Ready for Plan 03-05 (Wave 5):**

- Generators are extracted, tested, and proven byte-deterministic under seeded RNG.
- The music21 audit comment (D-23) is in place in chord/melody/bassline; Plan 03-05 can now add `tests/test_music21_isolation.py` to continuously guard that `roman.RomanNumeral`, `scale.MajorScale`, `scale.MinorScale`, and `pitch.Pitch` never mutate global random state.
- `music_gen.py` shim size is ~524 lines (generators gone, sampler re-imports, mixer/helpers/create_song still present). Phase 4 can pick up from here to extract `mix_and_save` / `create_effect` / `generate_pedalboard` and close the remaining bare-random sites in the mixer.
- Full test baseline: **368 passing** — stable foundation for Plan 03-05 additions.

## Self-Check: PASSED

All 12 key-files exist on disk. All 3 task-commit hashes present in git log:
- `9256012` (Task 1: feat — generators package)
- `5fa3f0c` (Task 2: test — seeded determinism + AST guard)
- `3224e4e` (Task 3: refactor — music_gen.py shim)

Final plan-level verification commands all pass (generator imports, `pytest tests/`, AST no-bare-random scan, shim `__module__` resolution).

---
*Phase: 03-package-skeleton-sampler-generators-extraction*
*Completed: 2026-04-18*
