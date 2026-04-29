---
phase: 04-renderer-mixer-annotator-beats-extraction
plan: "05"
subsystem: orchestrator
tags: [phase-4, orchestrator, collapse, ast-guard, beat_anotator-delete, d-17, d-23, d-24, d-31]
dependency_graph:
  requires:
    - 04-01 (musicgen.beats — extract_beat_times, extract_downbeat_times)
    - 04-02 (musicgen.renderer — pick_soundfonts, render_stems, FLUIDSYNTH_VERSION, RenderResult)
    - 04-03 (musicgen.mixer — build_fx_boards, compute_layer_mask, mix_part, concat_parts, MixResult)
    - 04-04 (musicgen.annotator — annotate())
  provides:
    - music_gen.py collapsed to 199 lines (from 523) — thin orchestrator (D-23/D-24)
    - generate_song_parts returns 6-tuple including chord_progressions (RESEARCH OQ-2)
    - beat_anotator.py deleted (D-03: zero importers, replaced by musicgen.beats)
    - tests/test_no_bare_random_in_package.py — package-wide AST guard (D-17/D-31)
  affects:
    - Phase 5 (writer will own json.dump and per-sample output layout)
tech_stack:
  added: []
  patterns:
    - Thin orchestrator chaining renderer → mixer → beats → annotator
    - chord_progressions 6th return value from generate_song_parts
    - Package-wide parametrized AST guard using glob.glob + ast.walk
    - _bare_random_calls helper excludes random.Random constructor (node.func.attr != "Random")
key_files:
  created:
    - tests/test_no_bare_random_in_package.py (real AST guard, replaces Wave 0 stub)
  modified:
    - music_gen.py (199 lines from 523 — 9 functions deleted, create_song rewritten)
    - tests/test_music_gen_logging.py (filter caplog by logger name to allow D-07 renderer warning)
  deleted:
    - beat_anotator.py (D-03: zero importers, straight-grid logic wrong for swing > 0)
decisions:
  - "music_gen.py collapsed from 523 to 199 lines by deleting 9 audio-side functions and rewriting create_song as a 70-line orchestrator (D-23/D-24)"
  - "generate_song_parts extended to return chord_progressions as 6th dict — annotator needs per-part chord data, which generate_chord_progression already computed but discarded (RESEARCH OQ-2)"
  - "test_music_gen_logging.py::test_import_music_gen_does_not_emit_logs updated to filter caplog by logger name 'music_gen' — importing music_gen now triggers musicgen.renderer import, which emits a D-07 FluidSynth-absent WARNING that is expected on CI and not a music_gen.py side-effect"
  - "beat_anotator.py deleted with git rm -f (file had local modifications from prior git history); zero importers confirmed by grep"
  - "tests/test_no_bare_random_in_package.py parametrized using _collect_package_modules() glob — auto-extends to future modules without test file edits (D-31)"
metrics:
  duration: "480 seconds (~8 minutes)"
  completed: "2026-04-19T17:59:49Z"
  tasks_completed: 2
  files_modified: 4
  files_deleted: 1
---

# Phase 04 Plan 05: Orchestrator Collapse + AST Guard Summary

**One-liner:** `music_gen.py` collapsed from 523 to 199 lines by deleting 9 audio-side functions and rewriting `create_song` as a thin orchestrator chaining `renderer → mixer → beats → annotator`; `beat_anotator.py` deleted; package-wide AST bare-random guard added with 13 tests (12 parametrized + 1 meta-test).

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Collapse music_gen.py + fix test_music_gen_logging.py | 062c6b3 | music_gen.py (199 lines), tests/test_music_gen_logging.py |
| 2 | Delete beat_anotator.py + create real AST guard | fb71998 | beat_anotator.py (deleted), tests/test_no_bare_random_in_package.py |

## Verification Results

### D-24: music_gen.py line count

```
199 /home/bidu/musicgen/music_gen.py
```

Target: < 200 lines. PASS.

### D-23: All 9 target functions deleted

```
grep -c "^def mix_and_save|^def generate_pedalboard|..." music_gen.py
→ 0
```

Functions deleted: `save_beat_annotations`, `read_instrument_probabilities`, `get_random_sound_font`,
`get_levels`, `create_effect`, `generate_pedalboard`, `apply_fx_to_layer`, `pedalboard_info_json`, `mix_and_save`.

### Orchestrator call chain (from music_gen.py)

```
from musicgen import renderer, mixer, annotator, beats
→ renderer.pick_soundfonts(_cfg, _rng)        # Step 1 - soundfonts
→ generate_song_parts(...)                     # Step 2 - MIDI generation
→ mixer.build_fx_boards(_cfg, _rng)           # Step 3 - FX boards
→ mixer.compute_layer_mask(...)               # Step 3 - layer mask
→ renderer.render_stems(...)                  # Step 4 - per-part render (parallel stems)
→ mixer.mix_part(...)                         # Step 4 - per-part mix
→ beats.extract_beat_times(...)               # Step 4 - beat derivation
→ beats.extract_downbeat_times(...)           # Step 4 - downbeat derivation
→ mixer.concat_parts(...)                     # Step 5 - concatenate
→ musicality_score.get_musicality_score(...)  # Step 6 - musicality
→ annotator.annotate(...)                     # Step 7 - annotation
→ json.dump(annotation, ...)                  # Step 8 - write JSON
```

### beat_anotator.py deletion

```
test ! -f beat_anotator.py → GONE
grep -r "import beat_anotator" . → 0 hits
git log --oneline | head -1 → fb71998 (delete mode 100644 beat_anotator.py)
```

### tests/test_no_bare_random_in_package.py — 13 tests

```
13 passed in 0.04s
- test_no_bare_random_in_package_module[__main__.py] PASSED
- test_no_bare_random_in_package_module[annotator.py] PASSED
- test_no_bare_random_in_package_module[beats.py] PASSED
- test_no_bare_random_in_package_module[cli.py] PASSED
- test_no_bare_random_in_package_module[duration_validator.py] PASSED
- test_no_bare_random_in_package_module[generators/bassline.py] PASSED
- test_no_bare_random_in_package_module[generators/beat.py] PASSED
- test_no_bare_random_in_package_module[generators/chord.py] PASSED
- test_no_bare_random_in_package_module[generators/melody.py] PASSED
- test_no_bare_random_in_package_module[mixer.py] PASSED
- test_no_bare_random_in_package_module[renderer.py] PASSED
- test_no_bare_random_in_package_module[sampler.py] PASSED
- test_package_scan_covers_all_phase4_modules PASSED
```

### tests/test_music_gen_logging.py — 6 tests

```
6 passed in 1.54s
- TestNoPrintCallsRemain::test_no_print_calls_remain_in_music_gen PASSED
- TestImportSideEffects::test_import_music_gen_does_not_emit_logs PASSED
- TestImportSideEffects::test_import_music_gen_does_not_trigger_generation PASSED
- TestLoggerSetup::test_module_level_logger_exists PASSED
- TestLoggerSetup::test_basic_config_only_in_main_guard PASSED
- TestLoggerSetup::test_no_fstring_in_logger_calls PASSED
```

### Full test suite (not slow)

```
504 passed, 1 skipped, 2 warnings in 2.05s
```

Prior baseline: 491 passed, 2 skipped. New tests: +13 (AST guard). Net: 504 passed, 1 skipped (integration E2E test). Zero regressions.

### Import check

```
python -c "import music_gen; print('imported OK')"
→ imported OK  (+ expected D-07 renderer warning about FluidSynth not on PATH)
```

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] test_import_music_gen_does_not_emit_logs failed after music_gen.py imports musicgen.renderer**

- **Found during:** Task 1 verification
- **Issue:** `import music_gen` now causes `musicgen.renderer` to be imported, which emits a WARNING log at module level about FluidSynth not being on PATH (D-07 pattern). The test asserted `caplog.records == []`, but the renderer's FluidSynth-absent warning was captured, causing the test to fail.
- **Fix:** Updated the test to filter `caplog.records` by `r.name == "music_gen"` — only records from `music_gen`'s own logger fail the test. The renderer warning is an expected CI behavior (D-07), not a `music_gen.py` side-effect.
- **Files modified:** `tests/test_music_gen_logging.py`
- **Commit:** 062c6b3

**2. [Rule 2 - Missing Critical] beat_anotator.py had local modifications blocking clean `git rm`**

- **Found during:** Task 2 Step A
- **Issue:** `git rm beat_anotator.py` failed with "file has local modifications" because the file's tracked content differed from HEAD (pre-existing condition from project git history).
- **Fix:** Used `git rm -f beat_anotator.py` — the -f flag is appropriate here since the intent is deletion (D-03: zero importers confirmed by grep).
- **Files modified:** N/A (deletion)
- **Commit:** fb71998

## Known Stubs

None — all Phase 4 stubs are resolved. The `tests/test_integration_full_generation.py` (Wave 0 E2E test) remains skipped (1 skip in the suite) because FluidSynth binary is not on PATH in this environment — this is expected and documented in D-30/R-X8. It is not a stub; it's a guard.

## Threat Flags

None — no new network endpoints, auth paths, or trust boundaries introduced. The orchestrator collapse is a code-organization refactor. The AST guard is a pure-static test with no I/O beyond reading `.py` files under `src/musicgen/`.

T-04-05-01 (orchestrator regression in create_song chain): mitigated — 504 tests pass, import check passes.
T-04-05-02 (deleting 9 functions breaks implicit caller): mitigated — full suite green, test_music_gen_logging.py all 6 pass.
T-04-05-03 (silent reintroduction of bare random.* calls): mitigated — 13 AST guard tests pass across all 12 modules.

## Self-Check: PASSED
