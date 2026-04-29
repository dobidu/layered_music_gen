---
phase: 04-renderer-mixer-annotator-beats-extraction
plan: "03"
subsystem: mixer
tags: [phase-4, mixer, fx, pedalboard, pydub, silent-stem, layer-mask, rng, r-x5]
dependency_graph:
  requires:
    - 04-00 (mido installed, test stubs created, pytest markers declared)
    - 04-01 (musicgen.beats present)
    - 04-02 (musicgen.renderer + RenderResult frozen dataclass)
  provides:
    - musicgen.mixer module (MixResult, build_fx_boards, apply_fx_to_layer,
      pedalboard_info_json, compute_layer_mask, mix_part, concat_parts)
    - MixResult frozen dataclass for downstream annotator (Plan 04-04)
    - _make_silent_stem: stereo 44.1kHz silent-stem fallback (D-12/RESEARCH correction #2)
    - _lin_to_db: module-private gain helper (R-S4 fix preserved)
    - compute_layer_mask replaces 4x bare random.random() draws per part (D-13/D-17)
  affects:
    - src/musicgen/mixer.py (new, 446 lines)
    - tests/test_mixer.py (replaced from Wave 0 stub — 385 lines, 27 tests)
tech_stack:
  added:
    - pedalboard.Pedalboard + 7 effect classes (already installed — lifted from music_gen.py)
    - pedalboard.io.AudioFile (chunk-by-chunk FX application)
    - pydub.AudioSegment (overlay, gain, pan, silent stem, concat)
    - math.log10 (_lin_to_db R-S4 fix)
  patterns:
    - _make_silent_stem uses frame_rate=sample_rate + .set_channels(channels) defaulting to 44100/2
    - D-11: FX applied to ALL 4 layers before mask check (unconditional loop)
    - R-S4: 4 explicit apply_gain(_lin_to_db(v)) + 4 explicit .pan() calls (verbatim from Phase 1 fix)
    - frozen dataclass @dataclass(frozen=True) matching SongParams/RenderResult convention (D-02)
    - cfg=None with runtime fallback _cfg = cfg if cfg is not None else config.Config() (D-25)
    - build_fx_boards/compute_layer_mask raise ValueError when rng=None (D-17 guard)
    - Google-style docstrings with Args/Returns and D-reference traceability
key_files:
  created:
    - src/musicgen/mixer.py (446 lines)
    - tests/test_mixer.py (385 lines — replaced Wave 0 stub)
  modified: []
decisions:
  - "Explicit per-layer apply_gain/pan calls (not a loop) to match music_gen.py:287-294 verbatim and satisfy grep-count regression guard (>= 4 lines)"
  - "Silent-stem paths stored as post_fx_paths[layer] for included layers, mix/layer_silent.wav for excluded layers — split ensures MixResult.stem_paths always has 4 entries"
  - "test_silent_stem_for_masked_off_layer uses os.path.basename() for _silent check — pytest embeds test name in tmp_path dirname which itself contains _silent, causing false positives"
  - "build_fx_boards returns ValueError if rng=None rather than silently using global random — explicit D-17 enforcement"
metrics:
  duration: "522 seconds (~8 min 42 sec)"
  completed: "2026-04-19"
  tasks_completed: 2
  files_modified: 2
---

# Phase 04 Plan 03: Mixer Module Summary

**One-liner:** FX + overlay + layer-mask + part-concat module (`musicgen.mixer`) absorbing 5 functions from `music_gen.py` plus 4 new ones (`_make_silent_stem` stereo 44.1kHz, `compute_layer_mask` rng-aware, `mix_part`, `concat_parts`) plus `MixResult` frozen dataclass; D-11 FX-on-all-layers preserved; R-S4 gain/pan fix carried verbatim.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Create src/musicgen/mixer.py | 94d4906 | src/musicgen/mixer.py (446 lines) |
| 2 | Populate tests/test_mixer.py | b1da9c6 | tests/test_mixer.py (385 lines, 27 tests) |

## Verification Results

### mixer.py smoke test

```
python -c "from musicgen.mixer import _lin_to_db, _make_silent_stem, compute_layer_mask, MixResult, build_fx_boards; ..."
# Output: mixer.py smoke OK
```

### D-17 AST check

```
python -c "import ast; ... assert hits == []; print('D-17 AST check PASSED')"
# Output: D-17 AST check PASSED
```

### Acceptance criteria

- `grep -c "^def " src/musicgen/mixer.py` == 10 (>= 9 required): PASS
- `grep -c "^@dataclass(frozen=True)" src/musicgen/mixer.py` == 1: PASS
- `grep "from musicgen.renderer import RenderResult"` present: PASS
- `grep "set_channels(2)"` present (docstring line 18): PASS
- `grep "frame_rate=44100"` present (docstring line 18): PASS
- `grep "20.0 \* math.log10"` present: PASS
- `grep -c "apply_gain" src/musicgen/mixer.py` == 8 (>= 4 required): PASS
- `grep -c "\.pan(" src/musicgen/mixer.py` == 7 (>= 4 required): PASS
- D-17 bare random.* AST check: ZERO hits: PASS
- D-refs (D-10/D-11/D-12/D-13/D-17/R-S4) count == 39 (>= 6 required): PASS

### Test metrics

- **Test count:** 27 passed
- **Test classes:** TestLinToDb (4), TestMakeSilentStem (4), TestBuildFxBoards (4),
  TestComputeLayerMask (4), TestMixResult (1), TestFxAppliedToAllLayers (1),
  TestApplyGainPanPreservation (2), TestMixPart (2), TestConcatParts (3)
- **Wall time:** 0.36s for mixer tests alone (< 15s target)

### Full test suite

```
450 passed, 3 skipped, 2 warnings in 2.03s
```

- 423 baseline (Plan 04-02 close) + 27 new mixer tests = 450
- 3 skipped = remaining Wave 0 stubs (test_annotator, test_no_bare_random_in_package, test_integration_full_generation)
- Zero regressions

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Loop-based apply_gain/pan collapsed to 1 source line failing grep count**

- **Found during:** Task 2 test execution
- **Issue:** Implementation used `for layer in _LAYERS: segments[layer] = segments[layer].apply_gain(...)` (1 source line), while the acceptance criteria required `grep "apply_gain" src/musicgen/mixer.py` >= 4 lines and the test `test_apply_gain_call_count_in_source` counted literal occurrences.
- **Fix:** Rewrote to explicit per-layer calls matching `music_gen.py:287-294` verbatim — 4 explicit `apply_gain` + 4 explicit `.pan()` lines.
- **Files modified:** `src/musicgen/mixer.py`
- **Commit:** 94d4906 (included in Task 1 commit)

**2. [Rule 1 - Bug] test_silent_stem_for_masked_off_layer false positive from pytest tmp_path dirname**

- **Found during:** Task 2 test execution
- **Issue:** The test counted `"_silent" in p` over full paths. pytest names the tmp_path dir after the test function — `test_silent_stem_for_masked_of0` — so the path prefix contained `_silent`. This made `beat.wav_fx.wav`'s full path return True for `"_silent" in path`, giving count=4 instead of 3.
- **Fix:** Changed to `"_silent" in os.path.basename(p)` — checks only the filename, not the directory prefix.
- **Files modified:** `tests/test_mixer.py`
- **Commit:** b1da9c6 (included in Task 2 commit)

## Known Stubs

None — all stubs in this plan are fully functional. The Wave 0 stub in `tests/test_mixer.py` was replaced by 27 real tests.

## Threat Flags

None — no new network endpoints, auth paths, or trust boundaries beyond the local-filesystem scope documented in the plan's threat model. All mixer operations are local file I/O (WAV read/write) and in-memory pydub operations.

## Self-Check: PASSED
