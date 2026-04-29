---
phase: 04-renderer-mixer-annotator-beats-extraction
verified: 2026-04-19T18:12:10Z
status: human_needed
score: 19/19 must-haves verified
overrides_applied: 0
human_verification:
  - test: "Run `pytest -m slow` on a machine with FluidSynth binary + populated sf/<layer>/ dirs"
    expected: "Both TestFullGenerationPipeline::test_one_part_full_pipeline and TestMidiReproducibility::test_same_seed_produces_same_midi PASS — 4 stems, 1 mix WAV, 4 MIDI files produced; annotation dict has all R-P4 Phase-4 fields non-None and Phase-5 TBD fields as None; beat MIDI is bit-identical across two seeded runs"
    why_human: "FluidSynth binary not on PATH in this environment; sf/<layer>/ dirs are empty; both E2E tests are guarded by module-level pytestmark skipif. Cannot verify FluidSynth subprocess path, real audio output, or WAV/MIDI artifact presence without the binary and soundfonts."
---

# Phase 4: Renderer + Mixer + Annotator + Beats Extraction — Verification Report

**Phase Goal:** Extract the audio-side of the pipeline. Decompose `mix_and_save` into `renderer`, `mixer`, and `annotator`. Replace `beat_anotator.py` with swing-aware beat derivation.
**Verified:** 2026-04-19T18:12:10Z
**Status:** human_needed
**Re-verification:** No — initial verification

## Goal Achievement

All 19 automated must-haves verified. The only open item is the E2E `@pytest.mark.slow` test execution, which requires a FluidSynth binary and populated soundfont directories not present in this environment.

### Observable Truths

| #  | Truth | Status | Evidence |
|----|-------|--------|----------|
| 1  | `src/musicgen/renderer.py` exists with `FLUIDSYNTH_VERSION`, `RenderResult`, `pick_soundfonts`, `render_stems` (R-X4) | VERIFIED | File exists at 209 lines; all 4 symbols confirmed present and importable |
| 2  | `src/musicgen/mixer.py` exists with `build_fx_boards`, `apply_fx_to_layer`, `pedalboard_info_json`, `_lin_to_db`, `MixResult`, `compute_layer_mask`, `mix_part`, `concat_parts`, `_make_silent_stem` (R-X5) | VERIFIED | File exists at 447 lines; `create_effect` → `_create_effect` (intentional private rename per 04-03-PLAN.md line 52; all listed public symbols confirmed) |
| 3  | `src/musicgen/annotator.py` exists with pure `annotate` function — no `open(`, no `json.dump`, no `os.makedirs` inside body (R-X6, D-14) | VERIFIED | Grep on `annotate()` body finds zero `open(`/`json.dump`/`os.makedirs`; `test_does_not_call_open` monkeypatches `builtins.open` to raise and confirms annotate passes |
| 4  | `src/musicgen/beats.py` exists with `beat_duration`, `extract_beat_times` (mido.tick2second), `extract_downbeat_times` (TIME-GRID, not stride-slice) (R-X7) | VERIFIED | File exists at 116 lines; docstring explicitly documents "TIME-GRID derivation, NOT stride-slice" per RESEARCH correction #1; TimeSignatureRegistry.lookup() used to compute measure duration |
| 5  | `beat_anotator.py` does NOT exist (D-03 delete) | VERIFIED | `test ! -f beat_anotator.py` → GONE; zero `import beat_anotator` hits in codebase; 04-05-SUMMARY confirms `git rm -f beat_anotator.py` with commit fb71998 |
| 6  | `music_gen.py` is < 200 lines (D-24); `mix_and_save` fully deleted (D-23 exit criterion met trivially) | VERIFIED | `wc -l music_gen.py` → 199; D-23 criterion "< 50 lines of orchestration" satisfied by full deletion |
| 7  | All 9 target functions gone from `music_gen.py`: `save_beat_annotations`, `read_instrument_probabilities`, `get_random_sound_font`, `get_levels`, `create_effect`, `generate_pedalboard`, `apply_fx_to_layer`, `pedalboard_info_json`, `mix_and_save` | VERIFIED | `grep -n "^def mix_and_save|^def generate_pedalboard|..."` returns NONE FOUND |
| 8  | `tests/test_no_bare_random_in_package.py` scans all `src/musicgen/**/*.py` and passes (D-17/D-31) | VERIFIED | 13 tests pass: 12 parametrized module tests + 1 meta-test; covers sampler, renderer, mixer, annotator, beats, duration_validator, generators/* |
| 9  | Integration test exists at `tests/test_integration_full_generation.py` with `@pytest.mark.slow` + `shutil.which("fluidsynth")` guard (R-X8) | VERIFIED | File exists at 246 lines; module-level `pytestmark` list with `pytest.mark.slow` + 2 `pytest.mark.skipif` guards (FluidSynth binary + sf2 pool); 2 test classes: `TestFullGenerationPipeline` + `TestMidiReproducibility` |
| 10 | Full suite green: 504 tests pass (133 above Phase 3 baseline of 371) | VERIFIED | `pytest -m "not slow" -q` → `504 passed, 2 deselected, 2 warnings in 2.01s` |
| 11 | `mido>=1.3.3` in `pyproject.toml [project].dependencies` (Wave 0 RESEARCH correction #3) | VERIFIED | Line present: `"mido>=1.3.3",` with comment explaining it is not a transitive dep; `mido.version_info` = 1.3.3 |
| 12 | `[tool.pytest.ini_options] markers` declares both `slow` and `integration` (Wave 0) | VERIFIED | Both markers present in pyproject.toml; no UnknownMarkWarning in suite output |
| 13 | RESEARCH correction #1 encoded: `extract_downbeat_times` uses time-grid formula (not `beat_times[::numerator]` stride-slice) | VERIFIED | beats.py docstring explicitly documents the decision; TimeSignatureRegistry.lookup() computes measure duration; `TestExtractDownbeatTimes::test_downbeat_grid_independent_of_beat_times_input` confirms output unchanged whether beat_times is `[]` or garbage |
| 14 | RESEARCH correction #2 encoded: `_make_silent_stem` uses `frame_rate=44100` and `.set_channels(2)` (stereo 44.1kHz) | VERIFIED | mixer.py line 89-90: `AudioSegment.silent(duration=duration_ms, frame_rate=sample_rate)` + `.set_channels(channels)` with defaults `sample_rate=44100, channels=2`; `TestMakeSilentStem::test_default_channels_2` and `test_default_frame_rate_44100` pass |
| 15 | D-11 preserved: FX applied to all 4 layers regardless of `layer_mask` — regression test passes | VERIFIED | `TestFxAppliedToAllLayers::test_fx_applied_to_all_4_layers_regardless_of_mask` patches `apply_fx_to_layer` with a counter, passes all-False mask, asserts call_counter == 4 |
| 16 | D-16 preserved: annotator dict has `seed`, `musicgen_version`, `split`, `pre_roll_offset_seconds` keys present as `None` (not missing) | VERIFIED | annotator.py lines 169-173 explicitly assign all four; `TestTbdFieldsAreNone` passes for all 4 keys |
| 17 | R-S4 preserved: `_lin_to_db` helper in `mixer.py` + `apply_gain` + `.pan()` return captured (not overwritten from Phase 1) | VERIFIED | mixer.py lines 55-68 (`_lin_to_db`); lines 358-365 (4 explicit `apply_gain` + 4 explicit `.pan()` capturing returns); `test_apply_gain_call_count_in_source` asserts `>= 4` for each |
| 18 | `beat_duration` in `beats.py` is primary definition; `generators/beat.py` re-exports it as compatibility alias (D-21) | VERIFIED | beats.py lines 29-44 define `beat_duration`; generators/beat.py line 31: `from musicgen.beats import beat_duration  # noqa: F401  (D-21 re-export)` |
| 19 | Phase 3 baseline regression: all Phase 3 tests still passing | VERIFIED | Full suite 504 pass (2 deselected slow) confirms zero regressions against Phase 3's 371-test baseline |

**Score:** 19/19 truths verified (automated); 1 item deferred to human verification (E2E slow test execution)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/musicgen/renderer.py` | R-X4 FluidSynth wrapper | VERIFIED | 209 lines; `FLUIDSYNTH_VERSION`, `RenderResult`, `pick_soundfonts`, `render_stems`; D-07 version capture with fallback |
| `src/musicgen/mixer.py` | R-X5 FX + overlay + concat | VERIFIED | 447 lines; all public symbols present; `_create_effect` private (intentional per 04-03-PLAN.md) |
| `src/musicgen/annotator.py` | R-X6 pure annotation function | VERIFIED | 182 lines; zero I/O inside `annotate()`; D-15/D-16 semantics enforced |
| `src/musicgen/beats.py` | R-X7 MIDI-tick derivation | VERIFIED | 116 lines; time-grid downbeat derivation; mido.tick2second; `beat_duration` primary definition |
| `tests/test_beats.py` | R-X7 swing-case unit tests | VERIFIED | 168 lines; 3 swing cases (0.5, 0.66, 0.75); time-grid downbeat tests; D-21 re-export tested implicitly |
| `tests/test_renderer.py` | R-X4 mocked unit tests | VERIFIED | 213 lines; FluidSynth mocked via `unittest.mock.patch`; seeded determinism tests |
| `tests/test_mixer.py` | R-X5 seeded + D-11 tests | VERIFIED | 386 lines; D-11 regression guard; R-S4 static grep; silent-stem stereo parity |
| `tests/test_annotator.py` | R-X6 fixture-driven pure tests | VERIFIED | 402 lines; purity contract test; D-16 None semantics; mode derivation; song_arrangement shape |
| `tests/test_no_bare_random_in_package.py` | D-17/D-31 AST guard | VERIFIED | 90 lines; 12 parametrized + 1 meta test; all 13 pass |
| `tests/test_integration_full_generation.py` | R-X8 E2E slow test | VERIFIED (structure) | 246 lines; substantive implementation (not stub); 2 test classes with full assertions; skipped on CI without FluidSynth |
| `music_gen.py` | Thin orchestrator < 200 lines, no deleted functions | VERIFIED | 199 lines; imports `renderer, mixer, annotator, beats`; `create_song` is 70-line orchestration chain |
| `beat_anotator.py` | Deleted (D-03) | VERIFIED (absent) | File absent; zero importers; git rm -f commit fb71998 |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `music_gen.py:create_song` | `musicgen.renderer.pick_soundfonts` | import + call | WIRED | Line 14 import; line 66 call |
| `music_gen.py:create_song` | `musicgen.renderer.render_stems` | import + call | WIRED | Line 103 call per-part |
| `music_gen.py:create_song` | `musicgen.mixer.build_fx_boards` | import + call | WIRED | Line 78 call |
| `music_gen.py:create_song` | `musicgen.mixer.compute_layer_mask` | import + call | WIRED | Line 81 call |
| `music_gen.py:create_song` | `musicgen.mixer.mix_part` | import + call | WIRED | Line 104 call per-part |
| `music_gen.py:create_song` | `musicgen.mixer.concat_parts` | import + call | WIRED | Line 118 call |
| `music_gen.py:create_song` | `musicgen.beats.extract_beat_times` | import + call | WIRED | Line 111 call per-part |
| `music_gen.py:create_song` | `musicgen.beats.extract_downbeat_times` | import + call | WIRED | Line 114 call per-part |
| `music_gen.py:create_song` | `musicgen.annotator.annotate` | import + call | WIRED | Line 128 call |
| `musicgen.annotator` | `musicgen.sampler.SongParams` | import | WIRED | annotator.py line 22 |
| `musicgen.annotator` | `musicgen.renderer.RenderResult` | import | WIRED | annotator.py line 23 |
| `musicgen.annotator` | `musicgen.mixer.MixResult` | import | WIRED | annotator.py line 24 |
| `musicgen.mixer` | `musicgen.renderer.RenderResult` | import | WIRED | mixer.py line 45 |
| `musicgen.beats` | `timesig.TimeSignatureRegistry` | import + `lookup()` call | WIRED | beats.py line 24; used in `extract_downbeat_times` |
| `generators/beat.py` | `musicgen.beats.beat_duration` | re-export alias | WIRED | generators/beat.py line 31: `from musicgen.beats import beat_duration  # noqa: F401  (D-21 re-export)` |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `music_gen.py:create_song` | `render_results` | `renderer.render_stems()` → real FluidSynth subprocess | Yes (guarded by human E2E test) | FLOWING (unit tests mock subprocess; E2E tests need FluidSynth binary) |
| `music_gen.py:create_song` | `mix_results` | `mixer.mix_part()` → real pydub overlay + pedalboard | Yes (confirmed by unit tests with synthetic WAVs) | FLOWING |
| `music_gen.py:create_song` | `beat_times_dict` | `beats.extract_beat_times()` → real mido MIDI parse | Yes (confirmed by TestExtractBeatTimes with real MIDI fixtures) | FLOWING |
| `music_gen.py:create_song` | `annotation` | `annotator.annotate()` → pure function | Yes (confirmed by TestAnnotateShape) | FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| renderer module importable, FLUIDSYNTH_VERSION is non-empty string | Python import + `isinstance(FLUIDSYNTH_VERSION, str)` | `FLUIDSYNTH_VERSION: 'unknown'` (fallback, CI-expected) | PASS |
| mixer symbols importable | `from musicgen.mixer import build_fx_boards, ...` | All imports succeed | PASS |
| beats.extract_downbeat_times is time-grid (independent of beat_times input) | `extract_downbeat_times([], "4/4", 3, 0.0, 120) == extract_downbeat_times([999.0], "4/4", 3, 0.0, 120)` | TestExtractDownbeatTimes::test_downbeat_grid_independent_of_beat_times_input passes | PASS |
| annotate() is pure (no open()) | monkeypatched open → raises in test | TestAnnotatorIsPure::test_does_not_call_open passes | PASS |
| Full suite minus slow: 504 pass | `pytest -m "not slow" -q` | `504 passed, 2 deselected` | PASS |
| E2E test execution with real FluidSynth | `pytest -m slow` on machine with fluidsynth binary | SKIPPED (binary absent) | SKIP — human needed |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| R-X4 | 04-02-renderer-module-PLAN.md | Extract `renderer.py` with ThreadPoolExecutor, FluidSynth version capture | SATISFIED | `src/musicgen/renderer.py` exists with all required symbols; 9 unit tests pass |
| R-X5 | 04-03-mixer-module-PLAN.md | Extract `mixer.py` with FX + pydub overlay + fixed gain/pan APIs | SATISFIED | `src/musicgen/mixer.py` exists; R-S4 gain/pan preserved; 27 unit tests pass |
| R-X6 | 04-04-annotator-module-PLAN.md | Pure `annotate()` function with R-P4 schema subset | SATISFIED | `src/musicgen/annotator.py` exists; zero I/O inside annotate; D-15/D-16 semantics; 32 unit tests pass |
| R-X7 | 04-01-beats-module-PLAN.md | Replace `beat_anotator.py` with MIDI-tick beats module | SATISFIED | `src/musicgen/beats.py` exists; beat_anotator.py deleted; time-grid downbeat; 27 unit tests pass |
| R-X8 | 04-06-e2e-integration-test-PLAN.md | One `@pytest.mark.slow` E2E test with file layout + annotation assertions | PARTIALLY SATISFIED | File exists with substantive implementation (not stub); 2 test classes; 2 guards (FluidSynth + sf2); EXECUTION awaits human verification |

Additional requirements addressed or preserved:

| Requirement | Status | Evidence |
|-------------|--------|----------|
| R-S4 (pydub gain/pan fix, Phase 1) | PRESERVED | `_lin_to_db` in mixer.py; 4 explicit `apply_gain` + 4 explicit `.pan()` returning segments; static grep test guards regression |
| R-P4 (sample.json schema, Phase 4 subset) | PARTIALLY SATISFIED | All Phase-4 fillable fields present in annotator output; Phase-5 TBD fields correctly present as None |
| R-P2 (stems-sum-to-mix) | DEFERRED TO PHASE 5 | Stems ARE persisted by mixer (enabling the assertion); numeric sum-check is Phase 5 per design |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `src/musicgen/mixer.py` | 15, 318 | `TODO is OUT OF SCOPE` comments | INFO | These are intentional documentation notes (D-11: FX-on-all-layers preserved by design); not implementation stubs |

No blockers found.

### Human Verification Required

#### 1. E2E Integration Test (`@pytest.mark.slow`)

**Test:** On a machine with FluidSynth binary on PATH and populated `sf/<layer>/` directories (at least 1 `.sf2` per layer), run:
```
pytest -m slow tests/test_integration_full_generation.py -v
```
**Expected:**
- `TestFullGenerationPipeline::test_one_part_full_pipeline` PASSES:
  - `intgen/` dir created with `intgen.wav` (mix) and `intgen.json` (annotation)
  - `intgen/intgen-intro/` subdir has >= 4 stem WAVs and >= 4 MIDI files
  - Annotation dict has all Phase-4 fields non-None (`key`, `mode`, `tempo_bpm`, `time_signature`, `time_signatures_per_part`, `measures_per_part`, `swing`, `song_arrangement`, `chord_progression`, `active_layers`, `soundfonts`, `fx_params`, `beat_times`, `downbeat_times`, `musicality_score`, `duration_seconds`, `fluidsynth_version`, `mix`, `stems`, `midi`)
  - Phase-5 TBD fields present as None (`seed`, `musicgen_version`, `split`, `pre_roll_offset_seconds`)
  - `analysis_failed` NOT in annotation (omitted on success)
  - `downbeat_times["intro"]` has exactly 2 entries (2 measures of 4/4 at 120 BPM)
  - JSON round-trip valid

- `TestMidiReproducibility::test_same_seed_produces_same_midi` PASSES:
  - Two `create_song(seed=42)` runs produce bit-identical beat/melody/harmony/bassline MIDI

**Why human:** FluidSynth binary not on PATH in this environment (`shutil.which("fluidsynth") is None`); sf/<layer>/ directories are empty. Both conditions trigger the module-level `pytestmark` skipif guard, causing 2 SKIPPED with reason "fluidsynth binary not on PATH" / "one or more sf/<layer>/ dirs is empty". Cannot verify real audio rendering, real musicality scoring, or real artifact layout without the system dependencies.

### Gaps Summary

No gaps. All 19 automated must-haves verified. The only outstanding item is the E2E FluidSynth-dependent test execution (R-X8 functional run), which is an environment dependency, not a code gap.

### Requirements Status Recommendation

After human verification passes, REQUIREMENTS.md R-X4, R-X5, R-X6, R-X7, R-X8 should be marked CLOSED, parallel to how R-X1/R-X2/R-X3 were marked when Phase 3 finished. Suggested closing reference lines:

- `R-X4`: **[Status: CLOSED 2026-04-19 by Plan 04-02.](/home/bidu/musicgen/.planning/phases/04-renderer-mixer-annotator-beats-extraction/04-02-SUMMARY.md)**
- `R-X5`: **[Status: CLOSED 2026-04-19 by Plan 04-03.](/home/bidu/musicgen/.planning/phases/04-renderer-mixer-annotator-beats-extraction/04-03-SUMMARY.md)**
- `R-X6`: **[Status: CLOSED 2026-04-19 by Plan 04-04.](/home/bidu/musicgen/.planning/phases/04-renderer-mixer-annotator-beats-extraction/04-04-SUMMARY.md)**
- `R-X7`: **[Status: CLOSED 2026-04-19 by Plan 04-01.](/home/bidu/musicgen/.planning/phases/04-renderer-mixer-annotator-beats-extraction/04-01-SUMMARY.md)**
- `R-X8`: **[Status: CLOSED 2026-04-19 by Plan 04-06.](/home/bidu/musicgen/.planning/phases/04-renderer-mixer-annotator-beats-extraction/04-06-SUMMARY.md)**

---

_Verified: 2026-04-19T18:12:10Z_
_Verifier: Claude (gsd-verifier)_
