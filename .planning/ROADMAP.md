# ROADMAP

**Project:** musicgen — synthetic music dataset generator
**Milestone v0.1:** First shippable dataset generator (Stabilize + Extract + Productize)
**Granularity:** Standard (5–8 phases, 3–5 plans each)
**Execution:** Parallel plans where independent

Each phase has a clear goal, must-have deliverables, requirement coverage from `.planning/REQUIREMENTS.md`, and dependencies. Extend and Research work is intentionally out of this milestone.

---

## Phase 1: Stabilize I — bug fixes and guardrails

**Goal:** `music_gen.py` becomes importable and stops silently corrupting outputs. Two real bugs found during research are fixed. First test suite lands.

**Depends on:** nothing (entry phase).

**Deliverables**

- `if __name__ == "__main__":` guard added (`music_gen.py:1158-1161`).
- Arrangement is computed once and passed into `mix_and_save`; the re-call at `music_gen.py:760` is removed.
- `pydub` gain/pan bug fixed at `music_gen.py:845-852` using `apply_gain(...)` and capturing the `.pan(...)` return.
- `from music21 import *` replaced with explicit imports.
- Bare `except:` blocks in `musicality_score.py` narrowed; `logger.exception` on failure.
- Dead imports and variables removed.
- `uuid` stub removed from `requirements.txt`.
- First pytest skeleton: `pytest` + `pytest-cov` installed via a dev-deps path (temporary; pyproject.toml lands in Phase 3). Unit tests for `verify_pattern_for_time_signature`, `verify_beat_pattern`, `validate_measures`, `DurationValidator`.

**Requirements covered:** R-S1, R-S2, R-S3, R-S4, R-S7 (partial), R-S8, R-Q2 (initial).

**Exit criteria:** (1) `python -c "import music_gen"` does not trigger generation. (2) A seeded run produces mix audio that reflects `levels.json` (pre-fix audio can be captured as a "before" fixture for comparison). (3) Pure-function unit tests all green.

**Plans:** 4 plans

Plans:
- [x] 01-01-importability-and-arrangement-fix-PLAN.md — Wrap top-level execution in `__main__` guard and fix the arrangement re-roll bug (PITFALLS P-A) by computing arrangement once in `create_song` and threading it into `mix_and_save`. (R-S1, R-S3)
- [x] 01-02-pydub-gain-pan-fix-PLAN.md — Fix pydub gain/pan no-op (PITFALLS P-B) so `levels.json` actually affects output. (R-S4)
- [x] 01-03-code-hygiene-PLAN.md — Explicit `music21` imports, narrowed exception handlers in `musicality_score.py`, dead-code/import removal in `music_gen.py`, and `uuid` stub removed from `requirements.txt`. (R-S2, R-S7, R-S8)
- [x] 01-04-pytest-skeleton-and-pure-function-tests-PLAN.md — First pytest skeleton via `dev-requirements.txt` plus unit tests for `verify_pattern_for_time_signature`, `verify_beat_pattern', `validate_measures`, and `DurationValidator`. (R-S7, R-Q2)

---

## Phase 2: Stabilize II — config + time-signature registry + logging

**Goal:** Kill the hardcoded-path and scattered-time-signature debt so Phase 3 extraction is safe. Finish the `print → logging` migration.

**Depends on:** Phase 1 (need clean imports before restructuring).

**Deliverables**

- `config.py` module owning all paths: `sf/<layer>/`, `*_fx.json`, `inst_probabilities.json`, `levels.json`, `song_structures.json`, `chord_patterns.txt`, `beat_roll_patterns_*.txt`. No path literals remain in `music_gen.py`.
- `timesig.py` module with a `TimeSignatureRegistry` of dataclass specs. `verify_pattern_for_time_signature`, `verify_beat_pattern`, `calculate_measures_for_time_signature`, `generate_random_time_signature`, and `DurationValidator` all delegate to the registry.
- All 32+ `print()` calls in `music_gen.py` replaced with `logging` using already-installed `python-json-logger`.
- Startup soundfont-pool detection: log count per `sf/<layer>/`, warn if < 3.
- Unit tests for `timesig` registry covering every currently supported signature (2/4, 3/4, 4/4, 5/4, 6/8, 7/8, 12/8).

**Requirements covered:** R-S5, R-S6, R-S7 (complete), R-S9.

**Exit criteria:** Adding a new time signature would require editing exactly one file. No grep hit for `sf/beat` / `*_fx.json` literals outside `config.py`.

**Plans:** 3 plans

Plans:
- [x] 02-01-PLAN.md — Config module with three-layer override precedence (CLI > env > defaults), path migration from music_gen.py, soundfont pool detection, and Wave 0 test skeletons. (R-S5, R-S9) — completed 2026-04-18
- [x] 02-02-PLAN.md — TimeSignatureRegistry in timesig.py, thin wrappers in music_gen.py, DurationValidator delegation, and full registry test suite. (R-S6) — completed 2026-04-18
- [x] 02-03-PLAN.md — Replace all 32 print() calls with semantically leveled logging, add basicConfig in __main__ guard, and AST-scan regression tests. (R-S7) — completed 2026-04-18

---

## Phase 3: Package skeleton + sampler + generators extraction

**Goal:** Stand up `src/musicgen/` as an installable Python package. Move pure-function logic (sampler, generators) out of the god file behind injected-RNG interfaces. This phase unblocks all productize work.

**Depends on:** Phase 2 (config + timesig registry must exist before generators can import them cleanly).

**Can run in parallel with:** Phase 4 (renderer/mixer/annotator extraction) — see plan-level dependencies below.

**Deliverables**

- `pyproject.toml` with `hatchling` backend, `requires-python >= 3.9`, runtime deps including new `typer>=0.12`, `[project.optional-dependencies].dev = [pytest, pytest-cov, pytest-xdist]`, entry point `musicgen = "musicgen.cli:app"`.
- `src/musicgen/` package layout: `__init__.py`, `__main__.py`, placeholder `cli.py`.
- `src/musicgen/sampler.py` with `SongParams` dataclass and all `generate_random_*` + `generate_song_arrangement` functions. Each takes `rng: random.Random`.
- `src/musicgen/generators/{chord,melody,bassline,beat}.py` extracted, each taking `SongParams` and injected `rng`.
- `music_gen.py` imports the new modules (thin backwards-compat shim).
- Unit tests for sampler and each generator with seeded RNG.

**Requirements covered:** R-X1, R-X2, R-X3.

**Exit criteria:** `pip install -e .` succeeds. `from musicgen.sampler import SongParams` works. Generators run without touching module-level `random`. Old `music_gen.py` still executable for smoke testing.

**Plans:** 5 plans

Plans:
- [x] 03-01-PLAN.md — Package scaffold + `pyproject.toml` (hatchling, typer>=0.12, dev extras), stub CLI, delete `requirements.txt`/`dev-requirements.txt`. Uses `requires-python=">=3.10"` per RESEARCH Risk #1 (CONTEXT D-13's `>=3.9` is infeasible with typer+hatchling pins). (R-X1, R-Q4) — **COMPLETE 2026-04-18** (commits 5d0a64f, 6409a8e, eb2a81a)
- [x] 03-02-PLAN.md — `git mv enhanced_duration_validator.py` → `src/musicgen/duration_validator.py` (D-10), rewrite the two consuming import sites (`music_gen.py:15`, `tests/test_duration_validator.py:10`). Prerequisite for Wave 2 generator extraction. (R-X1) — **COMPLETE 2026-04-18** (commit 447626f)
- [x] 03-03-PLAN.md — Extract sampler (SongParams frozen dataclass + `SongParams.sample` classmethod + 7 free functions + `validate_measures_dict`) into `src/musicgen/sampler.py` with injected `rng: random.Random`. Zero bare `random.*`. Seeded determinism tests + AST static guard. `music_gen.py` shim re-exports sampler symbols; call sites pass `_rng`. (R-X2) — **COMPLETE 2026-04-18** (commits eaa7ee5, cdaa3b8, 53d929d)
- [x] 03-04-PLAN.md — Extract 4 generators into `src/musicgen/generators/{chord,melody,bassline,beat}.py` with injected `rng`. 11 bare-random rewrites. music21 audit comment added to chord/melody/bassline. Per-generator seeded MIDI-byte-equal tests + AST no-bare-random guard. `music_gen.py` shim re-exports generator symbols; call sites pass `_rng`. (R-X3) — **COMPLETE 2026-04-18** (commits 9256012, 5fa3f0c, 3224e4e)
- [x] 03-05-PLAN.md — Add `tests/test_music21_isolation.py` regression guard (D-24, 3 tests — roman / scale / pitch). Delete `tests/conftest.py` (D-16 — `pyproject.toml pythonpath=["."]` from Plan 03-01 replaces it). Phase-gate verification: 371 tests green, full-package AST scan clean, `python music_gen.py` best-effort smoke reaches mix_and_save (env-only failure at soundfont selection). (R-X1, R-X2, R-X3) — **COMPLETE 2026-04-19** (commits edb0fe3, 88f2657)

---

## Phase 4: Renderer + mixer + annotator + beats extraction

**Goal:** Extract the audio-side of the pipeline. Decompose `mix_and_save` into `renderer`, `mixer`, and `annotator`. Replace `beat_anotator.py` with swing-aware beat derivation.

**Depends on:** Phase 2 (config + timesig). Can run in parallel with Phase 3 if the team capacity allows, but Phase 5 depends on both.

**Deliverables**

- `src/musicgen/renderer.py` wrapping FluidSynth, rendering the four stems per part via `ThreadPoolExecutor(max_workers=4)`. FluidSynth version captured at module load.
- `src/musicgen/mixer.py` — pedalboard FX, pydub overlay, part concatenation, silent-stem fallback for absent layers. Uses the fixed gain/pan APIs from Phase 1.
- `src/musicgen/annotator.py` — pure function producing the `sample.json` dict from stage outputs. Pre-reserves the schema defined in R-P4 (subset filled in now, missing fields = TBD flags that Phase 5 fills).
- `src/musicgen/beats.py` — derives beat and downbeat timestamps from MIDI ticks, swing-aware. Replaces `beat_anotator.py`.
- `music_gen.py` reduced to orchestration using the new modules.
- Unit tests for annotator (fixture-driven) and beats (swing 0.5, 0.66, 0.75 cases).
- One `@pytest.mark.slow` integration test that generates a single song end-to-end.

**Requirements covered:** R-X4, R-X5, R-X6, R-X7, R-X8.

**Exit criteria:** `mix_and_save` is < 50 lines of pure orchestration. Integration test produces mix audio whose stems sum to the mix within a loose tolerance (assertion is tightened in Phase 5). Beat timestamps match MIDI ticks rather than theoretical grid.

**Plans:** 7/7 plans executed — PHASE COMPLETE 2026-04-19

Plans:
- [x] 04-00-wave-0-infrastructure-PLAN.md — Wave 0: add mido>=1.3.3 dep + pytest markers + scaffold 6 new test files. (R-X4, R-X5, R-X6, R-X7, R-X8)
- [x] 04-01-beats-module-PLAN.md — Create src/musicgen/beats.py with beat_duration + extract_beat_times (mido.tick2second) + extract_downbeat_times (time-grid, RESEARCH correction #1); re-export beat_duration from generators/beat.py (D-21); tests/test_beats.py with 3 swing cases. (R-X7)
- [x] 04-02-renderer-module-PLAN.md — Create src/musicgen/renderer.py with FLUIDSYNTH_VERSION + RenderResult + pick_soundfonts(cfg, rng) + render_stems via ThreadPoolExecutor(max_workers=4); tests/test_renderer.py with mocked FluidSynth. (R-X4)
- [x] 04-03-mixer-module-PLAN.md — Create src/musicgen/mixer.py absorbing create_effect + generate_pedalboard (→ build_fx_boards) + apply_fx_to_layer + pedalboard_info_json + _lin_to_db (unnested); add compute_layer_mask + _make_silent_stem (stereo 44.1kHz, RESEARCH correction #2) + MixResult + mix_part + concat_parts; tests/test_mixer.py with seeded-RNG + D-11 FX-on-all-layers + R-S4 preservation. (R-X5)
- [x] 04-04-annotator-module-PLAN.md — Create src/musicgen/annotator.py with annotate(...) pure function producing R-P4 schema (Phase-4 fills + Phase-5 None semantics per D-15/D-16); tests/test_annotator.py fixture-driven golden-dict + purity contract. (R-X6)
- [x] 04-05-orchestrator-collapse-and-ast-guard-PLAN.md — Collapse music_gen.py from 523 to 199 lines: delete 9 audio functions (mix_and_save + 8 helpers); rewrite create_song to chain renderer → mixer → beats → annotator; delete beat_anotator.py outright (D-03); add tests/test_no_bare_random_in_package.py AST guard (D-17/D-31). 504 tests pass. (R-X4, R-X5, R-X6, R-X7) — **COMPLETE 2026-04-19** (commits 062c6b3, fb71998)
- [x] 04-06-e2e-integration-test-PLAN.md — Create tests/test_integration_full_generation.py with @pytest.mark.slow E2E test guarded on fluidsynth binary + sf2 pool; exercises full pipeline; asserts 4 stems + 1 mix + 4 MIDIs + R-P4 dict + MIDI reproducibility. (R-X8) — **COMPLETE 2026-04-19** (commit 69cd990)

---

## Phase 5: Productize I — writer, manifest, seed discipline, determinism

**Goal:** Per-sample output directory lands; seeds propagate end-to-end; determinism regression test passes. This is the heart of the productize milestone.

**Depends on:** Phase 3 + Phase 4.

**Deliverables**

- `src/musicgen/writer.py` implementing the index-based output layout from R-P1: `<dataset-root>/<index:06d>/` with `sample.json`, `mix.wav`, `stems/*.wav`, `midi/*.mid`. `sample.json` is always the last file written.
- `src/musicgen/manifest.py` implementing `manifest.jsonl` append under `multiprocessing.Manager().Lock()`, with resume detection (presence of `sample.json`) and status tracking (`ok` / `failed`).
- Full `sample.json` schema from R-P4: seed, `fluidsynth_version`, `musicgen_version`, all song params, arrangement with per-part timestamps, chord progressions, active layers, soundfonts, FX params, beat + downbeat times, musicality score, relative paths.
- End-to-end seed discipline: `derive_sample_seed`, `make_rngs`, no bare `random.*` in the pipeline. Workers seed on entry.
- Stem-sum-to-mix assertion in `writer.py` (or immediately before). Failure → sample marked `status: failed` + traceback logged + batch continues.
- Deterministic train/valid/test split via stable seed hash (default 80/10/10, configurable).
- Regression test: fixed seed → mix audio SHA-256 matches a checked-in golden. (See R-P8 — accept binary-dependent.)
- Fix UUID truncation by moving to index-based naming.

**Requirements covered:** R-P1, R-P2, R-P3, R-P4, R-P5, R-P6, R-P7, R-P8, R-Q3.

**Exit criteria:** Running the library API with the same seed twice produces bit-identical `sample.json` and MIDI, and bit-identical mix WAV under the pinned FluidSynth binary. Sum-of-stems assertion passes on > 95% of random seeds from a small smoke batch (remaining failures must be understood — likely gain rounding).

**Plans:** 6/6 plans executed — PHASE COMPLETE 2026-04-20

Plans:
- [x] 05-01-PLAN.md — Wave 0: test infrastructure (6 Wave 0 test stubs + tests/conftest.py --regen-goldens flag + tests/fixtures/determinism/README.md dir marker + AST guard widens allow-list for random.getstate/setstate + xfail meta-test adds seeds/writer/manifest/api/musicality to expected_present). (R-P1..R-P8, R-Q3 forward-guard) — **COMPLETE 2026-04-19** (commits 67a04e4, bc925b5)
- [x] 05-02-PLAN.md — Wave 1: src/musicgen/seeds.py (derive_sample_seed/make_rngs/save_random_state/assign_split per D-17/D-18/D-20/D-26 verbatim) + fully-populated test_seeds.py + test_split.py. (R-P6, R-P7) — **COMPLETE 2026-04-19** (commits c4b3d91, 62452fe, bc2e4fc)
- [x] 05-03-PLAN.md — Wave 1: git mv musicality_score.py → src/musicgen/musicality.py (D-03; closes Phase 3 D-11 / Phase 4 D-04 deferral); rewrite single import site in music_gen.py. (R-P4) — **COMPLETE 2026-04-19** (commit 48f71ac)
- [x] 05-04-PLAN.md — Wave 2: src/musicgen/manifest.py (ManifestWriter, threading.Lock default, sentinel-only is_sample_complete) + src/musicgen/writer.py (atomic per-sample layout, absolute-tick MIDI concat, int32 sum-of-stems assertion, deep-copy path rewrite) + config.py +7 fields + __post_init__ validation; populate test_manifest.py + test_writer.py + extend test_config.py. (R-P1, R-P2, R-P3, R-P4, R-P5, R-P6) — **COMPLETE 2026-04-20** (commits dbc2f01, 1ff9c73, bad47df, 9689464, ced0c6e, 8343ac9)
- [x] 05-05-PLAN.md — Wave 3: src/musicgen/api.py (generate + SampleResult + _generate_all_midi + resume short-circuit) + rewrite __init__.py (public exports) + collapse music_gen.py from 199 → 59 lines (delete create_song + generate_song_parts + generate_song per D-34) + migrate tests/test_integration_full_generation.py atomically + populate test_api.py + remove xfail from AST meta-test. (R-P1..R-P7, R-Q3) — **COMPLETE 2026-04-20** (commits de9b1ee, 0b5c6b5, a3bb349, 55804fc)
- [x] 05-06-PLAN.md — Wave 4: tests/test_determinism_golden.py (TestDeterminismGoldens parametrized over 6 artifacts + --regen-goldens flag + fluidsynth_version.txt xfail gate + TestSameProcessStability fast D-30 in-process cross-check) + refined fixtures README. (R-P8, R-Q3) — **COMPLETE 2026-04-20** (commit 04570b5)

---

## Phase 6: Productize II — FluidSynth calibration, batch generation, CLI, resumability

**Goal:** The library becomes usable as a library and as a CLI for real 1k–10k dataset runs. FluidSynth pre-roll is measured and corrected. Parallel batch generation works and can resume.

**Depends on:** Phase 5.

**Deliverables**

- `src/musicgen/calibrate.py` (or equivalent) — one-shot FluidSynth pre-roll measurement that caches result at `.musicgen/fluidsynth_preroll.json`. Pre-roll offset applied to beat times and note-onset annotations. Recorded in every `sample.json`.
- `src/musicgen/batch.py` — `generate_batch(config)` using `ProcessPoolExecutor(max_workers=config.workers or os.cpu_count())`. Per-worker seeding on entry. Progress logging (structured JSON lines).
- Resume logic: re-running against an existing dataset directory skips samples with a complete `sample.json`; retries failed ones; idempotent.
- Failure isolation: individual sample failures logged with traceback + manifest `status: failed`, batch continues. Summary report at end.
- `src/musicgen/cli.py` typer app:
  - `musicgen generate --count N --out DIR --seed S [--workers W] [--output-mode MODE] [-v/-q]`
  - `musicgen clean --failed`
  - `musicgen calibrate` (explicit pre-roll recompute)
- Output mode flag (`full`/`mix-only`/`stems-only`/`midi-only`) threaded through writer + annotator.
- Integration test: generate a batch of 4 samples with 2 workers, verify manifest, verify resume on second run.

**Requirements covered:** R-P9, R-P10, R-P11, R-P12, R-P13, R-P14, R-P15, R-P16.

**Exit criteria:** `musicgen generate --count 4 --out /tmp/ds --seed 42 --workers 2` produces a valid 4-sample dataset. Re-running the same command skips all 4. Killing the process mid-run and re-running resumes correctly.

**Plans:** 6/6 plans executed — PHASE COMPLETE 2026-04-28

- [x] 06-01-PLAN.md — Wave 0: test scaffolding stubs (5 new test files + extend test_config.py + update AST guard expected_present for calibrate.py + batch.py). (R-P9..R-P16, R-Q2 forward-guard) — **COMPLETE 2026-04-28** (commit 020fb76)
- [x] 06-02-PLAN.md — Wave 1: Config extension (output_mode + count fields) + OutputMode routing in writer.write_sample + pre_roll_offset_s parameter + api.py calibrate hook (fallback to 0.0 until Wave 2). (R-P14) — **COMPLETE 2026-04-28** (commits d34a368, 83a69b6)
- [x] 06-03-PLAN.md — Wave 2: src/musicgen/calibrate.py (measure_preroll + load_preroll + save_preroll + cache at .musicgen/fluidsynth_preroll.json) + test_calibrate.py. (R-P9) — **COMPLETE 2026-04-28** (commits 3ea34e8, f3ebc11)
- [x] 06-04-PLAN.md — Wave 3: src/musicgen/batch.py (generate_batch + BatchResult + ProcessPoolExecutor spawn + resume logic + failure isolation + JSON progress logs) + test_batch.py. (R-P10, R-P11, R-P15, R-P16) — **COMPLETE 2026-04-28** (commits 4bca822, bb47e62)
- [x] 06-05-PLAN.md — Wave 4: Full CLI rewrite (replace Phase 3 stub cli.py — typer commands generate + clean + calibrate) + test_cli.py. (R-P13) — **COMPLETE 2026-04-28** (commits a6d7014, 004b394)
- [x] 06-06-PLAN.md — Wave 5: Integration test (4-sample batch, 2 workers, verify manifest + resume + output_mode) + generate_batch/BatchResult export from musicgen.__init__. (R-P10, R-P11, R-P12, R-Q2) — **COMPLETE 2026-04-28** (commit da12dcf)

---

## Phase 7: Ship v0.1 — docs, polish, regression suite

**Goal:** Milestone-ready release. Documentation, test coverage, and a small canonical dataset for acceptance.

**Depends on:** Phase 6.

**Deliverables**

- README refresh per R-Q1: install, library example, CLI example, directory layout diagram, determinism contract, soundfont licensing caveat. ✓
- Test coverage target: pure-function coverage ≥ 80%. `pytest-cov` in CI. ✓ (87% actual)
- Determinism regression test from Phase 5 in CI. ✓ (TestSameProcessStability runs in every push)
- GitHub Actions CI on Python 3.10 + 3.12. ✓
- Tag `v0.1.0`. ✓

**Requirements covered:** R-Q1, R-Q2, R-Q3, R-Q4.

**Plans:** PHASE COMPLETE 2026-04-28 (commit f6cad19 — README + CI; tag v0.1.0 pushed)

**Exit criteria:** `v0.1.0` tagged. README documents the full workflow. A fresh 32-sample dataset generates cleanly with no manual intervention.

---

## v0.2 Integrations — Sibling Ecosystem

**Goal:** Connect musicgen to three sibling repos (soundfont_manager, midi_file_manager, audio_sample_manager) without adding hard dependencies.

**Depends on:** v0.1.0 shipped.

**Deliverables**

- Integration 1 — soundfont_manager: `pick_soundfonts()` becomes metadata-aware when `cfg.soundfont_manager_db` is set. Falls back to directory scan otherwise. Seed-deterministic via sorted candidate list + rng.choice. Layer-to-tag mapping (beat→drums/percussion, melody→melody/lead/piano/strings, harmony→harmony/chords/pads, bassline→bass).
- Integration 2 — midi_file_manager: `musicgen index-midi` CLI command. `index_midi_dataset()` library function. Walks dataset MIDI files, cross-references sample.json for ground-truth BPM/key/time_sig, stores enriched index in MidiManager JSON db. Optional CSV export.
- Integration 3 — audio_sample_manager: `musicgen index-audio` CLI command. `index_audio_dataset()` library function. Walks dataset WAV stems, applies ground-truth override (bpm/key/time_sig/scale), marks is_loop=False, stores in SampleManager db. Enables cross-library queries.

**Status:** ✓ COMPLETE (branch feat/soundfont-manager, 793 tests passing)

**Requirements covered:** R-E1 (extend soundfont pool), R-P10 (downstream consumers)

---

## Phase dependency graph

```
Phase 1 (Stabilize I)
    |
    v
Phase 2 (Stabilize II: config + timesig + logging)
    |
    +----------------------+
    v                      v
Phase 3 (Sampler +      Phase 4 (Renderer +
generators extraction)  mixer + annotator + beats)
    |                      |
    +----------+-----------+
               v
Phase 5 (Writer + manifest + seeds + determinism)
               |
               v
Phase 6 (FluidSynth calibration + batch + CLI + resume)
               |
               v
Phase 7 (Ship v0.1)
```

Phase 3 and Phase 4 are parallelizable once Phase 2 is done.

## Requirement coverage map

| Phase | Requirements |
|---|---|
| 1 | R-S1, R-S2, R-S3, R-S4, R-S7 (partial), R-S8, R-Q2 (initial) |
| 2 | R-S5, R-S6, R-S7 (complete), R-S9 |
| 3 | R-X1, R-X2, R-X3 |
| 4 | R-X4, R-X5, R-X6, R-X7, R-X8 |
| 5 | R-P1, R-P2, R-P3, R-P4, R-P5, R-P6, R-P7, R-P8, R-Q3 |
| 6 | R-P9, R-P10, R-P11, R-P12, R-P13, R-P14, R-P15, R-P16 |
| 7 | R-Q1, R-Q2 (final), R-Q3 (final), R-Q4 |

Every requirement in `REQUIREMENTS.md` maps to at least one phase.

## v0.2 — Extend: Genre System

**Branch:** `feat/genre-system`
**Goal:** Introduce a genre-aware generation system: a `GenreSpec` config layer that constrains sampler draws, chord vocab, drum patterns, FX profiles, and soundfont selection. Genres are composable from day one.

**Depends on:** v0.1.0 + v0.2 integrations shipped.

**Documentation rule:** Every phase that lands functional code must update `README.md` (CLI section, Config table, or Architecture) + relevant `.planning/` docs in the same PR. No functional merge without doc update.

---

### v0.2 Phase 1 — GenreSpec + composition engine

**Goal:** Define `GenreSpec` dataclass and the genre-merge algorithm. Wire `Config.genre` field. No sampler changes yet — infrastructure only.

**Deliverables:**
- `src/musicgen/genre.py` — `GenreSpec` dataclass with all dimension fields (tempo, time_sig weights, swing, scales, chord types, inversions, layer probs, soundfont tags, drum pool names, FX profile, arrangement shape)
- `genres/<name>/spec.json` schema (validated against `GenreSpec` at load time)
- `Config.genre: Optional[List[str]]` — list of genre names to compose
- `genre.py::load_genre(name) -> GenreSpec` — reads from `genres/<name>/spec.json`
- `genre.py::merge_genres(specs, weights=None) -> GenreSpec` — composition engine:
  - Hard numeric ranges (tempo, swing): intersection of all active genres
  - Soft weights (time sig, scale, chord types, inversions): normalized weighted average
  - Sets (soundfont tags, drum pool names): union
  - Priority: first genre listed wins ties on hard constraints
- `Config.load()` precedence updated: `CLI > env > genre-merged > defaults`
- Tests: merge algebra (intersection, union, weighted avg), precedence chain, unknown genre → `FileNotFoundError`

---

### v0.2 Phase 2 — Chord vocabulary expansion

**Goal:** Extend `generators/chord.py` to support richer chord types and inversions, genre-constrained.

**Deliverables:**
- New chord types: `maj`, `min`, `dom7`, `maj7`, `m7`, `m7b5`, `dim7`, `sus2`, `sus4`, `9`, `maj9`, `m9`, `add9`
- Inversion support: root / 1st / 2nd / 3rd — each generates correct MIDI voicing (note stack with octave shifts)
- Genre controls chord pool (hard: which types are allowed) and weights (soft: probability per type/inversion)
- `chord_patterns.txt` gains genre tag headers per pattern block: `# [genre: jazz, blues]`
- Chord pattern loader filters by active genre union; falls back to untagged patterns if no genre set
- Tests: voicing correctness per chord type + inversion, genre hard-filter removes disallowed types, soft weights shift distribution

---

### v0.2 Phase 3 — Drum pattern reorganization

**Goal:** Reorganize `beat_roll_patterns_*.txt` files from flat repo-root files into per-genre directories. Pattern loader merges pools from all active genres.

**New layout:**
```
genres/
  default/
    patterns_24.txt  patterns_34.txt  patterns_44.txt
    patterns_54.txt  patterns_68.txt  patterns_78.txt  patterns_128.txt
  jazz/
    patterns_34.txt  patterns_44.txt  patterns_54.txt  patterns_68.txt
  hip-hop/
    patterns_44.txt
  blues/  latin/  pop/  electronic/  reggae/  classical/
    patterns_*.txt (applicable sigs only)
```
- Old `beat_roll_patterns_*.txt` at repo root → moved to `genres/default/`
- `Config.beat_roll_pattern_dirs` list replaces single-file field; defaults to `["genres/default"]`
- Pattern loader: union of all active genre pattern files for the active time signature; deduplicates patterns
- Backward compat: no-genre config uses `genres/default/` identically to old behavior
- Tests: loader union, dedup, missing-genre file → skip (not error), fallback to default

---

### v0.2 Phase 4 — Sampler + FX genre integration

**Goal:** Sampler draws respect `GenreSpec` constraints. FX boards use genre FX profile as shifted default ranges.

**Deliverables:**
- `sampler.py` reads active `GenreSpec` for:
  - Tempo: hard bounds (`GenreSpec.tempo_min`, `tempo_max`) — `generate_random_tempo` clamps draw
  - Time signature: soft weights (`GenreSpec.time_sig_weights`) — `generate_random_time_signature` uses weighted choice
  - Swing: hard bounds (`GenreSpec.swing_min`, `swing_max`) — draw clamped
  - Scale/mode: soft weights (`GenreSpec.scale_weights`) — `generate_random_key` uses weighted choice
  - Layer active probabilities: `GenreSpec.layer_probs` overrides `inst_probabilities.json` defaults when genre set
- `mixer.py` `build_fx_boards`: genre FX profile shifts center of FX parameter ranges (soft — full range still accessible, but midpoint moves per genre)
- `api.py` resolves + merges genre specs once at start, passes merged `GenreSpec` down through all stages
- Tests: seeded draws within genre bounds, weighted distribution shift vs baseline, no-genre → identical behavior to pre-genre

---

### v0.2 Phase 5 — Soundfont genre integration

**Goal:** Genre `soundfont_tags` replaces static `_LAYER_TAGS` in `renderer.py` when genre is active.

**Deliverables:**
- `renderer._pick_via_soundfont_manager` accepts optional `layer_tags: Dict[str, List[str]]` override
- When genre active: `GenreSpec.soundfont_tags[layer]` → passed as `layer_tags`
- When no genre: static `_LAYER_TAGS` (existing behavior, backward compat)
- When SM not installed: fallback chain unchanged
- `sample.json` `soundfonts` field records which SF2 was used per layer (already present; no schema change)
- Tests: genre tags override static tags, fallback chain still works, no-SM still falls back to directory scan

---

### v0.2 Phase 6 — Built-in genre presets

**Goal:** Ship 8 genre presets, each with `spec.json` + genre-tagged drum pattern files.

**Genres:** `jazz`, `hip-hop`, `blues`, `pop`, `electronic`, `latin`, `reggae`, `classical`

**Per genre deliverables:**
- `genres/<name>/spec.json` — full `GenreSpec` with all dimensions filled
- `genres/<name>/patterns_<sig>.txt` — drum patterns for applicable time signatures
- Smoke test: generate 1 sample per genre, assert `sample.json` fields within genre bounds (tempo, swing, time_sig in allowed set)
- README: `## Built-in genres` table listing each genre + key characteristics

---

### v0.2 Phase 7 — Docs + CLI genre support

**Goal:** Expose genre system fully in CLI + document everything.

**Deliverables:**
- `musicgen generate --genre jazz` single genre
- `musicgen generate --genre jazz latin` composition (space-separated; typer `List[str]`)
- `musicgen generate --list-genres` prints available genres + one-line description each
- `genres/README.md` — spec format documentation, how to write custom genres, composition semantics
- README.md: Configuration table gains `genre` field; new "Genre system" section with usage examples
- `.planning/codebase/INTEGRATIONS.md` / `STRUCTURE.md` updated

---

### v0.2 Phase 8 — Jupyter notebook: feature showcase

**Goal:** End-to-end runnable notebook demonstrating all major features. Serves as living docs + acceptance test.

**File:** `notebooks/musicgen_demo.ipynb`

**Sections:**
1. Setup — install check, `Config` basics, FluidSynth presence check
2. Single sample generation — `generate()`, inspect `sample.json`, display fields
3. Batch generation — `generate_batch()`, manifest inspection, stats
4. Output modes — `full` / `stems-only` / `midi-only` side-by-side layout
5. Genre generation — single genre (`jazz`), compare `sample.json` fields vs no-genre baseline
6. Genre composition — `genre=["jazz", "latin"]`, show merged `GenreSpec` fields
7. Audio playback — `IPython.display.Audio` for mix + each stem
8. MIDI visualization — `pretty_midi` piano roll plot per layer
9. Musicality score — bar chart of component scores across small batch
10. MIDI indexing — `index_midi_dataset()`, query result from `MidiManager`
11. Audio indexing — `index_audio_dataset()`, cross-library query example
12. Determinism check — generate same `(seed, index)` twice, assert `sample.json` SHA-256 match

Cells requiring FluidSynth tagged `# requires: fluidsynth` with graceful skip if binary absent. Notebook runnable top-to-bottom on machine with FluidSynth + ≥ 1 `.sf2` per layer.

---

## v0.3 — Research: Higher-Order Markov

**Branch:** `feat/higher-order-markov`
**Goal:** Replace first-order / pattern-lookup chord and melody generation with configurable higher-order Markov chains. Add quality-gated regeneration loops.

**Depends on:** v0.2 genre system shipped (Markov matrices are per-genre).

**Documentation rule:** Same as v0.2 — every functional phase updates README.md + relevant planning docs in same PR.

---

### v0.3 Phase 1 — Chord progression: 2nd-order Markov

**Goal:** Replace `chord_patterns.txt` pattern-lookup with Markov transition matrix: `P(chord_N | chord_{N-1}, chord_{N-2})`.

**Deliverables:**
- `GenreSpec` gains `chord_transition_matrix: Optional[Dict]` — 2nd-order transition table (chord pair → next chord distribution)
- `genres/<name>/chord_transitions.json` — per-genre transition matrix (defined as Roman numeral pairs to avoid key lock-in)
- `generators/chord.py` `generate_chord_progression` checks for matrix; falls back to pattern-file if absent (backward compat)
- `Config.markov_order: int = 1` — 1 = current behavior, 2 = 2nd-order; higher orders via same code path
- Boundary handling: start of progression uses 1st-order until history is long enough
- Tests: matrix produces valid chord sequences, same seed = same sequence, fallback to patterns when no matrix, order=1 matches pre-refactor output

---

### v0.3 Phase 2 — Melody: higher-order Markov

**Goal:** Upgrade `generators/melody.py` to configurable-order Markov over scale-relative intervals.

**Deliverables:**
- Transition matrices defined in scale-relative interval space (not absolute pitches) — stays key-agnostic
- `GenreSpec` gains `melody_transition_matrix: Optional[Dict]` + `melody_markov_order: int = 2`
- `genres/<name>/melody_transitions.json` — per-genre matrices
- Fallback: no matrix → existing behavior
- Tests: relative-interval encoding correct, seed-deterministic, order-1 matches prior output

---

### v0.3 Phase 3 — Regeneration loops (quality gate)

**Goal:** Low-quality samples get re-rolled automatically before being written.

**Deliverables:**
- `Config.min_musicality_score: Optional[float] = None` — opt-in quality gate (disabled by default)
- `Config.max_attempts: int = 3` — max re-roll attempts
- Seed derivation for attempt k: `sha256(f"{global_seed}:{sample_index}:attempt={k}")` — deterministic, reproducible
- `api.py` loop: generate → score → if score < threshold and attempts remain → re-roll
- `sample.json` gains `attempt: int` field (1-indexed; 1 = no re-roll needed)
- Manifest gains `attempt` field + `status: "low_quality"` for samples that exhausted attempts but still didn't meet threshold
- Tests: gate disabled → no re-roll, gate enabled → re-roll until threshold or max_attempts, deterministic per attempt, manifest reflects final status

---

## Out of this milestone (future work)

- **v0.4 — ML-assisted generators:** ML-assisted chord/melody generation (trained on real MIDI corpora), model-guided regeneration.
- **v0.? — Public release:** soundfont license audit, CC0/MIT soundfont replacement, HF Datasets / WebDataset exporters, sharded directory layout for 100k+.

## Risks and mitigations (carried from research)

| Risk | Mitigation | Phase |
|---|---|---|
| FluidSynth cross-version non-determinism (P-1) | Pin binary; SHA regression test; document binary-dependent contract | 5, 7 |
| Sum-of-stems != mix after gain/pan fix (P-2) | Assertion + failure mode in writer | 5 |
| Beat annotation drift from swing + pre-roll (P-3, P-8) | MIDI-derived beats; calibrate pre-roll | 4, 6 |
| Multiprocessing RNG leakage (P-4) | Per-worker seeding on entry | 5, 6 |
| Narrow soundfont pool bias (P-5) | Detect in stabilize; broaden in v0.2 | 2 (detect) |
| `music21` global random state | Audit during Phase 3 extraction | 3 |
| Config churn breaks tests | Tests land before extraction in each phase | 1, 2, 3, 4 |
