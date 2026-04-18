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
- [ ] 03-04-PLAN.md — Extract 4 generators into `src/musicgen/generators/{chord,melody,bassline,beat}.py` with injected `rng`. 11 bare-random rewrites. music21 audit comment added to chord/melody/bassline. Per-generator seeded MIDI-byte-equal tests + AST no-bare-random guard. `music_gen.py` shim re-exports generator symbols; call sites pass `_rng`. (R-X3)
- [ ] 03-05-PLAN.md — Add `tests/test_music21_isolation.py` regression guard (D-24, 3 tests — roman / scale / pitch). Delete `tests/conftest.py` (D-16 — `pyproject.toml pythonpath=["."]` from Plan 03-01 replaces it). Phase-gate verification: ≥349 tests green, full-package AST scan clean, `python music_gen.py` best-effort smoke. (R-X1, R-X2, R-X3)

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

---

## Phase 7: Ship v0.1 — docs, polish, regression suite

**Goal:** Milestone-ready release. Documentation, test coverage, and a small canonical dataset for acceptance.

**Depends on:** Phase 6.

**Deliverables**

- README refresh per R-Q1: install, library example, CLI example, directory layout diagram, determinism contract, soundfont licensing caveat.
- Test coverage target: pure-function coverage ≥ 80%. `pytest-cov` in CI.
- Determinism regression test from Phase 5 in CI.
- Acceptance: run `musicgen generate --count 32 --seed 1 --workers 4` end-to-end. Manifest has 32 entries, all `status: ok`, sum-of-stems assertion holds. Manual sanity listen of a few samples.
- Tag `v0.1.0`.

**Requirements covered:** R-Q1, R-Q2, R-Q3, R-Q4.

**Exit criteria:** `v0.1.0` tagged. README documents the full workflow. A fresh 32-sample dataset generates cleanly with no manual intervention.

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

## Out of this milestone (future work)

- **v0.2 — Extend:** broader musical vocabulary, richer chord progressions, more drum/genre patterns, additional time signatures, broader soundfont pool.
- **v0.3 — Research:** smarter Markov / ML-assisted generators, musicality-aware regeneration experiments (opt-in, never default).
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
