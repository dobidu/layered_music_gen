# REQUIREMENTS

**Project:** musicgen — synthetic music dataset generator for ML training
**Milestone:** v0.1 — Stabilize + Productize (the first shippable dataset generator)

This milestone converts the existing `music_gen.py` monolith into a Python library + CLI that produces 1k–10k reproducible, fully-annotated music samples suitable for training music tagging, source separation, beat/tempo/downbeat detection, and audio→MIDI transcription models.

Scope decisions follow PROJECT.md priority order: **Stabilize → Productize → Extend → Research**. Extend and Research are explicitly deferred to a later milestone.

Inputs: `.planning/PROJECT.md`, `.planning/codebase/*`, `.planning/research/*`.

---

## R-S — Stabilize (blockers for importability, testability, correctness)

**R-S1 — Importability.** `music_gen.py` must be importable without triggering generation. Wrap the bottom-of-file execution (`music_gen.py:1158-1161`) in `if __name__ == "__main__":`.

**R-S2 — Explicit imports.** ✓ Complete (Plan 01-03). Replace `from music21 import *` (`music_gen.py:2`) with explicit symbol imports for only the names actually used.

**R-S3 — Bug fix: arrangement re-roll.** `mix_and_save` must not re-call `generate_song_arrangement()` (`music_gen.py:760`). Arrangement must be produced once and passed through the pipeline so the MIDI files and rendered audio describe the same structure. (PITFALLS P-A, confirmed by grep.)

**R-S4 — Bug fix: pydub gain/pan.** `music_gen.py:845-852` currently assigns to the read-only `.volume` property and discards the return of `.pan()` — `levels.json` has no effect today. Replace with `AudioSegment.apply_gain(...)` and `segment = segment.pan(...)`. (PITFALLS P-B, confirmed by grep.)

**R-S5 — Config centralization.** ✓ Complete (Plan 02-01). Extract a single config module (`config.py` or equivalent) that owns all paths to: soundfont directories (`sf/<layer>/`), FX JSON files (`*_fx.json`), `inst_probabilities.json`, `levels.json`, `song_structures.json`, `chord_patterns.txt`, `beat_roll_patterns_*.txt`. No path literals remain outside config.

**R-S6 — Time-signature registry.** Consolidate time-signature logic into one registry module. Today it is scattered across `verify_pattern_for_time_signature` (`music_gen.py:22`), `verify_beat_pattern` (`:42`), `calculate_measures_for_time_signature` (`:54`), `generate_random_time_signature`, and `enhanced_duration_validator.DurationValidator`. Adding a signature must touch one location. **Must land before generator extraction.**

**R-S7 — Structured logging.** ✓ Complete (Plan 02-03). All 32 `print()` calls in `music_gen.py` replaced with semantically leveled `logging` calls (16 DEBUG, 14 INFO, 2 WARNING) per D-07. Module-level `logger = logging.getLogger(__name__)` added. `logging.basicConfig` placed inside `__main__` guard only. AST-scan and import-side-effect tests confirm the migration is complete.

**R-S8 — Dead code removal.** ✓ Complete (Plan 01-03). Remove unused imports (`glob`, `Pool`, `cpu_count`, `time`) and unused variables (`beat_annotations`, `ha`, `ba`, `me`, `be`, `now`) from `music_gen.py`. Also remove the `uuid` PyPI entry from `requirements.txt` (stdlib in Python 3).

**R-S9 — Soundfont pool detection.** ✓ Complete (Plan 02-01). At startup / config load, log the number of soundfonts in each `sf/<layer>/` directory. Warn if any layer has < 3 options (PITFALLS P-5 detection).

**R-S10 — First test suite.** `pytest` suite covering the pure-function surface: time-signature validators, `DurationValidator`, random parameter samplers (seeded), chord pattern parsing. No audio-dependent tests yet. CI-runnable in < 10 s.

---

## R-X — Extract (refactor god-file into modules under `src/musicgen/`)

Extraction order follows ARCHITECTURE.md. Each extraction lands with unit tests.

**R-X1 — Package skeleton.** Create `src/musicgen/` with `__init__.py`, `__main__.py`, `cli.py`. Add `pyproject.toml` with `hatchling` build backend, `requires-python >= 3.9`, CLI entry point `musicgen = "musicgen.cli:app"`, and `[project.optional-dependencies].dev = ["pytest>=8.0", "pytest-cov>=5.0", "pytest-xdist>=3.5"]`. Add `typer>=0.12` as a runtime dependency. **[Status: CLOSED 2026-04-18 by Plan 03-01.](/home/bidu/musicgen/.planning/phases/03-package-skeleton-sampler-generators-extraction/03-01-SUMMARY.md)** Override applied: `requires-python = ">=3.10"` (typer+hatchling pins rule out 3.9; see RESEARCH Risk #1).

**R-X2 — Sampler.** Extract `sampler.py` with a `SongParams` dataclass and all `generate_random_*` + `generate_song_arrangement` functions. Functions take an explicit `rng: random.Random` parameter — no use of the module-level `random`. Unit-tested.

**R-X3 — Generators.** Extract `generators/{chord,melody,bassline,beat}.py`. Each takes `SongParams` + an injected `rng` + the relevant pattern files. Unit-tested.

**R-X4 — Renderer.** Extract `renderer.py` wrapping FluidSynth. Parallelizes the four per-part stem renders with `ThreadPoolExecutor(max_workers=4)`. Records the FluidSynth version at module load time.

**R-X5 — Mixer.** Extract `mixer.py` with pedalboard FX application and pydub overlay logic, using the fixed gain/pan APIs from R-S4. `mix_and_save` shrinks to orchestration only.

**R-X6 — Annotator.** Extract `annotator.py` as a pure function: `(SongParams, render_results, mix_results, beat_times, musicality_score) → dict`. Unit-tested with fixture data.

**R-X7 — Beats.** Replace standalone `beat_anotator.py` with `beats.py` that derives beat and downbeat timestamps from MIDI ticks (not theoretical grid), swing-aware. Unit-tested.

**R-X8 — Integration test.** One end-to-end `@pytest.mark.slow` test generates a single song and asserts file layout + annotation schema.

---

## R-P — Productize (library + CLI + reproducibility)

**R-P1 — Per-sample output layout.** Index-based directory: `<dataset-root>/<zero-padded-index>/` containing `sample.json`, `mix.wav`, `stems/{beat,melody,harmony,bassline}.wav`, `midi/{beat,melody,harmony,bassline}.mid`. `sample.json` is always written last (resume sentinel). Silent stem files are written for absent layers. (Fixes PITFALLS P-7.)

**R-P2 — Stem persistence + sum assertion.** Post-FX per-layer stems are persisted as WAVs. A post-generation assertion checks `max(|sum(stems) − mix|) < ε` for a small ε. If the assertion fails, the sample is marked `status: failed` in the manifest. (Fixes PITFALLS P-2.)

**R-P3 — MIDI persistence.** Per-layer MIDI files are persisted in `midi/` in the per-sample directory.

**R-P4 — `sample.json` schema.** One JSON per sample containing at minimum:
  - `seed`, `fluidsynth_version`, `musicgen_version`
  - `key`, `mode`, `tempo_bpm`, `time_signature`, `swing`, `duration_seconds`
  - `song_arrangement` (list of `{part, start_seconds, end_seconds}`)
  - per-part: `chord_progression`, `active_layers`, `soundfonts`, `fx_params`
  - `beat_times` (seconds), `downbeat_times` (seconds)
  - `musicality_score` (full analyzer output)
  - relative paths: `mix`, `stems.*`, `midi.*`
  - optional: `analysis_failed: true` if scoring raised

**R-P5 — `manifest.jsonl`.** Top-level index at `<dataset-root>/manifest.jsonl`, one line per sample, appended under a `multiprocessing.Manager().Lock()` for safe parallel writes. Each line includes sample index, seed, status (`ok` / `failed`), and relative path to `sample.json`.

**R-P6 — Deterministic train/valid/test split.** A stable hash of the seed assigns each sample to a split (default ratios configurable; e.g. 80/10/10). Split assignment is recorded in `sample.json` and `manifest.jsonl`.

**R-P7 — Seed discipline.**
  - `derive_sample_seed(global_seed, index) = int(sha256(f"{global}:{index}")[:8])`.
  - Named `random.Random` instances per domain (`params`, `generators`, `soundfonts`, `fx`, `mix`) derived from the sample seed.
  - No bare `random.*` anywhere in the pipeline.
  - Workers seed their local RNGs on entry; never inherit from the parent. (Fixes PITFALLS P-4.)
  - Global `random` state never touched; if `music21` or another dep mutates it, wrap affected calls in a save/restore context.

**R-P8 — Determinism contract.** Same global seed on the same machine with the same FluidSynth binary produces bit-identical MIDI, bit-identical `sample.json`, and bit-identical WAV. Cross-binary WAV identity is NOT guaranteed. A pytest regression test runs a fixed small seed and asserts `sha256(mix.wav)` matches a checked-in golden value. (Mitigates PITFALLS P-1.)

**R-P9 — FluidSynth pre-roll measurement.** A one-time calibration derives FluidSynth's startup silence offset for the installed binary and stores it in a cached file under `.musicgen/fluidsynth_preroll.json`. This offset is applied to all MIDI-anchored annotations (beat times, note onsets). Also recorded in `sample.json` for auditability. (Mitigates PITFALLS P-3, P-8.)

**R-P10 — Batch generation.** `musicgen.generate_batch(config)` runs N samples in parallel via `ProcessPoolExecutor(max_workers=config.workers or os.cpu_count())`. Each worker receives only `(sample_index, global_seed, config)` — no shared state. Completed samples are appended to `manifest.jsonl` as they finish.

**R-P11 — Resumability.** Re-running `generate_batch` against an existing dataset directory skips samples whose `sample.json` already exists. Failed samples (no `sample.json`, or manifest status `failed`) are retried. Resumption is idempotent.

**R-P12 — Library API.**
  - `musicgen.generate(config) -> SampleResult` — generate one sample.
  - `musicgen.generate_batch(config) -> BatchResult` — generate N samples, manifest-tracked.
  - Both accept a `Config` dataclass with at least: `global_seed`, `count`, `output_dir`, `workers`, `output_mode`, `split_ratios`, `overrides` (for biasing distributions).

**R-P13 — CLI.** `typer`-based entry point:
  - `musicgen generate --count N --out DIR --seed S [--workers W] [--output-mode MODE] [--verbose]`
  - `musicgen clean --failed` — remove incomplete per-sample directories.
  - Verbosity driven by `-v`/`-q` flags.

**R-P14 — Output mode flag.** `--output-mode` selects `full` (stems + midi + mix + annotations, default), `mix-only`, `stems-only`, or `midi-only`. Writer and annotator honor the flag.

**R-P15 — Progress + logging.** JSON-formatted structured logs during batch runs. Each sample logs start/finish with sample index, seed, and duration. Parallel workers append atomically.

**R-P16 — Failure isolation.** A single failing sample does not abort the batch. Failures are logged with traceback, manifest-marked `status: failed`, and generation continues. Aggregated failure count is reported at the end.

---

## R-Q — Quality + docs (ship-quality checklist for v0.1)

**R-Q1 — README refresh.** Update `README.md` to document: what the tool generates, how to install (`pip install -e .`), minimal library usage example, minimal CLI usage example, dataset directory layout, determinism contract, soundfont licensing caveat.

**R-Q2 — Test coverage target.** Pure-function coverage ≥ 80% (samplers, generators, annotator, beats, time-sig registry, DurationValidator). Integration tests cover one end-to-end sample + one end-to-end batch of 4 samples.

**R-Q3 — Regression test.** The determinism regression test from R-P8 is part of CI.

**R-Q4 — Version field.** `pyproject.toml` declares `version = "0.1.0"`. `musicgen_version` appears in every `sample.json`. **[Status: pyproject portion CLOSED 2026-04-18 by Plan 03-01](/home/bidu/musicgen/.planning/phases/03-package-skeleton-sampler-generators-extraction/03-01-SUMMARY.md); `musicgen_version` in `sample.json` remains for Phase 5.]**

---

## Out of scope for v0.1 (deferred milestones)

- **Extend (new musical vocabulary):** broader genres, richer chord progressions, new time signatures, more drum patterns, more instruments. Deferred to milestone v0.2.
- **Research (quality improvements):** smarter Markov, ML-assisted generators, regeneration loops. Deferred to milestone v0.3.
- **Broader soundfont pool:** detecting narrow pools lands in v0.1; broadening the actual `sf/` contents is v0.2.
- **License audit + CC0/MIT soundfont replacement:** gated by any plan to publish audio externally; not needed for internal-use datasets.
- **Sharded directory layout** (`dataset/<hex>/<id>/`): not required at 1k–10k scale; revisit if user targets 100k+.
- **Cloud / distributed generation, web UI, HTTP API:** explicit PROJECT.md anti-features.
- **Human-in-the-loop quality gating:** `MusicalityAnalyzer` stays a label, not a gate.

---

## Cross-reference map

| Requirement | PROJECT.md Active | PITFALLS | ARCHITECTURE | FEATURES |
|---|---|---|---|---|
| R-S1 | Stabilize #1 | — | Build order #1 | — |
| R-S3 | *(new from research)* | P-A | — | — |
| R-S4 | *(new from research)* | P-B | — | Table stakes: stems sum to mix |
| R-S6 | Stabilize #9 | — | Build order #5 | — |
| R-X1..X8 | Productize prereqs | — | Module layout | — |
| R-P1 | Productize #7, #8 | P-7 | Output layout | Table stakes: dataset structure |
| R-P2 | Productize #9 | P-2 | — | Table stakes: stems |
| R-P3 | Productize #9 | — | — | Table stakes: MIDI |
| R-P4 | Productize #8 | — | Data flow | Table stakes: annotations |
| R-P5 | Productize #10 | — | Manifest | Table stakes: manifest |
| R-P6 | *(new from research)* | — | — | Differentiator: reproducible split |
| R-P7 | Productize #3 | P-4 | Seed propagation | — |
| R-P8 | Productize #3 | P-1 | — | Differentiator: reproducibility |
| R-P9 | *(new from research)* | P-3, P-8 | — | Transcription task support |
| R-P10 | Productize #4, #5 | — | batch.py | — |
| R-P11 | Productize #6 | — | Resume logic | — |
