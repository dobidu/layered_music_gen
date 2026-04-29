# Testing

## Overview

The project has a full pytest-based test suite with 745 fast tests and a separate set of slow tests gated on the FluidSynth binary. Coverage sits at 87% across `src/musicgen/`. CI runs on every push to `main` and `productize-v0.1-pr` across Python 3.10 and 3.12.

## Running the tests

```
# Fast tests only (used by CI):
pytest -m "not slow" --cov=src/musicgen --cov-report=term-missing -q

# Slow (FluidSynth-dependent) tests:
pytest -m slow

# All tests:
pytest

# Regenerate SHA-256 golden fixtures (only on a pinned-FluidSynth host):
pytest -m slow --regen-goldens tests/test_determinism_golden.py
```

## Pytest marks

Two marks are registered in `pyproject.toml`:

- `slow` — FluidSynth rendering tests. Require the `fluidsynth` binary on `PATH` and at least one `.sf2` file per layer in `sf/<layer>/`. Tests that carry this mark are skipped in CI via `-m "not slow"`.
- `integration` — End-to-end tests requiring system dependencies. Currently overlaps with `slow` for the full-generation and batch integration tests.

## Coverage

87% line coverage across `src/musicgen/`. The CI step enforces a floor of 80% via `--cov-fail-under=80`. Coverage is measured on the fast test run only; slow tests are not counted in CI because they require system dependencies not available on the runner.

## CI workflow

`.github/workflows/ci.yml` runs on `ubuntu-latest` in a matrix of Python 3.10 and 3.12. Steps:

1. `pip install -e '.[dev]'` — installs the package in editable mode plus `pytest`, `pytest-cov`, `pytest-xdist`.
2. `pytest -m "not slow" --cov=src/musicgen --cov-report=term-missing -q` — runs the full fast suite.
3. `pytest -m "not slow" --cov=src/musicgen --cov-fail-under=80 -q --no-header` — enforces the coverage threshold.

No FluidSynth binary is installed on the runner; all slow tests are excluded.

## Test files and what they cover

| File | What it covers |
|---|---|
| `test_api.py` | `generate()` validation (missing seed, negative index), resume short-circuit, failure isolation, failed manifest entry |
| `test_batch.py` | `generate_batch()` skip logic, success/failure counting, manifest writes, progress events to stderr; ThreadPoolExecutor stand-in for spawn context |
| `test_cli.py` | `musicgen generate`, `clean`, `calibrate` commands via Typer test client; invalid `--output-mode` rejection |
| `test_config.py` | `Config` defaults, env-var layer (`MUSICGEN_SF_DIR`, `MUSICGEN_DATASET_ROOT`, `MUSICGEN_OUTPUT_MODE`), CLI override layer, validation errors (bad split ratios, invalid output mode, count < 1) |
| `test_sampler.py` | All five sampling functions; `SongParams.sample`; `validate_measures_dict`; AST bare-random guard for `sampler.py` |
| `test_seeds.py` | `derive_sample_seed` stability across calls; `make_rngs` domain isolation; `assign_split` train/val/test distribution; `save_random_state` context manager |
| `test_renderer.py` | `pick_soundfonts` determinism and error paths; `render_stems` with mocked `FluidSynth`; `FLUIDSYNTH_VERSION` fallback |
| `test_mixer.py` | `build_fx_boards`; `compute_layer_mask`; `mix_part` with synthetic WAVs; `concat_parts`; silent-stem fallback shape |
| `test_beats.py` | `extract_beat_times` and `extract_downbeat_times` against hand-crafted MIDI fixtures |
| `test_annotator.py` | Annotation dict structure and required fields across all supported time signatures and arrangements |
| `test_writer.py` | Atomic write ordering; MIDI concatenation (absolute-tick walk); stem WAV concatenation; sum-of-stems assertion (pass and fail); `_apply_preroll_offset`; `_rewrite_paths_relative`; `output_mode` file presence |
| `test_manifest.py` | `ManifestWriter.append`; `is_sample_complete` sentinel detection; concurrent append safety |
| `test_calibrate.py` | `load_preroll` cache hit/miss/version-stale paths; `measure_preroll` with mocked subprocess; `save_preroll` JSON structure |
| `test_duration_validator.py` | `DurationValidator` note duration validation across all registered time signatures |
| `test_time_signature.py` | `verify_pattern_for_time_signature`; `verify_beat_pattern`; `validate_measures` alias; all time signatures in registry |
| `test_timesig_registry.py` | `TimeSignatureRegistry.lookup`; `sample_random`; `measures_for`; `measure_count_valid`; `melody_duration_candidates`; all 7 registered time signatures |
| `test_split.py` | `assign_split` produces correct proportions over a large sample of indices |
| `test_output_mode.py` | All four `output_mode` values: correct files written, correct files absent, `sample.json` fields set appropriately |
| `test_determinism_golden.py` | Two test classes (see Determinism below) |
| `test_no_bare_random_in_package.py` | AST scan over every `*.py` under `src/musicgen/`; fails if any bare `random.<method>()` call is found (excluding the permitted set: `random.Random`, `random.getstate`, `random.setstate`) |
| `test_music21_isolation.py` | Checks that `music21` import does not pollute global random state or logging configuration |
| `test_music_gen_logging.py` | `music_gen.py` wrapper logging behaviour |
| `test_integration_full_generation.py` | `@pytest.mark.slow` — full pipeline end-to-end with real FluidSynth; asserts stem/mix/MIDI files exist; `sample.json` required fields; MIDI bit-identity across two same-seed runs |
| `test_integration_batch.py` | `@pytest.mark.slow` — real `generate_batch` with spawn context; multi-sample resume |
| `tests/test_generators/test_chord.py` | `generate_chord_progression` output MIDI structure |
| `tests/test_generators/test_melody.py` | `generate_melody` note range, scale validity, measure count |
| `tests/test_generators/test_bassline.py` | `generate_bassline` coordination with chord progression |
| `tests/test_generators/test_beat.py` | `generate_beat` pattern validity, swing offset, time signature coverage |
| `tests/test_generators/test_no_bare_random.py` | AST bare-random guard scoped to `generators/*.py` |

## Determinism test strategy

Determinism is tested at two levels, both in `test_determinism_golden.py`.

**Same-process stability (`TestSameProcessStability`, fast):** Calls `generate()` twice in one process with the same `global_seed` and `sample_index` but different `dataset_root` directories. `renderer.render_stems`, `renderer.pick_soundfonts`, and `musicality.get_musicality_score` are monkeypatched to return deterministic stubs that still consume the correct number of RNG draws. The SHA-256 of both `sample.json` files is compared for byte equality. This catches wall-clock leaks (`datetime.now`), entropy leaks (`os.urandom`, unseeded module-level `random`), and non-stable iteration order in `annotator` or `writer`.

**SHA-256 goldens (`TestDeterminismGoldens`, slow):** Runs the full pipeline with `global_seed=1, sample_index=0` and computes SHA-256 hashes of six artifacts: `mix.wav`, `sample.json`, and four MIDI files. Hashes are compared against fixture files in `tests/fixtures/determinism/`. The `mix.wav` golden is additionally gated on FluidSynth version: if the installed binary differs from `fluidsynth_version.txt`, the WAV test is marked `xfail` rather than failing, because FluidSynth synthesis is not bit-identical across binary versions. MIDI and `sample.json` goldens must pass unconditionally when the binary is present. To regenerate goldens after an intentional RNG order change or FluidSynth upgrade, run `pytest -m slow --regen-goldens tests/test_determinism_golden.py` on a host with the pinned binary.

The `conftest.py` registers the `--regen-goldens` flag so it is available across the test run.

## Bare-random enforcement

Two overlapping AST scan tests ensure that no bare `random.<method>()` calls are introduced in the pipeline code:

- `test_no_bare_random_in_package.py` — scans every `*.py` under `src/musicgen/` (parametrized, so new modules are covered automatically).
- `tests/test_generators/test_no_bare_random.py` — redundant scoped guard for `generators/*.py`.
- `test_sampler.py::test_no_bare_random_in_sampler` — scoped guard for `sampler.py`.

Permitted uses: `random.Random` (RNG factory), `random.getstate`, `random.setstate` (used by `seeds.save_random_state`). All other `random.*` attribute calls cause the test to fail. The correct pattern throughout the codebase is to accept an injected `rng: random.Random` parameter and call `rng.choice(...)`, `rng.random()`, etc.

## Fixtures and mocking patterns

Tests that exercise pipeline logic without FluidSynth use one of these approaches:

- **Monkeypatching `renderer.render_stems`** — replaced with a stub that writes stereo 44.1 kHz silent WAVs and returns a `RenderResult` with `duration_seconds=0.5`. The stub still draws from the correct RNG argument to preserve draw order.
- **Monkeypatching `renderer.pick_soundfonts`** — replaced with a stub that makes one `rng.choice` per layer (matching the real draw count) and returns fake paths.
- **Monkeypatching `musicality.get_musicality_score`** — returns `(0.5, {"rhythm": 0.5, "harmony": 0.5})` deterministically.
- **Synthetic WAV fixtures** — `test_writer.py` and `test_mixer.py` generate silent or tonal WAVs in `tmp_path` using `pydub.AudioSegment` and `scipy.io.wavfile`.
- **Tiny MIDI fixtures** — `test_writer.py` and `test_beats.py` write minimal 1–4-note MIDI files using `mido` directly.
- **`patch_executor` fixture in `test_batch.py`** — replaces `ProcessPoolExecutor` with `ThreadPoolExecutor` to avoid spawn-context overhead in fast tests.

`conftest.py` only registers the `--regen-goldens` option; fixtures are defined locally in each test module via `@pytest.fixture` or inline helpers.
