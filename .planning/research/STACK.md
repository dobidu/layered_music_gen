# Stack Research

**Scope:** Additional stack/tooling to turn `music_gen.py` into a reproducible, parallel, CLI+library dataset generator. Existing stack (`midiutil`, `music21`, `midi2audio` + FluidSynth, `pedalboard`, `pydub`, `librosa`) is NOT replaced.

**Confidence:** MEDIUM overall — no live PyPI verification was possible this session.

## Summary

- **One new runtime dependency needed:** `typer>=0.12` for CLI.
- **Parallelism:** stdlib `concurrent.futures` only. No `joblib`, `ray`, `dask`.
- **Manifest:** JSONL + per-sample `annotation.json`, stdlib `json` only.
- **Logging:** `python-json-logger` is **already in `requirements.txt`** — zero churn.
- **Packaging:** `pyproject.toml` with `hatchling` build backend.

## Recommended additions

### CLI — `typer>=0.12` (MEDIUM confidence)

Type-hint driven; the same `generate(seed, count, out_dir)` function is both library API and CLI command. Avoids argparse boilerplate and manual click decorators. `click>=8.0` is pulled in transitively.

**Not argparse:** verbose, no shell completion, manual sync to function signatures is fragile during refactors.
**Not click directly:** Typer compiles down to Click; writing Click by hand is more indirection.

### Test framework — `pytest>=8.0`, `pytest-cov>=5.0`, `pytest-xdist>=3.5` (HIGH)

Place in `[project.optional-dependencies].dev`.

- Pure unit tests (validators, samplers, generators): `random.seed(42)` fixture, call fn, assert on returned MIDIFile/list. Start here.
- Audio integration tests marked `@pytest.mark.slow` and `@pytest.mark.integration` — skip in fast CI.
- FluidSynth render tests: minimal MIDI → render to `tmp_path` → assert file exists and SHA-256 matches golden (when pinned).
- No dedicated audio-testing library (`auraloss` etc.) — `librosa.load` + `np.allclose` on short fixtures is sufficient.

### Parallelism — `concurrent.futures` (stdlib, HIGH)

**Cross-sample:** `ProcessPoolExecutor(max_workers=os.cpu_count())`. Each worker receives `seed = derive_sample_seed(global_seed, sample_index)` as an explicit argument. CPU-bound Python (sampling, MIDI authoring) parallelizes across cores.

**Intra-sample:** `ThreadPoolExecutor(max_workers=4)` for the four FluidSynth renders per part (`music_gen.py:828, 831, 834, 837`). FluidSynth is a subprocess — threads suffice; `ProcessPoolExecutor` adds fork overhead with no benefit.

Replaces the dead `multiprocessing.Pool` import at `music_gen.py:13`.

**Not joblib / ray / dask:** over-engineering for 1k–10k on one machine. PROJECT.md rules out cluster scale.

### Manifest + annotations — stdlib `json` (HIGH)

- Top-level `manifest.jsonl` — one JSON line per sample. `load_dataset("json", data_files="manifest.jsonl")` works directly in HuggingFace Datasets.
- Per-sample `annotation.json` — full rich metadata for the sample.
- Write with `json.dumps(record) + "\n"`. No `jsonlines` package.
- File locking for parallel appends: `multiprocessing.Manager().Lock()` (portable across POSIX + Windows; `fcntl.flock` is POSIX-only).

**Not Parquet:** mismatches mixed-type nested metadata, requires `pyarrow`/`pandas`.
**Not WebDataset:** designed for 100k+ streaming; local files are more debuggable at 1k–10k.

### Seed / RNG — stdlib only (HIGH)

- `random.Random(seed)` instance per domain; no use of the module-level `random.*` global.
- `numpy.random.default_rng(seed)` if numpy randomness is ever introduced (never `np.random.seed()`).
- Every source of randomness — soundfont selection, layer inclusion, FX param jitter, arrangement sampling — threads through an explicit `rng` parameter.

No new library. Do not pull in `faker`, `hypothesis` for generation reproducibility.

### Logging — `python-json-logger` (HIGH — already installed)

Already in `requirements.txt`. Replace 32+ `print()` calls in `music_gen.py` with module-level `logger = logging.getLogger(__name__)` + `logger.info(...)`. JSON-formatted lines survive interleaved parallel output (each line is atomic).

Configure verbosity from `--verbose`/`--quiet` CLI flags in the entry point.

**Not loguru / structlog:** churn without payoff when `python-json-logger` is present.

### FluidSynth — keep `midi2audio`, pin the binary (MEDIUM)

Stay on `midi2audio`. Do **not** switch to `pyfluidsynth` — it wraps the same underlying binary via ctypes; no determinism gain.

**Determinism contract:** "same FluidSynth binary version + same soundfont + same MIDI + same sample rate → bit-identical WAV." Cross-version, cross-platform bit identity is NOT guaranteed (DSP floating-point paths differ). Mitigations:

1. Pin FluidSynth version in a `Dockerfile` or install script.
2. Record the FluidSynth version string in every `annotation.json`.
3. Add a SHA-256 regression test in CI against a canonical render.
4. Accept "MIDI + annotations fully deterministic, audio deterministic only under pinned binary."

This matches the `⚠️ Revisit` entry already in PROJECT.md Key Decisions.

### Packaging — `pyproject.toml` + `hatchling>=1.21` (HIGH)

Required for: `pip install -e .`, CLI entry point, optional-dependencies split, Python version constraint.

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "musicgen"
version = "0.1.0"
requires-python = ">=3.9"
dependencies = [
    # existing requirements.txt contents
    "typer>=0.12",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-cov>=5.0", "pytest-xdist>=3.5"]

[project.scripts]
musicgen = "musicgen.cli:app"

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "slow: FluidSynth rendering tests (deselect with '-m not slow')",
    "integration: requires system FluidSynth binary",
]
```

### Annotations — custom `annotation.json`, not JAMS (HIGH)

JAMS is designed for multi-annotator disagreement on existing recordings. Here every label is exact and known from generator state. Custom schema keeps it simple; JAMS conversion is trivial if ever needed.

## Do NOT add

| Package | Why skipped |
|---|---|
| `joblib` | stdlib `concurrent.futures` covers parallelism |
| `ray`, `dask` | over-engineered for 1k–10k on one box |
| `pyfluidsynth` | same binary underneath; no determinism gain |
| `loguru`, `structlog` | `python-json-logger` already present |
| `jsonlines` | stdlib `json` + `"\n"` is one line of code |
| `pyarrow`, `pandas` | only needed if Parquet manifest is chosen (it isn't) |
| `jams`, `mirdata` | wrong use case (evaluation, not generation) |
| `faker`, `hypothesis` | not needed for deterministic seeded generation |
| `webdataset` | scale does not justify it |

## Open questions

1. **FluidSynth version currently installed** — measure before pinning. Cannot proceed to Productize determinism work without this.
2. **`requires-python` floor** — assumed 3.9; verify against the target system.
3. **`typer` / `click` version conflict** — verify no existing `click` pin is incompatible with `typer>=0.12`.
4. **`uuid` package in `requirements.txt`** — `uuid` is stdlib in Python 3. The pin in `requirements.txt` likely installs a PyPI stub and should be removed.

## Confidence summary

| Area | Confidence |
|---|---|
| CLI: typer | MEDIUM |
| Test: pytest | HIGH |
| Parallelism: concurrent.futures | HIGH |
| Manifest: JSONL + annotation.json | HIGH |
| Logging: python-json-logger (existing) | HIGH |
| Packaging: pyproject.toml | HIGH |
| Seed/RNG: stdlib | HIGH |
| FluidSynth: keep midi2audio | MEDIUM (empirical determinism test needed) |
| Annotations: custom schema, not JAMS | HIGH |
