# musicgen — synthetic music dataset generator

A Python library for generating **reproducible, fully-annotated** synthetic music samples for ML/MIR research. Each sample lands in a canonical per-sample directory with the mixed audio, per-layer stems, per-layer MIDI, and a rich JSON annotation describing every musical and synthesis parameter.

Suitable for training models that learn music tagging, source separation, beat/tempo/downbeat detection, and audio→MIDI transcription at the 1k–10k sample scale.

## Status

- **Milestone v0.1 — in progress.** 5 of 7 phases complete.
- **What works today (Phase 5):** single-sample library API (`musicgen.generate(Config)`), per-sample directory layout, deterministic seed propagation, bit-identical re-runs, sum-of-stems integrity check, manifest tracking, train/valid/test split.
- **Coming in Phase 6:** parallel batch runner, full `typer` CLI, FluidSynth pre-roll calibration, resumability, output-mode flag.
- **Test suite:** 690 fast tests passing (`pytest -m "not slow"`); 6 slow FluidSynth-gated tests collected separately.

## Core value

Every generated sample is a complete, reproducible, fully-labeled training example. If the stems drift from the mix, the MIDI doesn't match the audio, the seed doesn't reproduce, or the annotations are wrong, the dataset is worthless — no matter how musical it sounds.

The **determinism contract**: same `global_seed` + same `sample_index` → bit-identical MIDI and bit-identical `sample.json`, always. Bit-identical WAV when running on the same FluidSynth binary version.

## Requirements

- Python ≥ 3.10
- FluidSynth binary on `PATH` (for synthesis)
- SoundFont (`.sf2`) files in `sf/<layer>/` (one or more per layer)

All Python deps are in `pyproject.toml` and install automatically. The `requirements.txt` file was removed in Phase 3 — `pyproject.toml` is the single authoritative dependency manifest.

## Installation

```bash
git clone https://github.com/dobidu/layered_music_gen.git
cd layered_music_gen

python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e '.[dev]'
```

Install FluidSynth:

```bash
# Ubuntu/Debian
sudo apt-get install fluidsynth

# macOS
brew install fluidsynth

# Windows
# Download installer from https://www.fluidsynth.org/
```

Place SoundFont files (one per layer minimum; more is better — narrow pools cause models to overfit timbre):

```
sf/
├── beat/        *.sf2
├── melody/      *.sf2
├── harmony/     *.sf2
└── bassline/    *.sf2
```

## Usage

### Library API

The public surface is exactly four names:

```python
from musicgen import generate, Config, SampleResult, __version__
```

Generate one sample:

```python
from musicgen import generate, Config

result = generate(Config(
    global_seed=42,
    sample_index=0,
    dataset_root="./dataset",
))

print(result.sample_dir)         # "./dataset/000000"
print(result.split)              # "train" | "valid" | "test"
print(result.musicality_score)   # float
print(result.status)             # "ok" | "failed"
```

Re-running with the same `(global_seed, sample_index)` short-circuits if the sample is already on disk (sentinel = `sample.json` exists), so batch retries are idempotent.

### Smoke test

The repo-root `music_gen.py` is a 60-line smoke wrapper that calls `musicgen.generate(Config(global_seed=1, sample_index=0))`:

```bash
python music_gen.py
```

It will be replaced by the full `musicgen generate` CLI in Phase 6.

### Configuration

`Config` is a `@dataclass` with three precedence layers: **CLI args > env vars > defaults**.

| Field | Default | Notes |
|---|---|---|
| `global_seed` | `None` | **Required at `generate()` time** — raises `ValueError` if `None`. Library refuses to silently default; reproducibility is the core value prop. |
| `sample_index` | `0` | Per-sample identity within the dataset. `derive_sample_seed(global_seed, sample_index)` is the per-sample RNG basin. |
| `dataset_root` | `<repo>/dataset` | Where samples land. Override via `MUSICGEN_DATASET_ROOT` env var or pass directly. |
| `split_ratios` | `(0.8, 0.1, 0.1)` | Train/valid/test. Validated at `__post_init__` (must sum to 1.0 ± 1e-9, all non-negative). |
| `sum_of_stems_epsilon` | `1e-3` | Peak abs-difference tolerance (normalized float32) for the stems-sum-to-mix invariant. |
| `keep_working_dirs` | `False` | Set `True` to preserve per-sample working directories for debugging. |
| `workers` | `None` | Phase 6 reserved (batch runner). Ignored today. |
| `sf_dir` | `<repo>/sf` | Override via `MUSICGEN_SF_DIR` env var. |

Other paths (FX JSONs, levels, song structures, chord patterns, beat-roll patterns) are also `Config` fields with sensible defaults — see `config.py`.

#### Domain-specific config files

| File | Purpose |
|---|---|
| `song_structures.json` | Possible song arrangements (intro / verse / chorus / bridge / outro). |
| `chord_patterns.txt` | Chord progressions per song part. |
| `beat_roll_patterns_<sig>.txt` | Drum patterns per time signature (2/4, 3/4, 4/4, 5/4, 6/8, 7/8, 12/8). |
| `inst_probabilities.json` | Per-layer inclusion probabilities per part. |
| `levels.json` | Per-layer gain + pan (linear amplitudes 0–1, converted to dB at apply time). |
| `*_fx.json` | FX chain parameter ranges per layer (Compressor, Reverb, Delay, Chorus, Phaser, Filter, Gain). |

## Per-sample output layout

Each sample lands in a zero-padded numbered directory:

```
<dataset_root>/
├── manifest.jsonl                  # one line per sample (append-under-lock)
└── 000042/                         # 6-digit zero-padded index
    ├── sample.json                 # rich annotation — written LAST (resume sentinel)
    ├── mix.wav                     # final mixed audio
    ├── stems/                      # post-FX per-layer stems
    │   ├── beat.wav
    │   ├── melody.wav
    │   ├── harmony.wav
    │   └── bassline.wav
    └── midi/                       # per-layer MIDI (concatenated across song parts)
        ├── beat.mid
        ├── melody.mid
        ├── harmony.mid
        └── bassline.mid
```

`sample.json` is **always written last** — its presence means the sample is complete. Re-running `generate()` with the same `(global_seed, sample_index)` skips work when this sentinel exists.

`manifest.jsonl` accumulates one JSON object per sample with `sample_index`, `seed`, `status` (`ok`/`failed`), `split`, `path`, `musicality_score`, `duration_seconds`, and `wrote_at`.

### `sample.json` schema (R-P4)

Every sample carries:

- **Identity:** `seed`, `musicgen_version`, `fluidsynth_version`
- **Musical params:** `key`, `mode`, `tempo_bpm`, `time_signature`, `swing`, `duration_seconds`
- **Structure:** `song_arrangement` (list of `{part, start_seconds, end_seconds}`)
- **Per-part:** `chord_progression`, `active_layers`, `soundfonts`, `fx_params` (with sampled effect parameters), `time_signatures_per_part`, `measures_per_part`
- **Annotations:** `beat_times`, `downbeat_times` (seconds, swing-aware via MIDI-tick extraction)
- **Quality:** `musicality_score` (full analyzer output: tempo / harmony / rhythm / timbre / SNR)
- **Routing:** `split` (`train` / `valid` / `test`, deterministic from seed)
- **Paths:** `mix`, `stems.*`, `midi.*` (per-sample-dir-relative, e.g. `"mix.wav"`, `"stems/beat.wav"`)

`sample.json` is serialized canonically (`sort_keys=True`) so byte-identical re-runs are detectable via SHA-256 without parsing.

## Determinism

Same global seed + same sample index → bit-identical MIDI + bit-identical canonical `sample.json` **unconditionally**. WAV bit-identity holds when the FluidSynth binary version matches.

A two-tier regression test enforces this:

- **`tests/test_determinism_golden.py::TestDeterminismGoldens`** (`@pytest.mark.slow`) — parametrized over 6 SHA-256 artifacts (mix.wav + 4 MIDIs + canonical sample.json). Skips when FluidSynth is absent; xfails when FluidSynth version differs from the captured `tests/fixtures/determinism/fluidsynth_version.txt`. MIDI + sample.json hashes assert unconditionally.
- **`tests/test_determinism_golden.py::TestSameProcessStability`** (fast, no FluidSynth) — runs `generate()` twice in one process and asserts `sha256(sample.json)` matches across runs. Catches our-code-nondeterminism cheaply.

To capture the slow goldens on a FluidSynth-equipped host:

```bash
pytest -m slow --regen-goldens tests/test_determinism_golden.py
git add tests/fixtures/determinism/expected_*.sha256 tests/fixtures/determinism/fluidsynth_version.txt
git commit -m "test(05): capture determinism goldens (FluidSynth <version>)"
```

## Seed discipline (R-P7)

Five named `random.Random` instances per sample, derived deterministically from the sample seed:

```python
sample_seed = derive_sample_seed(global_seed, sample_index)   # sha256[:8]
rngs = make_rngs(sample_seed)
# {"params":     Random(seed ^ 0x01),   # key/tempo/time_sig/swing/measures/arrangement
#  "generators": Random(seed ^ 0x02),   # chord/melody/bassline/beat
#  "soundfonts": Random(seed ^ 0x03),   # pick_soundfonts
#  "fx":         Random(seed ^ 0x04),   # build_fx_boards
#  "mix":        Random(seed ^ 0x05)}   # compute_layer_mask
```

Zero bare `random.*` calls anywhere in `src/musicgen/` — enforced by an AST static guard (`tests/test_no_bare_random_in_package.py`). Global `random` state is never touched; the one `musicality_score` call is wrapped in `save_random_state()` for defense in depth against dependency-upgrade leaks.

## Tests

```bash
pytest -m "not slow"      # Fast suite (default CI) — 690 tests in ~5s
pytest -m slow            # Slow suite — requires FluidSynth + .sf2 pools
pytest                    # Everything
```

Coverage targets ≥ 80% on pure functions (samplers, generators, annotator, beats, time-sig registry, validators).

## Roadmap

| # | Phase | Status | Plans |
|---|---|---|---|
| 1 | Stabilize I — bug fixes and guardrails | ✓ COMPLETE | 4/4 |
| 2 | Stabilize II — config + time-signature registry + logging | ✓ COMPLETE | 3/3 |
| 3 | Package skeleton + sampler + generators extraction | ✓ COMPLETE | 5/5 |
| 4 | Renderer + mixer + annotator + beats extraction | ✓ COMPLETE | 7/7 |
| 5 | Productize I — writer, manifest, seed discipline, determinism | ✓ COMPLETE | 6/6 |
| 6 | Productize II — FluidSynth calibration, batch generation, CLI, resumability | ○ next | — |
| 7 | Ship v0.1 — docs, polish, regression suite | ○ pending | — |

### Phases delivered

- **Phase 1 — Stabilize I.** Importability fix (`if __name__ == '__main__'`), arrangement-reroll bug, pydub gain/pan no-op fix, narrow exception handlers, dead-code removal, first pytest skeleton.
- **Phase 2 — Stabilize II.** `config.py` owns all paths (CLI > env > defaults precedence). `timesig.py` registry consolidates 7 time signatures (one source of truth). All `print()` → structured `logging`.
- **Phase 3 — Package skeleton.** `src/musicgen/` installable package via `hatchling` + `pyproject.toml`. Sampler + generators extracted with injected `rng: random.Random` parameters. AST static guard against bare `random.*`. `enhanced_duration_validator.py` → `src/musicgen/duration_validator.py`. `requirements.txt` deleted.
- **Phase 4 — Audio-side extraction.** `src/musicgen/{renderer, mixer, annotator, beats}.py`. `ThreadPoolExecutor` parallel stem rendering. Pure-function annotator producing the R-P4 schema. Swing-aware MIDI-tick beat derivation (replaces the old grid-based `beat_anotator.py`, which is deleted). `mix_and_save` is gone; `music_gen.py` collapses to a thin orchestrator.
- **Phase 5 — Productize I.** `src/musicgen/{seeds, writer, manifest, api, musicality}.py`. Per-sample directory layout with atomic sentinel ordering. Sum-of-stems integrity check (int32 accumulator, ε=1e-3). Five-domain RNG hierarchy. `manifest.jsonl` with threading.Lock + `os.fsync`. Deterministic train/valid/test split. `musicgen.generate(Config) → SampleResult` library entry point. `music_gen.py` collapsed from 199 → 60 lines.

### Phases ahead

- **Phase 6 — Productize II.** `musicgen.generate_batch(config)` via `ProcessPoolExecutor`. FluidSynth pre-roll calibration (caches in `.musicgen/fluidsynth_preroll.json`, applied to beat-time annotations). Full `typer` CLI: `musicgen generate --count N --out DIR --seed S [--workers W] [--output-mode MODE]` and `musicgen clean --failed`. `--output-mode` flag (`full` / `mix-only` / `stems-only` / `midi-only`). Resumability beyond the sentinel check. Failure isolation (one bad sample doesn't kill a 10k batch).
- **Phase 7 — Ship v0.1.** README polish, ≥80% coverage, determinism regression test wired into CI, generate a 32-sample acceptance dataset, tag `v0.1.0`.

## Architecture (post-Phase 5)

```
src/musicgen/
├── __init__.py          # public exports: generate, Config, SampleResult, __version__
├── api.py               # generate(Config) — composition root
├── config.py (root)     # Config dataclass with CLI > env > defaults precedence
├── seeds.py             # derive_sample_seed, make_rngs, save_random_state, assign_split
├── sampler.py           # SongParams + key/tempo/time-sig/swing/measures/arrangement
├── generators/
│   ├── chord.py         # generate_chord_progression
│   ├── melody.py        # generate_melody (Markov-style over chord progressions)
│   ├── bassline.py      # generate_bassline (keyed to chords + melody)
│   └── beat.py          # generate_beat (drum patterns + swing offset)
├── renderer.py          # FluidSynth wrapper, ThreadPoolExecutor stem rendering
├── mixer.py             # FX (pedalboard), pydub overlay, layer mask, part concat
├── beats.py             # MIDI-tick beat + downbeat extraction (mido), swing-aware
├── annotator.py         # pure-function R-P4 schema assembler
├── musicality.py        # MusicalityAnalyzer (tempo, harmony, rhythm, timbre, SNR)
├── writer.py            # atomic per-sample dir, sum-of-stems assertion, MIDI/stem concat
├── manifest.py          # ManifestWriter (append-under-lock, JSONL)
└── duration_validator.py
```

Pipeline flow inside `generate(config)`:

```
sampler → generators → renderer (FluidSynth, parallel stems)
       → mixer (FX + overlay + concat)
       → beats (MIDI-tick extraction)
       → annotator (R-P4 dict)
       → writer (atomic sample dir + sum-of-stems)
       → manifest (JSONL append)
       → SampleResult
```

## Out of scope for v0.1

Deferred milestones (per `.planning/REQUIREMENTS.md`):

- **Extend (v0.2):** broader genres, richer chord vocab, additional time signatures, more drum patterns, broader soundfont pool.
- **Research (v0.3):** smarter Markov, ML-assisted generators, regeneration loops.
- License audit + CC0/MIT soundfont replacement (gated by external publication).
- Sharded directory layout (`dataset/<hex>/<id>/`) — only needed past 100k samples; 6-digit indices cover 1M.
- Cloud / distributed generation, web UI, HTTP API — explicit anti-features.

## Contributing

PRs welcome. Project planning lives under `.planning/` (PROJECT.md, REQUIREMENTS.md, ROADMAP.md, per-phase CONTEXT/RESEARCH/PLAN/SUMMARY/VERIFICATION/REVIEW). Phase work follows the GSD workflow (`.claude/get-shit-done/`).

## License

See [LICENSE](LICENSE).

## Acknowledgments

- [music21](https://web.mit.edu/music21/) for music theory primitives
- [FluidSynth](https://www.fluidsynth.org/) for soundfont synthesis
- [pedalboard](https://github.com/spotify/pedalboard) for audio effects
- [mido](https://mido.readthedocs.io/) for MIDI manipulation
- [librosa](https://librosa.org/) for audio analysis
