# musicgen — synthetic music dataset generator

A Python library for generating **reproducible, fully-annotated** synthetic music samples for ML/MIR research. Each sample lands in a canonical per-sample directory with the mixed audio, per-layer stems, per-layer MIDI, and a rich JSON annotation describing every musical and synthesis parameter.

Suitable for training models that learn music tagging, source separation, beat/tempo/downbeat detection, and audio→MIDI transcription at the 1k–10k sample scale.

## Status

- **v0.3.0 — released.** Higher-order Markov complete: 2nd-order chord transition matrices per genre, configurable-order melody Markov over scale-degree intervals, two-layer musicality quality gate (`check_midi_quality` + audio integrity), quality-gate regeneration loop (`Config.min_musicality_score`, `Config.max_attempts`), calibration harness (`run_midi_calibration`, `suggest_threshold`). Tag `v0.3.0`.
- **v0.2.0 — released.** Genre system complete: 8 built-in genres, `GenreSpec` composition engine, extended chord vocabulary, genre-constrained sampler/FX/soundfont selection, `list-genres` CLI, Jupyter demo notebook. Tag `v0.2.0`.
- **v0.2 integrations — complete.** Three opt-in sibling-ecosystem integrations: SoundfontManager-backed soundfont selection, MIDI indexing, and audio stem indexing. Zero new hard dependencies.
- **v0.1.0 — complete.** All 7 phases shipped: single-sample library API, parallel batch runner, full `typer` CLI, FluidSynth pre-roll calibration, resumability, output-mode routing, deterministic seed propagation, sum-of-stems integrity, manifest tracking, train/valid/test split.
- **Test suite:** 1197 fast tests passing (`pytest -m "not slow"`); slow FluidSynth-gated tests collected separately under `pytest -m slow`.

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

### CLI

```bash
# Generate 32 samples with 4 workers, seed 1
musicgen generate --count 32 --seed 1 --out ./dataset --workers 4

# Mix-only mode (no stems or MIDI written)
musicgen generate --count 32 --seed 1 --out ./dataset --output-mode mix-only

# Generate with a genre constraint
musicgen generate --count 8 --seed 1 --genre jazz

# Compose two genres (parameters merged)
musicgen generate --count 8 --seed 1 --genre jazz --genre latin

# List available genre presets
musicgen list-genres

# Clean up failed partial sample directories
musicgen clean --failed --out ./dataset

# Measure and cache FluidSynth pre-roll offset (run once per machine)
musicgen calibrate

# Index all generated MIDI files into a MidiManager database (optional — requires midi_file_manager)
musicgen index-midi --dataset ./dataset --out ./midi_db.json

# Index all generated WAV stems into a SampleManager database (optional — requires audio_sample_manager)
musicgen index-audio --dataset ./dataset --out ./audio_db.json
```

Output mode choices: `full` (default) | `mix-only` | `stems-only` | `midi-only`.

#### `index-midi` options

| Option | Default | Description |
|---|---|---|
| `--dataset PATH` | (required) | Root of a musicgen dataset (contains numbered sample dirs). |
| `--out PATH` | `./midi_db.json` | Output MidiManager database file. |
| `--midi-dir PATH` | `None` | Base directory for MIDI files if the db stores relative paths. |
| `--csv PATH` | `None` | If set, also export the database as CSV. |

#### `index-audio` options

| Option | Default | Description |
|---|---|---|
| `--dataset PATH` | (required) | Root of a musicgen dataset. |
| `--out PATH` | `./audio_db.json` | Output SampleManager database file. |
| `--samples-dir PATH` | `None` | Base directory for WAV files if the db stores relative paths. |
| `--csv PATH` | `None` | If set, also export the database as CSV. |

### Library API

```python
from musicgen import generate, generate_batch, Config, SampleResult, BatchResult, __version__
```

Generate one sample:

```python
from musicgen import generate, Config

result = generate(Config(global_seed=42, sample_index=0, dataset_root="./dataset"))

print(result.sample_dir)         # "./dataset/000000"
print(result.split)              # "train" | "valid" | "test"
print(result.musicality_score)   # float
print(result.status)             # "ok" | "failed"
```

Generate a batch:

```python
from musicgen import generate_batch, Config

result = generate_batch(Config(global_seed=1, count=32, dataset_root="./dataset", workers=4))

print(result.succeeded, result.failed, result.skipped)  # 32 0 0
print(result.duration_seconds)
```

Re-running with the same `(global_seed, sample_index)` short-circuits when `sample.json` already exists — batch retries are idempotent.

### Smoke test

`music_gen.py` at the repo root is a 60-line wrapper that calls the single-sample API:

```bash
python music_gen.py
```

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
| `workers` | `None` (all cores) | `generate_batch` parallel workers. Pass integer to cap. |
| `count` | `1` | Number of samples for `generate_batch`. Override via `MUSICGEN_COUNT`. |
| `output_mode` | `"full"` | `full` / `mix-only` / `stems-only` / `midi-only`. Override via `MUSICGEN_OUTPUT_MODE`. |
| `min_musicality_score` | `0.0` | Quality gate threshold. `0.0` disables the gate. Samples with `musicality_score < threshold` are re-generated up to `max_attempts` times. Override via `MUSICGEN_MIN_MUSICALITY_SCORE`. |
| `max_attempts` | `1` | Maximum re-roll attempts per sample when the quality gate is active. Must be ≥ 1. Override via `MUSICGEN_MAX_ATTEMPTS`. |
| `sf_dir` | `<repo>/sf` | Override via `MUSICGEN_SF_DIR` env var. |
| `soundfont_manager_db` | `None` | Path to a SoundfontManager JSON database. Set to activate metadata-aware soundfont selection (Integration 1). Override via `MUSICGEN_SOUNDFONT_MANAGER_DB`. |
| `soundfont_manager_sf_dir` | `None` | Base directory for `.sf2` files when the SM db stores relative paths. Override via `MUSICGEN_SOUNDFONT_MANAGER_SF_DIR`. |
| `genre` | `None` | List of genre names to constrain generation. Override via `MUSICGEN_GENRE` (comma-separated). |
| `genres_dir` | `<repo>/genres` | Root directory for genre spec files. Override via `MUSICGEN_GENRES_DIR`. |

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

## Genre system (v0.2)

musicgen ships 8 built-in genre presets that constrain generation parameters to produce stylistically coherent samples. Genres are composable — specify multiple to merge their constraints.

### Built-in genres

| Genre | Tempo | Swing | Time sigs | Style |
|---|---|---|---|---|
| `jazz` | 80–200 BPM | 0.60–0.75 | 4/4, 3/4, 6/8, 12/8 | Swing-heavy, ride patterns, maj7/m7 chords |
| `hip-hop` | 70–110 BPM | 0.50–0.65 | 4/4 dominant | Heavy kick-snare, compressed, minor-key bias |
| `blues` | 60–130 BPM | 0.55–0.70 | 4/4, 12/8 | Dominant 7ths, shuffle feel, guitar timbre |
| `pop` | 90–140 BPM | 0.50–0.55 | 4/4 dominant | Clean patterns, major-key bias, snare 2 & 4 |
| `electronic` | 110–160 BPM | 0.50–0.55 | 4/4 dominant | Four-on-floor, synth layers, heavy FX |
| `latin` | 90–140 BPM | 0.50–0.60 | 4/4, 3/4, 6/8 | Clave syncopation, percussion tags |
| `reggae` | 60–90 BPM | 0.50–0.58 | 4/4 dominant | One-drop (kick on beat 3), bass-heavy |
| `classical` | 50–160 BPM | 0.50–0.52 | 4/4, 3/4, 2/4, 6/8 | Wide dynamics, orchestral timbres |

### Usage

```bash
# Single genre
musicgen generate --seed 42 --genre jazz

# Genre composition (constraints merged)
musicgen generate --seed 42 --genre jazz --genre latin

# List all genres with descriptions
musicgen list-genres
```

### How genres constrain generation

- **Tempo** — hard bounds: drawn tempo is clamped to `[tempo_min, tempo_max]`
- **Swing** — hard bounds: drawn swing clamped to `[swing_min, swing_max]`
- **Time signature** — soft weights: `time_sig_weights` shifts registry draw probabilities
- **Key/scale** — soft weights: `scale_weights` shifts key selection probabilities
- **Chord type + inversions** — soft weights + optional hard filter via `chord_type_weights`, `inversion_weights`, `chord_type_hard_filter`
- **Drum patterns** — unioned from all active genre `patterns_*.txt` files + `genres/default/`
- **FX profile** — `fx_profile` multiplies effect probabilities (soft shift, full range still accessible)
- **Soundfonts** — `soundfont_tags` per layer replaces static tags when SoundfontManager is active

See [`genres/README.md`](genres/README.md) for the full `spec.json` format and instructions for writing custom genres.

## Musicality scoring and quality gate (v0.3)

The quality gate rejects samples that would contaminate a training distribution — not rank good music from very good music.

**Two-layer architecture:**

- **Layer 1 — symbolic (MIDI, pre-render, < 5 ms).** `check_midi_quality(midi_paths, key)` → `MIDIQualityResult`. Hard checks: empty layer, stuck pitch (> 80 % single note), extreme pitch range (> 36 semitones). Soft metrics on the melody layer: Krumhansl–Schmuckler key-profile correlation, scale adherence fraction, melodic step fraction, n-gram entropy, LZ compression ratio. Samples failing any hard check score 0.0 and are not rendered.

- **Layer 2 — audio integrity (post-render).** `get_musicality_score(filename, genre_spec)` → `(float, dict)`. Render-integrity penalty (clipping ratio, silence ratio) multiplied against weighted musical analysis (tempo stability/clarity, harmony KS correlation, rhythm regularity/strength, noise/spectral metrics). Weights: tempo 30 %, harmony 30 %, rhythm 25 %, noise 15 %.

**Quality-gate loop:**

```python
result = generate(Config(
    global_seed=42,
    sample_index=0,
    dataset_root="./dataset",
    min_musicality_score=0.6,   # reject below 0.6; 0.0 = disabled
    max_attempts=3,             # re-roll up to 3 times with distinct seeds
))
print(result.attempt)          # which attempt was accepted (1, 2, or 3)
```

**Calibration** — derive an empirical threshold without human labels:

```bash
musicgen calibrate          # runs run_midi_calibration + prints suggested threshold
```

```python
from musicgen.calibrate import run_midi_calibration, save_calibration

result = run_midi_calibration(n_good=100, n_bad=100, seed=42)
print(result.suggested_threshold, result.separation_ok)
save_calibration(result, "calibration.json")
```

See [`docs/musicality-scoring.md`](docs/musicality-scoring.md) for the theoretical basis, metric derivations, and literature references.

## Optional integrations (v0.2)

All three integrations are **opt-in** — zero new hard dependencies in `pyproject.toml`. Each external package is lazy-imported inside the function body; a clear `ImportError` with an install hint is raised if the package is absent.

### Integration 1 — SoundfontManager-backed soundfont selection

**Package:** [`dobidu/soundfont_manager`](https://github.com/dobidu/soundfont_manager)

**Install:**
```bash
pip install git+https://github.com/dobidu/soundfont_manager
```

**Purpose:** Replace blind `rng.choice(os.listdir(...))` with metadata-aware, tag-based selection from a SoundfontManager JSON database. Enables curated timbre selection ("pick a pad soundfont for the harmony layer") while preserving the determinism contract.

**Activation:** set `soundfont_manager_db` in `Config` (or env var `MUSICGEN_SOUNDFONT_MANAGER_DB`):

```python
from musicgen import generate, Config

result = generate(Config(
    global_seed=42,
    sample_index=0,
    dataset_root="./dataset",
    soundfont_manager_db="/path/to/soundfonts.json",
    soundfont_manager_sf_dir="/path/to/sf2/files",  # optional, for relative paths in db
))
```

Or via environment:
```bash
export MUSICGEN_SOUNDFONT_MANAGER_DB=/path/to/soundfonts.json
export MUSICGEN_SOUNDFONT_MANAGER_SF_DIR=/path/to/sf2/files
musicgen generate --count 32 --seed 1 --out ./dataset
```

**Layer → tag mapping:**

| Layer | Tags queried |
|---|---|
| `beat` | `["drums", "percussion"]` |
| `melody` | `["melody", "lead", "piano", "strings"]` |
| `harmony` | `["harmony", "chords", "pads", "pad"]` |
| `bassline` | `["bass"]` |

**Determinism:** Candidates are sorted by `sf.path` before `rng.choice()` — SM database insertion order has no effect on reproducibility.

**Fallback:** `ImportError`, empty tag result for any layer, or any exception → falls back to `sorted(os.listdir(sf/<layer>/))` + `rng.choice()`. Fallback is logged.

---

### Integration 2 — MIDI indexing into midi_file_manager

**Package:** [`dobidu/midi_file_manager`](https://github.com/dobidu/midi_file_manager)

**Install:**
```bash
pip install git+https://github.com/dobidu/midi_file_manager
```

**Purpose:** Index all generated MIDI files into a `MidiManager` database with ground-truth musicgen metadata (`tempo_bpm`, `key`, `time_signature`, `split`, `musicality_score`). Enables downstream ML pipelines to query by musical parameters.

**CLI:**
```bash
musicgen index-midi --dataset ./dataset --out ./midi_db.json
musicgen index-midi --dataset ./dataset --out ./midi_db.json --csv ./midi_db.csv
```

**Library:**
```python
from musicgen.midi_indexer import index_midi_dataset

count = index_midi_dataset(
    dataset_root="./dataset",
    out_db="./midi_db.json",
    midi_dir=None,      # optional base dir for relative paths
    export_csv=None,    # optional CSV export path
)
print(f"Indexed {count} MIDI files")
```

**What gets indexed:** `midi/<layer>.mid` for each of `beat`, `melody`, `harmony`, `bassline` across every complete sample directory. Ground-truth fields (`bpm`, `key`, `time_signature`) come from `sample.json` — no re-extraction. Category and tags per layer:

| Layer | MidiCategory | Tags |
|---|---|---|
| `beat` | `"drums"` | `["musicgen", "beat", <split>]` |
| `melody` | `"melody"` | `["musicgen", "melody", <split>]` |
| `harmony` | `"harmony"` | `["musicgen", "harmony", <split>]` |
| `bassline` | `"bass"` | `["musicgen", "bassline", <split>]` |

---

### Integration 3 — Audio indexing into audio_sample_manager

**Package:** [`dobidu/audio_sample_manager`](https://github.com/dobidu/audio_sample_manager)

**Install:**
```bash
pip install git+https://github.com/dobidu/audio_sample_manager
```

**Purpose:** Index generated WAV stems into a `SampleManager` database alongside external audio libraries (drum packs, synth loops, etc.). Enables unified cross-library queries — e.g., "all bass stems at 90 BPM in A minor" returns both musicgen-generated and externally-sourced samples.

**CLI:**
```bash
musicgen index-audio --dataset ./dataset --out ./audio_db.json
musicgen index-audio --dataset ./dataset --out ./audio_db.json --csv ./audio_db.csv
```

**Library:**
```python
from musicgen.audio_indexer import index_audio_dataset

count = index_audio_dataset(
    dataset_root="./dataset",
    out_db="./audio_db.json",
    samples_dir=None,   # optional base dir for relative paths
    export_csv=None,    # optional CSV export path
)
print(f"Indexed {count} WAV stems")
```

**What gets indexed:** `stems/<layer>.wav` for each of `beat`, `melody`, `harmony`, `bassline` per complete sample. `mix.wav` is not indexed — `SampleCategory` has no `full_song` value; the mix is better represented by its constituent stems.

Ground-truth fields (`bpm`, `key`, `time_signature`, `scale`) come from `sample.json`. Timbre/spectral features come from librosa analysis (not available in `sample.json`). `is_loop=False` — musicgen stems are full concatenated songs, not fixed-length loops.

| Layer | SampleCategory | Tags |
|---|---|---|
| `beat` | `"beat"` | `["musicgen", "beat", <split>]` |
| `melody` | `"melody"` | `["musicgen", "melody", <split>]` |
| `harmony` | `"harmony"` | `["musicgen", "harmony", <split>]` |
| `bassline` | `"bass"` | `["musicgen", "bassline", <split>]` |

---

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

`manifest.jsonl` accumulates one JSON object per sample with `sample_index`, `seed`, `status` (`ok`/`failed`), `split`, `path`, `musicality_score`, `duration_seconds`, `attempt`, and `wrote_at`.

### `sample.json` schema (R-P4)

Every sample carries:

- **Identity:** `seed`, `musicgen_version`, `fluidsynth_version`
- **Musical params:** `key`, `mode`, `tempo_bpm`, `time_signature`, `swing`, `duration_seconds`
- **Structure:** `song_arrangement` (list of `{part, start_seconds, end_seconds}`)
- **Per-part:** `chord_progression`, `active_layers`, `soundfonts`, `fx_params` (with sampled effect parameters), `time_signatures_per_part`, `measures_per_part`
- **Annotations:** `beat_times`, `downbeat_times` (seconds, swing-aware via MIDI-tick extraction)
- **Quality:** `musicality_score` (Layer 2 audio analyzer output: tempo / harmony / rhythm / noise, with render-integrity penalty); Layer 1 MIDI quality result available via `check_midi_quality` before render
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
pytest -m "not slow"      # Fast suite (default CI) — 1047 tests in ~4s
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
| 6 | Productize II — FluidSynth calibration, batch generation, CLI, resumability | ✓ COMPLETE | 6/6 |
| 7 | Ship v0.1 — docs, polish, regression suite | ✓ COMPLETE | — |
| v0.2-int | Sibling ecosystem integrations (SoundfontManager, MIDI indexer, audio indexer) | ✓ COMPLETE | — |
| v0.2 | Genre system — GenreSpec engine, 8 built-in genres, chord vocab, CLI, notebook | ✓ COMPLETE | 8/8 |
| v0.3 | Higher-order Markov — 2nd-order chords + melody, two-layer quality gate, calibration | ✓ COMPLETE | 3/3 |

### Phases delivered

- **Phase 1 — Stabilize I.** Importability fix (`if __name__ == '__main__'`), arrangement-reroll bug, pydub gain/pan no-op fix, narrow exception handlers, dead-code removal, first pytest skeleton.
- **Phase 2 — Stabilize II.** `config.py` owns all paths (CLI > env > defaults precedence). `timesig.py` registry consolidates 7 time signatures (one source of truth). All `print()` → structured `logging`.
- **Phase 3 — Package skeleton.** `src/musicgen/` installable package via `hatchling` + `pyproject.toml`. Sampler + generators extracted with injected `rng: random.Random` parameters. AST static guard against bare `random.*`. `enhanced_duration_validator.py` → `src/musicgen/duration_validator.py`. `requirements.txt` deleted.
- **Phase 4 — Audio-side extraction.** `src/musicgen/{renderer, mixer, annotator, beats}.py`. `ThreadPoolExecutor` parallel stem rendering. Pure-function annotator producing the R-P4 schema. Swing-aware MIDI-tick beat derivation (replaces the old grid-based `beat_anotator.py`, which is deleted). `mix_and_save` is gone; `music_gen.py` collapses to a thin orchestrator.
- **Phase 5 — Productize I.** `src/musicgen/{seeds, writer, manifest, api, musicality}.py`. Per-sample directory layout with atomic sentinel ordering. Sum-of-stems integrity check (int32 accumulator, ε=1e-3). Five-domain RNG hierarchy. `manifest.jsonl` with threading.Lock + `os.fsync`. Deterministic train/valid/test split. `musicgen.generate(Config) → SampleResult` library entry point. `music_gen.py` collapsed from 199 → 60 lines.

- **Phase 6 — Productize II.** `musicgen.generate_batch(config)` via `ProcessPoolExecutor`. FluidSynth pre-roll calibration (caches in `.musicgen/fluidsynth_preroll.json`, applied to beat-time annotations). Full `typer` CLI: `musicgen generate --count N --out DIR --seed S [--workers W] [--output-mode MODE]` and `musicgen clean --failed`. `--output-mode` flag (`full` / `mix-only` / `stems-only` / `midi-only`). Resumability via sentinel check. Failure isolation (one bad sample doesn't kill a 10k batch). Structured JSON progress events on stderr.
- **Phase 7 — Ship v0.1.** README refresh, GitHub Actions CI (87% coverage, determinism regression), `v0.1.0` tag.
- **v0.2 integrations — Sibling ecosystem.** Three opt-in integrations with zero new hard deps. (1) `soundfont_manager`: tag-based soundfont selection via `Config.soundfont_manager_db`; determinism preserved via `sf.path`-sorted candidates. (2) `midi_file_manager`: `musicgen index-midi` command indexes MIDI files into a `MidiManager` db with ground-truth enrichment. (3) `audio_sample_manager`: `musicgen index-audio` command indexes WAV stems into a `SampleManager` db for cross-library queries. All packages lazy-imported; `ImportError` falls back gracefully.
- **v0.2 — Genre system.** `GenreSpec` dataclass with hard bounds (tempo, swing), soft weight dicts (time-sig, scale, chord type, inversion, layer probs), FX profile multipliers, and per-layer soundfont tag overrides. `merge_genres` composition algebra (range intersection, weighted-average soft dicts, union of drum pools). 8 built-in genres (`jazz`, `hip-hop`, `blues`, `pop`, `electronic`, `latin`, `reggae`, `classical`), each with `spec.json` + time-sig drum pattern files. Extended chord vocabulary (maj7, m7, dom7, dim7, m7b5, sus2, sus4, add9, aug + all inversions). CLI: `--genre jazz`, `--genre jazz --genre latin`, `musicgen list-genres`. `notebooks/musicgen_demo.ipynb` 12-section demo notebook. Tag `v0.2.0`.
- **v0.3 — Higher-order Markov + quality gate.** 2nd-order chord transition matrices per genre (`genres/*/chord_transitions.json`), configurable-order melody Markov over scale-degree intervals (`generators/melody.py`). Two-layer musicality redesign: Layer 1 MIDI pre-filter (`check_midi_quality` — hard checks for empty/stuck/extreme-range + soft symbolic metrics: Krumhansl–Schmuckler key correlation, scale adherence, melodic step fraction, n-gram entropy, LZ compression ratio); Layer 2 audio integrity (`_render_integrity` — clipping, DC offset, silence, crest factor, applied as penalty to `MusicalityAnalyzer` output). Quality-gate regeneration loop: `Config.min_musicality_score` + `Config.max_attempts` — `generate()` re-rolls up to `max_attempts` times with distinct seeds; `SampleResult.attempt` and `manifest attempt` field track winning attempt. Calibration harness (`musicgen.calibrate.run_midi_calibration`, `suggest_threshold`) derives empirical thresholds from reference-good vs adversarial MIDI sets without human labels. See [`docs/musicality-scoring.md`](docs/musicality-scoring.md) for theoretical basis and implementation details. 1197 fast tests. Tag `v0.3.0`.

## Architecture (post-Phase 5)

```
src/musicgen/
├── __init__.py          # public exports: generate, generate_batch, Config, SampleResult, BatchResult, __version__
├── api.py               # generate(Config) — composition root; resolve_genre_spec
├── batch.py             # generate_batch(Config) → BatchResult via ProcessPoolExecutor
├── cli.py               # typer app: generate / clean / calibrate / index-midi / index-audio / list-genres
├── calibrate.py         # FluidSynth pre-roll offset measurement + .musicgen/ cache
├── config.py (root)     # Config dataclass with CLI > env > defaults precedence
├── seeds.py             # derive_sample_seed, make_rngs, save_random_state, assign_split
├── genre.py             # GenreSpec dataclass, load_genre, merge_genres, resolve_genres
├── sampler.py           # SongParams + key/tempo/time-sig/swing/measures/arrangement; genre-constrained draws
├── generators/
│   ├── chord.py         # generate_chord_progression; extended types (maj7/m7/dom7/dim7/sus/add9/aug) + inversions
│   ├── melody.py        # generate_melody (Markov-style over chord progressions)
│   ├── bassline.py      # generate_bassline (keyed to chords + melody)
│   └── beat.py          # generate_beat (drum patterns + swing offset); genre pattern union
├── renderer.py          # FluidSynth wrapper, ThreadPoolExecutor stem rendering; genre + SM-backed soundfont selection
├── mixer.py             # FX (pedalboard), pydub overlay, layer mask, part concat; genre FX profile
├── beats.py             # MIDI-tick beat + downbeat extraction (mido), swing-aware
├── annotator.py         # pure-function R-P4 schema assembler
├── musicality.py        # two-layer scorer: check_midi_quality (Layer 1, pre-render) + MusicalityAnalyzer (Layer 2, audio)
├── writer.py            # atomic per-sample dir, sum-of-stems assertion, MIDI/stem concat, output_mode routing
├── manifest.py          # ManifestWriter (append-under-lock, JSONL)
├── midi_indexer.py      # index_midi_dataset: indexes MIDI files into MidiManager db (opt-in)
├── audio_indexer.py     # index_audio_dataset: indexes WAV stems into SampleManager db (opt-in)
└── duration_validator.py
```

Pipeline flow inside `generate(config)`:

```
resolve_genre_spec → sampler (genre-constrained draws)
       → generators (chord/melody/bassline/beat, genre pattern pool)
       → check_midi_quality (Layer 1: hard + soft symbolic checks, < 5 ms)
       → [re-roll up to max_attempts if score < min_musicality_score]
       → renderer (FluidSynth, parallel stems, genre soundfont tags)
       → mixer (FX + overlay + concat, genre FX profile)
       → beats (MIDI-tick extraction)
       → get_musicality_score (Layer 2: audio integrity + musical analysis)
       → annotator (R-P4 dict + pre-roll offset)
       → writer (atomic sample dir + sum-of-stems + output_mode routing)
       → manifest (JSONL append, includes attempt field)
       → SampleResult
```

`generate_batch` wraps `generate` in a `ProcessPoolExecutor` (spawn context), emits JSON progress events on `stderr`, and returns `BatchResult`.

## Roadmap — upcoming

- **v0.4 — ML-assisted generators:** ML-guided chord/melody generation trained on real MIDI corpora, model-guided regeneration. Requires a reference dataset generated with the v0.3 stack as training corpus.
- **Public release:** soundfont license audit + CC0/MIT replacement, HuggingFace Datasets / WebDataset exporters, sharded directory layout for 100k+ samples.

Not planned: cloud / distributed generation, web UI, HTTP API.

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
