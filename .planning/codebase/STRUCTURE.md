# Structure

## Directory layout

```
musicgen/                               # repo root
├── config.py                           # Config dataclass (CLI > env > defaults)
├── timesig.py                          # TimeSignatureRegistry
├── music_gen.py                        # 59-line thin wrapper / smoke-test entry point
│
├── pyproject.toml                      # hatchling build; version 0.1.0; dev extras
├── README.md
├── LICENSE
├── .gitignore
│
├── src/
│   └── musicgen/                       # installable package
│       ├── __init__.py                 # public API re-exports
│       ├── __main__.py                 # python -m musicgen support
│       ├── api.py                      # generate(config) -> SampleResult
│       ├── batch.py                    # generate_batch(config) -> BatchResult
│       ├── cli.py                      # typer app: generate / clean / calibrate
│       ├── sampler.py                  # song-level parameter sampling; SongParams
│       ├── seeds.py                    # RNG hierarchy; derive_sample_seed; make_rngs
│       ├── renderer.py                 # FluidSynth wrapper; pick_soundfonts; RenderResult
│       ├── mixer.py                    # FX boards; layer mask; mix_part; concat_parts
│       ├── beats.py                    # MIDI-tick beat/downbeat time extraction
│       ├── annotator.py                # assembles sample.json annotation dict
│       ├── writer.py                   # atomic per-sample layout; sum-of-stems assertion
│       ├── manifest.py                 # JSONL manifest writer; is_sample_complete
│       ├── calibrate.py                # FluidSynth pre-roll measurement and cache
│       ├── musicality.py               # librosa-based audio quality scoring
│       ├── duration_validator.py       # time-signature-aware note duration validation
│       └── generators/
│           ├── __init__.py
│           ├── chord.py                # chord progression MIDI generation
│           ├── melody.py               # melodic line generation
│           ├── bassline.py             # bass line generation
│           └── beat.py                 # drum pattern generation with swing
│
├── tests/
│   ├── conftest.py                     # --regen-goldens flag; pytest hooks
│   ├── fixtures/
│   │   └── determinism/                # SHA-256 golden files for determinism regression
│   ├── test_api.py                     # generate() fast + slow tests
│   ├── test_batch.py                   # generate_batch() orchestration tests
│   ├── test_cli.py                     # typer CLI command tests
│   ├── test_config.py                  # Config dataclass and load() tests
│   ├── test_sampler.py                 # sampler functions; bare-random guard
│   ├── test_seeds.py                   # RNG hierarchy; derive/make/assign/save
│   ├── test_renderer.py                # render_stems; pick_soundfonts; FLUIDSYNTH_VERSION
│   ├── test_mixer.py                   # FX boards; mix_part; layer mask; concat_parts
│   ├── test_beats.py                   # beat/downbeat time extraction
│   ├── test_annotator.py               # annotation dict assembly
│   ├── test_writer.py                  # atomic write; MIDI concat; sum-of-stems
│   ├── test_manifest.py                # ManifestWriter; is_sample_complete
│   ├── test_calibrate.py               # pre-roll measurement and cache
│   ├── test_duration_validator.py      # DurationValidator class
│   ├── test_time_signature.py          # time-signature logic (validate_measures alias)
│   ├── test_timesig_registry.py        # TimeSignatureRegistry
│   ├── test_split.py                   # assign_split; train/val/test distribution
│   ├── test_output_mode.py             # full / mix-only / stems-only / midi-only paths
│   ├── test_determinism_golden.py      # SHA-256 goldens + same-process stability
│   ├── test_no_bare_random_in_package.py  # AST scan: zero bare random.* in src/musicgen/
│   ├── test_music21_isolation.py       # music21 import isolation checks
│   ├── test_music_gen_logging.py       # smoke wrapper logging
│   ├── test_integration_full_generation.py  # end-to-end slow integration test
│   ├── test_integration_batch.py       # batch integration with real spawn context
│   └── test_generators/
│       ├── test_chord.py
│       ├── test_melody.py
│       ├── test_bassline.py
│       ├── test_beat.py
│       └── test_no_bare_random.py      # AST scan: bare-random guard for generators/
│
├── .github/
│   └── workflows/
│       └── ci.yml                      # GitHub Actions: Python 3.10 + 3.12 matrix
│
├── .musicgen/
│   └── fluidsynth_preroll.json         # FluidSynth pre-roll cache (written by calibrate)
│
├── chord_patterns.txt                  # chord progression patterns
├── beats_annotations.txt
├── beat_roll_patterns_24.txt           # drum patterns per time signature
├── beat_roll_patterns_34.txt
├── beat_roll_patterns_44.txt
├── beat_roll_patterns_54.txt
├── beat_roll_patterns_68.txt
├── beat_roll_patterns_78.txt
├── beat_roll_patterns_128.txt
│
├── song_structures.json
├── inst_probabilities.json
├── levels.json
├── soundfonts.json
├── beat_fx.json
├── melody_fx.json
├── harmony_fx.json
├── bassline_fx.json
│
└── sf/                                 # soundfont assets, organized by layer
    ├── beat/
    ├── melody/
    ├── harmony/
    └── bassline/
```

## Key locations

| Looking for... | Look in |
|---|---|
| Public library API | `src/musicgen/__init__.py` — `generate`, `generate_batch`, `Config`, `SampleResult`, `BatchResult`, `__version__` |
| Single-sample orchestrator | `src/musicgen/api.py` — `generate(config)` and `_run_pipeline` |
| Batch generation | `src/musicgen/batch.py` — `generate_batch(config)` |
| CLI commands | `src/musicgen/cli.py` — `generate`, `clean`, `calibrate` commands |
| Config dataclass | `config.py` (repo root) — `Config.load(cli_overrides)` |
| Song parameter sampling | `src/musicgen/sampler.py` — `generate_random_key`, `generate_random_tempo`, etc. |
| RNG hierarchy | `src/musicgen/seeds.py` — `derive_sample_seed`, `make_rngs`, `assign_split`, `save_random_state` |
| FluidSynth rendering | `src/musicgen/renderer.py` — `render_stems`, `pick_soundfonts`, `FLUIDSYNTH_VERSION` |
| FX and mixing | `src/musicgen/mixer.py` — `build_fx_boards`, `compute_layer_mask`, `mix_part`, `concat_parts` |
| Beat time extraction | `src/musicgen/beats.py` — `extract_beat_times`, `extract_downbeat_times` |
| Annotation assembly | `src/musicgen/annotator.py` — `annotate(...)` |
| Atomic write + sum-of-stems | `src/musicgen/writer.py` — `write_sample(...)` |
| Manifest (JSONL) | `src/musicgen/manifest.py` — `ManifestWriter`, `is_sample_complete` |
| Pre-roll calibration | `src/musicgen/calibrate.py` — `load_preroll`, `measure_and_save_preroll` |
| Audio quality metrics | `src/musicgen/musicality.py` — `get_musicality_score` |
| Note duration validation | `src/musicgen/duration_validator.py` — `DurationValidator` |
| Time signature registry | `timesig.py` (repo root) — `TimeSignatureRegistry` |
| Chord generation | `src/musicgen/generators/chord.py` — `generate_chord_progression` |
| Melody generation | `src/musicgen/generators/melody.py` — `generate_melody` |
| Bass line generation | `src/musicgen/generators/bassline.py` — `generate_bassline` |
| Beat generation | `src/musicgen/generators/beat.py` — `generate_beat` |
| Smoke wrapper | `music_gen.py` (repo root) — calls `musicgen.generate(Config(global_seed=1, sample_index=0))` |

## Output layout (runtime)

Generated samples are written to `dataset_root/<idx:06d>/` (six-digit zero-padded index). Inside each sample directory:

- `sample.json` — completion sentinel; full annotation including beat times, chord progressions, split, musicality score, FluidSynth version.
- `mix.wav` — final concatenated mix (present for `output_mode` in `{"full", "mix-only"}`).
- `stems/<layer>.wav` — per-layer stem WAV concatenated across all arrangement parts (`{"full", "stems-only"}`).
- `midi/<layer>.mid` — per-layer MIDI concatenated across all arrangement parts (`{"full", "midi-only"}`).

At `dataset_root/manifest.jsonl`, each line is a JSON object with `sample_index`, `seed`, `sample_seed`, `status`, `split`, `path`, `musicality_score`, `duration_seconds`, and `wrote_at`. Failed samples append a line with `"status": "failed"` and an `"error"` field (capped at 2 KB).

## Naming conventions

- **Python files:** `snake_case.py` throughout `src/musicgen/` and `tests/`.
- **Package modules:** functional nouns (`sampler`, `renderer`, `mixer`, `writer`, `annotator`).
- **Functions:** `snake_case`. Action verbs (`generate_*`, `render_*`, `mix_*`, `write_*`, `compute_*`, `load_*`).
- **Dataclasses:** `PascalCase` — `Config`, `SampleResult`, `BatchResult`, `SongParams`, `RenderResult`, `MixResult`.
- **Constants:** `UPPER_SNAKE` for module-level constants (`FLUIDSYNTH_VERSION`, `RNG_PARAMS`, etc.).
- **Config files:** lowercase JSON/TXT at repo root, grouped by purpose (`*_fx.json`, `beat_roll_patterns_<sig>.txt`).
- **Soundfont directories:** `sf/<layer>/` where `<layer>` is one of `beat`, `melody`, `harmony`, `bassline`.
- **Time signature strings:** `"4/4"`, `"6/8"`, etc. Pattern files use the concatenated form without slash: `44`, `68`, `128`.

## Module boundaries

The `src/musicgen/` package owns the generation pipeline and all its stages. `config.py` and `timesig.py` live at the repo root because they serve both the package and the `music_gen.py` thin wrapper. Both are added to `sys.path` at import time by `musicgen/__init__.py` and `musicgen/cli.py` so they remain importable after `pip install -e .`.

The `generators/` subpackage contains the four MIDI-authoring modules. Each generator takes an explicit `rng: random.Random` parameter and returns file paths; none has side effects beyond writing MIDI files to the working directory.

No cross-stage imports flow backward: `writer` and `annotator` do not import from `renderer` or `mixer`; `batch` imports only from `api` and `manifest`; `api` orchestrates all other modules but nothing in those modules imports back from `api`.
