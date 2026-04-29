# Architecture Research

**Scope:** Proposed module layout, component boundaries, and build order for turning `music_gen.py` (1161-line monolith) into a Python library + CLI for reproducible dataset generation.

**Confidence:** HIGH — derived from direct codebase inspection plus well-established Python packaging and parallelism patterns.

## Key findings

- The current logic is right; the structure is wrong. `mix_and_save` (`music_gen.py:758-901`) conflates soundfont selection, FX chain construction, FluidSynth rendering, volume/panning, layer inclusion, and part concatenation into a single 143-line function. The extraction split is: **Sampler → Generators → Renderer → Mixer → Annotator → Writer → Manifest**, in dependency order.
- **Time-signature logic must be consolidated first** — five scattered locations today (`verify_pattern_for_time_signature`, `verify_beat_pattern`, `calculate_measures_for_time_signature`, `generate_random_time_signature`, `DurationValidator`). Every generator touches it. Extracting generators before consolidating creates N copies of a bug.
- **Seeding is completely absent.** Pattern: hierarchical deterministic seeding — `sample_seed[i] = sha256(global_seed:i)`, then named `random.Random` instances (`param_rng`, `generator_rng`, `soundfont_rng`, `fx_rng`, `mix_rng`) passed explicitly into each stage. Resumed runs produce the same sample N regardless of what other samples were generated.
- **Intra-sample parallelism uses threads, not processes.** FluidSynth is an external subprocess; `ThreadPoolExecutor(max_workers=4)` is cheaper than another fork level.
- **`manifest.jsonl` doubles as resume checkpoint.** Per-sample directories named by index (`0042/`), not timestamp. A partially-written sample is detected by absence of `sample.json` (the last file `Writer` writes).
- **CLI is a thin shell.** `src/musicgen/cli.py` + `__main__.py` do argument parsing and `Config` construction only. No business logic. The library API (`from musicgen import generate`) is usable independently.

## Proposed module layout

```
src/musicgen/
    __init__.py         # public API: generate(config) / generate_batch(config)
    __main__.py         # python -m musicgen entry point
    cli.py              # typer app; builds Config; calls library API
    config.py           # Config dataclass; all path resolution (replaces hardcoded literals)
    sampler.py          # SongParams dataclass; generate_random_* ; generate_song_arrangement
    timesig.py          # TimeSignatureRegistry (collapses 5 scattered time-sig locations)
    generators/
        __init__.py
        chord.py        # generate_chord_progression (extracted)
        melody.py       # generate_melody (extracted)
        bassline.py     # generate_bassline (extracted)
        beat.py         # generate_beat (extracted)
    renderer.py         # FluidSynth wrapper + ThreadPoolExecutor parallel stem rendering
    mixer.py            # pedalboard FX + pydub overlay + part concatenation
    annotator.py        # pure function: (SongParams + render + mix results) → annotation dict
    beats.py            # beat/downbeat timestamp derivation (replaces beat_anotator.py)
    writer.py           # per-sample output directory + file naming + lifecycle
    manifest.py         # manifest.jsonl append (locked) + resume detection
    batch.py            # ProcessPoolExecutor outer loop + resume dispatch
    scoring.py          # thin wrapper re-export of MusicalityAnalyzer

tests/                  # pytest suite
pyproject.toml
README.md
LICENSE
```

Existing top-level config files (`*_fx.json`, `inst_probabilities.json`, `levels.json`, `song_structures.json`, `chord_patterns.txt`, `beat_roll_patterns_*.txt`) and the `sf/` soundfont directory stay where they are; paths are resolved through `config.py`.

## Data flow

```
seed (int)
    │
    ▼
Sampler ─────────────────────────────────────► SongParams
                                                    │
                                             Generators (chord/melody/bass/beat)
                                                    │
                                               .mid files
                                                    │
                                             Renderer (FluidSynth × 4, ThreadPool)
                                                    │
                                           stem .wav files (persisted)
                                                    │
                                             Mixer (FX + overlay + concat parts)
                                                    │
                                           mix.wav + layer-inclusion dict
                                                    │
    ┌───────────────────────────────────────────────┤
    │ (SongParams + soundfonts + FX params           │
    │  + layer inclusion + musicality score          │
    │  + beat timestamps + file paths)               │
    │                                                ▼
    │                                          Annotator (pure fn)
    │                                                │
    │                                          annotation dict
    │                                                │
    └────────────────────────────────────────────────► Writer
                                                    │
                                              SampleResult (all paths)
                                                    │
                                                 Manifest
                                                    │
                                         manifest.jsonl (append, locked)
```

## Per-sample output layout

```
<dataset-root>/
├── manifest.jsonl
└── <sample-id>/               # zero-padded index, e.g. 00042/
    ├── sample.json            # annotation — last file written (resume sentinel)
    ├── mix.wav
    ├── stems/
    │   ├── beat.wav
    │   ├── melody.wav
    │   ├── harmony.wav
    │   └── bassline.wav
    └── midi/
        ├── beat.mid
        ├── melody.mid
        ├── harmony.mid
        └── bassline.mid
```

- Directory name is **index-based**, not timestamp. Enables unambiguous resume.
- `sample.json` is always written last. Presence = sample complete.
- Absent layers (probabilistic inclusion) still write silent stem files with correct duration so sum-of-stems equals mix.

## Seed / RNG propagation

```python
def derive_sample_seed(global_seed: int, sample_index: int) -> int:
    raw = hashlib.sha256(f"{global_seed}:{sample_index}".encode()).digest()
    return int.from_bytes(raw[:8], "big")

def make_rngs(sample_seed: int) -> dict[str, random.Random]:
    return {
        "params":     random.Random(sample_seed ^ 0x01),
        "generators": random.Random(sample_seed ^ 0x02),
        "soundfonts": random.Random(sample_seed ^ 0x03),
        "fx":         random.Random(sample_seed ^ 0x04),
        "mix":        random.Random(sample_seed ^ 0x05),
    }
```

**Rules:**

1. No bare `random.random()` / `random.choice()` anywhere in the pipeline — every call goes through a named `random.Random` instance passed as a parameter.
2. Global `random` state is never touched. If a dependency (e.g. `music21`) uses global `random`, wrap the call in `with save_random_state():` to prevent leakage.
3. Workers seed their `random.Random` instances from their assigned sample seed on entry — never inherit from the parent.
4. FluidSynth version is recorded in every `sample.json`. Bit-reproducibility of audio is best-effort; MIDI + annotations are always bit-reproducible.

## Build order

Build order respects stabilize → productize → extend priority. Each group must finish before the next starts.

### Stabilize (unblock importability + testability)

1. Wrap bottom-of-file execution in `if __name__ == '__main__':` (`music_gen.py:1158-1161`).
2. Replace `from music21 import *` with explicit imports (`music_gen.py:2`).
3. Fix two confirmed bugs (see PITFALLS.md):
   - Regenerated arrangement inside `mix_and_save` (line 760).
   - Broken `pydub` `.volume=` assignment and discarded `.pan()` return (lines 845-852).
4. Extract `config.py` — centralize hardcoded `sf/*`, `*_fx.json`, `inst_probabilities.json`, `levels.json`, `song_structures.json`, `chord_patterns.txt` paths.
5. Extract `timesig.py` registry — consolidate 5 scattered locations. **Do this before extracting generators.**
6. Replace 32+ `print()` calls with `logging` (using already-installed `python-json-logger`).
7. Replace bare `except:` blocks in `musicality_score.py:66, 94, 173, 205, 239` with narrow exceptions + `logger.exception`.
8. Remove dead imports (`glob`, `Pool`, `cpu_count`, `time`) and dead variables.

### Extract + Test

1. `sampler.py` — pure sampling functions, no I/O. Easiest to test first.
2. `generators/{chord,melody,bassline,beat}.py` — extract with injected `rng` parameters.
3. `renderer.py` — FluidSynth wrapper, `ThreadPoolExecutor` parallel stems.
4. `mixer.py` — FX + mixing extracted from `mix_and_save`.
5. `annotator.py` — pure function over all stage outputs.
6. `beats.py` — swing-aware beat/downbeat timestamp derivation (replaces `beat_anotator.py` straight-grid logic).
7. Pytest suite: unit tests for validators, samplers, generators, annotator; integration tests (marked `@pytest.mark.slow`) for renderer + mixer + full generation.

### Productize

1. `writer.py` — per-sample output directory, fix UUID truncation bug at `music_gen.py:1143`, silent-stem writer for absent layers.
2. `manifest.py` — `manifest.jsonl` append + resume detection + file locking (`multiprocessing.Manager().Lock()`).
3. `batch.py` — `ProcessPoolExecutor` outer loop, resume logic, progress tracking.
4. `cli.py` + `__main__.py` — `typer` app, `pyproject.toml` entry point.
5. Seeding wired throughout — replace all bare `random` calls with injected `random.Random` parameters.
6. FluidSynth version logging + pre-roll measurement.
7. Stem-sum-to-mix assertion as a post-generation check.

### Extend

Only after Productize. All new work lives in `generators/` and `sf/` — the clean boundaries mean it doesn't touch rendering, mixing, or orchestration.

## Refactors required before Productizing

These cannot be skipped; attempting Productize on the monolith creates tangled bugs:

1. **Importability guard** — cannot import for testing or library use without it.
2. **Config centralization** — cannot wire `Config` through the pipeline if paths are hardcoded.
3. **Time-sig registry** — cannot extract generators correctly while time-sig logic is in five locations.
4. **`mix_and_save` decomposition** — cannot add parallel rendering or stem persistence without separating Renderer from Mixer from orchestration.
5. **Seed injection** — cannot add deterministic seeding without replacing global `random` calls.

## Open questions

1. **FluidSynth bit-reproducibility** — empirical verification on target platform; may require binary pinning.
2. **`music21` global random usage** — audit whether `music21` touches global `random` state internally; if yes, `save_random_state()` wrapper needed.
3. **Output mode flag** — `keep-stems / mix-only / stems-only / midi-only` affects `Writer` and `Mixer` design. Decide during Productize.
4. **Manifest locking portability** — `multiprocessing.Manager().Lock()` recommended over `fcntl.flock` for Windows support, even though Windows isn't a current target.
