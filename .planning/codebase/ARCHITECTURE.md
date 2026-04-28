# Architecture

## Pattern

**Installable package / layered pipeline.** The project is an installable Python package (`src/musicgen/`) with a clean separation between library API, CLI, and internal pipeline stages. Each stage is its own module with explicit interfaces. Config lives in `config.py` at the repo root as a dataclass with three-layer precedence: CLI arguments override environment variables, which override hardcoded defaults.

The design is **stochastic-procedural**: at every stage, decisions are sampled from domain-specific `random.Random` instances derived from a global seed. No bare `random.*` calls exist anywhere in `src/musicgen/`; all draws go through injected RNG objects. This is enforced by a static AST scan test that runs on every CI push.

## Pipeline stages

The end-to-end run is driven by `generate(config) -> SampleResult` in `src/musicgen/api.py`. For each sample, `_run_pipeline` executes these stages in order:

1. **Seed derivation** — `seeds.derive_sample_seed(global_seed, sample_index)` produces a per-sample seed via SHA-256 of `"<global_seed>:<sample_index>"`. `seeds.make_rngs(sample_seed)` returns five named `random.Random` instances keyed by domain: `params`, `generators`, `soundfonts`, `fx`, `mix`.

2. **Parameter sampling** — `sampler.generate_random_key`, `generate_random_tempo`, `generate_random_time_signature`, `generate_random_swing` draw from `rngs["params"]`. Weighted distributions encode real-world Spotify key and tempo statistics. `generate_song_measures` and `generate_song_arrangement` also draw from `rngs["params"]` and loop until `validate_measures_dict` passes.

3. **MIDI generation** — For each unique song part, four generators run using `rngs["generators"]`:
   - `generators.chord.generate_chord_progression` — picks a chord pattern from `chord_patterns.txt`, validates against the time signature via `TimeSignatureRegistry`.
   - `generators.melody.generate_melody` — Markov-style melodic line over the chord progression.
   - `generators.bassline.generate_bassline` — bass line keyed to chords and melody contour.
   - `generators.beat.generate_beat` — drum pattern from `beat_roll_patterns_<sig>.txt` with swing offset applied.
   Each writes a `.mid` file to a per-sample working directory under `tempfile.mkdtemp`.

4. **Soundfont selection** — `renderer.pick_soundfonts(cfg, rngs["soundfonts"])` selects one `.sf2` file per layer from `sf/<layer>/`. The directory listing is sorted before selection to ensure consistent behavior across filesystems.

5. **FX and layer mask** — `mixer.build_fx_boards(cfg, rngs["fx"])` constructs a `pedalboard.Pedalboard` per layer from the `*_fx.json` specs. `mixer.compute_layer_mask(song_unique_parts, inst_proba, rngs["mix"])` probabilistically decides which layers are active in each part, driven by `inst_probabilities.json`.

6. **Per-part render and mix** — For each part in the arrangement:
   - `renderer.render_stems` dispatches four FluidSynth calls in parallel via `ThreadPoolExecutor(max_workers=4)`. Threads are used because FluidSynth is a subprocess; the GIL is not held during the wait.
   - `mixer.mix_part` applies FX, volume, and panning to each stem, suppresses inactive layers with a silent fallback, and overlays the results into a single `AudioSegment`.
   - `beats.extract_beat_times` and `beats.extract_downbeat_times` compute MIDI-anchored timing arrays from the beat MIDI file.

7. **Concatenation** — `mixer.concat_parts` concatenates the per-part mix WAVs into the final `mix.wav` in the working directory.

8. **Musicality scoring** — `musicality.get_musicality_score(final_wav)` computes audio-domain quality metrics using `librosa`. This call is wrapped in `seeds.save_random_state()` to prevent the librosa internals from perturbing the global random state.

9. **Annotation** — `annotator.annotate` assembles the full `sample.json` annotation dict from all pipeline outputs, including beat times, chord progressions, FluidSynth version, split assignment, and pre-roll offset.

10. **Atomic write** — `writer.write_sample` transitions from the working directory to the final dataset layout. Its ordering invariant is:
    1. Concatenate per-part MIDIs per layer (absolute-tick walk).
    2. Concatenate per-part stem WAVs per layer.
    3. Copy `mix.wav`.
    4. Sum-of-stems assertion (int32 accumulator, epsilon configurable, default 1e-3 normalized).
    5. Rewrite annotation paths to per-sample-dir-relative form.
    6. Serialize `sample.json` (sorted keys, indent=2).
    7. Atomic rename from `sample.json.tmp` to `sample.json` — the completion sentinel.

11. **Manifest append** — `manifest.ManifestWriter.append` writes a line to `manifest.jsonl` in `dataset_root`, whether the sample succeeded or failed. The manifest line is appended in both outcomes; failures record an `"error"` field.

## Module boundaries

| Module | Responsibility |
|---|---|
| `config.py` (repo root) | `Config` dataclass; three-layer precedence; soundfont pool report |
| `musicgen/__init__.py` | Package entry point; re-exports `generate`, `generate_batch`, `Config`, `SampleResult`, `BatchResult`, `__version__` |
| `musicgen/api.py` | `generate(config) -> SampleResult`; orchestrates all pipeline stages; resume short-circuit |
| `musicgen/batch.py` | `generate_batch(config) -> BatchResult`; ProcessPoolExecutor (spawn context); manifest writes in main process only |
| `musicgen/cli.py` | Typer app with `generate`, `clean`, `calibrate` commands; injects `config_root` onto `sys.path` |
| `musicgen/sampler.py` | All song-level parameter sampling; `SongParams` frozen dataclass; no I/O |
| `musicgen/seeds.py` | RNG hierarchy: `derive_sample_seed`, `make_rngs`, `assign_split`, `save_random_state`; no I/O |
| `musicgen/generators/chord.py` | Chord progression MIDI authoring via `midiutil.MIDIFile` |
| `musicgen/generators/melody.py` | Melodic line generation over chord progression |
| `musicgen/generators/bassline.py` | Bass line generation from chords and melody |
| `musicgen/generators/beat.py` | Drum pattern generation with swing; reads beat-roll pattern files |
| `musicgen/renderer.py` | `render_stems` via `ThreadPoolExecutor`; `pick_soundfonts`; `FLUIDSYNTH_VERSION` at import |
| `musicgen/mixer.py` | Pedalboard FX, layer mask, `mix_part`, `concat_parts`; `pydub.AudioSegment` overlay |
| `musicgen/beats.py` | MIDI-tick beat time extraction; no audio dependency |
| `musicgen/annotator.py` | Assembles the full annotation dict; stays pure (no path writes) |
| `musicgen/writer.py` | Atomic per-sample layout; MIDI concat; stem concat; sum-of-stems assertion; sentinel rename |
| `musicgen/manifest.py` | JSONL manifest writer; `is_sample_complete` resume check |
| `musicgen/calibrate.py` | FluidSynth pre-roll measurement; cache at `.musicgen/fluidsynth_preroll.json` |
| `musicgen/musicality.py` | `librosa`-based audio quality scoring; wrapped by `save_random_state` in caller |
| `musicgen/duration_validator.py` | Time-signature-aware note duration validation |
| `timesig.py` (repo root) | `TimeSignatureRegistry`; per-time-signature spec for patterns, durations, measure counts |

## Data flow

```
Config (CLI > env > defaults)
       │
       ▼
generate(config)
       │
       ├─ seeds.derive_sample_seed + make_rngs → {params, generators, soundfonts, fx, mix}
       │
       ├─ sampler → key, tempo, time_sig, swing, arrangement, measures
       │
       ├─ generators (per unique part) → .mid files in working dir
       │       chord → melody → bassline → beat
       │
       ├─ renderer.pick_soundfonts → {layer: .sf2 path}
       │
       ├─ mixer.build_fx_boards + compute_layer_mask
       │
       ├─ per part in arrangement:
       │       renderer.render_stems (ThreadPool 4x) → {layer: .wav}
       │       mixer.mix_part → part mix.wav + stem_paths
       │       beats.extract_beat_times / extract_downbeat_times
       │
       ├─ mixer.concat_parts → final mix.wav (working dir)
       │
       ├─ musicality.get_musicality_score (wrapped in save_random_state)
       │
       ├─ annotator.annotate → annotation dict
       │
       ├─ writer.write_sample → dataset_root/<idx:06d>/{mix.wav, stems/, midi/, sample.json}
       │
       └─ manifest_writer.append → dataset_root/manifest.jsonl
```

## Entry points

- **Library:** `from musicgen import generate, Config` — single-sample generation via `api.generate`.
- **Library (batch):** `from musicgen import generate_batch` — parallel generation via `batch.generate_batch`.
- **CLI:** `musicgen generate --seed S --count N --out DIR` — invokes `generate_batch` after building `Config` from CLI args.
- **CLI utilities:** `musicgen clean --failed` removes partial output dirs; `musicgen calibrate` measures and caches the FluidSynth pre-roll offset.
- **Thin wrapper script:** `python music_gen.py` calls `musicgen.generate(Config(global_seed=1, sample_index=0))` and logs the result. Used only as a smoke check.
- **`python -m musicgen`** — supported via `__main__.py`.

## Concurrency model

Two levels of concurrency are used, at different granularities:

- **Within a sample (ThreadPoolExecutor):** `renderer.render_stems` dispatches four FluidSynth subprocess calls in parallel using `ThreadPoolExecutor(max_workers=4)`. FluidSynth is an external process so the GIL is not contended during synthesis. Parts within a single sample remain sequential.

- **Across samples (ProcessPoolExecutor):** `batch.generate_batch` uses `ProcessPoolExecutor` with the `spawn` multiprocessing context. Each worker is a fresh interpreter that receives only `(global_seed, sample_index, config)` — no shared RNG state is inherited. Manifest writes happen in the main process only, via `as_completed`, to avoid inter-process coordination.

`_NullManifestWriter` is passed to each worker's `generate` call so the worker never writes manifest lines; the main process appends them after receiving each result.

## Resume invariant

If `writer.write_sample` raises before step 7 (the atomic sentinel rename), `sample.json` never exists and `manifest.ManifestWriter.is_sample_complete` returns `False`. On a resumed batch run, `generate_batch` re-queues that index. Once `sample.json` exists, `api.generate` short-circuits at step 3 and reconstructs the `SampleResult` from the existing file without re-running the pipeline.
