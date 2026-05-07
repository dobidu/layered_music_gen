# External Integrations

**Last updated:** 2026-04-29
**Status:** v0.1.0 shipped; v0.2 ecosystem integrations complete (branch feat/soundfont-manager)

---

## Runtime dependencies (always required)

| Library | Role | Key usage |
|---|---|---|
| midiutil | MIDI file generation | `generators/{chord,melody,bassline,beat}.py` — builds MIDI tracks from note data |
| music21 | Music theory primitives | `generators/chord.py`, `sampler.py` — scales, key selection, chord naming |
| midi2audio / FluidSynth | MIDI → WAV synthesis | `renderer.py` — `FluidSynth.midi_to_audio()` per layer stem |
| pedalboard | Audio FX chain | `mixer.py` — Compressor, Reverb, Delay, Chorus, Phaser, Filter, Gain applied per layer |
| pydub | Audio overlay + export | `mixer.py` — `mix_part`, `concat_parts`, WAV file I/O |
| librosa | Audio feature analysis | `musicality.py` — beat tracking, chroma, tonnetz, onset strength |
| mido | MIDI tick extraction | `beats.py` — swing-aware beat/downbeat time derivation |
| python-magic | File type detection | `writer.py` — WAV format validation |
| typer | CLI framework | `cli.py` — `generate`, `clean`, `calibrate`, `index-midi`, `index-audio` |

---

## Optional integrations (v0.2) — sibling ecosystem

All three integrations are **opt-in** with zero hard dependencies added to `pyproject.toml`. Each library is lazy-imported at call time; a clear `ImportError` is raised with an install hint if the library is absent.

### Integration 1 — soundfont_manager → musicgen

**Repo:** `dobidu/soundfont_manager`
**Purpose:** Replace blind `rng.choice(os.listdir(...))` in soundfont selection with metadata-aware, tag-based selection from a SoundfontManager JSON database.

**Activation:** set `cfg.soundfont_manager_db` (or env var `MUSICGEN_SOUNDFONT_MANAGER_DB`).

**Files changed:**
- `config.py` — added `soundfont_manager_db: Optional[str]` and `soundfont_manager_sf_dir: Optional[str]`
- `src/musicgen/renderer.py` — added `_LAYER_TAGS`, `_pick_via_soundfont_manager()`, updated `pick_soundfonts()`

**How it works:**

```
pick_soundfonts(cfg, rng)
  ├─ if cfg.soundfont_manager_db is set:
  │    _pick_via_soundfont_manager(cfg, rng)
  │      ├─ import soundfont_manager.SoundfontManager  (lazy)
  │      ├─ for each layer: sm.get_soundfonts_by_tags(_LAYER_TAGS[layer])
  │      ├─ sort candidates by sf.path  (cross-machine determinism)
  │      ├─ rng.choice(sorted_candidates)
  │      └─ return sm.get_absolute_path(chosen)  per layer
  │    if import fails or no matches → fall through
  └─ fallback: sorted(os.listdir(sf_layer_dir)) + rng.choice()
```

**Layer → tag mapping (`_LAYER_TAGS`):**

| Layer | Tags searched |
|---|---|
| `beat` | `["drums", "percussion"]` |
| `melody` | `["melody", "lead", "piano", "strings"]` |
| `harmony` | `["harmony", "chords", "pads", "pad"]` |
| `bassline` | `["bass"]` |

**Determinism contract:** same `(global_seed, sample_index)` → same soundfont pick. Candidates are sorted by `sf.path` (relative path from db) before `rng.choice`, so insertion order in the SM database does not affect the result.

**Fallback triggers:** `ImportError` (package not installed), empty tag result for any layer, any exception inside SM.

---

### Integration 2 — musicgen → midi_file_manager

**Repo:** `dobidu/midi_file_manager`
**Purpose:** Index all generated MIDI files into a MidiManager database with musicgen ground-truth metadata, enabling downstream ML pipelines to query by tempo, key, time signature, split, etc.

**Files added:**
- `src/musicgen/midi_indexer.py` — `index_midi_dataset()` library function
- `src/musicgen/cli.py` — `musicgen index-midi` command

**Dataset walk:**
```
dataset_root/
  000000/
    sample.json       ← ground-truth: tempo_bpm, key, time_signature, split, musicality_score
    midi/
      beat.mid        ← indexed
      melody.mid      ← indexed
      harmony.mid     ← indexed
      bassline.mid    ← indexed
  000001/
    ...
```

**Enrichment flow:**
```
for each sample_dir with sample.json + midi/:
  for each layer in (beat, melody, harmony, bassline):
    meta = mm.add_midi(path, analyze=True, save=False)   # MIDI structure analysis
    meta.bpm           = sample["tempo_bpm"]             # override MIDI-extracted
    meta.key           = sample["key"]
    meta.time_signature = sample["time_signature"]
    meta.category      = _LAYER_CATEGORY[layer]          # beat→drums, bassline→bass, …
    meta.tags          = ["musicgen", layer, split]
    meta.description   = f"musicgen:{entry} layer={layer} split={split} musicality={score:.3f}"
mm.save_midis()  # single write at end
```

**Layer → MidiCategory mapping:**

| Layer | MidiCategory |
|---|---|
| `beat` | `"drums"` |
| `melody` | `"melody"` |
| `harmony` | `"harmony"` |
| `bassline` | `"bass"` |

**Fallback:** `ImportError` if `midi_manager` package not installed. `FileNotFoundError` if `dataset_root` missing.

---

### Integration 3 — musicgen → audio_sample_manager

**Repo:** `dobidu/audio_sample_manager`
**Purpose:** Index generated WAV stems into a SampleManager database alongside external audio libraries, enabling unified cross-library queries and `SampleSelector.select_for_layer()` across both generated and externally-sourced audio.

**Primary use case:** A producer or researcher has external drum packs, synth loops, etc. already in a SampleManager db. Running `musicgen index-audio` adds musicgen's generated stems to the same db, so queries like "all bass stems at 90 BPM in Am minor" return both external and generated samples.

**Files added:**
- `src/musicgen/audio_indexer.py` — `index_audio_dataset()` library function
- `src/musicgen/cli.py` — `musicgen index-audio` command

**Dataset walk:**
```
dataset_root/
  000000/
    sample.json       ← ground-truth: tempo_bpm, key, mode, time_signature, split, musicality_score
    stems/
      beat.wav        ← indexed
      melody.wav      ← indexed
      harmony.wav     ← indexed
      bassline.wav    ← indexed
  000001/
    ...
```

**Enrichment flow:**
```
for each sample_dir with sample.json + stems/:
  for each layer in (beat, melody, harmony, bassline):
    meta = sm.add_sample(path, analyze=True, save=False)  # librosa feature extraction
    meta.bpm           = sample["tempo_bpm"]              # override librosa-extracted
    meta.key           = sample["key"]
    meta.time_signature = sample["time_signature"]
    meta.scale         = sample["mode"]                   # "major" | "minor"
    meta.category      = _LAYER_CATEGORY[layer]           # beat→beat, bassline→bass, …
    meta.tags          = ["musicgen", layer, split]
    meta.is_loop       = False                            # stems are full songs, not loops
    meta.description   = f"musicgen:{entry} layer={layer} split={split} musicality={score:.3f}"
sm.save_samples()  # single write at end
```

**Layer → SampleCategory mapping:**

| Layer | SampleCategory |
|---|---|
| `beat` | `"beat"` |
| `melody` | `"melody"` |
| `harmony` | `"harmony"` |
| `bassline` | `"bass"` |

**Ground-truth vs librosa:** `bpm`, `key`, `time_signature`, and `scale` come from `sample.json` (exact, no re-extraction). `waveform_features` and `spectral_features` come from `analyze_sample()` (librosa), which covers timbre characteristics not tracked by musicgen.

**`is_loop = False`:** musicgen stems span full songs (intro+verse+chorus+outro concatenated). They are not fixed-length loops.

**Note on `mix.wav`:** Not indexed. `SampleCategory` has no `full_song` value; the mix is better represented by the individual stems that compose it.

---

## Configuration reference

### soundfont_manager fields (Config)

| Field | Env var | Default | Description |
|---|---|---|---|
| `soundfont_manager_db` | `MUSICGEN_SOUNDFONT_MANAGER_DB` | `None` | Path to SoundfontManager JSON database. Set to activate SM-backed selection. |
| `soundfont_manager_sf_dir` | `MUSICGEN_SOUNDFONT_MANAGER_SF_DIR` | `None` | Base directory for `.sf2` files (SM's `sf2_directory` constructor arg). Set when db stores relative paths. |

### CLI commands

| Command | Lazy dep | Key options |
|---|---|---|
| `musicgen index-midi` | `midi_manager` (dobidu/midi_file_manager) | `--dataset`, `--out`, `--midi-dir`, `--csv` |
| `musicgen index-audio` | `sample_manager` (dobidu/audio_sample_manager) | `--dataset`, `--out`, `--samples-dir`, `--csv` |

---

## Internal audio pipeline (unchanged from v0.1)

```
sampler → generators (chord/melody/bassline/beat)
       → renderer (FluidSynth, parallel stems via ThreadPoolExecutor)
       → mixer (pedalboard FX + pydub overlay + concat)
       → beats (mido MIDI-tick extraction, swing-aware)
       → annotator (sample.json schema assembly)
       → writer (atomic dir, sum-of-stems assertion, output_mode routing)
       → manifest (JSONL append-under-lock)
```

---

## Dependency audit

### Hard dependencies (pyproject.toml)

All present and required for core generation pipeline. No new hard deps were added in v0.2.

### Optional (not in pyproject.toml)

| Package | Imported by | Install |
|---|---|---|
| `soundfont_manager` | `renderer._pick_via_soundfont_manager` | `pip install git+https://github.com/dobidu/soundfont_manager` |
| `midi_manager` | `midi_indexer.index_midi_dataset` | `pip install git+https://github.com/dobidu/midi_file_manager` |
| `sample_manager` | `audio_indexer.index_audio_dataset` | `pip install git+https://github.com/dobidu/audio_sample_manager` |

### System binaries

| Binary | Required for | Install |
|---|---|---|
| `fluidsynth` | WAV synthesis | `apt install fluidsynth` / `brew install fluidsynth` |
| `ffmpeg` or `avconv` | pydub format conversion | `apt install ffmpeg` |
| `libmagic` | python-magic (file type detection) | `apt install libmagic1` |

---

*Integration audit updated: 2026-04-29*
