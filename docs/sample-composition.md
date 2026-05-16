# Sample Composition in musicgen

Sample composition (features M3–M5) lets you blend pre-recorded audio samples into musicgen's generated output. Samples are drawn from a local library you index once, then selected, time-stretched, and mixed automatically for every generation run.

---

## 1. Installation

The sample composition subsystem requires a few extra dependencies. Install them with the `samples` extra:

```
pip install musicgen[samples]
```

This pulls in `audio-sample-manager`, `soundfile`, and (on Linux) `rubberband-stretch`. The `rubberband-stretch` package is required only for BPM normalization; if it is absent, musicgen emits a warning and skips time-stretching rather than failing hard. Key transposition depends on `librosa`, which is already a core dependency.

---

## 2. Quick Start

The minimal workflow is two CLI commands: build a sample library from a directory of audio files, then pass that library to `musicgen generate`.

```bash
# Step 1 — index a folder of samples into a JSON library
musicgen samples build \
    --dir ~/my_samples \
    --output ~/my_samples/library.json \
    --recursive \
    --musicality

# Step 2 — generate with samples mixed in
musicgen generate \
    --genre hiphop \
    --sample-db ~/my_samples/library.json \
    --sample-beat alongside \
    --sample-bassline substitution \
    --output ./out
```

The library JSON is reusable. Rebuild it only when you add or remove files.

---

## 3. Mixing Modes

Each layer (`beat`, `bassline`, `melody`, `harmony`) is controlled by an independent mode. Three modes are available.

### alongside

The sample plays in parallel with the generated audio for that layer. Both signals are mixed down together. This is the default for `beat` and `bassline`.

```bash
--sample-beat alongside
```

Use this when you want a live drum loop to reinforce the generated beat without replacing it.

### substitution

The generated audio for that layer is removed from the mix and replaced entirely by the sample. The sample is tiled to match the part duration.

```bash
--sample-bassline substitution
```

Use this when the sample quality for a layer is higher than what synthesis produces — for example, a recorded bass riff instead of a MIDI-driven bassline.

### adlib

The sample is placed once at a specific beat position rather than looping for the full part. The beat position is set through the Python API via `oneshot_at_beat`; the CLI places adlib samples at beat 1 by default.

```bash
--sample-melody adlib
```

Use this for one-shot stabs, fills, or transitions that should not loop.

### off

The layer is left to the generator only. No sample is selected or applied.

```bash
--sample-melody off   # default for melody and harmony
```

---

## 4. Sample Library Builder

`musicgen samples build` walks a directory, analyzes each audio file, and writes a JSON database used by all subsequent generation calls.

```
musicgen samples build \
    --dir DIR \
    --output JSON \
    [--category beat|bass|melody|harmony] \
    [--genre GENRE]... \
    [--mood MOOD]... \
    [--tag TAG]... \
    [--musicality] \
    [--recursive] \
    [-v] [-q]
```

| Flag | Purpose |
|---|---|
| `--dir DIR` | Root directory to search. |
| `--output JSON` | Output path for the library JSON file. |
| `--category` | Override category inference for every file in this batch. |
| `--genre GENRE` | Tag all samples in this batch with one or more genres. Repeatable. |
| `--mood MOOD` | Tag all samples in this batch with one or more moods. Repeatable. |
| `--tag TAG` | Arbitrary additional tag. Repeatable. |
| `--musicality` | Score each sample with `musicality.explain()` and store the score. Slow; optional. |
| `--recursive` | Descend into sub-directories. |
| `-v` / `-q` | Verbose / quiet logging. |

Supported audio formats: WAV, FLAC, OGG, AIF, MP3.

BPM and key are detected automatically for every file via `SampleManager.add_sample(analyze=True)`. Category is inferred from filename keywords unless `--category` overrides it (see section 10).

---

## 5. Integration with `musicgen generate`

After indexing a library you pass it to every `generate` call with `--sample-db`. The remaining `--sample-*` flags configure which layers use samples and how.

```
musicgen generate \
    [all existing generate flags] \
    --sample-db PATH \
    [--sample-beat    alongside|substitution|adlib|off] \
    [--sample-bassline alongside|substitution|adlib|off] \
    [--sample-melody  alongside|substitution|adlib|off] \
    [--sample-harmony alongside|substitution|adlib|off] \
    [--sample-gain FLOAT] \
    [--sample-min-score FLOAT]
```

**Defaults when `--sample-db` is set:**

| Layer | Default mode |
|---|---|
| beat | alongside |
| bassline | alongside |
| melody | off |
| harmony | off |

`--sample-gain` sets the gain in dB applied to every selected sample (default `-3.0`).

`--sample-min-score` filters the library so only samples with a `musicality_score` at or above the threshold are eligible. Requires `--musicality` to have been used during `samples build`.

---

## 6. Python API

### SampleLayerRule

Controls behavior for one layer.

```python
from musicgen.sample_composition import SampleLayerRule

@dataclass
class SampleLayerRule:
    layer: str                              # "beat" | "bassline" | "melody" | "harmony"
    mode: str                               # "alongside" | "substitution" | "adlib"
    loop_align_to_measure: bool = True
    oneshot_at_beat: Optional[int] = None   # required when mode="adlib"
    max_bpm_stretch_pct: float = 10.0
    min_musicality_score: Optional[float] = None
    gain_db: float = -3.0
    tags: Optional[List[str]] = None
    genre: Optional[List[str]] = None
    mood: Optional[List[str]] = None
```

`max_bpm_stretch_pct` caps how far the sample's BPM is allowed to stretch before it is rejected from the candidate pool. If no candidate survives this filter musicgen logs a warning and leaves the layer sample-free.

`tags`, `genre`, and `mood` act as allow-lists: only samples that carry at least one matching value are eligible. Omit them to allow any sample in that category.

### SampleCompositionConfig

Groups all layer rules and global settings for one generation run.

```python
from musicgen.sample_composition import SampleCompositionConfig

@dataclass
class SampleCompositionConfig:
    sample_db_path: str
    layer_rules: Dict[str, SampleLayerRule] = field(default_factory=dict)
    global_min_musicality: Optional[float] = None
    allow_transposition: bool = True
    allow_time_stretching: bool = True
```

`global_min_musicality` is applied before any per-layer `min_musicality_score`.

`allow_transposition` and `allow_time_stretching` are global kill-switches for the audio transforms (see section 8). Disable them to use samples exactly as recorded.

### Config integration

`SampleCompositionConfig` attaches to the top-level `MusicgenConfig`:

```python
from musicgen.config import MusicgenConfig
from musicgen.sample_composition import SampleCompositionConfig, SampleLayerRule

config = MusicgenConfig(
    genre="hiphop",
    sample_composition=SampleCompositionConfig(
        sample_db_path="/data/samples/library.json",
        global_min_musicality=0.6,
        layer_rules={
            "beat": SampleLayerRule(
                layer="beat",
                mode="alongside",
                genre=["hiphop", "trap"],
                gain_db=-6.0,
                max_bpm_stretch_pct=8.0,
            ),
            "bassline": SampleLayerRule(
                layer="bassline",
                mode="substitution",
                min_musicality_score=0.7,
            ),
            "melody": SampleLayerRule(
                layer="melody",
                mode="adlib",
                oneshot_at_beat=3,
                gain_db=-9.0,
                mood=["dark", "eerie"],
            ),
        },
    ),
)
```

### Full generate() example

```python
from musicgen.api import generate

results = generate(config)

for part in results:
    print(part.output_path)
    if part.sample_json_path:
        import json
        used = json.loads(open(part.sample_json_path).read())["used_samples"]
        for layer, info in used.items():
            print(f"  {layer}: {info['name']} (mode={info['mode']})")
```

---

## 7. Pipeline Integration

Sample selection and mixing are wired into `api.py` as follows:

1. `select_samples()` is called **once before the part loop**. The same sample is used for the entire composition, ensuring consistency across all parts.
2. For each part, `prepare_sample_wavs()` and `apply_substitutions()` run **before** `mix_part()`. Substitution mode removes the generated layer from the mix at this stage.
3. For each part, `apply_alongside()` runs **after** `mix_part()`, blending the transformed sample into the already-mixed part audio.
4. When sample composition is active, `used_samples` is written to `sample.json` in the output directory.

---

## 8. Audio Transforms

Three transforms are applied in order — stretch, then shift, then tile — before the sample is mixed.

### BPM normalization (stretch)

The sample's detected BPM is stretched or compressed to match the generation BPM using `rubberband-stretch`. Only applied when `allow_time_stretching=True` and the deviation between sample BPM and target BPM is within `max_bpm_stretch_pct`.

If the `rubberband-stretch` package is not installed, musicgen emits a warning and skips this step. The sample is used at its native BPM. Install it with:

```
pip install rubberband-stretch   # Linux; see rubberband docs for macOS/Windows
```

### Key transposition (shift)

After BPM normalization the sample is transposed to match the generation key using `librosa.effects.pitch_shift`. Only applied when `allow_transposition=True`. Because `librosa` is a core dependency this step is always available.

### Loop tiling (tile)

After transposition the sample is tiled with `numpy.tile` to exactly match the duration of the current part. For adlib mode the sample is placed at `oneshot_at_beat` and padded with silence; tiling does not apply.

---

## 9. Sample Usage Annotation

When sample composition is active, musicgen writes a `sample.json` file into the output directory alongside the generated audio. The `used_samples` key maps each active layer to metadata about the selected sample.

```json
{
  "used_samples": {
    "beat": {
      "id": 1,
      "name": "kick_120",
      "path": "/samples/kick_120.wav",
      "bpm": 120.0,
      "key": "G",
      "category": "beat",
      "musicality_score": 0.82,
      "mode": "alongside"
    },
    "bassline": {
      "id": 7,
      "name": "sub_bass_90",
      "path": "/samples/sub_bass_90.wav",
      "bpm": 90.0,
      "key": "A",
      "category": "bass",
      "musicality_score": 0.74,
      "mode": "substitution"
    }
  }
}
```

Layers in `off` mode are absent from `used_samples`. `musicality_score` is `null` when the library was built without `--musicality`.

---

## 10. Category Auto-Detection

When `--category` is not passed to `musicgen samples build`, the category is inferred from keywords in the filename. The lookup order is shown below; the first match wins. If no keyword matches, the sample is assigned `melody` as the fallback.

| Category | Filename keywords |
|---|---|
| `beat` | `beat`, `kick`, `hat`, `snare`, `drum`, `perc`, `clap` |
| `bass` | `bass`, `sub` |
| `harmony` | `pad`, `chord`, `harm`, `atmo`, `ambient`, `strings`, `keys`, `piano` |
| `melody` | `lead`, `melody`, `lick`, `riff`, `synth`, `arp` — and default fallback |

Keyword matching is case-insensitive. A file named `Kick_Trap_120bpm.wav` is assigned `beat`. A file named `MyLoop.wav` with no matching keywords is assigned `melody`.

To override inference for every file in a batch, pass `--category`:

```bash
musicgen samples build --dir ~/drums --output drums.json --category beat
```

---

## 11. Troubleshooting

### rubberband absent — BPM stretch skipped

**Symptom:** Warning log line containing `rubberband` not found; samples play at wrong tempo.

**Fix:** Install the package.

```
pip install rubberband-stretch
```

On macOS and Windows you may need to install the native `rubberband` library separately before the pip package. Alternatively, set `allow_time_stretching=False` in `SampleCompositionConfig` to disable the transform entirely and accept the tempo mismatch.

### No matching sample found for layer

**Symptom:** Log line `no eligible sample for layer <name>` and the layer is sample-free in the output.

**Causes and fixes:**

- `max_bpm_stretch_pct` is too restrictive for the target BPM. Raise the value or re-index samples closer in tempo to your generation BPM.
- `min_musicality_score` or `global_min_musicality` is filtering out all candidates. Lower the threshold or rebuild the library with `--musicality` if scores are absent.
- `genre`, `mood`, or `tags` filters on `SampleLayerRule` do not match any library entry. Check that the allow-lists in your rule match the metadata stored during `samples build`.
- The library has no samples for the requested category. Rebuild with files of the right type, or use `--category` to assign the correct category during indexing.

### Wrong category inferred

**Symptom:** A bass loop appears in `used_samples` under `melody` or vice versa.

**Fix:** Rename the file to include a recognized keyword, or force the correct category at index time:

```bash
musicgen samples build --dir ~/bass_loops --output bass.json --category bass
```

You can maintain separate JSON files per category and pass only the one you want via `--sample-db`, or merge them by rebuilding from a combined directory.
