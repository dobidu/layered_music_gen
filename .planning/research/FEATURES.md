# Features Research

**Scope:** What does a synthetic music dataset generator need to expose to support all four target ML tasks (tagging, source separation, beat/tempo/downbeat detection, audio→MIDI transcription) from a single reproducible generation seed?

**Prior art surveyed:** Slakh2100, MAESTRO, MusicNet, Lakh MIDI, GuitarSet, MedleyDB.

**Confidence:** HIGH for per-task feature mapping (follows from task definitions); MEDIUM for dataset API details (training cutoff Aug 2025).

## Key findings

- **No existing public dataset covers all four target tasks from a single reproducible seed.** This is the core differentiator.
- **Slakh2100 is closest prior art** but has two gaps: (a) no beat/downbeat annotations, (b) fixed MIDI corpus — not reproducible or extensible.
- **Three highest-leverage table-stakes features are one-line changes** in the existing code: persist stems before `mix_and_save` deletes them, persist per-part MIDI, and write per-sample `annotation.json`.
- **`beat_annotator.py` already exists** — integrating it into the pipeline gives beat annotations at near-zero new cost (but see PITFALLS.md: swing drift bug).
- **Stem-to-mix sum fidelity is the most failure-prone source-separation requirement.** Must be added as a post-generation assertion.
- **Swing as a quantified label is unique** — no public dataset exposes swing ground truth. Near-zero cost to add.

## Table-stakes features

### Audio

| Feature | Why | Complexity | Prior art |
|---|---|---|---|
| Mixed stereo WAV per sample | Observable signal for all 4 tasks | Low — already exists | All |
| Consistent sample rate (44.1 kHz) | Models break on mixed rates | Low — FluidSynth flag | Slakh, MAESTRO |
| Consistent bit depth (16-bit or 32-float) | Downstream loaders assume fixed dtype | Low | All |
| No clipping, consistent LUFS range | Clipping destroys labels; LUFS variance degrades separation | Medium | MedleyDB |

### Stems

| Feature | Why | Complexity | Prior art |
|---|---|---|---|
| Per-layer stems, sample-exact aligned to mix | Source separation impossible without; misalignment destroys the task | Medium — persist before deletion at `music_gen.py:898` | Slakh, MedleyDB |
| Stems sum to mix within numerical precision | Ground truth must be exact | Medium — volume/pan must be deterministic and bit-accurate | Slakh |
| Stem presence flags per sample | Separation models need to know which tracks are active | Low — already computed in `inst_probabilities` logic | Slakh |

### MIDI ground truth

| Feature | Why | Complexity | Prior art |
|---|---|---|---|
| Per-layer MIDI files aligned to audio | Transcription requires MIDI-to-audio alignment | Medium — FluidSynth render latency must be measured | MAESTRO, MusicNet, Slakh |
| Note-level onset/offset/pitch/velocity | Basic transcription label | Low — midiutil already writes this | MAESTRO |
| MIDI tempo map in file header | Required to convert ticks → seconds | Low | All |
| Instrument/program number per track | Multi-instrument transcription | Low | Slakh |

### Annotations JSON

| Feature | Why | Complexity |
|---|---|---|
| Tempo (BPM) | Beat detection, transcription | Low — already generated |
| Time signature | Meter / downbeat spacing | Low |
| Key (tonic + mode) | Tagging, harmony tasks | Low |
| Song duration (seconds) | Windowed inference | Low |
| Active layers list | Separation, tagging | Low |
| Seed value | Reproducibility | Low — **unique to procedural datasets** |
| FluidSynth version | Determinism auditing | Low |

### Dataset structure

| Feature | Why | Complexity |
|---|---|---|
| Top-level manifest (JSONL) | HF Datasets / WebDataset / torchdata all need it | Low |
| Deterministic train/valid/test split | Reproducible benchmarks | Low — hash seed into split |
| Canonical per-sample directory layout | Loaders find files by convention | Low |
| Canonical file naming | Fixes broken `song_name[:20]` truncation at `music_gen.py:1143` | Low |

## Differentiators

### Procedural reproducibility (core differentiator)

- **Seed-reproducible generation** — no public dataset offers this. Enables ablations, bug reproduction, dataset versioning.
- **Seed logged in every annotation** — users can regenerate any single sample or extend the dataset by extending seed range.
- **Configurable randomization parameters** — bias distributions (more 7/8 samples, specific key ranges) for targeted training sets.

### Beat / downbeat annotations (major gap in prior art)

- **Beat timestamps (seconds)** — GuitarSet has these; Slakh does not.
- **Downbeat timestamps (seconds)** — distinct task from beat detection.
- **Swing offset as a quantified label** — **unique**. Swing-aware beat detection is an open MIR problem.

### Harmonic / structural annotations (rare in prior art)

- **Chord progression per section** — Slakh lacks chords; MusicNet lacks structure.
- **Song section labels with timestamps** (intro/verse/chorus/bridge/outro) — essentially unique among synthetic datasets.
- **Per-section active layers** — sections often have different instrumentation; required for accurate separation training on section-level windows.

### Multi-task label coverage (core differentiator)

- **All four ML tasks from one sample** — uniquely possible here.
- **Musicality score as a meta-label** — `MusicalityAnalyzer` output stored per sample. Enables curriculum learning and downstream filtering.
- **Soundfont identity per layer** — timbre diversity as a controlled variable for timbre-robust separation research.
- **FX chain parameters per layer** — full effects provenance, enables FX-aware separation research.

## Anti-features (deliberately NOT build)

| Anti-feature | Why |
|---|---|
| Web UI / HTTP API | Out of scope per PROJECT.md |
| Real-time playback / live mode | This is a batch dataset generator, not a performer |
| Human-in-the-loop quality gating | Makes throughput unpredictable; hides non-determinism |
| Auto-filtering low-quality samples | Hides distribution shape from ML users |
| Training the downstream ML models | Out of scope; downstream concern |
| Vocal synthesis / lyrics | Complexity explosion; out of scope |
| Cloud storage integration (S3/GCS) | Adds credentials, networking; not needed at 1k–10k |
| Distributed / cluster-scale (100k+) | Over-engineering; single-box multiprocessing suffices |
| Lossy audio formats (MP3, OGG) | Degrade transcription/separation ground truth |
| Sample deduplication | Unique seeds → unique samples by construction |
| Versioned dataset diffing tooling | Seeds are implicit version IDs |
| Redistributable audio packaging | Blocked by soundfont license audit |

## Per-ML-task feature mapping

| Task | Model input | Required labels | Does NOT need |
|---|---|---|---|
| **Tagging / classification** | mix WAV | key, tempo, time sig, active layers, section structure, musicality score, swing, soundfont IDs, FX params | stems, MIDI, beat times |
| **Source separation** | mix WAV | per-layer stem WAVs, stem presence flags, volume/pan per layer (auditable sum identity) | MIDI, beat times, chord annotations |
| **Beat / tempo / downbeat** | mix WAV | beat timestamps (seconds), downbeat timestamps (seconds), tempo BPM, time signature | stems, MIDI (beyond duration) |
| **Transcription (audio → MIDI)** | mix WAV or stem WAV | per-layer MIDI, tempo map, program per track, FluidSynth-render-latency alignment offset | beat times, musicality score |

## Feature dependencies

```
Seed determinism
    └─► All 4 tasks (without this nothing is reproducible)

Persist per-layer stems (fix deletion at music_gen.py:898)
    ├─► Source separation
    └─► Stem-level transcription sub-task

Persist per-layer MIDI (currently written to intermediate files only)
    ├─► Transcription
    └─► Beat timestamp cross-check

Beat grid computation (tempo + time sig + swing → timestamps)
    ├─► Beat/downbeat detection
    └─► Section timestamp computation

Per-sample annotation.json
    ├─► All 4 tasks (label delivery vehicle)
    ├─► manifest.jsonl (aggregated index)
    └─► Train/valid/test split

Stem presence flags
    ├─► Separation
    └─► Tagging (active instrument labels)

FluidSynth render latency measurement
    └─► Transcription (MIDI ↔ audio alignment)
```

## MVP feature priority (strict order)

1. **Persist stems before deletion** — one-line change in `mix_and_save`; unlocks source separation entirely.
2. **Persist per-layer MIDI in final output directory** — unlocks transcription.
3. **Write per-sample `annotation.json`** — seed, tempo, key, time sig, swing, layers, soundfonts, section structure with timestamps, chord progression, musicality score, FluidSynth version.
4. **Integrate `beat_annotator.py`** (with swing-aware fix — see PITFALLS.md) — beat + downbeat timestamps.
5. **Write `manifest.jsonl`** — top-level index.
6. **Deterministic train/valid/test split** — hash seed into split assignment.

Deferred to Extend/Research:

- FX chain parameter serialization
- Configurable distribution targeting / stratified generation
- FluidSynth render-latency measurement + correction
- Swing ground truth beat warping

## Open questions

1. **FluidSynth render latency magnitude** — empirical, not documented; needs measurement during Productize.
2. **Stem-sum-to-mix precision** — `pydub` float arithmetic may not be bit-exact; needs verification.
3. **Soundfont licensing** — blocks public distribution; does not block generation for internal use.
4. **Beat annotation format** — recommend raw JSON arrays in per-sample annotations; JAMS conversion is a downstream concern.
