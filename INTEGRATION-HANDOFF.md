# Integration Handoff — musicgen v0.1.0 × Manager Ecosystem

**Date:** 2026-04-28
**Context:** musicgen v0.1.0 is shipped. Three sibling repos are candidates for integration.
**Next action:** Run the prompts below in fresh sessions to spec, plan, and execute each integration.

---

## Repos involved

| Repo | Role in integration |
|---|---|
| `dobidu/layered_music_gen` | Dataset generator (producer) |
| `dobidu/soundfont_manager` | SF2 selection layer (replaces blind `rng.choice(os.listdir(...))`) |
| `dobidu/midi_file_manager` | MIDI indexing + ML feature export (downstream consumer) |
| `dobidu/audio_sample_manager` | Audio sample indexing (downstream consumer, lower priority) |

---

## Integration 1 — soundfont_manager → musicgen (HIGHEST PRIORITY)

### Problem being solved

`renderer.pick_soundfonts()` currently does:
```python
files = sorted(os.listdir(cfg.sf_dir / layer))
return {layer: rng.choice(files)}
```
Blind random pick from a flat directory. PITFALLS P-5 in the research identified "narrow soundfont pool causes model timbre overfitting." soundfont_manager has metadata (note range, timbre tags, instrument family) and `get_random_soundfont(predicate)` / `get_soundfonts_by_tags()` APIs designed for exactly this use case.

### What the integration looks like

- `renderer.pick_soundfonts(cfg, rng)` becomes soundfont-manager-aware when `cfg.soundfont_manager_db` is set.
- Falls back to current directory scan when soundfont_manager is not installed or db path is unset (opt-in, no new hard dependency).
- Predicate-based selection: e.g., pick a soundfont tagged `"drums"` for the beat layer, `"bass"` for bassline, etc.
- Seed-deterministic: convert `rng.random()` float to an index into the filtered soundfont list.

### GSD prompt to use in a fresh session

```
I want to spec and plan an integration between two of my repos:

Producer: dobidu/layered_music_gen (musicgen v0.1.0)
  - Generates synthetic music datasets: MIDI + WAV stems + mix + sample.json annotations.
  - Soundfont selection lives in src/musicgen/renderer.py `pick_soundfonts(cfg, rng)`.
  - Currently: sorted(os.listdir("sf/<layer>/")) + rng.choice(). Blind random pick.
  - Config dataclass is in config.py at repo root with CLI > env > defaults precedence.

Consumer/provider: dobidu/soundfont_manager
  - Manages SF2 soundfont libraries with metadata (note range, timbre tags, instrument family).
  - Key API: SoundfontManager(db_path), get_random_soundfont(predicate), get_soundfonts_by_tags(tags), filter_soundfonts(**kwargs).
  - get_random_soundfont is designed to be called by external systems at runtime.

Goal: Replace the blind rng.choice(os.listdir(...)) in pick_soundfonts() with a
SoundfontManager-backed selection that is:
  1. Opt-in — falls back to current behavior when soundfont_manager not installed or db not configured.
  2. Seed-deterministic — same (global_seed, sample_index) must produce the same soundfont choice.
  3. Layer-aware — beat layer gets percussion-tagged SFs, bassline gets bass-tagged SFs, etc.
  4. No new hard dependency in pyproject.toml (optional extra at most).

Please:
  1. Read both repos to understand the current APIs.
  2. Discuss the integration design (where the seam is, how to preserve determinism, how opt-in works).
  3. Write a GSD-style CONTEXT + PLAN for the implementation.
  4. Execute the plan with TDD (RED commit then GREEN commit).
```

---

## Integration 2 — musicgen → midi_file_manager (MEDIUM PRIORITY)

### Problem being solved

musicgen generates per-layer `.mid` files (4 layers × N samples) with full musical metadata in `sample.json`. midi_file_manager can index these into a searchable database and export ML feature vectors. This turns the musicgen output into a first-class MIDI library that downstream models can query by tempo, key, time signature, chord pattern, etc.

### What the integration looks like

- A post-batch step (CLI subcommand or standalone script): `musicgen index-midi --dataset ./dataset --out ./midi_db.json`
- Calls `MidiManager.scan_directory(dataset_root)` or `midi_integration.py`'s adapter.
- Enriches each MIDI entry with musicgen's `sample.json` metadata (tempo, key, time_sig, split, musicality_score) so the MIDI index is searchable by musical parameters.
- Optional: export ML feature CSV merging MIDI features + musicgen annotations.

### GSD prompt to use in a fresh session

```
I want to spec and plan a downstream integration between two of my repos:

Producer: dobidu/layered_music_gen (musicgen v0.1.0)
  - generate_batch(Config) writes datasets to <dataset_root>/<idx:06d>/:
      sample.json  (seed, key, tempo, time_sig, swing, musicality_score, split, beat_times, ...)
      midi/<layer>.mid  (4 layers: beat, melody, harmony, bassline)
      stems/<layer>.wav
      mix.wav
  - manifest.jsonl tracks all samples with status, split, path.
  - Public API: from musicgen import generate_batch, Config, BatchResult

Consumer: dobidu/midi_file_manager
  - MidiManager: scan_directory, search, filter, export.
  - midi_integration.py: adapter API for ML pipelines, feature export.
  - Shares JSON-db + filter-operator pattern (field__gte, field__contains, etc).

Goal: Add a post-generation "index" step that:
  1. Takes a musicgen dataset_root as input.
  2. Feeds the generated MIDI files into MidiManager.
  3. Enriches each MIDI entry with musicgen's sample.json metadata (tempo, key, time_sig, split, musicality_score) so the index is searchable by musical parameters.
  4. Optionally exports a merged ML feature CSV (MIDI features + musicgen annotations).
  5. Lives either as a new `musicgen index-midi` CLI command or as a standalone adapter script.

Please:
  1. Read both repos to understand current APIs and data schemas.
  2. Discuss where the integration boundary should live (new CLI command vs adapter script vs library function).
  3. Write a GSD-style CONTEXT + PLAN.
  4. Execute with TDD.
```

---

## Integration 3 — musicgen → audio_sample_manager (LOWER PRIORITY)

### Problem being solved

musicgen produces WAV stems (4 layers) and a final mix per sample. audio_sample_manager can index these with librosa-extracted metadata (BPM, key, timbre) and make them searchable/filterable. Most useful if you want to query the generated audio library alongside external sample libraries, or feed the stems into audio_sample_manager's CompositionEngine for remixing.

### What the integration looks like

- Post-batch step: `musicgen index-audio --dataset ./dataset --out ./audio_db.json`
- Calls `SampleManager` on the stems and/or mix files.
- Enriches entries with musicgen's `sample.json` ground-truth metadata (avoids re-extracting BPM/key that musicgen already knows).
- Weakest fit of the three — musicgen's `manifest.jsonl` already covers most of what audio_sample_manager would add. Most useful for cross-library queries.

### GSD prompt to use in a fresh session

```
I want to spec and plan a downstream integration between two of my repos:

Producer: dobidu/layered_music_gen (musicgen v0.1.0)
  - generate_batch(Config) writes WAV stems (4 layers) + mix + sample.json per sample.
  - sample.json has ground-truth BPM, key, time_sig, split — no need to re-extract.
  - manifest.jsonl provides a flat index of all samples.

Consumer: dobidu/audio_sample_manager
  - SampleManager: index/search/filter audio samples with librosa-extracted metadata.
  - SampleSelector: compatible-sample selection by key/BPM/layer.
  - CompositionEngine: assemble samples into tracks.
  - sample_integration.py: adapter API for external tools.

Goal: Add a post-generation step that indexes musicgen-produced WAV files into
audio_sample_manager, preferring musicgen's ground-truth metadata over librosa re-extraction
where available (BPM, key, time_sig are already known; timbre features still need librosa).

Key question to resolve in design: where is the value? musicgen already indexes via
manifest.jsonl. The integration is only valuable if: (a) you want to query musicgen stems
alongside external sample libraries in one index, or (b) you want to use CompositionEngine
to remix the generated stems. Clarify which use case drives the design before speccing.

Please:
  1. Read both repos to understand current APIs and data schemas.
  2. Discuss the value proposition and narrow the scope to one primary use case.
  3. Write a GSD-style CONTEXT + PLAN only if the use case justifies the work.
  4. Execute with TDD if proceeding.
```

---

## Suggested sequencing

1. **Merge PR2** → v0.1.0 on main.
2. **soundfont_manager integration** (v0.2 scope) — improves generation quality immediately.
3. **midi_file_manager integration** (v0.2 scope) — adds downstream ML pipeline value.
4. **audio_sample_manager integration** (v0.3 or skip) — reassess value after #2 and #3 land.

## Key invariant to preserve across all integrations

musicgen's determinism contract: same `(global_seed, sample_index)` → bit-identical output. Any integration that touches soundfont selection or generation must preserve this. Integrations that are purely post-generation (indexing) do not affect it.
