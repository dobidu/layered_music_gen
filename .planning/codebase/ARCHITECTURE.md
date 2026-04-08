# Architecture

## Pattern

**Procedural pipeline / monolithic script.** A single orchestrator (`music_gen.py`) walks a fixed sequence of generation → rendering → mixing → scoring stages. There is no class hierarchy for the generators themselves; everything is module-level functions sharing global JSON/text configuration. Two helper modules wrap reusable analysis logic in classes (`enhanced_duration_validator.DurationValidator`, `musicality_score.MusicalityAnalyzer`).

The design is **stochastic-procedural**: at every stage, decisions are sampled from weighted distributions (key, tempo, time signature, swing, soundfonts, FX chains, layer inclusion). Outputs from one stage become inputs to the next.

## Pipeline stages

The end-to-end run is driven by `generate_song(id)` at `music_gen.py:1117`, which calls `create_song(...)` and then `mix_and_save(...)`. Stages:

1. **Parameter randomization** — `generate_random_key`, `generate_random_tempo`, `generate_random_time_signature`, `generate_random_swing` (see `music_gen.py:903` onward). Weighted ranges encode real-world Spotify key/tempo distributions.
2. **Song structure** — `generate_song_arrangement()` reads `song_structures.json` and yields a list of part names (intro/verse/chorus/bridge/outro). `generate_song_measures` and `validate_measures` produce per-part measure counts and per-part time signatures, then loop until valid.
3. **MIDI component generation** — for each unique part, four generators run (`music_gen.py:1095`+):
   - `generate_chord_progression(...)` — picks a chord pattern from `chord_patterns.txt`, validates against the time signature.
   - `generate_melody(...)` — Markov-style melodic line over the chord progression.
   - `generate_bassline(...)` — bass line keyed to chords + melody contour.
   - `generate_beat(...)` — drum pattern from `beat_roll_patterns_<sig>.txt`, with swing offset applied.
   Each writes a `.mid` file under a song-named output directory.
4. **Audio rendering (per part)** — `mix_and_save` (`music_gen.py:758`) renders each MIDI through FluidSynth using a randomly chosen soundfont per layer (`get_random_sound_font('sf/<layer>')`). Four sequential `FluidSynth(...).midi_to_audio(...)` calls per part (`music_gen.py:828, 831, 834, 837`).
5. **FX + mixing** — `apply_fx_to_layer(wav, board)` runs the WAV through a `pedalboard.Pedalboard` built by `generate_pedalboard(<layer>_fx.json)`. Per-layer volume and panning come from `levels.json`. Layer inclusion is probabilistic per part using `inst_probabilities.json`. The four (or fewer) layers are overlaid into a single `AudioSegment` mix per part.
6. **Concatenation + scoring** — Part mixes are concatenated into the final song WAV; intermediate per-part WAVs are deleted (`music_gen.py:898`). `musicality_score.MusicalityAnalyzer` computes audio-domain quality metrics (tempo, harmony, rhythm, timbre, noise) — informational only; nothing acts on the score.

## Layers / abstractions

There is **no formal layering**. Conceptually:

| Layer | Where |
|---|---|
| Configuration | `*.json`, `*.txt` pattern files at repo root |
| Music theory primitives | `music21` (imported via wildcard at `music_gen.py:2`) |
| MIDI authoring | `midiutil.MIDIFile` inside each `generate_*` function |
| Validation | `enhanced_duration_validator.DurationValidator` |
| Audio render | `midi2audio.FluidSynth` |
| Audio FX | `pedalboard` chains built from JSON specs |
| Audio assembly | `pydub.AudioSegment` |
| Quality scoring | `musicality_score.MusicalityAnalyzer` (uses `librosa`) |

The boundary between "generator" and "renderer" is the per-part MIDI file on disk. The boundary between "renderer" and "mixer" is the per-layer WAV file on disk. Both are real files, not in-memory handoffs.

## Data flow

```
JSON/TXT configs ──┐
                   ▼
generate_song ──► create_song ──► [per part: chord → melody → bass → beat] ──► .mid files
                                                                                  │
                                                                                  ▼
                          mix_and_save ──► FluidSynth render ──► .wav files
                                                                       │
                                                                       ▼
                                             pedalboard FX → pydub overlay → part mix
                                                                                  │
                                                                                  ▼
                                                              concat parts → song .wav
                                                                                  │
                                                                                  ▼
                                                              MusicalityAnalyzer (info)
```

## Entry points

- **Script entry:** the bottom of `music_gen.py:1158-1161` — a hardcoded `for i in range(1): generate_song(i)` loop. There is no CLI, no `if __name__ == '__main__'`, no argument parsing.
- **Public function:** `generate_song(id)` at `music_gen.py:1117`.
- **Standalone utilities:**
  - `beat_anotator.py` — beat-timing annotation utility, runnable separately.
  - `enhanced_duration_validator.py` — validator class, importable.
  - `musicality_score.py` — scorer class, imported by `music_gen.py:15`.

## Key abstractions

- **`DurationValidator`** (`enhanced_duration_validator.py`) — encapsulates time-signature-aware note duration logic. The only proper class abstraction in the generator pipeline.
- **`NoteValue`** enum (`enhanced_duration_validator.py`) — typed note durations.
- **`MusicalityAnalyzer`** (`musicality_score.py`) — wraps `librosa` calls for tempo/harmony/rhythm/timbre/noise scoring.
- **Pattern files** are the de-facto extension surface: adding a time signature means adding `beat_roll_patterns_<sig>.txt` plus several code edits.

## Concurrency

`from multiprocessing import Pool, cpu_count` is imported at `music_gen.py:13` but **never used**. All rendering is sequential.
