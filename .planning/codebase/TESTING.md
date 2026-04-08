# Testing

## Status

**There is no automated test suite.** No `tests/` directory, no `test_*.py` files, no `pytest.ini`, no `tox.ini`, no `unittest` imports anywhere in the project. CI is not configured.

`requirements.txt` does not include `pytest`, `unittest2`, `nose`, `hypothesis`, or any other testing framework.

## How "testing" currently happens

The de-facto test loop is:

1. Edit `music_gen.py`.
2. Run `python music_gen.py` (the script's bottom-of-file `for i in range(1): generate_song(i)` at `music_gen.py:1158-1161` triggers one full song generation).
3. Listen to the resulting WAV.
4. Inspect `MusicalityAnalyzer` output for tempo / harmony / rhythm / timbre / noise scores.

This is a **manual, end-to-end smoke test only**. There is no isolation of units, no fixtures, no assertion framework.

## In-code validation (not tests)

Several runtime validation hooks exist and act as the only safety net:

- `verify_pattern_for_time_signature(...)` — `music_gen.py:22`
- `verify_beat_pattern(...)` — `music_gen.py:42`
- `validate_measures(...)` — called in a `while True` loop in `generate_song` (`music_gen.py:1135`) to retry until valid.
- `enhanced_duration_validator.DurationValidator` — time-signature-aware note duration enforcement.
- `musicality_score.MusicalityAnalyzer` — post-generation quality scoring (descriptive only; results never trigger regeneration or fail a build).

These are validators, not tests — they run inside the generation pipeline at runtime.

## Coverage gaps (untested behavior)

Effectively everything. Highest-risk untested areas:

- `generate_chord_progression(...)` — pattern selection, time signature compatibility
- `generate_melody(...)` — Markov / scale-degree logic
- `generate_bassline(...)` — coordination with chords + melody
- `generate_beat(...)` — pattern selection, swing application
- `mix_and_save(...)` (`music_gen.py:758`) — the longest function in the codebase, ~143 lines, 4 sequential FluidSynth calls per part, probabilistic layer inclusion
- `apply_fx_to_layer(...)` — pedalboard application
- `generate_pedalboard(...)` — JSON → Pedalboard construction
- `DurationValidator` (entire class)
- `MusicalityAnalyzer` (entire class) — has multiple `except:` blocks at `musicality_score.py:66, 94, 173, 205, 239` that swallow errors silently

## Mocking / fixtures

None. Any future test suite would need to mock or stub:

- `FluidSynth(...).midi_to_audio(...)` — slow, requires soundfonts on disk
- `pydub.AudioSegment.from_wav(...)` — requires real audio files
- `librosa.load(...)` — same
- File system reads of pattern `.txt` files and `*.json` configs
- `random` — to make stochastic generators deterministic

A pragmatic split would be:

| Layer | Test type |
|---|---|
| Pattern parsers, validators, sampler functions | Pure unit tests with seeded `random` |
| MIDI generators | Unit tests asserting note count / range / rhythm structure on the produced `MIDIFile` object |
| FluidSynth render | Integration test, opt-in (slow) |
| Mixing pipeline | Integration test with tiny fixture WAVs |
| `MusicalityAnalyzer` | Unit tests with short fixture WAVs and tolerance ranges |

## Recommendations (for future work)

1. **Add `pytest` to `requirements.txt`** and create a `tests/` directory.
2. **Make `music_gen.py` importable** — wrap the bottom-of-file loop in `if __name__ == '__main__':` so importing it for tests doesn't trigger a full song generation.
3. **Seed `random`** in test fixtures so stochastic generators are deterministic.
4. **Replace bare `except:` blocks** in `musicality_score.py` (lines 66, 94, 173, 205, 239) with logged, narrow exceptions before testing — currently failures are invisible.
5. **Start with the validators** (`verify_pattern_for_time_signature`, `verify_beat_pattern`, `validate_measures`, `DurationValidator`) — they are pure functions / class methods, no audio dependencies, easy wins.
