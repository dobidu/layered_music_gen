---
status: partial
phase: 04-renderer-mixer-annotator-beats-extraction
source: [04-VERIFICATION.md]
started: 2026-04-19T17:30:00Z
updated: 2026-04-19T17:30:00Z
---

## Current Test

[awaiting human testing]

## Tests

### 1. E2E full-generation pipeline on a FluidSynth-equipped machine
expected: `pytest -m slow tests/test_integration_full_generation.py::TestFullGenerationPipeline::test_one_part_full_pipeline -v` exits 0 — produces 4 stem WAVs + 1 mix WAV + 4 MIDI files; annotation JSON has all Phase-4 R-P4 fields non-None (key/mode/tempo_bpm/time_signature/swing/song_arrangement/per-part chord_progression+active_layers+soundfonts+fx_params+beat_times+downbeat_times/musicality_score/duration_seconds/fluidsynth_version/paths) and Phase-5 TBD fields present as None (seed, musicgen_version, split, pre_roll_offset_seconds).
result: [pending]

### 2. MIDI reproducibility across two seeded runs
expected: `pytest -m slow tests/test_integration_full_generation.py::TestMidiReproducibility::test_same_seed_produces_same_midi -v` exits 0 — beat/melody/harmony/bassline MIDI bytes bit-identical between run #1 and run #2 with `_rng = random.Random(42)`.
result: [pending]

### 3. Smoke test — `python music_gen.py` reaches annotator stage
expected: Running `python music_gen.py` on a dev machine with FluidSynth + sf2 pool should produce the full generation chain through to `sample.json` emission. Environmental failure at `librosa` musicality scoring is acceptable — but the renderer → mixer → beats → annotator chain itself must complete, and `<dataset-root>/<song_name>/sample.json` must be written with all Phase-4 R-P4 fields populated.
result: [pending]

## Summary

total: 3
passed: 0
issues: 0
pending: 3
skipped: 0
blocked: 0

## Gaps

**Environmental gap (not a code gap):** This dev machine lacks the FluidSynth binary and has empty `sf/<layer>/` soundfont directories. All three items require that environment to execute. The code path is exercised by 504 unit/integration tests that mock away the binary, but the real subprocess call + real audio render can only be verified on an equipped machine. Same limitation as Phase 3 closure — tracked here so it surfaces in `/gsd-progress` and `/gsd-audit-uat` until the user runs the slow-marker suite on a dev machine.

Once the user runs the two slow-marker tests (items 1 and 2) and the smoke test (item 3) successfully, this UAT can be marked resolved.
