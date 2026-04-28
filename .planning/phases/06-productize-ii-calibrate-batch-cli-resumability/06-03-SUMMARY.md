# Plan 06-03 Summary — Wave 2: calibrate.py FluidSynth Pre-roll Measurement

**Status:** COMPLETE 2026-04-28
**Commits:** 3ea34e8 (RED), f3ebc11 (GREEN)

## What landed

`src/musicgen/calibrate.py` (201 lines) — FluidSynth pre-roll offset measurement and cache.

### Public API

```python
load_preroll(project_root: str) -> float          # returns 0.0 if no cache or FluidSynth absent
measure_and_save_preroll(project_root: str) -> float  # measures + persists
measure_preroll(project_root: str) -> float       # pure measurement, no save
save_preroll(project_root: str, offset_s: float) -> None
```

### Cache

Location: `<project_root>/.musicgen/fluidsynth_preroll.json`

```json
{
  "offset_s": 0.123456,
  "fluidsynth_version": "2.3.6",
  "measured_at": "2026-04-28T12:00:00Z"
}
```

Version gate: if `FLUIDSYNTH_VERSION` at load time differs from `fluidsynth_version` in cache, cache is stale → `load_preroll` returns 0.0 and logs a warning.

### Measurement algorithm

1. Write a 1-note calibration MIDI (`_write_calibration_midi`).
2. Render it with FluidSynth to a WAV file (`_render_calibration_midi`).
3. Walk WAV samples until abs(sample) > `SILENCE_THRESHOLD = 1e-4`; offset = sample_index / sample_rate.
4. Clamp to `[0, MAX_REASONABLE_OFFSET_S = 1.0]`.
5. Return 0.0 if FluidSynth absent (`FLUIDSYNTH_VERSION == "unknown"`) or no soundfont found.

### Soundfont search

`_find_any_soundfont(project_root)` walks `sf/*/` subdirectories and returns the first `.sf2` file found (sorted).

## Requirements closed

R-P9 (FluidSynth pre-roll measurement, cache, version gate).

## Outcome

Full suite passes. `musicgen calibrate` exits 0 on machines without FluidSynth (returns 0.0 with diagnostic message).
