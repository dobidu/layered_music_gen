# Determinism Fixtures

SHA-256 goldens captured on a pinned-FluidSynth host. See CONTEXT.md D-28.

**Files that land in Wave 5 (Plan 05-06):**
- `expected_mix.sha256` — SHA-256 of `mix.wav` for `Config(global_seed=1, sample_index=0)`.
- `expected_midi_beat.sha256`, `expected_midi_melody.sha256`, `expected_midi_harmony.sha256`, `expected_midi_bassline.sha256` — per-layer MIDI hashes.
- `expected_sample.sha256` — SHA-256 of the canonical `sample.json` bytes (D-23 `sort_keys=True, indent=2` ordering).
- `fluidsynth_version.txt` — first line of `fluidsynth --version` when the fixtures were captured.

**Regeneration:**
```
.venv/bin/pytest -m slow --regen-goldens tests/test_determinism_golden.py
```

Use when intentionally changing RNG draw order or when upgrading the pinned FluidSynth binary. **MIDI + sample.json hashes must pass unconditionally** — only `mix.wav` SHA-256 is version-guarded per R-P8.

Captured from: `pip install -e .` environment (so `importlib.metadata.version("musicgen") == "0.1.0"`, not `"0.1.0+uninstalled"`). See RESEARCH Pitfall 4.
