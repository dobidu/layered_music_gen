# Determinism Fixtures

SHA-256 goldens captured on a pinned-FluidSynth host. See CONTEXT.md D-28 and the test at `tests/test_determinism_golden.py`.

## Files (populated on first --regen-goldens run)

| File | Provenance | Version-gated? |
|------|------------|---------------|
| `expected_mix.sha256` | `sha256(<sample_dir>/mix.wav)` | **Yes** — skipped-xfail on FluidSynth version mismatch (R-P8) |
| `expected_midi_beat.sha256` | `sha256(<sample_dir>/midi/beat.mid)` | No — FluidSynth-independent |
| `expected_midi_melody.sha256` | `sha256(<sample_dir>/midi/melody.mid)` | No |
| `expected_midi_harmony.sha256` | `sha256(<sample_dir>/midi/harmony.mid)` | No |
| `expected_midi_bassline.sha256` | `sha256(<sample_dir>/midi/bassline.mid)` | No |
| `expected_sample.sha256` | `sha256(<sample_dir>/sample.json)` | No |
| `fluidsynth_version.txt` | First line of `fluidsynth --version` at capture time | — |

## Capture parameters

- `global_seed = 1`
- `sample_index = 0`
- `dataset_root = <pytest tmp_path>`
- Config defaults for everything else (split_ratios = 80/10/10, sum_of_stems_epsilon = 1e-3)
- `importlib.metadata.version("musicgen")` MUST resolve to `"0.1.0"` (run `pip install -e .` first — `"0.1.0+uninstalled"` would poison the sample.json hash per RESEARCH Pitfall 4)

## Regeneration

```
.venv/bin/pip install -e ".[dev]"    # ensure musicgen_version resolves to "0.1.0", NOT "+uninstalled"
.venv/bin/pytest -m slow --regen-goldens tests/test_determinism_golden.py
```

The `--regen-goldens` flag (registered in `tests/conftest.py` by Plan 05-01) turns every assertion into a capture operation. After regen, commit the 7 files (`expected_*.sha256` + `fluidsynth_version.txt`).

Use regeneration when:
- Intentionally changing RNG draw order (Phase 5+ decision).
- Upgrading the pinned FluidSynth binary.
- Fixing a sum-of-stems threshold that invalidated a prior golden.

## Interpretation

- If **MIDI** or **sample.json** hashes mismatch: our code has a non-determinism regression — debug in-process (run `TestSameProcessStability` first — it catches most of these without FluidSynth).
- If **mix.wav** hash mismatches AND `fluidsynth_version.txt` matches: FluidSynth on your platform produces different audio for the same MIDI+sf2. Rare — re-probe with a different sf2 or report upstream.
- If **mix.wav** hash mismatches AND `fluidsynth_version.txt` differs: expected behavior. R-P8 guarantees bit-identity only under the pinned binary. The test xfails cleanly.

## Why two classes?

- `TestDeterminismGoldens` is the "expensive oracle" — proves cross-run, cross-machine bit-identity under pinned FluidSynth. Runs under `-m slow`, needs the binary.
- `TestSameProcessStability` is the "cheap watchdog" (D-30) — runs in every `pytest` invocation, catches our-code non-determinism, does not need FluidSynth. Any stray `datetime.now()` in the annotator or writer fails this test the same day it's introduced.
