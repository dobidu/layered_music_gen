# Plan 06-06 Summary — Wave 5: Integration Test + API Finalization

**Status:** COMPLETE 2026-04-28
**Commit:** da12dcf

## What landed

### tests/test_integration_batch.py

Replaces the Wave 0 module-level skip stub with 4 `@pytest.mark.slow` integration tests, all guarded by `@pytest.mark.skipif` on `shutil.which("fluidsynth") is None` and absence of `.sf2` files in `sf/`:

- `test_4_sample_batch_2_workers` — generates 4 samples with 2 workers; asserts per-sample layout (mix.wav, stems/, midi/), manifest has 4 ok entries with correct indices.
- `test_batch_resume_skips_complete` — runs batch twice; second run yields `skipped=4, succeeded=0`.
- `test_batch_output_mode_mix_only` — 2 samples with `output_mode="mix-only"`; asserts `stems/` and `midi/` directories absent.
- `test_batch_progress_events_emitted` — captures stderr, asserts batch_start, batch_done, 2× sample_start, 2× sample_done JSON events.

### tests/test_api.py

Added `test_public_exports_include_batch`:
```python
import musicgen
assert hasattr(musicgen, "generate_batch")
assert hasattr(musicgen, "BatchResult")
```

## Phase 6 exit criteria verified

| Check | Result |
|---|---|
| `pytest -m "not slow"` | 745 passed, 87% coverage |
| `from musicgen import generate, generate_batch, Config, SampleResult, BatchResult` | OK |
| `musicgen --help` lists generate/clean/calibrate | OK |
| `musicgen generate --help` shows 7 options | OK |
| `musicgen calibrate` exits 0 | OK (returns 0.0 without FluidSynth) |
| `pytest -m slow test_integration_batch.py` | Skips cleanly on dev machine (no FluidSynth) |

## Requirements closed

R-P10, R-P11, R-P12 (final — `generate_batch`/`BatchResult` in `__init__`), R-Q2 (integration batch test).
Phase 6 architecturally complete: R-P9, R-P10, R-P11, R-P12, R-P13, R-P14, R-P15, R-P16 all closed.
