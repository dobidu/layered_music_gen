# Plan 06-02 Summary — Wave 1: Config Extension + OutputMode Routing + Pre-roll Hook

**Status:** COMPLETE 2026-04-28
**Commits:** d34a368 (RED), 83a69b6 (GREEN)

## What landed

### Config extensions (config.py)

Two new Phase 6 fields with full three-layer precedence (CLI > env > defaults):

- `output_mode: str = "full"` — validated in `__post_init__` against `{"full", "mix-only", "stems-only", "midi-only"}`.
- `count: int = 1` — validated `>= 1` in `__post_init__`.
- Env-var overrides: `MUSICGEN_OUTPUT_MODE`, `MUSICGEN_COUNT` added to `Config.load()`.

### writer.py output_mode routing

`write_sample(...)` extended with `output_mode: str = "full"` and `pre_roll_offset_s: float = 0.0` parameters.

Three routing booleans derived at call entry:
```python
_write_mix   = output_mode in ("full", "mix-only")
_write_stems = output_mode in ("full", "stems-only")
_write_midi  = output_mode in ("full", "midi-only")
```

Sum-of-stems assertion only runs when `_write_stems and _write_mix`.

`_apply_preroll_offset(anno_copy, offset_s)` helper: copies annotation, clips negative beat/downbeat times after subtracting offset.

### api.py calibrate hook

`_run_pipeline` loads pre-roll offset via:
```python
pre_roll_offset_s = calibrate.load_preroll(cfg.project_root)  # fallback 0.0
```
Applied to both `annotator.annotate(pre_roll_offset_seconds=...)` and `writer.write_sample(pre_roll_offset_s=...)`.

### annotator.py

Added `pre_roll_offset_seconds` kwarg, recorded in `sample.json` for auditability (R-P9).

## Requirements closed

R-P14 (output mode flag). R-P9 hook wired; measurement deferred to Plan 06-03.

## Outcome

745 → 700 fast tests (net: replaced stubs + new tests). Zero bare `random.*` violations.
