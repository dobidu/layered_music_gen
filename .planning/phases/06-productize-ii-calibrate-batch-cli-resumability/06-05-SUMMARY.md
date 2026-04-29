# Plan 06-05 Summary — Wave 4: Full CLI Rewrite

**Status:** COMPLETE 2026-04-28
**Commits:** a6d7014 (RED), 004b394 (GREEN)

## What landed

### src/musicgen/cli.py (rewrite of Phase 3 stub)

Three typer commands:

```
musicgen generate --seed S [--count N] [--out DIR] [--workers W] [--output-mode M] [-v/-q]
musicgen clean --failed [--out DIR]
musicgen calibrate [-v]
```

`generate` command:
- Validates `--output-mode` against `{"full", "mix-only", "stems-only", "midi-only"}` before dispatch; exits 1 with error message on invalid value.
- Builds `Config` via `Config.load(cli_overrides={...})`.
- Calls `generate_batch(cfg)`.
- Prints summary line; exits 1 with warning if `result.failed > 0`.

`clean` command:
- Requires `--failed` flag (guards against accidental use).
- Reads `manifest.jsonl`, builds last-status-wins map per `sample_index`.
- Removes `<dataset_root>/<idx:06d>/` directories where status is `"failed"` AND `sample.json` does not exist (partial output only).

`calibrate` command:
- Calls `calibrate.measure_and_save_preroll(cfg.project_root)`.
- Prints offset; prints diagnostic if FluidSynth absent.

### src/musicgen/__init__.py

sys.path fixup moved here (was incorrectly only in `cli.py`, causing `ModuleNotFoundError: No module named 'config'` when running as entry point). Fix runs before any package imports:

```python
_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
```

`generate_batch` and `BatchResult` added to exports and `__all__`.

### Tests

`tests/test_cli.py` — 8 tests via `typer.testing.CliRunner()` (no `mix_stderr` kwarg):
- `generate` missing `--seed` exits nonzero
- `generate` invalid `--output-mode` exits nonzero
- `generate --help` exits 0 and shows all 7 options
- `generate` with mocked `generate_batch` exits 0
- `clean` without `--failed` exits nonzero
- `clean --failed` on empty dataset exits 0
- `clean --failed` removes failed partial directory
- `calibrate` with mocked pre-roll measurement exits 0

### Bugs fixed during execution

1. `CliRunner(mix_stderr=False)` TypeError — typer's CliRunner doesn't accept that kwarg; removed.
2. Double-registration of `calibrate` command — removed manual `app.command(name=...)` call; kept `@app.command(name="calibrate")` decorator only.
3. `ModuleNotFoundError: No module named 'config'` for CLI entry point — fixed by moving sys.path fixup to `__init__.py` before all imports.

## Requirements closed

R-P13 (full typer CLI with generate/clean/calibrate). R-P12 partial (batch exports in `__init__.py`).
