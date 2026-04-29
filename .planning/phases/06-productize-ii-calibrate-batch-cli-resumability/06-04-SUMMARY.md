# Plan 06-04 Summary — Wave 3: batch.py BatchResult + generate_batch

**Status:** COMPLETE 2026-04-28
**Commits:** 4bca822 (RED), bb47e62 (GREEN)

## What landed

`src/musicgen/batch.py` (~200 lines) — parallel batch generation with resumability, failure isolation, and structured progress logs.

### Public API

```python
@dataclasses.dataclass(frozen=True, slots=True)
class BatchResult:
    total: int
    succeeded: int
    failed: int
    skipped: int
    results: Tuple[SampleResult, ...]
    duration_seconds: float

def generate_batch(config: Config) -> BatchResult: ...
```

### Key design decisions

**ProcessPoolExecutor with spawn context** — avoids fork-safety issues with FluidSynth and pydub. `_make_executor(max_workers, mp_ctx)` extracted as a separate helper so tests can monkeypatch to `ThreadPoolExecutor` without needing `mp_context` kwarg.

**Resume logic** — `ManifestWriter.is_sample_complete` checked before dispatch. Samples with a `sample.json` sentinel are skipped (`result.skipped` count). Samples with a manifest entry of `status: failed` but no sentinel are retried.

**Failure isolation** — `as_completed` loop wraps each future in try/except. Failed futures → `SampleResult(status="failed")` appended to manifest; batch continues. `BatchResult.failed` tracks count.

**Manifest writes in main process** — workers use `_NullManifestWriter` (no-op). Main process calls `_append_manifest` in the `as_completed` loop for both successes and failures.

**Structured JSON progress events** on stderr via `_log_event(event, **kwargs)`:
- `batch_start` — total count, workers, dataset_root, seed
- `sample_start` / `sample_skip` — per sample
- `sample_done` — with duration and status
- `batch_done` — aggregated counts and total duration

### Tests

`tests/test_batch.py` — 10 tests using `patch_executor` fixture (ThreadPoolExecutor stand-in) + `patch_generate` fixture (fake generate stub):
- BatchResult fields and frozen assertion
- count=1 and count=3 basic batch
- Skip complete samples
- Retry failed samples
- Failure isolation (one bad sample, rest continue)
- Manifest appended after batch
- batch_start and batch_done events emitted to stderr

## Requirements closed

R-P10 (batch generation), R-P11 (resumability), R-P15 (structured logs), R-P16 (failure isolation).
Also removed `@pytest.mark.xfail` from AST meta-test (both `calibrate.py` and `batch.py` now exist).
