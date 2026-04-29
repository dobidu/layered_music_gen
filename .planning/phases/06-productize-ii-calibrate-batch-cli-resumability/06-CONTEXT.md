# Phase 6: Productize II — FluidSynth calibration, batch generation, CLI, resumability - Context

**Gathered:** 2026-04-28 (auto mode — recommended defaults applied per gray area)
**Status:** Ready for planning

<domain>
## Phase Boundary

Turn the Phase 5 single-sample `generate()` primitive into a **production-grade batch generator with a full CLI**, FluidSynth pre-roll calibration, parallel workers, resume/retry, and output-mode control. Phase 5 already solved determinism, seed discipline, the writer, and the manifest abstraction; Phase 6 builds on those to cover parallel execution at scale.

Six concrete capabilities land:

1. **FluidSynth pre-roll calibration (R-P9).** `src/musicgen/calibrate.py` measures FluidSynth's startup silence by synthesizing a single-note MIDI, loading the output WAV, and finding the first non-silent frame. The offset is cached at `.musicgen/fluidsynth_preroll.json` in the project root. Applied to all beat-time annotations before `sample.json` is written. Recorded in `sample.json` as `pre_roll_offset_seconds`. `musicgen calibrate` CLI command re-runs measurement explicitly.

2. **Batch generation (R-P10, R-P12).** `src/musicgen/batch.py` — `generate_batch(config) -> BatchResult` via `ProcessPoolExecutor(max_workers=config.workers or os.cpu_count())`. Each worker child function receives `(global_seed, sample_index, config)` and calls `musicgen.generate(config_for_index)` — no shared mutable state. `generate_batch` is exported from `musicgen.__init__`.

3. **Resume + failure isolation (R-P11, R-P16).** Before submitting a sample to the pool, `batch.py` checks `ManifestWriter.is_sample_complete` — skip if complete, retry if failed or untouched. Per-sample `generate()` already isolates failure (Phase 5 D-24 try/except); the batch `as_completed` loop catches the `Future`'s exception and appends a failed manifest entry. Batch does not abort on a single failure.

4. **Full CLI rewrite (R-P13).** `src/musicgen/cli.py` replaces the Phase 3 stub with a real `typer` app. Three commands: `generate` (count, out, seed, workers, output-mode, verbose/quiet), `clean --failed` (remove dirs whose manifest entry is `status: failed` with no sentinel), and `calibrate` (explicit pre-roll recompute). Config is populated via `Config.load(cli_overrides={...})`.

5. **Output-mode flag (R-P14).** `--output-mode` selects `full` / `mix-only` / `stems-only` / `midi-only`. Threaded through `Config.output_mode` field → `writer.write_sample` → conditional file writes. Annotator's relative-path dict reflects only the written files.

6. **Structured JSON progress logs (R-P15).** Batch runs emit JSON lines to `stderr` for each batch event: `batch_start`, `sample_start`, `sample_done`, `batch_done`. Workers append their `sample_start` / `sample_done` events; the main process emits `batch_start` / `batch_done` with summary stats.

After Phase 6: `musicgen generate --count 4 --out /tmp/ds --seed 42 --workers 2` produces a valid 4-sample dataset. Re-running the same command skips all 4. `musicgen calibrate` measures and caches the pre-roll offset. `musicgen clean --failed` removes dead directories.

**This phase does NOT:** implement cloud/distributed generation, add new musical vocabulary (v0.2), add sharded directory layout (v0.2+ only), change the FluidSynth subprocess model, add HTTP API or web UI (explicit anti-features in PROJECT.md), or replace the soundfont pool (broadening is v0.2). The `workers` field in `Config` is no longer reserved — it is consumed this phase. Phase 7 handles README refresh, CI hardening, and the canonical 32-sample acceptance run.

</domain>

<decisions>
## Implementation Decisions

### Module split + file inventory

- **D-44:** Six file changes this phase under `src/musicgen/`:
  - `src/musicgen/calibrate.py` (new) — FluidSynth pre-roll measurement. Surface: `measure_preroll(cfg: Config) -> float` (seconds), `load_preroll(cfg: Config) -> float` (reads cache or measures), `save_preroll(cfg: Config, offset_s: float) -> None`, `PREROLL_CACHE_PATH` module constant. Gracefully returns `0.0` when FluidSynth is absent (same skip pattern as `renderer.py` FLUIDSYNTH_VERSION fallback).
  - `src/musicgen/batch.py` (new) — batch runner. Surface: `@dataclass(frozen=True) BatchResult` (fields: D-55), `generate_batch(config: Config) -> BatchResult`. No class — pure functions + frozen dataclass.
  - `src/musicgen/cli.py` (rewrite) — replace Phase 3 stub. Three typer commands: `generate`, `clean`, `calibrate`.
  - `config.py` (extend) — add `output_mode: str = "full"` field + `count: int = 1` field (R-P12 calls it `count` in the requirements). `output_mode` validated against the `{"full", "mix-only", "stems-only", "midi-only"}` set in `__post_init__`.
  - `src/musicgen/writer.py` (extend) — `write_sample` gets an `output_mode: str = "full"` parameter; conditional file writes per D-57.
  - `src/musicgen/__init__.py` (extend) — add `generate_batch` and `BatchResult` to `__all__`.

- **D-45:** `annotator.py` is NOT modified. The annotator produces a full annotation dict regardless of `output_mode`; the writer then writes only the requested files and the writer reuses the annotation dict's path fields to fill `sample.json`. The `sample.json` `stems` / `midi` dicts will only list the files that were actually written — the path rewrite step in `writer._rewrite_paths_relative` becomes output-mode-aware (D-57).

- **D-46:** `api.py` is minimally modified: (a) pass `output_mode=config.output_mode` to `writer.write_sample`; (b) pass `pre_roll_offset_s=calibrate.load_preroll(config)` into annotator kwargs so `pre_roll_offset_seconds` is no longer `None` (closes R-P9). No other changes to `api.py`'s pipeline shape.

### OutputMode representation

- **D-47:** `output_mode` is a **plain `str`** (Literal-like, NOT an Enum). Rationale: `Config` is a plain dataclass; using a `Literal["full", "mix-only", "stems-only", "midi-only"]` type hint provides IDE checking with zero runtime overhead. An Enum would require import of a custom type in `config.py`, add a conversion step in the CLI, and complicate JSON serialization. `str` wins on simplicity for Python 3.10+. Validation fires in `Config.__post_init__` with an explicit `ValueError` listing the valid set.
- **D-48:** **Default `output_mode = "full"`** — write everything (stems + midi + mix + annotation). The other modes reduce disk usage at the cost of some outputs (e.g., `mix-only` writes only `mix.wav` + `sample.json`). The default is the richest mode so a fresh run never silently omits expected files.
- **D-49:** **`config.count: int = 1`** — number of samples for `generate_batch`. Default 1 so `generate_batch(Config(global_seed=42))` is equivalent to generating index 0 only. CLI surfaces this as `--count N`. `api.generate()` ignores `count` (single-sample, always uses `config.sample_index`); `batch.py` iterates `range(config.count)`.

### Calibrate module design (R-P9)

- **D-50:** **Measurement algorithm**: synthesize a 1-note MIDI (middle C, 1 beat, 120 BPM, 4/4 time — always the same) via `FluidSynth.midi_to_audio` using the first available soundfont in any layer. Load the resulting WAV via `scipy.io.wavfile.read`; convert to float32; find `first_nonsilent = np.argmax(np.abs(signal) > SILENCE_THRESHOLD)`. Pre-roll offset = `first_nonsilent / sample_rate`. If `np.all(np.abs(signal) <= SILENCE_THRESHOLD)` (entirely silent — FluidSynth absent or soundfont missing), return `0.0`. `SILENCE_THRESHOLD = 1e-4` (−80 dBFS normalized float32, ~20× the floor noise; tight enough to skip genuine digital silence, loose enough to not miss a quiet attack). Typical FluidSynth pre-roll is 50–200 ms; anything > 1 s is treated as an error and logged.
- **D-51:** **Cache location**: `.musicgen/fluidsynth_preroll.json` relative to `config.project_root`. Full path: `os.path.join(config.project_root, ".musicgen", "fluidsynth_preroll.json")`. `.musicgen/` is a hidden dot-dir (analogous to `.git/` or `.tox/`) — keepable, gitignore-able, machine-specific. Format: `{"fluidsynth_version": "...", "offset_seconds": 0.123, "measured_at": "ISO-8601"}`. Module constant `PREROLL_CACHE_DIR = ".musicgen"`, `PREROLL_CACHE_FILE = "fluidsynth_preroll.json"`.
- **D-52:** **Cache validity**: the cached entry is valid if `fluidsynth_version` matches the current `renderer.FLUIDSYNTH_VERSION`. If the FluidSynth binary was updated, `load_preroll` re-measures automatically (i.e., `FLUIDSYNTH_VERSION` mismatch → `measure_preroll` → `save_preroll` → return new offset). This is cheap (one FluidSynth invocation, ~0.5 s) and ensures the offset is always current.
- **D-53:** **Pre-roll application in api.py**: `annotator.annotate(...)` already accepts `pre_roll_offset_seconds` as a kwarg (set to `None` in Phase 5). Phase 6 passes `pre_roll_offset_s = calibrate.load_preroll(config)`. The annotator stores the raw value in `sample.json`. Beat times are NOT shifted in the annotator (the annotator stays pure). Instead, `writer.write_sample` applies the offset to `beat_times` and `downbeat_times` fields in the annotation copy before writing `sample.json`: each time list becomes `[t - offset for t in times if t - offset >= 0]`. This keeps the annotator pure while making the persisted `beat_times` correct.
- **D-54:** **Graceful absence**: if FluidSynth is absent (`FLUIDSYNTH_VERSION == "unknown"` or no soundfonts available), `measure_preroll` returns `0.0` without writing the cache file. `load_preroll` returns `0.0` if the cache file doesn't exist. Effect: `pre_roll_offset_seconds = 0.0` in `sample.json` — no crash, no data corruption, just no calibration. The CLI `musicgen calibrate` command prints a friendly message explaining that FluidSynth was not found.

### BatchResult dataclass

- **D-55:** **`BatchResult` fields** (frozen=True, slots=True):
  ```python
  @dataclass(frozen=True, slots=True)
  class BatchResult:
      total: int           # total samples requested (config.count)
      succeeded: int       # samples with status == "ok"
      failed: int          # samples with status == "failed"
      skipped: int         # samples that existed and were skipped (resume)
      results: Tuple[SampleResult, ...]  # one per submitted sample (not skipped)
      duration_seconds: float  # wall-clock time for the whole batch
  ```
  `results` does not include skipped samples (they produced no `SampleResult`). `total = succeeded + failed + skipped`. The `Tuple` type is used (not `List`) because `BatchResult` is frozen.

### Worker seeding strategy

- **D-56:** **Worker function pattern**: the worker child function is a module-level (picklable) function, NOT a lambda or inner function:
  ```python
  def _worker(global_seed: int, sample_index: int, config: Config) -> SampleResult:
      """ProcessPoolExecutor worker — runs in a child process."""
      return generate(
          Config(
              **{**config.__dict__,
                 "global_seed": global_seed,
                 "sample_index": sample_index,
              }
          )
      )
  ```
  `Config` is a plain dataclass with only stdlib-typed fields; it is picklable by default. The worker imports `generate` from `musicgen.api` at module level (the import happens in the child process — no shared state from parent). Per R-P7 bullet 4: child workers seed their own `random.Random` instances on entry via `generate` → `make_rngs(derive_sample_seed(...))` — the parent process's RNG state is never inherited.

### ProcessPoolExecutor setup

- **D-57:** **`generate_batch` implementation pattern**:
  - `max_workers = config.workers or os.cpu_count() or 1` — fallback chain handles `None` and pathological `os.cpu_count() == None` (documented in stdlib: returns None on some platforms).
  - `multiprocessing.get_context("spawn")` is used explicitly when creating the executor on Linux to avoid fork-safety issues with `FluidSynth` subprocesses and `threading.Lock` from pedalboard. Pattern: `ProcessPoolExecutor(max_workers=max_workers, mp_context=multiprocessing.get_context("spawn"))`. On macOS 3.8+ this is already the default; on Linux fork is the default and is unsafe here.
  - `concurrent.futures.as_completed(futures)` drives progress reporting — each future completes asynchronously.
  - **Resume check before submit**: call `ManifestWriter.is_sample_complete(config.dataset_root, sample_index)` before `executor.submit()`. If complete, increment `skipped` counter and do NOT submit. This is the O(1) per-sample resume check (reads one file existence on disk).
  - **Retry logic**: a failed sample (no `sample.json` but possibly a manifest `failed` entry) is NOT skipped — it is submitted to the executor as normal. The failure manifest entry from a previous run becomes stale once the retry succeeds (`last-status-wins` semantics per Phase 5 D-15).
  - `futures_map: Dict[Future, int]` maps each future to its `sample_index` for progress reporting.

### ManifestWriter lock for multiprocessing

- **D-58:** **Shared manager lock**: `generate_batch` creates one `multiprocessing.Manager` and one `multiprocessing.Manager().Lock()` that is shared across all workers. The `ManifestWriter` instance is created once in the **main process** and is NOT passed to workers (it is not picklable as-is because it holds the lock proxy). Instead, each worker creates its own `ManifestWriter` instance but uses a `multiprocessing.Manager().Lock()` proxy passed via the `Config` — specifically, `config` gets a special `_manifest_lock` attribute that is set only during batch runs (type: `Optional[Any]`; normally `None` for single-sample). Simpler alternative: **workers do NOT append to the manifest during the run**; the main process appends after each `Future` completes (reading the result from the returned `SampleResult`). This is the chosen approach — it avoids passing a Manager lock proxy through pickling and matches the `as_completed` event loop naturally.
  - **Decision**: workers do NOT write to manifest. Main process writes manifest entries as `as_completed` yields results. `generate()` single-sample still writes manifest (Phase 5 behavior unchanged). `generate_batch` calls `generate()` in workers which writes the manifest — but waits for the worker's manifest append to complete before returning. **Actually**: Phase 5's `generate()` already appends to manifest using `threading.Lock()`. In a `ProcessPoolExecutor` (separate processes), each `generate()` call in a worker opens `manifest.jsonl` in append mode with a `threading.Lock()` that is LOCAL to that child process. Two workers can race on the file. Fix: pass the lock via `ManifestWriter(dataset_root, lock=multiprocessing_lock)`. This requires either (a) a Manager lock passed in config, or (b) main-process-only manifest writes. **Selected: option (b) — main-process-only manifest writes**. Workers do not append to manifest. After each `Future` completes in `as_completed`, the main process appends the manifest entry (success or failure). `generate()` is called with a `ManifestWriter` whose `append` is a no-op placeholder during batch (achieved via passing a `_no_op_manifest=True` flag or by subclassing — simpler: batch's `_worker` function replaces the manifest append in `generate()` by monkey-patching... No. Cleanest: **`generate()` accepts an optional `manifest_writer` parameter**; batch passes a sentinel that does nothing; main process owns all appends. Modify `api.generate` signature: `generate(config, *, manifest_writer=None)` — if `None`, creates one (single-sample behavior unchanged). Batch passes a `_NullManifestWriter` that silently discards appends. The main process uses its own `ManifestWriter(config.dataset_root)` to append after each future completes.
  - **`_NullManifestWriter`**: a private class in `batch.py` that implements the same `append()` interface but does nothing. Only needed inside `batch.py`.

### Structured JSON progress logs (R-P15)

- **D-59:** **Log event schema** — all events are written to `sys.stderr` as JSON lines (not via `logging`; this is structured progress output, not debug logs). Format:
  ```json
  {"event": "batch_start",  "total": 100, "workers": 4, "seed": 42, "timestamp": "ISO-8601"}
  {"event": "sample_start", "sample_index": 0, "seed": 42, "worker_pid": 12345, "timestamp": "ISO-8601"}
  {"event": "sample_done",  "sample_index": 0, "status": "ok", "duration_s": 12.3, "musicality_score": 0.87, "timestamp": "ISO-8601"}
  {"event": "batch_done",   "total": 100, "succeeded": 98, "failed": 2, "skipped": 0, "duration_s": 310.4, "timestamp": "ISO-8601"}
  ```
  `sample_start` is emitted by the **main process** before submitting the future (not the worker; workers don't write progress). `sample_done` is emitted by the main process after `as_completed` returns a result. This avoids multiprocess stderr interleaving. `worker_pid` is populated from the `SampleResult` or is set to 0 if unavailable.
- **D-60:** **Progress log function**: `_log_event(event: str, **kwargs) -> None` private helper in `batch.py` — builds the JSON dict with `"event"` + `"timestamp": datetime.now(timezone.utc).isoformat()` + all kwargs, `json.dumps(sort_keys=True)`, writes to `sys.stderr` with `\n`. No buffering issue (stderr is line-buffered in Python).

### CLI design (R-P13)

- **D-61:** **Three typer commands**:
  - `musicgen generate --count N --out DIR --seed S [--workers W] [--output-mode MODE] [-v/-q]`
  - `musicgen clean --failed [--out DIR]` — scans `manifest.jsonl` for `status: failed` entries, removes the corresponding per-sample dirs (if they exist) and optionally removes the manifest lines (append a `cleaned` status entry instead of rewriting).
  - `musicgen calibrate [--out DIR]` — runs `calibrate.measure_preroll(cfg)` + `calibrate.save_preroll(cfg, offset)`, prints the measured offset. `--out` defaults to the current working directory (no effect on cache location since cache is always under `config.project_root`).
- **D-62:** **CLI verbosity**: `-v` sets `logging.DEBUG` (or repeatable: `-vv` for even more detail — keep it simple, one `-v` for DEBUG, default INFO). `-q` sets `logging.WARNING`. Implemented via `logging.basicConfig(level=...)` inside the command functions. Do NOT use typer's built-in `--verbose` flag (it's a bool; `-v` as a count flag is more idiomatic for logging control).
- **D-63:** **Config construction in CLI**: `Config.load(cli_overrides={"dataset_root": out_dir, "global_seed": seed, "count": count, "workers": workers, "output_mode": output_mode})`. All CLI-provided values are passed; `None` values are excluded (the `Config.load` method already skips None overrides per Phase 2 D-01/D-02 pattern).
- **D-64:** **`musicgen clean --failed` implementation**: reads `manifest.jsonl` line-by-line, collects all unique `sample_index` values with `status: "failed"` AND NO corresponding `status: "ok"` line (last-status-wins: if a sample has both a failed and an ok entry, it is NOT cleaned). For each collected index, calls `shutil.rmtree(sample_dir, ignore_errors=True)`. Does NOT modify `manifest.jsonl` (append-only discipline from Phase 5 D-15). Reports count of cleaned dirs to stdout.
- **D-65:** **`--output-mode` choices**: typer validates against the literal list `["full", "mix-only", "stems-only", "midi-only"]` via `typer.Option(..., help="...", show_choices=True)`. Invalid choices produce a typer error before the config is constructed.

### OutputMode routing in writer (R-P14)

- **D-66:** **`write_sample` output-mode parameter**: `write_sample(..., output_mode: str = "full") -> dict`. Conditional writes:
  - `"full"`: write midi + stems + mix + `sample.json` (current behavior).
  - `"mix-only"`: write only `mix.wav` + `sample.json` (skip `midi/` and `stems/`).
  - `"stems-only"`: write `stems/*.wav` + `sample.json` (skip `midi/`; skip mix WAV).
  - `"midi-only"`: write `midi/*.mid` + `sample.json` (skip `stems/` and mix WAV).
  - `sample.json` is ALWAYS written regardless of output mode (it is the sentinel and annotation store). The `stems` / `midi` keys in `sample.json` reflect only the written files (empty dicts `{}` when skipped).
- **D-67:** **Paths dict from `write_sample`**: the returned paths dict includes only keys for actually-written files. `SampleResult` fields `stem_paths` / `midi_paths` will be empty dicts `{}` for omitted categories. Downstream code (tests, manifest entries) should handle empty dicts gracefully.

### Integration test (R-Q2)

- **D-68:** **`tests/test_integration_batch.py`** — `@pytest.mark.slow` + FluidSynth + sf2 guards (Phase 4 D-30 pattern). Test: `generate_batch(Config(global_seed=42, count=4, dataset_root=tmp_path, workers=2))` → assert `BatchResult.total == 4`, `BatchResult.succeeded == 4`, `BatchResult.failed == 0`, `BatchResult.skipped == 0`. Verify `manifest.jsonl` has 4 `"ok"` entries. Then run again → assert `BatchResult.skipped == 4` (all resume). Also tests `output_mode="mix-only"` in a separate run: assert no `stems/` or `midi/` dirs in the sample dir.

### Test strategy (new test files this phase)

- **D-69:** **`tests/test_output_mode.py`** — tests `write_sample` with each output mode. No FluidSynth: uses synthetic WAVs + MIDIs via `numpy` + `mido` (matching Phase 5 writer test pattern). 12 assertions: full has 10 files, mix-only has 2 files, stems-only has 6 files (4 stems + sample.json + no mix), midi-only has 5 files. Also tests that `sample.json` is always written and that `stem_paths` / `midi_paths` in returned dict are empty when omitted.
- **D-70:** **`tests/test_calibrate.py`** — uses `tmp_path`. Mocks `FluidSynth.midi_to_audio` (same pattern as `tests/test_renderer.py`). Tests: `measure_preroll` returns a float; `save_preroll` creates cache file with correct JSON; `load_preroll` reads and returns cached value; version mismatch triggers re-measure; FluidSynth absent returns 0.0. 8 assertions, zero FluidSynth on CI.
- **D-71:** **`tests/test_batch.py`** — uses `tmp_path` + mocked `generate` (monkeypatch). Tests: `generate_batch` calls `generate` N times; skips completed samples; retries failed; `BatchResult` fields are correct; `max_workers` fallback works; structured log events emitted to stderr. 10 assertions, no FluidSynth.
- **D-72:** **`tests/test_cli.py`** — uses `typer.testing.CliRunner`. Tests: `musicgen generate --count 1 --out DIR --seed 42` exits 0; `musicgen calibrate` exits 0 (mocked); `musicgen clean --failed` exits 0 on empty manifest; invalid `--output-mode` exits non-zero; `-v` and `-q` flags accepted. 8 assertions, no FluidSynth.
- **D-73:** **`tests/test_integration_batch.py`** — per D-68, `@pytest.mark.slow`.
- **D-74:** **Extend `tests/test_config.py`**: add tests for new `Config` fields: `output_mode` default is `"full"`, invalid output_mode raises ValueError, `count` default is `1`, `count` propagates through `Config.load`.

### Wave structure (6 plans in 5 waves)

- **D-75:** Wave breakdown:
  - **Wave 0 — 06-01**: Test scaffolding stubs (5 new test files with `pytest.mark.skip` stubs + extend test_config.py + update test_no_bare_random_in_package.py expected_present for `calibrate.py` and `batch.py`).
  - **Wave 1 — 06-02**: Config extension (`output_mode`, `count`) + OutputMode routing in writer + annotator no-op + `api.py` minimal changes. Closes the `pre_roll_offset_seconds = None` stub by hooking calibrate.
  - **Wave 2 — 06-03**: `calibrate.py` + `tests/test_calibrate.py`.
  - **Wave 3 — 06-04**: `batch.py` + `BatchResult` + `tests/test_batch.py`.
  - **Wave 4 — 06-05**: Full CLI rewrite (replace stub `cli.py`) + `tests/test_cli.py`.
  - **Wave 5 — 06-06**: Integration test + API finalization (`generate_batch` + `BatchResult` in `__init__.py`).

</decisions>

<deferred>
## Deferred Ideas

- **Sharded directory layout** (`dataset/<hex>/<id>/`) — v0.2+ per REQUIREMENTS "Out of scope". 6-digit index handles 1M with headroom; redesign at 100k+.
- **Cloud / distributed generation** — explicit PROJECT.md anti-feature. All parallelism is local-process `ProcessPoolExecutor`.
- **ML dataset exporters** (HF Datasets, WebDataset, LMDB) — v0.? per REQUIREMENTS.
- **Compressed annotations** (`sample.json.gz`) — v0.2+ if disk pressure emerges. Plain JSON wins on tooling compatibility at 10k scale.
- **Custom output-mode profiles** (e.g., `annotation-only` — write only `sample.json`, skip all audio) — v0.2+. The four modes in R-P14 are sufficient for v0.1.
- **GPU-accelerated synthesis** — out of scope; FluidSynth is CPU-bound.
- **Resume from interrupted batch mid-sample** — impossible by design: `sample.json` is the sentinel; a mid-sample interruption leaves no sentinel, so Phase 6 retries the whole sample. Fine at 10k scale.
- **Interactive progress bar** (tqdm / rich progress) — CLI aesthetics; deferred to Phase 7 polish pass (R-Q1 README + UX). Phase 6 ships JSON stderr lines which can be consumed by any progress renderer.
- **`musicgen calibrate --benchmark` mode** (run N measurements, report mean + stddev) — out of scope for v0.1; the single-measurement + cache approach is sufficient.
- **`--output-mode` per-layer control** (e.g., write beat MIDI only) — v0.2+. The four top-level modes cover the primary use cases.
- **Aggregated failure report file** (separate JSON summary next to `manifest.jsonl`) — Phase 7 polish. The `batch_done` JSON log event provides the numbers; Phase 7 can write a summary `.json` if needed.
- **`musicgen generate` parallelism for a single sample** (intra-sample thread count) — Phase 4 already uses `ThreadPoolExecutor(max_workers=4)` in `renderer.py` for per-layer rendering; no further intra-sample parallelism is needed for v0.1.

</deferred>

---

*Phase: 06-productize-ii-calibrate-batch-cli-resumability*
*Context gathered: 2026-04-28*
