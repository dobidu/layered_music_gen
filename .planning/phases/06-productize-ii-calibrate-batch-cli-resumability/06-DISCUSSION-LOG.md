# Phase 6: Productize II — Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in 06-CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-28
**Phase:** 06-productize-ii-calibrate-batch-cli-resumability
**Mode:** `--auto` (Claude selected recommended defaults for every gray area)
**Areas discussed:** OutputMode representation, calibrate cache location, worker pool strategy, CLI structure, manifest write strategy, progress log approach, test approach

---

## OutputMode representation

| Option | Description | Selected |
|--------|-------------|----------|
| `Literal["full", "mix-only", "stems-only", "midi-only"]` type hint on a `str` field | Simple, zero runtime overhead, JSON-serializable, validates in `__post_init__` | ✓ |
| `enum.Enum` subclass `OutputMode` | Strong typing, IDE support, but requires conversion in CLI + JSON serialization | |
| Plain string, no type hint, no validation | Simplest possible, but silently accepts invalid values | |

**Auto-selected:** Literal-hinted `str` with `__post_init__` validation. Matches Phase 5's pattern of config-level validation (D-27 split_ratios validator). Zero extra dependencies. D-47.

---

## Calibrate cache location

| Option | Description | Selected |
|--------|-------------|----------|
| `.musicgen/fluidsynth_preroll.json` under `config.project_root` | Machine-specific hidden dir; gitignore-able; analogous to `.git/`/`.tox/` | ✓ |
| `<dataset_root>/.fluidsynth_preroll.json` | Tied to dataset; re-measured if dataset moves | |
| User home `~/.musicgen/fluidsynth_preroll.json` | Global; shared across projects but wrong if multiple FluidSynth installs | |
| `config.py` top-level alongside source | Source dir pollution; not gitignored by default | |

**Auto-selected:** `.musicgen/` under project root. Follows the "hidden dot-dir for machine-specific runtime data" convention. Easy to gitignore, stable across dataset moves. D-51.

---

## Cache validity strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Re-measure when `fluidsynth_version` in cache != current FLUIDSYNTH_VERSION | Cheap (one FluidSynth invocation ~0.5s); always correct | ✓ |
| Time-based TTL (e.g. re-measure if cache is older than 7 days) | Fragile; FluidSynth can be upgraded without us noticing | |
| Never re-measure automatically; force with `musicgen calibrate` | Can use stale offset if binary is upgraded | |
| Re-measure on every batch run | Adds ~0.5s overhead per batch; overkill | |

**Auto-selected:** Version-gated re-measure. Same pattern as Phase 5's `fluidsynth_version.txt` golden-fixture gate. D-52.

---

## Worker pool strategy

| Option | Description | Selected |
|--------|-------------|----------|
| `ProcessPoolExecutor` with `mp_context=multiprocessing.get_context("spawn")` | Fork-safe; avoids FluidSynth/pedalboard subprocess corruption | ✓ |
| `ProcessPoolExecutor` with default fork context | Faster startup but unsafe with FluidSynth subprocesses on Linux | |
| `ThreadPoolExecutor` | Does not parallelize CPU-bound FluidSynth rendering (GIL not released by Python code) | |
| `multiprocessing.Pool` | Lower-level; `concurrent.futures` as_completed pattern is simpler | |

**Auto-selected:** `ProcessPoolExecutor` with explicit `spawn` context. R-P10 states ProcessPoolExecutor; spawn is the safe default on Linux where fork is the platform default. D-57.

---

## Manifest write strategy during batch

| Option | Description | Selected |
|--------|-------------|----------|
| Main process only: worker returns SampleResult; main appends manifest after as_completed | Simple; no inter-process lock needed | ✓ |
| Workers append using multiprocessing.Manager().Lock() passed via config | Concurrent writes; more complex; Manager adds subprocess overhead | |
| Workers append using file-level fcntl.flock() | Platform-dependent; non-portable on Windows | |
| Workers append using a threading.Lock() (wrong for multiprocessing) | Incorrect; lock is per-process | |

**Auto-selected:** Main-process-only manifest writes. The `as_completed` loop provides natural serialization. No Manager overhead. Workers use `_NullManifestWriter` that discards appends. D-58.

---

## Progress log destination

| Option | Description | Selected |
|--------|-------------|----------|
| JSON lines to `sys.stderr` | Standard for structured progress (doesn't pollute stdout data pipe); separate from Python `logging` | ✓ |
| Python `logging` with a structured JSON handler | Correct for debug/info, but intermixed with other log messages | |
| JSON lines to `stdout` | Pollutes stdout (user may pipe stdout to a file or tool) | |
| JSON lines to a progress file in `dataset_root` | Useful but adds file I/O; stderr is simpler for v0.1 | |

**Auto-selected:** JSON lines to `sys.stderr`. This is the standard pattern for CLI progress (e.g., `rclone`, `git`, structured CLI tools). Easily redirected: `musicgen generate ... 2>progress.jsonl`. D-59.

---

## `sample_start` event emitter (worker vs. main)

| Option | Description | Selected |
|--------|-------------|----------|
| Main process emits `sample_start` before submit, `sample_done` after as_completed returns | Simple; no inter-process communication needed | ✓ |
| Worker emits `sample_start` on entry | Race condition on stderr; requires worker-side stderr write | |

**Auto-selected:** Main-process-only progress log events. Matches main-process-only manifest write decision. D-59.

---

## CLI command structure

| Option | Description | Selected |
|--------|-------------|----------|
| Three separate commands: `generate`, `clean`, `calibrate` under one app | Matches R-P13 literal text | ✓ |
| Two commands: `generate` (covers calibrate via flag) + `clean` | Combines concerns; harder to document | |
| One command with subcommands as positional args | Non-idiomatic for typer | |
| `generate` + `generate-batch` as separate commands | Redundant; `--count` on one command is simpler | |

**Auto-selected:** Three named commands per R-P13. D-61.

---

## `clean --failed` manifest handling

| Option | Description | Selected |
|--------|-------------|----------|
| Append a "cleaned" status entry; never rewrite manifest | Append-only discipline from Phase 5 D-15 | ✓ |
| Rewrite manifest removing failed lines | Violates append-only contract; O(N) rewrite | |
| Delete manifest entirely and let next run rebuild | Destructive; loses history | |

**Auto-selected:** Append-only; `clean` does NOT modify `manifest.jsonl`. The cleaned dirs no longer have `sample.json`, so they naturally become "retry" candidates on the next run. D-64.

---

## Config pickling for ProcessPoolExecutor

| Option | Description | Selected |
|--------|-------------|----------|
| Pass full `Config` to worker; reconstruct per-sample config inside worker | Config is a plain dataclass with stdlib-typed fields — picklable by default | ✓ |
| Serialize config to JSON dict, pass dict, reconstruct inside worker | Extra serialization step; no benefit | |
| Pass only primitive fields (global_seed, sample_index, dataset_root) | Worker cannot access FX file paths etc. | |

**Auto-selected:** Pass full `Config`; per-sample `Config` created inside worker with `global_seed` and `sample_index` overrides. D-56.

---

## Test isolation for batch tests

| Option | Description | Selected |
|--------|-------------|----------|
| Monkeypatch `musicgen.api.generate` to a fast stub in test_batch.py | Fast; no FluidSynth dependency; tests batch logic only | ✓ |
| Use real generate() with mocked FluidSynth (like test_api.py fast path) | Complex; batch test should focus on orchestration, not pipeline | |
| Only have the slow integration test for batch | Insufficient; batch logic (resume, failure isolation) needs fast coverage | |

**Auto-selected:** Monkeypatched `generate` in `test_batch.py`; real `generate` only in `test_integration_batch.py`. D-71.

---

## OutputMode filtering: annotator vs. writer

| Option | Description | Selected |
|--------|-------------|----------|
| Writer filters writes; annotator produces full dict always; writer adjusts sample.json paths dict | Keeps annotator pure (Phase 4 D-14 invariant) | ✓ |
| Annotator receives output_mode; omits path fields for missing files | Annotator becomes output-mode-aware; harder to test in isolation | |
| Both annotator and writer are mode-aware | Redundant filtering; two places to maintain | |

**Auto-selected:** Writer-only filtering. Annotator stays a pure function. D-45, D-66.
