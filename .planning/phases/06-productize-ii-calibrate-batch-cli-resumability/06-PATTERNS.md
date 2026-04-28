# Phase 6: Productize II — Pattern Map

**Mapped:** 2026-04-28
**Files analyzed:** 12 (source + test)
**Primary analogs:** Phase 5 api.py, Phase 4 renderer.py, Phase 4 plan file format

---

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `src/musicgen/calibrate.py` | utility (FluidSynth measurement + cache) | request-response | `src/musicgen/renderer.py` (FluidSynth wrapper, FLUIDSYNTH_VERSION absent-graceful pattern) | exact |
| `src/musicgen/batch.py` | orchestrator (parallel runner) | batch | `src/musicgen/api.py` (`generate` is the primitive wrapped by `generate_batch`) | exact |
| `src/musicgen/cli.py` | interface (typer commands) | request-response | Phase 3 stub `cli.py` (being replaced) | exact |
| `config.py` | config (add `output_mode`, `count`) | — | `config.py` existing Phase 5 extension | exact |
| `src/musicgen/writer.py` | service (extend with output_mode) | batch | writer.py current (Phase 5 D-04/D-66) | exact |
| `src/musicgen/__init__.py` | config (add `generate_batch`, `BatchResult`) | — | `__init__.py` Phase 5 rewrite | exact |
| `src/musicgen/api.py` | orchestrator (add calibrate hook) | request-response | api.py current | exact |
| `tests/test_output_mode.py` | test (writer output-mode routing) | batch | `tests/test_writer.py` (synthetic WAV + mido fixtures) | exact |
| `tests/test_calibrate.py` | test (calibrate module) | request-response | `tests/test_renderer.py` (mocked FluidSynth) | exact |
| `tests/test_batch.py` | test (batch orchestration, mocked generate) | batch | `tests/test_manifest.py` (concurrency / orchestration) | role-match |
| `tests/test_cli.py` | test (typer CliRunner) | request-response | no prior CLI tests — new pattern | role-match |
| `tests/test_integration_batch.py` | test (E2E batch, @pytest.mark.slow) | batch | `tests/test_integration_full_generation.py` | exact |

---

## Pattern 1: ProcessPoolExecutor Batch Worker

**File:** `src/musicgen/batch.py`

Workers MUST be module-level functions (not lambdas or inner functions) to be picklable by the `spawn` context. The worker creates its own per-sample `Config` with overridden `global_seed` and `sample_index`.

```python
# VERBATIM PATTERN — executors copy this exactly.
from __future__ import annotations

import json
import logging
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from config import Config
from musicgen.api import SampleResult, generate
from musicgen.manifest import ManifestWriter

logger = logging.getLogger(__name__)

_LAYERS = ("beat", "melody", "harmony", "bassline")


class _NullManifestWriter:
    """Drop-in replacement for ManifestWriter that silently discards appends.

    Used by batch workers so the main process owns all manifest writes
    (avoids inter-process lock coordination — D-58).
    """
    def append(self, entry: dict) -> None:  # noqa: D102
        pass

    @staticmethod
    def is_sample_complete(dataset_root: str, sample_index: int, pad: int = 6) -> bool:
        from musicgen.manifest import ManifestWriter as _Real
        return _Real.is_sample_complete(dataset_root, sample_index, pad)


def _worker(global_seed: int, sample_index: int, config: Config) -> SampleResult:
    """ProcessPoolExecutor worker — runs in a child process (spawn context).

    Receives a full Config; overrides global_seed and sample_index.
    Calls generate() with a _NullManifestWriter so the main process
    owns all manifest appends (D-58).
    """
    worker_config = Config(
        **{k: v for k, v in config.__dict__.items()
           if k not in ("global_seed", "sample_index")},
        global_seed=global_seed,
        sample_index=sample_index,
    )
    return generate(worker_config, manifest_writer=_NullManifestWriter())


@dataclass(frozen=True, slots=True)
class BatchResult:
    """Aggregate result of a generate_batch() call (D-55)."""
    total: int
    succeeded: int
    failed: int
    skipped: int
    results: Tuple[SampleResult, ...]
    duration_seconds: float


def generate_batch(config: Config) -> BatchResult:
    """Run config.count samples in parallel via ProcessPoolExecutor (D-56/D-57).

    Resume logic: samples with an existing sample.json are skipped (D-57).
    Failure isolation: individual failures are logged; batch continues (R-P16).
    """
    start_t = time.monotonic()
    count = config.count
    max_workers = config.workers or os.cpu_count() or 1
    mp_ctx = multiprocessing.get_context("spawn")

    _log_event("batch_start", total=count, workers=max_workers,
               seed=config.global_seed)

    manifest_writer = ManifestWriter(config.dataset_root)
    results: List[SampleResult] = []
    succeeded = failed = skipped = 0

    with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_ctx) as pool:
        futures_map: Dict = {}

        for sample_index in range(count):
            if ManifestWriter.is_sample_complete(config.dataset_root, sample_index):
                skipped += 1
                logger.info("Sample %d: skipping (complete)", sample_index)
                continue

            _log_event("sample_start", sample_index=sample_index,
                       seed=config.global_seed)
            future = pool.submit(
                _worker, config.global_seed, sample_index, config,
            )
            futures_map[future] = sample_index

        for future in as_completed(futures_map):
            sample_index = futures_map[future]
            try:
                result: SampleResult = future.result()
            except Exception as exc:  # noqa: BLE001
                logger.exception("Sample %d future raised: %s", sample_index, exc)
                # Build a synthetic failed result for the manifest
                from musicgen.seeds import derive_sample_seed
                seed = derive_sample_seed(config.global_seed, sample_index)
                result = SampleResult(
                    sample_index=sample_index, seed=seed,
                    sample_dir=os.path.join(config.dataset_root,
                                            f"{sample_index:06d}"),
                    sample_json_path="", mix_path="",
                    stem_paths={}, midi_paths={}, split="",
                    status="failed", musicality_score=0.0,
                    duration_seconds=0.0,
                )

            # Main process appends manifest (D-58).
            _append_manifest(manifest_writer, result, config)

            _log_event("sample_done", sample_index=result.sample_index,
                       status=result.status,
                       duration_s=result.duration_seconds,
                       musicality_score=result.musicality_score)

            if result.status == "ok":
                succeeded += 1
            else:
                failed += 1
            results.append(result)

    duration = time.monotonic() - start_t
    _log_event("batch_done", total=count, succeeded=succeeded,
               failed=failed, skipped=skipped, duration_s=duration)

    return BatchResult(
        total=count, succeeded=succeeded, failed=failed, skipped=skipped,
        results=tuple(results), duration_seconds=duration,
    )


def _append_manifest(
    manifest_writer: ManifestWriter,
    result: SampleResult,
    config: Config,
) -> None:
    """Append one manifest entry from a SampleResult."""
    from datetime import datetime, timezone
    entry = {
        "sample_index": result.sample_index,
        "seed": config.global_seed,
        "sample_seed": result.seed,
        "status": result.status,
        "split": result.split,
        "path": (f"{result.sample_index:06d}/sample.json"
                 if result.status == "ok" else ""),
        "musicality_score": result.musicality_score,
        "duration_seconds": result.duration_seconds,
        "wrote_at": datetime.now(timezone.utc).isoformat(),
    }
    if result.status == "failed":
        entry["error"] = ""  # no traceback available from future result
    try:
        manifest_writer.append(entry)
    except Exception as exc:  # noqa: BLE001
        logger.error("Manifest append failed: %s", exc)


def _log_event(event: str, **kwargs) -> None:
    """Write a JSON progress event to stderr (D-59/D-60)."""
    payload = {"event": event,
               "timestamp": datetime.now(timezone.utc).isoformat(),
               **kwargs}
    sys.stderr.write(json.dumps(payload, sort_keys=True) + "\n")
    sys.stderr.flush()
```

---

## Pattern 2: OutputMode Filtering in writer.write_sample

**File:** `src/musicgen/writer.py`

`write_sample` receives `output_mode: str = "full"` and conditionally skips file categories. `sample.json` is ALWAYS written (sentinel invariant).

```python
# VERBATIM PATTERN — add to write_sample signature and body.
# (Existing parameters unchanged; output_mode added as keyword-only.)

def write_sample(
    dataset_root: str,
    sample_index: int,
    annotation: dict,
    final_wav: str,
    stem_paths_by_part: dict,
    midi_paths_by_part: dict,
    arrangement: list,
    tempo: int,
    part_durations_s: list,
    *,
    fluidsynth_version: str = "unknown",
    split: str = "train",
    sum_of_stems_epsilon: float = 1e-3,
    output_mode: str = "full",
) -> dict:
    ...
    _WRITE_MIX   = output_mode in ("full", "mix-only")
    _WRITE_STEMS = output_mode in ("full", "stems-only")
    _WRITE_MIDI  = output_mode in ("full", "midi-only")

    os.makedirs(sample_dir, exist_ok=True)

    written_midi_paths: dict = {}
    if _WRITE_MIDI:
        os.makedirs(midi_dir, exist_ok=True)
        for layer in _LAYERS:
            midi_out = os.path.join(midi_dir, f"{layer}.mid")
            _concat_layer_midis(midi_paths_by_part, layer, midi_out, arrangement, tempo, part_durations_s)
            written_midi_paths[layer] = midi_out

    written_stem_paths: dict = {}
    if _WRITE_STEMS:
        os.makedirs(stems_dir, exist_ok=True)
        for layer in _LAYERS:
            stem_out = os.path.join(stems_dir, f"{layer}.wav")
            _concat_layer_stems(stem_paths_by_part, layer, stem_out)
            written_stem_paths[layer] = stem_out

    written_mix_path = ""
    if _WRITE_MIX:
        shutil.copy2(final_wav, mix_out)
        written_mix_path = mix_out

    # Sum-of-stems assertion only when both stems and mix are written.
    if _WRITE_STEMS and _WRITE_MIX:
        _assert_sum_of_stems(written_mix_path, written_stem_paths, sum_of_stems_epsilon)

    # Path rewrite: only include written files in sample.json.
    anno_copy = _rewrite_paths_relative(
        annotation, written_mix_path, written_stem_paths, written_midi_paths,
    )
    # ... write sample.json sentinel as before ...
```

---

## Pattern 3: Pre-roll Offset Application in writer

**File:** `src/musicgen/writer.py` (inside `_rewrite_paths_relative` or a new helper)

The pre-roll offset is applied to `beat_times` and `downbeat_times` in the annotation copy BEFORE `sample.json` is written. The annotator stores the raw measured offset as `pre_roll_offset_seconds`; the writer shifts the time lists.

```python
# VERBATIM PATTERN — apply inside write_sample after annotation deep-copy.
def _apply_preroll_offset(anno_copy: dict, offset_s: float) -> dict:
    """Shift beat_times and downbeat_times by -offset_s (in-place on copy).

    Times < offset_s are dropped (they would be negative after shift).
    If offset_s == 0.0, returns anno_copy unchanged (fast path).
    """
    if offset_s == 0.0:
        return anno_copy
    for key in ("beat_times", "downbeat_times"):
        if key in anno_copy and isinstance(anno_copy[key], list):
            anno_copy[key] = [
                round(t - offset_s, 6)
                for t in anno_copy[key]
                if t - offset_s >= 0.0
            ]
    return anno_copy
```

The call site in `write_sample`:
```python
anno_copy = copy.deepcopy(annotation)
anno_copy = _apply_preroll_offset(anno_copy, pre_roll_offset_s)
anno_copy = _rewrite_paths_relative(anno_copy, ...)
```

---

## Pattern 4: Calibrate Module (FluidSynth Pre-roll Measurement)

**File:** `src/musicgen/calibrate.py`

```python
# VERBATIM PATTERN — calibrate.py structure.
"""Calibrate — FluidSynth pre-roll measurement and cache (R-P9, D-50/D-51/D-52/D-54).

Measures the startup silence FluidSynth inserts before audio output begins.
Result is cached in <project_root>/.musicgen/fluidsynth_preroll.json.
Applied to beat_times/downbeat_times in writer.write_sample.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone

import mido
import numpy as np

from musicgen.renderer import FLUIDSYNTH_VERSION

logger = logging.getLogger(__name__)

PREROLL_CACHE_DIR  = ".musicgen"
PREROLL_CACHE_FILE = "fluidsynth_preroll.json"
SILENCE_THRESHOLD  = 1e-4   # -80 dBFS normalized float32
MAX_REASONABLE_OFFSET_S = 1.0  # > 1s offset is suspicious; log warning


def _cache_path(project_root: str) -> str:
    return os.path.join(project_root, PREROLL_CACHE_DIR, PREROLL_CACHE_FILE)


def load_preroll(project_root: str) -> float:
    """Return cached pre-roll offset (seconds), re-measuring if version changed.

    Returns 0.0 if FluidSynth is absent or no soundfonts are available.
    """
    path = _cache_path(project_root)
    if os.path.exists(path):
        try:
            with open(path) as f:
                data = json.load(f)
            if data.get("fluidsynth_version") == FLUIDSYNTH_VERSION:
                return float(data["offset_seconds"])
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.warning("Pre-roll cache invalid (%s); re-measuring.", exc)
    # Cache miss or version mismatch — measure.
    return measure_and_save_preroll(project_root)


def measure_and_save_preroll(project_root: str) -> float:
    """Measure FluidSynth pre-roll and write cache. Returns offset in seconds."""
    offset_s = measure_preroll(project_root)
    save_preroll(project_root, offset_s)
    return offset_s


def measure_preroll(project_root: str) -> float:
    """Synthesize a 1-note MIDI and find the first non-silent sample."""
    if FLUIDSYNTH_VERSION == "unknown":
        logger.info("FluidSynth absent — pre-roll offset set to 0.0")
        return 0.0

    sf2_path = _find_any_soundfont(project_root)
    if sf2_path is None:
        logger.warning("No soundfonts found — pre-roll offset set to 0.0")
        return 0.0

    with tempfile.TemporaryDirectory(prefix="musicgen-calib-") as tmpdir:
        midi_path = os.path.join(tmpdir, "calib.mid")
        wav_path  = os.path.join(tmpdir, "calib.wav")
        _write_calibration_midi(midi_path)
        try:
            from midi2audio import FluidSynth as _FS
            _FS(sf2_path).midi_to_audio(midi_path, wav_path)
        except Exception as exc:
            logger.warning("FluidSynth calibration render failed: %s", exc)
            return 0.0

        try:
            from scipy.io import wavfile as _wf
            rate, data = _wf.read(wav_path)
        except Exception as exc:
            logger.warning("Calibration WAV read failed: %s", exc)
            return 0.0

        signal = data.astype(np.float32) / 32768.0
        if signal.ndim > 1:
            signal = signal[:, 0]  # mono from first channel
        nonsilent = np.where(np.abs(signal) > SILENCE_THRESHOLD)[0]
        if len(nonsilent) == 0:
            logger.info("Calibration WAV entirely silent — offset set to 0.0")
            return 0.0
        offset_s = float(nonsilent[0]) / rate
        if offset_s > MAX_REASONABLE_OFFSET_S:
            logger.warning(
                "Pre-roll offset %.3f s > %.1f s — suspicious; using as-measured.",
                offset_s, MAX_REASONABLE_OFFSET_S,
            )
        logger.info("Measured FluidSynth pre-roll: %.3f s", offset_s)
        return offset_s


def save_preroll(project_root: str, offset_s: float) -> None:
    """Write pre-roll offset to cache file."""
    path = _cache_path(project_root)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "fluidsynth_version": FLUIDSYNTH_VERSION,
        "offset_seconds": offset_s,
        "measured_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    logger.info("Pre-roll cache written to %s", path)


def _write_calibration_midi(path: str) -> None:
    """Write a single-note MIDI (middle C, 1 beat, 120 BPM, 4/4)."""
    mid = mido.MidiFile(type=0, ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))  # 120 BPM
    track.append(mido.Message("note_on",  channel=0, note=60, velocity=64, time=0))
    track.append(mido.Message("note_off", channel=0, note=60, velocity=0,  time=480))
    track.append(mido.MetaMessage("end_of_track", time=0))
    mid.save(path)


def _find_any_soundfont(project_root: str) -> "Optional[str]":
    """Return the path of the first .sf2 file found under sf/ in project_root."""
    sf_root = os.path.join(project_root, "sf")
    if not os.path.isdir(sf_root):
        return None
    for layer_dir in sorted(os.listdir(sf_root)):
        full_layer = os.path.join(sf_root, layer_dir)
        if not os.path.isdir(full_layer):
            continue
        for fname in sorted(os.listdir(full_layer)):
            if fname.endswith(".sf2"):
                return os.path.join(full_layer, fname)
    return None
```

---

## Pattern 5: Structured JSON Log Event

**File:** `src/musicgen/batch.py`

```python
# VERBATIM PATTERN — _log_event helper.
def _log_event(event: str, **kwargs) -> None:
    """Write a structured JSON progress event to stderr (D-59/D-60).

    Format: one JSON object per line, sorted keys, UTC timestamp.
    Caller: main process only (workers do not write progress events).
    """
    from datetime import datetime, timezone
    payload = {
        "event": event,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **kwargs,
    }
    sys.stderr.write(json.dumps(payload, sort_keys=True) + "\n")
    sys.stderr.flush()
```

Usage:
```python
_log_event("batch_start", total=100, workers=4, seed=42)
_log_event("sample_start", sample_index=0, seed=42)
_log_event("sample_done", sample_index=0, status="ok", duration_s=12.3, musicality_score=0.87)
_log_event("batch_done", total=100, succeeded=98, failed=2, skipped=0, duration_s=310.4)
```

---

## Pattern 6: CLI Command Registration

**File:** `src/musicgen/cli.py`

```python
# VERBATIM PATTERN — cli.py structure.
"""CLI — typer commands for musicgen (Phase 6, R-P13, D-61/D-62/D-63/D-64/D-65).

Commands:
  musicgen generate --count N --out DIR --seed S [--workers W] [--output-mode MODE] [-v/-q]
  musicgen clean --failed [--out DIR]
  musicgen calibrate [--out DIR]
"""
from __future__ import annotations

import logging
import shutil
import sys
from typing import Optional

import typer

from config import Config
from musicgen.batch import generate_batch
from musicgen.calibrate import load_preroll, measure_and_save_preroll

app = typer.Typer(
    help="musicgen — synthetic music dataset generator",
    no_args_is_help=True,
)

_VALID_OUTPUT_MODES = ["full", "mix-only", "stems-only", "midi-only"]


def _setup_logging(verbose: bool = False, quiet: bool = False) -> None:
    level = logging.DEBUG if verbose else (logging.WARNING if quiet else logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )


@app.command()
def generate(
    count:       int  = typer.Option(1,      "--count",       "-n", help="Number of samples to generate."),
    out:         str  = typer.Option("./dataset", "--out",    "-o", help="Dataset root directory."),
    seed:        int  = typer.Option(...,    "--seed",         "-s", help="Global seed (required)."),
    workers: Optional[int] = typer.Option(None, "--workers",  "-w", help="Parallel workers (default: cpu_count)."),
    output_mode: str  = typer.Option("full", "--output-mode", "-m",
                                     help=f"Output mode. Choices: {_VALID_OUTPUT_MODES}",
                                     show_choices=True),
    verbose:     bool = typer.Option(False, "--verbose", "-v", is_flag=True),
    quiet:       bool = typer.Option(False, "--quiet",   "-q", is_flag=True),
) -> None:
    """Generate a batch of samples."""
    _setup_logging(verbose, quiet)
    if output_mode not in _VALID_OUTPUT_MODES:
        typer.echo(f"Invalid --output-mode '{output_mode}'. Choose from: {_VALID_OUTPUT_MODES}", err=True)
        raise typer.Exit(code=1)
    cfg = Config.load(cli_overrides={
        "dataset_root": out,
        "global_seed":  seed,
        "count":        count,
        "workers":      workers,
        "output_mode":  output_mode,
    })
    result = generate_batch(cfg)
    typer.echo(
        f"Done: {result.succeeded} ok, {result.failed} failed, "
        f"{result.skipped} skipped of {result.total} total "
        f"({result.duration_seconds:.1f}s)"
    )
    if result.failed:
        raise typer.Exit(code=1)


@app.command()
def clean(
    out:    str  = typer.Option("./dataset", "--out", "-o", help="Dataset root directory."),
    failed: bool = typer.Option(False, "--failed", is_flag=True, help="Remove failed sample directories."),
) -> None:
    """Remove incomplete sample directories."""
    _setup_logging()
    if not failed:
        typer.echo("Specify --failed to remove failed sample directories.")
        raise typer.Exit(code=1)
    import json, os
    manifest_path = os.path.join(out, "manifest.jsonl")
    if not os.path.exists(manifest_path):
        typer.echo("No manifest.jsonl found — nothing to clean.")
        return
    # Collect last-status per sample_index.
    last_status: dict = {}
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                idx = entry.get("sample_index")
                if idx is not None:
                    last_status[idx] = entry.get("status", "")
            except json.JSONDecodeError:
                continue
    cleaned = 0
    for idx, status in last_status.items():
        if status == "failed":
            sample_dir = os.path.join(out, f"{idx:06d}")
            sentinel   = os.path.join(sample_dir, "sample.json")
            if os.path.exists(sample_dir) and not os.path.exists(sentinel):
                shutil.rmtree(sample_dir, ignore_errors=True)
                cleaned += 1
    typer.echo(f"Cleaned {cleaned} failed sample director{'ies' if cleaned != 1 else 'y'}.")


@app.command()
def calibrate(
    out: str = typer.Option("./dataset", "--out", "-o", help="Dataset root (unused; cache is under project_root)."),
    verbose: bool = typer.Option(False, "--verbose", "-v", is_flag=True),
) -> None:
    """Measure and cache FluidSynth pre-roll offset."""
    _setup_logging(verbose)
    cfg = Config.load()
    typer.echo("Measuring FluidSynth pre-roll offset...")
    offset_s = measure_and_save_preroll(cfg.project_root)
    typer.echo(f"Pre-roll offset: {offset_s:.3f} s")
    if offset_s == 0.0:
        typer.echo("(FluidSynth absent or no soundfonts — offset set to 0.0)")


if __name__ == "__main__":
    app()
```

---

## Pattern 7: api.py Minimal Changes for Phase 6

**File:** `src/musicgen/api.py` — two targeted edits only.

**Edit 1: Accept optional `manifest_writer` parameter (D-58)**

```python
def generate(config: Config, *, manifest_writer=None) -> SampleResult:
    """Generate one sample end-to-end (D-31).

    Args:
        config: Config with global_seed, sample_index, dataset_root, etc.
        manifest_writer: Optional ManifestWriter override. When None (default),
            a new ManifestWriter is created (single-sample behavior). Batch
            passes a _NullManifestWriter to suppress per-worker manifest writes.
    """
    ...
    # Inside generate(), replace:
    #   manifest_writer = ManifestWriter(config.dataset_root)
    # with:
    if manifest_writer is None:
        manifest_writer = ManifestWriter(config.dataset_root)
```

**Edit 2: Pass pre_roll_offset_s to annotator + writer (D-46/D-53)**

```python
# In _run_pipeline, add calibrate import + load:
from musicgen import calibrate as _calibrate
pre_roll_offset_s = _calibrate.load_preroll(_cfg.project_root)

# Pass to annotator (fills the pre_roll_offset_seconds field):
annotation = annotator.annotate(
    ...,
    pre_roll_offset_seconds=pre_roll_offset_s,
)

# Pass to writer (applies shift to beat_times/downbeat_times):
paths = writer.write_sample(
    ...,
    output_mode=_cfg.output_mode,
    pre_roll_offset_s=pre_roll_offset_s,
)
```
