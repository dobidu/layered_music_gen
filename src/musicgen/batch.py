"""Batch generation module (R-P10, R-P11, R-P12, R-P15, R-P16, D-55..D-60).

``generate_batch(config)`` runs N samples in parallel via ProcessPoolExecutor
(spawn context). Design choices:

- Workers receive only (global_seed, sample_index, config) — no shared state.
- Manifest writes happen in the main process only (via as_completed) to avoid
  inter-process lock coordination.
- Workers use _NullManifestWriter so generate() doesn't write manifest.
- Failure isolation: future.result() exceptions are caught per-sample; batch
  continues with the next sample.
- Structured JSON progress events to sys.stderr (R-P15).
"""
from __future__ import annotations

import dataclasses
import json
import logging
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from config import Config
from musicgen.api import SampleResult, generate
from musicgen.manifest import ManifestWriter
from musicgen.seeds import derive_sample_seed

logger = logging.getLogger(__name__)

_LAYERS = ("beat", "melody", "harmony", "bassline")


# ---------------------------------------------------------------------------
# Null manifest writer (D-58) — used in workers so manifest writes happen
# only in the main process after as_completed.
# ---------------------------------------------------------------------------


class _NullManifestWriter:
    """No-op manifest writer for batch worker processes (D-58)."""

    def append(self, entry: dict) -> None:
        pass

    @staticmethod
    def is_sample_complete(dataset_root: str, sample_index: int, pad: int = 6) -> bool:
        return ManifestWriter.is_sample_complete(dataset_root, sample_index, pad)


# ---------------------------------------------------------------------------
# BatchResult
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class BatchResult:
    """Result of a generate_batch() call (D-55, R-P12)."""
    total: int
    succeeded: int
    failed: int
    skipped: int
    results: Tuple[SampleResult, ...]
    duration_seconds: float


# ---------------------------------------------------------------------------
# Worker — must be a module-level function for multiprocessing picklability.
# ---------------------------------------------------------------------------


def _worker(global_seed: int, sample_index: int, config: Config) -> SampleResult:
    """Per-sample worker function — called in a subprocess (D-56).

    Each worker creates a per-sample Config via dataclasses.replace so
    it owns its own global_seed + sample_index. No parent RNG state is
    inherited (spawn context guarantees a fresh interpreter).
    """
    worker_config = dataclasses.replace(
        config, global_seed=global_seed, sample_index=sample_index,
    )
    return generate(worker_config, manifest_writer=_NullManifestWriter())


# ---------------------------------------------------------------------------
# generate_batch
# ---------------------------------------------------------------------------


def generate_batch(config: Config) -> BatchResult:
    """Generate config.count samples in parallel (R-P10, D-55..D-60).

    Resume: samples with an existing sample.json are skipped (R-P11).
    Failure isolation: individual sample errors are caught and logged (R-P16).
    Progress: JSON events to stderr (R-P15).
    """
    if config.global_seed is None:
        raise ValueError("global_seed is required for generate_batch")

    t0 = time.monotonic()
    max_workers = config.workers or os.cpu_count() or 1

    _log_event("batch_start", count=config.count, workers=max_workers,
               seed=config.global_seed, dataset_root=config.dataset_root)

    manifest_writer = ManifestWriter(config.dataset_root)
    os.makedirs(config.dataset_root, exist_ok=True)

    succeeded = 0
    failed = 0
    skipped = 0
    all_results: List[SampleResult] = []

    # Determine which samples to run.
    to_run: List[int] = []
    for idx in range(config.count):
        if ManifestWriter.is_sample_complete(config.dataset_root, idx):
            skipped += 1
            _log_event("sample_skip", index=idx)
            logger.info("Sample %d: already complete — skipping", idx)
        else:
            to_run.append(idx)

    if not to_run:
        elapsed = time.monotonic() - t0
        _log_event("batch_done", total=config.count, succeeded=0,
                   failed=0, skipped=skipped, elapsed_s=round(elapsed, 3))
        return BatchResult(
            total=config.count,
            succeeded=0, failed=0, skipped=skipped,
            results=tuple(all_results), duration_seconds=elapsed,
        )

    mp_ctx = multiprocessing.get_context("spawn")

    with _make_executor(max_workers, mp_ctx) as pool:
        futures: Dict = {}
        for idx in to_run:
            _log_event("sample_start", index=idx, seed=config.global_seed)
            fut = pool.submit(_worker, config.global_seed, idx, config)
            futures[fut] = idx

        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                result: SampleResult = fut.result()
                succeeded += 1
                all_results.append(result)
                _append_manifest(manifest_writer, result, config)
                _log_event("sample_done", index=idx, status="ok",
                           duration_s=result.duration_seconds)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Sample %d failed: %s", idx, exc)
                failed += 1
                sample_seed = derive_sample_seed(config.global_seed, idx)
                failed_result = SampleResult(
                    sample_index=idx,
                    seed=sample_seed,
                    sample_dir=os.path.join(config.dataset_root, f"{idx:06d}"),
                    sample_json_path="",
                    mix_path="",
                    stem_paths={},
                    midi_paths={},
                    split="",
                    status="failed",
                    musicality_score=0.0,
                    duration_seconds=0.0,
                )
                all_results.append(failed_result)
                _append_manifest(manifest_writer, failed_result, config,
                                 error=repr(exc)[:2048])
                _log_event("sample_done", index=idx, status="failed", error=repr(exc)[:200])

    elapsed = time.monotonic() - t0
    _log_event("batch_done", total=config.count, succeeded=succeeded,
               failed=failed, skipped=skipped, elapsed_s=round(elapsed, 3))

    if failed > 0:
        logger.warning(
            "Batch complete: %d/%d succeeded, %d failed, %d skipped",
            succeeded, config.count, failed, skipped,
        )
    else:
        logger.info(
            "Batch complete: %d/%d succeeded, %d skipped",
            succeeded, config.count, skipped,
        )

    return BatchResult(
        total=config.count,
        succeeded=succeeded, failed=failed, skipped=skipped,
        results=tuple(all_results), duration_seconds=elapsed,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_executor(max_workers: int, mp_ctx):
    """Create the process pool executor. Separated for testability (D-57)."""
    return ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_ctx)


def _append_manifest(
    manifest_writer: ManifestWriter,
    result: SampleResult,
    config: Config,
    error: Optional[str] = None,
) -> None:
    entry: Dict = {
        "sample_index": result.sample_index,
        "seed": config.global_seed,
        "sample_seed": result.seed,
        "status": result.status,
        "split": result.split,
        "path": f"{result.sample_index:06d}/sample.json" if result.status == "ok" else "",
        "musicality_score": result.musicality_score,
        "duration_seconds": result.duration_seconds,
        "wrote_at": datetime.now(timezone.utc).isoformat(),
    }
    if error is not None:
        entry["error"] = error
    try:
        manifest_writer.append(entry)
    except Exception as exc:
        logger.warning("Manifest append failed for sample %d: %s", result.sample_index, exc)


def _log_event(event: str, **kwargs) -> None:
    payload = {"event": event, "ts": datetime.now(timezone.utc).isoformat(), **kwargs}
    sys.stderr.write(json.dumps(payload, default=str) + "\n")
    sys.stderr.flush()
