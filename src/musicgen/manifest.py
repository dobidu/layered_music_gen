"""Manifest module — append-only JSONL with lock abstraction (R-P5, D-14/D-15/D-16).

Ships ``ManifestWriter``: opens ``<dataset_root>/manifest.jsonl`` in append
mode, serializes entries under an injected ``ContextManager`` (default
``threading.Lock()``), and writes with ``os.fsync`` for POSIX atomicity on
writes <= PIPE_BUF (4096 bytes). Our manifest lines are ~200 bytes, well
under the atomicity bound.

Phase 5 uses ``threading.Lock()`` (single-process correctness). Phase 6's
``generate_batch`` passes ``multiprocessing.Manager().Lock()`` — the
``ContextManager`` type bound accepts both (verified by RESEARCH probe).

``is_sample_complete`` is a **projection check**: it reads only the sentinel
file ``<dataset_root>/<idx:06d>/sample.json``, never the manifest. The
manifest is a projection of completion state, not the source of truth
(D-16). This keeps ``is_sample_complete`` lock-free and forward-compatible
with whatever Phase 6's resume logic wants to know.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from typing import ContextManager, Optional

logger = logging.getLogger(__name__)


class ManifestWriter:
    """Append-only JSONL writer with injectable lock.

    Args:
        dataset_root: Directory where ``manifest.jsonl`` lives. Created on
            first append (via ``os.makedirs(..., exist_ok=True)``).
        lock: Context manager acquired around every ``append`` call. Default
            ``threading.Lock()`` constructed per-instance (mutable-default
            pitfall avoided). Phase 6 passes ``multiprocessing.Manager().Lock()``.
    """

    def __init__(self, dataset_root: str, lock: Optional[ContextManager] = None):
        self.dataset_root = dataset_root
        self.manifest_path = os.path.join(dataset_root, "manifest.jsonl")
        self.lock = lock if lock is not None else threading.Lock()

    def append(self, entry: dict) -> None:
        """Append one JSON line under lock.

        Args:
            entry: JSON-serializable dict. Keys are sorted via
                ``json.dumps(..., sort_keys=True)`` for byte-stable output
                (D-15 invariant).

        Side effects:
            Creates ``dataset_root`` if missing. Writes one line + newline
            to ``manifest.jsonl``. Calls ``os.fsync(fileno)`` after write
            so a process crash after append does not leave a half-line.
        """
        os.makedirs(self.dataset_root, exist_ok=True)
        line = json.dumps(entry, sort_keys=True) + "\n"
        with self.lock:
            with open(self.manifest_path, "a") as f:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())

    @staticmethod
    def is_sample_complete(
        dataset_root: str, sample_index: int, pad: int = 6,
    ) -> bool:
        """True iff ``<dataset_root>/<idx:06d>/sample.json`` exists (D-16).

        Does NOT read or require ``manifest.jsonl``. The sentinel is the
        sole source of truth for "sample N finished successfully".

        Args:
            dataset_root: Dataset directory containing the per-sample dirs.
            sample_index: Zero-based sample index.
            pad: Zero-padding width for the index (default 6; matches D-05).

        Returns:
            True if the sentinel file exists, False otherwise.
        """
        sentinel = os.path.join(
            dataset_root, f"{sample_index:0{pad}d}", "sample.json"
        )
        return os.path.exists(sentinel)
