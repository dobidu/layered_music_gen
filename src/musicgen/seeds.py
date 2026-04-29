"""Seeds module — pure-function RNG hierarchy (R-P7, D-17/D-18/D-20, D-26).

Closes the "one module-level ``random.Random``" shortcut Phase 3 D-08 used.
Phase 5's contract: every RNG consumer in the pipeline gets a domain-specific
``random.Random`` from ``make_rngs(sample_seed)``; ``sample_seed`` itself is
a deterministic SHA-256 projection of ``(global_seed, sample_index)``.

Public surface:
  * ``derive_sample_seed(global_seed, sample_index) -> int``  — D-17
  * ``make_rngs(sample_seed) -> Dict[str, random.Random]``    — D-18
  * ``assign_split(sample_seed, ratios) -> str``              — D-26
  * ``save_random_state()`` context manager                   — D-20
  * Domain name constants ``RNG_PARAMS`` / ``RNG_GENERATORS`` / ``RNG_SOUNDFONTS`` / ``RNG_FX`` / ``RNG_MIX``

Zero I/O, zero dependencies outside stdlib. The AST bare-random guard
(``tests/test_no_bare_random_in_package.py``) was widened in Plan 05-01 to
permit ``random.Random`` / ``random.getstate`` / ``random.setstate`` — the
three stdlib primitives this module legitimately needs.
"""
from __future__ import annotations

import contextlib
import hashlib
import logging
import random
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

RNG_PARAMS = "params"
RNG_GENERATORS = "generators"
RNG_SOUNDFONTS = "soundfonts"
RNG_FX = "fx"
RNG_MIX = "mix"


def derive_sample_seed(global_seed: int, sample_index: int) -> int:
    """Deterministic per-sample seed from ``(global_seed, sample_index)`` (D-17).

    Args:
        global_seed: User-supplied integer seed (from ``Config.global_seed``).
        sample_index: Zero-based position within a batch run.

    Returns:
        A 64-bit unsigned integer suitable for ``random.Random(seed=...)``.
        Stable across processes, machines, and Python 3.10+ patch releases.

    Example:
        >>> derive_sample_seed(42, 0)  # doctest: +SKIP
        15620049066705523080
    """
    raw = hashlib.sha256(f"{global_seed}:{sample_index}".encode()).digest()
    return int.from_bytes(raw[:8], "big")


def make_rngs(sample_seed: int) -> Dict[str, random.Random]:
    """Five named domain RNGs from one sample_seed via XOR with small constants (D-18).

    Args:
        sample_seed: Integer seed from :func:`derive_sample_seed`.

    Returns:
        Dict mapping domain name (``RNG_PARAMS``, ``RNG_GENERATORS``,
        ``RNG_SOUNDFONTS``, ``RNG_FX``, ``RNG_MIX``) to an independent
        ``random.Random`` instance seeded with ``sample_seed ^ 0x0N``.
    """
    return {
        RNG_PARAMS:     random.Random(sample_seed ^ 0x01),
        RNG_GENERATORS: random.Random(sample_seed ^ 0x02),
        RNG_SOUNDFONTS: random.Random(sample_seed ^ 0x03),
        RNG_FX:         random.Random(sample_seed ^ 0x04),
        RNG_MIX:        random.Random(sample_seed ^ 0x05),
    }


def assign_split(sample_seed: int, ratios: Tuple[float, float, float]) -> str:
    """Deterministic train/valid/test assignment from sample_seed (D-26, R-P6).

    Args:
        sample_seed: Integer seed (typically from :func:`derive_sample_seed`).
        ratios: ``(train, valid, test)`` probabilities. Must sum to 1.0
            (validation lives on ``Config.__post_init__`` per D-27).

    Returns:
        One of ``"train"``, ``"valid"``, or ``"test"``. Lowercase is the
        ML-tooling idiom (PyTorch ``Dataset`` splits).
    """
    bucket = int.from_bytes(
        hashlib.sha256(f"split:{sample_seed}".encode()).digest()[:4], "big"
    ) % 10000 / 100.0  # float in [0, 100)
    train_cutoff = ratios[0] * 100
    valid_cutoff = (ratios[0] + ratios[1]) * 100
    if bucket < train_cutoff:
        return "train"
    if bucket < valid_cutoff:
        return "valid"
    return "test"


@contextlib.contextmanager
def save_random_state():
    """Snapshot + restore global ``random`` state — defense-in-depth (D-20).

    Wrap any call that might touch global random state (e.g.
    :func:`musicgen.musicality.get_musicality_score`) to guarantee
    per-sample RNG basins stay independent even if a transitive dep
    silently mutates ``random`` globals.

    Phase 3 D-24 audit proved ``music21`` and ``librosa`` do NOT mutate
    global random today; this wrapper is preventive insurance against
    dep-upgrade surprises.

    Usage:
        with save_random_state():
            score, components = get_musicality_score(wav_path)
    """
    state = random.getstate()
    try:
        yield
    finally:
        random.setstate(state)
