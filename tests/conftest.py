"""Pytest plugin hooks for Phase 5 determinism testing (D-32).

Registers the ``--regen-goldens`` flag consumed by
``tests/test_determinism_golden.py``. When present, the determinism test
regenerates ``tests/fixtures/determinism/*.sha256`` from its computed
artifact hashes instead of asserting equality. Use when intentionally
changing RNG draw order or when upgrading the pinned FluidSynth binary.
"""
from __future__ import annotations


def pytest_addoption(parser):
    parser.addoption(
        "--regen-goldens",
        action="store_true",
        default=False,
        help=(
            "Regenerate determinism fixtures "
            "(tests/fixtures/determinism/*.sha256) instead of asserting "
            "against them. Use when intentionally changing RNG order or "
            "FluidSynth version."
        ),
    )
