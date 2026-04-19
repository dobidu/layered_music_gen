"""Wave 0 stub — Wave 5 (Plan 05-06) fills in the real tests (D-41, R-P8, R-Q3).

Planned coverage: SHA-256 goldens for mix.wav (FluidSynth-version-guarded),
per-layer MIDI, canonicalized sample.json (unconditional). Non-slow
cross-check: generate twice in one process → identical sample.json bytes.
"""
from __future__ import annotations

import pytest

pytest.skip(
    "Wave 5 implements determinism goldens (D-41); this stub exists so Wave 0 "
    "test collection succeeds.",
    allow_module_level=True,
)
