"""Wave 0 stub — Wave 5 (Plan 06-06) fills in the real tests (D-68, R-P10/P11/P12/R-Q2).

Planned coverage: 4-sample batch with 2 workers (slow, needs FluidSynth + sf2 pools),
manifest verification, resume idempotence, output_mode routing end-to-end.
"""
from __future__ import annotations

import pytest

pytest.skip(
    "Wave 5 (Plan 06-06) implements integration batch test (D-68); "
    "this stub exists so Wave 0 test collection succeeds.",
    allow_module_level=True,
)
