"""Wave 0 stub — Wave 2 (Plan 06-03) fills in the real tests (D-50..D-54, R-P9).

Planned coverage: FluidSynth pre-roll measurement algorithm, cache read/write,
version-gate re-measure, graceful absent-FluidSynth path.
"""
from __future__ import annotations

import pytest

pytest.skip(
    "Wave 2 (Plan 06-03) implements calibrate.py (D-50..D-54); "
    "this stub exists so Wave 0 test collection succeeds.",
    allow_module_level=True,
)
