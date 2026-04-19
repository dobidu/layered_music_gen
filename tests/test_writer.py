"""Wave 0 stub — Wave 3 (Plan 05-04) fills in the real tests (D-37, R-P1/R-P2/R-P3).

Planned coverage: per-sample directory layout, relative path rewriting in
sample.json, silent-stem concatenation, sum-of-stems assertion (pass + fail
fault-injection), sentinel invariant (sample.json presence implies all files).
"""
from __future__ import annotations

import pytest

pytest.skip(
    "Wave 3 implements writer.py (D-37); this stub exists so Wave 0 test "
    "collection succeeds.",
    allow_module_level=True,
)
