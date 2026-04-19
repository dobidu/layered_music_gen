"""Wave 0 stub — Wave 3 (Plan 05-04) fills in the real tests (D-38, R-P5).

Planned coverage: append-under-lock correctness (10 threads × 100 entries),
is_sample_complete sentinel-only semantics (manifest contents irrelevant),
JSON-per-line well-formedness after concurrent appends.
"""
from __future__ import annotations

import pytest

pytest.skip(
    "Wave 3 implements manifest.py (D-38); this stub exists so Wave 0 test "
    "collection succeeds.",
    allow_module_level=True,
)
