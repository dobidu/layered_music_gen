"""Wave 0 stub — Wave 4 (Plan 05-05) fills in the real tests (D-40, R-P12 first bullet).

Planned coverage: Config global_seed required (ValueError on None),
generate resume short-circuit (pre-existing sample.json returns without
running pipeline), full-layout generation under @pytest.mark.slow,
idempotent re-runs produce identical SampleResult.
"""
from __future__ import annotations

import pytest

pytest.skip(
    "Wave 4 implements api.py (D-40); this stub exists so Wave 0 test "
    "collection succeeds.",
    allow_module_level=True,
)
