"""Wave 0 stub — Wave 1 (Plan 05-02) fills in the real tests (D-39, R-P6).

Planned coverage: assign_split determinism (same seed → same split),
empirical ratios within 2% of declared (10k samples → 80/10/10),
invalid ratio rejection at Config.__post_init__.
"""
from __future__ import annotations

import pytest

pytest.skip(
    "Wave 1 implements assign_split in seeds.py (D-39); this stub exists so "
    "Wave 0 test collection succeeds.",
    allow_module_level=True,
)
