"""Wave 0 stub — Wave 1 (Plan 05-02) fills in the real tests (D-36, R-P7).

Planned coverage: derive_sample_seed determinism, make_rngs five-domain shape,
save_random_state round-trip, assign_split deterministic + empirical ratios.
"""
from __future__ import annotations

import pytest

pytest.skip(
    "Wave 1 implements seeds.py (D-36); this stub exists so Wave 0 test "
    "collection succeeds.",
    allow_module_level=True,
)
