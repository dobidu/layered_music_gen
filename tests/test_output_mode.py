"""Wave 0 stub — Wave 1 (Plan 06-02) fills in the real tests (D-47, D-48, R-P14).

Planned coverage: Config.output_mode validation, writer.write_sample output_mode
routing (which files are written per mode), pre_roll_offset_s beat-time shift.
"""
from __future__ import annotations

import pytest

pytest.skip(
    "Wave 1 (Plan 06-02) implements output_mode in Config + writer (D-47/D-48/D-66); "
    "this stub exists so Wave 0 test collection succeeds.",
    allow_module_level=True,
)
