"""Beat generator tests (R-X3): seeded determinism across time signatures + swing values.

Also covers unit tests for the pure helpers ``beat_duration`` and ``calculate_swing_offset``.

Tests chdir into ``tmp_path`` and pass a *relative* ``name`` so the generator's
``name.split('-')[0]`` directory derivation (inherited verbatim from
``music_gen.py``) produces a valid path without interference from hyphens in
the pytest tmp-path prefix.
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import pytest

from musicgen.generators.beat import (
    beat_duration,
    calculate_swing_offset,
    generate_beat,
)


def test_beat_duration_44():
    # 120 BPM, 4/4 → 0.5s per beat
    assert beat_duration("4/4", 120) == pytest.approx(0.5)


def test_beat_duration_68():
    # 120 BPM, 6/8 → 0.25s per eighth note
    assert beat_duration("6/8", 120) == pytest.approx(0.25)


def test_calculate_swing_offset_basic():
    assert calculate_swing_offset(1.0, 0.5) == 0.0       # no swing
    assert calculate_swing_offset(1.0, 0.66) == pytest.approx(0.16)
    assert calculate_swing_offset(1.0, 0.75) == pytest.approx(0.25)


class TestGenerateBeatDeterministic:
    @pytest.mark.parametrize("ts,swing", [
        ("4/4", 0.5), ("4/4", 0.66), ("4/4", 0.75),
        ("3/4", 0.5), ("6/8", 0.66),
    ])
    def test_same_seed_same_output(self, tmp_path, monkeypatch, ts, swing):
        monkeypatch.chdir(tmp_path)
        args = dict(
            part="verse", tempo=120, time_signature=ts, measures=4,
            name="song-verse",
            swing_amount=swing,
        )
        mid1, ann1 = generate_beat(**args, rng=random.Random(42))
        bytes1 = Path(mid1).read_bytes()
        os.remove(mid1)
        mid2, ann2 = generate_beat(**args, rng=random.Random(42))
        assert ann1 == ann2, f"annotations diverged for ts={ts} swing={swing}"
        assert bytes1 == Path(mid2).read_bytes(), (
            f"MIDI bytes diverged for ts={ts} swing={swing}"
        )

    def test_missing_rng_raises_type_error(self, tmp_path, monkeypatch):
        """D-07/D-08: rng is required — calling without it must raise TypeError."""
        monkeypatch.chdir(tmp_path)
        with pytest.raises(TypeError):
            generate_beat(  # type: ignore[call-arg]
                part="verse", tempo=120, time_signature="4/4", measures=4,
                name="song-verse",
                swing_amount=0.5,
            )
