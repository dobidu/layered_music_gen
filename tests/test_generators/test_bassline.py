"""Bassline generator tests (R-X3): seeded determinism + byte-equal MIDI.

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

from musicgen.generators.bassline import generate_bassline


class TestGenerateBasslineDeterministic:
    @pytest.mark.parametrize("seed", [0, 42, 12345])
    def test_same_seed_same_output(self, tmp_path, monkeypatch, seed):
        monkeypatch.chdir(tmp_path)
        args = dict(
            key="C", tempo=120, time_signature="4/4", measures=4,
            name=f"song{seed}-verse",
            part="verse",
            chord_progression=["I", "IV", "V", "I"],
            melody=[60, 62, 64, 65, 67, 69, 71, 72],
        )
        path1 = generate_bassline(**args, rng=random.Random(seed))
        bytes1 = Path(path1).read_bytes()
        os.remove(path1)
        path2 = generate_bassline(**args, rng=random.Random(seed))
        assert bytes1 == Path(path2).read_bytes(), (
            f"MIDI bytes diverged for seed={seed}"
        )

    def test_missing_rng_raises_type_error(self, tmp_path, monkeypatch):
        """D-07/D-08: rng is required — calling without it must raise TypeError."""
        monkeypatch.chdir(tmp_path)
        with pytest.raises(TypeError):
            generate_bassline(  # type: ignore[call-arg]
                key="C", tempo=120, time_signature="4/4", measures=4,
                name="song-verse",
                part="verse",
                chord_progression=["I", "IV", "V", "I"],
                melody=[60, 62, 64, 65, 67, 69, 71, 72],
            )
