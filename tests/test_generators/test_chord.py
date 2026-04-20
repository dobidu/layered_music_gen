"""Chord generator tests (R-X3): seeded determinism + byte-equal MIDI.

The generators derive their output directory via ``name.split('-')[0]`` (a quirk
carried over verbatim from ``music_gen.py``), so the production call convention
is:

  song_name = "<20-digit-token>"           # no hyphens
  name      = f"{song_name}-{part}"        # exactly one hyphen

To match that convention under pytest's ``tmp_path`` (which contains hyphens
like ``pytest-17``), we ``chdir`` into ``tmp_path`` and pass a *relative*
``name``. The generator then writes into the pytest-managed temp tree with
no repo-root pollution.
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import pytest

from musicgen.generators.chord import generate_chord_progression

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
CHORD_PATTERNS = os.path.join(REPO_ROOT, "chord_patterns.txt")


class TestGenerateChordProgressionDeterministic:
    @pytest.mark.parametrize("seed", [0, 42, 12345])
    def test_same_seed_same_output(self, tmp_path, monkeypatch, seed):
        monkeypatch.chdir(tmp_path)
        args = dict(
            key="C", tempo=120, time_signature="4/4", measures=4,
            name=f"song{seed}-verse",
            part="verse",
            pattern_file=CHORD_PATTERNS,
        )
        p1, mid1 = generate_chord_progression(**args, rng=random.Random(seed))
        bytes1 = Path(mid1).read_bytes()
        os.remove(mid1)
        p2, mid2 = generate_chord_progression(**args, rng=random.Random(seed))
        assert p1 == p2, f"chord pattern diverged for seed={seed}"
        assert bytes1 == Path(mid2).read_bytes(), f"MIDI bytes diverged for seed={seed}"

    def test_missing_rng_raises_type_error(self, tmp_path, monkeypatch):
        """D-07/D-08: rng is required — calling without it must raise TypeError."""
        monkeypatch.chdir(tmp_path)
        with pytest.raises(TypeError):
            generate_chord_progression(  # type: ignore[call-arg]
                key="C", tempo=120, time_signature="4/4", measures=4,
                name="song-verse",
                part="verse",
                pattern_file=CHORD_PATTERNS,
            )
