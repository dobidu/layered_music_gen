"""Sampler tests (R-X2): seeded determinism + AST no-bare-random static guard.

These tests pin two contracts that Phase 3 locks in for Phase 5 to baseline against:

  1. Every extracted sampler function is deterministic under a seeded
     ``random.Random(seed)`` — two fresh RNGs seeded identically must produce
     identical output (D-07, RESEARCH Risk #2).
  2. ``src/musicgen/sampler.py`` contains ZERO bare ``random.<method>(...)``
     call nodes — only ``rng.<method>(...)`` draws and the ``random.Random``
     constructor are permitted (D-07/D-08 static guard).

Tests import directly from ``musicgen.sampler`` (not via the ``music_gen`` shim)
so they pass independently of Task 3's shim rewrite.
"""
from __future__ import annotations

import ast
import os
import random
from typing import List

import pytest

from musicgen.sampler import (
    SongParams,
    generate_random_key,
    generate_random_tempo,
    generate_random_time_signature,
    generate_random_swing,
    generate_song_arrangement,
    generate_song_measures,
    time_signature_alternative,
    validate_measures_dict,
)

SAMPLER_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, "src", "musicgen", "sampler.py")
)


# -- SongParams.sample -------------------------------------------------------


class TestSongParamsSample:
    """Seeded-determinism contract for the SongParams.sample classmethod."""

    def test_sample_returns_songparams(self):
        p = SongParams.sample(random.Random(42))
        assert isinstance(p, SongParams)

    def test_sample_is_frozen(self):
        p = SongParams.sample(random.Random(42))
        with pytest.raises((AttributeError, TypeError)):  # frozen dataclass
            p.key = "Z"  # type: ignore[misc]

    def test_sample_is_deterministic(self):
        a = SongParams.sample(random.Random(42))
        b = SongParams.sample(random.Random(42))
        assert a == b  # frozen dataclass structural equality

    @pytest.mark.parametrize("seed", [0, 1, 42, 12345])
    def test_sample_same_seed_same_output(self, seed):
        a = SongParams.sample(random.Random(seed))
        b = SongParams.sample(random.Random(seed))
        assert a == b

    def test_sample_different_seeds_different_output(self):
        a = SongParams.sample(random.Random(0))
        b = SongParams.sample(random.Random(999))
        assert a != b

    def test_sample_fields_shape(self):
        p = SongParams.sample(random.Random(42))
        # Every required field populated (Risk #2: SongParams is NOT an empty
        # shell — all 9 fields come from the canonical draw order).
        assert p.key
        assert p.tempo
        assert p.time_signature_base
        assert 0.0 <= p.time_signature_variation <= 1.0
        assert 0.5 <= p.swing_amount <= 0.75
        assert set(p.signatures_per_part) == {"intro", "verse", "chorus", "bridge", "outro"}
        assert set(p.measures_per_part) == {"intro", "verse", "chorus", "bridge", "outro"}
        assert isinstance(p.song_arrangement, list) and p.song_arrangement
        assert isinstance(p.song_unique_parts, list) and p.song_unique_parts


# -- Free-function determinism ----------------------------------------------


class TestSamplerFreeFunctions:
    """Seeded-determinism contract for the 7 extracted sampler free functions."""

    @pytest.mark.parametrize("seed", [0, 1, 42])
    def test_generate_random_key_deterministic(self, seed):
        a = generate_random_key(random.Random(seed))
        b = generate_random_key(random.Random(seed))
        assert a == b

    @pytest.mark.parametrize("seed", [0, 1, 42])
    def test_generate_random_tempo_deterministic(self, seed):
        a = generate_random_tempo(random.Random(seed))
        b = generate_random_tempo(random.Random(seed))
        assert a == b
        assert 60 <= a <= 170  # broad bound from tempo_ranges

    @pytest.mark.parametrize("seed", [0, 1, 42])
    def test_generate_random_swing_deterministic(self, seed):
        a = generate_random_swing(random.Random(seed))
        b = generate_random_swing(random.Random(seed))
        assert a == b
        assert 0.5 <= a <= 0.75

    @pytest.mark.parametrize("seed", [0, 1, 42])
    def test_generate_random_time_signature_deterministic(self, seed):
        a = generate_random_time_signature(random.Random(seed))
        b = generate_random_time_signature(random.Random(seed))
        assert a == b

    @pytest.mark.parametrize("seed", [0, 1, 42])
    def test_time_signature_alternative_deterministic(self, seed):
        a = time_signature_alternative("4/4", random.Random(seed))
        b = time_signature_alternative("4/4", random.Random(seed))
        assert a == b

    @pytest.mark.parametrize("seed", [0, 1, 42])
    def test_generate_song_arrangement_deterministic(self, seed):
        u1, r1 = generate_song_arrangement(random.Random(seed))
        u2, r2 = generate_song_arrangement(random.Random(seed))
        assert r1 == r2
        assert set(u1) == set(u2)

    @pytest.mark.parametrize("seed", [0, 1, 42])
    def test_generate_song_measures_deterministic(self, seed):
        m1, s1 = generate_song_measures("4/4", 1.0, random.Random(seed))
        m2, s2 = generate_song_measures("4/4", 1.0, random.Random(seed))
        assert m1 == m2
        assert s1 == s2
        assert set(m1) == {"intro", "verse", "chorus", "bridge", "outro"}
        assert set(s1) == {"intro", "verse", "chorus", "bridge", "outro"}


# -- validate_measures_dict --------------------------------------------------


class TestValidateMeasuresDict:
    """Pure-function validator (no RNG) — spot-check happy + unhappy paths."""

    def test_validate_measures_dict_happy_path_from_sample(self):
        # SongParams.sample loops until this returns True; a valid sample must
        # therefore satisfy it.
        p = SongParams.sample(random.Random(42))
        assert validate_measures_dict(p.measures_per_part, p.signatures_per_part) is True

    def test_validate_measures_dict_rejects_unknown_signature(self):
        # Passing a bogus time signature should raise via registry lookup
        # (contract already covered by test_timesig_registry; we just confirm
        # the sampler helper surfaces it).
        with pytest.raises(Exception):
            validate_measures_dict({"intro": 8}, {"intro": "99/99"})


# -- AST static guard --------------------------------------------------------


def _bare_random_calls(source: str) -> List[ast.Call]:
    """AST walk that returns ``random.<attr>(...)`` Call nodes, excluding the
    ``random.Random(...)`` constructor (which is the RNG *factory*, not a bare
    draw)."""
    tree = ast.parse(source)
    hits = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "random"
            and node.func.attr != "Random"
        ):
            hits.append(node)
    return hits


def test_no_bare_random_in_sampler():
    """D-07/D-08: sampler.py must not contain any bare ``random.<method>(...)``.

    The RNG must be threaded through via the ``rng: random.Random`` parameter
    so Phase 5 can inject deterministic per-sample RNGs without rewriting
    sampler internals.
    """
    with open(SAMPLER_PATH, "r") as f:
        source = f.read()
    hits = _bare_random_calls(source)
    assert hits == [], (
        f"Found {len(hits)} bare random.<method>() call(s) in sampler.py at "
        f"line(s) {[n.lineno for n in hits]} — use rng.<method>() per D-07."
    )
