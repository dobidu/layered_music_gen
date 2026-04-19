"""Tests for src/musicgen/seeds.py (D-36, R-P7).

Pure-function tests — zero I/O, zero FluidSynth. Runs in < 1s.
"""
from __future__ import annotations

import hashlib
import random

import pytest

from musicgen.seeds import (
    RNG_FX, RNG_GENERATORS, RNG_MIX, RNG_PARAMS, RNG_SOUNDFONTS,
    derive_sample_seed, make_rngs, save_random_state,
)


class TestDeriveSampleSeed:
    """D-17: sha256(f'{g}:{i}')[:8] as big-endian int."""

    @pytest.mark.parametrize("global_seed,sample_index", [
        (0, 0), (1, 0), (42, 5), (42, 100), (99999, 999999),
    ])
    def test_deterministic(self, global_seed, sample_index):
        a = derive_sample_seed(global_seed, sample_index)
        b = derive_sample_seed(global_seed, sample_index)
        assert a == b

    def test_different_indices_no_collision(self):
        """100 sequential indices for one global_seed → 100 distinct seeds."""
        seeds = {derive_sample_seed(42, i) for i in range(100)}
        assert len(seeds) == 100

    def test_order_sensitive(self):
        """derive(a, b) != derive(b, a) for non-equal args."""
        assert derive_sample_seed(1, 5) != derive_sample_seed(5, 1)

    def test_matches_documented_formula(self):
        """Byte-exact match of the D-17 formula."""
        expected = int.from_bytes(
            hashlib.sha256(b"42:0").digest()[:8], "big"
        )
        assert derive_sample_seed(42, 0) == expected

    def test_returns_64bit_unsigned(self):
        """Result fits in 64-bit unsigned range."""
        s = derive_sample_seed(42, 0)
        assert 0 <= s < 2**64


class TestMakeRngs:
    """D-18: 5 XOR-derived domain RNGs."""

    def test_five_domains(self):
        rngs = make_rngs(12345)
        assert set(rngs.keys()) == {
            RNG_PARAMS, RNG_GENERATORS, RNG_SOUNDFONTS, RNG_FX, RNG_MIX,
        }

    def test_xor_constants_match_spec(self):
        """make_rngs(0)[RNG_PARAMS] uses seed 0 ^ 0x01 == 1."""
        expected_first_float = random.Random(0 ^ 0x01).random()
        actual_first_float = make_rngs(0)[RNG_PARAMS].random()
        assert actual_first_float == expected_first_float

    @pytest.mark.parametrize("domain,xor_const", [
        (RNG_PARAMS, 0x01),
        (RNG_GENERATORS, 0x02),
        (RNG_SOUNDFONTS, 0x03),
        (RNG_FX, 0x04),
        (RNG_MIX, 0x05),
    ])
    def test_each_domain_xor_constant(self, domain, xor_const):
        """Every domain uses its D-18 XOR constant."""
        expected = random.Random(12345 ^ xor_const).random()
        actual = make_rngs(12345)[domain].random()
        assert actual == expected

    def test_domain_streams_distinct(self):
        """First 1000 draws from each domain are pairwise-distinct sequences."""
        rngs = make_rngs(12345)
        draws = {d: [rngs[d].random() for _ in range(1000)] for d in rngs}
        domains = list(draws.keys())
        for i, a in enumerate(domains):
            for b in domains[i + 1:]:
                assert draws[a] != draws[b], f"domains {a} and {b} produced identical sequences"

    def test_reseed_same_seed_same_draws(self):
        """make_rngs(s) twice → same first draws per domain."""
        a = make_rngs(12345)
        b = make_rngs(12345)
        for d in a:
            assert a[d].random() == b[d].random()


class TestSaveRandomState:
    """D-20: context manager round-trips global random state."""

    def test_restores_after_mutation(self):
        """Mutating random.seed inside the block leaves state intact after."""
        before = random.getstate()
        with save_random_state():
            random.seed(999)
            random.random()
        after = random.getstate()
        assert before == after

    def test_restores_on_exception(self):
        """State restored even when the with-body raises."""
        before = random.getstate()
        with pytest.raises(ValueError):
            with save_random_state():
                random.seed(999)
                raise ValueError("boom")
        after = random.getstate()
        assert before == after

    def test_nested_contexts(self):
        """Two nested save_random_state blocks compose correctly."""
        before = random.getstate()
        with save_random_state():
            random.seed(111)
            outer_state = random.getstate()
            with save_random_state():
                random.seed(222)
                random.random()
            # Inner restored outer_state
            assert random.getstate() == outer_state
        assert random.getstate() == before
