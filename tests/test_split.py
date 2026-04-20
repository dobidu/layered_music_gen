"""Tests for musicgen.seeds.assign_split (D-39, R-P6).

Pure-function tests — zero I/O. Runs in < 1s.
"""
from __future__ import annotations

from collections import Counter

import pytest

from musicgen.seeds import assign_split, derive_sample_seed


class TestAssignSplit:
    """D-26: sha256(f'split:{sample_seed}')[:4] modulo 10000 / 100.0 bucketed."""

    @pytest.mark.parametrize("sample_seed", [1, 42, 99, 12345, 999999999])
    def test_deterministic(self, sample_seed):
        a = assign_split(sample_seed, (0.8, 0.1, 0.1))
        b = assign_split(sample_seed, (0.8, 0.1, 0.1))
        assert a == b

    @pytest.mark.parametrize("sample_seed", range(100))
    def test_returns_valid_label(self, sample_seed):
        assert assign_split(sample_seed, (0.8, 0.1, 0.1)) in ("train", "valid", "test")

    def test_empirical_ratios_10k_seeds_default(self):
        """80/10/10 split across 10k seeds — empirical within 2% of declared."""
        labels = Counter()
        for i in range(10000):
            seed = derive_sample_seed(42, i)
            labels[assign_split(seed, (0.8, 0.1, 0.1))] += 1
        assert 7800 <= labels["train"] <= 8200, f"train={labels['train']} out of 80%±2%"
        assert 800 <= labels["valid"] <= 1200, f"valid={labels['valid']} out of 10%±2%"
        assert 800 <= labels["test"] <= 1200, f"test={labels['test']} out of 10%±2%"
        assert sum(labels.values()) == 10000

    def test_empirical_ratios_50_25_25(self):
        """50/25/25 split — empirical within 2% of declared."""
        labels = Counter()
        for i in range(10000):
            seed = derive_sample_seed(42, i)
            labels[assign_split(seed, (0.5, 0.25, 0.25))] += 1
        assert 4800 <= labels["train"] <= 5200
        assert 2300 <= labels["valid"] <= 2700
        assert 2300 <= labels["test"] <= 2700

    def test_prefix_disambiguates_from_seed_hash(self):
        """The 'split:' prefix ensures split bucket != sample_seed bucket."""
        # If the formulas accidentally used the same hash input, this pair
        # would correlate; with the "split:" prefix, they are independent.
        # (Weak check: not a statistical test, just a sanity probe.)
        sample_seed = derive_sample_seed(42, 0)
        # With the "split:" prefix in assign_split, the bucket is independent
        # of `sample_seed` used as the direct seed to random.Random.
        # We verify determinism here, not the statistical independence claim.
        label_a = assign_split(sample_seed, (0.8, 0.1, 0.1))
        label_b = assign_split(sample_seed, (0.8, 0.1, 0.1))
        assert label_a == label_b
