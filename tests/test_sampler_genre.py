"""RED tests — Phase 4: Sampler + FX genre integration (v0.2).

Covers:
  - generate_random_tempo respects GenreSpec.tempo_min/tempo_max hard bounds
  - generate_random_time_signature uses GenreSpec.time_sig_weights weighted draw
  - generate_random_swing respects GenreSpec.swing_min/swing_max hard bounds
  - generate_random_key uses GenreSpec.scale_weights weighted draw
  - SongParams.sample passes genre_spec through to all sub-draws
  - build_fx_boards accepts optional genre_spec (fx_profile soft shifts)
  - resolve_genre_spec(config) returns None or merged GenreSpec per config.genre
  - Backward compat: genre_spec=None → identical draws to pre-genre behavior
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import pytest

from musicgen.genre import GenreSpec
from musicgen.sampler import (
    generate_random_key,
    generate_random_swing,
    generate_random_tempo,
    generate_random_time_signature,
    SongParams,
)
from musicgen import mixer
from config import Config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _genre(**kwargs) -> GenreSpec:
    """Construct a GenreSpec with given overrides."""
    return GenreSpec(name="test", **kwargs)


def _rng(seed: int = 0) -> random.Random:
    return random.Random(seed)


# ---------------------------------------------------------------------------
# generate_random_tempo — hard bounds
# ---------------------------------------------------------------------------

class TestGenreTempoHardBounds:
    def test_tempo_always_within_genre_min_max(self):
        spec = _genre(tempo_min=180, tempo_max=200)
        for seed in range(50):
            t = generate_random_tempo(_rng(seed), genre_spec=spec)
            assert 180 <= t <= 200, f"seed {seed}: tempo {t} out of [180, 200]"

    def test_tempo_tight_range_single_value(self):
        spec = _genre(tempo_min=120, tempo_max=120)
        t = generate_random_tempo(_rng(0), genre_spec=spec)
        assert t == 120

    def test_tempo_none_genre_spec_unchanged(self):
        """genre_spec=None must produce same draws as before (backward compat)."""
        r1 = _rng(42)
        r2 = _rng(42)
        t1 = generate_random_tempo(r1, genre_spec=None)
        t2 = generate_random_tempo(r2)  # old call signature
        assert t1 == t2

    def test_tempo_low_range(self):
        spec = _genre(tempo_min=60, tempo_max=70)
        for seed in range(20):
            t = generate_random_tempo(_rng(seed), genre_spec=spec)
            assert 60 <= t <= 70

    def test_tempo_full_range_no_constraint(self):
        """Default GenreSpec (60–240) should produce same distribution as no-genre."""
        spec = _genre(tempo_min=60.0, tempo_max=240.0)
        # Just verify it doesn't crash and returns something in range
        t = generate_random_tempo(_rng(1), genre_spec=spec)
        assert 60 <= t <= 240


# ---------------------------------------------------------------------------
# generate_random_time_signature — soft weights
# ---------------------------------------------------------------------------

class TestGenreTimeSigWeights:
    def test_time_sig_deterministic_single_option(self):
        """When only one sig has weight, it must always win."""
        spec = _genre(time_sig_weights={"3/4": 1.0})
        for seed in range(20):
            ts = generate_random_time_signature(_rng(seed), genre_spec=spec)
            assert ts == "3/4", f"seed {seed}: expected 3/4, got {ts}"

    def test_time_sig_empty_weights_no_error(self):
        """Empty time_sig_weights → fall back to registry default."""
        spec = _genre(time_sig_weights={})
        ts = generate_random_time_signature(_rng(0), genre_spec=spec)
        assert isinstance(ts, str)
        assert "/" in ts

    def test_time_sig_none_genre_spec_unchanged(self):
        """genre_spec=None must produce same draws as before."""
        r1 = _rng(99)
        r2 = _rng(99)
        ts1 = generate_random_time_signature(r1, genre_spec=None)
        ts2 = generate_random_time_signature(r2)
        assert ts1 == ts2

    def test_time_sig_distribution_biased(self):
        """Heavy weight on 4/4 → 4/4 wins significantly more than 1/8 of the time."""
        spec = _genre(time_sig_weights={"4/4": 100.0, "3/4": 1.0})
        draws = [generate_random_time_signature(_rng(s), genre_spec=spec) for s in range(100)]
        assert draws.count("4/4") > 80


# ---------------------------------------------------------------------------
# generate_random_swing — hard bounds
# ---------------------------------------------------------------------------

class TestGenreSwingHardBounds:
    def test_swing_always_within_genre_min_max(self):
        spec = _genre(swing_min=0.65, swing_max=0.70)
        for seed in range(50):
            sw = generate_random_swing(_rng(seed), genre_spec=spec)
            assert 0.65 <= sw <= 0.70, f"seed {seed}: swing {sw} out of [0.65, 0.70]"

    def test_swing_tight_range(self):
        spec = _genre(swing_min=0.5, swing_max=0.5)
        sw = generate_random_swing(_rng(0), genre_spec=spec)
        assert sw == pytest.approx(0.5, abs=1e-9)

    def test_swing_none_genre_spec_unchanged(self):
        r1 = _rng(7)
        r2 = _rng(7)
        sw1 = generate_random_swing(r1, genre_spec=None)
        sw2 = generate_random_swing(r2)
        assert sw1 == pytest.approx(sw2, abs=1e-12)

    def test_swing_full_range_no_constraint(self):
        spec = _genre(swing_min=0.5, swing_max=0.75)
        sw = generate_random_swing(_rng(3), genre_spec=spec)
        assert 0.5 <= sw <= 0.75


# ---------------------------------------------------------------------------
# generate_random_key — soft scale weights
# ---------------------------------------------------------------------------

class TestGenreScaleWeights:
    def test_key_deterministic_single_scale(self):
        spec = _genre(scale_weights={"Am": 1.0})
        for seed in range(20):
            k = generate_random_key(_rng(seed), genre_spec=spec)
            assert k == "Am", f"seed {seed}: expected Am, got {k}"

    def test_key_empty_weights_no_error(self):
        spec = _genre(scale_weights={})
        k = generate_random_key(_rng(0), genre_spec=spec)
        assert isinstance(k, str)

    def test_key_none_genre_spec_unchanged(self):
        r1 = _rng(13)
        r2 = _rng(13)
        k1 = generate_random_key(r1, genre_spec=None)
        k2 = generate_random_key(r2)
        assert k1 == k2

    def test_key_distribution_biased(self):
        spec = _genre(scale_weights={"G": 100.0, "C": 1.0})
        draws = [generate_random_key(_rng(s), genre_spec=spec) for s in range(100)]
        assert draws.count("G") > 80

    def test_key_unknown_scale_falls_back(self):
        """scale_weights key not in 24-bucket list → handled gracefully."""
        spec = _genre(scale_weights={"G": 1.0, "INVALID_KEY": 0.5})
        k = generate_random_key(_rng(0), genre_spec=spec)
        assert isinstance(k, str)


# ---------------------------------------------------------------------------
# SongParams.sample — genre_spec kwarg
# ---------------------------------------------------------------------------

class TestSongParamsSampleGenre:
    def test_sample_accepts_genre_spec_kwarg(self):
        spec = _genre(tempo_min=140, tempo_max=160)
        cfg = Config()
        params = SongParams.sample(_rng(0), cfg, genre_spec=spec)
        assert 140 <= params.tempo <= 160

    def test_sample_tempo_within_genre_bounds_many_seeds(self):
        spec = _genre(tempo_min=90, tempo_max=100)
        cfg = Config()
        for seed in range(20):
            params = SongParams.sample(_rng(seed), cfg, genre_spec=spec)
            assert 90 <= params.tempo <= 100, f"seed {seed}: tempo {params.tempo}"

    def test_sample_time_sig_weighted_by_genre(self):
        spec = _genre(time_sig_weights={"6/8": 1.0})
        cfg = Config()
        for seed in range(10):
            params = SongParams.sample(_rng(seed), cfg, genre_spec=spec)
            assert params.time_signature_base == "6/8"

    def test_sample_swing_within_genre_bounds(self):
        spec = _genre(swing_min=0.60, swing_max=0.65)
        cfg = Config()
        for seed in range(20):
            params = SongParams.sample(_rng(seed), cfg, genre_spec=spec)
            assert 0.60 <= params.swing_amount <= 0.65, (
                f"seed {seed}: swing {params.swing_amount}"
            )

    def test_sample_none_genre_spec_backward_compat(self):
        """SongParams.sample(..., genre_spec=None) must match old call signature output."""
        cfg = Config()
        r1 = _rng(77)
        r2 = _rng(77)
        p1 = SongParams.sample(r1, cfg, genre_spec=None)
        p2 = SongParams.sample(r2, cfg)
        assert p1.key == p2.key
        assert p1.tempo == p2.tempo
        assert p1.time_signature_base == p2.time_signature_base
        assert p1.swing_amount == pytest.approx(p2.swing_amount, abs=1e-12)


# ---------------------------------------------------------------------------
# build_fx_boards — genre_spec optional kwarg (fx_profile soft shifts)
# ---------------------------------------------------------------------------

class TestBuildFxBoardsGenre:
    def test_build_fx_boards_accepts_genre_spec(self, tmp_path):
        """build_fx_boards must accept optional genre_spec without error."""
        spec = _genre(fx_profile={"reverb": 0.8, "chorus": 0.3})
        cfg = Config()
        # Only test signature acceptance — full FX requires audio hardware
        boards = mixer.build_fx_boards(cfg, _rng(0), genre_spec=spec)
        assert set(boards.keys()) == {"beat", "melody", "harmony", "bassline"}

    def test_build_fx_boards_none_genre_spec_backward_compat(self):
        """genre_spec=None must not change behavior vs old call signature."""
        cfg = Config()
        from pedalboard import Pedalboard
        boards = mixer.build_fx_boards(cfg, _rng(5), genre_spec=None)
        assert all(isinstance(b, Pedalboard) for b in boards.values())

    def test_build_fx_boards_empty_fx_profile_no_error(self):
        spec = _genre(fx_profile={})
        cfg = Config()
        boards = mixer.build_fx_boards(cfg, _rng(0), genre_spec=spec)
        assert len(boards) == 4


# ---------------------------------------------------------------------------
# resolve_genre_spec — api helper (new function)
# ---------------------------------------------------------------------------

class TestResolveGenreSpec:
    def test_resolve_returns_none_when_no_genre(self):
        """Config.genre is None → resolve_genre_spec returns None."""
        from musicgen.api import resolve_genre_spec
        cfg = Config()
        assert cfg.genre is None
        assert resolve_genre_spec(cfg) is None

    def test_resolve_returns_genre_spec_when_genre_set(self, tmp_path):
        """Config.genre=['mygenre'] with spec file → returns GenreSpec."""
        from musicgen.api import resolve_genre_spec
        import json
        genre_dir = tmp_path / "mygenre"
        genre_dir.mkdir()
        (genre_dir / "spec.json").write_text(json.dumps({
            "name": "mygenre",
            "tempo_min": 100.0,
            "tempo_max": 130.0,
        }))
        cfg = Config()
        cfg.genre = ["mygenre"]
        cfg.genres_dir = str(tmp_path)
        result = resolve_genre_spec(cfg)
        assert result is not None
        assert isinstance(result, GenreSpec)
        assert result.tempo_min == 100.0
        assert result.tempo_max == 130.0

    def test_resolve_merges_multiple_genres(self, tmp_path):
        """Config.genre=['a','b'] → merged GenreSpec returned."""
        from musicgen.api import resolve_genre_spec
        import json
        for name, tmin, tmax in [("a", 120.0, 160.0), ("b", 140.0, 200.0)]:
            d = tmp_path / name
            d.mkdir()
            (d / "spec.json").write_text(json.dumps({
                "name": name, "tempo_min": tmin, "tempo_max": tmax,
            }))
        cfg = Config()
        cfg.genre = ["a", "b"]
        cfg.genres_dir = str(tmp_path)
        result = resolve_genre_spec(cfg)
        assert result is not None
        # Merged hard range: max(120, 140)=140 .. min(160, 200)=160
        assert result.tempo_min == pytest.approx(140.0)
        assert result.tempo_max == pytest.approx(160.0)
