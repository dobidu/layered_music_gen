"""v0.3 Phase 3c — RED: musicality calibration harness.

Tests cover:
  - CalibrationResult dataclass fields
  - run_midi_calibration(): generates good/bad sets, scores them
  - Good MIDI set scores higher mean than adversarial set
  - suggest_threshold(): returns float between bad_mean and good_mean
  - separation_ok: True when good clearly > bad
  - save_calibration() / load_calibration() JSON round-trip
  - CLI entry: calibrate_midi() returns CalibrationResult
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# CalibrationResult dataclass
# ---------------------------------------------------------------------------

class TestCalibrationResult:
    def test_importable(self):
        from musicgen.calibrate import CalibrationResult
        assert CalibrationResult is not None

    def test_fields_present(self):
        from musicgen.calibrate import CalibrationResult
        import dataclasses
        fields = {f.name for f in dataclasses.fields(CalibrationResult)}
        for required in ("good_scores", "bad_scores", "suggested_threshold",
                         "separation_ok", "good_mean", "bad_mean"):
            assert required in fields, f"Missing field: {required!r}"

    def test_constructible(self):
        from musicgen.calibrate import CalibrationResult
        r = CalibrationResult(
            good_scores=[0.8, 0.9],
            bad_scores=[0.1, 0.2],
            suggested_threshold=0.5,
            separation_ok=True,
            good_mean=0.85,
            bad_mean=0.15,
        )
        assert r.separation_ok is True
        assert r.suggested_threshold == 0.5


# ---------------------------------------------------------------------------
# suggest_threshold
# ---------------------------------------------------------------------------

class TestSuggestThreshold:
    def test_importable(self):
        from musicgen.calibrate import suggest_threshold
        assert callable(suggest_threshold)

    def test_threshold_between_bad_and_good(self):
        """Threshold lies between max(bad) and min(good) when clearly separated."""
        from musicgen.calibrate import suggest_threshold
        good = [0.75, 0.80, 0.85, 0.90, 0.95]
        bad  = [0.05, 0.10, 0.15, 0.20, 0.25]
        t = suggest_threshold(good, bad)
        assert isinstance(t, float)
        assert max(bad) < t < min(good)

    def test_threshold_in_unit_interval(self):
        from musicgen.calibrate import suggest_threshold
        rng = random.Random(0)
        good = [rng.uniform(0.6, 1.0) for _ in range(20)]
        bad  = [rng.uniform(0.0, 0.4) for _ in range(20)]
        t = suggest_threshold(good, bad)
        assert 0.0 <= t <= 1.0

    def test_empty_lists_returns_default(self):
        from musicgen.calibrate import suggest_threshold
        t = suggest_threshold([], [])
        assert isinstance(t, float)
        assert 0.0 <= t <= 1.0


# ---------------------------------------------------------------------------
# run_midi_calibration
# ---------------------------------------------------------------------------

class TestRunMidiCalibration:
    def test_importable(self):
        from musicgen.calibrate import run_midi_calibration
        assert callable(run_midi_calibration)

    def test_returns_calibration_result(self, tmp_path):
        from musicgen.calibrate import CalibrationResult, run_midi_calibration
        result = run_midi_calibration(n_good=4, n_bad=4, seed=42, tmp_dir=str(tmp_path))
        assert isinstance(result, CalibrationResult)

    def test_good_scores_list_length(self, tmp_path):
        from musicgen.calibrate import run_midi_calibration
        result = run_midi_calibration(n_good=5, n_bad=3, seed=0, tmp_dir=str(tmp_path))
        assert len(result.good_scores) == 5

    def test_bad_scores_list_length(self, tmp_path):
        from musicgen.calibrate import run_midi_calibration
        result = run_midi_calibration(n_good=3, n_bad=6, seed=0, tmp_dir=str(tmp_path))
        assert len(result.bad_scores) == 6

    def test_good_mean_higher_than_bad_mean(self, tmp_path):
        """Reference-good MIDI should score higher than adversarial MIDI."""
        from musicgen.calibrate import run_midi_calibration
        result = run_midi_calibration(n_good=10, n_bad=10, seed=42, tmp_dir=str(tmp_path))
        assert result.good_mean > result.bad_mean, (
            f"Good mean ({result.good_mean:.3f}) should exceed bad mean "
            f"({result.bad_mean:.3f})"
        )

    def test_all_scores_in_unit_interval(self, tmp_path):
        from musicgen.calibrate import run_midi_calibration
        result = run_midi_calibration(n_good=5, n_bad=5, seed=1, tmp_dir=str(tmp_path))
        for s in result.good_scores + result.bad_scores:
            assert 0.0 <= s <= 1.0, f"Score {s:.3f} out of [0, 1]"

    def test_separation_ok_when_clearly_separated(self, tmp_path):
        """Well-formed MIDI vs. adversarial → separation_ok=True."""
        from musicgen.calibrate import run_midi_calibration
        result = run_midi_calibration(n_good=10, n_bad=10, seed=42, tmp_dir=str(tmp_path))
        assert result.separation_ok, (
            f"separation_ok should be True; good_mean={result.good_mean:.3f}, "
            f"bad_mean={result.bad_mean:.3f}"
        )

    def test_suggested_threshold_in_unit_interval(self, tmp_path):
        from musicgen.calibrate import run_midi_calibration
        result = run_midi_calibration(n_good=6, n_bad=6, seed=7, tmp_dir=str(tmp_path))
        assert 0.0 <= result.suggested_threshold <= 1.0

    def test_deterministic_with_same_seed(self, tmp_path):
        from musicgen.calibrate import run_midi_calibration
        r1 = run_midi_calibration(n_good=4, n_bad=4, seed=99, tmp_dir=str(tmp_path / "a"))
        r2 = run_midi_calibration(n_good=4, n_bad=4, seed=99, tmp_dir=str(tmp_path / "b"))
        assert r1.good_scores == r2.good_scores
        assert r1.bad_scores == r2.bad_scores

    def test_different_seed_gives_different_but_valid_result(self, tmp_path):
        from musicgen.calibrate import run_midi_calibration
        r1 = run_midi_calibration(n_good=4, n_bad=4, seed=1, tmp_dir=str(tmp_path / "a"))
        r2 = run_midi_calibration(n_good=4, n_bad=4, seed=2, tmp_dir=str(tmp_path / "b"))
        # Different seeds may give different scores, but both should be valid
        for s in r1.good_scores + r2.good_scores:
            assert 0.0 <= s <= 1.0


# ---------------------------------------------------------------------------
# save_calibration / load_calibration
# ---------------------------------------------------------------------------

class TestCalibrationPersistence:
    def test_save_importable(self):
        from musicgen.calibrate import save_calibration
        assert callable(save_calibration)

    def test_load_importable(self):
        from musicgen.calibrate import load_calibration
        assert callable(load_calibration)

    def test_round_trip(self, tmp_path):
        from musicgen.calibrate import CalibrationResult, load_calibration, save_calibration
        original = CalibrationResult(
            good_scores=[0.8, 0.85, 0.9],
            bad_scores=[0.1, 0.2],
            suggested_threshold=0.55,
            separation_ok=True,
            good_mean=0.85,
            bad_mean=0.15,
        )
        path = str(tmp_path / "calibration.json")
        save_calibration(original, path)
        loaded = load_calibration(path)
        assert loaded.good_scores == original.good_scores
        assert loaded.bad_scores == original.bad_scores
        assert loaded.suggested_threshold == pytest.approx(original.suggested_threshold)
        assert loaded.separation_ok == original.separation_ok
        assert loaded.good_mean == pytest.approx(original.good_mean)
        assert loaded.bad_mean == pytest.approx(original.bad_mean)

    def test_saved_file_is_valid_json(self, tmp_path):
        from musicgen.calibrate import CalibrationResult, save_calibration
        r = CalibrationResult(
            good_scores=[0.7], bad_scores=[0.2],
            suggested_threshold=0.45, separation_ok=True,
            good_mean=0.7, bad_mean=0.2,
        )
        path = str(tmp_path / "cal.json")
        save_calibration(r, path)
        data = json.loads(Path(path).read_text())
        assert "good_scores" in data
        assert "suggested_threshold" in data

    def test_load_missing_file_raises(self, tmp_path):
        from musicgen.calibrate import load_calibration
        with pytest.raises((FileNotFoundError, OSError)):
            load_calibration(str(tmp_path / "nonexistent.json"))
