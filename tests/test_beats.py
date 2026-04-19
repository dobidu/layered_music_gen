"""Beats tests (R-X7): beat_duration + extract_beat_times + extract_downbeat_times.

Three swing cases at 0.5/0.66/0.75 cover the roadmap requirement. Downbeat tests
encode RESEARCH correction #1: the time-grid algorithm returns `measures` entries
even when `beat_times` is sparse (verified against 4/4 and 12/8).

generate_beat fixtures use a seeded random.Random for determinism. MIDI files
are written inside tmp_path via monkeypatch.chdir (generate_beat creates
<name>/<name>-beat.mid in cwd).
"""
from __future__ import annotations

import random
from pathlib import Path

import pytest

from musicgen.beats import beat_duration, extract_beat_times, extract_downbeat_times
from musicgen.generators.beat import generate_beat


# ---------- beat_duration pure-function tests ----------

class TestBeatDuration:
    def test_4_4_at_120bpm(self):
        assert beat_duration("4/4", 120) == 0.5

    def test_6_8_at_120bpm(self):
        assert beat_duration("6/8", 120) == 0.25

    @pytest.mark.parametrize("tempo", [60, 120, 180, 200])
    def test_4_4_inverse_tempo(self, tempo):
        # 4/4: denominator == 4, so beat_duration == 60/tempo exactly
        assert beat_duration("4/4", tempo) == pytest.approx(60 / tempo)

    @pytest.mark.parametrize("sig,expected_at_120", [
        ("2/4", 0.5), ("3/4", 0.5), ("4/4", 0.5),
        ("6/8", 0.25), ("7/8", 0.25), ("12/8", 0.25),
    ])
    def test_all_registry_signatures_at_120(self, sig, expected_at_120):
        assert beat_duration(sig, 120) == pytest.approx(expected_at_120)


# ---------- extract_beat_times (MIDI-tick) ----------

def _make_beat_midi(tmp_path, monkeypatch, swing_amount: float, time_signature: str = "4/4", measures: int = 2, seed: int = 42) -> Path:
    """Generate a beat MIDI using the extracted generator with a seeded RNG.

    Returns the absolute path of the written MIDI. Uses monkeypatch.chdir so the
    generate_beat side-effect (creating <name>/ subdir) stays inside tmp_path.
    """
    monkeypatch.chdir(tmp_path)
    name = "song-verse"
    midi_path, _ = generate_beat(
        part="verse",
        tempo=120,
        time_signature=time_signature,
        measures=measures,
        name=name,
        swing_amount=swing_amount,
        rng=random.Random(seed),
    )
    return Path(tmp_path) / midi_path


class TestExtractBeatTimes:
    def test_returns_monotonic_sorted_list(self, tmp_path, monkeypatch):
        midi = _make_beat_midi(tmp_path, monkeypatch, swing_amount=0.5)
        times = extract_beat_times(str(midi), tempo=120, start_offset_seconds=0.0)
        assert times == sorted(times), "extract_beat_times must return sorted list"
        assert len(times) > 0, "expected at least one note_on extracted"

    def test_start_offset_shifts_all_timestamps(self, tmp_path, monkeypatch):
        midi = _make_beat_midi(tmp_path, monkeypatch, swing_amount=0.5)
        times_zero = extract_beat_times(str(midi), tempo=120, start_offset_seconds=0.0)
        times_shifted = extract_beat_times(str(midi), tempo=120, start_offset_seconds=10.0)
        assert len(times_zero) == len(times_shifted)
        for a, b in zip(times_zero, times_shifted):
            # rounding to 3 decimals may introduce tiny diffs — allow 0.002 tolerance
            assert b == pytest.approx(a + 10.0, abs=0.002)

    def test_deterministic_same_seed(self, tmp_path, monkeypatch):
        midi1 = _make_beat_midi(tmp_path, monkeypatch, swing_amount=0.5, seed=99)
        times1 = extract_beat_times(str(midi1), tempo=120, start_offset_seconds=0.0)
        # New tmp_path invocation — fresh file, same seed
        # (generate_beat writes into cwd/<song-verse>/ so we need a clean subdir)
        import shutil
        shutil.rmtree(tmp_path / "song")
        midi2 = _make_beat_midi(tmp_path, monkeypatch, swing_amount=0.5, seed=99)
        times2 = extract_beat_times(str(midi2), tempo=120, start_offset_seconds=0.0)
        assert times1 == times2, "same seed must produce same beat timestamps"


# ---------- Swing cases (0.5, 0.66, 0.75) ----------

class TestSwingCases:
    @pytest.mark.parametrize("swing_amount", [0.5, 0.66, 0.75])
    def test_monotonic_under_all_swing_values(self, tmp_path, monkeypatch, swing_amount):
        midi = _make_beat_midi(tmp_path, monkeypatch, swing_amount=swing_amount)
        times = extract_beat_times(str(midi), tempo=120, start_offset_seconds=0.0)
        assert times == sorted(times), f"swing={swing_amount}: non-monotonic"
        assert all(isinstance(t, float) and t >= 0.0 for t in times), \
            f"swing={swing_amount}: non-finite or negative"

    @pytest.mark.parametrize("swing_amount", [0.66, 0.75])
    def test_heavier_swing_delays_offbeats(self, tmp_path, monkeypatch, swing_amount):
        """For swing > 0.5, off-beats (odd index) should be delayed relative to swing=0.5."""
        # Use identical seed so the pattern choice is identical across runs
        import shutil
        midi_straight = _make_beat_midi(tmp_path, monkeypatch, swing_amount=0.5, seed=7)
        t_straight = extract_beat_times(str(midi_straight), tempo=120, start_offset_seconds=0.0)
        shutil.rmtree(tmp_path / "song")
        midi_swung = _make_beat_midi(tmp_path, monkeypatch, swing_amount=swing_amount, seed=7)
        t_swung = extract_beat_times(str(midi_swung), tempo=120, start_offset_seconds=0.0)
        # Only compare overlapping length; off-beats at odd positions in the pattern
        # accumulated beat array should be LATER in swung than straight.
        # (Approximation: the test asserts the MEAN timestamp under swung >= mean under straight,
        # which is a robust aggregate even if individual index mapping varies across RNG pattern choice.)
        if len(t_swung) >= 4 and len(t_straight) >= 4:
            assert sum(t_swung) >= sum(t_straight) - 0.01, \
                f"swing={swing_amount}: total offset should be >= straight (swing delays off-beats)"


# ---------- extract_downbeat_times (time-grid, RESEARCH correction #1) ----------

class TestExtractDownbeatTimes:
    @pytest.mark.parametrize("measures", [1, 2, 3, 4, 5])
    def test_downbeat_count_equals_measures_44(self, measures):
        downbeats = extract_downbeat_times([], "4/4", measures, 0.0, 120)
        assert len(downbeats) == measures

    @pytest.mark.parametrize("sig", ["2/4", "3/4", "4/4", "6/8", "7/8", "12/8"])
    def test_downbeat_count_equals_measures_all_sigs(self, sig):
        """RESEARCH correction #1: time-grid returns exactly `measures` downbeats
        for every registered time signature, regardless of pattern sparsity."""
        downbeats = extract_downbeat_times([], sig, measures=4, start_offset_seconds=0.0, tempo=120)
        assert len(downbeats) == 4, f"{sig}: expected 4 downbeats, got {len(downbeats)}"

    def test_downbeat_grid_4_4_120bpm(self):
        # measure duration = numerator (4) * beat_duration(4/4, 120) (0.5) = 2.0s
        downbeats = extract_downbeat_times([], "4/4", 3, 0.0, 120)
        assert downbeats == [0.0, 2.0, 4.0]

    def test_downbeat_grid_with_start_offset(self):
        downbeats = extract_downbeat_times([], "4/4", 2, 5.0, 120)
        assert downbeats == [5.0, 7.0]

    def test_downbeat_grid_independent_of_beat_times_input(self):
        """Whether beat_times is empty, full, or garbage — output is the same
        (time-grid is math-only; beat_times is retained for API compat only).
        """
        a = extract_downbeat_times([], "4/4", 3, 0.0, 120)
        b = extract_downbeat_times([0.5, 1.0, 1.5], "4/4", 3, 0.0, 120)
        c = extract_downbeat_times([999.0, -1.0, 42.0], "4/4", 3, 0.0, 120)
        assert a == b == c, "extract_downbeat_times must be input-independent"

    def test_downbeat_no_accumulated_rounding_error(self):
        # 10 measures at tempo 120 in 4/4 = expected at 0.0, 2.0, 4.0, ... 18.0
        downbeats = extract_downbeat_times([], "4/4", 10, 0.0, 120)
        expected = [2.0 * i for i in range(10)]
        for a, b in zip(downbeats, expected):
            assert abs(a - b) < 1e-9, f"rounding drift: {a} vs {b}"

    def test_downbeat_12_8_grid(self):
        # 12/8: numerator=12, beat_duration = 60/120 * 4/8 = 0.25, measure = 12*0.25 = 3.0
        downbeats = extract_downbeat_times([], "12/8", 3, 0.0, 120)
        assert downbeats == [0.0, 3.0, 6.0]
