"""
Registry-level unit tests for timesig.TimeSignatureRegistry (R-S6).

Pin ALL 7 supported signatures with exact field values. The wrapper-level
tests in tests/test_time_signature.py verify music_gen delegates correctly;
these tests verify the registry data and methods directly.
"""
import pytest
from timesig import TimeSignatureRegistry, TimeSignatureSpec


# ---------------------------------------------------------------------------
# TestRegistryContents — membership and field values
# ---------------------------------------------------------------------------

class TestRegistryContents:
    ALL_SIGS = {"2/4", "3/4", "4/4", "5/4", "6/8", "7/8", "12/8"}

    def test_registry_contains_all_seven_signatures(self):
        assert set(TimeSignatureRegistry.all_signatures()) == self.ALL_SIGS

    @pytest.mark.parametrize("sig", ["2/4", "3/4", "4/4", "5/4", "6/8", "7/8", "12/8"])
    def test_lookup_returns_spec(self, sig):
        spec = TimeSignatureRegistry.lookup(sig)
        assert isinstance(spec, TimeSignatureSpec)

    @pytest.mark.parametrize("sig,expected", [
        ("6/8", True),
        ("12/8", True),
        ("2/4", False),
        ("3/4", False),
        ("4/4", False),
        ("5/4", False),
        ("7/8", False),
    ])
    def test_is_compound_classification(self, sig, expected):
        assert TimeSignatureRegistry.lookup(sig).is_compound is expected

    @pytest.mark.parametrize("sig,expected_power", [
        ("2/4", 2),
        ("3/4", 2),
        ("4/4", 2),
        ("5/4", 2),
        ("6/8", 3),
        ("7/8", 3),
        ("12/8", 3),
    ])
    def test_midi_denominator_power(self, sig, expected_power):
        assert TimeSignatureRegistry.lookup(sig).midi_denominator_power == expected_power

    @pytest.mark.parametrize("sig", ["2/4", "3/4", "4/4", "5/4", "6/8", "7/8", "12/8"])
    def test_numerator_matches_name(self, sig):
        expected = int(sig.split("/")[0])
        assert TimeSignatureRegistry.lookup(sig).numerator == expected

    @pytest.mark.parametrize("sig", ["2/4", "3/4", "4/4", "5/4", "6/8", "7/8", "12/8"])
    def test_denominator_matches_name(self, sig):
        expected = int(sig.split("/")[1])
        assert TimeSignatureRegistry.lookup(sig).denominator == expected

    @pytest.mark.parametrize("sig,expected_bpm", [
        ("2/4", 2),
        ("3/4", 3),
        ("4/4", 4),
        ("5/4", 5),
        ("6/8", 2),    # 6/3 = 2 dotted-quarter beats
        ("7/8", 7),
        ("12/8", 4),   # 12/3 = 4 dotted-quarter beats
    ])
    def test_beats_per_measure(self, sig, expected_bpm):
        assert TimeSignatureRegistry.lookup(sig).beats_per_measure == expected_bpm

    @pytest.mark.parametrize("sig,expected_div", [
        ("2/4", 2.0),
        ("3/4", 2.0),
        ("4/4", 2.0),
        ("5/4", 2.0),
        ("6/8", 3.0),
        ("7/8", 2.0),
        ("12/8", 3.0),
    ])
    def test_primary_division(self, sig, expected_div):
        assert TimeSignatureRegistry.lookup(sig).primary_division == expected_div

    @pytest.mark.parametrize("sig,expected_pbd", [
        ("2/4", 1.0),
        ("3/4", 1.0),
        ("4/4", 1.0),
        ("5/4", 1.0),
        ("6/8", 0.5),
        ("7/8", 0.5),
        ("12/8", 0.5),
    ])
    def test_primary_beat_duration(self, sig, expected_pbd):
        assert TimeSignatureRegistry.lookup(sig).primary_beat_duration == expected_pbd


# ---------------------------------------------------------------------------
# TestChordPatternLengthValidation
# ---------------------------------------------------------------------------

class TestChordPatternLengthValidation:
    """Mirrors the exact parametrized matrix from test_time_signature.py."""

    # 6/8 — accepts {2,3,6}, rejects {1,4,5}
    @pytest.mark.parametrize("length", [2, 3, 6])
    def test_6_8_accepts_2_3_6(self, length):
        assert TimeSignatureRegistry.lookup("6/8").verify_chord_pattern_length(length) is True

    @pytest.mark.parametrize("length", [1, 4, 5])
    def test_6_8_rejects_1_4_5(self, length):
        assert TimeSignatureRegistry.lookup("6/8").verify_chord_pattern_length(length) is False

    # 12/8 — accepts {2,3,6}, rejects {1,4,5}
    @pytest.mark.parametrize("length", [2, 3, 6])
    def test_12_8_accepts_2_3_6(self, length):
        assert TimeSignatureRegistry.lookup("12/8").verify_chord_pattern_length(length) is True

    @pytest.mark.parametrize("length", [1, 4, 5])
    def test_12_8_rejects_1_4_5(self, length):
        assert TimeSignatureRegistry.lookup("12/8").verify_chord_pattern_length(length) is False

    # 4/4 — accepts {1,2,4}, rejects {3,5}
    @pytest.mark.parametrize("length", [1, 2, 4])
    def test_4_4_accepts_1_2_4(self, length):
        assert TimeSignatureRegistry.lookup("4/4").verify_chord_pattern_length(length) is True

    @pytest.mark.parametrize("length", [3, 5])
    def test_4_4_rejects_3_5(self, length):
        assert TimeSignatureRegistry.lookup("4/4").verify_chord_pattern_length(length) is False

    # 3/4 — accepts {1,3}, rejects {2,4}
    @pytest.mark.parametrize("length", [1, 3])
    def test_3_4_accepts_1_3(self, length):
        assert TimeSignatureRegistry.lookup("3/4").verify_chord_pattern_length(length) is True

    @pytest.mark.parametrize("length", [2, 4])
    def test_3_4_rejects_2_4(self, length):
        assert TimeSignatureRegistry.lookup("3/4").verify_chord_pattern_length(length) is False

    # 2/4 — accepts {1,2}, rejects {3,4}
    @pytest.mark.parametrize("length", [1, 2])
    def test_2_4_accepts_1_2(self, length):
        assert TimeSignatureRegistry.lookup("2/4").verify_chord_pattern_length(length) is True

    @pytest.mark.parametrize("length", [3, 4])
    def test_2_4_rejects_3_4(self, length):
        assert TimeSignatureRegistry.lookup("2/4").verify_chord_pattern_length(length) is False

    # 5/4 — no constraint (empty frozenset -> always True)
    @pytest.mark.parametrize("length", [1, 2, 3, 4, 5])
    def test_5_4_accepts_any_length(self, length):
        assert TimeSignatureRegistry.lookup("5/4").verify_chord_pattern_length(length) is True

    # 7/8 — no constraint (empty frozenset -> always True)
    @pytest.mark.parametrize("length", [1, 2, 3, 4, 5])
    def test_7_8_accepts_any_length(self, length):
        assert TimeSignatureRegistry.lookup("7/8").verify_chord_pattern_length(length) is True


# ---------------------------------------------------------------------------
# TestBeatPatternLengthValidation — cosmetic-if preservation
# ---------------------------------------------------------------------------

class TestBeatPatternLengthValidation:
    """CRITICAL: Both compound and simple return length == numerator, NOT numerator/2."""

    def test_6_8_requires_length_6_not_3(self):
        spec = TimeSignatureRegistry.lookup("6/8")
        assert spec.verify_beat_pattern_length(6) is True
        assert spec.verify_beat_pattern_length(3) is False  # cosmetic-if: len == numerator, not numerator/2

    def test_12_8_requires_length_12_not_6(self):
        spec = TimeSignatureRegistry.lookup("12/8")
        assert spec.verify_beat_pattern_length(12) is True
        assert spec.verify_beat_pattern_length(6) is False

    @pytest.mark.parametrize("sig", ["2/4", "3/4", "4/4", "5/4", "7/8"])
    def test_simple_sigs_require_numerator_length(self, sig):
        spec = TimeSignatureRegistry.lookup(sig)
        assert spec.verify_beat_pattern_length(spec.numerator) is True
        assert spec.verify_beat_pattern_length(spec.numerator + 1) is False

    @pytest.mark.parametrize("sig", ["3/4", "4/4", "5/4", "7/8"])
    def test_simple_sigs_reject_off_by_one_below(self, sig):
        spec = TimeSignatureRegistry.lookup(sig)
        assert spec.verify_beat_pattern_length(spec.numerator - 1) is False


# ---------------------------------------------------------------------------
# TestMeasuresMultiplier
# ---------------------------------------------------------------------------

class TestMeasuresMultiplier:
    """Mirrors calculate_measures_for_time_signature behavior with base_length=4."""

    @pytest.mark.parametrize("sig,expected", [
        ("4/4", 4),   # 4 * 1.0 = 4
        ("3/4", 5),   # int(4 * 4/3) = int(5.33) = 5
        ("2/4", 8),   # 4 * 2.0 = 8
        ("6/8", 8),   # 4 * 2.0 = 8
        ("12/8", 8),  # 4 * 2.0 = 8
        ("5/4", 4),   # 4 * 1.0 = 4
        ("7/8", 4),   # 4 * 1.0 = 4
    ])
    def test_measures_for_base_4(self, sig, expected):
        spec = TimeSignatureRegistry.lookup(sig)
        assert spec.measures_for(4) == expected


# ---------------------------------------------------------------------------
# TestMeasureCountValid
# ---------------------------------------------------------------------------

class TestMeasureCountValid:
    """Compound sigs and 2/4 require even measure counts."""

    @pytest.mark.parametrize("sig", ["6/8", "12/8", "2/4"])
    def test_requires_even_measures_even_counts_valid(self, sig):
        spec = TimeSignatureRegistry.lookup(sig)
        assert spec.measure_count_valid(2) is True
        assert spec.measure_count_valid(4) is True
        assert spec.measure_count_valid(8) is True

    @pytest.mark.parametrize("sig", ["6/8", "12/8", "2/4"])
    def test_requires_even_measures_odd_counts_invalid(self, sig):
        spec = TimeSignatureRegistry.lookup(sig)
        assert spec.measure_count_valid(1) is False
        assert spec.measure_count_valid(3) is False
        assert spec.measure_count_valid(5) is False

    @pytest.mark.parametrize("sig", ["3/4", "4/4", "5/4", "7/8"])
    def test_non_even_sigs_accept_any_count(self, sig):
        spec = TimeSignatureRegistry.lookup(sig)
        assert spec.measure_count_valid(1) is True
        assert spec.measure_count_valid(3) is True
        assert spec.measure_count_valid(5) is True
        assert spec.measure_count_valid(7) is True


# ---------------------------------------------------------------------------
# TestNoteDurations
# ---------------------------------------------------------------------------

class TestNoteDurations:
    """note_duration_map() must match get_note_durations() legacy output."""

    @pytest.mark.parametrize("sig", ["6/8", "12/8"])
    def test_compound_note_durations(self, sig):
        nd = TimeSignatureRegistry.lookup(sig).note_duration_map()
        assert nd["whole"] == 6.0
        assert nd["half"] == 3.0
        assert nd["quarter"] == 1.5
        assert nd["eighth"] == 0.5
        assert nd["sixteenth"] == 0.25

    @pytest.mark.parametrize("sig", ["2/4", "3/4", "4/4", "5/4", "7/8"])
    def test_simple_note_durations(self, sig):
        nd = TimeSignatureRegistry.lookup(sig).note_duration_map()
        assert nd["whole"] == 4.0
        assert nd["half"] == 2.0
        assert nd["quarter"] == 1.0
        assert nd["eighth"] == 0.5
        assert nd["sixteenth"] == 0.25

    @pytest.mark.parametrize("sig", ["2/4", "3/4", "4/4", "5/4", "6/8", "7/8", "12/8"])
    def test_note_duration_map_returns_plain_dict(self, sig):
        nd = TimeSignatureRegistry.lookup(sig).note_duration_map()
        assert isinstance(nd, dict)
        assert set(nd.keys()) == {"whole", "half", "quarter", "eighth", "sixteenth"}


# ---------------------------------------------------------------------------
# TestMelodyDurations
# ---------------------------------------------------------------------------

class TestMelodyDurations:
    """melody_duration_candidates must match get_melody_durations() legacy output."""

    @pytest.mark.parametrize("sig", ["6/8", "12/8"])
    def test_compound_melody_durations(self, sig):
        md = TimeSignatureRegistry.lookup(sig).melody_duration_candidates
        assert md == (0.5, 1.5, 3.0)

    def test_3_4_melody_durations(self):
        md = TimeSignatureRegistry.lookup("3/4").melody_duration_candidates
        assert md == (0.5, 1.0, 1.5)

    @pytest.mark.parametrize("sig", ["2/4", "4/4", "5/4", "7/8"])
    def test_default_melody_durations(self, sig):
        md = TimeSignatureRegistry.lookup(sig).melody_duration_candidates
        assert md == (0.5, 1.0, 2.0)


# ---------------------------------------------------------------------------
# TestValidDurations
# ---------------------------------------------------------------------------

class TestValidDurations:
    """valid_durations must match musicgen/duration_validator.py compound/simple sets."""

    @pytest.mark.parametrize("sig", ["6/8", "12/8"])
    def test_compound_valid_durations(self, sig):
        vd = TimeSignatureRegistry.lookup(sig).valid_durations
        assert vd == frozenset({1.5, 1.0, 0.75, 0.5, 0.25})

    @pytest.mark.parametrize("sig", ["2/4", "3/4", "4/4", "5/4", "7/8"])
    def test_simple_valid_durations(self, sig):
        vd = TimeSignatureRegistry.lookup(sig).valid_durations
        assert vd == frozenset({4.0, 2.0, 1.0, 0.5, 0.25, 3.0, 1.5, 0.75})


# ---------------------------------------------------------------------------
# TestAlternatives
# ---------------------------------------------------------------------------

class TestAlternatives:
    """alternatives field must match time_signature_alternative() legacy dict."""

    @pytest.mark.parametrize("sig,expected", [
        ("4/4", ("2/4", "3/4", "6/8", "12/8")),
        ("3/4", ("6/8", "4/4", "2/4", "12/8")),
        ("2/4", ("4/4", "6/8", "3/4")),
        ("6/8", ("12/8", "3/4", "4/4", "2/4")),
        ("12/8", ("6/8", "4/4", "3/4")),
        ("7/8", ("4/4", "6/8", "5/4")),
        ("5/4", ("4/4", "7/8", "3/4")),
    ])
    def test_alternatives_field_preserves_legacy_dict(self, sig, expected):
        assert TimeSignatureRegistry.lookup(sig).alternatives == expected

    @pytest.mark.parametrize("sig", ["2/4", "3/4", "4/4", "5/4", "6/8", "7/8", "12/8"])
    def test_all_alternatives_are_valid_signatures(self, sig):
        all_sigs = set(TimeSignatureRegistry.all_signatures())
        for alt in TimeSignatureRegistry.lookup(sig).alternatives:
            assert alt in all_sigs


# ---------------------------------------------------------------------------
# TestSamplingWeights
# ---------------------------------------------------------------------------

class TestSamplingWeights:
    def test_sampling_weights_sum_to_one(self):
        total = sum(
            TimeSignatureRegistry.lookup(s).sampling_weight
            for s in TimeSignatureRegistry.all_signatures()
        )
        assert abs(total - 1.0) < 1e-9

    def test_4_4_has_highest_weight(self):
        assert TimeSignatureRegistry.lookup("4/4").sampling_weight == 0.50

    def test_3_4_weight(self):
        assert TimeSignatureRegistry.lookup("3/4").sampling_weight == 0.15

    def test_2_4_weight(self):
        assert TimeSignatureRegistry.lookup("2/4").sampling_weight == 0.10

    def test_6_8_weight(self):
        assert TimeSignatureRegistry.lookup("6/8").sampling_weight == 0.10

    @pytest.mark.parametrize("sig", ["12/8", "7/8", "5/4"])
    def test_rare_sigs_have_weight_0_05(self, sig):
        assert TimeSignatureRegistry.lookup(sig).sampling_weight == 0.05

    def test_sample_random_returns_valid_signature(self):
        all_sigs = set(TimeSignatureRegistry.all_signatures())
        for _ in range(20):
            result = TimeSignatureRegistry.sample_random()
            assert result in all_sigs


# ---------------------------------------------------------------------------
# TestMaxDuration
# ---------------------------------------------------------------------------

class TestMaxDuration:
    @pytest.mark.parametrize("sig,expected", [
        ("4/4", 4.0),
        ("3/4", 3.0),
        ("2/4", 2.0),
        ("5/4", 5.0),
        ("6/8", 3.0),   # DOTTED_QUARTER * (6/3) = 1.5 * 2
        ("7/8", 7.0),
        ("12/8", 6.0),  # DOTTED_QUARTER * (12/3) = 1.5 * 4
    ])
    def test_max_duration(self, sig, expected):
        assert TimeSignatureRegistry.lookup(sig).max_duration == expected


# ---------------------------------------------------------------------------
# TestMinDuration
# ---------------------------------------------------------------------------

class TestMinDuration:
    @pytest.mark.parametrize("sig", ["2/4", "3/4", "4/4", "5/4", "6/8", "7/8", "12/8"])
    def test_min_duration_is_sixteenth_for_all(self, sig):
        assert TimeSignatureRegistry.lookup(sig).min_duration == 0.25


# ---------------------------------------------------------------------------
# TestSpecImmutability
# ---------------------------------------------------------------------------

class TestSpecImmutability:
    """TimeSignatureSpec is frozen=True — field reassignment must raise."""

    def test_frozen_spec_raises_on_field_assignment(self):
        spec = TimeSignatureRegistry.lookup("4/4")
        with pytest.raises(Exception):  # FrozenInstanceError (dataclasses) or AttributeError
            spec.numerator = 999  # type: ignore[misc]
