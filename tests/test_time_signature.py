"""
Pure-function unit tests for time-signature helpers in music_gen.py.

These tests pin the CURRENT behavior of the functions. Phase 2 will move
them into a TimeSignatureRegistry; the tests should still pass after that
move (delegation, not behavior change).
"""
import pytest

import music_gen


class TestVerifyPatternForTimeSignature:
    """music_gen.verify_pattern_for_time_signature(chord_pattern, time_signature)"""

    # Compound (6/8, 12/8) — valid pattern lengths are {2, 3, 6}
    @pytest.mark.parametrize("length", [2, 3, 6])
    def test_compound_6_8_accepts_2_3_6(self, length):
        pattern = ["I"] * length
        assert music_gen.verify_pattern_for_time_signature(pattern, "6/8") is True

    @pytest.mark.parametrize("length", [1, 4, 5, 7, 8])
    def test_compound_6_8_rejects_other_lengths(self, length):
        pattern = ["I"] * length
        assert music_gen.verify_pattern_for_time_signature(pattern, "6/8") is False

    @pytest.mark.parametrize("length", [2, 3, 6])
    def test_compound_12_8_accepts_2_3_6(self, length):
        pattern = ["I"] * length
        assert music_gen.verify_pattern_for_time_signature(pattern, "12/8") is True

    # 4/4 — valid pattern lengths are {1, 2, 4}
    @pytest.mark.parametrize("length", [1, 2, 4])
    def test_four_four_accepts_1_2_4(self, length):
        assert music_gen.verify_pattern_for_time_signature(["I"] * length, "4/4") is True

    @pytest.mark.parametrize("length", [3, 5, 6, 7])
    def test_four_four_rejects_other_lengths(self, length):
        assert music_gen.verify_pattern_for_time_signature(["I"] * length, "4/4") is False

    # 3/4 — valid pattern lengths are {1, 3}
    @pytest.mark.parametrize("length", [1, 3])
    def test_three_four_accepts_1_and_3(self, length):
        assert music_gen.verify_pattern_for_time_signature(["I"] * length, "3/4") is True

    @pytest.mark.parametrize("length", [2, 4, 5])
    def test_three_four_rejects_other_lengths(self, length):
        assert music_gen.verify_pattern_for_time_signature(["I"] * length, "3/4") is False

    # 2/4 — valid pattern lengths are {1, 2}
    @pytest.mark.parametrize("length", [1, 2])
    def test_two_four_accepts_1_and_2(self, length):
        assert music_gen.verify_pattern_for_time_signature(["I"] * length, "2/4") is True

    @pytest.mark.parametrize("length", [3, 4])
    def test_two_four_rejects_other_lengths(self, length):
        assert music_gen.verify_pattern_for_time_signature(["I"] * length, "2/4") is False

    # Default branch — anything not in the explicit cases returns True
    def test_unknown_signature_defaults_true(self):
        # 5/4 hits the default branch because numerator==5 isn't in {2,3,4} and
        # denominator==4 isn't compound.
        assert music_gen.verify_pattern_for_time_signature(["I", "IV", "V"], "5/4") is True


class TestVerifyBeatPattern:
    """music_gen.verify_beat_pattern(pattern, time_signature)

    Verified body (music_gen.py:42-52): BOTH branches return
    `len(pattern) == numerator`. The conditional is cosmetic. Effective rule:
    len(pattern) must equal the numerator of the time signature.
    """

    # Compound meters — len must equal numerator
    def test_compound_6_8_length_6_ok(self):
        assert music_gen.verify_beat_pattern([1, 0, 1, 0, 1, 0], "6/8") is True

    def test_compound_6_8_length_3_not_ok(self):
        # Compound rule requires len == numerator (6), not numerator/2
        assert music_gen.verify_beat_pattern([1, 0, 1], "6/8") is False

    def test_compound_12_8_length_12_ok(self):
        assert music_gen.verify_beat_pattern([1] * 12, "12/8") is True

    def test_compound_12_8_length_6_not_ok(self):
        assert music_gen.verify_beat_pattern([1] * 6, "12/8") is False

    # Simple meter 4/4 — len must equal 4
    def test_simple_4_4_length_4_ok(self):
        # Verified: denom=4, numerator=4, takes else branch → len(pattern) == 4
        assert music_gen.verify_beat_pattern([1, 0, 1, 0], "4/4") is True

    def test_simple_4_4_length_3_not_ok(self):
        # Verified: [1, 0, 1] has len 3, numerator is 4 → False
        assert music_gen.verify_beat_pattern([1, 0, 1], "4/4") is False

    def test_simple_4_4_length_5_not_ok(self):
        assert music_gen.verify_beat_pattern([1, 0, 1, 0, 1], "4/4") is False

    # Simple meter 3/4 — len must equal 3
    def test_simple_3_4_length_3_ok(self):
        assert music_gen.verify_beat_pattern([1, 0, 0], "3/4") is True

    def test_simple_3_4_length_4_not_ok(self):
        assert music_gen.verify_beat_pattern([1, 0, 0, 0], "3/4") is False


class TestValidateMeasures:
    """music_gen.validate_measures(measures, signatures)

    Verified body (music_gen.py:66-80): returns False ONLY when
      (a) compound meter (denom=8, num%3==0) with ODD measure_count, OR
      (b) 2/4 with ODD measure_count.
    Everything else returns True — including 4/4 with ANY int value (even 0),
    and 0-measures for compound/2-4 since 0 % 2 == 0.
    """

    def test_all_parts_with_positive_measures_valid_signatures_passes(self):
        # 4/4: denom=4 (not 8-compound), numerator=4 (not 2) → neither False
        # branch fires → returns True for any positive int count.
        measures = {"intro": 4, "verse": 8, "chorus": 8, "outro": 4}
        signatures = {"intro": "4/4", "verse": "4/4", "chorus": "4/4", "outro": "4/4"}
        assert music_gen.validate_measures(measures, signatures) is True

    def test_zero_measures_4_4_returns_true(self):
        # 4/4 path doesn't check measure_count at all → always True.
        measures = {"intro": 0}
        signatures = {"intro": "4/4"}
        assert music_gen.validate_measures(measures, signatures) is True

    def test_compound_6_8_even_measures_passes(self):
        # 6/8: denom=8, num=6, 6%3==0 → compound branch.
        # measure_count=4 is even → 4 % 2 == 0 → passes.
        assert music_gen.validate_measures({"verse": 4}, {"verse": "6/8"}) is True

    def test_compound_6_8_odd_measures_fails(self):
        # 6/8 compound with ODD count → returns False.
        assert music_gen.validate_measures({"verse": 3}, {"verse": "6/8"}) is False

    def test_compound_6_8_zero_measures_passes(self):
        # 0 % 2 == 0 → compound branch doesn't trigger False → returns True.
        assert music_gen.validate_measures({"verse": 0}, {"verse": "6/8"}) is True

    def test_two_four_odd_measures_fails(self):
        # 2/4 with odd count hits the `elif numerator == 2` False branch.
        assert music_gen.validate_measures({"intro": 3}, {"intro": "2/4"}) is False

    def test_two_four_even_measures_passes(self):
        assert music_gen.validate_measures({"intro": 4}, {"intro": "2/4"}) is True

    def test_three_four_any_count_passes(self):
        # 3/4: denom=4 (not compound), numerator=3 (not 2) → neither False
        # branch fires → any count returns True.
        assert music_gen.validate_measures({"bridge": 5}, {"bridge": "3/4"}) is True

    def test_empty_measures_dict_returns_true(self):
        # Empty loop body → returns True unconditionally.
        assert music_gen.validate_measures({}, {}) is True
