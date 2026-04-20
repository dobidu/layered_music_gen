"""
Pure-method unit tests for musicgen.duration_validator.DurationValidator.

Pin the CURRENT behavior of `get_suggested_duration` and `get_valid_duration`
across the supported time signatures (2/4, 3/4, 4/4, 5/4, 6/8, 7/8, 12/8) and
the four layer types (`melody`, `chord`, `bass`, `beat`).
"""
import pytest

from musicgen.duration_validator import DurationValidator, NoteValue


@pytest.fixture
def validator():
    return DurationValidator()


class TestGetSuggestedDuration:
    @pytest.mark.parametrize(
        "time_signature", ["2/4", "3/4", "4/4", "5/4", "6/8", "7/8", "12/8"]
    )
    @pytest.mark.parametrize("layer_type", ["melody", "chord", "bass", "beat"])
    def test_returns_positive_float(self, validator, time_signature, layer_type):
        result = validator.get_suggested_duration(time_signature, layer_type)
        assert isinstance(result, (int, float))
        assert result > 0

    # Pinned exact suggested-duration values from source:
    # simple meter → QUARTER(1.0)/HALF(2.0)/QUARTER(1.0)/QUARTER(1.0)
    # compound meter → DOTTED_QUARTER(1.5)/DOTTED_HALF(3.0)/DOTTED_QUARTER(1.5)/EIGHTH(0.5)
    def test_simple_4_4_melody_is_quarter(self, validator):
        assert validator.get_suggested_duration("4/4", "melody") == NoteValue.QUARTER.value

    def test_simple_4_4_chord_is_half(self, validator):
        assert validator.get_suggested_duration("4/4", "chord") == NoteValue.HALF.value

    def test_simple_3_4_bass_is_quarter(self, validator):
        assert validator.get_suggested_duration("3/4", "bass") == NoteValue.QUARTER.value

    def test_simple_2_4_beat_is_quarter(self, validator):
        assert validator.get_suggested_duration("2/4", "beat") == NoteValue.QUARTER.value

    def test_compound_6_8_melody_is_dotted_quarter(self, validator):
        assert validator.get_suggested_duration("6/8", "melody") == NoteValue.DOTTED_QUARTER.value

    def test_compound_6_8_chord_is_dotted_half(self, validator):
        assert validator.get_suggested_duration("6/8", "chord") == NoteValue.DOTTED_HALF.value

    def test_compound_12_8_bass_is_dotted_quarter(self, validator):
        assert validator.get_suggested_duration("12/8", "bass") == NoteValue.DOTTED_QUARTER.value

    def test_compound_6_8_beat_is_eighth(self, validator):
        assert validator.get_suggested_duration("6/8", "beat") == NoteValue.EIGHTH.value


class TestGetValidDuration:
    @pytest.mark.parametrize("time_signature", ["2/4", "3/4", "4/4", "6/8"])
    def test_duration_never_exceeds_remaining_beats(self, validator, time_signature):
        # Whatever the proposed duration is, it must be clamped to remaining_beats.
        result = validator.get_valid_duration(
            duration=2.0,
            time_signature=time_signature,
            remaining_beats=0.5,
            layer_type="melody",
        )
        assert result <= 0.5

    @pytest.mark.parametrize("layer_type", ["melody", "chord", "bass", "beat"])
    def test_returns_positive_float_for_each_layer(self, validator, layer_type):
        result = validator.get_valid_duration(
            duration=1.0,
            time_signature="4/4",
            remaining_beats=4.0,
            layer_type=layer_type,
        )
        assert isinstance(result, (int, float))
        assert result > 0

    def test_4_4_chord_duration_1_0_stays_quarter(self, validator):
        # 4/4 chord valid set is {WHOLE, HALF, QUARTER}; closest to 1.0 is QUARTER.
        result = validator.get_valid_duration(
            duration=1.0,
            time_signature="4/4",
            remaining_beats=4.0,
            layer_type="chord",
        )
        assert result == NoteValue.QUARTER.value

    def test_4_4_chord_duration_3_5_picks_whole_or_half(self, validator):
        # 4/4 chord valid set is {WHOLE(4.0), HALF(2.0), QUARTER(1.0)}.
        # Closest to 3.5 is WHOLE(4.0) (distance 0.5) over HALF(2.0) (distance 1.5).
        # Then clamped to remaining_beats=4.0 → still 4.0.
        result = validator.get_valid_duration(
            duration=3.5,
            time_signature="4/4",
            remaining_beats=4.0,
            layer_type="chord",
        )
        assert result == NoteValue.WHOLE.value

    def test_clamps_to_remaining_beats_even_when_valid_duration_is_larger(self, validator):
        # 4/4 chord nearest to 3.5 is WHOLE(4.0), but remaining_beats=1.0 clamps it.
        result = validator.get_valid_duration(
            duration=3.5,
            time_signature="4/4",
            remaining_beats=1.0,
            layer_type="chord",
        )
        assert result == 1.0


class TestNoteValue:
    def test_note_value_enum_has_expected_members(self):
        names = {member.name for member in NoteValue}
        # Spot-check well-known members from musicgen/duration_validator.py
        assert "WHOLE" in names
        assert "HALF" in names
        assert "QUARTER" in names
        assert "EIGHTH" in names
        assert "SIXTEENTH" in names
        assert "DOTTED_QUARTER" in names

    def test_note_value_numeric_values(self):
        assert NoteValue.WHOLE.value == 4.0
        assert NoteValue.HALF.value == 2.0
        assert NoteValue.QUARTER.value == 1.0
        assert NoteValue.EIGHTH.value == 0.5
        assert NoteValue.SIXTEENTH.value == 0.25
