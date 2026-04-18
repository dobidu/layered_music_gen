"""Time-signature registry for musicgen (R-S6).

Single source of truth for all time-signature metadata. Adding a new
time signature requires editing ONLY this file — add one entry to
TimeSignatureRegistry.REGISTRY per D-05.

Design decisions:
  D-04: Registry entries are frozen dataclasses.
  D-05: Registry owns ALL validation logic.
  D-06: Designed for flexibility — compound vs simple, unusual meters.

Note: Layer-specific duration sets (chord/melody/bass/beat) remain in
DurationValidator for Phase 2. Phase 3 generator extraction can
reconsider whether to hoist those into the registry.
"""
import logging
import random
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TimeSignatureSpec:
    """Immutable specification for a single time signature.

    All validation methods delegate to these fields so that the registry
    is the single source of truth. Do not mutate the note_durations dict
    (frozen=True blocks setattr, not dict content mutation).
    """
    name: str                                    # "4/4"
    numerator: int
    denominator: int
    is_compound: bool                            # denom==8 and num%3==0
    valid_chord_pattern_lengths: FrozenSet[int]  # {1,2,4} for 4/4; empty frozenset for 5/4, 7/8
    beat_pattern_length: int                     # == numerator for ALL sigs (cosmetic-if preserved)
    measure_multiplier: float                    # 1.0 for 4/4, 2.0 for compound/2-4, 4/3 for 3/4
    midi_denominator_power: int                  # 2 for denom=4, 3 for denom=8
    beats_per_measure: float                     # num/3 if compound else num
    valid_durations: FrozenSet[float]            # matches musicgen.duration_validator exactly
    primary_division: float                      # 3.0 compound, 2.0 simple
    max_duration: float                          # numerator for simple, DOTTED_QUARTER*(num/3) for compound
    min_duration: float                          # 0.25 (SIXTEENTH) for all
    primary_beat_duration: float                 # 0.5 for compound, 1.0 for simple
    note_durations: Dict[str, float]             # get_note_durations() return value frozen into field
    melody_durations: Tuple[float, ...]          # get_melody_durations() return value frozen into field
    sampling_weight: float                       # probability weight (sums to 1.0 across all 7)
    alternatives: Tuple[str, ...]                # time_signature_alternative() choices
    requires_even_measures: bool                 # True for compound and 2/4

    def verify_chord_pattern_length(self, length: int) -> bool:
        """Returns True if the chord pattern length is valid for this time signature.

        Empty valid_chord_pattern_lengths means 'no constraint' — returns True (5/4, 7/8).
        This preserves the default-True branch of music_gen.py:37.
        """
        if not self.valid_chord_pattern_lengths:
            return True
        return length in self.valid_chord_pattern_lengths

    def verify_beat_pattern_length(self, length: int) -> bool:
        """Returns True if beat pattern length equals numerator.

        CRITICAL: both compound and simple return length == numerator (cosmetic-if preserved).
        Tests in test_time_signature.py pin verify_beat_pattern([1,0,1], '6/8') is False,
        confirming that 6/8 requires exactly 6 beats, NOT 3.
        """
        return length == self.beat_pattern_length

    def measures_for(self, base_length: int) -> int:
        """Calculate adjusted measure count for this time signature."""
        return int(base_length * self.measure_multiplier)

    def measure_count_valid(self, count: int) -> bool:
        """Returns True if the measure count is valid (even when requires_even_measures)."""
        if self.requires_even_measures:
            return count % 2 == 0
        return True

    def note_duration_map(self) -> dict:
        """Return the note durations dict (whole/half/quarter/eighth/sixteenth)."""
        return dict(self.note_durations)

    @property
    def melody_duration_candidates(self) -> Tuple[float, ...]:
        """Return melody duration candidates as a tuple."""
        return self.melody_durations


# Note durations shared across compound and simple meters
_COMPOUND_NOTE_DURATIONS: Dict[str, float] = {
    "whole": 6.0,      # 6 eighth notes
    "half": 3.0,       # 3 eighth notes (dotted quarter)
    "quarter": 1.5,    # 1.5 eighth notes (dotted eighth)
    "eighth": 0.5,     # 1 eighth note
    "sixteenth": 0.25  # 1 sixteenth note
}

_SIMPLE_NOTE_DURATIONS: Dict[str, float] = {
    "whole": 4.0,      # 4 quarter notes
    "half": 2.0,       # 2 quarter notes
    "quarter": 1.0,    # 1 quarter note
    "eighth": 0.5,     # 1 eighth note
    "sixteenth": 0.25  # 1 sixteenth note
}

# Valid durations from musicgen/duration_validator.py (must match exactly)
_COMPOUND_VALID_DURATIONS: FrozenSet[float] = frozenset({1.5, 1.0, 0.75, 0.5, 0.25})
_SIMPLE_VALID_DURATIONS: FrozenSet[float] = frozenset({4.0, 2.0, 1.0, 0.5, 0.25, 3.0, 1.5, 0.75})


class TimeSignatureRegistry:
    """Registry of all supported time signatures.

    Adding a new time signature: add ONE entry to REGISTRY. All consumers
    (music_gen.py wrappers, DurationValidator adapter) automatically pick
    it up via lookup(). This satisfies R-S6.
    """

    REGISTRY: Dict[str, TimeSignatureSpec] = {
        "2/4": TimeSignatureSpec(
            name="2/4",
            numerator=2,
            denominator=4,
            is_compound=False,
            valid_chord_pattern_lengths=frozenset({1, 2}),
            beat_pattern_length=2,
            measure_multiplier=2.0,
            midi_denominator_power=2,
            beats_per_measure=2,
            valid_durations=_SIMPLE_VALID_DURATIONS,
            primary_division=2.0,
            max_duration=2.0,
            min_duration=0.25,
            primary_beat_duration=1.0,
            note_durations=_SIMPLE_NOTE_DURATIONS,
            melody_durations=(0.5, 1.0, 2.0),
            sampling_weight=0.10,
            alternatives=("4/4", "6/8", "3/4"),
            requires_even_measures=True,
        ),
        "3/4": TimeSignatureSpec(
            name="3/4",
            numerator=3,
            denominator=4,
            is_compound=False,
            valid_chord_pattern_lengths=frozenset({1, 3}),
            beat_pattern_length=3,
            measure_multiplier=4 / 3,
            midi_denominator_power=2,
            beats_per_measure=3,
            valid_durations=_SIMPLE_VALID_DURATIONS,
            primary_division=2.0,
            max_duration=3.0,
            min_duration=0.25,
            primary_beat_duration=1.0,
            note_durations=_SIMPLE_NOTE_DURATIONS,
            melody_durations=(0.5, 1.0, 1.5),
            sampling_weight=0.15,
            alternatives=("6/8", "4/4", "2/4", "12/8"),
            requires_even_measures=False,
        ),
        "4/4": TimeSignatureSpec(
            name="4/4",
            numerator=4,
            denominator=4,
            is_compound=False,
            valid_chord_pattern_lengths=frozenset({1, 2, 4}),
            beat_pattern_length=4,
            measure_multiplier=1.0,
            midi_denominator_power=2,
            beats_per_measure=4,
            valid_durations=_SIMPLE_VALID_DURATIONS,
            primary_division=2.0,
            max_duration=4.0,
            min_duration=0.25,
            primary_beat_duration=1.0,
            note_durations=_SIMPLE_NOTE_DURATIONS,
            melody_durations=(0.5, 1.0, 2.0),
            sampling_weight=0.50,
            alternatives=("2/4", "3/4", "6/8", "12/8"),
            requires_even_measures=False,
        ),
        "5/4": TimeSignatureSpec(
            name="5/4",
            numerator=5,
            denominator=4,
            is_compound=False,
            valid_chord_pattern_lengths=frozenset(),  # empty → default-True path (no constraint)
            beat_pattern_length=5,
            measure_multiplier=1.0,
            midi_denominator_power=2,
            beats_per_measure=5,
            valid_durations=_SIMPLE_VALID_DURATIONS,
            primary_division=2.0,
            max_duration=5.0,
            min_duration=0.25,
            primary_beat_duration=1.0,
            note_durations=_SIMPLE_NOTE_DURATIONS,
            melody_durations=(0.5, 1.0, 2.0),
            sampling_weight=0.05,
            alternatives=("4/4", "7/8", "3/4"),
            requires_even_measures=False,
        ),
        "6/8": TimeSignatureSpec(
            name="6/8",
            numerator=6,
            denominator=8,
            is_compound=True,
            valid_chord_pattern_lengths=frozenset({2, 3, 6}),
            beat_pattern_length=6,          # CRITICAL: must be 6, NOT 3 (cosmetic-if preserved)
            measure_multiplier=2.0,
            midi_denominator_power=3,
            beats_per_measure=2,            # 6/3 = 2 dotted-quarter beats
            valid_durations=_COMPOUND_VALID_DURATIONS,
            primary_division=3.0,
            max_duration=3.0,              # DOTTED_QUARTER * (6/3) = 1.5 * 2
            min_duration=0.25,
            primary_beat_duration=0.5,     # eighth note
            note_durations=_COMPOUND_NOTE_DURATIONS,
            melody_durations=(0.5, 1.5, 3.0),
            sampling_weight=0.10,
            alternatives=("12/8", "3/4", "4/4", "2/4"),
            requires_even_measures=True,
        ),
        "7/8": TimeSignatureSpec(
            name="7/8",
            numerator=7,
            denominator=8,
            is_compound=False,             # 7%3 != 0 → not compound despite denom=8
            valid_chord_pattern_lengths=frozenset(),  # empty → default-True path (no constraint)
            beat_pattern_length=7,
            measure_multiplier=1.0,
            midi_denominator_power=3,
            beats_per_measure=7,
            valid_durations=_SIMPLE_VALID_DURATIONS,
            primary_division=2.0,
            max_duration=7.0,
            min_duration=0.25,
            primary_beat_duration=0.5,     # eighth note (denom=8)
            note_durations=_SIMPLE_NOTE_DURATIONS,
            melody_durations=(0.5, 1.0, 2.0),
            sampling_weight=0.05,
            alternatives=("4/4", "6/8", "5/4"),
            requires_even_measures=False,
        ),
        "12/8": TimeSignatureSpec(
            name="12/8",
            numerator=12,
            denominator=8,
            is_compound=True,
            valid_chord_pattern_lengths=frozenset({2, 3, 6}),
            beat_pattern_length=12,         # CRITICAL: must be 12, NOT 6 (cosmetic-if preserved)
            measure_multiplier=2.0,
            midi_denominator_power=3,
            beats_per_measure=4,            # 12/3 = 4 dotted-quarter beats
            valid_durations=_COMPOUND_VALID_DURATIONS,
            primary_division=3.0,
            max_duration=6.0,              # DOTTED_QUARTER * (12/3) = 1.5 * 4
            min_duration=0.25,
            primary_beat_duration=0.5,     # eighth note
            note_durations=_COMPOUND_NOTE_DURATIONS,
            melody_durations=(0.5, 1.5, 3.0),
            sampling_weight=0.05,
            alternatives=("6/8", "4/4", "3/4"),
            requires_even_measures=True,
        ),
    }

    @classmethod
    def lookup(cls, time_signature: str) -> TimeSignatureSpec:
        """Look up a time signature spec. Raises KeyError if not found."""
        return cls.REGISTRY[time_signature]

    @classmethod
    def all_signatures(cls) -> List[str]:
        """Return all registered time signature strings."""
        return list(cls.REGISTRY.keys())

    @classmethod
    def sample_random(cls, rng: Optional[random.Random] = None) -> str:
        """Weighted random selection. Replaces generate_random_time_signature threshold-loop.

        Uses random.choices — always returns a value (fixes Pitfall 5 missing-return bug).
        rng parameter allows injected Random for Phase 5 seed discipline.
        """
        sigs = cls.all_signatures()
        weights = [cls.REGISTRY[s].sampling_weight for s in sigs]
        chooser = rng.choices if rng else random.choices
        return chooser(sigs, weights=weights, k=1)[0]
