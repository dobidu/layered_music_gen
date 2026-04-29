"""R-P7 / D-24 regression guard: music21 must not mutate global random.

If this test starts failing, add a save_random_state() wrapper in sampler.py
and every generator that touches music21 (melody, bassline, chord), and wrap
each music21 call in the new contextmanager. See 03-CONTEXT.md D-23 for rationale.

Empirical audit 2026-04-18 against music21 9.9.1 confirmed music21 does NOT
mutate random.getstate(). This test converts that audit into a permanent
regression guard.
"""
import random

import pytest


class TestMusic21DoesNotMutateGlobalRandom:
    def test_roman_numeral_preserves_global_state(self):
        from music21 import roman
        state0 = random.getstate()
        for key in ["C", "G", "D", "Am", "Em"]:
            for sym in ["I", "IV", "V", "vi", "ii"]:
                rn = roman.RomanNumeral(sym, key)
                _ = list(rn.pitches)
                for p in rn.pitches:
                    _ = p.midi
        assert random.getstate() == state0

    def test_scale_preserves_global_state(self):
        from music21 import scale
        state0 = random.getstate()
        _ = scale.MajorScale("C")
        _ = scale.MinorScale("A")
        _ = scale.MajorScale("G")
        _ = scale.MinorScale("E")
        assert random.getstate() == state0

    def test_pitch_midi_roundtrip_preserves_global_state(self):
        from music21 import pitch
        state0 = random.getstate()
        for midi_val in [36, 48, 60, 72]:
            p = pitch.Pitch()
            p.midi = midi_val
            p.octave = 2
            _ = p.midi
        assert random.getstate() == state0
