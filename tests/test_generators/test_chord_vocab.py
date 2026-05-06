"""Tests for extended chord vocabulary + inversions (v0.2 Phase 2).

Covers:
  - _pick_chord_type: hard filter exclusion, soft weight distribution, no-genre fallback
  - _pick_inversion: weighted draw, no-genre fallback (root position)
  - _build_chord_voicing: MIDI note layout per type + inversion
  - generate_chord_progression with genre_spec: respects hard filter, produces valid MIDI
  - Determinism: same seed + genre_spec → same output
  - Backward compat: no genre_spec → behavior identical to pre-Phase-2
"""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path
from collections import Counter

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "src"))

from musicgen.generators.chord import (
    generate_chord_progression,
    _pick_chord_type,
    _pick_inversion,
    _build_chord_voicing,
    CHORD_TYPES,
    INVERSION_NAMES,
)
from musicgen.genre import GenreSpec

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
CHORD_PATTERNS = os.path.join(REPO_ROOT, "chord_patterns.txt")


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

class TestChordTypeConstants:
    def test_all_expected_types_present(self):
        expected = {
            "maj", "min", "dom7", "maj7", "m7", "m7b5",
            "dim7", "sus2", "sus4", "9", "maj9", "m9", "add9",
        }
        assert expected.issubset(set(CHORD_TYPES))

    def test_inversion_names(self):
        assert set(INVERSION_NAMES) == {"root", "first", "second", "third"}


# ---------------------------------------------------------------------------
# _pick_chord_type
# ---------------------------------------------------------------------------

class TestPickChordType:
    def test_no_genre_returns_valid_type(self):
        rng = random.Random(42)
        t = _pick_chord_type(rng, "I", genre_spec=None)
        assert t in CHORD_TYPES

    def test_hard_filter_excludes_disallowed(self):
        spec = GenreSpec(
            name="jazz",
            chord_type_hard_filter=["maj7", "m7", "dom7"],
            chord_type_weights={"maj7": 0.5, "m7": 0.3, "dom7": 0.2},
        )
        rng = random.Random(0)
        results = {_pick_chord_type(rng, "I", genre_spec=spec) for _ in range(200)}
        assert results.issubset({"maj7", "m7", "dom7"})

    def test_soft_weights_shift_distribution(self):
        spec = GenreSpec(
            name="biased",
            chord_type_weights={"maj7": 0.99, "min": 0.01},
        )
        rng = random.Random(1)
        results = [_pick_chord_type(rng, "I", genre_spec=spec) for _ in range(500)]
        counts = Counter(results)
        assert counts["maj7"] > counts.get("min", 0) * 5

    def test_hard_filter_empty_falls_back_to_all_types(self):
        spec = GenreSpec(name="open", chord_type_hard_filter=None)
        rng = random.Random(7)
        results = {_pick_chord_type(rng, "I", genre_spec=spec) for _ in range(500)}
        assert len(results) > 3

    def test_deterministic_with_seed(self):
        spec = GenreSpec(name="jazz", chord_type_weights={"maj7": 0.6, "m7": 0.4})
        t1 = _pick_chord_type(random.Random(99), "I", genre_spec=spec)
        t2 = _pick_chord_type(random.Random(99), "I", genre_spec=spec)
        assert t1 == t2

    def test_hard_filter_single_type_always_returns_it(self):
        spec = GenreSpec(name="locked", chord_type_hard_filter=["dom7"])
        rng = random.Random(0)
        for _ in range(50):
            assert _pick_chord_type(rng, "I", genre_spec=spec) == "dom7"


# ---------------------------------------------------------------------------
# _pick_inversion
# ---------------------------------------------------------------------------

class TestPickInversion:
    def test_no_genre_returns_root(self):
        rng = random.Random(0)
        for _ in range(20):
            assert _pick_inversion(rng, genre_spec=None) == "root"

    def test_weighted_inversion_respected(self):
        spec = GenreSpec(
            name="jazz",
            inversion_weights={"root": 0.0, "first": 1.0, "second": 0.0, "third": 0.0},
        )
        rng = random.Random(0)
        for _ in range(50):
            assert _pick_inversion(rng, genre_spec=spec) == "first"

    def test_valid_inversion_name_returned(self):
        spec = GenreSpec(
            name="any",
            inversion_weights={"root": 0.25, "first": 0.25, "second": 0.25, "third": 0.25},
        )
        rng = random.Random(5)
        for _ in range(100):
            assert _pick_inversion(rng, genre_spec=spec) in INVERSION_NAMES

    def test_deterministic_with_seed(self):
        spec = GenreSpec(
            name="j",
            inversion_weights={"root": 0.5, "first": 0.3, "second": 0.2},
        )
        i1 = _pick_inversion(random.Random(42), genre_spec=spec)
        i2 = _pick_inversion(random.Random(42), genre_spec=spec)
        assert i1 == i2


# ---------------------------------------------------------------------------
# _build_chord_voicing
# ---------------------------------------------------------------------------

class TestBuildChordVoicing:
    """MIDI note layout correctness per chord type and inversion."""

    def _notes(self, chord_type, inversion="root", key="C", numeral="I"):
        return _build_chord_voicing(numeral, chord_type, inversion, key)

    def test_returns_list_of_ints(self):
        notes = self._notes("maj")
        assert all(isinstance(n, int) for n in notes)

    def test_maj_triad_root_three_notes(self):
        notes = self._notes("maj")
        assert len(notes) == 3

    def test_dom7_four_notes(self):
        notes = self._notes("dom7")
        assert len(notes) == 4

    def test_maj7_four_notes(self):
        notes = self._notes("maj7")
        assert len(notes) == 4

    def test_m7_four_notes(self):
        notes = self._notes("m7")
        assert len(notes) == 4

    def test_m7b5_four_notes(self):
        notes = self._notes("m7b5")
        assert len(notes) == 4

    def test_dim7_four_notes(self):
        notes = self._notes("dim7")
        assert len(notes) == 4

    def test_sus2_three_notes(self):
        notes = self._notes("sus2")
        assert len(notes) == 3

    def test_sus4_three_notes(self):
        notes = self._notes("sus4")
        assert len(notes) == 3

    def test_9_five_notes(self):
        notes = self._notes("9")
        assert len(notes) == 5

    def test_maj9_five_notes(self):
        notes = self._notes("maj9")
        assert len(notes) == 5

    def test_m9_five_notes(self):
        notes = self._notes("m9")
        assert len(notes) == 5

    def test_add9_four_notes(self):
        notes = self._notes("add9")
        assert len(notes) == 4

    def test_root_inversion_ascending(self):
        notes = self._notes("maj", inversion="root")
        assert notes == sorted(notes)

    def test_first_inversion_root_at_top(self):
        root_notes = self._notes("maj", inversion="root")
        first_notes = self._notes("maj", inversion="first")
        # First inversion: lowest note of root position moved up one octave
        assert first_notes[0] == root_notes[1]

    def test_second_inversion_shifts_correctly(self):
        root_notes = self._notes("maj", inversion="root")
        second_notes = self._notes("maj", inversion="second")
        assert second_notes[0] == root_notes[2]

    def test_third_inversion_only_for_four_note_chords(self):
        # Third inversion valid for 7th chords (4 notes)
        notes = self._notes("dom7", inversion="third")
        assert len(notes) == 4

    def test_third_inversion_on_triad_falls_back_to_second(self):
        # Triad has no 4th note; third inversion same as second
        notes_second = self._notes("maj", inversion="second")
        notes_third = self._notes("maj", inversion="third")
        assert notes_second == notes_third

    def test_notes_in_valid_midi_range(self):
        for chord_type in CHORD_TYPES:
            notes = self._notes(chord_type)
            assert all(0 <= n <= 127 for n in notes), f"{chord_type}: note out of MIDI range"

    def test_c_major_root_voicing(self):
        # C major I chord: C4=60, E4=64, G4=67
        notes = self._notes("maj", inversion="root", key="C", numeral="I")
        assert notes[0] == 60   # C4
        assert notes[1] == 64   # E4
        assert notes[2] == 67   # G4


# ---------------------------------------------------------------------------
# generate_chord_progression with genre_spec
# ---------------------------------------------------------------------------

class TestGenerateChordProgressionWithGenre:
    def test_no_genre_backward_compat(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        pattern, mid = generate_chord_progression(
            key="C", tempo=120, time_signature="4/4", measures=2,
            name="song0-verse", part="verse",
            pattern_file=CHORD_PATTERNS,
            rng=random.Random(42),
        )
        assert os.path.isfile(mid)
        assert isinstance(pattern, list)

    def test_genre_spec_produces_valid_midi(self, tmp_path, monkeypatch):
        spec = GenreSpec(
            name="jazz",
            chord_type_hard_filter=["maj7", "m7", "dom7"],
            chord_type_weights={"maj7": 0.5, "m7": 0.3, "dom7": 0.2},
            inversion_weights={"root": 0.6, "first": 0.4},
        )
        monkeypatch.chdir(tmp_path)
        pattern, mid = generate_chord_progression(
            key="C", tempo=120, time_signature="4/4", measures=2,
            name="song1-verse", part="verse",
            pattern_file=CHORD_PATTERNS,
            rng=random.Random(7),
            genre_spec=spec,
        )
        assert os.path.isfile(mid)
        assert Path(mid).stat().st_size > 0

    def test_determinism_with_genre_spec(self, tmp_path, monkeypatch):
        spec = GenreSpec(
            name="jazz",
            chord_type_weights={"maj7": 0.7, "m7": 0.3},
            inversion_weights={"root": 0.5, "first": 0.5},
        )
        monkeypatch.chdir(tmp_path)
        args = dict(
            key="Am", tempo=100, time_signature="4/4", measures=2,
            name="songdet-verse", part="verse",
            pattern_file=CHORD_PATTERNS,
            genre_spec=spec,
        )
        _, mid1 = generate_chord_progression(**args, rng=random.Random(55))
        b1 = Path(mid1).read_bytes()
        os.remove(mid1)
        _, mid2 = generate_chord_progression(**args, rng=random.Random(55))
        assert b1 == Path(mid2).read_bytes()

    def test_different_seeds_can_differ(self, tmp_path, monkeypatch):
        spec = GenreSpec(
            name="jazz",
            chord_type_weights={"maj7": 0.5, "m7": 0.3, "dom7": 0.2},
            inversion_weights={"root": 0.4, "first": 0.3, "second": 0.2, "third": 0.1},
        )
        monkeypatch.chdir(tmp_path)
        results = set()
        for seed in range(20):
            _, mid = generate_chord_progression(
                key="C", tempo=120, time_signature="4/4", measures=2,
                name=f"song{seed}-verse", part="verse",
                pattern_file=CHORD_PATTERNS,
                rng=random.Random(seed),
                genre_spec=spec,
            )
            results.add(Path(mid).read_bytes())
        assert len(results) > 1
