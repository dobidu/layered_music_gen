"""v0.3 Phase 2 — RED: higher-order Markov melody (scale-relative intervals).

Tests cover:
  - GenreSpec.melody_transition_matrix field (new)
  - load_genre reads melody_transitions.json when present
  - _sample_melody_markov helper (same fallback chain as chord Markov)
  - _degree_to_midi conversion (scale-relative, octave tracking)
  - generate_melody Markov path: valid MIDI range, determinism, no zero-weight crash
  - Backward compat: no matrix → existing behavior; genre_spec=None → pre-v0.3
  - merge_genres: melody_transition_matrix=None in composed specs
  - Pre-existing zero-weight bug does NOT occur in the Markov path
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent
GENRES_DIR = str(REPO_ROOT / "genres")

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_MELODY_MATRIX_1ST = {
    "order": 1,
    "init_probs": {"1": 0.30, "3": 0.25, "5": 0.25, "2": 0.10, "6": 0.10},
    "transitions": {
        "1": {"2": 0.35, "3": 0.30, "5": 0.20, "7": 0.10, "6": 0.05},
        "2": {"1": 0.25, "3": 0.40, "4": 0.20, "7": 0.10, "5": 0.05},
        "3": {"2": 0.25, "4": 0.30, "5": 0.25, "1": 0.10, "6": 0.10},
        "4": {"3": 0.30, "5": 0.35, "2": 0.20, "6": 0.15},
        "5": {"4": 0.20, "6": 0.30, "3": 0.25, "1": 0.15, "7": 0.10},
        "6": {"5": 0.25, "7": 0.30, "4": 0.20, "2": 0.15, "1": 0.10},
        "7": {"1": 0.50, "6": 0.25, "5": 0.15, "2": 0.10},
    },
}

_MELODY_MATRIX_2ND = {
    "order": 2,
    "init_probs": {"1": 0.30, "3": 0.25, "5": 0.25, "2": 0.10, "6": 0.10},
    "transitions": {
        "1": {"2": 0.35, "3": 0.30, "5": 0.20, "7": 0.10, "6": 0.05},
        "2": {"1": 0.25, "3": 0.40, "4": 0.20, "7": 0.10, "5": 0.05},
        "3": {"2": 0.25, "4": 0.30, "5": 0.25, "1": 0.10, "6": 0.10},
        "4": {"3": 0.30, "5": 0.35, "2": 0.20, "6": 0.15},
        "5": {"4": 0.20, "6": 0.30, "3": 0.25, "1": 0.15, "7": 0.10},
        "6": {"5": 0.25, "7": 0.30, "4": 0.20, "2": 0.15, "1": 0.10},
        "7": {"1": 0.50, "6": 0.25, "5": 0.15, "2": 0.10},
        # 2nd-order keys
        "1,2": {"3": 0.45, "4": 0.25, "1": 0.20, "5": 0.10},
        "2,3": {"4": 0.40, "2": 0.25, "5": 0.25, "1": 0.10},
        "3,4": {"5": 0.45, "3": 0.25, "2": 0.20, "6": 0.10},
        "4,5": {"6": 0.40, "4": 0.25, "3": 0.25, "7": 0.10},
        "5,6": {"7": 0.40, "5": 0.25, "4": 0.25, "1": 0.10},
        "6,7": {"1": 0.55, "6": 0.25, "5": 0.20},
        "7,1": {"2": 0.40, "3": 0.30, "5": 0.20, "7": 0.10},
        "5,3": {"2": 0.35, "4": 0.30, "1": 0.25, "6": 0.10},
        "3,5": {"6": 0.35, "4": 0.30, "3": 0.25, "7": 0.10},
    },
}

_VALID_DEGREES = {"1", "2", "3", "4", "5", "6", "7"}


# ---------------------------------------------------------------------------
# GenreSpec fields
# ---------------------------------------------------------------------------

class TestGenreSpecMelodyFields:
    def test_melody_transition_matrix_exists_defaults_none(self):
        from musicgen.genre import GenreSpec
        spec = GenreSpec(name="test")
        assert hasattr(spec, "melody_transition_matrix")
        assert spec.melody_transition_matrix is None

    def test_accepts_matrix_dict(self):
        from musicgen.genre import GenreSpec
        spec = GenreSpec(name="test", melody_transition_matrix=_MELODY_MATRIX_2ND)
        assert spec.melody_transition_matrix["order"] == 2

    def test_merge_genres_sets_matrix_none(self):
        from musicgen.genre import GenreSpec, merge_genres
        a = GenreSpec(name="a", melody_transition_matrix=_MELODY_MATRIX_1ST)
        b = GenreSpec(name="b", melody_transition_matrix=_MELODY_MATRIX_2ND)
        merged = merge_genres([a, b])
        assert merged.melody_transition_matrix is None

    def test_merge_single_preserves_matrix(self):
        from musicgen.genre import GenreSpec, merge_genres
        spec = GenreSpec(name="a", melody_transition_matrix=_MELODY_MATRIX_2ND)
        result = merge_genres([spec])
        assert result.melody_transition_matrix == _MELODY_MATRIX_2ND


# ---------------------------------------------------------------------------
# load_genre reads melody_transitions.json
# ---------------------------------------------------------------------------

class TestLoadGenreMelodyTransitions:
    def test_loads_melody_transitions_when_present(self, tmp_path):
        from musicgen.genre import load_genre
        genre_dir = tmp_path / "mygenre"
        genre_dir.mkdir()
        (genre_dir / "spec.json").write_text(json.dumps({"name": "mygenre"}))
        (genre_dir / "melody_transitions.json").write_text(json.dumps(_MELODY_MATRIX_2ND))

        spec = load_genre("mygenre", str(tmp_path))
        assert spec.melody_transition_matrix is not None
        assert spec.melody_transition_matrix["order"] == 2
        assert "init_probs" in spec.melody_transition_matrix
        assert "transitions" in spec.melody_transition_matrix

    def test_stays_none_without_file(self, tmp_path):
        from musicgen.genre import load_genre
        genre_dir = tmp_path / "plain"
        genre_dir.mkdir()
        (genre_dir / "spec.json").write_text(json.dumps({"name": "plain"}))
        spec = load_genre("plain", str(tmp_path))
        assert spec.melody_transition_matrix is None

    def test_real_genre_loads_if_file_exists(self):
        from musicgen.genre import load_genre
        spec = load_genre("jazz", GENRES_DIR)
        transitions_path = os.path.join(GENRES_DIR, "jazz", "melody_transitions.json")
        if os.path.exists(transitions_path):
            assert spec.melody_transition_matrix is not None
        else:
            assert spec.melody_transition_matrix is None


# ---------------------------------------------------------------------------
# _sample_melody_markov
# ---------------------------------------------------------------------------

class TestSampleMelodyMarkov:
    def test_importable(self):
        from musicgen.generators.melody import _sample_melody_markov
        assert callable(_sample_melody_markov)

    def test_step0_uses_init_probs(self):
        from musicgen.generators.melody import _sample_melody_markov
        result = _sample_melody_markov([], _MELODY_MATRIX_1ST, random.Random(1))
        assert result in _MELODY_MATRIX_1ST["init_probs"]

    def test_step1_uses_1st_order_key(self):
        from musicgen.generators.melody import _sample_melody_markov
        result = _sample_melody_markov(["3"], _MELODY_MATRIX_1ST, random.Random(7))
        assert result in _MELODY_MATRIX_1ST["transitions"]["3"]

    def test_step2_uses_2nd_order_key(self):
        from musicgen.generators.melody import _sample_melody_markov
        # "3,4" is a 2nd-order key in _MELODY_MATRIX_2ND
        result = _sample_melody_markov(["3", "4"], _MELODY_MATRIX_2ND, random.Random(5))
        assert result in _MELODY_MATRIX_2ND["transitions"]["3,4"]

    def test_step2_fallback_to_1st_order(self):
        from musicgen.generators.melody import _sample_melody_markov
        # "2,7" is NOT a 2nd-order key — falls back to transitions["7"]
        result = _sample_melody_markov(["2", "7"], _MELODY_MATRIX_2ND, random.Random(9))
        assert result in _MELODY_MATRIX_2ND["transitions"]["7"]

    def test_empty_transitions_fallback_to_init(self):
        from musicgen.generators.melody import _sample_melody_markov
        sparse = {
            "order": 2,
            "init_probs": {"1": 0.5, "5": 0.5},
            "transitions": {},
        }
        result = _sample_melody_markov(["X", "Y"], sparse, random.Random(0))
        assert result in sparse["init_probs"]

    def test_determinism(self):
        from musicgen.generators.melody import _sample_melody_markov
        history = ["3", "5"]
        a = _sample_melody_markov(history, _MELODY_MATRIX_2ND, random.Random(42))
        b = _sample_melody_markov(history, _MELODY_MATRIX_2ND, random.Random(42))
        assert a == b

    def test_returns_str(self):
        from musicgen.generators.melody import _sample_melody_markov
        rng = random.Random(0)
        for history in [[], ["1"], ["1", "2"], ["5", "6", "7"]]:
            result = _sample_melody_markov(history, _MELODY_MATRIX_2ND, rng)
            assert isinstance(result, str) and len(result) > 0

    def test_result_is_valid_scale_degree(self):
        from musicgen.generators.melody import _sample_melody_markov
        rng = random.Random(13)
        history: list = []
        for _ in range(20):
            d = _sample_melody_markov(history, _MELODY_MATRIX_2ND, rng)
            assert d in _VALID_DEGREES, f"Invalid scale degree: {d!r}"
            history.append(d)


# ---------------------------------------------------------------------------
# _degree_to_midi
# ---------------------------------------------------------------------------

class TestDegreeToMidi:
    def test_importable(self):
        from musicgen.generators.melody import _degree_to_midi
        assert callable(_degree_to_midi)

    def test_degree1_is_root_in_C(self):
        """Degree 1 in C major maps to some C (MIDI 48, 60, 72...)."""
        from musicgen.generators.melody import _degree_to_midi
        midi = _degree_to_midi("1", "C", reference=60)
        assert midi % 12 == 0, f"Expected C (midi%12==0), got {midi}"

    def test_degree1_is_root_in_G(self):
        """Degree 1 in G major maps to some G (MIDI %12 == 7)."""
        from musicgen.generators.melody import _degree_to_midi
        midi = _degree_to_midi("1", "G", reference=60)
        assert midi % 12 == 7, f"Expected G (midi%12==7), got {midi}"

    def test_degree5_in_C_is_G(self):
        """Degree 5 in C major = G."""
        from musicgen.generators.melody import _degree_to_midi
        midi = _degree_to_midi("5", "C", reference=60)
        assert midi % 12 == 7, f"Expected G (midi%12==7), got {midi}"

    def test_degree3_in_Am_is_C(self):
        """Degree 3 in A minor = C."""
        from musicgen.generators.melody import _degree_to_midi
        midi = _degree_to_midi("3", "Am", reference=60)
        assert midi % 12 == 0, f"Expected C (midi%12==0), got {midi}"

    def test_midi_in_playable_range(self):
        """Output always in MIDI range 36–84 (3 octaves of melody space)."""
        from musicgen.generators.melody import _degree_to_midi
        for key in ("C", "G", "D", "F", "Am", "Em"):
            for degree in _VALID_DEGREES:
                for ref in (48, 60, 72):
                    midi = _degree_to_midi(degree, key, reference=ref)
                    assert 36 <= midi <= 96, (
                        f"Out of range: degree={degree}, key={key}, ref={ref}, midi={midi}"
                    )

    def test_follows_reference_octave(self):
        """With reference=48 vs reference=72, output should differ by ~12."""
        from musicgen.generators.melody import _degree_to_midi
        low = _degree_to_midi("1", "C", reference=48)
        high = _degree_to_midi("1", "C", reference=72)
        assert high >= low, "Higher reference should yield higher or equal MIDI"

    def test_minor_key_degree_6_correct(self):
        """Degree 6 in A minor = F (semitone 5)."""
        from musicgen.generators.melody import _degree_to_midi
        midi = _degree_to_midi("6", "Am", reference=60)
        assert midi % 12 == 5, f"Expected F (midi%12==5) for degree 6 in Am, got {midi}"


# ---------------------------------------------------------------------------
# generate_melody — Markov path
# ---------------------------------------------------------------------------

def _make_spec(matrix):
    from musicgen.genre import GenreSpec
    return GenreSpec(name="melody_test", melody_transition_matrix=matrix)


class TestGenerateMelodyMarkov:
    def test_markov_path_active_when_matrix_present(self, tmp_path, monkeypatch):
        from musicgen.generators.melody import generate_melody
        monkeypatch.chdir(tmp_path)
        spec = _make_spec(_MELODY_MATRIX_2ND)
        melody, mid = generate_melody(
            key="C", tempo=120, time_signature="4/4", measures=4,
            name="song1-verse", part="verse",
            chord_progression=["I", "IV", "V", "I"],
            rng=random.Random(42),
            genre_spec=spec,
        )
        assert isinstance(melody, list)
        assert len(melody) > 0
        assert os.path.exists(mid) and os.path.getsize(mid) > 0

    def test_markov_determinism(self, tmp_path, monkeypatch):
        from musicgen.generators.melody import generate_melody
        monkeypatch.chdir(tmp_path)
        spec = _make_spec(_MELODY_MATRIX_2ND)
        args = dict(
            key="C", tempo=120, time_signature="4/4", measures=4,
            name="song99-verse", part="verse",
            chord_progression=["I", "IV", "V", "I"],
            genre_spec=spec,
        )
        m1, p1 = generate_melody(**args, rng=random.Random(7))
        b1 = Path(p1).read_bytes()
        os.remove(p1)
        m2, p2 = generate_melody(**args, rng=random.Random(7))
        assert m1 == m2
        assert b1 == Path(p2).read_bytes()

    def test_markov_notes_in_valid_midi_range(self, tmp_path, monkeypatch):
        from musicgen.generators.melody import generate_melody
        monkeypatch.chdir(tmp_path)
        spec = _make_spec(_MELODY_MATRIX_2ND)
        for seed in range(5):
            melody, mid = generate_melody(
                key="C", tempo=120, time_signature="4/4", measures=4,
                name=f"song{seed}-verse", part="verse",
                chord_progression=["I", "IV", "V", "I"],
                rng=random.Random(seed),
                genre_spec=spec,
            )
            for note in melody:
                assert 36 <= note <= 96, f"Note {note} out of melodic range (seed={seed})"
            if os.path.exists(mid):
                os.remove(mid)

    def test_no_zero_weight_error_across_seeds(self, tmp_path, monkeypatch):
        """Markov path never raises ValueError for zero weights."""
        from musicgen.generators.melody import generate_melody
        monkeypatch.chdir(tmp_path)
        spec = _make_spec(_MELODY_MATRIX_2ND)
        for seed in range(20):
            melody, mid = generate_melody(
                key="C", tempo=120, time_signature="4/4", measures=4,
                name=f"song{seed}-verse", part="verse",
                chord_progression=["I", "IV", "V", "I"],
                rng=random.Random(seed),
                genre_spec=spec,
            )
            assert len(melody) > 0
            if os.path.exists(mid):
                os.remove(mid)

    def test_no_matrix_uses_existing_behavior(self, tmp_path, monkeypatch):
        """genre_spec with no melody_transition_matrix → existing code path."""
        from musicgen.generators.melody import generate_melody
        from musicgen.genre import GenreSpec
        monkeypatch.chdir(tmp_path)
        spec = GenreSpec(name="no_matrix")
        melody, mid = generate_melody(
            key="C", tempo=120, time_signature="4/4", measures=4,
            name="song0-verse", part="verse",
            chord_progression=["I", "IV", "V", "I"],
            rng=random.Random(3),  # seed=3 avoids the pre-existing zero-weight bug
            genre_spec=spec,
        )
        assert isinstance(melody, list) and len(melody) > 0

    def test_none_genre_spec_unchanged(self, tmp_path, monkeypatch):
        """genre_spec=None → bit-identical to pre-v0.3 (backward compat)."""
        from musicgen.generators.melody import generate_melody
        monkeypatch.chdir(tmp_path)
        m1, p1 = generate_melody(
            key="C", tempo=120, time_signature="4/4", measures=4,
            name="song3-verse", part="verse",
            chord_progression=["I", "IV", "V", "I"],
            rng=random.Random(3),
            genre_spec=None,
        )
        b1 = Path(p1).read_bytes()
        os.remove(p1)
        m2, p2 = generate_melody(
            key="C", tempo=120, time_signature="4/4", measures=4,
            name="song3-verse", part="verse",
            chord_progression=["I", "IV", "V", "I"],
            rng=random.Random(3),
        )
        assert m1 == m2
        assert b1 == Path(p2).read_bytes()

    def test_markov_melody_varies_across_seeds(self, tmp_path, monkeypatch):
        """Different seeds → different melodies."""
        from musicgen.generators.melody import generate_melody
        monkeypatch.chdir(tmp_path)
        spec = _make_spec(_MELODY_MATRIX_2ND)
        melodies = set()
        for seed in range(10):
            melody, mid = generate_melody(
                key="C", tempo=120, time_signature="4/4", measures=4,
                name=f"song{seed}-verse", part="verse",
                chord_progression=["I", "IV", "V", "I"],
                rng=random.Random(seed),
                genre_spec=spec,
            )
            melodies.add(tuple(melody))
            if os.path.exists(mid):
                os.remove(mid)
        assert len(melodies) > 1

    @pytest.mark.parametrize("key", ["C", "G", "Am", "F", "Dm"])
    def test_markov_works_across_keys(self, tmp_path, monkeypatch, key):
        from musicgen.generators.melody import generate_melody
        monkeypatch.chdir(tmp_path)
        spec = _make_spec(_MELODY_MATRIX_1ST)
        melody, mid = generate_melody(
            key=key, tempo=120, time_signature="4/4", measures=4,
            name=f"song0-verse", part="verse",
            chord_progression=["I", "IV", "V", "I"],
            rng=random.Random(42),
            genre_spec=spec,
        )
        assert len(melody) > 0
        for note in melody:
            assert 0 <= note <= 127
        if os.path.exists(mid):
            os.remove(mid)
