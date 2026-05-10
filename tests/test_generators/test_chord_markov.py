"""v0.3 Phase 1 — RED: 2nd-order Markov chord progressions.

Tests cover:
  - GenreSpec.chord_transition_matrix field (new)
  - load_genre loads chord_transitions.json when present
  - _sample_chord_markov internal helper
  - generate_chord_progression uses Markov path when matrix present
  - Determinism (same seed → same sequence)
  - Boundary: init_probs at step 0; 1st-order at step 1; 2nd-order at step ≥ 2
  - Fallback chain: 2nd-order key not found → 1st-order key → uniform (no crash)
  - Backward compat: no matrix → pattern file path unchanged
  - merge_genres with matrix: merged spec has matrix=None (not merged)
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent
CHORD_PATTERNS = str(REPO_ROOT / "chord_patterns.txt")
GENRES_DIR = str(REPO_ROOT / "genres")


# ---------------------------------------------------------------------------
# Minimal matrix fixture — used across tests
# ---------------------------------------------------------------------------

_MATRIX_1ST_ORDER = {
    "order": 1,
    "init_probs": {"I": 0.5, "vi": 0.3, "IV": 0.2},
    "transitions": {
        "I":  {"IV": 0.4, "V": 0.4, "vi": 0.2},
        "IV": {"V": 0.5, "I": 0.3, "vi": 0.2},
        "V":  {"I": 0.6, "vi": 0.3, "IV": 0.1},
        "vi": {"IV": 0.4, "ii": 0.3, "V": 0.3},
        "ii": {"V": 0.6, "IV": 0.4},
    },
}

_MATRIX_2ND_ORDER = {
    "order": 2,
    "init_probs": {"I": 0.4, "IV": 0.3, "vi": 0.2, "ii": 0.1},
    "transitions": {
        # 1st-order fallback keys (single chord)
        "I":  {"IV": 0.35, "V": 0.35, "vi": 0.2, "ii": 0.1},
        "IV": {"V": 0.4,  "I": 0.3,  "ii": 0.2, "vi": 0.1},
        "V":  {"I": 0.5,  "vi": 0.3, "IV": 0.1, "ii": 0.1},
        "vi": {"IV": 0.35, "ii": 0.25, "V": 0.25, "I": 0.15},
        "ii": {"V": 0.5,  "IV": 0.3, "I": 0.1,  "vi": 0.1},
        # 2nd-order keys (prev,curr)
        "I,IV":  {"V": 0.5,  "I": 0.3, "vi": 0.2},
        "IV,V":  {"I": 0.7,  "vi": 0.2, "IV": 0.1},
        "V,I":   {"IV": 0.4, "ii": 0.3, "vi": 0.3},
        "vi,IV": {"V": 0.5,  "I": 0.3, "ii": 0.2},
    },
}

_ALL_CHORDS = {"I", "IV", "V", "vi", "ii"}


# ---------------------------------------------------------------------------
# GenreSpec field
# ---------------------------------------------------------------------------

class TestGenreSpecChordTransitionMatrix:
    def test_field_exists_and_defaults_none(self):
        from musicgen.genre import GenreSpec
        spec = GenreSpec(name="test")
        assert hasattr(spec, "chord_transition_matrix")
        assert spec.chord_transition_matrix is None

    def test_field_accepts_matrix_dict(self):
        from musicgen.genre import GenreSpec
        spec = GenreSpec(name="test", chord_transition_matrix=_MATRIX_1ST_ORDER)
        assert spec.chord_transition_matrix["order"] == 1

    def test_merge_genres_sets_matrix_none(self):
        """Merged spec: chord_transition_matrix → None (too complex to merge matrices)."""
        from musicgen.genre import GenreSpec, merge_genres
        a = GenreSpec(name="a", chord_transition_matrix=_MATRIX_1ST_ORDER)
        b = GenreSpec(name="b", chord_transition_matrix=_MATRIX_2ND_ORDER)
        merged = merge_genres([a, b])
        assert merged.chord_transition_matrix is None

    def test_merge_single_spec_preserves_matrix(self):
        """Single-spec merge: matrix preserved (no-op merge)."""
        from musicgen.genre import GenreSpec, merge_genres
        spec = GenreSpec(name="a", chord_transition_matrix=_MATRIX_2ND_ORDER)
        result = merge_genres([spec])
        assert result.chord_transition_matrix == _MATRIX_2ND_ORDER


# ---------------------------------------------------------------------------
# load_genre loads chord_transitions.json
# ---------------------------------------------------------------------------

class TestLoadGenreChordTransitions:
    def test_load_genre_with_transitions_file(self, tmp_path):
        """load_genre reads chord_transitions.json and sets chord_transition_matrix."""
        from musicgen.genre import load_genre
        genre_dir = tmp_path / "test_genre"
        genre_dir.mkdir()
        (genre_dir / "spec.json").write_text(json.dumps({
            "name": "test_genre",
            "description": "test",
        }))
        (genre_dir / "chord_transitions.json").write_text(json.dumps(_MATRIX_2ND_ORDER))

        spec = load_genre("test_genre", str(tmp_path))
        assert spec.chord_transition_matrix is not None
        assert spec.chord_transition_matrix["order"] == 2
        assert "init_probs" in spec.chord_transition_matrix
        assert "transitions" in spec.chord_transition_matrix

    def test_load_genre_without_transitions_file(self, tmp_path):
        """load_genre without chord_transitions.json: matrix stays None."""
        from musicgen.genre import load_genre
        genre_dir = tmp_path / "plain"
        genre_dir.mkdir()
        (genre_dir / "spec.json").write_text(json.dumps({"name": "plain"}))

        spec = load_genre("plain", str(tmp_path))
        assert spec.chord_transition_matrix is None

    def test_load_real_genre_with_transitions(self):
        """If jazz has chord_transitions.json, it loads. If not, matrix is None."""
        from musicgen.genre import load_genre
        spec = load_genre("jazz", GENRES_DIR)
        transitions_path = os.path.join(GENRES_DIR, "jazz", "chord_transitions.json")
        if os.path.exists(transitions_path):
            assert spec.chord_transition_matrix is not None
        else:
            assert spec.chord_transition_matrix is None


# ---------------------------------------------------------------------------
# _sample_chord_markov internal helper
# ---------------------------------------------------------------------------

class TestSampleChordMarkov:
    def test_importable(self):
        from musicgen.generators.chord import _sample_chord_markov
        assert callable(_sample_chord_markov)

    def test_step0_returns_init_chord(self):
        """Empty history → draw from init_probs."""
        from musicgen.generators.chord import _sample_chord_markov
        rng = random.Random(42)
        chord = _sample_chord_markov([], _MATRIX_1ST_ORDER, rng)
        assert chord in _MATRIX_1ST_ORDER["init_probs"]

    def test_step1_1st_order(self):
        """History of 1 → draw from transitions[history[-1]]."""
        from musicgen.generators.chord import _sample_chord_markov
        rng = random.Random(7)
        result = _sample_chord_markov(["I"], _MATRIX_1ST_ORDER, rng)
        assert result in _MATRIX_1ST_ORDER["transitions"]["I"]

    def test_step2_2nd_order_key(self):
        """History of 2 with matching 2nd-order key → uses it."""
        from musicgen.generators.chord import _sample_chord_markov
        rng = random.Random(99)
        # "I,IV" is a key in _MATRIX_2ND_ORDER
        result = _sample_chord_markov(["I", "IV"], _MATRIX_2ND_ORDER, rng)
        assert result in _MATRIX_2ND_ORDER["transitions"]["I,IV"]

    def test_step2_fallback_to_1st_order(self):
        """2nd-order key not found → falls back to 1st-order key."""
        from musicgen.generators.chord import _sample_chord_markov
        rng = random.Random(5)
        # "I,ii" is NOT a 2nd-order key; fallback to transitions["ii"]
        result = _sample_chord_markov(["I", "ii"], _MATRIX_2ND_ORDER, rng)
        assert result in _MATRIX_2ND_ORDER["transitions"]["ii"]

    def test_missing_key_fallback_uniform(self):
        """If neither 2nd- nor 1st-order key found → uniform over init_probs keys."""
        from musicgen.generators.chord import _sample_chord_markov
        sparse_matrix = {
            "order": 2,
            "init_probs": {"I": 0.5, "V": 0.5},
            "transitions": {},  # deliberately empty
        }
        rng = random.Random(1)
        result = _sample_chord_markov(["X", "Y"], sparse_matrix, rng)
        assert result in sparse_matrix["init_probs"]

    def test_determinism(self):
        """Same seed → same sequence."""
        from musicgen.generators.chord import _sample_chord_markov
        history = ["I", "IV"]
        chord_a = _sample_chord_markov(history, _MATRIX_2ND_ORDER, random.Random(42))
        chord_b = _sample_chord_markov(history, _MATRIX_2ND_ORDER, random.Random(42))
        assert chord_a == chord_b

    def test_result_always_str(self):
        """Return type must be str."""
        from musicgen.generators.chord import _sample_chord_markov
        rng = random.Random(0)
        for history in [[], ["I"], ["I", "IV"], ["IV", "V", "I"]]:
            result = _sample_chord_markov(history, _MATRIX_2ND_ORDER, rng)
            assert isinstance(result, str)
            assert len(result) > 0


# ---------------------------------------------------------------------------
# generate_chord_progression — Markov path
# ---------------------------------------------------------------------------

def _make_genre_spec_with_matrix(matrix: Dict) -> Any:
    from musicgen.genre import GenreSpec
    return GenreSpec(name="markov_test", chord_transition_matrix=matrix)


class TestGenerateChordProgressionMarkov:
    def test_uses_markov_when_matrix_present(self, tmp_path, monkeypatch):
        """When genre_spec.chord_transition_matrix is set, skip pattern file."""
        from musicgen.generators.chord import generate_chord_progression
        monkeypatch.chdir(tmp_path)
        spec = _make_genre_spec_with_matrix(_MATRIX_2ND_ORDER)
        pattern, mid = generate_chord_progression(
            key="C", tempo=120, time_signature="4/4", measures=4,
            name="song42-verse", part="verse",
            pattern_file=CHORD_PATTERNS,
            rng=random.Random(42),
            genre_spec=spec,
        )
        # Pattern should be a list of chords from the Markov chain
        assert isinstance(pattern, list)
        assert len(pattern) > 0
        all_chords = set(_MATRIX_2ND_ORDER["init_probs"]) | set(
            k.split(",")[-1]
            for k in _MATRIX_2ND_ORDER["transitions"]
        ) | set(_MATRIX_2ND_ORDER["transitions"].keys())
        for chord in pattern:
            # chord must come from Markov vocabulary
            assert chord in _MATRIX_2ND_ORDER["init_probs"] or any(
                chord in v for v in _MATRIX_2ND_ORDER["transitions"].values()
            ), f"Chord {chord!r} not in Markov vocabulary"

    def test_markov_midi_written(self, tmp_path, monkeypatch):
        """Markov path writes a valid MIDI file."""
        from musicgen.generators.chord import generate_chord_progression
        monkeypatch.chdir(tmp_path)
        spec = _make_genre_spec_with_matrix(_MATRIX_1ST_ORDER)
        _, mid = generate_chord_progression(
            key="G", tempo=100, time_signature="4/4", measures=4,
            name="song1-chorus", part="chorus",
            pattern_file=CHORD_PATTERNS,
            rng=random.Random(1),
            genre_spec=spec,
        )
        assert os.path.exists(mid)
        assert os.path.getsize(mid) > 0

    def test_markov_determinism(self, tmp_path, monkeypatch):
        """Same seed → same Markov chord sequence + identical MIDI bytes."""
        from musicgen.generators.chord import generate_chord_progression
        monkeypatch.chdir(tmp_path)
        spec = _make_genre_spec_with_matrix(_MATRIX_2ND_ORDER)
        args = dict(
            key="C", tempo=120, time_signature="4/4", measures=4,
            name="song99-verse", part="verse",
            pattern_file=CHORD_PATTERNS,
            genre_spec=spec,
        )
        p1, mid1 = generate_chord_progression(**args, rng=random.Random(17))
        b1 = Path(mid1).read_bytes()
        os.remove(mid1)
        p2, mid2 = generate_chord_progression(**args, rng=random.Random(17))
        assert p1 == p2
        assert b1 == Path(mid2).read_bytes()

    def test_markov_measures_count(self, tmp_path, monkeypatch):
        """Markov chord list length == measures (one chord per measure for 4/4)."""
        from musicgen.generators.chord import generate_chord_progression
        monkeypatch.chdir(tmp_path)
        spec = _make_genre_spec_with_matrix(_MATRIX_1ST_ORDER)
        for measures in [2, 4, 8]:
            pattern, mid = generate_chord_progression(
                key="C", tempo=120, time_signature="4/4", measures=measures,
                name=f"song{measures}-verse", part="verse",
                pattern_file=CHORD_PATTERNS,
                rng=random.Random(measures),
                genre_spec=spec,
            )
            assert len(pattern) == measures, (
                f"Expected {measures} chords, got {len(pattern)}"
            )
            if os.path.exists(mid):
                os.remove(mid)

    def test_no_matrix_uses_pattern_file(self, tmp_path, monkeypatch):
        """genre_spec without chord_transition_matrix → pattern file path (backward compat)."""
        from musicgen.generators.chord import generate_chord_progression
        from musicgen.genre import GenreSpec
        monkeypatch.chdir(tmp_path)
        spec = GenreSpec(name="no_matrix")  # chord_transition_matrix=None
        pattern, mid = generate_chord_progression(
            key="C", tempo=120, time_signature="4/4", measures=4,
            name="song0-verse", part="verse",
            pattern_file=CHORD_PATTERNS,
            rng=random.Random(0),
            genre_spec=spec,
        )
        assert isinstance(pattern, list)
        assert len(pattern) > 0

    def test_none_genre_spec_unchanged(self, tmp_path, monkeypatch):
        """genre_spec=None → exact pre-v0.3 behavior (backward compat)."""
        from musicgen.generators.chord import generate_chord_progression
        monkeypatch.chdir(tmp_path)
        p1, mid1 = generate_chord_progression(
            key="C", tempo=120, time_signature="4/4", measures=4,
            name="song0-verse", part="verse",
            pattern_file=CHORD_PATTERNS,
            rng=random.Random(0),
            genre_spec=None,
        )
        assert isinstance(p1, list)
        assert len(p1) > 0


# ---------------------------------------------------------------------------
# Chord sequence property tests
# ---------------------------------------------------------------------------

class TestMarkovChordProperties:
    def test_1st_order_sequence_stays_in_vocabulary(self, tmp_path, monkeypatch):
        """Every chord in a 1st-order run is reachable from init_probs + transitions."""
        from musicgen.generators.chord import generate_chord_progression
        monkeypatch.chdir(tmp_path)
        spec = _make_genre_spec_with_matrix(_MATRIX_1ST_ORDER)
        # Vocabulary = init_probs keys ∪ all values in transitions
        vocab: set = set(_MATRIX_1ST_ORDER["init_probs"])
        for v in _MATRIX_1ST_ORDER["transitions"].values():
            vocab.update(v)

        for seed in range(10):
            pattern, mid = generate_chord_progression(
                key="C", tempo=120, time_signature="4/4", measures=8,
                name=f"song{seed}-verse", part="verse",
                pattern_file=CHORD_PATTERNS,
                rng=random.Random(seed),
                genre_spec=spec,
            )
            for chord in pattern:
                assert chord in vocab, f"Chord {chord!r} not in vocabulary (seed={seed})"
            if os.path.exists(mid):
                os.remove(mid)

    def test_2nd_order_different_seeds_produce_variety(self, tmp_path, monkeypatch):
        """Different seeds produce different sequences (with high probability)."""
        from musicgen.generators.chord import generate_chord_progression
        monkeypatch.chdir(tmp_path)
        spec = _make_genre_spec_with_matrix(_MATRIX_2ND_ORDER)
        sequences = set()
        for seed in range(20):
            pattern, mid = generate_chord_progression(
                key="C", tempo=120, time_signature="4/4", measures=8,
                name=f"song{seed}-verse", part="verse",
                pattern_file=CHORD_PATTERNS,
                rng=random.Random(seed),
                genre_spec=spec,
            )
            sequences.add(tuple(pattern))
            if os.path.exists(mid):
                os.remove(mid)
        assert len(sequences) > 1, "All seeds produced the same sequence — insufficient variety"
