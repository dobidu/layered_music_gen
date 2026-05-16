"""Integration tests for the neural chord/melody generator backends (v0.5 Phase 3).

Strategy: train tiny in-memory models, monkeypatch the module-level caches in
chord.py and melody.py so no .pt files are needed on disk, then call
generate_chord_progression / generate_melody and assert:
  - output types/lengths are correct
  - chord symbols come from the trained vocabulary
  - melody notes are integers in valid MIDI range
  - determinism is preserved (same RNG seed → identical output)

Marked skip when torch is not installed.
"""
from __future__ import annotations

import json
import os
import random
import tempfile
from pathlib import Path
from typing import List
from unittest.mock import patch

import pytest

torch = pytest.importorskip(
    "torch", reason="torch not installed — run: pip install 'musicgen[neural]'"
)

from musicgen.neural.trainer import train
from musicgen.generators.chord import generate_chord_progression
from musicgen.generators.melody import generate_melody


# ---------------------------------------------------------------------------
# Shared fixture data
# ---------------------------------------------------------------------------

_GENRES = ["pop", "jazz"]
_CHORD_TOKENS = ["I", "ii", "IV", "V", "vi"]
_MELODY_TOKENS = ["1", "2", "3", "4", "5", "6", "7"]

_TINY_SEQUENCES = {
    "metadata": {"n_samples": 10, "genres": _GENRES, "musicgen_version": "0.5.0", "n_skipped": 0},
    "chord": [
        {
            "sample_index": i, "genre": [_GENRES[i % 2]], "key": "C",
            "full_sequence": ["I", "V", "vi", "IV", "I", "IV", "V", "I"] * 4,
        }
        for i in range(20)
    ],
    "melody": [
        {
            "sample_index": i, "genre": [_GENRES[i % 2]], "key": "C",
            "full_sequence": ["1", "2", "3", "4", "5", "4", "3", "2"] * 4,
        }
        for i in range(20)
    ],
}

_PATTERN_FILE_CONTENT = "verse:I,IV,V,vi\nchorus:I,V,vi,IV\n"


@pytest.fixture(scope="module")
def sequences_file(tmp_path_factory):
    p = tmp_path_factory.mktemp("data") / "sequences.json"
    p.write_text(json.dumps(_TINY_SEQUENCES))
    return str(p)


@pytest.fixture(scope="module")
def chord_sampler(sequences_file):
    return train(sequences_file, layer="chord", epochs=5, seed=0)


@pytest.fixture(scope="module")
def melody_sampler(sequences_file):
    return train(sequences_file, layer="melody", epochs=5, seed=0)


@pytest.fixture()
def pattern_file(tmp_path):
    f = tmp_path / "chord_patterns.txt"
    f.write_text(_PATTERN_FILE_CONTENT)
    return str(f)


# ---------------------------------------------------------------------------
# Fake Config helper
# ---------------------------------------------------------------------------

class _FakeCfg:
    """Minimal Config stand-in for tests that don't need real Config.load()."""

    def __init__(self, chord_backend="neural", melody_backend="neural", genre=None,
                 models_dir="/nonexistent"):
        self.chord_backend = chord_backend
        self.melody_backend = melody_backend
        self.genre = genre
        self.models_dir = models_dir


# ---------------------------------------------------------------------------
# Chord integration
# ---------------------------------------------------------------------------

class TestChordNeuralBackend:
    def test_returns_list_of_known_tokens(self, chord_sampler, pattern_file, tmp_path):
        cfg = _FakeCfg(chord_backend="neural", genre=["pop"])
        with patch("musicgen.generators.chord._chord_model_cache", {"/nonexistent/chord_pop.pt": None, "/nonexistent/chord.pt": chord_sampler}):
            with patch("musicgen.generators.chord._HAS_NEURAL_CHORD", True):
                result, filename = generate_chord_progression(
                    "C", 120, "4/4", 4,
                    str(tmp_path / "test-song"), "verse",
                    pattern_file, random.Random(1),
                    genre_spec=None, cfg=cfg,
                )
        assert isinstance(result, list)
        assert len(result) == 4
        for sym in result:
            assert sym in chord_sampler.token_to_idx, f"{sym!r} not in vocab"

    def test_deterministic_given_seed(self, chord_sampler, pattern_file, tmp_path):
        cfg = _FakeCfg(chord_backend="neural", genre=["pop"])
        cache = {"/nonexistent/chord_pop.pt": None, "/nonexistent/chord.pt": chord_sampler}
        results = []
        for run in range(2):
            out_dir = tmp_path / f"run{run}"
            out_dir.mkdir()
            with patch("musicgen.generators.chord._chord_model_cache", cache):
                with patch("musicgen.generators.chord._HAS_NEURAL_CHORD", True):
                    chords, _ = generate_chord_progression(
                        "C", 120, "4/4", 4,
                        str(out_dir / "song"), "verse",
                        pattern_file, random.Random(42),
                        genre_spec=None, cfg=cfg,
                    )
            results.append(chords)
        assert results[0] == results[1], "Neural chord output not deterministic"

    def test_falls_back_when_model_missing(self, pattern_file, tmp_path):
        cfg = _FakeCfg(chord_backend="neural", genre=None, models_dir="/nonexistent")
        # Cache miss → sampler=None → fallback to pattern file
        with patch("musicgen.generators.chord._chord_model_cache", {}):
            with patch("musicgen.generators.chord._HAS_NEURAL_CHORD", True):
                result, _ = generate_chord_progression(
                    "C", 120, "4/4", 4,
                    str(tmp_path / "song"), "verse",
                    pattern_file, random.Random(7),
                    genre_spec=None, cfg=cfg,
                )
        # Pattern-file fallback returns a valid list
        assert isinstance(result, list)
        assert len(result) > 0

    def test_markov_backend_unchanged(self, pattern_file, tmp_path):
        cfg = _FakeCfg(chord_backend="markov")
        result, _ = generate_chord_progression(
            "C", 120, "4/4", 4,
            str(tmp_path / "song"), "verse",
            pattern_file, random.Random(1),
            genre_spec=None, cfg=cfg,
        )
        assert isinstance(result, list)
        assert len(result) > 0

    def test_cfg_none_uses_pattern_file(self, pattern_file, tmp_path):
        result, _ = generate_chord_progression(
            "C", 120, "4/4", 4,
            str(tmp_path / "song"), "verse",
            pattern_file, random.Random(1),
            genre_spec=None, cfg=None,
        )
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Melody integration
# ---------------------------------------------------------------------------

class TestMelodyNeuralBackend:
    def test_returns_list_of_midi_ints(self, melody_sampler, tmp_path):
        cfg = _FakeCfg(melody_backend="neural", genre=["pop"])
        cache = {"/nonexistent/melody_pop.pt": None, "/nonexistent/melody.pt": melody_sampler}
        with patch("musicgen.generators.melody._melody_model_cache", cache):
            with patch("musicgen.generators.melody._HAS_NEURAL_MELODY", True):
                melody, filename = generate_melody(
                    "C", 120, "4/4", 2,
                    str(tmp_path / "song"), "verse",
                    ["I", "IV", "V", "I"], random.Random(1),
                    genre_spec=None, cfg=cfg,
                )
        assert isinstance(melody, list)
        assert len(melody) > 0
        for note in melody:
            assert isinstance(note, int)
            assert 0 <= note <= 127

    def test_deterministic_given_seed(self, melody_sampler, tmp_path):
        cfg = _FakeCfg(melody_backend="neural", genre=["pop"])
        cache = {"/nonexistent/melody_pop.pt": None, "/nonexistent/melody.pt": melody_sampler}
        results = []
        for run in range(2):
            out_dir = tmp_path / f"run{run}"
            out_dir.mkdir()
            with patch("musicgen.generators.melody._melody_model_cache", cache):
                with patch("musicgen.generators.melody._HAS_NEURAL_MELODY", True):
                    melody, _ = generate_melody(
                        "C", 120, "4/4", 2,
                        str(out_dir / "song"), "verse",
                        ["I", "IV", "V", "I"], random.Random(99),
                        genre_spec=None, cfg=cfg,
                    )
            results.append(melody)
        assert results[0] == results[1], "Neural melody output not deterministic"

    def test_falls_back_when_model_missing(self, tmp_path):
        cfg = _FakeCfg(melody_backend="neural", genre=None, models_dir="/nonexistent")
        with patch("musicgen.generators.melody._melody_model_cache", {}):
            with patch("musicgen.generators.melody._HAS_NEURAL_MELODY", True):
                melody, _ = generate_melody(
                    "C", 120, "4/4", 2,
                    str(tmp_path / "song"), "verse",
                    ["I", "IV", "V", "I"], random.Random(7),
                    genre_spec=None, cfg=cfg,
                )
        assert isinstance(melody, list)
        assert len(melody) > 0

    def test_markov_backend_unchanged(self, tmp_path):
        cfg = _FakeCfg(melody_backend="markov")
        melody, _ = generate_melody(
            "C", 120, "4/4", 2,
            str(tmp_path / "song"), "verse",
            ["I", "IV", "V", "I"], random.Random(1),
            genre_spec=None, cfg=cfg,
        )
        assert isinstance(melody, list)

    def test_cfg_none_uses_chord_pitch(self, tmp_path):
        melody, _ = generate_melody(
            "C", 120, "4/4", 2,
            str(tmp_path / "song"), "verse",
            ["I", "IV", "V", "I"], random.Random(1),
            genre_spec=None, cfg=None,
        )
        assert isinstance(melody, list)


# ---------------------------------------------------------------------------
# Config field validation
# ---------------------------------------------------------------------------

class TestConfigNeuralFields:
    def test_default_backends_are_markov(self):
        from config import Config
        cfg = Config()
        assert cfg.chord_backend == "markov"
        assert cfg.melody_backend == "markov"

    def test_models_dir_defaults_exist(self):
        from config import Config
        cfg = Config()
        assert isinstance(cfg.models_dir, str)
        assert len(cfg.models_dir) > 0

    def test_invalid_chord_backend_raises(self):
        from config import Config
        import dataclasses
        with pytest.raises(ValueError, match="chord_backend"):
            Config(chord_backend="invalid")

    def test_invalid_melody_backend_raises(self):
        from config import Config
        with pytest.raises(ValueError, match="melody_backend"):
            Config(melody_backend="transformer")

    def test_neural_backend_accepted(self):
        from config import Config
        cfg = Config(chord_backend="neural", melody_backend="neural")
        assert cfg.chord_backend == "neural"
        assert cfg.melody_backend == "neural"
