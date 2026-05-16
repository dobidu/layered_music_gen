"""Tests for the neural chord/melody LSTM backend (v0.5 Phase 2).

All tests run on CPU with tiny synthetic data. No GPU required, no FluidSynth.
Marked skip when torch is not installed (requires musicgen[neural]).
"""
from __future__ import annotations

import json
import random
import tempfile
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="torch not installed — run: pip install 'musicgen[neural]'")

from musicgen.neural.model import ChordLSTM, MelodyLSTM, NeuralSampler, _SequenceLSTM
from musicgen.neural.trainer import train, save_model, load_model
from musicgen.neural.sampler import sample_chord_neural, sample_melody_neural


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_CHORD_TOKENS = ["I", "ii", "IV", "V", "vi"]
_MELODY_TOKENS = ["1", "2", "3", "4", "5", "6", "7"]
_GENRES = ["pop", "jazz"]

_TINY_SEQUENCES = {
    "metadata": {"n_samples": 10, "genres": _GENRES, "musicgen_version": "0.5.0", "n_skipped": 0},
    "chord": [
        {"sample_index": i, "genre": [_GENRES[i % 2]], "key": "C",
         "full_sequence": ["I", "V", "vi", "IV", "I", "IV", "V", "I"] * 4}
        for i in range(20)
    ],
    "melody": [
        {"sample_index": i, "genre": [_GENRES[i % 2]], "key": "C",
         "full_sequence": ["1", "2", "3", "4", "5", "4", "3", "2"] * 4}
        for i in range(20)
    ],
}


@pytest.fixture()
def sequences_file(tmp_path):
    p = tmp_path / "sequences.json"
    p.write_text(json.dumps(_TINY_SEQUENCES))
    return str(p)


# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------


class TestModelArchitecture:
    def test_chord_lstm_forward_shape(self):
        model = ChordLSTM(vocab_size=10, genre_count=3)
        ctx = torch.zeros(2, 4, dtype=torch.long)
        gid = torch.zeros(2, dtype=torch.long)
        out = model(ctx, gid)
        assert out.shape == (2, 4, 10)

    def test_melody_lstm_forward_shape(self):
        model = MelodyLSTM(vocab_size=8, genre_count=2)
        ctx = torch.zeros(3, 4, dtype=torch.long)
        gid = torch.zeros(3, dtype=torch.long)
        out = model(ctx, gid)
        assert out.shape == (3, 4, 8)

    def test_chord_param_count(self):
        model = ChordLSTM(vocab_size=20, genre_count=5)
        n = sum(p.numel() for p in model.parameters())
        assert n < 100_000, f"ChordLSTM too large: {n} params"

    def test_melody_param_count(self):
        model = MelodyLSTM(vocab_size=8, genre_count=5)
        n = sum(p.numel() for p in model.parameters())
        assert n < 30_000, f"MelodyLSTM too large: {n} params"

    def test_chord_lstm_is_sequence_lstm(self):
        assert isinstance(ChordLSTM(10, 2), _SequenceLSTM)

    def test_melody_lstm_is_sequence_lstm(self):
        assert isinstance(MelodyLSTM(8, 2), _SequenceLSTM)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class TestTrainer:
    def test_train_chord_returns_sampler(self, sequences_file):
        sampler = train(sequences_file, layer="chord", epochs=5, seed=0)
        assert isinstance(sampler, NeuralSampler)
        assert sampler.layer == "chord"

    def test_train_melody_returns_sampler(self, sequences_file):
        sampler = train(sequences_file, layer="melody", epochs=5, seed=0)
        assert isinstance(sampler, NeuralSampler)
        assert sampler.layer == "melody"

    def test_vocab_contains_known_tokens(self, sequences_file):
        sampler = train(sequences_file, layer="chord", epochs=2, seed=0)
        assert "I" in sampler.token_to_idx
        assert "V" in sampler.token_to_idx

    def test_genre_vocab_built(self, sequences_file):
        sampler = train(sequences_file, layer="chord", epochs=2, seed=0)
        assert "pop" in sampler.genre_to_idx
        assert "jazz" in sampler.genre_to_idx

    def test_invalid_layer_raises(self, sequences_file):
        with pytest.raises(ValueError):
            train(sequences_file, layer="drums", epochs=2)

    def test_genre_filter(self, sequences_file):
        sampler = train(sequences_file, layer="chord", genres=["pop"], epochs=2, seed=0)
        assert sampler.genre_to_idx.get("pop") is not None

    def test_save_and_load_roundtrip(self, sequences_file, tmp_path):
        sampler = train(sequences_file, layer="chord", epochs=3, seed=0)
        pt_path = str(tmp_path / "chord.pt")
        save_model(sampler, pt_path)
        assert Path(pt_path).exists()
        assert Path(pt_path.replace(".pt", "_meta.json")).exists()

        loaded = load_model(pt_path)
        assert loaded is not None
        assert loaded.layer == "chord"
        assert set(loaded.token_to_idx) == set(sampler.token_to_idx)

    def test_load_missing_returns_none(self, tmp_path):
        result = load_model(str(tmp_path / "nonexistent.pt"))
        assert result is None

    def test_deterministic_training(self, sequences_file, tmp_path):
        s1 = train(sequences_file, layer="chord", epochs=5, seed=42)
        s2 = train(sequences_file, layer="chord", epochs=5, seed=42)
        # Same seed → same final weights
        for (k, v1), (_, v2) in zip(
            s1.model.state_dict().items(), s2.model.state_dict().items()
        ):
            assert torch.allclose(v1, v2), f"Weight mismatch at {k}"


# ---------------------------------------------------------------------------
# Sampler (inference)
# ---------------------------------------------------------------------------


class TestNeuralSampler:
    @pytest.fixture()
    def chord_sampler(self, sequences_file):
        return train(sequences_file, layer="chord", epochs=5, seed=0)

    @pytest.fixture()
    def melody_sampler(self, sequences_file):
        return train(sequences_file, layer="melody", epochs=5, seed=0)

    def test_sample_chord_returns_known_token(self, chord_sampler):
        rng = random.Random(1)
        result = sample_chord_neural(["I", "V"], ["pop"], chord_sampler, rng)
        assert result in chord_sampler.token_to_idx

    def test_sample_melody_returns_degree(self, melody_sampler):
        rng = random.Random(1)
        result = sample_melody_neural(["1", "2"], ["jazz"], melody_sampler, rng)
        assert result in ("1", "2", "3", "4", "5", "6", "7")

    def test_sample_chord_no_genre(self, chord_sampler):
        rng = random.Random(2)
        result = sample_chord_neural(["I"], None, chord_sampler, rng)
        assert result in chord_sampler.token_to_idx

    def test_sample_chord_empty_history(self, chord_sampler):
        rng = random.Random(3)
        result = sample_chord_neural([], ["pop"], chord_sampler, rng)
        assert result in chord_sampler.token_to_idx

    def test_sampling_deterministic_given_rng(self, chord_sampler):
        results = []
        for _ in range(3):
            rng = random.Random(99)
            r = sample_chord_neural(["I", "V", "vi"], ["pop"], chord_sampler, rng)
            results.append(r)
        assert len(set(results)) == 1, "Same RNG seed must yield same token"

    def test_sampling_varies_with_rng(self, chord_sampler):
        """Different RNG seeds should (almost always) produce at least 2 distinct tokens."""
        tokens = set()
        for seed in range(50):
            rng = random.Random(seed)
            tokens.add(sample_chord_neural(["I", "V"], ["pop"], chord_sampler, rng))
        assert len(tokens) >= 2, "Neural sampler never varied — weights degenerate?"
