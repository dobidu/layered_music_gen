"""LSTM models for chord and melody sequence generation (v0.5).

Both models share the same architecture: embedding + 2-layer LSTM + linear head.
Genre is injected as a one-hot vector concatenated to the embedding at every step.

NeuralSampler bundles a trained model with its token/genre vocabularies so the
trainer and inference code share a single serialisation unit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn


class _SequenceLSTM(nn.Module):
    """Shared backbone for chord and melody LSTMs."""

    def __init__(
        self,
        vocab_size: int,
        genre_count: int,
        embed_dim: int,
        hidden: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        lstm_input = embed_dim + genre_count  # token embedding ‖ genre one-hot
        self.lstm = nn.LSTM(
            lstm_input, hidden, num_layers=layers,
            batch_first=True, dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden, vocab_size)
        self._vocab_size = vocab_size
        self._genre_count = genre_count
        self._hidden = hidden

    def forward(
        self,
        token_ids: torch.Tensor,   # (batch, seq_len)
        genre_ids: torch.Tensor,   # (batch,) or (batch, seq_len)
    ) -> torch.Tensor:             # (batch, seq_len, vocab_size)
        emb = self.embed(token_ids)                      # (B, T, embed_dim)

        if genre_ids.dim() == 1:
            # Broadcast genre across sequence
            genre_oh = torch.zeros(
                genre_ids.size(0), self._genre_count,
                device=token_ids.device,
            )
            genre_oh.scatter_(1, genre_ids.unsqueeze(1), 1.0)
            genre_oh = genre_oh.unsqueeze(1).expand(-1, emb.size(1), -1)
        else:
            genre_oh = torch.zeros(
                genre_ids.size(0), genre_ids.size(1), self._genre_count,
                device=token_ids.device,
            )
            genre_oh.scatter_(2, genre_ids.unsqueeze(2), 1.0)

        x = torch.cat([emb, genre_oh], dim=-1)          # (B, T, embed+genre)
        out, _ = self.lstm(x)                            # (B, T, hidden)
        return self.head(out)                            # (B, T, vocab_size)


def ChordLSTM(vocab_size: int, genre_count: int) -> _SequenceLSTM:
    """2-layer LSTM for chord progressions (~35K parameters)."""
    return _SequenceLSTM(
        vocab_size=vocab_size,
        genre_count=genre_count,
        embed_dim=16,
        hidden=64,
        layers=2,
        dropout=0.2,
    )


def MelodyLSTM(vocab_size: int, genre_count: int) -> _SequenceLSTM:
    """2-layer LSTM for melody scale-degree sequences (~10K parameters)."""
    return _SequenceLSTM(
        vocab_size=vocab_size,
        genre_count=genre_count,
        embed_dim=8,
        hidden=32,
        layers=2,
        dropout=0.2,
    )


@dataclass
class NeuralSampler:
    """Trained model + vocabularies, used by trainer and inference code."""
    model: _SequenceLSTM
    token_to_idx: Dict[str, int]
    idx_to_token: Dict[int, str]
    genre_to_idx: Dict[str, int]
    layer: str                          # "chord" | "melody"
    context_len: int = 4               # how many previous tokens to feed

    # Derived convenience
    unknown_genre_idx: int = field(init=False)

    def __post_init__(self) -> None:
        self.unknown_genre_idx = self.genre_to_idx.get("__unknown__", 0)

    def genre_idx(self, genre: Optional[List[str]]) -> int:
        if not genre:
            return self.unknown_genre_idx
        for g in genre:
            if g in self.genre_to_idx:
                return self.genre_to_idx[g]
        return self.unknown_genre_idx
