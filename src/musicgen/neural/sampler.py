"""Neural sampling functions — drop-in replacements for Markov samplers (v0.5).

Both functions mirror the call signature of the Markov counterparts so the
generator code can switch backends with a single ``if`` branch.

Determinism: the model's forward pass is pure (fixed weights → fixed logits).
Sampling is done via ``rng.choices(tokens, weights=softmax(logits))``, which
uses the caller-supplied seeded ``random.Random`` — identical to the Markov
path.  The determinism contract is therefore fully preserved.
"""
from __future__ import annotations

import logging
import random
from typing import List, Optional

logger = logging.getLogger(__name__)


def sample_chord_neural(
    history: List[str],
    genre: Optional[List[str]],
    sampler,                # NeuralSampler (avoid circular import at type level)
    rng: random.Random,
) -> str:
    """Sample the next chord symbol using the trained LSTM.

    Args:
        history: Previous chord symbols (Roman numerals), most-recent last.
        genre: Genre list from the current generation config (may be None).
        sampler: :class:`~musicgen.neural.model.NeuralSampler` instance.
        rng: Seeded ``random.Random`` from the generator RNG pool.

    Returns:
        Roman numeral string (e.g. ``"I"``, ``"vi"``, ``"IV"``).
    """
    return _sample_neural(history, genre, sampler, rng)


def sample_melody_neural(
    history: List[str],
    genre: Optional[List[str]],
    sampler,
    rng: random.Random,
) -> str:
    """Sample the next scale degree using the trained LSTM.

    Args:
        history: Previous scale-degree strings (``"1"``–``"7"``), most-recent last.
        genre: Genre list from the current generation config.
        sampler: :class:`~musicgen.neural.model.NeuralSampler` instance.
        rng: Seeded ``random.Random`` from the generator RNG pool.

    Returns:
        Scale-degree string ``"1"``–``"7"``.
    """
    return _sample_neural(history, genre, sampler, rng)


def _sample_neural(
    history: List[str],
    genre: Optional[List[str]],
    sampler,
    rng: random.Random,
) -> str:
    import torch
    import torch.nn.functional as F

    t2i = sampler.token_to_idx
    i2t = sampler.idx_to_token
    ctx_len = sampler.context_len

    # Build context window — left-pad with <pad>=0 if history is short
    pad_id = t2i.get("<pad>", 0)
    ctx_tokens = [t2i.get(h, pad_id) for h in history[-ctx_len:]]
    ctx_tokens = [pad_id] * (ctx_len - len(ctx_tokens)) + ctx_tokens

    ctx_t = torch.tensor([ctx_tokens], dtype=torch.long)               # (1, T)
    genre_idx = sampler.genre_idx(genre)
    genre_t = torch.tensor([genre_idx], dtype=torch.long)              # (1,)

    with torch.no_grad():
        logits = sampler.model(ctx_t, genre_t)                         # (1, T, vocab)
        last_logits = logits[0, -1, :]                                 # (vocab,)
        probs = F.softmax(last_logits, dim=-1).tolist()

    tokens = list(i2t.keys())
    # Filter out <pad> (idx 0)
    eligible = [(idx, p) for idx, p in enumerate(probs) if i2t.get(idx, "<pad>") != "<pad>"]
    if not eligible:
        # Fallback: uniform over all tokens except <pad>
        eligible = [(idx, 1.0) for idx in i2t if i2t[idx] != "<pad>"]

    idxs = [e[0] for e in eligible]
    weights = [e[1] for e in eligible]
    chosen_idx = rng.choices(idxs, weights=weights, k=1)[0]
    return i2t[chosen_idx]
