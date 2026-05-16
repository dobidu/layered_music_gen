"""Training, saving, and loading for neural chord/melody models (v0.5).

Entry point::

    from musicgen.neural.trainer import train, save_model, load_model

Training is intentionally CPU-friendly — models are tiny (10K–35K parameters)
and a 1000-sample corpus trains in well under a minute on a laptop CPU.
"""
from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from musicgen.neural.model import ChordLSTM, MelodyLSTM, NeuralSampler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

_PAD = "<pad>"
_UNK_GENRE = "__unknown__"


class _SequenceDataset(Dataset):
    """Sliding-window dataset over token sequences."""

    def __init__(
        self,
        sequences: List[List[int]],
        genre_ids: List[int],
        context_len: int,
    ) -> None:
        self.windows: List[Tuple[List[int], int, int]] = []  # (ctx_tokens, genre_id, target)
        pad_id = 0  # token 0 is <pad>
        for seq, gid in zip(sequences, genre_ids):
            if len(seq) < 2:
                continue
            for i in range(len(seq) - 1):
                ctx_start = max(0, i - context_len + 1)
                ctx = seq[ctx_start: i + 1]
                # Left-pad to context_len
                padded = [pad_id] * (context_len - len(ctx)) + ctx
                self.windows.append((padded, gid, seq[i + 1]))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        ctx, gid, target = self.windows[idx]
        return (
            torch.tensor(ctx, dtype=torch.long),
            torch.tensor(gid, dtype=torch.long),
            torch.tensor(target, dtype=torch.long),
        )


# ---------------------------------------------------------------------------
# Vocabulary builders
# ---------------------------------------------------------------------------

def _build_token_vocab(sequences: List[List[str]]) -> Tuple[Dict[str, int], Dict[int, str]]:
    tokens = sorted({tok for seq in sequences for tok in seq})
    t2i = {_PAD: 0}
    for tok in tokens:
        t2i[tok] = len(t2i)
    i2t = {v: k for k, v in t2i.items()}
    return t2i, i2t


def _build_genre_vocab(genre_lists: List[Optional[List[str]]]) -> Dict[str, int]:
    genres = sorted({g for gl in genre_lists if gl for g in gl})
    g2i = {_UNK_GENRE: 0}
    for g in genres:
        g2i[g] = len(g2i)
    return g2i


# ---------------------------------------------------------------------------
# Perplexity
# ---------------------------------------------------------------------------

def _perplexity(model: nn.Module, loader: DataLoader, criterion: nn.CrossEntropyLoss) -> float:
    model.eval()
    total_loss = 0.0
    total_n = 0
    with torch.no_grad():
        for ctx, gid, target in loader:
            logits = model(ctx, gid)            # (B, T, vocab)
            last_logit = logits[:, -1, :]       # (B, vocab) — predict next token
            loss = criterion(last_logit, target)
            total_loss += loss.item() * ctx.size(0)
            total_n += ctx.size(0)
    return math.exp(total_loss / total_n) if total_n > 0 else float("inf")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def train(
    sequences_path: str,
    layer: str,
    genres: Optional[List[str]] = None,
    epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 64,
    context_len: int = 4,
    seed: int = 42,
    patience: int = 20,
) -> NeuralSampler:
    """Train a chord or melody LSTM on ``sequences_path``.

    Args:
        sequences_path: Path to ``sequences.json`` produced by
            :func:`musicgen.corpus_extractor.extract_sequences`.
        layer: ``"chord"`` or ``"melody"``.
        genres: Filter training data to these genres only (None = all).
        epochs: Maximum training epochs.
        lr: Adam learning rate.
        batch_size: Mini-batch size.
        context_len: Number of previous tokens fed to the model.
        seed: Reproducibility seed for torch (controls weight init and DataLoader).
        patience: Early-stopping patience in epochs.

    Returns:
        :class:`NeuralSampler` with trained model and vocabularies.
    """
    if layer not in ("chord", "melody"):
        raise ValueError(f"layer must be 'chord' or 'melody', got {layer!r}")

    torch.manual_seed(seed)

    with open(sequences_path) as f:
        data = json.load(f)

    entries = data.get(layer, [])
    if not entries:
        raise ValueError(f"No {layer!r} sequences in {sequences_path}")

    # Genre filter
    if genres:
        genre_set = set(genres)
        entries = [e for e in entries if e.get("genre") and set(e["genre"]) & genre_set]
        if not entries:
            raise ValueError(f"No entries for genres {genres} in {layer!r} sequences")

    raw_sequences = [e["full_sequence"] for e in entries]
    genre_lists = [e.get("genre") for e in entries]

    token_to_idx, idx_to_token = _build_token_vocab(raw_sequences)
    genre_to_idx = _build_genre_vocab(genre_lists)

    # Encode
    enc_sequences = [[token_to_idx.get(t, 0) for t in seq] for seq in raw_sequences]
    enc_genres = [
        (genre_to_idx.get(gl[0], 0) if gl else 0)
        for gl in genre_lists
    ]

    # Train/val split (90/10, deterministic)
    n_val = max(1, len(enc_sequences) // 10)
    val_seqs, val_genres = enc_sequences[:n_val], enc_genres[:n_val]
    trn_seqs, trn_genres = enc_sequences[n_val:], enc_genres[n_val:]

    if not trn_seqs:
        # Too few samples — use everything for training
        trn_seqs, trn_genres = enc_sequences, enc_genres
        val_seqs, val_genres = enc_sequences, enc_genres

    trn_ds = _SequenceDataset(trn_seqs, trn_genres, context_len)
    val_ds = _SequenceDataset(val_seqs, val_genres, context_len)

    if len(trn_ds) == 0:
        raise ValueError(f"No training windows extracted from {layer!r} sequences (sequences too short?)")

    trn_loader = DataLoader(trn_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    vocab_size = len(token_to_idx)
    genre_count = len(genre_to_idx)

    model = (ChordLSTM if layer == "chord" else MelodyLSTM)(vocab_size, genre_count)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 0 = <pad>

    best_val_ppl = float("inf")
    best_state: Optional[dict] = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for ctx, gid, target in trn_loader:
            optimizer.zero_grad()
            logits = model(ctx, gid)
            loss = criterion(logits[:, -1, :], target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        val_ppl = _perplexity(model, val_loader, criterion)

        if epoch == 1 or epoch % 20 == 0:
            logger.info("Epoch %d/%d — val perplexity: %.3f", epoch, epochs, val_ppl)

        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info("Early stop at epoch %d (patience=%d)", epoch, patience)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    logger.info(
        "Training complete — %s model, vocab=%d, genres=%d, best_val_ppl=%.3f",
        layer, vocab_size, genre_count, best_val_ppl,
    )

    return NeuralSampler(
        model=model,
        token_to_idx=token_to_idx,
        idx_to_token=idx_to_token,
        genre_to_idx=genre_to_idx,
        layer=layer,
        context_len=context_len,
    )


def save_model(sampler: NeuralSampler, path: str) -> None:
    """Save a NeuralSampler to ``path`` (.pt) with a companion ``_meta.json``.

    The meta file stores vocabularies and hyper-parameters so the model can
    be reconstructed without the training data.
    """
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    torch.save(sampler.model.state_dict(), path)

    meta_path = path.replace(".pt", "_meta.json")
    meta = {
        "layer": sampler.layer,
        "context_len": sampler.context_len,
        "token_to_idx": sampler.token_to_idx,
        "genre_to_idx": sampler.genre_to_idx,
        "model_config": {
            "vocab_size": len(sampler.token_to_idx),
            "genre_count": len(sampler.genre_to_idx),
        },
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("Saved %s model → %s (+ %s)", sampler.layer, path, meta_path)


def load_model(path: str) -> Optional[NeuralSampler]:
    """Load a NeuralSampler from ``path`` (.pt).

    Returns None (with a warning) if:
    - torch is not installed
    - the .pt file or companion _meta.json is missing
    - loading fails for any reason
    """
    try:
        import torch as _torch  # noqa — already imported at module level, re-check
    except ImportError:
        logger.warning("torch not installed — neural backend unavailable")
        return None

    meta_path = path.replace(".pt", "_meta.json")
    if not os.path.exists(path) or not os.path.exists(meta_path):
        return None

    try:
        with open(meta_path) as f:
            meta = json.load(f)

        layer = meta["layer"]
        context_len = meta.get("context_len", 4)
        token_to_idx: Dict[str, int] = meta["token_to_idx"]
        genre_to_idx: Dict[str, int] = meta["genre_to_idx"]
        cfg = meta["model_config"]

        model_fn = ChordLSTM if layer == "chord" else MelodyLSTM
        model = model_fn(cfg["vocab_size"], cfg["genre_count"])
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        model.eval()

        idx_to_token = {v: k for k, v in token_to_idx.items()}
        return NeuralSampler(
            model=model,
            token_to_idx=token_to_idx,
            idx_to_token=idx_to_token,
            genre_to_idx=genre_to_idx,
            layer=layer,
            context_len=context_len,
        )
    except Exception as exc:
        logger.warning("Failed to load neural model from %s: %s", path, exc)
        return None
