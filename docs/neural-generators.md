# Neural generators reference (v0.5)

ML-assisted chord and melody generation using small LSTMs trained on a self-generated corpus.

## Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick-start workflow](#quick-start-workflow)
- [CLI reference](#cli-reference)
  - [extract-sequences](#extract-sequences)
  - [train](#train)
  - [generate (neural flags)](#generate-neural-flags)
- [sequences.json schema](#sequencesjson-schema)
- [Model architecture](#model-architecture)
- [Training details](#training-details)
- [Inference and determinism](#inference-and-determinism)
- [Model file conventions](#model-file-conventions)
- [Python API](#python-api)
- [Fallback behaviour](#fallback-behaviour)
- [Config fields](#config-fields)

---

## Overview

v0.5 adds a second backend for chord and melody generation. The default `"markov"` backend is unchanged. The `"neural"` backend replaces the hand-crafted Markov transition matrices with two small LSTMs trained on a dataset produced by the Markov backend itself (or any other musicgen corpus):

- **ChordLSTM** — predicts the next chord Roman numeral given the previous `context_len` chords and a genre embedding
- **MelodyLSTM** — predicts the next scale degree (`"1"`–`"7"`) given the previous `context_len` degrees and a genre embedding

Both models are trained on sequences extracted from `sample.json` (chords) and `midi/melody.mid` (melody). The full pipeline is:

```
musicgen corpus  →  extract-sequences  →  train  →  generate --chord-backend neural
```

---

## Installation

```bash
pip install -e '.[neural]'
# installs: torch >= 2.0
```

torch is an optional dependency. All `import torch` calls are guarded; the neural package is a no-op (and the backends fall back to Markov) when torch is absent.

---

## Quick-start workflow

```bash
# 1. Generate a training corpus (500 samples, MIDI-only mode is fastest)
musicgen generate --count 500 --seed 1 --out ./corpus --output-mode midi-only

# 2. Extract sequences
musicgen extract-sequences --dataset ./corpus --output sequences.json

# 3. Train combined models (all genres)
musicgen train --sequences sequences.json --layer chord --output-dir ./models
musicgen train --sequences sequences.json --layer melody --output-dir ./models

# 4. Generate with neural backends
musicgen generate --count 32 --seed 42 --out ./dataset \
    --chord-backend neural --melody-backend neural \
    --models-dir ./models
```

---

## CLI reference

### `extract-sequences`

```
musicgen extract-sequences --dataset DIR --output FILE [--min-musicality FLOAT] [-v/-q]
```

| Option | Default | Description |
|---|---|---|
| `--dataset`, `-d` | (required) | Root of a musicgen dataset (`<dataset>/<idx>/sample.json`). |
| `--output`, `-o` | (required) | Output path for `sequences.json`. |
| `--min-musicality` | `0.0` | Skip samples with `musicality_score` below this threshold. `0.0` = include all. |
| `--verbose`, `-v` | `False` | DEBUG logging. |
| `--quiet`, `-q` | `False` | ERROR-only logging. |

Prints a summary on completion:

```
Extracted 487 samples (13 skipped). Chord sequences: 487. Melody sequences: 382.
```

Melody sequences may be fewer than chord sequences if some samples have no melody MIDI or the MIDI contains no note-on events.

---

### `train`

```
musicgen train --sequences FILE --layer LAYER [options] [-v/-q]
```

| Option | Default | Description |
|---|---|---|
| `--sequences`, `-s` | (required) | Path to `sequences.json` from `extract-sequences`. |
| `--layer`, `-l` | (required) | `chord` or `melody`. |
| `--genre`, `-g` | `None` | Filter training data to one or more genres (repeatable). `None` = train on all genres combined. |
| `--epochs`, `-e` | `200` | Maximum training epochs. Early stopping fires before this limit. |
| `--output-dir`, `-o` | `./models` | Directory where `.pt` and `_meta.json` files are written. |
| `--seed` | `42` | Reproducibility seed for torch weight initialization. |
| `--verbose`, `-v` | `False` | DEBUG logging (prints perplexity every 20 epochs). |
| `--quiet`, `-q` | `False` | ERROR-only logging. |

**Output files:**

| `--genre` arg | chord model files | melody model files |
|---|---|---|
| *(none — all genres)* | `models/chord.pt`, `models/chord_meta.json` | `models/melody.pt`, `models/melody_meta.json` |
| `--genre jazz` | `models/chord_jazz.pt`, `models/chord_jazz_meta.json` | ... |

**Example — genre-specific models:**

```bash
for GENRE in jazz hip-hop pop blues; do
    musicgen train -s sequences.json -l chord -g $GENRE -o ./models
    musicgen train -s sequences.json -l melody -g $GENRE -o ./models
done
# Also train combined fallback
musicgen train -s sequences.json -l chord -o ./models
musicgen train -s sequences.json -l melody -o ./models
```

---

### `generate` (neural flags)

```
musicgen generate --chord-backend neural --melody-backend neural --models-dir ./models [...]
```

| Option | Default | Description |
|---|---|---|
| `--chord-backend` | `markov` | `markov` (default) or `neural`. |
| `--melody-backend` | `markov` | `markov` (default) or `neural`. |
| `--models-dir PATH` | `<repo>/models` | Directory containing `.pt` model files. |

Both `--chord-backend` and `--melody-backend` are independent — you can use neural for one and Markov for the other.

---

## `sequences.json` schema

```json
{
  "metadata": {
    "n_samples": 487,
    "n_skipped": 13,
    "genres": ["jazz", "pop", "hip-hop"],
    "musicgen_version": "0.5.0"
  },
  "chord": [
    {
      "sample_index": 0,
      "genre": ["jazz"],
      "key": "C",
      "full_sequence": ["I", "vi", "ii", "V", "I", "IV", "V", "I", ...]
    },
    ...
  ],
  "melody": [
    {
      "sample_index": 0,
      "genre": ["jazz"],
      "key": "C",
      "full_sequence": ["1", "3", "5", "3", "2", "1", "7", "1", ...]
    },
    ...
  ]
}
```

**Chord sequences** are extracted directly from `sample.json → chord_progression` (Roman numerals per part, concatenated across all song parts).

**Melody sequences** are extracted from `midi/melody.mid` via mido:
1. All `note_on` events with `velocity > 0` are collected in tick order
2. Each MIDI note number is mapped to a scale degree string (`"1"`–`"7"`) using the key stored in `sample.json`
3. Chromatic (non-diatonic) notes are snapped to the nearest scale degree by semitone distance

---

## Model architecture

Both models share the `_SequenceLSTM` base class in `src/musicgen/neural/model.py`.

```
Input:
  token_ids  : (B, T)    — context window of T previous tokens
  genre_ids  : (B,)      — integer genre index

  embed(token_ids) → (B, T, embed_dim)
  genre_one_hot    → (B, genre_count) → expand → (B, T, genre_count)
  concat           → (B, T, embed_dim + genre_count)

LSTM:  num_layers=2, hidden, batch_first=True, dropout=0.2 (training only)
  → (B, T, hidden)

Linear head: hidden → vocab_size
  → (B, T, vocab_size)  logits

Only the last position logit → (B, vocab_size) is used for next-token prediction.
```

| Model | `embed_dim` | `hidden` | Approx. params |
|---|---|---|---|
| `ChordLSTM` | 16 | 64 | ~35 K |
| `MelodyLSTM` | 8 | 32 | ~10 K |

Both are instantiated via factory functions so they are easily replaced with larger configurations if corpus size warrants it:

```python
from musicgen.neural.model import ChordLSTM, MelodyLSTM

chord_model = ChordLSTM(vocab_size=20, genre_count=5)   # ~35K params
melody_model = MelodyLSTM(vocab_size=7, genre_count=5)  # ~10K params
```

---

## Training details

Handled by `musicgen.neural.trainer.train()`.

| Hyperparameter | Default | Notes |
|---|---|---|
| `epochs` | 200 | Maximum; early stopping usually fires earlier |
| `lr` | 1e-3 | Adam |
| `batch_size` | 64 | Sliding-window windows, not full sequences |
| `context_len` | 4 | Previous tokens fed to the LSTM |
| `patience` | 20 | Early stopping patience (epochs without val improvement) |
| `seed` | 42 | Controls torch weight initialization |

**Data pipeline:**

- 90 % train / 10 % val split (deterministic, first 10 % held out)
- Sliding-window dataset: for each position `i` in a sequence, the context is `seq[max(0, i-context_len+1) : i+1]` left-padded with `<pad>` (index 0), and the target is `seq[i+1]`
- `CrossEntropyLoss(ignore_index=0)` — `<pad>` tokens contribute no gradient
- Gradient clipping at norm 1.0
- Validation perplexity logged every 20 epochs; best checkpoint restored at end

**Corpus size guidance:**

| Corpus samples | Chord model | Melody model |
|---|---|---|
| 100–200 | Marginal improvement over Markov | ~Markov parity |
| 500–1 000 | Noticeably richer progressions | Good scale-degree continuity |
| 2 000+ | Recommended; early stopping fires around epoch 50–80 | Convergent within ~100 epochs |

Training on CPU takes under 1 minute for 500 samples at default hyperparameters.

---

## Inference and determinism

Sampling uses `musicgen.neural.sampler._sample_neural`:

1. Build a context window of length `context_len` (left-padded with `<pad>`)
2. Forward pass → logits `(1, context_len, vocab_size)` → take last position → `(vocab_size,)`
3. `softmax(last_logits)` → probability distribution over vocabulary
4. Filter out `<pad>` token
5. `rng.choices(token_indices, weights=probs, k=1)` — the same seeded `random.Random` used by the Markov path

Because step 2 is a pure function of fixed weights + the input context, and step 5 uses the caller-supplied seeded RNG, the **determinism contract is fully preserved**:

> Same `global_seed` + same `sample_index` + same trained model file → bit-identical MIDI and `sample.json` across any process invocation.

The model is loaded once per process into a module-level dict cache keyed by file path. Re-generating the same sample in the same process never re-reads the `.pt` file.

---

## Model file conventions

At inference, the generator looks for model files in `cfg.models_dir`:

```
models/
├── chord.pt              # combined (all genres)
├── chord_meta.json
├── chord_jazz.pt         # genre-specific (takes precedence when cfg.genre == ["jazz"])
├── chord_jazz_meta.json
├── melody.pt
├── melody_meta.json
├── melody_jazz.pt
└── melody_jazz_meta.json
```

**Lookup order** (first hit wins):

1. `{models_dir}/{layer}_{genre}.pt` — genre-specific (uses `cfg.genre[0]`)
2. `{models_dir}/{layer}.pt` — combined

If neither file exists, a warning is logged and the Markov backend is used for that layer.

Each `.pt` file has a companion `_meta.json` containing the vocabularies and hyperparameters needed to reconstruct the model without the training data:

```json
{
  "layer": "chord",
  "context_len": 4,
  "token_to_idx": {"<pad>": 0, "I": 1, "ii": 2, "IV": 3, "V": 4, "vi": 5},
  "genre_to_idx": {"__unknown__": 0, "jazz": 1, "pop": 2},
  "model_config": {"vocab_size": 6, "genre_count": 3}
}
```

---

## Python API

### Training

```python
from musicgen.neural.trainer import train, save_model, load_model

# Train
sampler = train(
    sequences_path="sequences.json",
    layer="chord",            # "chord" or "melody"
    genres=["jazz"],          # None = all genres
    epochs=200,
    lr=1e-3,
    batch_size=64,
    context_len=4,
    seed=42,
    patience=20,
)
print(sampler.layer)         # "chord"
print(list(sampler.token_to_idx))  # ["<pad>", "I", "ii", ...]

# Save / load
save_model(sampler, "models/chord_jazz.pt")
loaded = load_model("models/chord_jazz.pt")   # returns None on failure
```

### Sampling

```python
import random
from musicgen.neural.sampler import sample_chord_neural, sample_melody_neural
from musicgen.neural.trainer import load_model

chord_sampler = load_model("models/chord.pt")
rng = random.Random(42)

next_chord = sample_chord_neural(
    history=["I", "V", "vi"],   # previous chords (most recent last)
    genre=["jazz"],
    sampler=chord_sampler,
    rng=rng,
)
print(next_chord)   # e.g. "IV"

melody_sampler = load_model("models/melody.pt")
next_degree = sample_melody_neural(
    history=["1", "3", "5"],
    genre=["jazz"],
    sampler=melody_sampler,
    rng=rng,
)
print(next_degree)  # e.g. "4"
```

### Config

```python
from config import Config
from musicgen import generate

cfg = Config(
    global_seed=42,
    sample_index=0,
    dataset_root="./dataset",
    chord_backend="neural",    # "markov" | "neural"
    melody_backend="neural",   # "markov" | "neural"
    models_dir="./models",     # directory with .pt files
    genre=["jazz"],
)
result = generate(cfg)
```

---

## Fallback behaviour

The neural backend degrades gracefully in all error conditions:

| Condition | Behaviour |
|---|---|
| `torch` not installed | `_HAS_NEURAL_CHORD / _HAS_NEURAL_MELODY` is `False` at import; neural branch never entered |
| Model `.pt` file missing | `load_model()` returns `None`; warning logged; Markov backend used |
| `_meta.json` missing | Same as above |
| Load error (corrupt file, version mismatch) | `load_model()` catches exception, logs warning, returns `None` |
| Model returns empty eligible token list | `_sample_neural` falls back to uniform distribution over non-pad tokens |

In all cases the generator produces valid output and `sample.json` is written normally.

---

## Config fields

| Field | Type | Default | Description |
|---|---|---|---|
| `chord_backend` | `str` | `"markov"` | `"markov"` or `"neural"` |
| `melody_backend` | `str` | `"markov"` | `"markov"` or `"neural"` |
| `models_dir` | `str` | `<repo>/models` | Directory searched for `.pt` model files |

Both backend fields are validated at `Config.__post_init__` — any value other than `"markov"` or `"neural"` raises `ValueError`.
