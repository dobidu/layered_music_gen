"""Musicality — audio quality scoring for any audio file.

Standalone library. Works on WAV, MP3, FLAC, OGG, AIFF without a musicgen
pipeline. Also ships as part of the musicgen package.

Quick start::

    from musicality import score, explain, batch_score

    s = score("track.wav")                     # float in [0, 1]
    report = explain("track.wav")              # structured dict + labels
    results = batch_score(["a.wav", "b.mp3"])  # [(path, score), ...]

CLI::

    musicality score track.wav
    musicality explain track.wav
    musicality batch *.wav --json --out scores.csv
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

from musicality._core import (
    MusicalityAnalyzer,
    _render_integrity,
    _ks_key_correlation,
    get_musicality_score,
)

__all__ = [
    "score",
    "explain",
    "batch_score",
    "MusicalityAnalyzer",
    "get_musicality_score",
]

__version__ = "0.1.0"

# ---------------------------------------------------------------------------
# Score label thresholds
# ---------------------------------------------------------------------------

_LABELS = [(0.85, "strong"), (0.65, "moderate"), (0.40, "weak"), (0.0, "poor")]


def _label(s: float) -> str:
    for threshold, name in _LABELS:
        if s >= threshold:
            return name
    return "poor"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def score(path: str, genre: Optional[str] = None) -> float:
    """Score a single audio file.

    Parameters
    ----------
    path : str
        Path to audio file (WAV / MP3 / FLAC / OGG / AIFF / M4A).
    genre : str, optional
        Genre hint for tempo-range normalisation (e.g. "jazz", "electronic").
        When None, uses default 40–240 BPM range.

    Returns
    -------
    float
        Musicality score in [0, 1].  ≥ 0.5 considered acceptable quality.
    """
    s, _ = get_musicality_score(path)
    return s


def explain(path: str, genre: Optional[str] = None) -> Dict:
    """Score with sub-scores, labels, and integrity metrics.

    Returns a dict::

        {
            "score": 0.734,
            "label": "moderate",
            "components": {
                "tempo":   {"score": 0.82, "label": "strong",   "weight": 0.30,
                            "sub_scores": {"stability": …, "reasonableness": …, "clarity": …}},
                "harmony": {"score": 0.71, "label": "moderate", "weight": 0.30,
                            "sub_scores": {"key_clarity": …, "stability": …, "ks_correlation": …}},
                "rhythm":  {"score": 0.78, "label": "strong",   "weight": 0.25,
                            "sub_scores": {"regularity": …, "strength": …, "pattern": …, "density": …}},
                "noise":   {"score": 0.91, "label": "strong",   "weight": 0.15,
                            "sub_scores": {"music_signal_ratio": …}},
            },
            "integrity": {
                "clipping_ratio": 0.0,
                "dc_offset": 0.001,
                "silence_ratio": 0.02,
                "crest_db": 14.2,
                "penalty": 0.0,
            },
        }
    """
    import warnings
    import librosa
    import numpy as np

    analyzer = MusicalityAnalyzer()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y, sr = librosa.load(path)

    integrity = _render_integrity(y, sr)

    clipping_penalty = float(np.clip(integrity["clipping_ratio"] * 2.0, 0.0, 1.0))
    silence_frac = float(np.clip((integrity["silence_ratio"] - 0.5) * 2.0, 0.0, 1.0))
    dc_penalty = 1.0 if integrity["dc_offset"] > 0.01 else 0.0
    crest = integrity["crest_db"]
    crest_penalty = 1.0 if (crest < 3.0 or crest > 30.0) else 0.0
    integrity_penalty = max(clipping_penalty, silence_frac, dc_penalty, crest_penalty)

    tempo_sub = analyzer.analyze_tempo(y, sr)
    harmony_sub = analyzer.analyze_harmony(y, sr)
    rhythm_sub = analyzer.analyze_rhythm(y, sr)
    noise_sub = analyzer.analyze_noise(y, sr)

    weights = MusicalityAnalyzer.weights

    def _component(sub: dict, key: str) -> dict:
        s = float(np.clip(np.mean(list(sub.values())), 0, 1)) \
            if key != "noise" else float(np.clip(sub.get("music_signal_ratio", 0), 0, 1))
        return {"score": s, "label": _label(s), "weight": weights[key], "sub_scores": sub}

    import numpy as np
    components = {
        "tempo":   _component(tempo_sub, "tempo"),
        "harmony": _component(harmony_sub, "harmony"),
        "rhythm":  _component(rhythm_sub, "rhythm"),
        "noise":   _component(noise_sub, "noise"),
    }

    raw = sum(components[k]["score"] * weights[k] for k in weights)
    total = float(np.clip(raw * (1.0 - integrity_penalty), 0.0, 1.0))

    return {
        "score": total,
        "label": _label(total),
        "path": path,
        "components": components,
        "integrity": {**integrity, "penalty": integrity_penalty},
    }


def batch_score(
    paths: List[str],
    genre: Optional[str] = None,
    show_progress: bool = False,
) -> List[Tuple[str, float]]:
    """Score multiple audio files.

    Returns
    -------
    list of (path, score) tuples, in input order.
    Failed files return score=0.0.
    """
    results = []
    for i, p in enumerate(paths):
        try:
            s = score(p, genre=genre)
        except Exception:
            s = 0.0
        results.append((p, s))
        if show_progress and (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(paths)}]", flush=True)
    return results
