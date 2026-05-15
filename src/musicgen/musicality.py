"""Musicality gate — two-layer pipeline quality filter.

Layer 1 (MIDI, pre-render, <5 ms):  check_midi_quality()
Layer 2 (audio, post-render):       get_musicality_score()  ← from musicality._core

Layer 2 lives in the standalone ``musicality`` package so it can be
installed and used independently. This module re-exports everything for
backward compatibility.
"""
from __future__ import annotations

import logging
import sys
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import mido
import numpy as np

# Re-export Layer 2 from the standalone package (backward compat).
from musicality._core import (
    _KS_MAJOR, _KS_MINOR,
    _NOTE_TO_SEMI, _MAJOR_PCS, _MINOR_PCS,
    _ks_key_correlation,
    _scale_adherence_score,
    _melodic_step_fraction,
    _ngram_entropy,
    _lz_ratio,
    _render_integrity,
    MusicalityAnalyzer,
    get_musicality_score,
)
from musicality import explain as get_musicality_explain

__all__ = [
    # Layer 1
    "MIDIQualityResult",
    "check_midi_quality",
    # Layer 2 (re-exported from musicality)
    "MusicalityAnalyzer",
    "get_musicality_score",
    "get_musicality_explain",
    # Internals re-exported for tests / component_diagnostic
    "_render_integrity",
    "_ks_key_correlation",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Layer 1 — MIDI quality
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MIDIQualityResult:
    passed: bool
    score: float
    hard_failures: List[str]
    soft_scores: Dict[str, float]


def _extract_midi_pitches(path: str) -> List[int]:
    pitches: List[int] = []
    try:
        mid = mido.MidiFile(path)
        for track in mid.tracks:
            for msg in track:
                if msg.type == "note_on" and msg.velocity > 0:
                    pitches.append(msg.note)
    except Exception:
        pass
    return pitches


def check_midi_quality(
    midi_paths: Dict[str, str],
    key: str = "C",
) -> MIDIQualityResult:
    """Layer 1 quality check — pure MIDI, no audio rendering required."""
    hard_failures: List[str] = []
    all_pitches: Dict[str, List[int]] = {}

    for layer, path in midi_paths.items():
        pitches = _extract_midi_pitches(path)
        all_pitches[layer] = pitches
        if len(pitches) == 0:
            hard_failures.append(f"{layer}: empty layer (0 notes)")

    melody_pitches = all_pitches.get("melody", [])
    if melody_pitches:
        counts = Counter(melody_pitches)
        dominant_fraction = counts.most_common(1)[0][1] / len(melody_pitches)
        if dominant_fraction > 0.80:
            hard_failures.append(
                f"melody: stuck pitch ({dominant_fraction:.0%} same note)"
            )
        pitch_range = max(melody_pitches) - min(melody_pitches)
        if pitch_range > 36:
            hard_failures.append(
                f"melody: extreme pitch range ({pitch_range} semitones)"
            )

    soft_scores: Dict[str, float] = {}
    if melody_pitches:
        pc_hist = np.zeros(12)
        for p in melody_pitches:
            pc_hist[p % 12] += 1
        total = pc_hist.sum()
        if total > 0:
            pc_hist /= total
        soft_scores["ks_correlation"]       = _ks_key_correlation(pc_hist)
        soft_scores["scale_adherence"]      = _scale_adherence_score(melody_pitches, key)
        soft_scores["melodic_step_fraction"] = _melodic_step_fraction(melody_pitches)
        soft_scores["ngram_entropy"]        = _ngram_entropy(melody_pitches, n=3)
        soft_scores["lz_ratio"]             = _lz_ratio(melody_pitches)

    passed = not bool(hard_failures)
    if not passed:
        score = 0.0
    elif soft_scores:
        score = float(np.mean(list(soft_scores.values())))
    else:
        score = 0.5

    return MIDIQualityResult(
        passed=passed,
        score=float(np.clip(score, 0.0, 1.0)),
        hard_failures=hard_failures,
        soft_scores=soft_scores,
    )


# ---------------------------------------------------------------------------
# Legacy CLI (kept for `musicgen musicality <file>` entry point)
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: musicgen musicality <audio_file>  (or: musicality explain <file>)")
        sys.exit(1)
    filename = sys.argv[1]
    s, components = get_musicality_score(filename)
    print(f"\nMusicality: {s:.4f}")
    for k, v in components.items():
        print(f"  {k:>20}: {v:.4f}")


if __name__ == "__main__":
    main()
