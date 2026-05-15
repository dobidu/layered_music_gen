"""Layer 2 audio musicality analysis — standalone, no musicgen dependency.

All analysis operates on raw audio (path → librosa.load). No MIDI, no
FluidSynth. Suitable for scoring arbitrary WAV/MP3/FLAC/OGG/AIFF files.
"""
from __future__ import annotations

import logging
import warnings
import zlib
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Krumhansl–Schmuckler key profiles
# ---------------------------------------------------------------------------

_KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

_NOTE_TO_SEMI: Dict[str, int] = {
    "C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5,
    "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11,
    "Db": 1, "Eb": 3, "Gb": 6, "Ab": 8, "Bb": 10,
}
_MAJOR_PCS = frozenset([0, 2, 4, 5, 7, 9, 11])
_MINOR_PCS = frozenset([0, 2, 3, 5, 7, 8, 10])


# ---------------------------------------------------------------------------
# KS + scale helpers (also used by musicgen.musicality Layer 1)
# ---------------------------------------------------------------------------

def _ks_key_correlation(pc_hist: np.ndarray) -> float:
    """Max Pearson correlation of pc_hist against all 24 KS profiles → [0, 1]."""
    best = -1.0
    for profile in (_KS_MAJOR, _KS_MINOR):
        for shift in range(12):
            rotated = np.roll(profile, shift)
            if np.std(rotated) > 0 and np.std(pc_hist) > 0:
                corr = float(np.corrcoef(pc_hist, rotated)[0, 1])
                best = max(best, corr)
    return float(np.clip((best + 1.0) / 2.0, 0.0, 1.0))


def _scale_adherence_score(pitches: List[int], key: str) -> float:
    """Fraction of pitches belonging to key's diatonic scale → [0, 1]."""
    if not pitches:
        return 0.0
    is_minor = key.endswith("m") and len(key) > 1 and key[-2] not in ("#", "b")
    root_name = key[:-1] if is_minor else key
    root = _NOTE_TO_SEMI.get(root_name, 0)
    template = _MINOR_PCS if is_minor else _MAJOR_PCS
    scale_pcs = frozenset((root + pc) % 12 for pc in template)
    return float(sum(1 for p in pitches if p % 12 in scale_pcs) / len(pitches))


def _melodic_step_fraction(pitches: List[int]) -> float:
    """Fraction of consecutive intervals ≤ 2 semitones → [0, 1]."""
    if len(pitches) < 2:
        return 0.0
    intervals = [abs(pitches[i + 1] - pitches[i]) for i in range(len(pitches) - 1)]
    return float(sum(1 for iv in intervals if iv <= 2) / len(intervals))


def _ngram_entropy(symbols: List, n: int = 3) -> float:
    """Normalised Shannon entropy of n-gram distribution → [0, 1]."""
    if len(symbols) < n + 1:
        return 0.0
    ngrams = [tuple(symbols[i: i + n]) for i in range(len(symbols) - n + 1)]
    counts = Counter(ngrams)
    total = sum(counts.values())
    probs = np.array([c / total for c in counts.values()])
    raw = float(-np.sum(probs * np.log2(probs + 1e-12)))
    max_entropy = np.log2(len(ngrams)) if len(ngrams) > 1 else 1.0
    return float(np.clip(raw / max_entropy, 0.0, 1.0))


def _lz_ratio(symbols: List) -> float:
    """zlib compression ratio as LZ complexity proxy → [0, 1]."""
    if not symbols:
        return 0.0
    data = bytes(int(s) % 256 for s in symbols)
    return float(min(1.0, len(zlib.compress(data)) / len(data))) if data else 0.0


# ---------------------------------------------------------------------------
# Render integrity
# ---------------------------------------------------------------------------

def _render_integrity(y: np.ndarray, sr: int) -> Dict[str, float]:
    """Clipping, DC offset, 20 ms-frame silence ratio, crest factor."""
    clipping_ratio = float(np.mean(np.abs(y) >= 0.99))
    dc_offset = float(np.abs(np.mean(y)))

    _n = int(sr * 20 / 1000)
    if _n > 0 and len(y) >= _n:
        _frames = y[: (len(y) // _n) * _n].reshape(-1, _n)
        _rms_frames = np.sqrt(np.mean(_frames ** 2, axis=1) + 1e-12)
        _silence_floor = 10 ** (-60.0 / 20.0)
        silence_ratio = float(np.mean(_rms_frames < _silence_floor))
    else:
        silence_ratio = 0.0

    peak = float(np.max(np.abs(y))) if len(y) > 0 else 1e-10
    rms = float(np.sqrt(np.mean(y ** 2)))
    crest_db = float(20.0 * np.log10(peak / max(rms, 1e-10))) if peak > 0 else 0.0

    return {
        "clipping_ratio": clipping_ratio,
        "dc_offset": dc_offset,
        "silence_ratio": silence_ratio,
        "crest_db": crest_db,
    }


# ---------------------------------------------------------------------------
# MusicalityAnalyzer (Layer 2)
# ---------------------------------------------------------------------------

class MusicalityAnalyzer:
    """Audio-domain musicality scorer. Genre-aware when genre_spec supplied."""

    weights = {"tempo": 0.30, "harmony": 0.30, "rhythm": 0.25, "noise": 0.15}

    def analyze_tempo(self, y: np.ndarray, sr: int, genre_spec=None) -> Dict[str, float]:
        try:
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            tempo = float(np.squeeze(tempo))

            t_min = float(getattr(genre_spec, "tempo_min", None) or 40.0) \
                if genre_spec is not None else 40.0
            t_max = float(getattr(genre_spec, "tempo_max", None) or 240.0) \
                if genre_spec is not None else 240.0

            tempo_range = float(np.clip((tempo - t_min) / (t_max - t_min + 1e-6), 0, 1))
            tempo_reasonableness = 1.0 - abs(tempo_range - 0.5) * 2.0

            if len(beats) >= 2:
                beat_intervals = np.diff(beats)
                tempo_stability = 1.0 / (
                    1.0 + float(np.std(beat_intervals)) / (float(np.mean(beat_intervals)) + 1e-6)
                )
            else:
                tempo_stability = 0.0

            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            if len(onset_env) > 4 and float(np.max(onset_env)) > 0:
                _acf = librosa.autocorrelate(onset_env, max_size=len(onset_env) // 2)[1:]
                _mean_acf = float(np.mean(_acf))
                tempo_clarity = float(np.clip(
                    float(np.max(_acf)) / (_mean_acf + 1e-6) / 10.0, 0, 1
                )) if _mean_acf > 0 else 0.0
            else:
                tempo_clarity = 0.0

            return {
                "stability":       float(np.clip(tempo_stability, 0, 1)),
                "reasonableness":  float(np.clip(tempo_reasonableness, 0, 1)),
                "clarity":         float(np.clip(tempo_clarity, 0, 1)),
            }
        except Exception as exc:
            logger.debug("tempo analysis failed: %s", exc)
            return {"stability": 0.0, "reasonableness": 0.0, "clarity": 0.0}

    def analyze_harmony(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        try:
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr, bins_per_octave=36)
            mean_chroma = np.mean(chroma, axis=1)
            total = float(mean_chroma.sum())
            pc_hist = mean_chroma / total if total > 0 else np.ones(12) / 12.0
            ks_score = _ks_key_correlation(pc_hist)
            key_clarity = float(np.max(mean_chroma)) if total > 0 else 0.0
            chroma_diff = np.linalg.norm(np.diff(chroma, axis=1), axis=0)
            stability = float(np.exp(-float(np.mean(chroma_diff))))
            return {
                "key_clarity":    float(np.clip(key_clarity, 0, 1)),
                "stability":      float(np.clip(stability, 0, 1)),
                "ks_correlation": float(np.clip(ks_score, 0, 1)),
            }
        except Exception as exc:
            logger.debug("harmony analysis failed: %s", exc)
            return {"key_clarity": 0.0, "stability": 0.0, "ks_correlation": 0.0}

    def analyze_rhythm(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        try:
            onset_env = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.mean, n_mels=128)
            tempo, beats = librosa.beat.beat_track(
                onset_envelope=onset_env, sr=sr, hop_length=512, tightness=50
            )
            if len(beats) >= 4:
                beat_intervals = np.diff(beats)
                local_var = float(np.std(beat_intervals)) / (float(np.mean(beat_intervals)) + 1e-6)
                global_var = float(np.std(beat_intervals)) / (len(beat_intervals) + 1e-6)
                rhythm_regularity = 1.0 / (1.0 + 0.5 * (local_var + global_var))
            else:
                rhythm_regularity = 0.0

            if len(beats) > 0:
                peak_values = onset_env[beats]
                beat_strength = float(np.clip(
                    (float(np.mean(peak_values)) - float(np.mean(onset_env)))
                    / (float(np.max(onset_env)) + 1e-6), 0, 1
                ))
            else:
                beat_strength = 0.0

            if len(onset_env) > 0:
                acf = librosa.autocorrelate(onset_env)
                acf_search = acf[1: len(acf) // 2]
                if len(acf_search) > 0 and float(np.mean(acf_search)) > 0:
                    peak_lag = int(np.argmax(acf_search))
                    pattern_score = float(np.clip(
                        acf_search[peak_lag] / (float(np.mean(acf_search)) + 1e-6) / 5.0, 0, 1
                    ))
                else:
                    pattern_score = 0.0
            else:
                pattern_score = 0.0

            density_score = float(np.clip(
                1.0 - 0.5 * abs(len(beats) / (len(y) / sr + 1e-6) - 1.5) / 1.5, 0, 1
            )) if len(y) > 0 else 0.0

            return {
                "regularity": float(np.clip(rhythm_regularity, 0, 1)),
                "strength":   float(np.clip(beat_strength, 0, 1)),
                "pattern":    float(np.clip(pattern_score, 0, 1)),
                "density":    float(np.clip(density_score, 0, 1)),
            }
        except Exception as exc:
            logger.debug("rhythm analysis failed: %s", exc)
            return {"regularity": 0.0, "strength": 0.0, "pattern": 0.0, "density": 0.0}

    def analyze_noise(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        try:
            zcr_score = 1.0 - float(np.mean(librosa.feature.zero_crossing_rate(y)))
            flatness_score = 1.0 - float(np.mean(librosa.feature.spectral_flatness(y=y)))
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            max_c = float(np.max(contrast))
            contrast_score = float(np.mean(contrast)) / max_c if max_c > 0 else 0.0
            noise_score = float(np.clip((zcr_score + flatness_score + contrast_score) / 3, 0, 1))
            return {"noise_level": 1.0 - noise_score, "music_signal_ratio": noise_score}
        except Exception as exc:
            logger.debug("noise analysis failed: %s", exc)
            return {"noise_level": 1.0, "music_signal_ratio": 0.0}

    def calculate_musicality(
        self, filename: str, genre_spec=None
    ) -> Tuple[float, Dict[str, float]]:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                y, sr = librosa.load(filename)

            integrity = _render_integrity(y, sr)

            clipping_penalty = float(np.clip(integrity["clipping_ratio"] * 2.0, 0.0, 1.0))
            silence_frac = float(np.clip((integrity["silence_ratio"] - 0.5) * 2.0, 0.0, 1.0))
            dc_penalty = 1.0 if integrity["dc_offset"] > 0.01 else 0.0
            crest = integrity["crest_db"]
            crest_penalty = 1.0 if (crest < 3.0 or crest > 30.0) else 0.0
            integrity_penalty = max(clipping_penalty, silence_frac, dc_penalty, crest_penalty)

            tempo_scores = self.analyze_tempo(y, sr, genre_spec=genre_spec)
            harmony_scores = self.analyze_harmony(y, sr)
            rhythm_scores = self.analyze_rhythm(y, sr)
            noise_scores = self.analyze_noise(y, sr)

            base_scores = {
                "tempo":   float(np.clip(np.mean(list(tempo_scores.values())), 0, 1)),
                "harmony": float(np.clip(np.mean(list(harmony_scores.values())), 0, 1)),
                "rhythm":  float(np.clip(np.mean(list(rhythm_scores.values())), 0, 1)),
                "noise":   float(np.clip(noise_scores["music_signal_ratio"], 0, 1)),
            }

            raw_score = sum(base_scores[k] * self.weights[k] for k in self.weights)
            total = float(np.clip(raw_score * (1.0 - integrity_penalty), 0.0, 1.0))
            return total, {**base_scores, **integrity}

        except (FileNotFoundError, OSError, ValueError, RuntimeError) as exc:
            logger.exception("Error processing %s: %s", filename, exc)
            return 0.0, {}


# ---------------------------------------------------------------------------
# Functional API
# ---------------------------------------------------------------------------

def get_musicality_score(
    filename: str, genre_spec=None
) -> Tuple[float, Dict[str, float]]:
    """Return (score, components) for an audio file. score ∈ [0, 1]."""
    return MusicalityAnalyzer().calculate_musicality(filename, genre_spec=genre_spec)
