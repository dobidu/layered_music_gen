"""Two-layer musicality analysis (v0.3 Phase 3a).

Layer 1 (MIDI, pre-render, <5ms):
  - Hard checks: empty layer, stuck note, extreme pitch range
  - Soft symbolic metrics: KS key correlation, scale adherence,
    melodic step fraction, n-gram entropy, LZ compression ratio

Layer 2 (audio, post-render):
  - Render integrity: clipping, DC offset, silence ratio, crest factor
  - Audio analysis: genre-aware tempo, chroma-KS harmony, rhythm, noise
"""
import logging
import sys
import warnings
import zlib
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import librosa
import mido
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Krumhansl–Schmuckler key profiles
# ---------------------------------------------------------------------------

_KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# ---------------------------------------------------------------------------
# Scale constants
# ---------------------------------------------------------------------------

_NOTE_TO_SEMI: Dict[str, int] = {
    "C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5,
    "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11,
    "Db": 1, "Eb": 3, "Gb": 6, "Ab": 8, "Bb": 10,
}
_MAJOR_PCS = frozenset([0, 2, 4, 5, 7, 9, 11])
_MINOR_PCS = frozenset([0, 2, 3, 5, 7, 8, 10])


# ---------------------------------------------------------------------------
# Layer 1 — MIDI quality dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MIDIQualityResult:
    passed: bool
    score: float
    hard_failures: List[str]
    soft_scores: Dict[str, float]


# ---------------------------------------------------------------------------
# Soft metric helpers (internal but importable for unit tests)
# ---------------------------------------------------------------------------

def _ks_key_correlation(pc_hist: np.ndarray) -> float:
    """Max Pearson correlation of *pc_hist* against all 24 KS key profiles → [0, 1]."""
    best = -1.0
    for profile in (_KS_MAJOR, _KS_MINOR):
        for shift in range(12):
            rotated = np.roll(profile, shift)
            if np.std(rotated) > 0 and np.std(pc_hist) > 0:
                corr = float(np.corrcoef(pc_hist, rotated)[0, 1])
                best = max(best, corr)
    return float(np.clip((best + 1.0) / 2.0, 0.0, 1.0))


def _scale_adherence_score(pitches: List[int], key: str) -> float:
    """Fraction of *pitches* that belong to *key*'s diatonic scale → [0, 1]."""
    if not pitches:
        return 0.0
    is_minor = key.endswith("m") and len(key) > 1 and key[-2] not in ("#", "b")
    root_name = key[:-1] if is_minor else key
    root = _NOTE_TO_SEMI.get(root_name, 0)
    template = _MINOR_PCS if is_minor else _MAJOR_PCS
    scale_pcs = frozenset((root + pc) % 12 for pc in template)
    return float(sum(1 for p in pitches if p % 12 in scale_pcs) / len(pitches))


def _melodic_step_fraction(pitches: List[int]) -> float:
    """Fraction of consecutive intervals ≤ 2 semitones (steps vs leaps) → [0, 1]."""
    if len(pitches) < 2:
        return 0.0
    intervals = [abs(pitches[i + 1] - pitches[i]) for i in range(len(pitches) - 1)]
    return float(sum(1 for iv in intervals if iv <= 2) / len(intervals))


def _ngram_entropy(symbols: List, n: int = 3) -> float:
    """Normalized Shannon entropy of *n*-gram distribution → [0, 1]."""
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
    """zlib compression ratio of *symbols* as a proxy for LZ complexity → [0, 1]."""
    if not symbols:
        return 0.0
    data = bytes(int(s) % 256 for s in symbols)
    if len(data) == 0:
        return 0.0
    compressed = zlib.compress(data)
    return float(min(1.0, len(compressed) / len(data)))


def _render_integrity(y: np.ndarray, sr: int) -> Dict[str, float]:
    """Compute render-integrity metrics from raw audio samples."""
    clipping_threshold = 0.99
    clipping_ratio = float(np.mean(np.abs(y) >= clipping_threshold))
    dc_offset = float(np.abs(np.mean(y)))

    # Frame-based silence: fraction of 20ms frames with RMS below -60 dBFS.
    # Sample-level |y| < threshold counts zero-crossings of any non-silent wave,
    # making pure sinusoids appear ~2% silent — uninterpretable for a paper metric.
    _n = int(sr * 20 / 1000)  # 20 ms frame
    if _n > 0 and len(y) >= _n:
        _frames = y[: (len(y) // _n) * _n].reshape(-1, _n)
        _rms_frames = np.sqrt(np.mean(_frames ** 2, axis=1) + 1e-12)
        _silence_floor = 10 ** (-60.0 / 20.0)
        silence_ratio = float(np.mean(_rms_frames < _silence_floor))
    else:
        silence_ratio = 0.0

    peak = float(np.max(np.abs(y))) if len(y) > 0 else 1e-10
    rms = float(np.sqrt(np.mean(y ** 2)))
    rms_safe = max(rms, 1e-10)
    crest_db = float(20.0 * np.log10(peak / rms_safe)) if peak > 0 else 0.0
    return {
        "clipping_ratio": clipping_ratio,
        "dc_offset": dc_offset,
        "silence_ratio": silence_ratio,
        "crest_db": crest_db,
    }


# ---------------------------------------------------------------------------
# Layer 1 — MIDI note extraction + check_midi_quality
# ---------------------------------------------------------------------------

def _extract_midi_pitches(path: str) -> List[int]:
    """Extract ordered list of pitched note-on events from a MIDI file."""
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
    """Layer 1 quality check — pure MIDI, no audio rendering required.

    Hard failures make ``passed=False`` and ``score=0.0``.
    Soft metrics are always computed on the melody layer when available.
    """
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
        soft_scores["ks_correlation"] = _ks_key_correlation(pc_hist)
        soft_scores["scale_adherence"] = _scale_adherence_score(melody_pitches, key)
        soft_scores["melodic_step_fraction"] = _melodic_step_fraction(melody_pitches)
        soft_scores["ngram_entropy"] = _ngram_entropy(melody_pitches, n=3)
        soft_scores["lz_ratio"] = _lz_ratio(melody_pitches)

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
# Layer 2 — Audio analysis (redesigned MusicalityAnalyzer)
# ---------------------------------------------------------------------------

class MusicalityAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.weights = {
            "tempo": 0.30,
            "harmony": 0.30,
            "rhythm": 0.25,
            "noise": 0.15,
        }

    def analyze_tempo(
        self, y: np.ndarray, sr: int, genre_spec=None
    ) -> Dict[str, float]:
        try:
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            # librosa >= 0.10 returns tempo as ndarray([bpm]); extract scalar.
            tempo = float(np.squeeze(tempo))

            if genre_spec is not None:
                t_min = float(getattr(genre_spec, "tempo_min", None) or 40.0)
                t_max = float(getattr(genre_spec, "tempo_max", None) or 240.0)
            else:
                t_min, t_max = 40.0, 240.0

            if t_max > t_min:
                tempo_range = float(np.clip((tempo - t_min) / (t_max - t_min), 0, 1))
            else:
                tempo_range = 0.5
            tempo_reasonableness = 1.0 - abs(tempo_range - 0.5) * 2.0

            if len(beats) >= 2:
                beat_intervals = np.diff(beats)
                tempo_stability = 1.0 / (
                    1.0 + float(np.std(beat_intervals)) / (float(np.mean(beat_intervals)) + 1e-6)
                )
            else:
                tempo_stability = 0.0

            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            max_onset = float(np.max(onset_env))
            tempo_clarity = float(np.mean(onset_env)) / max_onset if max_onset > 0 else 0.0

            return {
                "stability": float(np.clip(tempo_stability, 0, 1)),
                "reasonableness": float(np.clip(tempo_reasonableness, 0, 1)),
                "clarity": float(np.clip(tempo_clarity, 0, 1)),
            }
        except Exception as exc:
            self.logger.debug("tempo analysis failed: %s", exc)
            return {"stability": 0.0, "reasonableness": 0.0, "clarity": 0.0}

    def analyze_harmony(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        try:
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr, bins_per_octave=36)
            mean_chroma = np.mean(chroma, axis=1)
            total = float(mean_chroma.sum())

            pc_hist = mean_chroma / total if total > 0 else np.ones(12) / 12.0
            ks_score = _ks_key_correlation(pc_hist)

            key_clarity = float(np.max(mean_chroma)) if total > 0 else 0.0
            # chroma_cqt values are continuous floats; exact equality between
            # consecutive frames is ~never true, so the old (diff != 0) test
            # returned harmonic_changes ≈ 1.0 for any real input, collapsing
            # stability to ≈ 0 regardless of actual harmonic smoothness.
            chroma_diff = np.linalg.norm(np.diff(chroma, axis=1), axis=0)
            stability = float(np.exp(-float(np.mean(chroma_diff))))

            return {
                "key_clarity": float(np.clip(key_clarity, 0, 1)),
                "stability": float(np.clip(stability, 0, 1)),
                "ks_correlation": float(np.clip(ks_score, 0, 1)),
            }
        except Exception as exc:
            self.logger.debug("harmony analysis failed: %s", exc)
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
                background = float(np.mean(onset_env))
                beat_strength = float(
                    np.clip(
                        (float(np.mean(peak_values)) - background) / (float(np.max(onset_env)) + 1e-6),
                        0, 1,
                    )
                )
            else:
                beat_strength = 0.0

            if len(onset_env) > 0:
                acf = librosa.autocorrelate(onset_env)
                # std/mean of the full half-ACF has no musical interpretation:
                # it varies nearly randomly across samples. Use first-peak
                # prominence instead: ratio of the strongest periodic lag to
                # the mean ACF level, normalised to [0,1] by dividing by 5.
                acf_search = acf[1 : len(acf) // 2]
                if len(acf_search) > 0 and float(np.mean(acf_search)) > 0:
                    peak_lag = int(np.argmax(acf_search))
                    pattern_score = float(
                        np.clip(
                            acf_search[peak_lag] / (float(np.mean(acf_search)) + 1e-6) / 5.0,
                            0, 1,
                        )
                    )
                else:
                    pattern_score = 0.0
            else:
                pattern_score = 0.0

            if len(y) > 0:
                density = len(beats) / (len(y) / sr)
                density_score = float(np.clip(1.0 - 0.5 * abs(density - 1.5) / 1.5, 0, 1))
            else:
                density_score = 0.0

            return {
                "regularity": float(np.clip(rhythm_regularity, 0, 1)) * 0.35,
                "strength": float(np.clip(beat_strength, 0, 1)) * 0.30,
                "pattern": float(np.clip(pattern_score, 0, 1)) * 0.20,
                "density": float(np.clip(density_score, 0, 1)) * 0.15,
            }
        except Exception as exc:
            self.logger.debug("rhythm analysis failed: %s", exc)
            return {"regularity": 0.0, "strength": 0.0, "pattern": 0.0, "density": 0.0}

    def analyze_noise(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        try:
            zcr = librosa.feature.zero_crossing_rate(y)
            zcr_score = 1.0 - float(np.mean(zcr))
            flatness = librosa.feature.spectral_flatness(y=y)
            flatness_score = 1.0 - float(np.mean(flatness))
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            max_contrast = float(np.max(contrast))
            contrast_score = float(np.mean(contrast)) / max_contrast if max_contrast > 0 else 0.0
            noise_score = float(np.clip((zcr_score + flatness_score + contrast_score) / 3, 0, 1))
            return {"noise_level": 1.0 - noise_score, "music_signal_ratio": noise_score}
        except Exception as exc:
            self.logger.debug("noise analysis failed: %s", exc)
            return {"noise_level": 1.0, "music_signal_ratio": 0.0}

    def calculate_musicality(
        self, filename: str, genre_spec=None
    ) -> Tuple[float, Dict[str, float]]:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                y, sr = librosa.load(filename)

            integrity = _render_integrity(y, sr)

            # Integrity penalty: clipping, silence, DC bias, and crest factor.
            clipping_penalty = float(np.clip(integrity["clipping_ratio"] * 2.0, 0.0, 1.0))
            silence_frac = float(np.clip((integrity["silence_ratio"] - 0.5) * 2.0, 0.0, 1.0))
            # DC offset > 0.01 (normalised float) indicates a render artifact.
            # Probe over 100 normal samples showed p99=0.0035, max=0.0035 —
            # 0.001 fired on 34% of normal output. 0.01 gives 0% false-reject
            # while still catching genuine DC bias (which typically exceeds 0.1).
            dc_penalty = 1.0 if integrity["dc_offset"] > 0.01 else 0.0
            # Crest factor outside [3, 30] dB is physically implausible audio.
            crest = integrity["crest_db"]
            crest_penalty = 1.0 if (crest < 3.0 or crest > 30.0) else 0.0
            integrity_penalty = max(clipping_penalty, silence_frac, dc_penalty, crest_penalty)

            tempo_scores = self.analyze_tempo(y, sr, genre_spec=genre_spec)
            harmony_scores = self.analyze_harmony(y, sr)
            rhythm_scores = self.analyze_rhythm(y, sr)
            noise_scores = self.analyze_noise(y, sr)

            base_scores = {
                "tempo": float(np.clip(np.mean(list(tempo_scores.values())), 0, 1)),
                "harmony": float(np.clip(np.mean(list(harmony_scores.values())), 0, 1)),
                "rhythm": float(np.clip(np.mean(list(rhythm_scores.values())), 0, 1)),
                "noise": float(np.clip(noise_scores["music_signal_ratio"], 0, 1)),
            }

            raw_score = sum(base_scores[k] * self.weights[k] for k in self.weights)
            total = float(np.clip(raw_score * (1.0 - integrity_penalty), 0.0, 1.0))

            components = {**base_scores, **integrity}
            return total, components

        except (FileNotFoundError, OSError, ValueError, RuntimeError) as exc:
            self.logger.exception("Error processing %s: %s", filename, exc)
            return 0.0, {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_musicality_score(
    filename: str, genre_spec=None
) -> Tuple[float, Dict[str, float]]:
    """Return (score, components) for an audio file. score ∈ [0, 1]."""
    analyzer = MusicalityAnalyzer()
    return analyzer.calculate_musicality(filename, genre_spec=genre_spec)


def main():
    if len(sys.argv) < 2:
        print("Usage: musicgen musicality <audio_file>")
        sys.exit(1)
    filename = sys.argv[1]
    score, components = get_musicality_score(filename)
    print("\nMusicality Analysis Results:")
    print("-" * 40)
    print(f"Overall Score: {score:.2f}")
    print("\nComponent Scores:")
    for k, v in components.items():
        print(f"  {k:>20}: {v:.4f}")


if __name__ == "__main__":
    main()
