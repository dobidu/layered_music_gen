"""Construct validity experiment (E2).

Validates that the musicality gate discriminates between well-formed music
and known-pathological signals. Two pathology modes:

  synthetic  — programmatically generated bad audio (silence, white noise,
               hard-clipped sine, DC offset, pure monotone sine)
  real       — controlled degradations of good musicgen outputs (silence gaps,
               noise overlay, tempo warp, pitch scramble, time reversal)
  both       — union of the above

Primary metric: AUROC treating gate score as binary classifier (good=1, bad=0).
Secondary: Cohen's d separation (mean_good - mean_bad) / pooled_std.
"""
from __future__ import annotations

import logging
import random
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_SR = 44100
_DURATION = 5.0  # seconds for synthetic samples

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PathologyResult:
    name: str
    scores: List[float]
    mean: float
    std: float
    auroc: float
    separation: float  # Cohen's d vs good set


@dataclass
class ConstructValidityResult:
    n_good: int
    good_scores: List[float]
    good_mean: float
    good_std: float
    pathologies: List[PathologyResult]
    overall_auroc: float
    overall_separation: float


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def _auroc(pos: List[float], neg: List[float]) -> float:
    if not pos or not neg:
        return 0.5
    count = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                count += 1.0
            elif p == n:
                count += 0.5
    return count / (len(pos) * len(neg))


def _cohens_d(good: List[float], bad: List[float]) -> float:
    if not good or not bad:
        return 0.0
    mg, mb = float(np.mean(good)), float(np.mean(bad))
    sg, sb = float(np.std(good)), float(np.std(bad))
    ng, nb = len(good), len(bad)
    pooled = np.sqrt(((ng - 1) * sg ** 2 + (nb - 1) * sb ** 2) / (ng + nb - 2 + 1e-9))
    return float((mg - mb) / (pooled + 1e-9))


# ---------------------------------------------------------------------------
# Audio I/O helpers
# ---------------------------------------------------------------------------


def _write_wav(y: np.ndarray, sr: int, path: str) -> str:
    import soundfile as sf
    sf.write(path, y.astype(np.float32), sr)
    return path


def _score_wav(wav_path: str) -> Optional[float]:
    try:
        from musicgen.musicality import get_musicality_score
        score, _ = get_musicality_score(wav_path)
        return float(score)
    except Exception as exc:
        logger.warning("score failed %s: %s", wav_path, exc)
        return None


# ---------------------------------------------------------------------------
# Synthetic bad generators
# ---------------------------------------------------------------------------

SYNTHETIC_TYPES: List[str] = ["silence", "noise", "clip", "dc", "monotone"]


def _synth_silence(path: str, sr: int = _SR, duration: float = _DURATION, rng=None) -> str:
    return _write_wav(np.zeros(int(duration * sr)), sr, path)


def _synth_noise(path: str, sr: int = _SR, duration: float = _DURATION, rng=None) -> str:
    _rng = rng if rng is not None else np.random.default_rng(0)
    y = _rng.standard_normal(int(duration * sr)).astype(np.float32) * 0.5
    return _write_wav(y, sr, path)


def _synth_clip(path: str, sr: int = _SR, duration: float = _DURATION, rng=None) -> str:
    t = np.linspace(0, duration, int(duration * sr))
    y = np.clip(np.sin(2 * np.pi * 440 * t) * 10.0, -0.1, 0.1)
    return _write_wav(y, sr, path)


def _synth_dc(path: str, sr: int = _SR, duration: float = _DURATION, rng=None) -> str:
    t = np.linspace(0, duration, int(duration * sr))
    y = 0.5 + 0.05 * np.sin(2 * np.pi * 440 * t)
    return _write_wav(y, sr, path)


def _synth_monotone(path: str, sr: int = _SR, duration: float = _DURATION, rng=None) -> str:
    t = np.linspace(0, duration, int(duration * sr))
    y = 0.5 * np.sin(2 * np.pi * 440 * t)
    return _write_wav(y, sr, path)


_SYNTH_GENERATORS = {
    "silence": _synth_silence,
    "noise": _synth_noise,
    "clip": _synth_clip,
    "dc": _synth_dc,
    "monotone": _synth_monotone,
}


# ---------------------------------------------------------------------------
# Real (degradation-of-good) generators
# ---------------------------------------------------------------------------

REAL_TYPES: List[str] = [
    "silence_gaps", "noise_overlay", "tempo_warp", "pitch_scramble", "time_reverse"
]


def _degrade_silence_gaps(y: np.ndarray, sr: int, rng) -> np.ndarray:
    """Zero out ~50% in 500ms chunks."""
    chunk = int(sr * 0.5)
    y = y.copy()
    n_chunks = max(1, int(len(y) / chunk * 0.5))
    for _ in range(n_chunks):
        start = int(rng.integers(0, max(1, len(y) - chunk)))
        y[start:start + chunk] = 0.0
    return y


def _degrade_noise_overlay(y: np.ndarray, sr: int, rng) -> np.ndarray:
    """White noise at ~3 dB SNR."""
    rms = float(np.sqrt(np.mean(y ** 2) + 1e-9))
    noise = rng.standard_normal(len(y)).astype(np.float32) * rms * 0.7
    return np.clip(y + noise, -1.0, 1.0)


def _degrade_tempo_warp(y: np.ndarray, sr: int, rng) -> np.ndarray:
    """4× speed-up — destroys rhythm perception."""
    import librosa
    return librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=sr // 4)


def _degrade_pitch_scramble(y: np.ndarray, sr: int, rng) -> np.ndarray:
    """1-second chunks each shifted by a random ±6 semitone amount."""
    import librosa
    chunk = int(sr * 1.0)
    out = np.zeros(len(y), dtype=np.float32)
    for start in range(0, len(y), chunk):
        seg = y[start:start + chunk].astype(np.float32)
        n_steps = float(rng.uniform(-6, 6))
        try:
            seg = librosa.effects.pitch_shift(seg, sr=sr, n_steps=n_steps)
        except Exception:
            pass
        end = min(start + len(seg), len(out))
        out[start:end] = seg[: end - start]
    return out


def _degrade_time_reverse(y: np.ndarray, sr: int, rng) -> np.ndarray:
    """Reverse time axis — preserves spectrum, destroys temporal structure."""
    return y[::-1].copy()


_REAL_DEGRADATIONS = {
    "silence_gaps": _degrade_silence_gaps,
    "noise_overlay": _degrade_noise_overlay,
    "tempo_warp": _degrade_tempo_warp,
    "pitch_scramble": _degrade_pitch_scramble,
    "time_reverse": _degrade_time_reverse,
}


# ---------------------------------------------------------------------------
# Good-sample generation
# ---------------------------------------------------------------------------


def _generate_good_wav(seed: int, idx: int, dataset_root: str) -> Optional[str]:
    from musicgen import Config, generate
    r = generate(Config(
        global_seed=seed,
        sample_index=idx,
        dataset_root=dataset_root,
        min_musicality_score=0.0,
        max_attempts=1,
    ))
    return r.mix_path if r.status == "ok" else None


# ---------------------------------------------------------------------------
# Main experiment entry point
# ---------------------------------------------------------------------------


def run_construct_validity_test(
    n_good: int = 50,
    n_bad: int = 20,
    bad_types: str = "both",
    synthetic_signals: Optional[List[str]] = None,
    real_signals: Optional[List[str]] = None,
    base_seed: int = 42,
) -> ConstructValidityResult:
    """Run E2 construct validity experiment.

    bad_types: "synthetic" | "real" | "both"
    synthetic_signals: subset of SYNTHETIC_TYPES (default: all)
    real_signals: subset of REAL_TYPES (default: all)
    """
    if synthetic_signals is None:
        synthetic_signals = list(SYNTHETIC_TYPES)
    if real_signals is None:
        real_signals = list(REAL_TYPES)

    np_rng = np.random.default_rng(base_seed)

    with tempfile.TemporaryDirectory(prefix="musicgen-e2-") as tmp:
        tmp_path = Path(tmp)

        # --- Good samples ---
        good_wav_paths: List[str] = []
        print(f"  Generating {n_good} good samples …", flush=True)
        for i in range(n_good):
            p = _generate_good_wav(base_seed + i, i, str(tmp_path / "good" / f"s{i}"))
            if p:
                good_wav_paths.append(p)
            if (i + 1) % 10 == 0 or (i + 1) == n_good:
                print(f"    [{i+1}/{n_good}]  ok={len(good_wav_paths)}", flush=True)

        good_scores = [s for s in (_score_wav(p) for p in good_wav_paths) if s is not None]
        if not good_scores:
            raise RuntimeError("No good samples scored successfully.")
        print(
            f"  Good: n={len(good_scores)}  mean={np.mean(good_scores):.3f}"
            f"  σ={np.std(good_scores):.3f}",
            flush=True,
        )

        pathology_results: List[PathologyResult] = []

        # --- Synthetic bad ---
        if bad_types in ("synthetic", "both"):
            for sig in synthetic_signals:
                gen_fn = _SYNTH_GENERATORS[sig]
                bad_dir = tmp_path / "synth" / sig
                bad_dir.mkdir(parents=True, exist_ok=True)
                scores: List[float] = []
                for j in range(n_bad):
                    wav = str(bad_dir / f"{sig}_{j}.wav")
                    gen_fn(wav, rng=np_rng)
                    s = _score_wav(wav)
                    if s is not None:
                        scores.append(s)
                auc = _auroc(good_scores, scores)
                d = _cohens_d(good_scores, scores)
                pathology_results.append(PathologyResult(
                    name=f"synth/{sig}",
                    scores=scores,
                    mean=float(np.mean(scores)) if scores else 0.0,
                    std=float(np.std(scores)) if scores else 0.0,
                    auroc=auc,
                    separation=d,
                ))
                print(
                    f"  synth/{sig:<14} n={len(scores):3d}  mean={np.mean(scores):.3f}"
                    f"  AUROC={auc:.3f}  d={d:.2f}",
                    flush=True,
                )

        # --- Real (degradation) bad ---
        if bad_types in ("real", "both"):
            import librosa
            import soundfile as sf

            good_audio: List[Tuple[np.ndarray, int]] = []
            for p in good_wav_paths[:n_bad]:
                y, sr = librosa.load(p, sr=None, mono=True)
                good_audio.append((y, int(sr)))

            real_rng = np.random.default_rng(base_seed + 9999)

            for sig in real_signals:
                deg_fn = _REAL_DEGRADATIONS[sig]
                bad_dir = tmp_path / "real" / sig
                bad_dir.mkdir(parents=True, exist_ok=True)
                scores = []
                for j, (y, sr) in enumerate(good_audio):
                    try:
                        y_bad = deg_fn(y, sr, real_rng)
                        wav = str(bad_dir / f"{sig}_{j}.wav")
                        sf.write(wav, y_bad.astype(np.float32), sr)
                        s = _score_wav(wav)
                        if s is not None:
                            scores.append(s)
                    except Exception as exc:
                        logger.warning("degrade %s/%d: %s", sig, j, exc)
                auc = _auroc(good_scores, scores)
                d = _cohens_d(good_scores, scores)
                pathology_results.append(PathologyResult(
                    name=f"real/{sig}",
                    scores=scores,
                    mean=float(np.mean(scores)) if scores else 0.0,
                    std=float(np.std(scores)) if scores else 0.0,
                    auroc=auc,
                    separation=d,
                ))
                print(
                    f"  real/{sig:<14} n={len(scores):3d}  mean={np.mean(scores):.3f}"
                    f"  AUROC={auc:.3f}  d={d:.2f}",
                    flush=True,
                )

        all_bad = [s for pr in pathology_results for s in pr.scores]
        overall_auroc = _auroc(good_scores, all_bad)
        overall_d = _cohens_d(good_scores, all_bad)

        return ConstructValidityResult(
            n_good=len(good_scores),
            good_scores=good_scores,
            good_mean=float(np.mean(good_scores)),
            good_std=float(np.std(good_scores)),
            pathologies=pathology_results,
            overall_auroc=overall_auroc,
            overall_separation=overall_d,
        )
