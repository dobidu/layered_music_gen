"""Construct validity experiment (E2).

Validates that the musicality gate discriminates between well-formed music
and known-pathological signals. Two pathology modes:

  synthetic  — programmatically generated bad audio (silence, white noise,
               hard-clipped sine, DC offset, pure monotone sine)
  real       — controlled degradations of good musicgen outputs (silence gaps,
               noise overlay, 4× tempo warp, per-chunk pitch scramble,
               time reversal)
  both       — union of the above

Primary metric: AUROC (gate score as binary classifier, good=1, bad=0)
with bootstrap 95% CI. Secondary: Cohen's d. When bad-set std=0, reports
rejection_rate (fraction scoring below threshold) instead of d.

Known limitation:
  real/silence_gaps scores at chance level (~0.56 AUROC). The gate measures
  per-frame audio quality; if sounding portions are musical the gate passes
  the sample regardless of silence distribution. Document as out-of-scope
  for a temporal-completeness check.
"""
from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_SR = 44100
_DURATION = 5.0
_REJECTION_THRESHOLD = 0.5


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
    ci_low: float
    ci_high: float
    separation: float           # Cohen's d (NaN when bad std = 0)
    rejection_rate: float       # fraction of bad scores < _REJECTION_THRESHOLD
    excluded: bool              # excluded from criterion AUROC
    exclusion_reason: Optional[str]  # key into EXCLUSION_NOTES, or None


@dataclass
class ConstructValidityResult:
    n_good: int
    good_scores: List[float]
    good_mean: float
    good_std: float
    pathologies: List[PathologyResult]
    overall_auroc: float                    # over all pathologies
    overall_auroc_ci: Tuple[float, float]
    overall_separation: float
    criterion_auroc: float                  # excluding EXCLUDED_FROM_CRITERION
    criterion_auroc_ci: Tuple[float, float]
    criterion_n_pathologies: int            # number of pathologies in criterion


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def _auroc_arr(pos: np.ndarray, neg: np.ndarray) -> float:
    """Vectorised Mann-Whitney AUROC."""
    p = pos[:, np.newaxis]
    n = neg[np.newaxis, :]
    count = float(np.sum(p > n)) + 0.5 * float(np.sum(p == n))
    return count / (len(pos) * len(neg))


def _auroc(pos: List[float], neg: List[float]) -> float:
    if not pos or not neg:
        return 0.5
    return _auroc_arr(np.array(pos), np.array(neg))


def _bootstrap_auroc_ci(
    pos: List[float],
    neg: List[float],
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float]:
    if len(pos) < 2 or len(neg) < 2:
        return (0.0, 1.0)
    rng = np.random.default_rng(seed)
    pos_arr = np.array(pos)
    neg_arr = np.array(neg)
    boot = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        p_b = rng.choice(pos_arr, size=len(pos_arr), replace=True)
        n_b = rng.choice(neg_arr, size=len(neg_arr), replace=True)
        boot[i] = _auroc_arr(p_b, n_b)
    return (
        float(np.percentile(boot, 100 * alpha / 2)),
        float(np.percentile(boot, 100 * (1 - alpha / 2))),
    )


def _cohens_d(good: List[float], bad: List[float]) -> float:
    if not good or not bad or float(np.std(bad)) == 0.0:
        return float("nan")
    mg, mb = float(np.mean(good)), float(np.mean(bad))
    sg, sb = float(np.std(good)), float(np.std(bad))
    ng, nb = len(good), len(bad)
    pooled = np.sqrt(((ng - 1) * sg ** 2 + (nb - 1) * sb ** 2) / (ng + nb - 2 + 1e-9))
    return float((mg - mb) / (pooled + 1e-9))


def _rejection_rate(scores: List[float], threshold: float = _REJECTION_THRESHOLD) -> float:
    if not scores:
        return 0.0
    return float(np.mean([s < threshold for s in scores]))


# ---------------------------------------------------------------------------
# Audio I/O
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
    return _write_wav(_rng.standard_normal(int(duration * sr)).astype(np.float32) * 0.5, sr, path)


def _synth_clip(path: str, sr: int = _SR, duration: float = _DURATION, rng=None) -> str:
    t = np.linspace(0, duration, int(duration * sr))
    return _write_wav(np.clip(np.sin(2 * np.pi * 440 * t) * 10.0, -0.1, 0.1), sr, path)


def _synth_dc(path: str, sr: int = _SR, duration: float = _DURATION, rng=None) -> str:
    t = np.linspace(0, duration, int(duration * sr))
    return _write_wav(0.5 + 0.05 * np.sin(2 * np.pi * 440 * t), sr, path)


def _synth_monotone(path: str, sr: int = _SR, duration: float = _DURATION, rng=None) -> str:
    t = np.linspace(0, duration, int(duration * sr))
    return _write_wav(0.5 * np.sin(2 * np.pi * 440 * t), sr, path)


_SYNTH_GENERATORS = {
    "silence":  _synth_silence,
    "noise":    _synth_noise,
    "clip":     _synth_clip,
    "dc":       _synth_dc,
    "monotone": _synth_monotone,
}


# ---------------------------------------------------------------------------
# Real (degradation-of-good) generators
# ---------------------------------------------------------------------------

REAL_TYPES: List[str] = [
    "silence_gaps", "noise_overlay", "tempo_warp", "pitch_scramble", "time_reverse"
]

# Types explicitly excluded from the criterion AUROC, keyed by short name.
# Value is a reason tag looked up in EXCLUSION_NOTES.
EXCLUDED_FROM_CRITERION: Dict[str, str] = {
    "silence_gaps": "temporal-completeness",
    "tempo_warp":   "pitch-tempo-confound",
}

EXCLUSION_NOTES: Dict[str, str] = {
    "temporal-completeness": (
        "Gate measures per-frame audio quality; silence distribution "
        "is out of scope for a temporal-completeness check."
    ),
    "pitch-tempo-confound": (
        "4x resample conflates pitch shift (+/-2 oct.) with tempo distortion. "
        "The gate partially detects the resulting spectral change (AUROC~0.74) "
        "but not reliably. A dedicated temporal-coherence check is out of scope."
    ),
}


def _degrade_silence_gaps(y: np.ndarray, sr: int, rng) -> np.ndarray:
    """Zero out ~50% of audio in 500 ms chunks."""
    chunk = int(sr * 0.5)
    y = y.copy()
    for _ in range(max(1, int(len(y) / chunk * 0.5))):
        start = int(rng.integers(0, max(1, len(y) - chunk)))
        y[start:start + chunk] = 0.0
    return y


def _degrade_noise_overlay(y: np.ndarray, sr: int, rng) -> np.ndarray:
    """Add white noise at ~3 dB SNR."""
    rms = float(np.sqrt(np.mean(y ** 2) + 1e-9))
    noise = rng.standard_normal(len(y)).astype(np.float32) * rms * 0.7
    return np.clip(y + noise, -1.0, 1.0)


def _degrade_tempo_warp(y: np.ndarray, sr: int, rng) -> np.ndarray:
    """4× speed-up — destroys rhythm perception."""
    import librosa
    return librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=sr // 4)


def _degrade_pitch_scramble(y: np.ndarray, sr: int, rng) -> np.ndarray:
    """1-second chunks each shifted by a random ±6-semitone amount."""
    import librosa
    chunk = int(sr * 1.0)
    out = np.zeros(len(y), dtype=np.float32)
    for start in range(0, len(y), chunk):
        seg = y[start:start + chunk].astype(np.float32)
        try:
            seg = librosa.effects.pitch_shift(seg, sr=sr, n_steps=float(rng.uniform(-6, 6)))
        except Exception:
            pass
        end = min(start + len(seg), len(out))
        out[start:end] = seg[:end - start]
    return out


def _degrade_time_reverse(y: np.ndarray, sr: int, rng) -> np.ndarray:
    """Reverse time axis — preserves spectrum, destroys temporal structure."""
    return y[::-1].copy()


_REAL_DEGRADATIONS = {
    "silence_gaps":    _degrade_silence_gaps,
    "noise_overlay":   _degrade_noise_overlay,
    "tempo_warp":      _degrade_tempo_warp,
    "pitch_scramble":  _degrade_pitch_scramble,
    "time_reverse":    _degrade_time_reverse,
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
# Main experiment
# ---------------------------------------------------------------------------


def run_construct_validity_test(
    n_good: int = 50,
    n_bad: int = 20,
    bad_types: str = "both",
    synthetic_signals: Optional[List[str]] = None,
    real_signals: Optional[List[str]] = None,
    base_seed: int = 42,
    bootstrap_n: int = 2000,
) -> ConstructValidityResult:
    """Run E2 construct validity.

    bad_types: "synthetic" | "real" | "both"
    n_bad: bad samples per pathology type. For real degradations, if n_bad >
           n_good, good samples are cycled (each reused with a different rng seed).
    bootstrap_n: bootstrap iterations for 95% CI (set 0 to skip).
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

        def _make_pathology(name: str, scores: List[float]) -> PathologyResult:
            short = name.split("/")[-1]
            excl_reason = EXCLUDED_FROM_CRITERION.get(short)
            auc = _auroc(good_scores, scores)
            ci = _bootstrap_auroc_ci(good_scores, scores, n_bootstrap=bootstrap_n) \
                if bootstrap_n > 0 and scores else (0.0, 1.0)
            d = _cohens_d(good_scores, scores)
            rr = _rejection_rate(scores)
            excluded = excl_reason is not None
            return PathologyResult(
                name=name, scores=scores,
                mean=float(np.mean(scores)) if scores else 0.0,
                std=float(np.std(scores)) if scores else 0.0,
                auroc=auc, ci_low=ci[0], ci_high=ci[1],
                separation=d, rejection_rate=rr,
                excluded=excluded, exclusion_reason=excl_reason,
            )

        pathology_results: List[PathologyResult] = []

        # --- Synthetic ---
        if bad_types in ("synthetic", "both"):
            for sig in synthetic_signals:
                bad_dir = tmp_path / "synth" / sig
                bad_dir.mkdir(parents=True, exist_ok=True)
                scores: List[float] = []
                for j in range(n_bad):
                    wav = str(bad_dir / f"{sig}_{j}.wav")
                    _SYNTH_GENERATORS[sig](wav, rng=np_rng)
                    s = _score_wav(wav)
                    if s is not None:
                        scores.append(s)
                pr = _make_pathology(f"synth/{sig}", scores)
                pathology_results.append(pr)
                _log_pathology(pr)

        # --- Real (degradation) ---
        if bad_types in ("real", "both"):
            import librosa
            import soundfile as sf

            # Load good audio; cycle if n_bad > n_good
            good_audio: List[Tuple[np.ndarray, int]] = []
            for p in good_wav_paths:
                y, sr = librosa.load(p, sr=None, mono=True)
                good_audio.append((y, int(sr)))

            real_rng = np.random.default_rng(base_seed + 9999)

            for sig in real_signals:
                deg_fn = _REAL_DEGRADATIONS[sig]
                bad_dir = tmp_path / "real" / sig
                bad_dir.mkdir(parents=True, exist_ok=True)
                scores = []
                for j in range(n_bad):
                    y, sr = good_audio[j % len(good_audio)]
                    try:
                        y_bad = deg_fn(y, sr, real_rng)
                        wav = str(bad_dir / f"{sig}_{j}.wav")
                        sf.write(wav, y_bad.astype(np.float32), sr)
                        s = _score_wav(wav)
                        if s is not None:
                            scores.append(s)
                    except Exception as exc:
                        logger.warning("degrade %s/%d: %s", sig, j, exc)
                pr = _make_pathology(f"real/{sig}", scores)
                pathology_results.append(pr)
                _log_pathology(pr)

        all_bad = [s for pr in pathology_results for s in pr.scores]
        overall_auroc = _auroc(good_scores, all_bad)
        overall_ci = _bootstrap_auroc_ci(good_scores, all_bad, n_bootstrap=bootstrap_n) \
            if bootstrap_n > 0 else (0.0, 1.0)
        overall_d = _cohens_d(good_scores, all_bad)

        criterion_prs = [pr for pr in pathology_results if not pr.excluded]
        criterion_bad = [s for pr in criterion_prs for s in pr.scores]
        criterion_auroc = _auroc(good_scores, criterion_bad) if criterion_bad else 0.5
        criterion_ci = _bootstrap_auroc_ci(good_scores, criterion_bad, n_bootstrap=bootstrap_n) \
            if bootstrap_n > 0 and criterion_bad else (0.0, 1.0)

        return ConstructValidityResult(
            n_good=len(good_scores),
            good_scores=good_scores,
            good_mean=float(np.mean(good_scores)),
            good_std=float(np.std(good_scores)),
            pathologies=pathology_results,
            overall_auroc=overall_auroc,
            overall_auroc_ci=overall_ci,
            overall_separation=overall_d,
            criterion_auroc=criterion_auroc,
            criterion_auroc_ci=criterion_ci,
            criterion_n_pathologies=len(criterion_prs),
        )


def _log_pathology(pr: PathologyResult) -> None:
    d_str = f"d={pr.separation:.2f}" if not (pr.separation != pr.separation) else \
        f"rej={pr.rejection_rate:.0%}"
    flag = f"[excl: {pr.exclusion_reason}]" if pr.excluded else ""
    print(
        f"  {pr.name:<22} n={len(pr.scores):4d}  mean={pr.mean:.3f}"
        f"  AUROC={pr.auroc:.3f} [{pr.ci_low:.3f}–{pr.ci_high:.3f}]"
        f"  {d_str}  {flag}",
        flush=True,
    )
