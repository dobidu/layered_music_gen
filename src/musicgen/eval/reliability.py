"""E1 — Reliability experiment (§2 of validation plan).

Three variability types:

  Type 1 — Determinism: same WAV scored N times. Expected σ < 1e-6.
  Type 2 — Render invariance: same MIDI, re-rendered with N soundfont seeds.
            Expected σ < 0.03 per sample.
  Type 3 — Seed distribution: same Config, N seeds per genre.
            No pass/fail — characterises the score distribution.

Public API
----------
  run_determinism_test(n_samples, n_calls, seed, genres, dataset_root) -> DeterminismResult
  run_render_invariance_test(n_samples, n_renders, seed, genres, dataset_root) -> RenderInvarianceResult
  run_seed_distribution_test(n_seeds, genres, base_seed, dataset_root) -> SeedDistributionResult
"""
from __future__ import annotations

import dataclasses
import logging
import random
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_ALL_GENRES = ["blues", "classical", "electronic", "hip-hop", "jazz", "latin", "pop", "reggae"]

# Acceptance criteria from §2.3
_DETERMINISM_SIGMA_MAX = 1e-6
_RENDER_SIGMA_MAX = 0.03


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class DeterminismResult:
    """Type 1: repeated scoring of the same WAV file."""
    n_samples: int
    n_calls: int
    max_sigma: float            # worst σ across all samples
    all_sigma: List[float]      # σ per sample
    passed: bool                # all σ < _DETERMINISM_SIGMA_MAX


@dataclasses.dataclass
class RenderVarianceEntry:
    sample_index: int
    genre: str
    sigma: float
    scores: List[float]


@dataclasses.dataclass
class RenderInvarianceResult:
    """Type 2: same MIDI, N renders with different soundfont seeds."""
    n_samples: int
    n_renders: int
    entries: List[RenderVarianceEntry]
    sigma_by_genre: Dict[str, float]    # mean σ per genre
    max_sigma: float
    passed: bool                        # all genre-mean σ < _RENDER_SIGMA_MAX


@dataclasses.dataclass
class SeedDistributionEntry:
    genre: str
    n_seeds: int
    mean: float
    median: float
    sigma: float
    iqr: float
    p10: float
    p90: float
    scores: List[float]


@dataclasses.dataclass
class SeedDistributionResult:
    """Type 3: distribution of scores across seeds per genre."""
    n_seeds: int
    genres: List[str]
    entries: List[SeedDistributionEntry]


# ---------------------------------------------------------------------------
# Type 1 — Determinism
# ---------------------------------------------------------------------------

def run_determinism_test(
    n_samples: int = 50,
    n_calls: int = 20,
    seed: int = 1,
    genres: Optional[List[str]] = None,
    dataset_root: Optional[str] = None,
) -> DeterminismResult:
    """Score the same WAV n_calls times; σ must be < 1e-6."""
    from musicgen import Config, generate
    from musicgen.musicality import get_musicality_score

    genres = genres or _ALL_GENRES
    all_sigma: List[float] = []

    with tempfile.TemporaryDirectory(prefix="musicgen-e1-det-") as tmp:
        root = dataset_root or tmp
        for i in range(n_samples):
            genre = genres[i % len(genres)]
            r = generate(Config(
                global_seed=seed, sample_index=i,
                dataset_root=root, genre=[genre], output_mode="mix-only",
            ))
            if r.status != "ok":
                logger.warning("sample %d failed (status=%s), skipping", i, r.status)
                continue
            mix_path = str(Path(r.sample_dir) / "mix.wav")
            scores = [get_musicality_score(mix_path)[0] for _ in range(n_calls)]
            sigma = float(np.std(scores))
            all_sigma.append(sigma)
            logger.debug("det sample %d genre=%s σ=%.2e", i, genre, sigma)

    max_sigma = max(all_sigma) if all_sigma else float("nan")
    passed = all(s < _DETERMINISM_SIGMA_MAX for s in all_sigma)
    return DeterminismResult(
        n_samples=len(all_sigma), n_calls=n_calls,
        max_sigma=max_sigma, all_sigma=all_sigma, passed=passed,
    )


# ---------------------------------------------------------------------------
# Type 2 — Render invariance
# ---------------------------------------------------------------------------

def run_render_invariance_test(
    n_samples: int = 50,
    n_renders: int = 20,
    seed: int = 1,
    genres: Optional[List[str]] = None,
    dataset_root: Optional[str] = None,
) -> RenderInvarianceResult:
    """Re-render the same MIDI with N soundfont seeds; measure score σ."""
    from musicgen import Config, generate
    from musicgen.musicality import get_musicality_score
    from musicgen.renderer import pick_soundfonts, render_stems
    import config as cfg_module

    genres = genres or _ALL_GENRES
    entries: List[RenderVarianceEntry] = []
    cfg = cfg_module.Config()

    with tempfile.TemporaryDirectory(prefix="musicgen-e1-rinv-") as tmp:
        root = dataset_root or tmp
        for i in range(n_samples):
            genre = genres[i % len(genres)]
            r = generate(Config(
                global_seed=seed, sample_index=i,
                dataset_root=root, genre=[genre], output_mode="midi-only",
            ))
            if r.status != "ok":
                logger.warning("sample %d failed (status=%s), skipping", i, r.status)
                continue

            midi_paths = r.midi_paths  # Dict[layer, path]
            scores: List[float] = []
            rng_base = random.Random(seed + i + 10000)

            for render_idx in range(n_renders):
                render_rng = random.Random(rng_base.randint(0, 2**32))
                render_out = Path(tmp) / f"rinv_{i}_{render_idx}"
                render_out.mkdir(parents=True, exist_ok=True)
                try:
                    sf = pick_soundfonts(cfg=cfg, rng=render_rng)
                    result = render_stems(midi_paths, sf, str(render_out), cfg=cfg)
                    # mix: simple sum of stems via pydub to match normal mix path
                    from pydub import AudioSegment
                    mix = AudioSegment.from_wav(result.stem_paths["beat"])
                    for layer in ("melody", "harmony", "bassline"):
                        mix = mix.overlay(AudioSegment.from_wav(result.stem_paths[layer]))
                    mix_path = str(render_out / "mix.wav")
                    mix.export(mix_path, format="wav")
                    sc, _ = get_musicality_score(mix_path)
                    scores.append(sc)
                except Exception as exc:
                    logger.warning("render %d/%d failed: %s", render_idx, n_renders, exc)

            if len(scores) < 2:
                continue
            sigma = float(np.std(scores))
            entries.append(RenderVarianceEntry(
                sample_index=i, genre=genre, sigma=sigma, scores=scores,
            ))
            logger.debug("rinv sample %d genre=%s σ=%.4f", i, genre, sigma)

    sigma_by_genre: Dict[str, float] = {}
    for g in genres:
        g_entries = [e.sigma for e in entries if e.genre == g]
        if g_entries:
            sigma_by_genre[g] = float(np.mean(g_entries))

    max_sigma = max((e.sigma for e in entries), default=float("nan"))
    passed = all(v < _RENDER_SIGMA_MAX for v in sigma_by_genre.values())
    return RenderInvarianceResult(
        n_samples=len(entries), n_renders=n_renders,
        entries=entries, sigma_by_genre=sigma_by_genre,
        max_sigma=max_sigma, passed=passed,
    )


# ---------------------------------------------------------------------------
# Type 3 — Seed distribution
# ---------------------------------------------------------------------------

def run_seed_distribution_test(
    n_seeds: int = 400,
    genres: Optional[List[str]] = None,
    base_seed: int = 1,
    dataset_root: Optional[str] = None,
) -> SeedDistributionResult:
    """Generate n_seeds samples per genre; characterise score distribution."""
    from musicgen import Config, generate
    from musicgen.musicality import get_musicality_score

    genres = genres or _ALL_GENRES
    entries: List[SeedDistributionEntry] = []

    with tempfile.TemporaryDirectory(prefix="musicgen-e1-sdist-") as tmp:
        root = dataset_root or tmp
        for genre in genres:
            scores: List[float] = []
            for s in range(n_seeds):
                r = generate(Config(
                    global_seed=base_seed + s,
                    sample_index=s,
                    dataset_root=str(Path(root) / genre),
                    genre=[genre],
                    output_mode="mix-only",
                ))
                if r.status != "ok":
                    continue
                mix_path = str(Path(r.sample_dir) / "mix.wav")
                sc, _ = get_musicality_score(mix_path)
                scores.append(sc)
                logger.debug("sdist genre=%s seed=%d score=%.4f", genre, base_seed + s, sc)

            if not scores:
                continue
            arr = np.array(scores)
            q25, q75 = float(np.percentile(arr, 25)), float(np.percentile(arr, 75))
            entries.append(SeedDistributionEntry(
                genre=genre,
                n_seeds=len(scores),
                mean=float(np.mean(arr)),
                median=float(np.median(arr)),
                sigma=float(np.std(arr)),
                iqr=q75 - q25,
                p10=float(np.percentile(arr, 10)),
                p90=float(np.percentile(arr, 90)),
                scores=scores,
            ))

    return SeedDistributionResult(n_seeds=n_seeds, genres=genres, entries=entries)
