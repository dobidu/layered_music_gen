"""M5 Part 1 — Sample library builder: annotate a WAV directory → SampleManager JSON.

Usage (via CLI):
    musicgen samples build --dir /my/samples --output library.json [--musicality]

Usage (as library):
    from musicgen.sample_builder import build_library
    count = build_library(wav_dir="/my/samples", output="library.json", compute_musicality=True)

Category inferred from filename patterns when not overridden. Musicality scoring
requires musicgen[samples] (musicality package) to be installed.
"""
from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Filename keyword → SampleManager category string (first match wins)
_CATEGORY_PATTERNS: List[tuple[str, str]] = [
    # beat patterns
    ("beat", "beat"), ("kick", "beat"), ("hat", "beat"), ("snare", "beat"),
    ("drum", "beat"), ("perc", "beat"), ("clap", "beat"), ("hh", "beat"),
    ("hihat", "beat"), ("cyma", "beat"), ("ride", "beat"),
    # bass patterns
    ("bass", "bass"), ("sub", "bass"),
    # harmony patterns (before melody so "chord" beats "lick")
    ("pad", "harmony"), ("chord", "harmony"), ("harm", "harmony"),
    ("atmo", "harmony"), ("ambient", "harmony"), ("strings", "harmony"),
    ("vox", "harmony"), ("choir", "harmony"), ("keys", "harmony"),
    ("piano", "harmony"), ("organ", "harmony"),
    # melody patterns
    ("lead", "melody"), ("melody", "melody"), ("lick", "melody"),
    ("riff", "melody"), ("synth", "melody"), ("arp", "melody"),
    ("melo", "melody"), ("hook", "melody"),
]

_SUPPORTED_EXTENSIONS = {".wav", ".aif", ".aiff", ".flac", ".ogg", ".mp3"}


def _infer_category(filename: str) -> str:
    """Return SampleManager category string inferred from filename keywords."""
    name_lower = os.path.splitext(filename)[0].lower()
    for keyword, cat in _CATEGORY_PATTERNS:
        if keyword in name_lower:
            return cat
    return "melody"  # safe default


def _collect_wav_files(wav_dir: str, recursive: bool) -> List[str]:
    paths: List[str] = []
    if recursive:
        for root, _, files in os.walk(wav_dir):
            for f in sorted(files):
                if os.path.splitext(f)[1].lower() in _SUPPORTED_EXTENSIONS:
                    paths.append(os.path.join(root, f))
    else:
        for f in sorted(os.listdir(wav_dir)):
            if os.path.splitext(f)[1].lower() in _SUPPORTED_EXTENSIONS:
                paths.append(os.path.join(wav_dir, f))
    return paths


def build_library(
    wav_dir: str,
    output: str,
    category_override: Optional[str] = None,
    genre: Optional[List[str]] = None,
    mood: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
    compute_musicality: bool = False,
    recursive: bool = False,
    skip_errors: bool = True,
) -> int:
    """Annotate a directory of WAV files into a SampleManager JSON database.

    Args:
        wav_dir: Directory containing WAV files (and optionally subdirs when
            recursive=True).
        output: Destination path for the SampleManager JSON database.
        category_override: Force all samples to this category (beat|bass|melody|harmony).
            When None, category is inferred from each filename.
        genre: Genre tags applied to every sample.
        mood: Mood tags applied to every sample.
        tags: Additional tags applied to every sample.
        compute_musicality: If True, score each sample with musicality.explain().
            Requires musicgen[samples] (musicality package).
        recursive: If True, walk subdirectories.
        skip_errors: If True, log warnings and continue on per-file failures.

    Returns:
        Number of samples successfully indexed.

    Raises:
        ImportError: if audio_sample_manager is not installed.
        FileNotFoundError: if wav_dir does not exist.
    """
    try:
        from sample_manager import SampleManager  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "audio_sample_manager not installed — cannot build library. "
            "Install with: pip install 'musicgen[samples]'"
        ) from exc

    if not os.path.isdir(wav_dir):
        raise FileNotFoundError(f"WAV directory not found: {wav_dir}")

    wav_files = _collect_wav_files(wav_dir, recursive)
    if not wav_files:
        logger.warning("No supported audio files found in %s", wav_dir)
        return 0

    logger.info("Found %d audio files in %s", len(wav_files), wav_dir)

    # Lazy-load musicality only if requested
    _explain = None
    if compute_musicality:
        try:
            from musicality import explain as _explain  # type: ignore[import]
            logger.info("Musicality scoring enabled")
        except ImportError:
            logger.warning("musicality package not installed — skipping musicality scores")

    sm = SampleManager(output)
    indexed = 0

    for wav_path in wav_files:
        fname = os.path.basename(wav_path)
        logger.info("Indexing %s", fname)

        try:
            meta = sm.add_sample(wav_path, analyze=True, save=False)
        except Exception as exc:
            if skip_errors:
                logger.warning("Failed to analyze %s: %s", fname, exc)
                continue
            raise

        # Apply overrides
        cat = category_override or _infer_category(fname)
        meta.category = cat
        if genre:
            meta.genre = list(genre)
        if mood:
            meta.mood = list(mood)
        if tags:
            meta.tags = list(tags)

        # Musicality scoring
        if _explain is not None:
            try:
                report = _explain(wav_path)
                meta.musicality_score = report["score"]
                meta.musicality_components = {
                    k: v["score"] if isinstance(v, dict) else float(v)
                    for k, v in report.get("components", {}).items()
                }
                logger.debug("%s musicality=%.4f", fname, meta.musicality_score)
            except Exception as exc:
                logger.warning("Musicality scoring failed for %s: %s", fname, exc)

        indexed += 1

    sm.save_samples()
    logger.info("Library written to %s (%d samples)", output, indexed)
    return indexed
