"""Calibration module — FluidSynth pre-roll + musicality harness (v0.3 Phase 3c).

Pre-roll (R-P9, D-50..D-54):
  Measures and caches the FluidSynth silence offset before first note.

Musicality calibration harness (v0.3 Phase 3c):
  Generates reference-good and adversarial MIDI sets, scores them with
  check_midi_quality, and derives an empirical min_musicality_score threshold.

Public surface:
  load_preroll(project_root) -> float
  measure_preroll(project_root) -> float
  measure_and_save_preroll(project_root) -> float
  save_preroll(project_root, offset_s) -> None

  CalibrationResult              — dataclass
  run_midi_calibration(...)      -> CalibrationResult
  suggest_threshold(good, bad)   -> float
  save_calibration(result, path) -> None
  load_calibration(path)         -> CalibrationResult
"""
from __future__ import annotations

import dataclasses
import json
import logging
import os
import random
import struct
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Optional

import mido
import numpy as np
import scipy.io.wavfile as wf

from musicgen.renderer import FLUIDSYNTH_VERSION

logger = logging.getLogger(__name__)

PREROLL_CACHE_DIR = ".musicgen"
PREROLL_CACHE_FILE = "fluidsynth_preroll.json"
SILENCE_THRESHOLD = 1e-4
MAX_REASONABLE_OFFSET_S = 1.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_preroll(project_root: str) -> float:
    """Return cached pre-roll offset (seconds), re-measuring if stale."""
    cache = _cache_path(project_root)
    if os.path.isfile(cache):
        try:
            data = json.loads(open(cache).read())
            cached_version = data.get("fluidsynth_version", "")
            if cached_version == FLUIDSYNTH_VERSION:
                offset = float(data["offset_s"])
                logger.debug(
                    "Pre-roll cache hit: %.4f s (FluidSynth %s)",
                    offset, FLUIDSYNTH_VERSION,
                )
                return offset
            logger.info(
                "Pre-roll cache version mismatch (%r vs %r) — re-measuring",
                cached_version, FLUIDSYNTH_VERSION,
            )
        except Exception as exc:
            logger.warning("Pre-roll cache read failed: %s — re-measuring", exc)

    # Cache absent, corrupt, or version-stale — measure and save.
    return measure_and_save_preroll(project_root)


def measure_and_save_preroll(project_root: str) -> float:
    """Measure pre-roll, persist to cache, return offset in seconds."""
    offset_s = measure_preroll(project_root)
    save_preroll(project_root, offset_s)
    return offset_s


def measure_preroll(project_root: str) -> float:
    """Synthesize a 1-note MIDI and detect first non-silent sample.

    Returns 0.0 when:
    - FluidSynth is not installed (FLUIDSYNTH_VERSION == "unknown")
    - No soundfont is available in the sf/ directories
    - The synthesized WAV is entirely silent
    - Any synthesis error occurs
    """
    if FLUIDSYNTH_VERSION == "unknown":
        logger.debug("FluidSynth absent — pre-roll measurement skipped, offset=0.0")
        return 0.0

    sf_path = _find_any_soundfont(project_root)
    if sf_path is None:
        logger.debug("No soundfont found — pre-roll measurement skipped, offset=0.0")
        return 0.0

    try:
        with tempfile.TemporaryDirectory(prefix="musicgen-calib-") as tmp:
            midi_path = os.path.join(tmp, "calib.mid")
            wav_path = os.path.join(tmp, "calib.wav")
            _write_calibration_midi(midi_path)
            _render_calibration_midi(project_root, sf_path, midi_path, wav_path)
            offset_s = _detect_offset(wav_path)
            logger.info(
                "FluidSynth pre-roll measured: %.4f s (FluidSynth %s)",
                offset_s, FLUIDSYNTH_VERSION,
            )
            return offset_s
    except Exception as exc:
        logger.warning("Pre-roll measurement failed: %s — using 0.0", exc)
        return 0.0


def save_preroll(project_root: str, offset_s: float) -> None:
    """Write pre-roll cache JSON."""
    cache = _cache_path(project_root)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    data = {
        "offset_s": offset_s,
        "fluidsynth_version": FLUIDSYNTH_VERSION,
    }
    with open(cache, "w") as f:
        json.dump(data, f, indent=2)
    logger.debug("Pre-roll cache written: %s", cache)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _cache_path(project_root: str) -> str:
    return os.path.join(project_root, PREROLL_CACHE_DIR, PREROLL_CACHE_FILE)


def _write_calibration_midi(path: str) -> None:
    """Produce a deterministic 1-note MIDI: middle C, 120 BPM, 480 PPQ."""
    mf = mido.MidiFile(ticks_per_beat=480, type=1)

    meta = mido.MidiTrack()
    meta.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(120), time=0))
    meta.append(mido.MetaMessage("end_of_track", time=0))
    mf.tracks.append(meta)

    notes = mido.MidiTrack()
    notes.append(mido.Message("note_on", note=60, velocity=100, time=0))
    notes.append(mido.Message("note_off", note=60, velocity=0, time=480))
    notes.append(mido.MetaMessage("end_of_track", time=0))
    mf.tracks.append(notes)

    mf.save(path)


def _render_calibration_midi(
    project_root: str, sf_path: str, midi_path: str, wav_path: str,
) -> None:
    """Render the calibration MIDI to WAV via FluidSynth CLI."""
    import subprocess
    cmd = [
        "fluidsynth",
        "-ni",  # no interactive, no shell
        "-F", wav_path,
        sf_path,
        midi_path,
    ]
    result = subprocess.run(
        cmd, capture_output=True, timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"FluidSynth calibration failed (rc={result.returncode}): "
            f"{result.stderr.decode(errors='replace')[:500]}"
        )


def _detect_offset(wav_path: str) -> float:
    """Return time (seconds) of first non-silent sample in the WAV."""
    sr, data = wf.read(wav_path)
    if data.ndim > 1:
        data = data[:, 0]  # use first channel
    norm = data.astype(np.float32) / 32768.0
    nonsilent = np.where(np.abs(norm) > SILENCE_THRESHOLD)[0]
    if len(nonsilent) == 0:
        return 0.0
    offset_s = float(nonsilent[0]) / sr
    if offset_s > MAX_REASONABLE_OFFSET_S:
        logger.warning(
            "Pre-roll offset %.3f s exceeds MAX_REASONABLE_OFFSET_S=%.1f — clamping to 0.0",
            offset_s, MAX_REASONABLE_OFFSET_S,
        )
        return 0.0
    return offset_s


def _find_any_soundfont(project_root: str) -> Optional[str]:
    """Return path to any .sf2 file under project_root/sf/, or None."""
    sf_base = os.path.join(project_root, "sf")
    if not os.path.isdir(sf_base):
        return None
    for root, _dirs, files in os.walk(sf_base):
        for fname in files:
            if fname.endswith(".sf2"):
                return os.path.join(root, fname)
    return None


# ---------------------------------------------------------------------------
# Musicality calibration harness (v0.3 Phase 3c)
# ---------------------------------------------------------------------------

@dataclass
class CalibrationResult:
    """Result of a MIDI-level musicality calibration run."""
    good_scores: List[float]
    bad_scores: List[float]
    suggested_threshold: float
    separation_ok: bool
    good_mean: float
    bad_mean: float


# --- Minimal MIDI writer (no external deps beyond struct) -------------------

def _var_len(n: int) -> bytes:
    result = [n & 0x7F]
    n >>= 7
    while n:
        result.insert(0, (n & 0x7F) | 0x80)
        n >>= 7
    return bytes(result)


def _midi_bytes(
    notes: List[tuple],  # (pitch, start_tick, dur_tick, velocity)
    tempo_bpm: int = 120,
    ticks_per_beat: int = 480,
) -> bytes:
    us = int(60_000_000 / tempo_bpm)
    events = [(0, bytes([0xFF, 0x51, 0x03]) + us.to_bytes(3, "big"))]
    for pitch, start, dur, vel in notes:
        events.append((start, bytes([0x90, pitch, vel])))
        events.append((start + dur, bytes([0x80, pitch, 0x00])))
    events.append((ticks_per_beat * 4, bytes([0xFF, 0x2F, 0x00])))
    events.sort(key=lambda e: e[0])
    track_bytes = b""
    prev = 0
    for tick, msg in events:
        delta = tick - prev
        prev = tick
        track_bytes += _var_len(delta) + msg
    header = struct.pack(">4sIHHH", b"MThd", 6, 0, 1, ticks_per_beat)
    track = struct.pack(">4sI", b"MTrk", len(track_bytes)) + track_bytes
    return header + track


def _write_midi(path: str, notes: list, tempo_bpm: int = 120) -> None:
    from pathlib import Path
    Path(path).write_bytes(_midi_bytes(notes, tempo_bpm))


# --- Reference set builders -------------------------------------------------

_C_MAJOR_SCALE = [60, 62, 64, 65, 67, 69, 71, 72]  # C4..C5


def _good_melody_notes(rng: random.Random, n_notes: int = 16) -> list:
    """Step-wise C-major melody, varied pitches, one-octave range."""
    scale = _C_MAJOR_SCALE
    notes = []
    for i in range(n_notes):
        pitch = scale[i % len(scale)]
        start = i * 240
        notes.append((pitch, start, 200, rng.randint(70, 100)))
    return notes


def _bad_melody_notes(rng: random.Random, variant: int) -> list:
    """Adversarial melody variants:
      0 = empty (0 notes)
      1 = stuck note (15/16 same pitch)
      2 = extreme range (4-octave span)
      3 = chromatic noise (random semitones, no key)
    """
    if variant == 0:
        return []
    if variant == 1:
        notes = [(60, i * 240, 200, 80) for i in range(15)]
        notes.append((62, 15 * 240, 200, 80))
        return notes
    if variant == 2:
        return [(36, 0, 200, 80), (84, 240, 200, 80)] * 8
    # variant 3+: chromatic noise
    return [(rng.randint(36, 96), i * 240, 200, 80) for i in range(16)]


def _write_sample_midi_paths(
    base_dir: str, index: int, melody_notes: list, rng: random.Random
) -> Dict[str, str]:
    """Write 4-layer MIDI for one calibration sample. Returns paths dict."""
    os.makedirs(base_dir, exist_ok=True)
    good_filler = [(60 + i % 5, i * 240, 200, 80) for i in range(16)]
    paths = {}
    for layer in ("beat", "harmony", "bassline"):
        p = os.path.join(base_dir, f"{index:04d}-{layer}.mid")
        _write_midi(p, good_filler)
        paths[layer] = p
    mel_p = os.path.join(base_dir, f"{index:04d}-melody.mid")
    _write_midi(mel_p, melody_notes)
    paths["melody"] = mel_p
    return paths


# --- Core calibration functions ---------------------------------------------

def suggest_threshold(good_scores: List[float], bad_scores: List[float]) -> float:
    """Derive threshold that separates good from bad score distributions.

    When clearly separated: midpoint between p90(bad) and p10(good).
    When overlapping: p25(good) as a conservative threshold.
    Returns 0.5 when either list is empty.
    """
    import numpy as np
    if not good_scores or not bad_scores:
        return 0.5
    good_low = float(np.percentile(good_scores, 10))
    bad_high = float(np.percentile(bad_scores, 90))
    if good_low > bad_high:
        return float((good_low + bad_high) / 2.0)
    return float(np.clip(np.percentile(good_scores, 25), 0.0, 1.0))


def run_midi_calibration(
    n_good: int = 20,
    n_bad: int = 20,
    seed: int = 42,
    tmp_dir: Optional[str] = None,
) -> CalibrationResult:
    """Generate reference-good and adversarial MIDI sets; score with check_midi_quality.

    Args:
        n_good: Number of reference-good samples to generate.
        n_bad:  Number of adversarial samples to generate.
        seed:   RNG seed for determinism.
        tmp_dir: Directory for temp MIDI files. Uses tempfile.mkdtemp if None.

    Returns:
        CalibrationResult with scores, threshold, and separation flag.
    """
    import numpy as np
    from musicgen.musicality import check_midi_quality

    rng = random.Random(seed)
    n_variants = 4  # number of bad variants (0=empty, 1=stuck, 2=extreme, 3=chromatic)

    cleanup = tmp_dir is None
    if tmp_dir is None:
        tmp_dir = tempfile.mkdtemp(prefix="musicgen-calib-")
    good_dir = os.path.join(tmp_dir, "good")
    bad_dir = os.path.join(tmp_dir, "bad")

    try:
        good_scores: List[float] = []
        for i in range(n_good):
            notes = _good_melody_notes(rng)
            paths = _write_sample_midi_paths(good_dir, i, notes, rng)
            result = check_midi_quality(midi_paths=paths, key="C")
            good_scores.append(result.score)

        bad_scores: List[float] = []
        for i in range(n_bad):
            variant = i % n_variants
            notes = _bad_melody_notes(rng, variant)
            paths = _write_sample_midi_paths(bad_dir, i, notes, rng)
            result = check_midi_quality(midi_paths=paths, key="C")
            bad_scores.append(result.score)

    finally:
        if cleanup:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)

    good_mean = float(np.mean(good_scores)) if good_scores else 0.0
    bad_mean = float(np.mean(bad_scores)) if bad_scores else 0.0
    threshold = suggest_threshold(good_scores, bad_scores)

    separation_margin = 0.2
    separation_ok = (good_mean - bad_mean) >= separation_margin

    return CalibrationResult(
        good_scores=good_scores,
        bad_scores=bad_scores,
        suggested_threshold=threshold,
        separation_ok=separation_ok,
        good_mean=good_mean,
        bad_mean=bad_mean,
    )


def save_calibration(result: CalibrationResult, path: str) -> None:
    """Persist CalibrationResult to a JSON file."""
    data = dataclasses.asdict(result)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    logger.debug("Calibration saved to %s", path)


def load_calibration(path: str) -> CalibrationResult:
    """Load CalibrationResult from a JSON file."""
    with open(path, "r") as f:
        data = json.load(f)
    return CalibrationResult(**data)
