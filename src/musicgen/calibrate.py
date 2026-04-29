"""FluidSynth pre-roll calibration (R-P9, D-50..D-54).

FluidSynth inserts a short silence before the first rendered note — the
"pre-roll" — that shifts all MIDI-anchored beat times forward. This module
measures the offset once, caches it in ``.musicgen/fluidsynth_preroll.json``,
and returns it for use in writer._apply_preroll_offset.

The cache is version-gated: if the installed FluidSynth binary changes, the
cached value is stale and the module re-measures automatically.

Public surface:
  load_preroll(project_root) -> float
  measure_preroll(project_root) -> float
  measure_and_save_preroll(project_root) -> float
  save_preroll(project_root, offset_s) -> None
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from typing import Optional

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
