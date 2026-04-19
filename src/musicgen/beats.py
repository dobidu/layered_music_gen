"""Beats module — MIDI-tick beat and downbeat extraction (R-X7).

Replaces ``beat_anotator.py`` (D-03/D-19). Uses ``mido`` MIDI-tick derivation so
beat timestamps are automatically swing-aware (swing is baked into MIDI onset
times by ``generators/beat.py:calculate_swing_offset`` at write time).

Design:
  D-19 — ``extract_beat_times`` uses ``mido.MidiFile`` + ``mido.tick2second``.
  D-20 — ``extract_downbeat_times`` uses TIME-GRID derivation, NOT stride-slice
         of ``beat_times``. RESEARCH correction #1 verified against all 6 beat
         pattern files: 4/4 ``intro: 0, 42, 38, 0`` has 2 non-zero entries per
         measure (stride-slice would return measures//2 downbeats); 12/8 has
         9-10 non-zero entries per measure (stride-slice returns wrong count).
  D-21 — ``beat_duration`` primary definition lives HERE; ``generators/beat.py``
         imports it as a re-export alias.
"""
from __future__ import annotations

import logging
from typing import List

import mido

from timesig import TimeSignatureRegistry

logger = logging.getLogger(__name__)


def beat_duration(signature: str, tempo: int) -> float:
    """Return the duration of one beat slot in seconds for (signature, tempo).

    Identical body to the pre-refactor ``generators/beat.py:beat_duration`` and
    ``beat_anotator.py:beat_duration`` — body is unchanged; only the home moves.

    Args:
        signature: Time signature string like ``"4/4"`` or ``"6/8"``.
        tempo: BPM, integer.

    Returns:
        Duration of one beat slot in seconds (float).
    """
    numerator, denominator = map(int, signature.split('/'))
    beat_length = 60 / tempo  # Duration of a quarter note
    return beat_length * (4 / denominator)


def extract_beat_times(
    midi_path: str,
    tempo: int,
    start_offset_seconds: float,
) -> List[float]:
    """Extract beat timestamps from MIDI ``note_on`` events (velocity > 0).

    Swing is already baked into MIDI onset times (``generators/beat.py``
    applies ``calculate_swing_offset`` at write time), so tick-derived
    extraction is automatically swing-aware (D-19).

    Args:
        midi_path: Path to the beat MIDI file.
        tempo: BPM integer.
        start_offset_seconds: Part start time in the full mix (from
            ``MixResult.transitions``).

    Returns:
        Sorted list of beat timestamps in seconds (rounded to 3 decimals).
    """
    midi = mido.MidiFile(midi_path)
    beats: List[float] = []
    time_elapsed = 0.0
    ticks_per_beat = midi.ticks_per_beat
    tempo_us = mido.bpm2tempo(tempo)
    for msg in midi:  # merged-track iteration; msg.time is delta ticks
        time_elapsed += mido.tick2second(msg.time, ticks_per_beat, tempo_us)
        if msg.type == 'note_on' and msg.velocity > 0:
            beats.append(round(time_elapsed + start_offset_seconds, 3))
    return sorted(beats)


def extract_downbeat_times(
    beat_times: List[float],
    time_signature: str,
    measures: int,
    start_offset_seconds: float,
    tempo: int,
) -> List[float]:
    """Derive downbeat timestamps as a pure time grid — one per measure.

    Does NOT stride-slice ``beat_times`` — ``beat_times`` is sparse for
    patterns with zero-valued slots (RESEARCH correction #1, verified against
    all 6 beat pattern files: 4/4 ``intro: 0, 42, 38, 0`` has 2 non-zero entries
    per measure; 12/8 has 9-10 non-zero entries per measure). The stride
    approach described in CONTEXT.md D-20 is an approximation that fails for
    these patterns; this time-grid implementation is correct by construction.

    ``beat_times`` is retained in the signature for API stability (future
    callers may want to cross-check the grid against actual MIDI onsets).

    Args:
        beat_times: Per-part beat timestamps (accepted but NOT sliced).
        time_signature: Time signature string like ``"4/4"``.
        measures: Number of measures in the part.
        start_offset_seconds: Part start time in the full mix.
        tempo: BPM integer.

    Returns:
        List of ``measures`` downbeat timestamps, one per measure,
        rounded to 3 decimals.
    """
    spec = TimeSignatureRegistry.lookup(time_signature)
    beat_slot_s = beat_duration(time_signature, tempo)
    measure_duration_s = spec.numerator * beat_slot_s
    return [
        round(start_offset_seconds + i * measure_duration_s, 3)
        for i in range(measures)
    ]
