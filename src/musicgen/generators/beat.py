"""Beat generator (extracted from music_gen.py per Plan 03-04 / R-X3).

Generates a drum-pattern MIDI file for one song part, with configurable swing.
Co-locates ``beat_duration`` and ``calculate_swing_offset`` helpers (pure
functions — no RNG).

Design:
  D-02 — cfg fallback: ``_beat_cfg = cfg if cfg is not None else config.Config()``.
  D-06 — Uses ``TimeSignatureRegistry.lookup(...)`` directly for attribute
         access (numerator, midi denominator power). The beat-pattern
         length check still routes through the spec via
         ``verify_beat_pattern_length``.
  D-07 — Zero bare ``random.<method>`` calls — all draws use injected ``rng``.
  D-22 — Takes per-part fields, not SongParams.

No music21 imports (beat is percussion-only, per S3).
"""
import logging
import os
import random
from typing import Dict, List, Optional, Tuple

from midiutil import MIDIFile

import config
from musicgen.duration_validator import DurationValidator
# D-21: beat_duration primary definition moves to musicgen.beats this phase.
# Import and re-export so existing callers (generators, tests, music_gen.py:34)
# are unaffected. calculate_swing_offset stays here (used only at MIDI write
# time, not at annotation time).
from musicgen.beats import beat_duration  # noqa: F401  (D-21 re-export)
from timesig import TimeSignatureRegistry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pattern loader (v0.2 Phase 3)
# ---------------------------------------------------------------------------

def _sig_to_flat(time_signature: str) -> str:
    """Convert "4/4" → "44", "12/8" → "128", etc."""
    return time_signature.replace("/", "")


def load_beat_patterns(
    time_signature: str,
    pattern_dirs: List[str],
    spec,
) -> Dict[str, List[List[int]]]:
    """Load and union beat patterns from multiple directories.

    Each dir is searched for patterns_<sig_flat>.txt (e.g. patterns_44.txt).
    Patterns from all dirs are unioned per part; identical patterns deduplicated.
    Missing dirs and missing files are silently skipped.

    Returns {part_name: [[n1, n2, ...], ...]} same structure as current loader.
    """
    flat = _sig_to_flat(time_signature)
    seen: Dict[str, set] = {}
    result: Dict[str, List[List[int]]] = {}

    for d in pattern_dirs:
        path = os.path.join(d, f"patterns_{flat}.txt")
        if not os.path.isfile(path):
            continue
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or ":" not in line:
                    continue
                part_name, pat_str = line.split(":", 1)
                part_name = part_name.strip()
                try:
                    pattern = [int(x.strip()) for x in pat_str.split(",") if x.strip()]
                except ValueError:
                    continue
                if not spec.verify_beat_pattern_length(len(pattern)):
                    continue
                key = tuple(pattern)
                if key not in seen.setdefault(part_name, set()):
                    seen[part_name].add(key)
                    result.setdefault(part_name, []).append(pattern)

    return result


def calculate_swing_offset(base_duration: float, swing_amount: float) -> float:
    """
    Calculates the offset time for a note with swing.

    Args:
        base_duration: Base note duration in beats
        swing_amount: Swing amount (0.0 to 1.0, where 0.5 is straight timing)

    Returns:
        float: Time offset in beats
    """
    # Swing amount of 0.5 means straight timing (in swing)
    # Values ​​> 0.5 delay the off-beat
    # Values ​​< 0.5 advance the off-beat
    return base_duration * (swing_amount - 0.5)


def generate_beat(
    part: str,
    tempo: int,
    time_signature: str,
    measures: int,
    name: str,
    swing_amount: float,
    rng: random.Random,
    cfg: Optional[config.Config] = None,
) -> Tuple[str, List[str]]:
    """
    Generates a drum pattern with configurable swing.

    Args:
        part: Part of the song (intro, verse, etc.)
        tempo: BPM
        time_signature: Time signature (ex: "4/4")
        measures: Number of bars
        name: Base name for generated files
        swing_amount: Swing amount (0.0 to 1.0, default 0.5 = no swing)
        rng: Injected random.Random (required; D-07/D-08)
        cfg: Optional Config override (D-02 fallback pattern)

    Returns:
        Tuple[str, List[str]]: (MIDI file name, list of time annotations)
    """
    validator = DurationValidator()
    _beat_cfg = cfg if cfg is not None else config.Config()

    mf = MIDIFile(1)
    track = 0
    time = 0

    mf.addTrackName(track, time, "Beat")
    mf.addTempo(track, time, tempo)

    spec = TimeSignatureRegistry.lookup(time_signature)
    numerator = spec.numerator
    midi_denominator = spec.midi_denominator_power
    mf.addTimeSignature(track, time, numerator, midi_denominator, 24, 8)

    base_duration = validator.get_suggested_duration(time_signature, 'beat')

    kick = 36
    snare = 38
    hihat = 42

    # Load patterns: prefer dirs-based loader (v0.2); fall back to file dict (legacy).
    pattern_dirs = getattr(_beat_cfg, "beat_roll_pattern_dirs", [])
    if pattern_dirs:
        beat_patterns = load_beat_patterns(time_signature, pattern_dirs, spec)
        if not beat_patterns:
            raise ValueError(
                f"No beat patterns found for {time_signature!r} in dirs: {pattern_dirs}"
            )
    else:
        beat_pattern_files = dict(_beat_cfg.beat_roll_pattern_files)
        filename = beat_pattern_files.get(time_signature)
        if not filename:
            raise ValueError(f"Time signature {time_signature} not supported.")
        beat_patterns = {}
        with open(filename, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split(":")
                    song_part = parts[0].strip()
                    pattern = [int(x) for x in parts[1].split(",")]
                    if spec.verify_beat_pattern_length(len(pattern)):
                        beat_patterns.setdefault(song_part, []).append(pattern)

    # default pattern
    if part not in beat_patterns or not beat_patterns[part]:
        base_pattern = [kick, hihat] if numerator == 2 else [kick, hihat, snare]
        beat_patterns[part] = [base_pattern]

    # Generates the pattern for the part
    beat_pattern = rng.choice(beat_patterns[part])
    beat = beat_pattern * (measures - 1)

    # Adds roll to the end
    roll_part = part + "_roll"
    roll_pattern = rng.choice(beat_patterns.get(roll_part, [beat_pattern]))
    beat.extend(roll_pattern)

    # Add notes to MIDI file with swing
    annotations = []
    current_time = 0.0

    for i, drum_hit in enumerate(beat):
        if drum_hit != 0:
            # Applies swing only to weak notes (odd indices)
            if i % 2 == 1:  # off-beat
                swing_offset = calculate_swing_offset(base_duration, swing_amount)
                actual_time = current_time + swing_offset
            else:
                actual_time = current_time

            mf.addNote(track, 9, drum_hit, actual_time, base_duration, 100)
            annotations.append(f"{actual_time:.3f}\t{len(annotations) + 1}")

        current_time += base_duration

    # Saves MIDI
    directory = name.split('-')[0]
    if not os.path.exists(directory):
        os.makedirs(directory)
    midi_filename = os.path.join(directory, f"{name}-beat.mid")

    with open(midi_filename, 'wb') as outf:
        mf.writeFile(outf)

    return midi_filename, annotations
