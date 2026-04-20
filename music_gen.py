"""music_gen.py — smoke wrapper calling ``musicgen.generate`` (D-33, D-34).

Time-sig wrappers + ``validate_measures`` alias stay one more phase per D-34;
Phase 6 replaces this file with the ``musicgen generate`` CLI (R-P13).
"""
from __future__ import annotations

import logging
from typing import List, Tuple

from musicgen.sampler import validate_measures_dict
from timesig import TimeSignatureRegistry

logger = logging.getLogger(__name__)
validate_measures = validate_measures_dict  # D-34 back-compat alias (test_time_signature.py)

# --- Time-signature delegation wrappers (D-34: stay for one more phase) ---

def verify_pattern_for_time_signature(chord_pattern: List[str], time_signature: str) -> bool:
    """Delegates to TimeSignatureRegistry per R-S6."""
    return TimeSignatureRegistry.lookup(time_signature).verify_chord_pattern_length(len(chord_pattern))

def verify_beat_pattern(pattern: List[int], time_signature: str) -> bool:
    """Delegates to TimeSignatureRegistry per R-S6."""
    return TimeSignatureRegistry.lookup(time_signature).verify_beat_pattern_length(len(pattern))

def calculate_measures_for_time_signature(base_length: int, time_signature: str) -> int:
    """Delegates to TimeSignatureRegistry per R-S6."""
    return TimeSignatureRegistry.lookup(time_signature).measures_for(base_length)

def get_midi_time_signature_values(time_signature: str) -> Tuple[int, int]:
    """Delegates to TimeSignatureRegistry per R-S6."""
    spec = TimeSignatureRegistry.lookup(time_signature)
    return spec.numerator, spec.midi_denominator_power

def get_note_duration(time_signature: str) -> float:
    """Delegates to TimeSignatureRegistry per R-S6."""
    return TimeSignatureRegistry.lookup(time_signature).primary_beat_duration

def get_note_durations(time_signature: str) -> dict:
    """Delegates to TimeSignatureRegistry per R-S6."""
    return TimeSignatureRegistry.lookup(time_signature).note_duration_map()

def get_melody_durations(time_signature: str) -> list:
    """Delegates to TimeSignatureRegistry per R-S6."""
    return list(TimeSignatureRegistry.lookup(time_signature).melody_duration_candidates)

# --- Smoke-test entry point (D-33) ---
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    from musicgen import Config, generate
    result = generate(Config(global_seed=1, sample_index=0))
    logger.info(
        "Sample %d (seed=%d) written to %s — status=%s",
        result.sample_index, result.seed, result.sample_dir, result.status,
    )
