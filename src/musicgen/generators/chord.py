"""Chord-progression generator (extracted from music_gen.py per Plan 03-04 / R-X3).

Generates a chord progression MIDI file for one song part, given a key, tempo,
time signature, pattern file, and an injected ``rng: random.Random``. All RNG
draws go through ``rng`` so Phase 5 can feed per-sample deterministic RNGs
without rewriting this generator.

Design:
  D-06 — Uses ``TimeSignatureRegistry.lookup(...)`` directly (no indirection
         through music_gen wrappers for attribute access; behavior wrappers
         stay in music_gen shim only where the registry doesn't expose a
         direct method).
  D-07 — Zero bare ``random.<method>`` calls — all draws use injected ``rng``.
  D-22 — Takes per-part fields, not SongParams.
  D-23 — music21 roman.RomanNumeral audited clean (does not mutate global
         random state).
"""
import logging
import os
import random
from typing import List, Tuple

from midiutil import MIDIFile
from music21 import roman, scale, pitch  # Plan 01-03 narrow-import commitment (S3)

from musicgen.duration_validator import DurationValidator
from timesig import TimeSignatureRegistry

logger = logging.getLogger(__name__)


def generate_chord_progression(
    key: str,
    tempo: int,
    time_signature: str,
    measures: int,
    name: str,
    part: str,
    pattern_file: str,
    rng: random.Random,
) -> Tuple[List[str], str]:
    """
    Generate a chord progression based on key, tempo, a pattern file and time signature.
    """
    validator = DurationValidator()
    mf = MIDIFile(1)
    track = 0
    time = 0

    # MIDI initial setup
    mf.addTrackName(track, time, "Chord Progression")
    mf.addTempo(track, time, tempo)
    spec = TimeSignatureRegistry.lookup(time_signature)
    numerator = spec.numerator
    midi_denominator = spec.midi_denominator_power
    mf.addTimeSignature(track, time, numerator, midi_denominator, 24, 8)

    beats_per_measure = numerator
    if time_signature.endswith('8'):  # Composed meters
        if numerator % 3 == 0:  # 6/8, 12/8
            beats_per_measure = numerator // 3  # Groups into units of 3 eighth notes

    # Reads and validates chord patterns
    chord_patterns = {}
    with open(pattern_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                part_name, pattern = line.split(':')
                pattern_chords = pattern.split(',')
                if spec.verify_chord_pattern_length(len(pattern_chords)):
                    chord_patterns.setdefault(part_name, []).append(pattern_chords)

    # default pattern if not found
    if part not in chord_patterns or not chord_patterns[part]:
        base_pattern = ['I'] if numerator in [2, 3] else ['I', 'IV', 'V', 'vi']
        chord_patterns[part] = [base_pattern]

    # chooses and applies a pattern
    chord_pattern = rng.choice(chord_patterns[part])
    base_duration = validator.get_suggested_duration(time_signature, 'chord')
    chord_duration = validator.get_valid_duration(
        base_duration,
        time_signature,
        validator._analyze_time_signature(time_signature).beats_per_measure,
        'chord'
    )

    # Aditional validation of chord duration
    if chord_duration <= 0:
        raise ValueError(f"Invalid chord duration calculated for time signature {time_signature}")

    # music21 global-random audit (Phase 3, D-23): music21 9.9.1's roman.RomanNumeral,
    # scale.MajorScale, scale.MinorScale, and pitch.Pitch do NOT mutate random.getstate().
    # Verified empirically 2026-04-18. If this changes in a future music21 release,
    # tests/test_music21_isolation.py will fail — wrap calls in save_random_state() then.
    # Generate chord progression
    chord_progression = []
    for chord_symbol in chord_pattern:
        chord = roman.RomanNumeral(chord_symbol.strip(), key)
        chord_progression.append(chord)

    # Add chords to MIDI file
    current_time = 0
    for _ in range(measures):
        for chord in chord_progression:
            for note in chord.pitches:
                mf.addNote(track, 0, note.midi, current_time, chord_duration, 100)
            current_time += chord_duration

    # Saves MIDI file
    directory = name.split('-')[0]
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = os.path.join(directory, name + "-chord_progression.mid")

    with open(filename, 'wb') as outf:
        mf.writeFile(outf)

    logger.debug("Chord progression: %s", chord_progression)

    return chord_pattern, filename
