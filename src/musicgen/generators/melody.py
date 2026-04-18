"""Melody generator (extracted from music_gen.py per Plan 03-04 / R-X3).

Generates a melody MIDI file for one song part, given a key, tempo,
time signature, chord progression, and an injected ``rng: random.Random``.

Design:
  D-06 — Uses ``TimeSignatureRegistry.lookup(...)`` directly for attribute
         access (numerator, midi denominator power).
  D-07 — Zero bare ``random.<method>`` calls — all draws use injected ``rng``.
  D-22 — Takes per-part fields, not SongParams.
  D-23 — music21 roman/scale/pitch audited clean (do not mutate global
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


def generate_melody(
    key: str,
    tempo: int,
    time_signature: str,
    measures: int,
    name: str,
    part: str,
    chord_progression: List[str],
    rng: random.Random,
) -> Tuple[List[int], str]:
    """
    Generate a melody based on key, tempo, a chord progression and time signature.
    """
    # Create a MIDI file with one track
    validator = DurationValidator()
    mf = MIDIFile(1)
    track = 0
    time = 0
    mf.addTrackName(track, time, "Melody")
    mf.addTempo(track, time, tempo)

    # Add time signature as a meta message
    # Add correct time signature
    spec = TimeSignatureRegistry.lookup(time_signature)
    numerator = spec.numerator
    midi_denominator = spec.midi_denominator_power
    mf.addTimeSignature(track, time, numerator, midi_denominator, 24, 8)

    # Get base note duration
    base_duration = validator.get_suggested_duration(time_signature, 'melody')
    beats_per_measure = numerator * base_duration
    total_beats = measures * beats_per_measure

    # music21 global-random audit (Phase 3, D-23): music21 9.9.1's roman.RomanNumeral,
    # scale.MajorScale, scale.MinorScale, and pitch.Pitch do NOT mutate random.getstate().
    # Verified empirically 2026-04-18. If this changes in a future music21 release,
    # tests/test_music21_isolation.py will fail — wrap calls in save_random_state() then.
    # Create scale based on key
    if key[-1] == 'm':
        scale_obj = scale.MinorScale(key[:-1])
    else:
        scale_obj = scale.MajorScale(key)

    # Create chord progression based on key and chord progression input
    chords = []
    for chord_symbol in chord_progression:
        chord_obj = roman.RomanNumeral(chord_symbol, key)
        chord_obj.key = key
        chords.append(chord_obj)

    # Determine which notes to use based on song part and chord progression
    if part == 'intro':
        notes_to_use = chords[0].pitches
    elif part == 'outro':
        notes_to_use = chords[-1].pitches
    else:
        notes_to_use = []
        for chord in chords:
            notes_to_use.extend(chord.pitches)

    # Define the Markov chain transition matrix
    # This matrix defines the probabilities of transitioning from one note to another
    transition_matrix = {}
    for note in notes_to_use:
        transition_matrix[note.midi] = {}
        for next_note in notes_to_use:
            if next_note in chord_obj.pitches:
                transition_matrix[note.midi][next_note.midi] = 1 / len(notes_to_use)
            else:
                transition_matrix[note.midi][next_note.midi] = 0

    # Generate a random melody using a Markov chain
    # Generate melody with correct note durations
    melody = []
    note_durations = []
    total_beats = measures * validator._analyze_time_signature(time_signature).beats_per_measure
    remaining_beats = total_beats

    current_note = rng.choice([note.midi for note in notes_to_use])

    # Choose the initial note randomly
    while remaining_beats > 0:
        current_note = rng.choices(
            population=list(transition_matrix[current_note].keys()),
            weights=list(transition_matrix[current_note].values())
        )[0]

        # Chooses proper note duration
        possible_durations = list(spec.melody_duration_candidates)
        # raw_duration = random.choice(possible_durations)
        raw_duration = rng.choice(list(spec.melody_duration_candidates))
        duration = validator.get_valid_duration(
            raw_duration,
            time_signature,
            remaining_beats,
            'melody'
        )

        note_duration = duration

        melody.append(current_note)
        note_durations.append(note_duration)
        remaining_beats -= note_duration

    if not validator.validate_layer_duration(possible_durations, time_signature, 'melody'):
        logger.warning("Generated melody has invalid timing structure")

    # Add notes to MIDI file
    for i in range(len(melody)):
        note = melody[i]
        velocity = rng.randint(70, 100)
        mf.addNote(track, 0, note, time, note_durations[i], velocity)
        time += note_durations[i]

    logger.debug("Melody: %s", melody)

    # Save MIDI file
    directory = name.split('-')[0]
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = os.path.join(directory, name + "-melody.mid")
    with open(filename, 'wb') as outf:
        mf.writeFile(outf)
    return melody, filename
