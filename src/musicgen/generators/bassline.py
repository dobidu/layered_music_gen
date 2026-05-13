"""Bassline generator (extracted from music_gen.py per Plan 03-04 / R-X3).

Generates a bassline MIDI file for one song part, given a key, tempo,
time signature, chord progression, melody, and an injected ``rng: random.Random``.

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
from typing import List

from midiutil import MIDIFile
from music21 import roman, scale, pitch  # Plan 01-03 narrow-import commitment (S3)

from musicgen.duration_validator import DurationValidator
from timesig import TimeSignatureRegistry

logger = logging.getLogger(__name__)


def generate_bassline(
    key: str,
    tempo: int,
    time_signature: str,
    measures: int,
    name: str,
    part: str,
    chord_progression: List[str],
    melody: List[int],
    rng: random.Random,
    genre_spec=None,
) -> str:
    """
    Generate a bassline based on key, tempo, a chord progression and time signature.
    """
    validator = DurationValidator()
    mf = MIDIFile(1)
    track = 0
    time = 0
    mf.addTrackName(track, time, "Bassline")
    mf.addTempo(track, time, tempo)

    spec = TimeSignatureRegistry.lookup(time_signature)
    numerator = spec.numerator
    midi_denominator = spec.midi_denominator_power
    mf.addTimeSignature(track, time, numerator, midi_denominator, 24, 8)

    base_duration = validator.get_suggested_duration(time_signature, 'bass')
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
        notes_to_use = [chords[0].pitches[0]]
    elif part == 'outro':
        notes_to_use = [chords[-1].pitches[0]]
    else:
        notes_to_use = [chord.pitches[0] for chord in chords]

    # Define the Markov chain transition matrix
    transition_matrix = {}
    for note in notes_to_use:
        transition_matrix[note.midi] = {}
        for next_note in notes_to_use:
            transition_matrix[note.midi][next_note.midi] = 1 / len(notes_to_use)

    # Generate a random bassline using a Markov chain
    bassline = []
    note_durations = []
    remaining_beats = total_beats

    # Initialize with a random note from the available ones
    current_note = rng.choice([note.midi for note in notes_to_use])

    while remaining_beats > 0:
        base_duration = validator.get_suggested_duration(time_signature, 'bass')
        duration = validator.get_valid_duration(
            base_duration,
            time_signature,
            remaining_beats,
            'bass'
        )
        current_note = rng.choices(
            population=list(transition_matrix[current_note].keys()),
            weights=list(transition_matrix[current_note].values())
        )[0]

        note_duration = duration

        # Adjust if it's the last beat
        if note_duration > remaining_beats:
            note_duration = remaining_beats

        bassline.append(current_note)
        note_durations.append(note_duration)
        remaining_beats -= note_duration

    # Make sure the bassline follows the melody
    for i in range(len(bassline)):
        if i < len(melody):
            if bassline[i] != melody[i]:
                if rng.random() < 0.5:
                    bassline[i] = melody[i]

    logger.debug("Bassline: %s", bassline)
    # Add notes to MIDI file
    current_time = 0
    for i in range(len(bassline)):
        note = bassline[i]
        # note_duration = note_durations[i]
        velocity = rng.randint(70, 100)

        note_obj = pitch.Pitch()
        note_obj.midi = note
        note_obj.octave = 2

        mf.addNote(track, 0, note_obj.midi, current_time, note_durations[i], velocity)
        current_time += note_durations[i]

    # Saves MIDI file
    directory = name.split('-')[0]
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = os.path.join(directory, name + "-bassline.mid")

    with open(filename, 'wb') as outf:
        mf.writeFile(outf)

    logger.debug("Bassline midi file: %s", filename)

    return filename
