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

from typing import Dict, Optional

from midiutil import MIDIFile
from music21 import roman, scale, pitch  # Plan 01-03 narrow-import commitment (S3)

from musicgen.duration_validator import DurationValidator
from timesig import TimeSignatureRegistry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Scale-relative Markov helpers (v0.3 Phase 2)
# ---------------------------------------------------------------------------

# Semitone offsets from root for each scale degree (1-based string keys).
_MAJOR_INTERVALS: Dict[str, int] = {"1": 0, "2": 2, "3": 4, "4": 5, "5": 7, "6": 9, "7": 11}
_MINOR_INTERVALS: Dict[str, int] = {"1": 0, "2": 2, "3": 3, "4": 5, "5": 7, "6": 8, "7": 10}

# Map note name → semitone 0–11
_NOTE_TO_SEMI: Dict[str, int] = {
    "C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5,
    "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11,
    "Db": 1, "Eb": 3, "Gb": 6, "Ab": 8, "Bb": 10,
}


def _key_root_and_intervals(key: str):
    """Return (root_semitone, intervals_dict) for a key string like "C", "Am", "F#"."""
    is_minor = key.endswith("m") and len(key) > 1 and key[-2] not in ("#", "b")
    root_name = key[:-1] if is_minor else key
    root_semi = _NOTE_TO_SEMI.get(root_name, 0)
    intervals = _MINOR_INTERVALS if is_minor else _MAJOR_INTERVALS
    return root_semi, intervals


def _degree_to_midi(degree: str, key: str, reference: int = 60) -> int:
    """Convert scale degree "1"–"7" to MIDI note, picking octave closest to reference.

    Output clamped to [36, 96] to stay in melodic range.
    """
    root_semi, intervals = _key_root_and_intervals(key)
    offset = intervals.get(degree, 0)
    target_semi = (root_semi + offset) % 12
    # Start from reference, find nearest pitch with correct semitone class
    base = (reference // 12) * 12 + target_semi
    # Choose closest octave
    candidates = [base - 12, base, base + 12]
    midi = min(candidates, key=lambda m: abs(m - reference))
    return max(36, min(96, midi))


def _sample_melody_markov(history: List[str], matrix: dict, rng: random.Random) -> str:
    """Sample next scale degree from a Markov transition matrix.

    Identical fallback chain to _sample_chord_markov:
    - step 0: draw from init_probs
    - step 1: transitions[history[-1]] → fallback to init_probs
    - step ≥ 2: transitions[prev,curr] → transitions[curr] → init_probs
    """
    transitions: dict = matrix.get("transitions", {})
    init_probs: dict = matrix.get("init_probs", {})

    def _draw(dist: dict) -> str:
        dist = {k: v for k, v in dist.items() if v > 0}
        if not dist:
            dist = {k: 1.0 for k in init_probs} if init_probs else {"1": 1.0}
        population = list(dist)
        return rng.choices(population, weights=[dist[k] for k in population], k=1)[0]

    if not history:
        return _draw(init_probs)

    curr = history[-1]
    if len(history) >= 2:
        prev = history[-2]
        key_2nd = f"{prev},{curr}"
        if key_2nd in transitions:
            return _draw(transitions[key_2nd])

    if curr in transitions:
        return _draw(transitions[curr])

    return _draw(init_probs)


def generate_melody(
    key: str,
    tempo: int,
    time_signature: str,
    measures: int,
    name: str,
    part: str,
    chord_progression: List[str],
    rng: random.Random,
    genre_spec=None,
) -> Tuple[List[int], str]:
    """Generate a melody based on key, tempo, chord progression and time signature.

    v0.3 Phase 2: if genre_spec.melody_transition_matrix is set, the melody is
    generated via a scale-degree Markov chain (key-agnostic, zero-weight-free).
    Otherwise falls back to the existing chord-pitch Markov path.
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

    matrix = getattr(genre_spec, "melody_transition_matrix", None) if genre_spec is not None else None

    if matrix is not None:
        # v0.3 Phase 2 — scale-degree Markov path (key-agnostic, zero-weight-free).
        melody: List[int] = []
        note_durations: List[float] = []
        total_beats = measures * validator._analyze_time_signature(time_signature).beats_per_measure
        remaining_beats = total_beats
        degree_history: List[str] = []
        reference_midi = 60  # track pitch continuity across notes

        while remaining_beats > 0:
            degree = _sample_melody_markov(degree_history, matrix, rng)
            degree_history.append(degree)
            midi_note = _degree_to_midi(degree, key, reference_midi)
            reference_midi = midi_note

            raw_duration = rng.choice(list(spec.melody_duration_candidates))
            duration = validator.get_valid_duration(
                raw_duration, time_signature, remaining_beats, "melody"
            )
            melody.append(midi_note)
            note_durations.append(duration)
            remaining_beats -= duration

        possible_durations = list(spec.melody_duration_candidates)
    else:
        # Existing chord-pitch Markov path (pre-v0.3 behavior).
        # music21 global-random audit (Phase 3, D-23): music21 9.9.1's roman.RomanNumeral,
        # scale.MajorScale, scale.MinorScale, and pitch.Pitch do NOT mutate random.getstate().
        # Verified empirically 2026-04-18. If this changes in a future music21 release,
        # tests/test_music21_isolation.py will fail — wrap calls in save_random_state() then.
        if key[-1] == 'm':
            scale_obj = scale.MinorScale(key[:-1])
        else:
            scale_obj = scale.MajorScale(key)

        chords = []
        for chord_symbol in chord_progression:
            chord_obj = roman.RomanNumeral(chord_symbol, key)
            chord_obj.key = key
            chords.append(chord_obj)

        if part == 'intro':
            notes_to_use = chords[0].pitches
        elif part == 'outro':
            notes_to_use = chords[-1].pitches
        else:
            notes_to_use = []
            for chord in chords:
                notes_to_use.extend(chord.pitches)

        transition_matrix = {}
        for note in notes_to_use:
            transition_matrix[note.midi] = {}
            for next_note in notes_to_use:
                if next_note in chord_obj.pitches:
                    transition_matrix[note.midi][next_note.midi] = 1 / len(notes_to_use)
                else:
                    transition_matrix[note.midi][next_note.midi] = 0

        melody = []
        note_durations = []
        total_beats = measures * validator._analyze_time_signature(time_signature).beats_per_measure
        remaining_beats = total_beats

        current_note = rng.choice([note.midi for note in notes_to_use])

        possible_durations = list(spec.melody_duration_candidates)
        while remaining_beats > 0:
            current_note = rng.choices(
                population=list(transition_matrix[current_note].keys()),
                weights=list(transition_matrix[current_note].values())
            )[0]
            raw_duration = rng.choice(possible_durations)
            duration = validator.get_valid_duration(
                raw_duration, time_signature, remaining_beats, "melody"
            )
            melody.append(current_note)
            note_durations.append(duration)
            remaining_beats -= duration

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
