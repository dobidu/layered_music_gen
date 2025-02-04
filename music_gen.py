from midiutil import MIDIFile
from music21 import roman, scale, pitch
from pydub import AudioSegment
from midi2audio import FluidSynth
from datetime import datetime
from pedalboard import Pedalboard, Compressor, Gain, Chorus, LadderFilter, Phaser, Delay, Reverb
from pedalboard.io import AudioFile
import time
import json
import random
import os
import glob
from multiprocessing import Pool, cpu_count
import uuid
import musicality_score
from enhanced_duration_validator import DurationValidator, NoteValue
from gd_upload import GDriveUploader

from typing import Tuple, Dict, List, Optional
import math

import pickle
import os.path


def verify_pattern_for_time_signature(chord_pattern: List[str], time_signature: str) -> bool:
    """
    Checks if the chord pattern is appropriate for the time signature.
    """
    numerator, denominator = map(int, time_signature.split('/'))
    
    # Para compassos compostos (6/8, 12/8)
    if denominator == 8 and numerator % 3 == 0:
        return len(chord_pattern) in [2, 3, 6]  # Permite divisões apropriadas
    # Para compassos simples
    elif numerator == 4:
        return len(chord_pattern) in [1, 2, 4]  # Permite divisões em 4
    elif numerator == 3:
        return len(chord_pattern) in [1, 3]     # Permite divisões em 3
    elif numerator == 2:
        return len(chord_pattern) in [1, 2]     # Permite divisões em 2
        
    return True


def verify_beat_pattern(pattern: List[int], time_signature: str) -> bool:
    """
    Checks if the beat pattern matches the time signature.
    """
    numerator, denominator = map(int, time_signature.split('/'))
    
    # Checks if the pattern length matches the signature
    if denominator == 8 and numerator % 3 == 0:
        return len(pattern) == numerator  # 6 beats - 6/8, etc.
    else:
        return len(pattern) == numerator  # 4 beats - 4/4, etc.

def calculate_measures_for_time_signature(base_length: int, time_signature: str) -> int:
    # TODO: couple improved compass control system
    numerator, denominator = map(int, time_signature.split('/'))
    
    if denominator == 8 and numerator % 3 == 0:
        return base_length * 2  # Doubling for compound measures
    elif numerator == 2:
        return base_length * 2  # Doubling for 2/4
    elif numerator == 3:
        return int(base_length * 4/3)  # Adjusts for 3/4        
    return base_length  # 4/4 keeps original

def validate_measures(measures: Dict[str, int], signatures: Dict[str, str]) -> bool:
    """
    Validate the number of measures for each part based on time signature.
    """
    for part, measure_count in measures.items():
        time_sig = signatures[part]
        numerator, denominator = map(int, time_sig.split('/'))
        
        if denominator == 8 and numerator % 3 == 0:
            if measure_count % 2 != 0:
                return False
        elif numerator == 2 and measure_count % 2 != 0:
            return False
    
    return True

def get_midi_time_signature_values(time_signature: str) -> Tuple[int, int]:
    """
    Convert musical time signature to MIDI format values.
    For example:
        "4/4" -> (4, 2)  # denominator 4 = 2^2
        "3/4" -> (3, 2)  # denominator 4 = 2^2
        "6/8" -> (6, 3)  # denominator 8 = 2^3
    """
    numerator, denominator = map(int, time_signature.split('/'))
    # Convert denominator to power of 2
    midi_denominator = {
        1: 0,  # 2^0
        2: 1,  # 2^1
        4: 2,  # 2^2
        8: 3,  # 2^3
        16: 4, # 2^4
        32: 5  # 2^5
    }.get(denominator)
    
    if midi_denominator is None:
        raise ValueError(f"Unsupported time signature denominator: {denominator}")
        
    return numerator, midi_denominator

def get_note_duration(time_signature: str) -> float:
    """
    Calculate base note duration for given time signature.
    Returns duration in beats.
    """
    numerator, denominator = map(int, time_signature.split('/'))
    if denominator == 8 and numerator % 3 == 0:
        # Compound meter (6/8, 9/8, 12/8)
        return 0.5  # eighth note = 0.5 beat
    else:
        # Simple meter (2/4, 3/4, 4/4)
        return 1.0  # quarter note = 1 beat

def get_note_durations(time_signature: str) -> dict:
    """
    Calculate note durations for different note values based on time signature.
    Returns dictionary with standard note durations.
    """
    numerator, denominator = map(int, time_signature.split('/'))
    
    if denominator == 8 and numerator % 3 == 0:
        # Compound meter (6/8, 9/8, 12/8)
        return {
            'whole': 6.0,          # 6 eighth notes
            'half': 3.0,           # 3 eighth notes (dotted quarter)
            'quarter': 1.5,        # 1.5 eighth notes (dotted eighth)
            'eighth': 0.5,         # 1 eighth note
            'sixteenth': 0.25      # 1 sixteenth note
        }
    else:
        # Simple meter (2/4, 3/4, 4/4)
        return {
            'whole': 4.0,          # 4 quarter notes
            'half': 2.0,           # 2 quarter notes
            'quarter': 1.0,        # 1 quarter note
            'eighth': 0.5,         # 1 eighth note
            'sixteenth': 0.25      # 1 sixteenth note
        }

def get_melody_durations(time_signature: str) -> list:
    """
    Get appropriate note durations for melody based on time signature.
    """
    numerator, denominator = map(int, time_signature.split('/'))
    
    if denominator == 8 and numerator % 3 == 0:  # Compassos compostos (6/8, 12/8)
        return [
            0.5,  # Eighth note (basic unit)
            1.5,  # Dotted eighth note (group of 3)
            3.0   # Dotted quarter note (full group)
        ]
    elif numerator == 3:  # 3/4
        return [
            0.5,  # Eighth note
            1.0,  # Quarter note
            1.5   # Dotted quarter note
        ]
    else:  # 2/4, 4/4
        return [
            0.5,  # Eighth note
            1.0,  # Quarter note
            2.0   # Half note
        ]

def generate_chord_progression(key, tempo, time_signature, measures, name, part, pattern_file):    
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
    numerator, midi_denominator = get_midi_time_signature_values(time_signature)
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
                if verify_pattern_for_time_signature(pattern_chords, time_signature):
                    chord_patterns.setdefault(part_name, []).append(pattern_chords)

    # default pattern if not found
    if part not in chord_patterns or not chord_patterns[part]:
        base_pattern = ['I'] if numerator in [2, 3] else ['I', 'IV', 'V', 'vi']
        chord_patterns[part] = [base_pattern]
    
    # chooses and applies a pattern
    chord_pattern = random.choice(chord_patterns[part])
    
    chord_pattern = random.choice(chord_patterns[part])
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
    
    print("\t\t\tChord progression: " + str(chord_progression))
    
    return chord_pattern, filename

def get_fallback_weights(valid_notes, part_type, scale_notes, current_note):
    """Generate varied fallback weights based on context"""
    strategy = random.choice(['equal', 'scale_based', 'interval_based', 'part_based'])
    
    if strategy == 'equal':
        return [1.0] * len(valid_notes)
        
    elif strategy == 'scale_based':
        return [2.0 if note in scale_notes else 1.0 
                for note in valid_notes]
        
    elif strategy == 'interval_based':
        return [2.0 if abs(note - current_note) <= 4 else 1.0 
                for note in valid_notes]
        
    else:  # part_based
        weights = [1.0] * len(valid_notes)
        if part_type == 'chorus':
            # Favor higher notes in chorus
            weights = [1.5 if note > current_note else 1.0 
                      for note in valid_notes]
        elif part_type == 'verse':
            # More varied intervals in verse
            weights = [1.5 if abs(note - current_note) <= 7 else 1.0 
                      for note in valid_notes]
        return weights

def get_valid_weights(transition_matrix, current_note, valid_notes, part_type, scale_notes):
    weights = list(transition_matrix[current_note].values())
    
    if sum(weights) <= 0:
        weights = get_fallback_weights(valid_notes, part_type, scale_notes, current_note)
        
    return weights


def generate_melody(key, tempo, time_signature, measures, name, part, chord_progression):
    validator = DurationValidator()
    mf = MIDIFile(1)
    track = 0
    time = 0
    
    mf.addTrackName(track, time, "Melody")
    mf.addTempo(track, time, tempo)
    numerator, midi_denominator = get_midi_time_signature_values(time_signature)
    mf.addTimeSignature(track, time, numerator, midi_denominator, 24, 8)

    # Get scale and chord notes
    scale_obj = scale.MajorScale(key) if not key.endswith('m') else scale.MinorScale(key[:-1])
    scale_notes = [n.midi for n in scale_obj.getPitches()]
    
    # Build chord notes with validation
    chord_notes = set()
    for chord_symbol in chord_progression:
        try:
            chord = roman.RomanNumeral(chord_symbol, key)
            chord_notes.update([n.midi for n in chord.pitches])
        except:
            continue
    
    # Fallback to scale if no chord notes
    if not chord_notes:
        chord_notes = set(scale_notes)
    
    # Ensure notes are in valid range
    valid_notes = sorted(list(chord_notes))
    if not valid_notes:
        valid_notes = [60, 64, 67, 71]  # Fallback to C major notes
        
    # Build transition matrix with guaranteed weights
    transition_matrix = {}
    for note in valid_notes:
        transition_matrix[note] = {}
        for next_note in valid_notes:
            weight = 1.0  # Base weight
            if abs(note - next_note) <= 4:  # Prefer small intervals
                weight += 1.0
            if next_note in chord_notes:  # Prefer chord tones
                weight += 0.5
            transition_matrix[note][next_note] = weight

    # Generate melody
    melody = []
    note_durations = []
    remaining_beats = measures * validator._analyze_time_signature(time_signature).beats_per_measure
    current_note = random.choice(valid_notes)

    while remaining_beats > 0:
        # Get valid weights
        weights = get_valid_weights(transition_matrix, current_note, valid_notes, part, scale_notes)
            
        current_note = random.choices(
            population=list(transition_matrix[current_note].keys()),
            weights=weights
        )[0]

        duration = validator.get_valid_duration(
            random.choice(get_melody_durations(time_signature)),
            time_signature,
            remaining_beats,
            'melody'
        )
        
        melody.append(current_note)
        note_durations.append(duration)
        remaining_beats -= duration

    # Write MIDI
    current_time = 0
    for i, note in enumerate(melody):
        velocity = random.randint(70, 100)
        mf.addNote(track, 0, note, current_time, note_durations[i], velocity)
        current_time += note_durations[i]

    directory = name.split('-')[0]
    os.makedirs(directory, exist_ok=True)
    filename = os.path.join(directory, f"{name}-melody.mid")
    
    with open(filename, 'wb') as outf:
        mf.writeFile(outf)
        
    return melody, filename

def generate_bassline(key, tempo, time_signature, measures, name, part, chord_progression, melody):
    """
    Generate a bassline based on key, tempo, a chord progression and time signature.
    """
    validator = DurationValidator()
    mf = MIDIFile(1)
    track = 0
    time = 0
    mf.addTrackName(track, time, "Bassline")
    mf.addTempo(track, time, tempo)

    numerator, midi_denominator = get_midi_time_signature_values(time_signature)
    mf.addTimeSignature(track, time, numerator, midi_denominator, 24, 8)
    
    base_duration = validator.get_suggested_duration(time_signature, 'bass')
    beats_per_measure = numerator * base_duration
    total_beats = measures * beats_per_measure

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
    current_note = random.choice([note.midi for note in notes_to_use])

    while remaining_beats > 0:
        base_duration = validator.get_suggested_duration(time_signature, 'bass')
        duration = validator.get_valid_duration(
            base_duration,
            time_signature,
            remaining_beats,
            'bass'
        )
        current_note = random.choices(
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
                if random.random() < 0.5:
                    bassline[i] = melody[i]

    print("\t\t\tBassline: " + str(bassline))
    # Add notes to MIDI file
    current_time = 0
    for i in range(len(bassline)):
        note = bassline[i]
        # note_duration = note_durations[i]
        velocity = random.randint(70, 100)
        
        note_obj = pitch.Pitch()
        note_obj.midi = note
        note_obj.octave = 2
        
        mf.addNote(track, 0, note_obj.midi, current_time, note_durations[i], velocity)
        current_time += note_duration

    # Saves MIDI file
    directory = name.split('-')[0]
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = os.path.join(directory, name + "-bassline.mid")
    
    with open(filename, 'wb') as outf:
        mf.writeFile(outf)
    
    print("\t\t\tBassline midi file: " + filename)
        
    return filename

def beat_duration(signature: str, tempo: int) -> float:
    """
    Calculates the duration of a beat based on the time signature and BPM.
    """
    numerator, denominator = map(int, signature.split('/'))
    beat_length = 60 / tempo  # Duration of a quarter note
    return beat_length * (4 / denominator)

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
    swing_amount: float = 0.5
) -> Tuple[str, List[float]]:
    """
    Generates a drum pattern with configurable swing.
    
    Args:
        part: Part of the song (intro, verse, etc.)
        tempo: BPM
        time_signature: Time signature (ex: "4/4")
        measures: Number of bars
        name: Base name for generated files
        swing_amount: Swing amount (0.0 to 1.0, default 0.5 = no swing)
        
    Returns:
        Tuple[str, List[float]]: (MIDI file name, list of time notes)
    """
    validator = DurationValidator()
    beat_pattern_files = {
        "2/4": "beat_roll_patterns_24.txt",
        "4/4": "beat_roll_patterns_44.txt",
        "3/4": "beat_roll_patterns_34.txt",
        "6/8": "beat_roll_patterns_68.txt",
        "7/8": "beat_roll_patterns_78.txt",
        "12/8": "beat_roll_patterns_128.txt",
        "5/4": "beat_roll_patterns_54.txt"
    }

    mf = MIDIFile(1)
    track = 0
    time = 0

    mf.addTrackName(track, time, "Beat")
    mf.addTempo(track, time, tempo)

    numerator, midi_denominator = get_midi_time_signature_values(time_signature)
    mf.addTimeSignature(track, time, numerator, midi_denominator, 24, 8)

    base_duration = validator.get_suggested_duration(time_signature, 'beat')

    # MIDI values ​​for percussion instruments
    kick = 36
    snare = 38
    hihat = 42

    # Reads patterns from file
    filename = beat_pattern_files.get(time_signature)
    if not filename:
        raise ValueError(f"Time signature {time_signature} not supported.")

    beat_patterns = {}
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(':')
                song_part = parts[0].strip()
                pattern = [int(x) for x in parts[1].split(',')]
                if verify_beat_pattern(pattern, time_signature):
                    beat_patterns.setdefault(song_part, []).append(pattern)

    # default pattern
    if part not in beat_patterns or not beat_patterns[part]:
        base_pattern = [kick, hihat] if numerator == 2 else [kick, hihat, snare]
        beat_patterns[part] = [base_pattern]

    # Generates the pattern for the part
    beat_pattern = random.choice(beat_patterns[part])
    beat = beat_pattern * (measures - 1)

    # Adds roll to the end
    roll_part = part + "_roll"
    roll_pattern = random.choice(beat_patterns.get(roll_part, [beat_pattern]))
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

def save_beat_annotations(name, beat_annotations):
    # Extract the directory name from the song name
    instance_dir = os.path.dirname(name)
    output_file = os.path.join(instance_dir, f"{name}-beats.txt")
    with open(output_file, 'w') as f:
        for part, annotations in beat_annotations.items():
            timestamps = [f"{timestamp:.3f}" for timestamp in annotations]
            f.write(f"{part}: {', '.join(timestamps)}\n")

    print(f"Beat annotations saved to: {output_file}")

def generate_song_arrangement(structures_file: str = 'song_structures.json') -> Tuple[List[str], List[str]]:
    """
    Generate a musical arrangement based on common music structures.
    
    Args:
        structures_file: Path to the JSON file containing the song structures
        
    Returns:
        Tuple containing (unique parts, complete arrangement)
    """
    try:
        if not os.path.exists(structures_file):
            raise FileNotFoundError(f"Structures file not found: {structures_file}")
            
        with open(structures_file, 'r') as f:
            data = json.load(f)
            
        if 'common_structures' not in data:
            raise KeyError("Missing 'common_structures' in JSON file")
            
        structures = data['common_structures']
        
        if not structures or not isinstance(structures, list):
            raise ValueError("Invalid or empty structures list")
            
        result = random.choice(structures)
        
        # Generate a list of unique elements in the arrangement
        unique_elements = list(set(result))
        
        return unique_elements, result
        
    except (json.JSONDecodeError, FileNotFoundError, KeyError, ValueError) as e:
        # Default structure in case of error
        default_structure = ['intro', 'verse', 'chorus', 'outro']
        print(f"Warning: Using default structure due to error: {str(e)}")
        return list(set(default_structure)), default_structure
    
def read_instrument_probabilities(file_path):
    with open(file_path) as f:
        inst_probabilities = json.load(f)
    return inst_probabilities  

def get_random_sound_font(directory_path):
    sound_fonts = [f for f in os.listdir(directory_path) if f.endswith('.sf2')]
    file_return = random.choice(sound_fonts)
    return os.path.join(directory_path, file_return)

def get_levels(file_path):
    with open(file_path) as f:
        levels = json.load(f)
    return levels  

def generate_random_swing() -> float:
    """
    Generates a random swing value with a weighted distribution.
    Favors more musically appropriate values.
    
    Returns:
        float: Swing value between 0.5 and 0.75
    """
    # Distribution that favors moderate swings
    # 0.5 = no swing
    # 0.66 = traditional jazz
    # 0.75 = extreme swing 
    swing_weights = [
        (0.5, 0.3),   # 30% no swing
        (0.66, 0.5),  # 50% traditional jazz
        (0.75, 0.2)   # 20% extreme swing
    ]
    
    base_swing = random.choices(
        [s[0] for s in swing_weights],
        weights=[s[1] for s in swing_weights]
    )[0]
    
    # random variations
    variation = random.uniform(-0.02, 0.02)
    return min(0.75, max(0.5, base_swing + variation))

def create_effect(effect_class, parameters):
    # Unpack the parameters
    probability = parameters['probability']
    value_range = parameters['value_range']
    
    if random.random() < probability:
        kwargs = {param: random.uniform(value_range[param][0], value_range[param][1])
                  for param in value_range}
        return effect_class(**kwargs)
    return None

def generate_pedalboard(effect_params_file):
    # Load effect parameters from the JSON file
    with open(effect_params_file, 'r') as json_file:
        effect_params = json.load(json_file)
    # Define available audio effects and their parameter configurations
    # Each effect is paired with its corresponding parameters from the JSON config
    effects = [
        (Compressor, effect_params['compressor']),     # Controls dynamic range
        (Gain, effect_params['gain']),                 # Adjusts amplitude level
        (Chorus, effect_params['chorus']),             # Creates chorus/ensemble effect
        (LadderFilter, effect_params['ladder_filter']), # Moog-style filter
        (Phaser, effect_params['phaser']),             # Creates sweeping filter effect
        (Delay, effect_params['delay']),               # Adds echo/delay
        (Reverb, effect_params['reverb']),             # Adds room ambience
    ]
    
    # Initialize pedalboard by creating effect instances
    # Only effects that are successfully created (not None) are added
    # create_effect() function handles individual effect parameter setup
    board = Pedalboard([effect for effect in (create_effect(effect_class, parameters)
                                          for effect_class, parameters in effects)
                    if effect is not None])
    return board
    
def apply_fx_to_layer(wav_file, board):
    # Apply the pedalboard effects to the input file        
    with AudioFile(wav_file) as af:
        with AudioFile(wav_file+'_fx.wav', 'w', af.samplerate, af.num_channels) as of:        
            while af.tell() < af.frames:
                chunk = af.read(af.samplerate)
                effected = board(chunk, af.samplerate, reset=False)
                of.write(effected)
              
    return wav_file+'_fx.wav'

def pedalboard_info_json(board):
    pedals_and_parameters = []
    for pedal in board:
        attributes = dir(pedal)

        # Filter out the attributes that are not parameters
        parameters = [attr for attr in attributes if not attr.startswith("_") and not callable(getattr(pedal, attr))]
        
        pedal_info = {
            "name": pedal.__class__.__name__,
            "parameters": {}
        }
        
        for parameter in parameters:
            value = getattr(pedal, parameter)
            pedal_info['parameters'][str(parameter)] = str(value)               
        pedals_and_parameters.append(pedal_info)    
    return pedals_and_parameters
    
# Mix song parts and save the result to WAV files
def mix_and_save(harm_filename, bass_filename, melo_filename, beat_filename, name):
    # TODO: only render and mix the parts that are used in the song arrangement
    song_unique_parts, song_arrangement = generate_song_arrangement()
    print("Song arrangement: "+ str(song_arrangement) + "\n")
    number_of_parts = len(song_arrangement)
    song_parts = []
    part_layers = {}
    part_layers['intro'] = []
    part_layers['verse'] = []
    part_layers['chorus'] = []
    part_layers['bridge'] = []
    part_layers['outro'] = []
    part_counter = 0
    soundfonts = {}
    pedalboards = {}
    #TODO: configure soundfont directory 
    beat_soundfont = get_random_sound_font(str(os.path.join('sf','beat')))
    melody_soundfont = get_random_sound_font(str(os.path.join('sf','melody')))
    harmony_soundfont = get_random_sound_font(str(os.path.join('sf','harmony')))
    bassline_soundfont = get_random_sound_font(str(os.path.join('sf','bassline')))
    soundfonts['beat'] = beat_soundfont
    soundfonts['melody'] = melody_soundfont
    soundfonts['harmony'] = harmony_soundfont
    soundfonts['bassline'] = bassline_soundfont    
    print("Beat soundfont: " + beat_soundfont)
    print("Melody soundfont: " + melody_soundfont)
    print("Harmony soundfont: " + harmony_soundfont)
    print("Bassline soundfont: " + bassline_soundfont)
    beat_board = generate_pedalboard('beat_fx.json')
    melody_board = generate_pedalboard('melody_fx.json')
    harmony_board = generate_pedalboard('harmony_fx.json')
    bassline_board = generate_pedalboard('bassline_fx.json')    
    pedalboards['beat'] = pedalboard_info_json(beat_board)
    pedalboards['melody'] = pedalboard_info_json(melody_board)
    pedalboards['harmony'] = pedalboard_info_json(harmony_board)
    pedalboards['bassline'] = pedalboard_info_json(bassline_board)
    print("Beat pedalboard: " + str(beat_board))
    print("Melody pedalboard: " + str(melody_board))
    print("Harmony pedalboard: " + str(harmony_board))
    print("Bassline pedalboard: " + str(bassline_board))
    # TODO: gather all file references in a single config file
    inst_proba = read_instrument_probabilities('inst_probabilities.json')
    levels = {}
    levels = get_levels('levels.json')
    print("Levels: " + str(levels))
    beat_part_mix = {}
    melody_part_mix = {}
    harmony_part_mix = {}
    bassline_part_mix = {}
    # Define which layers will be used for each part
    for part in song_unique_parts:
        beat_proba = float(inst_proba[part]['beat'])
        melody_proba = float(inst_proba[part]['melody'])
        harmony_proba = float(inst_proba[part]['harmony'])
        bassline_proba = float(inst_proba[part]['bassline'])
        beat_part_mix[part] = (random.random() <= beat_proba) 
        melody_part_mix[part] = (random.random() <= melody_proba)
        harmony_part_mix[part] = (random.random() <= harmony_proba)
        bassline_part_mix[part] = (random.random() <= bassline_proba)    
    print("Mixing song parts...")
    song_transitions = []
    song_time = 0
    for part in song_arrangement:
        this_transition = [part, song_time]
        song_transitions.append(this_transition)
        part_counter += 1
        print("Mixing part: " + part + (' (' + str(part_counter) + ' of ' + str(number_of_parts) + ')'))        
        # Render each MIDI file to an audio file using the chosen soundfont
        beat_wav = 'beat' + "-" + str(part_counter) + "-" + part + ".wav"
        beat_wav = os.path.join(name, beat_wav)
        FluidSynth(beat_soundfont).midi_to_audio(beat_filename[part], beat_wav)
        melo_wav = 'melody' + "-" + str(part_counter) + "-" + part + ".wav"
        melo_wav = os.path.join(name, melo_wav)
        FluidSynth(melody_soundfont).midi_to_audio(melo_filename[part], melo_wav)
        harm_wav = 'harmony' + "-" + str(part_counter) + "-" + part + ".wav"
        harm_wav = os.path.join(name, harm_wav)
        FluidSynth(harmony_soundfont).midi_to_audio(harm_filename[part], harm_wav)
        bass_wav = 'bassline' + "-" + str(part_counter) + "-" + part + ".wav"
        bass_wav = os.path.join(name, bass_wav)
        FluidSynth(bassline_soundfont).midi_to_audio(bass_filename[part], bass_wav)
        # Load the rendered audio files, applying the effects defined in the JSON files
        # TODO: optimize it so that the fx are only applied to the used layers
        beat = AudioSegment.from_wav(apply_fx_to_layer(beat_wav, beat_board))
        melody = AudioSegment.from_wav(apply_fx_to_layer(melo_wav, melody_board))
        harmony = AudioSegment.from_wav(apply_fx_to_layer(harm_wav, harmony_board))
        bassline = AudioSegment.from_wav(apply_fx_to_layer(bass_wav, bassline_board))
        # Volume and panning for each layer
        beat.volume = float(levels[part]['beat']['volume'])
        melody.volume = float(levels[part]['melody']['volume'])
        harmony.volume = float(levels[part]['harmony']['volume'])
        bassline.volume = float(levels[part]['bassline']['volume'])
        beat.pan(float(levels[part]['beat']['panning']))
        melody.pan(float(levels[part]['melody']['panning']))
        harmony.pan(float(levels[part]['harmony']['panning']))
        bassline.pan(float(levels[part]['bassline']['panning']))      
        # Create an empty AudioSegment to use as the initial mix
        mix = AudioSegment.silent(duration=beat.duration_seconds*1000)
        # Overlay each track onto the mix based on its probability value        
        # TODO: if layer is chosen, apply effects (considering probability) and mix
        # TODO: create a data structure to store which layers are used in each part
        # Mix the audio files together
        if beat_part_mix[part]:
            mix = mix.overlay(beat)
            if 'beat' not in part_layers[part]:
                part_layers[part].append('beat')
            print("Beat added to mix: "+part)
        if melody_part_mix[part]:
            mix = mix.overlay(melody)
            if 'melody' not in part_layers[part]:
                part_layers[part].append('melody')
            print("Melody added to mix: "+part)
        if harmony_part_mix[part]:
            mix = mix.overlay(harmony)  
            if 'harmony' not in part_layers[part]:
                part_layers[part].append('harmony')         
            print("Harmony added to mix: "+part)
        if bassline_part_mix[part]:
            mix = mix.overlay(bassline)
            if 'bassline' not in part_layers[part]:
                part_layers[part].append('bassline')
            print("Bassline added to mix: "+part)
        # Save the mixed audio to the output file
        part_mix_file = name + '-' + str(part_counter) + '.wav'
        part_mix_file = os.path.join(name, part_mix_file) 
        mix.export(part_mix_file, format='wav')
        song_parts.append(part_mix_file)
        song_time = song_time + mix.duration_seconds
    
    this_transition = ['end', song_time]
    song_transitions.append(this_transition)
    # Iterate through song_parts and concatenate them into a single file
    song = AudioSegment.from_wav(song_parts[0])
    for part_wav in song_parts[1:]:
        song += AudioSegment.from_wav(part_wav)
    # Save the song as a wav file
    song_file_wav = name + '.wav'
    song_file_wav = os.path.join(name, song_file_wav)
    song.export(song_file_wav, format='wav')
    print("Song saved as: " + song_file_wav)
    # Clean the wav parts
    for part_wav in song_parts:
        os.remove(part_wav)
        
    return song_file_wav, song_arrangement, song_transitions, soundfonts, pedalboards, part_layers

def generate_random_key():
    # https://www.digitaltrends.com/music/whats-the-most-popular-music-key-spotify/
    # https://web.archive.org/web/20190426230344/https://insights.spotify.com/us/2015/05/06/most-popular-keys-on-spotify/
    # https://forum.bassbuzz.com/t/most-used-keys-on-spotify/5886

    key_ranges = [(0.107, 'G'), (0.209, 'C'), (0.296, 'D'), (0.357, 'A'), (0.417, 'C#'), (0.47, 'F'),
                  (0.518, 'Am'), (0.561, 'G#'), (0.603, 'Em'), (0.645, 'Bm'), (0.681, 'E'), (0.716, 'A#'),
                  (0.748, 'A#m'), (0.778, 'Fm'), (0.805, 'F#'), (0.831, 'B'), (0.857, 'Gm'), (0.883, 'Dm'),
                  (0.908, 'F#m'), (0.932, 'D#'), (0.956, 'Cm'), (0.977, 'C#m'), (0.989, 'G#m'), (1.0, 'D#m')
    ]
    dice = random.random()
    for prob, key in key_ranges:
        if dice < prob:
            return key    

def generate_random_tempo():
    # https://blog.musiio.com/2021/08/19/which-musical-tempos-are-people-streaming-the-most/
    tempo_ranges = [(0.0183, 60, 70), (0.0454, 70, 80), (0.1849, 80, 90), (0.3721, 90, 100),
                    (0.4817, 100, 110), (0.5747, 110, 120), (0.7048, 120, 130), (0.7917, 130, 140),
                    (0.8958, 140, 150), (0.9739, 150, 160), (1.0, 160, 170)]
    dice = random.random()
    for prob, min_tempo, max_tempo in tempo_ranges:
        if dice < prob:
            return random.randint(min_tempo, max_tempo)

def generate_random_time_signature():
    """
    Generates a random time signature based on weighted probabilities.
    The probabilities are based on the frequency of use in popular music
    """
    time_signature_ranges = [
        (0.50, '4/4'),  # More commom (50% of the dataset will be 4/4)
        (0.65, '3/4'),  # Less common (15% of the dataset will be 3/4)
        (0.75, '2/4'),  # Less common (10% of the dataset will be 2/4)
        (0.85, '6/8'),  # Less common (10% of the dataset will be 6/8)
        (0.90, '12/8'), # Less common (5% of the dataset will be 12/8)
        (0.95, '7/8'),  # Less common (5% of the dataset will be 7/8)
        (1.00, '5/4')   # Less common (5% of the dataset will be 5/4)
    ]
    
    dice = random.random()
    for prob, time_signature in time_signature_ranges:
        if dice < prob:
            return time_signature

def time_signature_alternative(base_time_signature):
    """
    Generates a significant variation of the given time signature.
    Variations are based on common musical relationships and natural transitions.
    """
    variations = {
        "4/4": ["2/4", "3/4", "6/8", "12/8"],     
        "3/4": ["6/8", "4/4", "2/4", "12/8"],     
        "2/4": ["4/4", "6/8", "3/4"],             
        "6/8": ["12/8", "3/4", "4/4", "2/4"],     
        "12/8": ["6/8", "4/4", "3/4"],            
        "7/8": ["4/4", "6/8", "5/4"],             
        "5/4": ["4/4", "7/8", "3/4"]
    }

    # Returns a random variation if available
    if base_time_signature in variations:
        return random.choice(variations[base_time_signature])
    
    # Fallback to 4/4 if the time signature is not recognized
    return "4/4"

def generate_song_measures(time_signature: str, time_signature_variation: float) -> Tuple[Dict[str, int], Dict[str, str]]:
    """
    Generates a set of measures for each part of a song, based on a given time signature and variation.
    """

    base_lengths = {
        'intro': random.choice([8, 16]),
        'verse': random.choice([16, 32]),
        'chorus': random.choice([16, 32]),
        'bridge': random.choice([8, 16]),
        'outro': random.choice([8, 16])
    }
    
    # Sets time signatures for each part
    if random.random() < time_signature_variation:
        signatures = {
            'intro': random.choice([time_signature, time_signature_alternative(time_signature)]),
            'verse': random.choice([time_signature, time_signature_alternative(time_signature)]),
            'chorus': random.choice([time_signature, time_signature_alternative(time_signature)]),
            'bridge': random.choice([time_signature, time_signature_alternative(time_signature)]),
            'outro': random.choice([time_signature, time_signature_alternative(time_signature)])
        }
    else:
        signatures = {part: time_signature for part in base_lengths.keys()}
    
    # Adjusts number of bars based on time signature
    measures = {
        part: calculate_measures_for_time_signature(length, signatures[part])
        for part, length in base_lengths.items()
    }
    
    return measures, signatures
    

def create_song(
    key: str,
    tempo: int,
    song_signatures: Dict[str, str],
    measures: Dict[str, int],
    name: str,
    chord_pat_file: str,
    swing_amount: float
) -> Dict:

    song_info = {}
    song_info['key'] = key
    song_info['tempo'] = tempo
    song_info['measures'] = measures
    song_info['time_signatures'] = song_signatures
    song_info['name'] = name
    song_info['swing_amount'] = swing_amount  

    ha = {}
    ba = {}
    me = {}
    be = {}

    song_name = name
    start_time = time.time()
    
    print(f"Generating song with swing amount: {swing_amount}")
    
    # Generates the musical components
    ha, ba, me, be, an = generate_song_parts(
        key=key,
        tempo=tempo,
        song_signatures=song_signatures,
        song_measures=measures,
        name=song_name,
        chord_pat_file=chord_pat_file,
        swing_amount=swing_amount
    )
    
    # Mix components
    wav_name, arrangement, transitions, soundfonts, pedalboards, part_layers = mix_and_save(
        ha, ba, me, be, song_name
    )
    
    end_time = time.time()

    
    song_info['file_name'] = wav_name
    song_info['arrangement'] = arrangement
    song_info['transitions'] = transitions
    song_info['soundfonts'] = soundfonts
    song_info['pedalboards'] = pedalboards
    song_info['part_layers'] = part_layers
    
    # Calculates musicality score
    score, component_scores = musicality_score.get_musicality_score(wav_name)
    song_info['musicality'] = {
        'score': float(score),
        'components': {
            k: float(v) for k, v in component_scores.items()
        }
    }
    
    elapsed_time = end_time - start_time
    print(f'Elapsed time: {elapsed_time:.2f} seconds')
    print(f'Musicality Analysis:')
    print(f'Score: {score:.2f}')
    print('Component Scores:')
    for component, value in component_scores.items():
        print(f'{component:>10}: {value:.2f}')
    
    # Saves metadata
    json_file = os.path.join(name, name + '.json')
    with open(json_file, 'w') as outfile:
        json.dump(song_info, outfile, indent=4)
    
    
    return song_info

def generate_song_parts(
    key: str,
    tempo: int,
    song_signatures: Dict[str, str],
    song_measures: Dict[str, int],
    name: str,
    chord_pat_file: str,
    swing_amount: float
) -> Tuple[Dict, Dict, Dict, Dict, Dict]:

    harm_filename, bass_filename, melo_filename, beat_filename, beat_annotations = {}, {}, {}, {}, {}

    for part, measures in song_measures.items():
        print(f"Generating part: {part} ({measures} measures)")
        name_part = f"{name}-{part}"
        time_signature = song_signatures[part]

        chord_progression, harm_filename[part] = generate_chord_progression(
            key, tempo, time_signature, measures, name_part, part, chord_pat_file
        )
        
        melody, melo_filename[part] = generate_melody(
            key, tempo, time_signature, measures, name_part, part, chord_progression
        )
        
        bass_filename[part] = generate_bassline(
            key, tempo, time_signature, measures, name_part, part, chord_progression, melody
        )
        
        beat_filename[part], beat_annotations[part] = generate_beat(
            part, tempo, time_signature, measures, name_part, swing_amount
        )

    return harm_filename, bass_filename, melo_filename, beat_filename, beat_annotations

def generate_song(id: int):
    """
    Generates a complete song with all its components.
    
    Args:
        id: Music id
    """
    print(f"Generating song #{str(id)}")
    
    # Basic musical parameters
    key = generate_random_key()
    tempo = generate_random_tempo()
    time_signature = generate_random_time_signature()
    time_signature_variation = 1.0  # 100% chance of time signature variation
    swing_amount = generate_random_swing()
    swing_amount = min(0.75, max(0.5, float(swing_amount)))
    
    # Generates valid measurements and time signatures
    while True:
        measures, signatures = generate_song_measures(time_signature, time_signature_variation)
        if validate_measures(measures, signatures):
            break
    
    # Generates unique name for the song
    now = datetime.now()
    song_name = f"{datetime.now().strftime('%Y%m%d%H%M%S%f')}_{uuid.uuid4()}"
    song_name = song_name[:20]  # 20 chars
    
    # Generates the music components
    song_info = create_song(
        key=key,
        tempo=tempo,
        song_signatures=signatures,
        measures=measures,
        name=song_name,
        chord_pat_file='chord_patterns.txt',
        swing_amount=swing_amount
    )
    
    return song_info

def get_google_drive_service():
    """Set up and return Google Drive service."""
    SCOPES = ['https://www.googleapis.com/auth/drive.file']
    creds = None

    # Load existing credentials if available
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)

    # If credentials are invalid or don't exist, get new ones
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        
        # Save credentials
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    return build('drive', 'v3', credentials=creds)

def upload_to_drive(service, file_path, folder_id=None):
    """Upload a file to Google Drive."""
    file_metadata = {
        'name': os.path.basename(file_path),
        'parents': [folder_id] if folder_id else []
    }
    
    media = MediaFileUpload(
        file_path,
        resumable=True
    )
    
    file = service.files().create(
        body=file_metadata,
        media_body=media,
        fields='id'
    ).execute()
    
    return file.get('id')

def cleanup_local_files(directory):
    """Remove local files after upload."""
    if os.path.exists(directory):
        for root, dirs, files in os.walk(directory, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(directory)

# Example usage

for i in range(997):
    generate_song(i)
