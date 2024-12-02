from midiutil import MIDIFile
from music21 import *
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

from typing import Tuple, Dict, List
import math


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
    
    # Verifica se o comprimento do padrão corresponde à assinatura
    if denominator == 8 and numerator % 3 == 0:
        return len(pattern) == numerator  # 6 beats para 6/8, etc.
    else:
        return len(pattern) == numerator  # 4 beats para 4/4, etc.

def calculate_measures_for_time_signature(base_length: int, time_signature: str) -> int:
    # TODO: acoplar sistema melhorado de controle de compassos
    numerator, denominator = map(int, time_signature.split('/'))
    
    if denominator == 8 and numerator % 3 == 0:
        return base_length * 2  # Dobra para compassos compostos
    elif numerator == 2:
        return base_length * 2  # Dobra para 2/4
    elif numerator == 3:
        return int(base_length * 4/3)  # Ajusta para 3/4        
    return base_length  # 4/4 mantém original

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
            0.5,  # Colcheia (unidade básica)
            1.5,  # Colcheia pontuada (grupo de 3)
            3.0   # Semínima pontuada (grupo completo)
        ]
    elif numerator == 3:  # 3/4
        return [
            0.5,  # Colcheia
            1.0,  # Semínima
            1.5   # Semínima pontuada
        ]
    else:  # 2/4, 4/4
        return [
            0.5,  # Colcheia
            1.0,  # Semínima
            2.0   # Mínima
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
            beats_per_measure = numerator // 3  # Agrupa em unidades de 3 colcheias 
        
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

def generate_melody(key, tempo, time_signature, measures, name, part, chord_progression):
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
    numerator, midi_denominator = get_midi_time_signature_values(time_signature)
    mf.addTimeSignature(track, time, numerator, midi_denominator, 24, 8)
    
    # Get base note duration
    # base_duration = get_note_duration(time_signature) 
    base_duration = validator.get_suggested_duration(time_signature, 'melody')
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

    current_note = random.choice([note.midi for note in notes_to_use])

    # Choose the initial note randomly
    while remaining_beats > 0:
        current_note = random.choices(
            population=list(transition_matrix[current_note].keys()),
            weights=list(transition_matrix[current_note].values())
        )[0]

        # Chooses proper note duration
        possible_durations = get_melody_durations(time_signature)
        # raw_duration = random.choice(possible_durations)
        raw_duration = random.choice(get_melody_durations(time_signature))
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
        print("\n\nGenerated melody has invalid timing structure\n")
    
    # Add notes to MIDI file
    for i in range(len(melody)):
        note = melody[i]
        velocity = random.randint(70, 100)
        mf.addNote(track, 0, note, time, note_durations[i], velocity)
        time += note_duration

    print("\t\t\tMelody: " + str(melody))

    # Save MIDI file
    directory = name.split('-')[0]
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = os.path.join(directory, name + "-melody.mid")
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

def generate_beat(tempo, time_signature, measures, name, part):
    """
    Generate a drum beat based on a time signature.
    """
    validator = DurationValidator()
    beat_pattern_files = {
        "2/4": "beat_roll_patterns_24.txt",
        "4/4": "beat_roll_patterns_44.txt",
        "3/4": "beat_roll_patterns_34.txt",
        "5/4": "beat_roll_patterns_54.txt",
        "6/8": "beat_roll_patterns_68.txt",
        "7/8": "beat_roll_patterns_78.txt",
        "12/8": "beat_roll_patterns_128.txt"
    }

    mf = MIDIFile(1)
    track = 0
    time = 0

    mf.addTrackName(track, time, "Beat")
    mf.addTempo(track, time, tempo)

    # Adds correct time signature
    numerator, midi_denominator = get_midi_time_signature_values(time_signature)
    mf.addTimeSignature(track, time, numerator, midi_denominator, 24, 8)

    base_duration = validator.get_suggested_duration(time_signature, 'beat')

    kick = 36
    snare = 38
    hihat = 42


    # Reads beat patterns from file
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

    # Creates a basic pattern if not found
    if part not in beat_patterns or not beat_patterns[part]:
        base_pattern = [kick, hihat] if numerator == 2 else [kick, hihat, snare]
        beat_patterns[part] = [base_pattern]

    # Generates pattern for the part
    beat_pattern = random.choice(beat_patterns[part])
    beat = beat_pattern * (measures - 1)

    # Adds a roll at the end
    roll_part = part + "_roll"
    roll_pattern = random.choice(beat_patterns.get(roll_part, [beat_pattern]))
    beat.extend(roll_pattern)

    # Adds notes to MIDI file
    current_time = 0
    for drum_hit  in beat:
        if drum_hit != 0:
            mf.addNote(track, 9, drum_hit , current_time, base_duration, 100)
        current_time += base_duration

    # Saves MIDI file
    directory = name.split('-')[0]
    if not os.path.exists(directory):
        os.makedirs(directory)
    midi_filename = os.path.join(directory, f"{name}-beat.mid")

    with open(midi_filename, 'wb') as outf:
        mf.writeFile(outf)

    return midi_filename

def generate_song_parts(key, tempo, song_signatures, song_measures, name, chord_pat_file):
    """
    Generate all parts of a song based on key, tempo, time signatures and measures.
    """
    harm_filename, bass_filename, melo_filename, beat_filename = {}, {}, {}, {}

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
        beat_filename[part] = generate_beat(
            tempo, time_signature, measures, name_part, part
        )

    return harm_filename, bass_filename, melo_filename, beat_filename


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
    # Create a list of effects with their respective probabilities and value ranges
    effects = [
        (Compressor, effect_params['compressor']),
        (Gain, effect_params['gain']),
        (Chorus, effect_params['chorus']),
        (LadderFilter, effect_params['ladder_filter']),
        (Phaser, effect_params['phaser']),
        (Delay, effect_params['delay']),
        (Reverb, effect_params['reverb']),
    ]
    
    # Create a new pedalboard with the specified effects
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

# Create song file and metadata
def create_song(key, tempo, song_signatures, measures, name, chord_pat_file):
    song_info = {}
    song_info['key'] = key
    song_info['tempo'] = tempo
    # song_info['time_signature'] = time_signature
    song_info['measures'] = measures
    song_info['time_signatures'] = song_signatures
    song_info['name'] = name

    ha = {}
    ba = {}
    me = {}
    be = {}

    song_name = name

    start_time = time.time()
    
    ha, ba, me, be = generate_song_parts(key, tempo, song_signatures, measures, song_name, chord_pat_file)
    wav_name, arrangement, transitions, soundfonts, pedalboards, part_layers = mix_and_save(ha, ba, me, be, song_name)
    
    end_time = time.time()
    
    song_info['file_name'] = wav_name
    song_info['arrangement'] = arrangement
    song_info['transitions'] = transitions
    song_info['soundfonts'] = soundfonts
    song_info['pedalboards'] = pedalboards
    song_info['part_layers'] = part_layers
    # Novo sistema de musicality score
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
    json_file = os.path.join(name, name + '.json')
    
    print('Annotations: ' + json_file)
    
    with open(json_file, 'w') as outfile:
        json.dump(song_info, outfile, indent=4)

    # TODO: clean temp files in a better way (ATS)
    # TODO generate stems for each layer / the whole song
    midi_del = "*.mid"
    midi_path = os.path.join(name, midi_del)
    midi_files = glob.glob(midi_path)
    # for midi_file in midi_files:
        # os.remove(midi_file)
    
    wav_del = "beat-*.wav" 
    wav_path = os.path.join(name, wav_del)
    wav_files = glob.glob(wav_path)
    # for wav_file in wav_files:
        # os.remove(wav_file)

    wav_del = "bassline-*.wav" 
    wav_path = os.path.join(name, wav_del)
    wav_files = glob.glob(wav_path)
    # for wav_file in wav_files:
        # os.remove(wav_file)

    wav_del = "harmony-*.wav" 
    wav_path = os.path.join(name, wav_del)
    wav_files = glob.glob(wav_path)
    # for wav_file in wav_files:
        # os.remove(wav_file)

    wav_del = "melody-*.wav" 
    wav_path = os.path.join(name, wav_del)
    wav_files = glob.glob(wav_path)
    # for wav_file in wav_files:
        # os.remove(wav_file)

    
    return wav_name, json_file

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
    # Comprimentos base (para 4/4)
    base_lengths = {
        'intro': random.choice([8, 16]),
        'verse': random.choice([16, 32]),
        'chorus': random.choice([16, 32]),
        'bridge': random.choice([8, 16]),
        'outro': random.choice([8, 16])
    }
    
    # Define assinaturas de tempo para cada parte
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
    
    # Ajusta número de compassos baseado na assinatura de tempo
    measures = {
        part: calculate_measures_for_time_signature(length, signatures[part])
        for part, length in base_lengths.items()
    }
    
    return measures, signatures
    

def generate_song(id):
    print("Generating song #" + str(id))
    key = generate_random_key()
    tempo = generate_random_tempo()
    time_signature = generate_random_time_signature()
    time_signature_variation = 1.0  # 100% chance of varying the time signature
    # song_measures, song_signatures = generate_song_measures(time_signature, time_signature_variation)
    while True:
        measures, signatures = generate_song_measures(time_signature, time_signature_variation)
        if validate_measures(measures, signatures):
            break
    
    now = datetime.now()
    song_name = f"{datetime.now().strftime('%Y%m%d%H%M%S%f')}_{uuid.uuid4()}"
    # Truncate song name to 20 characters
    song_name = song_name[:20]
    create_song(key, tempo, signatures, measures, song_name, 'chord_patterns.txt')

# Example usage

for i in range(1):
    generate_song(i)
