from midiutil import MIDIFile
from music21 import roman, scale, pitch
from pydub import AudioSegment
from midi2audio import FluidSynth
from datetime import datetime
from pedalboard import Pedalboard, Compressor, Gain, Chorus, LadderFilter, Phaser, Delay, Reverb
from pedalboard.io import AudioFile
import logging
import time
import json
import random
import os
import uuid
import musicality_score
from musicgen.duration_validator import DurationValidator, NoteValue
import config
from timesig import TimeSignatureRegistry
from musicgen.sampler import (
    SongParams,
    generate_random_key,
    generate_random_tempo,
    generate_random_time_signature,
    generate_random_swing,
    generate_song_measures,
    time_signature_alternative,
    generate_song_arrangement,
    validate_measures_dict,
)
from musicgen.generators.chord import generate_chord_progression
from musicgen.generators.melody import generate_melody
from musicgen.generators.bassline import generate_bassline
from musicgen.generators.beat import (
    generate_beat,
    beat_duration,
    calculate_swing_offset,
)

from typing import Tuple, Dict, List, Optional
import math

logger = logging.getLogger(__name__)

# D-08: single module-level RNG threaded to every extracted sampler call site
# this phase. Phase 5 (R-P7) replaces this with derive_sample_seed + make_rngs
# so every generated sample gets its own deterministic RNG hierarchy.
_rng = random.Random()

# Preserve the pre-existing ``validate_measures`` name for back-compat — it's
# the same function as ``validate_measures_dict`` in musicgen.sampler.
validate_measures = validate_measures_dict


def verify_pattern_for_time_signature(chord_pattern: List[str], time_signature: str) -> bool:
    """Checks if the chord pattern is appropriate for the time signature.
    Delegates to TimeSignatureRegistry per R-S6."""
    return TimeSignatureRegistry.lookup(time_signature).verify_chord_pattern_length(len(chord_pattern))


def verify_beat_pattern(pattern: List[int], time_signature: str) -> bool:
    """Checks if the beat pattern matches the time signature.
    Delegates to TimeSignatureRegistry per R-S6."""
    return TimeSignatureRegistry.lookup(time_signature).verify_beat_pattern_length(len(pattern))

def calculate_measures_for_time_signature(base_length: int, time_signature: str) -> int:
    """Delegates to TimeSignatureRegistry per R-S6."""
    return TimeSignatureRegistry.lookup(time_signature).measures_for(base_length)

# ``validate_measures`` was extracted to ``musicgen.sampler.validate_measures_dict``
# and re-aliased at module top as ``validate_measures = validate_measures_dict``
# (03-03 shim) — keep callers working with zero changes to their arg lists.

def get_midi_time_signature_values(time_signature: str) -> Tuple[int, int]:
    """Convert musical time signature to MIDI format values.
    Delegates to TimeSignatureRegistry per R-S6."""
    spec = TimeSignatureRegistry.lookup(time_signature)
    return spec.numerator, spec.midi_denominator_power

def get_note_duration(time_signature: str) -> float:
    """Calculate base note duration for given time signature.
    Delegates to TimeSignatureRegistry per R-S6."""
    return TimeSignatureRegistry.lookup(time_signature).primary_beat_duration

def get_note_durations(time_signature: str) -> dict:
    """Calculate note durations for different note values.
    Delegates to TimeSignatureRegistry per R-S6."""
    return TimeSignatureRegistry.lookup(time_signature).note_duration_map()

def get_melody_durations(time_signature: str) -> list:
    """Get appropriate note durations for melody.
    Delegates to TimeSignatureRegistry per R-S6."""
    return list(TimeSignatureRegistry.lookup(time_signature).melody_duration_candidates)

# ``generate_chord_progression``, ``generate_melody``, ``generate_bassline``,
# ``generate_beat``, ``beat_duration``, and ``calculate_swing_offset`` were
# extracted to ``musicgen.generators.*`` (Plan 03-04). They are re-imported at
# module top; callers inside ``generate_song_parts`` pass ``_rng`` per D-08.

def save_beat_annotations(name, beat_annotations):
    # Extract the directory name from the song name
    instance_dir = os.path.dirname(name)
    output_file = os.path.join(instance_dir, f"{name}-beats.txt")
    with open(output_file, 'w') as f:
        for part, annotations in beat_annotations.items():
            timestamps = [f"{timestamp:.3f}" for timestamp in annotations]
            f.write(f"{part}: {', '.join(timestamps)}\n")

    logger.info("Beat annotations saved to: %s", output_file)

# ``generate_song_arrangement`` was extracted to ``musicgen.sampler`` (03-03).
# Callers pass ``_rng`` as the new first positional argument.

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

# ``generate_random_swing`` was extracted to ``musicgen.sampler`` (03-03).
# Callers pass ``_rng``.

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
def mix_and_save(harm_filename, bass_filename, melo_filename, beat_filename, name,
                 song_unique_parts, song_arrangement, *, cfg: config.Config = None):
    # Arrangement is now produced once upstream (see create_song) and threaded through.
    # See PITFALLS P-A / R-S3: do NOT re-roll the arrangement here; doing so
    # re-rolls RNG and can decouple the rendered audio from the MIDI structure.
    # TODO (later phase): only render and mix the parts that are used in the song arrangement
    logger.info("Song arrangement: %s", song_arrangement)
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
    _cfg = cfg if cfg is not None else config.Config()
    beat_soundfont = get_random_sound_font(_cfg.sf_layer_dir('beat'))
    melody_soundfont = get_random_sound_font(_cfg.sf_layer_dir('melody'))
    harmony_soundfont = get_random_sound_font(_cfg.sf_layer_dir('harmony'))
    bassline_soundfont = get_random_sound_font(_cfg.sf_layer_dir('bassline'))
    soundfonts['beat'] = beat_soundfont
    soundfonts['melody'] = melody_soundfont
    soundfonts['harmony'] = harmony_soundfont
    soundfonts['bassline'] = bassline_soundfont    
    logger.info("Beat soundfont: %s", beat_soundfont)
    logger.info("Melody soundfont: %s", melody_soundfont)
    logger.info("Harmony soundfont: %s", harmony_soundfont)
    logger.info("Bassline soundfont: %s", bassline_soundfont)
    beat_board = generate_pedalboard(_cfg.fx_files['beat'])
    melody_board = generate_pedalboard(_cfg.fx_files['melody'])
    harmony_board = generate_pedalboard(_cfg.fx_files['harmony'])
    bassline_board = generate_pedalboard(_cfg.fx_files['bassline'])    
    pedalboards['beat'] = pedalboard_info_json(beat_board)
    pedalboards['melody'] = pedalboard_info_json(melody_board)
    pedalboards['harmony'] = pedalboard_info_json(harmony_board)
    pedalboards['bassline'] = pedalboard_info_json(bassline_board)
    logger.debug("Beat pedalboard: %s", beat_board)
    logger.debug("Melody pedalboard: %s", melody_board)
    logger.debug("Harmony pedalboard: %s", harmony_board)
    logger.debug("Bassline pedalboard: %s", bassline_board)
    inst_proba = read_instrument_probabilities(_cfg.inst_probabilities_file)
    levels = {}
    levels = get_levels(_cfg.levels_file)
    logger.debug("Levels: %s", levels)
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
    logger.info("Mixing song parts...")
    song_transitions = []
    song_time = 0
    for part in song_arrangement:
        this_transition = [part, song_time]
        song_transitions.append(this_transition)
        part_counter += 1
        logger.info("Mixing part: %s (%s of %s)", part, part_counter, number_of_parts)        
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
        # Volume and panning for each layer (R-S4 / PITFALLS P-B fix).
        # `.volume =` was a no-op (read-only property); `.pan()` returns a new
        # segment so its return must be captured. Values in levels.json are
        # linear amplitudes; convert to dB via 20*log10(v) (clamped to avoid log(0)).
        def _lin_to_db(v: float) -> float:
            return 20.0 * math.log10(max(float(v), 1e-6))
        beat = beat.apply_gain(_lin_to_db(levels[part]['beat']['volume']))
        melody = melody.apply_gain(_lin_to_db(levels[part]['melody']['volume']))
        harmony = harmony.apply_gain(_lin_to_db(levels[part]['harmony']['volume']))
        bassline = bassline.apply_gain(_lin_to_db(levels[part]['bassline']['volume']))
        beat = beat.pan(float(levels[part]['beat']['panning']))
        melody = melody.pan(float(levels[part]['melody']['panning']))
        harmony = harmony.pan(float(levels[part]['harmony']['panning']))
        bassline = bassline.pan(float(levels[part]['bassline']['panning']))
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
            logger.debug("Beat added to mix: %s", part)
        if melody_part_mix[part]:
            mix = mix.overlay(melody)
            if 'melody' not in part_layers[part]:
                part_layers[part].append('melody')
            logger.debug("Melody added to mix: %s", part)
        if harmony_part_mix[part]:
            mix = mix.overlay(harmony)  
            if 'harmony' not in part_layers[part]:
                part_layers[part].append('harmony')         
            logger.debug("Harmony added to mix: %s", part)
        if bassline_part_mix[part]:
            mix = mix.overlay(bassline)
            if 'bassline' not in part_layers[part]:
                part_layers[part].append('bassline')
            logger.debug("Bassline added to mix: %s", part)
        # Save the mixed audio to the output file
        part_mix_file = name + '-' + str(part_counter) + '.wav'
        part_mix_file = os.path.join(name, part_mix_file) 
        mix.export(part_mix_file, format='wav')
        song_parts.append(part_mix_file)
        song_time = song_time + mix.duration_seconds
    
    this_transition = ['end', song_time]
    song_transitions.append(this_transition)
    # Iterate through song_parts and concatenate them into a single file
    if not song_parts:
        raise RuntimeError("No song parts were rendered; cannot assemble final mix")
    song = AudioSegment.from_wav(song_parts[0])
    for part_wav in song_parts[1:]:
        song += AudioSegment.from_wav(part_wav)
    # Save the song as a wav file
    song_file_wav = name + '.wav'
    song_file_wav = os.path.join(name, song_file_wav)
    song.export(song_file_wav, format='wav')
    logger.info("Song saved as: %s", song_file_wav)
    # Clean the wav parts
    for part_wav in song_parts:
        os.remove(part_wav)
        
    return song_file_wav, song_arrangement, song_transitions, soundfonts, pedalboards, part_layers

# The five sampler functions ``generate_random_key``, ``generate_random_tempo``,
# ``generate_random_time_signature``, ``time_signature_alternative``, and
# ``generate_song_measures`` were extracted to ``musicgen.sampler`` (03-03).
# They are re-imported at module top; callers pass ``_rng`` per D-08.

def create_song(
    key: str,
    tempo: int,
    song_signatures: Dict[str, str],
    measures: Dict[str, int],
    name: str,
    chord_pat_file: str,
    swing_amount: float,
    cfg: config.Config = None,
) -> Dict:

    song_info = {}
    song_info['key'] = key
    song_info['tempo'] = tempo
    song_info['measures'] = measures
    song_info['time_signatures'] = song_signatures
    song_info['name'] = name
    song_info['swing_amount'] = swing_amount

    song_name = name
    start_time = time.time()
    
    logger.info("Generating song with swing amount: %s", swing_amount)

    # Compute arrangement ONCE for the whole song (R-S3 / PITFALLS P-A).
    # Must happen before generate_song_parts so that all downstream RNG draws
    # (soundfont selection, FX, layer probabilities) sit deterministically after it.
    _structures_file = cfg.song_structures_file if cfg is not None else config.DEFAULT_SONG_STRUCTURES_FILE
    song_unique_parts, song_arrangement = generate_song_arrangement(
        _rng, structures_file=_structures_file
    )

    # Generates the musical components
    ha, ba, me, be, _ = generate_song_parts(
        key=key,
        tempo=tempo,
        song_signatures=song_signatures,
        song_measures=measures,
        name=song_name,
        chord_pat_file=chord_pat_file,
        swing_amount=swing_amount,
        cfg=cfg,
    )

    # Mix components
    wav_name, arrangement, transitions, soundfonts, pedalboards, part_layers = mix_and_save(
        ha, ba, me, be, song_name,
        song_unique_parts, song_arrangement,
        cfg=cfg,
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
    logger.info("Elapsed time: %.2f seconds", elapsed_time)
    logger.info("Musicality analysis:")
    logger.info("Musicality score: %.2f", score)
    logger.debug("Component scores: %s", component_scores)
    
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
    swing_amount: float,
    cfg: config.Config = None,
) -> Tuple[Dict, Dict, Dict, Dict, Dict]:

    harm_filename, bass_filename, melo_filename, beat_filename, beat_annotations = {}, {}, {}, {}, {}

    for part, measures in song_measures.items():
        logger.info("Generating part: %s (%s measures)", part, measures)
        name_part = f"{name}-{part}"
        time_signature = song_signatures[part]

        chord_progression, harm_filename[part] = generate_chord_progression(
            key, tempo, time_signature, measures, name_part, part, chord_pat_file, _rng
        )

        melody, melo_filename[part] = generate_melody(
            key, tempo, time_signature, measures, name_part, part, chord_progression, _rng
        )

        bass_filename[part] = generate_bassline(
            key, tempo, time_signature, measures, name_part, part, chord_progression, melody, _rng
        )

        beat_filename[part], beat_annotations[part] = generate_beat(
            part, tempo, time_signature, measures, name_part, swing_amount, _rng, cfg=cfg
        )

    return harm_filename, bass_filename, melo_filename, beat_filename, beat_annotations

def generate_song(id: int, cfg: config.Config):
    """
    Generates a complete song with all its components.

    Args:
        id: Music id
        cfg: Config instance with all path and override settings
    """
    logger.info("Generating song #%s", id)

    # Basic musical parameters (D-08: sampler draws threaded through module-level _rng)
    key = generate_random_key(_rng)
    tempo = generate_random_tempo(_rng)
    time_signature = generate_random_time_signature(_rng)
    time_signature_variation = 1.0  # 100% chance of time signature variation
    swing_amount = generate_random_swing(_rng)
    swing_amount = min(0.75, max(0.5, float(swing_amount)))

    # Generates valid measurements and time signatures
    while True:
        measures, signatures = generate_song_measures(time_signature, time_signature_variation, _rng)
        if validate_measures(measures, signatures):
            break

    # Generates unique name for the song
    song_name = f"{datetime.now().strftime('%Y%m%d%H%M%S%f')}_{uuid.uuid4()}"
    song_name = song_name[:20]  # 20 chars

    # Generates the music components
    song_info = create_song(
        key=key,
        tempo=tempo,
        song_signatures=signatures,
        measures=measures,
        name=song_name,
        chord_pat_file=cfg.chord_patterns_file,
        swing_amount=swing_amount,
        cfg=cfg,
    )

    return song_info

# Example usage

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    # Phase 6: swap to pythonjsonlogger.jsonlogger.JsonFormatter when --json flag arrives
    cfg = config.Config.load()
    for i in range(1):
        generate_song(i, cfg)
