from datetime import datetime
import logging, time, json, random, os, uuid
import musicality_score, config
from timesig import TimeSignatureRegistry
from musicgen.sampler import (
    SongParams, generate_random_key, generate_random_tempo,
    generate_random_time_signature, generate_random_swing,
    generate_song_measures, generate_song_arrangement, validate_measures_dict,
)
from musicgen.generators.chord import generate_chord_progression
from musicgen.generators.melody import generate_melody
from musicgen.generators.bassline import generate_bassline
from musicgen.generators.beat import generate_beat
from musicgen import renderer, mixer, annotator, beats
from typing import Tuple, Dict, List

logger = logging.getLogger(__name__)

_rng = random.Random()  # D-08/D-18: single module-level RNG; Phase 5 adds hierarchy.
validate_measures = validate_measures_dict  # back-compat alias

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

def create_song(
    key: str, tempo: int, song_signatures: Dict[str, str], measures: Dict[str, int],
    name: str, chord_pat_file: str, swing_amount: float, cfg: config.Config = None,
) -> Dict:
    """Orchestrator: generators → renderer → mixer → beats → annotator (D-23/D-24)."""
    _cfg = cfg if cfg is not None else config.Config()
    start_time = time.time()
    logger.info("Generating song '%s' with swing amount: %s", name, swing_amount)

    # R-S3: arrangement computed ONCE; stabilises downstream RNG draw order.
    song_unique_parts, song_arrangement = generate_song_arrangement(
        _rng, structures_file=_cfg.song_structures_file,
    )
    logger.info("Arrangement: %s", song_arrangement)

    soundfonts = renderer.pick_soundfonts(_cfg, _rng)
    for layer, sf_path in soundfonts.items():
        logger.info("%s soundfont: %s", layer.capitalize(), sf_path)

    harm_filename, bass_filename, melo_filename, beat_filename, _discarded, chord_progressions = (
        generate_song_parts(
            key=key, tempo=tempo, song_signatures=song_signatures,
            song_measures=measures, name=name, chord_pat_file=chord_pat_file,
            swing_amount=swing_amount, cfg=cfg,
        )
    )

    fx_boards = mixer.build_fx_boards(_cfg, _rng)
    with open(_cfg.inst_probabilities_file) as _f:
        inst_proba = json.load(_f)
    layer_mask = mixer.compute_layer_mask(song_unique_parts, inst_proba, _rng)
    with open(_cfg.levels_file) as _f:
        levels = json.load(_f)

    render_results: Dict[str, renderer.RenderResult] = {}
    mix_results: Dict[str, mixer.MixResult] = {}
    beat_times_dict: Dict[str, List[float]] = {}
    downbeat_times_dict: Dict[str, List[float]] = {}
    midi_paths_dict: Dict[str, Dict[str, str]] = {}
    part_mix_paths: List[str] = []
    song_time_start = 0.0
    for part_counter, part in enumerate(song_arrangement, start=1):
        logger.info("Mixing part: %s (%d of %d)", part, part_counter, len(song_arrangement))
        midi_paths = {
            "beat":     beat_filename[part],
            "melody":   melo_filename[part],
            "harmony":  harm_filename[part],
            "bassline": bass_filename[part],
        }
        midi_paths_dict[part] = midi_paths
        out_dir = os.path.join(name, "%s-%s" % (name, part))

        render_results[part] = renderer.render_stems(midi_paths, soundfonts, out_dir, cfg=_cfg)
        mix_results[part] = mixer.mix_part(
            render_result=render_results[part], levels=levels, fx_boards=fx_boards,
            layer_mask_for_part=layer_mask[part], part=part, out_dir=out_dir,
            soundfonts=soundfonts, part_counter=part_counter, song_time_start=song_time_start,
        )
        part_mix_paths.append(mix_results[part].mix_path)
        # Beats: D-22 post-mix, serial.
        beat_times_dict[part] = beats.extract_beat_times(
            midi_paths["beat"], tempo, song_time_start,
        )
        downbeat_times_dict[part] = beats.extract_downbeat_times(
            beat_times_dict[part], song_signatures[part], measures[part], song_time_start, tempo,
        )
        song_time_start += render_results[part].duration_seconds
    final_wav = mixer.concat_parts(part_mix_paths, os.path.join(name, name + ".wav"))
    logger.info("Song saved as: %s", final_wav)
    score, component_scores = musicality_score.get_musicality_score(final_wav)  # D-04
    musicality = {"score": float(score), "components": {k: float(v) for k, v in component_scores.items()}}
    song_params_obj = SongParams(
        key=key, tempo=tempo, time_signature_base=song_signatures.get("verse", "4/4"),
        time_signature_variation=1.0, swing_amount=swing_amount,
        signatures_per_part=song_signatures, measures_per_part=measures,
        song_unique_parts=list(song_unique_parts), song_arrangement=list(song_arrangement),
    )
    annotation = annotator.annotate(
        song_params=song_params_obj, render_results=render_results, mix_results=mix_results,
        beat_times=beat_times_dict, downbeat_times=downbeat_times_dict, musicality=musicality,
        chord_progressions=chord_progressions, midi_paths=midi_paths_dict, mix_path=final_wav,
        fluidsynth_version=renderer.FLUIDSYNTH_VERSION,
    )
    with open(os.path.join(name, name + ".json"), "w") as outfile:  # Phase 5 writer owns this.
        json.dump(annotation, outfile, indent=4)
    elapsed = time.time() - start_time
    logger.info("Elapsed time: %.2f seconds", elapsed)
    logger.info("Musicality score: %.2f", score)
    logger.debug("Component scores: %s", component_scores)
    return annotation


def generate_song_parts(
    key: str, tempo: int, song_signatures: Dict[str, str], song_measures: Dict[str, int],
    name: str, chord_pat_file: str, swing_amount: float, cfg: config.Config = None,
) -> Tuple[Dict, Dict, Dict, Dict, Dict, Dict]:
    """Generate MIDI files for all parts. 6th return value is chord_progressions (new in 04-05)."""
    harm_filename, bass_filename, melo_filename, beat_filename, beat_annotations = {}, {}, {}, {}, {}
    chord_progressions: Dict[str, List[str]] = {}

    for part, part_measures in song_measures.items():
        logger.info("Generating part: %s (%s measures)", part, part_measures)
        name_part, time_signature = "%s-%s" % (name, part), song_signatures[part]
        chord_progression, harm_filename[part] = generate_chord_progression(
            key, tempo, time_signature, part_measures, name_part, part, chord_pat_file, _rng,
        )
        chord_progressions[part] = list(chord_progression)
        melody, melo_filename[part] = generate_melody(
            key, tempo, time_signature, part_measures, name_part, part, chord_progression, _rng,
        )
        bass_filename[part] = generate_bassline(
            key, tempo, time_signature, part_measures, name_part, part, chord_progression, melody, _rng,
        )
        beat_filename[part], beat_annotations[part] = generate_beat(
            part, tempo, time_signature, part_measures, name_part, swing_amount, _rng, cfg=cfg,
        )

    return harm_filename, bass_filename, melo_filename, beat_filename, beat_annotations, chord_progressions

def generate_song(id: int, cfg: config.Config):
    """Generate a complete song with all components (sampler → create_song)."""
    logger.info("Generating song #%s", id)
    key = generate_random_key(_rng)
    tempo = generate_random_tempo(_rng)
    time_signature = generate_random_time_signature(_rng)
    swing_amount = min(0.75, max(0.5, float(generate_random_swing(_rng))))

    while True:
        measures, signatures = generate_song_measures(time_signature, 1.0, _rng)
        if validate_measures(measures, signatures):
            break

    song_name = "%s_%s" % (datetime.now().strftime('%Y%m%d%H%M%S%f'), uuid.uuid4())
    song_name = song_name[:20]
    return create_song(
        key=key, tempo=tempo, song_signatures=signatures, measures=measures,
        name=song_name, chord_pat_file=cfg.chord_patterns_file,
        swing_amount=swing_amount, cfg=cfg,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    cfg = config.Config.load()
    for i in range(1):
        generate_song(i, cfg)
