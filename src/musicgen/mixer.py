"""Mixer module — FX application, pydub overlay, layer mask, part concatenation (R-X5).

Absorbs 5 functions from ``music_gen.py`` (D-10): ``create_effect``,
``generate_pedalboard``, ``apply_fx_to_layer``, ``pedalboard_info_json``, and
the nested ``_lin_to_db`` from ``mix_and_save``. Adds 4 new functions:
``_make_silent_stem`` (D-12 + RESEARCH correction #2), ``compute_layer_mask``
(D-13/D-17), ``mix_part``, and ``concat_parts``.

Design:
  D-10 — FX pipeline (``create_effect``, ``generate_pedalboard``,
         ``apply_fx_to_layer``, ``pedalboard_info_json``) moved from
         ``music_gen.py``. ``_lin_to_db`` unnested from ``mix_and_save``.
  D-11 — FX applied to ALL four layers unconditionally. The
         ``music_gen.py:276`` "optimize so fx only applied to used layers"
         TODO is OUT OF SCOPE — changing the RNG draw order would invalidate
         the Phase 5 golden-seed baseline.
  D-12 — Silent-stem fallback via ``AudioSegment.silent(duration=ms,
         frame_rate=44100).set_channels(2)``. Stereo 44.1kHz is required to
         match FluidSynth renderer output; mono/11025 Hz defaults break
         overlay channel matching (RESEARCH correction #2).
  D-13 — ``compute_layer_mask`` replaces the 4 bare ``random.random()`` draws
         per part at ``music_gen.py:245-253``.
  D-17 — Zero bare ``random.<method>`` calls. All draws via injected ``rng``.
  D-25 — ``cfg: config.Config = None`` runtime fallback for public functions.

R-S4 preservation: the ``apply_gain(_lin_to_db(v))`` + ``segment = segment.pan(v)``
pattern from Phase 1 is preserved verbatim inside ``mix_part`` — levels.json
MUST keep affecting output (PITFALLS P-B regression guard).
"""
from __future__ import annotations

import json
import logging
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

from pedalboard import Pedalboard, Compressor, Gain, Chorus, LadderFilter, Phaser, Delay, Reverb
from pedalboard.io import AudioFile
from pydub import AudioSegment

import config
from musicgen.genre import GenreSpec
from musicgen.renderer import RenderResult

logger = logging.getLogger(__name__)


_LAYERS = ("beat", "melody", "harmony", "bassline")


# ---------- _lin_to_db (R-S4 preserved, unnested from mix_and_save) ----------

def _lin_to_db(v: float) -> float:
    """Convert linear amplitude (0.0-1.0) to dB. Clamps at 1e-6 to avoid log(0).

    R-S4 preservation — body identical to the nested version at
    ``music_gen.py:285-286``. Keeps ``levels.json`` values (linear amplitudes
    in range 0.5-1.0 observed) influencing output via ``apply_gain``.

    Args:
        v: Linear amplitude value (0.0-1.0).

    Returns:
        dB equivalent, clamped so that 0.0 input returns ~-120.0 dB.
    """
    return 20.0 * math.log10(max(float(v), 1e-6))


# ---------- _make_silent_stem (D-12 + RESEARCH correction #2) ----------

def _make_silent_stem(duration_ms: int, sample_rate: int = 44100, channels: int = 2) -> AudioSegment:
    """Create a silent AudioSegment matching rendered-stem spec (D-12).

    ``AudioSegment.silent()`` defaults to MONO 11025 Hz — this mismatches
    FluidSynth's stereo 44100 Hz output. Without explicit channels + sample
    rate, pydub overlay behaves unexpectedly and the Phase 5 R-P2
    stems-sum-to-mix assertion would break (RESEARCH correction #2).

    Args:
        duration_ms: Duration in milliseconds.
        sample_rate: Target frame rate in Hz; default 44100 matches FluidSynth.
        channels: Target channel count; default 2 (stereo).

    Returns:
        AudioSegment with the specified duration, sample rate, and channel count.
    """
    silent = AudioSegment.silent(duration=duration_ms, frame_rate=sample_rate)
    return silent.set_channels(channels)


# ---------- _create_effect (D-10/D-17) ----------

def _create_effect(effect_class, parameters: dict, rng: random.Random):
    """Probability-weighted effect construction (D-10).

    Body from ``music_gen.py:130-139`` with ``random.random()`` /
    ``random.uniform`` replaced by ``rng.*`` (D-17).

    Args:
        effect_class: Pedalboard effect class (e.g., Reverb).
        parameters: Dict with ``"probability"`` and ``"value_range"`` keys.
        rng: Injected ``random.Random`` (D-17).

    Returns:
        An instantiated effect object if the probability check passes, else None.
    """
    probability = parameters['probability']
    value_range = parameters['value_range']
    if rng.random() < probability:
        kwargs = {
            param: rng.uniform(value_range[param][0], value_range[param][1])
            for param in value_range
        }
        return effect_class(**kwargs)
    return None


# ---------- _generate_pedalboard (D-10/D-17) ----------

def _generate_pedalboard(
    effect_params_file: str,
    rng: random.Random,
    fx_profile: Optional[Dict[str, float]] = None,
) -> Pedalboard:
    """Construct a Pedalboard from the 7 effect specs in the FX JSON (D-10).

    Body from ``music_gen.py:141-160`` with injected ``rng`` threaded through
    ``_create_effect``. When ``fx_profile`` is supplied, each named effect's
    probability is multiplied by the matching profile weight (soft shift).

    Args:
        effect_params_file: Path to the FX JSON file (e.g., beat_fx.json).
        rng: Injected ``random.Random`` (D-17).
        fx_profile: Optional dict mapping effect name → weight multiplier (genre FX).

    Returns:
        Pedalboard with 0-7 effects depending on probability rolls.
    """
    with open(effect_params_file, 'r') as json_file:
        effect_params = json.load(json_file)
    _profile = fx_profile or {}
    effect_key_map = {
        "compressor":   Compressor,
        "gain":         Gain,
        "chorus":       Chorus,
        "ladder_filter": LadderFilter,
        "phaser":       Phaser,
        "delay":        Delay,
        "reverb":       Reverb,
    }
    effects = []
    for key, effect_class in effect_key_map.items():
        params = dict(effect_params[key])
        if key in _profile:
            params["probability"] = min(1.0, params["probability"] * _profile[key])
        effects.append((effect_class, params))

    board = Pedalboard([
        effect
        for effect in (
            _create_effect(effect_class, parameters, rng)
            for effect_class, parameters in effects
        )
        if effect is not None
    ])
    return board


# ---------- build_fx_boards (D-10/D-17) ----------

def build_fx_boards(
    cfg: Optional[config.Config] = None,
    rng: Optional[random.Random] = None,
    genre_spec: Optional[GenreSpec] = None,
) -> Dict[str, Pedalboard]:
    """Construct per-layer pedalboards (D-10/D-17).

    Replaces 4 sequential ``generate_pedalboard(fx_files[layer])`` calls in
    ``mix_and_save:224-227``. All four pedalboards share the same ``rng`` so
    their random draws happen in a stable order (preserves Phase 5 seeded
    determinism baseline).

    When ``genre_spec.fx_profile`` is non-empty, the effect probability for
    each named effect is multiplied by the genre's profile weight (soft shift —
    full range still accessible).

    Args:
        cfg: Optional Config (D-25 fallback to ``config.Config()`` if None).
        rng: Injected ``random.Random`` (required; D-17).
        genre_spec: Optional merged :class:`GenreSpec` for FX profile shifts.

    Returns:
        Dict mapping layer name → Pedalboard.

    Raises:
        ValueError: if ``rng`` is None (D-17 guard).
    """
    if rng is None:
        raise ValueError("build_fx_boards requires an injected rng (D-17)")
    _cfg = cfg if cfg is not None else config.Config()
    fx_profile = genre_spec.fx_profile if genre_spec is not None else {}
    return {
        layer: _generate_pedalboard(_cfg.fx_files[layer], rng, fx_profile=fx_profile)
        for layer in _LAYERS
    }


# ---------- apply_fx_to_layer (D-10, lifts verbatim) ----------

def apply_fx_to_layer(wav_file: str, board: Pedalboard) -> str:
    """Apply pedalboard FX to a WAV file chunk-by-chunk; writes ``<wav>_fx.wav``.

    Body lifts verbatim from ``music_gen.py:162-171`` — no RNG draws.

    Args:
        wav_file: Path to the source WAV file.
        board: Pedalboard to apply.

    Returns:
        Path to the FX-processed WAV (``wav_file + '_fx.wav'``).
    """
    with AudioFile(wav_file) as af:
        with AudioFile(wav_file + '_fx.wav', 'w', af.samplerate, af.num_channels) as of:
            while af.tell() < af.frames:
                chunk = af.read(af.samplerate)
                effected = board(chunk, af.samplerate, reset=False)
                of.write(effected)
    return wav_file + '_fx.wav'


# ---------- pedalboard_info_json (D-10, lifts verbatim) ----------

def pedalboard_info_json(board: Pedalboard) -> list:
    """Serialize a Pedalboard to a JSON-friendly list.

    Lifts verbatim from ``music_gen.py:173-190``. No RNG draws.

    Args:
        board: Pedalboard whose effects are serialized.

    Returns:
        List of dicts, each with ``"name"`` (class name) and ``"parameters"``
        (attribute → str(value)) keys.
    """
    pedals_and_parameters = []
    for pedal in board:
        attributes = dir(pedal)
        parameters = [
            attr for attr in attributes
            if not attr.startswith("_") and not callable(getattr(pedal, attr))
        ]
        pedal_info = {"name": pedal.__class__.__name__, "parameters": {}}
        for parameter in parameters:
            value = getattr(pedal, parameter)
            pedal_info['parameters'][str(parameter)] = str(value)
        pedals_and_parameters.append(pedal_info)
    return pedals_and_parameters


# ---------- compute_layer_mask (D-13/D-17) ----------

def compute_layer_mask(
    song_unique_parts: List[str],
    inst_proba: dict,
    rng: random.Random,
) -> Dict[str, Dict[str, bool]]:
    """Compute which of the 4 layers are INCLUDED in each unique part's mix.

    Replaces the 4 bare ``random.random() <= proba`` draws per part at
    ``music_gen.py:245-253`` (D-13/D-17).

    Args:
        song_unique_parts: List of distinct part names (e.g., ``["intro", "verse"]``).
        inst_proba: Dict keyed by part → Dict keyed by layer → probability
            string/float (matches ``inst_probabilities.json`` structure).
        rng: Injected ``random.Random``.

    Returns:
        Nested dict: ``{part: {layer: bool}}`` where ``True`` means the layer
        is overlaid in the final mix for that part.
    """
    return {
        part: {
            layer: rng.random() <= float(inst_proba[part][layer])
            for layer in _LAYERS
        }
        for part in song_unique_parts
    }


# ---------- MixResult (D-02) ----------

@dataclass(frozen=True)
class MixResult:
    """Per-part mix outputs (R-X5).

    Consumed by :func:`musicgen.annotator.annotate` and
    :func:`musicgen.mixer.concat_parts`.

    D-02: frozen dataclass; shape matches Phase 3's SongParams convention.

    Attributes:
        mix_path: Absolute path to the part's mixed WAV.
        stem_paths: Dict mapping layer name → post-FX stem WAV path
            (includes silent-stem stubs for layer_mask-False layers; D-12).
        part_layers: Dict mapping layer name → whether included in the mix
            for this part (from ``compute_layer_mask``).
        soundfonts: Dict mapping layer name → ``.sf2`` path used.
        pedalboards: Dict mapping layer name → ``pedalboard_info_json`` list.
        transitions: Pairs ``[part, start_seconds]`` + final ``["end", end_seconds]``.
    """
    mix_path: str
    stem_paths: Dict[str, str]
    part_layers: Dict[str, bool]
    soundfonts: Dict[str, str]
    pedalboards: Dict[str, list]
    transitions: list


# ---------- mix_part (D-10/D-11/D-12) ----------

def mix_part(
    render_result: RenderResult,
    levels: dict,
    fx_boards: Dict[str, Pedalboard],
    layer_mask_for_part: Dict[str, bool],
    part: str,
    out_dir: str,
    soundfonts: Dict[str, str],
    part_counter: int = 1,
    song_time_start: float = 0.0,
) -> MixResult:
    """Mix one part: FX (all 4 layers, D-11) → gain/pan → overlay (masked-in layers).

    R-S4 preservation: uses ``apply_gain(_lin_to_db(v))`` + ``segment.pan(v)``
    verbatim from the Phase 1 bug fix.

    D-11 preservation: applies FX to ALL 4 layers BEFORE the mask check, to
    keep the RNG draw count identical to the pre-refactor pipeline. The
    ``music_gen.py:276`` TODO is OUT OF SCOPE.

    D-12: silent stems for layer_mask-False layers are written via
    ``_make_silent_stem`` at stereo 44100 Hz.

    Args:
        render_result: Per-layer render output from :func:`musicgen.renderer.render_stems`.
        levels: Full levels dict (loaded from ``levels.json``). Keyed by
            part → layer → {volume: float, panning: float}.
        fx_boards: Per-layer pedalboards (from :func:`build_fx_boards`).
        layer_mask_for_part: Which layers to overlay (from
            :func:`compute_layer_mask`, indexed by ``part``).
        part: Current part name (e.g., ``"intro"``).
        out_dir: Directory for the part's mix WAV and silent-stem WAVs.
        soundfonts: Per-layer ``.sf2`` paths (from
            :func:`musicgen.renderer.pick_soundfonts`) — recorded in MixResult.
        part_counter: 1-based position in the arrangement (for filename stem).
        song_time_start: Cumulative mix time in seconds BEFORE this part — used
            to build the ``transitions`` entry.

    Returns:
        MixResult with the mix path, per-layer (post-FX + silent-stub) stem
        paths, part_layers mask, soundfonts used, pedalboards_info, and a
        transitions list with ``[[part, song_time_start], ["end", song_time_end]]``.
    """
    os.makedirs(out_dir, exist_ok=True)

    # D-11: FX applied to ALL 4 layers UNCONDITIONALLY (before mask check).
    # Preserves the RNG draw order vs. the pre-refactor music_gen.py pipeline.
    post_fx_paths: Dict[str, str] = {}
    segments: Dict[str, AudioSegment] = {}
    for layer in _LAYERS:
        stem_wav = render_result.stem_paths[layer]
        post_fx_paths[layer] = apply_fx_to_layer(stem_wav, fx_boards[layer])
        segments[layer] = AudioSegment.from_wav(post_fx_paths[layer])

    # R-S4: gain/pan applied verbatim from Phase 1 fix (music_gen.py:287-294).
    # apply_gain returns a new segment; pan() also returns a new segment.
    # Both returns are captured (the original P-B bug was discarding .pan() return).
    # Explicit per-layer calls (not a loop) to satisfy grep-based regression guard.
    segments["beat"] = segments["beat"].apply_gain(_lin_to_db(levels[part]["beat"]["volume"]))
    segments["melody"] = segments["melody"].apply_gain(_lin_to_db(levels[part]["melody"]["volume"]))
    segments["harmony"] = segments["harmony"].apply_gain(_lin_to_db(levels[part]["harmony"]["volume"]))
    segments["bassline"] = segments["bassline"].apply_gain(_lin_to_db(levels[part]["bassline"]["volume"]))
    segments["beat"] = segments["beat"].pan(float(levels[part]["beat"]["panning"]))
    segments["melody"] = segments["melody"].pan(float(levels[part]["melody"]["panning"]))
    segments["harmony"] = segments["harmony"].pan(float(levels[part]["harmony"]["panning"]))
    segments["bassline"] = segments["bassline"].pan(float(levels[part]["bassline"]["panning"]))

    # D-12: build silent-stem stubs for any layer that was rendered but will
    # be excluded from the mix — so MixResult.stem_paths always has 4 entries.
    duration_ms = int(render_result.duration_seconds * 1000)

    # Build empty mix at stereo 44.1 kHz (matches rendered stem spec).
    mix = _make_silent_stem(
        duration_ms,
        sample_rate=render_result.sample_rate,
        channels=render_result.channels,
    )

    final_stem_paths: Dict[str, str] = {}
    part_layers: Dict[str, bool] = {}
    for layer in _LAYERS:
        included = bool(layer_mask_for_part.get(layer, False))
        part_layers[layer] = included
        if included:
            mix = mix.overlay(segments[layer])
            final_stem_paths[layer] = post_fx_paths[layer]
            logger.debug("%s layer added to %s mix", layer, part)
        else:
            # Write a silent WAV stub at stereo 44.1 kHz (D-12 + RESEARCH correction #2).
            silent_path = os.path.join(out_dir, f"{layer}_silent.wav")
            _make_silent_stem(
                duration_ms=duration_ms,
                sample_rate=render_result.sample_rate,
                channels=render_result.channels,
            ).export(silent_path, format='wav')
            final_stem_paths[layer] = silent_path

    # Export the part mix.
    part_mix_name = f"part-{part_counter}-{part}.wav"
    part_mix_path = os.path.join(out_dir, part_mix_name)
    mix.export(part_mix_path, format='wav')

    pedalboards_info = {layer: pedalboard_info_json(fx_boards[layer]) for layer in _LAYERS}
    song_time_end = song_time_start + mix.duration_seconds
    transitions = [[part, float(song_time_start)], ["end", float(song_time_end)]]

    return MixResult(
        mix_path=part_mix_path,
        stem_paths=final_stem_paths,
        part_layers=part_layers,
        soundfonts=dict(soundfonts),
        pedalboards=pedalboards_info,
        transitions=transitions,
    )


# ---------- concat_parts ----------

def concat_parts(part_mix_paths: List[str], out_path: str) -> str:
    """Concatenate per-part mix WAVs into one final mix WAV.

    Body from ``music_gen.py:331-340`` — no RNG draws, pure audio concat.

    Args:
        part_mix_paths: List of per-part mix WAV paths in arrangement order.
        out_path: Destination path for the final mix WAV.

    Returns:
        ``out_path`` on success.

    Raises:
        ValueError: if ``part_mix_paths`` is empty.
    """
    if not part_mix_paths:
        raise ValueError("concat_parts: no part mix paths provided; cannot assemble final mix")

    song = AudioSegment.from_wav(part_mix_paths[0])
    for part_wav in part_mix_paths[1:]:
        song += AudioSegment.from_wav(part_wav)

    # Ensure out_path's directory exists.
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    song.export(out_path, format='wav')
    return out_path
