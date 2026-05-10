"""Library API — single-sample generate() entry point (R-P12 single-sample, D-31, D-34).

This module migrates the body of the old ``music_gen.create_song`` +
``generate_song_parts`` + ``generate_song`` triad into a single
``generate(config) -> SampleResult`` library call. The Phase 3+4 pipeline
(sampler → generators → renderer → mixer → beats → annotator) is unchanged
— only the orchestrator wrapping it changes:

  1. Single ``_rng`` replaced by ``rngs`` dict (5 domain-specific RNGs,
     D-18/D-19).
  2. ``musicality_score`` call wrapped in ``save_random_state()`` (D-20).
  3. Hand-rolled JSON dump replaced by ``writer.write_sample()`` with the
     atomic sentinel + sum-of-stems assertion + per-sample-dir-relative
     paths.
  4. Sentinel resume short-circuit (D-31 step 3).
  5. Manifest append regardless of success/failure (D-13).

``generate_batch`` is Phase 6 (R-P10); this module ships only the single
-sample primitive. D-43 lists the Phase 6 deferrals.
"""
from __future__ import annotations

import importlib.metadata
import json
import logging
import os
import random
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from config import Config  # Re-exported from musicgen.__init__

from musicgen import annotator, beats, mixer, musicality, renderer, writer
from musicgen.genre import GenreSpec, resolve_genres
from musicgen.generators.bassline import generate_bassline
from musicgen.generators.beat import generate_beat
from musicgen.generators.chord import generate_chord_progression
from musicgen.generators.melody import generate_melody
from musicgen.manifest import ManifestWriter
from musicgen.sampler import (
    SongParams,
    generate_random_key,
    generate_random_swing,
    generate_random_tempo,
    generate_random_time_signature,
    generate_song_arrangement,
    generate_song_measures,
    validate_measures_dict,
)
from musicgen.seeds import (
    RNG_FX,
    RNG_GENERATORS,
    RNG_MIX,
    RNG_PARAMS,
    RNG_SOUNDFONTS,
    assign_split,
    derive_sample_seed,
    make_rngs,
    save_random_state,
)

logger = logging.getLogger(__name__)

_LAYERS = ("beat", "melody", "harmony", "bassline")


def resolve_genre_spec(cfg: Config) -> Optional[GenreSpec]:
    """Return merged GenreSpec from cfg.genre, or None if no genre is configured.

    Calls :func:`musicgen.genre.resolve_genres` with cfg.genre list and
    cfg.genres_dir. Returns None when cfg.genre is None or empty so callers
    can use `genre_spec=None` for the no-genre code path.
    """
    if not cfg.genre:
        return None
    return resolve_genres(cfg.genre, cfg.genres_dir)

# D-22 / RESEARCH Pitfall 4: resolve once at import, fall back when uninstalled.
try:
    MUSICGEN_VERSION = importlib.metadata.version("musicgen")
except importlib.metadata.PackageNotFoundError:
    MUSICGEN_VERSION = "0.1.0+uninstalled"


@dataclass(frozen=True)
class SampleResult:
    """Result of a single ``generate(config)`` call (D-02).

    Produced by :func:`generate`. Twelve fields; all absolute paths.
    ``status`` is ``"ok"`` on success, ``"failed"`` on any pipeline
    exception (including sum-of-stems assertion failure).
    ``attempt`` is 1-based: which quality-gate attempt produced this result.
    """
    sample_index: int
    seed: int
    sample_dir: str
    sample_json_path: str
    mix_path: str
    stem_paths: Dict[str, str]
    midi_paths: Dict[str, str]
    split: str
    status: str
    musicality_score: float
    duration_seconds: float
    attempt: int = 1


def generate(config: Config, *, manifest_writer=None) -> SampleResult:
    """Generate one sample end-to-end (D-31, D-58).

    The sole library entry point for v0.1. Batch generation via
    ``generate_batch`` is Phase 6 (R-P10).

    Args:
        config: A ``Config`` with at minimum ``global_seed``, ``sample_index``,
            and ``dataset_root``. Split ratios default to (0.8, 0.1, 0.1).
        manifest_writer: Optional manifest writer override (D-58). When None
            a default ``ManifestWriter(config.dataset_root)`` is created. Pass
            a ``_NullManifestWriter`` from batch.py to skip per-sample manifest
            writes (batch.py appends after as_completed).

    Returns:
        SampleResult with ``status="ok"`` on success, ``"failed"`` on any
        pipeline exception (including sum-of-stems assertion failure).

    Raises:
        ValueError: ``config.global_seed is None`` or ``config.sample_index < 0``.
            These are CALLER ERRORS, not pipeline errors — they raise.
            Pipeline errors during generation are converted to failed results.
    """
    # Step 1: validate (raise on caller error per D-21).
    if config.global_seed is None:
        raise ValueError(
            "global_seed is required for deterministic generation; "
            "pass config.global_seed explicitly"
        )
    if config.sample_index < 0:
        raise ValueError(
            f"sample_index must be >= 0, got {config.sample_index}"
        )

    # Step 2: derive seeds + RNGs.
    sample_seed = derive_sample_seed(config.global_seed, config.sample_index)
    rngs = make_rngs(sample_seed)

    # Step 3: resume short-circuit (D-31 step 3).
    if ManifestWriter.is_sample_complete(
        config.dataset_root, config.sample_index
    ):
        logger.info(
            "Sample %d exists at %s — resume short-circuit",
            config.sample_index, config.dataset_root,
        )
        return _reconstruct_sample_result(
            config.dataset_root, config.sample_index, sample_seed,
        )

    # Step 4+: full pipeline under try/except — any failure → failed result.
    if manifest_writer is None:
        manifest_writer = ManifestWriter(config.dataset_root)
    working_dir = tempfile.mkdtemp(prefix="musicgen-")
    logger.debug("Sample %d working dir: %s", config.sample_index, working_dir)

    try:
        min_score = config.min_musicality_score
        max_att = config.max_attempts
        result = None

        for attempt in range(1, max_att + 1):
            attempt_seed = sample_seed + (attempt - 1)
            attempt_rngs = make_rngs(attempt_seed)
            result = _run_pipeline(
                config, attempt_seed, attempt_rngs, working_dir, attempt,
            )
            if result.musicality_score >= min_score or attempt == max_att:
                break

        # Write manifest once for the winning attempt.
        rel_path = f"{config.sample_index:06d}/sample.json"
        manifest_writer.append({
            "sample_index": config.sample_index,
            "seed": config.global_seed,
            "sample_seed": result.seed,
            "status": result.status,
            "split": result.split,
            "path": rel_path if result.status == "ok" else "",
            "musicality_score": result.musicality_score,
            "duration_seconds": result.duration_seconds,
            "attempt": result.attempt,
            "wrote_at": datetime.now(timezone.utc).isoformat(),
        })
        return result
    except Exception as exc:  # noqa: BLE001 — intentional broad catch per D-24
        logger.exception(
            "Sample %d failed: %s", config.sample_index, exc,
        )
        failed_result = SampleResult(
            sample_index=config.sample_index,
            seed=sample_seed,
            sample_dir=os.path.join(
                config.dataset_root, f"{config.sample_index:06d}",
            ),
            sample_json_path="",  # no sentinel on failure
            mix_path="",
            stem_paths={},
            midi_paths={},
            split="",
            status="failed",
            musicality_score=0.0,
            duration_seconds=0.0,
        )
        # Append failure entry to manifest (D-13 requires a line even on failure).
        try:
            manifest_writer.append({
                "sample_index": config.sample_index,
                "seed": config.global_seed,
                "sample_seed": sample_seed,
                "status": "failed",
                "split": "",
                "path": "",
                "musicality_score": 0.0,
                "duration_seconds": 0.0,
                "attempt": 1,
                "error": repr(exc)[:2048],  # D-13 2KB cap
            })
        except Exception as manifest_exc:  # noqa: BLE001
            logger.error(
                "Failed to append failure entry to manifest: %s", manifest_exc,
            )
        return failed_result
    finally:
        if not config.keep_working_dirs:
            shutil.rmtree(working_dir, ignore_errors=True)


def _run_pipeline(
    config: Config,
    attempt_seed: int,
    rngs: Dict[str, random.Random],
    working_dir: str,
    attempt: int = 1,
) -> SampleResult:
    """Run the full pipeline for one attempt. Called from generate() loop."""
    _cfg = config

    # Sample parameters from rngs[RNG_PARAMS] (D-19 — sampler calls).
    key = generate_random_key(rngs[RNG_PARAMS])
    tempo = generate_random_tempo(rngs[RNG_PARAMS])
    time_signature = generate_random_time_signature(rngs[RNG_PARAMS])
    swing_amount = min(0.75, max(0.5, float(generate_random_swing(rngs[RNG_PARAMS]))))

    while True:
        measures, signatures = generate_song_measures(
            time_signature, 1.0, rngs[RNG_PARAMS],
        )
        if validate_measures_dict(measures, signatures):
            break

    song_unique_parts, song_arrangement = generate_song_arrangement(
        rngs[RNG_PARAMS], structures_file=_cfg.song_structures_file,
    )

    # Per-song working-dir name — use a stable name under working_dir.
    name = os.path.join(working_dir, f"sample-{config.sample_index:06d}")
    os.makedirs(name, exist_ok=True)

    # Generators (D-19 — rngs[RNG_GENERATORS]).
    harm_paths, bass_paths, melo_paths, beat_paths, _discarded, chord_progressions = (
        _generate_all_midi(
            rngs, key, tempo, signatures, measures, name,
            _cfg.chord_patterns_file, swing_amount, _cfg,
        )
    )

    # Soundfont selection (D-19 — rngs[RNG_SOUNDFONTS]).
    soundfonts = renderer.pick_soundfonts(_cfg, rngs[RNG_SOUNDFONTS])
    for layer, sf_path in soundfonts.items():
        logger.info("%s soundfont: %s", layer.capitalize(), sf_path)

    # FX + layer mask (D-19 — rngs[RNG_FX] / rngs[RNG_MIX]).
    fx_boards = mixer.build_fx_boards(_cfg, rngs[RNG_FX])
    inst_proba = _cfg.load_inst_probabilities()
    layer_mask = mixer.compute_layer_mask(
        song_unique_parts, inst_proba, rngs[RNG_MIX],
    )
    levels = _cfg.load_levels()

    # Per-part render + mix + beat extraction.
    render_results: Dict[str, renderer.RenderResult] = {}
    mix_results: Dict[str, mixer.MixResult] = {}
    beat_times_dict: Dict[str, List[float]] = {}
    downbeat_times_dict: Dict[str, List[float]] = {}
    midi_paths_by_part: Dict[str, Dict[str, str]] = {}
    stem_paths_by_part: Dict[str, Dict[str, str]] = {}
    part_mix_paths: List[str] = []
    part_durations_s: List[float] = []
    song_time_start = 0.0

    for part_counter, part in enumerate(song_arrangement, start=1):
        logger.info(
            "Mixing part: %s (%d of %d)", part, part_counter, len(song_arrangement),
        )
        midi_paths = {
            "beat":     beat_paths[part],
            "melody":   melo_paths[part],
            "harmony":  harm_paths[part],
            "bassline": bass_paths[part],
        }
        midi_paths_by_part[part] = midi_paths
        out_dir = os.path.join(name, f"{config.sample_index:06d}-{part}")

        render_results[part] = renderer.render_stems(
            midi_paths, soundfonts, out_dir, cfg=_cfg,
        )
        mix_results[part] = mixer.mix_part(
            render_result=render_results[part], levels=levels,
            fx_boards=fx_boards, layer_mask_for_part=layer_mask[part],
            part=part, out_dir=out_dir, soundfonts=soundfonts,
            part_counter=part_counter, song_time_start=song_time_start,
        )
        stem_paths_by_part[part] = mix_results[part].stem_paths
        part_mix_paths.append(mix_results[part].mix_path)
        part_durations_s.append(render_results[part].duration_seconds)

        beat_times_dict[part] = beats.extract_beat_times(
            midi_paths["beat"], tempo, song_time_start,
        )
        downbeat_times_dict[part] = beats.extract_downbeat_times(
            beat_times_dict[part], signatures[part], measures[part],
            song_time_start, tempo,
        )
        song_time_start += render_results[part].duration_seconds

    # Concatenate final mix (working dir).
    final_wav = mixer.concat_parts(
        part_mix_paths, os.path.join(name, f"sample-{config.sample_index:06d}.wav"),
    )

    # Musicality scoring — D-20 defense-in-depth wrap.
    with save_random_state():
        score, component_scores = musicality.get_musicality_score(final_wav)
    musicality_dict = {
        "score": float(score),
        "components": {k: float(v) for k, v in component_scores.items()},
    }

    # Split assignment (D-26 / R-P6).
    split = assign_split(attempt_seed, _cfg.split_ratios)

    # Build SongParams for annotator.
    song_params_obj = SongParams(
        key=key, tempo=tempo,
        time_signature_base=signatures.get("verse", "4/4"),
        time_signature_variation=1.0, swing_amount=swing_amount,
        signatures_per_part=signatures, measures_per_part=measures,
        song_unique_parts=list(song_unique_parts),
        song_arrangement=list(song_arrangement),
    )

    # Load FluidSynth pre-roll offset (R-P9, D-53). Graceful fallback when
    # calibrate.py doesn't exist yet (Wave 2 of Phase 6 creates it).
    try:
        from musicgen import calibrate as _calibrate
        pre_roll_offset_s = _calibrate.load_preroll(_cfg.project_root)
    except Exception:
        pre_roll_offset_s = 0.0

    # Annotate — Phase-5 TBDs filled per D-22.
    annotation = annotator.annotate(
        song_params=song_params_obj,
        render_results=render_results, mix_results=mix_results,
        beat_times=beat_times_dict, downbeat_times=downbeat_times_dict,
        musicality=musicality_dict, chord_progressions=chord_progressions,
        midi_paths=midi_paths_by_part, mix_path=final_wav,
        fluidsynth_version=renderer.FLUIDSYNTH_VERSION,
        seed=attempt_seed,
        musicgen_version=MUSICGEN_VERSION,
        split=split,
        pre_roll_offset_seconds=pre_roll_offset_s,
    )

    # Write atomic per-sample layout.
    paths = writer.write_sample(
        config.dataset_root, config.sample_index, annotation,
        final_wav, stem_paths_by_part, midi_paths_by_part,
        list(song_arrangement), tempo, part_durations_s,
        fluidsynth_version=renderer.FLUIDSYNTH_VERSION,
        split=split,
        sum_of_stems_epsilon=_cfg.sum_of_stems_epsilon,
        output_mode=_cfg.output_mode,
        pre_roll_offset_s=pre_roll_offset_s,
    )

    duration_seconds = sum(part_durations_s)
    return SampleResult(
        sample_index=config.sample_index,
        seed=attempt_seed,
        sample_dir=paths["sample_dir"],
        sample_json_path=paths["sample_json"],
        mix_path=paths.get("mix", ""),
        stem_paths={layer: paths[f"stems_{layer}"] for layer in _LAYERS if f"stems_{layer}" in paths},
        midi_paths={layer: paths[f"midi_{layer}"] for layer in _LAYERS if f"midi_{layer}" in paths},
        split=split,
        status="ok",
        musicality_score=float(score),
        duration_seconds=duration_seconds,
        attempt=attempt,
    )


def _generate_all_midi(
    rngs: Dict[str, random.Random],
    key: str, tempo: int,
    song_signatures: Dict[str, str],
    song_measures: Dict[str, int],
    name: str,
    chord_pat_file: str,
    swing_amount: float,
    cfg: Config,
) -> Tuple[Dict, Dict, Dict, Dict, Dict, Dict]:
    """Per-part MIDI generation (extracted from music_gen.generate_song_parts, D-34).

    Migrated verbatim except ``_rng`` → ``rngs[RNG_GENERATORS]`` (D-19).
    """
    harm_filename: Dict = {}
    bass_filename: Dict = {}
    melo_filename: Dict = {}
    beat_filename: Dict = {}
    beat_annotations: Dict = {}
    chord_progressions: Dict[str, List[str]] = {}

    for part, part_measures in song_measures.items():
        logger.info("Generating part: %s (%s measures)", part, part_measures)
        name_part = f"{name}-{part}"
        time_signature = song_signatures[part]

        chord_progression, harm_filename[part] = generate_chord_progression(
            key, tempo, time_signature, part_measures, name_part, part,
            chord_pat_file, rngs[RNG_GENERATORS],
        )
        chord_progressions[part] = list(chord_progression)

        melody, melo_filename[part] = generate_melody(
            key, tempo, time_signature, part_measures, name_part, part,
            chord_progression, rngs[RNG_GENERATORS],
        )

        bass_filename[part] = generate_bassline(
            key, tempo, time_signature, part_measures, name_part, part,
            chord_progression, melody, rngs[RNG_GENERATORS],
        )

        beat_filename[part], beat_annotations[part] = generate_beat(
            part, tempo, time_signature, part_measures, name_part,
            swing_amount, rngs[RNG_GENERATORS], cfg=cfg,
        )

    return (
        harm_filename, bass_filename, melo_filename, beat_filename,
        beat_annotations, chord_progressions,
    )


def _reconstruct_sample_result(
    dataset_root: str, sample_index: int, sample_seed: int,
) -> SampleResult:
    """Load existing sample.json + build paths from convention (D-31 step 3, Pitfall 7)."""
    sample_dir = os.path.join(dataset_root, f"{sample_index:06d}")
    sample_json_path = os.path.join(sample_dir, "sample.json")
    with open(sample_json_path, "r") as f:
        data = json.load(f)
    stem_paths = {
        layer: os.path.join(sample_dir, "stems", f"{layer}.wav")
        for layer in _LAYERS
    }
    midi_paths = {
        layer: os.path.join(sample_dir, "midi", f"{layer}.mid")
        for layer in _LAYERS
    }
    # musicality_score may be stored as nested {"score": ..., "components": ...} or flat float.
    ms_raw = data.get("musicality_score", 0.0)
    if isinstance(ms_raw, dict):
        musicality_score = float(ms_raw.get("score", 0.0))
    else:
        musicality_score = float(ms_raw)
    return SampleResult(
        sample_index=sample_index,
        seed=sample_seed,
        sample_dir=sample_dir,
        sample_json_path=sample_json_path,
        mix_path=os.path.join(sample_dir, "mix.wav"),
        stem_paths=stem_paths,
        midi_paths=midi_paths,
        split=data.get("split", ""),
        status="ok",
        musicality_score=musicality_score,
        duration_seconds=float(data.get("duration_seconds", 0.0)),
    )
