"""Annotator module — pure-function R-P4 schema assembler (R-X6).

Design:
  D-14 — ``annotate()`` is a PURE function: zero I/O. No ``open``, no
         ``json.dump``, no filesystem writes, no subprocess. Caller is
         responsible for serializing the dict (Phase 5 writer owns lifecycle).
  D-15 — Phase 4 fills every R-P4 field that can be filled from stage outputs:
         key, mode, tempo_bpm, time_signature, time_signatures_per_part,
         measures_per_part, swing, song_arrangement (derived from transitions),
         chord_progression (per-part), active_layers, soundfonts, fx_params,
         beat_times, downbeat_times, musicality_score, duration_seconds,
         fluidsynth_version, mix/stems/midi paths.
  D-16 — Phase 5 TBD fields present as None (NOT missing, NOT "TBD" string):
         seed, musicgen_version, split, pre_roll_offset_seconds.
         The ``analysis_failed`` key is OMITTED on success (not set to False).
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

from musicgen.sampler import SongParams
from musicgen.renderer import RenderResult
from musicgen.mixer import MixResult

logger = logging.getLogger(__name__)


def _derive_mode(key: str) -> str:
    """Return "minor" if key ends in a lowercase 'm' (e.g., "Am", "C#m"),
    else "major" (D-15 mode derivation)."""
    return "minor" if key.endswith("m") else "major"


def _transitions_to_arrangement(
    mix_results: Dict[str, MixResult],
    song_arrangement_parts: List[str],
) -> List[Dict[str, object]]:
    """Convert per-part MixResult.transitions into the R-P4 song_arrangement shape.

    R-P4 schema: ``List[{part, start_seconds, end_seconds}]``. Each MixResult's
    ``transitions`` field is ``[[part, start_s], ["end", end_s]]`` — this
    helper flattens across parts in arrangement order, accumulating global
    offsets.

    Args:
        mix_results: Dict keyed by part name -> MixResult.
        song_arrangement_parts: List of parts in arrangement order (may repeat).

    Returns:
        List of per-instance dicts with keys part, start_seconds, end_seconds.
        Length equals ``len(song_arrangement_parts)``.
    """
    result = []
    cumulative = 0.0
    for part in song_arrangement_parts:
        mr = mix_results[part]
        # transitions[0] is [part_name, local_start_s]; transitions[-1] is ["end", local_end_s].
        # Local part duration = local_end_s - local_start_s.
        local_end = float(mr.transitions[-1][1])
        local_start = float(mr.transitions[0][1])
        part_duration = local_end - local_start
        result.append({
            "part": part,
            "start_seconds": round(cumulative, 3),
            "end_seconds": round(cumulative + part_duration, 3),
        })
        cumulative += part_duration
    return result


def _derive_total_duration(song_arrangement: List[Dict[str, object]]) -> float:
    """Total mix duration = end_seconds of the last arrangement entry."""
    if not song_arrangement:
        return 0.0
    return float(song_arrangement[-1]["end_seconds"])


def annotate(
    song_params: SongParams,
    render_results: Dict[str, RenderResult],
    mix_results: Dict[str, MixResult],
    beat_times: Dict[str, List[float]],
    downbeat_times: Dict[str, List[float]],
    musicality: dict,
    chord_progressions: Dict[str, List[str]],
    midi_paths: Dict[str, Dict[str, str]],
    mix_path: str,
    *,
    fluidsynth_version: str,
    musicgen_version: Optional[str] = None,
    seed: Optional[int] = None,
    split: Optional[str] = None,
    analysis_failed: Optional[bool] = None,
    pre_roll_offset_seconds: Optional[float] = None,
    genre: Optional[List[str]] = None,
) -> dict:
    """Produce the R-P4 annotation dict for one sample (Phase 4 subset, D-15/D-16).

    Pure function — zero I/O. Caller (Phase 4 orchestrator; Phase 5 writer)
    is responsible for ``json.dump`` on the returned dict.

    Args:
        song_params: Frozen SongParams from sampler (Phase 3).
        render_results: Dict of per-part RenderResult from renderer (Plan 04-02).
        mix_results: Dict of per-part MixResult from mixer (Plan 04-03).
        beat_times: Dict of per-part beat timestamps from
            :func:`musicgen.beats.extract_beat_times` (Plan 04-01).
        downbeat_times: Dict of per-part downbeat timestamps from
            :func:`musicgen.beats.extract_downbeat_times` (Plan 04-01).
        musicality: Dict with keys ``"score"`` (float) and ``"components"``
            (dict) from ``musicgen.musicality.get_musicality_score``.
        chord_progressions: Dict keyed by part -> list of chord strings.
            Threaded by the orchestrator since ``generate_song_parts``
            computes but discards this today (RESEARCH Open Question #2).
        midi_paths: Per-part per-layer MIDI paths — ``{part: {layer: path}}``.
        mix_path: Absolute path to the final mix WAV (from
            :func:`musicgen.mixer.concat_parts`).
        fluidsynth_version: Module-level ``renderer.FLUIDSYNTH_VERSION``.
        musicgen_version: Phase 5 fills. None this phase (D-16).
        seed: Phase 5 fills (R-P7 seed discipline). None this phase.
        split: Phase 5 fills (R-P6 train/valid/test split). None this phase.
        analysis_failed: Only set to ``True`` when musicality scoring raised.
            Default None (key omitted on success per D-16 clarification).

    Returns:
        Plain ``dict`` ready for ``json.dump``. Shape matches R-P4 schema with
        Phase-4-fill fields non-None and Phase-5 TBD fields as None.
    """
    # Per-part arrangement derivation.
    arrangement = _transitions_to_arrangement(mix_results, song_params.song_arrangement)
    total_duration = _derive_total_duration(arrangement)

    # Pedalboards dict: {part: {layer: info_list}} for R-P4 `fx_params`.
    fx_params = {part: dict(mr.pedalboards) for part, mr in mix_results.items()}

    # active_layers: {part: {layer: bool}} from MixResult.part_layers
    active_layers = {part: dict(mr.part_layers) for part, mr in mix_results.items()}

    # soundfonts: each MixResult records the same 4-layer dict (renderer picks
    # once at the top of the orchestrator); take from the first arrangement part.
    first_part = song_params.song_arrangement[0] if song_params.song_arrangement else None
    soundfonts_dict = dict(mix_results[first_part].soundfonts) if first_part and first_part in mix_results else {}

    # stems: per-part post-FX stem paths (Phase 5 writer will rewrite to relative paths).
    stems_per_part = {part: dict(mr.stem_paths) for part, mr in mix_results.items()}

    annotation = {
        # ---- Phase 4 FILLED (D-15) ----
        "key": song_params.key,
        "mode": _derive_mode(song_params.key),
        "tempo_bpm": song_params.tempo,
        "time_signature": song_params.time_signature_base,
        "time_signatures_per_part": dict(song_params.signatures_per_part),
        "measures_per_part": dict(song_params.measures_per_part),
        "swing": song_params.swing_amount,
        "duration_seconds": total_duration,
        "song_arrangement": arrangement,
        "chord_progression": {part: list(prog) for part, prog in chord_progressions.items()},
        "active_layers": active_layers,
        "soundfonts": soundfonts_dict,
        "fx_params": fx_params,
        "beat_times": {part: list(times) for part, times in beat_times.items()},
        "downbeat_times": {part: list(times) for part, times in downbeat_times.items()},
        "musicality_score": dict(musicality),
        "fluidsynth_version": fluidsynth_version,
        "mix": mix_path,
        "stems": stems_per_part,
        "midi": {part: dict(layers) for part, layers in midi_paths.items()},

        # ---- Phase 5 TBD (D-16: present as None, not missing) ----
        "seed": seed,
        "musicgen_version": musicgen_version,
        "split": split,
        "pre_roll_offset_seconds": pre_roll_offset_seconds,  # R-P9 Phase 6
        "genre": sorted(genre) if genre else None,
    }

    # D-16 clarification: analysis_failed is OMITTED on success, only present
    # when explicitly True. Do not emit {"analysis_failed": False}.
    if analysis_failed is True:
        annotation["analysis_failed"] = True

    return annotation
