"""Writer module — atomic per-sample layout (R-P1/R-P2/R-P3, D-04..D-12/D-22..D-25).

The writer OWNS the transition from Phase 4's working-dir layout
(``<working>/<name>-<part>/*``) to the final per-sample dataset layout
(``<dataset_root>/<idx:06d>/*``). Its ordering invariant is load-bearing:

    1. Per-layer MIDI concatenation from per-part MIDIs (absolute-tick walk).
    2. Per-layer stem WAV concatenation (pydub ``AudioSegment.from_wav +``).
    3. Final ``mix.wav`` copied from working-dir.
    4. Sum-of-stems assertion (int32 accumulator, ε=1e-3 normalized float).
    5. Path rewrite on the annotation dict (deep-copy; per-sample-dir-relative).
    6. Canonical ``sample.json`` serialization (sort_keys, indent=2).
    7. Atomic sentinel rename (``sample.json.tmp`` → ``sample.json``).

If any step before 7 raises, the sentinel never exists and the sample is
resume-invisible (Phase 6 retries the index). D-04's sentinel invariant is
the foundation of the Phase 6 resume path.
"""
from __future__ import annotations

import copy
import json
import logging
import os
import shutil
from typing import Dict, List, Tuple

import mido
import numpy as np
import scipy.io.wavfile as wf
from pydub import AudioSegment

logger = logging.getLogger(__name__)

_LAYERS = ("beat", "melody", "harmony", "bassline")


def write_sample(
    dataset_root: str,
    sample_index: int,
    annotation: dict,
    mix_working_path: str,
    stems_working_paths: Dict[str, Dict[str, str]],
    midi_working_paths: Dict[str, Dict[str, str]],
    song_arrangement: List[str],
    tempo_bpm: int,
    part_durations_s: List[float],
    *,
    fluidsynth_version: str,
    split: str,
    sum_of_stems_epsilon: float = 1e-3,
) -> Dict[str, str]:
    """Write the atomic per-sample layout (D-04 ordering).

    Args:
        dataset_root: Absolute path to the dataset directory.
        sample_index: Zero-based index. Zero-padded to width 6 (D-05).
        annotation: Dict from :func:`annotator.annotate` with kwargs already
            filled by api.py (seed, musicgen_version, split per D-22).
            Keys ``mix``, ``stems``, ``midi`` will be path-rewritten (D-11/D-12).
        mix_working_path: Absolute path to the final concatenated mix in
            the working dir (produced by ``mixer.concat_parts``).
        stems_working_paths: ``{part_name: {layer: abs_path}}`` — per-part
            per-layer post-FX stem paths (from ``mixer.mix_part``).
        midi_working_paths: ``{part_name: {layer: abs_path}}`` — per-part
            per-layer MIDI paths (from ``generators/*``).
        song_arrangement: List of part names in playback order — drives the
            concat iteration order for both stems and MIDIs.
        tempo_bpm: Integer tempo; used by ``_concat_layer_midis`` to convert
            seconds → ticks for the absolute-tick walk.
        part_durations_s: Per-part audio durations in seconds — same length
            and order as ``song_arrangement``. Used for MIDI tick offsets.
        fluidsynth_version: Captured at render time; included in the final
            annotation as-is (annotator already has it, writer just preserves).
        split: ``"train"``, ``"valid"``, or ``"test"`` from ``assign_split``.
        sum_of_stems_epsilon: Max allowed ``|sum(stems) - mix|`` in normalized
            float32. Default 1e-3 per D-25.

    Returns:
        Dict of final paths: ``{"sample_dir", "sample_json", "mix",
        "stems_<layer>", "midi_<layer>"}``.

    Raises:
        AssertionError: sum-of-stems check failed — sample.json NOT written.
        ValueError: from ``_concat_layer_midis`` if ``part_midi_paths`` empty,
            or from ``_assert_sum_of_stems`` if stem/mix shapes mismatch.
    """
    sample_dir = os.path.join(dataset_root, f"{sample_index:06d}")
    stems_dir = os.path.join(sample_dir, "stems")
    midi_dir = os.path.join(sample_dir, "midi")
    os.makedirs(stems_dir, exist_ok=True)
    os.makedirs(midi_dir, exist_ok=True)

    # Step 1: per-layer MIDI concat (absolute-tick walk, RESEARCH Pitfall 1).
    midi_final_paths: Dict[str, str] = {}
    for layer in _LAYERS:
        part_midi_paths = [
            midi_working_paths[part][layer] for part in song_arrangement
        ]
        midi_final_paths[layer] = _concat_layer_midis(
            part_midi_paths, part_durations_s, tempo_bpm,
            os.path.join(midi_dir, f"{layer}.mid"),
        )

    # Step 2: per-layer stem WAV concat (pydub, matches mixer.concat_parts).
    stem_final_paths: Dict[str, str] = {}
    for layer in _LAYERS:
        part_stem_paths = [
            stems_working_paths[part][layer] for part in song_arrangement
        ]
        stem_final_paths[layer] = _concat_layer_stems(
            part_stem_paths,
            os.path.join(stems_dir, f"{layer}.wav"),
        )

    # Step 3: mix.wav — copy from working dir (different filesystem may apply).
    mix_final = os.path.join(sample_dir, "mix.wav")
    shutil.copy2(mix_working_path, mix_final)

    # Step 4: sum-of-stems assertion (RESEARCH Pitfall 2 — int32 accumulator).
    passed, max_diff = _assert_sum_of_stems(
        mix_final, stem_final_paths, epsilon=sum_of_stems_epsilon,
    )
    if not passed:
        raise AssertionError(
            f"sum_of_stems_exceeded: max |Σstems − mix| = {max_diff:.6f} "
            f"> ε = {sum_of_stems_epsilon:.6f}"
        )

    # Step 5: path rewrite (deep-copy; D-11/D-12 — annotator stays pure).
    final_annotation = _rewrite_paths_relative(
        annotation, stem_final_paths.keys(), midi_final_paths.keys(),
    )
    final_annotation["split"] = split
    # fluidsynth_version preserved from annotator — no rewrite needed.

    # Step 6+7: canonical serialization + atomic sentinel rename.
    sample_json_tmp = os.path.join(sample_dir, "sample.json.tmp")
    sample_json_final = os.path.join(sample_dir, "sample.json")
    with open(sample_json_tmp, "w") as f:
        json.dump(
            final_annotation, f,
            sort_keys=True, indent=2, separators=(",", ": "),
        )
    os.rename(sample_json_tmp, sample_json_final)

    logger.info(
        "Wrote sample %d to %s (split=%s, max |Σstems-mix|=%.6f)",
        sample_index, sample_dir, split, max_diff,
    )

    result = {
        "sample_dir": sample_dir,
        "sample_json": sample_json_final,
        "mix": mix_final,
    }
    for layer in _LAYERS:
        result[f"stems_{layer}"] = stem_final_paths[layer]
        result[f"midi_{layer}"] = midi_final_paths[layer]
    return result


def _concat_layer_stems(
    part_stem_paths: List[str], out_path: str,
) -> str:
    """Concatenate per-part WAVs for one layer (D-06). Mirrors mixer.concat_parts."""
    if not part_stem_paths:
        raise ValueError(
            "_concat_layer_stems: no part stem paths provided"
        )
    song = AudioSegment.from_wav(part_stem_paths[0])
    for part_wav in part_stem_paths[1:]:
        song += AudioSegment.from_wav(part_wav)

    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    song.export(out_path, format="wav")
    return out_path


def _concat_layer_midis(
    part_midi_paths: List[str],
    part_durations_s: List[float],
    tempo_bpm: int,
    out_path: str,
) -> str:
    """Absolute-tick walk MIDI concat (RESEARCH Pitfall 1, Code Examples 497-554)."""
    if not part_midi_paths:
        raise ValueError(
            "_concat_layer_midis: no part midi paths provided"
        )
    if len(part_midi_paths) != len(part_durations_s):
        raise ValueError(
            f"_concat_layer_midis: length mismatch — "
            f"{len(part_midi_paths)} paths vs {len(part_durations_s)} durations"
        )

    first = mido.MidiFile(part_midi_paths[0])
    ticks_per_beat = first.ticks_per_beat
    tempo_us = mido.bpm2tempo(tempo_bpm)
    midi_type = first.type

    merged = mido.MidiFile(ticks_per_beat=ticks_per_beat, type=midi_type)
    num_tracks = len(first.tracks)

    def _track_to_absolute(track):
        t = 0
        abs_msgs = []
        for msg in track:
            t += msg.time
            if msg.is_meta and msg.type == "end_of_track":
                continue
            abs_msgs.append((t, msg.copy(time=0)))
        return abs_msgs

    def _absolute_to_track(abs_msgs, end_tick):
        abs_msgs = sorted(abs_msgs, key=lambda x: x[0])
        tr = mido.MidiTrack()
        prev = 0
        for t, msg in abs_msgs:
            tr.append(msg.copy(time=t - prev))
            prev = t
        tr.append(mido.MetaMessage("end_of_track", time=max(0, end_tick - prev)))
        return tr

    for tr_idx in range(num_tracks):
        merged_abs = []
        offset_ticks = 0
        for part_path, part_dur_s in zip(part_midi_paths, part_durations_s):
            part = mido.MidiFile(part_path)
            # Each part's track count may differ (e.g. silent MIDI may have
            # fewer tracks) — only iterate tracks that exist in this part.
            if tr_idx >= len(part.tracks):
                offset_ticks += int(
                    mido.second2tick(part_dur_s, ticks_per_beat, tempo_us)
                )
                continue
            abs_msgs = _track_to_absolute(part.tracks[tr_idx])
            merged_abs.extend([(t + offset_ticks, m) for (t, m) in abs_msgs])
            offset_ticks += int(
                mido.second2tick(part_dur_s, ticks_per_beat, tempo_us)
            )
        merged.tracks.append(_absolute_to_track(merged_abs, end_tick=offset_ticks))

    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    merged.save(out_path)
    return out_path


def _assert_sum_of_stems(
    mix_path: str,
    stem_paths: Dict[str, str],
    epsilon: float = 1e-3,
) -> Tuple[bool, float]:
    """Int32 accumulator sum-of-stems (RESEARCH Pitfall 2, Code Examples 556-584)."""
    _, mix_i16 = wf.read(mix_path)
    mix_i32 = mix_i16.astype(np.int32)
    sums_i32 = np.zeros_like(mix_i32)
    for layer, path in stem_paths.items():
        _, stem_i16 = wf.read(path)
        if stem_i16.shape != mix_i16.shape:
            raise ValueError(
                f"stem {layer!r} shape {stem_i16.shape} != "
                f"mix shape {mix_i16.shape}"
            )
        sums_i32 += stem_i16.astype(np.int32)
    max_abs_int = int(np.max(np.abs(sums_i32 - mix_i32)))
    max_abs_float = max_abs_int / 32768.0
    return (max_abs_float < epsilon, max_abs_float)


def _rewrite_paths_relative(
    annotation: dict,
    stem_layers,
    midi_layers,
) -> dict:
    """Deep-copy annotation + rewrite mix/stems/midi path fields (D-11/D-12)."""
    final = copy.deepcopy(annotation)
    final["mix"] = "mix.wav"
    final["stems"] = {layer: f"stems/{layer}.wav" for layer in stem_layers}
    final["midi"] = {layer: f"midi/{layer}.mid" for layer in midi_layers}
    return final
