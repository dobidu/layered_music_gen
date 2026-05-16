"""Corpus extractor — sequence extraction from musicgen datasets (v0.5, Phase 1).

Walks a musicgen dataset root, reads per-sample ``sample.json`` annotations and
melody MIDI files, and produces a ``sequences.json`` file used for training
the neural chord/melody backends.

Output schema::

    {
        "metadata": {
            "n_samples": int,
            "genres": [str, ...],        # sorted unique list of genres seen
            "musicgen_version": str | null
        },
        "chord": [
            {
                "sample_index": int,
                "genre": [str] | null,
                "key": str,
                "full_sequence": [str, ...]  # Roman numerals, arrangement order
            },
            ...
        ],
        "melody": [
            {
                "sample_index": int,
                "genre": [str] | null,
                "key": str,
                "full_sequence": [str, ...]  # scale degrees "1"–"7", arrangement order
            },
            ...
        ]
    }

The ``full_sequence`` fields concatenate all song parts in ``song_arrangement``
order, giving the LSTM an unbroken token stream per sample.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mido

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Scale-degree helpers (mirrors melody.py — kept local to avoid circular deps)
# ---------------------------------------------------------------------------

_MAJOR_INTERVALS: Dict[str, int] = {
    "1": 0, "2": 2, "3": 4, "4": 5, "5": 7, "6": 9, "7": 11,
}
_MINOR_INTERVALS: Dict[str, int] = {
    "1": 0, "2": 2, "3": 3, "4": 5, "5": 7, "6": 8, "7": 10,
}
_NOTE_TO_SEMI: Dict[str, int] = {
    "C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5,
    "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11,
    "Db": 1, "Eb": 3, "Gb": 6, "Ab": 8, "Bb": 10,
}


def _key_root_and_intervals(key: str) -> Tuple[int, Dict[str, int]]:
    is_minor = key.endswith("m") and len(key) > 1 and key[-2] not in ("#", "b")
    root_name = key[:-1] if is_minor else key
    root_semi = _NOTE_TO_SEMI.get(root_name, 0)
    intervals = _MINOR_INTERVALS if is_minor else _MAJOR_INTERVALS
    return root_semi, intervals


def _build_semi_to_degree(intervals: Dict[str, int]) -> Dict[int, str]:
    """Reverse map: semitone offset (0–11) → scale degree string."""
    return {v: k for k, v in intervals.items()}


def _midi_note_to_degree(midi_note: int, key: str) -> Optional[str]:
    """Map a MIDI note number to a scale-degree string ("1"–"7") for the given key.

    Returns None when the note cannot be mapped (e.g. chromatic passing note).
    Out-of-scale pitches are mapped to the nearest scale degree via minimum
    semitone distance (mod 12).
    """
    root_semi, intervals = _key_root_and_intervals(key)
    pitch_class = (midi_note - root_semi) % 12
    semi_to_deg = _build_semi_to_degree(intervals)

    if pitch_class in semi_to_deg:
        return semi_to_deg[pitch_class]

    # Nearest neighbour (chromatic passing note)
    best_deg = None
    best_dist = 13
    for semi, deg in semi_to_deg.items():
        dist = min(abs(pitch_class - semi), 12 - abs(pitch_class - semi))
        if dist < best_dist:
            best_dist = dist
            best_deg = deg
    return best_deg


# ---------------------------------------------------------------------------
# MIDI parsing
# ---------------------------------------------------------------------------

def _extract_melody_degrees_from_midi(midi_path: str, key: str) -> List[str]:
    """Parse a melody MIDI file and return an ordered list of scale degrees.

    Only ``note_on`` events with velocity > 0 on any channel are collected.
    The MIDI is assumed to be the concatenated per-layer melody file produced
    by ``writer.write_sample``.
    """
    try:
        mid = mido.MidiFile(midi_path)
    except Exception as exc:
        logger.warning("Failed to parse %s: %s", midi_path, exc)
        return []

    degrees: List[str] = []
    for track in mid.tracks:
        for msg in track:
            if msg.type == "note_on" and msg.velocity > 0:
                deg = _midi_note_to_degree(msg.note, key)
                if deg is not None:
                    degrees.append(deg)

    return degrees


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_sequences(
    dataset_root: str,
    output_path: str,
    min_musicality: float = 0.0,
    skip_errors: bool = True,
) -> Dict[str, int]:
    """Walk a musicgen dataset root and write ``sequences.json``.

    Args:
        dataset_root: Root directory produced by ``musicgen generate``.
        output_path: Path to write the output ``sequences.json``.
        min_musicality: Skip samples whose ``musicality_score.score`` is below
            this threshold (0.0 = keep all).
        skip_errors: If True, log warnings and skip malformed samples rather
            than raising.

    Returns:
        Dict with keys ``"chord"`` and ``"melody"`` counting extracted
        sequences per layer.
    """
    dataset_root = os.path.abspath(dataset_root)
    output_path = os.path.abspath(output_path)

    chord_entries: List[dict] = []
    melody_entries: List[dict] = []
    genres_seen: set = set()
    musicgen_version: Optional[str] = None
    n_skipped = 0

    sample_dirs = sorted(
        p for p in Path(dataset_root).iterdir()
        if p.is_dir() and p.name.isdigit()
    )

    for sample_dir in sample_dirs:
        sj_path = sample_dir / "sample.json"
        if not sj_path.exists():
            logger.debug("No sample.json in %s — skipping", sample_dir)
            n_skipped += 1
            continue

        try:
            sj = json.loads(sj_path.read_text())
        except Exception as exc:
            if skip_errors:
                logger.warning("Bad sample.json in %s: %s", sample_dir, exc)
                n_skipped += 1
                continue
            raise

        # Musicality gate
        ms = sj.get("musicality_score") or {}
        if isinstance(ms, dict):
            score = ms.get("score", 1.0)
        else:
            score = float(ms) if ms is not None else 1.0
        if score < min_musicality:
            n_skipped += 1
            continue

        key: str = sj.get("key", "C")
        genre: Optional[List[str]] = sj.get("genre")
        sample_index: int = int(sample_dir.name)
        if musicgen_version is None:
            musicgen_version = sj.get("musicgen_version")

        if genre:
            genres_seen.update(genre)

        # --- Chord sequence ---
        chord_prog: dict = sj.get("chord_progression") or {}
        arrangement: List[dict] = sj.get("song_arrangement") or []
        arrangement_parts = [a["part"] for a in arrangement]

        chord_full: List[str] = []
        for part in arrangement_parts:
            chord_full.extend(chord_prog.get(part) or [])

        if chord_full:
            chord_entries.append({
                "sample_index": sample_index,
                "genre": genre,
                "key": key,
                "full_sequence": chord_full,
            })

        # --- Melody sequence ---
        melody_midi = sample_dir / "midi" / "melody.mid"
        if melody_midi.exists():
            degrees = _extract_melody_degrees_from_midi(str(melody_midi), key)
            if degrees:
                melody_entries.append({
                    "sample_index": sample_index,
                    "genre": genre,
                    "key": key,
                    "full_sequence": degrees,
                })
        else:
            logger.debug("No melody MIDI in %s", sample_dir)

    metadata = {
        "n_samples": len(sample_dirs),
        "n_skipped": n_skipped,
        "genres": sorted(genres_seen),
        "musicgen_version": musicgen_version,
    }

    output = {
        "metadata": metadata,
        "chord": chord_entries,
        "melody": melody_entries,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(
        "Extracted %d chord sequences, %d melody sequences from %d samples → %s",
        len(chord_entries), len(melody_entries), len(sample_dirs), output_path,
    )
    return {"chord": len(chord_entries), "melody": len(melody_entries)}
