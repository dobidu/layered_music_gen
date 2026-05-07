"""Index a musicgen dataset into a MidiManager database (Integration 2).

Usage (via CLI):
    musicgen index-midi --dataset ./dataset --out ./midi_db.json [--csv ./index.csv]

Usage (as library):
    from musicgen.midi_indexer import index_midi_dataset
    count = index_midi_dataset(dataset_root="./dataset", out_db="./midi_db.json")

midi_manager is a lazy import — this module loads without it installed.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

_LAYERS = ("beat", "melody", "harmony", "bassline")

# musicgen layer name → MidiCategory value (from midi_utils.MidiCategory enum)
_LAYER_CATEGORY: dict[str, str] = {
    "beat":     "drums",
    "melody":   "melody",
    "harmony":  "harmony",
    "bassline": "bass",
}


def index_midi_dataset(
    dataset_root: str,
    out_db: str,
    midi_dir: Optional[str] = None,
    export_csv: Optional[str] = None,
) -> int:
    """Index all MIDI files in a musicgen dataset into a MidiManager database.

    Walks ``dataset_root/<idx:06d>/midi/<layer>.mid``, loads the paired
    ``sample.json`` for ground-truth metadata (key, bpm, time_sig, split,
    musicality), and stores the enriched index at ``out_db``.

    Ground-truth from sample.json overrides MIDI-extracted values so the index
    reflects the generative parameters, not re-extracted estimates.

    Args:
        dataset_root: Root of a musicgen dataset (zero-padded sample dirs).
        out_db: Path for the output MidiManager JSON database.
        midi_dir: Optional base dir for MidiManager relative paths. When None,
                  absolute paths are stored in the db.
        export_csv: Optional path to write a CSV of all indexed entries.

    Returns:
        Number of MIDI entries successfully indexed.

    Raises:
        ImportError: when midi_file_manager is not installed.
        FileNotFoundError: when dataset_root does not exist.
    """
    try:
        from midi_manager import MidiManager  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "midi_file_manager not installed — cannot run index-midi. "
            "Install dobidu/midi_file_manager to use this command."
        ) from exc

    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    mm = MidiManager(out_db, midi_dir)
    indexed = 0

    for entry in sorted(os.listdir(dataset_root)):
        sample_dir = os.path.join(dataset_root, entry)
        if not os.path.isdir(sample_dir):
            continue

        sample_json_path = os.path.join(sample_dir, "sample.json")
        if not os.path.isfile(sample_json_path):
            logger.debug("Skipping %s — no sample.json", sample_dir)
            continue

        try:
            with open(sample_json_path) as f:
                sample = json.load(f)
        except Exception as exc:
            logger.warning("Failed to load %s: %s", sample_json_path, exc)
            continue

        midi_subdir = os.path.join(sample_dir, "midi")
        if not os.path.isdir(midi_subdir):
            logger.debug("Skipping %s — no midi/ subdir", sample_dir)
            continue

        split = sample.get("split") or "unknown"
        musicality = sample.get("musicality_score", {})
        score = (
            musicality.get("score", 0.0)
            if isinstance(musicality, dict)
            else float(musicality or 0.0)
        )

        for layer in _LAYERS:
            midi_path = os.path.join(midi_subdir, f"{layer}.mid")
            if not os.path.isfile(midi_path):
                logger.debug("Missing MIDI for layer %r in %s", layer, sample_dir)
                continue

            try:
                meta = mm.add_midi(midi_path, analyze=True, save=False)
            except Exception as exc:
                logger.warning("Failed to index %s: %s", midi_path, exc)
                continue

            # Enrich with musicgen ground-truth (overrides MIDI-extracted values).
            # Direct attribute assignment avoids update_midi()'s per-call save_midis().
            meta.bpm = float(sample.get("tempo_bpm", meta.bpm))
            meta.key = sample.get("key") or meta.key
            meta.time_signature = sample.get("time_signature") or meta.time_signature
            meta.category = _LAYER_CATEGORY[layer]
            meta.tags = ["musicgen", layer, split]
            meta.description = (
                f"musicgen:{entry} layer={layer} split={split} musicality={score:.3f}"
            )
            indexed += 1

    mm.save_midis()

    if export_csv:
        mm.export_to_csv(export_csv)

    return indexed
