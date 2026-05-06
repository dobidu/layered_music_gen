"""Index musicgen WAV stems into an audio_sample_manager database (Integration 3).

Usage (via CLI):
    musicgen index-audio --dataset ./dataset --out ./audio_db.json [--csv ./index.csv]

Usage (as library):
    from musicgen.audio_indexer import index_audio_dataset
    count = index_audio_dataset(dataset_root="./dataset", out_db="./audio_db.json")

Primary value: query musicgen-generated stems alongside external audio libraries in
a unified SampleManager index. SampleSelector.select_for_layer() then works across
both generated and externally-sourced audio.

sample_manager is a lazy import — this module loads without it installed.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

_LAYERS = ("beat", "melody", "harmony", "bassline")

# musicgen layer name → SampleCategory value (beat/bass/melody/harmony)
_LAYER_CATEGORY: dict[str, str] = {
    "beat":     "beat",
    "melody":   "melody",
    "harmony":  "harmony",
    "bassline": "bass",
}


def index_audio_dataset(
    dataset_root: str,
    out_db: str,
    samples_dir: Optional[str] = None,
    export_csv: Optional[str] = None,
) -> int:
    """Index musicgen WAV stems into an audio_sample_manager database.

    Walks ``dataset_root/<idx>/stems/<layer>.wav``, loads ``sample.json`` for
    ground-truth BPM/key/time_sig/mode. Librosa-based timbre and spectral features
    are extracted by ``analyze_sample`` (not in sample.json, still need re-extraction).

    Args:
        dataset_root: Root of a musicgen dataset (zero-padded sample dirs).
        out_db: Path for the output SampleManager JSON database.
        samples_dir: Optional base dir for SampleManager relative paths.
        export_csv: Optional path to write a CSV of all indexed entries.

    Returns:
        Number of WAV entries successfully indexed.

    Raises:
        ImportError: when audio_sample_manager (sample_manager module) is not installed.
        FileNotFoundError: when dataset_root does not exist.
    """
    try:
        from sample_manager import SampleManager  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "audio_sample_manager not installed — cannot run index-audio. "
            "Install dobidu/audio_sample_manager to use this command."
        ) from exc

    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    sm = SampleManager(out_db, samples_dir)
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

        stems_subdir = os.path.join(sample_dir, "stems")
        if not os.path.isdir(stems_subdir):
            logger.debug("Skipping %s — no stems/ subdir", sample_dir)
            continue

        split = sample.get("split") or "unknown"
        mode = sample.get("mode", "major")  # "major" | "minor" from annotator._derive_mode
        musicality = sample.get("musicality_score", {})
        score = (
            musicality.get("score", 0.0)
            if isinstance(musicality, dict)
            else float(musicality or 0.0)
        )

        for layer in _LAYERS:
            wav_path = os.path.join(stems_subdir, f"{layer}.wav")
            if not os.path.isfile(wav_path):
                logger.debug("Missing stem for layer %r in %s", layer, sample_dir)
                continue

            try:
                meta = sm.add_sample(wav_path, analyze=True, save=False)
            except Exception as exc:
                logger.warning("Failed to index %s: %s", wav_path, exc)
                continue

            # Override with musicgen ground-truth (beats librosa re-extraction).
            # Direct attribute assignment avoids update_sample()'s per-call save_samples().
            meta.bpm = float(sample.get("tempo_bpm", meta.bpm))
            meta.key = sample.get("key") or meta.key
            meta.time_signature = sample.get("time_signature") or meta.time_signature
            meta.scale = mode
            meta.category = _LAYER_CATEGORY[layer]
            meta.tags = ["musicgen", layer, split]
            meta.is_loop = False
            meta.description = (
                f"musicgen:{entry} layer={layer} split={split} musicality={score:.3f}"
            )
            indexed += 1

    sm.save_samples()

    if export_csv:
        sm.export_csv(export_csv)

    return indexed
