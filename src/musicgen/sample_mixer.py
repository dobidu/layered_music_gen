"""M4 — SampleMixer: prepare and mix real audio samples into the pipeline.

Four public operations:

  prepare_sample_wav()      — stretch BPM, shift key, tile to part duration
  apply_substitutions()     — replace FluidSynth stem paths (substitution mode)
  apply_alongside()         — overlay sample on part mix WAV (alongside/adlib)
  integrate_samples()       — high-level call used by api._run_pipeline()

BPM time-stretching uses rubberband-stretch (pip install musicgen[samples]).
Key transposition uses librosa.effects.pitch_shift (already a core dependency).
Both operations degrade gracefully when the dependency or data is absent.
"""
from __future__ import annotations

import dataclasses
import logging
import math
import os
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Semitone offsets for root note names (enharmonic equivalents collapsed).
_ROOT_SEMI: Dict[str, int] = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "Fb": 4, "F": 5, "E#": 5, "F#": 6, "Gb": 6,
    "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10,
    "B": 11, "Cb": 11, "B#": 0,
}


def _parse_key_root(key: str) -> Optional[int]:
    """Return semitone (0-11) for a key string like 'Am', 'F#', 'Bb'. None on parse fail."""
    if not key:
        return None
    root = key.rstrip("m")  # strip minor suffix
    return _ROOT_SEMI.get(root)


def _semitone_distance(from_key: str, to_key: str) -> int:
    """Shortest semitone distance (−6..+6) between two key roots."""
    f = _parse_key_root(from_key)
    t = _parse_key_root(to_key)
    if f is None or t is None:
        return 0
    diff = (t - f) % 12
    if diff > 6:
        diff -= 12
    return diff


# ---------------------------------------------------------------------------
# Audio I/O helpers
# ---------------------------------------------------------------------------

def _load_audio(path: str) -> Tuple[np.ndarray, int]:
    """Load WAV to float32 ndarray (samples, channels). Raises on failure."""
    import soundfile as sf  # type: ignore[import]
    audio, sr = sf.read(path, always_2d=True, dtype="float32")
    return audio, sr


def _write_audio(path: str, audio: np.ndarray, sr: int) -> None:
    import soundfile as sf  # type: ignore[import]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    sf.write(path, audio, sr)


# ---------------------------------------------------------------------------
# Core transforms
# ---------------------------------------------------------------------------

def _normalize_bpm(audio: np.ndarray, sr: int, from_bpm: float, to_bpm: float) -> np.ndarray:
    """Time-stretch audio so it plays at to_bpm. Uses rubberband when available."""
    if abs(from_bpm - to_bpm) < 0.5 or from_bpm <= 0:
        return audio
    time_ratio = from_bpm / to_bpm  # <1 → speed up, >1 → slow down
    try:
        import rubberband as rb  # type: ignore[import]
        # rubberband.stretch(audio, sr, time_ratio, pitch_ratio)
        # audio shape: (samples, channels) or (samples,)
        return rb.stretch(audio, sr, time_ratio, 1.0)
    except ImportError:
        logger.warning("rubberband-stretch not installed; BPM normalization skipped")
        return audio
    except Exception as exc:
        logger.warning("rubberband stretch failed (%s); using original audio", exc)
        return audio


def _normalize_key(audio: np.ndarray, sr: int, from_key: str, to_key: str) -> np.ndarray:
    """Pitch-shift audio by the semitone distance between from_key and to_key."""
    semitones = _semitone_distance(from_key, to_key)
    if semitones == 0:
        return audio
    try:
        import librosa  # type: ignore[import]
        shifted = np.stack(
            [
                librosa.effects.pitch_shift(audio[:, ch], sr=sr, n_steps=float(semitones))
                for ch in range(audio.shape[1])
            ],
            axis=1,
        )
        return shifted
    except Exception as exc:
        logger.warning("librosa pitch_shift failed (%s); using original audio", exc)
        return audio


def _tile_to_duration(audio: np.ndarray, sr: int, duration_s: float) -> np.ndarray:
    """Tile/loop audio array to exactly duration_s seconds."""
    target = int(math.ceil(duration_s * sr))
    if target <= 0 or len(audio) == 0:
        return audio
    repeats = math.ceil(target / len(audio))
    tiled = np.tile(audio, (repeats, 1))
    return tiled[:target]


# ---------------------------------------------------------------------------
# prepare_sample_wav — public high-level transform
# ---------------------------------------------------------------------------

def prepare_sample_wav(
    sample_path: str,
    target_bpm: float,
    target_key: str,
    duration_s: float,
    out_path: str,
    sample_bpm: Optional[float] = None,
    sample_key: Optional[str] = None,
    allow_time_stretch: bool = True,
    allow_transpose: bool = True,
) -> str:
    """Load sample, normalize BPM + key, tile to duration, write WAV.

    Args:
        sample_path: source WAV path.
        target_bpm: composition BPM.
        target_key: composition key (e.g. 'Am', 'F#').
        duration_s: desired output duration in seconds.
        out_path: destination WAV path.
        sample_bpm: sample's native BPM (None → skip stretch).
        sample_key: sample's native key (None → skip shift).
        allow_time_stretch: False disables rubberband stretch.
        allow_transpose: False disables pitch shifting.

    Returns:
        out_path on success.

    Raises:
        RuntimeError: if soundfile is unavailable or the source cannot be read.
    """
    try:
        audio, sr = _load_audio(sample_path)
    except Exception as exc:
        raise RuntimeError(f"Cannot load sample {sample_path}: {exc}") from exc

    if allow_time_stretch and sample_bpm:
        audio = _normalize_bpm(audio, sr, sample_bpm, target_bpm)

    if allow_transpose and sample_key:
        audio = _normalize_key(audio, sr, sample_key, target_key)

    audio = _tile_to_duration(audio, sr, duration_s)
    _write_audio(out_path, audio, sr)
    return out_path


# ---------------------------------------------------------------------------
# Pipeline integration helpers
# ---------------------------------------------------------------------------

def apply_substitutions(
    render_result,
    sample_wavs: Dict[str, str],
    layer_rules: Dict[str, "SampleLayerRule"],  # noqa: F821
) -> object:
    """Return a new RenderResult with substituted stem paths for 'substitution' layers.

    The original RenderResult is frozen; this creates a dataclasses.replace() copy.
    """
    from musicgen.sample_composition import SampleLayerRule  # local import avoids circularity

    overrides: Dict[str, str] = {}
    for layer, rule in layer_rules.items():
        if rule.mode == "substitution" and layer in sample_wavs:
            overrides[layer] = sample_wavs[layer]

    if not overrides:
        return render_result

    new_stem_paths = {**render_result.stem_paths, **overrides}
    return dataclasses.replace(render_result, stem_paths=new_stem_paths)


def apply_alongside(
    mix_wav_path: str,
    sample_wavs: Dict[str, str],
    layer_rules: Dict[str, "SampleLayerRule"],  # noqa: F821
    bpm: float,
) -> str:
    """Overlay alongside/adlib sample WAVs onto the part mix WAV in-place.

    The mix_wav_path file is overwritten with the updated mix. Returns mix_wav_path.
    """
    from pydub import AudioSegment  # type: ignore[import]

    mix = AudioSegment.from_wav(mix_wav_path)
    modified = False

    for layer, rule in layer_rules.items():
        if rule.mode not in ("alongside", "adlib"):
            continue
        if layer not in sample_wavs:
            continue

        sample = AudioSegment.from_wav(sample_wavs[layer])
        sample = sample.apply_gain(rule.gain_db)

        if rule.mode == "alongside":
            mix = mix.overlay(sample)
            modified = True
        elif rule.mode == "adlib" and rule.oneshot_at_beat is not None:
            beat_ms = int((60.0 / bpm) * rule.oneshot_at_beat * 1000)
            mix = mix.overlay(sample, position=beat_ms)
            modified = True

    if modified:
        mix.export(mix_wav_path, format="wav")
    return mix_wav_path


# ---------------------------------------------------------------------------
# prepare_sample_wavs — called per part, before mix_part()
# ---------------------------------------------------------------------------

def prepare_sample_wavs(
    selected_samples: Dict[str, object],
    layer_rules: Dict[str, object],
    composition_key: str,
    composition_bpm: float,
    part_duration_s: float,
    out_dir: str,
    allow_transpose: bool = True,
    allow_time_stretch: bool = True,
) -> Dict[str, str]:
    """Stretch, shift, and tile each selected sample to match the current part.

    Called BEFORE mix_part() so the result can be used by both apply_substitutions()
    (pre-mix) and apply_alongside() (post-mix).

    Args:
        selected_samples: layer → SampleMetadata, from sample_composition.select_samples().
        layer_rules: layer → SampleLayerRule, from SampleCompositionConfig.layer_rules.
        composition_key: composition key string (e.g. 'Am', 'F#').
        composition_bpm: composition tempo in BPM.
        part_duration_s: this part's duration in seconds (used for tiling loops).
        out_dir: directory to write prepared WAVs into.
        allow_transpose: if False, skip pitch shifting.
        allow_time_stretch: if False, skip BPM stretching.

    Returns:
        Dict mapping layer name → absolute path of the prepared WAV.
        Layers that fail (missing file, soundfile error) are omitted.
    """
    sample_wavs: Dict[str, str] = {}

    for layer, sample_meta in selected_samples.items():
        if layer not in layer_rules:
            continue

        src_path = getattr(sample_meta, "file_path", None) or getattr(sample_meta, "path", None)
        if not src_path or not os.path.isfile(src_path):
            logger.warning("Layer %s: sample file not found at %r — skipped", layer, src_path)
            continue

        out_wav = os.path.join(out_dir, f"sample_{layer}.wav")
        try:
            prepare_sample_wav(
                sample_path=src_path,
                target_bpm=composition_bpm,
                target_key=composition_key,
                duration_s=part_duration_s,
                out_path=out_wav,
                sample_bpm=getattr(sample_meta, "bpm", None),
                sample_key=getattr(sample_meta, "key", None),
                allow_time_stretch=allow_time_stretch,
                allow_transpose=allow_transpose,
            )
            sample_wavs[layer] = out_wav
            logger.debug("Layer %s: prepared sample WAV → %s", layer, out_wav)
        except Exception as exc:
            logger.warning("Layer %s: prepare_sample_wav failed (%s) — skipped", layer, exc)

    return sample_wavs
