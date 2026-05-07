"""Renderer module — FluidSynth wrapper for per-layer stem rendering (R-X4).

Replaces the inline FluidSynth invocation inside ``music_gen.py:mix_and_save``
(lines 265-274) and the bare ``random.choice(sound_fonts)`` selection at
``music_gen.py:117-120``.

Design:
  D-02 — ``RenderResult`` is a frozen dataclass; shape matches Phase 3's
         ``SongParams`` convention.
  D-05 — Uses ``midi2audio.FluidSynth`` (NOT pyfluidsynth — same binary, zero
         determinism gain from switching).
  D-06 — ``ThreadPoolExecutor(max_workers=4)`` dispatches 4 per-layer renders
         in parallel. Threads suffice because FluidSynth is a subprocess (GIL
         is not held during the subprocess wait).
  D-07 — ``FLUIDSYNTH_VERSION`` captured at module import via
         ``subprocess.run(["fluidsynth", "--version"], ...)`` with a
         ``"unknown"`` fallback. NEVER raises at import (RESEARCH Pitfall 3).
  D-08 — ``pick_soundfonts(cfg, rng)`` replaces ``music_gen.py:get_random_sound_font``
         (4 bare ``random.choice`` draws move behind injected ``rng``).
  D-09 — ``render_stems`` takes a generic ``out_dir`` parameter; Phase 5 R-P1
         will handle the zero-padded-index layout.
  D-17 — Zero bare ``random.<method>`` calls. All draws via injected ``rng``.
  D-25 — ``cfg: config.Config = None`` with runtime fallback.
"""
from __future__ import annotations

import logging
import os
import random
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, Optional

from midi2audio import FluidSynth
from pydub import AudioSegment

import config
from musicgen.genre import GenreSpec

logger = logging.getLogger(__name__)


# ---------- FLUIDSYNTH_VERSION (D-07) ----------

# Capture at module import. NEVER raises — CI machines may not have FluidSynth
# installed and unit tests mock FluidSynth.midi_to_audio (Pitfall 3).
try:
    _fs_result = subprocess.run(
        ["fluidsynth", "--version"],
        capture_output=True, text=True, timeout=5,
    )
    # stdout may be empty on some platforms (A3: FluidSynth may output to stderr);
    # fall back to stderr if stdout is blank. Still tolerant of missing binary.
    _fs_output = _fs_result.stdout if _fs_result.stdout.strip() else _fs_result.stderr
    _fs_first_line = _fs_output.splitlines()[0] if _fs_output.splitlines() else ""
    FLUIDSYNTH_VERSION: str = _fs_first_line if _fs_first_line else "unknown"
    if FLUIDSYNTH_VERSION == "unknown":
        logger.warning("FluidSynth --version returned empty output; using fallback 'unknown'")
except Exception as exc:
    FLUIDSYNTH_VERSION = "unknown"
    logger.warning(
        "Could not capture FluidSynth version (%s: %s); renderer importable on CI "
        "without binary — FLUIDSYNTH_VERSION fallback to 'unknown'",
        type(exc).__name__, exc,
    )


# ---------- RenderResult frozen dataclass (D-02) ----------

@dataclass(frozen=True)
class RenderResult:
    """Per-part stem render outputs (R-X4).

    Produced by :func:`render_stems`. Consumed by :func:`musicgen.mixer.mix_part`
    and :func:`musicgen.annotator.annotate`.

    Attributes:
        stem_paths: Dict mapping layer name (``"beat"``, ``"melody"``,
            ``"harmony"``, ``"bassline"``) -> absolute path of the rendered
            stem WAV file.
        sample_rate: Output sample rate in Hz; always ``44100`` for
            FluidSynth via midi2audio default (research verified).
        channels: Output channel count; always ``2`` (stereo) for the default
            FluidSynth configuration (research verified).
        duration_seconds: Length of each stem in seconds (all four layers have
            the same duration because they share the same MIDI tempo grid).
        fluidsynth_version: The module-level ``FLUIDSYNTH_VERSION`` captured at
            import; surfaced here so each RenderResult carries provenance
            without re-querying the subprocess.
    """
    stem_paths: Dict[str, str]
    sample_rate: int
    channels: int
    duration_seconds: float
    fluidsynth_version: str


# ---------- pick_soundfonts (D-08/D-17) ----------

_LAYERS = ("beat", "melody", "harmony", "bassline")

# Preferred timbre tags per layer for SoundfontManager-backed selection.
# Tags are tried in the order soundfont_manager.get_soundfonts_by_tags receives them;
# any single match suffices (match_all=False).
_LAYER_TAGS: Dict[str, list] = {
    "beat":     ["drums", "percussion"],
    "melody":   ["melody", "lead", "piano", "strings"],
    "harmony":  ["harmony", "chords", "pads", "pad"],
    "bassline": ["bass"],
}


def _pick_via_soundfont_manager(
    cfg: config.Config,
    rng: random.Random,
    layer_tags: Optional[Dict[str, list]] = None,
) -> Optional[Dict[str, str]]:
    """SoundfontManager-backed selection. Returns None to signal directory-scan fallback.

    Lazy-imports soundfont_manager so the package remains an optional dependency.
    Candidates are sorted by path before rng.choice to guarantee cross-machine
    determinism (same seed → same pick regardless of DB insertion order).

    When ``layer_tags`` is supplied, each layer uses those tags instead of
    the static ``_LAYER_TAGS``. Layers absent from ``layer_tags`` fall back
    to ``_LAYER_TAGS[layer]``.
    """
    try:
        from soundfont_manager import SoundfontManager  # noqa: PLC0415
    except ImportError:
        logger.debug("soundfont_manager not installed; using directory-scan fallback")
        return None

    try:
        sm = SoundfontManager(cfg.soundfont_manager_db, cfg.soundfont_manager_sf_dir)
        result: Dict[str, str] = {}
        for layer in _LAYERS:
            tags = (layer_tags or {}).get(layer) or _LAYER_TAGS[layer]
            candidates = sm.get_soundfonts_by_tags(tags, match_all=False)
            if not candidates:
                logger.warning(
                    "soundfont_manager: no soundfonts tagged %r for layer %r; "
                    "using directory-scan fallback",
                    tags, layer,
                )
                return None
            candidates = sorted(candidates, key=lambda sf: sf.path)
            result[layer] = sm.get_absolute_path(rng.choice(candidates))
        return result
    except Exception as exc:
        logger.warning(
            "soundfont_manager selection failed (%s: %s); using directory-scan fallback",
            type(exc).__name__, exc,
        )
        return None


def pick_soundfonts(
    cfg: Optional[config.Config] = None,
    rng: Optional[random.Random] = None,
    genre_spec: Optional[GenreSpec] = None,
) -> Dict[str, str]:
    """Select one ``.sf2`` file per layer (D-08/D-17).

    When ``cfg.soundfont_manager_db`` is set and the ``soundfont_manager``
    package is installed, delegates to :func:`_pick_via_soundfont_manager` for
    tag-aware, metadata-rich selection. Falls back to a sorted directory scan
    when soundfont_manager is not installed, the db path is unset, or no
    soundfonts match the layer's tags.

    When ``genre_spec.soundfont_tags`` is non-empty, those tags are used
    per-layer for SM selection (layers absent from the dict fall back to
    static ``_LAYER_TAGS``). When ``genre_spec`` is None, behavior is identical
    to pre-genre (backward compat).

    Args:
        cfg: Optional Config (D-25 fallback to ``config.Config()`` if None).
        rng: Injected ``random.Random`` (required for determinism; D-17 forbids
            bare ``random.<method>`` at module scope).
        genre_spec: Optional merged :class:`GenreSpec` for soundfont tag overrides.

    Returns:
        Dict mapping layer name -> absolute ``.sf2`` path.

    Raises:
        ValueError: if ``rng`` is None (D-17 guard).
        FileNotFoundError: if no ``.sf2`` files exist for any layer (directory
            scan path only; SM path returns None on empty results).
    """
    if rng is None:
        raise ValueError("pick_soundfonts requires an injected rng (D-17)")
    _cfg = cfg if cfg is not None else config.Config()

    if getattr(_cfg, "soundfont_manager_db", None):
        layer_tags = (genre_spec.soundfont_tags or {}) if genre_spec else None
        sm_result = _pick_via_soundfont_manager(_cfg, rng, layer_tags=layer_tags or None)
        if sm_result is not None:
            return sm_result

    soundfonts: Dict[str, str] = {}
    for layer in _LAYERS:
        sf_dir = _cfg.sf_layer_dir(layer)
        sf2_files = sorted(f for f in os.listdir(sf_dir) if f.endswith(".sf2"))
        # Sorted for reproducibility — os.listdir order is filesystem-dependent;
        # the deterministic seed must produce identical choice across machines.
        if not sf2_files:
            raise FileNotFoundError(
                f"No .sf2 files found in {sf_dir} for layer {layer!r}"
            )
        soundfonts[layer] = os.path.join(sf_dir, rng.choice(sf2_files))
    return soundfonts


# ---------- render_stems (D-06/D-09) ----------

def render_stems(
    midi_paths: Dict[str, str],
    soundfonts: Dict[str, str],
    out_dir: str,
    cfg: Optional[config.Config] = None,
) -> RenderResult:
    """Render 4 per-layer stems in parallel via ThreadPoolExecutor (D-06).

    Dispatches one ``FluidSynth(sf).midi_to_audio(midi, wav)`` per layer
    through ``ThreadPoolExecutor(max_workers=4)``. Parts remain serial in the
    caller (D-06). No RNG draws (deterministic given the same soundfonts +
    MIDI files); the outer RNG drew in ``pick_soundfonts``.

    Args:
        midi_paths: Dict mapping layer name -> MIDI file path.
        soundfonts: Dict mapping layer name -> ``.sf2`` path (from
            :func:`pick_soundfonts`).
        out_dir: Destination directory for the 4 stem WAVs. Created if it does
            not exist (D-09: Phase 4 uses generic ``out_dir``; Phase 5 R-P1
            replaces with zero-padded-index layout).
        cfg: Optional Config (D-25; not read by renderer but kept for signature
            uniformity with the rest of Phase 4 modules).

    Returns:
        ``RenderResult`` with ``stem_paths``, ``sample_rate=44100``,
        ``channels=2``, ``duration_seconds`` read from the first written WAV,
        and ``fluidsynth_version`` from the module-level constant.

    Raises:
        KeyError: if ``midi_paths`` or ``soundfonts`` is missing any of the 4
            canonical layer keys.
    """
    _ = cfg if cfg is not None else config.Config()  # reserved for future use
    os.makedirs(out_dir, exist_ok=True)

    for layer in _LAYERS:
        if layer not in midi_paths:
            raise KeyError(f"midi_paths missing layer {layer!r}")
        if layer not in soundfonts:
            raise KeyError(f"soundfonts missing layer {layer!r}")

    def _render_one(layer: str) -> tuple[str, str]:
        wav_path = os.path.join(out_dir, f"{layer}.wav")
        FluidSynth(soundfonts[layer]).midi_to_audio(midi_paths[layer], wav_path)
        return layer, wav_path

    stem_paths: Dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(_render_one, layer): layer for layer in _LAYERS}
        for future in as_completed(futures):
            layer, wav_path = future.result()
            stem_paths[layer] = wav_path

    # Read duration from the first stem; all four have the same MIDI tempo
    # grid so durations match to within one sample.
    first_layer = _LAYERS[0]
    first_audio = AudioSegment.from_wav(stem_paths[first_layer])
    duration_seconds = first_audio.duration_seconds

    return RenderResult(
        stem_paths=stem_paths,
        sample_rate=44100,
        channels=2,
        duration_seconds=float(duration_seconds),
        fluidsynth_version=FLUIDSYNTH_VERSION,
    )
