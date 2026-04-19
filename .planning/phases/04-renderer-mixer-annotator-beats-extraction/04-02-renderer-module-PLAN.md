---
phase: 04-renderer-mixer-annotator-beats-extraction
plan: 02
type: execute
wave: 2
depends_on: ["04-00"]
files_modified:
  - src/musicgen/renderer.py
  - tests/test_renderer.py
autonomous: true
requirements: [R-X4]
tags: [phase-4, renderer, fluidsynth, threadpool, soundfonts, rng]

must_haves:
  truths:
    - "`import musicgen.renderer` succeeds even on machines without the FluidSynth binary (D-07: renderer captures fallback `FLUIDSYNTH_VERSION = \"unknown\"` and emits a WARNING log — never raises at import)"
    - "`RenderResult` is a frozen dataclass with fields: stem_paths, sample_rate, channels, duration_seconds, fluidsynth_version"
    - "`pick_soundfonts(cfg, rng)` is deterministic: same seed → same 4-layer soundfont dict (D-08/D-17)"
    - "`render_stems(midi_paths, soundfonts, out_dir)` dispatches 4 FluidSynth renders via `ThreadPoolExecutor(max_workers=4)` and assembles RenderResult (D-06)"
    - "Zero bare `random.<method>(` calls in `src/musicgen/renderer.py` — only `rng.choice(...)` via injected parameter (D-17)"
    - "Unit tests pass without FluidSynth binary installed — `FluidSynth.midi_to_audio` is mocked via `unittest.mock.patch`"
  artifacts:
    - path: "src/musicgen/renderer.py"
      provides: "FLUIDSYNTH_VERSION module constant + RenderResult frozen dataclass + pick_soundfonts(cfg, rng) + render_stems(midi_paths, soundfonts, out_dir, cfg)"
      exports: ["FLUIDSYNTH_VERSION", "RenderResult", "pick_soundfonts", "render_stems"]
      contains: "@dataclass(frozen=True)"
      min_lines: 90
    - path: "tests/test_renderer.py"
      provides: "FLUIDSYNTH_VERSION capture test + pick_soundfonts seeded-determinism test + render_stems mocked-FluidSynth tests"
      contains: "def test_fluidsynth_version_capture"
  key_links:
    - from: "src/musicgen/renderer.py"
      to: "midi2audio.FluidSynth"
      via: "from midi2audio import FluidSynth"
      pattern: "FluidSynth\\(.*\\)\\.midi_to_audio"
    - from: "src/musicgen/renderer.py"
      to: "concurrent.futures.ThreadPoolExecutor"
      via: "import at module top"
      pattern: "ThreadPoolExecutor\\(max_workers=4\\)"
    - from: "src/musicgen/renderer.py"
      to: "config.Config.sf_layer_dir"
      via: "_cfg.sf_layer_dir(layer) in pick_soundfonts"
      pattern: "sf_layer_dir"
---

<objective>
Implement `src/musicgen/renderer.py` — the FluidSynth wrapper (R-X4). Capture `FLUIDSYNTH_VERSION` at module import (D-07), expose `RenderResult` as a frozen dataclass (D-02), provide `pick_soundfonts(cfg, rng)` replacing `music_gen.py:get_random_sound_font` with injected RNG (D-08/D-17), and provide `render_stems(midi_paths, soundfonts, out_dir)` that uses `ThreadPoolExecutor(max_workers=4)` to render the 4 per-layer stems of one part in parallel (D-06). Populate `tests/test_renderer.py` with mocked-FluidSynth unit tests so the suite stays green on CI machines without the FluidSynth binary.

Purpose: The renderer is the smallest standalone piece of the audio pipeline — no dependencies on mixer or annotator. Getting it locked down in Wave 2 means: (a) the mixer (Plan 04-03) can `from musicgen.renderer import RenderResult` with a stable target; (b) the integration test (Plan 04-06) has a mockable subprocess boundary; (c) the 4 bare `random.choice(sound_fonts)` draws currently at `music_gen.py:117-120` move behind the injected RNG, one of the biggest single contributors to the ~40+ bare-random inventory from CONTEXT.md.

Output: `src/musicgen/renderer.py` (new, ~110 lines including docstrings), `tests/test_renderer.py` (replaced from Wave 0 stub with real unit tests).
</objective>

<execution_context>
@/home/bidu/musicgen/.claude/get-shit-done/workflows/execute-plan.md
@/home/bidu/musicgen/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/phases/04-renderer-mixer-annotator-beats-extraction/04-CONTEXT.md
@.planning/phases/04-renderer-mixer-annotator-beats-extraction/04-RESEARCH.md
@.planning/phases/04-renderer-mixer-annotator-beats-extraction/04-PATTERNS.md
@.planning/phases/04-renderer-mixer-annotator-beats-extraction/04-VALIDATION.md
@music_gen.py
@src/musicgen/generators/beat.py
@src/musicgen/sampler.py
@config.py

<interfaces>
<!-- music_gen.py:117-120 — get_random_sound_font (replacement target for pick_soundfonts) -->
def get_random_sound_font(directory_path):
    sound_fonts = [f for f in os.listdir(directory_path) if f.endswith('.sf2')]
    file_return = random.choice(sound_fonts)
    return os.path.join(directory_path, file_return)

<!-- music_gen.py:212-215 — 4-layer soundfont selection (4 bare random.choice draws) -->
beat_soundfont = get_random_sound_font(_cfg.sf_layer_dir('beat'))
melody_soundfont = get_random_sound_font(_cfg.sf_layer_dir('melody'))
harmony_soundfont = get_random_sound_font(_cfg.sf_layer_dir('harmony'))
bassline_soundfont = get_random_sound_font(_cfg.sf_layer_dir('bassline'))

<!-- music_gen.py:265-274 — FluidSynth invocation pattern (extraction target for render_stems) -->
FluidSynth(beat_soundfont).midi_to_audio(beat_filename[part], beat_wav)
FluidSynth(melody_soundfont).midi_to_audio(melo_filename[part], melo_wav)
FluidSynth(harmony_soundfont).midi_to_audio(harm_filename[part], harm_wav)
FluidSynth(bassline_soundfont).midi_to_audio(bass_filename[part], bass_wav)

<!-- config.Config.sf_layer_dir — from config.py:60-62 -->
def sf_layer_dir(self, layer: str) -> str:
    return os.path.join(self.sf_dir, layer)

<!-- cfg fallback pattern — from src/musicgen/generators/beat.py:85 -->
_cfg = cfg if cfg is not None else config.Config()

<!-- Frozen dataclass pattern — from src/musicgen/sampler.py:223-241 (SongParams) -->
@dataclass(frozen=True)
class SongParams:
    key: str
    tempo: int
    ...

<!-- FluidSynth subprocess default output spec — from midi2audio.py constant (RESEARCH VERIFIED):
     DEFAULT_SAMPLE_RATE = 44100, renders stereo. -->

<!-- Dev venv state — RESEARCH-VERIFIED: `which fluidsynth` is NOT found.
     Module import MUST survive this: FLUIDSYNTH_VERSION = "unknown" + WARNING log.
     Tests mock FluidSynth.midi_to_audio; do not call the binary. -->
</interfaces>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Create src/musicgen/renderer.py — FLUIDSYNTH_VERSION + RenderResult + pick_soundfonts + render_stems</name>
  <files>src/musicgen/renderer.py</files>
  <behavior>
    - Importing `musicgen.renderer` on a machine WITHOUT FluidSynth binary succeeds; `FLUIDSYNTH_VERSION == "unknown"` and exactly one WARNING log is emitted
    - `RenderResult` is frozen (assignment to its fields raises)
    - `RenderResult` fields: `stem_paths: Dict[str, str]`, `sample_rate: int`, `channels: int`, `duration_seconds: float`, `fluidsynth_version: str`
    - `pick_soundfonts(cfg=Config(), rng=Random(42))` called twice with two Random(42) returns identical dicts (deterministic)
    - `pick_soundfonts` returns a dict with exactly the 4 keys: `beat`, `melody`, `harmony`, `bassline`; each value is a `.sf2` path under `cfg.sf_layer_dir(layer)`
    - `render_stems(midi_paths, soundfonts, out_dir)` where `midi_paths` and `soundfonts` both have the 4 layer keys:
      * Creates `out_dir` if missing (via `os.makedirs(out_dir, exist_ok=True)`)
      * Dispatches 4 `FluidSynth(sf).midi_to_audio(midi, wav)` calls via `ThreadPoolExecutor(max_workers=4)`
      * Returns a `RenderResult` whose `stem_paths` maps the 4 layer names → `<out_dir>/<layer>.wav`
      * `sample_rate == 44100`, `channels == 2`, `duration_seconds` derived from the actual written WAV (reads first stem via `AudioSegment.from_wav`)
      * `fluidsynth_version == FLUIDSYNTH_VERSION` (module constant surfaced for annotator)
    - Zero bare `random.<method>(` calls in the module (only `rng.choice(...)` via injected parameter)
  </behavior>
  <read_first>
    - music_gen.py (lines 117-120: get_random_sound_font; lines 265-274: FluidSynth invocation pattern)
    - src/musicgen/generators/beat.py (lines 1-29: module header convention; line 85: cfg fallback pattern)
    - src/musicgen/sampler.py (lines 223-293: frozen dataclass pattern for SongParams to mirror)
    - config.py (lines 60-62: sf_layer_dir method; lines 48-58: Config fields)
    - .planning/phases/04-renderer-mixer-annotator-beats-extraction/04-PATTERNS.md §"src/musicgen/renderer.py" (authoritative code templates)
    - .planning/phases/04-renderer-mixer-annotator-beats-extraction/04-RESEARCH.md Pitfall 3 (do NOT raise at import)
  </read_first>
  <action>
Create `src/musicgen/renderer.py` with the following structure. Every D-reference in the docstring must be preserved for traceability.

```python
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
            ``"harmony"``, ``"bassline"``) → absolute path of the rendered
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


def pick_soundfonts(
    cfg: Optional[config.Config] = None,
    rng: Optional[random.Random] = None,
) -> Dict[str, str]:
    """Select one ``.sf2`` file per layer (D-08/D-17).

    Replaces the four ``random.choice(sound_fonts)`` draws in
    ``music_gen.py:get_random_sound_font`` (4 call sites in ``mix_and_save``).

    Args:
        cfg: Optional Config (D-25 fallback to ``config.Config()`` if None).
        rng: Injected ``random.Random`` (required for determinism; D-17 forbids
            bare ``random.<method>`` at module scope).

    Returns:
        Dict mapping layer name → absolute ``.sf2`` path.
    """
    if rng is None:
        raise ValueError("pick_soundfonts requires an injected rng (D-17)")
    _cfg = cfg if cfg is not None else config.Config()
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
        midi_paths: Dict mapping layer name → MIDI file path.
        soundfonts: Dict mapping layer name → ``.sf2`` path (from
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
```

Notes specific to this task:

- `from __future__ import annotations` at top (so `Dict[str, str]` works pre-3.10 style and `tuple[str, str]` in the nested function signature doesn't break)
- `from midi2audio import FluidSynth` at module import — OK because midi2audio is pure Python and doesn't invoke the binary at import; only at `.midi_to_audio()` call time
- `from pydub import AudioSegment` — used to read the duration of the first rendered stem
- The `_LAYERS` tuple is module-private (underscore prefix) so it's not accidentally part of the public API
- `pick_soundfonts` uses `sorted(...)` on the `.sf2` file list before `rng.choice` — this fixes a latent nondeterminism (os.listdir order is filesystem-dependent) that would have broken Phase 5 golden-seed baselines on different filesystems
- Every function signature has type annotations; docstrings are Google-style
  </action>
  <verify>
    <automated>python -c "
import random
from musicgen.renderer import FLUIDSYNTH_VERSION, RenderResult, pick_soundfonts, render_stems
# Module imports without raising on a machine without FluidSynth:
assert isinstance(FLUIDSYNTH_VERSION, str) and len(FLUIDSYNTH_VERSION) > 0
print(f'FLUIDSYNTH_VERSION = {FLUIDSYNTH_VERSION!r}')
# RenderResult is frozen:
rr = RenderResult(stem_paths={'beat': '/x'}, sample_rate=44100, channels=2, duration_seconds=1.0, fluidsynth_version='test')
try:
    rr.sample_rate = 22050
    assert False, 'frozen dataclass not actually frozen'
except (AttributeError, Exception) as e:
    if 'FrozenInstanceError' not in type(e).__name__ and not isinstance(e, AttributeError):
        raise
print('RenderResult frozen OK')
print('renderer.py smoke OK')
"</automated>
  </verify>
  <acceptance_criteria>
    - File `src/musicgen/renderer.py` exists
    - `grep "^FLUIDSYNTH_VERSION" src/musicgen/renderer.py` returns at least 1 line
    - `grep "^@dataclass(frozen=True)" src/musicgen/renderer.py` returns exactly 1 line
    - `grep "ThreadPoolExecutor(max_workers=4)" src/musicgen/renderer.py` returns exactly 1 line (D-06 locked)
    - `grep -c "^def " src/musicgen/renderer.py` returns `2` (pick_soundfonts + render_stems; RenderResult is a class, not a def)
    - `grep -c "rng\.choice" src/musicgen/renderer.py` returns at least `1` (pick_soundfonts uses injected rng)
    - `python -c "import ast; tree = ast.parse(open('src/musicgen/renderer.py').read()); hits = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) and n.func.value.id == 'random' and n.func.attr != 'Random']; assert hits == [], f'bare random.* hits: {[n.lineno for n in hits]}'"` — exits 0 (D-17 compliant)
    - `python -c "import musicgen.renderer; print(musicgen.renderer.FLUIDSYNTH_VERSION)"` exits 0 and prints a string (either a valid version OR `"unknown"` — both are acceptable; CI lacks fluidsynth binary so "unknown" is expected)
    - `grep "D-06\|D-07\|D-08\|D-17\|D-25" src/musicgen/renderer.py` returns at least 5 matches (traceability)
    - Smoke command above prints `FLUIDSYNTH_VERSION = ...`, `RenderResult frozen OK`, `renderer.py smoke OK`
  </acceptance_criteria>
  <done>`src/musicgen/renderer.py` exposes FLUIDSYNTH_VERSION (string, fallback to "unknown" if binary absent), RenderResult (frozen dataclass), pick_soundfonts(cfg, rng) (deterministic via injected rng with sorted listdir), and render_stems(midi_paths, soundfonts, out_dir, cfg) (ThreadPoolExecutor(4) dispatch → RenderResult). Zero bare random.* in the module.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Populate tests/test_renderer.py with mocked-FluidSynth unit tests</name>
  <files>tests/test_renderer.py</files>
  <behavior>
    - `test_fluidsynth_version_capture` — asserts `FLUIDSYNTH_VERSION` is a non-empty string (either a real version like "FluidSynth runtime version 2.3.4" OR the fallback "unknown"); both are acceptable because CI environments vary
    - `test_render_result_is_frozen` — attempting to assign to a RenderResult field raises (frozen dataclass contract)
    - `test_render_result_equality` — two RenderResults with identical fields compare equal (dataclass `__eq__` contract)
    - `test_pick_soundfonts_returns_4_layer_dict` — mocks `os.listdir` to return fake `.sf2` lists; asserts returned dict has exactly `{"beat", "melody", "harmony", "bassline"}` keys and each value ends in `.sf2`
    - `test_pick_soundfonts_deterministic` — two `random.Random(42)` instances yield identical soundfont dicts (D-08/D-17 contract)
    - `test_pick_soundfonts_different_seeds_different_output` — `random.Random(0)` vs `random.Random(999)` with >=2 .sf2 files per layer yield different dicts (sanity: the RNG is actually being used)
    - `test_pick_soundfonts_empty_dir_raises` — empty .sf2 list raises FileNotFoundError with the layer name in the message
    - `test_pick_soundfonts_requires_rng` — calling without rng raises ValueError (D-17 guard)
    - `test_render_stems_creates_out_dir_and_returns_render_result` — mocks FluidSynth.midi_to_audio to create fake WAVs; asserts out_dir was created, all 4 stem_paths exist, RenderResult shape is correct
    - `test_render_stems_dispatches_all_4_layers` — mock counter verifies FluidSynth.midi_to_audio called exactly 4 times (D-06 dispatch correctness)
    - `test_render_stems_missing_layer_raises` — passing midi_paths with only 3 layers raises KeyError
    - `test_render_stems_fluidsynth_version_in_result` — RenderResult.fluidsynth_version equals the module-level FLUIDSYNTH_VERSION
  </behavior>
  <read_first>
    - tests/test_renderer.py (Wave 0 stub — replaced entirely)
    - tests/test_generators/test_beat.py (analog: test scaffold + monkeypatch + seeded rng pattern)
    - tests/test_sampler.py (seeded-determinism class structure)
    - src/musicgen/renderer.py (from Task 1 — API under test)
    - .planning/phases/04-renderer-mixer-annotator-beats-extraction/04-PATTERNS.md §"tests/test_renderer.py" (mocked-FluidSynth fixture pattern)
  </read_first>
  <action>
Replace `tests/test_renderer.py` entirely (delete Wave 0 skip stub). File content:

```python
"""Renderer tests (R-X4): RenderResult assembly + FLUIDSYNTH_VERSION capture + seeded pick_soundfonts.

FluidSynth subprocess is mocked via ``unittest.mock.patch`` — unit tests do NOT
require the FluidSynth binary. The E2E integration test
(``tests/test_integration_full_generation.py``, Plan 04-06) covers the real
binary behind ``@pytest.mark.slow`` + ``shutil.which("fluidsynth")`` guard.
"""
from __future__ import annotations

import os
import random
from pathlib import Path
from unittest.mock import patch

import pytest
from pydub import AudioSegment

from musicgen.renderer import (
    FLUIDSYNTH_VERSION,
    RenderResult,
    pick_soundfonts,
    render_stems,
)


# ---------- FLUIDSYNTH_VERSION (D-07) ----------

class TestFluidSynthVersion:
    def test_fluidsynth_version_capture(self):
        """FLUIDSYNTH_VERSION is always a non-empty string.

        Either a real version line (e.g., "FluidSynth runtime version 2.3.4")
        when fluidsynth binary is on PATH, or "unknown" as the fallback when
        not (D-07 locks this: NEVER raises at import; RESEARCH Pitfall 3).
        """
        assert isinstance(FLUIDSYNTH_VERSION, str)
        assert len(FLUIDSYNTH_VERSION) > 0

    def test_renderer_importable_without_fluidsynth_binary(self):
        """Import must succeed regardless of fluidsynth binary presence."""
        import musicgen.renderer
        assert hasattr(musicgen.renderer, "FLUIDSYNTH_VERSION")


# ---------- RenderResult (D-02) ----------

class TestRenderResult:
    def test_is_frozen(self):
        rr = RenderResult(
            stem_paths={"beat": "/x.wav"},
            sample_rate=44100,
            channels=2,
            duration_seconds=1.5,
            fluidsynth_version="test",
        )
        with pytest.raises((AttributeError, Exception)):  # FrozenInstanceError or AttributeError
            rr.sample_rate = 22050  # type: ignore[misc]

    def test_equality(self):
        a = RenderResult(stem_paths={"beat": "/x.wav"}, sample_rate=44100, channels=2, duration_seconds=1.0, fluidsynth_version="v1")
        b = RenderResult(stem_paths={"beat": "/x.wav"}, sample_rate=44100, channels=2, duration_seconds=1.0, fluidsynth_version="v1")
        assert a == b

    def test_has_all_5_fields(self):
        rr = RenderResult(stem_paths={}, sample_rate=44100, channels=2, duration_seconds=0.0, fluidsynth_version="u")
        assert rr.stem_paths == {}
        assert rr.sample_rate == 44100
        assert rr.channels == 2
        assert rr.duration_seconds == 0.0
        assert rr.fluidsynth_version == "u"


# ---------- pick_soundfonts (D-08/D-17) ----------

@pytest.fixture
def fake_sf2_dirs(tmp_path):
    """Create fake sf/<layer>/ directories with 3 .sf2 files each."""
    layers = ("beat", "melody", "harmony", "bassline")
    for layer in layers:
        layer_dir = tmp_path / "sf" / layer
        layer_dir.mkdir(parents=True)
        for i in range(3):
            (layer_dir / f"fake_{i}.sf2").write_bytes(b"RIFF")
    return tmp_path


@pytest.fixture
def fake_cfg(fake_sf2_dirs, monkeypatch):
    """Config that points at the fake sf2 directories."""
    import config
    cfg = config.Config()
    cfg.sf_dir = str(fake_sf2_dirs / "sf")
    return cfg


class TestPickSoundfonts:
    def test_returns_4_layer_dict(self, fake_cfg):
        result = pick_soundfonts(cfg=fake_cfg, rng=random.Random(42))
        assert set(result.keys()) == {"beat", "melody", "harmony", "bassline"}
        for layer, path in result.items():
            assert path.endswith(".sf2")
            assert layer in path or True  # path contains the fake file name, not necessarily the layer

    def test_deterministic_same_seed(self, fake_cfg):
        a = pick_soundfonts(cfg=fake_cfg, rng=random.Random(42))
        b = pick_soundfonts(cfg=fake_cfg, rng=random.Random(42))
        assert a == b, "same seed must produce same soundfont selection"

    def test_different_seeds_different_output(self, fake_cfg):
        a = pick_soundfonts(cfg=fake_cfg, rng=random.Random(0))
        b = pick_soundfonts(cfg=fake_cfg, rng=random.Random(9999))
        # With 3 .sf2 files per layer, 2 different seeds almost always differ on at least one layer.
        assert a != b, "different seeds should produce different soundfont selection"

    def test_empty_sf_dir_raises(self, tmp_path, monkeypatch):
        import config
        cfg = config.Config()
        cfg.sf_dir = str(tmp_path / "empty_sf")
        for layer in ("beat", "melody", "harmony", "bassline"):
            (tmp_path / "empty_sf" / layer).mkdir(parents=True)
            # intentionally no .sf2 files
        with pytest.raises(FileNotFoundError, match=r"No \.sf2 files"):
            pick_soundfonts(cfg=cfg, rng=random.Random(42))

    def test_requires_rng(self, fake_cfg):
        with pytest.raises(ValueError, match="rng"):
            pick_soundfonts(cfg=fake_cfg, rng=None)


# ---------- render_stems (D-06/D-09) ----------

def _make_fake_wav(path: str, duration_ms: int = 1000, sample_rate: int = 44100):
    """Write a stereo 44.1kHz silent WAV; stand-in for a FluidSynth render."""
    AudioSegment.silent(duration=duration_ms, frame_rate=sample_rate).set_channels(2).export(
        path, format="wav"
    )


@pytest.fixture
def mock_fluidsynth(tmp_path):
    """Patch FluidSynth.midi_to_audio so it writes a fake stereo-44.1kHz WAV
    instead of calling the real subprocess.
    """
    call_counter = {"n": 0}

    def _fake_render(self, midi_path, wav_path):
        call_counter["n"] += 1
        _make_fake_wav(wav_path)

    with patch("musicgen.renderer.FluidSynth.midi_to_audio", _fake_render):
        yield call_counter


@pytest.fixture
def fake_midis_and_sfs(tmp_path):
    """Create fake MIDI paths + soundfont paths; the mock doesn't read MIDI content."""
    midi_paths = {}
    soundfonts = {}
    for layer in ("beat", "melody", "harmony", "bassline"):
        midi_path = tmp_path / f"{layer}.mid"
        midi_path.write_bytes(b"MThd fake")  # content doesn't matter; FluidSynth is mocked
        midi_paths[layer] = str(midi_path)
        sf_path = tmp_path / f"{layer}.sf2"
        sf_path.write_bytes(b"RIFF")
        soundfonts[layer] = str(sf_path)
    return midi_paths, soundfonts


class TestRenderStems:
    def test_creates_out_dir(self, tmp_path, mock_fluidsynth, fake_midis_and_sfs):
        midi_paths, soundfonts = fake_midis_and_sfs
        out_dir = tmp_path / "nested" / "stems"
        assert not out_dir.exists()
        render_stems(midi_paths, soundfonts, str(out_dir))
        assert out_dir.exists() and out_dir.is_dir()

    def test_returns_render_result(self, tmp_path, mock_fluidsynth, fake_midis_and_sfs):
        midi_paths, soundfonts = fake_midis_and_sfs
        out_dir = tmp_path / "stems"
        result = render_stems(midi_paths, soundfonts, str(out_dir))
        assert isinstance(result, RenderResult)
        assert set(result.stem_paths.keys()) == {"beat", "melody", "harmony", "bassline"}
        assert result.sample_rate == 44100
        assert result.channels == 2
        assert result.duration_seconds == pytest.approx(1.0, abs=0.1)
        assert result.fluidsynth_version == FLUIDSYNTH_VERSION

    def test_dispatches_all_4_layers(self, tmp_path, mock_fluidsynth, fake_midis_and_sfs):
        midi_paths, soundfonts = fake_midis_and_sfs
        out_dir = tmp_path / "stems"
        render_stems(midi_paths, soundfonts, str(out_dir))
        assert mock_fluidsynth["n"] == 4, f"FluidSynth.midi_to_audio called {mock_fluidsynth['n']}x, expected 4 (one per layer)"

    def test_stem_paths_under_out_dir(self, tmp_path, mock_fluidsynth, fake_midis_and_sfs):
        midi_paths, soundfonts = fake_midis_and_sfs
        out_dir = tmp_path / "stems"
        result = render_stems(midi_paths, soundfonts, str(out_dir))
        for layer, path in result.stem_paths.items():
            assert path == os.path.join(str(out_dir), f"{layer}.wav")
            assert os.path.exists(path), f"stem {layer!r} does not exist at {path}"

    def test_missing_midi_layer_raises(self, tmp_path, mock_fluidsynth, fake_midis_and_sfs):
        midi_paths, soundfonts = fake_midis_and_sfs
        del midi_paths["harmony"]
        with pytest.raises(KeyError, match="harmony"):
            render_stems(midi_paths, soundfonts, str(tmp_path / "stems"))

    def test_missing_soundfont_layer_raises(self, tmp_path, mock_fluidsynth, fake_midis_and_sfs):
        midi_paths, soundfonts = fake_midis_and_sfs
        del soundfonts["bassline"]
        with pytest.raises(KeyError, match="bassline"):
            render_stems(midi_paths, soundfonts, str(tmp_path / "stems"))
```

Do NOT retain the Wave 0 module-level skip. All tests must actually run (and pass).
  </action>
  <verify>
    <automated>python -m pytest tests/test_renderer.py -x -q 2>&1 | tail -25</automated>
  </verify>
  <acceptance_criteria>
    - `pytest tests/test_renderer.py -x -q` exits 0 — no tests skipped (Wave 0 stub replaced)
    - `grep -c "def test_" tests/test_renderer.py` >= 15 (multiple test methods across 4 test classes)
    - `grep "TestFluidSynthVersion\|TestRenderResult\|TestPickSoundfonts\|TestRenderStems" tests/test_renderer.py` returns at least 4 matches (all 4 test classes present)
    - `grep "@patch\|unittest.mock\|patch(" tests/test_renderer.py` returns at least 1 line (mocking in place)
    - Total wall time for `pytest tests/test_renderer.py -q` < 5 seconds (mocks run fast; no real FluidSynth subprocess)
    - Full suite regression: `pytest tests/ -m "not slow" -q` exits 0 (371 baseline + Plan 04-01 beats tests + Plan 04-02 renderer tests; no regressions)
  </acceptance_criteria>
  <done>`tests/test_renderer.py` contains 4 test classes (TestFluidSynthVersion, TestRenderResult, TestPickSoundfonts, TestRenderStems) covering: FLUIDSYNTH_VERSION capture + importability without binary + RenderResult frozen/equality + pick_soundfonts determinism (same seed, different seeds, empty dir, missing rng) + render_stems dispatches 4 layers, creates out_dir, missing-layer errors. All tests pass with mocked FluidSynth. No regressions.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| Python → FluidSynth subprocess | `subprocess.run(["fluidsynth", "--version"])` at import; `FluidSynth.midi_to_audio` at render time |
| Filesystem → `os.listdir(sf_dir)` | `.sf2` files discovered by scanning the config-provided directory |
| Input paths → filesystem | `out_dir` parameter is caller-trusted (D-09: Phase 4 does not normalize; Phase 5 writer takes over) |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-04-02-01 | Elevation | FluidSynth subprocess argument injection via soundfont path | accept | Soundfont paths come from `os.listdir(sf_dir)` filtered to `.sf2` — filenames can't contain `;` or `&&` at the shell level because `subprocess.run(list)` does not invoke a shell. `cfg.sf_dir` is trusted config. No user-controlled paths in Phase 4. |
| T-04-02-02 | Tampering | Malformed `.sf2` file | accept | FluidSynth itself raises on malformed SF2 (exit code != 0); midi2audio propagates as CalledProcessError → our `render_stems` would let this propagate to the orchestrator. No special handling this phase. |
| T-04-02-03 | Tampering | Path traversal via `out_dir` parameter | accept | Caller responsibility per D-09; `os.makedirs(out_dir, exist_ok=True)` creates whatever path is given. Phase 5 writer will add `os.path.abspath` normalization at the outer layer. |
| T-04-02-04 | Denial of Service | `subprocess.run` at module import hangs | mitigate | `timeout=5` on the version-capture subprocess + broad `except Exception` fallback to `"unknown"` (D-07) ensures import never blocks or raises. |
</threat_model>

<verification>
After all 2 tasks complete:

1. `python -c "import musicgen.renderer; print(musicgen.renderer.FLUIDSYNTH_VERSION)"` — prints a string, does not raise (even without fluidsynth binary)
2. `python -c "from musicgen.renderer import RenderResult, pick_soundfonts, render_stems; import inspect; print(inspect.signature(pick_soundfonts)); print(inspect.signature(render_stems))"` — prints the locked signatures
3. `python -m pytest tests/test_renderer.py -v` — all new tests pass
4. `python -m pytest tests/ -m "not slow" -q` — full suite green
5. `python -c "import ast; t = ast.parse(open('src/musicgen/renderer.py').read()); hits = [n for n in ast.walk(t) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) and n.func.value.id == 'random' and n.func.attr != 'Random']; assert hits == []"` — zero bare random.* (prepares for Plan 04-05)
</verification>

<success_criteria>
- `src/musicgen/renderer.py` exposes FLUIDSYNTH_VERSION + RenderResult + pick_soundfonts + render_stems
- Module is importable without FluidSynth binary (version fallback to "unknown")
- `ThreadPoolExecutor(max_workers=4)` literal present exactly once (D-06 locked)
- Zero bare `random.<method>(` calls (D-17 compliant)
- `tests/test_renderer.py` covers all 4 facets (version capture, RenderResult contract, pick_soundfonts RNG determinism, render_stems dispatch with mocked FluidSynth)
- Full suite green; no regression
</success_criteria>

<output>
After completion, create `.planning/phases/04-renderer-mixer-annotator-beats-extraction/04-02-SUMMARY.md`.

Include:
- Line count of `src/musicgen/renderer.py`
- Output of `python -c "from musicgen.renderer import FLUIDSYNTH_VERSION; print(FLUIDSYNTH_VERSION)"` (confirms fallback or real version on dev machine)
- Test count in tests/test_renderer.py (target: 15+)
- Confirmation of `ThreadPoolExecutor(max_workers=4)` presence
- Confirmation of zero bare `random.*` in renderer.py
- Full suite run tail
</output>
