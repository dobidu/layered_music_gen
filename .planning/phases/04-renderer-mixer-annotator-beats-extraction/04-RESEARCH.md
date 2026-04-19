# Phase 4: Renderer + Mixer + Annotator + Beats Extraction — Research

**Researched:** 2026-04-19
**Domain:** Audio pipeline extraction — FluidSynth rendering, pedalboard FX, pydub mixing, MIDI-tick beat annotation
**Confidence:** HIGH (all critical findings verified against live codebase, installed packages, and source inspection)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01:** Four new modules under `src/musicgen/`: `renderer.py`, `mixer.py`, `annotator.py`, `beats.py`. Exact surface APIs locked.
- **D-02:** `RenderResult` (frozen dataclass) in `renderer.py`; `MixResult` (frozen dataclass) in `mixer.py`. `@dataclass(frozen=True)`.
- **D-03:** `beat_anotator.py` deleted — no shim, no re-export.
- **D-04:** `musicality_score.py` stays at repo root this phase; annotator receives score dict as parameter.
- **D-05:** Keep `midi2audio`. Do NOT switch to `pyfluidsynth`.
- **D-06:** `ThreadPoolExecutor(max_workers=4)` inside `render_stems` for four per-stem renders per part. Parts remain serial.
- **D-07:** `FLUIDSYNTH_VERSION` captured via `subprocess.run(["fluidsynth","--version"], capture_output=True, text=True, timeout=5).stdout.splitlines()[0]` at module import with `"unknown"` fallback. Never raises at import.
- **D-08:** `pick_soundfonts(cfg, rng)` in `renderer.py` replaces bare `random.choice(sound_fonts)`. Renderer takes soundfonts dict as parameter.
- **D-09:** Renderer writes stems to `<working_dir>/<name>-<part>/stems/<layer>.wav`. Phase 5 owns index-based layout.
- **D-10:** `generate_pedalboard`, `create_effect`, `apply_fx_to_layer`, `pedalboard_info_json` all move to `mixer.py`.
- **D-11:** Apply FX to ALL four layers regardless of layer mask. The `music_gen.py:276` TODO is explicitly OUT OF SCOPE.
- **D-12:** Silent-stem fallback via `AudioSegment.silent(duration=ms)` matching rendered-stem duration + sample rate.
- **D-13:** `compute_layer_mask(song_unique_parts, inst_proba, rng)` in `mixer.py` with injected `rng`.
- **D-14:** `annotate(...)` is a pure function, zero I/O.
- **D-15/D-16:** Phase-4-filled fields populated; Phase-5 TBD fields present as `None` (not missing, not `"TBD"` string).
- **D-17:** Zero bare `random.choice/random/randint/choices/uniform` under `src/musicgen/**/*.py` after this phase.
- **D-18:** One module-level `_rng = random.Random()` threaded through `music_gen.py` orchestrator (Phase 5 adds hierarchy).
- **D-19:** `beats.py` uses `mido.MidiFile` + `mido.tick2second`. Replaces `beat_anotator.py` entirely.
- **D-20:** Downbeat derivation uses `beat_times[::numerator]` stride-slice with registry fallback for 6/8 grouping.
- **D-21:** `beat_duration` moves to `beats.py`; `generators/beat.py` keeps re-export alias. `calculate_swing_offset` stays in `generators/beat.py`.
- **D-22:** Beats called from orchestrator post-mix (serial), using `mix_result.transitions` for `start_offset_seconds`.
- **D-23:** `mix_and_save` deleted entirely. Responsibilities distributed.
- **D-24:** `music_gen.py` collapses to `< 200` lines of orchestration; `if __name__ == "__main__"` guard kept.
- **D-25:** Every new function uses `cfg: config.Config = None` with runtime fallback.
- **D-26 to D-32:** Test files: `test_annotator.py`, `test_beats.py`, `test_renderer.py`, `test_mixer.py`, `test_integration_full_generation.py`, `test_no_bare_random_in_package.py`. Swing cases at 0.5/0.66/0.75. E2E test is `@pytest.mark.slow` and guarded on `fluidsynth` binary.
- **D-33:** Phase 4 runs serially after Phase 3.

### Claude's Discretion

- Exact field names inside `RenderResult` / `MixResult` dataclasses (`duration_seconds` vs `duration_s`).
- Whether silent-stem WAVs are stereo or mono.
- Module docstring style (match `generators/beat.py` Google-style-with-D-reference).
- Whether `FLUIDSYNTH_VERSION` is kept as the full first line or parsed to `"2.3.4"`.
- Whether `pick_soundfonts` uses `os.listdir` + filter or `Path.glob`.
- Whether `beat_duration` in `beats.py` uses the exact same body as today's `generators/beat.py:beat_duration`.
- Whether `compute_layer_mask` takes `inst_proba` as a pre-loaded dict or loads it inside.

### Deferred Ideas (OUT OF SCOPE)

- Stem-sum-to-mix assertion (R-P2) — Phase 5.
- FluidSynth pre-roll calibration (R-P9) — Phase 6.
- Per-sample output layout with zero-padded index (R-P1) — Phase 5.
- `manifest.jsonl` append (R-P5) — Phase 5.
- `musicgen_version`, `seed`, `split` fields in annotation — Phase 5.
- `analysis_failed` true-path (R-P6) — Phase 5.
- FX-only-to-used-layers optimization (D-11 explicitly deferred).
- Moving `musicality_score.py` into package — Phase 5.
- Batch/process-pool outer parallelism (R-P10) — Phase 6.
- Downbeat grouping fallback via `TimeSignatureRegistry.primary_beat_group` — only if test fixture fails.
- `pyfluidsynth` migration — explicit no-go.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| R-X4 | Renderer: FluidSynth wrapper, ThreadPoolExecutor(4), FLUIDSYNTH_VERSION at module load | D-05..D-09 implementation verified; FluidSynth version capture pattern confirmed; mido required as new explicit dep |
| R-X5 | Mixer: pedalboard FX, pydub overlay, gain/pan fix (R-S4), silent-stem fallback | D-10..D-13 verified against music_gen.py:130-345 extraction targets; pedalboard determinism confirmed; silent-stem needs stereo match |
| R-X6 | Annotator: pure function returning R-P4 schema subset with Phase-5 TBD fields as None | R-P4 field inventory documented in full; D-14..D-16 shape locked |
| R-X7 | Beats: MIDI-tick beat + downbeat timestamps, swing-aware; replaces beat_anotator.py | D-19..D-21 implementation confirmed; D-20 stride assumption amended — see Pitfall 2 |
| R-X8 | Integration test: @pytest.mark.slow E2E test guarded on fluidsynth binary | D-30 strategy confirmed; FluidSynth not on PATH in CI environment; marker infra requires pyproject.toml update |
</phase_requirements>

---

## Summary

Phase 4 extracts the audio side of the pipeline from `music_gen.py` into four focused modules. The extraction targets are well-bounded: `mix_and_save` (lines 193–345) is the primary source, supplemented by `generate_pedalboard`/`create_effect`/`apply_fx_to_layer`/`pedalboard_info_json` (130–190) for the mixer, and `beat_anotator.py` as the deletion target replaced by `beats.py`.

**Three dependency findings require action before planning tasks can be written:**

1. `mido` is **not installed** in the project's venv and is **not listed** in `pyproject.toml`. It must be added as an explicit runtime dependency (`mido>=1.3.3`). The prior assumption that it was a transitive dependency of `midi2audio` is incorrect — `midi2audio` declares zero Python-level requirements.

2. `AudioSegment.silent()` defaults to **mono, 11025 Hz** — this mismatches FluidSynth-rendered stems which are **stereo, 44100 Hz**. Silent-stem fallback (D-12) must explicitly set `frame_rate=44100` and call `.set_channels(2)` (or use `RenderResult.sample_rate` / `RenderResult.channels` for the stereo match).

3. The D-20 downbeat stride assumption (`beat_times[::numerator]`) **fails** for 12/8 (patterns contain zeros → `extract_beat_times` returns sparse beat_times shorter than `numerator × measures`) and fails for any time signature when the first beat slot of a measure is 0 (e.g., 4/4 `intro: 0, 42, 38, 0` — verified in the pattern files). The correct approach is a **time-grid computation**, not a stride slice. See Pitfall 2 for the recommended fix.

**Primary recommendation:** Build in dependency order — `beats.py` first (no audio deps, pure MIDI logic), then `renderer.py` (FluidSynth wrapper, no mixer deps), then `mixer.py` (consumes RenderResult), then `annotator.py` (consumes all four). Modules can be tested independently; only the integration test requires all four to be complete.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| FluidSynth invocation + parallelism | `renderer.py` | `music_gen.py` orchestrator (calls renderer) | Renderer owns the subprocess boundary; orchestrator owns sequencing across parts |
| Soundfont selection | `renderer.py` (`pick_soundfonts`) | `config.Config` (provides sf_layer_dir) | Soundfont picking is render-time; cfg provides paths |
| FX chain construction (pedalboard) | `mixer.py` (`build_fx_boards`) | `config.Config` (provides fx_files) | FX is mix-time probability; cfg provides JSON paths |
| Layer-inclusion masking | `mixer.py` (`compute_layer_mask`) | `config.Config` (provides inst_proba) | Mix-time decision, not song-structure sampling |
| Gain/pan application (R-S4 fix) | `mixer.py` | — | Mix-level operation; isolated to mixer |
| Silent-stem fallback | `mixer.py` | — | Mixer owns all stem → output WAV transformations |
| Part concatenation | `mixer.py` (`concat_parts`) | — | Audio domain; mixer is the audio owner |
| MIDI-tick beat extraction | `beats.py` | — | Pure MIDI read; no audio dependency |
| Downbeat time derivation | `beats.py` | `TimeSignatureRegistry` (provides beats_per_measure) | Registry knows compound meter grouping |
| Annotation dict assembly | `annotator.py` | — | Pure function; receives all other stage outputs |
| Musicality scoring call | `music_gen.py` orchestrator | — | D-04: stays at repo root this phase |
| JSON dump (sample.json) | `music_gen.py` orchestrator | — | Phase 5 writer takes over this responsibility |
| FluidSynth version capture | `renderer.py` module scope | — | Captured once at import; surfaced in annotator |

---

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `midi2audio` | 0.1.1 [VERIFIED: pip show] | FluidSynth Python wrapper | Already installed; D-05 locks this |
| `mido` | 1.3.3 [VERIFIED: pip index] | MIDI file reading + `tick2second` conversion | Required for `beats.py`; beat_anotator.py already uses this API |
| `pedalboard` | 0.9.22 [VERIFIED: pip show] | Audio effects chain construction and application | Already installed; D-10 lifts verbatim |
| `pydub` | 0.25.1 [VERIFIED: pip show] | Audio overlay, gain, pan, concatenation, silent segments | Already installed; D-12/D-13 use AudioSegment |
| `concurrent.futures` | stdlib | `ThreadPoolExecutor(max_workers=4)` for parallel stem renders | D-06 locks this; FluidSynth is subprocess, threads suffice |
| `subprocess` | stdlib | `fluidsynth --version` capture at module load | D-07 pattern |
| `midiutil` | 1.2.1 [VERIFIED: pip list] | MIDI synthesis in test fixtures (beats tests) | Already installed; used in generators |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `ast` | stdlib | Package-wide bare-random static guard | D-31 extended AST scan |
| `glob` | stdlib | File discovery for AST guard parametrize | D-31 test |
| `math` | stdlib | `_lin_to_db` helper (20 * log10) | R-S4 gain fix in mixer |
| `json` | stdlib | FX param file loading, annotator output | Mixer + orchestrator |

### New Dependency Required

`mido` is **not installed** and **not in pyproject.toml**. `midi2audio` declares zero Python-level requirements and does not bring in `mido` as a transitive dep. [VERIFIED: `pip show midi2audio` shows `Requires:` is empty; `pip list` confirms mido absent from venv]

**Action required:** Add `"mido>=1.3.3"` to `[project] dependencies` in `pyproject.toml` and run `pip install -e '.[dev]'` before any beats.py work.

### Installation

```bash
# Add mido to pyproject.toml dependencies, then:
pip install -e '.[dev]'

# Verify:
python -c "import mido; print(mido.__version__)"
```

**Version verification:** All installed package versions confirmed against live venv on 2026-04-19. `mido>=1.3.3` chosen as minimum because 1.3.x is the current stable series and older releases had API changes to `tick2second`. [VERIFIED: pip index versions mido]

---

## Architecture Patterns

### System Architecture Diagram

```
music_gen.py orchestrator (create_song, ~180 lines after collapse)
    │
    ├─► [EXISTING] generate_song_parts → generators (chord/melody/bass/beat)
    │       │
    │       └─► .mid files on disk (per part, per layer)
    │
    ├─► renderer.pick_soundfonts(cfg, rng) → Dict[layer, sf_path]
    │
    ├─► Per part (serial):
    │       │
    │       ├─► renderer.render_stems(midi_paths, soundfonts, out_dir)
    │       │       │  ThreadPoolExecutor(4): beat/melody/harmony/bassline in parallel
    │       │       └─► RenderResult (frozen dataclass)
    │       │               ├─ stem_paths: Dict[layer, wav_path]
    │       │               ├─ sample_rate: int (44100)
    │       │               ├─ duration_seconds: float
    │       │               └─ fluidsynth_version: str (from FLUIDSYNTH_VERSION)
    │       │
    │       ├─► mixer.build_fx_boards(cfg, rng) → Dict[layer, Pedalboard]
    │       ├─► mixer.compute_layer_mask(song_unique_parts, inst_proba, rng)
    │       │
    │       ├─► mixer.mix_part(render_result, levels, fx_boards, layer_mask, part, out_dir)
    │       │       │  apply FX to ALL 4 layers (D-11) → gain/pan → overlay masked layers
    │       │       │  write silent stem for masked-out layers (D-12)
    │       │       └─► MixResult (frozen dataclass)
    │       │               ├─ mix_path: str
    │       │               ├─ stem_paths: Dict[layer, wav_path]  (post-FX)
    │       │               ├─ part_layers: Dict[layer, bool]  (layer mask)
    │       │               ├─ soundfonts: Dict[layer, str]
    │       │               └─ transitions: List (updated per-part timing)
    │       │
    │       └─► beats.extract_beat_times(midi_path, tempo, start_offset_seconds)
    │               beats.extract_downbeat_times(beat_times, time_signature, measures, start_offset)
    │
    ├─► mixer.concat_parts(part_mixes, out_path) → str (final mix WAV path)
    │
    ├─► musicality_score.get_musicality_score(wav_name) [stays at root, D-04]
    │
    └─► annotator.annotate(song_params, render_results, mix_result, beat_times,
                            downbeat_times, musicality, fluidsynth_version=...) → dict
```

### Recommended Project Structure

```
src/musicgen/
├── renderer.py        # NEW: FluidSynth wrapper + ThreadPoolExecutor + FLUIDSYNTH_VERSION
├── mixer.py           # NEW: FX + overlay + concat + layer mask + silent stems
├── annotator.py       # NEW: pure function → annotation dict
├── beats.py           # NEW: MIDI-tick beat/downbeat extraction
├── generators/
│   └── beat.py        # MODIFIED: re-export beat_duration from beats.py (D-21)
├── sampler.py         # UNCHANGED from Phase 3
├── duration_validator.py  # UNCHANGED
└── ...

tests/
├── test_renderer.py           # NEW (D-28)
├── test_mixer.py              # NEW (D-29)
├── test_annotator.py          # NEW (D-26)
├── test_beats.py              # NEW (D-27)
├── test_integration_full_generation.py  # NEW (D-30, @pytest.mark.slow)
└── test_no_bare_random_in_package.py   # NEW (D-31)

music_gen.py           # MODIFIED: collapsed to ~180 lines of orchestration (D-23/D-24)
beat_anotator.py       # DELETED (D-03)
pyproject.toml         # MODIFIED: add mido>=1.3.3; add markers section
```

### Pattern 1: Frozen Dataclass Output Contract

Match `SongParams` shape from Phase 3:

```python
# Source: src/musicgen/sampler.py (established pattern)
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass(frozen=True)
class RenderResult:
    """Per-part stem render outputs (R-X4).

    D-02: frozen dataclass; D-07: FLUIDSYNTH_VERSION captured at module load.
    """
    stem_paths: Dict[str, str]        # layer → wav path
    sample_rate: int                   # always 44100 (FluidSynth default)
    duration_seconds: float            # length of each stem (all equal)
    fluidsynth_version: str            # from module-level FLUIDSYNTH_VERSION

@dataclass(frozen=True)
class MixResult:
    """Per-part mix outputs (R-X5).

    D-02: frozen dataclass.
    """
    mix_path: str
    stem_paths: Dict[str, str]         # layer → post-FX wav path (incl. silent stubs)
    part_layers: Dict[str, bool]       # layer → included in mix
    soundfonts: Dict[str, str]         # layer → sf2 path
    pedalboards: Dict[str, list]       # layer → pedalboard_info_json output
    transitions: list                  # [[part, start_s], ..., ["end", end_s]]
```

### Pattern 2: RNG Threading (D-17/D-18)

```python
# All random draws take explicit rng parameter — no bare random.* in src/musicgen/
def pick_soundfonts(cfg: config.Config, rng: random.Random) -> Dict[str, str]:
    """Replace music_gen.py:117-120 get_random_sound_font × 4 (D-08/D-17)."""
    soundfonts = {}
    for layer in ("beat", "melody", "harmony", "bassline"):
        sf_dir = cfg.sf_layer_dir(layer)
        sf2_files = [f for f in os.listdir(sf_dir) if f.endswith(".sf2")]
        soundfonts[layer] = os.path.join(sf_dir, rng.choice(sf2_files))
    return soundfonts

def compute_layer_mask(
    song_unique_parts: List[str],
    inst_proba: dict,
    rng: random.Random,
) -> Dict[str, Dict[str, bool]]:
    """Replace music_gen.py:250-253 bare random.random() draws (D-13/D-17)."""
    mask = {}
    for part in song_unique_parts:
        mask[part] = {
            layer: rng.random() <= float(inst_proba[part][layer])
            for layer in ("beat", "melody", "harmony", "bassline")
        }
    return mask
```

### Pattern 3: FLUIDSYNTH_VERSION Capture (D-07)

```python
# Source: CONTEXT.md D-07 (locked)
import subprocess
import logging

logger = logging.getLogger(__name__)

try:
    _result = subprocess.run(
        ["fluidsynth", "--version"],
        capture_output=True, text=True, timeout=5,
    )
    FLUIDSYNTH_VERSION: str = _result.stdout.splitlines()[0]
except Exception:
    FLUIDSYNTH_VERSION = "unknown"
    logger.warning("Could not capture FluidSynth version; renderer importable on CI")
```

### Pattern 4: mido tick2second Beat Extraction (D-19)

Extracted from `beat_anotator.py:21-35` — this is the exact template:

```python
# Source: beat_anotator.py:21-35 (template per D-19)
import mido

def extract_beat_times(
    midi_path: str,
    tempo: int,
    start_offset_seconds: float,
) -> List[float]:
    """Extract beat timestamps from MIDI note_on events.

    Swing is already baked into MIDI onset times (generate_beat applies
    calculate_swing_offset at write time), so tick extraction is
    automatically swing-aware (D-19).
    """
    midi = mido.MidiFile(midi_path)
    beats = []
    time_elapsed = 0.0
    ticks_per_beat = midi.ticks_per_beat
    tempo_us = mido.bpm2tempo(tempo)
    for msg in midi:  # merged tracks; msg.time is delta ticks
        time_elapsed += mido.tick2second(msg.time, ticks_per_beat, tempo_us)
        if msg.type == "note_on" and msg.velocity > 0:
            beats.append(round(time_elapsed + start_offset_seconds, 3))
    return sorted(beats)
```

### Pattern 5: Silent-Stem Fallback (D-12) — STEREO MATCH REQUIRED

```python
# AudioSegment.silent() defaults to MONO 11025 Hz.
# FluidSynth renders STEREO 44100 Hz (midi2audio default).
# Must explicitly match to avoid channel-count mismatch in overlay.
# [VERIFIED: AudioSegment.silent(1000).channels == 1, .frame_rate == 11025]

def _make_silent_stem(duration_ms: int, sample_rate: int = 44100, channels: int = 2) -> AudioSegment:
    """Create a silent AudioSegment matching rendered stem spec (D-12)."""
    silent = AudioSegment.silent(duration=duration_ms, frame_rate=sample_rate)
    return silent.set_channels(channels)
```

### Pattern 6: Package-Wide AST Bare-Random Guard (D-31)

```python
# Source: tests/test_generators/test_no_bare_random.py (established pattern to extend)
import ast, glob, os, pytest

PACKAGE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, "src", "musicgen")
)

@pytest.mark.parametrize("path", sorted(
    p for p in glob.glob(os.path.join(PACKAGE_DIR, "**", "*.py"), recursive=True)
    if not p.endswith("__init__.py")
))
def test_no_bare_random_in_package_module(path):
    with open(path) as f:
        source = f.read()
    tree = ast.parse(source)
    hits = [
        node for node in ast.walk(tree)
        if (isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "random"
            and node.func.attr != "Random")   # random.Random() constructor is permitted
    ]
    assert hits == [], f"{os.path.basename(path)}: {len(hits)} bare random.<method>() at lines {[n.lineno for n in hits]}"
```

### Anti-Patterns to Avoid

- **Bare `random.*` draws in new modules:** D-17 is a hard regression signal. Use `rng.choice`, `rng.random`, `rng.uniform`, `rng.randint`.
- **`AudioSegment.silent()` without explicit stereo:** Produces mono 11025 Hz, mismatching FluidSynth's stereo 44100 Hz output. Always call `.set_channels(2)` and pass `frame_rate=44100`.
- **Raising at import in `renderer.py`:** `FLUIDSYNTH_VERSION` capture must be in a `try/except Exception` block. CI machines do not have FluidSynth installed; the module must be importable for mocked unit tests.
- **Generating beats from the theoretical grid:** The D-20 stride approach is documented as an approximation. The correct downbeat implementation is a time-grid (see Pitfall 2 below).
- **Testing FluidSynth behavior in unit tests:** Unit tests mock `FluidSynth.midi_to_audio`. Only the `@pytest.mark.slow` E2E test uses a real FluidSynth binary.
- **`_lin_to_db` as a nested function inside the loop:** The current `music_gen.py:285-286` defines it inside `mix_and_save`. Move it to module-private scope in `mixer.py`.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| MIDI-tick to seconds conversion | Custom tick math | `mido.tick2second(tick, ticks_per_beat, tempo_us)` | Handles tempo changes, edge cases |
| BPM to microseconds | `60_000_000 // bpm` approximation | `mido.bpm2tempo(bpm)` | Integer rounding difference matters at high precision |
| Audio effect chains | Custom DSP code | `pedalboard.Pedalboard` with existing JSON specs | 7 effect types, probability-weighted construction already in `create_effect` |
| Audio overlay with gain | Raw sample arithmetic | `pydub.AudioSegment.overlay` + `apply_gain` + `pan` | Phase counts on the R-S4 fix being identical behavior |
| Parallel subprocess dispatch | Custom thread pool | `concurrent.futures.ThreadPoolExecutor` | subprocess releases GIL; threads are optimal |
| MIDI reading | Binary MIDI parser | `mido.MidiFile` | Already used in `beat_anotator.py`; handles merged track iteration |

**Key insight:** The entire audio-processing pipeline is already built with established libraries. Phase 4 is a code-organization task, not a new-feature task. Every capability needed in the four new modules already exists in `music_gen.py` — the work is extraction and re-wiring, not invention.

---

## Dependency Verification Results

### 1. `mido` — NOT INSTALLED, NOT A TRANSITIVE DEP

[VERIFIED: `pip show midi2audio` shows `Requires: (empty)`; `pip list` shows no mido; `python -c "import mido"` raises `ModuleNotFoundError`]

`mido 1.3.3` is on PyPI and installable. It requires only `packaging` (already installed). It has zero audio system dependencies. Add `"mido>=1.3.3"` to `pyproject.toml` before writing `beats.py`.

### 2. `FluidSynth` binary — NOT ON PATH

[VERIFIED: `which fluidsynth` → not found; `/usr/bin/fluidsynth` → not found; `subprocess.run(["fluidsynth", "--version"])` fails]

FluidSynth is not installed in this WSL2 environment. The `FLUIDSYNTH_VERSION` capture will produce `"unknown"` with a logger.warning on this machine. The E2E integration test (D-30) must be guarded with a binary-presence check before invoking `FluidSynth.midi_to_audio`. Pattern: use `shutil.which("fluidsynth")` in a pytest fixture or `pytest.skip` call.

### 3. `pedalboard` — DETERMINISTIC

[VERIFIED: direct test — `Pedalboard([Reverb(room_size=0.5), Compressor(threshold_db=-20)])` applied to same numpy array twice produces bit-identical output with both `reset=True` and `reset=False`]

The FX application in `apply_fx_to_layer` is deterministic given the same input WAV and same Pedalboard configuration. Phase 5 golden-seed baselines on WAV content are valid once RNG draw order is locked.

### 4. `AudioSegment.silent()` — MONO BY DEFAULT

[VERIFIED: `AudioSegment.silent(duration=1000).channels == 1`, `.frame_rate == 11025`]

FluidSynth via `midi2audio` renders stereo (2 channels) at 44100 Hz (default `DEFAULT_SAMPLE_RATE = 44100` in midi2audio.py line 7; FluidSynth renders stereo by default). The D-12 silent-stem fallback MUST call `.set_channels(2)` and pass `frame_rate=44100`. The `RenderResult.sample_rate` and a new `RenderResult.channels` field (or hardcoded 2) should drive this. See Pitfall 1 below.

### 5. `pyproject.toml` markers — NOT DECLARED

[VERIFIED: `pyproject.toml` `[tool.pytest.ini_options]` contains only `testpaths` and `pythonpath`; no `markers =` list]

The `@pytest.mark.slow` and `@pytest.mark.integration` markers from STACK.md are NOT yet declared. Running tests with undeclared markers causes pytest warnings and may cause `--strict-markers` failures. Add a `markers =` list to `[tool.pytest.ini_options]` in `pyproject.toml` before the D-30 integration test lands.

---

## R-P4 Schema — Authoritative Field List

The annotator (D-14) must produce a dict with exactly these keys. Phase-4-filled fields get computed values; TBD fields get `None`. Changing field names after Phase 4 is expensive.

### Phase 4 FILLS these fields (non-None):

| Field | Source | Notes |
|-------|--------|-------|
| `key` | `SongParams.key` | e.g., `"Am"` |
| `mode` | derived from key | `"minor"` if key ends in `"m"`, else `"major"` |
| `tempo_bpm` | `SongParams.tempo` | integer |
| `time_signature` | `SongParams.time_signature_base` | e.g., `"4/4"` |
| `time_signatures_per_part` | `SongParams.signatures_per_part` | Dict[part, sig] |
| `measures_per_part` | `SongParams.measures_per_part` | Dict[part, int] |
| `swing` | `SongParams.swing_amount` | float in [0.5, 0.75] |
| `song_arrangement` | `MixResult.transitions` → List[{part, start_seconds, end_seconds}] | derived from transitions pairs |
| `chord_progression` | per-part, from generators output | Dict[part, List[str]] |
| `active_layers` | `MixResult.part_layers` | Dict[part, Dict[layer, bool]] |
| `soundfonts` | `MixResult.soundfonts` | Dict[layer, sf2_path] |
| `fx_params` | `MixResult.pedalboards` | Dict[layer, pedalboard_info_json output] |
| `beat_times` | `beats.extract_beat_times(...)` | Dict[part, List[float]] |
| `downbeat_times` | `beats.extract_downbeat_times(...)` | Dict[part, List[float]] |
| `musicality_score` | `musicality` dict parameter | `{"score": float, "components": {...}}` |
| `duration_seconds` | total mix duration | from `MixResult` concat output |
| `fluidsynth_version` | `FLUIDSYNTH_VERSION` module constant | `"FluidSynth runtime version 2.x.x"` or `"unknown"` |
| `mix` | path to mix WAV | absolute path (Phase 5 rewrites to relative) |
| `stems` | Dict[layer, path] | per-layer post-FX stem WAV paths |
| `midi` | Dict[layer, path] | per-layer MIDI paths |

### Phase 5 FILLS these fields (Phase 4 sets to None):

| Field | D-16 | Phase 5 requirement |
|-------|------|---------------------|
| `seed` | `None` | R-P7 seed discipline |
| `musicgen_version` | `None` | R-Q4 version field |
| `split` | `None` | R-P6 train/valid/test split |
| `pre_roll_offset_seconds` | `None` | R-P9 FluidSynth calibration |

### Optional field (Phase 4 omits entirely on success):

| Field | When present | Value |
|-------|-------------|-------|
| `analysis_failed` | Only if musicality scoring raises | `True` (D-16: omitted, not `False`, when scoring succeeds) |

Note from D-16: `analysis_failed` is `False` when present this phase and omitted when absent — but per CONTEXT.md D-16 clarification, the key itself is optional and omitted when scoring succeeds. Do not emit `"analysis_failed": False` unconditionally.

---

## Common Pitfalls

### Pitfall 1: AudioSegment.silent() Channel Mismatch

**What goes wrong:** `mixer.py` creates a silent stem with `AudioSegment.silent(duration=ms)`, producing a MONO 11025 Hz segment. When overlaid with stereo 44100 Hz rendered audio, pydub silently converts to mono or produces a mismatched result. The Phase 5 stem-sum-to-mix assertion (R-P2) fails or produces unexpected results.

**Why it happens:** `AudioSegment.silent()` has a default of 1 channel and 11025 Hz frame rate. FluidSynth renders at 2 channels, 44100 Hz.

**How to avoid:** Always construct silent stems as: `AudioSegment.silent(duration=ms, frame_rate=44100).set_channels(2)`. Use `RenderResult.sample_rate` for the frame_rate argument (allows future config override). Add an assertion in the mixer test that silent-stem channels == 2.

**Warning signs:** `TypeError` or `AssertionError` in pydub overlay; R-P2 assertion fails in Phase 5; RMS of "silent" stem is non-zero.

**Confidence:** HIGH [VERIFIED: tested with pydub 0.25.1]

---

### Pitfall 2: D-20 Downbeat Stride Assumption Breaks for Sparse Patterns

**What goes wrong:** `extract_downbeat_times(beat_times, time_signature)` implemented as `beat_times[::numerator]` produces wrong downbeat timestamps or crashes with `IndexError`.

**Root cause:** `extract_beat_times` only captures timestamps where `msg.type == "note_on" and msg.velocity > 0`. Beat patterns contain zeros (silent slots with no MIDI note). For 12/8, patterns have 9-10 non-zero entries per measure (not 12). For 4/4, patterns like `intro: 0, 42, 38, 0` have 2 non-zero entries per measure (not 4). The first beat slot is sometimes zero — verified in `beat_roll_patterns_44.txt` (e.g., `intro: 0, 42, 38, 0`).

**Evidence:**
- 12/8 `beat_roll_patterns_128.txt`: e.g., `intro: 36, 0, 42, 38, 42, 0, 36, 42, 38, 42, 0, 36` → 9 non-zero entries per measure (not 12) [VERIFIED by inspection]
- 4/4 `beat_roll_patterns_44.txt`: e.g., `intro: 0, 42, 38, 0` → first slot is zero → downbeat NOT in beat_times [VERIFIED by inspection]
- 6/8 `beat_roll_patterns_68.txt`: ALL entries non-zero → stride-6 would work for 6/8 specifically [VERIFIED]
- 7/8 `beat_roll_patterns_78.txt`: ALL entries non-zero → stride-7 would work for 7/8 specifically [VERIFIED]

**How to fix:** Compute downbeat times as a **time grid**, not a beat_times slice:

```python
def extract_downbeat_times(
    beat_times: List[float],        # kept for API compatibility
    time_signature: str,
    measures: int,
    start_offset_seconds: float,
    tempo: int,
) -> List[float]:
    """Derive downbeat timestamps as a pure time grid (one per measure).

    Does NOT stride-slice beat_times — beat_times is sparse for patterns
    with zero-valued slots. Instead computes measure_duration from tempo
    and time signature, then emits one timestamp per measure.

    The TimeSignatureRegistry.beats_per_measure field gives the musical
    beat count (2 for 6/8, 4 for 12/8), used only if per-pulse downbeats
    are needed for a compound-meter task (this function emits measure-level
    downbeats, one per measure).
    """
    spec = TimeSignatureRegistry.lookup(time_signature)
    # beat_duration() gives the duration of ONE beat slot in seconds
    beat_slot_s = beat_duration(time_signature, tempo)
    measure_duration_s = spec.numerator * beat_slot_s
    return [
        round(start_offset_seconds + i * measure_duration_s, 3)
        for i in range(measures)
    ]
```

**Impact on D-20:** The D-20 description (`beat_times[::numerator]`) is the approximation that was assumed for planning. The correct implementation is the time-grid approach above. The registry's `beats_per_measure` field exists and is correct (6/8: 2, 12/8: 4) — it's not needed for measure-level downbeats but can be used if per-pulse downbeats are needed in the future.

**Warning signs:** `downbeat_times` has wrong count (e.g., 9 or 10 instead of measures count for 12/8); first downbeat is not at `start_offset_seconds` for parts starting with a silent beat slot.

**Confidence:** HIGH [VERIFIED: inspected all 6 beat pattern files; confirmed zero presence by signature]

---

### Pitfall 3: FluidSynth Not Importable as Module

**What goes wrong:** `renderer.py`'s `FLUIDSYNTH_VERSION` capture block is not inside `try/except`. `ImportError` or `FileNotFoundError` propagates on import, breaking all mocked unit tests.

**Why it happens:** D-07 explicitly says "do not raise at import." The renderer must be importable on CI without FluidSynth installed.

**How to avoid:** The entire subprocess block goes inside `try/except Exception`. Use `logger.warning`, not `logger.error`. The fallback value is `"unknown"`. Unit tests mock `FluidSynth.midi_to_audio` and import `renderer` without needing the binary.

**Warning signs:** `test_renderer.py` fails with `FileNotFoundError` or `OSError` on import; `pytest --collect-only` fails.

---

### Pitfall 4: FX Applied Only to Used Layers (D-11 Violation)

**What goes wrong:** During refactor, someone moves the `apply_fx_to_layer` call inside the `if beat_part_mix[part]:` branch to "optimize." This changes the RNG draw order compared to today's code (all 4 FX calls happen unconditionally before any overlay).

**Why it happens:** The `music_gen.py:276` TODO comment ("optimize so fx only applied to used layers") looks like an invitation. It is explicitly locked as OUT OF SCOPE per D-11.

**How to avoid:** The mixer must apply FX to all 4 layers unconditionally, THEN decide which layers to overlay. The exact code from `music_gen.py:277-280` lifts verbatim. The test for D-11 compliance: mixer tests check that `build_fx_boards` is called once per layer (not conditionally).

**Warning signs:** Seeded mixer tests produce different FX params when layer mask changes; pedalboard construction count != 4.

---

### Pitfall 5: AST Guard False Positives on `random.Random` Constructor

**What goes wrong:** The D-31 package-wide AST guard flags `rng = random.Random()` or `_rng = random.Random()` as a bare random draw.

**Why it doesn't happen:** The existing `_bare_random_calls` helper in `tests/test_sampler.py` already handles this correctly: `node.func.attr != "Random"` excludes the constructor. The extended `test_no_bare_random_in_package.py` must copy this exact condition. [VERIFIED: existing guard passes on `music_gen.py` which contains `_rng = random.Random()`]

**Warning signs:** `tests/test_no_bare_random_in_package.py` fails on `music_gen.py` line 46 (`_rng = random.Random()`) or on sampler.py's test fixtures.

---

### Pitfall 6: music_gen.py Collapse Breaks AST Tests

**What goes wrong:** Reducing `music_gen.py` from 523 to ~180 lines removes functions that `tests/test_music_gen_logging.py` looks for.

**What the test actually checks:** (1) Zero `print()` calls in `music_gen.py`; (2) Import is side-effect-free; (3) `logger = logging.getLogger(__name__)` is present at module level; (4) `logging.basicConfig` appears only inside `if __name__ == "__main__"`. None of these assertions check for the presence of specific functions or minimum line count. [VERIFIED: read `tests/test_music_gen_logging.py`]

**How to avoid:** Keep `logger = logging.getLogger(__name__)` in the collapsed `music_gen.py`. Keep `logging.basicConfig` inside the `__main__` guard. No print calls. All four assertions will pass automatically as long as the shim is properly migrated.

---

### Pitfall 7: beat_anotator.py Deletion Test Regression

**What might go wrong:** Some test imports `beat_anotator`.

**Verification result:** Zero importers found. [VERIFIED: `grep -r "beat_anotator" /home/bidu/musicgen/ --include="*.py"` returns only a self-reference inside `beat_anotator.py`]. No migration work needed. Delete the file; the 371-test baseline is unaffected.

---

## Code Examples

### Extraction Map: music_gen.py Functions → Phase 4 Modules

| Source Location | Lines | Function | Destination | Action |
|----------------|-------|----------|-------------|--------|
| `music_gen.py` | 117–120 | `get_random_sound_font` | `renderer.pick_soundfonts` | Replace 4× bare `random.choice` with `rng.choice` |
| `music_gen.py` | 122–125 | `get_levels` | `mixer.py` (inline call or helper) | Move verbatim |
| `music_gen.py` | 112–115 | `read_instrument_probabilities` | `mixer.py` (inline call or helper) | Move verbatim |
| `music_gen.py` | 130–139 | `create_effect` | `mixer.py` | Replace `random.random()` + `random.uniform()` with `rng.random()` + `rng.uniform()` |
| `music_gen.py` | 141–160 | `generate_pedalboard` | `mixer.build_fx_boards` or `mixer.generate_pedalboard` | Move; add `rng` param; call `create_effect(..., rng)` |
| `music_gen.py` | 162–171 | `apply_fx_to_layer` | `mixer.py` | Move verbatim (no rng draws) |
| `music_gen.py` | 173–190 | `pedalboard_info_json` | `mixer.py` | Move verbatim (no rng draws) |
| `music_gen.py` | 285–286 | `_lin_to_db` (nested) | `mixer._lin_to_db` (module-private) | Unnest; make module-private |
| `music_gen.py` | 193–345 | `mix_and_save` | DELETED (D-23) | Responsibilities distributed per D-23 |
| `music_gen.py` | 98–107 | `save_beat_annotations` | DELETED | Phase 5 writer owns annotation lifecycle |
| `beat_anotator.py` | 6–12 | `beat_duration` | `beats.py` (primary); `generators/beat.py` (re-export alias) | D-21 |
| `beat_anotator.py` | 21–35 | `extract_midi_beats` | `beats.extract_beat_times` | Rename; add `start_offset_seconds` param (already there) |
| `beat_anotator.py` | entire | — | DELETED (D-03/D-19) | Zero importers confirmed |

### Orchestrator Shape Post-Phase

```python
# music_gen.py create_song() after Phase 4 collapse (~180 lines total)
def create_song(key, tempo, song_signatures, measures, name, chord_pat_file,
                swing_amount, cfg=None, *, song_unique_parts, song_arrangement):
    _cfg = cfg if cfg is not None else config.Config()

    # Step 1: soundfont selection (renderer, rng-aware)
    soundfonts = renderer.pick_soundfonts(_cfg, _rng)

    # Step 2: generators (unchanged from Phase 3)
    ha, ba, me, be, _ = generate_song_parts(...)

    # Step 3: FX boards + layer mask (mixer, rng-aware)
    fx_boards = mixer.build_fx_boards(_cfg, _rng)
    layer_mask = mixer.compute_layer_mask(song_unique_parts, _cfg.load_inst_probabilities(), _rng)
    levels = _cfg.load_levels()

    # Step 4: per-part render + mix (serial parts, parallel stems)
    render_results, mix_results, beat_times_dict, downbeat_times_dict = {}, {}, {}, {}
    part_counter = 0
    for part in song_arrangement:
        part_counter += 1
        midi_paths = {"beat": be[part], "melody": me[part], "harmony": ha[part], "bassline": ba[part]}
        out_dir = os.path.join(name, f"{name}-{part}")
        render_results[part] = renderer.render_stems(midi_paths, soundfonts, out_dir)
        mix_results[part] = mixer.mix_part(
            render_results[part], levels, fx_boards, layer_mask, part, out_dir
        )
        start_offset = mix_results[part].transitions[-1][1] if mix_results else 0.0
        beat_times_dict[part] = beats.extract_beat_times(be[part], tempo, start_offset)
        downbeat_times_dict[part] = beats.extract_downbeat_times(
            beat_times_dict[part], song_signatures[part], measures[part], start_offset, tempo
        )

    # Step 5: concatenate parts
    final_wav = mixer.concat_parts(list(mix_results.values()), os.path.join(name, name + ".wav"))

    # Step 6: musicality scoring (stays at root, D-04)
    score, components = musicality_score.get_musicality_score(final_wav)
    musicality = {"score": float(score), "components": {k: float(v) for k, v in components.items()}}

    # Step 7: annotation (pure function)
    annotation = annotator.annotate(
        song_params=SongParams(...),  # constructed from current params
        render_results=render_results,
        mix_result=mix_results,  # or pass the concat result
        beat_times=beat_times_dict,
        downbeat_times=downbeat_times_dict,
        musicality=musicality,
        fluidsynth_version=renderer.FLUIDSYNTH_VERSION,
    )

    # Step 8: write JSON (Phase 5 writer takes over this)
    with open(os.path.join(name, name + ".json"), "w") as f:
        json.dump(annotation, f, indent=4)

    return annotation
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Bare `random.*` in audio pipeline | Injected `rng: random.Random` | Phase 4 (this phase) | Enables Phase 5 golden-seed determinism |
| Theoretical grid beat annotations (`beat_anotator.py`) | MIDI-tick-derived beat times (`beats.py`) | Phase 4 (this phase) | Eliminates P-3 swing drift; beats reflect actual MIDI onsets |
| `mix_and_save` monolith | Distributed renderer/mixer/annotator/beats | Phase 4 (this phase) | Testable, parallelizable, seed-disciplined |
| Sequential FluidSynth renders (4 per part, serial) | `ThreadPoolExecutor(max_workers=4)` per part | Phase 4 (this phase) | ~4× speedup on 4-layer parts (FluidSynth is subprocess, GIL not a factor) |

**Deprecated/outdated after this phase:**
- `beat_anotator.py`: deleted; `generate_annotations` / `compare_beats` / `theoretical_beats` logic is wrong for swing > 0.
- `mix_and_save`: deleted; responsibilities distributed across renderer, mixer, orchestrator.
- `save_beat_annotations`: deleted; Phase 5 writer owns annotation lifecycle.
- `get_random_sound_font`: deleted; replaced by `renderer.pick_soundfonts(cfg, rng)`.

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | FluidSynth renders 2-channel stereo at 44100 Hz by default when invoked via `midi2audio` | Silent-stem parity, Pitfall 1 | Silent stems would mismatch; overlay behavior undefined. Mitigation: read `RenderResult.sample_rate` and `RenderResult.channels` from the actual rendered WAV in the mixer, not from hardcoded constants. |
| A2 | The CONTEXT.md `D-20` stride-slice approach is acceptable to amend to a time-grid approach | Pitfall 2, beats.py design | If planner treats D-20 as locked, the stride implementation is wrong for 12/8 and some 4/4 patterns. The CONTEXT.md D-20 states "Validated by test fixture: synthesize a known MIDI, run extract_beat_times + extract_downbeat_times" — this test will catch the bug. Flag to planner to amend D-20. |
| A3 | `FluidSynth --version` outputs to stdout (not stderr) on Linux/WSL2 | D-07 version capture | If FluidSynth outputs to stderr, `stdout.splitlines()[0]` returns empty string instead of "unknown". Mitigation: capture both and check `stdout or stderr`. |

---

## Open Questions

1. **Beat_times signature for annotator — Dict[part, List[float]] or List[float]?**
   - What we know: CONTEXT.md D-14 shows `beat_times: Dict[str, List[float]]` (per-part) in the annotator signature.
   - What's unclear: D-22 says beats run per-part from the orchestrator — so the orchestrator accumulates a dict before calling annotator. This matches D-14.
   - Recommendation: Orchestrator builds `{part: extract_beat_times(...)}` dict, passes the full dict to annotator.

2. **`chord_progression` source for annotator — currently not returned by generate_song_parts**
   - What we know: `generate_song_parts` returns `(harm_filename, bass_filename, melo_filename, beat_filename, beat_annotations)` — MIDI paths, not chord data. The chord progression is currently consumed internally inside generators.
   - What's unclear: The annotator needs `chord_progression` per-part (D-15 R-P4 schema). Either `generate_chord_progression` must return the progression alongside the MIDI path (it already returns it: `chord_progression, harm_filename[part]`), or it must be collected from the existing generator call sites.
   - Recommendation: `generate_song_parts` already calls `generate_chord_progression` and receives `chord_progression` in the return — it just discards it. The orchestrator refactor should collect chord progressions per-part and thread them to the annotator. No generator changes needed.

3. **`song_arrangement` vs `transitions` source for annotator**
   - What we know: D-15 says `song_arrangement` in the annotation is `List[{part, start_seconds, end_seconds}]` derived from `mix_result.transitions`. The current `transitions` is `[[part, start_s], ..., ["end", end_s]]`.
   - Recommendation: Annotator converts `transitions` to the required schema format internally. E.g., `[{"part": t[0], "start_seconds": t[1], "end_seconds": transitions[i+1][1]} for i, t in enumerate(transitions[:-1])]`.

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| `mido` | `beats.py` (MIDI reading) | NO — must install | 1.3.3 on PyPI | None — blocking; must add to pyproject.toml |
| `FluidSynth` binary | `renderer.render_stems` (unit tests: mocked) | NO (not on PATH) | — | Unit tests mock FluidSynth.midi_to_audio; E2E test skips via `pytest.skip` on `shutil.which("fluidsynth") is None` |
| `pedalboard` | `mixer.py` FX | YES | 0.9.22 | — |
| `pydub` | `mixer.py` audio assembly | YES | 0.25.1 | — |
| `midi2audio` | `renderer.py` (FluidSynth wrapper) | YES | 0.1.1 | — |
| `midiutil` | `test_beats.py` (MIDI fixture creation) | YES | 1.2.1 | — |
| `ffmpeg` | pydub `.export()` | NO (warns but not blocking for in-memory ops) | — | pydub warns on import; in-memory AudioSegment ops work without ffmpeg; WAV export via pydub uses ffmpeg if available but can fall back to soundfile |
| Python 3.12 | — | YES | 3.12.3 | — |

**Missing dependencies with no fallback:**
- `mido` — blocks `beats.py` implementation and test execution. Must install before Wave 1 of plans.

**Missing dependencies with fallback:**
- `FluidSynth` binary — unit tests mock away the binary. Only the `@pytest.mark.slow` E2E test requires the real binary. CI can run the full suite except `@pytest.mark.slow`.
- `ffmpeg` — pydub still works for `AudioSegment.silent()`, `.overlay()`, `.apply_gain()`, `.pan()`. WAV file export via `AudioSegment.export()` may work through alternative backends; test with actual render output.

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest 9.0.3 |
| Config file | `pyproject.toml` `[tool.pytest.ini_options]` |
| Quick run command | `python -m pytest tests/ -m "not slow" -q` |
| Full suite command | `python -m pytest tests/ -q` |
| Slow tests only | `python -m pytest tests/ -m slow -q` |

**pyproject.toml update required:** Add `markers` list before D-30 lands:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
markers = [
    "slow: FluidSynth rendering tests — requires fluidsynth binary (deselect with '-m not slow')",
    "integration: end-to-end tests requiring system dependencies",
]
```

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| R-X4 | `render_stems` dispatches 4 parallel FluidSynth renders → RenderResult | unit (mocked) | `pytest tests/test_renderer.py -x` | NO — Wave 0 |
| R-X4 | `FLUIDSYNTH_VERSION` captured at import, fallback to "unknown" | unit | `pytest tests/test_renderer.py::test_fluidsynth_version_capture -x` | NO — Wave 0 |
| R-X4 | `pick_soundfonts` uses injected rng, no bare random.* | unit + AST guard | `pytest tests/test_no_bare_random_in_package.py tests/test_renderer.py -x` | NO — Wave 0 |
| R-X5 | `build_fx_boards` + `compute_layer_mask` seeded-RNG determinism | unit | `pytest tests/test_mixer.py -x` | NO — Wave 0 |
| R-X5 | Silent-stem fallback channels == 2, frame_rate == 44100 | unit | `pytest tests/test_mixer.py::test_silent_stem_channels -x` | NO — Wave 0 |
| R-X5 | FX applied to all 4 layers regardless of mask (D-11) | unit | `pytest tests/test_mixer.py::test_fx_applied_to_all_layers -x` | NO — Wave 0 |
| R-X6 | Annotator returns dict with all Phase-4 fields non-None | unit (fixture) | `pytest tests/test_annotator.py -x` | NO — Wave 0 |
| R-X6 | Phase-5 TBD fields are present as None (not missing) | unit (fixture) | `pytest tests/test_annotator.py::test_tbd_fields_are_none -x` | NO — Wave 0 |
| R-X7 | `extract_beat_times` returns monotonic timestamps at correct BPM | unit | `pytest tests/test_beats.py -x` | NO — Wave 0 |
| R-X7 | Swing cases 0.5/0.66/0.75 — off-beats later than straight grid | unit | `pytest tests/test_beats.py::TestSwingCases -x` | NO — Wave 0 |
| R-X7 | `extract_downbeat_times` returns one timestamp per measure | unit | `pytest tests/test_beats.py::test_downbeat_count -x` | NO — Wave 0 |
| R-X8 | E2E: 4 stems + 1 mix + 4 MIDI exist; annotation dict complete | integration (slow) | `pytest tests/test_integration_full_generation.py -m slow -x` | NO — Wave 0 |
| D-17/D-31 | Zero bare `random.<method>()` across `src/musicgen/**/*.py` | static (AST) | `pytest tests/test_no_bare_random_in_package.py -x` | NO — Wave 0 |
| — | Existing 371-test baseline stays green | regression | `pytest tests/ -m "not slow" -q` | YES (371 tests) |

### Sampling Rate

- **Per task commit:** `python -m pytest tests/ -m "not slow" -q --tb=short`
- **Per wave merge:** `python -m pytest tests/ -m "not slow" -q`
- **Phase gate:** `python -m pytest tests/ -m "not slow" -q` green (371 + new tests); `@pytest.mark.slow` E2E skips on CI (no FluidSynth binary)

### Wave 0 Gaps

- [ ] `tests/test_renderer.py` — covers R-X4
- [ ] `tests/test_mixer.py` — covers R-X5
- [ ] `tests/test_annotator.py` — covers R-X6
- [ ] `tests/test_beats.py` — covers R-X7
- [ ] `tests/test_integration_full_generation.py` — covers R-X8 (`@pytest.mark.slow`)
- [ ] `tests/test_no_bare_random_in_package.py` — covers D-31
- [ ] `pyproject.toml` markers section — enables `@pytest.mark.slow` without warnings
- [ ] `pip install mido>=1.3.3` (add to pyproject.toml, re-install) — unblocks beats.py tests

---

## Security Domain

> `security_enforcement` not set to false in config.json — treating as enabled.

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | No | Not applicable — local tool, no auth |
| V3 Session Management | No | Not applicable |
| V4 Access Control | No | Not applicable — local filesystem |
| V5 Input Validation | Partial | `config.Config` paths normalized via `os.path.abspath`; FX JSON loaded with stdlib `json` (no exec); MIDI files read via `mido` (established library) |
| V6 Cryptography | No | No cryptographic operations in Phase 4 |

### Known Threat Patterns

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Path traversal via `out_dir` parameter | Tampering | Accept as caller responsibility; `renderer.py` writes only to paths under `out_dir`; Phase 5 writer will add `os.path.abspath` normalization |
| Malformed FX JSON file | Tampering | `json.load` raises `json.JSONDecodeError` which propagates naturally; no exec or eval |
| MIDI file with unusual content | Tampering | `mido.MidiFile` is a pure Python reader; malformed MIDI raises `mido.MidiException` or `ValueError` — let propagate to caller |
| FluidSynth subprocess injection via soundfont path | Elevation | Soundfont paths come from `os.listdir(sf_dir)` filtered to `.sf2` — no user-controlled path injection; `cfg.sf_layer_dir` is trusted |

---

## Implementation Order and Wave Grouping

Phase 4 has a strict dependency chain: beats.py → renderer.py → mixer.py → annotator.py. The E2E test and orchestrator collapse require all four. Within this chain, some parallelism is possible between modules with no shared code dependency.

### Wave 0: Infrastructure (blocking)
- Add `mido>=1.3.3` to `pyproject.toml` + re-install
- Add `markers` list to `pyproject.toml` `[tool.pytest.ini_options]`
- Create all 6 test file stubs with failing placeholder tests

### Wave 1: beats.py + test_beats.py (no audio deps)
- `src/musicgen/beats.py`: `beat_duration`, `extract_beat_times`, `extract_downbeat_times`
- `generators/beat.py`: add `beat_duration` re-export alias, keeping function body in beats.py
- `tests/test_beats.py`: 4/4 grid test + 3 swing cases + downbeat count assertions

### Wave 2: renderer.py + test_renderer.py (parallel with Wave 1 once beats done)
- `src/musicgen/renderer.py`: `FLUIDSYNTH_VERSION`, `RenderResult`, `pick_soundfonts`, `render_stems`
- `tests/test_renderer.py`: all unit tests with mocked `FluidSynth.midi_to_audio`

### Wave 3: mixer.py + test_mixer.py (requires RenderResult from Wave 2)
- `src/musicgen/mixer.py`: `create_effect`, `generate_pedalboard` → `build_fx_boards`, `apply_fx_to_layer`, `pedalboard_info_json`, `compute_layer_mask`, `_lin_to_db`, `MixResult`, `mix_part`, `concat_parts`
- `tests/test_mixer.py`: seeded-RNG tests + silent-stem channel assertion + D-11 FX-on-all-layers

### Wave 4: annotator.py + test_annotator.py (requires MixResult from Wave 3)
- `src/musicgen/annotator.py`: `annotate(...)` pure function
- `tests/test_annotator.py`: fixture-driven golden dict tests

### Wave 5: orchestrator collapse + AST guard (requires all four modules)
- `music_gen.py`: collapse from 523 to ~180 lines; delete `mix_and_save`, `save_beat_annotations`, `get_random_sound_font`; import and call new modules
- `beat_anotator.py`: delete
- `tests/test_no_bare_random_in_package.py`: package-wide AST guard

### Wave 6: E2E integration test + phase gate
- `tests/test_integration_full_generation.py`: `@pytest.mark.slow` test, skips if no FluidSynth binary
- Phase gate verification: 371 + new tests green; `python music_gen.py` smoke test reaches annotator (env failure at FluidSynth OK)

---

## Sources

### Primary (HIGH confidence)

- `music_gen.py` lines 117–345, 98–107 — direct source extraction targets [VERIFIED by Read tool]
- `beat_anotator.py` lines 6–35 — beats.py template [VERIFIED by Read tool]
- `src/musicgen/generators/beat.py` — beat_duration, calculate_swing_offset; D-21 source [VERIFIED by Read tool]
- `src/musicgen/sampler.py` — frozen dataclass pattern, RNG threading [VERIFIED by Read tool]
- `tests/test_sampler.py` — AST guard `_bare_random_calls` helper [VERIFIED by Read tool]
- `tests/test_generators/test_no_bare_random.py` — parametrized AST guard pattern [VERIFIED by Read tool]
- `tests/test_music_gen_logging.py` — AST assertions on music_gen.py collapse safety [VERIFIED by Read tool]
- `/home/bidu/musicgen/.venv/lib/python3.12/site-packages/midi2audio.py` — FluidSynth invocation details, `DEFAULT_SAMPLE_RATE = 44100` [VERIFIED by Read tool]
- `pyproject.toml` — current dependency list, markers absence [VERIFIED by Read tool]
- `config.py` — `sf_layer_dir`, `fx_files`, `inst_probabilities_file`, `levels_file` [VERIFIED by Read tool]
- `timesig.py` via venv Python — `beats_per_measure` field values for 6/8 (2), 12/8 (4), 7/8 (7) [VERIFIED by live execution]
- `beat_roll_patterns_128.txt`, `_68.txt`, `_78.txt`, `_44.txt` — zero-presence analysis [VERIFIED by inspection]

### Secondary (MEDIUM confidence)

- `pedalboard` determinism: tested directly — `Pedalboard([Reverb(0.5), Compressor(-20)])` applied twice to same input produces bit-identical output [VERIFIED by live execution]
- `AudioSegment.silent()` defaults: `channels=1`, `frame_rate=11025` [VERIFIED by live execution with pydub 0.25.1]
- `mido` availability on PyPI: `pip index versions mido` → latest `1.3.3` [VERIFIED by pip index]
- `FluidSynth` binary: not on PATH [VERIFIED by `which fluidsynth`; `subprocess` fails]

### Tertiary (LOW confidence)

- FluidSynth 2.x `--version` stdout format: `"FluidSynth runtime version X.Y.Z"` on first line — [ASSUMED] from FluidSynth documentation; not directly verifiable without binary. D-07 fallback to `"unknown"` handles wrong format gracefully.

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all libraries verified via pip/venv
- Architecture: HIGH — directly derived from reading source extraction targets
- Pitfalls: HIGH — Pitfalls 1, 2 verified by testing/inspection; Pitfalls 3-7 verified by reading existing code
- Dependency findings: HIGH — all three critical findings (mido missing, FluidSynth absent, silent-stem mono) verified by live execution

**Research date:** 2026-04-19
**Valid until:** 2026-05-19 (stable libraries; mido/pydub/pedalboard change slowly)
