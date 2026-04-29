---
phase: 04-renderer-mixer-annotator-beats-extraction
plan: 04
type: execute
wave: 4
depends_on: ["04-00", "04-01", "04-03"]
files_modified:
  - src/musicgen/annotator.py
  - tests/test_annotator.py
autonomous: true
requirements: [R-X6]
tags: [phase-4, annotator, r-p4-schema, pure-function, json, tbd-fields]

must_haves:
  truths:
    - "`annotate(...)` is a pure function — zero I/O: no open(), no json.dump, no os.* filesystem writes inside the function body"
    - "Return value is a plain dict (ready for json.dump by the caller)"
    - "Every Phase-4-fillable R-P4 field is populated with a non-None value (full list in D-15)"
    - "Every Phase-5 TBD field is present with value `None` (D-16 — NOT missing, NOT the string 'TBD')"
    - "The `analysis_failed` key is OMITTED on success, not set to False (D-16 clarification)"
    - "`mode` is derived from `key` (minor if key ends in 'm', else major)"
    - "`song_arrangement` field is a list of `{part, start_seconds, end_seconds}` dicts derived from `mix_result.transitions` — not a raw transitions list"
    - "Zero bare `random.<method>` calls in `src/musicgen/annotator.py` (the module does not need random at all)"
  artifacts:
    - path: "src/musicgen/annotator.py"
      provides: "annotate(song_params, render_results, mix_results, beat_times, downbeat_times, musicality, chord_progressions, midi_paths, mix_path, *, fluidsynth_version, musicgen_version=None, seed=None, split=None, analysis_failed=None) -> dict"
      exports: ["annotate"]
      min_lines: 100
      contains: "def annotate"
    - path: "tests/test_annotator.py"
      provides: "Fixture-driven golden-dict tests covering D-15 field coverage + D-16 None semantics + no-I/O contract"
      contains: "def test_tbd_fields_are_none"
  key_links:
    - from: "src/musicgen/annotator.py"
      to: "musicgen.sampler.SongParams + musicgen.renderer.RenderResult + musicgen.mixer.MixResult"
      via: "type-hint imports (runtime imports for module resolution)"
      pattern: "from musicgen\\.(sampler|renderer|mixer) import"
    - from: "tests/test_annotator.py"
      to: "musicgen.annotator.annotate"
      via: "fixture-constructed inputs → dict output assertions"
      pattern: "annotate\\("
---

<objective>
Implement `src/musicgen/annotator.py` — the pure-function R-P4 schema assembler (R-X6). The annotator takes stage outputs from sampler + renderer + mixer + beats + musicality scoring and returns a plain dict matching the full R-P4 schema, with Phase-5 TBD fields present as `None` (D-15/D-16). Zero I/O: the caller writes the JSON (Phase 5 writer owns that lifecycle).

Populate `tests/test_annotator.py` with fixture-driven tests that construct synthetic `SongParams` + `RenderResult` + `MixResult` + beat/downbeat dicts + stub musicality dict, call `annotate(...)`, and assert the returned dict shape + Phase-4-fill completeness + Phase-5 None semantics. Include a no-I/O contract test (monkeypatch `open` to raise if called — must not trigger).

Purpose: The annotator is the dataset's contract surface. Phase 4 locks the SHAPE of every future `sample.json` file. Phase 5 will fill the currently-None fields (seed, musicgen_version, split, pre_roll_offset_seconds); the shape MUST NOT change between now and then. Any field name change after this plan ships is an expensive migration (every downstream consumer breaks). The annotator is also a pure function — no file I/O, no subprocess, no RNG — so it is cheap to test exhaustively and easy to integrate into the orchestrator collapse (Plan 04-05).

Output: `src/musicgen/annotator.py` (new, ~130-180 lines incl. docstrings), `tests/test_annotator.py` (replaced from Wave 0 stub with real tests).
</objective>

<execution_context>
@/home/bidu/musicgen/.claude/get-shit-done/workflows/execute-plan.md
@/home/bidu/musicgen/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/REQUIREMENTS.md
@.planning/phases/04-renderer-mixer-annotator-beats-extraction/04-CONTEXT.md
@.planning/phases/04-renderer-mixer-annotator-beats-extraction/04-RESEARCH.md
@.planning/phases/04-renderer-mixer-annotator-beats-extraction/04-PATTERNS.md
@.planning/phases/04-renderer-mixer-annotator-beats-extraction/04-VALIDATION.md
@.planning/phases/04-renderer-mixer-annotator-beats-extraction/04-01-SUMMARY.md
@.planning/phases/04-renderer-mixer-annotator-beats-extraction/04-02-SUMMARY.md
@.planning/phases/04-renderer-mixer-annotator-beats-extraction/04-03-SUMMARY.md
@src/musicgen/sampler.py
@src/musicgen/renderer.py
@src/musicgen/mixer.py

<interfaces>
<!-- SongParams from Phase 3 (sampler.py:223-293) — 9 fields -->
@dataclass(frozen=True)
class SongParams:
    key: str
    tempo: int
    time_signature_base: str
    time_signature_variation: float
    swing_amount: float
    signatures_per_part: Dict[str, str]
    measures_per_part: Dict[str, int]
    song_unique_parts: List[str]
    song_arrangement: List[str]

<!-- RenderResult from Plan 04-02 -->
@dataclass(frozen=True)
class RenderResult:
    stem_paths: Dict[str, str]      # layer → wav path
    sample_rate: int                 # 44100
    channels: int                    # 2
    duration_seconds: float
    fluidsynth_version: str

<!-- MixResult from Plan 04-03 -->
@dataclass(frozen=True)
class MixResult:
    mix_path: str
    stem_paths: Dict[str, str]       # layer → post-FX wav path (incl. silent stubs)
    part_layers: Dict[str, bool]
    soundfonts: Dict[str, str]
    pedalboards: Dict[str, list]     # layer → pedalboard_info_json output
    transitions: list                # [[part, start_s], ["end", end_s]]

<!-- R-P4 authoritative field list from RESEARCH.md (lines 454-494):
     Phase 4 FILLS:
       key, mode, tempo_bpm, time_signature, time_signatures_per_part,
       measures_per_part, swing, song_arrangement (computed from transitions),
       chord_progression (dict per-part), active_layers (mix_result.part_layers),
       soundfonts (mix_result.soundfonts), fx_params (mix_result.pedalboards),
       beat_times (dict per-part), downbeat_times (dict per-part),
       musicality_score (full musicality dict), duration_seconds (total),
       fluidsynth_version, mix (path), stems (dict), midi (dict)
     Phase 5 FILLS (None this phase):
       seed, musicgen_version, split, pre_roll_offset_seconds
     Optional (omitted on success):
       analysis_failed -->
</interfaces>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Create src/musicgen/annotator.py — pure-function annotate() with full R-P4 shape</name>
  <files>src/musicgen/annotator.py</files>
  <behavior>
    - `annotate(...)` is a pure function — zero I/O (no `open`, no `json.dump`, no `os.makedirs`, no `subprocess`, no network)
    - Function signature: positional (song_params, render_results, mix_results, beat_times, downbeat_times, musicality, chord_progressions, midi_paths, mix_path), keyword-only (fluidsynth_version, musicgen_version=None, seed=None, split=None, analysis_failed=None)
    - Return value is a dict with at least the keys listed in D-15 "FILLED" section — all non-None
    - Return value contains the keys `seed`, `musicgen_version`, `split`, `pre_roll_offset_seconds` with value `None` (D-16)
    - `analysis_failed` key is OMITTED from the dict when scoring succeeded (default `analysis_failed=None` means omit; only when caller explicitly passes `analysis_failed=True` does the key appear as True)
    - `mode` derivation: `"minor"` when key ends in `"m"` — `"Am"` → minor, `"A"` → major, `"A#m"` → minor, `"C"` → major
    - `song_arrangement` is a `List[Dict[str, Union[str, float]]]` with keys `part`, `start_seconds`, `end_seconds` — derived from `mix_results[part].transitions` with cumulative global offsets
    - `time_signature` field (singular) is `song_params.time_signature_base`
    - `time_signatures_per_part` is a COPY of `song_params.signatures_per_part` (defensive — don't leak the frozen dataclass dict into the mutable output dict)
    - `measures_per_part` is a COPY of `song_params.measures_per_part`
    - `swing` is `song_params.swing_amount`
    - `tempo_bpm` is `song_params.tempo`
    - `chord_progression` is `chord_progressions` parameter (per-part dict of list)
    - `duration_seconds` equals the sum across the arrangement (final `end_seconds` entry)
    - Zero bare `random.<method>` calls (module does not import random)
  </behavior>
  <read_first>
    - src/musicgen/sampler.py (SongParams shape — lines 223-293)
    - src/musicgen/renderer.py (RenderResult shape — from Plan 04-02 summary)
    - src/musicgen/mixer.py (MixResult shape — from Plan 04-03 summary)
    - .planning/REQUIREMENTS.md R-P4 (authoritative schema — lines 68-76)
    - .planning/phases/04-renderer-mixer-annotator-beats-extraction/04-CONTEXT.md D-14, D-15, D-16 (signature + fill list + None semantics)
    - .planning/phases/04-renderer-mixer-annotator-beats-extraction/04-RESEARCH.md §"R-P4 Schema — Authoritative Field List" (field-by-field source derivation)
    - .planning/phases/04-renderer-mixer-annotator-beats-extraction/04-PATTERNS.md §"src/musicgen/annotator.py" (signature + TBD-None pattern)
  </read_first>
  <action>
Create `src/musicgen/annotator.py`. Full content:

```python
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
            (dict) from ``musicality_score.get_musicality_score``.
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
        "pre_roll_offset_seconds": None,  # R-P9 Phase 6 calibrate step fills
    }

    # D-16 clarification: analysis_failed is OMITTED on success, only present
    # when explicitly True. Do not emit {"analysis_failed": False}.
    if analysis_failed is True:
        annotation["analysis_failed"] = True

    return annotation
```

Design notes specific to this task:

- The signature extends CONTEXT.md D-14 with three ORCHESTRATOR-PROVIDED params that D-14 did not list but are necessary per RESEARCH Open Questions #2 and #3: `chord_progressions` (per-part, threaded by orchestrator since generate_song_parts discards this today), `midi_paths` (per-part per-layer), `mix_path` (the final concat output). The annotator cannot infer them from SongParams/MixResult alone.
- `_derive_mode` is the simplest possible derivation: trailing `"m"` → minor. Matches CONTEXT.md D-15 "derived from key: minor if key ends in 'm'".
- `_transitions_to_arrangement` accumulates global offsets. Each `MixResult.transitions` records LOCAL offsets; the helper walks the arrangement in order and builds the global view. Defensive — even if the orchestrator passes transitions that already record global offsets, the output here will be internally consistent (successive `end_seconds`/`start_seconds` pairs chain contiguously).
- `analysis_failed` is a tri-state kwarg: `None` (default, omit), `True` (set), `False` (also omit per D-16 clarification). The orchestrator wires Phase 5's failure pathway later; this phase just encodes the contract.
- `pre_roll_offset_seconds: None` is hardcoded since R-P9 (Phase 6) fills it; not exposed as a kwarg this phase.
- No `import random` anywhere (module does not need it).
  </action>
  <verify>
    <automated>python -c "
from musicgen.annotator import annotate, _derive_mode
assert _derive_mode('Am') == 'minor'
assert _derive_mode('A') == 'major'
assert _derive_mode('C#m') == 'minor'
assert _derive_mode('G') == 'major'
assert _derive_mode('F#m') == 'minor'
import inspect
sig = inspect.signature(annotate)
required_params = {'song_params', 'render_results', 'mix_results', 'beat_times', 'downbeat_times', 'musicality', 'chord_progressions', 'midi_paths', 'mix_path', 'fluidsynth_version'}
actual_params = set(sig.parameters.keys())
missing = required_params - actual_params
assert not missing, f'missing annotator params: {missing}'
print('annotator.py smoke OK')
"</automated>
  </verify>
  <acceptance_criteria>
    - File `src/musicgen/annotator.py` exists
    - `grep "^def annotate(" src/musicgen/annotator.py` returns exactly 1 line
    - `grep "from musicgen.sampler import SongParams" src/musicgen/annotator.py` returns at least 1 line
    - `grep "from musicgen.renderer import RenderResult" src/musicgen/annotator.py` returns at least 1 line
    - `grep "from musicgen.mixer import MixResult" src/musicgen/annotator.py` returns at least 1 line
    - `grep -c "import random\|from random" src/musicgen/annotator.py` returns `0` (module does not need RNG)
    - `grep -c "open(\|json.dump\|json.load\|os.makedirs\|os.path.exists" src/musicgen/annotator.py` returns `0` (pure-function / zero-I/O contract — D-14)
    - `grep "D-14\|D-15\|D-16" src/musicgen/annotator.py` returns at least 3 matches (traceability)
    - `grep "\"seed\"\s*:\s*seed\|'seed'\s*:\s*seed" src/musicgen/annotator.py` returns at least 1 match (Phase-5 TBD fields threaded from kwargs)
    - `python -c "import ast; t = ast.parse(open('src/musicgen/annotator.py').read()); hits = [n for n in ast.walk(t) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) and n.func.value.id == 'random' and n.func.attr != 'Random']; assert hits == [], f'bare random.* hits: {[n.lineno for n in hits]}'"` — exits 0
    - Automated smoke above exits 0 (prints `annotator.py smoke OK`)
  </acceptance_criteria>
  <done>`src/musicgen/annotator.py` exists with a single public function `annotate(...)` of the locked signature. All R-P4 fields Phase 4 can fill are populated from the input parameters; all Phase 5 TBD fields are present with value None; `analysis_failed` is tri-state per D-16. Module contains zero I/O, zero random.*, zero imports of `random`.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Populate tests/test_annotator.py with fixture-driven golden-dict tests</name>
  <files>tests/test_annotator.py</files>
  <behavior>
    - `test_annotate_returns_dict` — basic shape: returns a dict
    - `test_phase4_fields_filled` — all Phase-4 fillable fields are non-None after `annotate(...)` returns: key, mode, tempo_bpm, time_signature, time_signatures_per_part, measures_per_part, swing, song_arrangement, chord_progression, active_layers, soundfonts, fx_params, beat_times, downbeat_times, musicality_score, duration_seconds, fluidsynth_version, mix, stems, midi
    - `test_tbd_fields_are_none` — seed, musicgen_version, split, pre_roll_offset_seconds are present and == None
    - `test_analysis_failed_omitted_on_success` — default call (analysis_failed=None) → key NOT in result
    - `test_analysis_failed_present_when_true` — explicit analysis_failed=True → key present and == True
    - `test_analysis_failed_false_also_omitted` — explicit analysis_failed=False → key NOT in result (D-16 clarification)
    - `test_mode_derivation_major` — key="G" → mode=="major"
    - `test_mode_derivation_minor` — key="Am" → mode=="minor"; key="F#m" → minor; key="C#m" → minor
    - `test_song_arrangement_shape` — returned list-of-dicts with keys `part`, `start_seconds`, `end_seconds`; length == len(song_arrangement); start_seconds monotonically non-decreasing
    - `test_duration_seconds_equals_final_end` — duration_seconds == song_arrangement[-1].end_seconds (when arrangement is non-empty)
    - `test_stems_per_part_shape` — stems is {part: {layer: path}} with the 4 canonical layer keys
    - `test_annotate_is_pure_function` — monkeypatch `builtins.open` to raise on call; call `annotate(...)`; no exception raised (pure — does not call open)
    - `test_annotate_does_not_mutate_inputs` — capture input dicts, call annotate, assert inputs unchanged (defensive copies)
    - `test_deterministic_same_inputs_same_output` — two calls with identical fixtures return equal dicts
  </behavior>
  <read_first>
    - tests/test_annotator.py (Wave 0 stub — replaced entirely)
    - tests/test_sampler.py (analog: fixture + frozen-dataclass assertion pattern, lines 44-85 for test class structure)
    - src/musicgen/annotator.py (from Task 1 — API under test)
    - src/musicgen/sampler.py (SongParams construction — used in fixtures)
    - src/musicgen/renderer.py (RenderResult construction — used in fixtures)
    - src/musicgen/mixer.py (MixResult construction — used in fixtures)
    - .planning/phases/04-renderer-mixer-annotator-beats-extraction/04-PATTERNS.md §"tests/test_annotator.py" (fixture + TBD-None assertion template)
  </read_first>
  <action>
Replace `tests/test_annotator.py` entirely (delete Wave 0 stub). File content:

```python
"""Annotator tests (R-X6): fixture-driven pure-function contract with D-15/D-16 semantics.

Builds synthetic SongParams + RenderResult + MixResult + beat/downbeat lists +
stub musicality dict, calls annotate(...), asserts returned dict shape against
the R-P4 schema. Includes a no-I/O contract test (monkeypatch open to raise).
"""
from __future__ import annotations

import builtins
import copy
from unittest.mock import patch

import pytest

from musicgen.annotator import annotate
from musicgen.sampler import SongParams
from musicgen.renderer import RenderResult
from musicgen.mixer import MixResult


# ---------- Fixtures ----------

@pytest.fixture
def minimal_song_params():
    """Single-part 4/4 SongParams for fixture simplicity."""
    return SongParams(
        key="Am",
        tempo=120,
        time_signature_base="4/4",
        time_signature_variation=1.0,
        swing_amount=0.66,
        signatures_per_part={"intro": "4/4"},
        measures_per_part={"intro": 2},
        song_unique_parts=["intro"],
        song_arrangement=["intro"],
    )


@pytest.fixture
def minimal_render_results():
    return {
        "intro": RenderResult(
            stem_paths={
                "beat": "/tmp/intro_beat.wav",
                "melody": "/tmp/intro_melody.wav",
                "harmony": "/tmp/intro_harmony.wav",
                "bassline": "/tmp/intro_bassline.wav",
            },
            sample_rate=44100,
            channels=2,
            duration_seconds=4.0,
            fluidsynth_version="FluidSynth runtime version 2.3.4",
        ),
    }


@pytest.fixture
def minimal_mix_results():
    return {
        "intro": MixResult(
            mix_path="/tmp/intro_mix.wav",
            stem_paths={
                "beat": "/tmp/intro/beat_fx.wav",
                "melody": "/tmp/intro/melody_silent.wav",
                "harmony": "/tmp/intro/harmony_fx.wav",
                "bassline": "/tmp/intro/bassline_fx.wav",
            },
            part_layers={"beat": True, "melody": False, "harmony": True, "bassline": True},
            soundfonts={
                "beat": "/sf/beat/fake.sf2",
                "melody": "/sf/melody/fake.sf2",
                "harmony": "/sf/harmony/fake.sf2",
                "bassline": "/sf/bassline/fake.sf2",
            },
            pedalboards={"beat": [], "melody": [], "harmony": [], "bassline": []},
            transitions=[["intro", 0.0], ["end", 4.0]],
        ),
    }


@pytest.fixture
def minimal_beat_times():
    return {"intro": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]}


@pytest.fixture
def minimal_downbeat_times():
    return {"intro": [0.0, 2.0]}


@pytest.fixture
def minimal_musicality():
    return {"score": 0.75, "components": {"rhythm": 0.8, "melody": 0.7}}


@pytest.fixture
def minimal_chord_progressions():
    return {"intro": ["Am", "F", "C", "G"]}


@pytest.fixture
def minimal_midi_paths():
    return {
        "intro": {
            "beat": "/tmp/intro-beat.mid",
            "melody": "/tmp/intro-melody.mid",
            "harmony": "/tmp/intro-harmony.mid",
            "bassline": "/tmp/intro-bassline.mid",
        },
    }


@pytest.fixture
def annotate_kwargs(
    minimal_song_params, minimal_render_results, minimal_mix_results,
    minimal_beat_times, minimal_downbeat_times, minimal_musicality,
    minimal_chord_progressions, minimal_midi_paths,
):
    return dict(
        song_params=minimal_song_params,
        render_results=minimal_render_results,
        mix_results=minimal_mix_results,
        beat_times=minimal_beat_times,
        downbeat_times=minimal_downbeat_times,
        musicality=minimal_musicality,
        chord_progressions=minimal_chord_progressions,
        midi_paths=minimal_midi_paths,
        mix_path="/tmp/song.wav",
        fluidsynth_version="FluidSynth runtime version 2.3.4",
    )


# ---------- Shape (D-15) ----------

class TestAnnotateShape:
    def test_returns_dict(self, annotate_kwargs):
        result = annotate(**annotate_kwargs)
        assert isinstance(result, dict)

    def test_phase4_fields_filled(self, annotate_kwargs):
        result = annotate(**annotate_kwargs)
        phase4_fields = [
            "key", "mode", "tempo_bpm", "time_signature", "time_signatures_per_part",
            "measures_per_part", "swing", "song_arrangement", "chord_progression",
            "active_layers", "soundfonts", "fx_params", "beat_times", "downbeat_times",
            "musicality_score", "duration_seconds", "fluidsynth_version",
            "mix", "stems", "midi",
        ]
        for field in phase4_fields:
            assert field in result, f"R-P4 field {field!r} missing from annotate output"
            assert result[field] is not None, f"R-P4 field {field!r} is None; Phase 4 must fill it"

    def test_key_and_tempo_threaded(self, annotate_kwargs):
        result = annotate(**annotate_kwargs)
        assert result["key"] == "Am"
        assert result["tempo_bpm"] == 120

    def test_swing_threaded(self, annotate_kwargs):
        result = annotate(**annotate_kwargs)
        assert result["swing"] == 0.66

    def test_time_signature_base(self, annotate_kwargs):
        result = annotate(**annotate_kwargs)
        assert result["time_signature"] == "4/4"
        assert result["time_signatures_per_part"] == {"intro": "4/4"}

    def test_measures_per_part(self, annotate_kwargs):
        result = annotate(**annotate_kwargs)
        assert result["measures_per_part"] == {"intro": 2}

    def test_chord_progression_per_part(self, annotate_kwargs):
        result = annotate(**annotate_kwargs)
        assert result["chord_progression"] == {"intro": ["Am", "F", "C", "G"]}

    def test_soundfonts_from_first_mix_result(self, annotate_kwargs):
        result = annotate(**annotate_kwargs)
        assert set(result["soundfonts"].keys()) == {"beat", "melody", "harmony", "bassline"}

    def test_active_layers_per_part(self, annotate_kwargs):
        result = annotate(**annotate_kwargs)
        assert result["active_layers"] == {"intro": {"beat": True, "melody": False, "harmony": True, "bassline": True}}

    def test_beat_times_per_part(self, annotate_kwargs):
        result = annotate(**annotate_kwargs)
        assert result["beat_times"]["intro"] == [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]

    def test_downbeat_times_per_part(self, annotate_kwargs):
        result = annotate(**annotate_kwargs)
        assert result["downbeat_times"]["intro"] == [0.0, 2.0]

    def test_mix_path_threaded(self, annotate_kwargs):
        result = annotate(**annotate_kwargs)
        assert result["mix"] == "/tmp/song.wav"

    def test_midi_paths_per_part(self, annotate_kwargs):
        result = annotate(**annotate_kwargs)
        assert set(result["midi"]["intro"].keys()) == {"beat", "melody", "harmony", "bassline"}

    def test_fluidsynth_version_threaded(self, annotate_kwargs):
        result = annotate(**annotate_kwargs)
        assert result["fluidsynth_version"] == "FluidSynth runtime version 2.3.4"


# ---------- D-16 None semantics ----------

class TestTbdFieldsAreNone:
    def test_seed_is_none(self, annotate_kwargs):
        result = annotate(**annotate_kwargs)
        assert "seed" in result
        assert result["seed"] is None

    def test_musicgen_version_is_none(self, annotate_kwargs):
        result = annotate(**annotate_kwargs)
        assert "musicgen_version" in result
        assert result["musicgen_version"] is None

    def test_split_is_none(self, annotate_kwargs):
        result = annotate(**annotate_kwargs)
        assert "split" in result
        assert result["split"] is None

    def test_pre_roll_offset_seconds_is_none(self, annotate_kwargs):
        result = annotate(**annotate_kwargs)
        assert "pre_roll_offset_seconds" in result
        assert result["pre_roll_offset_seconds"] is None

    def test_seed_can_be_threaded_via_kwarg(self, annotate_kwargs):
        """Phase 5 will pass seed; the kwarg routing is already wired."""
        result = annotate(**{**annotate_kwargs, "seed": 42})
        assert result["seed"] == 42

    def test_split_can_be_threaded_via_kwarg(self, annotate_kwargs):
        result = annotate(**{**annotate_kwargs, "split": "train"})
        assert result["split"] == "train"


# ---------- analysis_failed (D-16 clarification) ----------

class TestAnalysisFailedKey:
    def test_omitted_on_success_default(self, annotate_kwargs):
        result = annotate(**annotate_kwargs)
        assert "analysis_failed" not in result

    def test_omitted_when_explicit_false(self, annotate_kwargs):
        """D-16 clarification: do NOT emit {"analysis_failed": False}."""
        result = annotate(**{**annotate_kwargs, "analysis_failed": False})
        assert "analysis_failed" not in result

    def test_present_and_true_when_explicit_true(self, annotate_kwargs):
        result = annotate(**{**annotate_kwargs, "analysis_failed": True})
        assert "analysis_failed" in result
        assert result["analysis_failed"] is True


# ---------- Mode derivation ----------

class TestModeDerivation:
    @pytest.mark.parametrize("key,expected_mode", [
        ("A", "major"),
        ("Am", "minor"),
        ("C", "major"),
        ("C#", "major"),
        ("C#m", "minor"),
        ("F#m", "minor"),
        ("G", "major"),
        ("D#m", "minor"),
    ])
    def test_mode_from_key(self, annotate_kwargs, key, expected_mode):
        sp = annotate_kwargs["song_params"]
        new_sp = SongParams(
            key=key, tempo=sp.tempo, time_signature_base=sp.time_signature_base,
            time_signature_variation=sp.time_signature_variation, swing_amount=sp.swing_amount,
            signatures_per_part=sp.signatures_per_part, measures_per_part=sp.measures_per_part,
            song_unique_parts=sp.song_unique_parts, song_arrangement=sp.song_arrangement,
        )
        result = annotate(**{**annotate_kwargs, "song_params": new_sp})
        assert result["mode"] == expected_mode


# ---------- song_arrangement shape ----------

class TestSongArrangement:
    def test_is_list_of_dicts(self, annotate_kwargs):
        result = annotate(**annotate_kwargs)
        sa = result["song_arrangement"]
        assert isinstance(sa, list)
        assert all(isinstance(entry, dict) for entry in sa)

    def test_each_entry_has_part_start_end(self, annotate_kwargs):
        result = annotate(**annotate_kwargs)
        for entry in result["song_arrangement"]:
            assert set(entry.keys()) == {"part", "start_seconds", "end_seconds"}

    def test_length_matches_arrangement(self, annotate_kwargs):
        result = annotate(**annotate_kwargs)
        assert len(result["song_arrangement"]) == len(annotate_kwargs["song_params"].song_arrangement)

    def test_start_seconds_monotonic(self, annotate_kwargs):
        # Build a 3-part arrangement
        sp = annotate_kwargs["song_params"]
        new_sp = SongParams(
            key=sp.key, tempo=sp.tempo, time_signature_base=sp.time_signature_base,
            time_signature_variation=sp.time_signature_variation, swing_amount=sp.swing_amount,
            signatures_per_part={"intro": "4/4", "verse": "4/4", "chorus": "4/4"},
            measures_per_part={"intro": 2, "verse": 4, "chorus": 4},
            song_unique_parts=["intro", "verse", "chorus"],
            song_arrangement=["intro", "verse", "chorus"],
        )
        new_mr = {
            p: MixResult(
                mix_path=f"/tmp/{p}.wav",
                stem_paths={l: f"/tmp/{p}/{l}.wav" for l in ("beat", "melody", "harmony", "bassline")},
                part_layers={l: True for l in ("beat", "melody", "harmony", "bassline")},
                soundfonts={l: f"/sf/{l}/fake.sf2" for l in ("beat", "melody", "harmony", "bassline")},
                pedalboards={l: [] for l in ("beat", "melody", "harmony", "bassline")},
                transitions=[[p, 0.0], ["end", 2.0]],
            )
            for p in ("intro", "verse", "chorus")
        }
        new_rr = {
            p: RenderResult(
                stem_paths={l: f"/tmp/{p}_{l}.wav" for l in ("beat", "melody", "harmony", "bassline")},
                sample_rate=44100, channels=2, duration_seconds=2.0, fluidsynth_version="v",
            )
            for p in ("intro", "verse", "chorus")
        }
        new_bt = {p: [] for p in ("intro", "verse", "chorus")}
        new_dbt = {p: [] for p in ("intro", "verse", "chorus")}
        new_cp = {p: [] for p in ("intro", "verse", "chorus")}
        new_midi = {
            p: {l: f"/tmp/{p}-{l}.mid" for l in ("beat", "melody", "harmony", "bassline")}
            for p in ("intro", "verse", "chorus")
        }
        result = annotate(
            song_params=new_sp, render_results=new_rr, mix_results=new_mr,
            beat_times=new_bt, downbeat_times=new_dbt,
            musicality={"score": 0.5, "components": {}},
            chord_progressions=new_cp, midi_paths=new_midi,
            mix_path="/tmp/full.wav", fluidsynth_version="v",
        )
        starts = [entry["start_seconds"] for entry in result["song_arrangement"]]
        assert starts == sorted(starts), f"start_seconds not monotonic: {starts}"
        # Chained: each end_seconds == next start_seconds
        for i in range(len(result["song_arrangement"]) - 1):
            cur_end = result["song_arrangement"][i]["end_seconds"]
            next_start = result["song_arrangement"][i + 1]["start_seconds"]
            assert cur_end == next_start, f"arrangement gap at index {i}: {cur_end} vs {next_start}"

    def test_duration_equals_final_end(self, annotate_kwargs):
        result = annotate(**annotate_kwargs)
        if result["song_arrangement"]:
            assert result["duration_seconds"] == result["song_arrangement"][-1]["end_seconds"]


# ---------- Purity contract (D-14: zero I/O) ----------

class TestAnnotatorIsPure:
    def test_does_not_call_open(self, annotate_kwargs):
        """D-14: annotate() must not call builtins.open — no file I/O allowed."""
        real_open = builtins.open

        def _fail_open(*args, **kwargs):
            raise AssertionError("annotate() called open() — D-14 purity violation")

        with patch("builtins.open", _fail_open):
            result = annotate(**annotate_kwargs)
            assert isinstance(result, dict)
        # Restoration is automatic via the context manager.

    def test_does_not_mutate_inputs(self, annotate_kwargs):
        """Annotator must not mutate input dicts (defensive copies expected)."""
        # Deep-copy inputs before call
        original_mix_results = copy.deepcopy(annotate_kwargs["mix_results"])
        original_chord_progressions = copy.deepcopy(annotate_kwargs["chord_progressions"])
        original_beat_times = copy.deepcopy(annotate_kwargs["beat_times"])
        annotate(**annotate_kwargs)
        # Verify no mutation
        assert annotate_kwargs["mix_results"] == original_mix_results
        assert annotate_kwargs["chord_progressions"] == original_chord_progressions
        assert annotate_kwargs["beat_times"] == original_beat_times


# ---------- Determinism ----------

class TestAnnotatorDeterminism:
    def test_same_inputs_same_output(self, annotate_kwargs):
        a = annotate(**annotate_kwargs)
        b = annotate(**annotate_kwargs)
        assert a == b


# ---------- stems/midi structure ----------

class TestStemsAndMidi:
    def test_stems_shape(self, annotate_kwargs):
        result = annotate(**annotate_kwargs)
        assert "intro" in result["stems"]
        assert set(result["stems"]["intro"].keys()) == {"beat", "melody", "harmony", "bassline"}

    def test_midi_shape(self, annotate_kwargs):
        result = annotate(**annotate_kwargs)
        assert "intro" in result["midi"]
        assert set(result["midi"]["intro"].keys()) == {"beat", "melody", "harmony", "bassline"}
```

Do NOT retain the Wave 0 module-level skip.
  </action>
  <verify>
    <automated>python -m pytest tests/test_annotator.py -x -q 2>&1 | tail -30</automated>
  </verify>
  <acceptance_criteria>
    - `pytest tests/test_annotator.py -x -q` exits 0 with no skipped tests (Wave 0 stub replaced)
    - `grep -c "def test_" tests/test_annotator.py` >= 25 (multiple test methods across 7+ test classes including parametrized mode cases)
    - `grep "TestAnnotateShape\|TestTbdFieldsAreNone\|TestAnalysisFailedKey\|TestModeDerivation\|TestSongArrangement\|TestAnnotatorIsPure\|TestAnnotatorDeterminism" tests/test_annotator.py` returns at least 7 matches (all test classes present)
    - `grep "analysis_failed" tests/test_annotator.py` returns at least 3 matches (tri-state coverage)
    - `grep "pre_roll_offset_seconds" tests/test_annotator.py` returns at least 1 match (D-16 coverage)
    - `grep "is None\|is not None" tests/test_annotator.py` returns at least 6 matches (None semantics well-tested)
    - Total wall time for `pytest tests/test_annotator.py -q` < 5 seconds (pure function, no I/O, fixtures are in-memory)
    - Full suite regression: `pytest tests/ -m "not slow" -q` exits 0 (no regression)
  </acceptance_criteria>
  <done>`tests/test_annotator.py` contains 7+ test classes with ≥ 25 test cases covering: Phase-4 field fill completeness (D-15), Phase-5 TBD None semantics (D-16), analysis_failed tri-state handling (D-16 clarification), mode derivation parametrized over 8 keys, song_arrangement shape and monotonicity, purity contract (open-panic monkeypatch), input non-mutation, determinism. All tests pass. No regressions.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| Input dicts → output dict | Caller supplies fixture-like data; no external trust boundaries crossed inside this function |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-04-04-01 | Information Disclosure | Annotation dict may surface absolute filesystem paths (mix, stems, midi keys) | accept | Intentional per D-15 — Phase 5 writer rewrites to relative paths at json.dump time; for now the paths are local dev paths with no secrets |
| T-04-04-02 | Tampering | Input dicts mutated silently during annotator call | mitigate | Tests include `test_does_not_mutate_inputs` (deep-copy before, equality after); internal dict construction uses `dict(x)` or `list(x)` comprehensions to defensively copy |
| T-04-04-03 | Repudiation | Annotation dict missing required fields for R-P4 | mitigate | Test `test_phase4_fields_filled` iterates the 20-item field list from D-15 and asserts each is present + non-None |
</threat_model>

<verification>
After all 2 tasks complete:

1. `python -c "from musicgen.annotator import annotate; import inspect; print(list(inspect.signature(annotate).parameters))"` — prints the locked signature parameter list
2. `python -m pytest tests/test_annotator.py -v` — all new tests pass (≥ 25)
3. `python -m pytest tests/ -m "not slow" -q` — full suite green
4. `python -c "import ast; t = ast.parse(open('src/musicgen/annotator.py').read()); ios = [n for n in ast.walk(t) if isinstance(n, ast.Call) and ((isinstance(n.func, ast.Name) and n.func.id == 'open') or (isinstance(n.func, ast.Attribute) and n.func.attr in ('dump', 'load', 'makedirs', 'mkdir')))]; assert ios == []"` — zero I/O calls (D-14)
5. `python -c "import ast; t = ast.parse(open('src/musicgen/annotator.py').read()); hits = [n for n in ast.walk(t) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) and n.func.value.id == 'random' and n.func.attr != 'Random']; assert hits == []"` — zero bare random.*
</verification>

<success_criteria>
- `src/musicgen/annotator.py` exposes a single public function `annotate(...)` (D-14)
- Function is pure: zero I/O, zero random.*, zero subprocess
- Every R-P4 Phase-4-fill field populated non-None (D-15)
- Every Phase-5 TBD field present with value None (D-16)
- `analysis_failed` is tri-state and correctly omitted on success/False (D-16 clarification)
- `tests/test_annotator.py` covers 25+ cases across 7+ test classes
- Full suite green; no regressions
</success_criteria>

<output>
After completion, create `.planning/phases/04-renderer-mixer-annotator-beats-extraction/04-04-SUMMARY.md`.

Include:
- Line count of `src/musicgen/annotator.py`
- Full list of 20 R-P4 fields and their source (SongParams field / MixResult field / caller kwarg / None)
- Confirmation of zero I/O in annotator.py (static AST check output)
- Confirmation of analysis_failed tri-state behavior (test output)
- Test count passing in test_annotator.py
- Full suite run tail
</output>
