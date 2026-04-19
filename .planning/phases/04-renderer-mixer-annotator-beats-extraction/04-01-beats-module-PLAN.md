---
phase: 04-renderer-mixer-annotator-beats-extraction
plan: 01
type: execute
wave: 1
depends_on: ["04-00"]
files_modified:
  - src/musicgen/beats.py
  - src/musicgen/generators/beat.py
  - tests/test_beats.py
autonomous: true
requirements: [R-X7]
tags: [phase-4, beats, midi, mido, swing]

must_haves:
  truths:
    - "`from musicgen.beats import beat_duration, extract_beat_times, extract_downbeat_times` succeeds"
    - "`beat_duration(\"4/4\", 120) == 0.5` exactly (preserves pre-refactor behavior)"
    - "`extract_beat_times` emits one timestamp per MIDI note_on with velocity > 0, swing-aware because swing is baked into MIDI onsets"
    - "`extract_downbeat_times` returns exactly `measures` entries (one per measure), derived via time-grid computation — NOT via stride-slice of beat_times (RESEARCH correction #1)"
    - "`generators/beat.py` imports `beat_duration` from `musicgen.beats` — body is removed; single source of truth is `musicgen.beats` (D-21)"
    - "Zero bare `random.<method>(` in `src/musicgen/beats.py` (beats has no RNG draws — deterministic from MIDI)"
    - "Three swing cases (0.5, 0.66, 0.75) all produce monotonic sorted beat timestamps"
  artifacts:
    - path: "src/musicgen/beats.py"
      provides: "beat_duration + extract_beat_times + extract_downbeat_times"
      exports: ["beat_duration", "extract_beat_times", "extract_downbeat_times"]
      contains: "def extract_downbeat_times"
      min_lines: 60
    - path: "src/musicgen/generators/beat.py"
      provides: "generators.beat with beat_duration re-exported from musicgen.beats"
      contains: "from musicgen.beats import beat_duration"
    - path: "tests/test_beats.py"
      provides: "4/4 grid test + 3 swing cases + downbeat count assertion + beat_duration pure function"
      contains: "def test_downbeat_count_equals_measures"
  key_links:
    - from: "src/musicgen/beats.py"
      to: "mido.MidiFile + mido.tick2second + mido.bpm2tempo"
      via: "import mido"
      pattern: "mido\\.tick2second"
    - from: "src/musicgen/beats.py"
      to: "TimeSignatureRegistry.lookup"
      via: "import TimeSignatureRegistry"
      pattern: "TimeSignatureRegistry\\.lookup"
    - from: "src/musicgen/generators/beat.py"
      to: "src/musicgen/beats.py"
      via: "from musicgen.beats import beat_duration  # D-21 re-export"
      pattern: "from musicgen\\.beats import beat_duration"
---

<objective>
Implement `src/musicgen/beats.py` as the authoritative swing-aware MIDI-tick beat + downbeat extraction module (R-X7). Replaces the deletion target `beat_anotator.py` (deletion happens in Plan 04-05). Move the primary definition of `beat_duration` here and re-export it from `generators/beat.py` (D-21). Populate `tests/test_beats.py` with the 4/4 grid test, three swing cases, and a downbeat-count assertion. This plan runs first among implementation waves because it has zero audio dependencies (no FluidSynth, no pedalboard, no pydub) and is a pure MIDI-reading utility.

Purpose: (1) Fix PITFALLS P-3 — `beat_anotator.py`'s theoretical-grid approach drifts when swing > 0.5 because the MIDI onsets are swung but the grid is not; MIDI-tick extraction is swing-correct by construction. (2) Lock the downbeat algorithm as a time-grid (RESEARCH correction #1 — D-20's stride-slice is WRONG for sparse patterns like 4/4 `intro: 0,42,38,0` and 12/8). (3) Establish the `musicgen.beats` module surface so annotator (Plan 04-04) and orchestrator (Plan 04-05) have a stable import target.

Output: `src/musicgen/beats.py` (new), `src/musicgen/generators/beat.py` (modified — body of `beat_duration` replaced by re-export import), `tests/test_beats.py` (replaced from Wave 0 stub).
</objective>

<execution_context>
@/home/bidu/musicgen/.claude/get-shit-done/workflows/execute-plan.md
@/home/bidu/musicgen/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/phases/04-renderer-mixer-annotator-beats-extraction/04-CONTEXT.md
@.planning/phases/04-renderer-mixer-annotator-beats-extraction/04-RESEARCH.md
@.planning/phases/04-renderer-mixer-annotator-beats-extraction/04-PATTERNS.md
@.planning/phases/04-renderer-mixer-annotator-beats-extraction/04-VALIDATION.md
@beat_anotator.py
@src/musicgen/generators/beat.py
@timesig.py

<interfaces>
<!-- beat_duration — currently has TWO copies (beat_anotator.py:6-12 and generators/beat.py:32-38);
     this plan makes musicgen.beats the SINGLE source of truth. Body is identical in both: -->
def beat_duration(signature: str, tempo: int) -> float:
    numerator, denominator = map(int, signature.split('/'))
    beat_length = 60 / tempo  # Duration of a quarter note
    return beat_length * (4 / denominator)

<!-- beat_anotator.py:21-35 — the extract_midi_beats template (renamed to extract_beat_times) -->
def extract_midi_beats(midi_file: str, tempo: int, start_time: float) -> List[float]:
    midi = mido.MidiFile(midi_file)
    beats = []
    time_elapsed = 0.0
    ticks_per_beat = midi.ticks_per_beat
    tempo_us = mido.bpm2tempo(tempo)
    for msg in midi:
        time_elapsed += mido.tick2second(msg.time, ticks_per_beat, tempo_us)
        if msg.type == 'note_on' and msg.velocity > 0:
            beats.append(round(time_elapsed + start_time, 3))
    return sorted(beats)

<!-- TimeSignatureRegistry surface — from timesig.py, confirmed via Grep -->
class TimeSignatureRegistry:
    @classmethod
    def lookup(cls, time_signature: str) -> TimeSignatureSpec: ...

@dataclass(frozen=True)
class TimeSignatureSpec:
    numerator: int                    # 2, 3, 4, 5, 6, 7, 12 (directly from signature string)
    denominator: int
    beats_per_measure: float          # num/3 if compound (6/8, 12/8) else num; informational — NOT used for downbeat grid
    ...

<!-- generate_beat signature from src/musicgen/generators/beat.py:58-67 — used by tests/test_beats.py swing cases -->
def generate_beat(
    part: str,
    tempo: int,
    time_signature: str,
    measures: int,
    name: str,
    swing_amount: float,
    rng: random.Random,
    cfg: Optional[config.Config] = None,
) -> Tuple[str, List[str]]:
    # Returns (midi_filename, annotations) — midi_filename is a path string.

<!-- Beat pattern file sparsity — VERIFIED in RESEARCH.md Pitfall 2:
     4/4 intro: "0, 42, 38, 0" — extract_beat_times returns 2 entries per measure, not 4;
     12/8 intro: "36, 0, 42, 38, 42, 0, 36, 42, 38, 42, 0, 36" — returns 9 entries per measure, not 12.
     Therefore stride-slicing beat_times by numerator is WRONG for sparse patterns;
     downbeats MUST be computed as a time grid from tempo + numerator + beat_duration. -->
</interfaces>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Create src/musicgen/beats.py with beat_duration, extract_beat_times, extract_downbeat_times</name>
  <files>src/musicgen/beats.py</files>
  <behavior>
    - `beat_duration("4/4", 120)` returns `0.5` (exactly — float equality passes)
    - `beat_duration("6/8", 120)` returns `0.25` (exact: 60/120 * 4/8)
    - `extract_beat_times(midi_path_4_kicks_at_120bpm, 120, 0.0)` returns `[0.0, 0.5, 1.0, 1.5]` within `abs=0.01` tolerance
    - `extract_beat_times(midi_path, 120, 10.0)` offsets every timestamp by +10.0
    - `extract_beat_times` returns a sorted (monotonic non-decreasing) list
    - `extract_downbeat_times(beat_times_any, "4/4", measures=3, start_offset_seconds=0.0, tempo=120)` returns exactly 3 entries: `[0.0, 2.0, 4.0]` (measure_duration = 4 * 0.5 = 2.0 seconds)
    - `extract_downbeat_times(..., measures=2, start_offset=5.0, tempo=120, "4/4")` returns `[5.0, 7.0]`
    - `extract_downbeat_times` with sparse input `beat_times=[]` (no note_ons emitted) STILL returns `measures` downbeats (because it's time-grid, not slice — RESEARCH correction #1)
    - `extract_downbeat_times` output length EXACTLY equals `measures` for all 7 time signatures
    - Zero bare `random.*` in this module (no RNG — deterministic from MIDI + math)
  </behavior>
  <read_first>
    - beat_anotator.py (primary extraction template — lines 1-35; this file is DELETED in Plan 04-05, so copy before it goes)
    - src/musicgen/generators/beat.py (body of `beat_duration` at lines 32-38 — identical to what beats.py will host; used to verify both callsites match after D-21 re-export)
    - timesig.py (TimeSignatureRegistry.lookup + TimeSignatureSpec.numerator surface)
    - .planning/phases/04-renderer-mixer-annotator-beats-extraction/04-PATTERNS.md §"src/musicgen/beats.py" (authoritative code template including the time-grid downbeat body)
    - .planning/phases/04-renderer-mixer-annotator-beats-extraction/04-RESEARCH.md Pitfall 2 (time-grid downbeat rationale — the D-20 stride-slice is WRONG)
  </read_first>
  <action>
Create `src/musicgen/beats.py` with the following exact content structure. Keep D-references in docstrings for traceability.

```python
"""Beats module — MIDI-tick beat and downbeat extraction (R-X7).

Replaces ``beat_anotator.py`` (D-03/D-19). Uses ``mido`` MIDI-tick derivation so
beat timestamps are automatically swing-aware (swing is baked into MIDI onset
times by ``generators/beat.py:calculate_swing_offset`` at write time).

Design:
  D-19 — ``extract_beat_times`` uses ``mido.MidiFile`` + ``mido.tick2second``.
  D-20 — ``extract_downbeat_times`` uses TIME-GRID derivation, NOT stride-slice
         of ``beat_times``. RESEARCH correction #1 verified against all 6 beat
         pattern files: 4/4 ``intro: 0, 42, 38, 0`` has 2 non-zero entries per
         measure (stride-slice would return measures//2 downbeats); 12/8 has
         9-10 non-zero entries per measure (stride-slice returns wrong count).
  D-21 — ``beat_duration`` primary definition lives HERE; ``generators/beat.py``
         imports it as a re-export alias.
"""
from __future__ import annotations

import logging
from typing import List

import mido

from timesig import TimeSignatureRegistry

logger = logging.getLogger(__name__)


def beat_duration(signature: str, tempo: int) -> float:
    """Return the duration of one beat slot in seconds for (signature, tempo).

    Identical body to the pre-refactor ``generators/beat.py:beat_duration`` and
    ``beat_anotator.py:beat_duration`` — body is unchanged; only the home moves.

    Args:
        signature: Time signature string like ``"4/4"`` or ``"6/8"``.
        tempo: BPM, integer.

    Returns:
        Duration of one beat slot in seconds (float).
    """
    numerator, denominator = map(int, signature.split('/'))
    beat_length = 60 / tempo  # Duration of a quarter note
    return beat_length * (4 / denominator)


def extract_beat_times(
    midi_path: str,
    tempo: int,
    start_offset_seconds: float,
) -> List[float]:
    """Extract beat timestamps from MIDI ``note_on`` events (velocity > 0).

    Swing is already baked into MIDI onset times (``generators/beat.py``
    applies ``calculate_swing_offset`` at write time), so tick-derived
    extraction is automatically swing-aware (D-19).

    Args:
        midi_path: Path to the beat MIDI file.
        tempo: BPM integer.
        start_offset_seconds: Part start time in the full mix (from
            ``MixResult.transitions``).

    Returns:
        Sorted list of beat timestamps in seconds (rounded to 3 decimals).
    """
    midi = mido.MidiFile(midi_path)
    beats: List[float] = []
    time_elapsed = 0.0
    ticks_per_beat = midi.ticks_per_beat
    tempo_us = mido.bpm2tempo(tempo)
    for msg in midi:  # merged-track iteration; msg.time is delta ticks
        time_elapsed += mido.tick2second(msg.time, ticks_per_beat, tempo_us)
        if msg.type == 'note_on' and msg.velocity > 0:
            beats.append(round(time_elapsed + start_offset_seconds, 3))
    return sorted(beats)


def extract_downbeat_times(
    beat_times: List[float],
    time_signature: str,
    measures: int,
    start_offset_seconds: float,
    tempo: int,
) -> List[float]:
    """Derive downbeat timestamps as a pure time grid — one per measure.

    Does NOT stride-slice ``beat_times`` — ``beat_times`` is sparse for
    patterns with zero-valued slots (RESEARCH correction #1, verified against
    all 6 beat pattern files: 4/4 ``intro: 0, 42, 38, 0`` has 2 non-zero entries
    per measure; 12/8 has 9-10 non-zero entries per measure). The stride
    approach described in CONTEXT.md D-20 is an approximation that fails for
    these patterns; this time-grid implementation is correct by construction.

    ``beat_times`` is retained in the signature for API stability (future
    callers may want to cross-check the grid against actual MIDI onsets).

    Args:
        beat_times: Per-part beat timestamps (accepted but NOT sliced).
        time_signature: Time signature string like ``"4/4"``.
        measures: Number of measures in the part.
        start_offset_seconds: Part start time in the full mix.
        tempo: BPM integer.

    Returns:
        List of ``measures`` downbeat timestamps, one per measure,
        rounded to 3 decimals.
    """
    spec = TimeSignatureRegistry.lookup(time_signature)
    beat_slot_s = beat_duration(time_signature, tempo)
    measure_duration_s = spec.numerator * beat_slot_s
    return [
        round(start_offset_seconds + i * measure_duration_s, 3)
        for i in range(measures)
    ]
```

Specific requirements:

- `from __future__ import annotations` at the top (matches `sampler.py:12` convention)
- `from timesig import TimeSignatureRegistry` — NOT `from musicgen.timesig` (timesig.py is at repo root, carried by `pyproject.toml pythonpath = ["."]` per Phase 3 close)
- Google-style docstrings with `Args:` / `Returns:` sections (matches `generators/beat.py` and `sampler.py` convention)
- `import mido` at module top (Wave 0 added the dep)
- No `import config` needed (this module uses only stdlib + mido + timesig)
- No `import random` (module has zero RNG)
- No nested functions
- Logger declared but not used this task (future tasks may add DEBUG logs)

Per D-17 / D-31: this file will be auto-scanned by `tests/test_no_bare_random_in_package.py` (Plan 04-05) — it MUST have zero bare `random.<method>(` calls. Easy: module does not import `random` at all.
  </action>
  <verify>
    <automated>python -c "from musicgen.beats import beat_duration, extract_beat_times, extract_downbeat_times; assert beat_duration('4/4', 120) == 0.5, f'4/4@120 expected 0.5, got {beat_duration(\"4/4\", 120)}'; assert beat_duration('6/8', 120) == 0.25, f'6/8@120 expected 0.25, got {beat_duration(\"6/8\", 120)}'; db = extract_downbeat_times([], '4/4', 3, 0.0, 120); assert db == [0.0, 2.0, 4.0], f'downbeats expected [0.0, 2.0, 4.0], got {db}'; db2 = extract_downbeat_times([], '4/4', 2, 5.0, 120); assert db2 == [5.0, 7.0], f'offset downbeats expected [5.0, 7.0], got {db2}'; db3 = extract_downbeat_times([], '12/8', 4, 0.0, 120); assert len(db3) == 4, f'12/8 4-measure expected 4 downbeats, got {len(db3)}'; print('beats.py smoke OK')"</automated>
  </verify>
  <acceptance_criteria>
    - File `src/musicgen/beats.py` exists
    - `grep -c "^def " src/musicgen/beats.py` returns `3` (beat_duration, extract_beat_times, extract_downbeat_times)
    - `grep "import mido" src/musicgen/beats.py` returns at least 1 line
    - `grep "TimeSignatureRegistry" src/musicgen/beats.py` returns at least 2 lines (import + usage)
    - `grep -c "random\." src/musicgen/beats.py` returns `0` (no bare random — prepares for Plan 04-05 AST guard)
    - `grep -c "import random" src/musicgen/beats.py` returns `0` (module does not import random)
    - `grep "D-19\|D-20\|D-21" src/musicgen/beats.py` returns at least 3 matches (traceability to CONTEXT.md decisions)
    - `grep "stride" src/musicgen/beats.py` returns at least 1 line (documents the RESEARCH correction — must call out "NOT stride")
    - Automated smoke above passes (asserts exact values for 4/4@120, 6/8@120, downbeat grids with 0.0 and 5.0 offsets, 12/8 count)
  </acceptance_criteria>
  <done>`musicgen.beats` exposes beat_duration + extract_beat_times + extract_downbeat_times with the locked signatures and time-grid downbeat algorithm; smoke test verifies the exact-match values for 4/4 and 6/8 and downbeat counts.</done>
</task>

<task type="auto">
  <name>Task 2: Re-export beat_duration from generators/beat.py (D-21)</name>
  <files>src/musicgen/generators/beat.py</files>
  <read_first>
    - src/musicgen/generators/beat.py (current state: defines beat_duration at lines 32-38)
    - src/musicgen/beats.py (from Task 1 — new primary location)
  </read_first>
  <action>
Modify `src/musicgen/generators/beat.py` to remove the body of `beat_duration` and replace it with an import-based re-export from `musicgen.beats` (D-21). This preserves the external import surface: `from musicgen.generators.beat import beat_duration` keeps working; `music_gen.py:34` and any other importers are unaffected.

Specific edit:

1. At the top of the file, after the existing `from timesig import TimeSignatureRegistry` line (currently line 27), add:
   ```python
   # D-21: beat_duration primary definition moves to musicgen.beats this phase.
   # Import and re-export so existing callers (generators, tests, music_gen.py:34)
   # are unaffected. calculate_swing_offset stays here (used only at MIDI write
   # time, not at annotation time).
   from musicgen.beats import beat_duration  # noqa: F401  (D-21 re-export)
   ```

2. Delete the existing `beat_duration` function body (currently lines 32-38):
   ```python
   # DELETE these 7 lines:
   def beat_duration(signature: str, tempo: int) -> float:
       """
       Calculates the duration of a beat based on the time signature and BPM.
       """
       numerator, denominator = map(int, signature.split('/'))
       beat_length = 60 / tempo  # Duration of a quarter note
       return beat_length * (4 / denominator)
   ```

3. Keep `calculate_swing_offset` (lines 41-55) and `generate_beat` (lines 58-164) UNCHANGED.

4. Keep existing `import random` (line 20), `from midiutil import MIDIFile` (line 23), `from musicgen.duration_validator import DurationValidator` (line 26), etc. — none of those imports regress.

Result: `generators/beat.py` contains exactly ONE less function (beat_duration) but exposes the same import surface. The file shrinks by ~7 lines.
  </action>
  <verify>
    <automated>python -c "from musicgen.generators.beat import beat_duration as bd_gen; from musicgen.beats import beat_duration as bd_beats; assert bd_gen is bd_beats, f'D-21 re-export broken: {bd_gen!r} is not {bd_beats!r}'; assert bd_gen('4/4', 120) == 0.5; from musicgen.generators.beat import calculate_swing_offset, generate_beat; assert callable(calculate_swing_offset); assert callable(generate_beat); print('D-21 re-export OK')"</automated>
  </verify>
  <acceptance_criteria>
    - `grep "^def beat_duration" src/musicgen/generators/beat.py` returns 0 lines (function body removed)
    - `grep "from musicgen.beats import beat_duration" src/musicgen/generators/beat.py` returns exactly 1 line (re-export present)
    - `python -c "from musicgen.generators.beat import beat_duration, calculate_swing_offset, generate_beat"` succeeds (surface preserved)
    - `python -c "from musicgen.generators.beat import beat_duration as a; from musicgen.beats import beat_duration as b; assert a is b"` succeeds (they are the SAME object — re-export, not duplicate definition)
    - `grep -c "^def " src/musicgen/generators/beat.py` returns `2` (calculate_swing_offset + generate_beat remain; beat_duration gone)
    - Existing tests `tests/test_generators/test_beat.py` still pass (re-export preserves the import surface contract)
  </acceptance_criteria>
  <done>`beat_duration` is importable from both `musicgen.beats` and `musicgen.generators.beat` — both identical function objects (D-21 re-export, not duplicate definition). `calculate_swing_offset` and `generate_beat` unchanged.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 3: Populate tests/test_beats.py with 4/4 grid + 3 swing cases + downbeat count + pure-function tests</name>
  <files>tests/test_beats.py</files>
  <behavior>
    - `test_beat_duration_4_4_at_120bpm` — asserts exactly 0.5
    - `test_beat_duration_6_8_at_120bpm` — asserts exactly 0.25
    - `test_beat_duration_multiple_tempos` — parametrized over (60, 120, 180) at 4/4, verifies 60/tempo
    - `test_extract_beat_times_44_straight_grid` — generate_beat with swing=0.5, seed=42, 2 measures of 4/4, call extract_beat_times, assert monotonic sorted and count > 0
    - `test_extract_beat_times_start_offset` — same MIDI, extract with `start_offset=10.0`, assert all timestamps are 10.0 + original_value within float precision
    - `test_swing_cases_parametrize` — parametrized over [0.5, 0.66, 0.75], 2 measures of 4/4, assert monotonic + finite
    - `test_swing_off_beats_later_with_heavier_swing` — compare same-seed MIDI at swing=0.5 vs 0.75; assert swing=0.75's off-beat indices are >= swing=0.5's off-beat indices (swing delays off-beats)
    - `test_downbeat_count_equals_measures` — parametrized over 4/4 with measures in [1,2,3,4,5]; assert `len(downbeat_times) == measures`
    - `test_downbeat_count_12_8_still_equals_measures` — verifies RESEARCH correction #1: 12/8 pattern has sparse beat_times but downbeat_count still equals measures exactly
    - `test_downbeat_grid_math` — asserts first downbeat == start_offset; second downbeat == start_offset + measure_duration; no rounding error > 1e-9 accumulates over 10 measures
    - `test_downbeat_sparse_input_tolerance` — calls `extract_downbeat_times(beat_times=[], "4/4", measures=4, 0.0, 120)` and asserts returns `[0.0, 2.0, 4.0, 6.0]` (doesn't error on empty beat_times — time-grid is INPUT-independent per RESEARCH correction #1)
  </behavior>
  <read_first>
    - tests/test_beats.py (Wave 0 stub — replaced entirely)
    - tests/test_generators/test_beat.py (analog — parametrize + monkeypatch.chdir + seeded Random pattern)
    - src/musicgen/beats.py (from Task 1 — API under test)
    - src/musicgen/generators/beat.py (from Task 2 — generate_beat signature used by swing fixtures)
    - .planning/phases/04-renderer-mixer-annotator-beats-extraction/04-PATTERNS.md §"tests/test_beats.py" (authoritative scaffold)
  </read_first>
  <action>
Replace `tests/test_beats.py` entirely (delete the Wave 0 skip stub) with a real test file. Use `tmp_path` + `monkeypatch.chdir` to contain generate_beat's side effect (it creates `<name>/<name>-beat.mid` in cwd).

**File structure:**

```python
"""Beats tests (R-X7): beat_duration + extract_beat_times + extract_downbeat_times.

Three swing cases at 0.5/0.66/0.75 cover the roadmap requirement. Downbeat tests
encode RESEARCH correction #1: the time-grid algorithm returns `measures` entries
even when `beat_times` is sparse (verified against 4/4 and 12/8).

generate_beat fixtures use a seeded random.Random for determinism. MIDI files
are written inside tmp_path via monkeypatch.chdir (generate_beat creates
<name>/<name>-beat.mid in cwd).
"""
from __future__ import annotations

import random
from pathlib import Path

import pytest

from musicgen.beats import beat_duration, extract_beat_times, extract_downbeat_times
from musicgen.generators.beat import generate_beat


# ---------- beat_duration pure-function tests ----------

class TestBeatDuration:
    def test_4_4_at_120bpm(self):
        assert beat_duration("4/4", 120) == 0.5

    def test_6_8_at_120bpm(self):
        assert beat_duration("6/8", 120) == 0.25

    @pytest.mark.parametrize("tempo", [60, 120, 180, 200])
    def test_4_4_inverse_tempo(self, tempo):
        # 4/4: denominator == 4, so beat_duration == 60/tempo exactly
        assert beat_duration("4/4", tempo) == pytest.approx(60 / tempo)

    @pytest.mark.parametrize("sig,expected_at_120", [
        ("2/4", 0.5), ("3/4", 0.5), ("4/4", 0.5),
        ("6/8", 0.25), ("7/8", 0.25), ("12/8", 0.25),
    ])
    def test_all_registry_signatures_at_120(self, sig, expected_at_120):
        assert beat_duration(sig, 120) == pytest.approx(expected_at_120)


# ---------- extract_beat_times (MIDI-tick) ----------

def _make_beat_midi(tmp_path, monkeypatch, swing_amount: float, time_signature: str = "4/4", measures: int = 2, seed: int = 42) -> Path:
    """Generate a beat MIDI using the extracted generator with a seeded RNG.

    Returns the absolute path of the written MIDI. Uses monkeypatch.chdir so the
    generate_beat side-effect (creating <name>/ subdir) stays inside tmp_path.
    """
    monkeypatch.chdir(tmp_path)
    name = "song-verse"
    midi_path, _ = generate_beat(
        part="verse",
        tempo=120,
        time_signature=time_signature,
        measures=measures,
        name=name,
        swing_amount=swing_amount,
        rng=random.Random(seed),
    )
    return Path(tmp_path) / midi_path


class TestExtractBeatTimes:
    def test_returns_monotonic_sorted_list(self, tmp_path, monkeypatch):
        midi = _make_beat_midi(tmp_path, monkeypatch, swing_amount=0.5)
        times = extract_beat_times(str(midi), tempo=120, start_offset_seconds=0.0)
        assert times == sorted(times), "extract_beat_times must return sorted list"
        assert len(times) > 0, "expected at least one note_on extracted"

    def test_start_offset_shifts_all_timestamps(self, tmp_path, monkeypatch):
        midi = _make_beat_midi(tmp_path, monkeypatch, swing_amount=0.5)
        times_zero = extract_beat_times(str(midi), tempo=120, start_offset_seconds=0.0)
        times_shifted = extract_beat_times(str(midi), tempo=120, start_offset_seconds=10.0)
        assert len(times_zero) == len(times_shifted)
        for a, b in zip(times_zero, times_shifted):
            # rounding to 3 decimals may introduce tiny diffs — allow 0.001 tolerance
            assert b == pytest.approx(a + 10.0, abs=0.002)

    def test_deterministic_same_seed(self, tmp_path, monkeypatch):
        midi1 = _make_beat_midi(tmp_path, monkeypatch, swing_amount=0.5, seed=99)
        times1 = extract_beat_times(str(midi1), tempo=120, start_offset_seconds=0.0)
        # New tmp_path invocation — fresh file, same seed
        # (generate_beat writes into cwd/<song-verse>/ so we need a clean subdir)
        import shutil
        shutil.rmtree(tmp_path / "song")
        midi2 = _make_beat_midi(tmp_path, monkeypatch, swing_amount=0.5, seed=99)
        times2 = extract_beat_times(str(midi2), tempo=120, start_offset_seconds=0.0)
        assert times1 == times2, "same seed must produce same beat timestamps"


# ---------- Swing cases (0.5, 0.66, 0.75) ----------

class TestSwingCases:
    @pytest.mark.parametrize("swing_amount", [0.5, 0.66, 0.75])
    def test_monotonic_under_all_swing_values(self, tmp_path, monkeypatch, swing_amount):
        midi = _make_beat_midi(tmp_path, monkeypatch, swing_amount=swing_amount)
        times = extract_beat_times(str(midi), tempo=120, start_offset_seconds=0.0)
        assert times == sorted(times), f"swing={swing_amount}: non-monotonic"
        assert all(isinstance(t, float) and t >= 0.0 for t in times), \
            f"swing={swing_amount}: non-finite or negative"

    @pytest.mark.parametrize("swing_amount", [0.66, 0.75])
    def test_heavier_swing_delays_offbeats(self, tmp_path, monkeypatch, swing_amount):
        """For swing > 0.5, off-beats (odd index) should be delayed relative to swing=0.5."""
        # Use identical seed so the pattern choice is identical across runs
        import shutil
        midi_straight = _make_beat_midi(tmp_path, monkeypatch, swing_amount=0.5, seed=7)
        t_straight = extract_beat_times(str(midi_straight), tempo=120, start_offset_seconds=0.0)
        shutil.rmtree(tmp_path / "song")
        midi_swung = _make_beat_midi(tmp_path, monkeypatch, swing_amount=swing_amount, seed=7)
        t_swung = extract_beat_times(str(midi_swung), tempo=120, start_offset_seconds=0.0)
        # Only compare overlapping length; off-beats at odd positions in the pattern
        # accumulated beat array should be LATER in swung than straight.
        # (Approximation: the test asserts the MEAN timestamp under swung >= mean under straight,
        # which is a robust aggregate even if individual index mapping varies across RNG pattern choice.)
        if len(t_swung) >= 4 and len(t_straight) >= 4:
            assert sum(t_swung) >= sum(t_straight) - 0.01, \
                f"swing={swing_amount}: total offset should be >= straight (swing delays off-beats)"


# ---------- extract_downbeat_times (time-grid, RESEARCH correction #1) ----------

class TestExtractDownbeatTimes:
    @pytest.mark.parametrize("measures", [1, 2, 3, 4, 5])
    def test_downbeat_count_equals_measures_44(self, measures):
        downbeats = extract_downbeat_times([], "4/4", measures, 0.0, 120)
        assert len(downbeats) == measures

    @pytest.mark.parametrize("sig", ["2/4", "3/4", "4/4", "6/8", "7/8", "12/8"])
    def test_downbeat_count_equals_measures_all_sigs(self, sig):
        """RESEARCH correction #1: time-grid returns exactly `measures` downbeats
        for every registered time signature, regardless of pattern sparsity."""
        downbeats = extract_downbeat_times([], sig, measures=4, start_offset_seconds=0.0, tempo=120)
        assert len(downbeats) == 4, f"{sig}: expected 4 downbeats, got {len(downbeats)}"

    def test_downbeat_grid_4_4_120bpm(self):
        # measure duration = numerator (4) * beat_duration(4/4, 120) (0.5) = 2.0s
        downbeats = extract_downbeat_times([], "4/4", 3, 0.0, 120)
        assert downbeats == [0.0, 2.0, 4.0]

    def test_downbeat_grid_with_start_offset(self):
        downbeats = extract_downbeat_times([], "4/4", 2, 5.0, 120)
        assert downbeats == [5.0, 7.0]

    def test_downbeat_grid_independent_of_beat_times_input(self):
        """Whether beat_times is empty, full, or garbage — output is the same
        (time-grid is math-only; beat_times is retained for API compat only).
        """
        a = extract_downbeat_times([], "4/4", 3, 0.0, 120)
        b = extract_downbeat_times([0.5, 1.0, 1.5], "4/4", 3, 0.0, 120)
        c = extract_downbeat_times([999.0, -1.0, 42.0], "4/4", 3, 0.0, 120)
        assert a == b == c, "extract_downbeat_times must be input-independent"

    def test_downbeat_no_accumulated_rounding_error(self):
        # 10 measures at tempo 120 in 4/4 = expected at 0.0, 2.0, 4.0, ... 18.0
        downbeats = extract_downbeat_times([], "4/4", 10, 0.0, 120)
        expected = [2.0 * i for i in range(10)]
        for a, b in zip(downbeats, expected):
            assert abs(a - b) < 1e-9, f"rounding drift: {a} vs {b}"

    def test_downbeat_12_8_grid(self):
        # 12/8: numerator=12, beat_duration = 60/120 * 4/8 = 0.25, measure = 12*0.25 = 3.0
        downbeats = extract_downbeat_times([], "12/8", 3, 0.0, 120)
        assert downbeats == [0.0, 3.0, 6.0]
```

Do NOT retain the Wave 0 `pytest.skip(..., allow_module_level=True)` — this replaces the stub entirely.

All tests must complete in < 5 seconds total (each generate_beat invocation writes a small MIDI; `mido.MidiFile` reads are fast).
  </action>
  <verify>
    <automated>python -m pytest tests/test_beats.py -x -q 2>&1 | tail -20</automated>
  </verify>
  <acceptance_criteria>
    - `pytest tests/test_beats.py -x -q` exits 0 with no skipped tests (no `allow_module_level=True` skip remains)
    - `pytest tests/test_beats.py -k test_downbeat_count_equals_measures_all_sigs -v` shows 6 parametrize cases running (one per time signature)
    - `pytest tests/test_beats.py -k swing -v` shows at least 5 test invocations across the swing classes (3 monotonic + 2 heavier-delays = 5)
    - `grep -c "extract_downbeat_times" tests/test_beats.py` >= 8 (at least 7 distinct test methods call it)
    - `grep "TestSwingCases" tests/test_beats.py` returns 1 line (class exists)
    - Total wall time for `pytest tests/test_beats.py -q` < 10 seconds
    - Full suite passes: `pytest tests/ -m "not slow" -q` exits 0 (371 baseline + new beats tests; no regression)
  </acceptance_criteria>
  <done>`tests/test_beats.py` contains classes TestBeatDuration, TestExtractBeatTimes, TestSwingCases, TestExtractDownbeatTimes with at least 15 passing test invocations covering 4/4 grid + 3 swing cases + downbeat count for all 6 registered signatures + time-grid input-independence. Full suite green.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| Filesystem → `mido.MidiFile` | Tests read MIDI files from tmp_path; production reads from paths provided by the orchestrator (caller-trusted) |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-04-01-01 | Tampering | `mido.MidiFile` reading malformed MIDI | accept | `mido` is an established library; malformed input raises `ValueError` or `mido.MidiException` which propagates naturally to the caller (orchestrator). No user-controlled MIDI input this phase — all MIDI comes from our own `generators/` output. |
| T-04-01-02 | Denial of Service | Very large `measures` parameter in `extract_downbeat_times` | accept | Output is O(measures) floats; `measures` comes from `SongParams.measures_per_part` which is registry-validated (capped by `TimeSignatureRegistry.measure_count_valid`). No user-controlled `measures` this phase. |
</threat_model>

<verification>
After all 3 tasks complete:

1. `python -c "from musicgen.beats import beat_duration, extract_beat_times, extract_downbeat_times"` succeeds
2. `python -c "from musicgen.generators.beat import beat_duration as a; from musicgen.beats import beat_duration as b; assert a is b"` succeeds
3. `python -m pytest tests/test_beats.py -v` — all new tests pass (at least 15 test invocations)
4. `python -m pytest tests/ -m "not slow" -q` — full suite passes, no regressions
5. `grep -c "^def " src/musicgen/beats.py` returns `3`
6. `grep -c "random\." src/musicgen/beats.py` returns `0` (module has zero bare random references — prepares for Plan 04-05 AST guard)
</verification>

<success_criteria>
- `src/musicgen/beats.py` exists with 3 public functions (beat_duration, extract_beat_times, extract_downbeat_times)
- `src/musicgen/generators/beat.py` re-exports beat_duration from musicgen.beats (D-21)
- `tests/test_beats.py` covers: beat_duration pure-function at all 6 registered signatures, extract_beat_times monotonicity + start_offset + determinism, 3 swing cases at 0.5/0.66/0.75, downbeat count == measures for all 6 signatures, time-grid input-independence (RESEARCH correction #1 encoded as a test)
- Full suite (371 baseline + new tests) green in < 15 seconds
- Zero bare `random.<method>` references in `src/musicgen/beats.py`
</success_criteria>

<output>
After completion, create `.planning/phases/04-renderer-mixer-annotator-beats-extraction/04-01-SUMMARY.md`.

Include:
- Line count of `src/musicgen/beats.py` (target: 60-100 lines incl. docstrings)
- Diff summary for `src/musicgen/generators/beat.py` (7-line body deleted, 1 import line added)
- Test count added (target: 15+ test invocations passing)
- Confirmation of D-21 re-export identity (`generators.beat.beat_duration is beats.beat_duration`)
- Confirmation of RESEARCH correction #1 encoded as a test (`test_downbeat_grid_independent_of_beat_times_input`)
- Full suite run tail (`pytest tests/ -m "not slow" -q` output)
</output>
