# Phase 3: Package skeleton + sampler + generators extraction — Pattern Map

**Mapped:** 2026-04-18
**Files analyzed:** 22 new/modified/deleted
**Analogs found:** 19 / 22 (3 have no in-repo analog — excerpts come from RESEARCH.md templates)

This file enumerates every file Phase 3 touches, classifies it, assigns the closest existing analog in the repository, and extracts the concrete code excerpts that downstream planners/executors should lift verbatim into task templates.

## File → Analog Map

| # | File | Change | Role | Closest Analog | Analog Path | Match |
|---|------|--------|------|----------------|-------------|-------|
| 1 | `pyproject.toml` | CREATE | build-config | *(none — template in RESEARCH.md §Technical Approach)* | — | none |
| 2 | `src/musicgen/__init__.py` | CREATE | package-marker | *(none — stub in RESEARCH.md §CLI Scaffolding)* | — | none |
| 3 | `src/musicgen/__main__.py` | CREATE | entry-point | *(none — 2-line stub in RESEARCH.md)* | — | none |
| 4 | `src/musicgen/cli.py` | CREATE | CLI stub (typer) | `config.py` (module docstring + module logger style) | `/home/bidu/musicgen/config.py` | style-only |
| 5 | `src/musicgen/sampler.py` | CREATE | package-module (pure fns + frozen dataclass) | `timesig.py` + sampler blocks of `music_gen.py` | `/home/bidu/musicgen/timesig.py`, `/home/bidu/musicgen/music_gen.py:508-548, 564-589, 808-847, 849-880` | exact (style) + exact (content) |
| 6 | `src/musicgen/generators/__init__.py` | CREATE | package-marker | `config.py` (module docstring style) | — | style-only |
| 7 | `src/musicgen/generators/chord.py` | CREATE | generator (pure fn, music21 + MIDIFile) | `music_gen.py:70-145` (`generate_chord_progression`) | `/home/bidu/musicgen/music_gen.py:70-145` | exact (content) |
| 8 | `src/musicgen/generators/melody.py` | CREATE | generator (pure fn, music21 + MIDIFile + markov) | `music_gen.py:147-255` (`generate_melody`) | `/home/bidu/musicgen/music_gen.py:147-255` | exact (content) |
| 9 | `src/musicgen/generators/bassline.py` | CREATE | generator (pure fn, music21 + MIDIFile + markov) | `music_gen.py:257-367` (`generate_bassline`) | `/home/bidu/musicgen/music_gen.py:257-367` | exact (content) |
| 10 | `src/musicgen/generators/beat.py` | CREATE | generator (pure fn, MIDIFile + swing helpers) | `music_gen.py:369-495` (`beat_duration`, `calculate_swing_offset`, `generate_beat`) | `/home/bidu/musicgen/music_gen.py:369-495` | exact (content) |
| 11 | `src/musicgen/duration_validator.py` | MOVE (`git mv`) | package-module (class-based validator) | `enhanced_duration_validator.py` (identical — pure rename) | `/home/bidu/musicgen/enhanced_duration_validator.py` | identical |
| 12 | `music_gen.py` | MODIFY (shim) | orchestration-shim (re-exports + orchestrators) | current `music_gen.py` top-of-file (lines 1-22) + `generate_song` (1002-1042) | `/home/bidu/musicgen/music_gen.py:1-22, 1002-1042` | self-reference |
| 13 | `tests/test_sampler.py` | CREATE | test (seeded RNG + AST static) | `tests/test_timesig_registry.py` + `tests/test_music_gen_logging.py` | `/home/bidu/musicgen/tests/test_timesig_registry.py`, `/home/bidu/musicgen/tests/test_music_gen_logging.py` | exact |
| 14 | `tests/test_generators/__init__.py` | CREATE | test-package-marker | `tests/__init__.py` | `/home/bidu/musicgen/tests/__init__.py` | identical |
| 15 | `tests/test_generators/test_chord.py` | CREATE | test (seeded RNG + MIDI byte-equal) | `tests/test_timesig_registry.py` (fixture + parametrize) + `tests/test_duration_validator.py` (fixture) | same | exact |
| 16 | `tests/test_generators/test_melody.py` | CREATE | test (seeded RNG + MIDI byte-equal) | same as #15 | same | exact |
| 17 | `tests/test_generators/test_bassline.py` | CREATE | test (seeded RNG + MIDI byte-equal) | same as #15 | same | exact |
| 18 | `tests/test_generators/test_beat.py` | CREATE | test (seeded RNG + MIDI byte-equal + annotations list) | same as #15 | same | exact |
| 19 | `tests/test_music21_isolation.py` | CREATE | test (regression guard, no RNG) | `tests/test_duration_validator.py` (pure unit-test shape) | `/home/bidu/musicgen/tests/test_duration_validator.py` | role-match |
| 20 | `tests/test_package_install.py` *(optional)* | CREATE | test (subprocess/venv smoke) | `tests/test_music_gen_logging.py` (subprocess-adjacent AST scan) | `/home/bidu/musicgen/tests/test_music_gen_logging.py` | role-match |
| 21 | `tests/conftest.py` | **DELETE** | *(replaced by)* `pyproject.toml` + `[tool.pytest.ini_options] pythonpath = ["."]` | `/home/bidu/musicgen/tests/conftest.py` (for reference) | replacement |
| 22 | `requirements.txt`, `dev-requirements.txt` | **DELETE** | *(replaced by)* `pyproject.toml` `[project].dependencies` + `[project.optional-dependencies].dev` | — | replacement |
| +1 | `tests/test_duration_validator.py` | MODIFY (1 line) | test import update | itself (current `from enhanced_duration_validator import ...`) | `/home/bidu/musicgen/tests/test_duration_validator.py:10` | self-reference |

Match legend:
- **exact** — analog serves the same role AND same data flow; copy structure verbatim and adapt names.
- **role-match** — analog serves the same role but a different data flow; copy structural pattern, adjust inner logic.
- **style-only** — analog shares docstring/import/logger conventions; borrow shape only.
- **identical** — file literally moves or is re-created at a new path with zero logic change.
- **self-reference** — "copy from the current in-place code" (extraction source; not a different analog).
- **replacement** — old file is removed and its responsibility reassigned to a different file.
- **none** — no in-repo analog exists; use the RESEARCH.md template.

---

## Analog Excerpts — Patterns to Copy

Each pattern below has a **source** (exact file + line numbers) and a list of **applied to** files. Planners/executors copy the source verbatim and rename identifiers per the target file.

### Pattern A — Module-level module shape (docstring, imports, module logger)

**Applied to:** `src/musicgen/sampler.py`, `src/musicgen/generators/{chord,melody,bassline,beat}.py`, `src/musicgen/duration_validator.py` (post-move), `src/musicgen/cli.py`.

**Source:** `/home/bidu/musicgen/timesig.py:1-21`

```python
"""Time-signature registry for musicgen (R-S6).

Single source of truth for all time-signature metadata. Adding a new
time signature requires editing ONLY this file — add one entry to
TimeSignatureRegistry.REGISTRY per D-05.

Design decisions:
  D-04: Registry entries are frozen dataclasses.
  D-05: Registry owns ALL validation logic.
  D-06: Designed for flexibility — compound vs simple, unusual meters.

Note: Layer-specific duration sets (chord/melody/bass/beat) remain in
DurationValidator for Phase 2. Phase 3 generator extraction can
reconsider whether to hoist those into the registry.
"""
import logging
import random
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Tuple

logger = logging.getLogger(__name__)
```

**Equivalent from `config.py:1-15`** (same convention — multi-paragraph module docstring referencing decisions, then grouped imports, then `logger`):

```python
"""Config module for musicgen — owns all filesystem paths and override layers.

D-01/D-02: three-layer precedence — CLI args > env vars > hardcoded defaults.
D-03: wraps existing JSON files; no new config file format.
D-09: soundfont pool detection fires at config load time (informational only).

Phase 2 (R-S5, R-S9). Phase 6 will populate `cli_overrides` from the typer CLI.
"""
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)
```

Rules to carry over (for all extracted Phase 3 modules):
- Module docstring states **purpose + referenced decision IDs** (D-xx / R-xx).
- `import` blocks grouped: stdlib → third-party → first-party (music_gen.py style at lines 1-20 shows `midiutil/music21/pydub/...` before `import config` / `from timesig import ...`).
- `typing` imports use `Dict`, `List`, `Optional`, `Tuple` — **not** PEP 604 pipe syntax (Python 3.9 compat per D-13, even if Risk #1 bumps floor to 3.10).
- Module-level `logger = logging.getLogger(__name__)` directly after imports.
- No `logging.basicConfig` at module scope (enforced by `tests/test_music_gen_logging.py::test_basic_config_only_in_main_guard`).

---

### Pattern B — Frozen dataclass with methods and classmethods

**Applied to:** `SongParams` in `src/musicgen/sampler.py`.

**Source:** `/home/bidu/musicgen/timesig.py:24-89`

```python
@dataclass(frozen=True)
class TimeSignatureSpec:
    """Immutable specification for a single time signature.

    All validation methods delegate to these fields so that the registry
    is the single source of truth. Do not mutate the note_durations dict
    (frozen=True blocks setattr, not dict content mutation).
    """
    name: str                                    # "4/4"
    numerator: int
    denominator: int
    is_compound: bool
    valid_chord_pattern_lengths: FrozenSet[int]
    beat_pattern_length: int
    measure_multiplier: float
    # ... (field comments describe provenance / invariants)

    def verify_chord_pattern_length(self, length: int) -> bool:
        """Returns True if the chord pattern length is valid for this time signature."""
        if not self.valid_chord_pattern_lengths:
            return True
        return length in self.valid_chord_pattern_lengths

    @property
    def melody_duration_candidates(self) -> Tuple[float, ...]:
        """Return melody duration candidates as a tuple."""
        return self.melody_durations
```

Rules:
- `@dataclass(frozen=True)` — **NOT** `slots=True` (CONTEXT.md Claude Discretion: 3.9 compat).
- Field comments right of the type annotation explain provenance/invariants (e.g. which legacy dict each field replaces).
- Short-inline instance methods use the fields to answer predicate questions (`verify_*`, `measure_count_valid`).
- `@property` used for accessor that returns a stable transformed view.

---

### Pattern C — Classmethod builder with layered precedence

**Applied to:** `SongParams.sample(rng, cfg, *, time_signature_variation=1.0)` in `src/musicgen/sampler.py`.

**Source A:** `/home/bidu/musicgen/timesig.py:271-291` — classmethod lookups + optional-rng sampler

```python
@classmethod
def lookup(cls, time_signature: str) -> TimeSignatureSpec:
    """Look up a time signature spec. Raises KeyError if not found."""
    return cls.REGISTRY[time_signature]

@classmethod
def all_signatures(cls) -> List[str]:
    """Return all registered time signature strings."""
    return list(cls.REGISTRY.keys())

@classmethod
def sample_random(cls, rng: Optional[random.Random] = None) -> str:
    """Weighted random selection. Replaces generate_random_time_signature threshold-loop.

    Uses random.choices — always returns a value (fixes Pitfall 5 missing-return bug).
    rng parameter allows injected Random for Phase 5 seed discipline.
    """
    sigs = cls.all_signatures()
    weights = [cls.REGISTRY[s].sampling_weight for s in sigs]
    chooser = rng.choices if rng else random.choices
    return chooser(sigs, weights=weights, k=1)[0]
```

Lift directly: the `chooser = rng.choices if rng else random.choices` shape is the D-09 contract the sampler preserves. `SongParams.sample` **requires** `rng` (no Optional), but `cfg: Optional[config.Config] = None` with internal fallback matches the `generate_beat` convention.

**Source B:** `/home/bidu/musicgen/config.py:68-100` — `Config.load` layered-precedence builder

```python
@classmethod
def load(cls, cli_overrides: Optional[Dict[str, object]] = None) -> "Config":
    """Load Config applying D-01/D-02 precedence: CLI > env > defaults.

    Phase 2 callers pass cli_overrides=None; only env + defaults apply.
    Phase 6 typer CLI will build a dict from parsed args and pass it here:
        cfg = Config.load(cli_overrides={"sf_dir": args.sf_dir, ...})

    Fires the D-09 soundfont pool report before returning.
    """
    cfg = cls()

    # env-var layer (D-02 middle layer)
    sf_env = os.environ.get("MUSICGEN_SF_DIR")
    if sf_env:
        cfg.sf_dir = os.path.abspath(sf_env)
    root_env = os.environ.get("MUSICGEN_PROJECT_ROOT")
    if root_env:
        cfg.project_root = os.path.abspath(root_env)

    # cli layer (D-02 top layer; framework-agnostic — avoids typer dep in Phase 2)
    if cli_overrides:
        for key, value in cli_overrides.items():
            if value is None:
                continue
            if not hasattr(cfg, key):
                continue
            if isinstance(value, str) and key in ("sf_dir", "project_root"):
                value = os.path.abspath(value)
            setattr(cfg, key, value)

    cfg._emit_soundfont_pool_report()
    return cfg
```

Rules for `SongParams.sample`:
- Docstring cites the exact decision (D-21) and enumerates the RNG-draw order as a numbered list (see RESEARCH.md §Sampler Extraction lines 289-307).
- Implementation fully constructs the instance inside the method; returns `cls(...)` at the end — frozen dataclass means no attribute mutation post-construction.
- Keep the `while True:` retry loop semantics of `generate_song_measures + validate_measures` exactly as the orchestration in `music_gen.py:1020-1024` — do NOT short-circuit it (Risk #2).

---

### Pattern D — Class-based module with injected logger, typed methods, Google-style docstrings

**Applied to:** `src/musicgen/duration_validator.py` (file is a literal `git mv`, shape unchanged — included here so planners confirm the shape survives the move).

**Source:** `/home/bidu/musicgen/enhanced_duration_validator.py:34-77`

```python
class DurationValidator:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._time_signature_cache: Dict[str, TimeSignatureInfo] = {}

    def _analyze_time_signature(self, time_signature: str) -> TimeSignatureInfo:
        """Analyzes a time signature by delegating to TimeSignatureRegistry.
        Returns legacy TimeSignatureInfo shape for compatibility with existing methods."""
        if time_signature in self._time_signature_cache:
            return self._time_signature_cache[time_signature]

        from timesig import TimeSignatureRegistry   # LOCAL import avoids circular
        spec = TimeSignatureRegistry.lookup(time_signature)
        info = TimeSignatureInfo(
            numerator=spec.numerator,
            denominator=spec.denominator,
            is_compound=spec.is_compound,
            valid_durations=set(spec.valid_durations),
            min_duration=spec.min_duration,
            max_duration=spec.max_duration,
            primary_division=spec.primary_division,
            beats_per_measure=spec.beats_per_measure,
        )
        self._time_signature_cache[time_signature] = info
        return info

    def get_valid_duration(self, duration: float, time_signature: str,
                          remaining_beats: float, layer_type: str) -> float:
        """
        Retorna uma duração válida para o contexto específico.

        Args:
            duration: Duração proposta
            time_signature: Assinatura de tempo
            remaining_beats: Beats restantes no compasso
            layer_type: Tipo de camada ('melody', 'chord', 'bass', 'beat')
        """
```

Notes:
- The `from timesig import TimeSignatureRegistry` **local import inside `_analyze_time_signature`** (line 45) is intentional — it avoids a circular import at module load. After the move to `src/musicgen/duration_validator.py`, this line stays IDENTICAL (per CONTEXT.md D-10 and RESEARCH.md §Duration Validator Relocation step 2).
- `self.logger = logging.getLogger(__name__)` is the **legacy instance-level** logger pattern. Other Phase-2/3 modules use module-level `logger`; CONTEXT.md D-10 rules this is a pure rename → DO NOT refactor the logger pattern in this phase.

---

### Pattern E — Generator function (music21 + MIDIFile + DurationValidator + file I/O)

**Applied to:** `src/musicgen/generators/chord.py`, `melody.py`, `bassline.py`.

**Source (structural reference):** `/home/bidu/musicgen/music_gen.py:70-145` (`generate_chord_progression`)

```python
def generate_chord_progression(key, tempo, time_signature, measures, name, part, pattern_file):
    """
    Generate a chord progression based on key, tempo, a pattern file and time signature.
    """
    validator = DurationValidator()
    mf = MIDIFile(1)
    track = 0
    time = 0

    # MIDI initial setup
    mf.addTrackName(track, time, "Chord Progression")
    mf.addTempo(track, time, tempo)
    numerator, midi_denominator = get_midi_time_signature_values(time_signature)
    mf.addTimeSignature(track, time, numerator, midi_denominator, 24, 8)

    beats_per_measure = numerator
    if time_signature.endswith('8'):
        if numerator % 3 == 0:
            beats_per_measure = numerator // 3

    # Reads and validates chord patterns
    chord_patterns = {}
    with open(pattern_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                part_name, pattern = line.split(':')
                pattern_chords = pattern.split(',')
                if verify_pattern_for_time_signature(pattern_chords, time_signature):
                    chord_patterns.setdefault(part_name, []).append(pattern_chords)

    # default pattern if not found
    if part not in chord_patterns or not chord_patterns[part]:
        base_pattern = ['I'] if numerator in [2, 3] else ['I', 'IV', 'V', 'vi']
        chord_patterns[part] = [base_pattern]

    # chooses and applies a pattern  ← BARE-RANDOM SITE #1 (music_gen.py:107)
    chord_pattern = random.choice(chord_patterns[part])        # → rng.choice(...)
    base_duration = validator.get_suggested_duration(time_signature, 'chord')
    chord_duration = validator.get_valid_duration(
        base_duration, time_signature,
        validator._analyze_time_signature(time_signature).beats_per_measure,
        'chord'
    )

    if chord_duration <= 0:
        raise ValueError(f"Invalid chord duration calculated for time signature {time_signature}")

    chord_progression = []
    for chord_symbol in chord_pattern:
        chord = roman.RomanNumeral(chord_symbol.strip(), key)
        chord_progression.append(chord)

    current_time = 0
    for _ in range(measures):
        for chord in chord_progression:
            for note in chord.pitches:
                mf.addNote(track, 0, note.midi, current_time, chord_duration, 100)
            current_time += chord_duration

    directory = name.split('-')[0]
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = os.path.join(directory, name + "-chord_progression.mid")

    with open(filename, 'wb') as outf:
        mf.writeFile(outf)

    logger.debug("Chord progression: %s", chord_progression)
    return chord_pattern, filename
```

Phase-3 rewrite rules applied to this pattern:
1. Add `rng: random.Random` as the final positional argument.
2. Rewrite `random.choice(chord_patterns[part])` → `rng.choice(chord_patterns[part])`.
3. Replace `from enhanced_duration_validator import DurationValidator` → `from musicgen.duration_validator import DurationValidator`.
4. Replace `get_midi_time_signature_values(time_signature)` with inline `TimeSignatureRegistry.lookup(time_signature)` + `spec.numerator, spec.midi_denominator_power` (D-06: no double indirection through music_gen shim wrappers).
5. Replace `verify_pattern_for_time_signature(pattern_chords, time_signature)` with direct `TimeSignatureRegistry.lookup(time_signature).verify_chord_pattern_length(len(pattern_chords))`.
6. Keep `logger = logging.getLogger(__name__)` module-level (from Pattern A). `logger.debug("Chord progression: %s", ...)` remains unchanged — percent-style args, not f-strings (enforced by `tests/test_music_gen_logging.py::test_no_fstring_in_logger_calls`).
7. Add the `# music21 global-random audit (Phase 3, D-23): ...` comment block (RESEARCH.md §music21 Global RNG Audit) above the first `roman.RomanNumeral(...)` call in `melody.py` and `bassline.py` (chord.py already flagged in research — add there too for consistency).

The **same structural shape** applies to melody (source `music_gen.py:147-255`, 4 bare-random sites at lines 210, 214-217, 222, 242) and bassline (source `music_gen.py:257-367`, 4 sites at 309, 319-322, 338, 347). Each imports music21 objects narrowly (`from music21 import roman, scale, pitch`) — that narrow-import commitment is Plan 01-03's decision and must be preserved.

---

### Pattern F — Beat generator with co-located helpers (swing + beat_duration)

**Applied to:** `src/musicgen/generators/beat.py`.

**Source:** `/home/bidu/musicgen/music_gen.py:369-495`

```python
def beat_duration(signature: str, tempo: int) -> float:
    """
    Calculates the duration of a beat based on the time signature and BPM.
    """
    numerator, denominator = map(int, signature.split('/'))
    beat_length = 60 / tempo
    return beat_length * (4 / denominator)

def calculate_swing_offset(base_duration: float, swing_amount: float) -> float:
    """
    Calculates the offset time for a note with swing.

    Args:
        base_duration: Base note duration in beats
        swing_amount: Swing amount (0.0 to 1.0, where 0.5 is straight timing)

    Returns:
        float: Time offset in beats
    """
    return base_duration * (swing_amount - 0.5)


def generate_beat(
    part: str,
    tempo: int,
    time_signature: str,
    measures: int,
    name: str,
    swing_amount: float = 0.5,
    cfg: config.Config = None,
) -> Tuple[str, List[float]]:
    """Generates a drum pattern with configurable swing."""
    validator = DurationValidator()
    _beat_cfg = cfg if cfg is not None else config.Config()
    beat_pattern_files = dict(_beat_cfg.beat_roll_pattern_files)

    mf = MIDIFile(1)
    track = 0
    time = 0

    mf.addTrackName(track, time, "Beat")
    mf.addTempo(track, time, tempo)

    numerator, midi_denominator = get_midi_time_signature_values(time_signature)
    mf.addTimeSignature(track, time, numerator, midi_denominator, 24, 8)

    base_duration = validator.get_suggested_duration(time_signature, 'beat')

    # MIDI values for percussion instruments
    kick = 36
    snare = 38
    hihat = 42

    filename = beat_pattern_files.get(time_signature)
    if not filename:
        raise ValueError(f"Time signature {time_signature} not supported.")

    beat_patterns = {}
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(':')
                song_part = parts[0].strip()
                pattern = [int(x) for x in parts[1].split(',')]
                if verify_beat_pattern(pattern, time_signature):
                    beat_patterns.setdefault(song_part, []).append(pattern)

    if part not in beat_patterns or not beat_patterns[part]:
        base_pattern = [kick, hihat] if numerator == 2 else [kick, hihat, snare]
        beat_patterns[part] = [base_pattern]

    beat_pattern = random.choice(beat_patterns[part])             # ← BARE-RANDOM #1 → rng.choice(...)
    beat = beat_pattern * (measures - 1)

    roll_part = part + "_roll"
    roll_pattern = random.choice(beat_patterns.get(roll_part, [beat_pattern]))  # ← BARE-RANDOM #2
    beat.extend(roll_pattern)

    annotations = []
    current_time = 0.0
    for i, drum_hit in enumerate(beat):
        if drum_hit != 0:
            if i % 2 == 1:
                swing_offset = calculate_swing_offset(base_duration, swing_amount)
                actual_time = current_time + swing_offset
            else:
                actual_time = current_time
            mf.addNote(track, 9, drum_hit, actual_time, base_duration, 100)
            annotations.append(f"{actual_time:.3f}\t{len(annotations) + 1}")
        current_time += base_duration

    directory = name.split('-')[0]
    if not os.path.exists(directory):
        os.makedirs(directory)
    midi_filename = os.path.join(directory, f"{name}-beat.mid")

    with open(midi_filename, 'wb') as outf:
        mf.writeFile(outf)

    return midi_filename, annotations
```

Phase-3 rewrite rules (in addition to Pattern E rules):
- Two bare-random sites → `rng.choice(...)` (lines 460, 465 of `music_gen.py`).
- `beat_duration` and `calculate_swing_offset` stay co-located in `beat.py` (CONTEXT.md Claude Discretion: only one caller — no `_swing.py` helper).
- `cfg: config.Config = None` with `_beat_cfg = cfg if cfg is not None else config.Config()` fallback — **preserve exactly** (Plan 02-01 D-02 pattern, RESEARCH.md Open Question 3).
- New signature (RESEARCH.md §Generator Extraction) adds `rng: random.Random` BEFORE `cfg` keyword-only arg.

---

### Pattern G — Sampler pure-function extraction (bare-random → rng.*)

**Applied to:** All free functions in `src/musicgen/sampler.py` (`generate_random_key`, `generate_random_tempo`, `generate_random_swing`, `generate_random_time_signature`, `time_signature_alternative`, `generate_song_measures`, `generate_song_arrangement`, `validate_measures_dict`).

**Source — `generate_random_key`:** `/home/bidu/musicgen/music_gen.py:808-823`

```python
def generate_random_key():
    key_ranges = [(0.107, 'G'), (0.209, 'C'), (0.296, 'D'), (0.357, 'A'), (0.417, 'C#'), (0.47, 'F'),
                  (0.518, 'Am'), (0.561, 'G#'), (0.603, 'Em'), (0.645, 'Bm'), (0.681, 'E'), (0.716, 'A#'),
                  (0.748, 'A#m'), (0.778, 'Fm'), (0.805, 'F#'), (0.831, 'B'), (0.857, 'Gm'), (0.883, 'Dm'),
                  (0.908, 'F#m'), (0.932, 'D#'), (0.956, 'Cm'), (0.977, 'C#m'), (0.989, 'G#m'), (1.0, 'D#m')
    ]
    dice = random.random()              # → rng.random()
    for prob, key in key_ranges:
        if dice < prob:
            return key
    return key_ranges[-1][1]  # explicit fallback
```

**Source — `generate_random_tempo`:** `music_gen.py:825-835` — 1 `random.random()` + 2 `random.randint(...)` calls (one in loop, one in fallback branch).

**Source — `generate_random_swing`:** `music_gen.py:564-589`

```python
def generate_random_swing() -> float:
    swing_weights = [
        (0.5, 0.3),   # 30% no swing
        (0.66, 0.5),  # 50% traditional jazz
        (0.75, 0.2)   # 20% extreme swing
    ]
    base_swing = random.choices(            # → rng.choices(...)
        [s[0] for s in swing_weights],
        weights=[s[1] for s in swing_weights]
    )[0]
    variation = random.uniform(-0.02, 0.02) # → rng.uniform(...)
    return min(0.75, max(0.5, base_swing + variation))
```

**Source — `generate_random_time_signature` (Phase-2 shape, already rng-aware):** `music_gen.py:837-841`

```python
def generate_random_time_signature():
    """Generates a random time signature based on weighted probabilities.
    Delegates to TimeSignatureRegistry.sample_random() per R-S6."""
    return TimeSignatureRegistry.sample_random()
```

Phase-3 rewrite: add `rng: random.Random` parameter and forward it — `return TimeSignatureRegistry.sample_random(rng)` (D-09 contract already supports this).

**Source — `time_signature_alternative`:** `music_gen.py:843-847`

```python
def time_signature_alternative(base_time_signature):
    spec = TimeSignatureRegistry.lookup(base_time_signature)
    return random.choice(spec.alternatives) if spec.alternatives else "4/4"
    # → rng.choice(spec.alternatives) if spec.alternatives else "4/4"
```

**Source — `generate_song_measures`:** `music_gen.py:849-880` — 5 `random.choice([8,16])`-style + 1 `random.random()` gate + 5 `random.choice([ts, ts_alt])` = 11 sites, counting the nested `time_signature_alternative` bare-random internally brings the per-iteration total to 16 draws when gate passes (RESEARCH.md §Sampler Extraction). Retry loop is in the CALLER (the new `SongParams.sample`), NOT in `generate_song_measures` itself — preserve that split.

**Source — `generate_song_arrangement`:** `music_gen.py:508-547`

```python
def generate_song_arrangement(structures_file: Optional[str] = None) -> Tuple[List[str], List[str]]:
    """Generate a musical arrangement based on common music structures."""
    if structures_file is None:
        structures_file = config.DEFAULT_SONG_STRUCTURES_FILE
    try:
        if not os.path.exists(structures_file):
            raise FileNotFoundError(f"Structures file not found: {structures_file}")
        with open(structures_file, 'r') as f:
            data = json.load(f)
        if 'common_structures' not in data:
            raise KeyError("Missing 'common_structures' in JSON file")
        structures = data['common_structures']
        if not structures or not isinstance(structures, list):
            raise ValueError("Invalid or empty structures list")
        result = random.choice(structures)          # → rng.choice(structures)
        unique_elements = list(set(result))
        return unique_elements, result
    except (json.JSONDecodeError, FileNotFoundError, KeyError, ValueError) as e:
        default_structure = ['intro', 'verse', 'chorus', 'outro']
        logger.warning("Using default structure due to error", exc_info=True)
        return list(set(default_structure)), default_structure
```

Phase-3 rewrite: new signature becomes `generate_song_arrangement(rng: random.Random, structures_file: Optional[str] = None)` — `rng` first (the only required positional), `structures_file` optional as today.

Rules unchanged in all sampler extractions:
- Preserve explicit-fallback returns (prevents the Pitfall-5 missing-return bug).
- Preserve `logger.warning(..., exc_info=True)` — percent-format, not f-string.
- Preserve all validation `raise ValueError(...)` lines as-is.

---

### Pattern H — Seeded-RNG unit test (pytest fixture + parametrize + method class)

**Applied to:** `tests/test_sampler.py`, `tests/test_generators/test_{chord,melody,bassline,beat}.py`.

**Source A (class + parametrize matrix):** `/home/bidu/musicgen/tests/test_timesig_registry.py:16-96`

```python
class TestRegistryContents:
    ALL_SIGS = {"2/4", "3/4", "4/4", "5/4", "6/8", "7/8", "12/8"}

    def test_registry_contains_all_seven_signatures(self):
        assert set(TimeSignatureRegistry.all_signatures()) == self.ALL_SIGS

    @pytest.mark.parametrize("sig", ["2/4", "3/4", "4/4", "5/4", "6/8", "7/8", "12/8"])
    def test_lookup_returns_spec(self, sig):
        spec = TimeSignatureRegistry.lookup(sig)
        assert isinstance(spec, TimeSignatureSpec)

    @pytest.mark.parametrize("sig,expected", [
        ("6/8", True), ("12/8", True),
        ("2/4", False), ("3/4", False), ("4/4", False), ("5/4", False), ("7/8", False),
    ])
    def test_is_compound_classification(self, sig, expected):
        assert TimeSignatureRegistry.lookup(sig).is_compound is expected
```

**Source B (fixture factory):** `/home/bidu/musicgen/tests/test_duration_validator.py:13-16`

```python
@pytest.fixture
def validator():
    return DurationValidator()
```

**Source C (sampling smoke — seeded rng returns valid output):** `tests/test_timesig_registry.py:367-371`

```python
def test_sample_random_returns_valid_signature(self):
    all_sigs = set(TimeSignatureRegistry.all_signatures())
    for _ in range(20):
        result = TimeSignatureRegistry.sample_random()
        assert result in all_sigs
```

Skeleton to lift for `tests/test_sampler.py`:

```python
import random
import pytest
from musicgen.sampler import (
    SongParams,
    generate_random_key, generate_random_tempo, generate_random_time_signature,
    generate_random_swing, generate_song_measures, time_signature_alternative,
    generate_song_arrangement,
)

class TestSongParamsSample:
    def test_sample_is_deterministic(self, tmp_path):
        params_a = SongParams.sample(random.Random(42))
        params_b = SongParams.sample(random.Random(42))
        assert params_a == params_b                           # frozen dataclass equality

    @pytest.mark.parametrize("seed", [0, 1, 42, 12345])
    def test_sample_same_seed_same_output(self, seed):
        a = SongParams.sample(random.Random(seed))
        b = SongParams.sample(random.Random(seed))
        assert (a.key, a.tempo, a.time_signature_base, a.swing_amount) == \
               (b.key, b.tempo, b.time_signature_base, b.swing_amount)

class TestGenerateRandomKey:
    @pytest.mark.parametrize("seed", [0, 1, 42])
    def test_deterministic_under_seed(self, seed):
        assert generate_random_key(random.Random(seed)) == generate_random_key(random.Random(seed))
```

Skeleton to lift for `tests/test_generators/test_melody.py` (byte-equal MIDI from RESEARCH.md:794-803):

```python
import os
import random
from pathlib import Path

import pytest
from musicgen.generators.melody import generate_melody

def test_generate_melody_is_deterministic_under_seed(tmp_path):
    args = dict(key="C", tempo=120, time_signature="4/4", measures=4,
                name=str(tmp_path / "song-verse"), part="verse",
                chord_progression=["I", "IV", "V", "I"])
    melody1, path1 = generate_melody(**args, rng=random.Random(42))
    bytes1 = Path(path1).read_bytes()
    os.remove(path1)
    melody2, path2 = generate_melody(**args, rng=random.Random(42))
    assert melody1 == melody2
    assert bytes1 == Path(path2).read_bytes()
```

Rules:
- Test classes `TestX` group related assertions; parametrize at the method level (not class).
- Use `tmp_path` for MIDI write targets (avoid polluting repo root; generators create `name.split('-')[0]` directories).
- `random.Random(seed)` is the canonical fresh-RNG construction — one per run.
- Use `Path(...).read_bytes()` for MIDI byte-equality assertions (RESEARCH.md A3).
- Keep assertions on the returned list (`melody1 == melody2`) AND the written file — double-gate catches both pure-logic drift and MIDIUtil-internal drift.

---

### Pattern I — AST-scan static regression test (no-bare-random guard)

**Applied to:** `tests/test_sampler.py::test_no_bare_random_in_sampler` and the same class can be extended to scan `src/musicgen/generators/*.py`.

**Source:** `/home/bidu/musicgen/tests/test_music_gen_logging.py:17-36` (structural template for AST-based file scanning)

```python
class TestNoPrintCallsRemain:
    """R-S7 exit criterion: zero print() calls in music_gen.py."""

    def test_no_print_calls_remain_in_music_gen(self):
        src_path = os.path.join(
            os.path.dirname(__file__), os.pardir, "music_gen.py"
        )
        with open(os.path.abspath(src_path)) as f:
            tree = ast.parse(f.read())
        print_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "print"
        ]
        assert print_calls == [], (
            f"Found {len(print_calls)} print() call(s) in music_gen.py "
            f"at line(s): {[n.lineno for n in print_calls]}"
        )
```

Adaptation for Phase 3 "no bare random.*" test:

```python
import ast
import os
import pytest

SAMPLER_PATH = os.path.join(os.path.dirname(__file__), os.pardir, "src", "musicgen", "sampler.py")

def _bare_random_call_nodes(source: str):
    """Return AST nodes representing `random.<method>(...)` (module-level, not `rng.<method>`)."""
    tree = ast.parse(source)
    hits = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "random"):
            hits.append(node)
    return hits

def test_no_bare_random_in_sampler():
    with open(os.path.abspath(SAMPLER_PATH)) as f:
        hits = _bare_random_call_nodes(f.read())
    assert hits == [], (
        f"Found {len(hits)} bare random.<method>() call(s) in sampler.py "
        f"at line(s): {[n.lineno for n in hits]} — use rng.<method>() per D-07."
    )
```

The same test can be parametrized over `sampler.py`, `generators/chord.py`, `generators/melody.py`, `generators/bassline.py`, `generators/beat.py`, `duration_validator.py`. Covers both R-X2 and R-X3 static requirements in one file. Consider placing in `tests/test_sampler.py` (Phase 3 scope) OR a new `tests/test_no_bare_random.py` (Phase 5 might upgrade it); CONTEXT.md doesn't mandate a file location.

---

### Pattern J — music21 global-RNG isolation test (no-RNG regression guard)

**Applied to:** `tests/test_music21_isolation.py`.

**Source:** **No direct analog in repo** — test body provided verbatim in RESEARCH.md §music21 Global RNG Audit (lines 694-735). Use `tests/test_duration_validator.py` (Pattern H) ONLY for the `class TestX:` shape:

```python
# Shape borrowed from tests/test_duration_validator.py:18-29:
class TestGetSuggestedDuration:
    @pytest.mark.parametrize(...)
    def test_returns_positive_float(self, validator, time_signature, layer_type):
        ...
```

Phase 3 body (from RESEARCH.md:693-735):

```python
"""R-P7 / D-24 regression guard: music21 must not mutate global random.

If this test starts failing, add a save_random_state() wrapper in sampler.py
and every generator that touches music21 (melody, bassline, chord), and wrap
each music21 call in the new contextmanager. See CONTEXT.md D-23 for rationale.
"""
import random
import pytest


class TestMusic21DoesNotMutateGlobalRandom:
    def test_roman_numeral_preserves_global_state(self):
        from music21 import roman
        state0 = random.getstate()
        for key in ["C", "G", "D", "Am", "Em"]:
            for sym in ["I", "IV", "V", "vi", "ii"]:
                rn = roman.RomanNumeral(sym, key)
                _ = list(rn.pitches)
                for p in rn.pitches:
                    _ = p.midi
        assert random.getstate() == state0

    def test_scale_preserves_global_state(self):
        from music21 import scale
        state0 = random.getstate()
        _ = scale.MajorScale("C"); _ = scale.MinorScale("A")
        _ = scale.MajorScale("G"); _ = scale.MinorScale("E")
        assert random.getstate() == state0

    def test_pitch_midi_roundtrip_preserves_global_state(self):
        from music21 import pitch
        state0 = random.getstate()
        for midi_val in [36, 48, 60, 72]:
            p = pitch.Pitch()
            p.midi = midi_val
            p.octave = 2
            _ = p.midi
        assert random.getstate() == state0
```

---

### Pattern K — Back-compat shim (re-exports in the thin orchestrator)

**Applied to:** `music_gen.py` post-extraction.

**Source A (current import block — what gets rewritten):** `/home/bidu/musicgen/music_gen.py:1-22`

```python
from midiutil import MIDIFile
from music21 import roman, scale, pitch
from pydub import AudioSegment
from midi2audio import FluidSynth
from datetime import datetime
from pedalboard import Pedalboard, Compressor, Gain, Chorus, LadderFilter, Phaser, Delay, Reverb
from pedalboard.io import AudioFile
import logging
import time
import json
import random
import os
import uuid
import musicality_score
from enhanced_duration_validator import DurationValidator, NoteValue   # ← DELETE, replace
import config
from timesig import TimeSignatureRegistry

from typing import Tuple, Dict, List, Optional
import math

logger = logging.getLogger(__name__)
```

**Source B (current orchestrator using old symbols — what calls the shim):** `/home/bidu/musicgen/music_gen.py:1012-1024`

```python
logger.info("Generating song #%s", id)

# Basic musical parameters
key = generate_random_key()
tempo = generate_random_tempo()
time_signature = generate_random_time_signature()
time_signature_variation = 1.0
swing_amount = generate_random_swing()
swing_amount = min(0.75, max(0.5, float(swing_amount)))

# Generates valid measurements and time signatures
while True:
    measures, signatures = generate_song_measures(time_signature, time_signature_variation)
    if validate_measures(measures, signatures):
        break
```

**Target shim (from RESEARCH.md §Risk #5 / Back-compat shim):**

```python
# Top of refactored music_gen.py (representative — actual organization follows
# the CONVENTIONS.md import grouping).
import random

from musicgen.sampler import (
    SongParams,
    generate_random_key, generate_random_tempo, generate_random_time_signature,
    generate_random_swing, generate_song_measures, time_signature_alternative,
    generate_song_arrangement, validate_measures_dict,
)
from musicgen.generators.chord import generate_chord_progression
from musicgen.generators.melody import generate_melody
from musicgen.generators.bassline import generate_bassline
from musicgen.generators.beat import generate_beat, beat_duration, calculate_swing_offset
from musicgen.duration_validator import DurationValidator, NoteValue

_rng = random.Random()   # D-08 — one unseeded RNG threaded to every call site
```

Plus: every call site inside `music_gen.py` for an extracted function adds a trailing `_rng` arg. Example post-rewrite of lines 1013-1022:

```python
key = generate_random_key(_rng)
tempo = generate_random_tempo(_rng)
time_signature = generate_random_time_signature(_rng)
time_signature_variation = 1.0
swing_amount = generate_random_swing(_rng)
swing_amount = min(0.75, max(0.5, float(swing_amount)))

while True:
    measures, signatures = generate_song_measures(time_signature, time_signature_variation, _rng)
    if validate_measures(measures, signatures):
        break
```

Rules:
- Keep `verify_pattern_for_time_signature`, `verify_beat_pattern`, `calculate_measures_for_time_signature`, `validate_measures`, `get_midi_time_signature_values`, `get_note_duration`, `get_note_durations`, `get_melody_durations` (the thin time-sig wrappers at lines 25-68) **unchanged** (CONTEXT.md D-06).
- Keep `mix_and_save`, `create_song`, `generate_song`, `generate_song_parts`, `generate_pedalboard`, `apply_fx_to_layer`, soundfont helpers, levels helpers, `read_instrument_probabilities`, `save_beat_annotations`, and the `if __name__ == "__main__":` guard **unchanged** (CONTEXT.md D-05).
- Post-rewrite: `grep -n 'random\.' music_gen.py` still returns hits inside `mix_and_save` / `create_effect` — those are Phase 4 scope, leave alone. The per-call-site `_rng` threading ONLY happens for calls into extracted code.

---

### Pattern L — CLI stub (typer app, minimal command)

**Applied to:** `src/musicgen/cli.py`, `src/musicgen/__main__.py`.

**Source:** **No direct analog** — templates in RESEARCH.md §CLI Scaffolding (lines 581-622). Style conventions borrow from Pattern A.

`src/musicgen/cli.py` (verbatim from RESEARCH.md:581-612):

```python
"""CLI entry point (stub — real CLI lands in Phase 6 per D-18).

Provides the `musicgen` console script so `pip install -e . && musicgen --help`
works. One stub command (`info`) demonstrates the plumbing; real batch/generate
commands are out of scope for Phase 3.
"""
from __future__ import annotations

import logging
from typing import Optional

import typer

app = typer.Typer(
    help="musicgen — synthetic music dataset generator",
    no_args_is_help=True,
)


@app.command()
def info() -> None:
    """Print package metadata and a friendly pointer at Phase 6's full CLI."""
    typer.echo("musicgen 0.1.0 — Phase 3 package skeleton")
    typer.echo(
        "Real CLI (generate / batch / clean / calibrate) arrives in Phase 6. "
        "Today: `python music_gen.py` runs one smoke-test song."
    )


if __name__ == "__main__":
    app()
```

`src/musicgen/__main__.py` (verbatim from RESEARCH.md:617-621):

```python
"""python -m musicgen entry — routes to the typer app."""
from musicgen.cli import app

app()
```

---

### Pattern M — Package markers (`__init__.py`)

**Applied to:** `src/musicgen/__init__.py`, `src/musicgen/generators/__init__.py`, `tests/test_generators/__init__.py`.

**Source A (empty test-package marker):** `/home/bidu/musicgen/tests/__init__.py` — empty file.

**Source B (package-marker docstring style):** RESEARCH.md §CLI Scaffolding lines 624-640.

`src/musicgen/__init__.py`:

```python
"""musicgen package — Phase 3 skeleton. Public API lands in Phase 5."""
# Intentionally empty — Phase 5 will add `from musicgen.sampler import SongParams`
# and the `generate` / `generate_batch` library entry points.
```

`src/musicgen/generators/__init__.py`:

```python
"""Per-layer MIDI generators (chord, melody, bassline, beat).

Each generator takes explicit fields + injected rng per D-22.
"""
# Empty marker — Phase 5 may add a convenience barrel re-export.
```

`tests/test_generators/__init__.py`: empty (matches current `tests/__init__.py`).

---

### Pattern N — pyproject.toml (hatchling + src-layout, NO in-repo analog)

**Applied to:** `pyproject.toml`.

**Source:** **No in-repo analog.** Template from RESEARCH.md §Technical Approach (lines 151-208) — copy verbatim with the 3 planner-verified fields:
- `requires-python` may bump to `>=3.10` per Risk #1.
- `[project].dependencies` migrates all entries from `requirements.txt` verbatim (RESEARCH.md §Standard Stack).
- `[tool.pytest.ini_options]` includes `testpaths = ["tests"]` AND (per RESEARCH.md §Risk #3 / Open Question 2) `pythonpath = ["."]` to guarantee `config`/`timesig` at repo root remain importable after `conftest.py` deletion.

No code excerpt duplicated here — the planner lifts the RESEARCH.md template directly.

---

## Shared / Cross-Cutting Patterns

These apply to more than one file. Executors should set them up once and reuse.

### S1. Logging — module-level logger, percent-format args

**Source:** every Phase-2 module (`config.py:15`, `timesig.py:21`, `music_gen.py:22`).

```python
import logging
logger = logging.getLogger(__name__)

# Later:
logger.debug("Chord progression: %s", chord_progression)
logger.warning("Using default structure due to error", exc_info=True)
```

Rules enforced by existing tests (`tests/test_music_gen_logging.py`):
- `logger = logging.getLogger(__name__)` at module scope.
- `logging.basicConfig(...)` ONLY inside `if __name__ == "__main__":`.
- **No f-strings as the first arg** to `logger.debug/info/warning/error/exception`. Use `%s`/`%d` placeholders + positional args.
- Import of `logging` at module-top (stdlib group).

Apply to: sampler.py, generators/*.py, duration_validator.py (post-move — already compliant).

### S2. Type hints — Python 3.9-compatible

**Source:** every module (see `/home/bidu/musicgen/config.py:13`, `timesig.py:19`, `music_gen.py:19`).

```python
from typing import Dict, FrozenSet, List, Optional, Tuple
```

Rules:
- Use `typing.Dict`, `typing.List`, `typing.Optional`, `typing.Tuple`, `typing.FrozenSet`.
- Do NOT use PEP 604 `|` union syntax (e.g. `int | None`).
- Every function signature has parameter and return type annotations (CONTEXT.md §Established Patterns).

### S3. Narrow music21 imports

**Source:** `/home/bidu/musicgen/music_gen.py:2`

```python
from music21 import roman, scale, pitch
```

Rules:
- Import only the three submodules actually used (per Plan 01-03 commitment).
- Inside generator bodies, access attributes via the submodule: `roman.RomanNumeral(...)`, `scale.MajorScale(...)`, `scale.MinorScale(...)`, `pitch.Pitch()`.
- Do NOT switch to `import music21` or `from music21 import *`.

### S4. Config fallback pattern (runtime `cfg` default)

**Source:** `/home/bidu/musicgen/music_gen.py:418`, `music_gen.py:672`.

```python
def generate_beat(
    part: str, tempo: int, time_signature: str, measures: int, name: str,
    swing_amount: float = 0.5,
    cfg: config.Config = None,   # Plan 02-01 D-02 contract
) -> Tuple[str, List[float]]:
    validator = DurationValidator()
    _beat_cfg = cfg if cfg is not None else config.Config()   # fallback
    beat_pattern_files = dict(_beat_cfg.beat_roll_pattern_files)
```

Rules:
- Signature declares `cfg: config.Config = None` (NOT `Optional[config.Config]` to match existing style) — **keep as-is** on the extraction; tightening to `Optional` is out of scope.
- First line of body: `_cfg = cfg if cfg is not None else config.Config()`.
- Use `_cfg.*` for all config attribute reads in the body.

Apply to: `sampler.py::SongParams.sample` (D-21 signature), `generators/beat.py::generate_beat`. Other generators don't receive `cfg` (they receive `pattern_file` / `chord_progression` directly).

### S5. Import grouping order

**Source:** `/home/bidu/musicgen/music_gen.py:1-20` (pre-Phase-3 shape) + `timesig.py:16-19` / `config.py:9-13` (cleaner Phase-2 shape).

Grouping convention (from CONVENTIONS.md via CONTEXT.md §Established Patterns):
1. stdlib (`import logging`, `import random`, `import json`, `import os`, `from dataclasses import ...`, `from typing import ...`)
2. third-party (`from midiutil import MIDIFile`, `from music21 import roman, scale, pitch`)
3. first-party (`import config`, `from timesig import TimeSignatureRegistry`, `from musicgen.duration_validator import DurationValidator`)

Blank line between groups. Alphabetical within a group is preferred but existing `music_gen.py` does not enforce it strictly (CONTEXT.md Claude Discretion: match existing style).

### S6. Explicit fallbacks on randomized branch logic

**Source:** `/home/bidu/musicgen/music_gen.py:821-823` and `music_gen.py:834-835`.

```python
for prob, key in key_ranges:
    if dice < prob:
        return key
# Explicit fallback in case float-rounding leaves dice >= final prob
return key_ranges[-1][1]
```

Rule: every threshold-loop that can miss its last branch due to float rounding has an explicit `return <last-branch>` after the loop (prevents Pitfall-5 missing-return bug). Preserve exactly when extracting.

---

## Replaced / Deleted Files

### `tests/conftest.py` — DELETED (D-16)

**Current content** (`/home/bidu/musicgen/tests/conftest.py`):

```python
"""Test configuration for Phase 1 pytest skeleton.

Phase 3 will introduce `pyproject.toml` and a proper `src/musicgen/` package,
at which point this conftest's sys.path shim becomes unnecessary and should
be deleted along with this file.
"""
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
```

**Replacement:** `pyproject.toml` `[tool.pytest.ini_options]` block:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]        # RESEARCH.md §Risk #3 fallback C — keeps root config/timesig importable
```

### `requirements.txt` — DELETED (D-14)

**Replacement:** `pyproject.toml [project].dependencies` (full list in RESEARCH.md §Technical Approach). Each entry migrates verbatim; `typer>=0.12` is ADDED.

### `dev-requirements.txt` — DELETED (D-14)

**Replacement:** `pyproject.toml [project.optional-dependencies].dev`:

```toml
[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "pytest-xdist>=3.5",
]
```

### `enhanced_duration_validator.py` — DELETED (D-10 via `git mv`)

**Replacement:** `src/musicgen/duration_validator.py` — literal file move, zero code changes inside. Consumers update imports:
- `tests/test_duration_validator.py:10` — `from enhanced_duration_validator import ...` → `from musicgen.duration_validator import ...`.
- `music_gen.py:15` — same rewrite.
- Generator modules import `from musicgen.duration_validator import DurationValidator`.

Verification command (RESEARCH.md:574): `grep -r "enhanced_duration_validator" --include="*.py" /home/bidu/musicgen/` must return zero hits.

---

## Files With No In-Repo Analog (Use RESEARCH.md Templates)

| File | Why no analog | Source of truth |
|------|---------------|-----------------|
| `pyproject.toml` | No `pyproject.toml` exists today; project uses flat `requirements.txt` | RESEARCH.md §Technical Approach lines 151-208 |
| `src/musicgen/__init__.py` | Package doesn't exist yet | RESEARCH.md §CLI Scaffolding lines 624-630 |
| `src/musicgen/__main__.py` | No existing `python -m X` entry in project | RESEARCH.md §CLI Scaffolding lines 617-621 |
| `src/musicgen/cli.py` | No typer/click/argparse CLI exists in project | RESEARCH.md §CLI Scaffolding lines 581-612 |
| `src/musicgen/generators/__init__.py` | Generators folder doesn't exist | RESEARCH.md §CLI Scaffolding lines 632-640 |
| `tests/test_music21_isolation.py` | No comparable global-state audit test exists | RESEARCH.md §music21 Global RNG Audit lines 693-735 |
| `tests/test_package_install.py` *(optional)* | No subprocess/venv tests exist | RESEARCH.md §Wave 0 Gaps — author bespoke from `subprocess.run` + `tmp_path` |

All other files have at least a role-match analog in the existing codebase.

---

## Metadata

**Analog search scope:** `/home/bidu/musicgen/` (excluding `.venv/`, `.planning/`, `__pycache__/`).
**Files scanned:** `config.py`, `timesig.py`, `music_gen.py`, `enhanced_duration_validator.py`, `musicality_score.py`, `tests/*.py`, `tests/conftest.py`, `requirements.txt`, `dev-requirements.txt`.
**Line-range anchors confirmed:** `music_gen.py:1-22`, `70-145`, `147-255`, `257-367`, `369-495`, `508-547`, `564-589`, `808-880`, `1002-1042`; `timesig.py:1-89`, `113-291`; `config.py:1-100`; `enhanced_duration_validator.py:1-143`; `tests/test_timesig_registry.py:1-413`; `tests/test_music_gen_logging.py:1-141`; `tests/test_duration_validator.py:1-129`; `tests/conftest.py:1-15`.
**Pattern extraction date:** 2026-04-18

## PATTERN MAPPING COMPLETE
