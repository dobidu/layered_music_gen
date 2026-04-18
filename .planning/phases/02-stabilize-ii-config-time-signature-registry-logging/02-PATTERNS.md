# Phase 2: Stabilize II — Pattern Map

**Mapped:** 2026-04-18
**Files analyzed:** 5 (timesig.py new, enhanced_duration_validator.py modified, music_gen.py modified, tests/test_timesig_registry.py new, tests/test_music_gen_logging.py new)
**Analogs found:** 5 / 5

Note: config.py and tests/test_config.py are already covered by Plan 02-01 and are excluded here. tests/test_time_signature.py and tests/test_duration_validator.py require import-line-only changes (no pattern mapping needed — preserve exactly as written in git at commits aedef2d and 5a4dd13).

---

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `timesig.py` | registry/service | transform (pure data) | `enhanced_duration_validator.py` (`TimeSignatureInfo` dataclass + `DurationValidator` class) | role-match |
| `enhanced_duration_validator.py` | service | transform | itself (modify `_analyze_time_signature` only) | self-ref |
| `music_gen.py` (time-sig wrappers) | module thin-wrappers | request-response | `music_gen.py` lines 20-166 (the functions being wrapped) | self-ref |
| `music_gen.py` (print→logging) | module | request-response | `musicality_score.py` (per-module logger, `%s` format-arg style) | role-match |
| `tests/test_timesig_registry.py` | test | transform | `tests/test_time_signature.py` (git: aedef2d) — parametrized `TestX` classes, `is True`/`is False` assertions, no audio deps | exact |
| `tests/test_music_gen_logging.py` | test | request-response | `tests/test_time_signature.py` (import structure) + new AST-scan pattern | partial |

---

## Pattern Assignments

### `timesig.py` (registry, pure-data transform)

**Analog:** `enhanced_duration_validator.py` — provides the dataclass + class pattern already in use for time-sig data

**Imports pattern** (`enhanced_duration_validator.py` lines 1-7):
```python
from dataclasses import dataclass
from typing import List, Dict, Optional, Set
from enum import Enum
import math
import logging
```
For `timesig.py`, use:
```python
import logging
import random
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Tuple
```
No `logging` is actually needed in `timesig.py` itself (pure data, no I/O); include it only if `TimeSignatureRegistry.lookup()` needs to warn on unknown signature.

**Dataclass pattern** (`enhanced_duration_validator.py` lines 23-32):
```python
@dataclass
class TimeSignatureInfo:
    numerator: int
    denominator: int
    is_compound: bool
    valid_durations: Set[float]
    min_duration: float
    max_duration: float
    primary_division: float
    beats_per_measure: float
```
For `TimeSignatureSpec`, use `@dataclass(frozen=True)` (D-06 design flexibility; immutable registry entries prevent accidental mutation). Follow same field-per-line style.

**Registry constant pattern** — no codebase analog exists; use RESEARCH.md Pattern 1 directly:
```python
class TimeSignatureRegistry:
    REGISTRY: Dict[str, "TimeSignatureSpec"] = {
        "4/4": TimeSignatureSpec(...),
        "3/4": TimeSignatureSpec(...),
        # one entry per signature
    }

    @classmethod
    def lookup(cls, time_signature: str) -> "TimeSignatureSpec":
        return cls.REGISTRY[time_signature]

    @classmethod
    def all_signatures(cls) -> List[str]:
        return list(cls.REGISTRY.keys())

    @classmethod
    def sample_random(cls) -> str:
        """Weighted random selection. Replaces generate_random_time_signature threshold-loop."""
        sigs = cls.all_signatures()
        weights = [cls.REGISTRY[s].sampling_weight for s in sigs]
        return random.choices(sigs, weights=weights, k=1)[0]
```

**Method pattern for spec validation** — mirrors `music_gen.py` lines 20-37 (compound branch, then numerator-branch, then default-True):
```python
# music_gen.py:20-37 — the exact logic to encode in TimeSignatureSpec.verify_chord_pattern_length
def verify_pattern_for_time_signature(chord_pattern, time_signature):
    numerator, denominator = map(int, time_signature.split('/'))
    if denominator == 8 and numerator % 3 == 0:
        return len(chord_pattern) in [2, 3, 6]
    elif numerator == 4:
        return len(chord_pattern) in [1, 2, 4]
    elif numerator == 3:
        return len(chord_pattern) in [1, 3]
    elif numerator == 2:
        return len(chord_pattern) in [1, 2]
    return True  # ← default-True: 5/4, 7/8 — MUST be preserved via empty-frozenset guard
```
Registry method must implement this as:
```python
def verify_chord_pattern_length(self, length: int) -> bool:
    if not self.valid_chord_pattern_lengths:   # empty frozenset → default-True for 5/4, 7/8
        return True
    return length in self.valid_chord_pattern_lengths
```

**Beat pattern validation** — mirrors `music_gen.py` lines 40-50 (cosmetic-if; BOTH branches return `len == numerator`):
```python
# music_gen.py:40-50 — cosmetic-if: both branches identical
def verify_beat_pattern(pattern, time_signature):
    numerator, denominator = map(int, time_signature.split('/'))
    if denominator == 8 and numerator % 3 == 0:
        return len(pattern) == numerator   # 6/8 → 6, NOT 3
    else:
        return len(pattern) == numerator
```
Registry method:
```python
def verify_beat_pattern_length(self, length: int) -> bool:
    # CRITICAL: preserve cosmetic-if — both branches return length == numerator.
    # Tests in test_time_signature.py pin verify_beat_pattern([1,0,1], "6/8") is False.
    return length == self.beat_pattern_length  # beat_pattern_length = numerator for ALL sigs
```

**measures_for pattern** — mirrors `music_gen.py` lines 52-62:
```python
# music_gen.py:52-62
def calculate_measures_for_time_signature(base_length, time_signature):
    numerator, denominator = map(int, time_signature.split('/'))
    if denominator == 8 and numerator % 3 == 0:
        return base_length * 2
    elif numerator == 2:
        return base_length * 2
    elif numerator == 3:
        return int(base_length * 4/3)
    return base_length
```
Registry field: `measure_multiplier: float` (1.0, 2.0, or 4/3). Method:
```python
def measures_for(self, base_length: int) -> int:
    return int(base_length * self.measure_multiplier)
```

**sampling_weight field** — from `music_gen.py` lines 943-951 threshold-loop (must sum to 1.0):
```
"4/4": 0.50, "3/4": 0.15, "2/4": 0.10, "6/8": 0.10, "12/8": 0.05, "7/8": 0.05, "5/4": 0.05
```

**alternatives field** — from `music_gen.py` lines 964-971:
```python
"4/4": ("2/4", "3/4", "6/8", "12/8"),
"3/4": ("6/8", "4/4", "2/4", "12/8"),
"2/4": ("4/4", "6/8", "3/4"),
"6/8": ("12/8", "3/4", "4/4", "2/4"),
"12/8": ("6/8", "4/4", "3/4"),
"7/8": ("4/4", "6/8", "5/4"),
"5/4": ("4/4", "7/8", "3/4"),
```
Field: `alternatives: Tuple[str, ...]`

**Error handling pattern** — `lookup()` raises `KeyError` on unknown signature (stdlib dict behavior). No explicit try/except needed in the registry; callers that need a fallback handle it. See `music_gen.py:99-101` for the raise-ValueError pattern used in `get_midi_time_signature_values`.

---

### `enhanced_duration_validator.py` — modify `_analyze_time_signature` only

**Analog:** `enhanced_duration_validator.py` itself (the full existing file is the reference; only lines 39-87 change)

**Current pattern to REPLACE** (`enhanced_duration_validator.py` lines 39-87):
```python
def _analyze_time_signature(self, time_signature: str) -> TimeSignatureInfo:
    if time_signature in self._time_signature_cache:
        return self._time_signature_cache[time_signature]
    numerator, denominator = map(int, time_signature.split('/'))
    is_compound = denominator == 8 and numerator % 3 == 0
    beats_per_measure = numerator / 3 if is_compound else numerator
    if is_compound:
        valid_durations = { NoteValue.DOTTED_QUARTER.value, NoteValue.QUARTER.value,
                            NoteValue.DOTTED_EIGHTH.value, NoteValue.EIGHTH.value,
                            NoteValue.SIXTEENTH.value }
        min_duration = NoteValue.SIXTEENTH.value
        max_duration = NoteValue.DOTTED_QUARTER.value * (numerator / 3)
        primary_division = 3.0
    else:
        valid_durations = { NoteValue.WHOLE.value, NoteValue.HALF.value,
                            NoteValue.QUARTER.value, NoteValue.EIGHTH.value,
                            NoteValue.SIXTEENTH.value, NoteValue.DOTTED_HALF.value,
                            NoteValue.DOTTED_QUARTER.value, NoteValue.DOTTED_EIGHTH.value }
        min_duration = NoteValue.SIXTEENTH.value
        max_duration = float(numerator)
        primary_division = 2.0
    info = TimeSignatureInfo(numerator=..., ...)
    self._time_signature_cache[time_signature] = info
    return info
```

**Replacement pattern** (delegate to registry, keep legacy `TimeSignatureInfo` shape):
```python
def _analyze_time_signature(self, time_signature: str) -> TimeSignatureInfo:
    if time_signature in self._time_signature_cache:
        return self._time_signature_cache[time_signature]
    from timesig import TimeSignatureRegistry          # local import avoids circular import
    spec = TimeSignatureRegistry.lookup(time_signature)
    info = TimeSignatureInfo(
        numerator=spec.numerator,
        denominator=spec.denominator,
        is_compound=spec.is_compound,
        valid_durations=set(spec.valid_durations),    # TimeSignatureInfo uses Set, not FrozenSet
        min_duration=spec.min_duration,
        max_duration=spec.max_duration,
        primary_division=spec.primary_division,
        beats_per_measure=spec.beats_per_measure,
    )
    self._time_signature_cache[time_signature] = info
    return info
```
NOTE: The local `from timesig import ...` inside the method (not at module top) is the safest pattern to avoid any potential circular-import risk (Pitfall 1 in RESEARCH.md). Alternatively, add `import timesig` at module top — either approach is acceptable as long as `config.py` does NOT also import `timesig`.

**Unchanged patterns to preserve** — everything else in `enhanced_duration_validator.py` stays identical:
- `NoteValue` enum (lines 8-20): keep exactly — numeric values are pinned by `tests/test_duration_validator.py::TestNoteValue`
- `TimeSignatureInfo` dataclass (lines 23-32): keep exactly — shape is the adapter interface
- `DurationValidator.__init__` (lines 35-37): keep exactly — logger and cache pattern
- `get_valid_duration` (lines 89-142): keep exactly — layer-specific duration sets stay here per RESEARCH.md recommendation
- `validate_layer_duration` (lines 144-155): keep exactly
- `get_suggested_duration` (lines 157-172): keep exactly

**Logger pattern** (`enhanced_duration_validator.py` line 36) — already correct, keep:
```python
self.logger = logging.getLogger(__name__)
```

---

### `music_gen.py` — time-signature thin wrappers

**Analog:** `music_gen.py` lines 20-166 (the functions being wrapped) — they become one-liners delegating to the registry

**Import addition** (insert after existing imports at lines 1-17):
```python
from timesig import TimeSignatureRegistry
```

**Thin wrapper pattern** — each module-level function becomes a one-liner. Reference for style: `music_gen.py:20` existing function signature preserved unchanged for test compatibility.
```python
def verify_pattern_for_time_signature(chord_pattern: List[str], time_signature: str) -> bool:
    """Delegates to TimeSignatureRegistry per R-S6. Kept as module-level function
    for backwards compatibility with tests/test_time_signature.py (Plan 01-04)."""
    return TimeSignatureRegistry.lookup(time_signature).verify_chord_pattern_length(len(chord_pattern))

def verify_beat_pattern(pattern: List[int], time_signature: str) -> bool:
    """Delegates to TimeSignatureRegistry."""
    return TimeSignatureRegistry.lookup(time_signature).verify_beat_pattern_length(len(pattern))

def calculate_measures_for_time_signature(base_length: int, time_signature: str) -> int:
    """Delegates to TimeSignatureRegistry."""
    return TimeSignatureRegistry.lookup(time_signature).measures_for(base_length)

def validate_measures(measures: Dict[str, int], signatures: Dict[str, str]) -> bool:
    """Cross-signature validation — cannot be a single-spec method (Pitfall 4 in RESEARCH.md)."""
    for part, measure_count in measures.items():
        spec = TimeSignatureRegistry.lookup(signatures[part])
        if not spec.measure_count_valid(measure_count):
            return False
    return True

def get_midi_time_signature_values(time_signature: str) -> Tuple[int, int]:
    spec = TimeSignatureRegistry.lookup(time_signature)
    return spec.numerator, spec.midi_denominator_power

def get_note_duration(time_signature: str) -> float:
    return TimeSignatureRegistry.lookup(time_signature).primary_beat_duration

def get_note_durations(time_signature: str) -> dict:
    return TimeSignatureRegistry.lookup(time_signature).note_duration_map()

def get_melody_durations(time_signature: str) -> list:
    return list(TimeSignatureRegistry.lookup(time_signature).melody_duration_candidates)

def generate_random_time_signature() -> str:
    return TimeSignatureRegistry.sample_random()

def time_signature_alternative(base_time_signature: str) -> str:
    spec = TimeSignatureRegistry.lookup(base_time_signature)
    return random.choice(spec.alternatives) if spec.alternatives else "4/4"
```

---

### `music_gen.py` — print→logging migration

**Analog:** `musicality_score.py` — established per-module logger and `%s` format-arg style

**Logger setup pattern** (`musicality_score.py` lines 6, 14 — but corrected per RESEARCH.md Pitfall 2):
```python
# musicality_score.py:6 — import already present
import logging
# musicality_score.py:14 — but placed WRONG (inside __init__). For music_gen.py use module-level:
logger = logging.getLogger(__name__)    # ← add at module top, OUTSIDE any class or function
```

**`basicConfig` placement pattern** — must be inside `if __name__ == "__main__":` only (never at module level). Analog: `music_gen.py` lines 1170-1172 (existing `__main__` guard, currently empty of logging setup):
```python
# music_gen.py:1170 — the existing guard to extend
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    # Phase 6: swap to pythonjsonlogger.jsonlogger.JsonFormatter when --json flag arrives
    for i in range(1):
        generate_song(i)
```

**`%s` format-arg style** (`musicality_score.py` line 67):
```python
self.logger.exception("Error in tempo analysis: %s", exc)
```
Apply to all 32 print conversions. Never use f-strings with logger calls. Full inventory in RESEARCH.md Print Call Inventory (32 calls with severity).

**DEBUG print conversion pattern** (e.g. `music_gen.py:241`):
```python
# BEFORE:
print("\t\t\tChord progression: " + str(chord_progression))
# AFTER:
logger.debug("Chord progression: %s", chord_progression)
```

**INFO print conversion pattern** (e.g. `music_gen.py:781`):
```python
# BEFORE:
print("Beat soundfont: " + beat_soundfont)
# AFTER:
logger.info("Beat soundfont: %s", beat_soundfont)
```

**WARNING with traceback inside except** (e.g. `music_gen.py:646`):
```python
# BEFORE:
except (...):
    print(f"Warning: Using default structure due to error: {str(e)}")
# AFTER:
except (...):
    logger.warning("Using default structure due to error", exc_info=True)
```
Source: `musicality_score.py:66-67` `.exception()` + `exc_info=True` pattern.

**Per-component loop conversion** (`music_gen.py:1082-1084`):
```python
# BEFORE (loop emitting N print calls):
print(f'Component Scores:')
for component, value in component_scores.items():
    print(f'{component:>10}: {value:.2f}')
# AFTER (aggregate into one DEBUG call):
logger.debug("Component scores: %s", component_scores)
```

---

### `tests/test_timesig_registry.py` (new, test, transform)

**Analog:** `tests/test_time_signature.py` (git commit aedef2d) — provides the exact class-per-function, parametrize-heavy, `is True`/`is False` assertion style

**Module docstring + import pattern** (git: aedef2d, lines 1-9):
```python
"""
Pure-function unit tests for time-signature helpers in music_gen.py.

These tests pin the CURRENT behavior of the functions. Phase 2 will move
them into a TimeSignatureRegistry; the tests should still pass after that
move (delegation, not behavior change).
"""
import pytest
import music_gen
```
For `test_timesig_registry.py`, adapt to:
```python
"""
Registry-level unit tests for timesig.TimeSignatureRegistry.

Pin ALL 7 supported signatures with exact field values. The analogous
wrapper-level tests in tests/test_time_signature.py verify that
music_gen delegates correctly; these tests verify the registry itself.
"""
import pytest
from timesig import TimeSignatureRegistry, TimeSignatureSpec
```

**Parametrized test class pattern** (git: aedef2d, lines 12-23):
```python
class TestVerifyPatternForTimeSignature:
    @pytest.mark.parametrize("length", [2, 3, 6])
    def test_compound_6_8_accepts_2_3_6(self, length):
        pattern = ["I"] * length
        assert music_gen.verify_pattern_for_time_signature(pattern, "6/8") is True
```
Apply same structure to registry tests:
```python
class TestTimeSignatureRegistryContents:
    ALL_SIGS = {"2/4", "3/4", "4/4", "5/4", "6/8", "7/8", "12/8"}

    def test_registry_contains_all_seven_signatures(self):
        assert set(TimeSignatureRegistry.all_signatures()) == self.ALL_SIGS

    @pytest.mark.parametrize("sig", ["2/4", "3/4", "4/4", "5/4", "6/8", "7/8", "12/8"])
    def test_lookup_returns_spec(self, sig):
        spec = TimeSignatureRegistry.lookup(sig)
        assert isinstance(spec, TimeSignatureSpec)

    @pytest.mark.parametrize("sig,expected", [("6/8", True), ("12/8", True),
                                               ("4/4", False), ("3/4", False),
                                               ("2/4", False), ("5/4", False), ("7/8", False)])
    def test_is_compound_classification(self, sig, expected):
        assert TimeSignatureRegistry.lookup(sig).is_compound is expected
```

**`is True`/`is False` assertion style** — matches Phase 01-04 convention exactly. Never use `assertEqual(result, True)` or `assert result`. Use `assert ... is True` / `assert ... is False`.

**No audio deps constraint** (per Plan 01-04 SUMMARY): no imports of `librosa`, `pedalboard`, `pydub`, `midi2audio`, `music_gen` in this test file.

**Cosmetic-if preservation test** (pinned from Plan 01-04 SUMMARY):
```python
class TestVerifyBeatPatternLength:
    def test_6_8_requires_length_6_not_3(self):
        spec = TimeSignatureRegistry.lookup("6/8")
        assert spec.verify_beat_pattern_length(6) is True
        assert spec.verify_beat_pattern_length(3) is False   # cosmetic-if: len == numerator, not numerator/2

    def test_12_8_requires_length_12_not_6(self):
        spec = TimeSignatureRegistry.lookup("12/8")
        assert spec.verify_beat_pattern_length(12) is True
        assert spec.verify_beat_pattern_length(6) is False
```

**Sampling weight sum test**:
```python
class TestSamplingWeights:
    def test_sampling_weights_sum_to_one(self):
        total = sum(TimeSignatureRegistry.lookup(s).sampling_weight
                    for s in TimeSignatureRegistry.all_signatures())
        assert abs(total - 1.0) < 1e-9
```

---

### `tests/test_music_gen_logging.py` (new, test, request-response)

**Analog for import style:** `tests/test_time_signature.py` (git: aedef2d) — `import music_gen` pattern and `conftest.py` sys.path shim

**Module docstring + import pattern**:
```python
"""
Regression tests for music_gen.py logging migration (R-S7).

1. Import side-effect guard — importing music_gen must not emit any logs.
2. AST-level print() scan — no print() calls may remain after Phase 2.
"""
import ast
import logging
import pytest
import music_gen
```

**AST-scan pattern** — no codebase analog exists; use stdlib `ast`:
```python
class TestNoPrintCallsRemain:
    def test_no_print_calls_remain_in_music_gen(self):
        import os
        src_path = os.path.join(os.path.dirname(__file__), os.pardir, "music_gen.py")
        with open(os.path.abspath(src_path)) as f:
            tree = ast.parse(f.read())
        print_calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "print"
        ]
        assert print_calls == [], f"Found {len(print_calls)} print() call(s) in music_gen.py"
```

**Import side-effect guard pattern** — test that `import music_gen` emits no log records. Use `caplog` (pytest built-in):
```python
class TestImportSideEffects:
    def test_import_music_gen_does_not_emit_logs(self, caplog):
        with caplog.at_level(logging.DEBUG):
            import importlib
            import music_gen as _mg
            importlib.reload(_mg)   # force re-execution to check for side-effect logs
        assert caplog.records == [], f"Unexpected log records: {caplog.records}"
```

---

## Shared Patterns

### Per-Module Logger (apply to `timesig.py` and `music_gen.py`)

**Source:** `musicality_score.py` line 6 (import) + corrected placement (NOT inside `__init__`)

```python
import logging
logger = logging.getLogger(__name__)   # module-level, not instance-level
```

Apply to: `timesig.py` (top of module), `music_gen.py` (top of module, after existing imports).
Do NOT add `logging.basicConfig(...)` at module level — confine to `if __name__ == "__main__":` in `music_gen.py` only.

### `%s` Format-Arg Style (apply to all logging calls)

**Source:** `musicality_score.py` line 67

```python
self.logger.exception("Error in tempo analysis: %s", exc)
```

All new `logger.*()` calls in `music_gen.py` must use `%s` format args, never f-strings. Rationale: defers string formatting until the handler decides to emit.

### Dataclass Field Convention (apply to `TimeSignatureSpec`)

**Source:** `enhanced_duration_validator.py` lines 23-32

```python
@dataclass
class TimeSignatureInfo:
    numerator: int
    denominator: int
    is_compound: bool
    valid_durations: Set[float]
    min_duration: float
    max_duration: float
    primary_division: float
    beats_per_measure: float
```

`TimeSignatureSpec` follows the same one-field-per-line style. Use `@dataclass(frozen=True)` for the registry entries (immutable). Use `FrozenSet[int]` / `FrozenSet[float]` instead of `Set` for fields that hold duration/pattern sets.

### Cache Pattern for `_analyze_time_signature`

**Source:** `enhanced_duration_validator.py` lines 41-42

```python
if time_signature in self._time_signature_cache:
    return self._time_signature_cache[time_signature]
```

Keep this cache in the modified `_analyze_time_signature` — it is still useful even though the body now delegates to the registry (avoids repeated `TimeSignatureRegistry.lookup()` calls from the same `DurationValidator` instance).

### `is True` / `is False` Assertion Convention (apply to all new tests)

**Source:** `tests/test_time_signature.py` (git: aedef2d) — every assertion uses `is True` or `is False`, never `assertEqual(result, True)` or bare `assert result`.

Apply to: all assertions in `test_timesig_registry.py` for boolean-returning methods.

---

## No Analog Found

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| `timesig.py` (registry class body) | registry | pure-data | No registry pattern exists yet — codebase uses scattered inline branches. RESEARCH.md Pattern 1 is the design; `enhanced_duration_validator.py`'s dataclass style is the nearest shape analog only. |
| `tests/test_music_gen_logging.py` (AST scan) | test | n/a | AST-level source scanning is not used anywhere in the codebase today. Pattern comes from RESEARCH.md / stdlib `ast` only. |

---

## Critical Constraints for Planner

These are not preferences — they are hard constraints from the pinned test suite:

1. **`verify_beat_pattern_length` must return `length == numerator` for ALL signatures, including 6/8 and 12/8.** Tests `test_compound_6_8_length_3_not_ok` and `test_compound_12_8_length_6_not_ok` pin this. Do NOT "fix" the cosmetic-if to return `length == numerator/2` for compound meters.

2. **`verify_chord_pattern_length` must return `True` for 5/4 and 7/8 regardless of pattern length.** Test `test_unknown_signature_defaults_true` pins this. Empty `frozenset()` for `valid_chord_pattern_lengths` must trigger the default-True guard in the method.

3. **`_analyze_time_signature` in `enhanced_duration_validator.py` must return a `TimeSignatureInfo` with `valid_durations` as a `set` (not `frozenset`).** The `TimeSignatureInfo` dataclass field is typed `Set[float]`; the registry stores `FrozenSet[float]`. The adapter must call `set(spec.valid_durations)`.

4. **`NoteValue` enum in `enhanced_duration_validator.py` must not change.** Tests pin exact numeric values (`WHOLE == 4.0`, `QUARTER == 1.0`, etc.).

5. **`logging.basicConfig` must NOT appear at module level in `music_gen.py`.** It belongs only inside `if __name__ == "__main__":`. Violating this reintroduces the import-side-effect bug closed in Phase 01-01.

6. **`timesig.py` must not import from `config.py` and `config.py` must not import from `timesig.py`.** They are independent modules; only `music_gen.py` imports both.

---

## Metadata

**Analog search scope:** `/home/bidu/musicgen/` (repo root flat layout — no `src/` package yet)
**Files scanned:** `music_gen.py`, `enhanced_duration_validator.py`, `musicality_score.py`, git objects at aedef2d/5a4dd13/fbb830f (test files deleted from working tree but available in git history)
**Tests deleted from working tree:** `tests/` directory shows as `D` in `git status` — planner must restore these files (git restore or equivalent) as part of Phase 2 wave 0 before any test can run
**Pattern extraction date:** 2026-04-18
