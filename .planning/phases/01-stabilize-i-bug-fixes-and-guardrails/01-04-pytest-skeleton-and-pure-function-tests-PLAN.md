---
phase: 01-stabilize-i-bug-fixes-and-guardrails
plan: 04
type: execute
wave: 2
depends_on: [01]
files_modified:
  - dev-requirements.txt
  - tests/__init__.py
  - tests/conftest.py
  - tests/test_time_signature.py
  - tests/test_duration_validator.py
autonomous: true
requirements: [R-Q2]
must_haves:
  truths:
    - "`pytest` and `pytest-cov` are installable via a documented dev path"
    - "`pytest tests/ -q` exits 0"
    - "Tests cover `verify_pattern_for_time_signature`, `verify_beat_pattern`, `validate_measures`, `DurationValidator`"
    - "Tests run in under 10 seconds and require no audio dependencies"
  artifacts:
    - path: "dev-requirements.txt"
      provides: "Test dependencies (pytest, pytest-cov), separated from runtime requirements.txt"
      contains: "pytest"
    - path: "tests/conftest.py"
      provides: "sys.path shim so tests can import music_gen / enhanced_duration_validator from repo root"
    - path: "tests/test_time_signature.py"
      provides: "Unit tests for verify_pattern_for_time_signature, verify_beat_pattern, validate_measures"
      min_lines: 60
    - path: "tests/test_duration_validator.py"
      provides: "Unit tests for DurationValidator"
      min_lines: 40
  key_links:
    - from: "tests/conftest.py"
      to: "music_gen.py + enhanced_duration_validator.py"
      via: "sys.path.insert(0, repo_root) so `import music_gen` works without installing the package (Phase 3 lands pyproject.toml)"
      pattern: "sys\\.path\\.insert"
---

<objective>
Land the first pytest skeleton and unit tests for the four pure-function targets named in the ROADMAP: `verify_pattern_for_time_signature`, `verify_beat_pattern`, `validate_measures`, and `DurationValidator`. Use a lightweight `dev-requirements.txt` install path because `pyproject.toml` doesn't land until Phase 3.

Purpose: This is the first executable safety net for the project. Phase 2 (config + timesig registry) and Phase 3 (package extraction) can only refactor safely if these pure functions have green tests pinning their current behavior. Also seeds R-Q2 test-coverage progress.

Output: `tests/` directory with two test files green under `pytest tests/ -q`, plus a `dev-requirements.txt` listing `pytest>=8.0` and `pytest-cov>=5.0`.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/REQUIREMENTS.md
@.planning/codebase/TESTING.md
@.planning/codebase/STRUCTURE.md
@music_gen.py
@enhanced_duration_validator.py

<interfaces>
**Pure functions under test — verified bodies (lines 42-80 of music_gen.py):**

```python
# music_gen.py:22
def verify_pattern_for_time_signature(chord_pattern: List[str], time_signature: str) -> bool:
    """
    Returns True if the chord_pattern length is compatible with the time_signature.
    Rules from current implementation:
      - 6/8, 12/8 (compound, denominator==8 and numerator%3==0): valid lens = {2, 3, 6}
      - 4/4 (numerator==4): valid lens = {1, 2, 4}
      - 3/4 (numerator==3): valid lens = {1, 3}
      - 2/4 (numerator==2): valid lens = {1, 2}
      - everything else: True (default)
    """

# music_gen.py:42 — VERIFIED BODY:
def verify_beat_pattern(pattern: List[int], time_signature: str) -> bool:
    numerator, denominator = map(int, time_signature.split('/'))
    if denominator == 8 and numerator % 3 == 0:
        return len(pattern) == numerator
    else:
        return len(pattern) == numerator
    # i.e. BOTH branches return `len(pattern) == numerator`. The conditional is
    # cosmetic — effectively the function just checks len(pattern) == numerator
    # for every time signature.

# music_gen.py:66 — VERIFIED BODY:
def validate_measures(measures: Dict[str, int], signatures: Dict[str, str]) -> bool:
    for part, measure_count in measures.items():
        time_sig = signatures[part]
        numerator, denominator = map(int, time_sig.split('/'))
        if denominator == 8 and numerator % 3 == 0:
            if measure_count % 2 != 0:
                return False
        elif numerator == 2 and measure_count % 2 != 0:
            return False
    return True
    # i.e. returns False ONLY when:
    #   (a) compound meter (denom=8, num%3==0) with ODD measure_count, OR
    #   (b) 2/4 with ODD measure_count.
    # Everything else (including 4/4 with any int >=0, and 0-measures for compound/2-4) returns True.

# enhanced_duration_validator.py — class DurationValidator
class DurationValidator:
    def get_suggested_duration(self, time_signature: str, layer_type: str) -> float: ...
    def get_valid_duration(self, duration: float, time_signature: str,
                           remaining_beats: float, layer_type: str) -> float: ...
```

**Importability prerequisite:** This plan depends on Plan 01 (`__main__` guard). Without it, `import music_gen` would trigger a full song generation as a side effect of `pytest` collection.

**No package install:** Phase 3 introduces `pyproject.toml`. For Phase 1, tests live in a top-level `tests/` directory and use a `conftest.py` `sys.path` shim to import `music_gen` and `enhanced_duration_validator` from the repo root. This is intentional throwaway scaffolding — Phase 3 will replace it with a proper editable install.
</interfaces>
</context>

<tasks>

<task type="auto" tdd="false">
  <name>Task 1: Create dev-requirements.txt and tests/ skeleton (conftest, __init__)</name>
  <files>dev-requirements.txt, tests/__init__.py, tests/conftest.py</files>
  <read_first>
    - .planning/codebase/TESTING.md
    - requirements.txt (to confirm we don't double-list anything)
  </read_first>
  <action>
**Create `dev-requirements.txt`** at repo root with this exact content:

```
# Development-only dependencies (Phase 1 scaffolding).
# Phase 3 will move these into pyproject.toml [project.optional-dependencies].dev
# and this file will be deleted.
-r requirements.txt
pytest>=8.0
pytest-cov>=5.0
```

**Create `tests/__init__.py`** (empty file — just an empty string).

**Create `tests/conftest.py`** with this exact content:

```python
"""
Test configuration for Phase 1 pytest skeleton.

Phase 3 will introduce `pyproject.toml` and a proper `src/musicgen/` package,
at which point this conftest's sys.path shim becomes unnecessary and should
be deleted along with this file.
"""
import os
import sys

# Make the repo root importable so `import music_gen` and
# `import enhanced_duration_validator` work without an editable install.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
```

Then install the dev dependencies so pytest is actually available:

```bash
pip install -r dev-requirements.txt
```

If `pip install` fails because of a sandboxed environment or missing system packages (e.g. `python-magic`, `librosa` native deps), run instead:

```bash
pip install pytest>=8.0 pytest-cov>=5.0
```

— pytest itself has no native deps and will install. The `-r requirements.txt` line in `dev-requirements.txt` is there for users running this on a clean machine; the executor only strictly needs `pytest` to verify Tasks 2 and 3.
  </action>
  <verify>
    <automated>cd /home/openclaw/musicgen &amp;&amp; test -f dev-requirements.txt &amp;&amp; test -f tests/__init__.py &amp;&amp; test -f tests/conftest.py &amp;&amp; grep -q "pytest" dev-requirements.txt &amp;&amp; grep -q "sys.path.insert" tests/conftest.py &amp;&amp; python -c "import pytest; print('pytest', pytest.__version__)"</automated>
  </verify>
  <acceptance_criteria>
    - `dev-requirements.txt` exists, contains `pytest>=8.0` and `pytest-cov>=5.0` and `-r requirements.txt`
    - `tests/__init__.py` exists (may be empty)
    - `tests/conftest.py` exists and contains `sys.path.insert(0, REPO_ROOT)` (or equivalent)
    - `python -c "import pytest"` exits 0 (i.e. pytest is installed in the environment)
    - `cd /home/openclaw/musicgen &amp;&amp; python -c "import sys, os; sys.path.insert(0, os.getcwd()); import music_gen; print('ok')"` exits 0 with no generation output
  </acceptance_criteria>
  <done>Test scaffolding is in place. `pytest` can be invoked. The next two tasks can write actual test files.</done>
</task>

<task type="auto" tdd="false">
  <name>Task 2: Write tests for verify_pattern_for_time_signature, verify_beat_pattern, validate_measures</name>
  <files>tests/test_time_signature.py</files>
  <read_first>
    - music_gen.py lines 22-105 (the three functions under test — the planner has already verified the bodies; assertions below are concrete)
  </read_first>
  <action>
Create `tests/test_time_signature.py` with the EXACT content below. All assertions are concrete and pinned against the verified function bodies (see the `<interfaces>` block above for the verified source code of `verify_beat_pattern` and `validate_measures`). Do NOT introduce `pass` placeholders. Do NOT weaken assertions to `isinstance(result, bool)` — every assertion must use `is True` or `is False`.

```python
"""
Pure-function unit tests for time-signature helpers in music_gen.py.

These tests pin the CURRENT behavior of the functions. Phase 2 will move
them into a TimeSignatureRegistry; the tests should still pass after that
move (delegation, not behavior change).
"""
import pytest

import music_gen


class TestVerifyPatternForTimeSignature:
    """music_gen.verify_pattern_for_time_signature(chord_pattern, time_signature)"""

    # Compound (6/8, 12/8) — valid pattern lengths are {2, 3, 6}
    @pytest.mark.parametrize("length", [2, 3, 6])
    def test_compound_6_8_accepts_2_3_6(self, length):
        pattern = ["I"] * length
        assert music_gen.verify_pattern_for_time_signature(pattern, "6/8") is True

    @pytest.mark.parametrize("length", [1, 4, 5, 7, 8])
    def test_compound_6_8_rejects_other_lengths(self, length):
        pattern = ["I"] * length
        assert music_gen.verify_pattern_for_time_signature(pattern, "6/8") is False

    @pytest.mark.parametrize("length", [2, 3, 6])
    def test_compound_12_8_accepts_2_3_6(self, length):
        pattern = ["I"] * length
        assert music_gen.verify_pattern_for_time_signature(pattern, "12/8") is True

    # 4/4 — valid pattern lengths are {1, 2, 4}
    @pytest.mark.parametrize("length", [1, 2, 4])
    def test_four_four_accepts_1_2_4(self, length):
        assert music_gen.verify_pattern_for_time_signature(["I"] * length, "4/4") is True

    @pytest.mark.parametrize("length", [3, 5, 6, 7])
    def test_four_four_rejects_other_lengths(self, length):
        assert music_gen.verify_pattern_for_time_signature(["I"] * length, "4/4") is False

    # 3/4 — valid pattern lengths are {1, 3}
    @pytest.mark.parametrize("length", [1, 3])
    def test_three_four_accepts_1_and_3(self, length):
        assert music_gen.verify_pattern_for_time_signature(["I"] * length, "3/4") is True

    @pytest.mark.parametrize("length", [2, 4, 5])
    def test_three_four_rejects_other_lengths(self, length):
        assert music_gen.verify_pattern_for_time_signature(["I"] * length, "3/4") is False

    # 2/4 — valid pattern lengths are {1, 2}
    @pytest.mark.parametrize("length", [1, 2])
    def test_two_four_accepts_1_and_2(self, length):
        assert music_gen.verify_pattern_for_time_signature(["I"] * length, "2/4") is True

    @pytest.mark.parametrize("length", [3, 4])
    def test_two_four_rejects_other_lengths(self, length):
        assert music_gen.verify_pattern_for_time_signature(["I"] * length, "2/4") is False

    # Default branch — anything not in the explicit cases returns True
    def test_unknown_signature_defaults_true(self):
        # 5/4 hits the default branch because numerator==5 isn't in {2,3,4} and
        # denominator==4 isn't compound.
        assert music_gen.verify_pattern_for_time_signature(["I", "IV", "V"], "5/4") is True


class TestVerifyBeatPattern:
    """music_gen.verify_beat_pattern(pattern, time_signature)

    Verified body (music_gen.py:42-52): BOTH branches return
    `len(pattern) == numerator`. The conditional is cosmetic. Effective rule:
    len(pattern) must equal the numerator of the time signature.
    """

    # Compound meters — len must equal numerator
    def test_compound_6_8_length_6_ok(self):
        assert music_gen.verify_beat_pattern([1, 0, 1, 0, 1, 0], "6/8") is True

    def test_compound_6_8_length_3_not_ok(self):
        # Compound rule requires len == numerator (6), not numerator/2
        assert music_gen.verify_beat_pattern([1, 0, 1], "6/8") is False

    def test_compound_12_8_length_12_ok(self):
        assert music_gen.verify_beat_pattern([1] * 12, "12/8") is True

    def test_compound_12_8_length_6_not_ok(self):
        assert music_gen.verify_beat_pattern([1] * 6, "12/8") is False

    # Simple meter 4/4 — len must equal 4
    def test_simple_4_4_length_4_ok(self):
        # Verified: denom=4, numerator=4, takes else branch → len(pattern) == 4
        assert music_gen.verify_beat_pattern([1, 0, 1, 0], "4/4") is True

    def test_simple_4_4_length_3_not_ok(self):
        # Verified: [1, 0, 1] has len 3, numerator is 4 → False
        assert music_gen.verify_beat_pattern([1, 0, 1], "4/4") is False

    def test_simple_4_4_length_5_not_ok(self):
        assert music_gen.verify_beat_pattern([1, 0, 1, 0, 1], "4/4") is False

    # Simple meter 3/4 — len must equal 3
    def test_simple_3_4_length_3_ok(self):
        assert music_gen.verify_beat_pattern([1, 0, 0], "3/4") is True

    def test_simple_3_4_length_4_not_ok(self):
        assert music_gen.verify_beat_pattern([1, 0, 0, 0], "3/4") is False


class TestValidateMeasures:
    """music_gen.validate_measures(measures, signatures)

    Verified body (music_gen.py:66-80): returns False ONLY when
      (a) compound meter (denom=8, num%3==0) with ODD measure_count, OR
      (b) 2/4 with ODD measure_count.
    Everything else returns True — including 4/4 with ANY int value (even 0),
    and 0-measures for compound/2-4 since 0 % 2 == 0.
    """

    def test_all_parts_with_positive_measures_valid_signatures_passes(self):
        # 4/4: denom=4 (not 8-compound), numerator=4 (not 2) → neither False
        # branch fires → returns True for any positive int count.
        measures = {"intro": 4, "verse": 8, "chorus": 8, "outro": 4}
        signatures = {"intro": "4/4", "verse": "4/4", "chorus": "4/4", "outro": "4/4"}
        assert music_gen.validate_measures(measures, signatures) is True

    def test_zero_measures_4_4_returns_true(self):
        # 4/4 path doesn't check measure_count at all → always True.
        measures = {"intro": 0}
        signatures = {"intro": "4/4"}
        assert music_gen.validate_measures(measures, signatures) is True

    def test_compound_6_8_even_measures_passes(self):
        # 6/8: denom=8, num=6, 6%3==0 → compound branch.
        # measure_count=4 is even → 4 % 2 == 0 → passes.
        assert music_gen.validate_measures({"verse": 4}, {"verse": "6/8"}) is True

    def test_compound_6_8_odd_measures_fails(self):
        # 6/8 compound with ODD count → returns False.
        assert music_gen.validate_measures({"verse": 3}, {"verse": "6/8"}) is False

    def test_compound_6_8_zero_measures_passes(self):
        # 0 % 2 == 0 → compound branch doesn't trigger False → returns True.
        assert music_gen.validate_measures({"verse": 0}, {"verse": "6/8"}) is True

    def test_two_four_odd_measures_fails(self):
        # 2/4 with odd count hits the `elif numerator == 2` False branch.
        assert music_gen.validate_measures({"intro": 3}, {"intro": "2/4"}) is False

    def test_two_four_even_measures_passes(self):
        assert music_gen.validate_measures({"intro": 4}, {"intro": "2/4"}) is True

    def test_three_four_any_count_passes(self):
        # 3/4: denom=4 (not compound), numerator=3 (not 2) → neither False
        # branch fires → any count returns True.
        assert music_gen.validate_measures({"bridge": 5}, {"bridge": "3/4"}) is True

    def test_empty_measures_dict_returns_true(self):
        # Empty loop body → returns True unconditionally.
        assert music_gen.validate_measures({}, {}) is True
```

**Executor notes:**

1. Every assertion above is concrete and pinned against the verified function bodies in the `<interfaces>` block. Do NOT weaken any assertion. Do NOT add `pass` placeholders. Do NOT add TODO comments saying "encode current behavior" — the behavior is already encoded.

2. After writing the file, run `pytest tests/test_time_signature.py -v` and ensure every test passes. If any test fails, the CORRECT response is to re-read `music_gen.py` lines 42-80 and verify the function body hasn't drifted from what's documented here; if the source has actually changed, update the test to match the new source (do NOT weaken to `isinstance` checks). If the source matches the documented body but the test still fails, something else is off — investigate before changing anything.

3. Tests must NOT touch the filesystem, must NOT call FluidSynth, must NOT import librosa or pedalboard. They are pure-function tests.

4. Total file should be at least 60 lines (the parametrize blocks plus the validate_measures cases get you well past 60).
  </action>
  <verify>
    <automated>cd /home/openclaw/musicgen &amp;&amp; python -m pytest tests/test_time_signature.py -q 2>&amp;1 | tail -20</automated>
  </verify>
  <acceptance_criteria>
    - File `tests/test_time_signature.py` exists with at least 60 lines
    - `pytest tests/test_time_signature.py -q` exits 0 (all tests pass)
    - File contains at least 3 `class Test*:` classes (one per function under test)
    - File contains at least 15 individual test cases (parametrize counts each parameter as one)
    - No `pass` placeholders remain — every test has a real assertion
    - No `isinstance(result, bool)` assertions for `validate_measures` or `verify_beat_pattern` — every assertion uses `is True` or `is False`
    - No `# ← REPLACE` or `# encode current behavior` TODO comments remain
    - No imports of `librosa`, `pedalboard`, `pydub`, or `midi2audio`
  </acceptance_criteria>
  <done>The three pure functions in `music_gen.py` have a regression-detection net with concrete, pinned assertions. Phase 2 can refactor them with confidence.</done>
</task>

<task type="auto" tdd="false">
  <name>Task 3: Write tests for DurationValidator</name>
  <files>tests/test_duration_validator.py</files>
  <read_first>
    - enhanced_duration_validator.py (full file — 171 lines, small)
  </read_first>
  <action>
Create `tests/test_duration_validator.py` with unit tests for `DurationValidator`. Read the full source first, then write tests that pin current behavior of the public methods.

```python
"""
Pure-method unit tests for enhanced_duration_validator.DurationValidator.

Pin the CURRENT behavior of `get_suggested_duration` and `get_valid_duration`
across the supported time signatures (2/4, 3/4, 4/4, 5/4, 6/8, 7/8, 12/8) and
the four layer types (`melody`, `chord`, `bass`, `beat`).
"""
import pytest

from enhanced_duration_validator import DurationValidator, NoteValue


@pytest.fixture
def validator():
    return DurationValidator()


class TestGetSuggestedDuration:
    @pytest.mark.parametrize("time_signature", ["2/4", "3/4", "4/4", "5/4", "6/8", "7/8", "12/8"])
    @pytest.mark.parametrize("layer_type", ["melody", "chord", "bass", "beat"])
    def test_returns_positive_float(self, validator, time_signature, layer_type):
        result = validator.get_suggested_duration(time_signature, layer_type)
        assert isinstance(result, (int, float))
        assert result > 0


class TestGetValidDuration:
    @pytest.mark.parametrize("time_signature", ["2/4", "3/4", "4/4", "6/8"])
    def test_duration_never_exceeds_remaining_beats(self, validator, time_signature):
        # Whatever the proposed duration is, it must be clamped to remaining_beats.
        result = validator.get_valid_duration(
            duration=2.0,
            time_signature=time_signature,
            remaining_beats=0.5,
            layer_type="melody",
        )
        assert result <= 0.5

    @pytest.mark.parametrize("layer_type", ["melody", "chord", "bass", "beat"])
    def test_returns_float_for_each_layer(self, validator, layer_type):
        result = validator.get_valid_duration(
            duration=1.0,
            time_signature="4/4",
            remaining_beats=4.0,
            layer_type=layer_type,
        )
        assert isinstance(result, (int, float))
        assert result > 0


class TestNoteValue:
    def test_note_value_enum_has_expected_members(self):
        # Pin the membership of NoteValue. Read the enum definition before writing
        # this assertion. At minimum, WHOLE / HALF / QUARTER / EIGHTH / SIXTEENTH
        # should exist (verify against source).
        names = {member.name for member in NoteValue}
        # Spot-check a few well-known names
        assert "WHOLE" in names
        assert "QUARTER" in names
```

**Executor notes:**

1. Read `enhanced_duration_validator.py` first. If `get_valid_duration`'s actual signature differs from what's used above (parameter names, order), update the test calls to match the actual signature. Pin behavior to source, not to this template.

2. If `NoteValue` doesn't have `WHOLE` or `QUARTER`, fix the assertion to match the actual member names. Do not change the source.

3. Run `pytest tests/test_duration_validator.py -v` and ensure all tests pass. Then run the full suite: `pytest tests/ -q`.

4. Total file should be at least 40 lines.
  </action>
  <verify>
    <automated>cd /home/openclaw/musicgen &amp;&amp; python -m pytest tests/ -q 2>&amp;1 | tail -20</automated>
  </verify>
  <acceptance_criteria>
    - File `tests/test_duration_validator.py` exists with at least 40 lines
    - `pytest tests/test_duration_validator.py -q` exits 0
    - `pytest tests/ -q` exits 0 (full suite green — no regression in test_time_signature.py)
    - At least 8 individual test cases (counting parametrize expansions)
    - No imports of `librosa`, `pedalboard`, `pydub`, `midi2audio`, `music_gen`
    - Total `pytest tests/ -q` runtime under 10 seconds (the ROADMAP target)
  </acceptance_criteria>
  <done>R-Q2 (initial) is satisfied: the four ROADMAP-named pure functions all have green tests. Phase 2 and Phase 3 inherit a working test harness.</done>
</task>

</tasks>

<verification>
After all three tasks:
1. `pytest tests/ -q` exits 0.
2. `time pytest tests/ -q` shows total runtime &lt; 10 seconds.
3. `tests/` contains: `__init__.py`, `conftest.py`, `test_time_signature.py`, `test_duration_validator.py`.
4. `dev-requirements.txt` exists at repo root.
</verification>

<success_criteria>
- pytest skeleton lands and is green (R-Q2 ✓ initial)
- Four ROADMAP-named pure functions have unit tests
- Test runtime under 10 s
- No audio dependencies pulled in by the tests
</success_criteria>

<output>
Create `.planning/phases/01-stabilize-i-bug-fixes-and-guardrails/01-04-SUMMARY.md` with:
- Final tree of `tests/`
- Output of `pytest tests/ -q` showing N passed
- Total test runtime
- Note that this scaffolding is intentional throwaway — Phase 3 replaces `dev-requirements.txt` with `pyproject.toml`'s `[project.optional-dependencies].dev` and removes the `conftest.py` sys.path shim once `pip install -e .` works.
</output>
