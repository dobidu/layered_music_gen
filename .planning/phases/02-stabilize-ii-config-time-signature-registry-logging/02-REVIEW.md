---
phase: 02-stabilize-ii-config-time-signature-registry-logging
reviewed: 2026-04-18T00:00:00Z
depth: standard
files_reviewed: 7
files_reviewed_list:
  - config.py
  - enhanced_duration_validator.py
  - music_gen.py
  - tests/test_config.py
  - tests/test_music_gen_logging.py
  - tests/test_timesig_registry.py
  - timesig.py
findings:
  critical: 2
  warning: 5
  info: 4
  total: 11
status: issues_found
---

# Phase 02: Code Review Report

**Reviewed:** 2026-04-18
**Depth:** standard
**Files Reviewed:** 7
**Status:** issues_found

## Summary

This phase introduces three subsystems: a centralized `Config` dataclass with three-layer override precedence (D-01/D-02), an immutable `TimeSignatureRegistry` (R-S6), and logging migration (R-S7). The registry design is solid — frozen dataclasses, single-source-of-truth dispatch, exhaustive test coverage across all 7 signatures. Config also has good test coverage and the env/CLI precedence logic is correct.

The critical issues are both in `music_gen.py:generate_melody` and involve the Markov chain transition matrix. A stale loop variable (`chord_obj`) causes the matrix to use the last chord's pitches as the weight criterion for all rows. For multi-chord verse/chorus progressions this produces all-zero weight rows, which causes `random.choices` to raise `ValueError` at runtime. This is a latent crash path that would fire on most non-trivial songs.

A secondary cluster of warnings covers: a broken `validate_layer_duration` call that validates a spec list instead of actual note durations (always logs a spurious warning), a `save_beat_annotations` function with a type error that would crash if called, a missing beat-roll pattern file for `5/4` time, and inconsistency in how the config exposes unsupported time signatures.

---

## Critical Issues

### CR-01: Markov transition matrix all-zero weights — crash in `generate_melody`

**File:** `music_gen.py:195-216`

**Issue:** The transition matrix is built in a loop over `notes_to_use` (all pitches from all chords in the progression). At line 198 the weight predicate is `if next_note in chord_obj.pitches`, but `chord_obj` is the loop variable from the preceding `for chord_symbol in chord_progression` loop (lines 177-180) — it is frozen to the **last** chord after that loop exits. For any note that is not a member of the final chord, every `next_note` in its row gets weight `0`, so the entire row is `{midi: 0, midi: 0, ...}`. When the melody generation loop (line 213) picks one of those notes as `current_note` and calls `random.choices(population=..., weights=[0, 0, 0, ...])`, Python raises `ValueError: Total of weights must be greater than zero`.

This crash path is triggered reliably for standard 4-chord progressions (`['I', 'IV', 'V', 'vi']`) in `verse` and `chorus` parts (the `else` branch at line 188 collects pitches from all chords, then the matrix is built with the wrong reference chord).

**Fix:**
```python
# Before (line 195-201) — stale chord_obj reference:
for note in notes_to_use:
    transition_matrix[note.midi] = {}
    for next_note in notes_to_use:
        if next_note in chord_obj.pitches:  # BUG: chord_obj is the last chord only
            transition_matrix[note.midi][next_note.midi] = 1 / len(notes_to_use)
        else:
            transition_matrix[note.midi][next_note.midi] = 0

# After — uniform weights give each candidate equal probability:
for note in notes_to_use:
    transition_matrix[note.midi] = {
        next_note.midi: 1 / len(notes_to_use)
        for next_note in notes_to_use
    }
```

If the chord-membership weighting is intentional, the check must be applied per-row using the chord whose root pitch is closest to `note`, not a single global `chord_obj`.

---

### CR-02: `save_beat_annotations` — `ValueError` on format string (`List[str]` passed to `:.3f`)

**File:** `music_gen.py:497-506`

**Issue:** `generate_beat` returns `annotations` as `List[str]` where each element is already formatted as `"0.000\t1"` (line 482: `f"{actual_time:.3f}\t{len(annotations) + 1}"`). `save_beat_annotations` then iterates `beat_annotations.items()` and does:

```python
timestamps = [f"{timestamp:.3f}" for timestamp in annotations]
```

Applying `:.3f` to a `str` raises `ValueError: Unknown format code 'f' for object of type 'str'`. The function is currently not called anywhere in the codebase (dead code), so this does not crash production today, but it will fail immediately if wired up.

**Fix — option A** (annotations stay as strings, drop redundant formatting):
```python
def save_beat_annotations(name, beat_annotations):
    output_file = os.path.join(name, f"{name}-beats.txt")
    with open(output_file, 'w') as f:
        for part, annotations in beat_annotations.items():
            f.write(f"{part}: {', '.join(annotations)}\n")
    logger.info("Beat annotations saved to: %s", output_file)
```

**Fix — option B** (change `generate_beat` to return `List[float]` timestamps and fix the path too):
```python
annotations.append(actual_time)  # store float, not formatted string
```

Note: the path construction also has a bug — see WR-04.

---

## Warnings

### WR-01: `validate_layer_duration` called with spec candidate list instead of generated note durations

**File:** `music_gen.py:220-236`

**Issue:** After the melody generation `while` loop, the validation check at line 236 is:

```python
if not validator.validate_layer_duration(possible_durations, time_signature, 'melody'):
```

`possible_durations` holds the last value assigned inside the loop body — it is the list of **spec candidate durations** (e.g., `[0.5, 1.0, 2.0]` for 4/4), not the `note_durations` list of actually generated durations. `validate_layer_duration` sums its first argument and checks divisibility by `beats_per_measure`. For 4/4: `sum([0.5, 1.0, 2.0]) = 3.5`, which is never divisible by 4, so this warning fires on **every** 4/4 melody regardless of whether the timing is actually correct. The actual note durations are in the list `note_durations`.

**Fix:**
```python
# Line 236 — replace possible_durations with note_durations:
if not validator.validate_layer_duration(note_durations, time_signature, 'melody'):
    logger.warning("Generated melody has invalid timing structure")
```

---

### WR-02: `5/4` time signature registered in `TimeSignatureRegistry` but missing beat-roll pattern file

**File:** `config.py:34-41` / `timesig.py:185-205`

**Issue:** `TimeSignatureRegistry` includes `5/4` (5% sampling weight, ~1 in 20 songs). `DEFAULT_BEAT_ROLL_PATTERN_FILES` has no `5/4` entry. When `generate_beat` is called for a `5/4` part, `beat_pattern_files.get("5/4")` returns `None` and the function raises `ValueError: Time signature 5/4 not supported.` This turns a random 5% of song generations into crashes.

**Fix:** Either add `beat_roll_patterns_54.txt` to the project and register it in `DEFAULT_BEAT_ROLL_PATTERN_FILES`, or remove `5/4` from the registry's sampling pool (set `sampling_weight=0.0` and redistribute weight to other signatures) until the file is created. The registry already supports `5/4` for chord/beat pattern validation, so the fix is purely the missing data file + config entry:

```python
# config.py DEFAULT_BEAT_ROLL_PATTERN_FILES:
"5/4":  os.path.join(DEFAULT_PROJECT_ROOT, "beat_roll_patterns_54.txt"),
```

---

### WR-03: Redundant double call to `get_melody_durations` inside melody generation loop

**File:** `music_gen.py:220-222`

**Issue:** Lines 220 and 222 both call `get_melody_durations(time_signature)` on every loop iteration. Line 221 is a commented-out version of the cleaner form that uses `possible_durations`:

```python
possible_durations = get_melody_durations(time_signature)
# raw_duration = random.choice(possible_durations)      # ← commented out
raw_duration = random.choice(get_melody_durations(time_signature))  # ← live duplicate
```

The second call is redundant, and the commented-out line is dead code left over from an incomplete refactor. The fix for WR-01 also resolves this: use `possible_durations` for `raw_duration` and `note_durations` for the validation call.

**Fix:**
```python
possible_durations = get_melody_durations(time_signature)
raw_duration = random.choice(possible_durations)
# remove the commented-out line
```

---

### WR-04: `save_beat_annotations` path construction loses the song directory

**File:** `music_gen.py:498-500`

**Issue:** `save_beat_annotations` builds the output path with:

```python
instance_dir = os.path.dirname(name)
output_file = os.path.join(instance_dir, f"{name}-beats.txt")
```

`name` is the song base name (e.g., `"20240101120000_abc1"`). `os.path.dirname` of a plain filename with no directory component returns `""`, so `output_file` resolves to `"20240101120000_abc1-beats.txt"` in the current working directory instead of the song's subdirectory. Every other file-writing function in `music_gen.py` uses `name.split('-')[0]` or `os.path.join(name, ...)` to place output inside the song directory.

**Fix:**
```python
def save_beat_annotations(name, beat_annotations):
    output_file = os.path.join(name, f"{name}-beats.txt")  # name is the song dir
    with open(output_file, 'w') as f:
        ...
```

---

### WR-05: `Config.beat_pattern_file` raises bare `KeyError` for unsupported time signatures

**File:** `config.py:64-66`

**Issue:** `beat_pattern_file(time_signature)` does `return self.beat_roll_pattern_files[time_signature]`, which raises a bare `KeyError` with only the key as the message if the signature is not found. `generate_beat` handles this case gracefully via `.get()` with an explicit `ValueError` and a helpful message. The inconsistency means callers using `Config.beat_pattern_file` directly (e.g., future CLI tooling) will get an opaque `KeyError: '5/4'` rather than a diagnostic message.

**Fix:**
```python
def beat_pattern_file(self, time_signature: str) -> str:
    """Return the beat-roll pattern file path for a time signature string like '4/4'."""
    try:
        return self.beat_roll_pattern_files[time_signature]
    except KeyError:
        raise KeyError(
            f"No beat-roll pattern file registered for time signature {time_signature!r}. "
            f"Registered: {list(self.beat_roll_pattern_files)}"
        ) from None
```

---

## Info

### IN-01: `note_durations` dict field in `TimeSignatureSpec` is mutable despite `frozen=True`

**File:** `timesig.py:46`

**Issue:** `frozen=True` on the dataclass prevents reassignment of fields (`spec.note_durations = ...` raises `FrozenInstanceError`) but does **not** prevent in-place mutation of the dict itself (`spec.note_durations["whole"] = 999` silently corrupts the shared `_SIMPLE_NOTE_DURATIONS` / `_COMPOUND_NOTE_DURATIONS` module-level dicts, which are shared across all specs). This is documented in the class docstring but is a latent correctness risk. The safe public API `note_duration_map()` returns a copy, so current consumers are protected. No callers currently access `spec.note_durations` directly outside of `timesig.py` itself.

**Suggested improvement:** Convert to `types.MappingProxyType` at construction time, or change the field type to `Tuple[Tuple[str, float], ...]` to make immutability structural.

---

### IN-02: f-string in `DurationValidator.validate_layer_duration` logger call

**File:** `enhanced_duration_validator.py:123`

**Issue:** `self.logger.warning(f"{layer_type} layer duration does not complete full measures")` uses an f-string. The project convention (enforced by `test_no_fstring_in_logger_calls` in `test_music_gen_logging.py`) is to use `%s` format args. The test only scans `music_gen.py`, so this instance is not caught.

**Fix:**
```python
self.logger.warning("%s layer duration does not complete full measures", layer_type)
```

---

### IN-03: Commented-out code left in `generate_melody`

**File:** `music_gen.py:221`

**Issue:** `# raw_duration = random.choice(possible_durations)` is dead code from an incomplete refactor. After WR-03 is resolved this line should be deleted entirely.

---

### IN-04: `Config.load` cli_overrides applies values without type validation

**File:** `config.py:89-97`

**Issue:** The cli_overrides loop calls `setattr(cfg, key, value)` for any recognized key without checking that the value type matches the dataclass field annotation. For example, passing `cli_overrides={"sf_layers": "single-string"}` would silently replace a `Tuple[str, ...]` field with a `str`. This is noted in the Phase 6 roadmap and is low risk now (no CLI caller exists), but worth documenting as a pre-condition for Phase 6 wiring.

---

_Reviewed: 2026-04-18_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
