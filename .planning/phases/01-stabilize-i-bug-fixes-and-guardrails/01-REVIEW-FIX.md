---
phase: 01-stabilize-i-bug-fixes-and-guardrails
fixed_at: 2026-04-08T00:00:00Z
review_path: .planning/phases/01-stabilize-i-bug-fixes-and-guardrails/01-REVIEW.md
iteration: 1
findings_in_scope: 5
fixed: 4
skipped: 1
status: partial
---

# Phase 01: Code Review Fix Report

**Fixed at:** 2026-04-08
**Source review:** .planning/phases/01-stabilize-i-bug-fixes-and-guardrails/01-REVIEW.md
**Iteration:** 1

**Summary:**
- Findings in scope (Critical + Warning): 5
- Fixed: 4
- Skipped: 1
- Pinned tests (95) pass after every fix.

## Fixed Issues

### WR-01: Duplicate `random.choice` re-rolls chord pattern

**Files modified:** `music_gen.py`
**Commit:** 5a84dba
**Applied fix:** Removed the duplicate `chord_pattern = random.choice(chord_patterns[part])` call at line 207, keeping a single draw. This restores a stable RNG-consumption pattern and eliminates the silent re-roll that decoupled downstream state.
**Verification:** Tier 1 re-read, Tier 2 `ast.parse`, Tier 3 `pytest tests/ -q` -> 95 passed.

### WR-02: `generate_melody` / `generate_bassline` advance time by stale `note_duration`

**Files modified:** `music_gen.py`
**Commit:** ab03df0
**Applied fix:** Changed `time += note_duration` to `time += note_durations[i]` in `generate_melody` (line 344) and `current_time += note_duration` to `current_time += note_durations[i]` in `generate_bassline` (line 454). Both loops now use the per-note duration captured during generation instead of the stale loop-variable value.
**Verification:** Tier 1 re-read, Tier 2 `ast.parse`, Tier 3 `pytest tests/ -q` -> 95 passed.
**Note:** This is a behavior-changing bugfix for generated MIDI output. The pinned tests in `tests/` do not cover `generate_melody` / `generate_bassline`, so pinning did not regress, but downstream MIDI timing will differ from the previous broken behavior. This is the intended correct behavior.

### WR-04: `generate_random_key` / `generate_random_tempo` can return `None`

**Files modified:** `music_gen.py`
**Commit:** 2a02af2
**Applied fix:** Added explicit fallback `return key_ranges[-1][1]` after the `generate_random_key` loop and `return random.randint(*tempo_ranges[-1][1:])` after the `generate_random_tempo` loop. Today both final weights are exactly `1.0`, so the fallback is unreachable and this fix is strictly defensive -- no behavior change in current runs.
**Verification:** Tier 1 re-read, Tier 2 `ast.parse`, Tier 3 `pytest tests/ -q` -> 95 passed.

### WR-05: `mix_and_save` has no fallback if `song_parts` is empty

**Files modified:** `music_gen.py`
**Commit:** 9dbad21
**Applied fix:** Added `if not song_parts: raise RuntimeError("No song parts were rendered; cannot assemble final mix")` immediately before `song = AudioSegment.from_wav(song_parts[0])`. Converts a latent `IndexError` into a clear, local contract violation. Unreachable today (default-structure fallback in `generate_song_arrangement` prevents empty lists), so no behavior change on happy path.
**Verification:** Tier 1 re-read, Tier 2 `ast.parse`, Tier 3 `pytest tests/ -q` -> 95 passed.

## Skipped Issues

### WR-03: `chord_obj` leaks out of preceding loop in melody transition matrix

**File:** `music_gen.py:294-301`
**Reason:** skipped: fix is behavior-changing and explicitly deferred to Phase 2. The reviewer's Fix text itself says "Defer to Phase 2's planned refactor." Phase 01 is a stabilization phase; the pinned test suite (95 tests) does not cover `generate_melody`'s transition matrix, so there is no regression safety net for this change. Any fix (either iterating over all chords per step or using the union of all chord pitches) alters the Markov transition weights and therefore alters generated melodies in an RNG-observable way. Applying it now would silently change generator output without the behavioral pins needed to verify intent. Leaving the latent bug in place for Phase 2 to address alongside the planned melody/bassline refactor, which is explicitly where the reviewer recommended it go.
**Original issue:** The transition matrix builder references `chord_obj` which is the loop variable left over from the `for chord_symbol in chord_progression:` loop on lines 277-280. So every entry in the matrix is "is `next_note` in the LAST chord of the progression?" instead of "is `next_note` in any active chord?". The melody Markov chain therefore biases toward only the final chord's notes regardless of which chord the song is currently on.

---

_Fixed: 2026-04-08_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
