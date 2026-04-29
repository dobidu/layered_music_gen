# Phase 3 — Deferred Items (Out-of-Scope Findings)

Findings discovered during Plan 03-05 execution that are NOT caused by Phase 3
changes and are therefore out of scope per the executor's scope-boundary rule.

## [Deferred] Pre-existing Markov melody zero-weights bug

- **File:** `src/musicgen/generators/melody.py:110` (formerly `music_gen.py` line ~317 in commit `94c19a0`)
- **Symptom:** `ValueError: Total of weights must be greater than zero` from `rng.choices(population=..., weights=...)` when the current `current_note`'s row in `transition_matrix` has all-zero entries. This happens when `current_note` (a MIDI int drawn from `notes_to_use`) is not among any of the chord's pitches, so every successor transition weight is 0.
- **Root cause (design-level):** The Markov transition matrix sets weight = 0 for any `next_note` not in `chord_obj.pitches`. If `current_note` itself is also not in `chord_obj.pitches`, every outbound weight is 0 and `random.choices` / `rng.choices` refuses to sample.
- **Pre-existing:** Verified in git at commit `94c19a0` (original repo upload) — the call `current_note = random.choices(population=..., weights=...)[0]` exists verbatim with the identical logic. Phase 3's extraction into `generators/melody.py` (Plan 03-04) rewrote `random.*` → `rng.*` but did not change the Markov construction. **Not a Phase 3 regression.**
- **Manifestation frequency:** Non-deterministic under pytest (caught by seeded tests in `test_generators/` that use controlled seeds) but triggers intermittently in `python music_gen.py` smoke runs depending on which chord progression + `notes_to_use` combination is sampled. Two of three consecutive smoke runs failed here; the third reached `mix_and_save` (environmental ffmpeg/soundfont failure).
- **Recommended fix (future phase):** Either
  1. Add a fallback branch: when the weights row sums to 0, uniform-sample from the chord_obj.pitches instead, or
  2. Make the transition matrix symmetric / always include self-loops with non-zero weight, or
  3. Seed `current_note` from `chord_obj.pitches` (not `notes_to_use`) so it's always in-chord.
- **Phase-5 relevance:** The fix must preserve the seeded-RNG draw order to keep Phase 5's determinism baseline stable. Consider landing the fix in Phase 4 (mix-and-save refactor) or Phase 5 (seed discipline) with an accompanying determinism-baseline update.
- **Status:** LOGGED, NOT FIXED in Plan 03-05 (scope-boundary rule — pre-existing bug not caused by current task).
