---
phase: 01-stabilize-i-bug-fixes-and-guardrails
plan: 02
subsystem: mixing
tags: [bugfix, pydub, mixing, audio, R-S4, PITFALLS-P-B]
requirements: [R-S4]
requires:
  - "Plan 01-01 (importability + arrangement fix)"
provides:
  - "levels.json values actually applied to mix audio (gain + pan)"
  - "Source-separation contract precondition (per-stem gain/pan matches the mix)"
affects:
  - "music_gen.py::mix_and_save"
  - "All future rendered audio (will differ from pre-fix output)"
tech-stack:
  added: []
  patterns:
    - "pydub apply_gain(dB) capture-return"
    - "pydub pan() capture-return"
    - "Linear-amplitude -> dB conversion via 20*log10(v), floor 1e-6"
key-files:
  created: []
  modified:
    - music_gen.py
decisions:
  - "Interpret levels.json `volume` fields as LINEAR amplitudes (range 0.5-1.0 observed) and convert to dB at apply time"
  - "Clamp with 1e-6 floor before log10 to avoid log(0) if future values include 0.0"
metrics:
  duration: "~5m"
  tasks_completed: 2
  files_modified: 1
  completed: 2026-04-08
---

# Phase 01 Plan 02: Pydub Gain/Pan Fix Summary

One-liner: Fixes the pydub `.volume = ...` / discarded `.pan()` no-op bug so `levels.json` volume and panning values actually affect the rendered mix audio.

## What Changed

Replaced the buggy block at `music_gen.py:847-855` (`mix_and_save`) with a correct capture-return pattern using `apply_gain(dB)` and `pan()`.

### Gain-unit decision

Task 1 inspection of `levels.json`:

- Volume range observed: **0.5 to 1.0**
- Sample values: `[0.8, 0.6, 0.7, 0.9, 0.9, 0.8, 0.6, 0.7]`
- Verdict: **LINEAR amplitudes** (not dB)

Selected **Version B** from the plan: convert to dB via `20 * math.log10(v)` (with a `1e-6` floor to avoid `log(0)` if a future value is ever 0).

### Diff

Before (`music_gen.py:847-855`):

```python
        # Volume and panning for each layer
        beat.volume = float(levels[part]['beat']['volume'])
        melody.volume = float(levels[part]['melody']['volume'])
        harmony.volume = float(levels[part]['harmony']['volume'])
        bassline.volume = float(levels[part]['bassline']['volume'])
        beat.pan(float(levels[part]['beat']['panning']))
        melody.pan(float(levels[part]['melody']['panning']))
        harmony.pan(float(levels[part]['harmony']['panning']))
        bassline.pan(float(levels[part]['bassline']['panning']))
```

After:

```python
        # Volume and panning for each layer (R-S4 / PITFALLS P-B fix).
        # `.volume =` was a no-op (read-only property); `.pan()` returns a new
        # segment so its return must be captured. Values in levels.json are
        # linear amplitudes; convert to dB via 20*log10(v) (clamped to avoid log(0)).
        def _lin_to_db(v: float) -> float:
            return 20.0 * math.log10(max(float(v), 1e-6))
        beat = beat.apply_gain(_lin_to_db(levels[part]['beat']['volume']))
        melody = melody.apply_gain(_lin_to_db(levels[part]['melody']['volume']))
        harmony = harmony.apply_gain(_lin_to_db(levels[part]['harmony']['volume']))
        bassline = bassline.apply_gain(_lin_to_db(levels[part]['bassline']['volume']))
        beat = beat.pan(float(levels[part]['beat']['panning']))
        melody = melody.pan(float(levels[part]['melody']['panning']))
        harmony = harmony.pan(float(levels[part]['harmony']['panning']))
        bassline = bassline.pan(float(levels[part]['bassline']['panning']))
```

## Tasks Executed

| Task | Name                                                              | Commit    | Files         |
| ---- | ----------------------------------------------------------------- | --------- | ------------- |
| 1    | Inspect levels.json to determine gain unit (dB vs linear)         | (no-op)   | levels.json (read only) |
| 2    | Replace .volume assignments + capture .pan() returns in mix_and_save | c6a257b | music_gen.py  |

Task 1 was read-only analysis; its finding was applied directly in the Task 2 edit (no separate commit, which matches the plan's instruction to "embed in Task 2's edit" and "Do NOT modify any file in this task").

## Verification Results

- `grep -nE "(beat|melody|harmony|bassline)\.volume\s*=" music_gen.py` → no matches
- `grep -c "apply_gain" music_gen.py` → `4`
- AST check: every `.pan(` call inside `mix_and_save` is on the right-hand side of `=` (capture-return pattern) — PASS
- `python3 -m py_compile music_gen.py` → syntax OK
- Block contains the R-S4 / PITFALLS P-B comment — PASS

## Deviations from Plan

None. Plan executed exactly as written. Version B selected per the plan's conditional instruction after Task 1's linear-unit verdict.

## Deferred Issues (Out of Scope)

- `python3 -c "import music_gen"` fails with `ModuleNotFoundError: No module named 'music21'` in the runner environment. This is a **pre-existing environment problem**, not a regression from this plan: the source compiles cleanly (`py_compile` passes), and the plan's explicit regression check ("still exits 0") assumes `music21` is installed. Logged in `deferred-items.md`. Does not affect the correctness of this plan — R-S4 is satisfied at the source level.

## Impact Notes

- **Every existing seed will now produce different audio.** This is expected and intentional — the previous output was silently ignoring `levels.json`. Phase 5 will pin a "post-fix golden" for the determinism regression test.
- **R-S4 satisfied**: `levels.json` volume and panning values now affect the final mix audio.
- **PITFALLS P-B closed**: the read-only `.volume` property and discarded `.pan()` returns are gone.
- **Phase 1 exit criterion #2 unblocked**: "seeded run produces mix audio that reflects levels.json."
- **Source-separation contract (Phase 5) unblocked**: per-stem gain/pan will now match the mix by construction.

## Self-Check: PASSED

- music_gen.py exists and contains the new block: FOUND
- Commit c6a257b exists: FOUND
- No `.volume =` assignments on layer vars: CONFIRMED
- 4 `apply_gain` calls present: CONFIRMED
