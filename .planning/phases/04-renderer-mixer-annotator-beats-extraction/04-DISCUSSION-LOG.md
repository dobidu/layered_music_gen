# Phase 4: Renderer + mixer + annotator + beats extraction - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-19
**Phase:** 04-renderer-mixer-annotator-beats-extraction
**Mode:** `--auto` (recommended defaults selected non-interactively)
**Areas discussed:** Module split, Renderer design, Mixer design, Annotator schema, RNG threading, Beats derivation, Orchestrator shape, Testing strategy, Parallelization

---

## Module split + file inventory

| Option | Description | Selected |
|--------|-------------|----------|
| Four new modules under `src/musicgen/` (renderer, mixer, annotator, beats) | Matches research/ARCHITECTURE.md layout exactly; one responsibility per module | ✓ |
| Collapse renderer+mixer into one `audio.py` | Fewer files but violates the one-responsibility-per-module pattern from Phase 3 | |
| Split further (renderer, fx, mixer, concat, annotator, beats) | Six files; more granular but introduces `fx.py` that just re-exports pedalboard helpers | |

**User's choice:** `--auto` — four modules, mirrors ARCHITECTURE.md.
**Notes:** D-01/D-02 in CONTEXT.md. `RenderResult` lives in renderer.py, `MixResult` in mixer.py; dataclass-per-producer.

---

## `beat_anotator.py` treatment

| Option | Description | Selected |
|--------|-------------|----------|
| Delete outright, no shim | No importers exist; its straight-grid logic is actively wrong on swing > 0 | ✓ |
| Keep as CLI-only tool, deprecate the straight-grid code | Preserves the `python beat_anotator.py <dir>` entry point | |
| Replace body with re-export of `musicgen.beats` | Keeps file path, forwards calls | |

**User's choice:** `--auto` — delete.
**Notes:** D-03. Zero importers. Phase 5 writer owns the annotation lifecycle end-to-end; standalone entry point is obsolete.

---

## Renderer — FluidSynth parallelism

| Option | Description | Selected |
|--------|-------------|----------|
| `ThreadPoolExecutor(max_workers=4)` for per-part stems | FluidSynth is subprocess; threads suffice; matches research/STACK.md | ✓ |
| `ProcessPoolExecutor` | Adds fork overhead, no determinism gain | |
| Serial rendering | Simpler but slower | |

**User's choice:** `--auto` — ThreadPoolExecutor.
**Notes:** D-05/D-06. Outer per-sample ProcessPoolExecutor is Phase 6 R-P10 and NOT added here.

---

## FluidSynth version capture

| Option | Description | Selected |
|--------|-------------|----------|
| Module-level `FLUIDSYNTH_VERSION` from `subprocess.run(["fluidsynth", "--version"])` at import | Captured once, available for annotator; `"unknown"` fallback on failure | ✓ |
| Per-render capture inside `render_stems` | N subprocess calls per sample; wasteful | |
| Configured via `Config.fluidsynth_version` (manual) | Relies on user to keep in sync | |

**User's choice:** `--auto` — module-level at import.
**Notes:** D-07. Does NOT raise at import; CI-friendly.

---

## Annotator schema coverage in Phase 4

| Option | Description | Selected |
|--------|-------------|----------|
| Fill every field Phase 4 has inputs for; Phase-5 fields = `None` | Locks schema shape early; Phase 5 only fills, doesn't restructure | ✓ |
| Minimal annotator (only key/tempo/time_sig); expand in Phase 5 | Less risk but Phase 5 does more structural work | |
| Skip annotator entirely; land in Phase 5 | Violates R-X6 which is in Phase 4 scope | |

**User's choice:** `--auto` — full coverage with Phase-5 TBD fields as `None`.
**Notes:** D-14/D-15.

---

## TBD-flag representation

| Option | Description | Selected |
|--------|-------------|----------|
| `None` values in the dict (key present) | Python/JSON nullable idiom; trivially updated by Phase 5 | ✓ |
| Missing keys (Phase 5 adds them) | Forces consumer code to handle KeyError | |
| Literal `"TBD"` string placeholder | Type-unsafe (would change from string to int) | |
| Sentinel object (e.g., `TBD = object()`) | Not JSON-serializable | |

**User's choice:** `--auto` — `None`.
**Notes:** D-16.

---

## Layer-inclusion RNG placement

| Option | Description | Selected |
|--------|-------------|----------|
| Mixer owns `compute_layer_mask(...)` with injected `rng` | Layer-inclusion is a mix-time probability, not a song-structure decision | ✓ |
| Move to sampler (co-locate with arrangement sampling) | Would disrupt RNG draw order committed in Plan 02-02 A3 / Phase 3 D-21 | |
| Keep in orchestrator | Leaks mix logic into music_gen.py | |

**User's choice:** `--auto` — mixer owns it.
**Notes:** D-13. Preserves Phase 3 RNG-order commitment.

---

## FX application scope

| Option | Description | Selected |
|--------|-------------|----------|
| Apply FX to all four layers regardless of mix inclusion (preserve current behavior) | Locks RNG draw order; defers the optimization safely | ✓ |
| Apply FX only to included layers (the long-standing TODO at `music_gen.py:276`) | Better efficiency but changes RNG consumption and breaks Phase 5 golden baseline | |

**User's choice:** `--auto` — preserve current behavior.
**Notes:** D-11. Phase 5+ can optimize after baselines are locked.

---

## Beats derivation approach

| Option | Description | Selected |
|--------|-------------|----------|
| MIDI-tick extraction via `mido.tick2second` (per R-X7 roadmap + ARCHITECTURE.md) | Swing-aware by construction (onsets are already swung in the MIDI) | ✓ |
| Theoretical-grid derivation (current `beat_anotator.py` approach) | Actively wrong on swing > 0 (PITFALLS P-3) | |
| Audio-onset detection via `librosa.onset.onset_detect` on the rendered WAV | Expensive, imprecise, and couples beats to audio path | |

**User's choice:** `--auto` — MIDI-tick extraction.
**Notes:** D-19.

---

## Downbeat derivation

| Option | Description | Selected |
|--------|-------------|----------|
| `beat_times[::numerator]` (group beats by numerator, stride-slice for downbeats) | Simple; validated by test fixture; falls back to registry `primary_beat_group` if assumption breaks | ✓ |
| Parse MIDI time-signature meta-events directly | More authoritative but couples to `mido` internals | |
| Extract from `TimeSignatureRegistry.lookup(sig).downbeat_pattern` | Requires registry extension not in scope | |

**User's choice:** `--auto` — stride-slice with test fixture validation.
**Notes:** D-20.

---

## Orchestrator shape post-phase

| Option | Description | Selected |
|--------|-------------|----------|
| Delete `mix_and_save` entirely; distribute body across renderer + mixer + orchestrator | Cleanest reading of the roadmap "< 50 lines of orchestration" exit criterion | ✓ |
| Keep `mix_and_save` as a ~40-line compose function | Preserves external signature but adds indirection | |
| Split into `mix_parts` + `concat_parts` but keep the name | Half-measure; still fuses responsibilities | |

**User's choice:** `--auto` — delete.
**Notes:** D-23/D-24. `music_gen.py` collapses to ~180 lines of pure orchestration.

---

## `musicality_score.py` location

| Option | Description | Selected |
|--------|-------------|----------|
| Keep at repo root this phase; move to package in Phase 5 | Annotator takes score dict as parameter, does not import musicality_score — no blocker | ✓ |
| Move to `src/musicgen/scoring.py` this phase | Adds churn for no Phase 4 consumer benefit | |

**User's choice:** `--auto` — defer to Phase 5.
**Notes:** D-04. Phase 3 D-11 already deferred this; re-deferred.

---

## Parallelization with Phase 3

| Option | Description | Selected |
|--------|-------------|----------|
| Run Phase 4 serially after Phase 3 | Phase 3 is COMPLETE; serial keeps the session simple | ✓ |
| Attempt Phase 3 ∥ Phase 4 per roadmap | Phase 3 already shipped; moot | |

**User's choice:** `--auto` — serial.
**Notes:** D-33.

---

## Testing strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Fixture-driven unit tests per module + one `@pytest.mark.slow` E2E integration test | Matches roadmap deliverable list exactly | ✓ |
| Full integration tests, skip unit tests | Fast but gives up regression coverage | |
| Heavy property-based tests (hypothesis) | Out of scope for the v0.1 milestone | |

**User's choice:** `--auto` — roadmap-matching test list.
**Notes:** D-26 through D-32. Annotator: fixture-driven. Beats: 3 swing cases (0.5 / 0.66 / 0.75). Renderer: mock `FluidSynth.midi_to_audio`. Mixer: seeded-RNG tests. E2E: 1 slow test guarded on FluidSynth binary. AST guard extended to scan `src/musicgen/**/*.py`.

---

## Auto-Resolved (summary)

All 13 gray areas above were auto-resolved in `--auto` mode with the recommended default. No corrections requested. No Unclear items required escalation because every decision had strong evidence from prior CONTEXT.md files, the roadmap, or research/ARCHITECTURE.md.

## External Research

No external research was performed — every decision is backed by existing `.planning/` artifacts (ROADMAP, REQUIREMENTS, PROJECT, codebase/*, research/*, prior phase CONTEXT files) and direct codebase inspection.

## Deferred Ideas

See CONTEXT.md `<deferred>` section. 10 items deferred, primarily to Phase 5 (writer/seed discipline/pre-roll) and Phase 6 (batch parallelism).

## Scope Creep Redirected

None encountered during this discussion.
