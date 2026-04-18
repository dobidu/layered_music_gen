# Phase 3: Package skeleton + sampler + generators extraction - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in `03-CONTEXT.md` — this log preserves the alternatives considered.

**Date:** 2026-04-18
**Phase:** 03-package-skeleton-sampler-generators-extraction
**Mode:** --auto (recommended defaults auto-selected, no interactive questioning)
**Areas discussed:** Package layout, Back-compat shim strategy, RNG plumbing depth, DurationValidator/scoring placement, pyproject.toml scope, Test migration, CLI placeholder scaffolding, SongParams shape, music21 global RNG audit, Parallelization

---

## Area 1 — Package layout

| Option | Description | Selected |
|--------|-------------|----------|
| src/ layout (`src/musicgen/`) | Modern, isolates package from tests/docs, works with hatchling out of the box | ✓ |
| Flat layout (`musicgen/` at repo root) | Less plumbing but mixes package with root-level helper scripts and data files | |

**User's choice:** src/ layout (auto-recommended). Matches `.planning/research/ARCHITECTURE.md` §Proposed module layout.

---

## Area 2 — Back-compat shim strategy for `music_gen.py`

| Option | Description | Selected |
|--------|-------------|----------|
| (A) Keep music_gen.py behavior intact; new modules shadow it via re-exports | Minimum diff, but double maintenance | |
| (B) Replace moved functions with re-imports from musicgen.*; music_gen.py becomes a thin shim | Keeps `python music_gen.py` working, honors the roadmap exit criterion, single-source of truth is now the package | ✓ |
| (C) No shim; update everything including beat_anotator + tests to import from package | Cleanest but largest diff; violates "old music_gen.py still executable for smoke testing" | |

**User's choice:** (B) (auto-recommended). Phase 3 exit criterion requires `python music_gen.py` still works.

---

## Area 3 — RNG plumbing depth

| Option | Description | Selected |
|--------|-------------|----------|
| (A) All-in: every extracted function takes `rng: random.Random`; zero bare `random.*` in extracted code | Phase 5 just builds the RNG hierarchy; signatures are ready | ✓ |
| (B) Minimal: generators accept `Optional[random.Random] = None`, fall back to module-level random | Less disruptive but requires another refactor pass in Phase 5 | |
| (C) Single default `_rng` at module scope, not yet injected | Worst of both — no new capability, still needs Phase 5 refactor | |

**User's choice:** (A) (auto-recommended). Roadmap explicitly says "Functions take an explicit `rng: random.Random` parameter — no use of the module-level `random`."

---

## Area 4 — DurationValidator + musicality_score placement

| Option | Description | Selected |
|--------|-------------|----------|
| (A-narrow) Move DurationValidator into package; leave musicality_score for Phase 4 | Matches what generators actually need this phase | ✓ |
| (A-wide) Move both DurationValidator and musicality_score into the package this phase | Cleaner end state, more churn this phase, no immediate Phase 3 consumer for musicality_score in the package | |
| (B) Leave both at repo root; import across the boundary during Phase 3 | Awkward cross-root imports from package code | |

**User's choice:** (A-narrow) (auto-recommended). Only move what this phase's extracted code imports.

---

## Area 5 — pyproject.toml scope

| Option | Description | Selected |
|--------|-------------|----------|
| (A) Full migration: delete requirements.txt + dev-requirements.txt; pyproject.toml is single source | Aligns with the modern pip-installable package contract; Plan 01-04 already flagged dev-requirements.txt as throwaway | ✓ |
| (B) Keep requirements.txt alongside pyproject.toml | Duplicate lists diverge in practice | |
| (C) Keep requirements.txt, skip pyproject.toml | Violates R-X1 ("pyproject.toml with hatchling backend") | |

**User's choice:** (A) (auto-recommended).

---

## Area 6 — Test migration strategy

| Option | Description | Selected |
|--------|-------------|----------|
| (A) Rewrite tests to import from musicgen.* where code moved; delete conftest.py sys.path shim | Matches Plan 01-04 commitment that shim is Phase-3-throwaway | ✓ |
| (B) Keep tests on root imports; add back-compat shims for moved modules | Test file churn avoided but keeps sys.path hack alive | |
| (C) Full rename + new test structure | Too much scope for Phase 3 | |

**User's choice:** (A) (auto-recommended).

---

## Area 7 — CLI placeholder scaffolding

| Option | Description | Selected |
|--------|-------------|----------|
| (A) Full typer app with a real `generate` command that runs one sample end-to-end | More useful smoke test; partially overlaps Phase 6 scope | |
| (B) Empty typer app with no commands | Satisfies entry-point but adds no value | |
| (C) Minimal typer app with one stub command pointing at Phase 6 | Satisfies `musicgen --help` and `musicgen = "musicgen.cli:app"` contract; no Phase 6 overlap | ✓ |

**User's choice:** (C) (auto-recommended). Roadmap explicitly says "placeholder cli.py".

---

## Area 8 — SongParams dataclass shape

| Option | Description | Selected |
|--------|-------------|----------|
| (A) Frozen dataclass embedding arrangement + per-part signatures + measures | Single object = "everything about this song"; Phase 4 annotator consumes directly | ✓ |
| (B) Minimal dataclass (key/tempo/time_sig/swing) + separate arrangement return | Matches today's call signature more closely but splits song state | |
| (C) Rich class with methods (`.with_variation()`, `.per_part(p)`) | Over-engineered for Phase 3; pure dataclass suffices | |

**User's choice:** (A) (auto-recommended). Matches the R-P4 sample.json schema shape, minimizes plumbing in Phase 4.

---

## Area 9 — music21 global RNG audit

| Option | Description | Selected |
|--------|-------------|----------|
| (A) Audit during Phase 3; wrap in `save_random_state()` if needed, else document | Clears the research open question, unblocks Phase 5 seed discipline | ✓ |
| (B) Defer audit to Phase 5 | Risk: Phase 5 discovers leakage and retro-fits the wrapper under time pressure | |
| (C) Always wrap unconditionally | Adds overhead and noise even if unnecessary | |

**User's choice:** (A) (auto-recommended). Two outcomes, both cheap: test passes → mark regression guard; test fails → write small wrapper.

---

## Area 10 — Phase 3 / Phase 4 parallelization

| Option | Description | Selected |
|--------|-------------|----------|
| (A) Run Phase 3 then Phase 4 serially | Phase 4 gets a stable package surface; simpler | ✓ |
| (B) Run Phase 3 and Phase 4 in parallel (roadmap permits this) | Would need upfront agreement on DurationValidator location, config threading, package root — all of which Phase 3 is currently deciding | |

**User's choice:** (A) (auto-recommended).

---

## Auto-Resolved

All 10 areas resolved via `--auto` mode using the recommended default (option A or A-narrow for most; option C for CLI placeholder since the roadmap explicitly calls it a "placeholder"). No user-facing prompts were presented.

## Corrections Made

None — auto mode did not surface contradictions against prior phase decisions.

## External Research

Not performed. The phase sits inside a well-mapped codebase with clear deliverables; decisions are grounded in `.planning/research/ARCHITECTURE.md`, `.planning/research/PITFALLS.md`, and prior phase CONTEXT/SUMMARY artifacts.

## Deferred Ideas (logged to CONTEXT.md)

- Moving `musicality_score.py` into the package (Phase 4 or 5).
- Moving `config.py` / `timesig.py` into the package (Phase 5).
- Replacing `beat_anotator.py` with `beats.py` (Phase 4 R-X7).
- Real seed derivation + per-domain RNG names (Phase 5 R-P7).
- Public library API `musicgen.generate / generate_batch` (Phase 5/6).
- Real typer CLI flags (Phase 6 R-P13).
- README rewrite for `pip install -e .` (Phase 7 R-Q1).
- Coverage gates (Phase 7 R-Q2).
