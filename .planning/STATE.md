# STATE

## Project Reference

See: `.planning/PROJECT.md` (updated 2026-04-08)

**Core value:** Every generated sample is a complete, reproducible, fully-labeled training example.
**Current focus:** Milestone v0.1 — not yet started. Phase 1 (Stabilize I) is next.

## Current position

- Initialized: 2026-04-08
- Milestone: v0.1 (Stabilize + Extract + Productize)
- Active phase: none yet — ready for `/gsd-plan-phase 1`
- Mode: Interactive
- Granularity: Standard
- Parallelization: enabled (Phase 3 ∥ Phase 4 after Phase 2)
- Model profile: Balanced
- Workflow agents: Researcher + Plan Checker + Verifier all enabled

## Artifacts

- `.planning/PROJECT.md` — project context, core value, requirements buckets, key decisions
- `.planning/REQUIREMENTS.md` — scoped v0.1 requirements (R-S, R-X, R-P, R-Q families)
- `.planning/ROADMAP.md` — 7 phases for v0.1, dependency graph, coverage map
- `.planning/config.json` — workflow preferences
- `.planning/codebase/` — 7 structured documents from `/gsd-map-codebase`
- `.planning/research/` — STACK, FEATURES, ARCHITECTURE, PITFALLS, SUMMARY from domain research

## Notable findings from research to keep in mind

- **Two confirmed-in-code bugs** verified by grep, both fixed in Phase 1:
  - `mix_and_save` re-calls `generate_song_arrangement()` at `music_gen.py:760` (PITFALLS P-A).
  - `pydub` `.volume =` and discarded `.pan()` return at `music_gen.py:845-852` mean `levels.json` has zero effect today (PITFALLS P-B).
- **One new runtime dependency:** `typer>=0.12`. Everything else is stdlib or already installed; `python-json-logger` is already in `requirements.txt`.
- **Determinism contract:** MIDI + metadata bit-identical; audio bit-identical only under pinned FluidSynth binary.

## Outstanding infra issue

- Git identity is not configured in this repo. All planning documents are written to disk but uncommitted. User must run `git config user.email` / `user.name` (or set them globally) before the GSD commit helper can commit. Files currently uncommitted: `.planning/codebase/*` (7 files), `.planning/PROJECT.md`, `.planning/config.json`, `.planning/REQUIREMENTS.md`, `.planning/ROADMAP.md`, `.planning/research/*` (5 files), `.planning/STATE.md`.

## Next command

`/gsd-plan-phase 1`

---
*Last updated: 2026-04-08 after initialization*
