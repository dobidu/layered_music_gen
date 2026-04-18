---
gsd_state_version: 1.0
milestone: v0.1
milestone_name: — docs, polish, regression suite
status: Executing Phase 03
last_updated: "2026-04-18T20:17:53Z"
progress:
  total_phases: 7
  completed_phases: 2
  total_plans: 12
  completed_plans: 8
  percent: 67
---

# STATE

## Project Reference

See: `.planning/PROJECT.md` (updated 2026-04-08)

**Core value:** Every generated sample is a complete, reproducible, fully-labeled training example.
**Current focus:** Phase 03 — package-skeleton-sampler-generators-extraction

## Current position

Phase: 03 (package-skeleton-sampler-generators-extraction) — EXECUTING
Plan: 2 of 5 (Plan 03-01 complete)

- Initialized: 2026-04-08
- Milestone: v0.1 (Stabilize + Extract + Productize)
- Active phase: 03-package-skeleton-sampler-generators-extraction — Wave 1 (Plan 03-01) executed 2026-04-18
- Resume file: .planning/phases/03-package-skeleton-sampler-generators-extraction/03-02-PLAN.md
- Progress: Phase 02 complete; Phase 03 Wave 1 complete (Plan 03-01 — pyproject.toml + src/musicgen/ skeleton + editable install green, 309 tests still pass). Ready for Wave 2 (Plan 03-02 — music21 isolation regression tests).
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

## Recent decisions

- **2026-04-08 (Plan 01-02):** `levels.json` volume fields interpreted as LINEAR amplitudes (range 0.5-1.0 observed) and converted to dB at apply time via `20*log10(v)` with a `1e-6` floor. R-S4 / PITFALLS P-B closed.
- **2026-04-08 (Plan 01-03):** music21 symbols narrowed to `{roman, scale, pitch}` (verified by grep). `musicality_score.py` exception handlers narrowed to `(ValueError, RuntimeError, IndexError, FloatingPointError)` for analysis methods, `(FileNotFoundError, OSError, ValueError, RuntimeError)` for the outer file handler; all use `logger.exception(...)`. Kept `import time`/`import uuid` (still used; `uuid` removal deferred to Phase 5 R-P1). `print`→`logging` migration in `music_gen.py` deferred to Phase 2 per ROADMAP. R-S2 / R-S7 (musicality_score portion) / R-S8 closed.
- **2026-04-08 (Plan 01-04):** pytest skeleton landed with 95 passing tests across `tests/test_time_signature.py` and `tests/test_duration_validator.py` (runtime 2.75s). `dev-requirements.txt` and `tests/conftest.py` sys.path shim are intentional throwaway scaffolding scheduled for deletion in Phase 3. Tests pin current behavior including the cosmetic-if in `verify_beat_pattern` — Phase 2 registry refactor must preserve `len(pattern) == numerator` for ALL time signatures. Local `.venv/` required because system Python is PEP-668 managed (Debian-family). R-Q2 initial coverage closed. Phase 01 complete.
- **2026-04-18 (Plan 02-01):** Config threaded as explicit parameter through call chain (not module-level singleton) — preserves importability (Plan 01-01 property). cfg parameters default to None with fallback to config.Config() inside functions. generate_song_arrangement default arg changed from literal 'song_structures.json' to None + runtime fallback (Pitfall 7). T-02-01: os.path.abspath() on env/CLI path inputs. T-02-02: FileNotFoundError and PermissionError caught in pool report. R-S5 and R-S9 closed.
- **2026-04-18 (Plan 02-02):** TimeSignatureSpec frozen dataclass with 19 fields; TimeSignatureRegistry with all 7 signatures as single source of truth. All 10 time-sig functions in music_gen.py are thin wrappers. DurationValidator._analyze_time_signature delegates to registry via local import adapter (avoids circular import). beat_pattern_length == numerator for ALL sigs (cosmetic-if preserved). Empty frozenset for 5/4/7/8 valid_chord_pattern_lengths triggers default-True. generate_random_time_signature missing-return bug (Pitfall 5) fixed via random.choices. 186 registry tests added. A3 (RNG order change from threshold-loop to random.choices) confirmed — Phase 5 must baseline against post-refactor behavior. R-S6 closed.

- **2026-04-18 (Plan 02-03):** All 32 print() calls in music_gen.py replaced with semantically leveled logging calls (16 DEBUG, 14 INFO, 2 WARNING) per D-07. Module-level logger = logging.getLogger(__name__) added. logging.basicConfig(INFO) added inside __main__ guard only (T-02-06 mitigated). Component scores loop (3 prints) aggregated into single logger.debug call (Pitfall 8). except-block warning uses exc_info=True. 6 AST-scan + import-guard tests added; full suite 309 passed. R-S7 closed.

- **2026-04-18 (Phase 03 CONTEXT, auto mode):** Phase 3 context captured with 25 decisions across 10 gray areas, all selected via --auto recommended defaults. Key locked decisions: src/ layout with `src/musicgen/` package (D-01/02); `music_gen.py` becomes a thin re-import shim to preserve the "old music_gen.py still executable for smoke testing" exit criterion (D-04/05); every extracted function takes an explicit `rng: random.Random` parameter with zero bare `random.*` in extracted code (D-07/08) — Phase 5 will build the RNG hierarchy on these ready-made signatures; `enhanced_duration_validator.py` moves to `src/musicgen/duration_validator.py` (D-10), `musicality_score.py` stays at root for Phase 4 (D-11); `pyproject.toml` is the single authoritative dependency manifest — delete `requirements.txt` and `dev-requirements.txt` (D-13/14); test imports rewrite to `musicgen.*` and delete `tests/conftest.py` sys.path shim (D-15/16); `SongParams` is a frozen dataclass embedding arrangement + per-part signatures + measures (D-20); music21 global-RNG audit lands this phase (D-23/24) to clear a Phase 5 blocker. `config.py` and `timesig.py` stay at repo root this phase (D-03) — will move in Phase 5. Phases 3 and 4 run serially, not parallel, so Phase 4 gets a stable package surface (D-25).

- **2026-04-18 (Plan 03-01):** `pyproject.toml` (hatchling + 14 runtime deps + typer>=0.12 + pytest/cov/xdist dev extras) landed at repo root; `src/musicgen/` skeleton created (empty `__init__.py`, `__main__.py` delegator, `cli.py` typer stub with `info` command). `pip install -e '.[dev]'` now succeeds; `musicgen --help` and `python -m musicgen --help` both exit 0. `requirements.txt` + `dev-requirements.txt` deleted — pyproject.toml is the sole dep manifest. Two Rule 1/2 auto-fixes during execution: (a) pedalboard floor relaxed from `>=1.0.0` to `>=0.9.0` — no 1.x release exists on PyPI (this was a pre-existing bug in requirements.txt that blocked the editable install); (b) README.md Installation step updated from `pip install -r requirements.txt` to `pip install -e '.[dev]'`. Applied the RESEARCH.md Risk #1 override: `requires-python = ">=3.10"` (CONTEXT.md D-13's `">=3.9"` infeasible — typer>=0.12 and hatchling need 3.10+; inline comment preserves traceability). `[tool.pytest.ini_options] pythonpath = ["."]` wired so root `config.py`/`timesig.py` remain importable when Plan 03-05 deletes `tests/conftest.py`. Baseline 309 tests still pass. R-X1 closed; R-Q4 (version field) closed.

## Next command

Phase 03 Wave 1 complete. Next: `/gsd-execute-phase 3 --resume` → Wave 2 (Plan 03-02 — music21 global-RNG audit + isolation regression tests, D-23/D-24).

---
*Last updated: 2026-04-18 after Plan 03-01 (package skeleton + pyproject.toml) execution.*
