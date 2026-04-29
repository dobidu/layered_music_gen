# Phase 3: Package skeleton + sampler + generators extraction - Context

**Gathered:** 2026-04-18 (auto mode — recommended defaults applied per gray area)
**Status:** Ready for planning

<domain>
## Phase Boundary

Stand up `src/musicgen/` as a pip-installable Python package. Extract all **pure-function** song-level logic — the sampler (random key/tempo/time-sig/swing + arrangement) and the four MIDI generators (chord, melody, bassline, beat) — out of `music_gen.py` into clean modules behind injected-RNG interfaces.

This phase does NOT extract the audio-side pipeline (renderer, mixer, annotator, beats). That's Phase 4. This phase does NOT wire deterministic seed discipline end-to-end (that's Phase 5) — but it lays the groundwork by plumbing `rng: random.Random` parameters through every extracted function so Phase 5 only has to build the RNG hierarchy, not refactor signatures.

After Phase 3: `pip install -e .` works, `from musicgen.sampler import SongParams` works, extracted generators have zero bare `random.*` calls, and `python music_gen.py` still generates one smoke-test song end-to-end via thin re-export shims.

</domain>

<decisions>
## Implementation Decisions

### Package layout
- **D-01:** `src/` layout (not flat). Package root: `src/musicgen/`. Follows the module layout proposed in `.planning/research/ARCHITECTURE.md`.
- **D-02:** Files created this phase:
  - `src/musicgen/__init__.py` — empty for Phase 3; public API exports land in Phase 5 when library API is real.
  - `src/musicgen/__main__.py` — `python -m musicgen` entry; delegates to `cli.py`.
  - `src/musicgen/cli.py` — typer app scaffolding (stub, see D-17).
  - `src/musicgen/sampler.py` — `SongParams` dataclass + all `generate_random_*` functions + `generate_song_arrangement`.
  - `src/musicgen/generators/__init__.py` — empty marker.
  - `src/musicgen/generators/chord.py` — `generate_chord_progression` extracted.
  - `src/musicgen/generators/melody.py` — `generate_melody` extracted.
  - `src/musicgen/generators/bassline.py` — `generate_bassline` extracted.
  - `src/musicgen/generators/beat.py` — `generate_beat` + `beat_duration` + `calculate_swing_offset` extracted.
  - `src/musicgen/duration_validator.py` — `DurationValidator` + `NoteValue` moved into package (see D-10).
- **D-03:** `config.py` and `timesig.py` stay at repo root for this phase and are imported by `src.musicgen.*` via back-compat aliases in `music_gen.py`. Moving them into the package is deferred — both are already clean, standalone modules and the churn/test-migration cost isn't justified in Phase 3. They can migrate in Phase 5 when the package takes full ownership of orchestration.

### Back-compat shim strategy
- **D-04:** `music_gen.py` stays at repo root and continues to run as the Phase 3 smoke-test entry point (roadmap exit criterion: "Old `music_gen.py` still executable for smoke testing"). Functions extracted into the package are replaced in `music_gen.py` with explicit re-import statements (`from musicgen.sampler import generate_random_key, ...`), so any existing consumer doing `from music_gen import generate_random_key` still resolves.
- **D-05:** `music_gen.py` keeps the currently-unextracted logic intact for this phase: `mix_and_save` (143-line function — Phase 4's extraction target), `create_song`, `generate_song`, `generate_song_parts`, `generate_pedalboard`, `apply_fx_to_layer`, FX/levels/soundfont helpers, and the `__main__` guard. All are Phase 4/5 targets, out of scope here.
- **D-06:** The thin time-signature wrappers at the top of `music_gen.py` (`verify_pattern_for_time_signature`, `verify_beat_pattern`, etc.) remain — they already delegate to `TimeSignatureRegistry` (Plan 02-02). Generators call `TimeSignatureRegistry` directly rather than the shim wrappers to avoid a useless indirection.

### RNG plumbing depth
- **D-07:** Every extracted function takes an explicit `rng: random.Random` parameter as its last positional arg (or keyword-only after an existing required arg). No bare `random.choice / random.random / random.randint / random.choices` anywhere inside extracted modules (sampler, generators, duration_validator). All 39 bare-random call sites in the migrated code become `rng.choice / rng.random / rng.randint / rng.choices`.
- **D-08:** Phase 3 does NOT implement `derive_sample_seed` or `make_rngs` — that's R-P7 / Phase 5. During Phase 3, the `music_gen.py` shim constructs a single unseeded module-level `_rng = random.Random()` and passes it to every extracted call site. Behaviorally equivalent to today's module-level `random`, but the pipeline signatures are Phase-5-ready.
- **D-09:** The existing Plan 02-02 entry `TimeSignatureRegistry.sample_random(rng: Optional[random.Random] = None)` already follows this contract — leave it as-is. Sampler's `generate_random_time_signature` stays a thin wrapper that forwards `rng` to the registry.

### DurationValidator + musicality_score placement
- **D-10:** Move `enhanced_duration_validator.py` → `src/musicgen/duration_validator.py` (renamed — drop the `enhanced_` prefix; it's no longer contrasting against a non-enhanced version). `DurationValidator` and `NoteValue` exported from the new location. Root-level `enhanced_duration_validator.py` is deleted (no back-compat shim — tests migrate per D-14). Generators import `from musicgen.duration_validator import DurationValidator`.
- **D-11:** `musicality_score.py` stays at repo root for Phase 3. It's invoked from `create_song` (orchestration, not a generator) and will naturally move into the package during Phase 4's renderer/mixer/annotator extraction, or Phase 5's writer. Moving it now adds cost for no Phase 3 benefit.
- **D-12:** `beat_anotator.py` is untouched in Phase 3. It's scheduled for replacement by `src/musicgen/beats.py` in Phase 4 (R-X7). Leaving it alone avoids collision with Phase 4 work.

### pyproject.toml scope
- **D-13:** `pyproject.toml` is the single authoritative dependency manifest after Phase 3. Contents:
  - `[build-system]` → `requires = ["hatchling"]`, `build-backend = "hatchling.build"`.
  - `[project]` → `name = "musicgen"`, `version = "0.1.0"` (R-Q4), `requires-python = ">=3.9"`, description, readme pointer, license pointer.
  - `[project].dependencies` — migrate every runtime dep currently in `requirements.txt` verbatim (numpy, scipy, midiutil, music21, librosa, pydub, midi2audio, pedalboard, python-json-logger, typing-extensions, numba, llvmlite, python-magic) PLUS new runtime dep `typer>=0.12`.
  - `[project.optional-dependencies].dev = ["pytest>=8.0", "pytest-cov>=5.0", "pytest-xdist>=3.5"]`.
  - `[project.scripts]` → `musicgen = "musicgen.cli:app"` (CLI entry, stub body lands this phase per D-17).
  - `[tool.hatch.build.targets.wheel]` → `packages = ["src/musicgen"]` (src layout).
- **D-14:** `requirements.txt` and `dev-requirements.txt` are deleted. The Plan 01-04 summary already flagged `dev-requirements.txt` as "intentional throwaway scaffolding scheduled for deletion in Phase 3." Users install via `pip install -e .` or `pip install -e '.[dev]'`. README update for the new install command is deferred to Phase 7 R-Q1 (the milestone-wide README refresh).

### Test migration
- **D-15:** Rewrite every existing test to import from the package:
  - `tests/test_time_signature.py` — imports `from musicgen` (time-sig wrappers still live in music_gen.py as shims, but the registry tests already live in `tests/test_timesig_registry.py` and import `from timesig` → update to `from musicgen` or keep root `timesig` import per D-03).
  - `tests/test_duration_validator.py` — `from musicgen.duration_validator import DurationValidator, NoteValue`.
  - `tests/test_config.py`, `tests/test_timesig_registry.py` — `config` and `timesig` stay at repo root (D-03), so these imports don't change. They continue to work because the package's editable install resolves project-root imports naturally under `pip install -e .`.
  - `tests/test_music_gen_logging.py` — AST-scan tests over `music_gen.py` at the project root; path continues to work.
- **D-16:** Delete `tests/conftest.py` — its sys.path shim was explicitly flagged in its own docstring as "Phase 3 will introduce `pyproject.toml` and a proper `src/musicgen/` package, at which point this conftest's sys.path shim becomes unnecessary and should be deleted along with this file." With `pip install -e .`, pytest discovers the package without any shim.
- **D-17:** New test files to add this phase:
  - `tests/test_sampler.py` — seeded-RNG tests covering `SongParams` construction, `generate_random_key`, `generate_random_tempo`, `generate_random_time_signature`, `generate_random_swing`, `generate_song_measures`, `time_signature_alternative`, `generate_song_arrangement`. Same-seed → same-output for every function.
  - `tests/test_generators/test_chord.py`, `test_melody.py`, `test_bassline.py`, `test_beat.py` — seeded-RNG tests per generator. Verify MIDI files produced are byte-equal across two runs with the same `rng` seed and inputs.
  - `tests/test_package_install.py` — smoke test: `pip install -e .` succeeds in a temp venv (optional — can be a CI job instead if that's cheaper).

### CLI scaffolding
- **D-18:** `src/musicgen/cli.py` this phase is a minimal typer app — enough to satisfy the `musicgen = "musicgen.cli:app"` entry-point contract and make `pip install -e . && musicgen --help` not error. Contents:
  - `import typer` + `app = typer.Typer(help="musicgen — synthetic music dataset generator")`.
  - One stub command (e.g. `musicgen info` or `musicgen generate`) that prints a "Phase 6 will implement real generation — for now use `python music_gen.py`" message, or runs one smoke-test sample through the new sampler + generator stack without writing the full dataset layout.
  - No Config loading, no batch logic, no output-mode flags. All of that is Phase 6 R-P13.
- **D-19:** `src/musicgen/__main__.py` is a one-line `from musicgen.cli import app; app()` so `python -m musicgen` routes to the same typer app.

### SongParams dataclass shape
- **D-20:** `SongParams` is a **frozen dataclass** in `src/musicgen/sampler.py` with these fields:
  - `key: str` — e.g. "G", "Am"
  - `tempo: int` — BPM
  - `time_signature_base: str` — the base time sig from `generate_random_time_signature`, e.g. "4/4"
  - `time_signature_variation: float` — probability used to vary per-part sigs (today: hardcoded 1.0 in `generate_song`)
  - `swing_amount: float` — 0.5–0.75 clamped
  - `signatures_per_part: Dict[str, str]` — per-part time sigs from `generate_song_measures`
  - `measures_per_part: Dict[str, int]` — per-part measure counts from `generate_song_measures`
  - `song_unique_parts: List[str]` — unique part names from `generate_song_arrangement`
  - `song_arrangement: List[str]` — full part sequence from `generate_song_arrangement`
- **D-21:** Builder signature: `SongParams.sample(rng: random.Random, cfg: config.Config, *, time_signature_variation: float = 1.0) -> SongParams`. One call draws everything, in this deterministic order: key → tempo → time_signature_base → swing → arrangement → measures+per-part-signatures (with validation retry loop, same order as current `generate_song`). Preserves Plan 02-02 A3's RNG-order commitment — Phase 5 golden test will baseline against post-refactor behavior.
- **D-22:** Generators DO NOT take a `SongParams` directly — they take the specific fields they need (as today: `key, tempo, time_signature, measures, name, part, chord_progression`, plus injected `rng`). Rationale: minimizes diff size, keeps generator signatures unit-testable without constructing a full SongParams, matches today's call pattern from `generate_song_parts`. The package-internal orchestrator (Phase 4+) unpacks `SongParams` and passes the per-part fields.

### music21 global RNG audit
- **D-23:** As part of the extraction, audit whether `music21.roman.RomanNumeral`, `music21.scale.MajorScale`, `music21.scale.MinorScale`, and `music21.pitch.Pitch` touch the global `random` state internally (research ARCHITECTURE.md open question #2, PITFALLS.md open question #3). Two acceptable outcomes:
  - If they do NOT touch global `random`: add a short comment in `src/musicgen/generators/melody.py` and `bassline.py` documenting the audit (so Phase 5 doesn't re-investigate), no code change.
  - If they DO touch global `random`: implement a `save_random_state()` context manager in `src/musicgen/_rng_safety.py` (or inline helper in sampler.py) and wrap every music21 call. This is a Phase 3 prerequisite for Phase 5's seed discipline — without it, injected `rng` leaks through global `random`.
- **D-24:** Audit mechanism: a small test in `tests/test_music21_isolation.py` that snapshots `random.getstate()`, constructs each music21 object currently used, and asserts `random.getstate()` is unchanged. If the test fails, implement the wrapper; if it passes, mark the test as a regression guard.

### Parallelization
- **D-25:** Phase 3 and Phase 4 are technically parallelizable per ROADMAP.md, but this phase plans to **run serially before Phase 4**. Rationale: Phase 4 (renderer + mixer) depends on the package shape Phase 3 establishes (src/musicgen/ layout, config threading pattern, duration_validator location). Finishing Phase 3 first means Phase 4 gets a stable surface to import from, rather than moving targets.

### Claude's Discretion
- Exact filename of `SongParams.sample` classmethod vs free function (`sample_song_params`) — shape is fixed (D-21), name is aesthetic.
- Whether `calculate_swing_offset` lives in `generators/beat.py` or a shared `generators/_swing.py` helper — only one caller today, so inlining into beat.py is fine.
- Module docstring style (one-liner vs Google-style full) — match whatever convention the existing `config.py`/`timesig.py` modules use.
- Exact import ordering inside extracted modules (alphabetical vs group-by-origin) — match existing `music_gen.py` style.
- Whether `SongParams` uses `@dataclass(frozen=True, slots=True)` or just `@dataclass(frozen=True)` — `slots=True` requires Python 3.10+; `requires-python = ">=3.9"` per D-13 rules it out. Use `@dataclass(frozen=True)`.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase scope and requirements
- `.planning/ROADMAP.md` — Phase 3 section (deliverables, exit criteria, R-X1/X2/X3 coverage, Phase 4 parallelization note, dependency graph).
- `.planning/REQUIREMENTS.md` — R-X1 (package skeleton + pyproject.toml), R-X2 (sampler extraction), R-X3 (generators extraction). R-Q4 (version = "0.1.0" in pyproject) applies here too.
- `.planning/PROJECT.md` — Productize priority, determinism constraints, 1k–10k target scale.

### Architecture and design input
- `.planning/research/ARCHITECTURE.md` — proposed module layout (follow the `src/musicgen/` tree verbatim), data flow, build order ("Extract + Test" section).
- `.planning/research/PITFALLS.md` — P-4 (RNG leakage), open questions 2–3 (music21 global random audit). Informs D-07/D-08/D-23.
- `.planning/research/SUMMARY.md` §build-order — reinforces "time-sig registry before generator extraction" (done in Phase 2, precondition met).

### Codebase context
- `.planning/codebase/STRUCTURE.md` — current repo layout (flat Python files at root), naming conventions, configuration surface friction.
- `.planning/codebase/ARCHITECTURE.md` — the god-file description, pipeline stages, layer boundaries. Generators live in "Pipeline stages" §3 (MIDI component generation).
- `.planning/codebase/CONVENTIONS.md` — naming, import organization, type hint style, logging patterns (`logger = logging.getLogger(__name__)`). Extracted modules must match.
- `.planning/codebase/CONCERNS.md` — #1 god file (what we're decomposing), #4 wildcard import (already fixed Phase 1), #7 hardcoded paths (already fixed Phase 2), #9 time-sig shotgun (already fixed Phase 2).

### Prior phase artifacts (decisions to carry forward)
- `.planning/phases/01-stabilize-i-bug-fixes-and-guardrails/01-04-SUMMARY.md` — `conftest.py` sys.path shim flagged as "scheduled for deletion in Phase 3" (D-16); `dev-requirements.txt` flagged as "throwaway scaffolding scheduled for deletion in Phase 3" (D-14).
- `.planning/phases/01-stabilize-i-bug-fixes-and-guardrails/01-03-SUMMARY.md` — music21 narrow imports `{roman, scale, pitch}` (follow in extracted generator modules); `uuid` removal deferred to Phase 5 R-P1 (don't touch here).
- `.planning/phases/02-stabilize-ii-config-time-signature-registry-logging/02-CONTEXT.md` — D-01/D-02/D-03 config threading as explicit parameter (continue this pattern; generators that need config paths accept `cfg: config.Config = None` with runtime fallback).
- `.planning/phases/02-stabilize-ii-config-time-signature-registry-logging/02-02-*-SUMMARY.md` — A3 commitment that `TimeSignatureRegistry.sample_random` changed RNG order from threshold-loop to `random.choices`; Phase 5 baselines post-refactor behavior. Phase 3 preserves this (D-09).
- `.planning/phases/02-stabilize-ii-config-time-signature-registry-logging/02-03-*-SUMMARY.md` — module-level `logger = logging.getLogger(__name__)` pattern, `logging.basicConfig` stays inside `__main__` guard only. Extracted modules follow.

### Source files to modify / extract from
- `music_gen.py` — god file; extraction source for sampler (lines 508–548, 564–590, 808–848, 849–881) + generators (70–146 chord, 147–255 melody, 257–367 bassline, 369–495 beat) + `DurationValidator`-using helpers.
- `enhanced_duration_validator.py` — moves to `src/musicgen/duration_validator.py` (D-10).
- `tests/conftest.py` — delete (D-16).
- `tests/test_*.py` — update imports per D-15.
- `requirements.txt`, `dev-requirements.txt` — delete (D-14).

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `config.Config` + `config.Config.load()` — already supports CLI > env > defaults precedence. Sampler's `generate_song_arrangement` takes `structures_file: Optional[str] = None`; preserve this contract in the extracted module.
- `timesig.TimeSignatureRegistry` — already exposes `.sample_random(rng=None)`, `.lookup(sig)`, `.all_signatures()`. Sampler uses these instead of hand-rolling the threshold-loop.
- `DurationValidator` — already the correct abstraction. Generators use `get_suggested_duration`, `get_valid_duration`, `_analyze_time_signature`, `validate_layer_duration`. Preserve these surfaces after move.
- `musicality_score.py` and `enhanced_duration_validator.py` logging pattern — both use `logger = logging.getLogger(__name__)` at module scope. Follow.
- Existing pytest suite (309 passing tests across 5 files) — provides regression safety net for the extraction.

### Established Patterns (to match in extracted modules)
- Type hints on every function signature (including return type).
- Module-level `logger = logging.getLogger(__name__)`; `logger.debug/info/warning` at the levels established in Plan 02-03 (D-07 of Phase 2).
- Narrow music21 imports: `from music21 import roman, scale, pitch` (Plan 01-03 commitment).
- Explicit exception raising with descriptive messages for invalid states (e.g. `ValueError(f"Unsupported time signature...")`).
- `Tuple`, `Dict`, `List`, `Optional` from `typing` (Python 3.9 compatible — don't switch to PEP 604 pipe syntax yet).
- Google-style docstrings with Args section (see `enhanced_duration_validator.py:91-99` for canonical shape).

### Integration Points (callers of extracted code)
- `music_gen.py` `create_song` / `generate_song_parts` / `generate_song` — call into extracted sampler + generators via re-import shim (D-04).
- `music_gen.py` `mix_and_save` — untouched by Phase 3; Phase 4 extracts it.
- `tests/test_time_signature.py`, `tests/test_duration_validator.py`, `tests/test_timesig_registry.py`, `tests/test_config.py`, `tests/test_music_gen_logging.py` — update imports, keep assertions unchanged.
- `beat_anotator.py` — standalone, not imported by anyone. Leave alone (Phase 4 replaces it).

### Notes on the 39 bare-random call sites in music_gen.py (D-07 target)
- `generate_random_key` (1 `random.random()`)
- `generate_random_tempo` (2 `random.random()`, `random.randint`)
- `generate_random_swing` (inferred — read to confirm)
- `generate_song_measures` (4 `random.choice`, 1 `random.random`)
- `time_signature_alternative` (1 `random.choice`)
- `generate_chord_progression` (1 `random.choice`)
- `generate_melody` (3 — `random.choice`, `random.choices`, `random.randint`)
- `generate_bassline` (3 — `random.choice`, `random.choices`, `random.random`, `random.randint`)
- `generate_beat` (2 `random.choice`)
- `generate_song_arrangement` (1 `random.choice`)
- `mix_and_save` (5+ — but that's Phase 4's problem)

Every call site inside a function that migrates this phase gets rewritten to `rng.<method>(...)`. The `mix_and_save` call sites are explicitly out of scope.

</code_context>

<specifics>
## Specific Ideas

- Phase 3 is the "unlock productize" phase — its quality directly determines how smoothly Phases 4/5/6/7 go. Prioritize clean module boundaries and explicit contracts over minimal diff.
- SongParams lives in `sampler.py` (not a separate `song_params.py`) because it's co-generated with sampling. Placing it in sampler.py makes `SongParams.sample(rng, cfg)` the obvious entry point for "start one song."
- The "old `music_gen.py` still executable for smoke testing" exit criterion is the Phase 3 integration test. A single `python music_gen.py` run that produces valid output with the shim in place proves the extraction is non-breaking.
- Seeding is the future. Every decision this phase makes should pass the test "does this make Phase 5 seed discipline easy or hard?" — if it makes Phase 5 harder, choose the other option.

</specifics>

<deferred>
## Deferred Ideas

- **Moving `musicality_score.py` into the package** — Phase 4 or Phase 5, whenever orchestration moves.
- **Moving `config.py` / `timesig.py` into the package** — Phase 5, when the root `music_gen.py` shim is fully collapsed into the package.
- **Replacing `beat_anotator.py` with `src/musicgen/beats.py`** — Phase 4 R-X7.
- **Real `derive_sample_seed` + `make_rngs` + per-domain RNG names (`params`, `generators`, `soundfonts`, `fx`, `mix`)** — Phase 5 R-P7.
- **Public library API (`from musicgen import generate, generate_batch`)** — Phase 5 R-P12 (single sample) / Phase 6 R-P10 (batch).
- **Real typer CLI with `--count`, `--out`, `--seed`, `--workers`, `--output-mode`, `-v/-q`** — Phase 6 R-P13.
- **README rewrite for `pip install -e .`** — Phase 7 R-Q1.
- **`pytest-cov` coverage gates ≥ 80%** — Phase 7 R-Q2.

</deferred>

---

*Phase: 03-package-skeleton-sampler-generators-extraction*
*Context gathered: 2026-04-18 (auto mode)*
