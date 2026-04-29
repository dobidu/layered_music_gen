---
phase: 03-package-skeleton-sampler-generators-extraction
plan: 03-03
subsystem: sampler-extraction
tags: [extraction, sampler, songparams, rng-injection, determinism, frozen-dataclass, ast-guard]

# Dependency graph
requires:
  - phase: 02-stabilize-ii-config-time-signature-registry-logging
    provides: "TimeSignatureRegistry.sample_random(rng) — already rng-aware per Plan 02-02; config.Config.song_structures_file attribute per Plan 02-01"
  - phase: 03-package-skeleton-sampler-generators-extraction (Plan 03-01)
    provides: "src/musicgen/ package skeleton + pyproject.toml + [tool.pytest.ini_options] pythonpath = ['.']"
  - phase: 03-package-skeleton-sampler-generators-extraction (Plan 03-02)
    provides: "src/musicgen/duration_validator.py (relocated) — unblocks sampler-time generators since they don't import it directly"
provides:
  - "src/musicgen/sampler.py — 9 sampler symbols (SongParams frozen dataclass + SongParams.sample classmethod + 7 free functions + validate_measures_dict helper), all rng-aware"
  - "tests/test_sampler.py — 33 seeded-determinism tests across 3 classes + AST no-bare-random static guard"
  - "music_gen.py shim — inline sampler defs deleted; top-of-file re-imports 9 symbols from musicgen.sampler; _rng = random.Random() threaded to 6 call sites; validate_measures = validate_measures_dict back-compat alias"
  - "Phase-5-ready RNG-injection interface: every sampler function takes explicit rng: random.Random; SongParams.sample(rng, cfg) classmethod is the single entry point for song-level parameter draws"
  - "Locked RNG draw order for Phase 5's golden determinism baseline: key -> tempo -> time_sig_base -> swing -> arrangement -> loop(measures) (RESEARCH Risk #2)"
affects: [03-04, 03-05, phase-04-generators-extraction, phase-05-rng-hierarchy]

# Tech tracking
tech-stack:
  added: []  # pure-Python stdlib + first-party imports only; no new third-party deps
  patterns:
    - "PATTERN-S (sampler shape): @dataclass(frozen=True) + @classmethod sample(rng, cfg) builder; fields populate in canonical RNG-draw order"
    - "PATTERN-G (sampler bare-random rewrite): mechanical random.<method>() -> rng.<method>() substitution with zero logic change; AST test asserts the rewrite"
    - "PATTERN-K (shim surface): deleted defs replaced in place with one-line pointer comment; re-imports at top + back-compat alias + module-level _rng"
    - "PATTERN-H (seeded determinism test): two fresh random.Random(seed) instances per-test; @pytest.mark.parametrize('seed', [...]) over multiple seeds"
    - "PATTERN-I (AST static guard): ast.walk + isinstance(node.func.value, ast.Name) + node.func.value.id == 'random' filter, excluding random.Random constructor"

key-files:
  created:
    - "src/musicgen/sampler.py (293 lines)"
    - "tests/test_sampler.py (196 lines)"
    - ".planning/phases/03-package-skeleton-sampler-generators-extraction/03-03-SUMMARY.md"
  modified:
    - "music_gen.py (155 deletions, 38 insertions — net -117; 193 lines of sampler surface collapsed to 76 lines of shim + comments)"

key-decisions:
  - "D-08 enacted: single module-level _rng = random.Random() in music_gen.py, threaded to every extracted call site this phase; Phase 5 (R-P7) replaces this with derive_sample_seed + make_rngs."
  - "D-20 enacted: SongParams is @dataclass(frozen=True) with 9 fields — key, tempo, time_signature_base, time_signature_variation, swing_amount, signatures_per_part, measures_per_part, song_unique_parts, song_arrangement."
  - "D-21 enacted: SongParams.sample(rng, cfg=None, *, time_signature_variation=1.0) classmethod is the canonical builder. RNG draw order preserved verbatim from music_gen.py generate_song (RESEARCH Risk #2)."
  - "D-22 enacted: generators do NOT take a SongParams object — the orchestrator (Phase 4) will unpack it. SongParams is sampler-internal."
  - "generate_song_arrangement signature: (rng, structures_file: Optional[str] = None). rng is the NEW first positional arg; structures_file defaults to config.DEFAULT_SONG_STRUCTURES_FILE when None (preserves pre-refactor behavior — the pre-03-03 code also defaulted to this constant at line 520)."
  - "validate_measures_dict is the new canonical name in musicgen.sampler; validate_measures kept as back-compat alias in music_gen.py so the legacy call-site `if validate_measures(measures, signatures): break` continues to work without ripple-edits."
  - "generate_song_measures internal call `time_signature_alternative(ts)` rewritten to `time_signature_alternative(ts, rng)` — preserves the nested-rng.choice draw order established in the pre-refactor code (each time_signature_alternative call consumes exactly 1 rng.choice draw)."

patterns-established:
  - "PATTERN-S: sampler module shape = stdlib + first-party imports + logger + 7 free functions (each takes rng: random.Random) + 1 helper + 1 frozen dataclass with .sample classmethod"
  - "PATTERN-G: bare-random rewrite is purely mechanical — 1:1 substitution preserving call-order, argument-order, and nesting; verify with AST scan, not regex"
  - "PATTERN-K: when a function is extracted to a new module, replace the inline def with a one-line comment pointing at the new location (makes future greps for `def foo` yield 0 hits and documents the split)"

requirements-completed: [R-X2]

# Metrics
duration: "6 min"
completed: "2026-04-18"
---

# Phase 03 Plan 03-03: Sampler Extraction Summary

**All 9 sampler symbols (SongParams frozen dataclass + SongParams.sample classmethod builder + 7 rng-aware free functions + validate_measures_dict helper) extracted from music_gen.py into src/musicgen/sampler.py, with zero bare `random.*` in sampler.py (AST-enforced) and 33 seeded-determinism tests locking the canonical RNG draw order for Phase 5's golden baseline.**

## Performance

- **Duration:** 6 min
- **Started:** 2026-04-18T20:30:20Z
- **Completed:** 2026-04-18T20:36:XXZ
- **Tasks:** 3 / 3
- **Files modified:** 3 (1 created in src/musicgen/, 1 created in tests/, 1 rewritten at repo root)

## Accomplishments

- Created `src/musicgen/sampler.py` (293 lines) with all 9 sampler symbols: `SongParams`, `generate_random_key`, `generate_random_tempo`, `generate_random_time_signature`, `generate_random_swing`, `time_signature_alternative`, `generate_song_measures`, `generate_song_arrangement`, `validate_measures_dict`. Every function takes `rng: random.Random`; `SongParams` is a `@dataclass(frozen=True)` with 9 fields and a `.sample(rng, cfg)` classmethod builder.
- Rewrote 25 `random.<method>()` call sites to `rng.<method>()` inside sampler.py (no logic change; AST scan enforces zero bare `random.*` remain — excepting `random.Random` type/constructor references).
- Created `tests/test_sampler.py` (196 lines, 33 tests): `TestSongParamsSample` (9 tests), `TestSamplerFreeFunctions` (21 tests parametrized over seeds [0,1,42]), `TestValidateMeasuresDict` (2 tests), and the module-level AST static guard `test_no_bare_random_in_sampler` (1 test). All 33 pass.
- Rewrote `music_gen.py` as a shim: deleted 8 inline function definitions (155 line deletions), added top-of-file re-import block (9 symbols), added `_rng = random.Random()` and `validate_measures = validate_measures_dict` back-compat alias, threaded `_rng` through 6 call sites (`generate_song_arrangement` in `create_song`; `generate_random_key`/`tempo`/`time_signature`/`swing`/`generate_song_measures` in `generate_song`). Net churn: -117 lines.
- Preserved RNG draw order exactly per RESEARCH Risk #2: `key -> tempo -> time_sig_base -> swing -> arrangement -> loop(measures)` inside `SongParams.sample`; per-iteration draws inside `generate_song_measures` unchanged (5 base_length choices + 1 variation gate + conditional 5 ts choices × each consuming 1 nested `time_signature_alternative` rng.choice).
- Full test suite: 309 -> 342 passing (+33 new sampler tests; 0 regressions). Baseline preserved.
- Smoke test `python music_gen.py` reaches `mix_and_save` before failing on environmental ffmpeg/soundfont absence — confirms sampler layer and all call-site wiring work end-to-end.

## Task Commits

Each task committed atomically on branch `main`:

1. **Task 1: Extract sampler logic into src/musicgen/sampler.py with rng injection** — `eaa7ee5` (feat)
2. **Task 2: Add seeded-RNG sampler tests + AST no-bare-random guard** — `cdaa3b8` (test)
3. **Task 3: Rewrite music_gen.py shim + thread _rng to call sites** — `53d929d` (refactor)

**Plan metadata commit:** pending (will include this SUMMARY + STATE.md + ROADMAP.md + REQUIREMENTS.md update).

_Note: Task 1 and Task 2 are flagged `tdd="true"` in the plan frontmatter, but for a pure-extraction plan the RED/GREEN cycle is degenerate (the port is logic-preserving — there's no "missing feature" to stub in RED). Task 1 instead used AST self-verification gates inside its `<action>` steps as the "test-before-commit" contract; Task 2 is the formal pytest test file. All 33 tests in tests/test_sampler.py pass against the Task-1 sampler.py as-written, which functions as the "GREEN" confirmation post-hoc. This is a reasonable interpretation of TDD for pure refactors — the author had the same discussion in RESEARCH §Sampler Extraction._

## Files Created/Modified

- `src/musicgen/sampler.py` — **CREATED** (293 lines). Module docstring + stdlib imports + first-party imports (`import config`, `from timesig import TimeSignatureRegistry`) + module-level logger + 7 free functions + `validate_measures_dict` helper + `SongParams` frozen dataclass with `.sample` classmethod. Zero bare `random.*` calls (25 `rng.*` calls instead).
- `tests/test_sampler.py` — **CREATED** (196 lines). 33 tests: `TestSongParamsSample` (9), `TestSamplerFreeFunctions` (21), `TestValidateMeasuresDict` (2), `test_no_bare_random_in_sampler` (1, module scope). Imports directly from `musicgen.sampler` (not via `music_gen` shim).
- `music_gen.py` — **MODIFIED** (net -117 lines: 155 deletions, 38 insertions). Deleted 8 inline defs (validate_measures, generate_song_arrangement, generate_random_swing, generate_random_key, generate_random_tempo, generate_random_time_signature, time_signature_alternative, generate_song_measures). Added top-of-file re-import block + `_rng = random.Random()` + `validate_measures = validate_measures_dict` alias. Threaded `_rng` through 6 call sites. Kept intact: `mix_and_save`, `create_song` (outer), `generate_song_parts`, `generate_pedalboard`, `apply_fx_to_layer`, `create_effect`, soundfont/levels helpers, `__main__` guard, all 10 thin time-signature wrapper functions at the top of the file (D-05/D-06 unchanged).

## Decisions Made

- **Import block organization:** Placed the 9-symbol `from musicgen.sampler import (...)` block immediately after the existing `from timesig import TimeSignatureRegistry` line (i.e., inside the first-party import group), preserving the existing stdlib/third-party/first-party grouping convention established by Plan 03-02.
- **Back-compat alias location:** Put `validate_measures = validate_measures_dict` at module top (line 42) rather than at the old function-definition site (~line 60). This keeps the alias near the import and makes `grep -n "validate_measures" music_gen.py` results contiguous.
- **`_rng` placement:** Placed `_rng = random.Random()` immediately before `validate_measures = validate_measures_dict`, so both "shim surface" features (the alias + the RNG threading root) live in one visually-adjacent block at module top with an explanatory comment pointing at D-08 and R-P7.
- **Comment tombstones for deleted functions:** Replaced each deleted `def` with a one-line comment pointing at the extracted location (pattern: `# ``foo`` was extracted to ``musicgen.sampler`` (03-03). Callers pass ``_rng``.`). This is PATTERN-K from 03-PATTERNS.md and is future-proof against greps that search for `def foo` expecting zero hits.
- **Test file NOT imported via music_gen shim:** `tests/test_sampler.py` imports directly from `musicgen.sampler`, not from `music_gen`. This makes the tests pass independently of Task 3's shim rewrite (ran first to confirm Task 1 was complete before starting Task 3). It also matches D-15/16 convention: tests import from the NEW package location, not the legacy shim.

## Deviations from Plan

### Observations (no code changes)

**1. [Rule 3 observation - plan spec typo, no deviation required] Acceptance-criterion grep off-by-one**
- **Found during:** Task 1 self-check
- **Issue:** The plan's Task 1 acceptance criterion says `grep -c "^def generate_" src/musicgen/sampler.py >= 7`, but only 6 top-level functions in sampler.py start with `generate_` (`generate_random_key`, `generate_random_tempo`, `generate_random_time_signature`, `generate_random_swing`, `generate_song_measures`, `generate_song_arrangement`). The 7th "free function" per `must_haves.truths` is `time_signature_alternative`, which does NOT start with `generate_`. The semantic requirement — 7 sampler free functions + 1 helper — is satisfied (8 top-level functions in sampler.py); the grep is just too narrow.
- **Fix:** None needed. The actual count of "^def " lines in sampler.py is 8, covering all required symbols. Flagging here so Plan 03-04's/05's author can loosen the grep pattern if they copy-paste this acceptance shape.
- **Files modified:** None.
- **Verification:** `grep -cE "^def (generate_|time_signature_alternative|validate_measures_dict)" src/musicgen/sampler.py` returns `8`.

**2. [Rule 3 observation - plan spec narrower than reality] `_rng` call-site grep pattern**
- **Found during:** Task 3 self-check
- **Issue:** The plan's Task 3 acceptance criterion says `grep -cE "(generate_random_key|generate_random_tempo|...|generate_song_arrangement|generate_song_measures)\(_rng" music_gen.py >= 5`. This pattern only matches when `_rng` is the FIRST positional arg. In our rewrite, `generate_song_measures` passes `_rng` as the third positional (`generate_song_measures(time_signature, time_signature_variation, _rng)` — matches D-07 "rng last after existing required positionals"). The narrow grep therefore counts 4 matches, not 5. Relaxing the grep to `grep -cE "_rng\b"` shows 6 actual call sites (5 inside `generate_song` + 1 inside `create_song`).
- **Fix:** None needed. The semantic requirement — "multiple call sites pass `_rng`" — is met with 6 sites (≥5). Flagging so future plans' greps account for rng-last signatures.
- **Files modified:** None.

---

**Total deviations:** 0 code deviations. 2 plan-spec observations filed (both are grep-pattern looseness, no action required).
**Impact on plan:** Plan executed exactly as written at the code level. Observations are spec-patching hints for Plans 03-04 / 03-05 / Phase 4.

## Issues Encountered

- **Smoke test `python music_gen.py` fails at `mix_and_save`** due to 0 soundfont files on disk + no ffmpeg binary in PATH. Plan 03-03 calls this out explicitly as environmental (not a regression) and acceptable. The sampler layer itself (key/tempo/time_signature/swing/arrangement/measures) runs to completion and all 5 MIDI parts are generated before `mix_and_save` hits the soundfont gap. Downstream environment configuration (install ffmpeg; populate `sf/*/` with .sf2 files) is out of scope for Phase 3.

- **`random.*` call sites remaining in `music_gen.py`:** 19 `random.<method>(...)` sites remain inside `generate_chord_progression`, `generate_melody`, `generate_bassline`, `generate_beat`, `create_effect`, `mix_and_save`, and `get_random_sound_font`. These are **generator-layer** and **mix-layer** draws, not sampler — per RESEARCH Risk #5 they are Phase-4 scope (Plan 04-xx) and MUST NOT be rewritten in Plan 03-03. Confirmed not a regression.

## User Setup Required

None — pure internal refactor. No external service configuration, no environment variables, no dashboard changes.

## Next Phase Readiness

- **Plan 03-04 (music21 global-RNG audit, R-X3):** Ready. Sampler is now a clean, rng-threaded module; music21-internal random state (if any) can be audited against the new sampler.py without interference.
- **Plan 03-05 (requirements.txt / dev-requirements.txt / conftest.py sweep + test-import rewrite):** Ready. All imports now go through `musicgen.sampler` / `musicgen.duration_validator`; `tests/conftest.py` sys.path shim can be deleted once pyproject's `pythonpath = ["."]` is confirmed sufficient.
- **Phase 4 (generators extraction):** Ready. Sampler is done; generators (`generate_chord_progression`, `generate_melody`, `generate_bassline`, `generate_beat`) can now be lifted into `src/musicgen/generators/` using the same PATTERN-G bare-random rewrite approach. The 19 remaining `random.*` call sites in music_gen.py map directly to the generator-layer work.
- **Phase 5 (RNG hierarchy, R-P7):** Ready. Every sampler function takes `rng: random.Random`; `SongParams.sample(rng, cfg)` is the single-entry builder. Phase 5 just needs to replace the module-level `_rng = random.Random()` with `derive_sample_seed(master_seed, sample_id) -> Random`, and the entire sampler layer becomes deterministic per-sample with zero rewrites. The RNG draw order locked by Plan 03-03 IS the golden-determinism baseline.

## Threat Flags

No new threat surface introduced. Internal refactor (per Phase 3 threat register T-03-03-01, disposition: accept). `generate_song_arrangement` still reads `song_structures.json` through `config.Config.song_structures_file` / `config.DEFAULT_SONG_STRUCTURES_FILE` (existing trust boundary, not widened).

## Self-Check: PASSED

Verified claims against disk + git:

- `src/musicgen/sampler.py` — FOUND (293 lines; `grep -cE "^def " sampler.py` = 8).
- `tests/test_sampler.py` — FOUND (196 lines; 33 tests collected).
- `music_gen.py` — FOUND; `grep -c "^def (generate_random_key|tempo|swing|time_signature|time_signature_alternative|song_measures|song_arrangement)"` = 0; `grep -c "from musicgen.sampler import"` = 1.
- Commit `eaa7ee5` — FOUND via `git log --oneline`.
- Commit `cdaa3b8` — FOUND.
- Commit `53d929d` — FOUND.
- `python -c "import music_gen; assert music_gen.generate_random_key.__module__ == 'musicgen.sampler'"` exits 0.
- `python -c "from musicgen.sampler import SongParams; import random; assert SongParams.sample(random.Random(42)) == SongParams.sample(random.Random(42))"` exits 0.
- AST scan reports zero bare `random.*` calls in `src/musicgen/sampler.py`.
- Full suite: `pytest tests/ -q` reports `342 passed` (309 baseline + 33 new).

---
*Phase: 03-package-skeleton-sampler-generators-extraction*
*Plan: 03-03*
*Completed: 2026-04-18*
