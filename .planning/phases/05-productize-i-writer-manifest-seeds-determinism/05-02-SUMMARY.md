---
phase: 05-productize-i-writer-manifest-seeds-determinism
plan: "02"
subsystem: seeds

tags: [phase-5, wave-1, seeds, rng-hierarchy, split, d-17, d-18, d-19, d-20, d-26, d-36, d-39]

# Dependency graph
requires:
  - phase: 05-productize-i-writer-manifest-seeds-determinism
    plan: "01"
    provides: "tests/test_seeds.py + tests/test_split.py Wave 0 stubs (pytest.skip(allow_module_level=True)); widened AST guard allow-list {Random, getstate, setstate} enabling seeds.save_random_state(); forward-guard expected_present set listing seeds.py"
provides:
  - "src/musicgen/seeds.py — pure-function RNG foundation (R-P7) exporting derive_sample_seed, make_rngs, assign_split, save_random_state, plus RNG_PARAMS/RNG_GENERATORS/RNG_SOUNDFONTS/RNG_FX/RNG_MIX constants"
  - "tests/test_seeds.py — 21 passing assertions across 3 classes (TestDeriveSampleSeed, TestMakeRngs, TestSaveRandomState); D-36 closure"
  - "tests/test_split.py — 108 passing assertions in TestAssignSplit (5 determinism + 100 parametrized valid-label + 2 empirical-ratio + 1 disambiguation); D-39 closure"
affects:
  - 05-04 (Wave 3 writer + manifest — stem sum-of-stems assertion relies on deterministic sample_seed propagation this plan unlocks)
  - 05-05 (Wave 4 api.py — direct consumer of derive_sample_seed + make_rngs + save_random_state + assign_split; musicality call site wraps in save_random_state per D-20)
  - 05-06 (Wave 5 determinism goldens — D-17/D-18/D-26 formulas byte-exact-locked this plan, Wave 5 hashes depend on them)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Pure-function + contextmanager module (stdlib-only): hashlib + random + contextlib + typing; zero package imports; module-header shape exact match to src/musicgen/beats.py"
    - "Byte-exact algorithm verification in tests: test_matches_documented_formula spot-checks derive_sample_seed(42, 0) == int.from_bytes(hashlib.sha256(b'42:0').digest()[:8], 'big') — guards against any drift of the D-17 formula"
    - "Per-XOR-constant parametrized test: one test case per (domain, 0x0N) tuple rather than one monolithic assertion — pytest surface names the exact failing mask on regression"

key-files:
  created:
    - src/musicgen/seeds.py
  modified:
    - tests/test_seeds.py
    - tests/test_split.py

key-decisions:
  - "derive_sample_seed uses int.from_bytes(raw[:8], 'big') exactly — byte-slice '[:8]' and endian 'big' are load-bearing for D-28 Wave 5 goldens; any drift would break the Phase 5 determinism contract."
  - "make_rngs XOR constants 0x01..0x05 assigned in the D-18 canonical order (params→generators→soundfonts→fx→mix) — downstream api.py (Wave 4) routes per-domain by the exact string keys set here."
  - "save_random_state is a @contextlib.contextmanager (not a class with __enter__/__exit__) — matches D-20 verbatim; try/finally guarantees restoration even on exception."
  - "assign_split uses sha256(f'split:{sample_seed}'.encode())[:4] % 10000 / 100.0 — the 'split:' prefix disambiguates from derive_sample_seed's hash input so the split bucket is statistically independent of the sample RNG basin."
  - "Config.__post_init__ validation of split_ratios (D-27 ValueError on sum != 1.0) is NOT in this plan — config.py field doesn't exist yet. Correctly deferred to Wave 3 Plan 05-04 where split_ratios becomes a Config field."

patterns-established:
  - "Parametrized determinism test: @pytest.mark.parametrize over (global_seed, sample_index) or sample_seed — two calls per case, assert equal. Phase 5 Wave 4/5 api/determinism tests will echo this shape."
  - "Empirical-ratio test with 10k-sample Counter: precomputes seed via derive_sample_seed(42, i) for i in range(10000), aggregates labels in collections.Counter, asserts bounds ±2%. Runs in ~60ms — no @pytest.mark.slow needed."
  - "class-based test grouping with single class per public symbol: TestDeriveSampleSeed / TestMakeRngs / TestSaveRandomState / TestAssignSplit — inherits the Phase 3 tests/test_sampler.py::TestSamplerFreeFunctions idiom."

requirements-completed: [R-P6, R-P7]

# Metrics
duration: ~3min
completed: 2026-04-19
---

# Phase 5 Plan 02: Wave 1 — seeds.py + assign_split + deterministic split tests Summary

**Pure-function RNG hierarchy (derive_sample_seed, make_rngs, save_random_state, assign_split) + 5 domain name constants landed in src/musicgen/seeds.py; Wave 0 stubs in tests/test_seeds.py + tests/test_split.py replaced with 129 real assertions proving D-17/D-18/D-20/D-26 formulas byte-exact and D-39 empirical-ratio contract holds.**

## Performance

- **Duration:** ~3 min
- **Started:** 2026-04-19T21:19:47Z
- **Completed:** 2026-04-19T21:23:00Z (approximate)
- **Tasks:** 3
- **Files modified:** 3 (1 created + 2 overwritten — Wave 0 stubs replaced)
- **Commits:** 3 task commits (c4b3d91, 62452fe, bc2e4fc) + 1 plan-metadata commit (pending)

## Accomplishments

- **Shipped the standalone RNG foundation that every downstream Phase 5 plan imports** — zero in-package dependencies, pure stdlib (hashlib + random + contextlib). Wave 3's writer tests and Wave 4's api.py orchestration both route through symbols landed here.
- **Proved D-17/D-18/D-20/D-26 formulas byte-exact in tests** — `test_matches_documented_formula` asserts `derive_sample_seed(42, 0) == int.from_bytes(hashlib.sha256(b'42:0').digest()[:8], 'big')`; `test_each_domain_xor_constant` parametrizes over all five (domain, 0x0N) pairs; `test_restores_on_exception` covers the try/finally path of save_random_state. Any future drift in these formulas fails a named test.
- **108 split tests pass in <200ms** — 5 determinism cases + 100 parametrized valid-label cases + 2 empirical-ratio (10k-seed) checks on default 80/10/10 and alternative 50/25/25 ratios + 1 disambiguation probe. Empirical ratios land comfortably within 2% tolerance: train=8015, valid=989, test=996 on default; train=4977, valid=2533, test=2490 on 50/25/25.
- **AST bare-random guard parametrized case for seeds.py passed on first run** — Plan 05-01's widened allow-list ({Random, getstate, setstate}) accepts the three primitives seeds.py legitimately uses. No additional guard edits needed.
- **Full suite transitioned 503 → 633 passed (+130 net), 8 → 6 skipped, 1 xfailed unchanged, 0 failed** — two Wave 0 stubs replaced (test_seeds.py + test_split.py), one new AST guard parametrize case added, 21 new seeds tests + 108 new split tests.

## Task Commits

1. **Task 1: Create src/musicgen/seeds.py with 4 functions + 5 constants** — `c4b3d91` (feat)
2. **Task 2: Populate tests/test_seeds.py with real tests (D-36)** — `62452fe` (test)
3. **Task 3: Populate tests/test_split.py with real tests (D-39)** — `bc2e4fc` (test)

## Files Created/Modified

### Created (1)

- `src/musicgen/seeds.py` — 121 lines. Module-header (from __future__ annotations + logger = logging.getLogger(__name__)) matches `src/musicgen/beats.py` pattern. Four public functions: `derive_sample_seed` (D-17 verbatim — `int.from_bytes(raw[:8], "big")`), `make_rngs` (D-18 verbatim — 5 XOR masks 0x01..0x05 in params/generators/soundfonts/fx/mix order), `assign_split` (D-26 verbatim — `"split:"` prefix + 4-byte slice % 10000 / 100.0), `save_random_state` (D-20 verbatim @contextlib.contextmanager with try/finally). Five module-level string constants `RNG_PARAMS`..`RNG_MIX`.

### Modified (2)

- `tests/test_seeds.py` — 129 lines (was 14-line Wave 0 stub). Three test classes: `TestDeriveSampleSeed` (5 parametrized determinism + 4 singular tests = 9 cases), `TestMakeRngs` (2 singular + 5 parametrized XOR-constant + 2 singular = 9 cases), `TestSaveRandomState` (3 cases: after-mutation, on-exception, nested). 21 assertions pass in ~30ms.
- `tests/test_split.py` — 59 lines (was 16-line Wave 0 stub). Single class `TestAssignSplit`: 5 parametrized determinism + 100 parametrized valid-label + 2 empirical-ratio (10k seeds each) + 1 prefix disambiguation = 108 cases. Runs in ~100ms.

## Decisions Made

- **Module-body verbatim from plan** — D-17/D-18/D-20/D-26 formulas copied character-for-character from the plan's `<action>` block, which itself transcribes the canonical references in 05-CONTEXT.md and .planning/research/ARCHITECTURE.md §"Seed / RNG propagation". No creative interpretation.
- **Test class boundaries match public symbols** — One class per function exported from seeds.py. Split tests live in tests/test_split.py per D-39 (consumer-facing separation) rather than in test_seeds.py.
- **Parametrized XOR-constant test** — Added `test_each_domain_xor_constant` beyond the plan's explicit `test_xor_constants_match_spec` spot check, because the plan spec listed it in both the `<behavior>` block and in Task 2's `<action>` verbatim code. Every domain gets its own named test case so a future regression surfaces the exact failing mask (e.g., "`test_each_domain_xor_constant[fx-4] FAILED`").

## Deviations from Plan

None — plan executed exactly as written.

- No Rule 1 auto-fix (no bugs found).
- No Rule 2 auto-add (no missing critical functionality — threat register T-05-02-01 through T-05-02-04 mitigations all already in the plan bodies: type-hint int coercion, SHA-256 collision resistance via cryptographic hash, try/finally on save_random_state, O(n) bounded empirical test).
- No Rule 3 auto-fix (no blocking issues).
- No Rule 4 architectural decisions needed.
- No auth gates.

## Issues Encountered

None — all three tasks executed cleanly on first attempt. The test suite transitioned from baseline 503 → 633 passed with zero failures, zero regressions, and zero fix attempts.

One false positive during verification: a naive `grep -c "failed"` on the pytest summary line matched "1 xfailed" as if it were a real failure. Re-ran with the tighter pattern ` N failed` (leading space, N digit groups, "failed" word) which correctly returned 0 — the xfailed is the Wave 0 forward-guard meta-test still awaiting Waves 3-4 modules.

## Self-Check: PASSED

Verification commands (all exit 0 / confirm FOUND):

### Task 1 — src/musicgen/seeds.py
- `[ -f src/musicgen/seeds.py ]` → FOUND (121 lines)
- `grep -c "def derive_sample_seed" src/musicgen/seeds.py` → 1 ✓
- `grep -c "def make_rngs" src/musicgen/seeds.py` → 1 ✓
- `grep -c "def assign_split" src/musicgen/seeds.py` → 1 ✓
- `grep -c "def save_random_state" src/musicgen/seeds.py` → 1 ✓
- `grep -c 'int.from_bytes(raw\[:8\], "big")' src/musicgen/seeds.py` → 1 ✓
- `grep -c 'sample_seed \^ 0x01' src/musicgen/seeds.py` → 1 ✓
- `grep -c 'sample_seed \^ 0x05' src/musicgen/seeds.py` → 1 ✓
- `grep -c 'f"split:{sample_seed}"' src/musicgen/seeds.py` → 1 ✓
- `grep -c "@contextlib.contextmanager" src/musicgen/seeds.py` → 1 ✓
- `grep -cE '^RNG_(PARAMS|GENERATORS|SOUNDFONTS|FX|MIX) = ' src/musicgen/seeds.py` → 5 ✓
- AST guard: `tests/test_no_bare_random_in_package.py[...seeds.py]` → PASSED ✓
- `.venv/bin/python -c "... derive_sample_seed(42, 0) == int.from_bytes(hashlib.sha256(b'42:0').digest()[:8], 'big')"` → exits 0 ✓
- `.venv/bin/python -c "... set(make_rngs(1).keys()) == {'params', 'generators', 'soundfonts', 'fx', 'mix'}"` → exits 0 ✓

### Task 2 — tests/test_seeds.py
- `grep -c "allow_module_level=True" tests/test_seeds.py` → 0 ✓ (stub removed)
- `grep -c "class TestDeriveSampleSeed" tests/test_seeds.py` → 1 ✓
- `grep -c "class TestMakeRngs" tests/test_seeds.py` → 1 ✓
- `grep -c "class TestSaveRandomState" tests/test_seeds.py` → 1 ✓
- PASSED count: 21 (>= 15) ✓
- Failed: 0 ✓
- `test_matches_documented_formula` present and runs ✓

### Task 3 — tests/test_split.py
- `grep -c "allow_module_level=True" tests/test_split.py` → 0 ✓ (stub removed)
- `grep -c "class TestAssignSplit" tests/test_split.py` → 1 ✓
- `grep -c "test_empirical_ratios_10k_seeds_default" tests/test_split.py` → 1 ✓
- PASSED count: 108 (>= 7) ✓
- Failed: 0 ✓

### Commits present in history
- `git log --oneline -5 | grep -c c4b3d91` → FOUND
- `git log --oneline -5 | grep -c 62452fe` → FOUND
- `git log --oneline -5 | grep -c bc2e4fc` → FOUND

### Full-suite regression check
- `.venv/bin/pytest tests/ -q` → `633 passed, 6 skipped, 1 xfailed, 2 warnings in 1.44s` ✓ (zero failures; 503 prior + 21 test_seeds + 108 test_split + 1 new AST guard parametrize for seeds.py = 633, arithmetic checks)
- `.venv/bin/pytest tests/test_seeds.py tests/test_split.py tests/test_no_bare_random_in_package.py -v` → `142 passed, 1 xfailed` ✓ (xfailed is the forward-guard meta-test, expected per Plan 05-01 — clears in Plan 05-05 after api.py lands)

## Next Plan Readiness

- **Plan 05-02 fully unblocks Wave 3 and Wave 4 downstream consumers.** The expected_present forward-guard meta-test in `tests/test_no_bare_random_in_package.py` now partially clears (seeds.py is present; writer/manifest/api/musicality still pending — xfail correctly stays xfail until Waves 3-4 land).
- **Wave 3 (Plan 05-04)** can now import `derive_sample_seed`, `make_rngs`, `assign_split`, `save_random_state`, and the 5 domain constants without test-infrastructure work. The writer's sum-of-stems assertion doesn't directly touch these symbols, but the resume sentinel contract depends on deterministic `sample_index` → `sample_seed` projection which is now proven byte-exact.
- **Wave 4 (Plan 05-05)** — api.py's `generate(config)` will:
  1. Validate `config.global_seed is not None`
  2. `sample_seed = derive_sample_seed(config.global_seed, config.sample_index)`
  3. `rngs = make_rngs(sample_seed)` — route to the 5 domain call sites per D-19
  4. Wrap `get_musicality_score(wav_path)` in `with save_random_state():`
  5. `split = assign_split(sample_seed, config.split_ratios)`

  All four symbols land in `src/musicgen/__init__.py` indirectly via api.py's import.
- **Wave 5 (Plan 05-06)** — determinism goldens test will invoke the full pipeline and assert `sha256(mix.wav)`, 4× `sha256(midi/*.mid)`, and canonical `sha256(sample.json)` match fixtures. The D-17/D-18/D-26 formulas this plan proved byte-exact are the foundation those goldens sit on.
- **Blockers:** none.

---
*Phase: 05-productize-i-writer-manifest-seeds-determinism*
*Completed: 2026-04-19*
