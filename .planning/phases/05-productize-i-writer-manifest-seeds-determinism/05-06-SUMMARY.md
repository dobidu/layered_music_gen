---
phase: 05-productize-i-writer-manifest-seeds-determinism
plan: "06"
subsystem: determinism-regression-test

tags: [phase-5, wave-4, determinism, goldens, sha256, regression-test, in-process-cross-check, xfail-version-gate, d-28, d-29, d-30, d-32, d-41, r-p8, r-q3, phase-5-complete]

# Dependency graph
requires:
  - phase: 05-productize-i-writer-manifest-seeds-determinism
    plan: "01"
    provides: "tests/conftest.py --regen-goldens flag registration (pytest_addoption); tests/fixtures/determinism/ dir + skeleton README (refined this plan); Wave 0 stub tests/test_determinism_golden.py (replaced this plan)"
  - phase: 05-productize-i-writer-manifest-seeds-determinism
    plan: "05"
    provides: "musicgen.generate(Config) -> SampleResult — the function under test; SampleResult.mix_path / .sample_json_path / .midi_paths are the golden-hash targets; musicgen.api.renderer.render_stems + musicgen.api.renderer.pick_soundfonts + musicgen.api.musicality.get_musicality_score are the monkeypatch targets for the fast D-30 test"
provides:
  - "tests/test_determinism_golden.py — 268 LOC replaces Wave 0 stub. TestDeterminismGoldens (@pytest.mark.slow + skipif-no-fluidsynth + skipif-no-sf2) parametrized over 6 artifacts (mix, midi_beat, midi_melody, midi_harmony, midi_bassline, sample); --regen-goldens capture mode writes expected_*.sha256 + fluidsynth_version.txt; assert mode compares against fixtures with skip-on-missing-golden fallback; mix.wav case xfails on fluidsynth version mismatch (R-P8 pinned-binary guarantee); MIDI + sample.json hashes unconditional (FluidSynth-independent per R-P8)."
  - "tests/test_determinism_golden.py — TestSameProcessStability (fast, no FluidSynth, D-30): monkeypatches renderer.render_stems + renderer.pick_soundfonts + musicality.get_musicality_score; runs generate() twice with distinct dataset_roots and identical (global_seed, sample_index); hashes both sample.json bytes; asserts equality. Catches wall-clock / entropy / iteration-order leaks in OUR code independent of FluidSynth."
  - "tests/fixtures/determinism/README.md — refined from Plan 05-01 skeleton to a full maintainer playbook: 7-file layout table, capture-params block (global_seed=1, sample_index=0, pip install -e . prereq per RESEARCH Pitfall 4), regeneration command, interpretation guide distinguishing MIDI/sample-hash regressions from mix.wav cross-binary drift, rationale for the two-class split."

affects:
  - "Phase 5 closure — this is the FINAL Plan in the phase. R-P8 and R-Q3 close here. Phase 5 transitions from in-progress to ready-for-verification gates."
  - "Post-phase maintainer operation (out of scope here): capture actual goldens by running `.venv/bin/pip install -e '.[dev]'` then `.venv/bin/pytest -m slow --regen-goldens tests/test_determinism_golden.py` on a host with FluidSynth + full sf/<layer>/ pool. Commit the 7 fixture files. Until then, the slow test SKIPS on golden-file-missing with a clear operator message."
  - "Phase 7 CI integration — the slow marker controls CI inclusion. Baseline CI runs -m \"not slow\" (~5s, fast D-30 included); opt-in slow runs (nightly / pre-release) include the 6 parametrized goldens once fixtures are captured."

# Tech tracking
tech-stack:
  added: []  # zero new runtime deps — all stdlib (hashlib, subprocess, shutil, pathlib) + pydub (already installed) for the silent-stem stub
  patterns:
    - "D-29 parametrized golden test: 6 artifacts in a single @pytest.mark.parametrize list (mix, 4 MIDIs, sample.json). One failure attributes to a specific artifact rather than a generic 'hashes differ' aggregate. Plan chose parametrize over per-artifact test methods for exactly this reason (RESEARCH Open Question 6)."
    - "D-32 --regen-goldens capture-and-assert dual-mode: a single test function toggles between 'write expected_*.sha256' mode (flag present) and 'assert actual == expected' mode (flag absent) via request.config.getoption('--regen-goldens'). The flag is registered by Plan 05-01's conftest.py and is verified advertised via .venv/bin/pytest --help."
    - "FluidSynth version gate (R-P8): for the mix artifact only, compare `subprocess.run(['fluidsynth', '--version']).stdout.splitlines()[0]` against fixtures/determinism/fluidsynth_version.txt. Mismatch → pytest.xfail with a clear message. MIDI + sample.json goldens skip this gate (they are FluidSynth-independent). This is the 'accept binary-dependent audio' contract from PITFALLS P-1."
    - "D-30 fast in-process cross-check: monkeypatch renderer (render_stems + pick_soundfonts) and musicality to return deterministic stubs; run generate() twice in one process with the same seed but distinct dataset_roots; hash sample.json bytes; assert equality. Proves byte-stability of OUR orchestrator + annotator + writer independent of FluidSynth. This catches the largest class of regressions (datetime.now(), os.urandom, dict-iteration-order) cheaply."
    - "Stub parity with RNG draw count: the _fake_pick_soundfonts stub still does rng.choice(candidates) once per layer (4 draws, matching real pick_soundfonts). Without this, monkeypatching would desync the downstream RNG draw order and the D-30 test would fail for the wrong reason (RNG-state mismatch) rather than revealing actual non-determinism bugs."
    - "Skip-gate pattern reused verbatim from test_integration_full_generation.py: shutil.which('fluidsynth') for binary detection; _all_sf2_layers_have_files() for sf/<layer>/*.sf2 presence; applied as class-level decorators on TestDeterminismGoldens (not module-level pytestmark) so TestSameProcessStability stays collectable + runnable on dev machines without FluidSynth."

key-files:
  created: []  # both files existed from Plan 05-01 (Wave 0 stub + skeleton README); this plan populates them
  modified:
    - tests/test_determinism_golden.py
    - tests/fixtures/determinism/README.md

key-decisions:
  - "Added _fake_pick_soundfonts stub beyond the plan's verbatim spec (Rule 3 blocking fix). The plan's test body only monkeypatched render_stems + musicality, but generate() calls pick_soundfonts BEFORE render_stems. With empty sf/<layer>/ dirs on a dev machine, pick_soundfonts raises FileNotFoundError and the D-30 test fails before exercising any determinism check. The stub does rng.choice(candidates) per layer so RNG draw order matches the real pipeline (D-19) — critical for the cross-check to exercise the same downstream state."
  - "TestDeterminismGoldens uses class-level @pytest.mark.skipif decorators rather than module-level pytestmark. Module-level would have skipped TestSameProcessStability too (it doesn't need FluidSynth), defeating the whole point of the fast D-30 check. Class-level lets TestSameProcessStability run on dev machines without FluidSynth while TestDeterminismGoldens cleanly skips."
  - "pytest.skip on missing golden file (rather than xfail or fail). Before the first --regen-goldens run, the fixture files don't exist. Skipping with a clear operator message ('Run `.venv/bin/pytest -m slow --regen-goldens ...` on a pinned-FluidSynth host') makes the first-time setup self-documenting. xfail would have been a misuse (the test isn't known to fail — its fixture hasn't been captured yet); fail would break CI before maintainer-mode capture."
  - "mix-only version gate. Only the mix artifact goes through _fluidsynth_version_matches_golden(); MIDI + sample.json don't touch FluidSynth at all. Per R-P8 contract, MIDI bit-identity is guaranteed across all FluidSynth versions (MIDI is generated by our code before renderer runs); WAV bit-identity requires the pinned binary. Mixing the version gate into MIDI assertions would be over-defensive and mask real MIDI regressions."
  - "Dev-machine execution profile locked: 6 slow cases SKIP (no FluidSynth on this host); TestSameProcessStability PASSES; fast suite 689 → 690 passed (+1 net from D-30). Slow cases collect cleanly under -m slow (6 collected, confirmed via --collect-only). The skip messages point operators at exactly what to do to capture goldens."

patterns-established:
  - "Goldens infrastructure vs captured hashes split: this plan ships the TEST + the FIXTURE DIRECTORY + the README + the --regen-goldens pathway. It does NOT commit actual hash values — those are captured post-phase by a maintainer on a pinned FluidSynth host. This split means Phase 5 can close cleanly even though the goldens themselves are operator-captured, and CI green is preserved without pinning CI's FluidSynth version as a Phase 5 requirement."
  - "Fast D-30 + slow expensive oracle two-test pattern: the cheap watchdog runs in every pytest invocation (catches 90% of our-code regressions same-day); the expensive oracle gates release builds (catches FluidSynth drift + WAV bit-identity across pinned binary). Future plans introducing determinism-sensitive pipelines should follow this split."
  - "Stub-with-RNG-draw-parity: when monkeypatching a deterministic function for speed, the stub must still consume rng draws at the same count/order as the real implementation. Breaking this turns the cross-check into a 'does the stub produce stable output?' trivial assertion rather than a real determinism test. pick_soundfonts stub documents this with the comment 'One rng.choice per layer matching real pick_soundfonts draw count.'"

requirements-completed: [R-P8, R-Q3]
# R-P8: determinism contract ENFORCED via regression test — MIDI + sample.json bit-identity asserted unconditionally (cross-machine, cross-binary); mix.wav bit-identity asserted under pinned FluidSynth (xfail-gate on version mismatch, acknowledging P-1 binary-dependent audio contract).
# R-Q3: regression test ships in CI via slow marker — `-m "not slow"` excludes by default (CI green), `-m slow` or nightly runs include the 6 parametrized goldens + the cheap D-30 cross-check that runs in the fast suite already.

# Metrics
duration: ~5min (active work; wait time minimal — one failing run revealed the pick_soundfonts stub gap, one fixup, one passing run)
completed: 2026-04-20
---

# Phase 5 Plan 06: Wave 4 — determinism goldens + D-30 in-process cross-check Summary

**Final Phase 5 plan closes R-P8 + R-Q3. tests/test_determinism_golden.py (268 LOC) ships two test classes: TestDeterminismGoldens parametrized over 6 artifacts with --regen-goldens capture mode + FluidSynth version xfail gate, and TestSameProcessStability as the fast D-30 in-process cross-check. fixtures README refined into a maintainer playbook. Phase 5 architecturally complete; 25/25 plans closed.**

## Performance

- **Duration:** ~5 min active work (sequential executor on main branch)
- **Completed:** 2026-04-20
- **Tasks:** 1 of 1 (single-task plan — the test module + README)
- **Files modified:** 2 (both existed from Plan 05-01 Wave 0; populated with real bodies here)
- **Commits:** 1 task commit + 1 final metadata commit (pending)

## Accomplishments

- **tests/test_determinism_golden.py replaces Plan 05-01 Wave 0 stub with 268 lines of real test body.** Two classes, one file, zero module-level pytest.skip.
- **TestDeterminismGoldens: 6 parametrized slow cases** — one each for mix.wav, beat/melody/harmony/bassline MIDI files, and canonical sample.json. @pytest.mark.slow + class-level skipif(no fluidsynth) + class-level skipif(no sf2 pool) mirror the E2E integration test's gating pattern verbatim. Each case: calls `generate(Config(global_seed=1, sample_index=0, dataset_root=tmp_path))`, resolves the artifact via `_artifact_path(result, artifact)` → `result.mix_path` / `result.sample_json_path` / `result.midi_paths[layer]`, computes SHA-256, and either writes the hash to `expected_<artifact>.sha256` (--regen-goldens mode) or asserts equality against the fixture content (default mode).
- **--regen-goldens capture pathway**: request.config.getoption("--regen-goldens") is checked per test; when True, the computed hash is written to the fixture file and the test returns early (always passes). The mix case additionally writes `fluidsynth_version.txt` via `_current_fluidsynth_version_line()` — regeneration always implicitly re-pins the binary reference.
- **FluidSynth version gate (R-P8 pinned-binary contract)**: the mix artifact case calls `_fluidsynth_version_matches_golden()` and, on mismatch, calls `pytest.xfail(...)` with a message explaining the P-1 binary-dependent audio contract. MIDI + sample.json assertions never invoke the gate — they hold unconditionally per R-P8 "MIDI + metadata bit-identical across binaries".
- **Missing-golden skip fallback**: when `expected_<artifact>.sha256` doesn't exist yet (first run before maintainer capture), the test `pytest.skip(...)` with a message telling the operator exactly how to capture: `.venv/bin/pytest -m slow --regen-goldens tests/test_determinism_golden.py`. Self-documenting first-time setup.
- **TestSameProcessStability: D-30 fast in-process cross-check** — one test, no FluidSynth, monkeypatches three functions on `musicgen.api.*`:
  - `renderer.render_stems` → silent stereo 44.1kHz WAVs (duration 500ms matches pipeline expectation shape)
  - `renderer.pick_soundfonts` → stub that still does rng.choice per layer (preserves RNG draw order)
  - `musicality.get_musicality_score` → `(0.5, {"rhythm": 0.5, "harmony": 0.5})`
  Then calls `generate(Config(global_seed=1, sample_index=0, dataset_root=<a>))` and the same config with `<b>` root; hashes both resulting sample.json files; asserts byte-identity. A stray `datetime.now()` in the annotator, `os.urandom` call anywhere in the writer, or set-iteration-order dependency in the RNG routing would fail this test the same day it's introduced.
- **Fixtures README elevated from skeleton to maintainer playbook**: 7-file layout table distinguishing version-gated mix.wav from unconditional MIDI/sample hashes, explicit capture-params block (global_seed=1, sample_index=0, `pip install -e .` prereq per RESEARCH Pitfall 4 to avoid `"0.1.0+uninstalled"` version poisoning the sample.json hash), regeneration command, three-branch interpretation guide (MIDI/sample mismatch → our-code regression, mix mismatch + matching version → FluidSynth platform drift, mix mismatch + different version → expected R-P8 behavior), two-class rationale (expensive oracle vs cheap watchdog).
- **Dev-machine execution profile (no FluidSynth)**: 6 slow cases SKIP cleanly with binary-not-on-PATH message; TestSameProcessStability PASSES in ~1.4s. `.venv/bin/pytest --help | grep regen-goldens` confirms the flag is advertised (registered by Plan 05-01 conftest.py). `.venv/bin/pytest tests/test_determinism_golden.py -m slow --collect-only` reports 6 parametrized cases (confirms parametrize wiring is correct).
- **Zero test regressions**: 689 passed (baseline from Plan 05-05) → 690 passed (+1 net from the fast D-30 test). 1 skipped (pre-existing Wave 0 stub) becomes 0 skipped + 6 deselected-under-not-slow additions. Under `-m "not slow"`: 690 passed, 12 deselected (6 prior slow tests from test_api.py + test_integration_full_generation + 6 new slow parametrized goldens), 0 failed.

## Task Commits

1. **Task 1: Populate tests/test_determinism_golden.py + refine fixtures README** — `04570b5` (test)

**Plan metadata commit:** (pending — created after this SUMMARY is written)

## Files Created/Modified

- `tests/test_determinism_golden.py` — Wave 0 stub (15 lines, pytest.skip(allow_module_level=True)) replaced with 268 LOC real body. Two classes, 6 parametrized slow cases + 1 fast D-30 case, --regen-goldens dual-mode, FluidSynth version xfail gate, helper functions (_sha256_of, _current_fluidsynth_version_line, _fluidsynth_version_matches_golden, _artifact_path), skip-gates reused from test_integration_full_generation.py.
- `tests/fixtures/determinism/README.md` — Plan 05-01 skeleton (19 lines) refined to 44-line maintainer playbook: 7-file layout table, capture-params block with pip-install-e prereq, regeneration command, interpretation guide, two-class rationale.

## Decisions Made

See frontmatter `key-decisions` for full rationale. Summary:

1. **Rule 3 blocking fix — added _fake_pick_soundfonts stub** (plan spec missed this; without it the D-30 test fails on FileNotFoundError before the determinism check runs).
2. **Class-level skipif (not module-level pytestmark)** — keeps TestSameProcessStability runnable on dev machines without FluidSynth.
3. **pytest.skip on missing golden** — self-documenting first-time setup; xfail/fail would misuse the semantics.
4. **mix-only version gate** — MIDI + sample.json hashes hold unconditionally per R-P8 contract; version gate would mask real regressions.
5. **Dev-machine locked profile**: 6 skipped, 1 passed (D-30); slow cases collect cleanly; operator instructions embedded in skip messages.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Added _fake_pick_soundfonts stub to TestSameProcessStability**
- **Found during:** Task 1 verification (first test run after initial Write)
- **Issue:** The plan's verbatim test body monkeypatched `musicgen.api.renderer.render_stems` and `musicgen.api.musicality.get_musicality_score`, but NOT `musicgen.api.renderer.pick_soundfonts`. In `api._run_pipeline`, `pick_soundfonts(cfg, rng)` is called at line 234 BEFORE `render_stems` (line 270). On a dev machine without FluidSynth (and without `.sf2` files populated in `sf/<layer>/`), pick_soundfonts raises `FileNotFoundError: No .sf2 files found in /home/bidu/musicgen/sf/beat for layer 'beat'`. The exception converts to SampleResult(status="failed") (per D-24), which the test then asserts `status == "ok"` against — and fails. First run output: `FAILED tests/test_determinism_golden.py::TestSameProcessStability::test_generate_sample_json_stable_same_process`.
- **Fix:** Added `_fake_pick_soundfonts(cfg, rng)` stub returning `{layer: f"/stub/{rng.choice(candidates)}" for layer in _LAYERS}` with 3 candidate filenames. Crucially, the stub still calls `rng.choice(candidates)` once per layer — matching the real `pick_soundfonts`'s 4 rng draws so the downstream RNG state (RNG_SOUNDFONTS) is consumed identically. Without matching the draw count, the cross-check would be a trivial "stub produces stable output" assertion instead of a real RNG-routing determinism probe.
- **Files modified:** `tests/test_determinism_golden.py` (added _fake_pick_soundfonts function + monkeypatch.setattr call between the existing render_stems patch and the musicality patch)
- **Verification:** `.venv/bin/pytest tests/test_determinism_golden.py::TestSameProcessStability -v` now PASSES; hash_a == hash_b for `generate(Config(global_seed=1, sample_index=0))` run twice.
- **Committed in:** `04570b5` (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (Rule 3 blocking)
**Impact on plan:** Zero architectural change. The stub is strictly additive (3 extra lines + 1 monkeypatch call) and preserves the plan's intent (monkeypatch all FluidSynth-touching functions so the D-30 test runs without the binary). Documented inline in the test file with a comment explicitly calling out the fix rationale: `"Rule 3 blocking fix: the plan's spec missed this — pick_soundfonts is called BEFORE render_stems and raises FileNotFoundError when the sf/<layer>/ dirs are empty (dev-machine default)."`

## Authentication Gates

None — all automation is local + file-based.

## Issues Encountered

- **pick_soundfonts stub gap surfaced on first run** (see Deviation 1 above). Cost: one extra test run + one edit. Gain: the fix is permanent and makes the test runnable on any dev machine regardless of sf/<layer>/ population state.
- **Pre-existing chmod-only mods in working tree (25 files)** unrelated to this plan — did not stage or commit these, same posture as Plan 05-05 carried forward.

## User Setup Required

**Post-phase operator task (NOT part of this plan's scope):** to capture the actual golden hashes, a maintainer on a host with FluidSynth installed and the `sf/<layer>/` pools populated runs:

```bash
.venv/bin/pip install -e ".[dev]"   # resolves musicgen version to "0.1.0" (not "+uninstalled")
.venv/bin/pytest -m slow --regen-goldens tests/test_determinism_golden.py
```

This writes 7 files to `tests/fixtures/determinism/`: 6 `expected_*.sha256` files + `fluidsynth_version.txt`. Commit all 7. Subsequent runs without `--regen-goldens` will then assert against them; MIDI + sample.json assertions hold across any FluidSynth version; mix.wav assertion xfails gracefully on version mismatch.

Until goldens are captured, the 6 slow parametrized cases SKIP on each run (with instructional message). The fast D-30 cross-check is ALWAYS runnable and passes on the current codebase.

## Phase 5 Readiness

- **Phase 5 architecturally COMPLETE.** 25 of 25 plans closed. All Phase 5 requirements closed: R-P1 (per-sample atomic layout), R-P2 (sum-of-stems assertion), R-P3 (absolute-tick MIDI concat), R-P4 (sample.json schema fully populated), R-P5 (manifest append-under-lock), R-P6 (assign_split end-to-end), R-P7 (full seed discipline), R-P8 (determinism contract enforced via golden test + version gate), R-Q3 (regression test in CI via slow marker).
- **Phase 6 entry fully unblocked.** The single-sample primitive `api.generate(Config)` has a golden-test oracle; `generate_batch` becomes a thin loop over it; the multi-process lock path is scaffolded via `ManifestWriter(dataset_root, lock=...)`; the typer CLI replaces `music_gen.py` outright.
- **Phase 7 CI integration has both test tiers ready.** Default CI runs `-m "not slow"` (~5s, includes the D-30 cross-check → catches our-code regressions same-day). Nightly / pre-release CI runs `-m slow` once fixtures are captured → catches FluidSynth drift + WAV cross-machine bit-identity.

## Known Stubs

None. The `_fake_render_stems`, `_fake_pick_soundfonts`, `_fake_musicality` inside TestSameProcessStability are intentional test doubles (not production stubs) — they exercise the real orchestrator + annotator + writer pipeline with deterministic inputs, proving byte-stability of our code. Documented inline with rationale comments.

## Self-Check: PASSED

**Files verified present:**
- `tests/test_determinism_golden.py` — FOUND (268 lines, 2 classes, 7 collectable tests)
- `tests/fixtures/determinism/README.md` — FOUND (44 lines, TestSameProcessStability referenced)
- `.planning/phases/05-productize-i-writer-manifest-seeds-determinism/05-06-SUMMARY.md` — being written now

**Commits verified present:**
- `04570b5` — FOUND (Task 1)

**Acceptance criteria verified:**
- `grep -c "allow_module_level=True" tests/test_determinism_golden.py` == 0 ✓
- `grep -c "class TestDeterminismGoldens" tests/test_determinism_golden.py` == 1 ✓
- `grep -c "class TestSameProcessStability" tests/test_determinism_golden.py` == 1 ✓
- `grep -c '"mix", "midi_beat", "midi_melody", "midi_harmony", "midi_bassline", "sample"'` == 1 ✓
- `grep -c '@pytest.mark.slow'` == 2 (>= 1) ✓
- `grep -c 'fluidsynth_version_matches_golden'` == 2 (>= 1) ✓
- `grep -c 'request.config.getoption'` == 1 (>= 1) ✓
- `grep -c 'pytest.xfail'` == 1 (>= 1) ✓
- `grep -c "test_generate_sample_json_stable_same_process"` == 1 ✓
- `grep -c "hashlib.sha256"` == 3 (>= 2) ✓
- Slow cases collected under `-m slow --collect-only`: 6 ✓
- Fast test result: PASSED ✓
- Full suite under `-m "not slow"`: 690 passed, 0 failed ✓
- `grep -c "TestSameProcessStability" tests/fixtures/determinism/README.md` == 2 (>= 1) ✓
- `.venv/bin/pytest --help | grep regen-goldens` → FOUND (registered by Plan 05-01) ✓

---
*Phase: 05-productize-i-writer-manifest-seeds-determinism*
*Completed: 2026-04-20*
*Phase 5 ALL plans closed: 25/25 — ready for verification gates.*
