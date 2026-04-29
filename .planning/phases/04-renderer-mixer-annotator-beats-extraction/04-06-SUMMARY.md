---
phase: 04-renderer-mixer-annotator-beats-extraction
plan: "06"
subsystem: integration-test
tags: [phase-4, integration-test, e2e, slow-marker, fluidsynth-guard, r-x8]
dependency_graph:
  requires:
    - 04-01 (musicgen.beats)
    - 04-02 (musicgen.renderer)
    - 04-03 (musicgen.mixer)
    - 04-04 (musicgen.annotator)
    - 04-05 (music_gen.py orchestrator collapse)
  provides:
    - tests/test_integration_full_generation.py — 2 test classes, @pytest.mark.slow, FluidSynth + sf2 guards (R-X8)
  affects:
    - Phase 5 (golden-seed WAV test — Phase 4 only asserts MIDI bit-identity)
tech_stack:
  added: []
  patterns:
    - Module-level pytestmark list for multi-condition skip (slow + no-fluidsynth + no-sf2-pool)
    - Absolute chord_pat_file path via Path(__file__).parent.parent for monkeypatch.chdir safety
    - Seeded _rng via music_gen._rng.seed(42) for MIDI reproducibility across two runs
key_files:
  created: []
  modified:
    - tests/test_integration_full_generation.py (239 insertions — Wave 0 stub replaced)
decisions:
  - "Module-level pytestmark applies all three guards (slow + skipif-no-fluidsynth + skipif-no-sf2) to both test classes — TestMidiReproducibility also requires FluidSynth because it calls create_song() which invokes renderer.render_stems before returning"
  - "chord_pat_file uses absolute path (Path(__file__).parent.parent / 'chord_patterns.txt') — monkeypatch.chdir(tmp_path) changes cwd, so a bare filename would resolve to tmp_path, not repo root"
  - "TestMidiReproducibility names the output dirs 'rep1' and 'rep2' and reads MIDI bytes directly from part subdir paths, not from the annotation dict"
metrics:
  duration: "108 seconds (~2 minutes)"
  completed: "2026-04-19T18:07:00Z"
  tasks_completed: 1
  files_modified: 1
  files_deleted: 0
---

# Phase 04 Plan 06: E2E Integration Test Summary

**One-liner:** `tests/test_integration_full_generation.py` populated with `TestFullGenerationPipeline` + `TestMidiReproducibility` — two `@pytest.mark.slow` tests gated on FluidSynth binary + sf2 pool availability, replacing the Wave 0 stub (R-X8 closed).

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Populate tests/test_integration_full_generation.py with @pytest.mark.slow E2E test | 69cd990 | tests/test_integration_full_generation.py |

## Verification Results

### pytest -v tests/test_integration_full_generation.py (dev machine — no FluidSynth)

```
collecting ... collected 2 items
tests/test_integration_full_generation.py::TestFullGenerationPipeline::test_one_part_full_pipeline SKIPPED [ 50%]
tests/test_integration_full_generation.py::TestMidiReproducibility::test_same_seed_produces_same_midi SKIPPED [100%]

2 skipped in 0.06s
```

Skip reasons visible in `-v` output:
- `fluidsynth binary not on PATH — skipping E2E integration test`
- `one or more sf/<layer>/ dirs is empty (no .sf2) — skipping E2E integration test`

### pytest --collect-only -m slow tests/test_integration_full_generation.py

```
<Class TestFullGenerationPipeline>
  <Function test_one_part_full_pipeline>
<Class TestMidiReproducibility>
  <Function test_same_seed_produces_same_midi>

2 tests collected in 0.05s
```

No `UnknownMarkWarning` — `slow` marker is recognized from pyproject.toml.

### pytest tests/ -m "not slow" -q (full suite)

```
504 passed, 2 deselected, 2 warnings in 1.31s
```

Prior baseline (Plan 04-05): 504 passed, 1 skipped. Now: 504 passed, 2 deselected (by `-m "not slow"` filter). Zero regressions.

### Phase gate checks

```
wc -l music_gen.py   → 199 (< 200 target: PASS)
test ! -f beat_anotator.py → CONFIRMED: gone
grep -r "import beat_anotator" --include="*.py" → 0 hits
pytest tests/test_no_bare_random_in_package.py → 13 passed
```

## Acceptance Criteria Verification

| Criterion | Result |
|-----------|--------|
| `@pytest.mark.slow` present | 2 matches (pytestmark + internal ref) |
| `shutil.which` guard present | 1 match |
| `def test_` count >= 2 | 2 (one per class) |
| `phase4_fields` / R-P4 field checklist | 4 matches |
| `analysis_failed` asserted | 4 matches (assert not in annotation) |
| `pytest -m "not slow"` excludes test | 2 deselected — confirmed |
| `pytest --collect-only -m slow` lists 2 tests | confirmed |
| No `UnknownMarkWarning` | confirmed |

## Deviations from Plan

### None — plan executed exactly as written.

The only design note: `TestMidiReproducibility` is also gated by the module-level `pytestmark` (requiring FluidSynth + sf2). The plan's `<test_design>` section suggested it "can run even without fluidsynth because MIDI generation is deterministic and doesn't need audio render" — however, `create_song()` calls `renderer.render_stems` (FluidSynth subprocess) in the same pipeline before returning. There is no code path to generate MIDI only without rendering. Gating both classes on FluidSynth is correct behavior. Documented as a design clarification, not a deviation.

## Known Stubs

None — no stubs. The E2E test is a full implementation. The SKIP on this dev machine is a guard (FluidSynth not installed), not a stub.

## Threat Flags

None — no new network endpoints, auth paths, or trust boundaries introduced. The test writes to pytest's `tmp_path` which is auto-cleaned.

T-04-06-01 (CI hang from missing FluidSynth): mitigated — `shutil.which("fluidsynth") is None` module-level skipif guard confirmed working (2 SKIPPED, clean report).
T-04-06-02 (test outputs on filesystem): mitigated — all artifacts under `tmp_path` (monkeypatch.chdir).
T-04-06-03 (slow test > 60s): accepted — 1-part 2-measure render; opt-in with `-m slow`.

## Self-Check: PASSED
