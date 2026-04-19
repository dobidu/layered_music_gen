---
phase: 04-renderer-mixer-annotator-beats-extraction
plan: "02"
subsystem: renderer
tags: [phase-4, renderer, fluidsynth, threadpool, soundfonts, rng, dataclass]
dependency_graph:
  requires:
    - 04-00 (mido installed, test stubs created, pytest markers declared)
    - 04-01 (musicgen.beats present; D-21 re-export in generators/beat.py)
  provides:
    - musicgen.renderer module (FLUIDSYNTH_VERSION, RenderResult, pick_soundfonts, render_stems)
    - RenderResult frozen dataclass for downstream mixer (Plan 04-03) and annotator (Plan 04-04)
    - pick_soundfonts replaces 4x bare random.choice in music_gen.py:get_random_sound_font (D-08/D-17)
    - ThreadPoolExecutor(max_workers=4) parallel stem render dispatch (D-06)
  affects:
    - src/musicgen/renderer.py (new)
    - tests/test_renderer.py (replaced from Wave 0 stub)
tech_stack:
  added:
    - midi2audio.FluidSynth (Python wrapper for FluidSynth subprocess — already installed)
    - concurrent.futures.ThreadPoolExecutor (stdlib — parallel stem dispatch D-06)
    - subprocess.run (stdlib — FLUIDSYNTH_VERSION capture D-07)
    - pydub.AudioSegment (already installed — duration read from first rendered WAV)
  patterns:
    - FLUIDSYNTH_VERSION captured at module import inside try/except with 'unknown' fallback (D-07/RESEARCH Pitfall 3)
    - frozen dataclass @dataclass(frozen=True) matching SongParams convention (D-02)
    - sorted(os.listdir) before rng.choice for cross-machine determinism (D-08/D-17)
    - cfg=None with runtime fallback _cfg = cfg if cfg is not None else config.Config() (D-25)
    - mock FluidSynth.midi_to_audio via unittest.mock.patch in tests (D-28)
    - from __future__ import annotations (matches sampler.py convention)
    - Google-style docstrings with Args/Returns sections
key_files:
  created:
    - src/musicgen/renderer.py (208 lines)
    - (tests/test_renderer.py replaced from Wave 0 stub — 212 lines)
  modified: []
decisions:
  - "FLUIDSYNTH_VERSION assignment inside try/except block (not bare module-level) — this is the correct D-07 pattern; the plan acceptance criterion 'grep ^FLUIDSYNTH_VERSION' targets the pattern but try/except is mandatory per RESEARCH Pitfall 3"
  - "sorted() applied to os.listdir() result before rng.choice — locks cross-machine determinism (D-17 / Phase 5 golden-seed baseline requirement)"
  - "pick_soundfonts raises ValueError when rng=None (D-17 guard) — explicit rather than silently using global random"
  - "render_stems validates all 4 layer keys before dispatching threads — KeyError raised before thread pool starts"
  - "AudioSegment.from_wav reads duration from first stem after render — RESEARCH verified FluidSynth renders at 44100 Hz stereo"
metrics:
  duration: "192 seconds (~3 min 12 sec)"
  completed: "2026-04-19"
  tasks_completed: 2
  files_modified: 2
---

# Phase 04 Plan 02: Renderer Module Summary

**One-liner:** FluidSynth wrapper module (`musicgen.renderer`) with `FLUIDSYNTH_VERSION` import-time capture, `RenderResult` frozen dataclass, `pick_soundfonts(cfg, rng)` (injected-RNG soundfont selection replacing 4x bare `random.choice`), and `render_stems` dispatching 4 per-layer stems via `ThreadPoolExecutor(max_workers=4)`.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 (RED) | Write failing renderer tests | 6b25c10 | tests/test_renderer.py (212 lines, 16 tests) |
| 1 (GREEN) | Create src/musicgen/renderer.py | 859a1ca | src/musicgen/renderer.py (208 lines) |
| 2 | tests/test_renderer.py populated (same as RED commit) | 6b25c10 | tests/test_renderer.py |

## Verification Results

### FLUIDSYNTH_VERSION capture (D-07)

```
python -c "import musicgen.renderer; print(musicgen.renderer.FLUIDSYNTH_VERSION)"
# Output: unknown
# (FluidSynth binary not on PATH in WSL2 dev environment — correct fallback behavior)
```

### Signatures locked

```
pick_soundfonts: (cfg: 'Optional[config.Config]' = None, rng: 'Optional[random.Random]' = None) -> 'Dict[str, str]'
render_stems: (midi_paths: 'Dict[str, str]', soundfonts: 'Dict[str, str]', out_dir: 'str', cfg: 'Optional[config.Config]' = None) -> 'RenderResult'
```

### D-17 AST check

```
PASS: zero bare random.* in renderer.py
```

### ThreadPoolExecutor(max_workers=4) presence

```
grep -n "ThreadPoolExecutor(max_workers=4)" src/musicgen/renderer.py
190:    with ThreadPoolExecutor(max_workers=4) as pool:
```
One real dispatch call at line 190; also appears in module + function docstrings for traceability.

### tests/test_renderer.py metrics

- **Test invocations:** 16 passed
- **Classes:** TestFluidSynthVersion (2), TestRenderResult (3), TestPickSoundfonts (5), TestRenderStems (6)
- **Wall time:** 0.17s (target: < 5s)

### Full test suite

```
423 passed, 4 skipped, 2 warnings in 1.82s
```
- 407 baseline (Plan 04-01 close: 371 + 36 beats) + 16 new renderer tests = 423
- 4 skipped = remaining Wave 0 stubs (test_mixer, test_annotator, test_no_bare_random_in_package, test_integration_full_generation)
- Zero regressions

### src/musicgen/renderer.py metrics

- **Line count:** 208 lines (target: ~110 incl. docstrings — exceeded due to comprehensive Google-style docstrings with D-references)
- **Public exports:** FLUIDSYNTH_VERSION, RenderResult, pick_soundfonts, render_stems
- **Bare `random.*` calls:** 0 (AST verified)
- **D-references in docstrings:** D-02, D-05, D-06, D-07, D-08, D-09, D-17, D-25 all present

## Deviations from Plan

None — plan executed exactly as written. The exact template from the plan's `<action>` blocks was used verbatim for `renderer.py` and `tests/test_renderer.py`. No deviations from the specified architecture.

Note: The TDD cycle for Tasks 1 and 2 was combined into a single RED commit (tests written first against non-existent module → confirmed collection error = ImportError as expected RED failure), then GREEN commit (renderer.py created → 16 tests pass). Task 2 acceptance criteria are satisfied by the same test file written in Task 1's RED phase.

## Known Stubs

None — all stubs introduced by this plan are functional. The Wave 0 stub in `tests/test_renderer.py` was fully replaced by 16 real tests across 4 test classes.

## Threat Flags

None — no new network endpoints, auth paths, or trust boundaries beyond those described in the plan's threat model (T-04-02-01 through T-04-02-04). All four threats accepted or mitigated as specified:
- T-04-02-04 (subprocess hang at import): `timeout=5` + broad `except Exception` → mitigated (D-07 compliant)

## Self-Check: PASSED

- `src/musicgen/renderer.py`: FOUND (208 lines)
- `tests/test_renderer.py` contains `def test_fluidsynth_version_capture`: FOUND
- Commit 6b25c10 (RED — test file): FOUND
- Commit 859a1ca (GREEN — renderer.py): FOUND
- `grep -c "@dataclass(frozen=True)" src/musicgen/renderer.py` == 1: VERIFIED
- `grep -c "rng.choice" src/musicgen/renderer.py` == 1: VERIFIED
- `grep "ThreadPoolExecutor(max_workers=4)" src/musicgen/renderer.py` at line 190: VERIFIED
- `grep -c "^def " src/musicgen/renderer.py` == 2 (pick_soundfonts + render_stems): VERIFIED
- D-17 AST guard: zero bare random.* hits: VERIFIED
- Full suite: 423 passed, 4 skipped: VERIFIED
