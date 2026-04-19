---
phase: 4
slug: renderer-mixer-annotator-beats-extraction
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-19
---

# Phase 4 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> Derived from `04-RESEARCH.md §Validation Architecture` (post-research corrections).

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 9.0.3 |
| **Config file** | `pyproject.toml` `[tool.pytest.ini_options]` |
| **Quick run command** | `python -m pytest tests/ -m "not slow" -q --tb=short` |
| **Full suite command** | `python -m pytest tests/ -q` |
| **Slow tests only** | `python -m pytest tests/ -m slow -q` |
| **Estimated runtime (quick)** | ~8 seconds (371 baseline + ~30 new Phase 4 tests) |
| **Estimated runtime (full)** | ~15–60 seconds (adds the E2E `@pytest.mark.slow` FluidSynth render) |

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest tests/ -m "not slow" -q --tb=short`
- **After every plan wave:** Run `python -m pytest tests/ -m "not slow" -q`
- **Before `/gsd-verify-work`:** Full suite (minus `@pytest.mark.slow` on CI without FluidSynth) must be green
- **Max feedback latency:** 15 seconds (quick run)

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 04-00-01 | 00 | 0 | Infra | — | N/A | static | `pip show mido` (expect "Name: mido" + "Version: >=1.3.3") | ❌ W0 | ⬜ pending |
| 04-00-02 | 00 | 0 | Infra | — | N/A | static | `python -c "import tomllib,pathlib; d=tomllib.loads(pathlib.Path('pyproject.toml').read_text()); assert 'markers' in d['tool']['pytest']['ini_options']"` | ❌ W0 | ⬜ pending |
| 04-01-01 | 01 | 1 | R-X7 | — | Deterministic pure function | unit | `pytest tests/test_beats.py::test_beat_duration -x` | ❌ W0 | ⬜ pending |
| 04-01-02 | 01 | 1 | R-X7 | — | MIDI-tick extraction swing-aware | unit | `pytest tests/test_beats.py::TestExtractBeatTimes -x` | ❌ W0 | ⬜ pending |
| 04-01-03 | 01 | 1 | R-X7 | — | 3 swing cases (0.5/0.66/0.75) | unit | `pytest tests/test_beats.py::TestSwingCases -x` | ❌ W0 | ⬜ pending |
| 04-01-04 | 01 | 1 | R-X7 | — | Downbeat count == measure count | unit | `pytest tests/test_beats.py::test_downbeat_count -x` | ❌ W0 | ⬜ pending |
| 04-02-01 | 02 | 2 | R-X4 | — | FluidSynth version captured at import | unit | `pytest tests/test_renderer.py::test_fluidsynth_version_capture -x` | ❌ W0 | ⬜ pending |
| 04-02-02 | 02 | 2 | R-X4 | T-path-traversal | `out_dir` trusted, writes stay under it | unit (mocked FluidSynth) | `pytest tests/test_renderer.py::TestRenderStems -x` | ❌ W0 | ⬜ pending |
| 04-02-03 | 02 | 2 | R-X4 | — | `pick_soundfonts` uses injected rng | unit | `pytest tests/test_renderer.py::test_pick_soundfonts_deterministic -x` | ❌ W0 | ⬜ pending |
| 04-03-01 | 03 | 3 | R-X5 | — | `build_fx_boards` seeded determinism | unit | `pytest tests/test_mixer.py::test_build_fx_boards_deterministic -x` | ❌ W0 | ⬜ pending |
| 04-03-02 | 03 | 3 | R-X5 | — | `compute_layer_mask` seeded determinism | unit | `pytest tests/test_mixer.py::test_compute_layer_mask_deterministic -x` | ❌ W0 | ⬜ pending |
| 04-03-03 | 03 | 3 | R-X5 | — | Silent-stem channels==2, frame_rate==44100 | unit | `pytest tests/test_mixer.py::test_silent_stem_channels -x` | ❌ W0 | ⬜ pending |
| 04-03-04 | 03 | 3 | R-X5 (D-11) | — | FX applied to all 4 layers regardless of mask | unit | `pytest tests/test_mixer.py::test_fx_applied_to_all_layers -x` | ❌ W0 | ⬜ pending |
| 04-03-05 | 03 | 3 | R-S4 (preserve) | — | `apply_gain` path works (regression) | unit | `pytest tests/test_mixer.py::test_apply_gain_pan_fix -x` | ❌ W0 | ⬜ pending |
| 04-04-01 | 04 | 4 | R-X6 | — | Annotator returns dict; Phase-4 fields non-None | unit (fixture) | `pytest tests/test_annotator.py::test_phase4_fields_filled -x` | ❌ W0 | ⬜ pending |
| 04-04-02 | 04 | 4 | R-X6 (D-16) | — | Phase-5 TBD fields present as None | unit (fixture) | `pytest tests/test_annotator.py::test_tbd_fields_are_none -x` | ❌ W0 | ⬜ pending |
| 04-04-03 | 04 | 4 | R-X6 | — | No I/O inside annotator | unit (monkeypatch) | `pytest tests/test_annotator.py::test_annotator_is_pure -x` | ❌ W0 | ⬜ pending |
| 04-05-01 | 05 | 5 | D-23 | — | `mix_and_save` deleted from music_gen.py | static | `pytest tests/test_music_gen_logging.py -x` + `python -c "from music_gen import mix_and_save" 2>&1` exits non-zero | ❌ W0 | ⬜ pending |
| 04-05-02 | 05 | 5 | D-03 (beat_anotator delete) | — | `beat_anotator.py` absent | static | `test ! -f beat_anotator.py` | ❌ W0 | ⬜ pending |
| 04-05-03 | 05 | 5 | D-17/D-31 | — | Zero bare `random.<method>` in `src/musicgen/**/*.py` | static (AST) | `pytest tests/test_no_bare_random_in_package.py -x` | ❌ W0 | ⬜ pending |
| 04-05-04 | 05 | 5 | Smoke preservation | — | `python music_gen.py` reaches annotator/mixer | manual | `python music_gen.py` → logs reach mix+annotator stage before env failure | ❌ W0 | ⬜ pending |
| 04-06-01 | 06 | 6 | R-X8 | — | E2E integration test: 4 stems + 1 mix + 4 MIDI exist + annotation complete | integration (slow) | `pytest tests/test_integration_full_generation.py -m slow -x` | ❌ W0 | ⬜ pending |
| 04-REG | — | all | Regression baseline | — | 371 baseline tests + new tests green | regression | `pytest tests/ -m "not slow" -q` | YES (371 tests) | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `pyproject.toml` — add `mido>=1.3.3` to `[project].dependencies`
- [ ] `pyproject.toml` — add `markers` list under `[tool.pytest.ini_options]` (declares `slow` + `integration`)
- [ ] `pip install -e '.[dev]'` — refresh the editable install so `import mido` works
- [ ] `tests/test_beats.py` — 4/4 grid + 3 swing cases + downbeat count (stubs before Wave 1 starts)
- [ ] `tests/test_renderer.py` — mocked FluidSynth unit tests (stubs before Wave 2 starts)
- [ ] `tests/test_mixer.py` — seeded-RNG + silent-stem + FX-all-layers (stubs before Wave 3 starts)
- [ ] `tests/test_annotator.py` — fixture-driven golden dict (stubs before Wave 4 starts)
- [ ] `tests/test_no_bare_random_in_package.py` — package-wide AST guard (stub before Wave 5 starts)
- [ ] `tests/test_integration_full_generation.py` — `@pytest.mark.slow` E2E (stub before Wave 6)

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| `python music_gen.py` smoke test reaches annotator | Smoke preservation (roadmap exit) | Requires FluidSynth binary + `.sf2` files in `sf/<layer>/`; CI environment has neither | Run `python music_gen.py` on a dev machine with FluidSynth installed and at least one `.sf2` per layer; confirm logs reach "mix part" + "annotation" stage (may fail at musicality scoring if `librosa` deps missing — acceptable) |
| Auditory sanity check (Phase 5 pre-flight) | Subjective quality guard | FX / mix quality cannot be asserted via code | Listen to one `mix.wav` produced during a dev smoke test; confirm four layers audible when all four are in the mask |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies listed above
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify (max gap is Wave 5 music_gen.py collapse → E2E bridge, covered by 04-05-03 AST + 04-05-04 smoke + 04-06-01 E2E)
- [ ] Wave 0 covers all ❌ W0 references in the per-task map
- [ ] No watch-mode flags (all commands exit deterministically)
- [ ] Feedback latency < 15s quick, < 60s full
- [ ] `nyquist_compliant: true` will flip once Wave 0 lands and every task has its automated command exercising green

**Approval:** pending (auto-mode — will be flipped to approved by executor's verification pass)

---

## Deviations from CONTEXT.md Locked by Research

Research (`04-RESEARCH.md`) surfaced three CONTEXT.md decisions that need amendment:

1. **D-20 downbeat stride-slice is incorrect.** Beat patterns have zeros (e.g., `4/4: intro: 0, 42, 38, 0`), so `extract_beat_times` returns sparse beat_times. Correct approach: derive downbeats via time-grid computation (`beat_duration(signature, tempo) × numerator × measure_index`) rather than stride-slicing the sparse MIDI-derived list. `TimeSignatureRegistry` already exposes `beats_per_measure` for compound-meter pulse handling.
2. **D-12 silent-stem fallback needs explicit channels + frame_rate.** `AudioSegment.silent()` defaults to mono 11025 Hz; FluidSynth renders stereo 44100 Hz. Silent stems must call `.set_channels(2)` and pass `frame_rate=44100` for sum-of-stems parity (Phase 5 R-P2 precondition).
3. **`mido` is NOT a transitive dep.** `midi2audio` declares zero Python-level requirements. Wave 0 must add `mido>=1.3.3` to `pyproject.toml`. `beat_anotator.py` only worked because `mido` was installed incidentally in the current dev venv.

These deviations are research-driven improvements, not scope changes. Planner must adopt them verbatim.
