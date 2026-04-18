---
phase: 03
slug: package-skeleton-sampler-generators-extraction
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-18
---

# Phase 03 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> Derived from `03-RESEARCH.md` §Validation Architecture.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.x + pytest-cov 5.x + pytest-xdist 3.5+ (declared in `[project.optional-dependencies].dev`) |
| **Config file** | `pyproject.toml` (under `[tool.pytest.ini_options]`); `tests/conftest.py` is DELETED per D-16 |
| **Quick run command** | `.venv/bin/python -m pytest tests/ -q` |
| **Full suite command** | `.venv/bin/python -m pytest tests/ -v --tb=short` |
| **Estimated runtime** | ~5–10s (existing 309 tests + new sampler/generator/music21 tests) |

---

## Sampling Rate

- **After every task commit:** Run `.venv/bin/python -m pytest tests/ -q` — must report ≥309 passed (regression baseline) plus any new tests the task introduced.
- **After every plan wave:** Full suite (`pytest tests/ -v --tb=short`) + `python music_gen.py` smoke (best-effort if ffmpeg/FluidSynth installed).
- **Before `/gsd-verify-work`:**
  - `pytest tests/ -q` green.
  - `pip install -e '.[dev]'` succeeds in a fresh venv.
  - `musicgen --help` exits 0.
  - `python -c "from musicgen.sampler import SongParams"` succeeds.
  - AST scan / grep over `src/musicgen/` returns zero bare `random.*` calls (enforces D-07).
- **Max feedback latency:** ≤10 seconds for quick run; ≤2 minutes including best-effort smoke.

---

## Per-Task Verification Map

Plans are created by `gsd-planner`; this table will be regenerated with concrete `{N}-PP-TT` task IDs once PLAN.md files exist. For now, each requirement has at least one automated command:

| Requirement | Behavior Under Test | Test Type | Automated Command | File Exists? | Status |
|-------------|--------------------|-----------|-------------------|--------------|--------|
| R-X1 | `pip install -e .` succeeds; package is importable | integration-smoke | `python -c "import musicgen, musicgen.sampler, musicgen.generators.chord, musicgen.generators.melody, musicgen.generators.bassline, musicgen.generators.beat, musicgen.duration_validator"` | ❌ W0 (add `tests/test_package_install.py`) | ⬜ pending |
| R-X1 | `musicgen --help` works | smoke | `musicgen --help` (exit 0; help text contains "musicgen") | ❌ W0 | ⬜ pending |
| R-X1 | `python -m musicgen --help` works | smoke | `python -m musicgen --help` (exit 0) | ❌ W0 | ⬜ pending |
| R-X1 | `pyproject.toml` exists with hatchling + version 0.1.0 + entry point | static | `python -c "import tomllib; d=tomllib.load(open('pyproject.toml','rb')); assert d['project']['version']=='0.1.0'; assert d['project']['scripts']['musicgen']=='musicgen.cli:app'"` | ❌ W0 | ⬜ pending |
| R-X2 | `from musicgen.sampler import SongParams` works | import-check | `python -c "from musicgen.sampler import SongParams; assert hasattr(SongParams,'sample')"` | ❌ W0 | ⬜ pending |
| R-X2 | `SongParams.sample(rng, cfg)` is deterministic (same seed → same output) | unit (seeded) | `pytest tests/test_sampler.py::TestSongParamsSample::test_sample_is_deterministic -x` | ❌ W0 | ⬜ pending |
| R-X2 | All seven sampler functions (`generate_random_key`, `_tempo`, `_swing`, `generate_song_arrangement`, `generate_random_time_signature`, `generate_song_measures`, `time_signature_alternative`) are rng-aware | unit (seeded) | `pytest tests/test_sampler.py -x` (expected ≥7 tests, one per function) | ❌ W0 | ⬜ pending |
| R-X2 | Zero bare `random.*` in `src/musicgen/sampler.py` | static-AST | `pytest tests/test_sampler.py::test_no_bare_random_in_sampler -x` | ❌ W0 | ⬜ pending |
| R-X3 | Each generator (chord/melody/bassline/beat) takes injected `rng` and produces deterministic MIDI | unit (seeded) | `pytest tests/test_generators/ -x` (one file per generator) | ❌ W0 | ⬜ pending |
| R-X3 | Zero bare `random.*` in `src/musicgen/generators/*.py` | static-AST | `pytest tests/test_generators/test_no_bare_random_in_generators -x` OR unified scan under sampler test | ❌ W0 | ⬜ pending |
| D-10 | `DurationValidator` + `NoteValue` importable from `musicgen.duration_validator`; existing behavior preserved | unit | `pytest tests/test_duration_validator.py -x` (existing 37 tests must still pass after one import update) | ✅ (exists) | ⬜ pending |
| D-23 | music21 does not mutate global `random` state | unit (regression) | `pytest tests/test_music21_isolation.py -x` (≥3 tests: roman, scale, pitch) | ❌ W0 | ⬜ pending |
| D-24 | Regression test file exists and is discoverable | existence | `pytest tests/test_music21_isolation.py --collect-only -q` shows ≥3 tests | ❌ W0 | ⬜ pending |
| D-04 / Exit criterion | `python music_gen.py` runs one song end-to-end using the shim | integration-manual-or-slow | Best-effort `python music_gen.py` OR `@pytest.mark.slow` equivalent. Passes iff ffmpeg + FluidSynth are installed; otherwise documented as manual. | ❌ W0 (manual step in plan) | ⬜ pending |
| Regression | All 309 pre-existing tests continue to pass | regression | `pytest tests/ -q` (baseline: 309 passed 2026-04-18 per research §Regression) | ✅ (exists) | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

Files that must exist (even as empty scaffolds) before any task can claim green:

- [ ] `pyproject.toml` at repo root (hatchling + deps + entry point + pytest config)
- [ ] `src/musicgen/__init__.py` (empty)
- [ ] `src/musicgen/__main__.py` (one-line delegator to `musicgen.cli:app`)
- [ ] `src/musicgen/cli.py` (stub typer app, one `info` command per D-18)
- [ ] `src/musicgen/sampler.py` (scaffolded in W0, filled in W1)
- [ ] `src/musicgen/generators/__init__.py` (empty)
- [ ] `src/musicgen/generators/{chord,melody,bassline,beat}.py` (scaffolded in W0, filled in W1)
- [ ] `src/musicgen/duration_validator.py` (created via `git mv enhanced_duration_validator.py ...` — preserves history)
- [ ] `tests/test_sampler.py` (new)
- [ ] `tests/test_generators/__init__.py` + `test_chord.py`, `test_melody.py`, `test_bassline.py`, `test_beat.py` (new)
- [ ] `tests/test_music21_isolation.py` (new, D-24)
- [ ] `tests/test_package_install.py` (optional per D-17; recommended as `@pytest.mark.slow`)
- [ ] `tests/conftest.py` DELETED (D-16)
- [ ] `requirements.txt` DELETED (D-14)
- [ ] `dev-requirements.txt` DELETED (D-14)
- [ ] Verify editable install end-to-end: `pip install -e '.[dev]'` in `.venv`, then `python -c "import musicgen.sampler"` and `musicgen --help` succeed.

**Framework install:** no new framework introduced — pytest stack migrates from `dev-requirements.txt` (deleted) into `[project.optional-dependencies].dev`.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| End-to-end `python music_gen.py` produces a song directory with MIDI + WAV files | D-04 exit criterion | Requires ffmpeg + FluidSynth on host; CI may not have them. Audio outputs are too large to compare byte-for-byte here (Phase 5 handles determinism regression). | 1. Activate `.venv`. 2. `pip install -e '.[dev]'`. 3. `python music_gen.py`. 4. Confirm a dated output directory is created with per-part `.mid` and a final `.wav`. Non-zero exit or missing files = fail. |
| Fresh-venv install works (not just editable from current dev venv) | R-X1 | Requires creating a second venv — CI-worthy but adds runtime; easy to skip if flaky. | 1. `python -m venv /tmp/test-install`. 2. `/tmp/test-install/bin/pip install -e /home/bidu/musicgen`. 3. `/tmp/test-install/bin/musicgen --help`. 4. `/tmp/test-install/bin/python -c "import musicgen"`. Remove `/tmp/test-install` after. |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or declared Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without an automated verify command
- [ ] Wave 0 covers all ❌ MISSING references in the Per-Task Verification Map
- [ ] No watch-mode flags in automated commands (CI stability)
- [ ] Feedback latency < 10 seconds for quick run
- [ ] `nyquist_compliant: true` can be set in frontmatter once all above are green

**Approval:** pending (draft created 2026-04-18 — will be approved when PLAN.md tasks reference this file and all rows map to concrete task IDs)
