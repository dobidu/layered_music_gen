# Phase 3: Package skeleton + sampler + generators extraction — Research

**Researched:** 2026-04-18
**Domain:** Python packaging (hatchling + src-layout + editable install), pure-function extraction, RNG plumbing, music21 side-effect audit
**Confidence:** HIGH (empirical audit of music21 global random state via project venv; direct source inspection of music_gen.py, config.py, timesig.py, duration_validator; current test-suite baseline verified: 309 passing)

## Summary

Phase 3 is a mechanical extraction, not a design problem — CONTEXT.md has already locked 25 decisions across 10 gray areas. The research job is to surface the **concrete technical facts** the planner needs to cut tasks:

1. **hatchling / typer / Python 3.9 version clash.** `requires-python = ">=3.9"` (D-13) is incompatible with the latest hatchling and typer. Hatchling 1.28.0 dropped Python 3.9, and typer 0.12+ has required Python 3.10+ since its 0.12.0 release. Phase 3 must either (a) use `hatchling<1.28` + `typer<0.12` [rejected — D-13 pins `typer>=0.12`], or (b) relax `requires-python` to `>=3.10`, or (c) accept that the floor is 3.10 in practice despite the spec. This conflict is material and surfaced as Risk #1 below. **Recommended mitigation:** relax `requires-python` to `>=3.10` during planning; the active Python is 3.12 and no stakeholder has claimed 3.9 compatibility.

2. **music21 global-RNG audit is a pass.** Empirically verified in the project's venv (`music21 v9.9.1`) that `roman.RomanNumeral`, `scale.MajorScale`, `scale.MinorScale`, and `pitch.Pitch` roundtrip (the full set touched by `generate_melody`, `generate_bassline`, `generate_chord_progression`) do NOT mutate `random.getstate()` across 5 keys × 5 roman symbols + pitch MIDI roundtrip + 20 repeated constructions. D-23 clean-result branch applies: add audit comment + regression test `tests/test_music21_isolation.py`, skip the `save_random_state()` wrapper.

3. **SongParams.sample RNG trace is straightforward but the retry loop is a trap.** `generate_song` today does key → tempo → time_sig → swing → WHILE(measures+signatures → validate_measures) — the retry loop CAN consume additional RNG draws on rejection. The sampler must preserve this semantics (not short-circuit to one draw) to avoid Phase 5 golden-test divergence.

**Primary recommendation:** Treat Phase 3 as six sequential tasks executed in this order — (T1) pyproject.toml + src/musicgen scaffold, (T2) duration_validator relocation + test import fix, (T3) sampler extraction (SongParams + all generate_random_* + generate_song_arrangement), (T4) generator extraction (chord → melody → bassline → beat), (T5) music21 isolation test + audit comment, (T6) test suite migration + conftest.py delete + requirements.txt delete + music_gen.py re-import shim. All tasks must keep `pytest tests/ -q` green and `python music_gen.py` runnable end-to-end.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Package metadata + build | Build backend (hatchling) | — | Declared in pyproject.toml; only build-system concern |
| Random parameter sampling (key/tempo/ts/swing/measures/arrangement) | `src/musicgen/sampler.py` (pure-fn library) | — | All logic is pure; no I/O except `song_structures.json` read |
| Per-part MIDI generation | `src/musicgen/generators/{chord,melody,bassline,beat}.py` (pure-fn library) | `src/musicgen/duration_validator.py` | Each generator consumes `DurationValidator` + time-sig registry; writes one `.mid` file per call |
| Duration validation | `src/musicgen/duration_validator.py` | `timesig` (via adapter) | Moved from root; delegates analysis to registry |
| CLI dispatch (phase 3 stub) | `src/musicgen/cli.py` (typer app) | `src/musicgen/__main__.py` | Entry-point contract only; real CLI lands Phase 6 |
| Time-signature registry lookup | `timesig.py` at repo root (Phase 2 — unchanged) | — | D-03: stays at root this phase |
| Path/config resolution | `config.py` at repo root (Phase 2 — unchanged) | — | D-03: stays at root this phase |
| Orchestration (`create_song`, `generate_song`, `mix_and_save`) | `music_gen.py` at repo root (shim) | — | D-04/05: stays in god file; Phase 4 extracts `mix_and_save` |
| Smoke test | `music_gen.py` `__main__` | — | Exit criterion: `python music_gen.py` still produces a song |

## Standard Stack

### Core (build & packaging)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| hatchling | `>=1.24,<1.28` (see Risk #1) | Build backend | [CITED: packaging.python.org writing-pyproject-toml] Modern, zero-config default for src-layout; minimal boilerplate vs setuptools |
| typer | `>=0.12,<0.13` (see Risk #1) | CLI framework | [VERIFIED: CONTEXT.md D-13, D-18] Locked by CONTEXT.md |
| pytest | `>=8.0` | Test runner | [VERIFIED: CONTEXT.md D-13] Already in use via dev-requirements.txt |
| pytest-cov | `>=5.0` | Coverage plugin | [VERIFIED: CONTEXT.md D-13] Locked; Phase 7 enforces ≥80% |
| pytest-xdist | `>=3.5` | Parallel test execution | [VERIFIED: CONTEXT.md D-13] Locked; benefits appear as suite grows |

### Runtime deps migrated verbatim from requirements.txt (D-13)

[VERIFIED: `/home/bidu/musicgen/requirements.txt` read 2026-04-18]

| Library | Current spec | Notes |
|---------|-------------|-------|
| numpy | `>=1.20.0` | Unchanged |
| scipy | `>=1.7.0` | Unchanged |
| midiutil | `>=1.2.1` | Unchanged |
| music21 | `>=7.3.3` | Currently installed: 9.9.1 (verified) |
| librosa | `>=0.9.2` | Unchanged |
| pydub | `>=0.25.1` | Unchanged |
| midi2audio | `>=0.1.1` | Unchanged |
| pedalboard | `>=1.0.0` | Unchanged |
| python-json-logger | `>=2.0.7` | Already installed (Phase 2 used it) |
| typing-extensions | `>=4.4.0` | Unchanged |
| numba | `>=0.56.4` | Unchanged |
| llvmlite | `>=0.39.1` | Unchanged |
| python-magic | `>=0.4.27` | Unchanged |
| **typer** (NEW) | `>=0.12` | [VERIFIED: CONTEXT.md D-13] Phase 3 new dependency |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| hatchling | setuptools | [ASSUMED] Setuptools supports Python 3.9 through current versions; would dodge Risk #1. But CONTEXT.md D-13 locks hatchling. |
| typer | argparse / click | argparse: stdlib (no Python-version clash) but verbose. click: similar to typer. CONTEXT.md D-13 locks typer. |

**Version verification performed 2026-04-18:**

```bash
# Hatchling 1.28.0 dropped Python 3.9 (2025-11-26)
# Last 3.9-compatible hatchling: 1.27.0 (2024-11-26)
# [CITED: https://hatch.pypa.io/dev/history/hatchling/]

# Typer 0.12.0+ requires Python 3.10+
# [CITED: https://pypi.org/project/typer/]
```

**Installation:**

```bash
pip install -e .            # runtime deps only
pip install -e '.[dev]'     # runtime + pytest + pytest-cov + pytest-xdist
```

## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01:** `src/` layout with package root `src/musicgen/`.
- **D-02:** Files created: `__init__.py` (empty), `__main__.py`, `cli.py`, `sampler.py`, `generators/__init__.py`, `generators/chord.py`, `generators/melody.py`, `generators/bassline.py`, `generators/beat.py`, `duration_validator.py`.
- **D-03:** `config.py` and `timesig.py` STAY at repo root this phase.
- **D-04:** `music_gen.py` stays as a re-import shim; `python music_gen.py` must still generate a song end-to-end.
- **D-05:** Keep `mix_and_save`, `create_song`, `generate_song`, `generate_song_parts`, `generate_pedalboard`, `apply_fx_to_layer`, FX/levels/soundfont helpers, and `__main__` guard in `music_gen.py`.
- **D-06:** Time-sig wrappers in `music_gen.py` remain; generators call `TimeSignatureRegistry` directly (no double indirection).
- **D-07:** Every extracted function takes `rng: random.Random` explicitly. Zero bare `random.*` inside extracted modules (sampler, generators, duration_validator).
- **D-08:** Phase 3 does NOT implement `derive_sample_seed` or `make_rngs`; `music_gen.py` shim constructs `_rng = random.Random()` and threads it through.
- **D-09:** `TimeSignatureRegistry.sample_random(rng=None)` already follows the contract — leave it alone.
- **D-10:** Move `enhanced_duration_validator.py` → `src/musicgen/duration_validator.py` (rename; no back-compat shim at root).
- **D-11:** `musicality_score.py` stays at repo root.
- **D-12:** `beat_anotator.py` untouched.
- **D-13:** `pyproject.toml` is single dep manifest: hatchling backend, `requires-python = ">=3.9"` (but see Risk #1), `version = "0.1.0"`, full runtime deps + `typer>=0.12`, dev extras, `musicgen = "musicgen.cli:app"` script, `[tool.hatch.build.targets.wheel] packages = ["src/musicgen"]`.
- **D-14:** Delete `requirements.txt` and `dev-requirements.txt`.
- **D-15:** Rewrite tests to import from `musicgen.*` where code moved.
- **D-16:** Delete `tests/conftest.py`.
- **D-17:** New test files: `tests/test_sampler.py`, `tests/test_generators/test_{chord,melody,bassline,beat}.py`, optional `tests/test_package_install.py`, plus D-24's `tests/test_music21_isolation.py`.
- **D-18/D-19:** Minimal typer CLI with one stub command; `__main__.py` is a one-line delegator.
- **D-20:** `SongParams` is `@dataclass(frozen=True)` with 9 fields (see Sampler Extraction section).
- **D-21:** `SongParams.sample(rng: random.Random, cfg: config.Config, *, time_signature_variation: float = 1.0) -> SongParams`. Deterministic order: key → tempo → time_sig_base → swing → arrangement → measures+per-part-signatures (with retry loop).
- **D-22:** Generators take individual fields (not `SongParams` object).
- **D-23:** Audit music21 global-RNG. If clean, add audit comment; if leaky, add `save_random_state()` wrapper.
- **D-24:** Add `tests/test_music21_isolation.py` regression test.
- **D-25:** Phase 3 runs serially before Phase 4.

### Claude's Discretion

- Classmethod name: `SongParams.sample(...)` vs free `sample_song_params(...)` — shape fixed, name aesthetic.
- Placement of `calculate_swing_offset` (inlined into `generators/beat.py` vs `_swing.py` helper — only one caller; inline is fine).
- Module docstring style — match `config.py`/`timesig.py`.
- Import ordering — match existing `music_gen.py` style.
- Use `@dataclass(frozen=True)` (NOT `slots=True` — requires 3.10+ and spec pins `>=3.9`; but see Risk #1).

### Deferred Ideas (OUT OF SCOPE)

- Moving `musicality_score.py` into the package (Phase 4 or 5).
- Moving `config.py` / `timesig.py` into the package (Phase 5).
- Replacing `beat_anotator.py` with `src/musicgen/beats.py` (Phase 4 R-X7).
- Real `derive_sample_seed` / `make_rngs` / per-domain RNG hierarchy (Phase 5 R-P7).
- Public library API (`from musicgen import generate, generate_batch`) — Phase 5/6.
- Real typer CLI with `--count`, `--out`, `--seed`, etc. — Phase 6 R-P13.
- README rewrite for `pip install -e .` — Phase 7 R-Q1.
- `pytest-cov` ≥ 80% gate — Phase 7 R-Q2.

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| R-X1 | Package skeleton + pyproject.toml (hatchling, typer>=0.12, dev extras, `musicgen = "musicgen.cli:app"`) | Technical Approach section (hatchling pyproject.toml template); CLI Scaffolding section; Risk #1 (version clash) |
| R-X2 | `sampler.py` with `SongParams` + all `generate_random_*` + `generate_song_arrangement` taking explicit `rng: random.Random` | Sampler Extraction section (full SongParams field list, `SongParams.sample()` signature + rng-call sequence) |
| R-X3 | `generators/{chord,melody,bassline,beat}.py` each taking SongParams + injected rng + pattern files, unit tested | Generator Extraction section (per-generator signature + bare-random count + music21 use) |
| R-Q4 (partial) | `version = "0.1.0"` in pyproject.toml | Technical Approach section (pyproject.toml template includes `version = "0.1.0"`) |

## Technical Approach

### pyproject.toml template (hatchling, src-layout, Phase 3 scope)

[CITED: https://packaging.python.org/en/latest/guides/writing-pyproject-toml/, https://hatch.pypa.io/1.13/config/build/]

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "musicgen"
version = "0.1.0"              # R-Q4
description = "Synthetic music dataset generator for ML training"
readme = "README.md"
license = { file = "LICENSE" }
requires-python = ">=3.9"       # See Risk #1 — probably needs to be >=3.10 for hatchling+typer compatibility
authors = [{ name = "Carlos Eduardo Batista" }]
dependencies = [
    # Core
    "numpy>=1.20.0",
    "scipy>=1.7.0",
    # MIDI and Audio Processing
    "midiutil>=1.2.1",
    "music21>=7.3.3",
    "librosa>=0.9.2",
    "pydub>=0.25.1",
    "midi2audio>=0.1.1",
    # Audio Effects Processing
    "pedalboard>=1.0.0",
    # Utility Libraries
    "python-json-logger>=2.0.7",
    "typing-extensions>=4.4.0",
    # Performance
    "numba>=0.56.4",
    "llvmlite>=0.39.1",
    # File handling
    "python-magic>=0.4.27",
    # CLI (NEW in Phase 3 per D-13)
    "typer>=0.12",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "pytest-xdist>=3.5",
]

[project.scripts]
musicgen = "musicgen.cli:app"

[tool.hatch.build.targets.wheel]
packages = ["src/musicgen"]

[tool.pytest.ini_options]
testpaths = ["tests"]
# Phase 3 uses the default import mode; --import-mode=importlib is a Phase 7
# consideration when we add parallel test isolation via pytest-xdist in CI.
```

### src-layout + editable install + pytest discovery (no shim needed)

[CITED: https://docs.pytest.org/en/stable/explanation/goodpractices.html — WebSearch summary 2026-04-18]

Standard pattern confirmed by pytest docs:

1. `pip install -e .` populates `.venv/lib/pythonX.Y/site-packages/musicgen.*-py3.X-editable.pth` (or similar egg-link pointing to `src/musicgen/`).
2. `import musicgen.sampler` resolves via the editable install — NO sys.path shim needed.
3. Tests live in `tests/` at repo root; pytest discovers them via `testpaths = ["tests"]`.
4. `config` and `timesig` still import directly (they remain at repo root per D-03) because the editable install *also* exposes the repo root to `sys.path` — confirmed by the existing Plan 01-04 behavior where `import music_gen` / `import config` / `import timesig` worked without shims once the package entry existed.

**Key subtlety:** editable installs place `src/musicgen/` on `sys.path` (hatchling-editables standard), BUT `config.py`/`timesig.py` at repo root are NOT automatically on `sys.path` after the shim deletion unless:

- They're in the same directory as files the editable points to (YES — repo root contains `pyproject.toml` and hatchling-editables makes `src/` importable, not repo root), OR
- A pytest `rootdir` mechanism puts them on path.

**Verification step that MUST appear as a task:** after deleting `tests/conftest.py` and installing `pip install -e .`, run `python -c "import config; import timesig; from musicgen.sampler import SongParams"` in a fresh venv. If `config`/`timesig` fail to import, add `[tool.hatch.build.targets.wheel.force-include]` mapping for those files, OR add them to `packages` list as secondary modules, OR restore a minimal `conftest.py` that does `sys.path.insert(0, <repo-root>)`. **Preferred fallback:** add a minimal `tests/conftest.py` that ONLY puts repo root on path; revisit in Phase 5 when `config`/`timesig` move into the package.

**Test config:** use `testpaths = ["tests"]` in `[tool.pytest.ini_options]`. No `pythonpath` directive needed if editable install + conftest fallback works. No `--import-mode=importlib` required this phase (deferred to Phase 7).

### Data files handling

[CITED: https://github.com/pypa/hatch/discussions/427]

**Decision:** `beat_roll_patterns_*.txt`, `*_fx.json`, `sf/` stay at repo root as dev-time assets — they are NOT packaged in the wheel this phase. Rationale:

- CONTEXT.md D-03 keeps `config.py` at repo root; config paths are computed relative to `os.path.dirname(os.path.abspath(config.__file__))` which resolves to repo root.
- Phase 3 wheel only needs to expose Python code under `src/musicgen/`.
- Phase 5 (when config moves into the package) will re-examine whether data files become package-relative assets.

**Explicit default:** hatchling does NOT include non-Python files by default; no extra `exclude` or `include` config required for Phase 3.

## Sampler Extraction

### SongParams full dataclass definition

**Location:** `src/musicgen/sampler.py`

```python
"""Sampler module — pure-function song-level parameter sampling.

All functions take an explicit `rng: random.Random` parameter per D-07.
No bare `random.*` anywhere in this module.
"""
from __future__ import annotations  # Phase 3 keeps Python 3.9 syntax compatible

import json
import logging
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import config
from timesig import TimeSignatureRegistry

logger = logging.getLogger(__name__)


@dataclass(frozen=True)   # D-20, NOT slots=True per Claude discretion (Python 3.9 compat)
class SongParams:
    """Frozen song-level parameters produced by the sampler.

    Fields are populated by :meth:`SongParams.sample` in this deterministic order
    (D-21): key, tempo, time_signature_base, swing_amount, song_unique_parts,
    song_arrangement, measures_per_part, signatures_per_part. The retry loop
    for measures/signatures can consume additional RNG draws on rejection —
    see the Risks section.
    """
    key: str                                  # e.g. "G", "Am"
    tempo: int                                # BPM
    time_signature_base: str                  # e.g. "4/4"
    time_signature_variation: float           # probability used to vary per-part; hardcoded 1.0 today
    swing_amount: float                       # clamped 0.5–0.75
    signatures_per_part: Dict[str, str]       # e.g. {'intro': '4/4', 'verse': '6/8', ...}
    measures_per_part: Dict[str, int]         # e.g. {'intro': 16, 'verse': 32, ...}
    song_unique_parts: List[str]              # set(arrangement) preserving order or as list
    song_arrangement: List[str]               # full sequence, e.g. ['intro','verse','chorus','verse','outro']

    @classmethod
    def sample(
        cls,
        rng: random.Random,
        cfg: Optional[config.Config] = None,
        *,
        time_signature_variation: float = 1.0,
    ) -> "SongParams":
        """Draw all song-level parameters using the given rng.

        Order (must match current generate_song RNG order to preserve Phase 5
        determinism baseline):
          1. key           — generate_random_key(rng)
          2. tempo         — generate_random_tempo(rng)
          3. time_sig_base — generate_random_time_signature(rng)
          4. swing         — generate_random_swing(rng)
          5. arrangement   — generate_song_arrangement(rng, structures_file=cfg.song_structures_file)
          6. LOOP UNTIL validate_measures:
                 measures, signatures = generate_song_measures(time_sig_base, time_sig_var, rng)
        """
        _cfg = cfg if cfg is not None else config.Config()
        key = generate_random_key(rng)
        tempo = generate_random_tempo(rng)
        time_sig_base = generate_random_time_signature(rng)
        swing = generate_random_swing(rng)
        unique_parts, arrangement = generate_song_arrangement(
            rng, structures_file=_cfg.song_structures_file
        )
        # Retry loop — identical semantics to music_gen.py:1020-1024
        while True:
            measures, signatures = generate_song_measures(
                time_sig_base, time_signature_variation, rng
            )
            if validate_measures_dict(measures, signatures):
                break
        return cls(
            key=key,
            tempo=tempo,
            time_signature_base=time_sig_base,
            time_signature_variation=time_signature_variation,
            swing_amount=swing,
            signatures_per_part=signatures,
            measures_per_part=measures,
            song_unique_parts=unique_parts,
            song_arrangement=arrangement,
        )
```

### SongParams.sample RNG call sequence (canonical order)

[VERIFIED: traced from `music_gen.py:1010-1024` + `generate_random_key:808-823` + `generate_random_tempo:825-835` + `generate_random_swing:564-589` + `generate_song_measures:849-880` + `generate_song_arrangement:508-547`]

Per `SongParams.sample(rng)` invocation the rng is consumed in this exact order:

| Step | Function | RNG calls | Notes |
|------|----------|-----------|-------|
| 1 | `generate_random_key(rng)` | `rng.random()` | one draw; maps to weighted key ranges |
| 2 | `generate_random_tempo(rng)` | `rng.random()`, then `rng.randint(min_tempo, max_tempo)` | exactly 2 draws always (both branches) |
| 3 | `generate_random_time_signature(rng)` | `rng.choices(sigs, weights=..., k=1)` | ONE draw via `TimeSignatureRegistry.sample_random(rng)` — existing D-09 contract |
| 4 | `generate_random_swing(rng)` | `rng.choices([0.5,0.66,0.75], weights=...)` + `rng.uniform(-0.02, 0.02)` | 2 draws always |
| 5 | `generate_song_arrangement(rng, ...)` | `rng.choice(structures)` | 1 draw; error fallback path does 0 draws |
| 6 (loop body) | `generate_song_measures(time_sig, var, rng)` | **minimum 6 draws, up to 11 per iteration.** See below. | Iterates until `validate_measures` passes. |

**Inside `generate_song_measures` per iteration** (ref `music_gen.py:854-880`):

```
rng.choice([8,16])                      # intro base_length
rng.choice([16,32])                     # verse base_length
rng.choice([16,32])                     # chorus base_length
rng.choice([8,16])                      # bridge base_length
rng.choice([8,16])                      # outro base_length
rng.random()                            # time_signature_variation gate
# IF gate passed (ts_var == 1.0 → always true):
rng.choice([ts, ts_alt])                # intro (time_signature_alternative also does rng.choice if alternatives)
rng.choice([ts, ts_alt])                # verse
rng.choice([ts, ts_alt])                # chorus
rng.choice([ts, ts_alt])                # bridge
rng.choice([ts, ts_alt])                # outro
```

**Critical:** `time_signature_alternative(ts)` ALSO does `rng.choice(spec.alternatives)` (music_gen.py:847) — so the "6 or 11 draws" count depends on whether the gate passes AND how many ts_alt calls are evaluated. With `time_signature_variation=1.0` (today's hardcoded value), all 5 per-part signature lines evaluate `rng.choice([ts, time_signature_alternative(ts, rng)])` — the `time_signature_alternative` eagerly evaluates first, adding 5 additional draws, then `rng.choice([ts, ts_alt])` does 5 more. So: 5 base_length draws + 1 gate draw + 5 ts_alt draws + 5 choice draws = **16 draws per iteration** when ts_var == 1.0.

**Retry trap:** `validate_measures` requires even measure counts for time signatures where `requires_even_measures=True` (2/4, 6/8, 12/8). If the gate path assigned a 6/8 to a part where `base_length * measure_multiplier` lands on odd, the whole iteration is rejected and RE-DRAWN — consuming another full 16 RNG draws. Phase 5's golden test must baseline AFTER Phase 3 extraction with this retry loop intact.

### Free-function sampler signatures (extracted, D-22 note: NOT SongParams consumers)

```python
def generate_random_key(rng: random.Random) -> str: ...
def generate_random_tempo(rng: random.Random) -> int: ...
def generate_random_time_signature(rng: random.Random) -> str:
    # Delegates to TimeSignatureRegistry.sample_random(rng) per D-09
    return TimeSignatureRegistry.sample_random(rng)
def generate_random_swing(rng: random.Random) -> float: ...
def time_signature_alternative(base_time_signature: str, rng: random.Random) -> str: ...
def generate_song_measures(
    time_signature: str,
    time_signature_variation: float,
    rng: random.Random,
) -> Tuple[Dict[str, int], Dict[str, str]]: ...
def generate_song_arrangement(
    rng: random.Random,
    structures_file: Optional[str] = None,
) -> Tuple[List[str], List[str]]: ...
def validate_measures_dict(
    measures: Dict[str, int],
    signatures: Dict[str, str],
) -> bool: ...   # NOTE: `validate_measures` lives in music_gen.py today (line 40) and does NOT use RNG — pure delegation to TimeSignatureRegistry. Migrated unchanged (no rng param needed).
```

**Free function names match the current music_gen.py symbols verbatim** (except last positional arg becomes `rng`). This is required so the shim `from musicgen.sampler import generate_random_key, ...` continues to satisfy any code that does `from music_gen import generate_random_key`.

### Sampler bare-random rewrite count

[VERIFIED: `Grep random\\. music_gen.py` 2026-04-18]

| Function (in music_gen.py) | Bare-random sites | Rewrite |
|---------------------------|-------------------|---------|
| `generate_random_key` (808) | 1 × `random.random()` | `rng.random()` |
| `generate_random_tempo` (825) | 1 × `random.random()` + 2 × `random.randint(...)` | `rng.random()` + `rng.randint(...)` |
| `generate_random_swing` (564) | 1 × `random.choices(...)` + 1 × `random.uniform(...)` | `rng.choices(...)` + `rng.uniform(...)` |
| `generate_song_measures` (849) | 5 × `random.choice([8/16/32])` + 1 × `random.random()` + 5 × `random.choice([ts, ts_alt])` | all → `rng.*` |
| `time_signature_alternative` (843) | 1 × `random.choice(spec.alternatives)` | `rng.choice(spec.alternatives)` |
| `generate_song_arrangement` (508) | 1 × `random.choice(structures)` | `rng.choice(structures)` |
| **Total (sampler)** | **~18 sites** | all rewritten |

## Generator Extraction

[VERIFIED: `music_gen.py:70-495` read 2026-04-18]

### `src/musicgen/generators/chord.py`

**New signature:**

```python
def generate_chord_progression(
    key: str,
    tempo: int,
    time_signature: str,
    measures: int,
    name: str,
    part: str,
    pattern_file: str,
    rng: random.Random,
) -> Tuple[List[str], str]:
```

**Bare-random sites to rewrite:** 1 (`music_gen.py:107` — `random.choice(chord_patterns[part])`).

**music21 usage:** `roman.RomanNumeral(chord_symbol, key)` + `chord.pitches` + `note.midi`. [VERIFIED: no global random mutation — see music21 audit section.]

**Internal changes beyond rng threading:**
- Import `from musicgen.duration_validator import DurationValidator` (was `from enhanced_duration_validator import DurationValidator`).
- Import `from timesig import TimeSignatureRegistry` and call `TimeSignatureRegistry.lookup(time_signature)` to avoid going through the `get_midi_time_signature_values` wrapper at `music_gen.py:49` (D-06: no double indirection).

### `src/musicgen/generators/melody.py`

**New signature:**

```python
def generate_melody(
    key: str,
    tempo: int,
    time_signature: str,
    measures: int,
    name: str,
    part: str,
    chord_progression: List[str],
    rng: random.Random,
) -> Tuple[List[int], str]:
```

**Bare-random sites to rewrite:** 3
- `music_gen.py:210` — `random.choice([note.midi for note in notes_to_use])` → `rng.choice(...)`
- `music_gen.py:214-217` — `random.choices(population=..., weights=...)` → `rng.choices(...)`
- `music_gen.py:222` — `random.choice(get_melody_durations(time_signature))` → `rng.choice(...)`
- `music_gen.py:242` — `random.randint(70, 100)` → `rng.randint(...)`

**music21 usage:** `scale.MajorScale(key)` / `scale.MinorScale(key[:-1])` + `roman.RomanNumeral(chord_symbol, key)` + `chord_obj.key = key` setter + `chord_obj.pitches` + `note.midi`. [VERIFIED: no global random mutation.]

### `src/musicgen/generators/bassline.py`

**New signature:**

```python
def generate_bassline(
    key: str,
    tempo: int,
    time_signature: str,
    measures: int,
    name: str,
    part: str,
    chord_progression: List[str],
    melody: List[int],
    rng: random.Random,
) -> str:
```

**Bare-random sites to rewrite:** 4
- `music_gen.py:309` — `random.choice([note.midi for note in notes_to_use])` → `rng.choice(...)`
- `music_gen.py:319-322` — `random.choices(population=..., weights=...)` → `rng.choices(...)`
- `music_gen.py:338` — `random.random() < 0.5` → `rng.random() < 0.5`
- `music_gen.py:347` — `random.randint(70, 100)` → `rng.randint(...)`

**music21 usage:** `scale.MajorScale` / `scale.MinorScale`, `roman.RomanNumeral`, `chord_obj.key =` setter, `pitch.Pitch()` + `.midi` setter + `.octave =` setter (line 349-351). [VERIFIED: no global random mutation.]

### `src/musicgen/generators/beat.py`

**New signature:**

```python
def generate_beat(
    part: str,
    tempo: int,
    time_signature: str,
    measures: int,
    name: str,
    swing_amount: float,
    rng: random.Random,
    cfg: Optional[config.Config] = None,
) -> Tuple[str, List[str]]:
```

**Bare-random sites to rewrite:** 2
- `music_gen.py:460` — `random.choice(beat_patterns[part])` → `rng.choice(...)`
- `music_gen.py:465` — `random.choice(beat_patterns.get(roll_part, [beat_pattern]))` → `rng.choice(...)`

**music21 usage:** NONE. Beat generator uses only MIDIFile + config-provided beat_roll_pattern_files.

**Helpers co-located (per Claude's discretion, see CONTEXT.md):**
- `calculate_swing_offset(base_duration: float, swing_amount: float) -> float` — inlined from `music_gen.py:377`. No RNG. No change needed beyond location.
- `beat_duration(signature: str, tempo: int) -> float` — inlined from `music_gen.py:369`. No RNG.

### Generator bare-random total

| Generator | Sites rewritten |
|-----------|----------------|
| chord | 1 |
| melody | 4 |
| bassline | 4 |
| beat | 2 |
| **Total** | **11 sites** |

Combined with sampler's ~18 sites = ~29 sites rewritten in Phase 3. The remaining ~10 sites from CONTEXT.md's "39 bare-random call sites" inventory live in `mix_and_save` (`music_gen.py:711-714, 596-597, 556`) and are out of scope (Phase 4).

## Duration Validator Relocation

[VERIFIED: `enhanced_duration_validator.py` read 2026-04-18; imports traced via Grep]

### Import graph BEFORE Phase 3

```
music_gen.py:15
    from enhanced_duration_validator import DurationValidator, NoteValue
enhanced_duration_validator.py:45 (inside _analyze_time_signature)
    from timesig import TimeSignatureRegistry            # local import — avoids circular
tests/test_duration_validator.py:10
    from enhanced_duration_validator import DurationValidator, NoteValue
```

### Import graph AFTER Phase 3

```
src/musicgen/duration_validator.py:45 (inside _analyze_time_signature — unchanged)
    from timesig import TimeSignatureRegistry            # timesig still at repo root per D-03
src/musicgen/generators/chord.py
    from musicgen.duration_validator import DurationValidator
src/musicgen/generators/melody.py
    from musicgen.duration_validator import DurationValidator
src/musicgen/generators/bassline.py
    from musicgen.duration_validator import DurationValidator
src/musicgen/generators/beat.py
    from musicgen.duration_validator import DurationValidator
music_gen.py  (REWRITTEN shim)
    from musicgen.duration_validator import DurationValidator, NoteValue   # re-import to preserve `from music_gen import DurationValidator` usage IF any
tests/test_duration_validator.py  (REWRITTEN)
    from musicgen.duration_validator import DurationValidator, NoteValue
```

### Migration steps (no back-compat shim — D-10)

1. `git mv enhanced_duration_validator.py src/musicgen/duration_validator.py`.
2. Rename is a pure file move — zero code changes inside the file (the `from timesig import TimeSignatureRegistry` local import at line 45 continues to work: `timesig` stays at repo root per D-03, and the repo root is importable alongside the editable `src/musicgen` install).
3. Update `tests/test_duration_validator.py:10` from `from enhanced_duration_validator import ...` to `from musicgen.duration_validator import ...`.
4. Update `music_gen.py:15` from `from enhanced_duration_validator import ...` to `from musicgen.duration_validator import ...`.
5. Confirm no other grep hits for `enhanced_duration_validator` exist in the project (spec check — none expected).

**Verification command per task:** `grep -r "enhanced_duration_validator" --include="*.py" /home/bidu/musicgen/` must return zero hits after migration. Optional: `find /home/bidu/musicgen -name 'enhanced_duration_validator*' -not -path '*/\.venv/*'` must return only the stale `__pycache__` entries (safely deletable).

## CLI Scaffolding

### `src/musicgen/cli.py` (D-18 minimal stub)

```python
"""CLI entry point (stub — real CLI lands in Phase 6 per D-18).

Provides the `musicgen` console script so `pip install -e . && musicgen --help`
works. One stub command (`info`) demonstrates the plumbing; real batch/generate
commands are out of scope for Phase 3.
"""
from __future__ import annotations

import logging
from typing import Optional

import typer

app = typer.Typer(
    help="musicgen — synthetic music dataset generator",
    no_args_is_help=True,
)


@app.command()
def info() -> None:
    """Print package metadata and a friendly pointer at Phase 6's full CLI."""
    typer.echo("musicgen 0.1.0 — Phase 3 package skeleton")
    typer.echo(
        "Real CLI (generate / batch / clean / calibrate) arrives in Phase 6. "
        "Today: `python music_gen.py` runs one smoke-test song."
    )


if __name__ == "__main__":  # direct-invocation fallback
    app()
```

### `src/musicgen/__main__.py` (D-19)

```python
"""python -m musicgen entry — routes to the typer app."""
from musicgen.cli import app

app()
```

### `src/musicgen/__init__.py` (D-02)

```python
"""musicgen package — Phase 3 skeleton. Public API lands in Phase 5."""
# Intentionally empty — Phase 5 will add `from musicgen.sampler import SongParams`
# and the `generate` / `generate_batch` library entry points.
```

### `src/musicgen/generators/__init__.py` (D-02)

```python
"""Per-layer MIDI generators (chord, melody, bassline, beat).

Each generator takes explicit fields + injected rng per D-22.
"""
# Empty marker — Phase 5 may add a convenience barrel re-export.
```

## music21 Global RNG Audit

### Empirical result (D-23 / D-24)

[VERIFIED: live audit in project venv 2026-04-18]

Command run:

```bash
/home/bidu/musicgen/.venv/bin/python -c "
import random
s0 = random.getstate()
from music21 import roman, scale, pitch
for k in ['C', 'G', 'D', 'Am', 'Em']:
    if k.endswith('m'):
        sc = scale.MinorScale(k[:-1])
    else:
        sc = scale.MajorScale(k)
    for sym in ['I', 'IV', 'V', 'vi', 'ii']:
        rn = roman.RomanNumeral(sym, k)
        _ = list(rn.pitches)
        for p in rn.pitches:
            _ = p.midi
    p = pitch.Pitch()
    p.midi = 36
    p.octave = 2
    _ = p.midi
s1 = random.getstate()
print('STATE CHANGED' if s0 != s1 else 'STATE UNCHANGED')
"
```

Output: `STATE UNCHANGED across 5 keys x 5 romans x pitch roundtrip`. Also verified across 20 repeated RomanNumeral constructions: state remained unchanged.

**music21 version tested:** 9.9.1. CONTEXT.md requires `music21>=7.3.3` — 9.9.1 satisfies it. [ASSUMED: behavior on music21 7.x and 8.x matches 9.9.1 — no regression test across historical versions was performed in this phase. Mitigation: `test_music21_isolation.py` regression test detects any future drift.]

### Outcome (D-23 clean-result path)

Add a short comment in `src/musicgen/generators/melody.py` and `bassline.py`:

```python
# music21 global-random audit (Phase 3, D-23): music21 9.9.1's roman.RomanNumeral,
# scale.MajorScale, scale.MinorScale, and pitch.Pitch do NOT mutate random.getstate().
# Verified empirically 2026-04-18. If this changes in a future music21 release,
# tests/test_music21_isolation.py will fail — wrap calls in save_random_state() then.
```

**No `save_random_state()` contextmanager needed this phase.**

### Regression test `tests/test_music21_isolation.py` (D-24)

```python
"""R-P7 / D-24 regression guard: music21 must not mutate global random.

If this test starts failing, add a save_random_state() wrapper in sampler.py
and every generator that touches music21 (melody, bassline, chord), and wrap
each music21 call in the new contextmanager. See CONTEXT.md D-23 for rationale.
"""
import random

import pytest


class TestMusic21DoesNotMutateGlobalRandom:
    def test_roman_numeral_preserves_global_state(self):
        from music21 import roman
        state0 = random.getstate()
        for key in ["C", "G", "D", "Am", "Em"]:
            for sym in ["I", "IV", "V", "vi", "ii"]:
                rn = roman.RomanNumeral(sym, key)
                _ = list(rn.pitches)
                for p in rn.pitches:
                    _ = p.midi
        assert random.getstate() == state0

    def test_scale_preserves_global_state(self):
        from music21 import scale
        state0 = random.getstate()
        _ = scale.MajorScale("C")
        _ = scale.MinorScale("A")
        _ = scale.MajorScale("G")
        _ = scale.MinorScale("E")
        assert random.getstate() == state0

    def test_pitch_midi_roundtrip_preserves_global_state(self):
        from music21 import pitch
        state0 = random.getstate()
        for midi_val in [36, 48, 60, 72]:
            p = pitch.Pitch()
            p.midi = midi_val
            p.octave = 2
            _ = p.midi
        assert random.getstate() == state0
```

### `save_random_state` contextmanager skeleton (fallback, only if regression test ever fails)

```python
import contextlib
import random
from typing import Iterator

@contextlib.contextmanager
def save_random_state() -> Iterator[None]:
    """Save/restore global random.getstate() across a block.

    Use this ONLY if a dependency (music21 release after 9.9.1, pedalboard,
    etc.) starts mutating global random state. Phase 3 audit (D-23) confirmed
    no wrapping is needed at music21 9.9.1 — this helper is reserved for future
    regressions flagged by tests/test_music21_isolation.py.
    """
    state = random.getstate()
    try:
        yield
    finally:
        random.setstate(state)
```

## Test Migration Map

### Existing test files (5 files, 309 passing tests total — verified 2026-04-18)

| Path | Current imports | Required post-Phase-3 imports | Change scope |
|------|----------------|-------------------------------|--------------|
| `tests/conftest.py` | `sys.path.insert(0, REPO_ROOT)` shim | **DELETED** (D-16). If `config`/`timesig` fail to import in a fresh venv after editable install, restore a minimal sys.path shim for repo root — but prefer fixing via pyproject.toml includes. | DELETE file |
| `tests/__init__.py` | empty | unchanged | none |
| `tests/test_time_signature.py` | `import music_gen` | `import music_gen` — unchanged (shim still exposes `verify_pattern_for_time_signature`, `verify_beat_pattern`, `validate_measures` as re-exports per D-04/D-06). Alternatively can import `from timesig import TimeSignatureRegistry` directly. | **none required**; optional tightening |
| `tests/test_duration_validator.py` | `from enhanced_duration_validator import DurationValidator, NoteValue` | `from musicgen.duration_validator import DurationValidator, NoteValue` | **1 import line** |
| `tests/test_timesig_registry.py` | `from timesig import TimeSignatureRegistry, TimeSignatureSpec` | unchanged — `timesig` stays at repo root per D-03 | **none** |
| `tests/test_config.py` | `import config` | unchanged — `config` stays at repo root per D-03 | **none** |
| `tests/test_music_gen_logging.py` | `ast.parse(open('../music_gen.py').read())` | unchanged — the AST scan reads `music_gen.py` at repo root by relative path | **none** |

**Post-migration baseline:** all 309 existing tests must still pass. If `test_duration_validator.py` is the only edit needed (one line), baseline is preserved trivially.

### New test files (D-17, D-24)

| New file | Purpose | Coverage |
|----------|---------|----------|
| `tests/test_sampler.py` | Seeded-RNG unit tests: `SongParams.sample(rng, cfg)`, `generate_random_key`, `generate_random_tempo`, `generate_random_time_signature`, `generate_random_swing`, `generate_song_measures`, `time_signature_alternative`, `generate_song_arrangement`. Same seed → same output (pinned values). | R-X2 |
| `tests/test_generators/__init__.py` | Package marker | — |
| `tests/test_generators/test_chord.py` | Seeded-RNG test: `generate_chord_progression(..., rng)` — produces same `.mid` bytes across two runs with the same seed and inputs. | R-X3 |
| `tests/test_generators/test_melody.py` | Seeded-RNG test: `generate_melody(..., rng)` — produces same `.mid` bytes across two seeded runs. | R-X3 |
| `tests/test_generators/test_bassline.py` | Seeded-RNG test: `generate_bassline(..., rng)` — produces same `.mid` bytes across two seeded runs. | R-X3 |
| `tests/test_generators/test_beat.py` | Seeded-RNG test: `generate_beat(..., rng)` across 2/4, 4/4, 6/8 × swing ∈ {0.5, 0.66, 0.75} — `.mid` reproducibility + annotations list structure. | R-X3 |
| `tests/test_music21_isolation.py` | D-24 regression guard — 3 test methods, see music21 section above | R-P7 (deferred; landed here as a Phase 3 prerequisite per D-23) |
| `tests/test_package_install.py` *(optional)* | Smoke test: create a temp venv, run `pip install -e .`, assert `import musicgen.sampler` + `musicgen --help` succeeds. Gated behind `@pytest.mark.slow` — optional per D-17. | R-X1 |

### Byte-equal `.mid` strategy for generator tests

MIDIUtil's output is deterministic given the same input sequence. Two runs of `generate_melody(..., rng=random.Random(42))` with identical arguments produce byte-equal `.mid` files. The test pattern:

```python
def test_generate_melody_is_deterministic_under_seed(tmp_path):
    args = dict(key="C", tempo=120, time_signature="4/4", measures=4,
                name=str(tmp_path / "song-verse"), part="verse",
                chord_progression=["I", "IV", "V", "I"])
    melody1, path1 = generate_melody(**args, rng=random.Random(42))
    os.remove(path1)
    melody2, path2 = generate_melody(**args, rng=random.Random(42))
    assert melody1 == melody2
    with open(path1, "rb") as f: assert Path(path1).read_bytes() == Path(path2).read_bytes()
```

## Risks and Mitigations

### Risk #1 — hatchling + typer + Python 3.9 requires-python conflict (HIGH)

**Problem:** CONTEXT.md D-13 locks `requires-python = ">=3.9"` + `typer>=0.12`. Verified today:
- `typer>=0.12` has required Python 3.10+ since 0.12.0 release [CITED: https://pypi.org/project/typer/].
- `hatchling>=1.28` drops Python 3.9 [CITED: https://hatch.pypa.io/dev/history/hatchling/]. Last 3.9-compatible hatchling is 1.27.0.

**Why it matters:** If `requires-python = ">=3.9"` stays, `pip install -e .` on Python 3.9 will fail to resolve `typer>=0.12` (wheel/sdist rejects 3.9). If we pin `hatchling==1.27.0` we get a 2024-era build backend; if we unpin hatchling it installs 1.29 which refuses to build on 3.9. Either way, the `>=3.9` claim is not honest.

**Mitigations (ranked):**
1. **PREFERRED:** Relax `requires-python` to `>=3.10` in pyproject.toml and document in the plan. Current project venv is Python 3.12; no stakeholder in CONTEXT.md / ROADMAP claims 3.9 support. Risk cost: near zero — nobody has been running on 3.9 anyway.
2. **Alternative:** Keep `>=3.9`, pin `typer>=0.12,<0.13` (which may still reject 3.9 per pypi metadata) AND `hatchling<1.28`. Result: `pip install -e .` fails on 3.9 at dep-resolution time despite the claim. Delivers a broken promise.
3. **Escalate to user:** Surface the conflict in the plan-level DISCUSSION-LOG and ask whether Python 3.9 support is an actual requirement. The CONTEXT.md auto-mode may have locked `>=3.9` as a guess.

**Recommendation for planner:** include a task "Verify Python floor by running `pip install -e .` in a throwaway venv; if typer/hatchling reject 3.9 (expected), bump `requires-python` to `>=3.10` and note the change in STATE.md." This converts the guess into a verified decision.

### Risk #2 — RNG order divergence breaks Phase 5 golden test (MEDIUM)

**Problem:** Phase 5 will add a determinism regression test — same seed → same MIDI bytes. If Phase 3's extraction subtly changes the order of `rng.*` calls (e.g., because `time_signature_alternative` is called at a different point in the sampler vs. in the shim, or because the retry loop in `generate_song_measures` is collapsed to one iteration), Phase 5's first run baselines against the new order but Phase 5's regression test detects any subsequent drift as a bug. The goal is: Phase 3 order == today's order, so Phase 5 starts from a meaningful baseline.

**Mitigation:**
- Sampler's `SongParams.sample` calls the same 6 functions in the same order as today's `generate_song` (D-21). Audit checklist:
  - [ ] Step 1: `generate_random_key(rng)` — matches music_gen.py:1013.
  - [ ] Step 2: `generate_random_tempo(rng)` — matches 1014.
  - [ ] Step 3: `generate_random_time_signature(rng)` — matches 1015.
  - [ ] Step 4: `generate_random_swing(rng)` — matches 1017 (note: 1018 does `min/max` clamp — no RNG).
  - [ ] Step 5: `generate_song_arrangement(rng, cfg.song_structures_file)` — matches `create_song` at 911 (order MOVES UP in the sampler — originally happens AFTER `generate_song_parts` is called, but Phase 1 R-S3 fix already moved it up into `create_song`; sampler version keeps it in step 5). Verify no rng consumer runs between step 4 and step 5.
  - [ ] Step 6: while-loop `generate_song_measures(ts, var, rng) + validate_measures` — matches 1021-1024. Preserve full 16-draw-per-iteration + retry semantics; DO NOT collapse to a single draw.
- Add a single smoke test in `tests/test_sampler.py` that pins exact output for `SongParams.sample(random.Random(42), config.Config())` — captures the post-refactor baseline as a golden tuple. Phase 5 is then free to compare its rendered WAV sha256 against this baseline.

### Risk #3 — `config`/`timesig` import fails after `conftest.py` deletion (MEDIUM)

**Problem:** `tests/conftest.py` today inserts repo root on `sys.path`. After deletion (D-16), Python resolves `import config` via whatever `sys.path` looks like after `pip install -e .`. Hatchling's editable install puts `src/musicgen/` on path (via a `.pth` file), NOT the repo root. If `config.py` / `timesig.py` cannot be found, every test that directly imports them (e.g., `tests/test_config.py`, `tests/test_timesig_registry.py`, and indirectly sampler/generator tests) fails.

**Empirical uncertainty:** The current venv has `.venv/bin/python -m pytest tests/` succeeding (309 passed) — but that's WITH the conftest.py shim. Post-shim behavior requires testing.

**Mitigations:**
1. Verify empirically after the conftest delete. The planner MUST include a task "After `pip install -e .` and `rm tests/conftest.py`, run `pytest tests/ -q` — expect 309+ passed." If it fails:
2. Fallback A: re-add a minimal `tests/conftest.py` that ONLY does `sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))`. Document that Phase 5 removes this when `config`/`timesig` move into the package.
3. Fallback B: add `[tool.hatch.build.targets.wheel.force-include]` or list `config` and `timesig` as standalone single-file packages in pyproject.toml. Warning: pollutes the wheel namespace — prefer fallback A.
4. Fallback C: add `[tool.pytest.ini_options] pythonpath = ["."]` — a pytest-level sys.path override that doesn't require conftest.py. **This is the cleanest option** and is recommended in the plan.

**Recommendation for planner:** go with fallback C proactively (add `pythonpath = ["."]` to pyproject.toml). Zero cost, eliminates the empirical risk.

### Risk #4 — music_gen.py shim imports break if Phase 3 rename anything inadvertently (LOW)

**Problem:** `music_gen.py` currently imports `from enhanced_duration_validator import DurationValidator, NoteValue` (line 15) and `import musicality_score` (line 14). After D-10 rename the first must become `from musicgen.duration_validator import ...`. A forgotten site leaves `music_gen.py` broken and the Phase 3 smoke-test exit criterion fails.

**Mitigation:** single-shot search across `music_gen.py`:
```bash
grep -n "enhanced_duration_validator\|from enhanced_" /home/bidu/musicgen/music_gen.py
```
Expected hits: 1 (line 15). After edit expected hits: 0.

### Risk #5 — `python music_gen.py` smoke-test drifts because shim misses a rebuilt symbol (LOW)

**Problem:** CONTEXT.md's exit criterion is "Old `music_gen.py` still executable for smoke testing." The shim must re-export every previously-top-level symbol that `__main__` or any currently-importing code references. Forgetting a re-export (e.g., `verify_pattern_for_time_signature`) doesn't fail the import but causes `test_music_gen_logging.py`'s AST scan to reach an undefined reference, or `generate_chord_progression` at `music_gen.py:984` to NameError.

**Mitigation:** After extraction, `music_gen.py` must be a grep-verifiable re-export surface:

```python
# Top of refactored music_gen.py (representative — actual organization follows
# the CONVENTIONS.md import grouping).
from musicgen.sampler import (
    SongParams,
    generate_random_key, generate_random_tempo, generate_random_time_signature,
    generate_random_swing, generate_song_measures, time_signature_alternative,
    generate_song_arrangement, validate_measures_dict,
)
from musicgen.generators.chord import generate_chord_progression
from musicgen.generators.melody import generate_melody
from musicgen.generators.bassline import generate_bassline
from musicgen.generators.beat import generate_beat, beat_duration, calculate_swing_offset
from musicgen.duration_validator import DurationValidator, NoteValue

_rng = random.Random()   # D-08 — one unseeded RNG threaded to every call site
```

Each currently-top-level `def` that moves MUST have a corresponding `from musicgen.X import Y` line. Mechanical check: compare `git show HEAD:music_gen.py | grep "^def "` with the post-Phase-3 `grep "^def\|^from musicgen" music_gen.py` — the set of names defined OR re-imported must be a superset of the pre-refactor `^def` set.

**Planner task:** add a verification step after the shim rewrite: `python music_gen.py` must complete one `generate_song(0, cfg)` call without `NameError`/`AttributeError`, then `ls <song_dir>` must show a `.wav` per part. This is the exit criterion AND the regression test.

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest 8.x + pytest-cov 5.x + pytest-xdist 3.5+ (declared in `[project.optional-dependencies].dev`) |
| Config file | `pyproject.toml` (under `[tool.pytest.ini_options]`); `tests/conftest.py` is DELETED per D-16 |
| Quick run command | `.venv/bin/python -m pytest tests/ -q` (expected <5s for existing 309 tests; <10s with new test files) |
| Full suite command | `.venv/bin/python -m pytest tests/ -v --tb=short` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| R-X1 | `pip install -e .` succeeds | integration-smoke | `python -c "import musicgen; import musicgen.sampler; import musicgen.generators.chord; import musicgen.generators.melody; import musicgen.generators.bassline; import musicgen.generators.beat; import musicgen.duration_validator"` after editable install | ❌ Wave 0 (add `tests/test_package_install.py`) |
| R-X1 | `musicgen --help` works | smoke | `musicgen --help` (exit code 0, help text contains "musicgen") | ❌ Wave 0 (add to `tests/test_package_install.py`) |
| R-X1 | `python -m musicgen --help` works | smoke | `python -m musicgen --help` | ❌ Wave 0 |
| R-X2 | `from musicgen.sampler import SongParams` works | import-check | `python -c "from musicgen.sampler import SongParams; assert hasattr(SongParams, 'sample')"` | ❌ Wave 0 |
| R-X2 | `SongParams.sample(rng, cfg)` is deterministic | unit (seeded) | `pytest tests/test_sampler.py::TestSongParamsSample::test_sample_is_deterministic -x` | ❌ Wave 0 |
| R-X2 | `generate_random_key` / `_tempo` / `_swing` / `_time_signature` / `generate_song_measures` / `time_signature_alternative` / `generate_song_arrangement` are rng-aware | unit (seeded) | `pytest tests/test_sampler.py -x` | ❌ Wave 0 |
| R-X2 | Zero bare `random.*` in `src/musicgen/sampler.py` | static-AST | `pytest tests/test_sampler.py::test_no_bare_random_in_sampler -x` OR (Phase 5 upgrade) one `test_no_bare_random_in_package` AST scan covering all of `src/musicgen/` | ❌ Wave 0 (NOT explicitly required by CONTEXT.md but strongly recommended — validates D-07 contract) |
| R-X3 | Each generator takes `rng` and produces deterministic MIDI | unit (seeded) | `pytest tests/test_generators/ -x` | ❌ Wave 0 |
| R-X3 | Zero bare `random.*` in `src/musicgen/generators/*.py` | static-AST | same test as R-X2 — one scan over all of `src/musicgen/` | ❌ Wave 0 |
| D-10 | `DurationValidator` importable from new location; tests still pass | unit | `pytest tests/test_duration_validator.py -x` | ✅ (exists; 1 import line must be updated) |
| D-23 | music21 does not mutate global random | unit | `pytest tests/test_music21_isolation.py -x` | ❌ Wave 0 |
| D-24 | Test file exists as regression guard | existence | `pytest tests/test_music21_isolation.py --collect-only` shows 3 tests | ❌ Wave 0 |
| D-04 / Exit criterion | `python music_gen.py` runs end-to-end | integration-manual-or-slow | `python music_gen.py` in CI with ffmpeg installed OR `@pytest.mark.slow` test that runs one `generate_song(0, cfg)` iteration. | ❌ Wave 0 (can defer to Phase 4's R-X8 integration test, but we need SOMETHING — recommend a manual step in the plan) |
| Regression (Phase 2 baseline) | All 309 pre-existing tests still pass | regression | `pytest tests/ -q` (expected ≥309 passed) | ✅ (exists) |

### Sampling Rate

- **Per task commit:** `.venv/bin/python -m pytest tests/ -q` — must report ≥309 passed (regression baseline) + any new tests from the current task.
- **Per wave merge:** Full suite + smoke `python music_gen.py` (produces a song directory + wav files). Manual ear check is NOT required this phase (musicality_score stays decorative — CONCERNS.md #10).
- **Phase gate:** Before `/gsd-verify-work`:
  - `pytest tests/ -q` green.
  - `pip install -e '.[dev]'` succeeds in a fresh venv.
  - `musicgen --help` exits 0.
  - `python music_gen.py` completes one iteration without error (timing ~1–3 min including FluidSynth rendering).
  - `git grep -n 'random\\.' src/musicgen/ tests/test_sampler.py tests/test_generators/` returns zero hits outside of `rng` object (AST scan ideal; grep acceptable).

### Wave 0 Gaps

- [ ] `pyproject.toml` — **does not exist**; must be created at repo root as part of R-X1.
- [ ] `src/musicgen/__init__.py` — must be created (empty).
- [ ] `src/musicgen/__main__.py` — must be created (one-liner delegator).
- [ ] `src/musicgen/cli.py` — must be created (stub typer app + one `info` command).
- [ ] `src/musicgen/sampler.py` — must be created (empty scaffold in Wave 0; filled in Wave 1).
- [ ] `src/musicgen/generators/__init__.py`, `chord.py`, `melody.py`, `bassline.py`, `beat.py` — must be created (empty scaffolds in Wave 0).
- [ ] `src/musicgen/duration_validator.py` — must exist (created by `git mv`).
- [ ] `tests/test_sampler.py` — must be created.
- [ ] `tests/test_generators/__init__.py`, `test_chord.py`, `test_melody.py`, `test_bassline.py`, `test_beat.py` — must be created.
- [ ] `tests/test_music21_isolation.py` — must be created (D-24).
- [ ] `tests/test_package_install.py` — optional per D-17 (recommended as a `@pytest.mark.slow` test).
- [ ] Verify editable install end-to-end: `pip install -e .` in a fresh venv, then `python -c "import musicgen.sampler"` and `musicgen --help` must succeed.

**Framework install:** no new test framework is introduced; pytest stack already in use via `dev-requirements.txt` (to be deleted per D-14 and migrated to `[project.optional-dependencies].dev`).

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | music21 behavior on versions 7.3.3–9.8.x matches the 9.9.1 audit result (no global random mutation) | music21 audit | LOW — `tests/test_music21_isolation.py` runs in CI and catches drift in any version; the regression guard makes the assumption self-correcting |
| A2 | hatchling editable install places repo root on sys.path implicitly, enabling `config`/`timesig` imports | Technical Approach / Risk #3 | MEDIUM — mitigated by proactively adding `pythonpath = ["."]` to pyproject.toml (fallback C) which is recommended regardless |
| A3 | MIDIUtil output is bit-deterministic given the same MIDI operations in the same order | Test Migration Map (byte-equal .mid strategy) | LOW — widely documented for MIDIUtil; if false, generator tests can compare on `melody` list or timing arrays instead of raw bytes |
| A4 | The CONTEXT.md "39 bare-random call sites" figure is approximately correct; Phase 3 scope hits ~29 of them | Generator Extraction summary | LOW — the grep-verified count is 11 in generators + 18 in sampler functions = 29, matching |
| A5 | Phase 3 DOES NOT need to add `test_package_install.py` — D-17 marks it optional | Validation Architecture / Wave 0 | LOW — "optional" is explicit in CONTEXT.md; planner may include as slow-marked CI nice-to-have |
| A6 | No stakeholder is actively running this project on Python 3.9 | Risk #1 | LOW — no CI config for 3.9, no Dockerfile pinned to 3.9, project venv is 3.12. Mitigation: surface in DISCUSSION-LOG for explicit user confirmation before finalizing `requires-python` |

## Open Questions (RESOLVED)

> All three open questions were answered during planning; each `RESOLVED:` marker below cites the implementing plan/task. Preserved for audit trail per `dimension 11 (Research Resolution)`.

1. **Should `requires-python` be `>=3.9` or `>=3.10`?**
   - What we know: CONTEXT.md D-13 pins `>=3.9`; typer>=0.12 and hatchling>=1.28 both require 3.10+.
   - What's unclear: whether D-13's `>=3.9` was a considered decision or an auto-mode default.
   - Recommendation: include a plan task that tests `pip install -e .` on a fresh 3.9 venv. If it fails (predicted yes), bump to `>=3.10` and log the change. If a stakeholder later objects, it's reversible.
   - **RESOLVED:** `>=3.10` per Plan 03-01 Task 1 (Risk #1). typer>=0.12 and hatchling>=1.28 require 3.10+, and there is no stakeholder on Python 3.9 (A6). D-13's `>=3.9` was overridden with explicit justification logged in 03-01-SUMMARY.md.

2. **Does `tests/conftest.py` deletion actually work without `pythonpath` fallback?**
   - What we know: today's conftest.py's shim is explicitly flagged for deletion; editable install should expose `src/musicgen/`.
   - What's unclear: whether root-level `config.py` / `timesig.py` resolve without extra configuration.
   - Recommendation: add `pythonpath = ["."]` to `[tool.pytest.ini_options]` as a belt-and-suspenders measure. Zero cost, eliminates the empirical risk — preferred over adding a `conftest.py` fallback.
   - **RESOLVED:** Plan 03-01 adds `pythonpath = ["."]` to `[tool.pytest.ini_options]` in `pyproject.toml`; Plan 03-05 deletes `tests/conftest.py` only **after** that precondition is verified. The two-step sequencing eliminates the empirical risk entirely (belt-and-suspenders).

3. **Should `SongParams.sample` accept `cfg=None` with a runtime `config.Config()` fallback?**
   - What we know: Plan 02-01 established the "cfg=None with runtime fallback" pattern across the project (Phase 2 D-02 for generators).
   - What's unclear: whether Phase 3 should continue the pattern in the sampler since Phase 5 will thread the real Config anyway.
   - Recommendation: follow the existing pattern (`cfg: Optional[config.Config] = None` with `_cfg = cfg if cfg is not None else config.Config()` fallback inside the method). Consistent with `generate_beat` today.
   - **RESOLVED:** Plan 03-03 Task 1 applies the Phase 2 `cfg=None` + runtime fallback pattern in `SongParams.sample` (see `<interfaces>` skeleton: `_cfg = cfg if cfg is not None else config.Config()`). Consistent with `generate_beat` today; Phase 5 will thread the real Config without breaking the signature.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python | Runtime | ✓ | 3.12.3 | — |
| `/home/bidu/musicgen/.venv/` (local venv) | Phase 3 test run | ✓ | preserved from Plan 01-04 | — |
| music21 | Generator audit + melody/bassline/chord extraction | ✓ | 9.9.1 (installed in .venv) | — |
| midiutil | All generators | ✓ | installed in .venv | — |
| pytest | Test suite | ✓ | installed in .venv | — |
| pytest-cov | Coverage (Phase 7; Phase 3 skeleton only) | ✓ | installed in .venv | — |
| pytest-xdist | Parallel tests (future) | [?] | check with `pip show pytest-xdist` | `pytest -n 0` (serial) — zero code impact for Phase 3 |
| typer | NEW runtime dep (D-13) | ✗ | not installed; `pip show typer` returned `not found` 2026-04-18 | `pip install -e '.[dev]'` installs it once pyproject.toml lands |
| hatchling | Build backend (D-13) | ✗ | not installed; `pip show hatchling` returned `not found` 2026-04-18 | `pip install hatchling` is implicit under PEP 517 `pip install -e .` — no manual step needed |
| ffmpeg / avconv | pydub runtime (`mix_and_save`) | ✗ | `pydub/utils.py:170: RuntimeWarning: Couldn't find ffmpeg or avconv` seen during test run | smoke `python music_gen.py` may fail at the pydub export step; install `ffmpeg` system-wide OR skip the smoke test as part of Phase 3 (verify sampler+generators without full mix) |
| FluidSynth binary | `FluidSynth(...).midi_to_audio` inside `mix_and_save` | [?] — assumed missing | — | Phase 3 does NOT exercise mix_and_save in automated tests; smoke test is manual. Phase 5/6 handle FluidSynth provisioning properly |

**Missing dependencies with no fallback:** none blocking Phase 3 tasks (typer + hatchling auto-install via pyproject.toml).

**Missing dependencies with fallback:** ffmpeg + FluidSynth — the smoke test `python music_gen.py` may fail on the audio-rendering step. Mitigation: verify sampler + generators via automated tests only; treat `python music_gen.py` as a "best-effort smoke" that passes if the machine has the audio stack and is skipped otherwise. Phase 3's strict exit criterion is "import resolves and generators produce MIDI"; end-to-end audio was always a nice-to-have.

## Security Domain

Skipped — Phase 3 is internal code reorganization. No user input, no network, no auth, no crypto, no secrets, no data validation boundary, no privilege escalation paths. `security_enforcement` status in `.planning/config.json` is unset (default: enabled); but there is no applicable ASVS category for "rename enhanced_duration_validator.py and thread an rng parameter." If a future phase adds a generation REST API or user-supplied config, the security domain becomes relevant then.

## Sources

### Primary (HIGH confidence)

- `/home/bidu/musicgen/music_gen.py` — direct source inspection, all 1054 lines read 2026-04-18
- `/home/bidu/musicgen/enhanced_duration_validator.py` — full source
- `/home/bidu/musicgen/config.py` — full source (Phase 2 artifact)
- `/home/bidu/musicgen/timesig.py` — full source (Phase 2 artifact)
- `/home/bidu/musicgen/tests/*.py` — full source of all 5 test modules
- `/home/bidu/musicgen/requirements.txt` — full source
- `/home/bidu/musicgen/.planning/phases/03-package-skeleton-sampler-generators-extraction/03-CONTEXT.md` — authoritative phase decisions
- `/home/bidu/musicgen/.planning/REQUIREMENTS.md`, `ROADMAP.md`, `STATE.md` — scope
- Empirical audit via `/home/bidu/musicgen/.venv/bin/python` of music21 9.9.1 global-random behavior — REPRODUCIBLE command included above
- `pytest tests/ -q` run 2026-04-18 — baseline 309 passing tests confirmed

### Secondary (MEDIUM confidence)

- [Writing your pyproject.toml — Python Packaging User Guide](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/)
- [Hatch build configuration 1.13](https://hatch.pypa.io/1.13/config/build/)
- [Hatchling history](https://hatch.pypa.io/dev/history/hatchling/) — cross-referenced version-drop dates
- [typer on PyPI](https://pypi.org/project/typer/) — Python requirement metadata
- [pytest Good Integration Practices](https://docs.pytest.org/en/stable/explanation/goodpractices.html) — src-layout guidance (fetched via WebSearch summary)
- [pytest import mechanisms and sys.path/PYTHONPATH](https://docs.pytest.org/en/stable/explanation/pythonpath.html) — conftest + pythonpath behavior

### Tertiary (LOW confidence)

- None — every claim marked `[ASSUMED]` is enumerated in the Assumptions Log with its risk/mitigation.

## Metadata

**Confidence breakdown:**

- Standard stack: **HIGH** — hatchling + typer + pytest are widely used; version-clash with 3.9 is VERIFIED via PyPI/hatch docs.
- Architecture: **HIGH** — every module boundary is dictated by CONTEXT.md; research only concretizes signatures.
- Pitfalls: **HIGH** — every identified risk is traced to VERIFIED code or VERIFIED tool behavior; no hand-waving.
- music21 audit: **HIGH** — empirical verification performed in the actual project venv; regression test ships with Phase 3.
- RNG order trace: **HIGH** — line-by-line walk of `music_gen.py:1010-1024` + `generate_song_measures:849-880`; every draw enumerated.

**Research date:** 2026-04-18
**Valid until:** 2026-05-18 (30 days) — primary deps (music21, hatchling, typer) are slow-moving; the music21 audit is only invalidated by a music21 upgrade, which our regression test catches.

## RESEARCH COMPLETE
