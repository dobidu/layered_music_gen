---
phase: 04-renderer-mixer-annotator-beats-extraction
plan: 00
type: execute
wave: 0
depends_on: []
files_modified:
  - pyproject.toml
  - tests/test_beats.py
  - tests/test_renderer.py
  - tests/test_mixer.py
  - tests/test_annotator.py
  - tests/test_no_bare_random_in_package.py
  - tests/test_integration_full_generation.py
autonomous: true
requirements: [R-X4, R-X5, R-X6, R-X7, R-X8]
tags: [phase-4, infrastructure, dependencies, test-scaffolds]

must_haves:
  truths:
    - "mido 1.3.3+ is installed and importable"
    - "pytest markers `slow` and `integration` are declared — no UnknownMarkWarning"
    - "All 6 new test files exist and are discoverable by pytest (as skipping stubs)"
    - "371-test baseline still passes after this plan"
  artifacts:
    - path: "pyproject.toml"
      provides: "mido>=1.3.3 dep + markers list"
      contains: 'mido>=1.3.3'
    - path: "pyproject.toml"
      provides: "pytest markers declaration"
      contains: 'markers ='
    - path: "tests/test_beats.py"
      provides: "Beats test scaffold (skipped stubs)"
    - path: "tests/test_renderer.py"
      provides: "Renderer test scaffold (skipped stubs)"
    - path: "tests/test_mixer.py"
      provides: "Mixer test scaffold (skipped stubs)"
    - path: "tests/test_annotator.py"
      provides: "Annotator test scaffold (skipped stubs)"
    - path: "tests/test_no_bare_random_in_package.py"
      provides: "Package-wide AST guard scaffold (may pass as-is since existing src/musicgen/ already clean)"
    - path: "tests/test_integration_full_generation.py"
      provides: "E2E integration test scaffold (@pytest.mark.slow skipped stub)"
  key_links:
    - from: "pyproject.toml [project].dependencies"
      to: "mido"
      via: "pinned minimum version"
      pattern: '"mido>=1\\.3\\.3"'
    - from: "pyproject.toml [tool.pytest.ini_options]"
      to: "markers list"
      via: "markers ="
      pattern: 'slow:.*fluidsynth'
---

<objective>
Wave 0: unblock the entire Phase 4 chain by (1) adding the `mido>=1.3.3` runtime dep that was a false-assumed transitive dep of `midi2audio` (RESEARCH correction #3), (2) declaring the `slow` and `integration` pytest markers so Wave 6's `@pytest.mark.slow` integration test does not emit UnknownMarkWarnings, (3) refreshing the editable install so `import mido` works in subsequent waves, and (4) scaffolding the 6 new test files with skipping stubs so later waves can fill them in without having to create files under context pressure.

Purpose: Wave 1 needs `mido` to implement `beats.py`; Wave 6 needs declared markers. Both are blocking. Creating stubs in Wave 0 also lets every later wave land its tests as in-place file edits (lower context cost) rather than cold-writing 200-line files under implementation pressure.

Output: `pyproject.toml` updated with 1 dep line + markers list; `pip install -e '.[dev]'` run; 6 new test files exist containing skipping stubs (`pytest.skip("Wave N placeholder")`). Regression baseline green.
</objective>

<execution_context>
@/home/bidu/musicgen/.claude/get-shit-done/workflows/execute-plan.md
@/home/bidu/musicgen/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/phases/04-renderer-mixer-annotator-beats-extraction/04-CONTEXT.md
@.planning/phases/04-renderer-mixer-annotator-beats-extraction/04-RESEARCH.md
@.planning/phases/04-renderer-mixer-annotator-beats-extraction/04-VALIDATION.md

<interfaces>
<!-- Current pyproject.toml [project].dependencies (from Read) -->
dependencies = [
    "numpy>=1.20.0",
    "scipy>=1.7.0",
    "midiutil>=1.2.1",
    "music21>=7.3.3",
    "librosa>=0.9.2",
    "pydub>=0.25.1",
    "midi2audio>=0.1.1",
    "pedalboard>=0.9.0",
    "python-json-logger>=2.0.7",
    "typing-extensions>=4.4.0",
    "numba>=0.56.4",
    "llvmlite>=0.39.1",
    "python-magic>=0.4.27",
    "typer>=0.12",
]

<!-- Current pyproject.toml [tool.pytest.ini_options] (from Read) -->
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
<!-- No markers section yet -->
</interfaces>
</context>

<tasks>

<task type="auto">
  <name>Task 0: pyproject.toml — add mido dep + pytest markers list</name>
  <files>pyproject.toml</files>
  <read_first>
    - pyproject.toml (current state: mido missing, markers section missing)
    - .planning/phases/04-renderer-mixer-annotator-beats-extraction/04-RESEARCH.md §"New Dependency Required" and §"pyproject.toml markers — NOT DECLARED"
  </read_first>
  <action>
Edit `pyproject.toml` to make two additions:

**Addition 1** — In the `[project] dependencies` list, insert `"mido>=1.3.3",` after the `"midi2audio>=0.1.1",` line (alphabetical-ish placement near sibling MIDI lib). Rationale comment BEFORE the line (inline TOML comment): `# RESEARCH correction #3: mido is NOT a transitive dep of midi2audio (which declares zero Python-level requirements); beats.py needs it for MIDI-tick extraction.`

**Addition 2** — In the `[tool.pytest.ini_options]` block (currently at line 43-45 with only `testpaths` and `pythonpath`), append a `markers =` list after `pythonpath = ["."]`:

```toml
markers = [
    "slow: FluidSynth rendering tests — requires fluidsynth binary on PATH (deselect with '-m \"not slow\"')",
    "integration: end-to-end tests requiring system dependencies",
]
```

Do NOT add `--strict-markers` to `addopts` — rollout is additive; strict mode is a Phase 6+ concern.

This task implements CONTEXT.md D-26..D-32 infra precondition and the two RESEARCH corrections (§Dependency Verification 1 + 5).
  </action>
  <verify>
    <automated>python -c "import tomllib, pathlib; d = tomllib.loads(pathlib.Path('pyproject.toml').read_text()); deps = d['project']['dependencies']; assert any(x.startswith('mido>=1.3') for x in deps), f'mido>=1.3.x not in deps: {deps}'; ini = d['tool']['pytest']['ini_options']; assert 'markers' in ini, f'markers key missing from ini_options: {list(ini)}'; assert any('slow' in m for m in ini['markers']), f'slow marker missing: {ini[\"markers\"]}'; assert any('integration' in m for m in ini['markers']), f'integration marker missing: {ini[\"markers\"]}'; print('OK')"</automated>
  </verify>
  <acceptance_criteria>
    - `python -c "import tomllib, pathlib; d=tomllib.loads(pathlib.Path('pyproject.toml').read_text()); print(d['project']['dependencies'])"` output contains the substring `mido>=1.3.3`
    - `python -c "import tomllib, pathlib; d=tomllib.loads(pathlib.Path('pyproject.toml').read_text()); print(d['tool']['pytest']['ini_options'].get('markers'))"` output contains BOTH `slow` and `integration` tokens
    - Existing `testpaths = ["tests"]` and `pythonpath = ["."]` entries remain intact (not regressed)
    - File is parseable by `tomllib` (automated command above would error on parse failure)
  </acceptance_criteria>
  <done>pyproject.toml has `mido>=1.3.3` in [project].dependencies and a `markers =` list with `slow` + `integration` entries under [tool.pytest.ini_options]; file is valid TOML.</done>
</task>

<task type="auto">
  <name>Task 1: pip install -e '.[dev]' — refresh editable install so import mido works</name>
  <files>.venv/ (package state side effect — no file edits)</files>
  <read_first>
    - pyproject.toml (confirm mido is now listed)
  </read_first>
  <action>
Run `pip install -e '.[dev]'` from repo root to install the new `mido` dep and refresh the editable install. Expected output: `Successfully installed mido-1.3.x ...` (or "Requirement already satisfied" lines if mido happened to be installed; either is fine).

Verify by running `python -c "import mido; print(mido.__version__)"` — must print a version >= 1.3.0 with no ModuleNotFoundError.

No code files are modified by this task — it is pure dependency-refresh. The task exists because `pyproject.toml` edits alone do not mutate the venv; Wave 1's beats.py needs `import mido` to work at test time.
  </action>
  <verify>
    <automated>pip install -e '.[dev]' >/dev/null 2>&1 && python -c "import mido; v = tuple(int(x) for x in mido.__version__.split('.')[:3]); assert v >= (1, 3, 0), f'mido version {mido.__version__} < 1.3.0'; print(f'mido {mido.__version__} OK')"</automated>
  </verify>
  <acceptance_criteria>
    - `pip show mido` exits 0 and prints `Name: mido` plus `Version: 1.3.x` (or higher 1.x)
    - `python -c "import mido; mido.bpm2tempo(120)"` succeeds with no error (verifies the specific API beats.py will use)
    - `python -c "from midi2audio import FluidSynth"` still succeeds (smoke: editable reinstall did not break the existing deps)
  </acceptance_criteria>
  <done>`import mido` succeeds in the active .venv; `mido.bpm2tempo` and `mido.tick2second` are both callable.</done>
</task>

<task type="auto">
  <name>Task 2: Create the 6 test file scaffolds as skipping stubs</name>
  <files>
    tests/test_beats.py,
    tests/test_renderer.py,
    tests/test_mixer.py,
    tests/test_annotator.py,
    tests/test_no_bare_random_in_package.py,
    tests/test_integration_full_generation.py
  </files>
  <read_first>
    - tests/test_sampler.py (reference: scaffold shape, AST-guard helper pattern at lines 165-180, 183-196)
    - tests/test_generators/test_no_bare_random.py (reference: parametrize-over-glob pattern, lines 13-44)
    - tests/test_generators/test_beat.py (reference: per-module scaffold shape — if this file exists)
    - .planning/phases/04-renderer-mixer-annotator-beats-extraction/04-PATTERNS.md §"tests/test_*.py" sections (authoritative scaffolds)
  </read_first>
  <action>
Create each of the 6 test files as a MINIMAL skipping-stub file with a module-level `pytest.skip("Wave N placeholder — populated by Plan 04-0N")` guard followed by a single `def test_placeholder(): pass`. The body is intentionally empty — later waves REPLACE these files entirely with real tests. The stubs exist so pytest collection discovers the files immediately (clean baseline: 371 passing, 6 skipped) rather than ImportError on "file does not exist" in later waves' automated verify commands.

**Stub template (copy for each file, substitute marked fields):**

```python
"""[MODULE PURPOSE] — Plan 04-0N placeholder scaffold (Wave 0).

This file is a Wave 0 skipping stub. Plan 04-0N (Wave N) replaces the entire
body with real tests. The stub exists so pytest collection succeeds from Wave 0
onward; collection failure breaks pre-commit AST tests that iterate tests/.
"""
import pytest

pytest.skip(
    "Plan 04-00 (Wave 0) scaffold — Plan 04-0N (Wave N) will replace this stub.",
    allow_module_level=True,
)


def test_placeholder():
    """Placeholder; will be replaced by Plan 04-0N's real tests."""
    pass
```

**Per-file header substitutions (use these verbatim as the docstring first line):**

1. `tests/test_beats.py` — `"""Beats tests (R-X7): extract_beat_times + extract_downbeat_times + swing cases."""` — stub text: `Plan 04-01 (Wave 1) will replace this stub.`
2. `tests/test_renderer.py` — `"""Renderer tests (R-X4): RenderResult assembly + FLUIDSYNTH_VERSION capture."""` — stub text: `Plan 04-02 (Wave 2) will replace this stub.`
3. `tests/test_mixer.py` — `"""Mixer tests (R-X5): seeded-RNG determinism + silent-stem channel parity + D-11 guard."""` — stub text: `Plan 04-03 (Wave 3) will replace this stub.`
4. `tests/test_annotator.py` — `"""Annotator tests (R-X6): fixture-driven pure-function contract with D-16 None semantics."""` — stub text: `Plan 04-04 (Wave 4) will replace this stub.`
5. `tests/test_no_bare_random_in_package.py` — `"""Static guard: zero bare random.<method>() in src/musicgen/**/*.py (D-17/D-31)."""` — stub text: `Plan 04-05 (Wave 5) will replace this stub.`
6. `tests/test_integration_full_generation.py` — `"""Integration test (R-X8): full pipeline smoke test with real FluidSynth binary (@pytest.mark.slow)."""` — stub text: `Plan 04-06 (Wave 6) will replace this stub.`

All 6 files use `pytest.skip(..., allow_module_level=True)` so pytest reports them as `SKIPPED`, not as test failures. They contribute to collection counts without running any assertions.

This follows CONTEXT.md D-26..D-32 and lets each downstream wave's automated verify command `pytest tests/test_X.py -x` produce a clean skip result as baseline, flipping to pass as real tests are added.
  </action>
  <verify>
    <automated>python -m pytest tests/test_beats.py tests/test_renderer.py tests/test_mixer.py tests/test_annotator.py tests/test_no_bare_random_in_package.py tests/test_integration_full_generation.py --collect-only -q 2>&1 | tail -20 && python -m pytest tests/test_beats.py tests/test_renderer.py tests/test_mixer.py tests/test_annotator.py tests/test_no_bare_random_in_package.py tests/test_integration_full_generation.py -q 2>&1 | tail -5</automated>
  </verify>
  <acceptance_criteria>
    - All 6 files exist at the specified paths
    - Each file contains `pytest.skip(..., allow_module_level=True)` at module scope (verifiable via grep)
    - Each file's module docstring first line matches the R-X# reference above (verifiable via head -1)
    - `pytest tests/test_beats.py tests/test_renderer.py tests/test_mixer.py tests/test_annotator.py tests/test_no_bare_random_in_package.py tests/test_integration_full_generation.py -q` exits 0 with output indicating "6 skipped" or "skipped" (not "error" or "failed")
    - `pytest tests/ -m "not slow" -q` passes with the 371 baseline tests unchanged plus the new 6 skips; exit code 0
  </acceptance_criteria>
  <done>All 6 test files exist, are skipping stubs, pytest collection succeeds on each, and the 371-test baseline is preserved (pytest suite passes overall with 6 skips added).</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| PyPI → local venv | `pip install -e '.[dev]'` pulls `mido` from PyPI into the dev environment |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-04-00-01 | Tampering | `mido` package from PyPI (supply chain) | accept | Version pin `mido>=1.3.3` matches RESEARCH-verified version; PyPI is the standard dep source already used for all other listed libraries; no new trust surface introduced vs existing deps (pedalboard, pydub, midi2audio) |
| T-04-00-02 | Information Disclosure | pyproject.toml contents | accept | No secrets in pyproject.toml; all fields are project metadata already committed to the repo |
</threat_model>

<verification>
After all 3 tasks complete:

1. `python -c "import mido; print(mido.__version__)"` — prints a 1.3+ version
2. `python -m pytest tests/ -m "not slow" -q --tb=short` — all 371 existing tests pass; 6 new test files show as skipped (not errored)
3. `python -c "import tomllib, pathlib; d = tomllib.loads(pathlib.Path('pyproject.toml').read_text()); assert any('mido' in x for x in d['project']['dependencies']); assert 'markers' in d['tool']['pytest']['ini_options']"` — exits 0
4. No test file is EMPTY (all 6 have `pytest.skip(...)` at module level)
</verification>

<success_criteria>
- `mido>=1.3.3` is in `pyproject.toml [project].dependencies`
- `markers` list with `slow` and `integration` entries is in `[tool.pytest.ini_options]`
- `import mido` works in the active .venv
- All 6 scaffold test files exist with skipping stubs
- 371 baseline tests still pass; pytest suite overall passes with 6 new skips
- Zero new failing tests introduced by this plan
</success_criteria>

<output>
After completion, create `.planning/phases/04-renderer-mixer-annotator-beats-extraction/04-00-SUMMARY.md`.

Include:
- Final state of `pyproject.toml` diff (mido line + markers block)
- Output of `pip show mido` and `python -c "import mido; print(mido.__version__)"`
- Output of `pytest tests/ -m "not slow" -q --tb=short` tail (X passed, 6 skipped)
- Confirmation that each of the 6 stubs is discoverable via `pytest --collect-only`
- Any deviations from the plan (Rule 1/2 fixes)
</output>
