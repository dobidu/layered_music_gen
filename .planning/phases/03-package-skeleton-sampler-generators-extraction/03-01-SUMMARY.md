---
phase: 03-package-skeleton-sampler-generators-extraction
plan: 03-01
subsystem: infra
tags: [packaging, pyproject, hatchling, typer, cli, python-3.10, src-layout]

# Dependency graph
requires:
  - phase: 01-stabilize-i-bug-fixes-and-guardrails
    provides: ".venv at /home/bidu/musicgen/.venv (Plan 01-04); pytest skeleton + 95 tests"
  - phase: 02-stabilize-ii-config-time-signature-registry-logging
    provides: "309-passing test baseline; print→logging migration; timesig registry; config threading"
provides:
  - "pyproject.toml as single authoritative dependency manifest (hatchling + 14 runtime deps + typer + dev extras)"
  - "src/musicgen/ package skeleton (empty __init__.py, __main__.py delegator, cli.py stub)"
  - "Editable install working: `pip install -e '.[dev]'` succeeds; `musicgen --help` exits 0; `python -m musicgen --help` exits 0"
  - "`[tool.pytest.ini_options] pythonpath = ['.']` — keeps root config.py/timesig.py importable after conftest.py deletion in Plan 03-05"
  - "Legacy requirements.txt and dev-requirements.txt deleted"
affects: [03-02, 03-03, 03-04, 03-05, 04-xx, 05-xx, 06-xx]

# Tech tracking
tech-stack:
  added: [hatchling, typer>=0.12, pytest-xdist>=3.5]
  patterns: [src-layout package, pyproject.toml-only dep manifest, typer CLI stub, hatchling editable install]

key-files:
  created:
    - pyproject.toml
    - src/musicgen/__init__.py
    - src/musicgen/__main__.py
    - src/musicgen/cli.py
  modified:
    - README.md  # installation instructions updated to `pip install -e '.[dev]'`
  deleted:
    - requirements.txt
    - dev-requirements.txt

key-decisions:
  - "Applied RESEARCH.md Risk #1 override: requires-python='>=3.10' (CONTEXT.md D-13 ambition of >=3.9 infeasible — typer>=0.12 and hatchling require 3.10+). Inline comment in pyproject.toml preserves traceability."
  - "Rule 1 fix: relaxed `pedalboard>=1.0.0` (the spec in legacy requirements.txt) to `pedalboard>=0.9.0`. No 1.x release has ever shipped to PyPI (latest is 0.9.22, already installed in .venv); the original pin was aspirational and would have blocked the editable install."
  - "Rule 2 fix: updated README.md Installation section to instruct `pip install -e '.[dev]'` instead of `pip install -r requirements.txt`. Required because Task 3 deletes requirements.txt."
  - "typer single-command apps collapse the command into the root (no `info` subcommand shown in --help), so the plan's acceptance criterion 'stdout contains both `musicgen` and `info`' was softened to the plan's top-level must-have truth ('prints help text containing `musicgen`') and the Task 2 <automated> verify command (`grep -q musicgen`). The `info` command still runs correctly when invoked (`.venv/bin/musicgen` prints the stub text)."

patterns-established:
  - "Pattern (src-layout): package code lives under src/musicgen/, hatchling wheel built from src/musicgen/"
  - "Pattern (single dep manifest): pyproject.toml is the ONLY file declaring deps; no requirements.txt duplicate"
  - "Pattern (typer CLI stub): `app = typer.Typer(...)` + `@app.command()` + entry point `musicgen = 'musicgen.cli:app'` in [project.scripts]"
  - "Pattern (pytest pythonpath shim): `[tool.pytest.ini_options] pythonpath = ['.']` lets root-level modules remain importable from tests without tests/conftest.py"

requirements-completed: [R-X1, R-Q4]

# Metrics
duration: 3min
completed: 2026-04-18
---

# Phase 3 Plan 03-01: Package Skeleton + pyproject.toml Summary

**Stood up `src/musicgen/` src-layout package backed by a hatchling-based `pyproject.toml` manifest; `pip install -e '.[dev]'` now succeeds, `musicgen --help` exits 0, and requirements.txt/dev-requirements.txt are deleted — pyproject.toml is the single authoritative dependency source for all downstream plans.**

## Performance

- **Duration:** 3 min
- **Started:** 2026-04-18T20:14:11Z
- **Completed:** 2026-04-18T20:17:53Z
- **Tasks:** 3 / 3
- **Files modified/created/deleted:** 7 (4 created, 1 modified, 2 deleted)

## Accomplishments

- Editable install pipeline works end-to-end: `pip install -e '.[dev]'` completes cleanly in the project venv (Python 3.12.3); hatchling builds the wheel from `src/musicgen/`; the console-script entry point registers `musicgen` to `musicgen.cli:app`.
- `musicgen --help` exits 0 with help text containing "musicgen"; `python -m musicgen --help` exits 0; `import musicgen` and `from musicgen.cli import app` both succeed.
- `requirements.txt` and `dev-requirements.txt` removed (via `git rm`); pyproject.toml is the single authoritative manifest. All 14 runtime deps + 3 dev deps migrated; `typer>=0.12` and `pytest-xdist>=3.5` newly added per D-13.
- `[tool.pytest.ini_options].pythonpath = ["."]` wired — pre-bakes the fix for when Plan 03-05 deletes `tests/conftest.py` sys.path shim (RESEARCH.md Risk #3 fallback C).
- Regression baseline preserved: 309 tests still pass (before and after each task).

## Task Commits

1. **Task 1: Create src/musicgen/ package skeleton + pyproject.toml** — `5d0a64f` (feat)
2. **Task 2: Install editable + smoke-test CLI + module imports** — `6409a8e` (fix — pedalboard floor relaxed to unblock install; see Deviations)
3. **Task 3: Delete legacy requirements.txt and dev-requirements.txt** — `eb2a81a` (chore — also updated README.md Installation step, see Deviations)

## Files Created/Modified

- `pyproject.toml` — hatchling build backend; 14 runtime deps + typer>=0.12; dev extras (pytest, pytest-cov, pytest-xdist); `musicgen = "musicgen.cli:app"` console script; `[tool.hatch.build.targets.wheel] packages = ["src/musicgen"]`; `[tool.pytest.ini_options]` with testpaths and pythonpath.
- `src/musicgen/__init__.py` — Phase-3 empty package marker; Phase 5 will add public re-exports.
- `src/musicgen/__main__.py` — routes `python -m musicgen` to `musicgen.cli:app`.
- `src/musicgen/cli.py` — typer stub with one `info` command; real CLI lands in Phase 6 per D-18.
- `README.md` — Installation step 2 updated from `pip install -r requirements.txt` to `pip install -e '.[dev]'` (Rule 2 fix; the old instruction would break for new users after Task 3).
- `requirements.txt` — DELETED (superseded by pyproject.toml).
- `dev-requirements.txt` — DELETED (superseded by pyproject.toml [project.optional-dependencies].dev).

## Decisions Made

- **requires-python = ">=3.10" (Risk #1 override):** CONTEXT.md D-13 specified ">=3.9" but RESEARCH.md Risk #1 identified this as infeasible (typer>=0.12 and hatchling>=1.28 both require Python 3.10+). Applied the research override with an inline comment in pyproject.toml so the deviation is traceable. Project venv is 3.12.3, well within the new floor.
- **pedalboard floor relaxed to >=0.9.0:** The legacy requirements.txt declared `pedalboard>=1.0.0`, but no 1.x release has ever shipped to PyPI (latest version is 0.9.22). The strict copy-verbatim rule from the plan would have broken `pip install`; Rule 1 applies (bug that prevents completing the task's must-have truth). See Deviations.
- **README.md install instructions updated in this plan:** Task 3 deletes `requirements.txt`, which the README was telling users to `pip install -r`. Leaving that stale would silently break new-user onboarding — Rule 2.
- **typer single-command behavior accepted:** With one registered command, typer flattens the command into the root, so `musicgen --help` shows the `info` docstring directly without listing `info` as a subcommand. The plan's top-level must-have truth ("prints help text containing 'musicgen'") and Task 2's automated verify (`grep -q musicgen`) both pass; only the acceptance-criteria prose ("stdout contains both `musicgen` and `info`") was inaccurate. Invoking `.venv/bin/musicgen` with no args still runs `info` correctly.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 — Bug] `pedalboard>=1.0.0` blocks `pip install -e '.[dev]'` (no 1.x release exists on PyPI)**
- **Found during:** Task 2 (editable install)
- **Issue:** Legacy `requirements.txt` pinned `pedalboard>=1.0.0`. Copying that verbatim into pyproject.toml per the plan's strict "version specifiers match requirements.txt exactly" rule made pip fail with `ERROR: Could not find a version that satisfies the requirement pedalboard>=1.0.0 (from versions: 0.8.2 ... 0.9.22)`. The pin was aspirational; pedalboard has never released 1.x. The project's own `.venv` already had pedalboard 0.9.22 installed (compatible with the current codebase).
- **Fix:** Relaxed the spec in pyproject.toml to `pedalboard>=0.9.0`, which is satisfied by 0.9.22 without forcing a reinstall. Added an inline comment referencing the rule and the pre-existing requirements.txt bug.
- **Files modified:** `pyproject.toml`
- **Verification:** `pip install -e '.[dev]'` succeeds (exit 0); pytest 309 passed.
- **Committed in:** `6409a8e` (Task 2 commit).

**2. [Rule 2 — Missing Critical] README.md Installation section would break for new users after Task 3**
- **Found during:** Task 3 (post-deletion scan for stale references to requirements.txt)
- **Issue:** README.md Installation step 2 said `pip install -r requirements.txt`. Task 3 deletes that file; following the README post-Task-3 would fail with "No such file or directory".
- **Fix:** Updated step 2 to `python -m venv .venv && source .venv/bin/activate && pip install -e '.[dev]'`, with a note that `requirements.txt` was removed in Phase 3 and pyproject.toml is now authoritative.
- **Files modified:** `README.md` (lines 40-47).
- **Verification:** Rendered correctly; `pip install -e '.[dev]'` is the literal command that works (already verified in Task 2).
- **Committed in:** `eb2a81a` (Task 3 commit).

**3. [Rule 1 — Acceptance-criteria discrepancy] typer collapses single-command app; `info` doesn't appear in `--help`**
- **Found during:** Task 2 (CLI smoke test)
- **Issue:** Task 2 acceptance criterion said "stdout contains both `musicgen` and `info` (the stub command)". In typer 0.24.1, a `Typer` app with exactly one `@app.command()` collapses that command into the root — the `info` docstring becomes the root docstring, and there is no `info` subcommand entry in `--help` output. So `grep -c info /tmp/musicgen-help.log` returns `0` while `grep -c musicgen` returns `1`.
- **Decision:** The plan's top-level must-have truth ("prints help text containing 'musicgen'") and the Task 2 `<automated>` verify command (`grep -q musicgen /tmp/musicgen-help.log`) both pass. The `info` command still executes correctly — invoking `.venv/bin/musicgen` (no args) prints the intended stub text. The acceptance-criterion prose was typer-behavior-misprediction; no code fix needed.
- **Files modified:** None (documentation/discrepancy only).
- **Verification:** `.venv/bin/musicgen` produces `musicgen 0.1.0 — Phase 3 package skeleton\nReal CLI (generate / batch / clean / calibrate) arrives in Phase 6. ...`. Future plans adding a second typer command will naturally make both command names appear in `--help`.
- **Committed in:** N/A (no code change).

---

**Total deviations:** 3 (2 auto-fixes, 1 documentation-only discrepancy). All necessary to unblock the plan's must-have truths; no scope creep.
**Impact on plan:** Zero requirements impact — R-X1 and R-Q4 both closed. No architectural decisions deferred.

## Issues Encountered

- **Pre-existing git status noise:** The working tree had many pre-existing mode-only changes (644→755) on unrelated files (e.g., `beat_roll_patterns_*.txt`, soundfont manifests, `LICENSE`, `README.md`, etc.) before this plan started. Handled by staging only the specific files this plan touched (never used `git add .` or `git add -A`). The mode-change on `README.md` ended up in Task 3's commit because I also edited the README content there — unavoidable and cosmetic.
- **`dev-requirements.txt` was already locally deleted (but git-tracked):** At plan start, the file was shown as `D dev-requirements.txt` in git status (deleted from working tree but still tracked). `git rm -f dev-requirements.txt` handled it cleanly. Added a sanity-check using `git show HEAD:dev-requirements.txt` to verify all tracked entries were present in pyproject.toml dev extras before deletion.

## Install-Log Warnings Worth Calling Out

- `pytest` emitted 2 warnings (pre-existing, unrelated to this plan):
  - `DeprecationWarning: 'audioop' is deprecated and slated for removal in Python 3.13` (pydub 0.25.1 internal — not our code).
  - `RuntimeWarning: Couldn't find ffmpeg or avconv - defaulting to ffmpeg, but may not work` (pydub utilities — environmental, only matters when we actually render audio).
- No ERROR lines in `/tmp/musicgen-install.log`. The install pulled: `annotated-doc 0.0.4, click 8.3.2, execnet 2.1.2, markdown-it-py 4.0.0, mdurl 0.1.2, musicgen 0.1.0 (editable), pytest-xdist 3.8.0, python-magic 0.4.27, rich 15.0.0, shellingham 1.5.4, typer 0.24.1`. All other runtime deps were already satisfied from the Phase 1 Plan 01-04 venv.

## User Setup Required

None — pyproject.toml fully drives the editable install; no environment variables or external services added.

## Next Phase Readiness

- **Plan 03-02** (music21 global-RNG audit regression tests) can now `from musicgen.` (though the plan will write `src/musicgen/...`-targeting tests directly). The package import surface is live.
- **Plans 03-03, 03-04, 03-05** all depend on `pip install -e '.[dev]'` working before they can `from musicgen.sampler import ...` / `from musicgen.generators.* import ...` / `from musicgen.duration_validator import ...`. That prerequisite is now satisfied.
- **Baseline test count: 309.** Downstream plans must preserve (not regress) this count. Plan 03-02 will add ≥3 music21 isolation tests; Plan 03-03/03-04 will add sampler + per-generator tests; Plan 03-05 will delete `tests/conftest.py` (the `pythonpath=["."]` directive already in pyproject.toml will keep root-level `config.py`/`timesig.py` importable).
- **Carryover caution for Plan 03-05:** When `tests/conftest.py` is removed, re-verify `pytest tests/ -q` still sees `config.py` and `timesig.py`. If there's any import drift, the fallback is to move those two files under `src/musicgen/` (R-S5 / R-S6 test mocks currently patch them at the root path — that would also need updating). Out of scope for this plan.

## Self-Check: PASSED

**Created files verified:**
- FOUND: `/home/bidu/musicgen/pyproject.toml`
- FOUND: `/home/bidu/musicgen/src/musicgen/__init__.py`
- FOUND: `/home/bidu/musicgen/src/musicgen/__main__.py`
- FOUND: `/home/bidu/musicgen/src/musicgen/cli.py`

**Deleted files verified absent:**
- ABSENT: `/home/bidu/musicgen/requirements.txt`
- ABSENT: `/home/bidu/musicgen/dev-requirements.txt`

**Commits verified in git log:**
- FOUND: `5d0a64f` (Task 1)
- FOUND: `6409a8e` (Task 2)
- FOUND: `eb2a81a` (Task 3)

**Automated verification:**
- `pip install -e '.[dev]'`: exit 0.
- `musicgen --help`: exit 0, contains "musicgen".
- `python -m musicgen --help`: exit 0.
- `python -c "import musicgen; from musicgen.cli import app; print(type(app).__name__)"`: prints `Typer`.
- `pytest tests/ -q`: 309 passed.
- Risk #1 traceability comment present in pyproject.toml line 11.

---
*Phase: 03-package-skeleton-sampler-generators-extraction*
*Plan: 03-01*
*Completed: 2026-04-18*
