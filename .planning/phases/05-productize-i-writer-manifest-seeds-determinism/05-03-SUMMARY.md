---
phase: 05-productize-i-writer-manifest-seeds-determinism
plan: 05-03
subsystem: package-move
tags: [phase-5, wave-2, git-mv, musicality, package-relocation, d-03, r-p4]

# Dependency graph
requires:
  - phase: 03-package-skeleton-sampler-generators-extraction
    provides: "Plan 03-02 precedent — git mv rename pattern for `enhanced_duration_validator.py` → `src/musicgen/duration_validator.py` (R100, no back-compat shim, docstring/comment sweep under Rule 2)"
  - phase: 04-renderer-mixer-annotator-beats-extraction
    provides: "Plan 04-04 annotator emits the R-P4 `musicality_score` JSON field (Phase 4 D-15); Plan 04-05 orchestrator imports `musicality_score` at music_gen.py:3 — this plan's rewrite target"
  - phase: 05-productize-i-writer-manifest-seeds-determinism
    provides: "Plan 05-01 test infrastructure (conftest.py, AST guard widened allow-list); Plan 05-02 seeds.py in place with internal docstring references to `musicgen.musicality.get_musicality_score` already using the new path"
provides:
  - "`src/musicgen/musicality.py` importable as `from musicgen.musicality import get_musicality_score` (relocated from repo root via git mv — R100 rename, full `git log --follow` history back through commit `350bf60` / `94c19a0`)"
  - "Old path `musicality_score.py` raises `ModuleNotFoundError` (no back-compat shim per D-03, matching the Plan 03-02 precedent exactly)"
  - "music_gen.py temporary bridge — `from musicgen import musicality` + `musicality.get_musicality_score(...)` call site (Wave 4 Plan 05-05 will delete the entire create_song body that uses this)"
  - "Clean precondition for Wave 4 Plan 05-05 (api.py) — the import line `from musicgen import renderer, mixer, annotator, beats, writer, musicality` (05-PATTERNS.md line 273) is now resolvable"
affects: [05-05-api, 05-06-init-exports, 06-batch-cli]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Wave 2 module move via `git mv` (100% rename; byte-identical content; history preserved) — identical shape to Plan 03-02 (`enhanced_duration_validator.py` → `src/musicgen/duration_validator.py`), now a repeatable Phase 5 pattern"
    - "R-P4 schema field key preservation discipline: `\"musicality_score\"` as a JSON dict key in the Phase 4 R-P4 annotation schema is a distinct concept from the `musicality_score` module name and is preserved through any future module renames (Rule 2 scope clarification, documented in this plan's deviations)"

key-files:
  created:
    - "src/musicgen/musicality.py (via rename — same bytes as the prior musicality_score.py; MusicalityAnalyzer class + get_musicality_score free function preserved verbatim)"
  modified:
    - "music_gen.py (line 3 `import musicality_score, config` → `import config` + `from musicgen import musicality`; line 121 call site `musicality_score.get_musicality_score(...)` → `musicality.get_musicality_score(...)`)"
    - "src/musicgen/annotator.py (line 110 docstring reference: `musicality_score.get_musicality_score` → `musicgen.musicality.get_musicality_score`)"
    - "tests/test_integration_full_generation.py (line 17 module docstring pipeline diagram: `musicality_score.get_musicality_score` → `musicgen.musicality.get_musicality_score`)"
  deleted:
    - "musicality_score.py (repo root — moved, not deleted; git sees 100% rename)"

key-decisions:
  - "Rename verb via `git mv` — confirmed by `git diff --cached --name-status -M` showing `R100` (byte-identical) and `git log --follow` traversing back through commit `350bf60` (Plan 01-03 exception handler narrowing) and `94c19a0` (initial upload). Matches Plan 03-02's verification exactly."
  - "Extended scope under Rule 2 to rewrite 2 docstring module-name references (annotator.py:110 and test_integration_full_generation.py:17) — both described the pipeline using the old module name in comment/docstring position. Mechanical rewrite to the fully qualified `musicgen.musicality.get_musicality_score` matches 03-02's precedent (Plan 03-02 rewrote 4 such references)."
  - "R-P4 JSON schema field key `\"musicality_score\"` PRESERVED in `src/musicgen/annotator.py:11` (module docstring R-P4 field listing), `src/musicgen/annotator.py:163` (annotation dict literal), `tests/test_annotator.py:146` (schema assertion list), and `tests/test_integration_full_generation.py:158` (schema assertion list). These are NOT module-name references; they are the frozen Phase 4 D-15 / D-16 schema contract. Changing them would break R-X6 (Phase 4), the forthcoming R-P4 Wave 3 writer contract, the manifest D-13 entry shape (`\"musicality_score\": 0.87`), and the Wave 5 sample.json byte-identity goldens. Scope boundary applied — the plan's grep invariant `wc -l == 0` was overstated relative to its stated rewrite rule (`→ musicality` or `→ musicgen.musicality`, whichever is syntactically correct), and a schema field key fits neither replacement."
  - "Did NOT touch the moved file's internal references (`src/musicgen/musicality.py` lines 243/250/254 — `def get_musicality_score(...)`, `print('Usage: python musicality_score_ii.py …')`, `score, component_scores = get_musicality_score(filename)`). Per plan's explicit carveout: the moved file's own internal docstring/comment references are allowed, matching 03-02-SUMMARY's '4 docstring/comment references were EXTERNAL to duration_validator.py' pattern."

patterns-established:
  - "Phase 5 Wave 2 module move pattern: `git mv <root> src/musicgen/<new>` + rewrite syntactic module-name references + PRESERVE JSON schema field keys with the same textual name. Reusable whenever a package file's legacy module name textually collides with a schema/contract key name."
  - "Module reference vs. schema field disambiguation: a `musicality_score` token in code is a module reference iff it appears in an `import` statement, a dotted attribute access (`musicality_score.X`), or a docstring/comment describing a module or its function. It is a schema field key iff it appears as a string literal in a dict (`\"musicality_score\": ...`) or in an R-P4 field assertion list. Only the former class is rewritten during a module rename."

requirements-completed: [R-P4]  # partial — R-P4's musicality field landing path closes here by making `from musicgen import musicality` resolvable; full R-P4 schema completion awaits Wave 4 api.py + Wave 3 writer.py.

# Metrics
duration: 3 min
completed: 2026-04-19
---

# Phase 5 Plan 05-03: Musicality Module Relocation Summary

**Moved `musicality_score.py` → `src/musicgen/musicality.py` via `git mv` (R100, history preserved), rewrote the single live import site in `music_gen.py`, swept 2 docstring module-name references — full suite 634/634 still green; R-P4 JSON schema field key `"musicality_score"` preserved as a deliberate scope clarification.**

## Performance

- **Duration:** 3 min
- **Started:** 2026-04-19T21:29:18Z
- **Completed:** 2026-04-19T21:32:14Z
- **Tasks:** 1/1
- **Files modified:** 4 (1 renamed + 3 edited)

## Accomplishments

- `musicality_score.py` → `src/musicgen/musicality.py` via `git mv`. Git confirms `rename musicality_score.py => src/musicgen/musicality.py (100%)` in the commit output; `git diff --cached --name-status -M` reported `R100`; `git log --follow src/musicgen/musicality.py` correctly traverses the rename back through `350bf60` (Plan 01-03 — narrow exception handlers in musicality_score) and `94c19a0` (initial upload). The D-03 deferral originally noted in Phase 3 D-11 (re-deferred as Phase 4 D-04) is finally closed.
- Both live references in `music_gen.py` rewritten atomically in the same commit: line 3 split `import musicality_score, config` into `import config` + `from musicgen import musicality` (clean grep-safe idiom); line 121 rewritten to `musicality.get_musicality_score(final_wav)`. The pre-existing local-variable-shadowing pattern on line 122 (`musicality = {"score": …, "components": …}`) is unchanged and still safe — the module reference on 121 is the last use before the shadowing; the shadowed name (a plain dict) is what `annotator.annotate` wants two frames down.
- `from musicality_score import …` now raises `ModuleNotFoundError: No module named 'musicality_score'` — verified. No back-compat shim at the old path (D-03).
- Moved file is **byte-identical** to the original — R100 is confirmed by git. The mode change (`100644 → 100755`) that was already staged pre-session (chmod +x artifact) piggybacked into the same commit; content diff remains zero bytes.
- Regression safety: `.venv/bin/pytest tests/ -q` reports **634 passed, 6 skipped, 1 xfailed**. Net +1 over Plan 05-02's 633-passed baseline — the AST guard (`tests/test_no_bare_random_in_package.py`) auto-picked up `src/musicgen/musicality.py` via its `glob.glob('src/musicgen/**/*.py')` parametrize and passed (no bare stdlib `random.*` calls in the moved file — it uses `numpy.random` transitively via librosa, not stdlib `random`).

## Task Commits

1. **Task 1: git mv musicality_score.py → src/musicgen/musicality.py + rewrite music_gen.py import site + sweep stale references** — `48f71ac` (refactor)

_No plan metadata commit yet — that happens in the finalization step after STATE.md / ROADMAP.md land._

## Files Created/Modified

- `src/musicgen/musicality.py` — MusicalityAnalyzer class (5 analyze_* methods + calculate_musicality) and the module-level `get_musicality_score` free function (relocated from repo root; contents unchanged; a `main()` + `if __name__ == '__main__':` block also moves — it keeps the moved file runnable as a CLI for ad-hoc debugging).
- `music_gen.py` — line 3 import rewritten (split into `import config` + `from musicgen import musicality`); line 121 call site rewritten (`musicality.get_musicality_score(...)`).
- `src/musicgen/annotator.py` — docstring on line 110 updated from `musicality_score.get_musicality_score` → `musicgen.musicality.get_musicality_score` (module-name reference sweep; no logic change).
- `tests/test_integration_full_generation.py` — module-docstring pipeline diagram on line 17 updated from `musicality_score.get_musicality_score` → `musicgen.musicality.get_musicality_score` (module-name reference sweep; no logic change).
- `musicality_score.py` — removed (via rename target, NOT a separate deletion; git's R100 confirms this).

## Decisions Made

- **Extended the edit scope beyond `music_gen.py` to sweep 2 external docstring module-name references under Rule 2.** The plan's Step C explicitly scripts this: grep for `musicality_score` across the tracked Python tree, rewrite every surviving hit that is NOT internal to the moved file to `musicgen.musicality`. Found 2 such hits (`src/musicgen/annotator.py:110`, `tests/test_integration_full_generation.py:17`), both describing the module in docstring position. Rewrite is the exact 03-02 precedent — 4 doc-comment refs were swept there; 2 here.
- **PRESERVED the R-P4 JSON schema field key `"musicality_score"` in 4 locations.** This is a named scope clarification that I applied as Rule 2 (missing critical — the plan's grep invariant would have forced me to rename a frozen Phase 4 schema contract). The plan's own rewrite spec says "→ `musicality` (the local import name) or `musicgen.musicality` (fully qualified), whichever is syntactically correct" — a JSON string literal `"musicality_score"` is neither syntactically correct target. The alternative (renaming the schema field) would cascade into Plan 04-04's annotator contract, Plan 05-01's test infrastructure, Wave 3 writer.py, Wave 5 determinism goldens, and the manifest schema (05-CONTEXT D-13's `"musicality_score": 0.87`). That cascade is architectural (Rule 4 territory), so I pinned the preserve decision to the narrower concept-disambiguation fix instead. Documented here, in the deviation section below, and as a `patterns-established` entry so future phases encountering similar textual collisions can apply the same rule.
- **Did NOT touch the moved file's internal references** at `src/musicgen/musicality.py` lines 243 (`def get_musicality_score`), 250 (`print('Usage: python musicality_score_ii.py …')`), 254 (`get_musicality_score(filename)`). Lines 243/254 are the free-function name (with `_score` suffix) — not a module reference. Line 250 is an internal CLI usage docstring pointing at a legacy filename (`musicality_score_ii.py`), preserved under the plan's explicit carveout for internal-to-moved-file docstrings (mirrors 03-02's preservation of duration_validator.py's internal structure).
- **Did NOT create a back-compat shim** at the old path. D-03 binding; 03-02 precedent exact.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 — Missing Critical] Swept 2 docstring module-name references for the zero-grep-hits acceptance criterion**

- **Found during:** Task 1 (Step C sweep, after Step B's music_gen.py rewrite).
- **Issue:** The plan's Step C explicitly anticipated "surprise hits" in comment/docstring position and prescribed rewriting them to `musicgen.musicality`. Found 2 such hits on grep: `src/musicgen/annotator.py:110` (inside `annotate`'s `musicality` parameter docstring, phrased as `(dict) from \`\`musicality_score.get_musicality_score\`\``) and `tests/test_integration_full_generation.py:17` (module docstring listing the pipeline stages as `... → musicality_score.get_musicality_score → annotator.annotate → ...`). Leaving them would leave dangling stale module-name references in the tree even though the acceptance grep's scope (executable code only) technically permits comment-only references.
- **Fix:** Rewrote each occurrence from `musicality_score` → `musicgen.musicality`. Zero-byte change to executable code; only docstring/comment text updated. Matches the 03-02 precedent (4 doc-comment refs were swept there, satisfying the then-binding zero-grep-hits gate).
- **Files modified:** `src/musicgen/annotator.py` (1 docstring line), `tests/test_integration_full_generation.py` (1 module-docstring line).
- **Verification:** `grep -rn "musicality_score\.get_musicality_score" --include="*.py" --exclude-dir=.venv --exclude-dir=__pycache__ --exclude-dir=.planning .` returns empty. Full test suite 634 passed / 6 skipped / 1 xfailed.
- **Committed in:** `48f71ac` (folded into Task 1 commit).

**2. [Rule 2 — Missing Critical / Scope Clarification] Preserved the R-P4 `"musicality_score"` JSON schema field key**

- **Found during:** Task 1 (Step C sweep, inspecting the 12 total grep hits).
- **Issue:** The plan's Step C grep invariant (`wc -l == 0` when excluding only `src/musicgen/musicality.py:` lines) would, if enforced literally, force a rename of the R-P4 schema field `"musicality_score"` in 4 places: `src/musicgen/annotator.py:11` (module docstring R-P4 field listing), `src/musicgen/annotator.py:163` (annotation dict literal), `tests/test_annotator.py:146` (phase4_fields assertion list), and `tests/test_integration_full_generation.py:158` (phase4_fields assertion list). This schema field is a **frozen Phase 4 contract** (D-15 R-P4 schema, asserted by Plan 04-04's `tests/test_annotator.py::TestAnnotateShape::test_phase4_fields_filled` — all pass today). Renaming the field would: (a) break Plan 04-04's R-X6 closure; (b) cascade into Wave 3 (writer.py's sample.json canonicalization), Wave 4 (api.py orchestration), Wave 5 (determinism goldens — sample.json bytes identity depends on this key name); (c) invalidate 05-CONTEXT D-13's manifest.jsonl entry shape (`{"musicality_score": 0.87, ...}`). That cascade is Rule 4 architectural territory, not a Rule 1/2 scope.
- **Fix:** Disambiguated the plan's grep invariant semantically: module-name references (`import musicality_score`, `musicality_score.get_musicality_score`, `\`\`musicality_score.get_musicality_score\`\``) were rewritten; JSON schema field keys (`"musicality_score"` in dict literals / field assertion lists) were preserved. Every remaining grep hit either (i) lives inside the moved file itself (explicit plan carveout), (ii) uses the already-rewritten `musicgen.musicality.get_musicality_score` path and matches only via the `_score` suffix substring, or (iii) is the R-P4 JSON field key. Zero surviving syntactic module references.
- **Files modified:** None (the decision is NOT to modify these 4 files' schema field references).
- **Verification:** Visual audit of all 9 non-moved-file grep hits (see the `Self-Check` section below for the tabular breakdown). `.venv/bin/python -c "from musicgen.musicality import get_musicality_score"` exits 0; `.venv/bin/python -c "import musicality_score"` raises `ModuleNotFoundError`; full test suite 634 passed.
- **Committed in:** (no code change — decision recorded here and in `patterns-established`).

---

**Total deviations:** 2 auto-fixed (both Rule 2: 1 scope-extension like 03-02 precedent, 1 scope-clarification disambiguating module ref vs. schema field).
**Impact on plan:** No scope creep — the scope extension (deviation 1) is exactly the 03-02 precedent the plan explicitly referenced. The scope clarification (deviation 2) prevented unbudgeted architectural churn that would have cascaded into Wave 3/4/5 contracts. Net result: 1 commit, 4 files, 100% rename preserved, all tests pass, zero syntactic module references to `musicality_score` remain anywhere in the tree.

## Issues Encountered

- **Pre-session file mode change on `musicality_score.py`:** `git diff` against HEAD showed a `100644 → 100755` mode change on the source file before this session began (chmod +x applied externally; zero content diff). The `git mv` operation unified the rename with the mode change in one commit (`48f71ac` reports `mode change 100644 => 100755` alongside the 100% rename). This is cosmetic; the moved file is now executable but the content is byte-identical. No action needed. Documented for traceability.
- **Pre-existing untracked items in working tree:** `git status` shows `?? .claude/` (Claude Code cache dir), `?? 20260419*/` (8 timestamped dirs — likely output artifacts from ad-hoc `python music_gen.py` runs before the dataset layout landed), `?? musicgen.txt` (unknown pre-existing file). All pre-date this session. Out of scope — these are session artifacts / user scratch, not produced or touched by this plan.
- **Pre-existing `.venv`-adjacent modifications:** `git status` shows `M` on `.continue-here.md`, `.gitignore`, `LICENSE`, `*.json` config files, `beat_roll_patterns_*.txt`, etc. — all pre-existing local user edits, unrelated to 05-03. Not staged, not committed. Out of scope per plan boundaries.

## User Setup Required

None — pure internal refactor, no external services touched, no dependencies added, no new env vars.

## Next Phase Readiness

- **Plan 05-04 (Wave 3 — writer.py / manifest.py) unblocked:** writer.py doesn't directly import `musicality`, but its sibling `api.py` in Wave 4 does — having `musicgen.musicality` in place before Wave 4 starts keeps the Wave 3 → Wave 4 handoff clean (05-CONTEXT's stated rationale for placing the move in its own Wave 2 plan).
- **Plan 05-05 (Wave 4 — api.py) has a resolvable import line:** `from musicgen import renderer, mixer, annotator, beats, writer, musicality` (05-PATTERNS.md:273) — every module in that line now exists. The temporary `from musicgen import musicality` added to `music_gen.py:3` in this plan is explicitly scheduled for deletion when Wave 4 deletes the entire `create_song` body (05-CONTEXT D-34).
- **Phase 5 Wave 1 now architecturally complete:** seeds.py (Plan 05-02) + musicality.py (this plan) land together as the Phase 5 Wave 1 RNG+musicality foundation. Wave 2 (writer + manifest, Plan 05-04) can now proceed without any Wave 1 blockers.
- **No blockers for Phase 5 Wave 2.** Baseline test count (634) preserved / advanced; no new imports added; no behavior change; R-P4 schema contract intact.

## Self-Check

- [x] `test -f src/musicgen/musicality.py` → PASS
- [x] `test ! -f musicality_score.py` → PASS
- [x] `git diff --cached --name-status -M HEAD~1..HEAD -- musicality_score.py src/musicgen/musicality.py` shows `R100` — verified by commit output `rename musicality_score.py => src/musicgen/musicality.py (100%)`
- [x] `.venv/bin/python -c "from musicgen.musicality import get_musicality_score; print('ok:', get_musicality_score.__name__)"` → `ok: get_musicality_score`
- [x] `.venv/bin/python -c "import musicality_score"` → `ModuleNotFoundError: No module named 'musicality_score'` (D-03 satisfied)
- [x] `.venv/bin/python -c "import music_gen"` → exits 0 (with expected FluidSynth-absent WARNING from renderer per Plan 04-05 D-07)
- [x] `grep -c "from musicgen import musicality" music_gen.py` → 1
- [x] `grep -c "musicality.get_musicality_score" music_gen.py` → 1
- [x] `git log --follow -- src/musicgen/musicality.py | wc -l` → 3 commits (`48f71ac` this plan → `350bf60` Plan 01-03 narrow exceptions → `94c19a0` initial upload) — rename history preserved
- [x] Task 1 committed as `48f71ac` (refactor); all 4 files (1 renamed + 3 edited) folded into the one task commit
- [x] Full test suite: `.venv/bin/pytest tests/ -q` → `634 passed, 6 skipped, 1 xfailed, 2 warnings in 1.53s` — zero regressions vs Plan 05-02's 633-passed baseline; +1 from AST guard auto-picking up `src/musicgen/musicality.py`
- [x] AST guard (`tests/test_no_bare_random_in_package.py`) PASSED on `src/musicgen/musicality.py` — zero bare stdlib `random.*` calls in the moved file (uses numpy.random transitively via librosa, not stdlib random)
- [x] Remaining grep hits classified (12 total):
  - 3 inside `src/musicgen/musicality.py` (moved file internal refs — allowed by plan carveout): lines 243 (def), 250 (CLI usage string), 254 (call)
  - 3 already using the new path (match only via `_score` suffix substring): `music_gen.py:121`, `src/musicgen/annotator.py:110`, `src/musicgen/seeds.py:105` all read `musicgen.musicality.get_musicality_score` or `musicality.get_musicality_score`
  - 2 function-name substring matches: `src/musicgen/seeds.py:115` and `tests/test_integration_full_generation.py:17` — the token `get_musicality_score` contains `_score` but is a function name, not a module reference
  - 4 R-P4 JSON schema field key references (PRESERVED as frozen Phase 4 contract): `src/musicgen/annotator.py:11`, `src/musicgen/annotator.py:163`, `tests/test_annotator.py:146`, `tests/test_integration_full_generation.py:158`
  - **Zero remaining syntactic module references** to `musicality_score` as a module name

## Self-Check: PASSED

---
*Phase: 05-productize-i-writer-manifest-seeds-determinism*
*Completed: 2026-04-19*
