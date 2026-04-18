---
phase: 02-stabilize-ii-config-time-signature-registry-logging
plan: 01
subsystem: config
tags: [config, paths, refactor, python, r-s5, r-s9]

requires:
  - phase: 01-stabilize-i-bug-fixes-and-guardrails
    provides: "Plan 01-01 __main__ guard making music_gen.py side-effect-free on import"
  - phase: 01-stabilize-i-bug-fixes-and-guardrails
    provides: "Plan 01-04 pytest skeleton and conftest.py sys.path shim"

provides:
  - "config.py module with Config dataclass, DEFAULT_* constants, load() classmethod, _emit_soundfont_pool_report()"
  - "Three-layer override precedence: CLI > env vars > defaults (D-01/D-02)"
  - "R-S5 closed: zero path literals in music_gen.py — all 18 sites migrated to config accessors"
  - "R-S9 closed: Config.load() fires _emit_soundfont_pool_report() emitting layer counts and WARNINGs when < 3 .sf2 files"
  - "Wave 0 test skeletons (tests/test_timesig_registry.py, tests/test_music_gen_logging.py) for Plans 02 and 03"
  - "22 new tests in tests/test_config.py all passing"

affects:
  - "music_gen.py — generate_song, create_song, mix_and_save, generate_beat, generate_song_parts, generate_song_arrangement all accept cfg parameter"
  - "02-02 plan (timesig registry) — imports config.Config from this module"
  - "02-03 plan (logging migration) — imports config.Config from this module"

tech-stack:
  added: []
  patterns:
    - "Config dataclass with field() factory defaults for mutable dict fields"
    - "Three-layer override: cls() → env vars → cli_overrides dict (framework-agnostic)"
    - "os.path.abspath() normalization for user-controlled path inputs (T-02-01)"
    - "FileNotFoundError/PermissionError catch in _emit_soundfont_pool_report (T-02-02)"
    - "module-level logger = logging.getLogger(__name__) — no basicConfig at import"
    - "cfg threaded as parameter through generate_song → create_song → mix_and_save/generate_beat"

key-files:
  created:
    - config.py
    - tests/test_config.py
    - tests/test_timesig_registry.py
    - tests/test_music_gen_logging.py
  modified:
    - music_gen.py

key-decisions:
  - "Config threaded as explicit parameter through call chain (not module-level singleton) — preserves importability (Plan 01-01 property)"
  - "cfg parameters default to None with fallback to config.Config() inside functions — allows callers that don't have cfg to still work"
  - "generate_song_arrangement default arg changed from literal 'song_structures.json' to None + runtime fallback (Pitfall 7 mitigation)"
  - "T-02-01: os.path.abspath() applied to both env var and CLI path inputs"
  - "T-02-02: FileNotFoundError and PermissionError caught in _emit_soundfont_pool_report; never raises"
  - "Venv recreated (bin/ directory was missing from previous session) — installed all runtime deps"

metrics:
  duration: "~9 minutes"
  completed: "2026-04-18"
  tasks: 3
  files_created: 4
  files_modified: 1
  tests_added: 22
  tests_total: 117
  tests_skipped: 2
---

# Phase 02 Plan 01: Config Module and Path Migration Summary

**Config dataclass with three-layer D-01/D-02 override precedence centralizing all 18 path literals from music_gen.py and soundfont pool detection firing at load time**

## Performance

- **Duration:** ~9 minutes
- **Started:** 2026-04-18T15:47:29Z
- **Completed:** 2026-04-18
- **Tasks:** 3
- **Files created:** 4 (config.py + 3 test files)
- **Files modified:** 1 (music_gen.py)
- **Test runtime:** 0.74s for 117 tests (well under 10s R-Q2 budget)

## Accomplishments

- Landed `config.py` with `Config` dataclass, module-level `DEFAULT_*` constants, `load()` classmethod with CLI > env > defaults precedence, and `_emit_soundfont_pool_report()` D-09 hook
- Migrated all 18 enumerated path literals from `music_gen.py` to config accessors (R-S5 closed)
- Threaded `cfg: config.Config` through `generate_song` → `create_song` → `mix_and_save` and `generate_song_parts` → `generate_beat`
- Fixed `generate_song_arrangement` default arg from `'song_structures.json'` literal to `None` with `config.DEFAULT_SONG_STRUCTURES_FILE` runtime fallback (Pitfall 7 resolved)
- R-S9 closed: D-09 soundfont pool detection emits `WARNING` per layer when < 3 `.sf2` files; verified on live sandbox (all 4 layers at 0 files → 4 WARNINGs)
- Landed Wave 0 test skeletons for Plans 02 and 03
- 22 new passing tests in `tests/test_config.py`; all 95 Phase 1 pinned tests still pass

## Task Commits

1. **Task 1: Wave 0 test skeletons** — `67b2ce0` (test)
2. **Task 2: config.py** — `9828fca` (feat)
3. **Task 3: music_gen.py path migration** — `b89108e` (feat)

## Final Config.py API

### Module-level constants (DEFAULT_* layer)

| Constant | Value |
|----------|-------|
| `DEFAULT_PROJECT_ROOT` | `os.path.dirname(os.path.abspath(__file__))` |
| `DEFAULT_SF_DIR` | `{root}/sf` |
| `DEFAULT_SF_LAYERS` | `("beat", "melody", "harmony", "bassline")` |
| `DEFAULT_FX_FILES` | `{layer: "{root}/{layer}_fx.json"}` for all 4 layers |
| `DEFAULT_INST_PROBABILITIES_FILE` | `{root}/inst_probabilities.json` |
| `DEFAULT_LEVELS_FILE` | `{root}/levels.json` |
| `DEFAULT_SONG_STRUCTURES_FILE` | `{root}/song_structures.json` |
| `DEFAULT_CHORD_PATTERNS_FILE` | `{root}/chord_patterns.txt` |
| `DEFAULT_BEAT_ROLL_PATTERN_FILES` | `{sig: "{root}/beat_roll_patterns_{NN}.txt"}` for 6 sigs |
| `SOUNDFONT_POOL_WARN_THRESHOLD` | `3` |

### Config dataclass fields and methods

| API | Type | Description |
|-----|------|-------------|
| `Config.project_root` | `str` | Repo root path |
| `Config.sf_dir` | `str` | Soundfont base directory |
| `Config.sf_layers` | `Tuple[str,...]` | Layer names tuple |
| `Config.fx_files` | `Dict[str, str]` | Per-layer FX JSON paths |
| `Config.inst_probabilities_file` | `str` | Path to inst_probabilities.json |
| `Config.levels_file` | `str` | Path to levels.json |
| `Config.song_structures_file` | `str` | Path to song_structures.json |
| `Config.chord_patterns_file` | `str` | Path to chord_patterns.txt |
| `Config.beat_roll_pattern_files` | `Dict[str, str]` | Per-time-sig beat pattern paths |
| `Config.sf_layer_dir(layer)` | `str` | Returns `os.path.join(sf_dir, layer)` |
| `Config.beat_pattern_file(ts)` | `str` | Returns `beat_roll_pattern_files[ts]` |
| `Config.load(cli_overrides=None)` | `Config` | Three-layer precedence load |
| `Config._emit_soundfont_pool_report()` | `None` | D-09 pool check (never raises) |
| `Config.load_levels()` | `dict` | JSON wrapper for levels.json |
| `Config.load_inst_probabilities()` | `dict` | JSON wrapper for inst_probabilities.json |
| `Config.load_song_structures()` | `dict` | JSON wrapper for song_structures.json |

### cli_overrides shape for Phase 6

Phase 6 `typer` CLI will call:
```python
cfg = Config.load(cli_overrides={
    "sf_dir": "/path/to/sf",
    "project_root": "/path/to/repo",
    # Any Config field name as key, value as string or appropriate type
})
```
Keys must match `Config` dataclass field names exactly. Unknown keys are silently ignored (`hasattr` check in `load()`). Path fields `sf_dir` and `project_root` are normalized via `os.path.abspath()` (T-02-01).

## Path Literals Removed and Where They Moved

| music_gen.py site | Old literal | New config accessor |
|-------------------|-------------|---------------------|
| `generate_beat` (lines 515-522) | 6-entry inline dict of `beat_roll_patterns_NN.txt` | `dict(cfg.beat_roll_pattern_files)` |
| `generate_song_arrangement` default arg (line 611) | `'song_structures.json'` | `config.DEFAULT_SONG_STRUCTURES_FILE` (runtime fallback) |
| `mix_and_save` line 773 | `os.path.join('sf','beat')` | `cfg.sf_layer_dir('beat')` |
| `mix_and_save` line 774 | `os.path.join('sf','melody')` | `cfg.sf_layer_dir('melody')` |
| `mix_and_save` line 775 | `os.path.join('sf','harmony')` | `cfg.sf_layer_dir('harmony')` |
| `mix_and_save` line 776 | `os.path.join('sf','bassline')` | `cfg.sf_layer_dir('bassline')` |
| `mix_and_save` line 785 | `'beat_fx.json'` | `cfg.fx_files['beat']` |
| `mix_and_save` line 786 | `'melody_fx.json'` | `cfg.fx_files['melody']` |
| `mix_and_save` line 787 | `'harmony_fx.json'` | `cfg.fx_files['harmony']` |
| `mix_and_save` line 788 | `'bassline_fx.json'` | `cfg.fx_files['bassline']` |
| `mix_and_save` line 798 | `'inst_probabilities.json'` | `cfg.inst_probabilities_file` |
| `mix_and_save` line 800 | `'levels.json'` | `cfg.levels_file` |
| `generate_song` line 1162 | `'chord_patterns.txt'` | `cfg.chord_patterns_file` |

All 18 enumerated sites from 02-RESEARCH.md §Path Literal Inventory are resolved. The 4 grep exit criteria return zero hits.

## Soundfont Pool Report Behavior (D-09)

Verified on sandbox with 0 `.sf2` files per layer:

```
WARNING:config:Soundfont pool thin for layer beat: 0 .sf2 files in /home/bidu/musicgen/sf/beat (expected >= 3)
WARNING:config:Soundfont pool thin for layer melody: 0 .sf2 files in /home/bidu/musicgen/sf/melody (expected >= 3)
WARNING:config:Soundfont pool thin for layer harmony: 0 .sf2 files in /home/bidu/musicgen/sf/harmony (expected >= 3)
WARNING:config:Soundfont pool thin for layer bassline: 0 .sf2 files in /home/bidu/musicgen/sf/bassline (expected >= 3)
```

With >= 3 files per layer (tested via `tmp_path` in `tests/test_config.py`): no WARNING, INFO only. Missing directories (tested via `tmp_path`): WARNING with "missing" text. Neither case raises an exception.

## Assumptions Validated During Implementation

| Assumption | Status | Notes |
|------------|--------|-------|
| A1 (compound valid_durations set) | N/A for this plan | Config has no time-sig knowledge |
| A2 (basicConfig in __main__ doesn't break import check) | Validated | `import music_gen` still side-effect-free after adding `import config` |
| A5 (conftest.py sys.path shim covers new modules) | Validated | `import config` works in all tests via existing shim |
| A7 (generate_random_time_signature has no fallback return) | N/A for this plan | Pitfall 5 is Plan 02's concern |

## Deviations from Plan

### Environment Setup (Rule 3 - Blocking)

**1. Recreated .venv — bin/ directory was missing**
- **Found during:** Task 1 pre-verification
- **Issue:** `.venv/` existed with `lib/` and `include/` but no `bin/` directory, so `.venv/bin/python` was absent. The venv was created by a previous session but the executable symlinks were not present (likely a WSL filesystem interaction issue).
- **Fix:** Ran `python3 -m venv /home/bidu/musicgen/.venv --clear` to recreate the venv, then installed all runtime + test dependencies via `.venv/bin/pip install pytest pytest-cov music21 midiutil pydub midi2audio pedalboard python-json-logger librosa`.
- **Files modified:** None (venv is gitignored)
- **Committed in:** N/A — environment, not code.

### CRLF line endings in music_gen.py (cosmetic)
- **Found during:** Task 3 acceptance check for `grep -c "^import config$"`
- **Issue:** `music_gen.py` uses Windows CRLF line endings (`\r\n`). The grep pattern `^import config$` fails because the `$` matches before `\r`. Python ignores `\r` in source files so functionality is unaffected.
- **Fix:** Not fixed — changing line endings would produce a large noisy diff unrelated to the task. The `import config` line is present and functional (verified via `grep -n "import config" music_gen.py` and `python -c "import music_gen; print('ok')`).
- **Deferred:** Line ending normalization can be a one-time cleanup in a later plan or via `.gitattributes`.

## Known Stubs

None. All config accessors are fully wired. The Wave 0 skeleton tests (`test_timesig_registry.py`, `test_music_gen_logging.py`) are intentionally marked `@pytest.mark.skip` — they are skeleton files for Plans 02 and 03, not stubs that block this plan's goals.

## Threat Flags

None. No new network endpoints, auth paths, or file-write surfaces introduced. The `MUSICGEN_SF_DIR` env var is the only new user-controlled input; it flows through `os.path.abspath()` and only reaches `os.path.join` and `os.listdir` — both shell-safe.

## Self-Check: PASSED

- FOUND: config.py — `ls config.py` ✓
- FOUND: tests/test_config.py — `ls tests/test_config.py` ✓
- FOUND: tests/test_timesig_registry.py — `ls tests/test_timesig_registry.py` ✓
- FOUND: tests/test_music_gen_logging.py — `ls tests/test_music_gen_logging.py` ✓
- FOUND commit: 67b2ce0 (Task 1 test skeletons) ✓
- FOUND commit: 9828fca (Task 2 config.py) ✓
- FOUND commit: b89108e (Task 3 music_gen.py migration) ✓
- VERIFIED: `pytest tests/ -q` → 117 passed, 2 skipped in 0.74s ✓
- VERIFIED: R-S5 grep gate → zero hits ✓
- VERIFIED: `import music_gen` → side-effect-free ✓
- VERIFIED: D-09 soundfont pool detection → 4 WARNINGs on sandbox ✓

---
*Phase: 02-stabilize-ii-config-time-signature-registry-logging*
*Plan: 01*
*Completed: 2026-04-18*
