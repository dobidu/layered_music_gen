---
phase: 02-stabilize-ii-config-time-signature-registry-logging
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - config.py
  - music_gen.py
  - tests/test_config.py
  - tests/test_timesig_registry.py
  - tests/test_music_gen_logging.py
autonomous: true
requirements:
  - R-S5
  - R-S9
tags: [config, paths, refactor, python]

must_haves:
  truths:
    - "`config.py` exists at repo root and owns every path literal previously embedded in music_gen.py."
    - "Running the existing pinned test suite (`tests/test_time_signature.py`, `tests/test_duration_validator.py`) stays green — the 95 Phase 1 tests are not broken."
    - "No path literal for `sf/<layer>/`, `*_fx.json`, `inst_probabilities.json`, `levels.json`, `song_structures.json`, `chord_patterns.txt`, or `beat_roll_patterns_<NN>.txt` remains in `music_gen.py` outside of import lines."
    - "`Config.load(cli_overrides={'sf_dir': ...})` returns a Config with the CLI override applied, proving D-01/D-02 precedence (CLI > env > defaults) works."
    - "On config load, each soundfont layer directory (`sf/beat`, `sf/melody`, `sf/harmony`, `sf/bassline`) is probed and a `WARNING` is logged when the `.sf2` count is < 3 (D-09)."
    - "`import music_gen` still does not trigger generation (Plan 01-01 importability property preserved)."
  artifacts:
    - path: "config.py"
      provides: "Config dataclass, DEFAULT_* constants, load() classmethod, _emit_soundfont_pool_report()"
      contains: "class Config"
      contains_also: "def load"
      contains_also2: "SOUNDFONT_POOL_WARN_THRESHOLD"
    - path: "tests/test_config.py"
      provides: "Unit tests for config defaults, env/CLI override precedence, and soundfont pool warning"
      contains: "test_cli_overrides_take_precedence_over_env"
      contains_also: "test_soundfont_pool_warning_fires_below_threshold"
    - path: "tests/test_timesig_registry.py"
      provides: "Wave 0 skeleton (empty import guard) for Plan 02 to populate"
    - path: "tests/test_music_gen_logging.py"
      provides: "Wave 0 skeleton (empty import guard) for Plan 03 to populate"
    - path: "music_gen.py"
      provides: "Path-literal-free orchestration; Config is loaded inside the __main__ guard and threaded through generate_song → create_song → mix_and_save"
  key_links:
    - from: "music_gen.py generate_song(id, cfg)"
      to: "config.Config.load()"
      via: "cfg instance passed as second argument"
      pattern: "generate_song\\(.*cfg"
    - from: "music_gen.py mix_and_save(...)"
      to: "config.Config.sf_layer_dir / fx_files / levels_file / inst_probabilities_file"
      via: "cfg.sf_layer_dir('beat') etc. replace os.path.join('sf','beat') and friends"
      pattern: "cfg\\.sf_layer_dir|cfg\\.fx_files|cfg\\.levels_file|cfg\\.inst_probabilities_file"
    - from: "music_gen.py generate_beat(...)"
      to: "config.Config.beat_pattern_file(time_signature)"
      via: "config lookup replaces the inline dict of six beat_roll_patterns_*.txt literals"
      pattern: "cfg\\.beat_pattern_file"
    - from: "config.Config.load()"
      to: "config._emit_soundfont_pool_report()"
      via: "called at end of load() before returning cfg"
      pattern: "_emit_soundfont_pool_report"
---

<objective>
Build the single source of truth for filesystem paths and the D-01/D-02 override layers so every downstream Phase 2 task — and the Phase 3 extraction that follows — can import paths instead of hardcoding them. This plan also lands the three Wave 0 test skeletons that Plans 02 and 03 will populate, and lands the R-S9 soundfont pool detection hook at config-load time.

Purpose: R-S5 (config centralization) and R-S9 (soundfont pool detection) are phase exit criteria; they must close before the time-signature registry or the logging migration can land, because both consume `Config` and the logger attached to `config.py` must already exist.

Output: `config.py` module, `tests/test_config.py` (populated), `tests/test_timesig_registry.py` (skeleton), `tests/test_music_gen_logging.py` (skeleton), and `music_gen.py` with every path literal replaced by a `cfg.*` accessor and Config plumbed through `generate_song(id, cfg)` → `create_song(..., cfg)` → `mix_and_save(..., cfg)`.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/ROADMAP.md
@.planning/phases/02-stabilize-ii-config-time-signature-registry-logging/02-CONTEXT.md
@.planning/phases/02-stabilize-ii-config-time-signature-registry-logging/02-RESEARCH.md
@.planning/phases/02-stabilize-ii-config-time-signature-registry-logging/02-VALIDATION.md
@.planning/codebase/STRUCTURE.md
@.planning/codebase/CONVENTIONS.md
@musicality_score.py
@tests/conftest.py
@music_gen.py

<interfaces>
<!-- Key reference shapes the executor needs. Do not explore; use these directly. -->

Plan 01-01 hard-won property (Phase 1 Plan 01-01 SUMMARY):
  `import music_gen` must remain side-effect-free. NEVER call `logging.basicConfig`
  at module import time in any new module. The `if __name__ == "__main__":` guard
  lives at music_gen.py:1170.

Existing path literals in music_gen.py that MUST move to config.py (verified in
02-RESEARCH.md §Path Literal Inventory):
  music_gen.py:516-521  — six beat_roll_patterns_<NN>.txt files (dict inside generate_beat)
  music_gen.py:611      — 'song_structures.json' as default arg of generate_song_arrangement
  music_gen.py:773-776  — os.path.join('sf','beat'|'melody'|'harmony'|'bassline')
  music_gen.py:785-788  — beat_fx.json / melody_fx.json / harmony_fx.json / bassline_fx.json
  music_gen.py:798      — 'inst_probabilities.json'
  music_gen.py:800      — 'levels.json'
  music_gen.py:1162     — 'chord_patterns.txt' passed to create_song as chord_pat_file

Existing music_gen.py call chain (pre-refactor):
  if __name__ == "__main__":                     # line 1170
      for i in range(1):
          generate_song(i)                       # line 1128-1166 → no cfg arg today
  def generate_song(id: int):                    # line 1128
      ...
      song_info = create_song(
          key=..., tempo=..., song_signatures=...,
          measures=..., name=song_name,
          chord_pat_file='chord_patterns.txt',   # line 1162 — literal
          swing_amount=swing_amount,
      )
  def create_song(..., chord_pat_file: str, swing_amount: float):  # line 1014
      ...
      # calls mix_and_save(name, beat_filename, melo_filename, harm_filename,
      #                    bass_filename, song_unique_parts, song_arrangement)
  def mix_and_save(name, beat_filename, melo_filename, harm_filename,
                   bass_filename, song_unique_parts, song_arrangement):  # line 754

Registry pattern (from musicality_score.py:14 — correct pattern):
  import logging
  logger = logging.getLogger(__name__)   # module-level, NOT instance-level,
                                         # NOT basicConfig anywhere outside __main__

Existing tests/conftest.py (Plan 01-04 scaffolding):
  Adds repo root to sys.path so `import config`, `import timesig`, and
  `import music_gen` all resolve when running pytest from repo root. Do NOT
  modify conftest.py in this plan.
</interfaces>
</context>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| env var → Config.load() | `MUSICGEN_SF_DIR` / `MUSICGEN_PROJECT_ROOT` come from process environment — technically user-controlled when a human runs the script. |
| filesystem → `_emit_soundfont_pool_report()` | `os.listdir(sf_layer_dir)` reads directory contents; results are only logged, never shelled out. |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-02-01 | Tampering | `Config.load()` env var ingestion | mitigate | Normalize incoming path values with `os.path.abspath()` before storing. Never pass the resulting path to `os.system`, `subprocess(shell=True)`, or `eval`. Only `os.path.join` and `os.listdir` consume them — both are shell-safe. |
| T-02-02 | Denial of Service | `os.listdir(sf_layer_dir)` on missing / permission-denied directory | mitigate | Wrap `os.listdir` in `try/except (FileNotFoundError, PermissionError)`; log a `WARNING` and continue. Never raise out of `_emit_soundfont_pool_report`. |
| T-02-03 | Information Disclosure | Soundfont filenames and counts logged at INFO level | accept | Soundfont names are non-sensitive repo assets (shipped in `sf/`); logging them aids operator debugging and matches D-09 intent. |
</threat_model>

<tasks>

<task type="auto">
  <name>Task 1 (Wave 0): Create the three new test files with minimal runnable skeletons</name>
  <files>tests/test_config.py, tests/test_timesig_registry.py, tests/test_music_gen_logging.py</files>
  <read_first>
    - tests/conftest.py (to confirm sys.path shim is in place)
    - tests/test_time_signature.py (for the project's existing pytest style)
    - tests/test_duration_validator.py (for the class-based test organisation convention)
    - .planning/phases/02-stabilize-ii-config-time-signature-registry-logging/02-RESEARCH.md §Test Migration and §Validation Architecture (the exact test names that must exist)
  </read_first>
  <action>
    Create three new test files. DO NOT modify `tests/conftest.py` — it already adds the repo root to `sys.path` (Plan 01-04 scaffolding).

    File 1 — `tests/test_config.py` (populate fully in this plan):

    ```python
    """
    Unit tests for config.py — R-S5 (config centralization) and R-S9 (soundfont pool detection).

    Covers:
    - Default paths match the legacy file layout (four sf layers, four fx files,
      six beat_roll_patterns files, inst_probabilities / levels / song_structures /
      chord_patterns).
    - Env-var override precedence (MUSICGEN_SF_DIR, MUSICGEN_PROJECT_ROOT).
    - CLI-overrides-take-precedence-over-env (D-01/D-02).
    - Soundfont pool WARNING fires below threshold (<3 .sf2 files per layer).
    - Soundfont pool WARNING does NOT fire at/above threshold.
    - Missing layer directory emits a WARNING (not a crash).
    """
    import logging
    import os

    import pytest

    import config


    class TestConfigDefaults:
        @pytest.mark.parametrize("layer", ["beat", "melody", "harmony", "bassline"])
        def test_default_sf_layer_dir_ends_with_layer_name(self, layer):
            cfg = config.Config()
            assert cfg.sf_layer_dir(layer).endswith(os.path.join("sf", layer))

        def test_default_fx_files_has_four_layers(self):
            cfg = config.Config()
            assert set(cfg.fx_files.keys()) == {"beat", "melody", "harmony", "bassline"}

        @pytest.mark.parametrize(
            "time_signature,expected_suffix",
            [
                ("2/4",  "beat_roll_patterns_24.txt"),
                ("3/4",  "beat_roll_patterns_34.txt"),
                ("4/4",  "beat_roll_patterns_44.txt"),
                ("6/8",  "beat_roll_patterns_68.txt"),
                ("7/8",  "beat_roll_patterns_78.txt"),
                ("12/8", "beat_roll_patterns_128.txt"),
            ],
        )
        def test_beat_pattern_file_per_signature(self, time_signature, expected_suffix):
            cfg = config.Config()
            assert cfg.beat_pattern_file(time_signature).endswith(expected_suffix)

        def test_default_levels_file_name(self):
            cfg = config.Config()
            assert cfg.levels_file.endswith("levels.json")

        def test_default_song_structures_file_name(self):
            cfg = config.Config()
            assert cfg.song_structures_file.endswith("song_structures.json")

        def test_default_inst_probabilities_file_name(self):
            cfg = config.Config()
            assert cfg.inst_probabilities_file.endswith("inst_probabilities.json")

        def test_default_chord_patterns_file_name(self):
            cfg = config.Config()
            assert cfg.chord_patterns_file.endswith("chord_patterns.txt")


    class TestOverridePrecedence:
        def test_env_override_applies_to_sf_dir(self, monkeypatch, tmp_path):
            fake = tmp_path / "fake_sf"
            fake.mkdir()
            monkeypatch.setenv("MUSICGEN_SF_DIR", str(fake))
            cfg = config.Config.load()
            assert cfg.sf_dir == str(fake)

        def test_cli_overrides_take_precedence_over_env(self, monkeypatch, tmp_path):
            env_path = tmp_path / "env_sf"
            env_path.mkdir()
            cli_path = tmp_path / "cli_sf"
            cli_path.mkdir()
            monkeypatch.setenv("MUSICGEN_SF_DIR", str(env_path))
            cfg = config.Config.load(cli_overrides={"sf_dir": str(cli_path)})
            assert cfg.sf_dir == str(cli_path)

        def test_load_without_overrides_returns_defaults(self, monkeypatch):
            monkeypatch.delenv("MUSICGEN_SF_DIR", raising=False)
            monkeypatch.delenv("MUSICGEN_PROJECT_ROOT", raising=False)
            cfg = config.Config.load()
            assert cfg.sf_dir == config.DEFAULT_SF_DIR


    class TestSoundfontPoolDetection:
        def _make_sf_tree(self, root, counts):
            """counts: dict {layer_name: int number of .sf2 files to create}."""
            for layer, n in counts.items():
                layer_dir = root / layer
                layer_dir.mkdir(parents=True, exist_ok=True)
                for i in range(n):
                    (layer_dir / f"{layer}_{i}.sf2").write_bytes(b"")
            return root

        def test_soundfont_pool_warning_fires_below_threshold(self, tmp_path, caplog):
            root = self._make_sf_tree(tmp_path, {"beat": 2, "melody": 2, "harmony": 2, "bassline": 2})
            cfg = config.Config.load(cli_overrides={"sf_dir": str(root)})
            with caplog.at_level(logging.WARNING, logger="config"):
                cfg._emit_soundfont_pool_report()
            warning_lines = [r for r in caplog.records if r.levelno == logging.WARNING]
            assert len(warning_lines) >= 4  # one per layer

        def test_soundfont_pool_no_warning_at_threshold(self, tmp_path, caplog):
            root = self._make_sf_tree(tmp_path, {"beat": 3, "melody": 3, "harmony": 3, "bassline": 3})
            cfg = config.Config.load(cli_overrides={"sf_dir": str(root)})
            caplog.clear()
            with caplog.at_level(logging.WARNING, logger="config"):
                cfg._emit_soundfont_pool_report()
            warning_lines = [r for r in caplog.records if r.levelno == logging.WARNING]
            assert warning_lines == []

        def test_soundfont_pool_missing_layer_logs_warning(self, tmp_path, caplog):
            root = tmp_path / "empty_sf"
            root.mkdir()
            # no layer subdirectories created
            cfg = config.Config.load(cli_overrides={"sf_dir": str(root)})
            with caplog.at_level(logging.WARNING, logger="config"):
                cfg._emit_soundfont_pool_report()
            messages = " ".join(r.message for r in caplog.records)
            assert "missing" in messages.lower() or "not found" in messages.lower() or "does not exist" in messages.lower()


    class TestImportSideEffects:
        def test_importing_config_does_not_emit_warnings_at_module_load(self):
            # Regression guard: `import config` itself MUST NOT call Config.load(),
            # because load() runs _emit_soundfont_pool_report which does os.listdir.
            # Phase 1 Plan 01-01 property: imports must be side-effect-free.
            import importlib
            import config as cfg_module
            importlib.reload(cfg_module)
            # If importing triggered load(), the soundfont pool report would fire
            # and log warnings. Verify nothing emitted by checking the module has
            # a Config class but no module-level Config instance.
            assert hasattr(cfg_module, "Config")
            assert not any(
                isinstance(getattr(cfg_module, name, None), cfg_module.Config)
                for name in dir(cfg_module)
            )
    ```

    File 2 — `tests/test_timesig_registry.py` (Wave 0 skeleton — do NOT populate assertions here; Plan 02 will):

    ```python
    """
    Wave 0 skeleton for Plan 02 (timesig.py TimeSignatureRegistry).

    This file exists so Plan 02 can populate it without creating a new file.
    Plan 02 will add the full assertion matrix described in
    02-RESEARCH.md §Test Migration.
    """
    import pytest


    @pytest.mark.skip(reason="Plan 02 will populate timesig registry tests")
    def test_placeholder():
        pass
    ```

    File 3 — `tests/test_music_gen_logging.py` (Wave 0 skeleton — do NOT populate AST assertions here; Plan 03 will):

    ```python
    """
    Wave 0 skeleton for Plan 03 (print → logging migration in music_gen.py).

    This file exists so Plan 03 can populate it without creating a new file.
    Plan 03 will add the AST-based test that asserts zero print() calls remain
    in music_gen.py and the import-side-effect-free regression guard.
    """
    import pytest


    @pytest.mark.skip(reason="Plan 03 will populate music_gen logging tests")
    def test_placeholder():
        pass
    ```

    Before running any test command, confirm `config.py` does NOT yet exist — the full test suite will fail to collect `test_config.py` until Task 2 lands. That is expected and acceptable at this intermediate step; the acceptance check for this task is limited to file existence and syntax.
  </action>
  <verify>
    <automated>.venv/bin/python -c "import ast; ast.parse(open('tests/test_config.py').read()); ast.parse(open('tests/test_timesig_registry.py').read()); ast.parse(open('tests/test_music_gen_logging.py').read()); print('OK')"</automated>
  </verify>
  <acceptance_criteria>
    - `ls tests/test_config.py tests/test_timesig_registry.py tests/test_music_gen_logging.py` lists all three files.
    - `grep -c "test_cli_overrides_take_precedence_over_env" tests/test_config.py` returns at least `1`.
    - `grep -c "test_soundfont_pool_warning_fires_below_threshold" tests/test_config.py` returns at least `1`.
    - `grep -c "test_soundfont_pool_no_warning_at_threshold" tests/test_config.py` returns at least `1`.
    - `grep -c "test_soundfont_pool_missing_layer_logs_warning" tests/test_config.py` returns at least `1`.
    - `grep -c "@pytest.mark.skip" tests/test_timesig_registry.py` returns at least `1`.
    - `grep -c "@pytest.mark.skip" tests/test_music_gen_logging.py` returns at least `1`.
    - `.venv/bin/python -m pytest tests/test_time_signature.py tests/test_duration_validator.py -q` still returns 0 (Phase 1 pinned tests unaffected).
  </acceptance_criteria>
  <done>All three test files exist, parse as valid Python, contain the required test names, and existing Phase 1 pinned tests still pass.</done>
</task>

<task type="auto">
  <name>Task 2: Create config.py with Config dataclass, three-layer precedence load(), and soundfont pool probe</name>
  <files>config.py</files>
  <read_first>
    - musicality_score.py (canonical pattern for `logger = logging.getLogger(__name__)`)
    - enhanced_duration_validator.py (dataclass conventions; logger usage)
    - .planning/phases/02-stabilize-ii-config-time-signature-registry-logging/02-RESEARCH.md §Pattern 2 (full config.py skeleton, lines 229-326) and §Pitfall 2 (basicConfig placement) and §CLI Override Scaffolding
    - .planning/phases/02-stabilize-ii-config-time-signature-registry-logging/02-CONTEXT.md (D-01 through D-09 in full)
  </read_first>
  <action>
    Create a new file `config.py` at the repo root (NOT inside `src/` — Phase 3 will move it later). Exact requirements:

    1. **Module header** — imports only. Module-level logger. NO `logging.basicConfig` anywhere (Pitfall 2).

    ```python
    """Config module for musicgen — owns all filesystem paths and override layers.

    D-01/D-02: three-layer precedence — CLI args > env vars > hardcoded defaults.
    D-03: wraps existing JSON files; no new config file format.
    D-09: soundfont pool detection fires at config load time (informational only).

    Phase 2 (R-S5, R-S9). Phase 6 will populate `cli_overrides` from the typer CLI.
    """
    import json
    import logging
    import os
    from dataclasses import dataclass, field
    from typing import Dict, Optional, Tuple

    logger = logging.getLogger(__name__)
    ```

    2. **Module-level constants** — the "hardcoded defaults" layer per D-02.

    ```python
    DEFAULT_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    DEFAULT_SF_DIR = os.path.join(DEFAULT_PROJECT_ROOT, "sf")
    DEFAULT_SF_LAYERS: Tuple[str, ...] = ("beat", "melody", "harmony", "bassline")

    DEFAULT_FX_FILES: Dict[str, str] = {
        "beat":     os.path.join(DEFAULT_PROJECT_ROOT, "beat_fx.json"),
        "melody":   os.path.join(DEFAULT_PROJECT_ROOT, "melody_fx.json"),
        "harmony":  os.path.join(DEFAULT_PROJECT_ROOT, "harmony_fx.json"),
        "bassline": os.path.join(DEFAULT_PROJECT_ROOT, "bassline_fx.json"),
    }

    DEFAULT_INST_PROBABILITIES_FILE = os.path.join(DEFAULT_PROJECT_ROOT, "inst_probabilities.json")
    DEFAULT_LEVELS_FILE              = os.path.join(DEFAULT_PROJECT_ROOT, "levels.json")
    DEFAULT_SONG_STRUCTURES_FILE     = os.path.join(DEFAULT_PROJECT_ROOT, "song_structures.json")
    DEFAULT_CHORD_PATTERNS_FILE      = os.path.join(DEFAULT_PROJECT_ROOT, "chord_patterns.txt")

    DEFAULT_BEAT_ROLL_PATTERN_FILES: Dict[str, str] = {
        "2/4":  os.path.join(DEFAULT_PROJECT_ROOT, "beat_roll_patterns_24.txt"),
        "3/4":  os.path.join(DEFAULT_PROJECT_ROOT, "beat_roll_patterns_34.txt"),
        "4/4":  os.path.join(DEFAULT_PROJECT_ROOT, "beat_roll_patterns_44.txt"),
        "6/8":  os.path.join(DEFAULT_PROJECT_ROOT, "beat_roll_patterns_68.txt"),
        "7/8":  os.path.join(DEFAULT_PROJECT_ROOT, "beat_roll_patterns_78.txt"),
        "12/8": os.path.join(DEFAULT_PROJECT_ROOT, "beat_roll_patterns_128.txt"),
    }

    SOUNDFONT_POOL_WARN_THRESHOLD = 3  # D-09: warn when a layer has fewer than 3 .sf2 files
    ```

    3. **`Config` dataclass** — one field per deliverable path, helper accessors, no side-effect-on-construction.

    ```python
    @dataclass
    class Config:
        project_root: str = DEFAULT_PROJECT_ROOT
        sf_dir: str = DEFAULT_SF_DIR
        sf_layers: Tuple[str, ...] = DEFAULT_SF_LAYERS
        fx_files: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_FX_FILES))
        inst_probabilities_file: str = DEFAULT_INST_PROBABILITIES_FILE
        levels_file: str = DEFAULT_LEVELS_FILE
        song_structures_file: str = DEFAULT_SONG_STRUCTURES_FILE
        chord_patterns_file: str = DEFAULT_CHORD_PATTERNS_FILE
        beat_roll_pattern_files: Dict[str, str] = field(
            default_factory=lambda: dict(DEFAULT_BEAT_ROLL_PATTERN_FILES)
        )

        def sf_layer_dir(self, layer: str) -> str:
            """Return the on-disk directory for a single soundfont layer."""
            return os.path.join(self.sf_dir, layer)

        def beat_pattern_file(self, time_signature: str) -> str:
            """Return the beat-roll pattern file path for a time signature string like '4/4'."""
            return self.beat_roll_pattern_files[time_signature]

        @classmethod
        def load(cls, cli_overrides: Optional[Dict[str, object]] = None) -> "Config":
            """Load Config applying D-01/D-02 precedence: CLI > env > defaults.

            Phase 2 callers pass cli_overrides=None; only env + defaults apply.
            Phase 6 typer CLI will build a dict from parsed args and pass it here.

            Fires the D-09 soundfont pool report before returning.
            """
            cfg = cls()

            # env-var layer (D-02 middle layer)
            sf_env = os.environ.get("MUSICGEN_SF_DIR")
            if sf_env:
                cfg.sf_dir = os.path.abspath(sf_env)  # T-02-01 mitigation: normalize path
            root_env = os.environ.get("MUSICGEN_PROJECT_ROOT")
            if root_env:
                cfg.project_root = os.path.abspath(root_env)

            # cli layer (D-02 top layer; framework-agnostic — avoids typer dep in Phase 2)
            if cli_overrides:
                for key, value in cli_overrides.items():
                    if value is None:
                        continue
                    if not hasattr(cfg, key):
                        continue
                    if isinstance(value, str) and key in ("sf_dir", "project_root"):
                        value = os.path.abspath(value)  # T-02-01 mitigation
                    setattr(cfg, key, value)

            cfg._emit_soundfont_pool_report()  # D-09
            return cfg

        def _emit_soundfont_pool_report(self) -> None:
            """Log soundfont counts per layer. WARNING when below D-09 threshold."""
            for layer in self.sf_layers:
                layer_dir = self.sf_layer_dir(layer)
                try:
                    entries = os.listdir(layer_dir)
                except FileNotFoundError:
                    logger.warning("Soundfont layer directory missing: %s", layer_dir)
                    continue
                except PermissionError:
                    logger.warning(
                        "Soundfont layer directory permission denied: %s", layer_dir
                    )
                    continue
                count = sum(1 for e in entries if e.endswith(".sf2"))
                if count < SOUNDFONT_POOL_WARN_THRESHOLD:
                    logger.warning(
                        "Soundfont pool thin for layer %s: %d .sf2 files in %s "
                        "(expected >= %d)",
                        layer, count, layer_dir, SOUNDFONT_POOL_WARN_THRESHOLD,
                    )
                else:
                    logger.info(
                        "Soundfont pool for layer %s: %d .sf2 files", layer, count
                    )

        # --- JSON loader helpers (thin wrappers over existing files; D-03) ---

        def load_levels(self) -> dict:
            with open(self.levels_file) as f:
                return json.load(f)

        def load_inst_probabilities(self) -> dict:
            with open(self.inst_probabilities_file) as f:
                return json.load(f)

        def load_song_structures(self) -> dict:
            with open(self.song_structures_file) as f:
                return json.load(f)
    ```

    4. **Anti-patterns the executor must NOT introduce** (from 02-RESEARCH.md §Anti-Patterns and §Pitfalls):
       - Do NOT import `timesig` in this file (Pitfall 1 — circular import).
       - Do NOT call `Config.load()` at module scope (Pitfall 2 — side effect at import).
       - Do NOT add `logging.basicConfig(...)` anywhere in this file.
       - Do NOT add a `Config.from_argparse` or `Config.from_cli_args` method (02-RESEARCH.md §CLI Override Scaffolding — typer arrives in Phase 6, argparse is not the target).
       - Do NOT hoist any time-signature-specific knowledge (numerators, compound flags) into `config.py`. Beat-pattern files are indexed by the string `"4/4"` only because that is how music_gen.py already addresses them today.

    5. Final check: `.venv/bin/python -c "import config; print(config.Config().sf_layer_dir('beat'))"` must print a path ending with `sf/beat`.
  </action>
  <verify>
    <automated>.venv/bin/python -m pytest tests/test_config.py -x -q</automated>
  </verify>
  <acceptance_criteria>
    - `ls config.py` shows the file exists.
    - `.venv/bin/python -c "import config"` exits 0 with no output (no log lines emitted at import).
    - `grep -c "^class Config" config.py` returns `1`.
    - `grep -c "def load" config.py` returns at least `1`.
    - `grep -c "_emit_soundfont_pool_report" config.py` returns at least `2` (definition + call inside `load`).
    - `grep -c "SOUNDFONT_POOL_WARN_THRESHOLD = 3" config.py` returns `1`.
    - `grep -c "logging.basicConfig" config.py` returns `0` (Pitfall 2).
    - `grep -c "import timesig" config.py` returns `0` (Pitfall 1).
    - `.venv/bin/python -m pytest tests/test_config.py -x -q` exits 0 — all tests in `tests/test_config.py` pass.
    - `.venv/bin/python -m pytest tests/test_time_signature.py tests/test_duration_validator.py -q` exits 0 (Phase 1 pinned tests unaffected).
  </acceptance_criteria>
  <done>`config.py` exists, passes all `tests/test_config.py` assertions, emits no log lines at import time, and contains no circular-import or basicConfig anti-patterns.</done>
</task>

<task type="auto">
  <name>Task 3: Replace path literals in music_gen.py with Config accessors and thread cfg through generate_song → create_song → mix_and_save</name>
  <files>music_gen.py</files>
  <read_first>
    - config.py (the module created in Task 2 — know the exact accessor names)
    - music_gen.py lines 1-20 (import block)
    - music_gen.py lines 500-530 (generate_beat's beat_pattern_files dict at lines 516-521)
    - music_gen.py lines 609-648 (generate_song_arrangement default arg on line 611; its except block has a print on 646 — that print stays as-is for Plan 03 to convert)
    - music_gen.py lines 754-907 (mix_and_save — path literals on lines 773-776, 785-788, 798, 800)
    - music_gen.py lines 1014-1170 (create_song signature, generate_song, __main__ guard)
    - .planning/phases/02-stabilize-ii-config-time-signature-registry-logging/02-RESEARCH.md §Path Literal Inventory (the exact 18-site table) and §Soundfont Pool Layout (the "pass cfg through generate_song" recommendation) and §Pitfall 7 (generate_song_arrangement default arg)
  </read_first>
  <action>
    Thread a single `Config` instance through the call chain. Load Config ONCE inside the `if __name__ == "__main__":` guard. NEVER at module top — that would reintroduce the Plan 01-01 importability footgun.

    **Step 1 — imports.** Add a single line at the top of `music_gen.py` (after the existing stdlib/third-party imports, before `from typing import ...`):

    ```python
    import config
    ```

    Place it next to the other local imports (`import musicality_score`, `from enhanced_duration_validator import ...`).

    **Step 2 — `generate_song` signature.** Change line 1128 from:

    ```python
    def generate_song(id: int):
    ```

    to:

    ```python
    def generate_song(id: int, cfg: config.Config):
    ```

    Inside `generate_song` (line 1162), change the call to `create_song` — replace:

    ```python
    chord_pat_file='chord_patterns.txt',
    ```

    with:

    ```python
    chord_pat_file=cfg.chord_patterns_file,
    ```

    Also pass `cfg` to `create_song`:

    ```python
    song_info = create_song(
        key=key,
        tempo=tempo,
        song_signatures=signatures,
        measures=measures,
        name=song_name,
        chord_pat_file=cfg.chord_patterns_file,
        swing_amount=swing_amount,
        cfg=cfg,
    )
    ```

    **Step 3 — `create_song` signature.** At line 1014, add `cfg: config.Config` as a new keyword-only parameter. Search the body of `create_song` for calls to `mix_and_save(...)` and add `cfg=cfg` as the final keyword argument. Also pass `cfg` to `generate_song_arrangement(...)` if called from `create_song` — use `structures_file=cfg.song_structures_file`.

    **Step 4 — `mix_and_save` signature.** At line 754, add `cfg: config.Config` as a new keyword-only parameter at the end:

    ```python
    def mix_and_save(name, beat_filename, melo_filename, harm_filename,
                     bass_filename, song_unique_parts, song_arrangement, *,
                     cfg: config.Config):
    ```

    Then replace the hardcoded paths inside `mix_and_save`:

    | Line (old) | Old literal | New expression |
    |---|---|---|
    | 773 | `str(os.path.join('sf','beat'))` | `cfg.sf_layer_dir('beat')` |
    | 774 | `str(os.path.join('sf','melody'))` | `cfg.sf_layer_dir('melody')` |
    | 775 | `str(os.path.join('sf','harmony'))` | `cfg.sf_layer_dir('harmony')` |
    | 776 | `str(os.path.join('sf','bassline'))` | `cfg.sf_layer_dir('bassline')` |
    | 785 | `'beat_fx.json'` | `cfg.fx_files['beat']` |
    | 786 | `'melody_fx.json'` | `cfg.fx_files['melody']` |
    | 787 | `'harmony_fx.json'` | `cfg.fx_files['harmony']` |
    | 788 | `'bassline_fx.json'` | `cfg.fx_files['bassline']` |
    | 798 | `read_instrument_probabilities('inst_probabilities.json')` | `read_instrument_probabilities(cfg.inst_probabilities_file)` |
    | 800 | `get_levels('levels.json')` | `get_levels(cfg.levels_file)` |

    Delete the two `# TODO: configure soundfont directory` and `# TODO: gather all file references in a single config file` comments on lines 772 and 797 — they are resolved by this refactor. Leave the other TODO comments alone; they belong to later phases.

    **Step 5 — `generate_beat` beat-pattern dict (music_gen.py:516-521).** Replace the inline dict with a Config lookup. Open `music_gen.py` and find the `generate_beat` function. Locate the dict that maps time signature strings to `beat_roll_patterns_NN.txt` literals. Replace the six-entry dict literal with:

    ```python
    beat_pattern_files = dict(cfg.beat_roll_pattern_files)
    ```

    The `generate_beat` function needs to accept `cfg` — add `cfg: config.Config` to its signature as a trailing keyword-only parameter and update the call site inside `create_song` to pass `cfg=cfg`.

    **Step 6 — `generate_chord_progression` and any other generator that currently reads `chord_patterns.txt`.** If the function takes `pattern_file` as an argument (it does — verified at music_gen.py:168), that argument is already threaded through `create_song`'s `chord_pat_file` kwarg. No additional change is needed because Step 2 already swapped `'chord_patterns.txt'` → `cfg.chord_patterns_file`.

    **Step 7 — `generate_song_arrangement` default arg (Pitfall 7).** At line 611, change the signature from:

    ```python
    def generate_song_arrangement(structures_file: str = 'song_structures.json') -> Tuple[List[str], List[str]]:
    ```

    to:

    ```python
    def generate_song_arrangement(structures_file: Optional[str] = None) -> Tuple[List[str], List[str]]:
    ```

    And add, as the very first statement of the function body (before the `try:` block):

    ```python
    if structures_file is None:
        structures_file = config.DEFAULT_SONG_STRUCTURES_FILE
    ```

    Rationale: `generate_song_arrangement` is called from `create_song` in Phase 1's plan 01-01 refactor. The calling code should pass `cfg.song_structures_file` explicitly; the default is preserved as a safety net that points at the config module's default constant, NOT the literal string, satisfying the grep exit criterion.

    **Step 8 — `__main__` guard.** At line 1170, load Config once and pass it into `generate_song`:

    ```python
    if __name__ == "__main__":
        cfg = config.Config.load()
        for i in range(1):
            generate_song(i, cfg)
    ```

    Do NOT add `logging.basicConfig` in this task — Plan 03 owns that. Keep this task strictly about path migration.

    **Step 9 — verify imports still work.**

    ```bash
    .venv/bin/python -c "import music_gen; print('ok')"
    ```

    Expected output: `ok` with no other lines.

    **Anti-patterns the executor must NOT do:**
    - Do NOT call `config.Config.load()` at module scope in `music_gen.py`. Load inside the `__main__` guard ONLY.
    - Do NOT replace `os.path.join` calls that are about runtime-generated filenames (like `beat_wav = os.path.join(name, beat_wav)` at line 826) — those aren't config paths, they're output paths.
    - Do NOT touch any `print()` call in this task — Plan 03 owns the print→logging migration.
    - Do NOT delete the `# TODO: only render and mix the parts that are used in the song arrangement` comment at line 759 — that's a Phase 3+ concern.
    - Do NOT modify `enhanced_duration_validator.py` or `musicality_score.py` in this task.

    **After the refactor, all four grep commands below must return zero matches** (the ROADMAP exit criterion expanded per 02-RESEARCH.md §Path Literal Inventory):

    ```bash
    grep -nE "os\.path\.join\('sf'" music_gen.py                                # sf/ layer dirs
    grep -nE "(beat|melody|harmony|bassline)_fx\.json" music_gen.py             # fx json files
    grep -nE "beat_roll_patterns_(24|34|44|54|68|78|128)\.txt" music_gen.py     # beat roll files
    grep -nE "'inst_probabilities\.json'|'levels\.json'|'song_structures\.json'|'chord_patterns\.txt'" music_gen.py
    ```
  </action>
  <verify>
    <automated>.venv/bin/python -c "import music_gen; print('import-ok')" && .venv/bin/python -m pytest tests/ -q</automated>
  </verify>
  <acceptance_criteria>
    - `.venv/bin/python -c "import music_gen"` exits 0 and prints nothing (or only `import-ok` when run with the print). No "Generating song #0" output — the `__main__` guard must still work.
    - `grep -nE "os\\.path\\.join\\('sf'" music_gen.py` returns 0 lines.
    - `grep -nE "(beat\|melody\|harmony\|bassline)_fx\\.json" music_gen.py` returns 0 lines.
    - `grep -nE "beat_roll_patterns_(24\|34\|44\|54\|68\|78\|128)\\.txt" music_gen.py` returns 0 lines.
    - `grep -nE "'inst_probabilities\\.json'\|'levels\\.json'\|'song_structures\\.json'\|'chord_patterns\\.txt'" music_gen.py` returns 0 lines.
    - `grep -c "^import config$" music_gen.py` returns `1` (new top-level import).
    - `grep -c "cfg: config\\.Config" music_gen.py` returns at least `3` (generate_song, create_song, mix_and_save signatures).
    - `grep -c "cfg = config\\.Config\\.load()" music_gen.py` returns exactly `1` (the __main__ guard).
    - `grep -n "logging.basicConfig" music_gen.py` returns 0 matches (Plan 03 owns this).
    - `.venv/bin/python -m pytest tests/test_time_signature.py tests/test_duration_validator.py -q` exits 0 — 95 Phase 1 pinned tests still pass.
    - `.venv/bin/python -m pytest tests/test_config.py -q` exits 0 — config tests still pass.
    - `.venv/bin/python -m pytest tests/ -q` exits 0 overall.
  </acceptance_criteria>
  <done>music_gen.py no longer contains any of the 18 enumerated path literals, Config is loaded only inside `__main__`, cfg is threaded through generate_song → create_song → mix_and_save → generate_beat, the 95 Phase 1 tests plus all new config tests pass, and `import music_gen` remains side-effect-free.</done>
</task>

</tasks>

<verification>
Plan-level verification (run after all three tasks):

1. **Grep exit criterion (R-S5 main gate):**
   ```bash
   grep -nE "os\.path\.join\('sf'|(beat|melody|harmony|bassline)_fx\.json|beat_roll_patterns_(24|34|44|54|68|78|128)\.txt|'inst_probabilities\.json'|'levels\.json'|'song_structures\.json'|'chord_patterns\.txt'" music_gen.py
   ```
   Expected: zero hits.

2. **Importability regression (Plan 01-01 property preserved):**
   ```bash
   .venv/bin/python -c "import music_gen"
   ```
   Expected: exits 0, no log/print output.

3. **Full test suite green:**
   ```bash
   .venv/bin/python -m pytest tests/ -q
   ```
   Expected: 95 Phase 1 tests + ~12 new config tests all pass, under 10s total.

4. **Soundfont pool D-09 behavior (manual sanity):**
   ```bash
   .venv/bin/python -c "import logging; logging.basicConfig(level=logging.INFO); import config; cfg = config.Config.load()"
   ```
   Expected: WARNING lines for each sf/<layer>/ directory that has < 3 `.sf2` files (all four layers in this sandbox per 02-RESEARCH.md §Soundfont Pool Layout).
</verification>

<success_criteria>
- R-S5 closed: `config.py` owns every path; music_gen.py has zero enumerated literal.
- R-S9 closed: `Config.load()` emits layer-by-layer soundfont counts with WARNING on < 3.
- All 95 Phase 1 pinned tests still pass without modification.
- `tests/test_config.py` ~12 new tests all pass.
- `import music_gen` remains side-effect-free.
- Wave 0 test skeletons exist for Plans 02 and 03 to populate.
</success_criteria>

<output>
After completion, create `.planning/phases/02-stabilize-ii-config-time-signature-registry-logging/02-01-SUMMARY.md` documenting:
- The final config.py API (fields + methods exposed)
- Exact list of path literals removed and where they moved
- Soundfont pool report behavior verified in the sandbox (4 layers, N .sf2 each)
- Any assumption from 02-RESEARCH.md §Assumptions Log that was validated or invalidated during implementation
- The `cli_overrides` dict shape that Phase 6 will populate
</output>
