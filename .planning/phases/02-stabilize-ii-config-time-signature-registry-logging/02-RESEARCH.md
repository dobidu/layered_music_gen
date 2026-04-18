# Phase 2: Stabilize II — config + time-signature registry + logging - Research

**Researched:** 2026-04-10
**Domain:** Python refactoring — configuration centralization, domain registry, structured logging
**Confidence:** HIGH

## Summary

Phase 2 takes a 1172-line god file (`music_gen.py`) and extracts two new internal modules — `config.py` and `timesig.py` — while completing the `print → logging` migration. None of the three workstreams introduce new technology: `python-json-logger>=2.0.7` is already pinned in `requirements.txt` and installed in `.venv/`, and the logging / dataclass / `os.path.join` patterns are already in use elsewhere in the repo (`musicality_score.py`, `enhanced_duration_validator.py`). The risk surface is entirely about preserving the 95 pinned tests from Plan 01-04 and not leaking new imports that would reintroduce the "`import music_gen` triggers generation" footgun that Plan 01-01 just closed.

The phase's crux is the time-signature registry design (D-04..D-06). Seven supported signatures live in five different places today (`music_gen.py:22,42,54` and `:944-951`, plus `enhanced_duration_validator.py:39-87`) and tests pin exact outputs for all of them — including a cosmetic-if quirk in `verify_beat_pattern` that Phase 01-04 explicitly flagged as must-preserve. The registry must be pure data + thin delegation so the pinned tests pass unchanged with only their import line rewritten.

**Primary recommendation:** Build `timesig.py` as a single dataclass-per-signature registry keyed on the canonical string ("4/4", "6/8", ...), with the four existing module-level functions kept as thin wrappers in `music_gen.py` that delegate to `registry.lookup(ts).<method>()`. Build `config.py` as a plain module exposing constants (for hardcoded defaults) plus a `Config` dataclass with a `load()` classmethod that layers env vars over defaults (no CLI argparse yet — the hook point is documented but unused until Phase 6). Replace `print()` with `logger = logging.getLogger(__name__)` per module, matching `musicality_score.py`. No new dependencies.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Config Module Design**
- **D-01:** Config module must support three override layers with precedence: CLI args > environment variables > config file defaults. This lays the groundwork for the robust CLI coming in Phase 6.
- **D-02:** Precedence order is standard: CLI args override env vars, env vars override file defaults, file defaults override hardcoded defaults.
- **D-03:** The config module provides a unified access layer on top of the existing JSON files (`levels.json`, `inst_probabilities.json`, `song_structures.json`, `*_fx.json`, etc.). No new config file format introduced — the module wraps existing files with override capability.

**Time-Signature Registry Data Model**
- **D-04:** Registry entries are dataclasses containing fields like `valid_chord_lengths`, `beat_count`, `measure_multiplier`, `is_compound`, and any other per-signature metadata currently scattered across functions.
- **D-05:** The registry owns ALL validation logic. `verify_pattern_for_time_signature`, `verify_beat_pattern`, `calculate_measures_for_time_signature` either become thin wrappers delegating to the registry or are absorbed entirely. Adding a new time signature must touch exactly one location.
- **D-06:** Design for flexibility and precision — the system should make it easy to add unusual meters and handle compound vs. simple signatures cleanly.

**Logging Style**
- **D-07:** Log levels follow semantic differentiation:
  - `DEBUG` — internal state dumps (chord chosen, pattern selected, intermediate values)
  - `INFO` — progress milestones (mixing part N, song saved, generation started)
  - `WARNING` — recoverable oddities (soundfont pool thin, fallback path used)
  - `ERROR` — failures that affect output quality
- **D-08:** `python-json-logger` is already a dependency. JSON logging format is available but activation deferred to Phase 6 batch mode. Default format stays human-readable for interactive use.

**Soundfont Pool Detection**
- **D-09:** Soundfont pool check is informational only (not a hard error). Fires at config load time. Uses `logging.warning` when a layer has < 3 soundfonts available.

### Claude's Discretion
- Internal module naming (`config.py` vs `settings.py`, `timesig.py` vs `time_signatures.py`)
- Dataclass field names and exact registry API shape
- Whether to use `@dataclass` or `@dataclass(frozen=True)` for registry entries
- Logger naming convention (`__name__` per module vs centralized)

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| R-S5 | Config centralization — single module owns all paths to `sf/<layer>/`, `*_fx.json`, `inst_probabilities.json`, `levels.json`, `song_structures.json`, `chord_patterns.txt`, `beat_roll_patterns_*.txt`. No path literals outside config. | Path Literal Inventory (below) enumerates every `file:line` — 18 sites total. D-03 locks "wrap existing files, no new format." |
| R-S6 | Time-signature registry — consolidate into one module so adding a signature touches one location. Must land before generator extraction (Phase 3). | Time-Signature Surface Area (below) maps 7 functions + `DurationValidator._analyze_time_signature` + 2 pattern-file tables. D-04/D-05 lock dataclass-based registry owning all validation. |
| R-S7 | Structured logging — replace all 32+ `print()` calls in `music_gen.py` with `logging`. `musicality_score.py` bare-except fix is already done in Phase 1 Plan 01-03 (verified below). | `print()` Call Inventory (below) lists 32 calls with suggested severity per D-07. |
| R-S9 | Soundfont pool detection — log count per `sf/<layer>/`, warn if < 3. | Soundfont Pool Layout (below) identifies `sf/{beat,melody,harmony,bassline}/` and the `get_random_sound_font()` / `mix_and_save()` hook points. D-09 locks informational-only behavior. |
</phase_requirements>

## Project Constraints (from CLAUDE.md)

No `CLAUDE.md` exists in the repo (verified: `Read` of `/home/openclaw/musicgen/CLAUDE.md` returned "File does not exist"). No project-level coding directives to honor beyond the conventions captured in `.planning/codebase/CONVENTIONS.md`.

No `.claude/skills/` or `.agents/skills/` directories exist in the repo — verified via `ls`. No skill rules to load.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `dataclasses` (stdlib) | Python 3.9+ | Registry entry type + Config container | `DurationValidator` already uses `@dataclass` for `TimeSignatureInfo` (`enhanced_duration_validator.py:23`). Project convention. [VERIFIED: grep] |
| `logging` (stdlib) | — | All logging calls | Already used by `musicality_score.py` and `enhanced_duration_validator.py` via `logging.getLogger(__name__)`. [VERIFIED: grep `musicality_score.py:14`] |
| `python-json-logger` | ≥2.0.7 | Optional JSON formatter (activated in Phase 6) | Already pinned in `requirements.txt` and importable in `.venv/`: `/home/openclaw/musicgen/.venv/bin/python -c "import pythonjsonlogger"` succeeds. [VERIFIED: runtime import check] |
| `os.path` (stdlib) | — | Path construction in `config.py` | `mix_and_save` already uses `os.path.join('sf','beat')` etc. (`music_gen.py:773-776`). Project has no pathlib adoption yet — keep consistent. [VERIFIED: grep] |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `os.environ` (stdlib) | — | Env-var override layer | `Config.load()` reads env vars like `MUSICGEN_SF_DIR` as override inputs per D-02 |
| `typing` (stdlib) | — | Type hints on registry/config | Project uses `Tuple, Dict, List, Optional` already (`music_gen.py:16`) |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `@dataclass` registry entries | `pydantic.BaseModel` | pydantic gives validation/coercion but adds a new dep — rejected, D-04 says "dataclasses" and the repo has no pydantic today. [VERIFIED: `requirements.txt`] |
| Module-level constants in `config.py` | `ConfigParser` / `tomllib` / YAML | D-03 explicitly says "no new config file format" — we wrap existing JSON files only. |
| `argparse` stub in `config.py` for D-01 CLI layer | `typer` | `typer>=0.12` is a Phase 6 dependency per ROADMAP Phase 3 deliverables ("Add `typer>=0.12` as a runtime dependency"). Do NOT introduce typer in Phase 2 — leave a docstring hook for `Config.from_cli(args)` to be wired in Phase 6. [VERIFIED: `.planning/ROADMAP.md` line 74] |
| `logging.config.dictConfig` at module import | Per-module `getLogger(__name__)` + one `basicConfig` call in `__main__` guard | `basicConfig` inside `if __name__ == "__main__":` is what `musicality_score.py` does at line 13 (inside `__init__` today, but should be moved out — noted below). Keeping imports side-effect-free is hard-learned from Plan 01-01. [VERIFIED: Plan 01-01 SUMMARY] |

**Installation:** No new deps. Already-installed verification:
```bash
# Both run clean in .venv/ — already available
/home/openclaw/musicgen/.venv/bin/python -c "import pythonjsonlogger; import logging; from dataclasses import dataclass"
```

**Version verification:** `python-json-logger>=2.0.7` is pinned in `requirements.txt`. `.venv/` has it importable. No registry lookup needed — the project's existing pin already governs. [VERIFIED: cat requirements.txt + runtime import]

## Architecture Patterns

### Recommended Project Structure

Phase 2 introduces two new sibling modules at the repo root (not yet in `src/musicgen/` — that's Phase 3). Structure stays flat per current layout:

```
musicgen/                           # repo root
├── music_gen.py                    # shrinks: all path literals + print() removed, time-sig funcs become thin wrappers
├── config.py                       # NEW — owns paths + Config dataclass + load_config() entrypoint
├── timesig.py                      # NEW — TimeSignatureRegistry + TimeSignatureSpec dataclass
├── enhanced_duration_validator.py  # MODIFIED — DurationValidator delegates to timesig.registry.lookup()
├── musicality_score.py             # unchanged
└── tests/
    ├── test_time_signature.py      # MODIFIED — imports re-pointed to timesig module
    ├── test_duration_validator.py  # MODIFIED if needed — import paths only
    ├── test_config.py              # NEW — config load/override precedence + soundfont count warning
    └── test_timesig_registry.py    # NEW — covers all 7 signatures per D-04
```

**Why flat, not `src/musicgen/`:** Package extraction is Phase 3's job (R-X1). Phase 2 must keep `import music_gen` working and the `tests/conftest.py` sys.path shim intact (Plan 01-04 flagged it as throwaway scheduled for Phase 3 deletion).

### Pattern 1: Registry as Pure Data + Thin Wrappers
**What:** Build `timesig.py` with a single `TimeSignatureRegistry.REGISTRY: Dict[str, TimeSignatureSpec]` constant, dataclass fields for all per-signature attributes, and methods like `verify_chord_pattern_length(length)`, `verify_beat_pattern_length(length)`, `measures_multiplier(base)`. Then `music_gen.py` keeps the existing module-level function names as one-liners:

```python
# music_gen.py after refactor
from timesig import registry

def verify_pattern_for_time_signature(chord_pattern, time_signature):
    return registry.lookup(time_signature).verify_chord_pattern_length(len(chord_pattern))
```

**When to use:** Always for D-05 (registry owns validation logic) combined with preserving the Plan 01-04 test pinning discipline. The tests import `music_gen.verify_pattern_for_time_signature` and call it with the current signature; thin wrappers preserve the public surface with zero test rewrites beyond potentially adding tests for the new module.

**Example (proposed — matches current behavior exactly):**
```python
# timesig.py
from dataclasses import dataclass, field
from typing import FrozenSet, Optional

@dataclass(frozen=True)
class TimeSignatureSpec:
    name: str                           # "4/4"
    numerator: int
    denominator: int
    is_compound: bool                   # denominator == 8 and numerator % 3 == 0
    valid_chord_pattern_lengths: FrozenSet[int]  # {1,2,4} for 4/4
    beat_pattern_length: int            # must equal numerator per tests (verified cosmetic-if quirk)
    measure_multiplier: float           # 1.0 for 4/4, 2.0 for compound/2-4, 4/3 for 3/4
    midi_denominator_power: int         # MIDI format: 2 for denom=4, 3 for denom=8
    # DurationValidator-facing fields
    beats_per_measure: float            # numerator/3 if compound else numerator
    valid_durations: FrozenSet[float]   # simple vs compound set from enhanced_duration_validator.py:49-73
    primary_division: float             # 3.0 compound, 2.0 simple
    max_duration: float                 # numerator or DOTTED_QUARTER*(numerator/3)
    min_duration: float = 0.25          # SIXTEENTH

    def verify_chord_pattern_length(self, length: int) -> bool:
        # Mirror music_gen.py:27-37 exactly — default-True for sigs not in the registry's explicit set
        if not self.valid_chord_pattern_lengths:
            return True
        return length in self.valid_chord_pattern_lengths

    def verify_beat_pattern_length(self, length: int) -> bool:
        # Preserve cosmetic-if quirk: len == numerator for BOTH compound and simple paths
        # (Plan 01-04 tests pin this — test_compound_6_8_length_3_not_ok)
        return length == self.beat_pattern_length

    def measures_for(self, base_length: int) -> int:
        # Mirror music_gen.py:56-62 — compound doubles, 2/4 doubles, 3/4 * 4/3, 4/4 identity
        return int(base_length * self.measure_multiplier)


class TimeSignatureRegistry:
    REGISTRY = {
        "4/4":  TimeSignatureSpec(name="4/4",  numerator=4,  denominator=4, is_compound=False,
                                   valid_chord_pattern_lengths=frozenset({1,2,4}),
                                   beat_pattern_length=4, measure_multiplier=1.0,
                                   midi_denominator_power=2, beats_per_measure=4,
                                   valid_durations=frozenset({4.0,2.0,1.0,0.5,0.25,3.0,1.5,0.75}),
                                   primary_division=2.0, max_duration=4.0),
        "3/4":  TimeSignatureSpec(name="3/4",  numerator=3,  denominator=4, is_compound=False,
                                   valid_chord_pattern_lengths=frozenset({1,3}),
                                   beat_pattern_length=3, measure_multiplier=4/3,
                                   midi_denominator_power=2, beats_per_measure=3,
                                   valid_durations=frozenset({4.0,2.0,1.0,0.5,0.25,3.0,1.5,0.75}),
                                   primary_division=2.0, max_duration=3.0),
        "2/4":  TimeSignatureSpec(name="2/4",  numerator=2,  denominator=4, is_compound=False,
                                   valid_chord_pattern_lengths=frozenset({1,2}),
                                   beat_pattern_length=2, measure_multiplier=2.0,
                                   midi_denominator_power=2, beats_per_measure=2,
                                   valid_durations=frozenset({4.0,2.0,1.0,0.5,0.25,3.0,1.5,0.75}),
                                   primary_division=2.0, max_duration=2.0),
        "5/4":  TimeSignatureSpec(name="5/4",  numerator=5,  denominator=4, is_compound=False,
                                   valid_chord_pattern_lengths=frozenset(),   # empty → default-True path
                                   beat_pattern_length=5, measure_multiplier=1.0,
                                   midi_denominator_power=2, beats_per_measure=5,
                                   valid_durations=frozenset({4.0,2.0,1.0,0.5,0.25,3.0,1.5,0.75}),
                                   primary_division=2.0, max_duration=5.0),
        "7/8":  TimeSignatureSpec(name="7/8",  numerator=7,  denominator=8, is_compound=False,  # 7%3 != 0
                                   valid_chord_pattern_lengths=frozenset(),   # default-True path
                                   beat_pattern_length=7, measure_multiplier=1.0,
                                   midi_denominator_power=3, beats_per_measure=7,
                                   valid_durations=frozenset({4.0,2.0,1.0,0.5,0.25,3.0,1.5,0.75}),
                                   primary_division=2.0, max_duration=7.0),
        "6/8":  TimeSignatureSpec(name="6/8",  numerator=6,  denominator=8, is_compound=True,
                                   valid_chord_pattern_lengths=frozenset({2,3,6}),
                                   beat_pattern_length=6, measure_multiplier=2.0,
                                   midi_denominator_power=3, beats_per_measure=2,  # 6/3
                                   valid_durations=frozenset({1.5,1.0,0.75,0.5,0.25}),
                                   primary_division=3.0, max_duration=3.0),  # DOTTED_QUARTER*(6/3)
        "12/8": TimeSignatureSpec(name="12/8", numerator=12, denominator=8, is_compound=True,
                                   valid_chord_pattern_lengths=frozenset({2,3,6}),
                                   beat_pattern_length=12, measure_multiplier=2.0,
                                   midi_denominator_power=3, beats_per_measure=4,  # 12/3
                                   valid_durations=frozenset({1.5,1.0,0.75,0.5,0.25}),
                                   primary_division=3.0, max_duration=6.0),  # DOTTED_QUARTER*(12/3)
    }

    @classmethod
    def lookup(cls, time_signature: str) -> TimeSignatureSpec:
        return cls.REGISTRY[time_signature]

    @classmethod
    def all_signatures(cls) -> list:
        return list(cls.REGISTRY.keys())
```

### Pattern 2: Config as Hardcoded Defaults + Env Override
**What:** `config.py` exposes module-level constants for the hardcoded defaults (the Phase 6 "lowest precedence" layer per D-02), plus a `Config` dataclass and a `load()` classmethod that layers env vars over those defaults. No file-based config (D-03 says "wrap existing files, no new format").

**When to use:** Immediately, as the first step in Phase 2 so every downstream task (timesig wiring, print→logging, soundfont check) imports paths from `config.py` instead of hardcoding.

**Example:**
```python
# config.py
import os
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List

logger = logging.getLogger(__name__)

# --- Hardcoded defaults (lowest precedence layer per D-02) ---
DEFAULT_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SF_DIR = os.path.join(DEFAULT_PROJECT_ROOT, "sf")
DEFAULT_SF_LAYERS = ("beat", "melody", "harmony", "bassline")

DEFAULT_FX_FILES = {
    "beat":     os.path.join(DEFAULT_PROJECT_ROOT, "beat_fx.json"),
    "melody":   os.path.join(DEFAULT_PROJECT_ROOT, "melody_fx.json"),
    "harmony":  os.path.join(DEFAULT_PROJECT_ROOT, "harmony_fx.json"),
    "bassline": os.path.join(DEFAULT_PROJECT_ROOT, "bassline_fx.json"),
}
DEFAULT_INST_PROBABILITIES_FILE = os.path.join(DEFAULT_PROJECT_ROOT, "inst_probabilities.json")
DEFAULT_LEVELS_FILE              = os.path.join(DEFAULT_PROJECT_ROOT, "levels.json")
DEFAULT_SONG_STRUCTURES_FILE     = os.path.join(DEFAULT_PROJECT_ROOT, "song_structures.json")
DEFAULT_CHORD_PATTERNS_FILE      = os.path.join(DEFAULT_PROJECT_ROOT, "chord_patterns.txt")

DEFAULT_BEAT_ROLL_PATTERN_FILES = {
    "2/4":  os.path.join(DEFAULT_PROJECT_ROOT, "beat_roll_patterns_24.txt"),
    "3/4":  os.path.join(DEFAULT_PROJECT_ROOT, "beat_roll_patterns_34.txt"),
    "4/4":  os.path.join(DEFAULT_PROJECT_ROOT, "beat_roll_patterns_44.txt"),
    "6/8":  os.path.join(DEFAULT_PROJECT_ROOT, "beat_roll_patterns_68.txt"),
    "7/8":  os.path.join(DEFAULT_PROJECT_ROOT, "beat_roll_patterns_78.txt"),
    "12/8": os.path.join(DEFAULT_PROJECT_ROOT, "beat_roll_patterns_128.txt"),
}
SOUNDFONT_POOL_WARN_THRESHOLD = 3  # D-09

@dataclass
class Config:
    project_root: str = DEFAULT_PROJECT_ROOT
    sf_dir: str = DEFAULT_SF_DIR
    sf_layers: tuple = DEFAULT_SF_LAYERS
    fx_files: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_FX_FILES))
    inst_probabilities_file: str = DEFAULT_INST_PROBABILITIES_FILE
    levels_file: str = DEFAULT_LEVELS_FILE
    song_structures_file: str = DEFAULT_SONG_STRUCTURES_FILE
    chord_patterns_file: str = DEFAULT_CHORD_PATTERNS_FILE
    beat_roll_pattern_files: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_BEAT_ROLL_PATTERN_FILES))

    def sf_layer_dir(self, layer: str) -> str:
        return os.path.join(self.sf_dir, layer)

    def beat_pattern_file(self, time_signature: str) -> str:
        return self.beat_roll_pattern_files[time_signature]

    @classmethod
    def load(cls, cli_overrides: dict = None) -> "Config":
        """Three-layer precedence per D-01/D-02: CLI > env > defaults.

        CLI layer is the `cli_overrides` dict parameter. Phase 6 `typer` will
        populate this; Phase 2 callers pass `None` and only env + defaults apply.
        """
        cfg = cls()
        # env-var layer
        if sf := os.environ.get("MUSICGEN_SF_DIR"):
            cfg.sf_dir = sf
        if root := os.environ.get("MUSICGEN_PROJECT_ROOT"):
            cfg.project_root = root
        # (add more env vars as the CLI grows; keep them out of fast path)
        # cli layer
        if cli_overrides:
            for k, v in cli_overrides.items():
                if hasattr(cfg, k) and v is not None:
                    setattr(cfg, k, v)
        cfg._emit_soundfont_pool_report()  # D-09 fires at load
        return cfg

    def _emit_soundfont_pool_report(self) -> None:
        for layer in self.sf_layers:
            layer_dir = self.sf_layer_dir(layer)
            try:
                count = len([f for f in os.listdir(layer_dir) if f.endswith(".sf2")])
            except FileNotFoundError:
                logger.warning("Soundfont layer directory missing: %s", layer_dir)
                continue
            if count < SOUNDFONT_POOL_WARN_THRESHOLD:
                logger.warning(
                    "Soundfont pool thin for layer %s: %d .sf2 files in %s (expected >= %d)",
                    layer, count, layer_dir, SOUNDFONT_POOL_WARN_THRESHOLD,
                )
            else:
                logger.info("Soundfont pool for layer %s: %d .sf2 files", layer, count)
```

### Pattern 3: Per-Module Logger (confirm D-07 default)
**What:** Every module does `logger = logging.getLogger(__name__)` at module top. No centralized logger. Matches `musicality_score.py:14` and `enhanced_duration_validator.py:36` exactly.

**When to use:** Always for `config.py`, `timesig.py`, and the post-migration `music_gen.py`.

**Gotcha:** `musicality_score.py:12-13` has `logging.basicConfig(level=logging.INFO)` inside `MusicalityAnalyzer.__init__` — that's wrong placement (configures global logging every time the class is instantiated) but out of scope here. Do not replicate the mistake in `config.py`. For `music_gen.py`, put `logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")` inside the `if __name__ == "__main__":` guard at `music_gen.py:1170`. That preserves importability (Plan 01-01's hard-won property).

### Anti-Patterns to Avoid
- **Anti-pattern: Importing `timesig` at the top of `config.py` or vice versa.** Creates a circular-import risk if `config.py` grows to reference time-sig-specific constants. Keep them independent: `config.py` holds path data only; `timesig.py` holds music-theory data only. `music_gen.py` imports both but they never import each other.
- **Anti-pattern: Removing the module-level `verify_pattern_for_time_signature` / `verify_beat_pattern` / `validate_measures` functions from `music_gen.py`.** Tests in `tests/test_time_signature.py` call them via `music_gen.verify_pattern_for_time_signature(...)`. Keeping them as thin wrappers means zero test diff (except possibly adding new `test_timesig_registry.py` alongside). This satisfies D-05 ("delegating to the registry" wording) while preserving Plan 01-04's 46 pinned assertions.
- **Anti-pattern: Using `frozenset()` comparison semantics incorrectly.** The 5/4 and 7/8 entries use empty frozenset for `valid_chord_pattern_lengths` to preserve the "default branch returns True" behavior from `music_gen.py:37`. Test `test_unknown_signature_defaults_true` pins this. The wrapper method must special-case empty set → True.
- **Anti-pattern: `logging.basicConfig` at module import.** Triggers the "importing music_gen configures global logging" footgun. Confine it to `if __name__ == "__main__":`.
- **Anti-pattern: `print()` replacements as f-strings passed to `.info()`.** `logger.info("Mixing part: %s", part)` is preferred over `logger.info(f"Mixing part: {part}")` — it defers formatting until the handler decides to emit. Match `musicality_score.py:67` style (`self.logger.exception("Error in tempo analysis: %s", exc)`).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Parse "num/den" time signature strings | A new utility function | `TimeSignatureRegistry.lookup(ts)` | Seven sites in `music_gen.py` do `map(int, ts.split('/'))` today; registry replaces all of them with a dict lookup. Parsing is O(1) constant-time per lookup anyway. |
| Validate time-signature "correctness" | Custom branch logic with `numerator == 3` etc. | Dataclass fields on the `TimeSignatureSpec` | D-05 locks this. |
| Log level plumbing (verbose/quiet toggle) | Custom verbosity int | `logging.basicConfig(level=...)` with `-v/-q` flag parsing (Phase 6) | Phase 2 just needs named levels. Toggle arrives with `typer` in Phase 6. |
| JSON log formatter | Custom `json.dumps(record)` | `pythonjsonlogger.jsonlogger.JsonFormatter` | Dep already pinned. D-08 defers activation to Phase 6 — do not wire it yet, just leave a commented `# Phase 6: swap to JsonFormatter` hook in the `__main__` logging setup. |
| Config file format | TOML / YAML loader | `os.environ` + module constants | D-03 locks this explicitly. |
| "Did logging config get applied?" detection | `logging.getLogger().handlers` inspection | Trust that the `__main__` `basicConfig` runs once; library-mode callers configure their own root logger | Standard Python logging idiom; no work needed. |

**Key insight:** This phase has zero genuine algorithmic work. Every deliverable is *relocation of existing logic* behind a cleaner interface. The risks are all in the plumbing (test imports, circular imports, basicConfig placement), not in the design.

## Path Literal Inventory

Every path literal in `music_gen.py` that must move to `config.py`, verified by grep (`grep -n "sf/|_fx\.json|inst_probabilities|levels\.json|song_structures|chord_patterns\.txt|beat_roll_patterns" music_gen.py`):

| # | file:line | Literal | Lives in | New config attribute |
|---|-----------|---------|----------|---------------------|
| 1 | `music_gen.py:516` | `"beat_roll_patterns_24.txt"` | `generate_beat.beat_pattern_files["2/4"]` | `config.beat_pattern_file("2/4")` |
| 2 | `music_gen.py:517` | `"beat_roll_patterns_44.txt"` | `generate_beat.beat_pattern_files["4/4"]` | `config.beat_pattern_file("4/4")` |
| 3 | `music_gen.py:518` | `"beat_roll_patterns_34.txt"` | `generate_beat.beat_pattern_files["3/4"]` | `config.beat_pattern_file("3/4")` |
| 4 | `music_gen.py:519` | `"beat_roll_patterns_68.txt"` | `generate_beat.beat_pattern_files["6/8"]` | `config.beat_pattern_file("6/8")` |
| 5 | `music_gen.py:520` | `"beat_roll_patterns_78.txt"` | `generate_beat.beat_pattern_files["7/8"]` | `config.beat_pattern_file("7/8")` |
| 6 | `music_gen.py:521` | `"beat_roll_patterns_128.txt"` | `generate_beat.beat_pattern_files["12/8"]` | `config.beat_pattern_file("12/8")` |
| 7 | `music_gen.py:611` | `'song_structures.json'` | `generate_song_arrangement` default arg | `config.song_structures_file` |
| 8 | `music_gen.py:773` | `os.path.join('sf','beat')` | `mix_and_save` soundfont selection | `config.sf_layer_dir("beat")` |
| 9 | `music_gen.py:774` | `os.path.join('sf','melody')` | `mix_and_save` | `config.sf_layer_dir("melody")` |
| 10 | `music_gen.py:775` | `os.path.join('sf','harmony')` | `mix_and_save` | `config.sf_layer_dir("harmony")` |
| 11 | `music_gen.py:776` | `os.path.join('sf','bassline')` | `mix_and_save` | `config.sf_layer_dir("bassline")` |
| 12 | `music_gen.py:785` | `'beat_fx.json'` | `mix_and_save` pedalboard load | `config.fx_files["beat"]` |
| 13 | `music_gen.py:786` | `'melody_fx.json'` | `mix_and_save` | `config.fx_files["melody"]` |
| 14 | `music_gen.py:787` | `'harmony_fx.json'` | `mix_and_save` | `config.fx_files["harmony"]` |
| 15 | `music_gen.py:788` | `'bassline_fx.json'` | `mix_and_save` | `config.fx_files["bassline"]` |
| 16 | `music_gen.py:798` | `'inst_probabilities.json'` | `mix_and_save` | `config.inst_probabilities_file` |
| 17 | `music_gen.py:800` | `'levels.json'` | `mix_and_save` | `config.levels_file` |
| 18 | `music_gen.py:1162` | `'chord_patterns.txt'` | `generate_song` → `create_song` kwarg | `config.chord_patterns_file` |

**Sibling files (not paths but related reference points):**
- `music_gen.py:611` — `generate_song_arrangement(structures_file: str = 'song_structures.json')`. The default arg value needs updating or the signature should lose its default (preferred: remove default, require caller to pass from config).
- `enhanced_duration_validator.py` — no path literals. Purely computational. [VERIFIED: grep]
- `musicality_score.py` — no path literals. [VERIFIED: grep]
- `tests/test_time_signature.py` / `tests/test_duration_validator.py` — no path literals. [VERIFIED: grep]

**Exit-criterion grep commands (ROADMAP):**
```bash
# Must return 0 hits after Phase 2 (only config.py may contain these)
grep -n "sf/beat\|sf/melody\|sf/harmony\|sf/bassline" music_gen.py
grep -n "_fx\.json" music_gen.py
grep -nE "beat_roll_patterns_(24|34|44|68|78|128)\.txt" music_gen.py
grep -n "inst_probabilities\.json\|levels\.json\|song_structures\.json\|chord_patterns\.txt" music_gen.py
```
The ROADMAP's stated exit criterion is "No grep hit for `sf/beat` / `*_fx.json` literals outside `config.py`." Expand it to the full set above.

## Time-Signature Surface Area

### Module-level functions in `music_gen.py` that parse / branch on time signature

All verified via `grep -n "time_signature.split('/')" music_gen.py`:

| # | file:line | Function | What it does | Registry method |
|---|-----------|----------|--------------|-----------------|
| 1 | `music_gen.py:20-37` | `verify_pattern_for_time_signature(chord_pattern, ts)` | Returns bool — pattern length valid for compound (`{2,3,6}`), 4 (`{1,2,4}`), 3 (`{1,3}`), 2 (`{1,2}`), else True | `spec.verify_chord_pattern_length(len(pattern))` |
| 2 | `music_gen.py:40-50` | `verify_beat_pattern(pattern, ts)` | Returns bool — `len(pattern) == numerator` for both branches (cosmetic-if) | `spec.verify_beat_pattern_length(len(pattern))` |
| 3 | `music_gen.py:52-62` | `calculate_measures_for_time_signature(base, ts)` | Integer measure count adjustment: compound×2, 2/4×2, 3/4×4/3, else×1 | `spec.measures_for(base)` |
| 4 | `music_gen.py:64-78` | `validate_measures(measures, signatures)` | False only when compound-odd or 2/4-odd | `spec.measure_count_valid(count)` or inline loop calling registry |
| 5 | `music_gen.py:80-102` | `get_midi_time_signature_values(ts)` | Returns `(numerator, midi_power_of_2)` — raises on unsupported denominator | `spec.midi_denominator_power` (field access, numerator from field) |
| 6 | `music_gen.py:104-115` | `get_note_duration(ts)` | 0.5 for compound, 1.0 for simple | `spec.primary_beat_duration` (new field; derivable from is_compound) |
| 7 | `music_gen.py:117-141` | `get_note_durations(ts)` | Dict of whole/half/quarter/eighth/sixteenth durations — compound vs simple | `spec.note_duration_map()` — dict method returning per-note-name durations |
| 8 | `music_gen.py:143-166` | `get_melody_durations(ts)` | List of valid melody durations — compound, 3/4, else (2/4 and 4/4 share) | `spec.melody_duration_candidates` (FrozenSet) |
| 9 | `music_gen.py:938-956` | `generate_random_time_signature()` | Weighted choice across all 7 signatures | Uses `random.choices(registry.all_signatures(), weights=registry.sampling_weights())`. The probability table moves into registry entries as a `sampling_weight: float` field. |
| 10 | `music_gen.py:958-978` | `time_signature_alternative(base)` | Adjacent-signatures dict | `spec.alternatives: Tuple[str, ...]` field; lookup returns random choice |

### Duplicated logic in `enhanced_duration_validator.py`

`DurationValidator._analyze_time_signature` (`enhanced_duration_validator.py:39-87`) parses `"num/den"`, computes `is_compound = denominator == 8 and numerator % 3 == 0`, and builds a `TimeSignatureInfo` dataclass with fields that overlap heavily with our proposed `TimeSignatureSpec`:

| `TimeSignatureInfo` field | Already in registry as |
|---------------------------|------------------------|
| `numerator` | `spec.numerator` |
| `denominator` | `spec.denominator` |
| `is_compound` | `spec.is_compound` |
| `valid_durations` | `spec.valid_durations` |
| `min_duration` | `spec.min_duration` |
| `max_duration` | `spec.max_duration` |
| `primary_division` | `spec.primary_division` |
| `beats_per_measure` | `spec.beats_per_measure` |

**Plan:** `DurationValidator._analyze_time_signature` becomes:
```python
def _analyze_time_signature(self, time_signature: str) -> TimeSignatureInfo:
    # Thin adapter — delegate to registry, return legacy TimeSignatureInfo shape
    # so existing get_valid_duration / get_suggested_duration methods keep working.
    spec = TimeSignatureRegistry.lookup(time_signature)
    return TimeSignatureInfo(
        numerator=spec.numerator,
        denominator=spec.denominator,
        is_compound=spec.is_compound,
        valid_durations=set(spec.valid_durations),
        min_duration=spec.min_duration,
        max_duration=spec.max_duration,
        primary_division=spec.primary_division,
        beats_per_measure=spec.beats_per_measure,
    )
```
This keeps `tests/test_duration_validator.py`'s 49 assertions passing unchanged — the public methods (`get_suggested_duration`, `get_valid_duration`) still operate on `TimeSignatureInfo`, just sourced from the registry. D-05's "registry owns all validation" intent is met because the shape now comes from one place.

**Caveat:** `DurationValidator.get_valid_duration` (`enhanced_duration_validator.py:89-142`) has its own per-layer-type `valid_durations` sets (lines 107-137) that are NOT in the registry. These are layer-specific (melody vs chord vs bass vs beat) — not time-signature-specific in the same way. Leaving them in `DurationValidator` is defensible for Phase 2. If the planner decides to also hoist those into the registry, each `TimeSignatureSpec` needs a `layer_duration_candidates: Dict[str, FrozenSet[float]]`. **Recommendation:** leave them in `DurationValidator` for Phase 2 (minimizes diff, preserves tests) and let Phase 3's generator extraction decide whether to split further. Document the deferral explicitly in the registry docstring.

### Compound vs simple classification per signature

| Signature | Compound? | Why |
|-----------|-----------|-----|
| 2/4 | simple | denom=4 |
| 3/4 | simple | denom=4 |
| 4/4 | simple | denom=4 |
| 5/4 | simple | denom=4 |
| 6/8 | **compound** | denom=8 AND num%3==0 |
| 7/8 | simple (!) | denom=8 but 7%3 != 0 — handled by default branch in all code paths today |
| 12/8 | **compound** | denom=8 AND num%3==0 |

The `is_compound` test is `denominator == 8 and numerator % 3 == 0` — verified in both `music_gen.py:27` and `enhanced_duration_validator.py:45`. 7/8 is an irregular simple meter that falls through to default branches today. Registry preserves this exactly.

### Test-suite pinning constraints (from Plan 01-04)

Plan 01-04 SUMMARY explicitly flags:
1. **"Preserve cosmetic-if branch of verify_beat_pattern."** Both branches return `len == numerator`. The registry method must do the same — NOT `len == numerator/2` for compound.
2. **"NoteValue exact numeric values pinned."** Registry must not drift rhythm grid — reuse `NoteValue` enum or match its float values exactly.
3. **"Pin exact values, not shapes."** Registry `valid_durations` must match the exact floats `DurationValidator._analyze_time_signature` emits today.

## `print()` Call Inventory in `music_gen.py`

Full list from `grep -n "print(" music_gen.py` — **32 total** (matches ROADMAP "32+"):

| # | file:line | Snippet | Suggested level (D-07) | Notes |
|---|-----------|---------|------------------------|-------|
| 1 | `:241` | `print("\t\t\tChord progression: " + str(chord_progression))` | DEBUG | Internal state dump |
| 2 | `:335` | `print("\n\nGenerated melody has invalid timing structure\n")` | WARNING | Recoverable oddity — validation failure |
| 3 | `:344` | `print("\t\t\tMelody: " + str(melody))` | DEBUG | Internal state dump |
| 4 | `:439` | `print("\t\t\tBassline: " + str(bassline))` | DEBUG | Internal state dump |
| 5 | `:463` | `print("\t\t\tBassline midi file: " + filename)` | DEBUG | Internal state dump (path value) |
| 6 | `:609` | `print(f"Beat annotations saved to: {output_file}")` | INFO | Progress milestone |
| 7 | `:646` | `print(f"Warning: Using default structure due to error: {str(e)}")` | WARNING | Fallback path used. Also: switch to `logger.warning(..., exc_info=True)` to capture traceback since this is in an `except` block. |
| 8 | `:760` | `print("Song arrangement: " + str(song_arrangement) + "\n")` | INFO | Progress milestone |
| 9 | `:781` | `print("Beat soundfont: " + beat_soundfont)` | INFO | Progress — records which soundfont was chosen |
| 10 | `:782` | `print("Melody soundfont: " + melody_soundfont)` | INFO | Progress |
| 11 | `:783` | `print("Harmony soundfont: " + harmony_soundfont)` | INFO | Progress |
| 12 | `:784` | `print("Bassline soundfont: " + bassline_soundfont)` | INFO | Progress |
| 13 | `:793` | `print("Beat pedalboard: " + str(beat_board))` | DEBUG | Internal state dump (FX chain detail) |
| 14 | `:794` | `print("Melody pedalboard: " + str(melody_board))` | DEBUG | Internal state dump |
| 15 | `:795` | `print("Harmony pedalboard: " + str(harmony_board))` | DEBUG | Internal state dump |
| 16 | `:796` | `print("Bassline pedalboard: " + str(bassline_board))` | DEBUG | Internal state dump |
| 17 | `:801` | `print("Levels: " + str(levels))` | DEBUG | Internal state dump |
| 18 | `:816` | `print("Mixing song parts...")` | INFO | Progress milestone |
| 19 | `:823` | `print("Mixing part: " + part + ...)` | INFO | Progress — per-part milestone |
| 20 | `:867` | `print("Beat added to mix: "+part)` | DEBUG | Per-layer granular detail — INFO would flood |
| 21 | `:872` | `print("Melody added to mix: "+part)` | DEBUG | " |
| 22 | `:877` | `print("Harmony added to mix: "+part)` | DEBUG | " |
| 23 | `:882` | `print("Bassline added to mix: "+part)` | DEBUG | " |
| 24 | `:902` | `print("Song saved as: " + song_file_wav)` | INFO | Progress milestone |
| 25 | `:1035` | `print(f"Generating song with swing amount: {swing_amount}")` | INFO | Progress milestone |
| 26 | `:1079` | `print(f'Elapsed time: {elapsed_time:.2f} seconds')` | INFO | Progress metric |
| 27 | `:1080` | `print(f'Musicality Analysis:')` | INFO | Progress heading |
| 28 | `:1081` | `print(f'Score: {score:.2f}')` | INFO | Progress metric |
| 29 | `:1082` | `print('Component Scores:')` | DEBUG | Subheader of DEBUG table |
| 30 | `:1084` | `print(f'{component:>10}: {value:.2f}')` | DEBUG | Per-component state dump |
| 31 | `:1106` | `print(f"Generating part: {part} ({measures} measures)")` | INFO | Progress milestone |
| 32 | `:1135` | `print(f"Generating song #{str(id)}")` | INFO | Progress milestone |

**Severity rollup:**
- INFO: 14 calls (progress milestones — 8, 9-12, 18, 19, 24, 25, 26, 27, 28, 31, 32)
- DEBUG: 16 calls (internal state dumps — 1, 3, 4, 5, 13-17, 20-23, 29, 30)
- WARNING: 2 calls (2, 7)
- ERROR: 0 calls

**Batching recommendation for the planner:** Tasks can be split by severity for cleaner commits:
1. Task A — replace all 16 DEBUG prints (bulk `s/print(/logger.debug(/`).
2. Task B — replace all 14 INFO prints.
3. Task C — replace the 2 WARNING prints (special: use `exc_info=True` inside `except` block at line 646).
4. Task D — add the `logger = logging.getLogger(__name__)` at module top + `basicConfig(...)` inside `if __name__ == "__main__":`.

**Format-string conversion pattern:** Change `print("Beat soundfont: " + beat_soundfont)` to `logger.info("Beat soundfont: %s", beat_soundfont)`. Change `print(f"Generating song #{str(id)}")` to `logger.info("Generating song #%s", id)`. Match `musicality_score.py:67` style.

## Soundfont Pool Layout

**Verified by `ls sf/`:**

```
sf/
├── beat/
│   └── soundfonts.txt      # in this sandbox — 0 *.sf2 files
├── melody/
├── harmony/
├── bassline/
└── soundfonts.txt
```

Current sandbox count: 0 `.sf2` files in each layer directory (runtime can't render audio here, but the pool-size check can still run and will emit a WARNING for each layer — expected behavior that demonstrates D-09 working).

### Current probing location

`music_gen.py:654-657`:
```python
def get_random_sound_font(directory_path):
    sound_fonts = [f for f in os.listdir(directory_path) if f.endswith('.sf2')]
    file_return = random.choice(sound_fonts)
    return os.path.join(directory_path, file_return)
```

This is called per-part inside `mix_and_save` at lines 773-776 with the four hardcoded layer paths. It does not currently count; it just picks one randomly. If a layer is empty, `random.choice([])` raises `IndexError` deep in generation today — the D-09 startup check would surface the problem much earlier.

### Best hook point for the startup check

Three options, ranked:

1. **Inside `Config.load()` at the end, via `_emit_soundfont_pool_report()` helper** (recommended — see Pattern 2 example). Fires at config load time per D-09. Call site: `music_gen.py`'s `__main__` guard calls `cfg = Config.load()` before `generate_song(i)`.
2. Inside `mix_and_save` just before the four `get_random_sound_font` calls. Rejected — fires per-song, not at startup.
3. As a new standalone function in `config.py` called manually. Rejected — fragile, easy to forget.

**Important:** For Phase 2, `generate_song` / `create_song` should receive a `Config` parameter (or the module holds a single module-level `config = Config.load()` loaded lazily). The planner should pick one and document it; lazy-module-level is simpler but breaks the import-without-side-effects rule from Plan 01-01 (listdir at import time is a side effect). **Recommendation:** pass a `Config` instance through `generate_song` → `create_song` → `mix_and_save` → registry/path consumers, loaded once inside the `if __name__ == "__main__":` guard. This keeps imports pure.

## CLI Override Scaffolding (Phase 6 compatibility)

ROADMAP Phase 3 (`R-X1`) adds `typer>=0.12` as a runtime dependency. Phase 6 actually wires up `musicgen.cli:app`. The Phase 2 config API needs to anticipate this without introducing typer.

**Recommended shape (survives Phase 6 unchanged):**

```python
# config.py
@dataclass
class Config:
    ...
    @classmethod
    def load(cls, cli_overrides: dict = None) -> "Config":
        """
        Precedence (D-01/D-02): cli_overrides > env vars > hardcoded defaults.

        Phase 2: callers pass cli_overrides=None; only env + defaults apply.
        Phase 6: typer command will build a dict from parsed args and pass it here:
            cfg = Config.load(cli_overrides={"sf_dir": args.sf_dir, "workers": args.workers})
        """
```

**Why this survives Phase 6:** `typer` commands traditionally build a dict or kwargs bundle from typed parameters; passing that dict into `Config.load()` is idiomatic. No refactor will be needed when Phase 6 lands — the CLI just starts populating the `cli_overrides` arg that's currently always `None`.

**Anti-pattern:** Do NOT write `Config.from_argparse(ns)` in Phase 2 — argparse is not the target CLI framework. A generic `cli_overrides: dict` is framework-agnostic.

**Anti-pattern:** Do NOT add a `Config.from_cli_args(argv: list[str])` method that parses anything. Phase 2 doesn't own CLI parsing.

## Existing Logging Pattern (canonical reference)

**`musicality_score.py`:**
```python
# Line 12-14 (inside MusicalityAnalyzer.__init__)
logging.basicConfig(level=logging.INFO)
self.logger = logging.getLogger(__name__)
```
- Usage: `self.logger.exception("Error in tempo analysis: %s", exc)` (line 67).
- `basicConfig` placement is WRONG (re-runs on every instantiation). Don't copy this literally.

**`enhanced_duration_validator.py`:**
```python
# Line 6 (module top), then line 36 inside __init__
import logging
...
self.logger = logging.getLogger(__name__)
```
- Usage: `self.logger.warning(f"{layer_type} layer duration does not complete full measures")` (line 152).
- Uses f-string, not format args. Acceptable but not optimal. New code should prefer `%s` style.

**Recommended convention for Phase 2:**

```python
# At module top of config.py, timesig.py, music_gen.py:
import logging
logger = logging.getLogger(__name__)   # module-level, not instance-level
```

```python
# Inside the if __name__ == "__main__": guard of music_gen.py:
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    # Phase 6: swap to pythonjsonlogger.jsonlogger.JsonFormatter when --json flag arrives
    for i in range(1):
        generate_song(i)
```

**Logger naming convention:** Use `logging.getLogger(__name__)` per module. Three loggers will exist: `config`, `timesig`, `music_gen`. Output will be prefixed with the module name, which helps grepping logs. Matches the explicit D-07 "Claude's Discretion" note — this IS the obvious answer and I confirm it over any alternative (centralized `logging.getLogger("musicgen")` would be worse because it couples modules to a name that might collide with the Phase 3 `src/musicgen/` package).

## Test Migration

### Current imports in tests

`tests/test_time_signature.py:10`:
```python
import music_gen
```
Uses: `music_gen.verify_pattern_for_time_signature(...)`, `music_gen.verify_beat_pattern(...)`, `music_gen.validate_measures(...)` — all module-level.

`tests/test_duration_validator.py:10`:
```python
from enhanced_duration_validator import DurationValidator, NoteValue
```
Uses `DurationValidator()` instance methods directly.

### Minimal-diff migration strategy

**Recommendation: Keep `tests/test_time_signature.py` UNCHANGED.** The registry goes in `timesig.py` but `music_gen.py` keeps module-level functions of the same names that are one-line delegates. All 46 assertions pass with zero diff.

**For `tests/test_duration_validator.py`:** Keep UNCHANGED. `DurationValidator` still exposes the same public methods; internal `_analyze_time_signature` now delegates to the registry but returns the same `TimeSignatureInfo` shape. All 49 assertions pass with zero diff.

### New tests Phase 2 must add

Per R-S6 ("Unit tests for `timesig` registry covering every currently supported signature (2/4, 3/4, 4/4, 5/4, 6/8, 7/8, 12/8)"):

`tests/test_timesig_registry.py` — new file, new tests:
- `test_registry_contains_all_seven_signatures` — assert `set(TimeSignatureRegistry.all_signatures()) == {"2/4","3/4","4/4","5/4","6/8","7/8","12/8"}`
- `test_lookup_returns_spec_per_signature` — parametrized over 7 sigs
- `test_is_compound_classification` — 6/8 and 12/8 True, rest False
- `test_midi_denominator_power` — 2 for denom=4, 3 for denom=8
- `test_chord_pattern_length_validation_mirrors_legacy` — parametrized matrix: every (signature, length) tuple the Plan 01-04 test checks, asserted via `spec.verify_chord_pattern_length(length)` directly
- `test_beat_pattern_length_validation_preserves_cosmetic_if` — explicit tests that `spec("6/8").verify_beat_pattern_length(3) is False` and `is True` only at 6; ditto 12/8 at 12 and not 6
- `test_measures_multiplier_values` — 4/4 → 1, 3/4 → 4/3 (int-truncated), 2/4 → 2, 6/8 → 2, 12/8 → 2
- `test_alternatives_field_preserves_legacy_dict` — `spec("4/4").alternatives == ("2/4","3/4","6/8","12/8")` etc.
- `test_sampling_weights_sum_to_one` — ensures `generate_random_time_signature()` refactor won't drift distribution

`tests/test_config.py` — new file, new tests:
- `test_config_defaults_match_legacy_paths` — assert `Config().sf_layer_dir("beat").endswith("sf/beat")` etc. (parametrized over four layers)
- `test_config_fx_files_all_present_as_keys` — `set(Config().fx_files.keys()) == {"beat","melody","harmony","bassline"}`
- `test_config_beat_pattern_file_per_signature` — parametrized over 6 beat-pattern files
- `test_env_override_applies_to_sf_dir` — monkeypatch `MUSICGEN_SF_DIR`, assert `Config.load().sf_dir == monkeypatched`
- `test_cli_overrides_take_precedence_over_env` — monkeypatch env, call `Config.load(cli_overrides={"sf_dir": "/cli/path"})`, assert wins
- `test_soundfont_pool_warning_fires_below_threshold` — `caplog.at_level(logging.WARNING)`, create a tmp_path with 2 dummy `.sf2` files, point config at it, assert WARNING logged
- `test_soundfont_pool_no_warning_at_threshold` — 3 `.sf2` files, assert no WARNING
- `test_soundfont_pool_missing_layer_logs_warning` — layer dir doesn't exist, assert WARNING logged with "missing" phrasing

`tests/test_music_gen_logging.py` — new file, minimal:
- `test_import_music_gen_does_not_emit_logs` — assert `import music_gen` still doesn't print or log anything (regression guard for the __main__ boundary)
- `test_no_print_calls_remain_in_music_gen` — AST-level scan: walk `music_gen.py`, assert `ast.Call(func=ast.Name(id='print'))` count == 0. This is the direct R-S7 exit criterion.

### Total new tests target
Roughly 45-60 new tests (registry × 7 sigs parametrized expands quickly). Runtime target < 1s additional (under R-Q2 < 10s budget).

## Common Pitfalls

### Pitfall 1: Importing `config.py` from `timesig.py` (or vice versa)
**What goes wrong:** Circular import when `music_gen.py` imports both. Python's import machinery partially loads one module, then the other tries to reference symbols not yet bound.
**Why it happens:** `config.py` might be tempted to reference `TimeSignatureRegistry.all_signatures()` to build `DEFAULT_BEAT_ROLL_PATTERN_FILES` keys. `timesig.py` might be tempted to reference `config.DEFAULT_*` for something.
**How to avoid:** Keep them strictly independent. `config.py` has no knowledge of time signatures beyond the six hardcoded keys in `DEFAULT_BEAT_ROLL_PATTERN_FILES`. `timesig.py` has no knowledge of filesystem paths. `music_gen.py` is the only importer of both.
**Warning signs:** `ImportError: cannot import name 'X' from partially initialized module 'Y'`.

### Pitfall 2: `basicConfig` at module import
**What goes wrong:** `import music_gen` from a test or downstream tool causes `logging.basicConfig` to run, reconfiguring the root logger and potentially swallowing the caller's existing log handlers.
**Why it happens:** Copying `musicality_score.py:12-13` pattern without realizing it's a bug.
**How to avoid:** Put `logging.basicConfig(...)` ONLY inside `if __name__ == "__main__":`. Module-level code emits via `getLogger(__name__)` and lets library consumers configure the root.
**Warning signs:** Test output contains log lines when it shouldn't; downstream tools complain about duplicate handlers.

### Pitfall 3: Registry's 7/8 and 5/4 chord-pattern lengths
**What goes wrong:** Test `test_unknown_signature_defaults_true` asserts `verify_pattern_for_time_signature(["I","IV","V"], "5/4") is True`. The source (`music_gen.py:20-37`) implements this via a default-branch-returns-True fallthrough. A naive registry implementation that says `length in spec.valid_chord_pattern_lengths` returns False for 5/4 because the set is empty, breaking the test.
**Why it happens:** Empty `frozenset()` semantics differ from default-branch-returns-True semantics.
**How to avoid:** Either (a) special-case `if not spec.valid_chord_pattern_lengths: return True` in `verify_chord_pattern_length`, OR (b) use `Optional[FrozenSet[int]]` with `None` meaning "no constraint → True". Option (a) is simpler and matches the legacy branch exactly. Document the choice in the method docstring.
**Warning signs:** `test_unknown_signature_defaults_true` fails after refactor.

### Pitfall 4: `validate_measures` iterates dict — cannot be a simple spec method
**What goes wrong:** `validate_measures(measures, signatures)` takes two parallel dicts and returns False if ANY part fails. It's a cross-signature function — you can't just hang it on a single `TimeSignatureSpec`.
**Why it happens:** Over-enthusiastic "push everything into the spec" thinking.
**How to avoid:** Keep `validate_measures` as a module-level function in `timesig.py` (or as a classmethod on `TimeSignatureRegistry`). Its body becomes a loop that calls `registry.lookup(sig).measure_count_valid(count)` for each (part, count) tuple. Wrapper in `music_gen.py` delegates.
**Warning signs:** Plan proposes moving `validate_measures` inside `TimeSignatureSpec`.

### Pitfall 5: `generate_random_time_signature` returning None on edge case
**What goes wrong:** Current implementation (`music_gen.py:938-956`) has no explicit fallback return if `dice >= 1.00` (float rounding). `generate_random_key` and `generate_random_tempo` were fixed in Plan 01 to add explicit fallback returns (see commits `2a02af2` and `e1c9503`). The same pitfall exists at `music_gen.py:956` — there is no `return time_signature_ranges[-1][1]` after the loop.
**Why it happens:** Float rounding leaves `dice >= 1.00` → loop falls through → implicit `return None` → caller receives None and crashes in `split('/')`.
**How to avoid:** When refactoring to use `registry.sampling_weights()`, use `random.choices(...)[0]` (which always returns) instead of a threshold-loop. Also: add a regression test parametrized over seeded `random.seed()` values that guarantees the function never returns None. Document this as an implicit bug-fix that falls out of the refactor — flag to the planner so it can mention it in the SUMMARY.
**Warning signs:** Random hangs once every few runs; or `AttributeError: 'NoneType' object has no attribute 'split'`.

### Pitfall 6: `tests/conftest.py` sys.path shim still applies
**What goes wrong:** Tests import `timesig` and `config` as top-level modules (e.g. `from timesig import TimeSignatureRegistry`). The Plan 01-04 conftest.py sys.path shim inserts the repo root, so these imports resolve. But `pytest tests/ -q` from a different directory might not — the shim only covers repo-root-relative.
**Why it happens:** The shim is intentional throwaway (Plan 01-04 SUMMARY flags it for Phase 3 deletion).
**How to avoid:** Use the existing shim. Don't delete or modify it in Phase 2. Phase 3 replaces it with `pip install -e .`. New test files can safely `import config` and `import timesig` alongside the existing `import music_gen`.
**Warning signs:** `ModuleNotFoundError: No module named 'config'` when running `pytest`.

### Pitfall 7: `generate_song_arrangement` default arg
**What goes wrong:** `music_gen.py:611` has `def generate_song_arrangement(structures_file: str = 'song_structures.json')`. Python evaluates default args at function definition time, so the string literal sits in the function object forever — not a runtime grep target but semantically a path literal.
**Why it happens:** Easy to miss when grepping by file name.
**How to avoid:** Change the signature to `structures_file: str = None` and inside the function do `if structures_file is None: structures_file = config.song_structures_file`. Or require the caller (create_song) to pass it. Latter is cleaner but slightly more diff.
**Warning signs:** Grep exit-criterion check misses this because it's in a function signature, not a string literal inside the function body.

### Pitfall 8: `f"{component:>10}: {value:.2f}"` inside a loop
**What goes wrong:** `music_gen.py:1082-1084` loops and prints per-component. Converting naively to `logger.debug(f"{component:>10}: {value:.2f}")` emits N log records where one would do. Also: format spec like `>10` is harder with `%`-style args.
**Why it happens:** Tempting to one-to-one replace each `print`.
**How to avoid:** Either aggregate into a single `logger.debug("Component scores: %s", component_scores_dict)` call, or accept the N records as cheap at DEBUG level. Both work. Planner's choice.
**Warning signs:** None — this is purely stylistic.

## Code Examples

### Pattern: Module-top logger + `__main__` basicConfig
```python
# music_gen.py (top)
import logging
logger = logging.getLogger(__name__)

# ... rest of module ...

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    for i in range(1):
        generate_song(i)
```
Source: `musicality_score.py:14` pattern + the correction from Pitfall 2 + Plan 01-01 importability contract. [CITED: Plan 01-01 SUMMARY]

### Pattern: Thin wrapper delegating to registry
```python
# music_gen.py (replaces lines 20-37)
from timesig import TimeSignatureRegistry

def verify_pattern_for_time_signature(chord_pattern: List[str], time_signature: str) -> bool:
    """Checks if the chord pattern is appropriate for the time signature.

    Delegates to TimeSignatureRegistry per R-S6; kept as a module-level function
    for backwards compatibility with tests/test_time_signature.py (Plan 01-04).
    """
    spec = TimeSignatureRegistry.lookup(time_signature)
    return spec.verify_chord_pattern_length(len(chord_pattern))
```

### Pattern: `%`-style logging with format args
```python
# Replaces music_gen.py:781
# BEFORE: print("Beat soundfont: " + beat_soundfont)
# AFTER:
logger.info("Beat soundfont: %s", beat_soundfont)
```
Source: `musicality_score.py:67` pattern.

### Pattern: Warning with traceback inside except
```python
# Replaces music_gen.py:646
# BEFORE:
#   print(f"Warning: Using default structure due to error: {str(e)}")
# AFTER:
except (json.JSONDecodeError, FileNotFoundError, KeyError, ValueError):
    logger.warning("Using default structure due to error", exc_info=True)
    return list(set(default_structure)), default_structure
```
Source: `musicality_score.py:67` `.exception()` is for ERROR; here we want WARNING with traceback → use `logger.warning(..., exc_info=True)`.

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Wildcard `from music21 import *` | Explicit symbol imports (`from music21 import roman, scale, pitch`) | Phase 1 Plan 01-03 | Already done — verified `music_gen.py:2` |
| `print()` for status | Per-module `logging.getLogger(__name__)` | Phase 1 partial (musicality_score.py), Phase 2 completes | `music_gen.py` is the last holdout |
| Inline path literals | `config.py` module | Phase 2 | This phase |
| Scattered time-sig branches | `TimeSignatureRegistry` dataclass | Phase 2 | This phase |
| Hand-rolled arg parsing | `typer>=0.12` CLI | Phase 6 (R-P13) | Do NOT anticipate in Phase 2; leave hook only |

**Deprecated/outdated:**
- `logging.basicConfig` inside class `__init__` (as `musicality_score.py:13` does). Should live in `__main__` guards only. Don't copy.
- `from music21 import *` already removed in Phase 1.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| `python-json-logger` | D-08 (deferred Phase 6 wiring) | Yes (in `.venv/`) | ≥2.0.7 (from `requirements.txt`) | — |
| `pytest` / `pytest-cov` | New test files | Yes (in `.venv/` per Plan 01-04) | ≥8.0 / ≥5.0 | — |
| `music21`, `midiutil`, `pydub`, `pedalboard`, `midi2audio`, `librosa` | `music_gen.py` imports | Yes (in `.venv/` per Plan 01-04) | — | — |
| Python stdlib (`dataclasses`, `logging`, `os`, `typing`) | `config.py`, `timesig.py` | Yes (Python 3.9+) | — | — |

**Missing dependencies with no fallback:** None.

**Missing dependencies with fallback:** None.

Sandbox note: `/home/openclaw/musicgen/sf/{beat,melody,harmony,bassline}/` directories exist but contain 0 `.sf2` files. Tests for D-09 (soundfont pool warning) should use `tmp_path` fixtures rather than the real `sf/` tree, so the sandbox condition doesn't block test development.

## Validation Architecture

Workflow has `nyquist_validation: true` in `.planning/config.json`. This section is mandatory.

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest ≥8.0 + pytest-cov ≥5.0 (installed via `dev-requirements.txt`) |
| Config file | none (tests directly in `tests/`; `tests/conftest.py` holds sys.path shim from Plan 01-04) |
| Quick run command | `.venv/bin/python -m pytest tests/ -q` |
| Full suite command | `.venv/bin/python -m pytest tests/ -v` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| R-S5 | `config.py` owns all paths; no path literals in `music_gen.py` | grep regression + unit | `grep -nE "sf/(beat\|melody\|harmony\|bassline)\|_fx\.json\|beat_roll_patterns_\|inst_probabilities\.json\|levels\.json\|song_structures\.json\|chord_patterns\.txt" music_gen.py` returns 0 hits | ❌ Wave 0 (new `tests/test_config.py`) |
| R-S5 | Default paths match legacy file layout | unit | `.venv/bin/python -m pytest tests/test_config.py::test_config_defaults_match_legacy_paths -x` | ❌ Wave 0 |
| R-S5 | Env var override precedence | unit | `.venv/bin/python -m pytest tests/test_config.py::test_env_override_applies_to_sf_dir -x` | ❌ Wave 0 |
| R-S5 | CLI override precedence (D-01/D-02) | unit | `.venv/bin/python -m pytest tests/test_config.py::test_cli_overrides_take_precedence_over_env -x` | ❌ Wave 0 |
| R-S6 | Registry has all 7 signatures | unit | `.venv/bin/python -m pytest tests/test_timesig_registry.py::test_registry_contains_all_seven_signatures -x` | ❌ Wave 0 (new `tests/test_timesig_registry.py`) |
| R-S6 | Per-signature validation behavior matches legacy | unit (parametrized) | `.venv/bin/python -m pytest tests/test_timesig_registry.py -k "chord_pattern_length or beat_pattern_length or measures_multiplier" -x` | ❌ Wave 0 |
| R-S6 | Legacy thin-wrapper tests still green (regression) | unit | `.venv/bin/python -m pytest tests/test_time_signature.py -x` | ✅ (Plan 01-04) |
| R-S6 | DurationValidator still green (regression) | unit | `.venv/bin/python -m pytest tests/test_duration_validator.py -x` | ✅ (Plan 01-04) |
| R-S6 | Adding a signature touches one file | manual + grep | `git diff --name-only HEAD~1 HEAD -- timesig.py music_gen.py` — expect only `timesig.py` changed when adding a new spec | manual |
| R-S7 | No `print()` calls remain in `music_gen.py` | AST | `.venv/bin/python -m pytest tests/test_music_gen_logging.py::test_no_print_calls_remain_in_music_gen -x` | ❌ Wave 0 |
| R-S7 | `import music_gen` still side-effect-free | unit | `.venv/bin/python -m pytest tests/test_music_gen_logging.py::test_import_music_gen_does_not_emit_logs -x` | ❌ Wave 0 |
| R-S7 | grep check: no literal `print(` tokens in `music_gen.py` | grep | `grep -c "^[[:space:]]*print(" music_gen.py` returns 0 | n/a |
| R-S9 | Soundfont pool WARNING fires below threshold | unit (caplog) | `.venv/bin/python -m pytest tests/test_config.py::test_soundfont_pool_warning_fires_below_threshold -x` | ❌ Wave 0 |
| R-S9 | No WARNING at/above threshold | unit (caplog) | `.venv/bin/python -m pytest tests/test_config.py::test_soundfont_pool_no_warning_at_threshold -x` | ❌ Wave 0 |
| R-S9 | Missing layer dir logs WARNING | unit (caplog) | `.venv/bin/python -m pytest tests/test_config.py::test_soundfont_pool_missing_layer_logs_warning -x` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `.venv/bin/python -m pytest tests/ -q` (full suite, < 5s expected). Phase 1's 95 tests run in 2.75s; Phase 2 adds ~60 more, still under the R-Q2 10s budget.
- **Per wave merge:** `.venv/bin/python -m pytest tests/ -v` + the full grep-exit-criterion block.
- **Phase gate:** Full suite green + all four grep commands return 0 hits + AST check passes before `/gsd-verify-work`.

### Wave 0 Gaps
- [ ] `tests/test_config.py` — covers R-S5, R-S9 (new file)
- [ ] `tests/test_timesig_registry.py` — covers R-S6 registry surface (new file)
- [ ] `tests/test_music_gen_logging.py` — covers R-S7 (new file, AST-based)
- [ ] `config.py` — the module itself (not a test but blocks all config tests)
- [ ] `timesig.py` — the module itself (blocks all registry tests)

No framework installation needed — pytest already in `.venv/` from Plan 01-04. No new deps.

## Security Domain

`.planning/config.json` does not set `security_enforcement`, so treat as the default. This is a refactoring phase with zero external input, zero network, zero user-supplied data flowing to the new modules. ASVS surface is minimal.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | — (no auth surface) |
| V3 Session Management | no | — |
| V4 Access Control | no | — |
| V5 Input Validation | yes (narrow) | Validate `MUSICGEN_SF_DIR` env var is a real directory path (not user-supplied shell injection); we use `os.path.join` and `os.listdir`, both of which don't invoke a shell |
| V6 Cryptography | no | — |
| V12 Files and Resources | yes (narrow) | `os.listdir(sf_layer_dir)` can fail on missing / permission-denied; handle `FileNotFoundError` and `PermissionError` in `_emit_soundfont_pool_report` with a WARNING log, not a crash |

### Known Threat Patterns for Python path handling

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Path injection via env var | Tampering | Use `os.path.abspath` / `os.path.normpath`; never pass path values to `os.system` or `shell=True` subprocess — we don't do either here |
| Listing files from untrusted dir | Information Disclosure | `os.listdir` result is filtered to `.sf2` suffix and the values are not echoed except in INFO logs; no exfil risk |
| `json.load` on user-controlled file | Tampering | The only JSON files loaded are shipped in-repo (`levels.json`, `*_fx.json`); env-var override of their paths could point to attacker-controlled JSON, but this requires local filesystem access — threat model out of scope for v0.1 |
| `logging.basicConfig` root handler takeover | Tampering (library consumers) | Confined to `__main__` guard per Pitfall 2 |

**Summary:** No new attack surface. The `MUSICGEN_SF_DIR` env var is the one new user-controlled input; it's consumed only by `os.path.join` and `os.listdir`, both of which are path-traversal-safe in the sense that they don't shell out. If the user points it at `/etc`, they get a WARNING log about no `.sf2` files. No privilege escalation possible.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | The 6/8 and 12/8 `valid_durations` set matches `DurationValidator._analyze_time_signature` compound branch exactly (`{1.5, 1.0, 0.75, 0.5, 0.25}`) | Pattern 1 example code | If drifted, `tests/test_duration_validator.py::test_compound_6_8_*` fails. **Verified by reading `enhanced_duration_validator.py:49-56` directly — this is [VERIFIED]**, moving out of Assumptions Log. |
| A2 | Adding `logging.basicConfig` to the `__main__` guard does not break existing Plan 01-01 AST-level verification that `import music_gen` has no side effects | Pitfall 2 / Pattern 1 | Low — the `if __name__ == "__main__":` guard specifically prevents the `basicConfig` call from running at import time. Plan 01-01 AST check verifies the guard exists; it doesn't inspect what's inside. [ASSUMED: verification will pass] |
| A3 | `TimeSignatureSpec` `sampling_weight` field sums to 1.0 across all 7 signatures and can replace `generate_random_time_signature`'s threshold-loop with `random.choices()` without changing the distribution | Pitfall 5 / Pattern 1 | Medium — floating-point drift between cumulative-threshold and `random.choices` could produce different RNG sequences. Since Phase 5 will lock seed contracts AFTER this phase, drift here is acceptable but must be noted in Phase 2 SUMMARY so Phase 5 baselines against post-refactor RNG order. [ASSUMED] |
| A4 | `pythonjsonlogger.jsonlogger.JsonFormatter` is the correct class path for `python-json-logger>=2.0.7` | Don't Hand-Roll table | Low — public API since 2.0. Activation is Phase 6 only; Phase 2 just references it in a comment. [ASSUMED based on package docs convention] |
| A5 | Plan 01-04's `tests/conftest.py` sys.path shim adds repo root to `sys.path[0]` such that `import config` and `import timesig` will resolve the same way `import music_gen` does today | Pitfall 6 / Test Migration | Low — Plan 01-04 SUMMARY explicitly describes the shim as "sys.path shim inserting repo root." Any top-level `.py` file in the repo will be importable. [ASSUMED from plan docs; could be [VERIFIED] by reading `tests/conftest.py` directly] |
| A6 | The current `generate_random_time_signature` float-rounding gap (no explicit fallback return) is actually present — not already patched in a commit not surfaced in STATE.md | Pitfall 5 | Medium — if already patched, refactor still works but the "bug fix falls out of refactor" framing is wrong. Planner should re-verify with `grep -A 20 "def generate_random_time_signature" music_gen.py` before committing. [VERIFIED via direct read of music_gen.py:938-956 — no fallback return after the loop. Moving out of Assumptions Log.] |
| A7 | Plan 01's fixes `2a02af2` (generate_random_key fallback) and `e1c9503` (generate_random_tempo fallback) did NOT also patch `generate_random_time_signature` | Pitfall 5 | Low — WR-04 in the referenced commit message suggests it covered key + tempo only. Recent commit log confirms: `2a02af2 fix(01): WR-04 add explicit fallback returns in generate_random_key/tempo`. [VERIFIED via git log in session] |

**Net assumptions requiring user confirmation:** A2, A3, A4, A5. A2 and A5 are low-risk plumbing. **A3 is the one to flag in discuss-phase or the plan itself:** the RNG order change from refactor is silent and impacts Phase 5's seed contract baseline.

## Open Questions

1. **Should `generate_random_time_signature` move into `timesig.py` or stay in `music_gen.py` as a wrapper?**
   - What we know: The weights are per-signature data (belongs in registry). The function is a thin RNG-over-registry call. Phase 3 will extract it into `sampler.py` anyway (R-X2).
   - What's unclear: Does Phase 2 move it now (cleaner) or keep it in `music_gen.py` as a wrapper until Phase 3 extracts it (less churn)?
   - Recommendation: Add `classmethod TimeSignatureRegistry.sample_random(rng=None)` in `timesig.py` with the full weights table; keep `generate_random_time_signature()` in `music_gen.py` as a one-line wrapper `return TimeSignatureRegistry.sample_random()` for zero test churn. Phase 3 can inline the wrapper later.

2. **Should `DurationValidator` move into `timesig.py` or stay in `enhanced_duration_validator.py`?**
   - What we know: `enhanced_duration_validator.py` is already a proper module (Plan 01-04 tests pin it). Moving it would rewrite test imports.
   - What's unclear: Does D-05 "registry owns all validation logic" imply absorbing `DurationValidator`?
   - Recommendation: Leave `DurationValidator` where it is. `_analyze_time_signature` becomes a thin adapter that calls `TimeSignatureRegistry.lookup(...)` and returns a legacy-shaped `TimeSignatureInfo`. Layer-specific duration sets (chord/melody/bass/beat) remain in `DurationValidator` since they're orthogonal to time signatures. Minimizes diff, preserves all 49 pinned tests. Phase 3 can reconsider.

3. **Does the planner want a single `config.py` or a `config/` package?**
   - What we know: D-03 says "unified access layer." The Claude-discretion note leaves naming open.
   - What's unclear: Flat single-file is simpler but 200+ lines. Package would allow `config.paths`, `config.overrides`, etc.
   - Recommendation: Single-file `config.py` for Phase 2. It will grow modestly (~150 lines). Package refactor can wait for Phase 3 when everything moves into `src/musicgen/config.py` anyway.

4. **`logger.debug` vs `logger.info` for the per-layer "added to mix" lines (prints 20-23)?**
   - D-07 defines INFO as "progress milestones" and DEBUG as "internal state dumps."
   - Per-layer adds (4 lines per part × 5 parts = 20 log lines per song at INFO) feels too noisy for "progress milestone." DEBUG is more defensible.
   - Recommendation (already captured in inventory): DEBUG. Flag for planner confirmation since it's a close call.

## Sources

### Primary (HIGH confidence)
- `/home/openclaw/musicgen/music_gen.py` lines 1-1172 — direct source read via `Read` tool
- `/home/openclaw/musicgen/enhanced_duration_validator.py` lines 1-172 — direct source read
- `/home/openclaw/musicgen/musicality_score.py` lines 1-264 — direct source read (logging pattern reference)
- `/home/openclaw/musicgen/tests/test_time_signature.py` lines 1-159 — direct source read
- `/home/openclaw/musicgen/tests/test_duration_validator.py` lines 1-129 — direct source read
- `/home/openclaw/musicgen/.planning/ROADMAP.md` Phase 2 section — direct read
- `/home/openclaw/musicgen/.planning/REQUIREMENTS.md` R-S5, R-S6, R-S7, R-S9 — direct read
- `/home/openclaw/musicgen/.planning/phases/02-stabilize-ii-config-time-signature-registry-logging/02-CONTEXT.md` — direct read
- `/home/openclaw/musicgen/.planning/phases/01-stabilize-i-bug-fixes-and-guardrails/01-01-importability-and-arrangement-fix-SUMMARY.md` — direct read (informs Pitfall 2)
- `/home/openclaw/musicgen/.planning/phases/01-stabilize-i-bug-fixes-and-guardrails/01-04-SUMMARY.md` — direct read (pinned test contract)
- `/home/openclaw/musicgen/.planning/codebase/STRUCTURE.md`, `CONVENTIONS.md`, `CONCERNS.md` — direct read
- `/home/openclaw/musicgen/requirements.txt` — direct read (confirms `python-json-logger>=2.0.7` already pinned)
- `/home/openclaw/musicgen/.venv/bin/python -c "import pythonjsonlogger"` — runtime verification

### Secondary (MEDIUM confidence)
- `python-json-logger` package public API (`pythonjsonlogger.jsonlogger.JsonFormatter`) — based on package conventions; not verified via Context7 in this session

### Tertiary (LOW confidence)
- None — Phase 2 is entirely refactoring of existing code. All knowledge is direct-source-derived.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — every dependency already in `requirements.txt` and verified runnable in `.venv/`
- Architecture: HIGH — directly derived from reading `music_gen.py` and the existing tests
- Pitfalls: HIGH — Pitfalls 1, 2, 6, 7 come from direct source inspection; Pitfall 5 from commit log; Pitfall 3 from test file inspection
- Test migration: HIGH — based on reading both test files and Plan 01-04 SUMMARY
- Validation architecture: HIGH — mapped directly to existing test infrastructure

**Research date:** 2026-04-10
**Valid until:** 2026-05-10 (30 days; no fast-moving dependencies, all source local)
