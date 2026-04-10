# Phase 2: Stabilize II — config + time-signature registry + logging - Context

**Gathered:** 2026-04-10
**Status:** Ready for planning

<domain>
## Phase Boundary

Extract a config module that centralizes all hardcoded paths, build a time-signature registry that makes adding a new signature a single-file edit, and replace all `print()` calls in `music_gen.py` with semantically leveled `logging` calls. This phase unblocks Phase 3 extraction by removing path literals and scattered time-signature logic from the god file.

</domain>

<decisions>
## Implementation Decisions

### Config Module Design
- **D-01:** Config module must support three override layers with precedence: CLI args > environment variables > config file defaults. This lays the groundwork for the robust CLI coming in Phase 6.
- **D-02:** Precedence order is standard: CLI args override env vars, env vars override file defaults, file defaults override hardcoded defaults.
- **D-03:** The config module provides a unified access layer on top of the existing JSON files (`levels.json`, `inst_probabilities.json`, `song_structures.json`, `*_fx.json`, etc.). No new config file format introduced — the module wraps existing files with override capability.

### Time-Signature Registry Data Model
- **D-04:** Registry entries are dataclasses containing fields like `valid_chord_lengths`, `beat_count`, `measure_multiplier`, `is_compound`, and any other per-signature metadata currently scattered across functions.
- **D-05:** The registry owns ALL validation logic. `verify_pattern_for_time_signature`, `verify_beat_pattern`, `calculate_measures_for_time_signature` either become thin wrappers delegating to the registry or are absorbed entirely. Adding a new time signature must touch exactly one location.
- **D-06:** Design for flexibility and precision — the system should make it easy to add unusual meters and handle compound vs. simple signatures cleanly.

### Logging Style
- **D-07:** Log levels follow semantic differentiation:
  - `DEBUG` — internal state dumps (chord chosen, pattern selected, intermediate values)
  - `INFO` — progress milestones (mixing part N, song saved, generation started)
  - `WARNING` — recoverable oddities (soundfont pool thin, fallback path used)
  - `ERROR` — failures that affect output quality
- **D-08:** `python-json-logger` is already a dependency. JSON logging format is available but activation deferred to Phase 6 batch mode. Default format stays human-readable for interactive use.

### Soundfont Pool Detection
- **D-09:** Soundfont pool check is informational only (not a hard error). Fires at config load time. Uses `logging.warning` when a layer has < 3 soundfonts available.

### Claude's Discretion
- Internal module naming (`config.py` vs `settings.py`, `timesig.py` vs `time_signatures.py`)
- Dataclass field names and exact registry API shape
- Whether to use `@dataclass` or `@dataclass(frozen=True)` for registry entries
- Logger naming convention (`__name__` per module vs centralized)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase scope and requirements
- `.planning/ROADMAP.md` Phase 2 section — deliverables, exit criteria, requirements covered
- `.planning/REQUIREMENTS.md` R-S5, R-S6, R-S7, R-S9 — the four requirements this phase closes

### Codebase context
- `.planning/codebase/STRUCTURE.md` — current file layout, key locations, config surface friction
- `.planning/codebase/CONVENTIONS.md` — naming patterns, import organization, logging patterns
- `.planning/codebase/CONCERNS.md` — prioritized tech debt inventory

### Prior phase artifacts
- `.planning/phases/01-stabilize-i-bug-fixes-and-guardrails/01-01-importability-and-arrangement-fix-SUMMARY.md` — arrangement fix approach (relevant to registry design)
- `.planning/phases/01-stabilize-i-bug-fixes-and-guardrails/01-04-SUMMARY.md` — pytest skeleton and test patterns (tests must be preserved/migrated)

### Source files to modify
- `music_gen.py` — god file containing all time-sig functions, print calls, and hardcoded paths
- `enhanced_duration_validator.py` — `DurationValidator` has its own time-sig logic that should delegate to the registry
- `musicality_score.py` — already has logging; reference for pattern consistency

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `musicality_score.py` logging pattern: `self.logger = logging.getLogger(__name__)` — established pattern to follow
- `enhanced_duration_validator.py` `DurationValidator` class — already a dataclass-like design, good reference for registry shape
- Phase 1 test suite (`tests/test_time_signature.py`, `tests/test_duration_validator.py`) — 95 passing tests that pin current behavior

### Established Patterns
- Time-sig string parsing: `numerator, denominator = map(int, time_signature.split('/'))` used in 5+ locations — registry eliminates this
- Weighted probability tables: `generate_random_time_signature` and `generate_random_tempo` use the same `(threshold, value)` tuple pattern
- Config files at repo root: JSON files loaded inline with `json.load(open(...))` — config module wraps these

### Integration Points
- All 5 time-sig functions called from within `music_gen.py` — no external consumers yet
- `DurationValidator` in `enhanced_duration_validator.py` has its own time-sig parsing that should delegate to registry
- Phase 1 tests import time-sig functions directly from `music_gen` — imports will need updating
- 32 `print()` calls in `music_gen.py` to replace (0 remaining after Phase 1 cleanup of `musicality_score.py`)

</code_context>

<specifics>
## Specific Ideas

- Config module is explicitly positioned as groundwork for the Phase 6 CLI — design should anticipate `typer` integration
- Time-signature registry should make the system "more flexible and precise" — not just a refactor but an improvement in how meters are handled
- Logging semantics matter because the domain is complex — operators need to understand what the generator is doing at each stage

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 02-stabilize-ii-config-time-signature-registry-logging*
*Context gathered: 2026-04-10*
