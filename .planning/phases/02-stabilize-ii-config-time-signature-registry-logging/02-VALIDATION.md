---
phase: 2
slug: stabilize-ii-config-time-signature-registry-logging
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-10
---

# Phase 2 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> Populated from 02-RESEARCH.md §Validation Architecture by gsd-planner.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | `pytest.ini` (installed Phase 1) |
| **Quick run command** | `pytest tests/test_timesig_registry.py tests/test_config.py tests/test_music_gen_logging.py -x -q` |
| **Full suite command** | `pytest -q` |
| **Estimated runtime** | ~{TBD by planner} seconds |

---

## Sampling Rate

- **After every task commit:** Run `{quick run command}`
- **After every plan wave:** Run `{full suite command}`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** {TBD} seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| {TBD by planner} | | | | | | | | | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_config.py` — stubs for R-S5 (config module owns all paths)
- [ ] `tests/test_timesig_registry.py` — stubs for R-S6 (registry is single source of truth)
- [ ] `tests/test_music_gen_logging.py` — stubs for R-S7 (no print() in music_gen.py)
- [ ] Existing `tests/test_time_signature.py` + `tests/test_duration_validator.py` stay green via thin wrappers

*Framework already installed in Phase 1 (plan 01-04).*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Soundfont pool warning fires when `sf/<layer>/` has <3 files | R-S9 | Requires manipulating the on-disk `sf/` tree | Rename files in `sf/beat/` to leave <3, run `python -m music_gen`, confirm WARNING logged |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter (planner fills task rows first)

**Approval:** pending
