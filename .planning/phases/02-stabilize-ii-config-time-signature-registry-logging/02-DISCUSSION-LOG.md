# Phase 2: Stabilize II — config + time-signature registry + logging - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-10
**Phase:** 02-stabilize-ii-config-time-signature-registry-logging
**Areas discussed:** Config module design, Time-signature registry data model, Logging style, Soundfont pool detection

---

## Config Module Design

| Option | Description | Selected |
|--------|-------------|----------|
| Flat constants module | Simple module with path constants, no override support | |
| Dataclass with overrides | Supports env vars + CLI args with layered precedence | ✓ |
| New config file format | Introduce musicgen.toml/yaml as unified config | |

**User's choice:** Dataclass with env var + CLI arg override support. Unified access layer on top of existing JSON files (no new config format). Positioned as groundwork for robust CLI.
**Notes:** Precedence confirmed as CLI > env > file > defaults (standard).

---

## Time-Signature Registry Data Model

| Option | Description | Selected |
|--------|-------------|----------|
| Data-only registry | Registry provides data, existing functions query it | |
| Full-ownership registry | Registry owns all validation logic, functions become thin wrappers or disappear | ✓ |

**User's choice:** Full-ownership dataclass registry. Adding a time signature touches one file only. Design for flexibility and precision.
**Notes:** User emphasized making the system "more flexible and precise" — not just a refactor.

---

## Logging Style

| Option | Description | Selected |
|--------|-------------|----------|
| All INFO | Replace all prints with logging.info | |
| Semantic levels | DEBUG/INFO/WARNING/ERROR differentiated by purpose | ✓ |
| JSON default | python-json-logger as default output format | |

**User's choice:** Semantic log levels. DEBUG for state dumps, INFO for progress, WARNING for recoverable oddities, ERROR for quality-affecting failures. JSON format deferred to Phase 6 batch mode.
**Notes:** User confirmed the proposed mapping without changes.

---

## Soundfont Pool Detection

| Option | Description | Selected |
|--------|-------------|----------|
| Hard error | Raise exception if pool too small | |
| Informational warning at config load | Log warning, continue normally | ✓ |
| Lazy check on first use | Warn only when soundfont actually requested | |

**User's choice:** Informational only, fires at config load time.
**Notes:** No follow-up needed — clear decision.

---

## Claude's Discretion

- Internal module naming
- Dataclass field names and registry API shape
- Frozen vs mutable dataclasses
- Logger naming convention

## Deferred Ideas

None
