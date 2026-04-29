# Research Summary

Synthesis of the four parallel research outputs for musicgen dataset generator.

## Top 5 decisions that should enter the roadmap

1. **Fix two confirmed bugs before any other refactor** — the arrangement re-roll inside `mix_and_save` (PITFALLS P-A) and the `pydub` `.volume`/`.pan` misuse (P-B). Both are verified by direct code inspection. Leaving them in corrupts every sample the project will ever generate and invalidates all downstream separation ground truth.

2. **Consolidate time-signature logic into a single registry before extracting generators.** Time-sig rules live in 5 scattered locations today; every generator touches them. Extract-first would create 4 copies of a bug.

3. **Only one new runtime dependency: `typer`.** Everything else is stdlib or already installed — especially `python-json-logger`, which is already in `requirements.txt` and replaces all 32+ `print()` calls with no churn. `concurrent.futures` handles parallelism without `joblib`.

4. **Determinism contract: MIDI + metadata always bit-identical; audio bit-identical only under pinned FluidSynth binary.** Attempting stronger guarantees is unverified and likely impossible across FluidSynth versions and CPU microarchitectures. Record the FluidSynth version in every `sample.json`.

5. **Per-sample output layout is index-based, not timestamp-based.** `dataset/00042/sample.json` + `stems/` + `midi/`. `sample.json` is always written last as a resume sentinel. Fixes the UUID truncation bug and enables clean resumable runs.

## Core differentiators (from FEATURES.md)

No existing public dataset combines: (a) seed-reproducible generation, (b) all four target ML tasks from one sample (tagging, separation, beat detection, transcription), (c) swing as a quantified label, (d) per-section chord + structure annotations in a synthetic dataset. Slakh2100 is closest prior art but has no beat/downbeat annotations and no reproducibility.

## Critical risks (from PITFALLS.md)

- P-1 FluidSynth cross-version non-determinism → pin binary, SHA regression test.
- P-2 Stems not persisted + broken pydub gain → stem persistence + sum-of-stems assertion.
- P-3 Beat annotation drift from swing + FluidSynth pre-roll → MIDI-tick-derived beats, empirical pre-roll correction.
- P-4 Multiprocessing RNG leakage → per-worker seeding via `sha256(global_seed, index)`.
- P-5 Narrow soundfont pool → timbre overfitting on trained models.

## Build order (from ARCHITECTURE.md)

**Stabilize → Extract+Test → Productize → Extend → Research**, in strict order. Five refactors are blockers for Productize and cannot be skipped: importability guard, config centralization, time-sig registry, `mix_and_save` decomposition, and seed injection.

## Proposed module layout

```
src/musicgen/
  cli.py  __main__.py  config.py
  sampler.py  timesig.py
  generators/{chord,melody,bassline,beat}.py
  renderer.py  mixer.py  annotator.py  beats.py
  writer.py  manifest.py  batch.py  scoring.py
tests/
pyproject.toml
```

## Open questions to resolve during phase planning

1. FluidSynth version currently installed and whether bit-reproducibility holds for single-seed re-runs on one binary.
2. Does `music21` (wildcard-imported) use global `random` state? Audit before seed injection.
3. Python version floor (`>=3.9` assumed).
4. `uuid` stub entry in `requirements.txt` should likely be removed (uuid is stdlib).
5. Absent-layer stem policy — explicit silent file or accounted for in the sum assertion.

## Cross-doc references

- STACK.md — libraries, versions, rationale, anti-choices
- FEATURES.md — per-task feature mapping, anti-features, MVP priority list
- ARCHITECTURE.md — module boundaries, data flow diagram, seed propagation, build order
- PITFALLS.md — phase-mapped risk catalog including two confirmed-in-code bugs
