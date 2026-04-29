# musicgen

## What This Is

A tool for generating **synthetic music datasets** for machine-learning research. It takes an existing stochastic-procedural music generator (randomized key/tempo/structure → MIDI generators → FluidSynth render → pedalboard FX → mixed WAV) and turns it into a library + CLI that produces 1k–10k reproducible, fully-annotated music samples suitable for training music tagging, source separation, beat/tempo detection, and audio→MIDI transcription models.

## Core Value

**Every generated sample is a complete, reproducible, fully-labeled training example.** If the stems drift from the mix, the MIDI doesn't match the audio, the seed doesn't reproduce, or the annotations are wrong, the dataset is worthless — no matter how musical it sounds.

## Requirements

### Validated

<!-- Existing capabilities inferred from the codebase map. -->

- ✓ Stochastic song-level parameter sampling (key, tempo, time signature, swing) — `music_gen.py:903+`
- ✓ Song arrangement generation from `song_structures.json` (intro / verse / chorus / bridge / outro)
- ✓ Time-signature-aware chord progression generation from `chord_patterns.txt`
- ✓ Markov-style melody generation over chord progressions
- ✓ Bassline generation keyed to chords + melody
- ✓ Drum/beat generation from `beat_roll_patterns_<sig>.txt` with swing offset
- ✓ Per-layer MIDI output (beat / melody / harmony / bassline)
- ✓ FluidSynth rendering with random per-layer soundfont selection from `sf/<layer>/`
- ✓ Per-layer pedalboard FX chains from `*_fx.json` specs
- ✓ Probabilistic per-part layer inclusion from `inst_probabilities.json`
- ✓ Per-layer volume + panning from `levels.json`
- ✓ Per-part mixdown + full-song concatenation → single WAV
- ✓ Time-signature-aware note duration validation (`enhanced_duration_validator.DurationValidator`)
- ✓ Post-generation audio quality scoring via `librosa` (`musicality_score.MusicalityAnalyzer`) — tempo, harmony, rhythm, timbre, noise
- ✓ Support for 2/4, 3/4, 4/4, 5/4, 6/8, 7/8, 12/8 time signatures

### Active

<!-- Priority order set by user: 1 Stabilize → 3 Productize → 2 Extend → 4 Research. -->

**Stabilize (priority 1)**

- [ ] Wrap bottom-of-file execution in `if __name__ == '__main__':` so `music_gen.py` is importable without side effects (`music_gen.py:1158-1161`)
- [ ] Replace `from music21 import *` with explicit imports (`music_gen.py:2`)
- [ ] Replace bare `except:` blocks in `musicality_score.py:66, 94, 173, 205, 239` with narrow exceptions + structured logging
- [ ] Introduce `logging` module throughout `music_gen.py` (replace 32+ `print` calls)
- [ ] Add a `pytest`-based test suite; unit-test the validators, samplers, and pattern parsers first
- [ ] Extract hardcoded paths (soundfont dirs, FX files, config files) into a single config module
- [ ] Refactor `mix_and_save` (`music_gen.py:758-901`, ~143 lines) into smaller functions
- [ ] Consolidate time-signature handling so adding a signature touches one registry instead of five locations
- [ ] Remove dead imports (`glob`, `Pool`, `cpu_count`, `time`) and dead variables

**Productize (priority 2)**

- [ ] Python library API: e.g. `musicgen.generate(seed, config) -> GenerationResult`
- [ ] CLI tool: `python -m musicgen generate --count N --out DIR --seed S [...]`
- [ ] Fully deterministic generation — same seed produces bit-identical WAV + MIDI + metadata
- [ ] Parallel sample generation across CPUs (wire up the already-imported `multiprocessing.Pool`)
- [ ] Parallel intra-sample FluidSynth rendering (the 4 sequential `midi_to_audio` calls at `music_gen.py:828-837`)
- [ ] Resumable batch runs (checkpoint progress; restart skips completed samples)
- [ ] Per-sample output directory with deterministic naming (fix the broken `song_name[:20]` truncation at `music_gen.py:1143` that drops the UUID)
- [ ] Per-sample rich annotations JSON: key, tempo, time signature, swing, song arrangement, per-part chord progression, beat times, active layers, soundfonts used, FX chain params, musicality scores
- [ ] Per-layer stem WAVs persisted alongside final mix (required for source-separation training)
- [ ] MIDI files persisted alongside audio (required for transcription training)
- [ ] Top-level dataset manifest (`manifest.jsonl`) indexing all samples
- [ ] Configurable output mode: keep-stems / mix-only / stems-only / midi-only

**Extend (priority 3)**

- [ ] Broaden musical vocabulary so the dataset covers more of the label space ML models care about (genres, instrumentation, moods)
- [ ] Richer beat / drum patterns per genre
- [ ] Richer chord vocabulary and progression logic
- [ ] Additional time signatures or uncommon meters as needed for dataset diversity
- [ ] (TBD — scope set after stabilize + productize)

**Research (priority 4)**

- [ ] Improve musical quality without gating: smarter Markov, better harmonic logic, or ML-assisted generators
- [ ] Keep `MusicalityAnalyzer` decorative — score every sample and store it as a label; do **not** regenerate on low scores (users filter downstream)
- [ ] (TBD — scope set after productize)

### Out of Scope

- **Real-time playback / live performance** — this is a batch dataset tool, not a performer.
- **Web UI / HTTP API / queue service** — Python library + CLI is the entire interface surface. Explicitly rejected.
- **Human-in-the-loop quality gating / regeneration loop** — `MusicalityAnalyzer` stays informational. Samples are labeled with their score, never rejected. Chosen to keep generation throughput predictable.
- **Commercial-grade musical realism** — "good enough to train on," not "good enough to ship as a song."
- **Training the ML models themselves** — this project produces datasets; consuming them is downstream.
- **Distributed / cluster-scale generation (100k+)** — target is 1k–10k per run. Design should not preclude scaling later, but no queue/scheduler work in this project.
- **Redistributable audio dataset (for now)** — soundfonts under `sf/` have mixed / unclear licensing. Generated audio is assumed personal / internal use until a license audit happens. MIDI + annotations are safe to publish; audio is not.
- **Vocal layer / lyrics / singing synthesis** — out of scope unless explicitly added to Extend later.
- **Rewriting the generator in another language** — staying in Python.

## Context

**Codebase origin.** This started as a single-file hobby music generator (`music_gen.py`, 1161 lines, ~44 KB). It already produces reasonable-sounding WAVs end-to-end but was never designed as a library: bottom-of-file execution, wildcard imports, hardcoded paths, zero tests, `print`-based logging, scattered TODOs, a `multiprocessing` import that's never used, and a `mix_and_save` function that owns too much of the pipeline. A full technical-debt inventory lives in `.planning/codebase/CONCERNS.md` (20 prioritized issues).

**Codebase map.** `.planning/codebase/` contains 7 structured documents (STACK, INTEGRATIONS, ARCHITECTURE, STRUCTURE, CONVENTIONS, TESTING, CONCERNS) written during `/gsd-map-codebase`. Planning should treat these as authoritative for current-state questions.

**ML user perspective.** The target user is someone (initially the author) building music-ML training sets. They care about: (1) reproducibility — same seed → same dataset; (2) label correctness — annotations must describe the audio exactly; (3) throughput — generating 1k–10k samples in a reasonable wall-clock time; (4) label diversity — enough variance in key/tempo/time-sig/genre/instrumentation to train a useful model.

**Determinism is hard with FluidSynth.** FluidSynth rendering is not guaranteed bit-reproducible across versions/platforms. "Fully deterministic" must be verified empirically; if FluidSynth drifts, we either pin versions, accept "MIDI + metadata deterministic, audio approximately deterministic," or investigate alternative renderers. This will be decided during the Productize phase, not now.

**Existing validators should become tests.** `verify_pattern_for_time_signature`, `verify_beat_pattern`, `validate_measures`, and `DurationValidator` are already the right shape for unit tests — they are pure functions or pure methods. The first test suite should start there.

## Constraints

- **Tech stack**: Python 3, `midiutil`, `music21`, `midi2audio` + FluidSynth, `pedalboard`, `pydub`, `librosa` — all already in `requirements.txt`. Don't swap the stack during stabilization.
- **Dependencies**: FluidSynth must be installed on the host system (it's a native binary, not a pip package). CI and dataset generators need it too.
- **Performance**: A single song currently takes seconds–minutes depending on arrangement length and FluidSynth speed. 4 sequential FluidSynth calls per part × N parts is the dominant cost. 1k–10k targets require parallelism to be practical.
- **Reproducibility**: Every source of randomness (Python `random`, numpy if introduced, FluidSynth) must be seed-controlled. This includes soundfont selection, layer inclusion probabilities, FX chain construction, and arrangement generation.
- **Backwards compatibility**: Not a concern — there are no external users yet. Refactors can rename, move, and delete freely during stabilization.
- **Licensing**: Generated audio cannot be assumed redistributable until `sf/` soundfonts are audited. Plan artifacts that assume public release should flag this.
- **Scope**: 1k–10k samples per run is the design target. Don't over-engineer for 100k+.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Build a dataset-generation tool (not a performer or product) | User's actual goal is ML training data | — Pending |
| Priority order: Stabilize → Productize → Extend → Research | Can't productize a buggy god-file; can't extend a non-library; research is premature before the pipeline is stable | — Pending |
| Target 1k–10k samples, not 100k+ | Keeps architecture simple; batch + multiprocessing suffices, no queue/scheduler needed | — Pending |
| Python library + CLI (no web UI, no HTTP API) | Matches user workflow; minimizes surface area | — Pending |
| Persist stems + MIDI + mix + annotations for every sample | Required to support all 4 ML target tasks (tagging, separation, beat detection, transcription) simultaneously | — Pending |
| `MusicalityAnalyzer` stays decorative — score-as-label, never regenerate | Keeps throughput predictable; ML users can filter downstream | — Pending |
| Fully-deterministic generation is a requirement | Reproducible ML benchmarks are non-negotiable for datasets | ⚠️ Revisit — FluidSynth bit-reproducibility unverified; may need to weaken to "MIDI + metadata deterministic, audio approximately deterministic" if FluidSynth drifts |
| Per-sample JSON + top-level `manifest.jsonl` index | Simple, inspectable, convertible to HF Datasets / WebDataset later if needed | — Pending |
| Soundfont license audit deferred; treat audio as non-redistributable for now | User flagged `sf/` licenses as mixed / unclear; blocks public release but not internal use | — Pending |
| Start tests at the validators | They are pure functions, no audio deps, and are the safest footholds | — Pending |
| Keep the existing Python stack (midiutil, music21, pedalboard, pydub, librosa, FluidSynth) | No migration during stabilization; churn would dwarf the debt payoff | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-04-08 after initialization*
