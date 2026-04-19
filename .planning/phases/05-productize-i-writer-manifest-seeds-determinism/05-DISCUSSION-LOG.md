# Phase 5: Productize I — writer, manifest, seed discipline, determinism - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in 05-CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-19
**Phase:** 05-productize-i-writer-manifest-seeds-determinism
**Mode:** `--auto` (Claude selected recommended defaults for every gray area)
**Areas discussed:** Module split, Writer design, Manifest design, Seed discipline, Annotator schema completion, Sum-of-stems assertion, Split assignment, Determinism regression test, Library API (generate), Orchestrator migration, Testing strategy, Phase 6 boundary

---

## Module split + file inventory

| Option | Description | Selected |
|--------|-------------|----------|
| 3 modules (writer, manifest, seeds) + re-export from `__init__.py` | Minimal surface; API lives inline in `__init__.py` | |
| 4 modules (writer, manifest, seeds, api) + export | Explicit `api.py`; `__init__.py` is a thin re-export | ✓ |
| 6+ modules (split each further) | More files, more import boilerplate | |

**Auto-selected:** Option 2 (4 modules). Matches the Phase 3 pattern of per-concern files (one module per responsibility). D-01.

### Should `musicality_score.py` move into the package this phase?

| Option | Description | Selected |
|--------|-------------|----------|
| Yes, move to `src/musicgen/musicality.py` | Close the Phase 3 D-11 / Phase 4 D-04 deferral | ✓ |
| No, stay at repo root until Phase 6 | Keep the scope tighter | |

**Auto-selected:** Move now. Library API `from musicgen import generate` requires all pipeline components under the package. D-03.

---

## Writer design

### Zero-padding width for index-based directory names

| Option | Description | Selected |
|--------|-------------|----------|
| 4 digits (0000–9999) | Covers 1k–10k target exactly | |
| 6 digits (000000–999999) | 100× headroom for v0.2+ | ✓ |
| 8 digits | Excessive for current scale | |

**Auto-selected:** 6 digits. Matches ARCHITECTURE.md diagram; supports 1M-sample corpora without a sharded-layout redesign. D-05.

### File-write order (atomicity vs. resume-sentinel invariant)

| Option | Description | Selected |
|--------|-------------|----------|
| Write all files, then `sample.json` last (sentinel) | Simple; `sample.json` presence = complete sample | ✓ |
| Atomic `.tmp` + rename per file | More overhead; no correctness gain since we have the sentinel | |
| Write manifest first, `sample.json` later | Reverses D-16 projection | |

**Auto-selected:** `sample.json` last as sentinel. Matches R-P1 literal text ("always written last (resume sentinel)"). D-04.

### Stem-and-MIDI concatenation across parts

| Option | Description | Selected |
|--------|-------------|----------|
| Concatenate via pydub `AudioSegment` (WAV) + mido (MIDI) | Matches existing `mixer.concat_parts` idiom | ✓ |
| Keep per-part stems in the sample dir (`stems/verse/beat.wav`) | Violates R-P1 literal layout | |
| Raw int16 array concat via `scipy.io.wavfile` | Faster but divergent from mixer style | |

**Auto-selected:** pydub + mido. Style consistency with `mixer.concat_parts`; correctness over microoptimization. D-06, D-07.

### Path representation in `sample.json`

| Option | Description | Selected |
|--------|-------------|----------|
| Absolute paths | Easy to read, but unportable | |
| Dataset-root-relative | E.g. `"000042/mix.wav"` | |
| Per-sample-dir-relative | E.g. `"mix.wav"`, `"stems/beat.wav"` | ✓ |

**Auto-selected:** Per-sample-dir-relative. Matches the idiom of ML dataset loaders that open one `sample.json` per dir. D-11.

---

## Manifest design

### Lock abstraction

| Option | Description | Selected |
|--------|-------------|----------|
| `threading.Lock()` hardcoded | Fine for Phase 5 only | |
| Accept `lock: ContextManager` parameter | Phase 6 swaps in `multiprocessing.Manager().Lock()` | ✓ |
| `multiprocessing.Lock()` from day 1 | Overhead without benefit this phase | |

**Auto-selected:** Parametrized lock. R-P5 literal says "appended under a `multiprocessing.Manager().Lock()`", but the safe way to land this now is an abstraction that Phase 6 fills — avoids single-process spawning a Manager unnecessarily. D-14.

### Manifest semantics: append-only vs. compaction

| Option | Description | Selected |
|--------|-------------|----------|
| Append-only, last-status-wins on retries | Simple; O(1) writes | ✓ |
| Rewrite lines in place on retry | Requires line-level file edits; racy | |
| Separate failed-manifest file | Extra read path complexity | |

**Auto-selected:** Append-only. Simplest invariant that preserves R-P11 correctness. D-15.

### `is_sample_complete` check: filesystem vs. manifest

| Option | Description | Selected |
|--------|-------------|----------|
| Check `sample.json` existence (filesystem) | Sentinel-driven; manifest-state-insensitive | ✓ |
| Grep the manifest for matching index | Requires locking the manifest on reads | |

**Auto-selected:** Filesystem. Manifest is a projection (D-16); sentinel is the truth. D-16.

---

## Seed discipline

### `derive_sample_seed` and `make_rngs` signatures

| Option | Description | Selected |
|--------|-------------|----------|
| Verbatim from ARCHITECTURE.md | `sha256` + XOR pattern spec'd in research | ✓ |
| Custom variant with different domain keys | Drift from research | |
| PCG-based RNG instead of `random.Random` | Numpy-style, but diverges from existing `rng: random.Random` signatures | |

**Auto-selected:** Verbatim per research. D-17, D-18.

### RNG domain-to-callsite mapping

| Option | Description | Selected |
|--------|-------------|----------|
| Map by pipeline stage: `params`/`generators`/`soundfonts`/`fx`/`mix` | Matches research domain names | ✓ |
| Single RNG for all | Defeats the per-domain independence rationale | |
| One RNG per function | Too granular; harder to reason about | |

**Auto-selected:** Five-domain map. D-19.

### music21 / librosa `save_random_state()` wrap

| Option | Description | Selected |
|--------|-------------|----------|
| Skip — Phase 3 proved no global-random leak today | Tightest scope | |
| Add wrap around `musicality_score.get_musicality_score` | Defense in depth — free insurance against dep upgrades | ✓ |
| Wrap every third-party call site | Excessive | |

**Auto-selected:** Wrap musicality only. REQUIREMENTS R-P7 bullet 5 explicitly mandates the save/restore pattern. D-20.

### Global seed default when `config.global_seed is None`

| Option | Description | Selected |
|--------|-------------|----------|
| Raise `ValueError` — require explicit seed | Enforces determinism contract visibly | ✓ |
| Default to `secrets.randbits(64)`, log it | Convenient but seed is invisible downstream | |
| Default to `time.time_ns()` | Cheap but the same problem | |

**Auto-selected:** Require explicit. Library's core value prop is reproducibility; silent defaults defeat that. D-21.

---

## Annotator schema completion

### Which field value goes into `seed`?

| Option | Description | Selected |
|--------|-------------|----------|
| `global_seed` | Matches "global" mental model | |
| `sample_seed` | Uniquely identifies this sample's RNG basin | ✓ |
| Both (`seed` + `global_seed`) | Schema bloat; `sample_index` already implies this | |

**Auto-selected:** `sample_seed`. The sample-derived seed is what RNGs are seeded with; it's the most specific identifier for "re-run this exact sample". D-22.

### `musicgen_version` source

| Option | Description | Selected |
|--------|-------------|----------|
| Hardcoded `"0.1.0"` literal | Drifts from pyproject.toml | |
| `importlib.metadata.version("musicgen")` | Single source of truth | ✓ |
| `__version__` in `__init__.py` | Duplicates pyproject | |

**Auto-selected:** `importlib.metadata`. D-22.

### `sample.json` serialization

| Option | Description | Selected |
|--------|-------------|----------|
| Default `json.dump(f, indent=2)` | Human-readable but key order varies | |
| `sort_keys=True, indent=2, separators=(",", ": ")` | Canonical, hashable | ✓ |
| Single-line compact output | Harder to debug | |

**Auto-selected:** Canonical with sorting. Load-bearing for D-28's byte-identical-across-runs assertion. D-23.

---

## Sum-of-stems assertion

### Assertion location

| Option | Description | Selected |
|--------|-------------|----------|
| Inside `writer.write_sample`, before sentinel | Fails early; sentinel not written on failure | ✓ |
| After sentinel write (post-hoc check) | Breaks the sentinel invariant | |
| In a separate `verify_sample` module | Overkill; adds an import layer | |

**Auto-selected:** Inside writer, pre-sentinel. D-24.

### Epsilon tolerance

| Option | Description | Selected |
|--------|-------------|----------|
| `1e-3` (~−60 dBFS) | Catches real bugs, survives int16 quantization | ✓ |
| `1e-4` (~−80 dBFS) | Risks flaky test on a different pedalboard version | |
| `1e-6` (~−120 dBFS) | Below numerical precision; guaranteed flaky | |

**Auto-selected:** `1e-3`. Configurable via `config.sum_of_stems_epsilon`. D-25.

### Failure behavior

| Option | Description | Selected |
|--------|-------------|----------|
| Write `status: failed` in manifest, skip `sample.json` | Matches R-P2 + D-04 sentinel invariant | ✓ |
| Write partial `sample.json` with failure marker | Breaks sentinel invariant | |
| Raise — abort the caller | Phase 6 batch needs per-sample isolation; library API too | |

**Auto-selected:** Manifest-failed + no sentinel. D-24.

---

## Split assignment

### Hash algorithm

| Option | Description | Selected |
|--------|-------------|----------|
| `sha256(f"split:{sample_seed}")[:4] % 10000 / 100.0` | Domain-prefixed hash, 0.01% resolution | ✓ |
| `sample_seed % 10000 / 100.0` | Correlates with seed structure | |
| `hash(sample_seed)` (Python built-in) | Nondeterministic across Python versions | |

**Auto-selected:** Prefixed sha256. D-26.

### Default ratios

| Option | Description | Selected |
|--------|-------------|----------|
| `(0.8, 0.1, 0.1)` | REQUIREMENTS R-P6 example | ✓ |
| `(0.7, 0.15, 0.15)` | More common in some benchmarks | |
| No default, require config | Friction for quickstart | |

**Auto-selected:** 80/10/10 default, configurable via `config.split_ratios`. D-27.

---

## Determinism regression test

### Golden files per generation artifact

| Option | Description | Selected |
|--------|-------------|----------|
| SHA-256 of mix + 4 MIDIs + sample.json + FluidSynth version guard | Full coverage per R-P8 | ✓ |
| SHA-256 of mix only | Misses MIDI / annotation drift | |
| Binary blob commit (actual WAV) | Large artifacts in git | |

**Auto-selected:** 6 SHA-256 files + version guard. D-28, D-29.

### Golden regeneration workflow

| Option | Description | Selected |
|--------|-------------|----------|
| `pytest --regen-goldens` flag | Explicit maintainer action | ✓ |
| Auto-update on first run | Invisible drift | |
| Manual script | More friction, more docs | |

**Auto-selected:** pytest flag. D-32.

### Hash-stability in-process cross-check (independent of FluidSynth)

| Option | Description | Selected |
|--------|-------------|----------|
| Second test: run `generate` twice, hash `sample.json` bytes | Catches our-code-nondeterminism without FluidSynth | ✓ |
| Rely on the slow golden test alone | Can't run on FluidSynth-less CI | |

**Auto-selected:** Add both tests. D-30.

---

## Library API (generate)

### Scope this phase

| Option | Description | Selected |
|--------|-------------|----------|
| `generate(config)` single-sample only | Phase 6 adds `generate_batch` | ✓ |
| `generate` + `generate_batch` both now | Blurs phase boundary; harder to isolate determinism bugs | |
| Just the internal functions, no public API | Deferred `__init__.py` exposure | |

**Auto-selected:** Single-sample only. D-31.

### Working directory strategy

| Option | Description | Selected |
|--------|-------------|----------|
| `tempfile.mkdtemp(prefix="musicgen-")` | Safe under concurrency; survives the call | ✓ |
| `<dataset_root>/.tmp/<index>/` | Pollutes dataset root if interrupted | |
| In-memory only | Possible for MIDI but not WAVs at scale | |

**Auto-selected:** `tempfile.mkdtemp`. D-31 step 4.

### Resume behavior

| Option | Description | Selected |
|--------|-------------|----------|
| Short-circuit when `sample.json` exists (reconstruct SampleResult) | Matches sentinel invariant | ✓ |
| Always re-run | Defeats the point of resumability | |
| Error on collision | Friction for batch restarts | |

**Auto-selected:** Sentinel short-circuit. D-31 step 3.

---

## Orchestrator migration

### Fate of `music_gen.py` + `create_song`

| Option | Description | Selected |
|--------|-------------|----------|
| Delete `create_song` / `generate_song_parts` / `generate_song`; keep smoke-test entry point | Close the god-file concern step-by-step | ✓ |
| Delete the whole file | Breaks the existing `python music_gen.py` workflow | |
| Keep everything, make `api.generate` a wrapper | Perpetuates duplicate orchestration | |

**Auto-selected:** Delete the three orchestration functions, keep `music_gen.py` as a 40-line smoke wrapper. Phase 6 deletes the file fully when the CLI replaces the smoke path. D-33, D-34.

### `__init__.py` exports

| Option | Description | Selected |
|--------|-------------|----------|
| Export `generate`, `Config`, `SampleResult`, `__version__` | Clean library surface | ✓ |
| Keep `__init__.py` empty | Users must import via submodules | |
| Also export internal helpers (`derive_sample_seed` etc.) | API surface creep | |

**Auto-selected:** Four public names. D-35.

---

## Testing strategy

Six new test files, auto-selected as a unit (D-36..D-41):

- `test_seeds.py` — pure-function unit tests (derive_sample_seed, make_rngs, save_random_state).
- `test_writer.py` — `tmp_path` + synthesized WAV/MIDI fixtures; verifies layout, relative paths, silent-stem concat, sum-of-stems fault injection.
- `test_manifest.py` — concurrent-append thread safety + `is_sample_complete` sentinel behavior.
- `test_split.py` — 10k-sample empirical ratio check within 2% of declared.
- `test_api.py` — split fast/slow; resume short-circuit; idempotence; layout validation.
- `test_determinism_golden.py` — the `@pytest.mark.slow` regression test gated on FluidSynth version (D-29). Plus a fast same-process cross-check (D-30).

No alternative seriously considered — each maps 1:1 to a deliverable, and the Phase 4 testing playbook is the template.

---

## Claude's Discretion

Areas where the user would say "you decide":
- `SampleResult` field names (shape fixed by D-02, naming aesthetic)
- Exact pydub API for stem concatenation (AudioSegment vs. scipy.wavfile)
- Dataset-root field name (`dataset_root` chosen over `output_dir` — matches terminology across REQUIREMENTS/ARCHITECTURE)
- `musicgen_version` uninstalled fallback (`"0.1.0+uninstalled"` vs. raise — chose fallback)
- Working-dir location (tempfile vs. dataset-root-hidden — chose tempfile)
- Silent-MIDI representation (zero-velocity note vs. empty track with end_of_track meta)
- Split name casing (lowercase chosen)
- Split-bucket modulo base (10000 chosen for 0.01% precision)

## Deferred Ideas

All deferred ideas (moved to 05-CONTEXT.md `<deferred>` section):
- `generate_batch` via ProcessPoolExecutor (Phase 6)
- `typer` full CLI (Phase 6)
- FluidSynth pre-roll calibration (Phase 6)
- `--output-mode` flag (Phase 6)
- Structured JSON progress logs (Phase 6)
- `musicgen clean --failed` (Phase 6)
- Sharded layout (v0.2+)
- Custom split labels (v0.2+)
- Compressed annotations (v0.2+)
- Automated golden regeneration CI (post-v0.1)
