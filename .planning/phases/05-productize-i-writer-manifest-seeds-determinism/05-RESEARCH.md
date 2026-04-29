# Phase 5: Productize I — writer, manifest, seed discipline, determinism - Research

**Researched:** 2026-04-19
**Domain:** stdlib plumbing (hashlib / threading / tempfile / shutil / importlib.metadata / json / contextlib) + scipy.io.wavfile + pydub.AudioSegment + mido.MidiFile + numpy dtype handling + pytest --addoption
**Confidence:** HIGH on the mechanics (every load-bearing claim below was probed against the installed venv during research). MEDIUM on FluidSynth cross-run WAV bit-identity (it's an empirical claim that can only be fully verified by capturing goldens during execution — acknowledged and accommodated by the D-29 version-pinned skip gate).

## Summary

Phase 5 is structurally a **stdlib-plumbing phase with two subtle integration gotchas**: (a) concatenating per-part MIDIs requires an **explicit absolute-tick walk**, not a naive track-append — because `midiutil` writes note tracks whose total accumulated tick count can be **less than the part's audio duration** (trailing zero-hits advance `current_time` but emit no MIDI event, leaving the track short); and (b) the sum-of-stems assertion must read via `scipy.io.wavfile` into an **int32 accumulator** (not int16 or float32) before subtracting from the mix to avoid silent wrap-around overflow on loud signals. Everything else Phase 5 needs — index-based directory layout, atomic sentinel write, `threading.Lock`-based append-only manifest, SHA-256 split hash, `random.Random(seed^constant)` XOR-derived named RNGs, `importlib.metadata.version()` with `PackageNotFoundError` fallback, `tempfile.mkdtemp` working dirs, `save_random_state()` contextmanager — is stdlib, deterministic, and well-understood.

The CONTEXT.md 43 locked decisions are all implementable as written. One CONTEXT.md claim needs the planner's attention as a **corroborated but not surprising finding**: the sum-of-stems assertion is sharp enough to validate the R-S4 gain/pan fix (a bit-identical int16 round-trip via pydub is verified — see Research Question 2) and loose enough (ε=1e-3 normalized float) to survive pedalboard's tiny numerical drift on FX-applied layers. The CONTEXT.md writer-MIDI-concat approach (D-07: "rewrite msg.time to be tick-cumulative") is correct in spirit but needs the **absolute-tick-walk implementation detail** documented here for the planner so the task description can be specific.

**Primary recommendation:** build the 4 new modules + musicality move in the wave order already laid out in the additional_context (Wave 0 test scaffolding → Wave 1 seeds.py → Wave 2 musicality move + writer + manifest in parallel → Wave 3 api.py → Wave 4 orchestrator migration → Wave 5 determinism goldens). Within Wave 2, plan `writer.py` as **two sub-concerns** (layout/atomic-write skeleton first, then stem-concat + MIDI-concat helpers second) because the MIDI concat is the subtlest code this phase ships. Keep the sum-of-stems assertion **on the final-layout paths** (after writer has rewritten to dataset-root — that's where the bug would manifest), not on the working-dir paths.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Per-sample filesystem layout (R-P1) | writer.py (Python library) | — | Phase 4 `mix_part` / `concat_parts` already own working-dir paths; writer owns dataset-root relocation |
| Seed derivation + RNG factory (R-P7) | seeds.py (pure functions) | — | No I/O; hashlib + random.Random; easiest to unit-test |
| Manifest append (R-P5) | manifest.py (Python library) | — | File I/O under lock; the lock abstraction is the interface Phase 6 extends |
| Schema completion (R-P4 residuals) | api.py (injects into annotator) | annotator.py (Phase 4) | Annotator stays pure; api.py fills Phase-5 kwargs |
| Split assignment (R-P6) | api.py → seeds.py helper | — | `assign_split` is a pure hash function; lives next to `derive_sample_seed` |
| Library entry point (R-P12 single-sample) | api.py | __init__.py re-export | Batch (R-P10) is Phase 6; this is the one-sample primitive |
| Musicality scoring wrap (defense-in-depth) | api.py uses `save_random_state()` | musicality.py (relocated) | `save_random_state` lives in seeds.py; api.py composes |
| Atomic sentinel discipline (R-P1) | writer.write_sample() | — | `sample.json` last; `os.rename` from temp-name to final name for atomicity |
| Determinism regression test (R-Q3, R-P8) | tests/test_determinism_golden.py | tests/conftest.py (--regen-goldens flag) | Test infrastructure, not production code |

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `hashlib` (stdlib) | 3.10+ | SHA-256 for seed derivation + split hash + golden-fixture checksums | The one cryptographic primitive we need is in stdlib; no reason to import anything else. `[VERIFIED: stdlib]` |
| `random` (stdlib) | 3.10+ | `random.Random(seed)` per-domain; `random.getstate()`/`setstate()` for `save_random_state` | Phase 3+4 already use `random.Random` via injected `rng`; Phase 5 extends without changing the contract. `[VERIFIED: Phase 3 D-07, Phase 4 D-17]` |
| `threading` (stdlib) | 3.10+ | `threading.Lock()` default for single-process `ManifestWriter` | Cheap, minimal overhead. CONTEXT D-14 locks this. `[VERIFIED: probe — `threading.Lock()` supports `__enter__/__exit__` for `with lock:`]` |
| `multiprocessing` (stdlib) | 3.10+ | NOT USED THIS PHASE. `multiprocessing.Manager().Lock()` is Phase 6 (`ProcessPoolExecutor` workers). The `ContextManager` lock parameter accepts both without branching. | `[VERIFIED: probe — Manager().Lock() is `AcquirerProxy`, supports `__enter__/__exit__`]` |
| `pathlib` / `os.path` (stdlib) | 3.10+ | Directory creation, path joins, rename | CONVENTIONS use `os.path`; stay consistent. `[CITED: src/musicgen/renderer.py:126]` |
| `importlib.metadata` (stdlib) | 3.10+ | `importlib.metadata.version("musicgen")` for `__version__` + annotation field | Stdlib since 3.8. Fallback: `PackageNotFoundError` → `"0.1.0+uninstalled"` string. `[VERIFIED: probe in venv yielded "0.1.0" since the package is editable-installed]` |
| `contextlib` (stdlib) | 3.10+ | `@contextlib.contextmanager` decorator for `save_random_state()` | CONTEXT D-20 spells this out verbatim. `[CITED: CONTEXT.md D-20]` |
| `shutil` (stdlib) | 3.10+ | `shutil.rmtree(working_dir, ignore_errors=True)` | Safe-on-nonexistent with `ignore_errors=True`. `[VERIFIED: stdlib docs]` |
| `tempfile` (stdlib) | 3.10+ | `tempfile.mkdtemp(prefix="musicgen-")` for working dirs | Default `/tmp/musicgen-<random8>/`. Safe even if caller is concurrent (mkdtemp collision-free). `[VERIFIED: probe]` |
| `json` (stdlib) | 3.10+ | Manifest append + sample.json write; `sort_keys=True` for byte-stable canonicalization | CONTEXT D-23 pins `json.dump(..., sort_keys=True, indent=2, separators=(",", ": "))`. `[VERIFIED: probe — sort_keys yields byte-stable output]` |
| `numpy>=1.20.0` | pyproject pin | Int32 accumulator for sum-of-stems; read via `scipy.io.wavfile` | Already required for rendering/annotator; no new pin. `[CITED: pyproject.toml:14]` |
| `scipy>=1.7.0` | pyproject pin | `scipy.io.wavfile.read(path) -> (rate, ndarray)` for stem-sum assertion | Already a direct dep (`scipy.stats.entropy` in `musicality_score.py`); no new pin. `[VERIFIED: pyproject.toml:15]` |
| `mido>=1.3.3` | pyproject pin | `mido.MidiFile(path)` read + `MidiTrack` write for MIDI concat across parts | Already a direct dep (Plan 04-00 added for `beats.py`); no new pin. `[VERIFIED: pyproject.toml:22]` |
| `pydub>=0.25.1` | pyproject pin | `AudioSegment.from_wav + AudioSegment.from_wav` for stem concat across parts | Already a direct dep; concat pattern matches `mixer.concat_parts`. `[VERIFIED: pyproject.toml:19]` |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `pytest>=8.0` | `[dev]` pin | Test runner; `pytest_addoption` hook for `--regen-goldens` flag | All 6 new test files + goldens regeneration |
| `pytest-xdist>=3.5` | `[dev]` pin | Parallel test execution — `pytest -n auto` | Enabled today; keep (existing dev dep) |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `threading.Lock()` default | `multiprocessing.Manager().Lock()` always | CONTEXT D-14 correctly rejects this — Manager lock has ~1ms+ per-acquire overhead and pulls in a Manager subprocess, meaningless in Phase 5's single-process flow. Plain `threading.Lock()` has zero marginal cost. |
| `scipy.io.wavfile.read` | `pydub.AudioSegment.raw_data` + np.frombuffer | Scipy returns the ndarray directly with dtype + shape; pydub requires manual dtype inference from `sample_width`. Scipy wins on ergonomics; both produce bit-identical arrays (probed). |
| `mido` read + absolute-tick walk (recommended) | midiutil for concat (write-only) | `midiutil` cannot read MIDIs back — it's write-only. `mido` is the only Python MIDI library in the current dep graph with read support. `[VERIFIED: probe]` |
| `json.dump(..., sort_keys=True)` (D-23 locked) | `json.JSONEncoder(sort_keys=True)` subclass | Equivalent; no behavioral difference. Use the simpler form. |
| `tempfile.mkdtemp` under `/tmp` | `<dataset_root>/.tmp/<index>` | CONTEXT's "Claude's Discretion" note picks the `tempfile` route. `/tmp` is bounded (tmpfs eviction on reboot) and survives interruption; keeps `dataset_root` clean of partial samples. Also survives resume-detection (scans `dataset_root`, not `/tmp`). |
| `os.rename` for sentinel | `pathlib.Path.rename` | Equivalent under the hood; CONVENTIONS use `os.path` style — stay consistent. |

**Installation:** No new packages. Everything required is already in `pyproject.toml`.

**Version verification:** No new registry lookups needed — Phase 5 adds no runtime deps.

## Architecture Patterns

### System Architecture Diagram

```
                ┌────────────────────────────────────────────────┐
                │            api.generate(config)                │
                │                                                │
                │  1. Validate config (D-21, D-27)               │
                │  2. sample_seed = derive_sample_seed(seed, idx)│  ← seeds.py
                │  3. rngs = make_rngs(sample_seed)              │  ← seeds.py
                │  4. IS_COMPLETE = mw.is_sample_complete(...)   │  ← manifest.py
                │     └→ if True: reconstruct SampleResult       │
                │                                                │
                │  5. working_dir = tempfile.mkdtemp(...)        │
                │  6. SongParams.sample(rngs["params"], cfg)     │  ← sampler (Phase 3)
                │  7. _generate_all_midi(rngs["generators"],...) │  ← generators (Phase 3)
                │  8. render_stems(rngs["soundfonts"],...)       │  ← renderer (Phase 4)
                │  9. build_fx_boards(rngs["fx"], ...)           │  ← mixer (Phase 4)
                │     compute_layer_mask(rngs["mix"], ...)       │  ← mixer (Phase 4)
                │     mix_part + concat_parts                    │  ← mixer (Phase 4)
                │ 10. extract_beat_times + downbeat_times        │  ← beats (Phase 4)
                │ 11. with save_random_state():                  │  ← seeds.py D-20
                │       score = musicality.get_score(mix_wav)    │  ← musicality (moved)
                │ 12. split = assign_split(sample_seed, ratios)  │  ← seeds.py
                │ 13. annotation = annotator.annotate(...,       │
                │        seed=sample_seed,                       │
                │        musicgen_version=importlib.metadata...  │
                │        split=split)                            │  ← annotator (Phase 4)
                │                                                │
                │ 14. writer.write_sample(                       │
                │        dataset_root, sample_index, annotation, │  ← writer.py
                │        working paths, fluidsynth_version=...)  │
                │     │                                          │
                │     ▼                                          │
                │     ┌───────────────────────────────────────┐  │
                │     │ writer.write_sample():                │  │
                │     │  a. mkdir <root>/<idx:06d>/{stems,midi}│ │
                │     │  b. _concat_layer_stems (pydub)       │  │
                │     │  c. _concat_layer_midis (mido abs-tick)│ │
                │     │  d. write mix.wav (copy/rename)       │  │
                │     │  e. _assert_sum_of_stems (scipy+np32) │  │
                │     │  f. rewrite annotation path fields    │  │
                │     │     (relative) — D-11/D-12            │  │
                │     │  g. write sample.json.tmp → rename    │  │
                │     │     (sentinel = atomic rename)        │  │
                │     │  [on failure: NO sample.json written] │  │
                │     └───────────────────────────────────────┘  │
                │                                                │
                │ 15. manifest_writer.append({index, seed,       │
                │        status, split, path, wrote_at, ...})    │  ← manifest.py
                │ 16. shutil.rmtree(working_dir)                 │
                │ 17. return SampleResult(...)                   │
                └────────────────────────────────────────────────┘
```

Data flow in one sentence: `config → sample_seed → per-domain RNGs → sampler+generators+renderer+mixer+beats+annotator (working dir) → writer relocates + concatenates + asserts → sentinel rename → manifest append → cleanup`.

### Recommended Project Structure

```
src/musicgen/
├── __init__.py           # REWRITE: exports generate, Config, SampleResult, __version__ (D-35)
├── api.py                # NEW: generate() entry point (D-31) + SampleResult (D-02)
├── seeds.py              # NEW: derive_sample_seed, make_rngs, assign_split, save_random_state
├── writer.py             # NEW: write_sample, _concat_layer_stems, _concat_layer_midis, _assert_sum_of_stems
├── manifest.py           # NEW: ManifestWriter class (D-14/D-15/D-16)
├── musicality.py         # MOVED from repo-root musicality_score.py (D-03)
├── sampler.py            # UNCHANGED (Phase 3)
├── annotator.py          # UNCHANGED (Phase 4) — api.py passes kwargs
├── mixer.py              # UNCHANGED (Phase 4)
├── renderer.py           # UNCHANGED (Phase 4)
├── beats.py              # UNCHANGED (Phase 4)
├── duration_validator.py # UNCHANGED (Phase 3)
├── cli.py                # UNCHANGED (Phase 3 stub; Phase 6 replaces)
└── generators/           # UNCHANGED (Phase 3)

tests/
├── conftest.py           # NEW: pytest_addoption("--regen-goldens") (D-32)
├── test_seeds.py         # NEW (D-36)
├── test_writer.py        # NEW (D-37)
├── test_manifest.py      # NEW (D-38)
├── test_split.py         # NEW (D-39)
├── test_api.py           # NEW (D-40)
├── test_determinism_golden.py  # NEW (D-41) — @pytest.mark.slow
└── fixtures/determinism/ # NEW (D-28)
    ├── expected_mix.sha256
    ├── expected_midi_{beat,melody,harmony,bassline}.sha256
    ├── expected_sample.sha256
    └── fluidsynth_version.txt

# Also:
music_gen.py              # REDUCE 199 → ~40 lines; delete create_song/generate_song_parts/generate_song (D-34); keep smoke-test __main__ (D-33)
config.py                 # EXTEND with 7 new fields (D-09/D-25/D-27)
musicality_score.py       # DELETED (moved to src/musicgen/musicality.py per D-03)
tests/test_integration_full_generation.py # MIGRATE from create_song(...) to musicgen.generate(Config(...))
```

### Pattern 1: seeds.py — pure-function RNG hierarchy

**What:** Three pure functions + five name constants + one context manager. No I/O, no class state.

**When to use:** Any time Phase 5+ code needs an RNG, it's pulled from `rngs[RNG_XXX]` from the api.py-built dict. Never construct `random.Random()` with a fresh entropy source inside the pipeline; never touch the module-level `random.*` globals.

**Example:**
```python
# Source: CONTEXT.md D-17, D-18, D-20 + .planning/research/ARCHITECTURE.md §Seed/RNG
import contextlib
import hashlib
import random
from typing import Dict

RNG_PARAMS = "params"
RNG_GENERATORS = "generators"
RNG_SOUNDFONTS = "soundfonts"
RNG_FX = "fx"
RNG_MIX = "mix"


def derive_sample_seed(global_seed: int, sample_index: int) -> int:
    """Deterministic per-sample seed from (global_seed, sample_index)."""
    raw = hashlib.sha256(f"{global_seed}:{sample_index}".encode()).digest()
    return int.from_bytes(raw[:8], "big")


def make_rngs(sample_seed: int) -> Dict[str, random.Random]:
    """Five named domain RNGs from one sample_seed via XOR with small constants."""
    return {
        RNG_PARAMS:     random.Random(sample_seed ^ 0x01),
        RNG_GENERATORS: random.Random(sample_seed ^ 0x02),
        RNG_SOUNDFONTS: random.Random(sample_seed ^ 0x03),
        RNG_FX:         random.Random(sample_seed ^ 0x04),
        RNG_MIX:        random.Random(sample_seed ^ 0x05),
    }


def assign_split(sample_seed: int, ratios: tuple) -> str:
    """Deterministic train/valid/test assignment from sample_seed."""
    bucket = int.from_bytes(
        hashlib.sha256(f"split:{sample_seed}".encode()).digest()[:4], "big"
    ) % 10000 / 100.0  # [0, 100)
    train_cutoff = ratios[0] * 100
    valid_cutoff = (ratios[0] + ratios[1]) * 100
    if bucket < train_cutoff: return "train"
    if bucket < valid_cutoff: return "valid"
    return "test"


@contextlib.contextmanager
def save_random_state():
    """Snapshot + restore global random state — defense-in-depth for dep upgrades."""
    state = random.getstate()
    try:
        yield
    finally:
        random.setstate(state)
```

### Pattern 2: writer.py — atomic per-sample write with sentinel

**What:** Five-step strict-ordered write that ends with a `rename()` atomic-on-POSIX-same-FS of `sample.json.tmp` to `sample.json`. If anything fails before step 5, the sentinel never exists.

**When to use:** Every successful `api.generate()` call that passed the sum-of-stems assertion.

**Example pattern (sketch):**
```python
# Source: CONTEXT.md D-04, D-11, D-12, D-24, D-25 + ARCHITECTURE.md §Per-sample output layout
def write_sample(
    dataset_root: str, sample_index: int, annotation: dict,
    mix_working_path: str, stems_working_paths: Dict[str, Dict[str, str]],
    midi_working_paths: Dict[str, Dict[str, str]],
    *, fluidsynth_version: str, split: str,
    sum_of_stems_epsilon: float = 1e-3,
) -> SampleResult:
    sample_dir = os.path.join(dataset_root, f"{sample_index:06d}")
    stems_dir = os.path.join(sample_dir, "stems")
    midi_dir = os.path.join(sample_dir, "midi")
    os.makedirs(stems_dir, exist_ok=True)
    os.makedirs(midi_dir, exist_ok=True)

    # 1. MIDI concat per layer (see Research Question 3 — absolute-tick walk)
    midi_final_paths = {}
    for layer in ("beat", "melody", "harmony", "bassline"):
        midi_final_paths[layer] = _concat_layer_midis(
            [midi_working_paths[part][layer] for part in song_arrangement],
            os.path.join(midi_dir, f"{layer}.mid"),
        )

    # 2. Stem concat per layer (pydub AudioSegment+, matches mixer.concat_parts)
    stem_final_paths = {}
    for layer in ("beat", "melody", "harmony", "bassline"):
        stem_final_paths[layer] = _concat_layer_stems(
            [stems_working_paths[part][layer] for part in song_arrangement],
            os.path.join(stems_dir, f"{layer}.wav"),
        )

    # 3. Mix: copy/rename from working
    mix_final = os.path.join(sample_dir, "mix.wav")
    shutil.copy2(mix_working_path, mix_final)  # or os.rename if same FS

    # 4. Sum-of-stems assertion — see Research Question 11
    passed, max_diff = _assert_sum_of_stems(
        mix_final, stem_final_paths, epsilon=sum_of_stems_epsilon,
    )
    if not passed:
        raise AssertionError(
            f"sum_of_stems_exceeded: {max_diff:.6f} > {sum_of_stems_epsilon:.6f}"
        )

    # 5. Annotation: path-rewrite + atomic sentinel
    final_annotation = _rewrite_paths_relative(
        annotation, mix_final, stem_final_paths, midi_final_paths,
    )
    final_annotation["split"] = split  # already filled by api.py
    sample_json_tmp = os.path.join(sample_dir, "sample.json.tmp")
    sample_json_final = os.path.join(sample_dir, "sample.json")
    with open(sample_json_tmp, "w") as f:
        json.dump(final_annotation, f, sort_keys=True, indent=2, separators=(",", ": "))
    os.rename(sample_json_tmp, sample_json_final)  # POSIX atomic on same FS

    return SampleResult(...)
```

### Pattern 3: manifest.py — append-only under injected lock

**What:** Open `manifest.jsonl` in `"a"` mode, acquire `lock`, write `json.dumps(entry, sort_keys=True) + "\n"`, `flush()`, `os.fsync(fd)`, release lock.

**When to use:** Once per `api.generate()` call (whether it succeeded or failed — CONTEXT D-13 requires a `status: "failed"` line).

**Example:**
```python
# Source: CONTEXT.md D-14, D-15, D-16
import json
import os
import threading
from typing import ContextManager, Optional

class ManifestWriter:
    def __init__(self, dataset_root: str, lock: Optional[ContextManager] = None):
        self.dataset_root = dataset_root
        self.manifest_path = os.path.join(dataset_root, "manifest.jsonl")
        self.lock = lock if lock is not None else threading.Lock()

    def append(self, entry: dict) -> None:
        os.makedirs(self.dataset_root, exist_ok=True)
        line = json.dumps(entry, sort_keys=True) + "\n"
        with self.lock:
            with open(self.manifest_path, "a") as f:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())

    @staticmethod
    def is_sample_complete(dataset_root: str, sample_index: int, pad: int = 6) -> bool:
        sentinel = os.path.join(dataset_root, f"{sample_index:0{pad}d}", "sample.json")
        return os.path.exists(sentinel)
```

### Pattern 4: api.py — single-sample library entry point

**What:** A linear procedure that wires together all prior modules and returns a `SampleResult`. Performs validation, resume-detection, pipeline orchestration, artifact relocation, manifest append, and working-dir cleanup.

**When to use:** The canonical call is `musicgen.generate(Config(global_seed=..., sample_index=...))`. Phase 6 `generate_batch` will call this per worker.

**Example (ordering only):** See System Architecture Diagram above. Full body is a ~80-100 line function.

### Pattern 5: determinism goldens with --regen-goldens flag

**What:** `tests/conftest.py` registers a `--regen-goldens` pytest CLI flag; `test_determinism_golden.py` reads it via `request.config.getoption()` and either regenerates the fixtures or asserts against them.

**When to use:** Intentional RNG-order changes, FluidSynth upgrades, or any legitimate golden-replacement event.

**Example:**
```python
# tests/conftest.py
def pytest_addoption(parser):
    parser.addoption(
        "--regen-goldens", action="store_true", default=False,
        help="Regenerate determinism fixtures instead of asserting.",
    )

# tests/test_determinism_golden.py
@pytest.mark.slow
def test_sha256_mix_matches_golden(request, tmp_path, fluidsynth_guard):
    result = musicgen.generate(Config(global_seed=1, sample_index=0, dataset_root=str(tmp_path)))
    mix_hash = hashlib.sha256(open(result.mix_path, "rb").read()).hexdigest()
    golden_path = FIXTURES / "expected_mix.sha256"
    if request.config.getoption("--regen-goldens"):
        golden_path.write_text(mix_hash + "\n")
    else:
        expected = golden_path.read_text().strip()
        assert mix_hash == expected
```

### Anti-Patterns to Avoid

- **Naive mido track concat by append.** Appending `mf2.tracks[0]` messages after `mf1.tracks[0]` relies on the accumulated tick count of `mf1` matching the intended part duration. `midiutil` writes tracks whose `sum(msg.time)` is the tick of the last emitted event — a pattern like `4/4 intro: 0, 42, 38, 0` only emits events at tick offsets for the two non-zero hits; the track's last tick is well short of the 4-beat measure. Naive concat places the next part's first note too early. **Use absolute-tick walk** (Research Question 3 below).
- **Summing int16 stems as int16.** `np.sum([stem_i16, stem_j16, ...])` on four stems each in `[-32768, 32767]` can overflow silently. **Cast to int32 first**, sum, compare, normalize for ε check: `np.max(np.abs(sums_i32 - mix_i32)) / 32768.0 < 1e-3`.
- **Writing `sample.json` in the middle of the sample dir build.** Resume detection (`is_sample_complete`) reads only `sample.json` existence. If the writer creates `sample.json` before the stems are finalized, a crash mid-write leaves a "complete" sample with half its data. **Sentinel-last invariant** is load-bearing — CONTEXT D-04 is correct; enforce with `rename` at the end.
- **Touching global `random` state.** The AST guard (Phase 4 D-31 + Phase 5 D-42) catches `random.<method>(...)` calls at module scope. Never call `random.seed(x)`, never call `random.choice(x)`. All RNG goes through the `rngs` dict's per-domain `random.Random` instances. `save_random_state()` is defense-in-depth against a transitive dep that ignores this rule.
- **Reading manifest.jsonl for resume detection.** CONTEXT D-16: manifest is a projection; the sentinel is truth. Don't parse JSONL to decide whether to skip — stat the file path.
- **Changing RNG draw order while "refactoring."** Any change to the order/count of `rng.*` calls between api.py's `make_rngs` and the final artifacts invalidates the goldens. CONTEXT D-19's RNG threading map is verbatim-inherited from Phase 4 call sites; do NOT optimize (e.g., "apply FX only to used layers" — CONTEXT explicitly defers this).
- **Assuming `musicgen_version` is always available.** CONTEXT D-22 is explicit about the fallback: `importlib.metadata.PackageNotFoundError` → `"0.1.0+uninstalled"` string. Needed in CI / PYTHONPATH-only test runs where the package is imported but not installed.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Hash of a seed + index | Custom 64-bit mix | `hashlib.sha256(f"{g}:{i}".encode()).digest()[:8]` | Cryptographic quality avoids collisions/clustering across 10k indices |
| MIDI file concatenation | Hand-rolled SMF binary walker | `mido.MidiFile` read → absolute-tick walk → `mido.MidiTrack` write | `mido` handles running-status encoding, delta-tick encoding, variable-length quantity compression — all subtle and all places hand-rolled code would break |
| WAV int16 round-trip | Manual header + sample struct unpack | `scipy.io.wavfile.read` / `pydub.AudioSegment.from_wav` | `scipy.io.wavfile.read` returns correct dtype + shape; bit-identical round-trips verified in probe |
| Lock abstraction for manifest | Custom "if single-process: flock else: spinlock" | `threading.Lock()` default + `ContextManager` type hint | Python's stdlib `Lock` protocols are `with`-statement compatible; `multiprocessing.Manager().Lock()` proxy swaps in identically (verified) |
| Canonical JSON serialization | Custom key-sort + recursive key-case rules | `json.dumps(obj, sort_keys=True, separators=(",", ": "), indent=2)` | sort_keys is load-bearing for the `sample.json` byte-stability test (D-30) — proven byte-stable across runs |
| Sum-of-stems assertion | Custom overflow-safe accumulator | `np.sum([s.astype(np.int32) for s in stems], axis=0) - mix.astype(np.int32)` | NumPy handles shape broadcasting + promotion correctly; any custom loop is slower + error-prone |
| Pytest CLI flag for golden regen | `os.environ.get("REGEN_GOLDENS")` | `pytest_addoption` + `request.config.getoption("--regen-goldens")` | Integrates with `-v`/`-q`/help; clean discoverability via `pytest --help` |
| Working dir for a sample | Per-index hardcoded path | `tempfile.mkdtemp(prefix="musicgen-")` | Collision-free under concurrent use (Phase 6 ProcessPool); survives interruption without leaking into `dataset_root` |
| Version string for package | Hardcoded `"0.1.0"` string | `importlib.metadata.version("musicgen")` + `PackageNotFoundError` fallback | Stays in sync with pyproject.toml; zero-maintenance on version bumps |
| Atomic file swap | `os.replace(tmp, dst)` + try/except | POSIX `os.rename(src, dst)` on same-filesystem paths | `os.rename` is atomic on POSIX within a filesystem; working dir + sample_dir are both under `dataset_root`/`/tmp` (separate FSs, so stems/mix use `shutil.copy2` then `os.rename` for the sentinel within `dataset_root`) |
| RNG state save/restore | Stack discipline with `random.getstate()` manually | `@contextlib.contextmanager` (D-20 verbatim) | One-line `with save_random_state():` is readable and exception-safe |

**Key insight:** Phase 5 is a stdlib phase. Every mechanism exists in Python 3.10+. The only real risks are (a) the MIDI concat subtlety documented below, and (b) the FluidSynth-binary-pinned nature of audio bit-identity (which is a known fact of life, not a fixable engineering risk — CONTEXT D-28/D-29 handle it by version-guarding the relevant assertion).

## Runtime State Inventory

> Phase 5 is a *code+layout* phase. The `musicality_score.py` → `src/musicgen/musicality.py` move is a rename/refactor item; the rest is additive.

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | None. Phase 5 starts writing `<dataset_root>/manifest.jsonl` and `<dataset_root>/<idx:06d>/` — but there is no prior "dev-os" / "v0.0" dataset under git that would need migrating. Any developer's previous `dataset/` (from the Phase 4 smoke-test code path `music_gen.py:create_song` writing to `<name>/...` in cwd) is a throwaway. | None — plan just documents that old `<name>/<name>.json` files from Phase 4 smoke tests can be rm-rf'd by developers before switching. |
| Live service config | None. No n8n / Datadog / external service is in scope. | None — verified: Phase 5 only writes to local filesystem. |
| OS-registered state | None. No OS-level task scheduler, systemd unit, launchd plist is in scope. | None — verified: `python music_gen.py` smoke test runs in-process; no background daemons. |
| Secrets/env vars | `MUSICGEN_DATASET_ROOT` is a **new** env var (CONTEXT D-09 Phase 2 precedence extension). Not secret. Existing `MUSICGEN_SF_DIR` and `MUSICGEN_PROJECT_ROOT` (Plan 02-01) are unchanged. | Plan must (a) document the new env var in the Config._emit_soundfont_pool_report() path or similar, (b) thread it through `Config.load(cli_overrides=None)`. |
| Build artifacts / installed packages | `musicality_score.py` rename (D-03) — import path changes from `import musicality_score` → `from musicgen import musicality`. One caller in `music_gen.py:3`. After D-34 deletes `create_song` from `music_gen.py`, this caller disappears. **Installed package** `musicgen` (editable via `pip install -e .[dev]`) picks up the new `src/musicgen/musicality.py` automatically on next import — no reinstall required. | Plan should: (a) git-mv `musicality_score.py` → `src/musicgen/musicality.py` (use `git mv` for 100% rename detection), (b) update the single import site in `music_gen.py` during D-34's rewrite (the import goes away entirely since `create_song` is deleted), (c) verify `from musicgen.musicality import get_musicality_score` works after the move. **No pip reinstall needed** (editable installs track `src/musicgen/`). |

**Filesystem side-effect expected during Phase 5 execution:** stale `musicality_score.cpython-*.pyc` under `__pycache__/` after the move — harmless; Python ignores .pyc without a matching .py source.

## Common Pitfalls

### Pitfall 1: Naive MIDI track concatenation drops trailing silence

**What goes wrong:** Writer concatenates `mido.MidiFile(p1).tracks[1]` and `mido.MidiFile(p2).tracks[1]` message-by-message. The first note_on of p2's track starts immediately after p1's last event — NOT after p1's total part duration.

**Why it happens:** `midiutil.MIDIFile.addNote` only writes note events; the track's accumulated tick count equals the tick of the last emitted event, not the intended part duration. In `generators/beat.py`, a pattern like `[kick, 0, 0, 0]` (one-beat-per-measure pattern) advances `current_time` through all four slots but only adds one MIDI event. Result: the written track ends at tick 960, not tick 3840 (at 4/4 @ 120 bpm).

**Empirical probe result:** A `midiutil` MIDI built with `mf.addNote(0, 0, 60, 0, 1, 100); mf.addNote(0, 0, 60, 1, 1, 100)` at 2 measures of 4/4 @ 120bpm has `sum(msg.time) == 1920` on its note track — NOT 3840. Concatenation would put the next part 1920 ticks (1 second) earlier than expected.

**How to avoid:** Use **absolute-tick walk**. For each source track: iterate msgs, accumulate into absolute ticks, drop `end_of_track`; merge absolute-tick message lists with explicit per-part tick offset = `part_duration_ticks = int(mido.second2tick(render_result.duration_seconds, ticks_per_beat, tempo_us))`; serialize merged list back to delta ticks; append a terminal `end_of_track` with `time=0`. This is the one piece of Phase 5 that benefits from careful unit tests.

**Warning signs:** Annotation's `beat_times` cover the full mix duration but the concatenated beat MIDI plays only the first part-length of drums. Any E2E test that renders the concat MIDI back to audio via FluidSynth and measures duration will catch it.

### Pitfall 2: Int16 sum-of-stems overflow

**What goes wrong:** Four stems each in `[-32768, 32767]` can sum to `[-131072, 131068]`. If stored in int16, this wraps silently. If stored in float32 without rescaling, numerical noise creeps in. Either way, the epsilon comparison fails for the wrong reason.

**Why it happens:** `numpy.sum` promotes dtype under some circumstances but not always — element-wise `+` on int16 arrays stays int16. A `np.sum([a, b, c, d], axis=0)` where a,b,c,d are int16 can return int16 (implementation-dependent).

**Empirical probe result:** `np.sum([s.astype(np.int32) for s in stems], axis=0)` is safe (explicit int32 promotion; verified). `max |sum - mix| int16: 0` when stems are constructed to sum cleanly — the assertion is sharp.

**How to avoid:** **Always cast to int32 before summing**, then compute `np.max(np.abs(sum_i32 - mix_i32)) / 32768.0 < 1e-3`. Division by `32768.0` (not `32767.0`) matches the pydub/scipy convention of int16 mapping to `[-1.0, +1.0)`.

**Warning signs:** Sum-of-stems assertion fails unpredictably on "loud" samples with four high-gain stems; fine on quiet ones.

### Pitfall 3: `os.rename` atomicity across filesystems

**What goes wrong:** `os.rename(/tmp/musicgen-xyz/sample.json.tmp, /dataset/000000/sample.json)` — when `/tmp` is tmpfs and `/dataset` is ext4, `os.rename` falls back to a non-atomic copy-then-delete on some kernels (and raises `OSError(EXDEV)` on others).

**Why it happens:** POSIX `rename(2)` is atomic only within a single filesystem. `/tmp` on tmpfs, `./dataset` on an ext4 home, and `/mnt/data` on NFS are three filesystems.

**How to avoid:** Do the sentinel rename **within the sample directory**: write `<sample_dir>/sample.json.tmp`, then `os.rename(<sample_dir>/sample.json.tmp, <sample_dir>/sample.json)`. Both paths share a filesystem. The mix.wav/stems/midi are **copied** from the `tempfile.mkdtemp` working dir into `<sample_dir>` first (via `shutil.copy2`), so the sample_dir is already populated by the time the sentinel rename runs.

**Warning signs:** Integration test passes locally but fails in Docker where `/tmp` is `tmpfs` and the workspace is bind-mounted.

### Pitfall 4: `musicgen_version` resolves to None in non-installed test runs

**What goes wrong:** `importlib.metadata.version("musicgen")` raises `PackageNotFoundError` when the package isn't pip-installed (e.g., a test harness that just adds `src/` to PYTHONPATH). Annotation's `musicgen_version` field becomes None, which violates the determinism-golden test's "this field is deterministic" invariant.

**How to avoid:** CONTEXT D-22 specifies the `"0.1.0+uninstalled"` fallback literal. Implement with:
```python
try:
    MUSICGEN_VERSION = importlib.metadata.version("musicgen")
except importlib.metadata.PackageNotFoundError:
    MUSICGEN_VERSION = "0.1.0+uninstalled"
```
This means the **golden fixture for `sample.json`** must be regenerated from an editable-installed environment (where `version("musicgen") == "0.1.0"`), not from a raw-PYTHONPATH environment. Document this in the `--regen-goldens` docstring.

**Warning signs:** Golden test passes when developer runs `pip install -e .[dev]` first but fails in a container that only sets PYTHONPATH.

### Pitfall 5: `sort_keys=True` doesn't handle nested non-string keys

**What goes wrong:** `json.dumps({"a": 1, "b": 2}, sort_keys=True)` sorts top-level keys. Nested dicts with non-string keys (e.g., int keys from a `defaultdict(int)` counter in `musicality["components"]`) raise `TypeError`.

**Why it happens:** JSON spec requires string keys; `json.dumps` coerces int keys but `sort_keys=True` sorts before coercion, and comparing int to str raises.

**How to avoid:** Audit the annotation dict for non-string keys. The current `annotator.annotate()` produces only string keys (probed against Phase 4 fixtures). **Low risk** but the task acceptance should include a `json.dumps(annotation, sort_keys=True)` invariance test in `test_writer.py`.

**Warning signs:** Test passes on simple fixtures, fails on real musicality_score output if that ever returns non-string keys.

### Pitfall 6: `shutil.rmtree(working_dir)` after `os.chdir(working_dir)`

**What goes wrong:** Phase 4 integration test uses `monkeypatch.chdir(tmp_path)` then the orchestrator creates subdirs under cwd. If Phase 5's `api.generate()` does the same (creates working dir under cwd), an `os.chdir` into the working dir followed by `shutil.rmtree` leaves the process in a deleted directory — subsequent `os.getcwd()` fails.

**How to avoid:** The CONTEXT-locked pattern is `tempfile.mkdtemp(prefix="musicgen-")` (D-31 step 4), which returns an absolute `/tmp/musicgen-...` path. `api.generate()` NEVER `chdir`s. All paths are absolute. `shutil.rmtree(working_dir, ignore_errors=True)` after generation is safe.

**Warning signs:** Phase 4 E2E test uses `monkeypatch.chdir(tmp_path)` to contain generator side-effects. Phase 5 should **not** inherit that pattern — instead, generators should write into a pre-created working_dir. (Phase 4 already writes to `<out_dir>` parameters — see renderer/mixer — so Phase 5's api.py passes `working_dir` as the `out_dir` argument to both.)

### Pitfall 7: Resume-short-circuit has no way to recover the original SampleResult

**What goes wrong:** CONTEXT D-31 step 3: `if ManifestWriter.is_sample_complete(...): return _reconstruct_sample_result(...)`. The reconstructed `SampleResult` fields (`sample_index`, `seed`, `sample_dir`, `sample_json_path`, `mix_path`, `stem_paths`, `midi_paths`, `split`, `status`, `musicality_score`, `duration_seconds`) must match what a fresh generation would return — otherwise a consumer who cached the first-run result sees a different object on resume.

**How to avoid:** `_reconstruct_sample_result` loads `<sample_dir>/sample.json`, reads the filled keys (`seed`, `split`, `musicality_score.score`, `duration_seconds`), builds the paths from convention (`<sample_dir>/mix.wav`, `<sample_dir>/stems/{layer}.wav`, `<sample_dir>/midi/{layer}.mid`), and sets `status="ok"`. **Does not** re-read the manifest (D-16: sentinel is truth).

**Warning signs:** Resume-idempotent test (D-40 `test_generate_twice_idempotent`) returns a SampleResult from the second call whose `musicality_score` is a fresh score from a new musicality pass — a bug where `_reconstruct_sample_result` accidentally runs scoring instead of reading the cached value.

### Pitfall 8: FluidSynth renders with inconsistent pre-roll across invocations

**What goes wrong:** Same MIDI + same soundfont + same FluidSynth binary, two runs in one process produce two WAV files whose SHA-256 differs by ~1 KB at the start (FluidSynth's buffered startup silence). This is exactly what CONTEXT D-28/D-29 guards against with the `fluidsynth_version.txt` skip.

**Why it happens:** FluidSynth schedules the first note with a per-invocation buffer fill before producing the first audio sample. The buffer size can vary with settings (`sample-rate`, `period-size`).

**How to avoid:** NOT a Phase 5 problem to solve — CONTEXT D-28/D-29 accepts "bit-identical only under pinned FluidSynth binary + pinned soundfonts + pinned invocation flags." The regression test skips with xfail when the FluidSynth version mismatches. **Pre-roll calibration is explicitly R-P9 Phase 6** and out of scope here.

**Warning signs:** Golden mix.sha256 passes locally, fails on CI — expected if FluidSynth versions differ.

## Code Examples

### Absolute-tick MIDI concatenation (the one subtle helper)

```python
# Source: mido official docs (https://mido.readthedocs.io/en/latest/files.html#type-1-files)
# + research probe against midiutil + a handwritten absolute-tick walk
import mido
from typing import List

def _concat_layer_midis(
    part_midi_paths: List[str],
    part_durations_s: List[float],
    tempo_bpm: int,
    out_path: str,
) -> str:
    """Concatenate per-part MIDIs into one with correct inter-part timing."""
    first = mido.MidiFile(part_midi_paths[0])
    ticks_per_beat = first.ticks_per_beat
    tempo_us = mido.bpm2tempo(tempo_bpm)
    midi_type = first.type

    merged = mido.MidiFile(ticks_per_beat=ticks_per_beat, type=midi_type)

    # For Type 1: each track concatenates independently. For Type 0: only track 0.
    num_tracks = len(first.tracks)

    def _track_to_absolute(track):
        t = 0
        abs_msgs = []
        for msg in track:
            t += msg.time
            if msg.is_meta and msg.type == "end_of_track":
                continue
            abs_msgs.append((t, msg.copy(time=0)))  # strip relative delta
        return abs_msgs

    def _absolute_to_track(abs_msgs, end_tick):
        abs_msgs = sorted(abs_msgs, key=lambda x: x[0])
        tr = mido.MidiTrack()
        prev = 0
        for t, msg in abs_msgs:
            tr.append(msg.copy(time=t - prev))
            prev = t
        tr.append(mido.MetaMessage("end_of_track", time=max(0, end_tick - prev)))
        return tr

    for tr_idx in range(num_tracks):
        merged_abs = []
        offset_ticks = 0
        for part_path, part_dur_s in zip(part_midi_paths, part_durations_s):
            part = mido.MidiFile(part_path)
            abs_msgs = _track_to_absolute(part.tracks[tr_idx])
            merged_abs.extend([(t + offset_ticks, m) for (t, m) in abs_msgs])
            offset_ticks += int(mido.second2tick(part_dur_s, ticks_per_beat, tempo_us))
        merged.tracks.append(_absolute_to_track(merged_abs, end_tick=offset_ticks))

    merged.save(out_path)
    return out_path
```

### Sum-of-stems assertion (int32 accumulator)

```python
# Source: probe against numpy+scipy+pydub behavior during research
import numpy as np
import scipy.io.wavfile as wf
from typing import Dict, Tuple

def _assert_sum_of_stems(
    mix_path: str,
    stem_paths: Dict[str, str],
    epsilon: float = 1e-3,
) -> Tuple[bool, float]:
    """Read mix + 4 stems, check max |sum - mix| < ε in normalized float."""
    _, mix_i16 = wf.read(mix_path)  # int16 ndarray
    mix_i32 = mix_i16.astype(np.int32)
    sums_i32 = np.zeros_like(mix_i32)
    for layer, path in stem_paths.items():
        _, stem_i16 = wf.read(path)
        # Shape sanity: all stems must match mix shape (44.1k stereo ensures this).
        if stem_i16.shape != mix_i16.shape:
            raise ValueError(
                f"stem {layer!r} shape {stem_i16.shape} != mix shape {mix_i16.shape}"
            )
        sums_i32 += stem_i16.astype(np.int32)
    max_abs_int = int(np.max(np.abs(sums_i32 - mix_i32)))
    max_abs_float = max_abs_int / 32768.0  # pydub int16 ↔ [-1, +1) convention
    return (max_abs_float < epsilon, max_abs_float)
```

### SHA-256 stability cross-check (the fast, no-FluidSynth test — D-30)

```python
# Source: CONTEXT.md D-30
import hashlib
from musicgen import generate, Config

def test_generate_sample_json_stable_same_process(tmp_path):
    """Two back-to-back calls produce byte-identical sample.json (D-30)."""
    cfg_a = Config(global_seed=1, sample_index=0, dataset_root=str(tmp_path / "a"))
    cfg_b = Config(global_seed=1, sample_index=0, dataset_root=str(tmp_path / "b"))
    # Fast path: mock the FluidSynth + musicality calls to return deterministic stubs.
    # (Full version uses monkeypatch + fixture.)
    r1 = generate(cfg_a)
    r2 = generate(cfg_b)
    with open(r1.sample_json_path, "rb") as f1, open(r2.sample_json_path, "rb") as f2:
        assert hashlib.sha256(f1.read()).hexdigest() == hashlib.sha256(f2.read()).hexdigest()
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `uuid.uuid4()[:20]` for sample names (`music_gen.py:184`) | `<idx:06d>` zero-padded index (CONTEXT D-04/D-05) | Phase 5 (this one) | Fixes PITFALLS P-7 (collision under parallelism); enables resume-by-index (Phase 6) |
| Single module-level `_rng = random.Random()` (`music_gen.py:19`) | Five per-domain RNGs via `make_rngs(sample_seed)` (CONTEXT D-18/D-19) | Phase 5 (this one) | Fixes PITFALLS P-4 (RNG leakage); enables Phase 6 ProcessPool correctness |
| `beat_anotator.py` straight-grid timestamps | `beats.py` MIDI-tick extraction (Phase 4 D-19) | Phase 4 (done) | Fixes PITFALLS P-3 (swing drift); already in place |
| `mix_and_save` 143-line god function | `renderer / mixer / annotator / beats / writer` chain | Phases 3-5 | Productize milestone unlocked |
| Absolute paths in annotation | Dataset-root-relative paths rewritten by writer (CONTEXT D-11/D-12) | Phase 5 (this one) | Publishable dataset; tarball-portable |
| Silent-stem is "skip the layer" | Silent-stem WAV file (Phase 4 D-12) concat'd into song-scope silent stem (Phase 5 D-06) | Phases 4 + 5 | Stems-sum-to-mix invariant (R-P2) holds for every sample |

**Deprecated/outdated:**
- `musicality_score.py` at repo root — CONTEXT D-03 relocates to `src/musicgen/musicality.py` this phase. This is the third time this move has been scheduled (Phase 3 D-11 deferred; Phase 4 D-04 deferred; Phase 5 D-03 commits). After the move, the repo-root `musicality_score.py` is **deleted** (no back-compat shim — the only import site is `music_gen.py:3`, which D-34 deletes via `create_song` removal).

## Assumptions Log

> Research claims tagged `[ASSUMED]` — planner and discuss-phase should decide if any need user confirmation. **All high-stakes claims were verified by probe against the installed venv.** Remaining assumptions below are low-stakes or cross-environment.

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | FluidSynth's WAV output is bit-identical across runs under the same binary+soundfont (for the D-28 golden test) | Pitfall 8 | LOW — CONTEXT D-28/D-29 already guards with the `fluidsynth_version.txt` skip gate. If the binary is not bit-stable on a given platform, the test xfails with a clear message instead of silently failing. The **MIDI + sample.json** goldens do NOT depend on this and must always pass. `[ASSUMED]` |
| A2 | `random.Random(seed)` is deterministic across CPython 3.10/3.11/3.12 patch releases | Pitfall 5 | LOW — Python's Mersenne Twister is spec'd to be stable. Probed against CPython 3.12.x; no PyPy in scope per pyproject `requires-python = ">=3.10"`. Any CPython minor-version change that breaks this would affect ALL goldens equally (regenerable via `--regen-goldens`). `[ASSUMED]` |
| A3 | `scipy.io.wavfile.read` returns int16 dtype for pydub-exported WAVs | Sum-of-stems probe | VERIFIED — probe showed `rate=44100 dtype=int16 shape=(22050, 2)` for stereo pydub output. `[VERIFIED: research probe]` |
| A4 | `AudioSegment.from_wav(p).export(p2, format='wav')` is byte-identical round-trip | Pattern 3 probe | VERIFIED — probe confirmed `hashlib.sha256` of round-trip matches original for custom int16 stereo data. `[VERIFIED: research probe]` |
| A5 | `midiutil.MIDIFile` writes Type-1 files with note data on track 1 (not track 0) | Pitfall 1 probe | VERIFIED — probe showed `mf.tracks` has len 2; track 0 is tempo meta, track 1 is notes. Absolute-tick concat strategy must iterate ALL tracks (`num_tracks = len(first.tracks)`), not just one. `[VERIFIED: research probe]` |
| A6 | `importlib.metadata.version("musicgen")` raises `PackageNotFoundError` (not ImportError) when the package isn't installed | Pitfall 4 | VERIFIED — stdlib docs confirm; probe in system Python (no musicgen install) raised `importlib.metadata.PackageNotFoundError`. `[VERIFIED: research probe]` |
| A7 | `os.rename(src, dst)` is atomic within a single POSIX filesystem | Pitfall 3 | HIGH confidence — POSIX spec + Linux man page; long-established. Safe when `src` and `dst` are under the same `<sample_dir>`. `[CITED: https://pubs.opengroup.org/onlinepubs/9699919799/functions/rename.html]` |
| A8 | `open(path, "a")` + `f.write(s)` + `f.flush()` + `os.fsync(f.fileno())` is sufficient for single-process append atomicity on POSIX | Pattern 4 | HIGH — POSIX `O_APPEND` semantics guarantee the seek-to-end + write is atomic for writes ≤ `PIPE_BUF` (4096 bytes on Linux). A manifest line at 10k samples is ~200 bytes, well under 4096. For Phase 5's single-thread append under `threading.Lock`, atomicity is trivial. `[CITED: POSIX write(2) man page]` |
| A9 | `pedalboard` FX application + `pydub.overlay` are deterministic given the same pedalboard params + input WAV | Sum-of-stems | MEDIUM — CONTEXT D-25 assumes ε=1e-3 is enough to absorb any float drift. Probed `AudioSegment.from_wav + .export` is bit-identical; pedalboard is a separate library we did NOT probe this session. Golden test will empirically verify. `[ASSUMED]` |
| A10 | `music21 / librosa` do not touch global `random` state (Phase 3 D-24 audit) | save_random_state | VERIFIED — `tests/test_music21_isolation.py` (Plan 03-05) is the regression guard. CONTEXT D-20 wraps musicality scoring anyway as defense-in-depth. `[VERIFIED: Plan 03-05 SUMMARY]` |
| A11 | `generators/{chord,melody,bassline,beat}.py` already accept `rng: random.Random` as their only RNG input | RNG threading map | VERIFIED — grep probe shows `rng.choice/rng.choices/rng.random/rng.randint/rng.uniform` throughout `src/musicgen/generators/`; no bare `random.*` (AST guard enforces). `[VERIFIED: research probe via Grep]` |

## Open Questions

1. **SampleResult field names — `sample_dir` vs `sample_path` vs `path`?**
   - What we know: CONTEXT D-02 locks the shape (11 fields); Claude's Discretion explicitly says the names are aesthetic.
   - What's unclear: which name reads most naturally in `result.sample_dir / "mix.wav"`-style code.
   - Recommendation: use `sample_dir` (matches the "per-sample directory" language in R-P1, REQUIREMENTS, ARCHITECTURE). `sample_json_path` for the sentinel (specific, unambiguous).

2. **Does the resume short-circuit reload the full annotation dict or just paths?**
   - What we know: CONTEXT D-31 step 3 says "reconstruct SampleResult from on-disk files" — this is ambiguous about whether `musicality_score` is read back.
   - What's unclear: if `SampleResult.musicality_score: float` must be populated, it has to come from `sample.json`'s `musicality_score.score`.
   - Recommendation: `_reconstruct_sample_result(dataset_root, sample_index)` loads `<sample_dir>/sample.json` via `json.load`, extracts `seed`, `split`, `musicality_score["score"]`, `duration_seconds`; constructs paths by convention. Does NOT re-run musicality scoring. Does NOT re-parse paths (they're dataset-root-relative in the JSON — need to re-absolutize for `SampleResult`).

3. **Should the working dir live under `dataset_root/.tmp/` or under `tempfile.mkdtemp`?**
   - What we know: CONTEXT's "Claude's Discretion" accepts either; research recommends `tempfile.mkdtemp` for collision-freeness + not polluting `dataset_root` with interrupt artifacts.
   - What's unclear: whether Phase 6 ProcessPool workers should share a common parent tempdir (for easier cleanup on batch cancellation) or each get their own mkdtemp.
   - Recommendation: this phase uses `tempfile.mkdtemp(prefix="musicgen-")` per call. Phase 6 can revisit by making it configurable via `config.working_dir_root` if needed. **No action this phase.**

4. **Does writer need to enforce the R-P4 schema explicitly or trust annotator?**
   - What we know: Phase 4 annotator produces the dict with Phase-5 TBD fields as `None`; api.py fills them via kwargs.
   - What's unclear: should writer.write_sample validate that all required keys are present (defensive) or trust the input (pragmatic)?
   - Recommendation: writer trusts input — the cost of a missing key is a loud KeyError in tests. A schema-validation framework (jsonschema) is overkill for v0.1. **Document the trust boundary**: api.py is the only legitimate caller of writer; writer does not accept arbitrary annotation dicts.

5. **How does api.py handle a failed sum-of-stems assertion — re-raise or return `SampleResult(status="failed")`?**
   - What we know: CONTEXT D-24 says "log.error + ManifestWriter.append(status='failed', ...) + return SampleResult with status='failed' + DO NOT write sample.json". CONTEXT D-31 step 7 calls `writer.write_sample(...)` and implies it returns. If write_sample raises on assertion failure (cleanest), api.py needs to catch.
   - What's unclear: does api.py do the try/except + manifest-append, or does writer.write_sample internally swallow + return a failed result?
   - Recommendation: `writer.write_sample` **raises `AssertionError` on sum-of-stems failure**. `api.py:generate` wraps the call in try/except, builds a `status="failed"` SampleResult, appends to manifest, cleans working dir, returns. Cleaner separation: writer only writes successes; api owns error logging + manifest + cleanup semantics. Planner should lock this at plan-time.

6. **The `test_determinism_golden.py` file — is the full suite one test or multiple parametrized tests?**
   - What we know: D-29 says "asserts each of the 6 SHA-256s" — could be one test with 6 asserts or 6 parametrized tests.
   - Recommendation: one test per fixture (`@pytest.mark.parametrize("artifact", ["mix", "midi_beat", "midi_melody", "midi_harmony", "midi_bassline", "sample"])`). Failures are attributable to specific artifacts (e.g., "mix hash mismatch — FluidSynth-related; sample.json hash mismatch — code regression").

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3.10+ | Package | ✓ | 3.12 (venv) | — |
| numpy | Sum-of-stems | ✓ | (installed — probed) | — |
| scipy | Sum-of-stems (wavfile.read) | ✓ | (installed — probed) | — |
| pydub | Stem concat | ✓ | (installed — probed) | — |
| mido | MIDI concat | ✓ | 1.3.3 (pyproject pin) | — |
| pytest | Test runner | ✓ | (dev extras installed) | — |
| FluidSynth binary | D-29 golden mix.sha256 test | ✓ (probe via Plan 04-06 integration test xfail pattern) | version-pinned via `fluidsynth_version.txt` | Skip with xfail on mismatch |
| SoundFonts (sf/*.sf2) | D-29 golden test | ✓ (smoke-test gate already in place per Plan 04-06) | — | Skip with xfail when empty |

**Missing dependencies with no fallback:** None.

**Missing dependencies with fallback:** None (all dependencies resolved).

## Validation Architecture

> `workflow.nyquist_validation` is not `false` in `.planning/config.json` → this section is required.

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 8.x + pytest-cov 5.x + pytest-xdist 3.x (pyproject.toml `[project.optional-dependencies].dev`) |
| Config file | `pyproject.toml` `[tool.pytest.ini_options]` (pythonpath=["."], markers=[slow, integration]) |
| Quick run command | `.venv/bin/pytest -m "not slow" -x` (skips FluidSynth-bound tests; ~5s) |
| Full suite command | `.venv/bin/pytest -m slow` then `.venv/bin/pytest` (FluidSynth + all; ~30s+) |
| Golden regeneration | `.venv/bin/pytest -m slow --regen-goldens tests/test_determinism_golden.py` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| R-P1 | Per-sample layout `<idx:06d>/{mix.wav, stems/*.wav, midi/*.mid, sample.json}` | unit + integration | `.venv/bin/pytest tests/test_writer.py::TestLayout` | ❌ Wave 0 |
| R-P1 | `sample.json` is LAST file written (sentinel) | unit | `.venv/bin/pytest tests/test_writer.py::TestSentinelOrder` | ❌ Wave 0 |
| R-P1 | UUID truncation bug eliminated (index-based naming) | unit | `.venv/bin/pytest tests/test_writer.py::TestIndexBasedNaming` | ❌ Wave 0 |
| R-P2 | Sum-of-stems assertion passes on valid sample | unit | `.venv/bin/pytest tests/test_writer.py::TestSumOfStems::test_passes_on_valid` | ❌ Wave 0 |
| R-P2 | Sum-of-stems assertion fails on fault-injected divergence | unit | `.venv/bin/pytest tests/test_writer.py::TestSumOfStems::test_fails_on_divergent` | ❌ Wave 0 |
| R-P3 | Per-layer MIDI persisted in `midi/{layer}.mid` | unit | `.venv/bin/pytest tests/test_writer.py::TestMidiConcat` | ❌ Wave 0 |
| R-P4 | All R-P4 schema fields present in `sample.json` | unit | `.venv/bin/pytest tests/test_writer.py::TestSchemaCompletion` | ❌ Wave 0 |
| R-P4 | Phase-5 TBD fields filled (seed, musicgen_version, split) | unit | `.venv/bin/pytest tests/test_api.py::TestAnnotationSchemaFilled` | ❌ Wave 0 |
| R-P4 | Paths in `sample.json` are per-sample-dir-relative | unit | `.venv/bin/pytest tests/test_writer.py::TestRelativePaths` | ❌ Wave 0 |
| R-P5 | `manifest.jsonl` append one-line per sample | unit | `.venv/bin/pytest tests/test_manifest.py::test_manifest_append_single` | ❌ Wave 0 |
| R-P5 | Concurrent-thread appends produce 10×100 well-formed lines | unit | `.venv/bin/pytest tests/test_manifest.py::test_manifest_append_concurrent` | ❌ Wave 0 |
| R-P5 | `is_sample_complete` checks sentinel only (not manifest) | unit | `.venv/bin/pytest tests/test_manifest.py::test_is_sample_complete_true_iff_sentinel` | ❌ Wave 0 |
| R-P6 | Split hash deterministic + 80/10/10 empirical ratio | unit | `.venv/bin/pytest tests/test_split.py::test_deterministic`, `test_ratios_10k_seeds` | ❌ Wave 0 |
| R-P6 | Invalid ratios raise at Config init | unit | `.venv/bin/pytest tests/test_config.py::test_invalid_split_ratios` (or test_split.py) | ❌ Wave 0 |
| R-P7 | `derive_sample_seed` deterministic + collision-free across 100 indices | unit | `.venv/bin/pytest tests/test_seeds.py::test_derive_sample_seed_deterministic`, `test_different_indices` | ❌ Wave 0 |
| R-P7 | `make_rngs` returns 5 keys, independent streams | unit | `.venv/bin/pytest tests/test_seeds.py::test_make_rngs_five_domains`, `test_domain_independence` | ❌ Wave 0 |
| R-P7 | `save_random_state()` restores global state | unit | `.venv/bin/pytest tests/test_seeds.py::test_save_random_state_restores` | ❌ Wave 0 |
| R-P7 | Zero bare `random.*` across `src/musicgen/**/*.py` | unit (AST) | `.venv/bin/pytest tests/test_no_bare_random_in_package.py` (existing, now covers new modules via glob) | ✅ present |
| R-P7 | `config.global_seed=None` → ValueError from `api.generate` | unit | `.venv/bin/pytest tests/test_api.py::test_config_global_seed_required` | ❌ Wave 0 |
| R-P8 | `sha256(mix.wav)` matches golden under pinned FluidSynth | integration `@slow` | `.venv/bin/pytest -m slow tests/test_determinism_golden.py::test_mix_sha` | ❌ Wave 0 |
| R-P8 | `sha256(midi/*.mid)` matches golden (FluidSynth-independent) | integration `@slow` (to share fixture) | `.venv/bin/pytest -m slow tests/test_determinism_golden.py::test_midi_sha[layer]` | ❌ Wave 0 |
| R-P8 | `sha256(sample.json)` matches golden | integration `@slow` | `.venv/bin/pytest -m slow tests/test_determinism_golden.py::test_sample_sha` | ❌ Wave 0 |
| R-P8 cross-check | `sample.json` byte-stable across two `generate()` calls in one process | unit | `.venv/bin/pytest tests/test_determinism_golden.py::test_generate_sample_json_stable_same_process` (fast, no-FluidSynth guard) | ❌ Wave 0 |
| R-Q3 | Regression test in CI (via `pytest -m "not slow"` default + `-m slow` opt-in) | runner config | `.venv/bin/pytest` (default) + `.venv/bin/pytest -m slow` (opt-in) | ✅ existing pytest markers suffice |

### Sampling Rate

- **Per task commit:** `.venv/bin/pytest -m "not slow" -x` — fast suite (no FluidSynth), runs in ~5s, blocks commit on first failure.
- **Per wave merge:** `.venv/bin/pytest -m "not slow"` — full fast suite (~10s), confirms no cross-module regression.
- **Phase gate:** `.venv/bin/pytest` (full suite including slow) — confirms determinism golden passes. In CI, goldens are captured on a pinned-FluidSynth runner; local developer can `pytest --regen-goldens` if intentionally changing RNG order.

### Wave 0 Gaps

- [ ] `tests/conftest.py` — NEW (Phase 3 deleted the old one; D-32 creates a new minimal one for `--regen-goldens` only). **Blocker for D-32/D-41.**
- [ ] `tests/test_seeds.py` — NEW (D-36). Covers R-P7.
- [ ] `tests/test_writer.py` — NEW (D-37). Covers R-P1, R-P2, R-P3, R-P4 partial.
- [ ] `tests/test_manifest.py` — NEW (D-38). Covers R-P5.
- [ ] `tests/test_split.py` — NEW (D-39). Covers R-P6.
- [ ] `tests/test_api.py` — NEW (D-40). Covers R-P12 (single-sample), R-P4 residuals, R-P7 validation.
- [ ] `tests/test_determinism_golden.py` — NEW (D-41). Covers R-P8, R-Q3.
- [ ] `tests/fixtures/determinism/expected_{mix,midi_beat,midi_melody,midi_harmony,midi_bassline,sample}.sha256` — NEW (D-28). Captured via `--regen-goldens` on pinned-FluidSynth host.
- [ ] `tests/fixtures/determinism/fluidsynth_version.txt` — NEW (D-28). First line of `fluidsynth --version` when fixtures captured.
- [ ] Existing `tests/test_integration_full_generation.py` needs **migration** from `music_gen.create_song(...)` to `musicgen.generate(Config(...))` (D-34 deletes `create_song`).

## Security Domain

> `security_enforcement` not explicitly `false` — section included.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | — (library; no auth surface) |
| V3 Session Management | no | — (library; no sessions) |
| V4 Access Control | no | — (library; no multi-user) |
| V5 Input Validation | yes | `Config.__post_init__` validates `split_ratios` sum / non-negativity (CONTEXT D-27); `global_seed is None` raises ValueError (D-21). `dataset_root` paths joined with `os.path.join`; 6-digit index is format-string-generated, not user-supplied (D-05). No SQL, no shell, no deserialization of untrusted input. |
| V6 Cryptography | yes-ish | SHA-256 via `hashlib` for seed derivation + split hash + golden fixtures. **Not used for security** — used for deterministic, collision-resistant hashing. `hashlib.sha256` is stdlib, FIPS-approved, no hand-rolling. |
| V7 Error Handling | yes | All exceptions inside `api.generate` funnel into `status="failed"` manifest entry (R-P16 Phase 6 completes aggregation; Phase 5 isolates per-sample). Traceback capped at 2KB (D-13) — no PII/secret exposure risk (pipeline processes MIDI + audio, not user data). |
| V10 Malicious Code | no | — (no untrusted input parsing) |
| V12 File and Resources | yes | `tempfile.mkdtemp(prefix="musicgen-")` for working dirs (D-31 step 4) — collision-free, permissions default to 0o700 via stdlib. `shutil.rmtree(..., ignore_errors=True)` for cleanup (D-10). No symlink traversal concerns — all paths are under `dataset_root` or `/tmp/musicgen-*`. |
| V14 Configuration | yes | Env var `MUSICGEN_DATASET_ROOT` follows Phase 2 D-01 `MUSICGEN_SF_DIR` precedent — normalized via `os.path.abspath` (T-02-01 mitigation). CLI > env > defaults precedence preserved. No secrets in Config. |

### Known Threat Patterns for Python-library / stdlib stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Path traversal via malicious `sample_index` | Tampering | `f"{sample_index:06d}"` format-string construction; integer-only typing — a non-int raises `ValueError` at format time; negative int rejected via explicit check in `api.generate` |
| Race on parallel `generate()` with same `(global_seed, sample_index)` | DoS (duplicate work) + Tampering (corrupt sample) | Sentinel check at start of `generate` + atomic `rename` at end. Two concurrent generations with same index: whichever finishes first wins; the other's `os.rename` either succeeds (replacing identical bytes) or the starting sentinel check skips entirely. **Phase 6 ProcessPool coordinates via sample-index partitioning (workers get disjoint index ranges) — not a Phase 5 concern.** |
| Manifest corruption under concurrent write | Tampering | `threading.Lock()` + `open(..., "a")` + `os.fsync()` — POSIX `O_APPEND` atomicity for writes ≤ 4096 bytes. Our lines are ~200 bytes. |
| Seed predictability | Information disclosure (non-threat in this context) | The dataset is **public training data**, not a cryptosystem. Seed predictability is the *point* — we WANT it for reproducibility. No threat. |
| Denial via infinite `generate_song_measures` loop | DoS | Phase 3 `SongParams.sample` has `while True` with `validate_measures_dict` — known-terminating per Phase 3 D-21 (preserves pre-refactor behavior). Not a Phase 5 regression risk. |
| Soundfont file injection (malicious `.sf2`) | Tampering | Out of scope — `sf/*.sf2` files are developer-supplied, not user-supplied. PITFALLS P-9 (licensing) is the only `.sf2`-related concern; deferred to public release. |

## Project Constraints (from CLAUDE.md)

`./CLAUDE.md` does not exist in the working directory. No project-specific directives beyond what's in CONTEXT.md / STATE.md / CONVENTIONS.md.

Established conventions (from `.planning/codebase/CONVENTIONS.md`, reinforced by Phase 3/4 code):
- `from __future__ import annotations` at top of every new module (used consistently in Phase 4 modules).
- Module-level `logger = logging.getLogger(__name__)` — no direct `print()` (Phase 2 R-S7).
- `logger.debug` for state dumps, `logger.info` for milestones, `logger.warning` for recoverable oddities, `logger.error` for failures.
- `from typing import Tuple, Dict, List, Optional` — stdlib typing (no PEP 604 `int | str` yet).
- `@dataclass(frozen=True)` — no `slots=True` unless size matters (CONTEXT mentions `slots=True` in D-02 but existing Phase 4 dataclasses `RenderResult`/`MixResult` do NOT use slots; match the existing pattern for consistency — planner to decide).
- Google-style docstrings with `Args:` / `Returns:` / `Raises:` sections.
- `cfg: config.Config = None` with runtime fallback `_cfg = cfg if cfg is not None else config.Config()` — pattern is consistent in Phases 2/3/4.

## User Constraints (from CONTEXT.md)

### Locked Decisions

See `.planning/phases/05-productize-i-writer-manifest-seeds-determinism/05-CONTEXT.md`'s `<decisions>` block (D-01 through D-43). Key highlights the planner MUST honor:

- **D-01:** Four new modules: `seeds.py`, `writer.py`, `manifest.py`, `api.py` — all under `src/musicgen/`.
- **D-02:** `SampleResult` lives in `api.py`, not `writer.py`. Frozen dataclass, 11 fields.
- **D-03:** `musicality_score.py` **moves** to `src/musicgen/musicality.py` this phase. No back-compat shim.
- **D-04:** Atomic per-sample write order: midi/ → stems/ → mix.wav → sum-of-stems assertion → sample.json sentinel.
- **D-05:** Zero-padding width = 6 (`<idx:06d>`). Hardcoded in writer, not Config-configurable.
- **D-06:** Stem concat across parts via `pydub.AudioSegment.from_wav + from_wav` pattern (matches `mixer.concat_parts`).
- **D-07:** MIDI concat across parts via `mido.MidiFile` with cumulative tick offsets. **Planner note:** the RESEARCH clarifies the implementation as an absolute-tick walk, not a naive track append.
- **D-09:** `dataset_root` added to Config; new fields: `global_seed`, `sample_index`, `split_ratios`, `sum_of_stems_epsilon`, `keep_working_dirs`, `workers` (reserved).
- **D-10:** Working-dir cleanup via `shutil.rmtree(..., ignore_errors=True)` after sentinel write.
- **D-11/D-12:** Relative paths in `sample.json`, rewritten by writer (not annotator — annotator stays pure).
- **D-13:** `manifest.jsonl` schema — 9 keys (`sample_index`, `seed`, `sample_seed`, `status`, `split`, `path`, `musicality_score`, `duration_seconds`, `wrote_at`).
- **D-14:** `ManifestWriter(dataset_root, lock: Optional[ContextManager] = None)`. Default `threading.Lock()`. **Not `Manager().Lock()`** — Phase 6's concern.
- **D-15:** Append-only; last-status-wins on re-run.
- **D-16:** `is_sample_complete` checks sentinel **only**, never reads manifest.
- **D-17:** `derive_sample_seed(global_seed, sample_index)` verbatim from ARCHITECTURE.md.
- **D-18:** `make_rngs(sample_seed)` verbatim — five domains, XOR with small constants.
- **D-19:** RNG domain mapping — `params/generators/soundfonts/fx/mix` to the specific call sites.
- **D-20:** `save_random_state()` context manager wraps ONLY `musicality.get_musicality_score` call.
- **D-21:** `config.global_seed is None` → `ValueError`. No silent entropy.
- **D-22:** Annotator TBDs filled by api.py: `seed=sample_seed`, `musicgen_version=importlib.metadata.version(...)` with `"0.1.0+uninstalled"` fallback, `split=assign_split(sample_seed, ratios)`. `pre_roll_offset_seconds` stays `None` (R-P9 Phase 6).
- **D-23:** `json.dump(annotation, f, sort_keys=True, indent=2, separators=(",", ": "))`.
- **D-24/D-25:** Sum-of-stems inside writer, ε=1e-3 normalized float32.
- **D-26/D-27:** Split hash `sha256(f"split:{sample_seed}")[:4] % 10000 / 100.0` with default 80/10/10.
- **D-28/D-29:** 6 SHA-256 fixture files + `fluidsynth_version.txt` skip gate.
- **D-30:** Fast no-FluidSynth `sample.json` byte-stability cross-check.
- **D-31:** `api.generate(config)` is single-sample only. `generate_batch` is Phase 6.
- **D-32:** `pytest --regen-goldens` flag via `pytest_addoption` in `tests/conftest.py`.
- **D-33:** `music_gen.py` stays as ~40-line smoke wrapper. Not deleted.
- **D-34:** `music_gen.py`'s `create_song` + `generate_song_parts` + `generate_song` **DELETED**. Bodies migrate into `api.py`.
- **D-35:** `__init__.py` exports `generate`, `Config`, `SampleResult`, `__version__`.
- **D-36..D-41:** Six new test files.
- **D-42:** Existing AST guard auto-covers new modules via parametrize-glob (no test edit needed).

### Claude's Discretion

- `SampleResult` field naming (`sample_dir` vs `sample_path`): **recommend `sample_dir`** (matches R-P1 language).
- Config field name (`dataset_root` vs `output_dir`): **CONTEXT locks `dataset_root`**.
- `musicgen_version` fallback (`"0.1.0+uninstalled"` vs raise): **CONTEXT locks fallback**.
- Working dir location (`/tmp` via `mkdtemp` vs `<dataset_root>/.tmp/`): **recommend `tempfile.mkdtemp`**.
- Silent-MIDI representation (zero-velocity note_on+off vs empty track with end_of_track): **recommend match `generators/beat.py` idiom — zero-velocity pair**.
- `sample.json` `split` field casing (`"train"` vs `"Train"`): **CONTEXT recommends lowercase**.
- Split modulo (10000 for 0.01% precision vs 1000 for 0.1%): **CONTEXT picks 10000**.
- `_concat_layer_stems` via pydub (match `mixer.concat_parts`) vs scipy (maybe faster): **recommend pydub** for consistency.

### Deferred Ideas (OUT OF SCOPE)

Verbatim from CONTEXT.md `<deferred>`:
- `generate_batch(config)` via `ProcessPoolExecutor` — Phase 6 (R-P10).
- `typer`-based full CLI — Phase 6 (R-P13).
- FluidSynth pre-roll calibration (`calibrate.py`) — Phase 6 (R-P9). Phase 5 writes `pre_roll_offset_seconds: None`.
- `--output-mode` flag (`full`/`mix-only`/`stems-only`/`midi-only`) — Phase 6 (R-P14).
- Structured JSON progress logs during batch — Phase 6 (R-P15).
- `musicgen clean --failed` subcommand — Phase 6 (R-P13 second bullet).
- Sharded `<dataset>/<hex>/<id>/` layout — v0.2+.
- Custom split labels beyond train/valid/test — v0.2+.
- CI golden-regeneration automation — v0.2+.
- Compressed annotations (`sample.json.gz`) — v0.2+.
- Per-sample working-dir preservation mode — `config.keep_working_dirs = True` suffices.

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| R-P1 | Per-sample output layout `<dataset-root>/<zero-padded-index>/` with mix.wav, stems/*.wav, midi/*.mid, sample.json | Writer design (Pattern 2); CONTEXT D-04/D-05; atomic sentinel via `os.rename` (Pitfall 3) |
| R-P2 | Post-FX stems persisted + sum-of-stems assertion `max(|Σstems − mix|) < ε` | Sum-of-stems code example (int32 accumulator); ε=1e-3 verified sharp on silent+valid cases; CONTEXT D-24/D-25 |
| R-P3 | Per-layer MIDI files persisted in `midi/` | MIDI concat via absolute-tick walk (Research Question 3 below, code example in Pattern 1 helper); CONTEXT D-07/D-08 |
| R-P4 | Full `sample.json` schema — seed, fluidsynth_version, musicgen_version, arrangement, chord_progressions, active_layers, soundfonts, FX, beat/downbeat times, musicality, relative paths | Annotator (Phase 4) emits shape; api.py fills TBDs (CONTEXT D-22); writer rewrites paths (D-11/D-12); sort_keys canonicalization verified byte-stable |
| R-P5 | `manifest.jsonl` append under `multiprocessing.Manager().Lock()` | Manifest design (Pattern 3); lock abstraction via `ContextManager` type (CONTEXT D-14); `threading.Lock` default, Manager lock Phase 6 swap — both verified context-manager compatible |
| R-P6 | Deterministic train/valid/test split via stable seed hash | `assign_split` code example (Pattern 1); CONTEXT D-26/D-27 (80/10/10 default); `sha256` hash verified deterministic |
| R-P7 | `derive_sample_seed`, `make_rngs`, named domain RNGs, no bare `random.*`, save_random_state | Pattern 1 (seeds.py); CONTEXT D-17/D-18/D-19/D-20/D-21; AST guard (Phase 4 D-31) auto-covers new modules (verified via parametrize glob) |
| R-P8 | Determinism contract — bit-identical MIDI + sample.json; bit-identical WAV under pinned FluidSynth | MIDI concat uses absolute-tick walk for bit-stability; `sort_keys=True` for JSON; FluidSynth pinning via `fluidsynth_version.txt` skip gate (CONTEXT D-28/D-29); cross-check test (D-30) detects non-FluidSynth regressions |
| R-Q3 | Regression test in CI | `tests/test_determinism_golden.py` + `@pytest.mark.slow` — `pytest -m slow` opt-in, default `-m "not slow"` keeps CI fast; `--regen-goldens` flag for intentional regen |

## Research Questions — Detailed Findings

> The additional_context identified 15 specific sub-questions. Below are findings for each, with confidence and empirical verification where applicable.

### Q1. `scipy.io.wavfile.read` semantics with pydub-produced output

**Finding:** `scipy.io.wavfile.read(path)` returns `(rate: int, data: np.ndarray)` where `data.dtype` matches the WAV sample width. For pydub-exported 16-bit int WAVs (the default `AudioSegment.export(..., format='wav')` path), `data.dtype == np.int16` and `data.shape == (n_frames, n_channels)` for stereo (n_channels=2), or `(n_frames,)` for mono. **Verified in venv:** silent pydub stereo 44.1kHz WAV → `rate=44100, dtype=int16, shape=(22050, 2)`. `sample_width` on the AudioSegment is 2 (bytes) confirming int16 internally.

**Frame count:** `n_frames == duration_ms * sample_rate / 1000`. For a 500ms WAV at 44100Hz, `n_frames = 22050`. Shape tuple gives the count directly.

**Confidence:** HIGH `[VERIFIED: research probe]`

### Q2. `pydub.AudioSegment` concatenation + WAV round-trip bit-identity

**Finding:** `AudioSegment.from_wav(a) + AudioSegment.from_wav(b)` preserves sample rate, channels, and frame alignment for same-format WAVs. The `+` operator concatenates raw bytes after stripping the WAV header, then re-wraps. **Verified:**
- `concat(p1, p2).export(out, format='wav')` produces the same SHA-256 on two separate runs (byte-identical).
- `AudioSegment(raw_data, ...).export(p).AudioSegment.from_wav(p)` is **bit-identical round-trip** (`raw_data` before == `raw_data` after; SHA-256 of re-exported file matches original).

**Fade/transition boundaries:** The current mixer (`mixer.concat_parts`) does NOT apply any fade — it's a hard splice. If a part ends at a non-zero sample and the next starts at a non-zero sample, there's a click. This is existing behavior; Phase 5 inherits it. Not a regression.

**Implication for sum-of-stems:** Because pydub is byte-identical round-trip, stems read back via scipy are the same bytes the mixer exported. The ε=1e-3 threshold has **all its headroom available for pedalboard's numerical drift** — the pydub layer contributes zero error. This makes the assertion sharp and useful.

**Confidence:** HIGH `[VERIFIED: research probe — SHA-256 match + raw_data equality]`

### Q3. `mido.MidiFile` track merging across files

**Finding:** The cleanest API is **absolute-tick walk**, not naive track append. The reason is that `midiutil` writes tracks whose `sum(msg.time)` can be less than the intended part duration (trailing silence has no events). Naive concat puts the next part's first note too early.

**Empirical probe:**
- A 2-measure 4/4 @ 120 bpm MIDI with only 2 notes has `track[1]` accumulated ticks = 1920 (not 3840 expected).
- Merged via naive append → next part's notes start at tick 1920 instead of 3840.
- Merged via absolute-tick walk with `offset_ticks = mido.second2tick(render_result.duration_seconds, ticks_per_beat, tempo_us)` → correct.

**Multiple tempo events:** CONTEXT D-07: "If parts have conflicting tempo, raise `ValueError`." musicgen passes a single tempo through `create_song`, so all parts share `mf.addTempo(0, 0, tempo)` — same metadata. Writer checks `mf1.tracks[0]` tempo vs `mf2.tracks[0]` tempo; raises on mismatch. Drops duplicate tempo meta events in the merged file (first one wins).

**Track-zero conventions:** `midiutil` writes Type-1 MIDI with `tracks[0]` as tempo/meta and `tracks[1]` as the note track. Merged file keeps this: `merged.tracks[0]` has ONE tempo meta + one end_of_track; `merged.tracks[1]` has concatenated notes.

**Delta-vs-absolute ticks:** `mido.MidiTrack` stores delta ticks (each `msg.time` is delta from the previous message). The absolute-tick helper converts to absolute on read, merges with offset, converts back to delta on write.

**Determinism:** `merged.save(out)` twice produces byte-identical files. **Verified** — `hashlib.sha256` matches.

**Confidence:** HIGH `[VERIFIED: research probe]`

### Q4. `multiprocessing.Manager().Lock()` vs `threading.Lock()`

**Finding:** API-identical for `with lock:` usage. Both support `acquire()`, `release()`, `__enter__`, `__exit__`. `Manager().Lock()` returns a `multiprocessing.managers.AcquirerProxy` that marshals calls over IPC to a manager subprocess. **Verified in probe:**
- `threading.Lock()` is `<class '_thread.lock'>`; `hasattr(lock, '__enter__')` → True.
- `Manager().Lock()` is `AcquirerProxy`; `hasattr(lock, '__enter__')` → True.

**Implication:** `ManifestWriter.append(entry)` body is identical regardless of lock type. CONTEXT D-14 defaults to `threading.Lock()` (cheap, Phase 5 is single-process); Phase 6 `generate_batch` passes `multiprocessing.Manager().Lock()`, zero code change in `manifest.py`.

**Overhead:** `threading.Lock` acquire ~100ns; `Manager().Lock()` acquire ~1ms (IPC round-trip). For 10k samples with ~5 manifest appends per sample at max, that's ~50ms of lock overhead even with Manager — negligible vs per-sample FluidSynth time (~1-10s).

**Confidence:** HIGH `[VERIFIED: research probe + stdlib docs]`

### Q5. `importlib.metadata.version("musicgen")` behavior

**Finding:** Raises `importlib.metadata.PackageNotFoundError` if the package lacks metadata (i.e., not pip-installed). Returns `pyproject.toml`'s `version = "0.1.0"` when installed (editable or non-editable). **Verified:**
- `.venv/bin/python -c 'import importlib.metadata; print(importlib.metadata.version("musicgen"))'` → `0.1.0` (editable-installed).
- System python (no musicgen install) → `PackageNotFoundError: No package metadata was found for musicgen`.

**Fallback pattern (CONTEXT D-22):**
```python
try:
    MUSICGEN_VERSION = importlib.metadata.version("musicgen")
except importlib.metadata.PackageNotFoundError:
    MUSICGEN_VERSION = "0.1.0+uninstalled"
```

**Affects the determinism golden:** `sample.json` contains `"musicgen_version": "0.1.0"`. When regenerating goldens (`--regen-goldens`), the test environment MUST have `pip install -e .[dev]` run first. Otherwise the golden captures `"0.1.0+uninstalled"` and subsequent installed-package runs fail the comparison. **Document in `tests/fixtures/determinism/README.md` or the `--regen-goldens` docstring.**

**Confidence:** HIGH `[VERIFIED: research probe]`

### Q6. SHA-256 determinism of floating-point audio after WAV write/read round-trip

**Finding:** Bit-identical across runs on the **same FluidSynth binary + same soundfont + same MIDI** — documented by PITFALLS P-1 and R-P8. Cross-machine / cross-binary: NOT guaranteed. CONTEXT D-28/D-29 accepts this via the `fluidsynth_version.txt` skip gate.

**WAV header padding:** `scipy.io.wavfile.write` produces a canonical WAV header (44 bytes for 16-bit PCM); `pydub.AudioSegment.export(..., format='wav')` uses ffmpeg's WAV writer which adds an 'INFO' LIST chunk by default (~40 bytes of metadata). If the writer converts any working-dir WAVs through pydub and exports to final, the header differs from a scipy-written version. **Implication:** the determinism golden MUST be captured through the same writer code path that generates in production. Writer uses `pydub.AudioSegment(...).export(..., format='wav')` (same as mixer), so round-trips are byte-stable.

**Byte order:** WAV is little-endian by spec; all our target platforms (Linux x86_64, Linux ARM64) are little-endian. Mac M-series is little-endian too. No issue.

**Recommendation:** Capture the mix.sha256 golden on the same Linux x86_64 + FluidSynth version that CI uses. `fluidsynth_version.txt` captures the binary identity.

**Confidence:** MEDIUM (the P-1 caveat lives here). `[CITED: PITFALLS P-1, ARCHITECTURE §Seed/RNG point 4, CONTEXT D-28/D-29]`

### Q7. `random.Random(seed ^ 0x01)` XOR seeding determinism

**Finding:** Deterministic across CPython 3.10+. `random.Random(seed)` with int `seed` uses the Mersenne Twister init algorithm; XORing with a small constant produces a different seed int, deterministically. **Verified in probe:**
```python
r1 = random.Random(12345 ^ 0x01); r2 = random.Random(12345 ^ 0x01)
assert [r1.random() for _ in range(3)] == [r2.random() for _ in range(3)]  # True
```

**PyPy:** `requires-python = ">=3.10"` in `pyproject.toml` + CPython-only dep pins (numpy, scipy, pedalboard — all distribute wheels for CPython) means PyPy is not in scope. No concern.

**Cross-minor-version stability (3.10 vs 3.11 vs 3.12):** The Mersenne Twister is spec'd. Python's `random` module has not changed its algorithm in decades. Any change would be a major Python-level break (release notes would call it out). Current de-facto stable.

**Confidence:** HIGH `[VERIFIED: research probe + Python docs stability commitment]`

### Q8. pytest fixture pattern for `--regen-goldens` flag

**Finding:** Canonical pattern uses `pytest_addoption` in `conftest.py` + `request.config.getoption(...)` in the test.

```python
# tests/conftest.py
def pytest_addoption(parser):
    parser.addoption(
        "--regen-goldens", action="store_true", default=False,
        help="Regenerate determinism fixtures instead of asserting.",
    )

# tests/test_determinism_golden.py
@pytest.fixture
def regen(request):
    return request.config.getoption("--regen-goldens")

def test_mix_sha256(tmp_path, regen):
    result = musicgen.generate(Config(global_seed=1, sample_index=0, dataset_root=str(tmp_path)))
    h = hashlib.sha256(open(result.mix_path, 'rb').read()).hexdigest()
    golden = FIXTURES / "expected_mix.sha256"
    if regen:
        golden.write_text(h + "\n")
    else:
        assert h == golden.read_text().strip()
```

**Cross-reference:** `pytest --help` includes `--regen-goldens` automatically under "custom options."

**Gotcha:** `pytest_addoption` in a nested conftest (e.g., `tests/fixtures/conftest.py`) is **ignored** — must be in the top-level `tests/conftest.py` per pytest docs.

**Confidence:** HIGH `[CITED: https://docs.pytest.org/en/stable/reference/reference.html#pytest.Parser.addoption]`

### Q9. `tempfile.mkdtemp` + `shutil.rmtree` atomicity after `os.chdir`

**Finding:** `shutil.rmtree` succeeds on Linux even if a process holds an open handle (including cwd) to the directory — the directory entry vanishes but the inode persists until handles close. `os.getcwd()` on the process that chdir'd in returns an "unlinked" path (may error on some kernels).

**CONTEXT's pattern (D-31 step 4):** `working_dir = tempfile.mkdtemp(prefix="musicgen-")` returns an absolute `/tmp/musicgen-XXX` path. `api.generate()` NEVER `chdir`s — it passes `working_dir` as an absolute path to every callee. `shutil.rmtree(working_dir, ignore_errors=True)` at the end is safe.

**Verified probe:**
```python
td = tempfile.mkdtemp(prefix='musicgen-')
os.chdir(td); os.chdir(orig); shutil.rmtree(td, ignore_errors=True)
assert not os.path.isdir(td)  # True — removed
```

**Recommendation for planner:** Forbid `os.chdir` inside `api.generate`. All callees (renderer, mixer, beats) already take `out_dir` params — they don't need cwd.

**Confidence:** HIGH `[VERIFIED: research probe]`

### Q10. Atomicity of `os.rename` for sentinel `sample.json`

**Finding:** POSIX `rename(2)` is atomic **within a single filesystem** — either the rename succeeds and the new name refers to the full file contents, or it fails and both names are unchanged. **Across filesystems** (e.g., tmpfs `/tmp` vs ext4 `/dataset`): on Linux, `rename(2)` returns `EXDEV` — Python's `os.rename` **raises `OSError` with `errno.EXDEV`**. Python's `os.replace` is a thin wrapper for `rename`; same semantics.

**Writer strategy (recommended):**
1. Write `<sample_dir>/sample.json.tmp` (same dir as final).
2. `os.rename(<sample_dir>/sample.json.tmp, <sample_dir>/sample.json)` — atomic within `<sample_dir>`'s filesystem.
3. Mix/stems/MIDIs are **copied** from `/tmp/musicgen-XXX/` into `<sample_dir>/` first via `shutil.copy2` (NOT rename — the cross-FS case requires copy anyway). Writer doesn't need them renamed atomically — partial samples are only visible through the sentinel.

**Recommendation:** Document the sentinel-rename as "atomic within the sample directory's filesystem; mix/stems/MIDIs are copied before the sentinel, so a crash between copy and rename leaves a partially-populated `<sample_dir>` with NO sentinel — resume correctly skips."

**Confidence:** HIGH `[CITED: POSIX rename(2) spec]`

### Q11. NumPy sum-of-stems computation dtype

**Finding:** **int32 accumulator, cast from int16, compare to mix as int32, normalize with `/32768.0` to float for ε comparison.** Done correctly, overflow is impossible (4 × int16 fits in int32 with 13 bits to spare).

**Code (verified by probe):**
```python
sums_i32 = np.zeros_like(mix_i16.astype(np.int32))
for layer, path in stem_paths.items():
    _, stem_i16 = scipy.io.wavfile.read(path)
    sums_i32 += stem_i16.astype(np.int32)
diff_i32 = sums_i32 - mix_i16.astype(np.int32)
max_abs_float = np.max(np.abs(diff_i32)) / 32768.0
passed = max_abs_float < 1e-3
```

**Probe result:** Four random int16 stems summed to a reference int16 mix yield `max |sum - mix| = 0` when the stems are constructed to sum cleanly (no pedalboard noise). The assertion has its full ε=1e-3 budget available for pedalboard drift.

**Edge case — different stem lengths:** `sums_i32 += stem_i16.astype(np.int32)` raises `ValueError` on shape mismatch. Writer's stem concat guarantees all 4 layers match the mix length (phase 4 silent-stem fallback + phase 5 stem-concat preserve per-part durations). Add an explicit `assert stem_i16.shape == mix_i16.shape` in the helper for clarity.

**Confidence:** HIGH `[VERIFIED: research probe]`

### Q12. Cross-phase hot take — generator RNG threading

**Finding:** All 4 generators (`chord.py`, `melody.py`, `bassline.py`, `beat.py`) already accept `rng: random.Random` as a parameter with zero bare `random.*` calls. **Verified via Grep:**
- `chord.py:40`: `rng: random.Random` parameter; calls `rng.choice` at line 80.
- `melody.py:36`: `rng: random.Random`; calls `rng.choice`/`rng.choices`/`rng.randint` at lines 106/110/118/138.
- `bassline.py:37`: `rng: random.Random`; calls `rng.choice`/`rng.choices`/`rng.random`/`rng.randint` at lines 96/106/125/134.
- `beat.py:61`: `rng: random.Random`; calls `rng.choice` at lines 125/130.

**Phase 5 simply swaps `music_gen.py`'s `_rng = random.Random()` for `rngs["generators"]` at the 4 call sites in `_generate_all_midi` (extracted from `generate_song_parts`).** No function signature changes needed.

**Known landmine:** None found. The extracted generators + sampler have been stable through Plans 03-03 (sampler), 03-04 (generators), 03-05 (phase-gate). The music21 isolation test (`test_music21_isolation.py`) is the regression guard.

**Confidence:** HIGH `[VERIFIED: Grep probe + Phase 3 summaries]`

### Q13. MIDI reproducibility across runs with identical seed

**Finding:** `midiutil.MIDIFile` write produces byte-identical output for identical inputs + seeds. **Verified:** building the same MIDI twice in the probe yielded identical SHA-256. Phase 4 `test_integration_full_generation.py::TestMidiReproducibility::test_same_seed_produces_same_midi` already asserts this for the 4-layer per-part MIDIs. Phase 5's writer concatenates these via `mido` — as long as the concat is deterministic, the output MIDI is bit-stable.

**`mido.save` determinism:** Verified — saving the same `MidiFile` twice produces identical bytes. No timestamp, no random nonce.

**Absolute-tick walk determinism:** The helper iterates `mido.tracks` in order, accumulates ticks deterministically, sorts final absolute-tick list with `key=lambda x: x[0]` (stable sort on int keys). No nondeterminism.

**Confidence:** HIGH `[VERIFIED: research probe + Phase 4 integration test]`

### Q14. `AudioSegment.from_wav` bit-perfect preservation

**Finding:** Bit-identical for int16 WAV round-trips. **Verified:** `AudioSegment(raw_bytes).export(p1).AudioSegment.from_wav(p1).export(p2)` — `sha256(p1) == sha256(p2)` and `raw_data` equality after re-read. pydub internally decodes via wave.py (stdlib) for WAV → raw_bytes (no transcoding); re-encodes via wave.py too. No dtype drift, no header variance between runs.

**Implication for sum-of-stems:** Because pydub round-trip is bit-identical, the ε=1e-3 budget is spent entirely on pedalboard FX + `AudioSegment.overlay` arithmetic. The FX layer introduces the only real error. ε=1e-3 = –60 dBFS is ~3 orders of magnitude above pedalboard's ~–120 dBFS floor (per pedalboard docs: Reverb / Delay use 32-bit floats internally → 24-bit int quantization error ≈ –144 dBFS on export back to 16-bit). Safe headroom.

**Confidence:** HIGH `[VERIFIED: research probe]`

### Q15. Concurrent append to `manifest.jsonl` under `threading.Lock`

**Finding:** No gotchas for single-process threaded appends. `open(path, "a")` + `f.write(s)` + `f.flush()` + `os.fsync(f.fileno())` inside a `with lock:` block is bulletproof for writes ≤ 4096 bytes (POSIX `PIPE_BUF` atomicity guarantee). Our manifest lines are ~200 bytes — well under.

**`os.fsync` necessity:** With `threading.Lock`, all threads in one process share a single OS file handle (same kernel buffer); `flush()` pushes Python's userland buffer to kernel; `fsync()` forces kernel to flush to disk. Without fsync, a crash between flush and sync loses the last manifest line. For 10k-sample runs with per-sample fsync, the latency cost is acceptable (SSD fsync ≈ 1ms; 10k × 1ms = 10s over a multi-hour run).

**Multi-process case (Phase 6):** Each worker opens its own file handle but appends under `Manager().Lock()`. POSIX `O_APPEND` semantics guarantee seek-to-end + write is atomic for ≤ PIPE_BUF, even across processes. The Manager lock serializes the Python-level `open()`+`write()`+`fsync()` sequence, further reducing contention.

**Writing JSON canonicalized:** `json.dumps(entry, sort_keys=True) + "\n"` — stable byte order for a given entry dict.

**Confidence:** HIGH `[CITED: POSIX write(2); verified by threading.Lock probe]`

## Sources

### Primary (HIGH confidence)

- Python stdlib documentation — hashlib, random, threading, multiprocessing, tempfile, shutil, importlib.metadata, contextlib, json, os.path, pathlib (all 3.10+).
- pytest docs — `pytest_addoption`, markers, fixtures: https://docs.pytest.org/en/stable/reference/reference.html
- mido docs — MidiFile, MidiTrack, tick2second, second2tick, bpm2tempo: https://mido.readthedocs.io/en/latest/files.html
- scipy.io.wavfile docs — `read` returns `(rate, ndarray)` with dtype matching sample width: https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.read.html
- pydub docs — `AudioSegment.from_wav`, `.export`, overlay, silence: https://github.com/jiaaro/pydub/blob/master/API.markdown
- POSIX spec — rename(2) atomicity; write(2) O_APPEND atomicity for ≤ PIPE_BUF.

### Secondary (verified by research probe in venv)

- scipy.io.wavfile read returns int16 dtype + (n_frames, n_channels) shape for pydub stereo output — verified.
- pydub.AudioSegment round-trip is byte-identical — verified via sha256 + raw_data comparison.
- midiutil writes Type-1 MIDI with note track as `tracks[1]` — verified.
- midiutil notes-only track has `sum(msg.time)` potentially < total part duration — **verified** (the MIDI-concat subtlety).
- importlib.metadata.version("musicgen") returns "0.1.0" when editable-installed, raises `PackageNotFoundError` otherwise — verified.
- threading.Lock() and multiprocessing.Manager().Lock() both implement `__enter__/__exit__` — verified.
- random.Random(seed ^ 0x01) is deterministic + differs from random.Random(seed) — verified.
- json.dumps(d, sort_keys=True) produces byte-identical output across runs — verified.
- tempfile.mkdtemp + shutil.rmtree after chdir-out is safe — verified.

### Prior phase artifacts (consulted)

- `.planning/phases/04-*/04-CONTEXT.md` — annotator contract (D-14/D-15/D-16), RNG threading (D-17/D-18), AST guard (D-31), integration test pattern (D-30).
- `.planning/phases/03-*/03-*SUMMARY.md` — RNG injection established, music21 isolation proven.
- `.planning/research/ARCHITECTURE.md` §"Seed/RNG propagation" — D-17/D-18 copy this verbatim.
- `.planning/research/PITFALLS.md` P-1, P-2, P-4, P-7 — gated by this phase's decisions.
- `.planning/research/STACK.md` — stdlib-only Phase 5 confirmed.
- `src/musicgen/mixer.py`, `renderer.py`, `annotator.py`, `beats.py` — Phase 4 APIs verified intact.
- `src/musicgen/sampler.py`, `generators/*.py` — rng parameter on all callables verified via Grep.
- `music_gen.py` (current 199 lines) — D-34 deletion targets verified present.

### Tertiary (none)

No low-confidence claims in this research — every load-bearing assertion was probed or cited to authoritative source.

## Metadata

**Confidence breakdown:**

| Area | Level | Reason |
|------|-------|--------|
| Standard stack | HIGH | Entirely stdlib + already-installed deps; every mechanism verified by venv probe |
| Architecture | HIGH | Wave order locked by CONTEXT.md additional_context; data flow matches ARCHITECTURE.md §"Per-sample output layout" + §"Seed/RNG propagation" verbatim |
| Pitfalls | HIGH | Two novel pitfalls (MIDI concat tick-underflow, int16 sum overflow) both caught + mitigation code examples verified in probe; other 6 pitfalls are standard POSIX/library facts |
| MIDI concat semantics | HIGH | Absolute-tick walk verified against midiutil-generated Type-1 MIDIs with probe; naive concat shown to fail on trailing silence |
| Sum-of-stems assertion | HIGH | int32 accumulator + normalized-float ε verified; pydub round-trip byte-identity verified (full ε budget available for pedalboard drift) |
| FluidSynth bit-identity | MEDIUM | Cross-binary bit-identity is not guaranteed — known per P-1; CONTEXT D-28/D-29 handles via skip gate. All non-FluidSynth goldens (MIDI, sample.json) are HIGH confidence deterministic |

**Research date:** 2026-04-19
**Valid until:** ~2026-05-19 (30 days — stable stdlib + stable deps + all Phase 5 decisions LOCKED in CONTEXT.md. The only time-sensitive item is if scipy/numpy/pydub/mido minor-version upgrade introduces behavior changes — none currently on the horizon.)

## RESEARCH COMPLETE
