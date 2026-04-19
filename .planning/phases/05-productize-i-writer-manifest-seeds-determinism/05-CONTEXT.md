# Phase 5: Productize I — writer, manifest, seed discipline, determinism - Context

**Gathered:** 2026-04-19 (auto mode — recommended defaults applied per gray area)
**Status:** Ready for planning

<domain>
## Phase Boundary

Turn the Phase 4 pipeline into a **library-grade, reproducible, single-sample generator** whose output lands in a canonical per-sample directory layout. The mid-tier productize work lands this phase; the batch runner + CLI + pre-roll calibration is Phase 6.

Five concrete capabilities land:

1. **Per-sample output layout (R-P1).** `<dataset-root>/<zero-padded-index>/` with `sample.json`, `mix.wav`, `stems/{beat,melody,harmony,bassline}.wav`, `midi/{beat,melody,harmony,bassline}.mid`. `sample.json` is always written last (resume sentinel). Silent stems are written for absent layers.
2. **`manifest.jsonl` (R-P5).** Append-under-lock at `<dataset-root>/manifest.jsonl`, one line per sample, lock-abstraction ready for Phase 6's `multiprocessing.Manager().Lock()`.
3. **Seed discipline end-to-end (R-P7).** `derive_sample_seed(global_seed, index)` + `make_rngs(sample_seed) -> dict[str, random.Random]` with named domains (`params`, `generators`, `soundfonts`, `fx`, `mix`). `music_gen.py`'s single `_rng` is replaced; every Phase 3 / Phase 4 call site receives its domain-specific RNG. Zero bare `random.*` under `src/musicgen/` remains enforced by the AST guard.
4. **Full `sample.json` schema (R-P4).** Phase 5 fills what Phase 4 left as `None`: `seed`, `musicgen_version`, `split`, plus `pre_roll_offset_seconds` stub (actual calibration is R-P9 Phase 6). Absolute paths → dataset-root-relative paths.
5. **Determinism + integrity (R-P2, R-P6, R-P8, R-Q3).** Sum-of-stems assertion inside the writer (`max(|sum(stems) − mix|) < ε`). Deterministic `train / valid / test` split from a seed hash (default 80/10/10). Regression test `test_determinism_golden.py` asserts `sha256(mix.wav)` + `sha256(midi/*.mid)` + canonicalized `sha256(sample.json)` match checked-in goldens under the pinned FluidSynth binary.

After Phase 5: `python -c "from musicgen import generate; generate(Config(global_seed=42, sample_index=0, dataset_root='./dataset'))"` produces a fully-annotated sample at `./dataset/000000/`. Running it twice is bit-identical. The UUID-truncation bug (`music_gen.py:184`'s `song_name[:20]`) is gone.

**This phase does NOT:** implement `generate_batch` (R-P10, Phase 6), run `ProcessPoolExecutor` parallelism (R-P10, Phase 6), add the `typer` CLI (R-P13, Phase 6), calibrate FluidSynth pre-roll (R-P9, Phase 6), implement the `--output-mode` flag (R-P14, Phase 6), ship failure-isolation-in-a-batch (R-P16, Phase 6), or hand-roll resume logic beyond the sentinel check (R-P11, Phase 6). `generate()` is **single-sample** this phase; `generate_batch()` wraps it next phase.

</domain>

<decisions>
## Implementation Decisions

### Module split + file inventory

- **D-01:** Four new package modules this phase, all under `src/musicgen/`:
  - `src/musicgen/seeds.py` — pure functions. Surface: `derive_sample_seed(global_seed: int, sample_index: int) -> int`, `make_rngs(sample_seed: int) -> Dict[str, random.Random]`, `save_random_state()` context manager (D-18), plus the five domain-name constants (`RNG_PARAMS`, `RNG_GENERATORS`, `RNG_SOUNDFONTS`, `RNG_FX`, `RNG_MIX`).
  - `src/musicgen/writer.py` — owns the per-sample directory layout. Surface: `write_sample(dataset_root, sample_index, annotation, mix_working_path, stems_working_paths, midi_working_paths, *, fluidsynth_version) -> SampleResult` (atomic, sum-of-stems-asserting, returns paths for manifest append).
  - `src/musicgen/manifest.py` — append-under-lock. Surface: `class ManifestWriter(dataset_root: str, lock: ContextManager = threading.Lock())` with methods `append(entry: dict)` and staticmethod `is_sample_complete(dataset_root: str, sample_index: int, pad: int = 6) -> bool`.
  - `src/musicgen/api.py` — library entry point. Surface: `@dataclass Config`, `@dataclass SampleResult`, and `def generate(config: Config) -> SampleResult`. Re-exported from `src/musicgen/__init__.py` as `musicgen.generate` / `musicgen.Config`.
- **D-02:** No new dataclasses in `writer.py` / `manifest.py`. Writer returns a `SampleResult` owned by `api.py` (not `writer.py`) — the writer is a verb, the result belongs to the API. `SampleResult` fields: `sample_index`, `seed`, `sample_dir`, `sample_json_path`, `mix_path`, `stem_paths: Dict[str, str]`, `midi_paths: Dict[str, str]`, `split`, `status` (`"ok"` / `"failed"`), `musicality_score: float`, `duration_seconds: float`. Frozen, `slots=True` (Python 3.10+ per Phase 3 D-13).
- **D-03:** `musicality_score.py` **finally moves** into the package this phase (Phase 3 D-11 / Phase 4 D-04 both deferred this). Destination: `src/musicgen/musicality.py`. It is a leaf dependency and owning it under the package is a prerequisite for the `generate()` API surface being importable as `from musicgen import generate` with no repo-root scripts in play. Move preserves `get_musicality_score(wav_path) -> Tuple[float, Dict[str, float]]`; `api.py` owns the call site (not `music_gen.py` — see D-36).

### Writer design (R-P1, R-P2, R-P3)

- **D-04:** **Atomic per-sample write**. Writer creates `<dataset-root>/<index:06d>/` then writes in this **strict order**: midi/{layer}.mid → stems/{layer}.wav → mix.wav → **(sum-of-stems assertion, D-22)** → sample.json. `sample.json` is the resume sentinel; if the assertion fires the sentinel is NOT written. Partial samples are resumable-invisible (their `sample.json` is missing → Phase 6 resume retries them).
- **D-05:** **Zero-padding width = 6.** `<index:06d>` → `000000` through `999999`. Supports the 1k–10k v0.1 target with 100× headroom for a v0.2+ 100k dataset (the sharded layout becomes necessary at ~1M, out of scope per REQUIREMENTS "Out of scope"). Width is **hardcoded in writer** — not config-configurable — because changing it invalidates prior-run paths. Justified: never breaks, constant compile-time; the manifest records the index as an int anyway so a future reflow (Phase 6+) can rewrite filesystem names without schema churn.
- **D-06:** **Stem concatenation across parts**. Phase 4's `mixer.mix_part` writes per-part post-FX stems at `<working>/<name>-<part>/stems/<layer>.wav`; the writer concatenates same-layer WAVs across the arrangement using `AudioSegment.from_wav(...) + AudioSegment.from_wav(...)` (the exact primitive `mixer.concat_parts` uses for the mix at `music_gen.py`'s legacy code, preserved in `mixer.py:concat_parts`). New writer helper: `_concat_layer_stems(stem_paths_per_part: Dict[str, Dict[str, str]], layer: str, out_path: str) -> str`. Produces one WAV per layer at song scope. **Absent layers are already silent stems** (Phase 4 D-12) — they concatenate into silent song-scope stems of the correct total duration.
- **D-07:** **MIDI concatenation across parts.** Per-layer `.mid` files from generators are per-part (`<name>-<part>/*.mid`). Writer flattens to one `midi/{layer}.mid` per sample using `mido.MidiFile` — merge tracks across parts with cumulative tick offsets. Per-layer implementation: load each part's MIDI, rewrite `msg.time` to be tick-cumulative across the arrangement, emit one output MIDI with identical PPQ + tempo metadata as the first part's file. If parts have conflicting tempo (they shouldn't — `create_song` passes a single tempo through), raise `ValueError` — this is a sampler regression, not a writer concern.
- **D-08:** **Silent-layer MIDI**. If a layer is masked out for a given part (Phase 4 D-13 `layer_mask[part][layer] == False`), the writer still outputs a same-duration silent MIDI segment for that part within the per-sample `midi/{layer}.mid`. This keeps the **"one MIDI per layer, length matches mix"** invariant. Implementation: a zero-velocity `note_on` at tick 0 + `note_off` at tick=part_duration_ticks. Test fixture validates that summing note events in the silent MIDI == 1 `note_on`+`note_off` pair per masked part.
- **D-09:** **`dataset_root` is a new `Config` field** (not a separate module constant): `dataset_root: str = os.path.join(project_root, "dataset")`. Threaded through `api.py:generate(config)` → `writer.write_sample(config.dataset_root, ...)`. `config.Config` gets this field plus `global_seed: Optional[int]`, `sample_index: int = 0`, and `split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1)`. Env-var override `MUSICGEN_DATASET_ROOT` is supported (Phase 2 config precedence pattern — D-25 Phase 4 continues).
- **D-10:** **Working-dir cleanup.** After the sentinel `sample.json` is written, the writer deletes the per-sample working directory created by Phase 4's orchestrator (where intermediate part-level stems live). Failure-tolerant: `shutil.rmtree(working_dir, ignore_errors=True)`. Rationale: avoids doubling disk usage across a 10k dataset run. `config.keep_working_dirs: bool = False` for debugging (Phase 6 CLI surfaces this as `--keep-working`).
- **D-11:** **Relative paths in `sample.json`.** Annotator's Phase 4 output embeds absolute working-dir paths (`mix`, `stems.*`, `midi.*`). Phase 5 writer **rewrites these fields** to dataset-root-relative paths before `json.dump`: `mix = "mix.wav"`, `stems = {"beat": "stems/beat.wav", ...}`, `midi = {"beat": "midi/beat.mid", ...}` — per-sample-dir-relative, not dataset-root-relative. Rationale: a per-sample consumer (ML training loader) opens `<sample_dir>/sample.json`, reads `sample["mix"]`, and joins against `<sample_dir>` — the natural idiom.
- **D-12:** **Path rewrite is in the writer, not the annotator.** The annotator stays a **pure function** (Phase 4 D-14) — it does not know about the dataset root. Writer receives the annotation dict, deep-copies it (to avoid mutating the caller's view), rewrites the path fields, then serializes. This keeps the annotator unit-testable without filesystem (Phase 4 testing contract preserved).

### Manifest design (R-P5)

- **D-13:** **`manifest.jsonl` schema** — one JSON object per line, UTF-8, LF line ending:
  ```json
  {"sample_index": 0, "seed": 42, "sample_seed": 12345678901234567, "status": "ok", "split": "train", "path": "000000/sample.json", "musicality_score": 0.87, "duration_seconds": 47.3, "wrote_at": "2026-04-19T14:00:00Z"}
  ```
  `path` is dataset-root-relative (always `<index:06d>/sample.json`). `wrote_at` is ISO-8601 UTC. `status` is `"ok"` or `"failed"`. On failure: `sample.json` does not exist (D-04 sentinel invariant), but the manifest entry DOES appear with `status: "failed"` and a nullable `error` string (traceback repr capped at 2KB). This is critical for the Phase 6 resume path — the manifest tells you which indices were attempted vs. untouched.
- **D-14:** **Lock abstraction.** `ManifestWriter(dataset_root: str, lock: Optional[ContextManager] = None)`. Default `lock = threading.Lock()` (single-process Phase 5 correctness). Phase 6 passes `multiprocessing.Manager().Lock()` from the `ProcessPoolExecutor`'s manager. Rationale: Phase 5 is single-process; we do NOT pay the cross-process `Manager` overhead now. But the API contract is right from day one — Phase 6 adds one parameter, not a refactor.
- **D-15:** **Append-only, never rewrite.** `ManifestWriter.append(entry)` opens `manifest.jsonl` in `"a"` mode, under the lock, writes `json.dumps(entry, sort_keys=True) + "\n"`, flushes and `os.fsync(fileno)`. Never mutates prior lines. **Resumption guarantee:** re-running `generate()` for `sample_index=N` where `<dataset_root>/<N>/sample.json` already exists is a no-op (`ManifestWriter.is_sample_complete` returns True → `api.generate` returns early with the existing `SampleResult` reconstructed from the on-disk files). Re-running for a failed index (no `sample.json`, manifest has `failed` entry) **appends a new** `status: "ok"` line on success; downstream readers take the last-status-wins semantics. Simpler than compaction; acceptable at 10k scale (~1 MB manifest).
- **D-16:** **No manifest file locking on read.** `ManifestWriter.is_sample_complete` checks **only** `<dataset_root>/<index:06d>/sample.json` existence — NOT the manifest contents. The sentinel is the source of truth; the manifest is a projection. This sidesteps the whole "what if the manifest is mid-write when we read it" concern — the writer never reads the manifest for correctness.

### Seed discipline (R-P7)

- **D-17:** **`derive_sample_seed` verbatim per `.planning/research/ARCHITECTURE.md` §"Seed / RNG propagation"**:
  ```python
  def derive_sample_seed(global_seed: int, sample_index: int) -> int:
      raw = hashlib.sha256(f"{global_seed}:{sample_index}".encode()).digest()
      return int.from_bytes(raw[:8], "big")
  ```
  Using `global_seed` as an `int` (not `str`) — the f-string renders it consistently. 8 bytes → 64-bit unsigned — fits comfortably in a Python `int` and in `random.Random.seed` without loss.
- **D-18:** **`make_rngs` verbatim per ARCHITECTURE.md**:
  ```python
  def make_rngs(sample_seed: int) -> Dict[str, random.Random]:
      return {
          "params":     random.Random(sample_seed ^ 0x01),
          "generators": random.Random(sample_seed ^ 0x02),
          "soundfonts": random.Random(sample_seed ^ 0x03),
          "fx":         random.Random(sample_seed ^ 0x04),
          "mix":        random.Random(sample_seed ^ 0x05),
      }
  ```
  Domain names match the five call-site groups (D-19). XOR with small bit constants is equivalent to distinct seeds — cheap and deterministic.
- **D-19:** **RNG threading map** — `music_gen.py`'s single `_rng` is replaced by `rngs` in `api.py:generate()`. Call sites:
  - `rngs["params"]` → `generate_random_key`, `generate_random_tempo`, `generate_random_time_signature`, `generate_random_swing`, `generate_song_measures`, `generate_song_arrangement` (all in `src/musicgen/sampler.py`)
  - `rngs["generators"]` → `generate_chord_progression`, `generate_melody`, `generate_bassline`, `generate_beat` (all in `src/musicgen/generators/*`)
  - `rngs["soundfonts"]` → `renderer.pick_soundfonts`
  - `rngs["fx"]` → `mixer.build_fx_boards`
  - `rngs["mix"]` → `mixer.compute_layer_mask`
  
  No function signatures change — they all already accept `rng: random.Random` (Phase 3 D-07, Phase 4 D-17). `api.py:generate` picks the right RNG per call. **RNG-consumption order is preserved** from Phase 4's `music_gen.py:create_song` so the Phase 4 golden-seed smoke test (`tests/test_integration_full_generation.py::test_midi_reproducibility_same_seed`) stays green.
- **D-20:** **`save_random_state()` context manager** — defined in `seeds.py`:
  ```python
  @contextlib.contextmanager
  def save_random_state():
      state = random.getstate()
      try:
          yield
      finally:
          random.setstate(state)
  ```
  **Applied at exactly one call site**: `musicgen.musicality.get_musicality_score`, which imports `music21` via `librosa`/`scipy`/none-of-the-above (transitive chain unclear). Phase 3's `tests/test_music21_isolation.py` already proved `music21` does NOT mutate global `random` today; the context manager is **defense in depth** — a cheap prophylactic so a future `music21`/`librosa` release that quietly starts calling `random.seed(...)` never poisons sample N+1. AST guard (D-32) does NOT catch dep mutations, so this layer is load-bearing.
- **D-21:** **Global seed policy.** `config.global_seed: Optional[int] = None`. If `None` when `generate(config)` is called: raise `ValueError("global_seed is required for deterministic generation; pass config.global_seed explicitly")`. Rationale: the library's core value prop is reproducibility; silently generating from `time.time_ns()` makes the seed invisible to the annotation and breaks the determinism contract. Phase 6's CLI (`musicgen generate --seed S`) requires the flag; there is no "random mode". Users who want random seeds pass `--seed $(python -c 'import secrets; print(secrets.randbits(32))')` — one shell-line, explicit, auditable.

### Annotator schema completion (R-P4)

- **D-22:** **Phase 5 fills the TBD fields** in the annotation dict. Annotator signature is already kwargs-accepting (Phase 4 D-15). `api.py:generate` passes:
  - `seed = sample_seed` (NOT `global_seed` — the sample-level seed is what uniquely identifies this sample's RNG basin)
  - `musicgen_version = importlib.metadata.version("musicgen")` (resolves `0.1.0` per pyproject; falls back to the literal `"0.1.0+uninstalled"` string if the `PackageNotFoundError` fires — happens in tests that run without `pip install -e .`; CI install path always resolves)
  - `split = assign_split(sample_seed, config.split_ratios)` (D-24)
  - `pre_roll_offset_seconds = None` **still** this phase — R-P9 calibration is Phase 6; Phase 5 writes `None` unambiguously, matching Phase 4 D-16's contract and avoiding a phantom-zero-offset lie. The annotator continues to emit `pre_roll_offset_seconds: None` until Phase 6 threads it.
- **D-23:** **Schema ordering in `sample.json`** — `json.dump(annotation, f, sort_keys=True, indent=2, separators=(",", ": "))`. `sort_keys=True` is the determinism lever: byte-identical `sample.json` across runs requires stable key ordering. `indent=2` for human readability; `separators` is the default except we pin it explicitly so a future `json` default shift can't surprise us. The regression test (D-26) hashes this canonicalized text.

### Sum-of-stems assertion (R-P2)

- **D-24:** **Implementation location: inside `writer.write_sample`**, **after** stems and mix are written to final paths and **before** the sentinel `sample.json` rename. Helper: `_assert_sum_of_stems(mix_path: str, stem_paths: Dict[str, str], epsilon: float) -> Tuple[bool, float]` returns `(passed, max_abs_diff)`. Implementation: read each WAV into `numpy` (via `scipy.io.wavfile.read` — already a transitive dep of `librosa`, confirmed present), convert int16 → float32 in `[-1, 1)` (`samples.astype(np.float32) / 32768.0`), sum the four stem arrays element-wise, subtract the mix array, take `np.max(np.abs(diff))`. Fails → `log.error` + `ManifestWriter.append(status="failed", error=f"sum_of_stems_exceeded: {diff:.6f}")` + return SampleResult with `status="failed"` + DO NOT write sample.json. Caller (`api.generate`) returns the failed result to the user.
- **D-25:** **Epsilon = `1e-3`** (peak sample absolute difference, normalized float32). Derivation: pedalboard + pydub are known to introduce ~−100 dBFS numerical drift through the resample/overlay path; `1e-3 ≈ −60 dBFS`, ~6 orders of magnitude above numerical noise, ~3 orders of magnitude below any audible signal. Tight enough to catch real bugs (wrong stem wired to wrong layer, silence where there shouldn't be, a skipped FX application), loose enough to survive the known int16 quantization loop. Configurable via `config.sum_of_stems_epsilon: float = 1e-3`; the Phase 5 regression test (D-26) uses the default and is expected to pass.

### Train/valid/test split (R-P6)

- **D-26:** **Stable split hash.**
  ```python
  def assign_split(sample_seed: int, ratios: Tuple[float, float, float]) -> str:
      bucket = int.from_bytes(
          hashlib.sha256(f"split:{sample_seed}".encode()).digest()[:4], "big"
      ) % 10000 / 100.0  # float in [0, 100)
      train_cutoff = ratios[0] * 100
      valid_cutoff = (ratios[0] + ratios[1]) * 100
      if bucket < train_cutoff:     return "train"
      if bucket < valid_cutoff:     return "valid"
      return "test"
  ```
  Inputs: `sample_seed` (NOT `global_seed` + `sample_index`; `sample_seed` is already a deterministic projection of both per D-17). Prefix `"split:"` disambiguates this hash from the seed-derivation hash so the RNG basins and split bucket are statistically independent.
- **D-27:** **Default ratios `(0.8, 0.1, 0.1)`** via `config.split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1)`. Validator: must sum to `1.0 ± 1e-9`; must all be `>= 0`. Raise `ValueError` on invalid at `Config.__post_init__`. Split names are fixed (`"train"`, `"valid"`, `"test"`); custom labels are v0.2+.

### Determinism regression test (R-P8, R-Q3)

- **D-28:** **Golden fixture file layout** — `tests/fixtures/determinism/`:
  - `expected_mix.sha256` — single line with the SHA-256 of `mix.wav` for `global_seed=1, sample_index=0, default-config`
  - `expected_midi_beat.sha256`, `expected_midi_melody.sha256`, `expected_midi_harmony.sha256`, `expected_midi_bassline.sha256` — one per layer
  - `expected_sample.sha256` — SHA-256 of the canonical `sample.json` bytes (per D-23 ordering)
  - `fluidsynth_version.txt` — first line of `fluidsynth --version` when the golden was captured; tests skip (xfail with clear message) on mismatch per R-P8's "cross-binary WAV identity is NOT guaranteed"
- **D-29:** **`tests/test_determinism_golden.py`** — marked `@pytest.mark.slow` (parity with Phase 4 D-30 integration test). Skips if `fluidsynth` not on PATH OR if `FLUIDSYNTH_VERSION != fixtures/determinism/fluidsynth_version.txt` (xfail with a message pointing at how to re-generate goldens: `python -m pytest -m slow --regen-goldens` — flag added per D-32). Runs: `generate(Config(global_seed=1, sample_index=0, dataset_root=tmp_path))`; asserts each of the 6 SHA-256s. MIDI + `sample.json` hashes MUST match unconditionally (R-P8: "bit-identical MIDI, bit-identical sample.json"); WAV hash is guarded by the FluidSynth version match.
- **D-30:** **`sample.json` hash-stability cross-check.** A second non-slow test runs `generate` twice in one process and asserts `sha256(sample.json_run_1) == sha256(sample.json_run_2)` — catches any non-determinism in our own code (e.g., a stray `datetime.now()` sneaking into annotation) without depending on FluidSynth at all. Critical: the `wrote_at` field in `manifest.jsonl` IS wall-clock (and therefore nondeterministic) — the test hashes `sample.json`, not the manifest.

### Library API (R-P12 partial — single-sample only this phase)

- **D-31:** **`api.py:generate(config: Config) -> SampleResult`** — the one function, single-sample. Responsibilities:
  1. Validate `config.global_seed is not None` (D-21), `config.sample_index >= 0`, `config.split_ratios` sums to 1.0.
  2. `sample_seed = derive_sample_seed(config.global_seed, config.sample_index)`; `rngs = make_rngs(sample_seed)`.
  3. Resume short-circuit: `if ManifestWriter.is_sample_complete(config.dataset_root, config.sample_index): return _reconstruct_sample_result(...)` (read `sample.json` + infer paths). Idempotent — re-running the same call does zero work.
  4. Build `working_dir = tmp_dir` (use `tempfile.mkdtemp(prefix="musicgen-")` rather than a deterministic-named working dir — two concurrent calls with the same index would collide otherwise; Phase 6 ProcessPool inherits this).
  5. Call the Phase 4 pipeline directly (sampler → generators → renderer → mixer → beats → annotator), replacing the Phase-3-module-level `_rng` with `rngs`. Wrap musicality scoring in `with save_random_state():` (D-20).
  6. `split = assign_split(sample_seed, config.split_ratios)`.
  7. `writer.write_sample(config.dataset_root, config.sample_index, annotation, mix_working_path, stems_working_paths, midi_working_paths, seed=sample_seed, musicgen_version=<resolved>, split=split, config=config)`.
  8. `manifest_writer.append(<dict>)`.
  9. `shutil.rmtree(working_dir)` (unless `config.keep_working_dirs`).
  10. Return `SampleResult`.
- **D-32:** **`--regen-goldens` pytest flag** — registered in `tests/conftest.py` via `pytest_addoption`. When present, `test_determinism_golden.py` writes its computed SHA-256s back to `tests/fixtures/determinism/*.sha256` + updates `fluidsynth_version.txt`, then always passes. Used by the maintainer when intentionally changing the RNG order or FluidSynth version. Without the flag, the test asserts equality.

### Orchestrator migration (R-P7 closure)

- **D-33:** **`music_gen.py` stays at repo root as a thin smoke-test wrapper.** Not deleted this phase. It becomes a ~40-line script: construct a `Config(global_seed=<fallback>, sample_index=<from argv>)`, call `musicgen.generate(config)`, print the result. Existing `python music_gen.py` workflow keeps working (Phase 3 D-24 / Phase 4 D-24 invariant). Phase 6 replaces it with the `musicgen generate` CLI; the file is deleted at that point, not now.
- **D-34:** **`music_gen.py`'s `create_song` + `generate_song_parts` + `generate_song` are DELETED** — their bodies moved into `api.py:generate()` (`create_song` ≈ steps 4-7 of D-31; `generate_song_parts` is extracted as a private helper `_generate_all_midi(rngs, params, cfg) -> Tuple[Dict, Dict]`; `generate_song` is `api.generate` itself). The Phase 4 delegations in `music_gen.py` (e.g. `verify_pattern_for_time_signature`) stay for back-compat ONE more phase (code that `from music_gen import verify_pattern_for_time_signature` — there is none in `src/` per grep, but the Phase 2 logging tests may touch them); deletion is Phase 6 cleanup.
- **D-35:** **Package-level `__init__.py` exports** — `src/musicgen/__init__.py` becomes:
  ```python
  from musicgen.api import generate, Config, SampleResult
  __version__ = importlib.metadata.version("musicgen")
  __all__ = ["generate", "Config", "SampleResult", "__version__"]
  ```
  Phase 3's empty `__init__.py` is replaced; consumers can now `from musicgen import generate`.

### Testing strategy

- **D-36:** **`tests/test_seeds.py`** — pure-function tests. `test_derive_sample_seed_deterministic` (same inputs → same output; two known vectors). `test_derive_sample_seed_different_indices` (N=100 indices for one global_seed → all distinct, no collisions). `test_make_rngs_five_domains` (output dict has exactly the five domain keys). `test_make_rngs_domain_independence` (draw 1000 floats from each domain; correlation coefficient across all pairs < 0.05). `test_save_random_state_restores` (mutate `random` inside, global state intact after). Zero I/O, zero FluidSynth. ~10 assertions.
- **D-37:** **`tests/test_writer.py`** — uses `tmp_path`. Synthesize known WAVs + MIDIs (one per layer per part, using `numpy` + `scipy.io.wavfile.write` + `midiutil.MIDIFile`) → call `write_sample` → assert: directory layout correct (000000/mix.wav, 000000/stems/*.wav, 000000/midi/*.mid, 000000/sample.json), `sample.json` paths are relative (D-11 contract), silent-stem concatenation produces silence (zero RMS), sum-of-stems assertion passes when stems are constructed to sum to mix + fails when they are not (fault-injection test). Sample.json existence implies all other files present (sentinel invariant). ~15 assertions, zero FluidSynth.
- **D-38:** **`tests/test_manifest.py`** — `tmp_path`. `test_manifest_append_single` (one entry → one line). `test_manifest_append_concurrent` (10 threads each append 100 entries → 1000 lines, all well-formed JSON, no corruption — the lock works). `test_is_sample_complete_true_iff_sentinel` (write `sample.json` → returns True; remove → returns False; manifest state irrelevant per D-16). ~8 assertions.
- **D-39:** **`tests/test_split.py`** — 10k seed samples, assert empirical ratios within 2% of declared ratios (80/10/10 → 7840-8160 train, 800-1200 each for valid/test). `test_split_deterministic` (same seed → same split). `test_invalid_ratios` (sum != 1.0 → ValueError at Config init). ~6 assertions.
- **D-40:** **`tests/test_api.py`** — uses `tmp_path` + `@pytest.mark.slow` guard on the FluidSynth subpath; splits into fast and slow cases. Fast: `test_config_global_seed_required` (missing seed → ValueError). `test_generate_resume_short_circuits` (pre-create `sample.json` → `generate` returns without running the pipeline; counter on mocked sampler stays at 0). Slow: `test_generate_produces_layout` (real pipeline, assert the 10 expected files + valid JSON). `test_generate_twice_idempotent` (run, capture `SampleResult`, run again, identical result).
- **D-41:** **`tests/test_determinism_golden.py`** — per D-29. Separately, a non-slow `test_generate_sample_json_stable_same_process` (D-30 cross-check) runs `generate` twice and hashes `sample.json` bytes.
- **D-42:** **AST guard update** — `tests/test_no_bare_random_in_package.py` (Phase 4 D-31) stays as-is; its glob `src/musicgen/**/*.py` now picks up `seeds.py`, `writer.py`, `manifest.py`, `api.py`, `musicality.py` automatically. No test file change needed. Manual grep-spot-check post-phase: `grep -rE "\brandom\.(choice|random|randint|choices|uniform)\(" src/musicgen/ | grep -v "_rng" | grep -v "rng\."` → must return zero rows.

### Phase 6 boundary (what this phase does NOT touch)

- **D-43:** **Explicitly out of scope** — `src/musicgen/batch.py` (R-P10), `src/musicgen/calibrate.py` (R-P9), `typer`-based full CLI in `src/musicgen/cli.py` (R-P13), `--output-mode` flag (R-P14), structured JSON progress logs during batch runs (R-P15), aggregated failure-count reporting (R-P16), and the `musicgen clean --failed` subcommand (R-P13). **Infrastructure hooks land now** so Phase 6 extends rather than refactors: `ManifestWriter` accepts a `lock` parameter (D-14) so `multiprocessing.Manager().Lock()` slots in, `Config` has `workers: Optional[int] = None` reserved but unused, `generate()` is the one-sample primitive `generate_batch` will wrap with `ProcessPoolExecutor`. Resume logic beyond `is_sample_complete` (the sentinel check) is Phase 6.

### Claude's Discretion

- Exact field names inside `SampleResult` (`sample_dir` vs. `sample_path` vs. `path`). Shape is fixed (D-02); names are aesthetic.
- Whether `_concat_layer_stems` uses `pydub.AudioSegment.from_wav(...) + ...` (matches `mixer.concat_parts` style) or `scipy.io.wavfile` array concatenation (possibly faster, same byte output for int16). Match `mixer.concat_parts` style unless a later benchmark reveals a bottleneck.
- Exact name of the new config field (`dataset_root` vs. `output_dir`). Chosen: `dataset_root` — matches REQUIREMENTS/ARCHITECTURE terminology.
- Whether `musicgen_version` falls back to `"0.1.0+uninstalled"` literal (D-22) or raises. Falling back keeps test runs green without `pip install -e .`; raising would surface missing installs faster. Falling back is the pragmatic pick for v0.1.
- Whether the working directory is under `<tempfile>/musicgen-<uuid>/` (default, matches `tempfile.mkdtemp`) or under `<dataset_root>/.tmp/<index>/`. Either works; `tempfile` keeps `dataset_root` clean of half-finished samples during interruption.
- Silent MIDI representation (D-08): zero-velocity `note_on`+`note_off` at tick 0+duration, OR an empty track with a final `meta end_of_track` message at `duration_ticks`. Both produce a layout-valid `.mid` file; first is what the existing generators emit for a zero-note beat, so use it for consistency.
- Whether `sample.json`'s `split` field is lowercased (`"train"`) or title-cased (`"Train"`). Lowercase — matches the typical ML tooling idiom (PyTorch `Dataset` splits).
- Whether `assign_split` modulos by `10000` (D-26: 0.01% split precision) or `1000` (0.1% precision). 10000 gives four decimal places of resolution, enough for 0.01% splits like 99.98/0.01/0.01 that nobody is asking for but are cheap to support.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase scope and requirements

- `.planning/ROADMAP.md` — Phase 5 section (deliverables, exit criteria, R-P1..P-8 + R-Q3 coverage). **Depends on Phase 3 + Phase 4** — both must be merged before this phase starts. Phase 6 depends on this phase for the `generate()` primitive and manifest abstraction.
- `.planning/REQUIREMENTS.md` — R-P1 (output layout), R-P2 (stem persistence + sum-of-stems), R-P3 (MIDI persistence), R-P4 (sample.json schema), R-P5 (manifest.jsonl), R-P6 (deterministic split), R-P7 (seed discipline — `derive_sample_seed` / `make_rngs`), R-P8 (determinism contract — bit-identical output under pinned FluidSynth), R-P12 **first bullet only** (`generate(config)` — `generate_batch` is Phase 6), R-Q3 (regression test in CI).
- `.planning/PROJECT.md` — Core value ("every sample reproducible, fully-labeled"), determinism non-negotiable, 1k–10k scale target, "Productize priority 2" checklist items #3 (deterministic generation), #7 (per-sample output directory + UUID truncation fix), #8 (rich annotations), #9 (stems + MIDI persistence), #10 (manifest).

### Architecture and design input

- `.planning/research/ARCHITECTURE.md` §"Per-sample output layout" (R-P1 layout diagram — writer follows verbatim), §"Seed / RNG propagation" (D-17/D-18 copy the `derive_sample_seed` + `make_rngs` code verbatim), §"Build order" #6-7 (this phase).
- `.planning/research/PITFALLS.md` — P-1 (FluidSynth bit-reproducibility — pinned-binary gate on the golden WAV hash, D-28/D-29), P-2 (stems not persisted / sum ≠ mix — closed by D-24/D-25), P-4 (RNG leakage in multiprocessing — closed for single-process by D-17/D-18/D-19; Phase 6 extends to workers), P-7 (per-sample directory + UUID truncation — closed by D-04/D-05).
- `.planning/research/STACK.md` §"Seed / RNG" (stdlib only — no new deps), §"Manifest + annotations" (stdlib json, JSONL append-under-lock), §"FluidSynth" (pin the binary — D-28 version-file).
- `.planning/research/SUMMARY.md` — build order confirmation.

### Codebase context

- `.planning/codebase/ARCHITECTURE.md` §"Pipeline stages" — the post-Phase-4 pipeline shape (sampler → generators → renderer → mixer → annotator); Phase 5 adds writer + manifest at the tail.
- `.planning/codebase/STRUCTURE.md` — current file layout (src/musicgen/ has sampler, generators/, renderer, mixer, annotator, beats, duration_validator, cli stub). Phase 5 adds seeds, writer, manifest, api, musicality.
- `.planning/codebase/CONVENTIONS.md` — naming, logging (`logger = logging.getLogger(__name__)`), typing, import organization, `@dataclass(frozen=True, slots=True)` pattern. Phase 5 modules match.
- `.planning/codebase/CONCERNS.md` — #7 (UUID truncation — closed by D-04/D-05), #1/#3 (god-file decomposition — Phase 5 finishes by deleting `create_song`/`generate_song_parts`/`generate_song` per D-34).
- `.planning/codebase/TESTING.md` — pytest layout, `@pytest.mark.slow` marker (already declared in `pyproject.toml`, used by Phase 4 integration test — Phase 5 D-29/D-40/D-41 extend).
- `.planning/codebase/INTEGRATIONS.md` — FluidSynth binary integration (version pinning gate for D-28/D-29).

### Prior phase artifacts (decisions to carry forward)

- `.planning/phases/01-stabilize-i-bug-fixes-and-guardrails/01-02-*-SUMMARY.md` — R-S4 fix (apply_gain + pan capture). Mixer's `mix_part` preserves this; sum-of-stems assertion only passes if R-S4 is preserved correctly (regression guard).
- `.planning/phases/02-stabilize-ii-config-time-signature-registry-logging/02-CONTEXT.md` D-01/D-02/D-03 — config precedence (CLI > env > defaults). D-09 `dataset_root` follows this.
- `.planning/phases/03-package-skeleton-sampler-generators-extraction/03-CONTEXT.md` D-07/D-08 (single module-level `_rng` — replaced by `rngs` dict per D-18/D-19), D-11 (musicality_score.py move deferred — **closed this phase** by D-03), D-13 (Python ≥ 3.10, hatchling), D-22 (per-field dataclasses), D-24 (music21 global RNG audit PASSED — D-20 is defense in depth anyway).
- `.planning/phases/04-renderer-mixer-annotator-beats-extraction/04-CONTEXT.md` — **all decisions consulted**, most relevant: D-04 (musicality_score.py stays at root — deferral **closed** by D-03 this phase), D-12 (silent-stem fallback — writer concatenates these per-part silents into song-level silents per D-06), D-14/D-15/D-16 (annotator signature + Phase 5 TBD fields — filled by D-22 this phase), D-17/D-18 (RNG threading — Phase 4 installed single-`_rng`; Phase 5 swaps for the named hierarchy per D-18/D-19), D-23 (mix_and_save deleted — orchestrator shell stays until D-34 this phase), D-31 (AST bare-random guard — Phase 5 D-42 confirms it covers new files), D-33 (serial Phase 3→4; Phase 5 runs serially after Phase 4).
- `.planning/phases/04-renderer-mixer-annotator-beats-extraction/04-06-SUMMARY.md` — E2E integration test pattern (`@pytest.mark.slow` + skip-if-no-fluidsynth + seeded-RNG reproducibility). D-29/D-41 reuse the skip pattern.

### Source files to modify / extract from

- `music_gen.py` — extraction source for Phase 5:
  - `create_song` (52-141): **deleted** per D-34; body migrates into `api.generate()`.
  - `generate_song_parts` (143-168): extracted to `api._generate_all_midi` private helper per D-34.
  - `generate_song` (170-189): **deleted** per D-34. The `uuid`-based `song_name[:20]` at line 184 is the truncation bug fixed by D-04's `<index:06d>` naming.
  - `if __name__ == "__main__":` block (192-199): **stays** as smoke-test wrapper per D-33 — rewrites to call `musicgen.generate(Config(global_seed=1, sample_index=0))`.
  - Module-level delegations (20-48): stay for one more phase (D-34).
- `src/musicgen/__init__.py` — rewrite per D-35.
- `src/musicgen/annotator.py` — unchanged signature; new call sites pass the previously-None kwargs (`seed`, `musicgen_version`, `split`) per D-22.
- `src/musicgen/mixer.py` — unchanged. Per-part stems and mix stay in the working dir; writer reads them out (D-06).
- `src/musicgen/renderer.py` — unchanged.
- `src/musicgen/beats.py` — unchanged (pure derivation).
- `musicality_score.py` — **moves** to `src/musicgen/musicality.py` per D-03. Import references in `music_gen.py` (line 3) and any test fixture updated; back-compat re-export from `musicality_score.py` (shim that imports from `src.musicgen.musicality`) kept until Phase 6 CLI replaces the smoke-test entry point.
- `config.py` — **adds** fields per D-09: `dataset_root: str`, `global_seed: Optional[int]`, `sample_index: int`, `split_ratios: Tuple[float, float, float]`, `sum_of_stems_epsilon: float`, `keep_working_dirs: bool`, `workers: Optional[int]` (reserved, Phase 6). Env-var overrides follow Phase 2 D-01 pattern.
- `pyproject.toml` — **no new runtime deps** (stdlib-only for seeds, writer, manifest, splits, determinism — `hashlib`, `threading`, `multiprocessing`, `pathlib`, `importlib.metadata`, `contextlib`, `shutil`, `tempfile`, `json`). `scipy.io.wavfile` is already transitive through `librosa`. `mido` for MIDI concatenation is already a direct dep (Phase 4 R-X7 added it). Dev deps unchanged.
- `tests/` — new files per D-36..D-41: `test_seeds.py`, `test_writer.py`, `test_manifest.py`, `test_split.py`, `test_api.py`, `test_determinism_golden.py`. Plus `tests/fixtures/determinism/` directory with golden SHA-256 files per D-28.
- `tests/conftest.py` — adds `--regen-goldens` pytest flag per D-32. First conftest.py in the project (Phase 3's conftest was deleted per Phase 3 05-SUMMARY); this one is new, minimal, only for the regeneration flag.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets

- `src/musicgen/annotator.py:annotate` — Phase 4 D-14/D-15/D-16. Accepts `seed`, `musicgen_version`, `split`, `analysis_failed` kwargs that are `None` today; Phase 5 passes real values (D-22). No signature change needed.
- `src/musicgen/mixer.py:concat_parts` (line 418) — `AudioSegment.from_wav + AudioSegment.from_wav` pattern. Writer's `_concat_layer_stems` (D-06) mirrors this idiom per-layer instead of the full mix.
- `src/musicgen/mixer.py:MixResult.stem_paths` (line 291) — per-part per-layer post-FX WAV paths. Writer reads these to build per-layer song-level stems.
- `src/musicgen/mixer.py:_make_silent_stem` (line 73) — stereo 44.1kHz silent AudioSegment. Writer doesn't call this directly (Phase 4 already emits silent per-part stems when masked); mentioned because the sum-of-stems invariant depends on silent-stem stereo parity that D-06 preserves across concatenation.
- `musicality_score.get_musicality_score(wav_path)` — returns `(score: float, components: Dict[str, float])`. Wraps `librosa.load` + five feature extractors. Phase 3 D-24 confirmed music21 / librosa do NOT touch global `random` today, BUT `save_random_state()` (D-20) wraps defensively.
- `config.Config` (dataclass, Phase 2) — three-layer precedence machinery already in place. Phase 5 D-09 adds fields without changing the precedence pattern.
- `src/musicgen/sampler.SongParams` (dataclass, Phase 3 D-22) — frozen. `api.generate` constructs this and passes downstream.
- `tests/test_integration_full_generation.py` — Phase 4 D-30's `@pytest.mark.slow` skip-if-no-fluidsynth pattern. D-29/D-40 reuse the skip guard verbatim.

### Established Patterns

- **RNG injection** (Phase 3 D-07 / Phase 4 D-17): every function accepting randomness takes `rng: random.Random` as a named parameter — never hits `random.*` directly. Phase 5 swaps `_rng` (single module-level Random) for a 5-key `rngs` dict (D-18/D-19) without touching ANY downstream function signature. This is the load-bearing invariant that makes this phase low-risk.
- **AST static guard** (Phase 3 D-25 / Phase 4 D-31): `tests/test_no_bare_random_in_package.py` walks `src/musicgen/**/*.py` for any `random.<method>` call outside test scope. Phase 5 D-42 confirms the existing guard covers new modules (glob is parametrized).
- **Config three-layer precedence** (Phase 2 D-01/D-02): CLI args > env vars > dataclass defaults. D-09 extends Config fields keeping this pattern intact (`MUSICGEN_DATASET_ROOT` env var follows `MUSICGEN_SF_DIR` precedent).
- **Logging** (Phase 2 D-07): `logger = logging.getLogger(__name__)`; level semantics DEBUG/INFO/WARNING/ERROR. New modules (seeds, writer, manifest, api) match.
- **Frozen dataclasses + `slots=True`** (Phase 3 D-13 Python ≥ 3.10): `SampleResult` (D-02), updated `Config` (D-09) follow.
- **Working-dir layout** (Phase 4 D-09 + orchestrator): per-sample `<name>/<name>-<part>/stems/<layer>.wav` + `<name>/<name>-<part>/*.mid` + `<name>/<name>.wav`. Writer reads from this working structure and lands at the final layout (D-04).
- **pytest markers** (pyproject.toml): `slow` and `integration` declared. D-29/D-40 use `slow` (FluidSynth-bound).

### Integration Points

- **`music_gen.py` → `api.py` migration (D-34):** `create_song` + `generate_song_parts` + `generate_song` bodies relocate; the smoke-test `if __name__ == "__main__":` block rewrites to `musicgen.generate(Config(global_seed=1, sample_index=0))` (D-33). This is the first phase where `music_gen.py` is no longer the "definition-of-the-pipeline" file — it's a wrapper.
- **`annotator.annotate(...)` → `writer.write_sample(...)`**: writer calls annotator (indirectly, via `api.py`'s orchestration), receives the dict, path-rewrites per D-11/D-12, then serializes. Annotator remains pure.
- **`mixer.mix_part(...)` → `writer._concat_layer_stems(...)`**: writer consumes `MixResult.stem_paths` from each part's MixResult and concatenates per-layer across the arrangement (D-06). No mixer API change.
- **Phase 4 golden integration test (`tests/test_integration_full_generation.py::test_midi_reproducibility_same_seed`)** — this Phase 4 test asserts `music_gen.create_song(...)` with the same seed yields identical MIDI. Phase 5's D-34 deletes `create_song`. The test is either (a) updated to call `musicgen.generate(Config(global_seed=..., sample_index=0))` or (b) marked deprecated and replaced by D-29's golden test. Recommended: migrate the test (a) — simpler, preserves Phase 4 intent. D-29's golden is additive, not replacing.
- **`multiprocessing.Manager().Lock()` (R-P5 literal):** D-14's `lock` parameter accepts any `ContextManager` — `threading.Lock()` for Phase 5, `multiprocessing.Manager().Lock()` for Phase 6. The difference is invisible to `ManifestWriter.append`.
- **`importlib.metadata.version("musicgen")` (D-22, D-35):** resolves from `pyproject.toml` after `pip install -e .`. If the package isn't installed (bare `PYTHONPATH` + repo root), `PackageNotFoundError`; D-22 catches and substitutes `"0.1.0+uninstalled"`.
- **`tempfile.mkdtemp(prefix="musicgen-")` (D-31 step 4):** working dirs land in `/tmp/musicgen-XXX/`. Survives a generate call. Phase 6's ProcessPool inherits this and each worker gets its own mkdtemp call.

</code_context>

<specifics>
## Specific Ideas

- **Index-based naming is non-negotiable**: the whole Phase 5 spine depends on `000000/` being predictable from `(global_seed, sample_index)` alone. No timestamps in paths, no UUIDs, no hash-based directory names. Phase 4 D-09's working-dir `<name>-<part>/` can stay UUID-flavored (it's temp); dataset layout is clean indices.
- **Bit-identical `sample.json` bytes across runs** is the cheap-and-critical determinism test (D-30). If this ever fails we don't need FluidSynth to debug — the bug is in our code. Prioritize this test's stability.
- **"The manifest is a projection, not the truth"** (D-16). `sample.json` is the sentinel; manifest exists to speed up batch-scale queries. Never reverse the dependency.
- **Defense in depth for music21** (D-20): Phase 3 proved the global-random leak isn't there *today*, but wrapping is cheap insurance for dep upgrades. Mentioned by the user explicitly in REQUIREMENTS R-P7 bullet 5: "if music21 or another dep mutates it, wrap affected calls in a save/restore context".
- **Single-sample `generate()` now, batch later**: splitting the single-sample / batch work across phases 5/6 lets us prove determinism end-to-end in Phase 5 without also debugging ProcessPool RNG propagation. The batch work in Phase 6 is then pure parallelization — all the RNG / writer / manifest hardpoints are already solid.
- **No new runtime deps** (D-43 implication): Phase 5 is stdlib + already-present deps. Keeps the dependency surface narrow as the library matures.

</specifics>

<deferred>
## Deferred Ideas

- **`generate_batch(config)` via `ProcessPoolExecutor`** — Phase 6 (R-P10). Wraps Phase 5's `generate()`. Worker seeding pattern already proven by the single-sample determinism test (D-29); Phase 6 extends to "each worker seeds its local `random.Random` on entry — never inherit from the parent" per R-P7 bullet 4.
- **`typer`-based CLI (`musicgen generate --count N --out DIR --seed S`)** — Phase 6 (R-P13). Phase 3 shipped a stub `cli.py`; Phase 6 replaces with the full surface.
- **FluidSynth pre-roll calibration (`src/musicgen/calibrate.py`)** — Phase 6 (R-P9). Writes `.musicgen/fluidsynth_preroll.json`; pre-roll offset flows into `sample.json.pre_roll_offset_seconds` (today a `None` placeholder per D-22).
- **`--output-mode` flag (`full` / `mix-only` / `stems-only` / `midi-only`)** — Phase 6 (R-P14). Writer gets a `mode` parameter; conditional writes.
- **Structured JSON progress logs during batch** — Phase 6 (R-P15).
- **`musicgen clean --failed` subcommand** — Phase 6 (R-P13 second bullet).
- **Sharded `<dataset>/<hex>/<id>/` layout** — v0.2+ only (REQUIREMENTS "Out of scope"). 6-digit index suffices for 1k–10k scale; 100k+ pushes this.
- **Custom split labels beyond train/valid/test** — v0.2+ (D-27 fixes labels).
- **Golden-regeneration automation beyond `--regen-goldens`** (D-32) — a CI job that regenerates goldens on the pinned-FluidSynth runner when a developer explicitly tags a commit. Not needed for v0.1; `--regen-goldens` local workflow is sufficient.
- **Compressed annotations** (`sample.json.gz` instead of plain JSON) — v0.2+ if dataset size pressure emerges. Plain JSON wins on tooling compatibility at 10k scale.
- **Per-sample working-dir preservation for debugging** — `config.keep_working_dirs = True` is enough (D-10); no separate "--debug-sample" mode needed this phase.

</deferred>

---

*Phase: 05-productize-i-writer-manifest-seeds-determinism*
*Context gathered: 2026-04-19*
