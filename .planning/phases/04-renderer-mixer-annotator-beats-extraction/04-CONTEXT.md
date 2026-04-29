# Phase 4: Renderer + mixer + annotator + beats extraction - Context

**Gathered:** 2026-04-19 (auto mode — recommended defaults applied per gray area)
**Status:** Ready for planning

<domain>
## Phase Boundary

Extract the **audio-side** of the pipeline out of `music_gen.py` into four package modules:

1. `src/musicgen/renderer.py` — FluidSynth wrapper; parallelizes the four per-part stem renders with `ThreadPoolExecutor(max_workers=4)`; captures FluidSynth binary version at module load.
2. `src/musicgen/mixer.py` — pedalboard FX application + pydub overlay + part concatenation; uses the fixed `apply_gain` / `.pan()` capture APIs (Phase 1 R-S4 fix); writes silent stems for absent layers.
3. `src/musicgen/annotator.py` — pure function producing the Phase 4 subset of the `sample.json` dict (R-P4 schema; fields Phase 5 fills get TBD flags — see D-14).
4. `src/musicgen/beats.py` — MIDI-tick-derived beat + downbeat timestamps, swing-aware; **replaces** the standalone `beat_anotator.py` (which uses a theoretical grid that drifts with swing > 0, PITFALLS P-3).

After Phase 4: `mix_and_save` is gone (or < 50 lines of pure orchestration, per the roadmap exit criterion); `music_gen.py` is reduced to a thin orchestrator calling sampler → generators → renderer → mixer → annotator; `python music_gen.py` still produces one smoke-test sample end-to-end via the Phase 3 shim; annotator schema is pre-reserved but incomplete (seed/version/split/beat-times fields are Phase 5); beats produced by `beats.py` match MIDI ticks not theoretical grid.

This phase does NOT: wire per-sample output layout (R-P1, Phase 5), implement `derive_sample_seed` / `make_rngs` hierarchy (R-P7, Phase 5), produce `manifest.jsonl` (R-P5, Phase 5), run batch generation (R-P10, Phase 6), or calibrate FluidSynth pre-roll (R-P9, Phase 5). Stem persistence to final dataset directory layout is Phase 5; Phase 4 writes stems to a per-sample working dir but does not own the zero-padded-index naming.

</domain>

<decisions>
## Implementation Decisions

### Module split + file inventory

- **D-01:** Four new package modules this phase, all under `src/musicgen/`:
  - `src/musicgen/renderer.py` — FluidSynth wrapper, parallel stem renders, version capture. Surface: `render_stems(midi_paths: Dict[str, str], soundfonts: Dict[str, str], out_dir: str) -> RenderResult` + module-level `FLUIDSYNTH_VERSION: str`.
  - `src/musicgen/mixer.py` — FX + overlay + part concat. Surface: `mix_part(render_result, levels, fx_boards, layer_mask, part, out_dir) -> MixResult` and `concat_parts(part_mixes, out_path) -> str`.
  - `src/musicgen/annotator.py` — pure function. Surface: `annotate(song_params, render_results, mix_results, beat_times, downbeat_times, musicality_score) -> dict`. Zero I/O inside the function (caller writes the JSON in Phase 5).
  - `src/musicgen/beats.py` — MIDI-tick beat derivation. Surface: `extract_beat_times(midi_path, tempo, start_offset_seconds) -> List[float]` and `extract_downbeat_times(beat_times, time_signature) -> List[float]` + `beat_duration(signature, tempo)` moved here (removed from `generators/beat.py` once callers migrate — see D-19).
- **D-02:** Shared dataclasses live in the module that produces them:
  - `RenderResult` (dataclass, frozen) in `renderer.py` — per-layer dict of stem WAV paths + sample rate + duration + FluidSynth version.
  - `MixResult` (dataclass, frozen) in `mixer.py` — mix WAV path, per-layer post-FX stem WAV paths, `part_layers` mask, soundfonts dict, pedalboards dict, transitions list.
  - Both use `@dataclass(frozen=True)` (Python 3.10+ per Phase 3 D-13 override; `slots=True` allowed but not required).
- **D-03:** `beat_anotator.py` is **deleted** this phase. No shim, no re-export. Grep confirms it has zero importers (it is run as `python beat_anotator.py <dir>` after generation, not imported). Its `if __name__ == "__main__"` entry point is deprecated — Phase 5 writer owns the annotation lifecycle end-to-end.
- **D-04:** `musicality_score.py` stays at repo root for this phase. Phase 3 D-11 deferred its move; `create_song` still calls `musicality_score.get_musicality_score(wav_name)` directly. The annotator takes the score dict as an input parameter — it does not import `musicality_score`. Module move deferred to Phase 5 when the writer owns orchestration.

### Renderer design (R-X4)

- **D-05:** `midi2audio` stays. Do NOT switch to `pyfluidsynth` — same binary underneath, no determinism gain (research/STACK.md §FluidSynth). Renderer thinly wraps `FluidSynth(soundfont).midi_to_audio(midi_path, out_wav)`.
- **D-06:** Parallelism: `ThreadPoolExecutor(max_workers=4)` inside `render_stems` — submits four per-stem renders (beat/melody/harmony/bassline) for **one part** in parallel. Parts remain serial (today's order). Threads suffice because FluidSynth is a subprocess; `ProcessPoolExecutor` would add fork overhead for no determinism win. The outer per-sample parallelism (across parts or across samples) is Phase 6's ProcessPoolExecutor (R-P10) and is NOT added here.
- **D-07:** `FLUIDSYNTH_VERSION` captured at module import time as `subprocess.run(["fluidsynth", "--version"], capture_output=True, text=True, timeout=5).stdout.splitlines()[0]` (or `"unknown"` with a `logger.warning` on subprocess error — do not raise at import; renderer module must be importable on CI without FluidSynth installed for unit tests that fake the subprocess). Recorded once; re-reading is not needed. Annotator surfaces this in every annotation dict (R-P4 `fluidsynth_version`).
- **D-08:** Soundfont selection moves into renderer orchestration **upstream** of `render_stems` — `music_gen.py`'s orchestrator calls `pick_soundfonts(cfg, rng) -> Dict[str, str]` (new thin helper in `renderer.py`) before `render_stems`. The helper replaces the bare `random.choice(sound_fonts)` in today's `get_random_sound_font` — `_rng` is threaded in (D-17). Renderer itself takes the soundfonts dict as a parameter; it does not pick.
- **D-09:** Renderer writes stems to `<working_dir>/<name>-<part>/stems/<layer>.wav` (working-dir layout owned by the Phase 4 orchestrator; Phase 5 R-P1 rewrites this to `<dataset_root>/<zero-padded-index>/stems/<layer>.wav`). Phase 4 does not care about the index-based naming yet — it takes `out_dir` as a parameter.

### Mixer design (R-X5)

- **D-10:** FX application, pydub overlay, part concatenation, layer-inclusion masking **all move** into `mixer.py`. The mixer owns:
  1. FX pedalboard construction (`generate_pedalboard` moves from `music_gen.py` into `mixer.py`; `create_effect`, `apply_fx_to_layer`, `pedalboard_info_json` move with it). Construction takes an explicit `rng: random.Random` parameter (D-17) — currently bare `random.random()` / `random.uniform()`.
  2. FX application — wraps `apply_fx_to_layer` per post-render stem.
  3. Level application — the R-S4 fix (Phase 1) moves verbatim; `apply_gain(lin_to_db(v))` and `segment = segment.pan(...)`. The `_lin_to_db` helper becomes a module-private function in `mixer.py`.
  4. Layer-inclusion overlay — reads the `layer_mask: Dict[str, bool]` parameter; overlays only masked-in layers.
  5. Part concatenation — `concat_parts(part_mixes, out_path)` wraps the `AudioSegment.from_wav(...) + ...` loop today at `music_gen.py:333-340`.
- **D-11:** **Preserve current behavior**: apply FX to ALL four layers regardless of whether they are included in the mix. The `music_gen.py:276` "TODO: optimize it so that the fx are only applied to the used layers" is explicitly OUT OF SCOPE for Phase 4. Rationale: the current code's RNG draws inside FX construction happen unconditionally today; changing this changes RNG consumption and breaks the Phase 5 golden-seed baseline before it's established. Phase 5 or later can optimize after baselines are locked.
- **D-12:** **Silent-stem fallback:** when `layer_mask[layer] == False`, mixer writes a silent WAV of the same duration + sample rate as the rendered layer (source: `RenderResult.duration_seconds` + `RenderResult.sample_rate`) using `AudioSegment.silent(duration=ms)`. This guarantees `stems + mix` accounting works for every sample (Phase 5 R-P2 assertion) without special-casing absent layers. Silent stems still go through the `apply_gain` / `.pan()` path so their final amplitude is a deterministic zero, not an uninitialized NaN.
- **D-13:** Layer-inclusion mask (`part_layers` dict in today's code) is produced by the **mixer**, not the orchestrator, and uses an injected `rng` (D-17). Currently at `music_gen.py:250-253` as `random.random() <= beat_proba`. The mixer exposes `compute_layer_mask(song_unique_parts, inst_proba, rng) -> Dict[str, Dict[str, bool]]`. Rationale: it is mix-time probability, not song-structure sampling (unlike arrangement, which correctly sits in `sampler.py`). Moving it upstream into sampler would drag FX/soundfont RNG earlier in the stream and disrupt the RNG order commitment from Plan 02-02 A3 / Phase 3 D-21.

### Annotator design (R-X6)

- **D-14:** Annotator is a **pure function**, no I/O inside. Signature:
  ```python
  def annotate(
      song_params: SongParams,
      render_result: RenderResult,
      mix_result: MixResult,
      beat_times: Dict[str, List[float]],         # per-part
      downbeat_times: Dict[str, List[float]],     # per-part
      musicality: dict,                           # {"score": float, "components": {...}}
      *,
      fluidsynth_version: str,
      musicgen_version: Optional[str] = None,     # Phase 5 fills
      seed: Optional[int] = None,                 # Phase 5 fills
      split: Optional[str] = None,                # Phase 5 fills (R-P6)
  ) -> dict
  ```
  Returns a plain dict ready to `json.dump`. Caller writes the file (Phase 5 writer owns lifecycle).
- **D-15:** **Schema coverage this phase — fill what Phase 4 can, leave the rest as explicit TBD flags.**
  - **FILLED:** `key`, `mode` (derived from key: minor if key ends in "m"), `tempo_bpm`, `time_signature` (song-level base sig), `time_signatures_per_part`, `measures_per_part`, `swing`, `song_arrangement` (list of `{part, start_seconds, end_seconds}` derived from `mix_result.transitions`), per-part `chord_progression`, per-part `active_layers`, per-part `soundfonts`, per-part `fx_params`, per-part `beat_times`, per-part `downbeat_times`, `musicality_score` (full analyzer output), `duration_seconds` (total), `fluidsynth_version`, relative paths (`mix`, `stems.*`, `midi.*`).
  - **TBD (Phase 5 fills):** `seed`, `musicgen_version`, `split` (R-P6), `pre_roll_offset_seconds` (R-P9), `analysis_failed` flag (R-P6 failure pathway), sample-index, dataset-root-relative vs absolute paths (Phase 4 emits absolute paths from the working dir; Phase 5 relocates + rewrites as relative).
- **D-16:** **TBD-flag representation:** fields Phase 5 fills appear in the dict with value `None` (not missing, not `"TBD"` string). Rationale: matches Python/JSON nullability idiom, lets Phase 5 code just reassign by key, and downstream test fixtures can `assert d["seed"] is None` unambiguously. The annotator's docstring lists which keys will be `None` in Phase 4 and must be non-`None` by Phase 5 exit. `analysis_failed` is `False` (not `None`) when present this phase — the key itself is optional (R-P6 says "optional: analysis_failed: true if scoring raised") and omitted when scoring succeeds.

### RNG threading (carry Phase 3 D-07/D-08 forward)

- **D-17:** Every new function in renderer, mixer, annotator, and beats that performs a random draw takes an explicit `rng: random.Random` parameter. Zero bare `random.choice / random.random / random.uniform / random.randint` anywhere under `src/musicgen/` after this phase. The AST static guard from Phase 3 (`tests/test_sampler.py::test_no_bare_random_in_sampler`) is generalized to scan `src/musicgen/**/*.py` as part of this phase (extend to full package), catching any regression at test time. Call sites carrying RNG draws this phase:
  - `pick_soundfonts` (renderer) — `random.choice` → `rng.choice`
  - `generate_pedalboard` → `create_effect` (mixer) — `random.random()` + `random.uniform()` → `rng.random()` / `rng.uniform()`
  - `compute_layer_mask` (mixer) — `random.random()` → `rng.random()` for the four `<=proba` draws per part
  - `annotator` — **no random draws**; pure function
  - `beats` — **no random draws**; deterministic derivation from MIDI ticks
- **D-18:** Phase 4 still uses **one** module-level `_rng = random.Random()` threaded through `music_gen.py`'s orchestrator, same as Phase 3 D-08. No `derive_sample_seed` / `make_rngs` in this phase — that is Phase 5 R-P7. The orchestrator passes `_rng` into `pick_soundfonts`, `generate_pedalboard` (×4), `compute_layer_mask`. When Phase 5 lands the RNG hierarchy, these call sites become named-RNG aware (`rngs["soundfonts"]`, `rngs["fx"]`, `rngs["mix"]`) with zero signature changes — the parameter name stays `rng`.

### Beats module design (R-X7)

- **D-19:** `beats.py` **replaces** `beat_anotator.py` entirely. The derivation approach is MIDI-tick-based using `mido` (already a transitive dep via `midi2audio`, confirmed present):
  ```python
  def extract_beat_times(midi_path: str, tempo: int, start_offset_seconds: float) -> List[float]:
      # Iterate mido.MidiFile(midi_path) track, accumulate mido.tick2second(msg.time, ...)
      # Emit a timestamp for every note_on event with velocity > 0 (what beat_anotator.py already does).
      # Swing is already baked into the MIDI onset times (generators/beat.py applies calculate_swing_offset
      # when writing the file), so extracting from MIDI ticks is automatically swing-aware.
  ```
  This is what `beat_anotator.py:extract_midi_beats` already does — the replacement deletes the dead `compare_beats` / `theoretical_beats` code paths and the `generate_annotations` entry point (Phase 5 writer replaces that).
- **D-20:** **Downbeat derivation:** `extract_downbeat_times(beat_times, time_signature)` groups beats by measure based on the numerator. Assumption: `beat_times` contains exactly `numerator × measures` entries (kick-on-every-beat in the current beat pattern files). **Validated by test fixture**: synthesize a known MIDI, run `extract_beat_times` + `extract_downbeat_times`, assert downbeats == `beat_times[::numerator]`. If the assumption fails for some signature (e.g., 6/8 is actually 2 dotted-quarter beats, not 6 eighths), the mapping becomes `beat_times[::numerator_group]` with `numerator_group` from `TimeSignatureRegistry` — add a `primary_beat_group` field to the registry only if the test fails (document as a Phase-4 discovery then).
- **D-21:** `beats.py` owns `beat_duration(signature, tempo)` — it is imported by `generators/beat.py` today (D-19 of Phase 3 / file `generators/beat.py:32`). After Phase 4, `generators/beat.py` imports from `musicgen.beats` (re-export kept as compatibility alias in `generators/beat.py` so the rest of generators' contract stays stable). `calculate_swing_offset` stays in `generators/beat.py` — it is used only at MIDI generation time, not at annotation time.
- **D-22:** `beats.py` is called from the **orchestrator**, not the mixer or renderer. Orchestrator sequence: render → mix → for each part, run `extract_beat_times(midi_paths["beat"], tempo, part_start_offset)` + `extract_downbeat_times(beat_times, signature)` → pass both dicts to annotator. Rationale: beats consume MIDI (not audio), so they could run in parallel with render, but keeping it post-mix lets us use `mix_result.transitions` for `start_offset_seconds`. The speedup from parallelizing is marginal (mido scan << FluidSynth render). Keep it serial for clarity.

### Orchestrator shape post-phase

- **D-23:** **`mix_and_save` is deleted**, not trimmed. Its responsibilities split as:
  - Soundfont picking → `renderer.pick_soundfonts(cfg, rng)` (D-08)
  - Rendering loop → `renderer.render_stems(midi_paths, soundfonts, out_dir)` per part
  - FX board construction → `mixer.build_fx_boards(cfg, rng)` (returns `Dict[str, Pedalboard]`)
  - Layer mask → `mixer.compute_layer_mask(song_unique_parts, inst_proba, rng)`
  - Mix per part → `mixer.mix_part(render_result, levels, fx_boards, layer_mask, part, out_dir)`
  - Part concat → `mixer.concat_parts(part_mixes, out_path)`
  - Beats → `beats.extract_beat_times(...)` / `beats.extract_downbeat_times(...)`
  - Annotation dict assembly → `annotator.annotate(...)`
- **D-24:** `music_gen.py` becomes a **thin orchestration shim** — `create_song` is the top-level function that chains sampler → generators → renderer → mixer → beats → annotator → musicality scoring → JSON dump. Target: `music_gen.py` total length < 200 lines (currently 523). Exit criterion from roadmap is "`mix_and_save` is < 50 lines of pure orchestration" — we satisfy it trivially by removing `mix_and_save` entirely. `music_gen.py` keeps the `if __name__ == "__main__":` guard as the smoke-test entry point (same role it plays in Phase 3). Final collapse into `src/musicgen/cli.py` / `src/musicgen/__init__.py` is Phase 5 (library API) and Phase 6 (full CLI).
- **D-25:** Config threading: every new Phase 4 function accepts `cfg: config.Config = None` with runtime fallback `_cfg = cfg if cfg is not None else config.Config()`. Same pattern as Phase 2 D-01/D-02 and Phase 3 generators (`generators/beat.py` D-02). Keeps modules unit-testable without forcing a Config construction in every test.

### Testing strategy

- **D-26:** **Annotator tests** (`tests/test_annotator.py`) — fixture-driven pure-function tests. Build a handcrafted `SongParams` + synthetic `RenderResult` + `MixResult` + beat/downbeat lists + a stub `musicality` dict → call `annotate(...)` → assert the returned dict matches a golden dict (stored inline in the test, not a file). Cover: all filled fields present, Phase-5 TBD fields are `None`, `analysis_failed` handling, nested per-part structures correct.
- **D-27:** **Beats tests** (`tests/test_beats.py`) — synthesize a small MIDI fixture (one measure of 4/4 at 120 bpm via `midiutil`), call `extract_beat_times`, assert timestamps at [0.0, 0.5, 1.0, 1.5]. Swing cases required by the roadmap — three cases: `swing_amount = 0.5` (straight), `0.66` (moderate), `0.75` (heavy). For each, generate a MIDI with `generators.beat.generate_beat(...)` (seeded RNG), call `extract_beat_times` + `extract_downbeat_times`, assert: (a) extracted times are monotonic; (b) swung off-beats are later than the straight grid; (c) downbeats == `beats[::numerator]`.
- **D-28:** **Renderer tests** (`tests/test_renderer.py`) — FluidSynth subprocess is the hard part. Approach: mock `FluidSynth.midi_to_audio` to create a dummy WAV of the correct duration at a known sample rate (use `AudioSegment.silent(...).export(...)` as the fake render). Assert `render_stems` dispatches all four layers, reads back their durations/sample rates, and assembles `RenderResult` correctly. **The FluidSynth-subprocess integration** is covered by the single E2E slow test (D-30), not by unit tests.
- **D-29:** **Mixer tests** (`tests/test_mixer.py`) — seeded-RNG tests for `build_fx_boards` and `compute_layer_mask` (same-seed → same-output). For `mix_part` / `concat_parts`, synthesize short silent-and-tonal `AudioSegment`s in-memory, write to `tmp_path`, call the mixer, assert output WAV duration + channel count + (for silent-stem layers) that output RMS is zero. Do not assert on sample-by-sample FX output — that is pedalboard's contract, not ours.
- **D-30:** **E2E integration test** (`tests/test_integration_full_generation.py`, marked `@pytest.mark.slow`) — one-part smoke test: build a minimal `SongParams` (1 part, 1 measure, 4/4, swing 0.5), thread a seeded RNG through the full pipeline (sampler already done in unit scope — feed params directly), call generators → renderer → mixer → beats → annotator, assert: output files exist (4 stems, 1 mix, 4 midi), annotator dict has all Phase-4 filled fields, TBD fields are None. Single seed produces reproducible output (bit-identical MIDI; WAV identity is the Phase 5 golden test, not this one). Skipped when `fluidsynth` is not on PATH — the test guards with `pytest.importorskip`-style check on the FluidSynth binary.
- **D-31:** **AST static guard** — extend Phase 3's `test_no_bare_random_in_sampler` to a new `tests/test_no_bare_random_in_package.py` that scans all `src/musicgen/**/*.py` via `ast.walk` for `ast.Attribute(value=ast.Name(id="random"), attr in {"choice", "random", "randint", "choices", "uniform"})` outside of guarded test-only paths. Failing this test is a hard regression signal for D-17.
- **D-32:** Test migration from existing suite: `beat_anotator.py` has no tests today (verified) → nothing to migrate. The 371-test Phase 3 baseline stays green; Phase 4 adds ~20-30 new tests (annotator fixtures + beats swing cases + mixer RNG + renderer subprocess mocks + E2E slow test) for an expected suite size of ~400.

### Parallelization vs Phase 3

- **D-33:** Phase 4 runs **serially after** Phase 3 (same decision as Phase 3 D-25 — Phase 3 depended on Phase 2, Phase 4 depends on Phase 3 for the src/musicgen/ package skeleton + the fixed sampler/generator contracts). The roadmap's Phase 3 ∥ Phase 4 parallelism is theoretical for a multi-agent scenario; single-session execution keeps them serial.

### Claude's Discretion

- Exact field names inside `RenderResult` / `MixResult` dataclasses (`duration_seconds` vs `duration_s` vs `duration`). Shape is fixed (D-02); names are aesthetic.
- Whether silent-stem WAVs are written in stereo (matching rendered stems) or mono. Match whatever `AudioSegment.silent(...).export()` produces by default unless sum-of-stems assertion forces stereo parity.
- Module docstring style — match the existing `generators/beat.py` Google-style-with-D-reference style.
- Whether `FLUIDSYNTH_VERSION` is captured as the full first line of `fluidsynth --version` output (e.g., "FluidSynth runtime version 2.3.4") or parsed down to `"2.3.4"`. Keep the full line for provenance; the regex parse is a Phase 5 concern if calibration cares.
- Whether `pick_soundfonts` reads `.sf2` files via `os.listdir` + filter or via `Path.glob`. Match existing `get_random_sound_font` style (`os.listdir` + `.endswith('.sf2')`).
- Whether `beat_duration` in `beats.py` is the exact same function body as today's `generators/beat.py:beat_duration` (D-21) or uses `TimeSignatureRegistry` for the quarter-note base. Using the registry is cleaner; preserving the current body is lower-risk — pick whichever keeps the baseline tests green.
- Whether `compute_layer_mask` takes `inst_proba` as a pre-loaded dict or loads it inside (via `cfg`). Pre-loaded is marginally cleaner for testability; loading inside matches the current code's pattern — match the current pattern unless a test forces otherwise.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase scope and requirements

- `.planning/ROADMAP.md` — Phase 4 section (deliverables, exit criteria, R-X4/X5/X6/X7/X8 coverage). Depends on Phase 2; can run parallel with Phase 3 but this session runs serial (D-33).
- `.planning/REQUIREMENTS.md` — R-X4 (renderer), R-X5 (mixer), R-X6 (annotator), R-X7 (beats MIDI-tick derivation), R-X8 (integration test). R-P4 (sample.json schema — subset filled now per D-15) and R-S4 (pydub gain/pan fix, Phase 1 — mixer inherits this behavior). R-P2 (stems-sum-to-mix — not enforced this phase, but stem persistence here enables the Phase 5 assertion).
- `.planning/PROJECT.md` — Core value ("every sample reproducible, fully-labeled"), determinism constraint, 1k–10k scale target.

### Architecture and design input

- `.planning/research/ARCHITECTURE.md` — proposed module layout (renderer / mixer / annotator / beats). Build order §"Extract + Test" #3-6. Data flow diagram (sampler → generators → renderer → mixer → annotator → writer). Seed propagation §"Seed / RNG propagation" (Phase 5 implements the hierarchy, but Phase 4 extraction must leave signatures ready — D-17/D-18).
- `.planning/research/PITFALLS.md` — P-2 (stems not persisted, sum-of-stems ≠ mix — persistence lands this phase, assertion Phase 5), P-3 (beat annotations drift from rendered audio — D-19/D-20 fix the swing-grid drift half; FluidSynth pre-roll is Phase 5), P-4 (RNG leakage — D-17 forbids bare random), P-8 (MIDI-to-WAV offset — Phase 5), P-1 (FluidSynth bit-reproducibility — recorded in annotator per D-14/D-15).
- `.planning/research/STACK.md` §FluidSynth — "keep midi2audio, pin the binary"; ThreadPoolExecutor(max_workers=4) is the correct parallelism primitive (D-05/D-06). Pytest markers §"slow" + "integration" for the integration test (D-30).
- `.planning/research/SUMMARY.md` — build order + "extract renderer/mixer before annotator because annotator consumes their outputs".

### Codebase context

- `.planning/codebase/ARCHITECTURE.md` §"Pipeline stages" — the god-file layering. §4 (audio rendering) and §5 (mixing) are the Phase 4 extraction targets.
- `.planning/codebase/STRUCTURE.md` — current file layout (sampler + generators extracted to `src/musicgen/` already; audio-side still at root in `music_gen.py`).
- `.planning/codebase/CONVENTIONS.md` — naming, logging pattern (`logger = logging.getLogger(__name__)`), typing conventions, import organization. New modules must match.
- `.planning/codebase/CONCERNS.md` — #1 (god file, being further decomposed), #6 (parallel FluidSynth never wired — R-X4 closes this), #8 (audio-mixing logic ownership — R-X5 closes this).
- `.planning/codebase/TESTING.md` — pytest layout, `@pytest.mark.slow` marker (already declared per Phase 2 pyproject), fixture conventions.

### Prior phase artifacts (decisions to carry forward)

- `.planning/phases/02-stabilize-ii-config-time-signature-registry-logging/02-CONTEXT.md` D-01/D-02/D-03 — config precedence and threading pattern (D-25 continues this). D-07 — logging level semantics (DEBUG state dumps, INFO milestones, WARNING recoverable, ERROR failures). Phase 4 modules follow both.
- `.planning/phases/02-stabilize-ii-config-time-signature-registry-logging/02-02-*-SUMMARY.md` — TimeSignatureRegistry API surface (`lookup`, `verify_chord_pattern_length`, `verify_beat_pattern_length`, `primary_beat_duration`, etc.) — beats.py may consult the registry for downbeat grouping (D-20 fallback path).
- `.planning/phases/03-package-skeleton-sampler-generators-extraction/03-CONTEXT.md` — D-07/D-08 (RNG threading, single module-level `_rng` via shim — continues this phase per D-17/D-18), D-13 (Python >= 3.10, hatchling build), D-22 (per-field dataclasses, not SongParams passed to leaf functions — mixer/renderer follow this), D-24 (music21 global RNG audit PASSED — no wrapper needed, confirmed by `tests/test_music21_isolation.py`).
- `.planning/phases/03-package-skeleton-sampler-generators-extraction/03-04-SUMMARY.md` — generator extraction pattern (file-per-generator, injected rng, cfg optional, `TimeSignatureRegistry` for attribute access). New Phase 4 modules mirror this structure.
- `.planning/phases/03-package-skeleton-sampler-generators-extraction/03-05-SUMMARY.md` — AST static guard pattern for bare `random.*` detection (extended in D-31); `tests/conftest.py` is deleted (pyproject `pythonpath=["."]` carries repo root).
- `.planning/phases/01-stabilize-i-bug-fixes-and-guardrails/01-01-*-SUMMARY.md` — R-S3 fix (arrangement produced once upstream, threaded into `mix_and_save`). Mixer inherits this threading — arrangement comes in as a parameter, never re-rolled.
- `.planning/phases/01-stabilize-i-bug-fixes-and-guardrails/01-02-*-SUMMARY.md` — R-S4 fix (`apply_gain(lin_to_db(v))` + `segment = segment.pan(...)`). Mixer D-10 moves this logic verbatim; the `_lin_to_db` helper is preserved.

### Source files to modify / extract from

- `music_gen.py` — extraction source. Phase 4 targets:
  - `mix_and_save` (193-345): deleted per D-23; responsibilities split across renderer + mixer + orchestrator
  - `generate_pedalboard` (141-160): moves to `mixer.py` per D-10
  - `create_effect` (130-139): moves to `mixer.py` per D-10
  - `apply_fx_to_layer` (162-171): moves to `mixer.py` per D-10
  - `pedalboard_info_json` (173-190): moves to `mixer.py` per D-10
  - `get_random_sound_font` (117-120): replaced by `renderer.pick_soundfonts(cfg, rng)` per D-08
  - `get_levels` (122-125), `read_instrument_probabilities` (112-115): move to `mixer.py` (mixer's callers) or stay as simple helpers in config — simpler path is move with `generate_pedalboard`
  - `save_beat_annotations` (98-107): replaced by annotator + Phase 5 writer; safe to delete this phase since `beat_anotator.py` is being deleted anyway
  - `create_song` (352-433): slimmed to orchestration; calls the new modules
  - `generate_song_parts` (435-469): unchanged (it already calls extracted generators)
  - `generate_song` (471-511): unchanged
- `beat_anotator.py` — **deleted** per D-03 / D-19
- `src/musicgen/generators/beat.py` — `beat_duration` moves to `musicgen.beats`; kept as re-export alias per D-21. `calculate_swing_offset` stays.
- `tests/` — new files per D-26/D-27/D-28/D-29/D-30/D-31; existing 371 tests stay green.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets

- **`midiutil.MIDIFile`** + **`mido.MidiFile`** — `midiutil` is write-only (used by generators); `mido` is read-capable (needed for `beats.py` MIDI-tick extraction). `beat_anotator.py:25-35` demonstrates the exact `mido.tick2second` pattern `beats.py` inherits.
- **`pedalboard.Pedalboard` + Compressor/Gain/Chorus/LadderFilter/Phaser/Delay/Reverb** — already constructed via `create_effect` probability + `value_range` dict pattern (`music_gen.py:130-160`). Lifts verbatim into `mixer.py`.
- **`AudioSegment.silent(duration=ms)`** — pydub primitive for silent-stem fallback (D-12). Already idiomatic via `music_gen.py:296`.
- **`AudioSegment.apply_gain` + `AudioSegment.pan`** — the Phase 1 R-S4 fix at `music_gen.py:285-294`. Mixer reuses the exact `_lin_to_db` helper.
- **`TimeSignatureRegistry.lookup(signature)`** — available for downbeat grouping fallback (D-20); already used by `generators/beat.py`.
- **`musicgen.duration_validator.DurationValidator`** — beats.py does NOT need this (derivation from MIDI ticks doesn't require duration validation); mixer doesn't need this either.
- **`config.Config` with `sf_layer_dir`, `fx_files`, `inst_probabilities_file`, `levels_file`** — renderer + mixer consume all four config surfaces; pattern established in `generators/beat.py`.

### Established Patterns (to match in new modules)

- Type hints on every signature, return type included.
- Module-level `logger = logging.getLogger(__name__)`; DEBUG for internal state, INFO for part-by-part progress, WARNING for recoverable oddities (FluidSynth version unknown), ERROR for failures.
- Google-style docstrings with `Args:` / `Returns:` sections (canonical shape at `src/musicgen/duration_validator.py:91-99` and `src/musicgen/generators/beat.py`).
- `from typing import Tuple, Dict, List, Optional` — no PEP 604 pipe syntax yet (Phase 3 D-13 settled on 3.10+ but stdlib typing remains the convention for symmetry with older modules).
- Narrow `music21` imports already landed (`from music21 import roman, scale, pitch`) — renderer/mixer/annotator/beats don't import `music21` at all.
- `cfg: config.Config = None` with runtime fallback — every extracted module function.
- Frozen dataclasses via `@dataclass(frozen=True)` — shape match Phase 3's `SongParams`.

### Integration Points

- `music_gen.py:create_song` (current line 352-433) — becomes the Phase 4 orchestrator. Calls into renderer, mixer, beats, annotator sequentially. Returns the annotation dict (replacing today's `song_info` dict).
- `music_gen.py:generate_song_parts` (435-469) — **unchanged this phase**. Already calls extracted generators with injected `_rng`. Feeds MIDI paths into the new renderer.
- `musicality_score.get_musicality_score(wav_name)` — caller stays in `create_song` for now (D-04). Returns `(score, component_scores)` which the orchestrator passes as a dict into `annotator.annotate(..., musicality={"score": ..., "components": ...})`.
- `tests/test_music_gen_logging.py` — AST-scan over `music_gen.py`. Shrinking `music_gen.py` from 523 to ~180 lines is expected; the tests assert no bare `print()` calls and the `if __name__ == "__main__"` guard — both must still pass after reduction.
- `tests/test_sampler.py::test_no_bare_random_in_sampler` — extended to `tests/test_no_bare_random_in_package.py` per D-31.

### Notes on the audio-side bare-random call sites (D-17 target)

Today's `music_gen.py` audio-side bare `random.*` inventory (all eliminated this phase):

- `get_random_sound_font:119` — `random.choice(sound_fonts)` × 4 layers = 4 call sites
- `create_effect:135-137` — `random.random()` + `random.uniform(...)` inside a dict comprehension = up to 8 call sites per pedalboard construction × 4 layers = 32 call sites per mix
- `mix_and_save:250-253` — `random.random()` × 4 layers per part = 4 call sites per part × N parts

Total: ~40 + 4·N_parts bare-random draws relocated behind the injected `rng` in Phase 4. The Phase 5 golden seed baseline is only valid after these are threaded, so Phase 4 locks the RNG draw order for every audio-side operation.

</code_context>

<specifics>
## Specific Ideas

- **Phase 4 is the "reproducibility enabler" phase.** Every bare `random.*` call in the audio path moves behind an injected RNG. Phase 5 then just wires the hierarchy — it does not refactor signatures. Every design decision should pass the same test Phase 3 used: "does this make Phase 5 seed discipline easy or hard?"
- **The annotator is the dataset's contract surface.** Phase 4 does not ship a fully populated `sample.json`, but it locks the **shape** — field names, nesting, types, null semantics (D-16). Phase 5 fills null fields; Phase 6 consumes the shape from the CLI. Changing the shape after Phase 4 is expensive.
- **`mix_and_save` deletion is the structural win.** The roadmap exit criterion ("< 50 lines of pure orchestration") is trivially satisfied by removing it entirely and distributing its body into renderer + mixer + orchestrator. This is the cleanest reading of the criterion.
- **`beat_anotator.py` deletion preserves correctness, not compatibility.** The file has no importers. Its straight-grid logic is actively wrong on swung samples (PITFALLS P-3). Replacing it with `beats.py` (MIDI-tick-derived) eliminates a bug, not just a god-file.
- **FluidSynth version capture is a one-line subprocess, not a config field.** Records the installed binary's version string at import time so every annotation's `fluidsynth_version` key is populated by the renderer module itself, not by plumbing through `Config`. If FluidSynth isn't installed at CI import time, annotator still works ("unknown" string + warning log).
- **Tests for the integration slow-test are gated on the FluidSynth binary.** The unit tests mock `FluidSynth.midi_to_audio` so they run everywhere; the E2E slow test requires FluidSynth on PATH and is marked `@pytest.mark.slow`. CI can opt-in (`pytest -m slow`) or opt-out (`pytest -m "not slow"`, the default from Phase 3's pyproject.toml).

</specifics>

<deferred>
## Deferred Ideas

- **Stem-sum-to-mix assertion** (R-P2 fail-check on `max(|sum(stems) − mix|) < ε`). Stems are persisted this phase (enables the assertion) but the check itself lands in Phase 5 alongside the writer + golden-seed regression test.
- **FluidSynth pre-roll calibration** (R-P9) — measuring and applying the subprocess startup offset to beat timestamps. Phase 5. Phase 4's beat timestamps are MIDI-tick-accurate but NOT WAV-onset-corrected.
- **Per-sample output layout** with zero-padded index directories (R-P1). Phase 5 writer owns lifecycle. Phase 4 renderer/mixer take a generic `out_dir` parameter and write there.
- **`manifest.jsonl` append** (R-P5) — Phase 5 writer.
- **`musicgen_version` + `seed` + `split` fields in annotation dict** — Phase 5 fills (D-16).
- **`analysis_failed` true-path** (R-P6 + R-P16 failure isolation) — Phase 5 wiring. Phase 4 annotator accepts the key but doesn't set it.
- **FX-only-to-used-layers optimization** (the long-standing `music_gen.py:276` TODO) — deferred; preserving current behavior locks the Phase 5 RNG baseline (D-11).
- **Moving `musicality_score.py` into the package** (Phase 3 D-11 already deferred to Phase 4/5; re-deferring to Phase 5 per D-04 because annotator does not depend on the module — caller passes the score dict in).
- **Batch/process-pool outer parallelism** (R-P10) — Phase 6.
- **Downbeat grouping fallback via `TimeSignatureRegistry.primary_beat_group`** — only added if the 6/8 assumption in D-20 fails at test time. Currently speculative.
- **pyfluidsynth migration** — explicit no-go (research/STACK.md §FluidSynth alternatives; same binary underneath, zero determinism gain).

</deferred>

---

*Phase: 04-renderer-mixer-annotator-beats-extraction*
*Context gathered: 2026-04-19 (auto mode — recommended defaults applied across 13 gray areas: module split, renderer design, mixer design, annotator schema coverage, TBD-flag representation, RNG threading, beats derivation, downbeat grouping, orchestrator shape, config threading, test strategy, parallelization, deferred-scope boundary)*
