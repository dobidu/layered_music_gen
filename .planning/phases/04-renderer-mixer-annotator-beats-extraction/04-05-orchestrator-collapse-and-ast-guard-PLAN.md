---
phase: 04-renderer-mixer-annotator-beats-extraction
plan: 05
type: execute
wave: 5
depends_on: ["04-00", "04-01", "04-02", "04-03", "04-04"]
files_modified:
  - music_gen.py
  - beat_anotator.py
  - tests/test_no_bare_random_in_package.py
autonomous: true
requirements: [R-X4, R-X5, R-X6, R-X7]
tags: [phase-4, orchestrator, collapse, ast-guard, beat_anotator-delete, d-17, d-23, d-24]

must_haves:
  truths:
    - "`music_gen.py` total length < 200 lines (D-24)"
    - "`music_gen.py` no longer contains `mix_and_save`, `generate_pedalboard`, `create_effect`, `apply_fx_to_layer`, `pedalboard_info_json`, `get_random_sound_font`, `save_beat_annotations`, `get_levels`, `read_instrument_probabilities` (D-23)"
    - "`beat_anotator.py` does not exist (D-03: deleted entirely)"
    - "`tests/test_no_bare_random_in_package.py` exists and asserts zero bare `random.<method>(` calls across `src/musicgen/**/*.py` (D-17/D-31)"
    - "`python music_gen.py` smoke test reaches the renderer+mixer stage (will fail at env — no fluidsynth / no .sf2 — but must NOT fail earlier due to import/orchestration breakage)"
    - "`tests/test_music_gen_logging.py` still passes (zero print(), __main__ guard, module-level logger all preserved)"
    - "`create_song` in music_gen.py calls renderer → mixer → beats → annotator in order (D-23)"
    - "All 371 baseline tests + Plan 04-01..04-04 new tests (~60+ total Phase 4 tests) remain green"
  artifacts:
    - path: "music_gen.py"
      provides: "Thin orchestration shim — create_song chains sampler → generators → renderer → mixer → beats → annotator"
      contains: "from musicgen import renderer, mixer, annotator, beats"
      contains_not: "def mix_and_save"
    - path: "tests/test_no_bare_random_in_package.py"
      provides: "Package-wide AST guard (parametrized over every *.py in src/musicgen/)"
      contains: "test_no_bare_random_in_package_module"
  key_links:
    - from: "music_gen.py create_song"
      to: "musicgen.renderer.pick_soundfonts, render_stems"
      via: "from musicgen import renderer"
      pattern: "renderer\\.(pick_soundfonts|render_stems)"
    - from: "music_gen.py create_song"
      to: "musicgen.mixer.build_fx_boards, compute_layer_mask, mix_part, concat_parts"
      via: "from musicgen import mixer"
      pattern: "mixer\\.(build_fx_boards|compute_layer_mask|mix_part|concat_parts)"
    - from: "music_gen.py create_song"
      to: "musicgen.beats.extract_beat_times, extract_downbeat_times"
      via: "from musicgen import beats"
      pattern: "beats\\.(extract_beat_times|extract_downbeat_times)"
    - from: "music_gen.py create_song"
      to: "musicgen.annotator.annotate"
      via: "from musicgen import annotator"
      pattern: "annotator\\.annotate"
---

<objective>
Collapse `music_gen.py` from 523 lines to < 200 lines of pure orchestration (D-23/D-24). Delete 9 functions that have been extracted to the Phase 4 modules:

1. `save_beat_annotations` (98-107) — DELETED (Phase 5 writer owns the annotation lifecycle)
2. `read_instrument_probabilities` (112-115) — DELETED (caller uses `_cfg.load_inst_probabilities()` — already exists in config.Config, or inline `json.load`)
3. `get_random_sound_font` (117-120) — DELETED (replaced by `renderer.pick_soundfonts`)
4. `get_levels` (122-125) — DELETED (caller uses `_cfg.load_levels()` — already exists, or inline `json.load`)
5. `create_effect` (130-139) — DELETED (moved to `mixer._create_effect`)
6. `generate_pedalboard` (141-160) — DELETED (replaced by `mixer.build_fx_boards`)
7. `apply_fx_to_layer` (162-171) — DELETED (moved to `mixer.apply_fx_to_layer`)
8. `pedalboard_info_json` (173-190) — DELETED (moved to `mixer.pedalboard_info_json`)
9. `mix_and_save` (193-345) — DELETED (distributed across renderer + mixer + orchestrator)

Rewrite `create_song` (currently lines 352-433) to orchestrate the new modules: sampler → generators → renderer (pick_soundfonts, render_stems) → mixer (build_fx_boards, compute_layer_mask, mix_part, concat_parts) → beats (extract_beat_times, extract_downbeat_times) → musicality → annotator.annotate → json.dump.

Delete `beat_anotator.py` outright (D-03 — zero importers, swing-drift bug, replaced by `musicgen.beats`).

Create `tests/test_no_bare_random_in_package.py` as a package-wide AST static guard that iterates every `*.py` in `src/musicgen/**` and asserts zero bare `random.<method>(` calls (D-17/D-31).

Purpose: This is the STRUCTURAL win of Phase 4. Roadmap exit criterion `"mix_and_save < 50 lines of pure orchestration"` is satisfied by removing it entirely. ~40+ bare `random.*` draws from the audio side have now moved behind injected RNG (confirmed by the AST guard). The god-file decomposition is complete for the audio pipeline.

Output: `music_gen.py` collapsed to < 200 lines (from 523), `beat_anotator.py` gone, `tests/test_no_bare_random_in_package.py` new.
</objective>

<execution_context>
@/home/bidu/musicgen/.claude/get-shit-done/workflows/execute-plan.md
@/home/bidu/musicgen/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/phases/04-renderer-mixer-annotator-beats-extraction/04-CONTEXT.md
@.planning/phases/04-renderer-mixer-annotator-beats-extraction/04-RESEARCH.md
@.planning/phases/04-renderer-mixer-annotator-beats-extraction/04-PATTERNS.md
@.planning/phases/04-renderer-mixer-annotator-beats-extraction/04-VALIDATION.md
@.planning/phases/04-renderer-mixer-annotator-beats-extraction/04-01-SUMMARY.md
@.planning/phases/04-renderer-mixer-annotator-beats-extraction/04-02-SUMMARY.md
@.planning/phases/04-renderer-mixer-annotator-beats-extraction/04-03-SUMMARY.md
@.planning/phases/04-renderer-mixer-annotator-beats-extraction/04-04-SUMMARY.md
@music_gen.py
@beat_anotator.py
@tests/test_music_gen_logging.py
@tests/test_sampler.py
@tests/test_generators/test_no_bare_random.py

<interfaces>
<!-- musicgen.renderer public surface (Plan 04-02) -->
FLUIDSYNTH_VERSION: str
class RenderResult:
    stem_paths: Dict[str, str]; sample_rate: int; channels: int; duration_seconds: float; fluidsynth_version: str
def pick_soundfonts(cfg=None, rng=None) -> Dict[str, str]
def render_stems(midi_paths, soundfonts, out_dir, cfg=None) -> RenderResult

<!-- musicgen.mixer public surface (Plan 04-03) -->
class MixResult:
    mix_path, stem_paths, part_layers, soundfonts, pedalboards, transitions
def build_fx_boards(cfg=None, rng=None) -> Dict[str, Pedalboard]
def compute_layer_mask(song_unique_parts, inst_proba, rng) -> Dict[str, Dict[str, bool]]
def mix_part(render_result, levels, fx_boards, layer_mask_for_part, part, out_dir, soundfonts, part_counter=1, song_time_start=0.0) -> MixResult
def concat_parts(part_mix_paths, out_path) -> str
def apply_fx_to_layer(wav_file, board) -> str
def pedalboard_info_json(board) -> list

<!-- musicgen.beats public surface (Plan 04-01) -->
def beat_duration(signature, tempo) -> float
def extract_beat_times(midi_path, tempo, start_offset_seconds) -> List[float]
def extract_downbeat_times(beat_times, time_signature, measures, start_offset_seconds, tempo) -> List[float]

<!-- musicgen.annotator public surface (Plan 04-04) -->
def annotate(song_params, render_results, mix_results, beat_times, downbeat_times, musicality, chord_progressions, midi_paths, mix_path, *, fluidsynth_version, musicgen_version=None, seed=None, split=None, analysis_failed=None) -> dict

<!-- Current music_gen.py surface the orchestrator MUST preserve (or keep as-is):
     - logger = logging.getLogger(__name__)   (line 41) — required by tests/test_music_gen_logging.py
     - _rng = random.Random()                  (line 46) — D-08/D-18 single threaded rng
     - validate_measures = validate_measures_dict  (line 50) — back-compat alias
     - verify_pattern_for_time_signature       (53-56) — registry delegate
     - verify_beat_pattern                     (59-62) — registry delegate
     - calculate_measures_for_time_signature   (64-66) — registry delegate
     - get_midi_time_signature_values          (72-76) — registry delegate
     - get_note_duration                       (78-81) — registry delegate
     - get_note_durations                      (83-86) — registry delegate
     - get_melody_durations                    (88-91) — registry delegate
     - create_song                             (352-433) — REWRITTEN by this plan
     - generate_song_parts                     (435-469) — UNCHANGED (already calls extracted generators with _rng)
     - generate_song                           (471-511) — UNCHANGED
     - if __name__ == "__main__" guard         (515-523) — UNCHANGED (logging.basicConfig + loop) -->

<!-- generate_chord_progression return tuple: (chord_progression, harm_filename_path)
     Currently generate_song_parts captures both but only stores the filename.
     Orchestrator refactor must collect chord_progression per part for annotator. -->

<!-- tests/test_music_gen_logging.py assertions (must continue passing):
     1. Zero print() calls in music_gen.py
     2. Import side-effect free
     3. logger = logging.getLogger(__name__) at module level
     4. logging.basicConfig inside __main__ guard only
     5. No f-strings in logger calls -->

<!-- tests/test_sampler.py::test_no_bare_random_in_sampler pattern (lines 165-196)
     — the base AST guard that the new test_no_bare_random_in_package.py generalizes -->

<!-- tests/test_generators/test_no_bare_random.py (lines 1-44) — already parametrizes
     over generators/*.py; the new package-wide guard parametrizes over all of src/musicgen/**/*.py -->
</interfaces>
</context>

<tasks>

<task type="auto">
  <name>Task 1: Collapse music_gen.py — delete 9 functions, rewrite create_song, add new imports</name>
  <files>music_gen.py</files>
  <read_first>
    - music_gen.py (current state — 523 lines; NOTE exact line ranges of the 9 functions to delete)
    - src/musicgen/renderer.py (from Plan 04-02 — know the pick_soundfonts + render_stems signatures)
    - src/musicgen/mixer.py (from Plan 04-03 — know the 7 public symbols + MixResult shape)
    - src/musicgen/beats.py (from Plan 04-01 — know extract_beat_times + extract_downbeat_times signatures)
    - src/musicgen/annotator.py (from Plan 04-04 — know the annotate(...) signature with 9 positional + 5 kwarg params)
    - src/musicgen/sampler.py (SongParams construction — orchestrator needs to build one from in-scope params)
    - src/musicgen/generators/chord.py (generate_chord_progression return tuple — orchestrator must capture chord_progression for annotator)
    - .planning/phases/04-renderer-mixer-annotator-beats-extraction/04-PATTERNS.md §"music_gen.py — orchestrator collapse" (authoritative collapse template)
    - tests/test_music_gen_logging.py (regression tests that must still pass)
  </read_first>
  <action>
Rewrite `music_gen.py` as a thin orchestration shim. Target: < 200 lines total (currently 523). Keep ONLY:

1. **Imports** — most current imports are still needed; add `from musicgen import renderer, mixer, annotator, beats`; DELETE unused imports (`from midi2audio import FluidSynth`, `from pedalboard import ...`, `from pedalboard.io import AudioFile`, `from pydub import AudioSegment` — all moved to mixer.py and renderer.py).
2. **Module-level constants** — `logger = logging.getLogger(__name__)` (line 41), `_rng = random.Random()` (line 46), `validate_measures = validate_measures_dict` (line 50).
3. **Registry delegates (UNCHANGED)** — `verify_pattern_for_time_signature`, `verify_beat_pattern`, `calculate_measures_for_time_signature`, `get_midi_time_signature_values`, `get_note_duration`, `get_note_durations`, `get_melody_durations` (lines 53-91).
4. **`create_song` — REWRITTEN** (new body ~50-70 lines of orchestration).
5. **`generate_song_parts` — MODIFIED** (must now ALSO return chord_progression per part for the annotator — today it captures and discards). Minimal change: add a 6th return value `chord_progressions: Dict[str, List[str]]` and update the call site in `create_song`.
6. **`generate_song` — UNCHANGED** (lines 471-511).
7. **`if __name__ == "__main__":` guard — UNCHANGED** (lines 515-523).

**DELETE these 9 functions (exact line ranges to remove):**

- `save_beat_annotations` (lines 98-107) — 10 lines
- `read_instrument_probabilities` (lines 112-115) — 4 lines
- `get_random_sound_font` (lines 117-120) — 4 lines
- `get_levels` (lines 122-125) — 4 lines
- `create_effect` (lines 130-139) — 10 lines
- `generate_pedalboard` (lines 141-160) — 20 lines
- `apply_fx_to_layer` (lines 162-171) — 10 lines
- `pedalboard_info_json` (lines 173-190) — 18 lines
- `mix_and_save` (lines 193-345) — 153 lines

Total deletion: ~233 lines. Also delete commentary-only lines (e.g., 93-96, 109-110, 127-128, 191 separator comments).

**DELETE these unused imports (moved to extracted modules):**

- `from midi2audio import FluidSynth` (line 4) — used only by mix_and_save/render; now inside renderer.py
- `from pedalboard import Pedalboard, Compressor, Gain, Chorus, LadderFilter, Phaser, Delay, Reverb` (line 6) — now inside mixer.py
- `from pedalboard.io import AudioFile` (line 7) — now inside mixer.py
- `from pydub import AudioSegment` (line 3) — now inside mixer.py

Keep these imports: `from midiutil import MIDIFile` (still used by generators indirectly — actually NOT used at the music_gen.py level; check and remove if grep confirms no references in the trimmed file), `from music21 import roman, scale, pitch` (used by generators — check and remove at music_gen.py level if no references remain post-trim), `from datetime import datetime`, `import logging, time, json, random, os, uuid, math, musicality_score`, the `from musicgen.*` imports (sampler, generators, duration_validator), `import config`, `from timesig import TimeSignatureRegistry`, `from typing import Tuple, Dict, List, Optional`.

After trim, re-run a quick review: if any of `FluidSynth`, `Pedalboard`, `AudioFile`, `AudioSegment`, `music21 import`, `MIDIFile` have ZERO references in the remaining music_gen.py body, DELETE them (keeping unused imports breaks Phase 1 R-S8 dead-code cleanup commitments).

**New `create_song` body (ORCHESTRATOR SHAPE — paste verbatim, adapting variable names as needed):**

```python
def create_song(
    key: str,
    tempo: int,
    song_signatures: Dict[str, str],
    measures: Dict[str, int],
    name: str,
    chord_pat_file: str,
    swing_amount: float,
    cfg: config.Config = None,
) -> Dict:
    _cfg = cfg if cfg is not None else config.Config()
    start_time = time.time()
    logger.info("Generating song '%s' with swing amount: %s", name, swing_amount)

    # R-S3: arrangement computed ONCE upstream. Must happen before any
    # downstream RNG-drawing steps so the RNG draw ordering is stable.
    song_unique_parts, song_arrangement = generate_song_arrangement(
        _rng, structures_file=_cfg.song_structures_file,
    )
    logger.info("Arrangement: %s", song_arrangement)

    # Step 1 — soundfont selection (renderer, D-08/D-17).
    soundfonts = renderer.pick_soundfonts(_cfg, _rng)
    for layer, sf_path in soundfonts.items():
        logger.info("%s soundfont: %s", layer.capitalize(), sf_path)

    # Step 2 — generate MIDI layers (existing generators).
    harm_filename, bass_filename, melo_filename, beat_filename, _beat_annotations_discarded, chord_progressions = generate_song_parts(
        key=key, tempo=tempo, song_signatures=song_signatures,
        song_measures=measures, name=name, chord_pat_file=chord_pat_file,
        swing_amount=swing_amount, cfg=cfg,
    )

    # Step 3 — FX boards + layer mask (mixer, D-10/D-13/D-17).
    fx_boards = mixer.build_fx_boards(_cfg, _rng)
    with open(_cfg.inst_probabilities_file) as _f:
        inst_proba = json.load(_f)
    layer_mask = mixer.compute_layer_mask(song_unique_parts, inst_proba, _rng)
    with open(_cfg.levels_file) as _f:
        levels = json.load(_f)

    # Step 4 — per-part render + mix (serial parts, D-06 parallel stems).
    render_results: Dict[str, renderer.RenderResult] = {}
    mix_results: Dict[str, mixer.MixResult] = {}
    beat_times_dict: Dict[str, List[float]] = {}
    downbeat_times_dict: Dict[str, List[float]] = {}
    midi_paths_dict: Dict[str, Dict[str, str]] = {}
    part_mix_paths: List[str] = []
    song_time_start = 0.0

    for part_counter, part in enumerate(song_arrangement, start=1):
        logger.info("Mixing part: %s (%d of %d)", part, part_counter, len(song_arrangement))
        midi_paths = {
            "beat":     beat_filename[part],
            "melody":   melo_filename[part],
            "harmony":  harm_filename[part],
            "bassline": bass_filename[part],
        }
        midi_paths_dict[part] = midi_paths

        out_dir = os.path.join(name, f"{name}-{part}")

        render_results[part] = renderer.render_stems(midi_paths, soundfonts, out_dir, cfg=_cfg)
        mix_results[part] = mixer.mix_part(
            render_result=render_results[part],
            levels=levels,
            fx_boards=fx_boards,
            layer_mask_for_part=layer_mask[part],
            part=part,
            out_dir=out_dir,
            soundfonts=soundfonts,
            part_counter=part_counter,
            song_time_start=song_time_start,
        )
        part_mix_paths.append(mix_results[part].mix_path)

        # Beats derivation (D-22: post-mix, serial).
        beat_times_dict[part] = beats.extract_beat_times(
            midi_paths["beat"], tempo, song_time_start,
        )
        downbeat_times_dict[part] = beats.extract_downbeat_times(
            beat_times_dict[part], song_signatures[part], measures[part],
            song_time_start, tempo,
        )
        song_time_start += render_results[part].duration_seconds

    # Step 5 — concatenate part mixes.
    final_wav = mixer.concat_parts(part_mix_paths, os.path.join(name, name + ".wav"))
    logger.info("Song saved as: %s", final_wav)

    # Step 6 — musicality scoring (stays at root, D-04).
    score, component_scores = musicality_score.get_musicality_score(final_wav)
    musicality = {
        "score": float(score),
        "components": {k: float(v) for k, v in component_scores.items()},
    }

    # Step 7 — annotation (pure function, D-14).
    song_params_obj = SongParams(
        key=key, tempo=tempo, time_signature_base=song_signatures.get("verse", "4/4"),
        time_signature_variation=1.0, swing_amount=swing_amount,
        signatures_per_part=song_signatures, measures_per_part=measures,
        song_unique_parts=list(song_unique_parts), song_arrangement=list(song_arrangement),
    )
    annotation = annotator.annotate(
        song_params=song_params_obj,
        render_results=render_results,
        mix_results=mix_results,
        beat_times=beat_times_dict,
        downbeat_times=downbeat_times_dict,
        musicality=musicality,
        chord_progressions=chord_progressions,
        midi_paths=midi_paths_dict,
        mix_path=final_wav,
        fluidsynth_version=renderer.FLUIDSYNTH_VERSION,
    )

    # Step 8 — write JSON (Phase 5 writer will eventually own this lifecycle).
    json_file = os.path.join(name, name + ".json")
    with open(json_file, "w") as outfile:
        json.dump(annotation, outfile, indent=4)

    elapsed = time.time() - start_time
    logger.info("Elapsed time: %.2f seconds", elapsed)
    logger.info("Musicality score: %.2f", score)
    logger.debug("Component scores: %s", component_scores)

    return annotation
```

**`generate_song_parts` — MODIFIED:** Add chord_progressions to the return tuple. Current implementation (line 435-469) captures `chord_progression` from `generate_chord_progression` but only stores `harm_filename[part]`. Change:

```python
def generate_song_parts(
    key: str, tempo: int, song_signatures: Dict[str, str], song_measures: Dict[str, int],
    name: str, chord_pat_file: str, swing_amount: float, cfg: config.Config = None,
) -> Tuple[Dict, Dict, Dict, Dict, Dict, Dict]:  # <-- 6th return dict added
    harm_filename, bass_filename, melo_filename, beat_filename, beat_annotations = {}, {}, {}, {}, {}
    chord_progressions: Dict[str, List[str]] = {}  # <-- NEW

    for part, measures in song_measures.items():
        logger.info("Generating part: %s (%s measures)", part, measures)
        name_part = f"{name}-{part}"
        time_signature = song_signatures[part]

        chord_progression, harm_filename[part] = generate_chord_progression(
            key, tempo, time_signature, measures, name_part, part, chord_pat_file, _rng,
        )
        chord_progressions[part] = list(chord_progression)  # <-- NEW

        melody, melo_filename[part] = generate_melody(
            key, tempo, time_signature, measures, name_part, part, chord_progression, _rng,
        )
        bass_filename[part] = generate_bassline(
            key, tempo, time_signature, measures, name_part, part, chord_progression, melody, _rng,
        )
        beat_filename[part], beat_annotations[part] = generate_beat(
            part, tempo, time_signature, measures, name_part, swing_amount, _rng, cfg=cfg,
        )

    return harm_filename, bass_filename, melo_filename, beat_filename, beat_annotations, chord_progressions
```

The `_beat_annotations_discarded` local in `create_song` acknowledges that the 5th return value (beat_annotations) is no longer used since beats come from `musicgen.beats.extract_beat_times` (not from `generate_beat`'s annotation output).

**`generate_song` — UNCHANGED.** Its call to `create_song` has the same arg shape.

**`if __name__ == "__main__":` guard — UNCHANGED.**

After deletion + rewrite, confirm total line count:

```bash
wc -l music_gen.py   # Must be < 200
```

If over 200, audit for dead imports and commentary that can be compressed.
  </action>
  <verify>
    <automated>wc -l music_gen.py && python -c "
# Verify all 9 target functions are gone
import ast
with open('music_gen.py') as f:
    tree = ast.parse(f.read())
deleted_names = {'save_beat_annotations', 'read_instrument_probabilities', 'get_random_sound_font',
                 'get_levels', 'create_effect', 'generate_pedalboard', 'apply_fx_to_layer',
                 'pedalboard_info_json', 'mix_and_save'}
present_defs = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
leftover = deleted_names & present_defs
assert not leftover, f'D-23 violation: these functions should be deleted but remain: {leftover}'
# Verify new modules are imported
with open('music_gen.py') as f:
    src = f.read()
for module in ('renderer', 'mixer', 'annotator', 'beats'):
    assert f'from musicgen import' in src and module in src, f'orchestrator does not import musicgen.{module}'
# Verify key orchestrator calls present
for call in ('renderer.pick_soundfonts', 'renderer.render_stems', 'mixer.build_fx_boards',
             'mixer.compute_layer_mask', 'mixer.mix_part', 'mixer.concat_parts',
             'beats.extract_beat_times', 'beats.extract_downbeat_times', 'annotator.annotate'):
    assert call in src, f'orchestrator missing expected call: {call}'
# Verify __main__ guard preserved
assert 'if __name__ == \"__main__\"' in src
assert 'logging.basicConfig' in src
assert 'logger = logging.getLogger(__name__)' in src
# Verify line count < 200
with open('music_gen.py') as f:
    line_count = len(f.readlines())
assert line_count < 200, f'D-24 violation: music_gen.py is {line_count} lines, must be < 200'
print(f'music_gen.py collapse OK: {line_count} lines, all 9 target functions deleted')
" && python -m pytest tests/test_music_gen_logging.py -x -q 2>&1 | tail -10</automated>
  </verify>
  <acceptance_criteria>
    - `wc -l music_gen.py` returns < 200 (D-24)
    - `grep -c "^def mix_and_save\|^def generate_pedalboard\|^def create_effect\|^def apply_fx_to_layer\|^def pedalboard_info_json\|^def get_random_sound_font\|^def save_beat_annotations\|^def get_levels\|^def read_instrument_probabilities" music_gen.py` returns `0` (all 9 functions deleted)
    - `grep "from musicgen import" music_gen.py | grep -E "renderer|mixer|annotator|beats"` returns at least 1 line matching all four modules (or 4 separate `from musicgen import X` lines totaling all four)
    - `grep "renderer.pick_soundfonts\|renderer.render_stems" music_gen.py` returns at least 2 lines
    - `grep "mixer.build_fx_boards\|mixer.compute_layer_mask\|mixer.mix_part\|mixer.concat_parts" music_gen.py` returns at least 4 lines
    - `grep "beats.extract_beat_times\|beats.extract_downbeat_times" music_gen.py` returns at least 2 lines
    - `grep "annotator.annotate" music_gen.py` returns at least 1 line
    - `grep "logger = logging.getLogger(__name__)" music_gen.py` returns exactly 1 line (test_music_gen_logging.py dependency)
    - `grep "if __name__" music_gen.py` returns exactly 1 line (__main__ guard preserved)
    - `grep "print(" music_gen.py` returns `0` (test_music_gen_logging.py::TestNoPrintCallsRemain)
    - `pytest tests/test_music_gen_logging.py -x -q` exits 0 (all 4 regression tests still pass)
    - `python -c "import music_gen; print('imported OK')"` prints `imported OK` (import side-effect free; R-S1 preserved)
    - `pytest tests/ -m "not slow" -q` exits 0 — full suite green (no regression)
  </acceptance_criteria>
  <done>`music_gen.py` collapses to < 200 lines; 9 audio-side functions deleted; new orchestrator body chains renderer → mixer → beats → annotator; `generate_song_parts` returns `chord_progressions` as a 6th dict; `__main__` guard, module-level logger, and all 7 registry-delegate functions preserved; all regression tests (test_music_gen_logging.py + full suite) green.</done>
</task>

<task type="auto">
  <name>Task 2: Delete beat_anotator.py + create tests/test_no_bare_random_in_package.py AST guard</name>
  <files>beat_anotator.py, tests/test_no_bare_random_in_package.py</files>
  <read_first>
    - beat_anotator.py (confirm it exists and has zero importers before deletion)
    - tests/test_no_bare_random_in_package.py (Wave 0 stub — replaced entirely)
    - tests/test_sampler.py (lines 165-196 — the base _bare_random_calls helper + assertion pattern to copy)
    - tests/test_generators/test_no_bare_random.py (lines 1-44 — the parametrize-over-glob generator-scoped pattern to generalize)
    - src/musicgen/ tree structure (the paths the parametrize will iterate)
  </read_first>
  <action>
**Step A — Delete `beat_anotator.py` outright (D-03/D-19).**

Pre-delete safety check: run `grep -rn "beat_anotator" . --include="*.py" 2>/dev/null` to confirm zero importers. Expected output: ONLY the self-reference inside `beat_anotator.py` itself (the `__main__` block prints `"Usage: python beat_annotator.py <instance_dir>"`). No `import beat_anotator` or `from beat_anotator` lines anywhere.

Then delete:
```bash
rm beat_anotator.py
```

Use `git rm beat_anotator.py` (not raw `rm`) so the deletion is staged correctly and `git log --follow` still traces history.

**Step B — Replace `tests/test_no_bare_random_in_package.py` Wave 0 stub with a real package-wide AST guard (D-17/D-31).**

Full file content:

```python
"""Static guard: zero bare random.<method>() in src/musicgen/**/*.py (D-17/D-31).

Generalizes the existing scoped guards:
  - tests/test_sampler.py::test_no_bare_random_in_sampler  (sampler.py only)
  - tests/test_generators/test_no_bare_random.py           (generators/*.py only)

This package-wide version is PARAMETRIZED over every ``*.py`` under
``src/musicgen/`` (recursive; excludes ``__init__.py``), so adding any new
module automatically extends the guard — no test file edits required.

The ``random.Random`` constructor IS permitted (it's the RNG factory, not a
bare draw). Every other ``random.<attr>(...)`` call is forbidden.
"""
from __future__ import annotations

import ast
import glob
import os

import pytest

PACKAGE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, "src", "musicgen")
)


def _bare_random_calls(source: str):
    """Return ``random.<attr>(...)`` Call nodes excluding the ``random.Random``
    constructor (matches the helper in tests/test_sampler.py lines 165-180).
    """
    tree = ast.parse(source)
    hits = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "random"
            and node.func.attr != "Random"
        ):
            hits.append(node)
    return hits


def _collect_package_modules():
    """List every *.py under src/musicgen/ (recursive; excludes __init__.py)."""
    return sorted(
        p for p in glob.glob(os.path.join(PACKAGE_DIR, "**", "*.py"), recursive=True)
        if not p.endswith("__init__.py")
    )


@pytest.mark.parametrize("path", _collect_package_modules())
def test_no_bare_random_in_package_module(path):
    """Every module under src/musicgen/ must use the injected rng (D-17).

    Failing this test means a ``random.choice``, ``random.random``,
    ``random.randint``, ``random.choices``, or ``random.uniform`` call
    slipped into the package. Use ``rng.<method>(...)`` with the injected
    ``rng: random.Random`` parameter instead.
    """
    with open(path, "r") as f:
        source = f.read()
    hits = _bare_random_calls(source)
    assert hits == [], (
        f"{os.path.basename(path)}: {len(hits)} bare random.<method>() at lines "
        f"{[n.lineno for n in hits]} — use rng.<method>() per D-17."
    )


def test_package_scan_covers_all_phase4_modules():
    """Meta-test: the scan collects at least sampler + generators/ + the 4 Phase 4 modules.

    Catches the case where PACKAGE_DIR mis-resolves and the parametrize returns
    an empty list (which would trivially 'pass' the bare-random test above).
    """
    modules = _collect_package_modules()
    relative = [os.path.relpath(m, PACKAGE_DIR) for m in modules]
    # Must cover all 4 Phase 4 new modules + Phase 3 sampler + generators
    expected_present = {
        "sampler.py", "renderer.py", "mixer.py", "annotator.py", "beats.py",
        "duration_validator.py",
        os.path.join("generators", "beat.py"),
        os.path.join("generators", "chord.py"),
        os.path.join("generators", "melody.py"),
        os.path.join("generators", "bassline.py"),
    }
    missing = expected_present - set(relative)
    assert not missing, f"package scan missed expected modules: {missing}"
```

This file does NOT need a `pytest.skip(..., allow_module_level=True)` — it's a real test from the moment Plan 04-05 executes. The parametrize approach auto-discovers future modules.

Do NOT keep any Wave 0 skip stub content in this file — replace entirely.
  </action>
  <verify>
    <automated>test ! -f beat_anotator.py && echo 'beat_anotator.py deleted OK' && python -m pytest tests/test_no_bare_random_in_package.py -v 2>&1 | tail -30</automated>
  </verify>
  <acceptance_criteria>
    - `test -f beat_anotator.py` exits NON-zero (file is gone)
    - `git status --short | grep "beat_anotator"` shows deletion staging (if using git rm) — the file is tracked as deleted
    - `grep -r "import beat_anotator\|from beat_anotator" . --include="*.py" 2>/dev/null` returns no lines (no stale references)
    - `pytest tests/test_no_bare_random_in_package.py -v` — passes with parametrize count >= 10 (at least the 10 expected modules from the meta-test)
    - `pytest tests/test_no_bare_random_in_package.py::test_package_scan_covers_all_phase4_modules -v` — passes (PACKAGE_DIR resolves correctly)
    - `pytest tests/test_no_bare_random_in_package.py -v` runs parametrized over `test_no_bare_random_in_package_module[<path>]` — each invocation is PASSED (no bare random.* anywhere under src/musicgen/)
    - `grep -c "ast.Attribute\|ast.Call\|ast.walk\|ast.Name" tests/test_no_bare_random_in_package.py` returns at least 4 matches (AST walk implemented)
    - `grep "random.Random" tests/test_no_bare_random_in_package.py` returns at least 2 matches (constructor exclusion documented)
    - Full suite: `pytest tests/ -m "not slow" -q` exits 0 (no regressions from beat_anotator deletion or guard addition)
  </acceptance_criteria>
  <done>`beat_anotator.py` deleted (no importers existed per RESEARCH Pitfall 7); `tests/test_no_bare_random_in_package.py` implements the parametrized-over-glob package-wide AST guard that discovers every `src/musicgen/**/*.py` file dynamically plus a meta-test asserting scan coverage. All parametrized test cases pass (zero bare `random.*` across the full package after Phase 4 work).</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| Filesystem → `json.load(_cfg.inst_probabilities_file)` / `json.load(_cfg.levels_file)` | Inline JSON loading inside the orchestrator (caller-trusted paths from cfg) |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-04-05-01 | Tampering | Orchestrator-collapse introduces a regression in create_song chain | mitigate | Smoke test `python music_gen.py` must reach the render+mix stage (env-only failure acceptable per ROADMAP Phase 3 closure precedent) |
| T-04-05-02 | Tampering | Deleting 9 functions breaks an implicit caller outside the refactored scope | mitigate | `pytest tests/test_music_gen_logging.py` regression suite catches import side-effects + logger contract; full-suite run catches any remaining indirect callers |
| T-04-05-03 | Integrity | Silent reintroduction of `random.<method>()` calls in a future edit | mitigate | `tests/test_no_bare_random_in_package.py` is parametrized over the full package, auto-extends to new modules, and runs as part of the default test suite (not a slow/opt-in marker) |
</threat_model>

<verification>
After both tasks complete:

1. `wc -l music_gen.py` — outputs < 200
2. `python -c "import music_gen"` — exits 0 (R-S1 importability preserved)
3. `test ! -f beat_anotator.py` — exits 0 (file is gone)
4. `pytest tests/test_no_bare_random_in_package.py -v` — all parametrize cases pass; meta-test passes
5. `pytest tests/test_music_gen_logging.py -v` — all 4 regression tests pass
6. `pytest tests/ -m "not slow" -q` — full suite green
7. Manual (optional) — `python music_gen.py` — should reach at least the "Mixing part" or "soundfont" log line before failing on env (no FluidSynth / empty sf/beat/). Failure BEFORE that point is a Phase 4 regression.
</verification>

<success_criteria>
- `music_gen.py` < 200 lines (D-24)
- 9 target audio-side functions deleted (D-23)
- `beat_anotator.py` deleted (D-03)
- New orchestrator chains renderer → mixer → beats → annotator
- `generate_song_parts` returns `chord_progressions` as a 6th value (fixes RESEARCH Open Question #2)
- `tests/test_no_bare_random_in_package.py` parametrized over every `src/musicgen/**/*.py`; all cases pass (D-17/D-31)
- `tests/test_music_gen_logging.py` continues passing (zero print, __main__ guard, module-level logger)
- Full suite green; no regressions
</success_criteria>

<output>
After completion, create `.planning/phases/04-renderer-mixer-annotator-beats-extraction/04-05-SUMMARY.md`.

Include:
- Final line count of music_gen.py (target: < 200)
- Diff summary: 9 deleted functions, N modified imports, create_song rewrite shape
- Confirmation of beat_anotator.py deletion (git status + test ! -f output)
- Output of `pytest tests/test_no_bare_random_in_package.py -v` (parametrize count + meta-test pass)
- Output of `pytest tests/test_music_gen_logging.py -v` (regression tests pass)
- Manual smoke result (`python music_gen.py` — how far it got before env failure)
- Full suite run tail
</output>
