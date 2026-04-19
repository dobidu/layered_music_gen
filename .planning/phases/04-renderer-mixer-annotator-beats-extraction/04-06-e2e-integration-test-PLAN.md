---
phase: 04-renderer-mixer-annotator-beats-extraction
plan: 06
type: execute
wave: 6
depends_on: ["04-00", "04-01", "04-02", "04-03", "04-04", "04-05"]
files_modified:
  - tests/test_integration_full_generation.py
autonomous: true
requirements: [R-X8]
tags: [phase-4, integration-test, e2e, slow-marker, fluidsynth-guard]

must_haves:
  truths:
    - "`tests/test_integration_full_generation.py` contains at least one `@pytest.mark.slow` test function covered by R-X8"
    - "Test module-level guard: `shutil.which(\"fluidsynth\") is None` → SKIPS (not fails), so CI without the binary still produces a clean report"
    - "Running `pytest tests/ -m \"not slow\" -q` does NOT include this test (confirmed by module-level `@pytest.mark.slow`)"
    - "Running `pytest -m slow tests/test_integration_full_generation.py` on a machine WITH fluidsynth + sf2 files exercises: sampler → generators → renderer → mixer → beats → annotator end-to-end"
    - "After running the slow test: 4 stems + 1 mix + 4 MIDI files exist on disk at the expected paths"
    - "Annotation dict from `annotator.annotate(...)` has all Phase-4 fillable fields non-None and Phase-5 TBD fields present as None"
    - "Test uses seeded `random.Random(42)` so the MIDI-level output is reproducible within the slow run"
    - "Baseline 371 tests + Plan 04-01..04-05 tests (~60 new) + this plan's integration test (counted in slow suite only) all green"
  artifacts:
    - path: "tests/test_integration_full_generation.py"
      provides: "One @pytest.mark.slow E2E test with shutil.which guard"
      contains: "@pytest.mark.slow"
      contains_2: "shutil.which"
  key_links:
    - from: "tests/test_integration_full_generation.py"
      to: "music_gen.create_song"
      via: "full pipeline invocation"
      pattern: "create_song\\("
    - from: "tests/test_integration_full_generation.py"
      to: "pyproject.toml [tool.pytest.ini_options].markers"
      via: "@pytest.mark.slow declared"
      pattern: "@pytest.mark.slow"
---

<objective>
Implement `tests/test_integration_full_generation.py` (R-X8) — one `@pytest.mark.slow` end-to-end test that builds a minimal 1-part 4/4 song and threads it through the full Phase 4 pipeline: sampler → generators → renderer (real FluidSynth subprocess) → mixer (real pedalboard + pydub) → beats (real mido) → annotator. Asserts the expected artifact layout (4 stems + 1 mix + 4 MIDIs) and the annotator dict shape (all Phase-4 fields non-None + all Phase-5 TBD fields None).

Gate on `shutil.which("fluidsynth")` — skip the test cleanly when the binary is absent (RESEARCH: the dev machine does NOT have fluidsynth on PATH; CI will not either; only the developer-workstation slow run exercises it). Also gate on the presence of `.sf2` files in the four `sf/<layer>/` directories — skip if any layer is empty (RESEARCH verified the current dev layout has some empty layers from the Phase 3 close).

Purpose: This test is the phase-gate proof that the full audio pipeline works end-to-end under real conditions. It is the first `@pytest.mark.slow` test in the suite (no analog exists per PATTERNS.md). The markers in Wave 0 made this possible without warnings. Because the test is marked `slow` and skipped on CI, it runs only when the developer explicitly opts in — it is the "acceptance gate" for Phase 4 closure.

Output: `tests/test_integration_full_generation.py` (replaces the Wave 0 stub).
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
@.planning/phases/04-renderer-mixer-annotator-beats-extraction/04-05-SUMMARY.md
@music_gen.py

<interfaces>
<!-- music_gen.create_song signature (post Plan 04-05 collapse) -->
def create_song(
    key: str, tempo: int,
    song_signatures: Dict[str, str], measures: Dict[str, int],
    name: str, chord_pat_file: str, swing_amount: float,
    cfg: config.Config = None,
) -> Dict:  # Returns the annotator dict.

<!-- music_gen._rng — module-level Random the create_song call uses.
     For reproducibility across two runs, the test must seed it BEFORE calling create_song:
         import music_gen
         music_gen._rng.seed(42)
     This is the Phase 4 approach; Phase 5 R-P7 introduces derive_sample_seed per-sample. -->

<!-- Phase 4 module surfaces (all loaded) -->
musicgen.renderer: FLUIDSYNTH_VERSION, RenderResult, pick_soundfonts, render_stems
musicgen.mixer:    MixResult, build_fx_boards, compute_layer_mask, mix_part, concat_parts, apply_fx_to_layer, pedalboard_info_json
musicgen.beats:    beat_duration, extract_beat_times, extract_downbeat_times
musicgen.annotator: annotate

<!-- Skip gate patterns -->
import shutil
fluidsynth_available = shutil.which("fluidsynth") is not None

@pytest.mark.slow
@pytest.mark.skipif(not fluidsynth_available, reason="fluidsynth binary not on PATH")
class TestFullGenerationPipeline:
    ...

<!-- RESEARCH Environment Availability (lines 747-763): fluidsynth is NOT on PATH on the dev
     machine and will not be on CI. sf/ directories have some empty layers (Phase 3 smoke
     confirmed "IndexError: empty sequence" in get_random_sound_font). Test must therefore
     also guard on sf2 availability. -->

<!-- pyproject.toml markers (Plan 04-00) — already added:
     [tool.pytest.ini_options].markers = [
         "slow: FluidSynth rendering tests — requires fluidsynth binary ...",
         "integration: end-to-end tests requiring system dependencies",
     ]  -->
</interfaces>
</context>

<tasks>

<task type="auto">
  <name>Task 1: Populate tests/test_integration_full_generation.py with @pytest.mark.slow E2E test</name>
  <files>tests/test_integration_full_generation.py</files>
  <read_first>
    - tests/test_integration_full_generation.py (Wave 0 stub — replaced entirely)
    - music_gen.py (post Plan 04-05 — final create_song signature + _rng module-level attribute)
    - src/musicgen/sampler.py (SongParams.sample classmethod for seeded parameter generation)
    - config.py (Config class and load() classmethod)
    - .planning/phases/04-renderer-mixer-annotator-beats-extraction/04-PATTERNS.md §"tests/test_integration_full_generation.py" (scaffold and marker pattern)
    - .planning/phases/04-renderer-mixer-annotator-beats-extraction/04-RESEARCH.md §Environment Availability (what skip conditions to check)
    - pyproject.toml (confirm Plan 04-00 added slow + integration markers)
  </read_first>
  <action>
Replace `tests/test_integration_full_generation.py` entirely (delete Wave 0 skip stub). File content:

```python
"""Integration test (R-X8): full Phase 4 pipeline end-to-end with real FluidSynth binary.

@pytest.mark.slow — ONLY runs when pytest is invoked with ``-m slow`` (or no
marker filter). CI's default ``pytest -m "not slow"`` skips this test.

Skip conditions (ANY of these triggers module-level skip):
  1. ``fluidsynth`` binary not on PATH (RESEARCH Environment Availability #2).
  2. Any ``sf/<layer>/`` dir is empty — ``pick_soundfonts`` will raise
     ``FileNotFoundError`` otherwise, which is an environment issue not a
     code regression (same as the Phase 3 closure smoke test).

Pipeline exercised: sampler.SongParams.sample (via music_gen._rng) →
generate_song_parts (real MIDI writes) → renderer.render_stems (real FluidSynth
subprocess) → mixer.mix_part + concat_parts (real pedalboard + pydub) →
beats.extract_beat_times + extract_downbeat_times (real mido) →
musicality_score.get_musicality_score → annotator.annotate → json.dump.

Assertions after the pipeline runs:
  - 4 stem WAVs + 1 mix WAV + 4 MIDI files exist on disk at the expected paths.
  - Annotation dict has all Phase-4 fill fields non-None (D-15).
  - Annotation dict has Phase-5 TBD fields as None (D-16).
  - MIDI files are bit-identical across two runs with the same seed (WAV golden
    test is Phase 5's scope; Phase 4 only asserts MIDI reproducibility).
"""
from __future__ import annotations

import json
import os
import random
import shutil
from pathlib import Path

import pytest

# ---------- Skip gates ----------

fluidsynth_available = shutil.which("fluidsynth") is not None


def _all_sf2_layers_have_files() -> bool:
    """Return True iff every layer (beat/melody/harmony/bassline) has at least one .sf2."""
    try:
        import config as _cfg_mod
        _cfg = _cfg_mod.Config()
        for layer in ("beat", "melody", "harmony", "bassline"):
            sf_dir = _cfg.sf_layer_dir(layer)
            if not os.path.isdir(sf_dir):
                return False
            files = [f for f in os.listdir(sf_dir) if f.endswith(".sf2")]
            if not files:
                return False
        return True
    except Exception:
        return False


sf2_pool_ready = _all_sf2_layers_have_files()


pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not fluidsynth_available, reason="fluidsynth binary not on PATH"),
    pytest.mark.skipif(not sf2_pool_ready, reason="one or more sf/<layer>/ dirs is empty (no .sf2)"),
]


# ---------- The E2E test ----------

class TestFullGenerationPipeline:
    def test_one_part_full_pipeline(self, tmp_path, monkeypatch):
        """One-part smoke test: seeded create_song produces all artifacts + valid annotation.

        Uses the real FluidSynth binary, real pedalboard FX, real pydub overlay,
        real mido tick extraction. Runs inside tmp_path (monkeypatch.chdir) so
        the per-song directory (created by generate_* and create_song) is
        isolated from the repo root.
        """
        monkeypatch.chdir(tmp_path)

        import music_gen

        # Seed the module-level RNG for reproducibility. Phase 5 R-P7 will
        # replace this with derive_sample_seed; here we just seed directly.
        music_gen._rng.seed(42)

        # Minimal single-part 4/4 SongParams. Build by hand (not SongParams.sample)
        # so we can bound the test to exactly one part + 1-2 measures (keeps the
        # FluidSynth render time to a few seconds).
        song_name = "intgen"
        signatures = {"intro": "4/4"}
        measures = {"intro": 2}

        annotation = music_gen.create_song(
            key="C",
            tempo=120,
            song_signatures=signatures,
            measures=measures,
            name=song_name,
            chord_pat_file="chord_patterns.txt",  # Repo-root file (carried by pyproject pythonpath)
            swing_amount=0.5,
        )

        # ---- Artifact layout assertions ----
        song_dir = Path(tmp_path) / song_name
        assert song_dir.is_dir(), f"expected song dir at {song_dir}"

        # Mix WAV exists
        mix_wav = song_dir / f"{song_name}.wav"
        assert mix_wav.is_file(), f"mix WAV missing at {mix_wav}"
        assert mix_wav.stat().st_size > 0

        # Annotation JSON exists
        annotation_json = song_dir / f"{song_name}.json"
        assert annotation_json.is_file(), f"annotation JSON missing at {annotation_json}"

        # Per-part subdir contains 4 stems (either _fx.wav for overlay-on layers
        # or _silent.wav for overlay-off layers; both count toward the 4-stem
        # total since D-12 writes a silent stub for masked-off layers).
        part_subdir = song_dir / f"{song_name}-intro"
        stem_files = list(part_subdir.rglob("*.wav"))
        # At least 4 stems (post-FX or silent) — the actual count may be higher
        # because renderer first writes raw stems, then apply_fx_to_layer writes
        # _fx.wav variants. We assert >= 4 stems exist.
        assert len(stem_files) >= 4, (
            f"expected >= 4 stem WAVs in {part_subdir}, found {len(stem_files)}: "
            f"{[s.name for s in stem_files]}"
        )

        # MIDI files: 4 per part (beat, melody, harmony, bassline)
        midi_files = list(part_subdir.rglob("*.mid"))
        assert len(midi_files) >= 4, (
            f"expected >= 4 MIDI files in {part_subdir}, found {len(midi_files)}: "
            f"{[m.name for m in midi_files]}"
        )

        # ---- Annotation dict shape assertions (D-15) ----
        assert isinstance(annotation, dict)
        phase4_fields = [
            "key", "mode", "tempo_bpm", "time_signature", "time_signatures_per_part",
            "measures_per_part", "swing", "song_arrangement", "chord_progression",
            "active_layers", "soundfonts", "fx_params", "beat_times", "downbeat_times",
            "musicality_score", "duration_seconds", "fluidsynth_version",
            "mix", "stems", "midi",
        ]
        for field in phase4_fields:
            assert field in annotation, f"R-P4 field {field!r} missing from annotator output"
            assert annotation[field] is not None, f"R-P4 field {field!r} is None (Phase 4 must fill)"

        # ---- Phase 5 TBD fields present as None (D-16) ----
        for tbd in ("seed", "musicgen_version", "split", "pre_roll_offset_seconds"):
            assert tbd in annotation, f"Phase-5 TBD field {tbd!r} missing (D-16: must be present as None)"
            assert annotation[tbd] is None, f"Phase-5 TBD field {tbd!r} should be None, got {annotation[tbd]!r}"

        # analysis_failed should be OMITTED on success (D-16 clarification)
        assert "analysis_failed" not in annotation, (
            "analysis_failed should be OMITTED on success, not set to False"
        )

        # ---- Per-part shape assertions ----
        assert annotation["key"] == "C"
        assert annotation["tempo_bpm"] == 120
        assert annotation["time_signature"] == "4/4"
        assert annotation["swing"] == 0.5
        assert annotation["mode"] == "major"  # "C" is major

        # song_arrangement is a list of {part, start_seconds, end_seconds} dicts
        assert isinstance(annotation["song_arrangement"], list)
        assert len(annotation["song_arrangement"]) >= 1
        for entry in annotation["song_arrangement"]:
            assert set(entry.keys()) == {"part", "start_seconds", "end_seconds"}

        # beat_times and downbeat_times are dicts keyed by part
        assert "intro" in annotation["beat_times"]
        assert "intro" in annotation["downbeat_times"]
        # 2 measures of 4/4 → 2 downbeats
        assert len(annotation["downbeat_times"]["intro"]) == 2

        # JSON is valid + round-trips
        with open(annotation_json) as f:
            loaded = json.load(f)
        assert loaded["key"] == "C"
        assert loaded["tempo_bpm"] == 120


class TestMidiReproducibility:
    """MIDI bit-identity under the same seed (WAV identity is Phase 5's golden test)."""

    def test_same_seed_produces_same_midi(self, tmp_path, monkeypatch):
        """Two runs with the same seed produce bit-identical beat/melody/harmony/bassline MIDI."""
        monkeypatch.chdir(tmp_path)
        import music_gen

        def _run(name: str, seed: int):
            music_gen._rng.seed(seed)
            music_gen.create_song(
                key="C", tempo=120,
                song_signatures={"intro": "4/4"}, measures={"intro": 2},
                name=name, chord_pat_file="chord_patterns.txt",
                swing_amount=0.5,
            )
            part_dir = Path(tmp_path) / name / f"{name}-intro"
            return {
                layer: (part_dir / f"{name}-intro-{layer}.mid").read_bytes()
                for layer in ("beat", "melody", "harmony", "bassline")
            }

        # Two runs with same seed; MIDI bytes must be identical.
        a = _run("rep1", seed=42)
        b = _run("rep2", seed=42)
        for layer in ("beat", "melody", "harmony", "bassline"):
            assert a[layer] == b[layer], (
                f"MIDI reproducibility broken for layer {layer!r}: "
                f"seed=42 run1 ({len(a[layer])} bytes) != run2 ({len(b[layer])} bytes)"
            )
```

Notes specific to this task:

- **Module-level `pytestmark` list** applies ALL three decorators (slow + skipif-no-fluidsynth + skipif-no-sf2) to every test in the module. This is cleaner than stacking them on each class.
- **`_all_sf2_layers_have_files`** is computed at module-import time (not inside the test), so the skip decision is final and visible in pytest's collection output.
- **`music_gen._rng.seed(42)`** — after Plan 04-05, `music_gen` has `_rng = random.Random()` at module level; seeding it before each `create_song` call gives reproducibility within Phase 4 without needing Phase 5's `derive_sample_seed`.
- **`chord_patterns.txt` at repo root** — Phase 3 Plan 03-01's `pyproject.toml pythonpath = ["."]` keeps the repo root on `sys.path`, but the test file path usage of `chord_patterns.txt` is RELATIVE to the test's working directory (tmp_path after monkeypatch.chdir). Because `chord_patterns.txt` lives at repo root, the test must pass an ABSOLUTE path OR fall back on the config default. Check `create_song`'s signature — if it accepts `cfg.chord_patterns_file` via the config, we can pass `cfg=config.Config()` and drop the `chord_pat_file` literal. If not, thread the absolute path via `str(Path(__file__).resolve().parent.parent / "chord_patterns.txt")`.

Actually, per the post-Plan 04-05 `create_song` shape, `chord_pat_file` is still a positional param (extracted from the original signature unchanged). The cleanest fix: pass `str(Path(__file__).resolve().parent.parent / "chord_patterns.txt")` as the `chord_pat_file` argument. Update the test body accordingly.

Actual fix (replace the `chord_pat_file="chord_patterns.txt"` line):

```python
chord_pat_file=str(Path(__file__).resolve().parent.parent / "chord_patterns.txt"),
```

The second test class `TestMidiReproducibility` exists because: (a) MIDI bit-identity is the strongest reproducibility assertion that Phase 4 can make (WAV identity depends on FluidSynth binary version — Phase 5 golden test), and (b) it exercises the seed-threading contract end-to-end (catching any regression where `_rng.seed(42)` is not honored by the extracted modules).

Do NOT retain the Wave 0 stub.
  </action>
  <verify>
    <automated>python -m pytest tests/test_integration_full_generation.py -v 2>&1 | tail -30 && python -m pytest tests/ -m "not slow" -q 2>&1 | tail -5</automated>
  </verify>
  <acceptance_criteria>
    - File `tests/test_integration_full_generation.py` exists with real content (no `pytest.skip(allow_module_level=True)` Wave 0 stub)
    - `grep "@pytest.mark.slow\|pytest.mark.slow" tests/test_integration_full_generation.py` returns at least 1 match
    - `grep "shutil.which" tests/test_integration_full_generation.py` returns at least 1 match (fluidsynth binary guard)
    - `grep -c "def test_" tests/test_integration_full_generation.py` returns at least 2 (TestFullGenerationPipeline + TestMidiReproducibility)
    - `grep "phase4_fields\|Phase-4 fields" tests/test_integration_full_generation.py` returns at least 1 match (R-P4 field checklist applied)
    - `grep "analysis_failed" tests/test_integration_full_generation.py` returns at least 1 match (D-16 clarification asserted)
    - Running `pytest tests/ -m "not slow" -q` does NOT include the integration test in the run count (the module skips at collection per the pytestmark decorator)
    - `pytest tests/test_integration_full_generation.py -v` on a machine WITHOUT fluidsynth: produces "SKIPPED" results with clear reason strings ("fluidsynth binary not on PATH" or "sf/<layer>/ dirs empty")
    - `pytest tests/test_integration_full_generation.py -v -m slow` on a machine WITH fluidsynth + full sf2 pool: passes both test classes
    - Full suite: `pytest tests/ -m "not slow" -q` exits 0 (371 baseline + Plan 04-01..04-05 tests + this plan's skipped integration test)
  </acceptance_criteria>
  <done>`tests/test_integration_full_generation.py` contains 2 test classes (TestFullGenerationPipeline + TestMidiReproducibility), both gated on `@pytest.mark.slow` + `shutil.which("fluidsynth")` + sf2-pool readiness. When skipped, provides clear reason strings. When run with `-m slow` on a machine with FluidSynth + full sf2 pool, exercises the full pipeline and asserts 4 stems + 1 mix + 4 MIDIs exist, annotation dict is R-P4 shaped (Phase-4 fields non-None + Phase-5 TBD fields None), and same-seed MIDI is bit-identical across two runs.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| Python → FluidSynth subprocess | `FluidSynth(sf).midi_to_audio(...)` calls the real binary when not skipped |
| Filesystem → sf/ directories | `pick_soundfonts` iterates `.sf2` files |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-04-06-01 | Tampering | Integration test runs on CI and hangs due to missing FluidSynth | mitigate | `shutil.which("fluidsynth")` module-level skip gate; test SKIPs (not fails) when binary absent. Safe default for CI. |
| T-04-06-02 | Information Disclosure | Test outputs (stems, mix) written to filesystem | accept | Written inside pytest's `tmp_path` which is auto-cleaned at test end |
| T-04-06-03 | Denial of Service | Slow test takes >60s to run (FluidSynth render can be slow) | accept | Test is `@pytest.mark.slow` and opt-in; developer knows about slow runs. 1-part 2-measure render is < 10s on a typical dev machine |
</threat_model>

<verification>
After task 1 completes:

1. `pytest tests/test_integration_full_generation.py -v` — on the dev machine (no fluidsynth): produces SKIPPED results with readable reason
2. `pytest tests/ -m "not slow" -q` — full suite green; integration test skipped from collection
3. `pytest --collect-only tests/test_integration_full_generation.py -m slow` — shows 2 test items collected (TestFullGenerationPipeline::test_one_part_full_pipeline + TestMidiReproducibility::test_same_seed_produces_same_midi)
4. Manual (on a machine with fluidsynth + full sf2 pool): `pytest tests/test_integration_full_generation.py -v -m slow` — both tests pass in < 60s total

**Phase gate verification** (run by the developer after Plan 04-06 closes, completing Phase 4):

- `pytest tests/ -m "not slow" -q` — ~430 tests pass (371 baseline + ~60 Phase 4 new)
- `python music_gen.py` smoke — reaches the annotator stage before any env failure
- `wc -l music_gen.py` — under 200 lines
- `grep -r "beat_anotator" . --include="*.py"` — returns no lines (file and all refs gone)
- `pytest tests/test_no_bare_random_in_package.py -v` — all parametrized cases pass
</verification>

<success_criteria>
- `tests/test_integration_full_generation.py` contains at least one `@pytest.mark.slow` end-to-end test (R-X8)
- Module-level guards (fluidsynth + sf2 pool) skip cleanly when env is incomplete
- On a machine with FluidSynth + full sf2 pool, the test passes end-to-end
- Assertions cover: 4 stems + 1 mix + 4 MIDIs exist; annotation dict R-P4 shaped; Phase-5 TBD as None; MIDI reproducibility under same seed
- Default `pytest -m "not slow"` still excludes this test
- Full suite (non-slow) green
</success_criteria>

<output>
After completion, create `.planning/phases/04-renderer-mixer-annotator-beats-extraction/04-06-SUMMARY.md`.

Include:
- Output of `pytest tests/test_integration_full_generation.py -v` (expected SKIPPED on dev machine without fluidsynth; document the exact reason strings)
- Output of `pytest --collect-only tests/test_integration_full_generation.py -m slow` (shows 2 tests collected)
- Output of `pytest tests/ -m "not slow" -q` tail (full suite green, integration test excluded)
- If run on a machine with FluidSynth: the full-suite-with-slow output showing PASSED
- Phase gate summary: total passing tests, `wc -l music_gen.py`, AST guard results, `beat_anotator.py` absence
- Any deviations (Rule 1/2 fixes)

**This SUMMARY closes Phase 4.** After writing it, the developer should update `.planning/STATE.md` with the Phase 4 close and prepare for Phase 5 (Productize I).
</output>
