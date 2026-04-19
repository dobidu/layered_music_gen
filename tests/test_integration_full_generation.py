"""Integration test (R-X8): full Phase 4 pipeline end-to-end with real FluidSynth binary.

@pytest.mark.slow — ONLY runs when pytest is invoked with ``-m slow`` (or no
marker filter on a machine with FluidSynth). CI's default
``pytest -m "not slow"`` skips this test.

Skip conditions (ALL apply at module level via pytestmark):
  1. ``fluidsynth`` binary not on PATH (RESEARCH Environment Availability #2).
  2. Any ``sf/<layer>/`` dir is empty — ``pick_soundfonts`` will raise
     ``FileNotFoundError`` otherwise, which is an environment issue not a
     code regression (same as the Phase 3 closure smoke test).

Pipeline exercised: sampler.SongParams (via music_gen._rng) →
generate_song_parts (real MIDI writes) → renderer.render_stems (real FluidSynth
subprocess) → mixer.mix_part + concat_parts (real pedalboard + pydub) →
beats.extract_beat_times + extract_downbeat_times (real mido) →
musicality_score.get_musicality_score → annotator.annotate → json.dump.

Assertions after the pipeline runs:
  - 4 stem WAVs + 1 mix WAV + 4 MIDI files exist on disk at the expected paths.
  - Annotation dict has all Phase-4 fill fields non-None (D-15).
  - Annotation dict has Phase-5 TBD fields as None (D-16).
  - analysis_failed is OMITTED on success (D-16 clarification).
  - MIDI files are bit-identical across two runs with the same seed (WAV golden
    test is Phase 5's scope; Phase 4 only asserts MIDI reproducibility).
"""
from __future__ import annotations

import json
import os
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

# Module-level mark: both classes require FluidSynth + full sf2 pool.
# TestMidiReproducibility calls create_song() which runs renderer.render_stems
# (FluidSynth subprocess) before MIDI paths are accessible, so it also needs
# the binary present.
pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not fluidsynth_available,
        reason="fluidsynth binary not on PATH — skipping E2E integration test",
    ),
    pytest.mark.skipif(
        not sf2_pool_ready,
        reason="one or more sf/<layer>/ dirs is empty (no .sf2) — skipping E2E integration test",
    ),
]

# Absolute path to chord_patterns.txt at repo root — the test chdirs to tmp_path
# so a bare filename would not resolve. Path(__file__).parent.parent points to
# the repo root where chord_patterns.txt lives (alongside music_gen.py).
_REPO_ROOT = Path(__file__).resolve().parent.parent
_CHORD_PAT_FILE = str(_REPO_ROOT / "chord_patterns.txt")


# ---------- The E2E test ----------

class TestFullGenerationPipeline:
    """One-part smoke test: seeded create_song produces all artifacts + valid annotation."""

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
        # replace this with derive_sample_seed; here we seed directly.
        music_gen._rng.seed(42)

        # Minimal single-part 4/4 song: 1 part ("intro"), 2 measures.
        # Keeps the FluidSynth render time to a few seconds.
        song_name = "intgen"
        signatures = {"intro": "4/4"}
        measures = {"intro": 2}

        annotation = music_gen.create_song(
            key="C",
            tempo=120,
            song_signatures=signatures,
            measures=measures,
            name=song_name,
            chord_pat_file=_CHORD_PAT_FILE,
            swing_amount=0.5,
        )

        # ---- Artifact layout assertions ----
        song_dir = Path(tmp_path) / song_name
        assert song_dir.is_dir(), f"expected song dir at {song_dir}"

        # Mix WAV exists
        mix_wav = song_dir / f"{song_name}.wav"
        assert mix_wav.is_file(), f"mix WAV missing at {mix_wav}"
        assert mix_wav.stat().st_size > 0, "mix WAV is empty"

        # Annotation JSON exists
        annotation_json = song_dir / f"{song_name}.json"
        assert annotation_json.is_file(), f"annotation JSON missing at {annotation_json}"

        # Per-part subdir contains stems (post-FX .wav or _silent.wav for masked layers).
        # create_song writes out_dir = os.path.join(name, f"{name}-{part}").
        part_subdir = song_dir / f"{song_name}-intro"
        assert part_subdir.is_dir(), f"expected part subdir at {part_subdir}"

        stem_files = list(part_subdir.rglob("*.wav"))
        # At least 4 stems (post-FX or silent); renderer also writes raw stems first.
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
            assert annotation[field] is not None, (
                f"R-P4 field {field!r} is None (Phase 4 must fill)"
            )

        # ---- Phase 5 TBD fields present as None (D-16) ----
        for tbd in ("seed", "musicgen_version", "split", "pre_roll_offset_seconds"):
            assert tbd in annotation, (
                f"Phase-5 TBD field {tbd!r} missing (D-16: must be present as None)"
            )
            assert annotation[tbd] is None, (
                f"Phase-5 TBD field {tbd!r} should be None, got {annotation[tbd]!r}"
            )

        # D-16 clarification: analysis_failed is OMITTED on success, not set to False.
        assert "analysis_failed" not in annotation, (
            "analysis_failed should be OMITTED on success, not set to False"
        )

        # ---- Per-part value assertions ----
        assert annotation["key"] == "C"
        assert annotation["tempo_bpm"] == 120
        assert annotation["time_signature"] == "4/4"
        assert annotation["swing"] == 0.5
        assert annotation["mode"] == "major"  # "C" has no trailing "m" → major

        # song_arrangement is a list of {part, start_seconds, end_seconds} dicts
        assert isinstance(annotation["song_arrangement"], list)
        assert len(annotation["song_arrangement"]) >= 1
        for entry in annotation["song_arrangement"]:
            assert set(entry.keys()) == {"part", "start_seconds", "end_seconds"}, (
                f"song_arrangement entry has unexpected keys: {set(entry.keys())}"
            )

        # beat_times and downbeat_times are dicts keyed by part
        assert "intro" in annotation["beat_times"], (
            "'intro' not in beat_times; keys: %s" % list(annotation["beat_times"].keys())
        )
        assert "intro" in annotation["downbeat_times"], (
            "'intro' not in downbeat_times; keys: %s" % list(annotation["downbeat_times"].keys())
        )
        # 2 measures of 4/4 → 2 downbeats (time-grid downbeat derivation, Plan 04-01)
        assert len(annotation["downbeat_times"]["intro"]) == 2, (
            f"expected 2 downbeats for 2 measures of 4/4, got "
            f"{len(annotation['downbeat_times']['intro'])}"
        )

        # JSON round-trip is valid
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
                name=name, chord_pat_file=_CHORD_PAT_FILE,
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
