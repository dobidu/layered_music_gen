"""Integration test (R-X8): full Phase 4+5 pipeline end-to-end with real FluidSynth binary.

@pytest.mark.slow — ONLY runs when pytest is invoked with ``-m slow`` (or no
marker filter on a machine with FluidSynth). CI's default
``pytest -m "not slow"`` skips this test.

Skip conditions (ALL apply at module level via pytestmark):
  1. ``fluidsynth`` binary not on PATH (RESEARCH Environment Availability #2).
  2. Any ``sf/<layer>/`` dir is empty — ``pick_soundfonts`` will raise
     ``FileNotFoundError`` otherwise, which is an environment issue not a
     code regression (same as the Phase 3 closure smoke test).

Pipeline exercised: api.generate (Plan 05-05) →
sampler.SongParams → generate_chord/melody/bassline/beat (real MIDI writes) →
renderer.render_stems (real FluidSynth subprocess) →
mixer.mix_part + concat_parts (real pedalboard + pydub) →
beats.extract_beat_times + extract_downbeat_times (real mido) →
musicgen.musicality.get_musicality_score → annotator.annotate →
writer.write_sample (atomic per-sample layout) → manifest append.

Assertions after the pipeline runs:
  - 4 stem WAVs + 1 mix WAV + 4 MIDI files exist on disk in the per-sample dir.
  - sample.json has all Phase-4 fill fields non-None (D-15).
  - sample.json has Phase-5 filled fields (seed, musicgen_version, split) non-None.
  - sample.json has pre_roll_offset_seconds as None (R-P9 Phase 6).
  - analysis_failed is OMITTED on success (D-16 clarification).
  - MIDI files are bit-identical across two runs with the same seed.
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
# TestMidiReproducibility calls musicgen.generate() which runs
# renderer.render_stems (FluidSynth subprocess) before MIDI paths are
# accessible, so it also needs the binary present.
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


_LAYERS = ("beat", "melody", "harmony", "bassline")


# ---------- The E2E test ----------

class TestFullGenerationPipeline:
    """Smoke test: seeded musicgen.generate produces all artifacts + valid sample.json."""

    def test_one_part_full_pipeline(self, tmp_path):
        """Seeded musicgen.generate produces all artifacts + valid sample.json.

        Uses the real FluidSynth binary, real pedalboard FX, real pydub overlay,
        real mido tick extraction. Writes under tmp_path (no chdir needed — the
        api.generate pipeline uses tempfile.mkdtemp for working dirs and the
        dataset_root param for final layout).
        """
        from musicgen import Config, generate

        result = generate(Config(
            global_seed=1, sample_index=0, dataset_root=str(tmp_path),
        ))

        # ---- Top-level status ----
        assert result.status == "ok", (
            f"expected status='ok', got {result.status!r}"
        )

        # ---- Artifact layout assertions (D-05 per-sample dir) ----
        sample_dir = Path(result.sample_dir)
        assert sample_dir.is_dir(), f"expected sample dir at {sample_dir}"

        mix_wav = sample_dir / "mix.wav"
        assert mix_wav.is_file(), f"mix WAV missing at {mix_wav}"
        assert mix_wav.stat().st_size > 0, "mix WAV is empty"

        sample_json_path = sample_dir / "sample.json"
        assert sample_json_path.is_file(), f"sample.json missing at {sample_json_path}"

        # 4 stems + 4 MIDIs in stems/ and midi/ subdirs.
        for layer in _LAYERS:
            stem_wav = sample_dir / "stems" / f"{layer}.wav"
            assert stem_wav.is_file(), f"stem WAV missing at {stem_wav}"
            assert stem_wav.stat().st_size > 0, f"stem WAV empty: {stem_wav}"

            midi_file = sample_dir / "midi" / f"{layer}.mid"
            assert midi_file.is_file(), f"MIDI missing at {midi_file}"
            assert midi_file.stat().st_size > 0, f"MIDI empty: {midi_file}"

        # ---- sample.json shape assertions ----
        annotation = json.loads(sample_json_path.read_text())
        assert isinstance(annotation, dict)

        # Phase-4 fill fields must be present + non-None (D-15).
        phase4_fields = [
            "key", "mode", "tempo_bpm", "time_signature", "time_signatures_per_part",
            "measures_per_part", "swing", "song_arrangement", "chord_progression",
            "active_layers", "soundfonts", "fx_params", "beat_times", "downbeat_times",
            "musicality_score", "duration_seconds", "fluidsynth_version",
            "mix", "stems", "midi",
        ]
        for field in phase4_fields:
            assert field in annotation, f"R-P4 field {field!r} missing from sample.json"
            assert annotation[field] is not None, (
                f"R-P4 field {field!r} is None (Phase 4 must fill)"
            )

        # Phase-5 FILLED fields (D-22 — api.generate populates these).
        assert annotation["seed"] == result.seed, (
            f"sample.json seed {annotation['seed']!r} != result.seed {result.seed!r}"
        )
        assert annotation["musicgen_version"] in ("0.1.0", "0.1.0+uninstalled")
        assert annotation["split"] in ("train", "valid", "test")

        # Phase-5 TBD field (pre_roll_offset_seconds) stays None per D-22 — R-P9 is Phase 6.
        assert annotation.get("pre_roll_offset_seconds") is None, (
            f"pre_roll_offset_seconds should be None (Phase 6 fills), "
            f"got {annotation.get('pre_roll_offset_seconds')!r}"
        )

        # D-16 clarification: analysis_failed is OMITTED on success, not set to False.
        assert "analysis_failed" not in annotation, (
            "analysis_failed should be OMITTED on success, not set to False"
        )

        # ---- Path rewrite sanity (D-11/D-12 — writer rewrites to relative paths) ----
        assert annotation["mix"] == "mix.wav"
        for layer in _LAYERS:
            assert annotation["stems"][layer] == f"stems/{layer}.wav"
            assert annotation["midi"][layer] == f"midi/{layer}.mid"

        # ---- Manifest append (D-13) ----
        manifest_path = Path(tmp_path) / "manifest.jsonl"
        assert manifest_path.is_file(), "manifest.jsonl missing"
        lines = manifest_path.read_text().splitlines()
        assert len(lines) >= 1, "manifest.jsonl has no entries"
        last_entry = json.loads(lines[-1])
        assert last_entry["status"] == "ok"
        assert last_entry["sample_index"] == 0
        assert last_entry["split"] == annotation["split"]


class TestMidiReproducibility:
    """MIDI bit-identity under the same seed (WAV identity is Phase 5 Plan 06's golden test)."""

    def test_same_seed_produces_same_midi(self, tmp_path):
        """Two runs with the same (global_seed, sample_index) produce bit-identical MIDI."""
        from musicgen import Config, generate

        def _run(seed: int, sample_index: int, root: Path):
            result = generate(Config(
                global_seed=seed, sample_index=sample_index, dataset_root=str(root),
            ))
            return {
                layer: Path(result.midi_paths[layer]).read_bytes()
                for layer in _LAYERS
            }

        # Two runs with the same seed in two distinct dataset_roots.
        # Both should produce bit-identical MIDI bytes (Phase 5 RNG hierarchy
        # guarantees determinism).
        a = _run(seed=42, sample_index=0, root=tmp_path / "a")
        b = _run(seed=42, sample_index=0, root=tmp_path / "b")
        for layer in _LAYERS:
            assert a[layer] == b[layer], (
                f"MIDI reproducibility broken for layer {layer!r}: "
                f"seed=42 run1 ({len(a[layer])} bytes) != run2 ({len(b[layer])} bytes)"
            )
