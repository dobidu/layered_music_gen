"""Tests for src/musicgen/api.py (D-40, R-P12 single-sample).

Split into TestApiFast (no FluidSynth, mocks the pipeline) and TestApiSlow
(real pipeline under @pytest.mark.slow with FluidSynth + sf2 guards).
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest


# ---------------- Skip gates for slow cases (reused from integration test) ----------------

fluidsynth_available = shutil.which("fluidsynth") is not None


def _all_sf2_layers_have_files() -> bool:
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


# ---------------- TestApiFast — no FluidSynth, pipeline mocked ----------------


class TestApiFast:
    """Fast tests — validation + resume short-circuit only."""

    def test_config_global_seed_required(self, tmp_path):
        """D-21: global_seed=None raises ValueError at api.generate."""
        from musicgen import Config, generate
        cfg = Config(global_seed=None, sample_index=0, dataset_root=str(tmp_path))
        with pytest.raises(ValueError, match="global_seed"):
            generate(cfg)

    def test_negative_sample_index_rejected(self, tmp_path):
        """sample_index < 0 raises ValueError."""
        from musicgen import Config, generate
        cfg = Config(global_seed=1, sample_index=-1, dataset_root=str(tmp_path))
        with pytest.raises(ValueError, match="sample_index"):
            generate(cfg)

    def test_generate_resume_short_circuits(self, tmp_path, monkeypatch):
        """Pre-create sample.json → generate returns without running pipeline."""
        from musicgen import Config, generate
        # Pre-write the sentinel.
        sample_dir = tmp_path / "000000"
        sample_dir.mkdir()
        (sample_dir / "sample.json").write_text(json.dumps({
            "seed": 12345,
            "split": "train",
            "musicality_score": {"score": 0.5, "components": {}},
            "duration_seconds": 10.0,
        }))

        # Poison the renderer — if the pipeline runs, this raises.
        def _poison(*args, **kwargs):
            raise RuntimeError("pipeline must not run on resume short-circuit")
        monkeypatch.setattr("musicgen.api.renderer.render_stems", _poison)

        cfg = Config(global_seed=1, sample_index=0, dataset_root=str(tmp_path))
        result = generate(cfg)
        assert result.status == "ok"
        assert result.sample_index == 0
        assert result.split == "train"

    def test_reconstruct_from_sample_json(self, tmp_path, monkeypatch):
        """Short-circuit populates SampleResult fields from sample.json."""
        from musicgen import Config, generate
        sample_dir = tmp_path / "000042"
        sample_dir.mkdir()
        (sample_dir / "sample.json").write_text(json.dumps({
            "seed": 999999,
            "split": "valid",
            "musicality_score": {"score": 0.91, "components": {"rhythm": 0.88}},
            "duration_seconds": 47.3,
        }))

        def _poison(*args, **kwargs):
            raise RuntimeError("must not run")
        monkeypatch.setattr("musicgen.api.renderer.render_stems", _poison)

        cfg = Config(global_seed=1, sample_index=42, dataset_root=str(tmp_path))
        result = generate(cfg)
        assert result.split == "valid"
        assert result.musicality_score == 0.91
        assert result.duration_seconds == 47.3
        assert result.sample_dir == str(sample_dir)

    def test_seed_field_is_sample_seed_not_global(self, tmp_path, monkeypatch):
        """D-22: SampleResult.seed is the derived sample_seed, not config.global_seed."""
        from musicgen import Config, generate
        from musicgen.seeds import derive_sample_seed
        sample_dir = tmp_path / "000000"
        sample_dir.mkdir()
        (sample_dir / "sample.json").write_text(json.dumps({
            "seed": derive_sample_seed(42, 0),
            "split": "train",
            "musicality_score": {"score": 0.0, "components": {}},
            "duration_seconds": 0.0,
        }))

        def _poison(*args, **kwargs):
            raise RuntimeError("must not run")
        monkeypatch.setattr("musicgen.api.renderer.render_stems", _poison)

        cfg = Config(global_seed=42, sample_index=0, dataset_root=str(tmp_path))
        result = generate(cfg)
        assert result.seed == derive_sample_seed(42, 0)
        assert result.seed != 42  # NOT the global seed

    def test_sample_result_shape(self):
        """D-02: SampleResult has the 11 documented fields, all frozen."""
        from musicgen import SampleResult
        import dataclasses
        fields = {f.name for f in dataclasses.fields(SampleResult)}
        assert fields == {
            "sample_index", "seed", "sample_dir", "sample_json_path",
            "mix_path", "stem_paths", "midi_paths", "split", "status",
            "musicality_score", "duration_seconds",
        }

    def test_public_exports(self):
        """D-35: the public surface resolves via top-level musicgen imports."""
        from musicgen import generate, Config, SampleResult, __version__
        assert generate.__name__ == "generate"
        assert Config.__name__ == "Config"
        assert SampleResult.__name__ == "SampleResult"
        assert isinstance(__version__, str)
        assert __version__  # non-empty


# ---------------- TestApiSlow — real pipeline, guarded ----------------


@pytest.mark.slow
@pytest.mark.skipif(
    not fluidsynth_available,
    reason="fluidsynth binary not on PATH — skipping slow api tests",
)
@pytest.mark.skipif(
    not sf2_pool_ready,
    reason="one or more sf/<layer>/ dirs is empty — skipping slow api tests",
)
class TestApiSlow:
    """Real pipeline — FluidSynth + sf2 required."""

    def test_generate_produces_layout(self, tmp_path):
        """End-to-end: the 10-file per-sample layout appears on disk."""
        from musicgen import Config, generate
        result = generate(Config(
            global_seed=1, sample_index=0, dataset_root=str(tmp_path),
        ))
        assert result.status == "ok"
        sample_dir = Path(result.sample_dir)
        assert (sample_dir / "mix.wav").is_file()
        assert (sample_dir / "sample.json").is_file()
        for layer in ("beat", "melody", "harmony", "bassline"):
            assert (sample_dir / "stems" / f"{layer}.wav").is_file()
            assert (sample_dir / "midi" / f"{layer}.mid").is_file()

    def test_sample_json_has_phase5_fields(self, tmp_path):
        """D-22: seed, musicgen_version, split are filled by api.generate."""
        from musicgen import Config, generate
        result = generate(Config(
            global_seed=1, sample_index=0, dataset_root=str(tmp_path),
        ))
        annotation = json.loads(Path(result.sample_json_path).read_text())
        assert annotation["seed"] == result.seed
        assert annotation["musicgen_version"] in ("0.1.0", "0.1.0+uninstalled")
        assert annotation["split"] in ("train", "valid", "test")
        # Phase-5 TBD (R-P9 Phase 6) — stays None:
        assert annotation.get("pre_roll_offset_seconds") is None

    def test_generate_twice_idempotent(self, tmp_path):
        """Second call short-circuits via sentinel — same paths, same seed."""
        from musicgen import Config, generate
        cfg = Config(global_seed=1, sample_index=0, dataset_root=str(tmp_path))
        r1 = generate(cfg)
        r2 = generate(cfg)
        assert r1.sample_index == r2.sample_index
        assert r1.seed == r2.seed
        assert r1.sample_dir == r2.sample_dir
        assert r1.split == r2.split

    def test_manifest_has_ok_entry(self, tmp_path):
        """Manifest append: one line, status=ok, correct sample_index."""
        from musicgen import Config, generate
        generate(Config(
            global_seed=1, sample_index=0, dataset_root=str(tmp_path),
        ))
        manifest_path = Path(tmp_path) / "manifest.jsonl"
        assert manifest_path.is_file()
        lines = manifest_path.read_text().splitlines()
        # At least one entry (the sample we just generated):
        assert len(lines) >= 1
        last_entry = json.loads(lines[-1])
        assert last_entry["status"] == "ok"
        assert last_entry["sample_index"] == 0
        assert last_entry["split"] in ("train", "valid", "test")
