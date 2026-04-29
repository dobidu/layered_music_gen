"""Integration test: 4-sample batch with 2 workers (R-Q2, D-68).

Guarded by @pytest.mark.slow + FluidSynth binary + sf2 pool checks.
Tests: generate 4 samples, verify manifest, verify resume, verify output_mode.
"""
from __future__ import annotations

import json
import os
import shutil

import pytest

from config import Config
from musicgen import BatchResult, generate_batch

pytestmark = pytest.mark.slow

_fluidsynth_missing = shutil.which("fluidsynth") is None
_sf2_missing = not any(
    fname.endswith(".sf2")
    for root, _, files in os.walk(os.path.join(
        os.path.dirname(__file__), "..", "sf"
    ))
    for fname in files
)


class TestBatchIntegration:
    @pytest.mark.skipif(_fluidsynth_missing, reason="FluidSynth not installed")
    @pytest.mark.skipif(_sf2_missing, reason="No .sf2 soundfonts in sf/")
    def test_4_sample_batch_2_workers(self, tmp_path):
        """Generate 4 samples with 2 workers; verify layout and manifest (R-Q2)."""
        cfg = Config(
            global_seed=42,
            count=4,
            dataset_root=str(tmp_path / "dataset"),
            workers=2,
        )
        result = generate_batch(cfg)

        assert isinstance(result, BatchResult)
        assert result.total == 4
        assert result.succeeded == 4
        assert result.failed == 0
        assert result.skipped == 0
        assert len(result.results) == 4
        assert result.duration_seconds > 0

        for idx in range(4):
            sample_dir = tmp_path / "dataset" / f"{idx:06d}"
            assert (sample_dir / "sample.json").exists(), f"Missing sentinel for {idx}"
            assert (sample_dir / "mix.wav").exists()
            for layer in ("beat", "melody", "harmony", "bassline"):
                assert (sample_dir / "stems" / f"{layer}.wav").exists()
                assert (sample_dir / "midi" / f"{layer}.mid").exists()

        manifest_path = tmp_path / "dataset" / "manifest.jsonl"
        assert manifest_path.exists()
        entries = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
        assert len(entries) == 4
        assert all(e["status"] == "ok" for e in entries)
        assert sorted(e["sample_index"] for e in entries) == [0, 1, 2, 3]

    @pytest.mark.skipif(_fluidsynth_missing, reason="FluidSynth not installed")
    @pytest.mark.skipif(_sf2_missing, reason="No .sf2 soundfonts in sf/")
    def test_batch_resume_skips_complete(self, tmp_path):
        """Re-running the same batch skips all complete samples (R-P11)."""
        cfg = Config(
            global_seed=42,
            count=4,
            dataset_root=str(tmp_path / "dataset"),
            workers=2,
        )
        result1 = generate_batch(cfg)
        assert result1.succeeded == 4

        result2 = generate_batch(cfg)
        assert result2.total == 4
        assert result2.skipped == 4
        assert result2.succeeded == 0
        assert result2.failed == 0

    @pytest.mark.skipif(_fluidsynth_missing, reason="FluidSynth not installed")
    @pytest.mark.skipif(_sf2_missing, reason="No .sf2 soundfonts in sf/")
    def test_batch_output_mode_mix_only(self, tmp_path):
        """mix-only mode: no stems/ or midi/ dirs written (R-P14)."""
        cfg = Config(
            global_seed=42,
            count=2,
            dataset_root=str(tmp_path / "dataset"),
            workers=1,
            output_mode="mix-only",
        )
        result = generate_batch(cfg)
        assert result.succeeded == 2

        for idx in range(2):
            sample_dir = tmp_path / "dataset" / f"{idx:06d}"
            assert (sample_dir / "sample.json").exists()
            assert (sample_dir / "mix.wav").exists()
            assert not (sample_dir / "stems").exists(), "stems/ should not exist in mix-only mode"
            assert not (sample_dir / "midi").exists(), "midi/ should not exist in mix-only mode"

    @pytest.mark.skipif(_fluidsynth_missing, reason="FluidSynth not installed")
    @pytest.mark.skipif(_sf2_missing, reason="No .sf2 soundfonts in sf/")
    def test_batch_progress_events_emitted(self, tmp_path, capsys):
        """Structured JSON events appear on stderr (R-P15)."""
        cfg = Config(
            global_seed=42,
            count=2,
            dataset_root=str(tmp_path / "dataset"),
            workers=1,
        )
        generate_batch(cfg)
        captured = capsys.readouterr()
        events = [json.loads(line) for line in captured.err.splitlines() if line.strip()]
        event_types = [e["event"] for e in events]
        assert "batch_start" in event_types
        assert "batch_done" in event_types
        assert event_types.count("sample_start") == 2
        assert event_types.count("sample_done") == 2
