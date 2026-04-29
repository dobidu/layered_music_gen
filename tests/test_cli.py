"""Tests for src/musicgen/cli.py full CLI (D-61..D-65, R-P13).

Uses typer.testing.CliRunner — runs in-process, fast, no FluidSynth.
"""
from __future__ import annotations

import json
import os

import pytest
from typer.testing import CliRunner

from musicgen.cli import app

runner = CliRunner()


class TestGenerateCommand:
    def test_generate_missing_seed_exits_nonzero(self):
        result = runner.invoke(app, ["generate", "--count", "1"])
        assert result.exit_code != 0

    def test_generate_invalid_output_mode_exits_nonzero(self):
        result = runner.invoke(app, [
            "generate", "--seed", "42", "--output-mode", "invalid",
        ])
        assert result.exit_code != 0

    def test_generate_help_exits_zero(self):
        result = runner.invoke(app, ["generate", "--help"])
        assert result.exit_code == 0
        assert "--count" in result.output
        assert "--seed" in result.output

    def test_generate_runs_with_mocked_batch(self, tmp_path, monkeypatch):
        from musicgen.api import SampleResult
        from musicgen.batch import BatchResult

        fake_result = BatchResult(
            total=1, succeeded=1, failed=0, skipped=0,
            results=(SampleResult(
                sample_index=0, seed=0,
                sample_dir=str(tmp_path / "000000"),
                sample_json_path=str(tmp_path / "000000/sample.json"),
                mix_path="", stem_paths={}, midi_paths={},
                split="train", status="ok",
                musicality_score=0.5, duration_seconds=1.0,
            ),),
            duration_seconds=0.1,
        )
        monkeypatch.setattr("musicgen.cli.generate_batch", lambda cfg: fake_result)
        result = runner.invoke(app, [
            "generate", "--count", "1", "--out", str(tmp_path), "--seed", "42",
        ])
        assert result.exit_code == 0
        assert "1" in result.output  # succeeded count


class TestCleanCommand:
    def test_clean_no_failed_flag_exits_nonzero(self, tmp_path):
        result = runner.invoke(app, ["clean", "--out", str(tmp_path)])
        assert result.exit_code != 0

    def test_clean_empty_dataset_exits_zero(self, tmp_path):
        result = runner.invoke(app, ["clean", "--failed", "--out", str(tmp_path)])
        assert result.exit_code == 0

    def test_clean_removes_failed_dir(self, tmp_path):
        dataset = tmp_path / "dataset"
        dataset.mkdir()
        # Create manifest.jsonl with a failed entry
        manifest = dataset / "manifest.jsonl"
        manifest.write_text(
            json.dumps({"sample_index": 0, "status": "failed", "path": ""}) + "\n"
        )
        # Create a dead directory (no sample.json)
        dead_dir = dataset / "000000"
        dead_dir.mkdir()
        (dead_dir / "mix.wav").touch()  # partial files, no sentinel

        result = runner.invoke(app, ["clean", "--failed", "--out", str(dataset)])
        assert result.exit_code == 0
        assert not dead_dir.exists()


class TestCalibrateCommand:
    def test_calibrate_exits_zero(self, monkeypatch):
        monkeypatch.setattr(
            "musicgen.cli.calibrate.measure_and_save_preroll", lambda root: 0.0
        )
        result = runner.invoke(app, ["calibrate"])
        assert result.exit_code == 0
        assert "0.0" in result.output or "pre-roll" in result.output.lower()
