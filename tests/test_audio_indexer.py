"""Tests for musicgen.audio_indexer — Integration 3 (audio_sample_manager).

SampleManager is lazy-imported; tests inject a fake sample_manager module via
sys.modules to avoid requiring the external package or librosa analysis.
"""
from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from musicgen.cli import app

runner = CliRunner()


# ---------------------------------------------------------------------------
# Fake sample_manager module helpers
# ---------------------------------------------------------------------------

class _FakeSampleMetadata:
    def __init__(self, sample_id: int, path: str):
        self.id = sample_id
        self.path = path
        self.bpm = 120.0
        self.key = None
        self.time_signature = "4/4"
        self.scale = "major"
        self.category = "beat"
        self.tags: list = []
        self.is_loop = True
        self.description = ""

    def to_dict(self):
        return {
            "id": self.id, "path": self.path, "bpm": self.bpm,
            "key": self.key, "time_signature": self.time_signature,
            "scale": self.scale, "category": self.category,
            "tags": self.tags, "is_loop": self.is_loop,
            "description": self.description,
        }


def _make_sample_manager_module():
    """Build a fake sample_manager module whose SampleManager records all calls."""
    mod = types.ModuleType("sample_manager")

    class FakeSM:
        def __init__(self, json_path: str, samples_directory=None):
            self.json_path = json_path
            self.samples_directory = samples_directory
            self.samples: list[_FakeSampleMetadata] = []
            self._next_id = 1
            self.saved = False
            self.csv_exported: str | None = None

        def add_sample(self, sample_path: str, analyze: bool = True, save: bool = True):
            meta = _FakeSampleMetadata(self._next_id, sample_path)
            self._next_id += 1
            self.samples.append(meta)
            return meta

        def save_samples(self):
            self.saved = True
            os.makedirs(os.path.dirname(os.path.abspath(self.json_path)), exist_ok=True)
            with open(self.json_path, "w") as f:
                json.dump([m.to_dict() for m in self.samples], f)

        def export_csv(self, csv_path: str):
            self.csv_exported = csv_path
            import csv
            with open(csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["id", "path", "bpm", "key", "category", "scale", "tags", "is_loop"])
                for m in self.samples:
                    w.writerow([m.id, m.path, m.bpm, m.key, m.category, m.scale,
                                ",".join(m.tags), m.is_loop])

    mod.SampleManager = FakeSM
    return mod


_FAKE_SM_MOD = _make_sample_manager_module()


# ---------------------------------------------------------------------------
# Dataset fixture helpers
# ---------------------------------------------------------------------------

def _write_sample(sample_dir: Path, tempo_bpm: float, key: str, mode: str,
                  time_sig: str, split: str, musicality: float,
                  layers=("beat", "melody", "harmony", "bassline")):
    sample_dir.mkdir(parents=True, exist_ok=True)
    (sample_dir / "sample.json").write_text(json.dumps({
        "tempo_bpm": tempo_bpm,
        "key": key,
        "mode": mode,
        "time_signature": time_sig,
        "split": split,
        "musicality_score": {"score": musicality, "components": {}},
    }))
    stems_dir = sample_dir / "stems"
    stems_dir.mkdir(exist_ok=True)
    for layer in layers:
        (stems_dir / f"{layer}.wav").write_bytes(b"RIFF\x00\x00\x00\x00WAVE")


@pytest.fixture
def two_sample_dataset(tmp_path):
    """Dataset with 2 valid samples × 4 layers each."""
    _write_sample(tmp_path / "dataset" / "000000", 120.0, "C", "major", "4/4", "train", 0.8)
    _write_sample(tmp_path / "dataset" / "000001", 90.0, "Am", "minor", "3/4", "valid", 0.6)
    return tmp_path / "dataset"


# ---------------------------------------------------------------------------
# Unit tests for index_audio_dataset
# ---------------------------------------------------------------------------

class TestIndexAudioDataset:
    def test_indexes_all_layers_per_sample(self, two_sample_dataset, tmp_path):
        from musicgen.audio_indexer import index_audio_dataset
        out_db = str(tmp_path / "audio_db.json")
        with patch.dict(sys.modules, {"sample_manager": _FAKE_SM_MOD}):
            count = index_audio_dataset(str(two_sample_dataset), out_db)
        assert count == 8  # 2 samples × 4 layers

    def test_creates_db_file(self, two_sample_dataset, tmp_path):
        from musicgen.audio_indexer import index_audio_dataset
        out_db = str(tmp_path / "audio_db.json")
        with patch.dict(sys.modules, {"sample_manager": _FAKE_SM_MOD}):
            index_audio_dataset(str(two_sample_dataset), out_db)
        assert os.path.isfile(out_db)

    def _run_and_capture(self, dataset, tmp_path):
        """Helper: run indexer with a capturing SM, return (count, sm_instance)."""
        from musicgen.audio_indexer import index_audio_dataset
        out_db = str(tmp_path / "audio_db.json")
        instances = []
        mod = _make_sample_manager_module()
        orig_cls = mod.SampleManager
        class CapturingSM(orig_cls):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                instances.append(self)
        mod.SampleManager = CapturingSM
        with patch.dict(sys.modules, {"sample_manager": mod}):
            count = index_audio_dataset(str(dataset), out_db)
        return count, instances[0]

    def test_applies_ground_truth_bpm(self, two_sample_dataset, tmp_path):
        _, sm = self._run_and_capture(two_sample_dataset, tmp_path)
        bpms = {m.bpm for m in sm.samples}
        assert 120.0 in bpms
        assert 90.0 in bpms

    def test_applies_ground_truth_key(self, two_sample_dataset, tmp_path):
        _, sm = self._run_and_capture(two_sample_dataset, tmp_path)
        keys = {m.key for m in sm.samples}
        assert "C" in keys
        assert "Am" in keys

    def test_applies_ground_truth_time_signature(self, two_sample_dataset, tmp_path):
        _, sm = self._run_and_capture(two_sample_dataset, tmp_path)
        time_sigs = {m.time_signature for m in sm.samples}
        assert "4/4" in time_sigs
        assert "3/4" in time_sigs

    def test_applies_mode_as_scale(self, two_sample_dataset, tmp_path):
        _, sm = self._run_and_capture(two_sample_dataset, tmp_path)
        scales = {m.scale for m in sm.samples}
        assert "major" in scales
        assert "minor" in scales

    def test_split_in_tags(self, two_sample_dataset, tmp_path):
        _, sm = self._run_and_capture(two_sample_dataset, tmp_path)
        all_tags = [tag for m in sm.samples for tag in m.tags]
        assert "train" in all_tags
        assert "valid" in all_tags
        assert "musicgen" in all_tags

    def test_layer_category_mapping(self, two_sample_dataset, tmp_path):
        _, sm = self._run_and_capture(two_sample_dataset, tmp_path)
        categories = {m.category for m in sm.samples}
        assert "beat" in categories
        assert "bass" in categories
        assert "melody" in categories
        assert "harmony" in categories

    def test_is_loop_false(self, two_sample_dataset, tmp_path):
        """musicgen stems are full song segments, not loops."""
        _, sm = self._run_and_capture(two_sample_dataset, tmp_path)
        assert all(not m.is_loop for m in sm.samples)

    def test_skips_dir_without_sample_json(self, tmp_path):
        from musicgen.audio_indexer import index_audio_dataset
        dataset = tmp_path / "dataset"
        dataset.mkdir()
        (dataset / "000000" / "stems").mkdir(parents=True)
        (dataset / "000000" / "stems" / "beat.wav").write_bytes(b"RIFF")
        out_db = str(tmp_path / "audio_db.json")
        with patch.dict(sys.modules, {"sample_manager": _FAKE_SM_MOD}):
            count = index_audio_dataset(str(dataset), out_db)
        assert count == 0

    def test_skips_dir_without_stems_subdir(self, tmp_path):
        from musicgen.audio_indexer import index_audio_dataset
        dataset = tmp_path / "dataset"
        sample_dir = dataset / "000000"
        sample_dir.mkdir(parents=True)
        (sample_dir / "sample.json").write_text(json.dumps({
            "tempo_bpm": 120, "key": "C", "mode": "major",
            "time_signature": "4/4", "split": "train",
            "musicality_score": {"score": 0.7},
        }))
        out_db = str(tmp_path / "audio_db.json")
        with patch.dict(sys.modules, {"sample_manager": _FAKE_SM_MOD}):
            count = index_audio_dataset(str(dataset), out_db)
        assert count == 0

    def test_csv_export_when_requested(self, two_sample_dataset, tmp_path):
        from musicgen.audio_indexer import index_audio_dataset
        out_db = str(tmp_path / "audio_db.json")
        out_csv = str(tmp_path / "index.csv")
        with patch.dict(sys.modules, {"sample_manager": _FAKE_SM_MOD}):
            index_audio_dataset(str(two_sample_dataset), out_db, export_csv=out_csv)
        assert os.path.isfile(out_csv)

    def test_raises_import_error_without_sample_manager(self, two_sample_dataset, tmp_path):
        from musicgen.audio_indexer import index_audio_dataset
        out_db = str(tmp_path / "audio_db.json")
        with patch.dict(sys.modules, {"sample_manager": None}):
            with pytest.raises(ImportError, match="audio_sample_manager"):
                index_audio_dataset(str(two_sample_dataset), out_db)

    def test_partial_layers_indexed_when_some_missing(self, tmp_path):
        """Dataset with only 2 of 4 layers → only those 2 are indexed."""
        from musicgen.audio_indexer import index_audio_dataset
        dataset = tmp_path / "dataset"
        _write_sample(dataset / "000000", 120.0, "C", "major", "4/4", "train", 0.8,
                      layers=("beat", "melody"))
        out_db = str(tmp_path / "audio_db.json")
        with patch.dict(sys.modules, {"sample_manager": _FAKE_SM_MOD}):
            count = index_audio_dataset(str(dataset), out_db)
        assert count == 2


# ---------------------------------------------------------------------------
# CLI tests for index-audio command
# ---------------------------------------------------------------------------

class TestIndexAudioCLICommand:
    def test_index_audio_help_exits_zero(self):
        result = runner.invoke(app, ["index-audio", "--help"])
        assert result.exit_code == 0
        assert "--dataset" in result.output

    def test_index_audio_missing_dataset_exits_nonzero(self, tmp_path):
        result = runner.invoke(app, [
            "index-audio",
            "--dataset", str(tmp_path / "nonexistent"),
            "--out", str(tmp_path / "out.json"),
        ])
        assert result.exit_code != 0

    def test_index_audio_runs_with_fake_sm(self, two_sample_dataset, tmp_path):
        out_db = str(tmp_path / "audio_db.json")
        with patch.dict(sys.modules, {"sample_manager": _FAKE_SM_MOD}):
            result = runner.invoke(app, [
                "index-audio",
                "--dataset", str(two_sample_dataset),
                "--out", out_db,
            ])
        assert result.exit_code == 0, result.output
        assert "8" in result.output

    def test_index_audio_csv_flag(self, two_sample_dataset, tmp_path):
        out_db = str(tmp_path / "audio_db.json")
        out_csv = str(tmp_path / "index.csv")
        with patch.dict(sys.modules, {"sample_manager": _FAKE_SM_MOD}):
            result = runner.invoke(app, [
                "index-audio",
                "--dataset", str(two_sample_dataset),
                "--out", out_db,
                "--csv", out_csv,
            ])
        assert result.exit_code == 0, result.output
        assert os.path.isfile(out_csv)
