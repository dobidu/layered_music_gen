"""Tests for musicgen.midi_indexer — Integration 2 (midi_file_manager).

MidiManager is lazy-imported; tests inject a fake midi_manager module via
sys.modules to avoid requiring the external package.
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
# Fake midi_manager module helpers
# ---------------------------------------------------------------------------

class _FakeMidiMetadata:
    def __init__(self, mid_id: int, path: str):
        self.id = mid_id
        self.path = path
        self.bpm = 120.0
        self.key = None
        self.time_signature = "4/4"
        self.category = "melody"
        self.tags: list = []
        self.description = ""

    def to_dict(self):
        return {
            "id": self.id, "path": self.path, "bpm": self.bpm,
            "key": self.key, "time_signature": self.time_signature,
            "category": self.category, "tags": self.tags,
            "description": self.description,
        }


def _make_midi_manager_module():
    """Build a fake midi_manager module whose MidiManager records all calls."""
    mod = types.ModuleType("midi_manager")

    class FakeMM:
        def __init__(self, json_path: str, midi_directory=None):
            self.json_path = json_path
            self.midi_directory = midi_directory
            self.midis: list[_FakeMidiMetadata] = []
            self._next_id = 1
            self.saved = False
            self.csv_exported: str | None = None

        def add_midi(self, midi_path: str, analyze: bool = True, save: bool = True):
            meta = _FakeMidiMetadata(self._next_id, midi_path)
            self._next_id += 1
            self.midis.append(meta)
            return meta

        def save_midis(self):
            self.saved = True
            os.makedirs(os.path.dirname(os.path.abspath(self.json_path)), exist_ok=True)
            with open(self.json_path, "w") as f:
                json.dump([m.to_dict() for m in self.midis], f)

        def export_to_csv(self, csv_path: str):
            self.csv_exported = csv_path
            import csv
            with open(csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["id", "path", "bpm", "key", "category", "tags"])
                for m in self.midis:
                    w.writerow([m.id, m.path, m.bpm, m.key, m.category, ",".join(m.tags)])

    mod.MidiManager = FakeMM
    return mod


_FAKE_MM_MOD = _make_midi_manager_module()


# ---------------------------------------------------------------------------
# Dataset fixture helpers
# ---------------------------------------------------------------------------

def _write_sample(sample_dir: Path, tempo_bpm: float, key: str, time_sig: str,
                  split: str, musicality: float, layers=("beat", "melody", "harmony", "bassline")):
    sample_dir.mkdir(parents=True, exist_ok=True)
    (sample_dir / "sample.json").write_text(json.dumps({
        "tempo_bpm": tempo_bpm,
        "key": key,
        "time_signature": time_sig,
        "split": split,
        "musicality_score": {"score": musicality, "components": {}},
    }))
    midi_dir = sample_dir / "midi"
    midi_dir.mkdir(exist_ok=True)
    for layer in layers:
        (midi_dir / f"{layer}.mid").write_bytes(b"MThd")  # fake MIDI bytes


@pytest.fixture
def two_sample_dataset(tmp_path):
    """Dataset with 2 valid samples × 4 layers each."""
    _write_sample(tmp_path / "dataset" / "000000", 120.0, "C", "4/4", "train", 0.8)
    _write_sample(tmp_path / "dataset" / "000001", 90.0, "Am", "3/4", "valid", 0.6)
    return tmp_path / "dataset"


# ---------------------------------------------------------------------------
# Unit tests for index_midi_dataset
# ---------------------------------------------------------------------------

class TestIndexMidiDataset:
    def test_indexes_all_layers_per_sample(self, two_sample_dataset, tmp_path):
        from musicgen.midi_indexer import index_midi_dataset
        out_db = str(tmp_path / "midi_db.json")
        with patch.dict(sys.modules, {"midi_manager": _FAKE_MM_MOD}):
            count = index_midi_dataset(str(two_sample_dataset), out_db)
        assert count == 8  # 2 samples × 4 layers

    def test_creates_db_file(self, two_sample_dataset, tmp_path):
        from musicgen.midi_indexer import index_midi_dataset
        out_db = str(tmp_path / "midi_db.json")
        with patch.dict(sys.modules, {"midi_manager": _FAKE_MM_MOD}):
            index_midi_dataset(str(two_sample_dataset), out_db)
        assert os.path.isfile(out_db)

    def test_applies_ground_truth_bpm(self, two_sample_dataset, tmp_path):
        from musicgen.midi_indexer import index_midi_dataset
        out_db = str(tmp_path / "midi_db.json")
        with patch.dict(sys.modules, {"midi_manager": _make_midi_manager_module()}) as mods:
            mm_mod = mods["midi_manager"]
            # Capture the MidiManager instance
            instances = []
            orig_cls = mm_mod.MidiManager
            class CapturingMM(orig_cls):
                def __init__(self, *a, **kw):
                    super().__init__(*a, **kw)
                    instances.append(self)
            mm_mod.MidiManager = CapturingMM
            index_midi_dataset(str(two_sample_dataset), out_db)
        mm = instances[0]
        bpms = {m.bpm for m in mm.midis}
        assert 120.0 in bpms
        assert 90.0 in bpms

    def test_applies_ground_truth_key(self, two_sample_dataset, tmp_path):
        from musicgen.midi_indexer import index_midi_dataset
        out_db = str(tmp_path / "midi_db.json")
        instances = []
        with patch.dict(sys.modules, {"midi_manager": _make_midi_manager_module()}) as mods:
            mm_mod = mods["midi_manager"]
            orig_cls = mm_mod.MidiManager
            class CapturingMM(orig_cls):
                def __init__(self, *a, **kw):
                    super().__init__(*a, **kw)
                    instances.append(self)
            mm_mod.MidiManager = CapturingMM
            index_midi_dataset(str(two_sample_dataset), out_db)
        mm = instances[0]
        keys = {m.key for m in mm.midis}
        assert "C" in keys
        assert "Am" in keys

    def test_applies_ground_truth_time_signature(self, two_sample_dataset, tmp_path):
        from musicgen.midi_indexer import index_midi_dataset
        out_db = str(tmp_path / "midi_db.json")
        instances = []
        with patch.dict(sys.modules, {"midi_manager": _make_midi_manager_module()}) as mods:
            mm_mod = mods["midi_manager"]
            orig_cls = mm_mod.MidiManager
            class CapturingMM(orig_cls):
                def __init__(self, *a, **kw):
                    super().__init__(*a, **kw)
                    instances.append(self)
            mm_mod.MidiManager = CapturingMM
            index_midi_dataset(str(two_sample_dataset), out_db)
        mm = instances[0]
        time_sigs = {m.time_signature for m in mm.midis}
        assert "4/4" in time_sigs
        assert "3/4" in time_sigs

    def test_split_in_tags(self, two_sample_dataset, tmp_path):
        from musicgen.midi_indexer import index_midi_dataset
        out_db = str(tmp_path / "midi_db.json")
        instances = []
        with patch.dict(sys.modules, {"midi_manager": _make_midi_manager_module()}) as mods:
            mm_mod = mods["midi_manager"]
            orig_cls = mm_mod.MidiManager
            class CapturingMM(orig_cls):
                def __init__(self, *a, **kw):
                    super().__init__(*a, **kw)
                    instances.append(self)
            mm_mod.MidiManager = CapturingMM
            index_midi_dataset(str(two_sample_dataset), out_db)
        mm = instances[0]
        all_tags = [tag for m in mm.midis for tag in m.tags]
        assert "train" in all_tags
        assert "valid" in all_tags
        assert "musicgen" in all_tags

    def test_layer_category_mapping(self, two_sample_dataset, tmp_path):
        from musicgen.midi_indexer import index_midi_dataset
        out_db = str(tmp_path / "midi_db.json")
        instances = []
        with patch.dict(sys.modules, {"midi_manager": _make_midi_manager_module()}) as mods:
            mm_mod = mods["midi_manager"]
            orig_cls = mm_mod.MidiManager
            class CapturingMM(orig_cls):
                def __init__(self, *a, **kw):
                    super().__init__(*a, **kw)
                    instances.append(self)
            mm_mod.MidiManager = CapturingMM
            index_midi_dataset(str(two_sample_dataset), out_db)
        mm = instances[0]
        categories = {m.category for m in mm.midis}
        assert "drums" in categories    # beat layer
        assert "bass" in categories     # bassline layer
        assert "melody" in categories
        assert "harmony" in categories

    def test_skips_dir_without_sample_json(self, tmp_path):
        from musicgen.midi_indexer import index_midi_dataset
        dataset = tmp_path / "dataset"
        dataset.mkdir()
        # dir without sample.json
        (dataset / "000000").mkdir()
        (dataset / "000000" / "midi").mkdir()
        (dataset / "000000" / "midi" / "beat.mid").write_bytes(b"MThd")
        out_db = str(tmp_path / "midi_db.json")
        with patch.dict(sys.modules, {"midi_manager": _FAKE_MM_MOD}):
            count = index_midi_dataset(str(dataset), out_db)
        assert count == 0

    def test_skips_dir_without_midi_subdir(self, tmp_path):
        from musicgen.midi_indexer import index_midi_dataset
        dataset = tmp_path / "dataset"
        dataset.mkdir()
        sample_dir = dataset / "000000"
        sample_dir.mkdir()
        (sample_dir / "sample.json").write_text(json.dumps({
            "tempo_bpm": 120, "key": "C", "time_signature": "4/4",
            "split": "train", "musicality_score": {"score": 0.7},
        }))
        # no midi/ subdir
        out_db = str(tmp_path / "midi_db.json")
        with patch.dict(sys.modules, {"midi_manager": _FAKE_MM_MOD}):
            count = index_midi_dataset(str(dataset), out_db)
        assert count == 0

    def test_csv_export_when_requested(self, two_sample_dataset, tmp_path):
        from musicgen.midi_indexer import index_midi_dataset
        out_db = str(tmp_path / "midi_db.json")
        out_csv = str(tmp_path / "index.csv")
        with patch.dict(sys.modules, {"midi_manager": _FAKE_MM_MOD}):
            index_midi_dataset(str(two_sample_dataset), out_db, export_csv=out_csv)
        assert os.path.isfile(out_csv)

    def test_raises_import_error_without_midi_manager(self, two_sample_dataset, tmp_path):
        from musicgen.midi_indexer import index_midi_dataset
        out_db = str(tmp_path / "midi_db.json")
        with patch.dict(sys.modules, {"midi_manager": None}):
            with pytest.raises(ImportError, match="midi_file_manager"):
                index_midi_dataset(str(two_sample_dataset), out_db)

    def test_partial_layers_indexed_when_some_missing(self, tmp_path):
        """Dataset with only 2 of 4 layers → only those 2 are indexed."""
        from musicgen.midi_indexer import index_midi_dataset
        dataset = tmp_path / "dataset"
        _write_sample(dataset / "000000", 120.0, "C", "4/4", "train", 0.8,
                      layers=("beat", "melody"))  # only 2 layers
        out_db = str(tmp_path / "midi_db.json")
        with patch.dict(sys.modules, {"midi_manager": _FAKE_MM_MOD}):
            count = index_midi_dataset(str(dataset), out_db)
        assert count == 2


# ---------------------------------------------------------------------------
# CLI tests for index-midi command
# ---------------------------------------------------------------------------

class TestIndexMidiCLICommand:
    def test_index_midi_help_exits_zero(self):
        result = runner.invoke(app, ["index-midi", "--help"])
        assert result.exit_code == 0
        assert "--dataset" in result.output

    def test_index_midi_missing_dataset_exits_nonzero(self, tmp_path):
        result = runner.invoke(app, [
            "index-midi",
            "--dataset", str(tmp_path / "nonexistent"),
            "--out", str(tmp_path / "out.json"),
        ])
        assert result.exit_code != 0

    def test_index_midi_runs_with_fake_mm(self, two_sample_dataset, tmp_path):
        out_db = str(tmp_path / "midi_db.json")
        with patch.dict(sys.modules, {"midi_manager": _FAKE_MM_MOD}):
            result = runner.invoke(app, [
                "index-midi",
                "--dataset", str(two_sample_dataset),
                "--out", out_db,
            ])
        assert result.exit_code == 0, result.output
        assert "8" in result.output  # 8 entries indexed

    def test_index_midi_csv_flag(self, two_sample_dataset, tmp_path):
        out_db = str(tmp_path / "midi_db.json")
        out_csv = str(tmp_path / "index.csv")
        with patch.dict(sys.modules, {"midi_manager": _FAKE_MM_MOD}):
            result = runner.invoke(app, [
                "index-midi",
                "--dataset", str(two_sample_dataset),
                "--out", out_db,
                "--csv", out_csv,
            ])
        assert result.exit_code == 0, result.output
        assert os.path.isfile(out_csv)
