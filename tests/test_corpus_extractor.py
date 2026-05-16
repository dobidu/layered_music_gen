"""Tests for corpus_extractor (v0.5 Phase 1).

Uses a synthetic in-memory fixture dataset — no FluidSynth, no real audio.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import mido
import pytest

from musicgen.corpus_extractor import (
    _midi_note_to_degree,
    _extract_melody_degrees_from_midi,
    extract_sequences,
)


# ---------------------------------------------------------------------------
# Unit tests — helpers
# ---------------------------------------------------------------------------


class TestMidiNoteToDegree:
    def test_c_major_c4(self):
        assert _midi_note_to_degree(60, "C") == "1"  # C4 in C major = degree 1

    def test_c_major_e4(self):
        assert _midi_note_to_degree(64, "C") == "3"  # E4 in C major = degree 3

    def test_c_major_g4(self):
        assert _midi_note_to_degree(67, "C") == "5"  # G4 in C major = degree 5

    def test_a_minor_a3(self):
        assert _midi_note_to_degree(57, "Am") == "1"  # A3 in Am = degree 1

    def test_a_minor_c4(self):
        assert _midi_note_to_degree(60, "Am") == "3"  # C4 in Am = minor 3rd

    def test_g_major_b4(self):
        assert _midi_note_to_degree(71, "G") == "3"  # B4 in G major = 3rd

    def test_octave_invariant(self):
        # C5 (72) in C major is still degree 1
        assert _midi_note_to_degree(72, "C") == "1"
        assert _midi_note_to_degree(48, "C") == "1"

    def test_flat_key(self):
        assert _midi_note_to_degree(60, "C") == "1"
        # F major: root F=5, degree 1 = F (65, 53, 77...)
        assert _midi_note_to_degree(65, "F") == "1"

    def test_chromatic_snaps_to_nearest(self):
        # C# (61) in C major — not in scale, should snap to 1 (C=0) or 2 (D=2), dist=1 each
        # The function picks the first minimum; either "1" or "2" is acceptable
        result = _midi_note_to_degree(61, "C")
        assert result in ("1", "2")


class TestExtractMelodyDegreesFromMidi:
    def test_empty_midi(self, tmp_path):
        midi_path = tmp_path / "melody.mid"
        mid = mido.MidiFile()
        track = mido.MidiTrack()
        mid.tracks.append(track)
        mid.save(str(midi_path))
        assert _extract_melody_degrees_from_midi(str(midi_path), "C") == []

    def test_single_note_c4(self, tmp_path):
        midi_path = tmp_path / "melody.mid"
        mid = mido.MidiFile(type=0)
        track = mido.MidiTrack()
        track.append(mido.Message("note_on", note=60, velocity=80, time=0))
        track.append(mido.Message("note_off", note=60, velocity=0, time=480))
        mid.tracks.append(track)
        mid.save(str(midi_path))
        result = _extract_melody_degrees_from_midi(str(midi_path), "C")
        assert result == ["1"]

    def test_multiple_notes(self, tmp_path):
        midi_path = tmp_path / "melody.mid"
        mid = mido.MidiFile(type=0)
        track = mido.MidiTrack()
        # C4=60 (1), E4=64 (3), G4=67 (5) in C major
        for note in (60, 64, 67):
            track.append(mido.Message("note_on", note=note, velocity=80, time=0))
            track.append(mido.Message("note_off", note=note, velocity=0, time=240))
        mid.tracks.append(track)
        mid.save(str(midi_path))
        result = _extract_melody_degrees_from_midi(str(midi_path), "C")
        assert result == ["1", "3", "5"]

    def test_note_off_velocity_zero_ignored(self, tmp_path):
        midi_path = tmp_path / "melody.mid"
        mid = mido.MidiFile(type=0)
        track = mido.MidiTrack()
        track.append(mido.Message("note_on", note=60, velocity=80, time=0))
        track.append(mido.Message("note_on", note=60, velocity=0, time=480))  # note_off via velocity=0
        mid.tracks.append(track)
        mid.save(str(midi_path))
        result = _extract_melody_degrees_from_midi(str(midi_path), "C")
        assert result == ["1"]

    def test_bad_path_returns_empty(self):
        result = _extract_melody_degrees_from_midi("/nonexistent/path.mid", "C")
        assert result == []


# ---------------------------------------------------------------------------
# Integration — extract_sequences
# ---------------------------------------------------------------------------

def _write_sample_json(sample_dir: Path, idx: int, **overrides) -> None:
    """Write a minimal sample.json for testing."""
    default = {
        "key": "C",
        "mode": "major",
        "tempo_bpm": 120,
        "time_signature": "4/4",
        "song_arrangement": [
            {"part": "verse", "start_seconds": 0.0, "end_seconds": 8.0},
            {"part": "chorus", "start_seconds": 8.0, "end_seconds": 16.0},
        ],
        "chord_progression": {
            "verse": ["I", "V", "vi", "IV"],
            "chorus": ["I", "IV", "V", "I"],
        },
        "musicality_score": {"score": 0.75, "components": {}},
        "genre": None,
        "seed": idx,
        "split": "train",
        "musicgen_version": "0.5.0",
        "active_layers": {},
        "beat_times": {},
        "downbeat_times": {},
        "measures_per_part": {"verse": 4, "chorus": 4},
        "time_signatures_per_part": {"verse": "4/4", "chorus": "4/4"},
        "swing": 0.5,
        "duration_seconds": 16.0,
        "fluidsynth_version": "2.3.4",
        "soundfonts": {},
        "fx_params": {},
        "mix": "mix.wav",
        "stems": {},
        "midi": {},
        "pre_roll_offset_seconds": 0.0,
    }
    default.update(overrides)
    (sample_dir / "sample.json").write_text(json.dumps(default))


def _write_melody_midi(sample_dir: Path, notes=(60, 64, 67, 65)) -> None:
    """Write a tiny melody MIDI with the given note numbers."""
    midi_dir = sample_dir / "midi"
    midi_dir.mkdir(exist_ok=True)
    mid = mido.MidiFile(type=0)
    track = mido.MidiTrack()
    for note in notes:
        track.append(mido.Message("note_on", note=note, velocity=80, time=0))
        track.append(mido.Message("note_off", note=note, velocity=0, time=240))
    mid.tracks.append(track)
    mid.save(str(midi_dir / "melody.mid"))


class TestExtractSequences:
    def _make_dataset(self, tmp_path, n=3) -> Path:
        dataset = tmp_path / "dataset"
        dataset.mkdir()
        for i in range(n):
            sd = dataset / f"{i:06d}"
            sd.mkdir()
            _write_sample_json(sd, i)
            _write_melody_midi(sd)
        return dataset

    def test_basic_extraction(self, tmp_path):
        dataset = self._make_dataset(tmp_path, n=3)
        out = tmp_path / "sequences.json"
        counts = extract_sequences(str(dataset), str(out))
        assert counts["chord"] == 3
        assert counts["melody"] == 3
        assert out.exists()

    def test_sequences_json_schema(self, tmp_path):
        dataset = self._make_dataset(tmp_path, n=2)
        out = tmp_path / "sequences.json"
        extract_sequences(str(dataset), str(out))
        data = json.loads(out.read_text())
        assert "metadata" in data
        assert "chord" in data
        assert "melody" in data
        assert data["metadata"]["n_samples"] == 2
        assert len(data["chord"]) == 2
        assert len(data["melody"]) == 2

    def test_chord_full_sequence_concatenated(self, tmp_path):
        dataset = self._make_dataset(tmp_path, n=1)
        out = tmp_path / "sequences.json"
        extract_sequences(str(dataset), str(out))
        data = json.loads(out.read_text())
        # verse=["I","V","vi","IV"] + chorus=["I","IV","V","I"] = 8 tokens
        assert data["chord"][0]["full_sequence"] == ["I", "V", "vi", "IV", "I", "IV", "V", "I"]

    def test_melody_full_sequence_nonempty(self, tmp_path):
        dataset = self._make_dataset(tmp_path, n=1)
        out = tmp_path / "sequences.json"
        extract_sequences(str(dataset), str(out))
        data = json.loads(out.read_text())
        seq = data["melody"][0]["full_sequence"]
        assert len(seq) > 0
        assert all(d in ("1", "2", "3", "4", "5", "6", "7") for d in seq)

    def test_musicality_gate(self, tmp_path):
        dataset = tmp_path / "dataset"
        dataset.mkdir()
        # Sample 0: high score
        sd0 = dataset / "000000"; sd0.mkdir()
        _write_sample_json(sd0, 0, musicality_score={"score": 0.8, "components": {}})
        _write_melody_midi(sd0)
        # Sample 1: low score
        sd1 = dataset / "000001"; sd1.mkdir()
        _write_sample_json(sd1, 1, musicality_score={"score": 0.3, "components": {}})
        _write_melody_midi(sd1)

        out = tmp_path / "sequences.json"
        counts = extract_sequences(str(dataset), str(out), min_musicality=0.6)
        assert counts["chord"] == 1
        assert counts["melody"] == 1
        data = json.loads(out.read_text())
        assert data["chord"][0]["sample_index"] == 0

    def test_genre_recorded(self, tmp_path):
        dataset = tmp_path / "dataset"
        dataset.mkdir()
        sd = dataset / "000000"; sd.mkdir()
        _write_sample_json(sd, 0, genre=["pop", "electronic"])
        _write_melody_midi(sd)
        out = tmp_path / "sequences.json"
        extract_sequences(str(dataset), str(out))
        data = json.loads(out.read_text())
        assert data["chord"][0]["genre"] == ["pop", "electronic"]
        assert set(data["metadata"]["genres"]) == {"pop", "electronic"}

    def test_missing_sample_json_skipped(self, tmp_path):
        dataset = tmp_path / "dataset"
        dataset.mkdir()
        # Create a dir with no sample.json
        bad = dataset / "000000"; bad.mkdir()
        # Good sample
        sd = dataset / "000001"; sd.mkdir()
        _write_sample_json(sd, 1)
        _write_melody_midi(sd)
        out = tmp_path / "sequences.json"
        counts = extract_sequences(str(dataset), str(out))
        assert counts["chord"] == 1

    def test_missing_melody_midi_ok(self, tmp_path):
        dataset = tmp_path / "dataset"
        dataset.mkdir()
        sd = dataset / "000000"; sd.mkdir()
        _write_sample_json(sd, 0)
        # No melody MIDI written
        out = tmp_path / "sequences.json"
        counts = extract_sequences(str(dataset), str(out))
        assert counts["chord"] == 1
        assert counts["melody"] == 0

    def test_empty_dataset(self, tmp_path):
        dataset = tmp_path / "dataset"
        dataset.mkdir()
        out = tmp_path / "sequences.json"
        counts = extract_sequences(str(dataset), str(out))
        assert counts == {"chord": 0, "melody": 0}

    def test_key_in_output(self, tmp_path):
        dataset = tmp_path / "dataset"
        dataset.mkdir()
        sd = dataset / "000000"; sd.mkdir()
        _write_sample_json(sd, 0, key="Am")
        _write_melody_midi(sd)
        out = tmp_path / "sequences.json"
        extract_sequences(str(dataset), str(out))
        data = json.loads(out.read_text())
        assert data["chord"][0]["key"] == "Am"
        assert data["melody"][0]["key"] == "Am"
