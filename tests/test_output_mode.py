"""Tests for output_mode routing in writer.write_sample (D-47, D-48, D-66, R-P14).

All tests synthesize WAVs + MIDIs in tmp_path — zero FluidSynth, no network.
Mirrors the synth_sample fixture from test_writer.py.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict

import mido
import numpy as np
import pytest
import scipy.io.wavfile as wf
from pydub import AudioSegment

from musicgen.writer import write_sample

_LAYERS = ("beat", "melody", "harmony", "bassline")


def _write_silent_wav(path: str, duration_ms: int = 500) -> None:
    seg = AudioSegment.silent(duration=duration_ms, frame_rate=44100).set_channels(2)
    seg.export(path, format="wav")


def _write_tiny_midi(path: str, ticks_per_beat: int = 480) -> None:
    mf = mido.MidiFile(ticks_per_beat=ticks_per_beat, type=1)
    meta = mido.MidiTrack()
    meta.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(120), time=0))
    meta.append(mido.MetaMessage("end_of_track", time=0))
    mf.tracks.append(meta)
    notes = mido.MidiTrack()
    notes.append(mido.Message("note_on", note=60, velocity=100, time=0))
    notes.append(mido.Message("note_off", note=60, velocity=0, time=480))
    notes.append(mido.MetaMessage("end_of_track", time=0))
    mf.tracks.append(notes)
    mf.save(path)


@pytest.fixture
def synth_sample(tmp_path):
    """Produce a minimal working-dir layout for output_mode tests."""
    working = tmp_path / "working"
    working.mkdir()
    song_arrangement = ["intro"]
    part_durations_s = [0.5]

    stems_paths: Dict[str, Dict[str, str]] = {}
    midi_paths: Dict[str, Dict[str, str]] = {}
    for part in song_arrangement:
        part_dir = working / f"part-{part}"
        (part_dir / "stems").mkdir(parents=True)
        (part_dir / "midi").mkdir(parents=True)
        stems_paths[part] = {}
        midi_paths[part] = {}
        for layer in _LAYERS:
            wav_p = str(part_dir / "stems" / f"{layer}.wav")
            mid_p = str(part_dir / "midi" / f"{layer}.mid")
            _write_silent_wav(wav_p, duration_ms=500)
            _write_tiny_midi(mid_p)
            stems_paths[part][layer] = wav_p
            midi_paths[part][layer] = mid_p

    mix_working = str(working / "final_mix.wav")
    seg = AudioSegment.silent(duration=500, frame_rate=44100).set_channels(2)
    seg.export(mix_working, format="wav")

    annotation = {
        "seed": 1,
        "musicgen_version": "0.1.0",
        "tempo_bpm": 120,
        "song_arrangement": [{"part": "intro", "start_seconds": 0.0, "end_seconds": 0.5}],
        "mix": mix_working,
        "stems": {l: stems_paths["intro"][l] for l in _LAYERS},
        "midi": {l: midi_paths["intro"][l] for l in _LAYERS},
        "musicality_score": {"score": 0.5, "components": {}},
        "duration_seconds": 0.5,
        "fluidsynth_version": "test",
        "pre_roll_offset_seconds": None,
        "beat_times": [0.0, 0.5],
        "downbeat_times": [0.0],
    }

    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()

    return {
        "dataset_root": str(dataset_root),
        "annotation": annotation,
        "mix_working": mix_working,
        "stems_paths": stems_paths,
        "midi_paths": midi_paths,
        "song_arrangement": song_arrangement,
        "part_durations_s": part_durations_s,
    }


def _call_write_sample(synth, output_mode="full", pre_roll_offset_s=0.0, sample_index=0):
    return write_sample(
        synth["dataset_root"], sample_index, synth["annotation"],
        synth["mix_working"], synth["stems_paths"], synth["midi_paths"],
        synth["song_arrangement"],
        tempo_bpm=120, part_durations_s=synth["part_durations_s"],
        fluidsynth_version="test", split="train",
        output_mode=output_mode, pre_roll_offset_s=pre_roll_offset_s,
    )


class TestOutputModeFull:
    """full mode: all 10 files written."""

    def test_mix_exists(self, synth_sample):
        _call_write_sample(synth_sample)
        assert (Path(synth_sample["dataset_root"]) / "000000" / "mix.wav").is_file()

    def test_stems_exist(self, synth_sample):
        _call_write_sample(synth_sample)
        for layer in _LAYERS:
            assert (Path(synth_sample["dataset_root"]) / "000000" / "stems" / f"{layer}.wav").is_file()

    def test_midi_exist(self, synth_sample):
        _call_write_sample(synth_sample)
        for layer in _LAYERS:
            assert (Path(synth_sample["dataset_root"]) / "000000" / "midi" / f"{layer}.mid").is_file()

    def test_sample_json_exists(self, synth_sample):
        _call_write_sample(synth_sample)
        assert (Path(synth_sample["dataset_root"]) / "000000" / "sample.json").is_file()


class TestOutputModeMixOnly:
    """mix-only: only mix.wav + sample.json written; stems/ and midi/ absent."""

    def test_mix_exists(self, synth_sample):
        _call_write_sample(synth_sample, output_mode="mix-only")
        assert (Path(synth_sample["dataset_root"]) / "000000" / "mix.wav").is_file()

    def test_sample_json_exists(self, synth_sample):
        _call_write_sample(synth_sample, output_mode="mix-only")
        assert (Path(synth_sample["dataset_root"]) / "000000" / "sample.json").is_file()

    def test_stems_absent(self, synth_sample):
        _call_write_sample(synth_sample, output_mode="mix-only")
        stems_dir = Path(synth_sample["dataset_root"]) / "000000" / "stems"
        assert not stems_dir.is_dir() or not any(stems_dir.iterdir())

    def test_midi_absent(self, synth_sample):
        _call_write_sample(synth_sample, output_mode="mix-only")
        midi_dir = Path(synth_sample["dataset_root"]) / "000000" / "midi"
        assert not midi_dir.is_dir() or not any(midi_dir.iterdir())


class TestOutputModeStemsOnly:
    """stems-only: 4 stems + sample.json; no mix.wav, no midi/."""

    def test_stems_exist(self, synth_sample):
        _call_write_sample(synth_sample, output_mode="stems-only")
        for layer in _LAYERS:
            assert (Path(synth_sample["dataset_root"]) / "000000" / "stems" / f"{layer}.wav").is_file()

    def test_mix_absent(self, synth_sample):
        _call_write_sample(synth_sample, output_mode="stems-only")
        assert not (Path(synth_sample["dataset_root"]) / "000000" / "mix.wav").is_file()

    def test_sample_json_exists(self, synth_sample):
        _call_write_sample(synth_sample, output_mode="stems-only")
        assert (Path(synth_sample["dataset_root"]) / "000000" / "sample.json").is_file()


class TestOutputModeMidiOnly:
    """midi-only: 4 midi + sample.json; no mix.wav, no stems/."""

    def test_midi_exist(self, synth_sample):
        _call_write_sample(synth_sample, output_mode="midi-only")
        for layer in _LAYERS:
            assert (Path(synth_sample["dataset_root"]) / "000000" / "midi" / f"{layer}.mid").is_file()

    def test_mix_absent(self, synth_sample):
        _call_write_sample(synth_sample, output_mode="midi-only")
        assert not (Path(synth_sample["dataset_root"]) / "000000" / "mix.wav").is_file()

    def test_sample_json_exists(self, synth_sample):
        _call_write_sample(synth_sample, output_mode="midi-only")
        assert (Path(synth_sample["dataset_root"]) / "000000" / "sample.json").is_file()


class TestSampleJsonAlwaysWritten:
    """All output modes produce sample.json (the sentinel)."""

    @pytest.mark.parametrize("mode", ["full", "mix-only", "stems-only", "midi-only"])
    def test_sentinel_written_for_all_modes(self, synth_sample, mode):
        _call_write_sample(synth_sample, output_mode=mode)
        sentinel = Path(synth_sample["dataset_root"]) / "000000" / "sample.json"
        assert sentinel.is_file()


class TestPrerollOffsetApplied:
    """pre_roll_offset_s shifts beat_times and downbeat_times (D-53)."""

    def test_beat_times_shifted(self, synth_sample):
        _call_write_sample(synth_sample, pre_roll_offset_s=0.1)
        sentinel = Path(synth_sample["dataset_root"]) / "000000" / "sample.json"
        data = json.loads(sentinel.read_text())
        # Original beat_times=[0.0, 0.5]; offset=0.1 → [0.4] (0.0-0.1 < 0 removed)
        assert data["beat_times"] == pytest.approx([0.4], abs=1e-5)

    def test_downbeat_times_shifted(self, synth_sample):
        _call_write_sample(synth_sample, pre_roll_offset_s=0.1)
        sentinel = Path(synth_sample["dataset_root"]) / "000000" / "sample.json"
        data = json.loads(sentinel.read_text())
        # Original downbeat_times=[0.0]; offset=0.1 → [] (0.0-0.1 < 0 removed)
        assert data["downbeat_times"] == []

    def test_zero_offset_is_noop(self, synth_sample):
        _call_write_sample(synth_sample, pre_roll_offset_s=0.0)
        sentinel = Path(synth_sample["dataset_root"]) / "000000" / "sample.json"
        data = json.loads(sentinel.read_text())
        assert data["beat_times"] == pytest.approx([0.0, 0.5], abs=1e-5)
        assert data["downbeat_times"] == pytest.approx([0.0], abs=1e-5)
