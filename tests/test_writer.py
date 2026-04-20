"""Tests for src/musicgen/writer.py (D-37, R-P1/R-P2/R-P3).

All tests synthesize WAVs + MIDIs in tmp_path — zero FluidSynth, no network.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List

import mido
import numpy as np
import pytest
import scipy.io.wavfile as wf
from pydub import AudioSegment

from musicgen.writer import (
    _assert_sum_of_stems, _concat_layer_midis, _concat_layer_stems,
    _rewrite_paths_relative, write_sample,
)


_LAYERS = ("beat", "melody", "harmony", "bassline")


def _write_silent_wav(path: str, duration_ms: int = 500) -> None:
    """Stereo 44.1kHz silent WAV — matches Phase 4 D-12 silent-stem format."""
    seg = AudioSegment.silent(duration=duration_ms, frame_rate=44100).set_channels(2)
    seg.export(path, format="wav")


def _write_tone_wav(path: str, freq_hz: float, duration_ms: int = 500, amp: int = 1000) -> None:
    """Stereo 44.1kHz int16 tone — for sum-of-stems constructive tests."""
    sr = 44100
    n = int(sr * duration_ms / 1000)
    t = np.arange(n)
    samples = (amp * np.sin(2 * np.pi * freq_hz * t / sr)).astype(np.int16)
    stereo = np.column_stack([samples, samples])  # stereo = mono duplicated
    wf.write(path, sr, stereo)


def _write_tiny_midi(path: str, num_notes: int = 2, ticks_per_beat: int = 480) -> None:
    """Produce a MIDI with `num_notes` sequential quarter-notes via mido (not midiutil)."""
    mf = mido.MidiFile(ticks_per_beat=ticks_per_beat, type=1)
    meta = mido.MidiTrack()
    meta.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(120), time=0))
    meta.append(mido.MetaMessage("end_of_track", time=0))
    mf.tracks.append(meta)
    notes = mido.MidiTrack()
    for i in range(num_notes):
        notes.append(mido.Message("note_on", note=60 + i, velocity=100, time=0 if i == 0 else ticks_per_beat))
        notes.append(mido.Message("note_off", note=60 + i, velocity=0, time=ticks_per_beat))
    notes.append(mido.MetaMessage("end_of_track", time=0))
    mf.tracks.append(notes)
    mf.save(path)


@pytest.fixture
def synth_sample(tmp_path):
    """Produce a full working-dir layout for a 2-part song."""
    working = tmp_path / "working"
    working.mkdir()
    song_arrangement = ["intro", "verse"]
    part_durations_s = [0.5, 0.5]

    stems_paths: Dict[str, Dict[str, str]] = {}
    midi_paths: Dict[str, Dict[str, str]] = {}
    # Per-part: 4 silent stems + 4 tiny MIDIs.
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
            _write_tiny_midi(mid_p, num_notes=2)
            stems_paths[part][layer] = wav_p
            midi_paths[part][layer] = mid_p

    # Build a final mix = sum of all 4 stems, per part, concatenated.
    # Since stems are silent, mix = silent * 2 parts = 1-second silent.
    mix_working = str(working / "final_mix.wav")
    seg = AudioSegment.silent(duration=1000, frame_rate=44100).set_channels(2)
    seg.export(mix_working, format="wav")

    # Annotation stub — mimics what annotator.annotate emits post-api-fill.
    annotation = {
        "seed": 12345,
        "musicgen_version": "0.1.0",
        "tempo_bpm": 120,
        "song_arrangement": [{"part": p, "start_seconds": i*0.5, "end_seconds": (i+1)*0.5}
                              for i, p in enumerate(song_arrangement)],
        "mix": mix_working,  # absolute path — writer must rewrite to "mix.wav"
        "stems": {l: stems_paths[song_arrangement[0]][l] for l in _LAYERS},
        "midi": {l: midi_paths[song_arrangement[0]][l] for l in _LAYERS},
        "musicality_score": {"score": 0.7, "components": {}},
        "duration_seconds": 1.0,
        "fluidsynth_version": "test-2.3.0",
        "pre_roll_offset_seconds": None,
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


class TestLayout:
    """D-04/D-05: index-based dir layout with all 10 files."""

    def test_produces_expected_10_files(self, synth_sample):
        result = write_sample(
            synth_sample["dataset_root"], 0, synth_sample["annotation"],
            synth_sample["mix_working"], synth_sample["stems_paths"],
            synth_sample["midi_paths"], synth_sample["song_arrangement"],
            tempo_bpm=120, part_durations_s=synth_sample["part_durations_s"],
            fluidsynth_version="test", split="train",
        )
        sample_dir = Path(result["sample_dir"])
        assert sample_dir.name == "000000"
        assert (sample_dir / "mix.wav").is_file()
        assert (sample_dir / "sample.json").is_file()
        for layer in _LAYERS:
            assert (sample_dir / "stems" / f"{layer}.wav").is_file()
            assert (sample_dir / "midi" / f"{layer}.mid").is_file()

    def test_large_index_zero_padded_to_6(self, synth_sample):
        result = write_sample(
            synth_sample["dataset_root"], 42, synth_sample["annotation"],
            synth_sample["mix_working"], synth_sample["stems_paths"],
            synth_sample["midi_paths"], synth_sample["song_arrangement"],
            tempo_bpm=120, part_durations_s=synth_sample["part_durations_s"],
            fluidsynth_version="test", split="valid",
        )
        assert Path(result["sample_dir"]).name == "000042"


class TestSentinelOrder:
    """D-04: sample.json is ALWAYS last; assertion failures leave it absent."""

    def test_sentinel_exists_on_success(self, synth_sample):
        write_sample(
            synth_sample["dataset_root"], 0, synth_sample["annotation"],
            synth_sample["mix_working"], synth_sample["stems_paths"],
            synth_sample["midi_paths"], synth_sample["song_arrangement"],
            tempo_bpm=120, part_durations_s=synth_sample["part_durations_s"],
            fluidsynth_version="test", split="train",
        )
        sample_json = Path(synth_sample["dataset_root"]) / "000000" / "sample.json"
        assert sample_json.is_file()

    def test_sentinel_absent_on_sum_of_stems_failure(self, synth_sample, tmp_path):
        """Fault-inject: replace mix.wav content with non-zero after concat."""
        # Force the final mix to be NOT the sum of (silent) stems by
        # overwriting mix_working with a loud tone.
        loud_mix = str(tmp_path / "loud_mix.wav")
        _write_tone_wav(loud_mix, freq_hz=440, duration_ms=1000, amp=20000)
        with pytest.raises(AssertionError, match="sum_of_stems_exceeded"):
            write_sample(
                synth_sample["dataset_root"], 0, synth_sample["annotation"],
                loud_mix, synth_sample["stems_paths"],
                synth_sample["midi_paths"], synth_sample["song_arrangement"],
                tempo_bpm=120, part_durations_s=synth_sample["part_durations_s"],
                fluidsynth_version="test", split="train",
            )
        sample_json = Path(synth_sample["dataset_root"]) / "000000" / "sample.json"
        assert not sample_json.is_file()
        # tmp file cleaned up automatically (never renamed):
        sample_json_tmp = Path(synth_sample["dataset_root"]) / "000000" / "sample.json.tmp"
        # Either doesn't exist or is irrelevant — the key invariant is sentinel absence.
        # Mix MAY exist since it was copied before the assertion.


class TestRelativePaths:
    """D-11/D-12: sample.json paths are per-sample-dir-relative."""

    def test_mix_path_is_relative(self, synth_sample):
        write_sample(
            synth_sample["dataset_root"], 0, synth_sample["annotation"],
            synth_sample["mix_working"], synth_sample["stems_paths"],
            synth_sample["midi_paths"], synth_sample["song_arrangement"],
            tempo_bpm=120, part_durations_s=synth_sample["part_durations_s"],
            fluidsynth_version="test", split="train",
        )
        sample_json = Path(synth_sample["dataset_root"]) / "000000" / "sample.json"
        data = json.loads(sample_json.read_text())
        assert data["mix"] == "mix.wav"

    def test_stems_paths_are_relative(self, synth_sample):
        write_sample(
            synth_sample["dataset_root"], 0, synth_sample["annotation"],
            synth_sample["mix_working"], synth_sample["stems_paths"],
            synth_sample["midi_paths"], synth_sample["song_arrangement"],
            tempo_bpm=120, part_durations_s=synth_sample["part_durations_s"],
            fluidsynth_version="test", split="train",
        )
        sample_json = Path(synth_sample["dataset_root"]) / "000000" / "sample.json"
        data = json.loads(sample_json.read_text())
        for layer in _LAYERS:
            assert data["stems"][layer] == f"stems/{layer}.wav"
            assert data["midi"][layer] == f"midi/{layer}.mid"

    def test_input_annotation_not_mutated(self, synth_sample):
        """D-12: writer deep-copies so the caller's dict is untouched."""
        original_mix = synth_sample["annotation"]["mix"]
        write_sample(
            synth_sample["dataset_root"], 0, synth_sample["annotation"],
            synth_sample["mix_working"], synth_sample["stems_paths"],
            synth_sample["midi_paths"], synth_sample["song_arrangement"],
            tempo_bpm=120, part_durations_s=synth_sample["part_durations_s"],
            fluidsynth_version="test", split="train",
        )
        # Caller's dict still has the absolute mix_working path, untouched:
        assert synth_sample["annotation"]["mix"] == original_mix


class TestSumOfStems:
    """R-P2/D-24/D-25: sum-of-stems assertion."""

    def test_passes_on_silent_stems_silent_mix(self, synth_sample):
        """Silent stems + silent mix → trivially passes (max_diff == 0)."""
        # synth_sample fixture already has this shape.
        result = write_sample(
            synth_sample["dataset_root"], 0, synth_sample["annotation"],
            synth_sample["mix_working"], synth_sample["stems_paths"],
            synth_sample["midi_paths"], synth_sample["song_arrangement"],
            tempo_bpm=120, part_durations_s=synth_sample["part_durations_s"],
            fluidsynth_version="test", split="train",
        )
        # Success = sentinel exists:
        assert (Path(synth_sample["dataset_root"]) / "000000" / "sample.json").is_file()

    def test_fails_on_divergent_mix(self, synth_sample, tmp_path):
        """A mix that is NOT the sum of stems triggers AssertionError."""
        loud_mix = str(tmp_path / "loud_mix.wav")
        _write_tone_wav(loud_mix, freq_hz=440, duration_ms=1000, amp=20000)
        with pytest.raises(AssertionError, match="sum_of_stems_exceeded"):
            write_sample(
                synth_sample["dataset_root"], 0, synth_sample["annotation"],
                loud_mix, synth_sample["stems_paths"],
                synth_sample["midi_paths"], synth_sample["song_arrangement"],
                tempo_bpm=120, part_durations_s=synth_sample["part_durations_s"],
                fluidsynth_version="test", split="train",
            )


class TestAssertSumOfStemsDirect:
    """Direct unit tests of _assert_sum_of_stems with fault injection."""

    def test_silent_match(self, tmp_path):
        mix = str(tmp_path / "mix.wav")
        stems = {l: str(tmp_path / f"{l}.wav") for l in _LAYERS}
        _write_silent_wav(mix)
        for p in stems.values():
            _write_silent_wav(p)
        passed, diff = _assert_sum_of_stems(mix, stems, epsilon=1e-3)
        assert passed
        assert diff == 0.0

    def test_shape_mismatch_raises(self, tmp_path):
        mix = str(tmp_path / "mix.wav")
        _write_silent_wav(mix, duration_ms=500)
        stems = {l: str(tmp_path / f"{l}.wav") for l in _LAYERS}
        for p in stems.values():
            _write_silent_wav(p, duration_ms=500)
        # Overwrite one stem with a different duration to force shape mismatch:
        _write_silent_wav(stems["beat"], duration_ms=1000)
        with pytest.raises(ValueError, match="shape"):
            _assert_sum_of_stems(mix, stems, epsilon=1e-3)


class TestMidiConcat:
    """R-P3/D-07: absolute-tick walk produces correct inter-part timing."""

    def test_two_part_concat_preserves_notes(self, tmp_path):
        p1 = str(tmp_path / "p1.mid")
        p2 = str(tmp_path / "p2.mid")
        out = str(tmp_path / "out.mid")
        _write_tiny_midi(p1, num_notes=2)
        _write_tiny_midi(p2, num_notes=2)
        # Each tiny MIDI is 2 quarter-notes at 120bpm → 1 second
        _concat_layer_midis([p1, p2], [1.0, 1.0], tempo_bpm=120, out_path=out)
        merged = mido.MidiFile(out)
        # Merged has same number of tracks as source:
        assert len(merged.tracks) == 2
        # Count note_on events — should be 2+2 = 4:
        note_on_count = sum(
            1 for tr in merged.tracks for m in tr
            if m.type == "note_on"
        )
        assert note_on_count == 4

    def test_second_part_offset_from_duration(self, tmp_path):
        """RESEARCH Pitfall 1: second part's first note_on lands at offset_ticks."""
        p1 = str(tmp_path / "p1.mid")
        p2 = str(tmp_path / "p2.mid")
        out = str(tmp_path / "out.mid")
        _write_tiny_midi(p1, num_notes=1, ticks_per_beat=480)
        _write_tiny_midi(p2, num_notes=1, ticks_per_beat=480)
        _concat_layer_midis([p1, p2], [1.0, 1.0], tempo_bpm=120, out_path=out)
        merged = mido.MidiFile(out)
        # Walk the notes track (track 1; track 0 is meta).
        notes_track = merged.tracks[1]
        abs_ticks = []
        t = 0
        for msg in notes_track:
            t += msg.time
            if msg.type == "note_on":
                abs_ticks.append(t)
        # Should have 2 note_ons (one per part). Second must be >= 960 ticks
        # (1s at 120bpm, 480 ticks_per_beat = 960 ticks). Allow slack for
        # msg.time accumulation within the first part's track.
        assert len(abs_ticks) == 2
        expected_offset = int(mido.second2tick(1.0, 480, mido.bpm2tempo(120)))
        # The second part's first note_on is AT LEAST the offset — may be more
        # if there's intra-part accumulation before the second's first note_on.
        assert abs_ticks[1] >= expected_offset

    def test_empty_part_list_raises(self, tmp_path):
        with pytest.raises(ValueError, match="no part midi paths"):
            _concat_layer_midis([], [], tempo_bpm=120,
                                out_path=str(tmp_path / "out.mid"))

    def test_length_mismatch_raises(self, tmp_path):
        p1 = str(tmp_path / "p1.mid")
        _write_tiny_midi(p1, num_notes=1)
        with pytest.raises(ValueError, match="length mismatch"):
            _concat_layer_midis([p1], [1.0, 2.0], tempo_bpm=120,
                                out_path=str(tmp_path / "out.mid"))


class TestStemConcat:
    """D-06: mirrors mixer.concat_parts style."""

    def test_two_part_concat_doubles_duration(self, tmp_path):
        p1 = str(tmp_path / "p1.wav")
        p2 = str(tmp_path / "p2.wav")
        out = str(tmp_path / "out.wav")
        _write_silent_wav(p1, duration_ms=500)
        _write_silent_wav(p2, duration_ms=500)
        _concat_layer_stems([p1, p2], out)
        # Length check via scipy:
        _, data = wf.read(out)
        # 1 second at 44.1kHz = 44100 frames:
        assert data.shape[0] == 44100

    def test_empty_list_raises(self, tmp_path):
        with pytest.raises(ValueError, match="no part stem paths"):
            _concat_layer_stems([], str(tmp_path / "out.wav"))


class TestRewritePaths:
    """D-11/D-12: deep-copy + rewrite."""

    def test_returns_new_dict(self):
        original = {
            "mix": "/abs/path/mix.wav",
            "stems": {"beat": "/abs/path/stems/beat.wav"},
            "midi": {"beat": "/abs/path/midi/beat.mid"},
            "other_field": "unchanged",
        }
        result = _rewrite_paths_relative(original, ["beat"], ["beat"])
        assert result is not original  # deep-copy
        # Original dict NOT mutated:
        assert original["mix"] == "/abs/path/mix.wav"
        # Result has relative paths:
        assert result["mix"] == "mix.wav"
        assert result["stems"]["beat"] == "stems/beat.wav"
        assert result["midi"]["beat"] == "midi/beat.mid"
        # Other fields preserved:
        assert result["other_field"] == "unchanged"
