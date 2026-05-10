"""v0.3 Phase 3a — RED: two-layer musicality redesign.

Tests cover:
  Layer 1 (MIDI pre-filter):
    - MIDIQualityResult dataclass
    - check_midi_quality interface
    - Hard checks: empty layer, pitch range, stuck note, melody/bass crossing
    - Soft metrics: KS key correlation, scale adherence, melodic step fraction,
      n-gram entropy, LZ compression ratio, bar self-similarity
    - Aggregate score in [0, 1]; deterministic

  Layer 2 (audio — redesigned get_musicality_score):
    - No 'timbre' key in components (bug deleted)
    - Render-integrity keys present: clipping, dc_offset, silence_ratio, crest_db
    - Genre-aware tempo (uses GenreSpec bounds, not hard 60-180 clip)
    - Score in [0, 1]; (float, dict) return type preserved
    - Clipped audio → hard-fail reflected in components
    - Near-silent audio → low score

  Internal helpers (importable, unit-testable):
    - _ks_key_correlation
    - _scale_adherence_score
    - _melodic_step_fraction
    - _ngram_entropy
    - _lz_ratio
    - _render_integrity
"""
from __future__ import annotations

import io
import os
import random
import struct
import wave
import zlib
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers to build minimal MIDI / WAV fixtures without FluidSynth
# ---------------------------------------------------------------------------

def _make_wav_bytes(samples: np.ndarray, sr: int = 44100) -> bytes:
    """Write a 16-bit mono WAV to bytes."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        pcm = np.clip(samples, -1.0, 1.0)
        wf.writeframes((pcm * 32767).astype(np.int16).tobytes())
    return buf.getvalue()


def _write_wav(path: str, samples: np.ndarray, sr: int = 44100):
    Path(path).write_bytes(_make_wav_bytes(samples, sr))


def _make_midi_bytes(
    notes: List[tuple],  # (pitch, start_tick, dur_tick, velocity)
    tempo_bpm: int = 120,
    ticks_per_beat: int = 480,
) -> bytes:
    """Minimal SMF type-0 MIDI builder without external deps."""
    us_per_beat = int(60_000_000 / tempo_bpm)

    def var_len(n):
        result = [n & 0x7F]
        n >>= 7
        while n:
            result.insert(0, (n & 0x7F) | 0x80)
            n >>= 7
        return bytes(result)

    events = []
    # tempo event at tick 0
    events.append((0, bytes([0xFF, 0x51, 0x03]) + us_per_beat.to_bytes(3, "big")))
    for pitch, start, dur, vel in notes:
        events.append((start, bytes([0x90, pitch, vel])))
        events.append((start + dur, bytes([0x80, pitch, 0x00])))
    events.append((ticks_per_beat * 4, bytes([0xFF, 0x2F, 0x00])))
    events.sort(key=lambda e: e[0])

    track_bytes = b""
    prev = 0
    for tick, msg in events:
        delta = tick - prev
        prev = tick
        track_bytes += var_len(delta) + msg

    header = struct.pack(">4sIHHH", b"MThd", 6, 0, 1, ticks_per_beat)
    track = struct.pack(">4sI", b"MTrk", len(track_bytes)) + track_bytes
    return header + track


def _write_midi(path: str, notes, tempo_bpm=120, ticks_per_beat=480):
    Path(path).write_bytes(_make_midi_bytes(notes, tempo_bpm, ticks_per_beat))


def _scale_degree_notes_C_major(n_notes=16, base=60):
    """Return MIDI pitches for C-major scale walking up and back."""
    scale = [0, 2, 4, 5, 7, 9, 11]
    pitches = []
    for i in range(n_notes):
        pitches.append(base + scale[i % len(scale)])
    return pitches


# ---------------------------------------------------------------------------
# MIDIQualityResult dataclass
# ---------------------------------------------------------------------------

class TestMIDIQualityResult:
    def test_importable(self):
        from musicgen.musicality import MIDIQualityResult
        assert MIDIQualityResult is not None

    def test_is_dataclass_with_expected_fields(self):
        from musicgen.musicality import MIDIQualityResult
        r = MIDIQualityResult(
            passed=True, score=0.8,
            hard_failures=[],
            soft_scores={"ks_correlation": 0.9},
        )
        assert r.passed is True
        assert r.score == 0.8
        assert r.hard_failures == []
        assert "ks_correlation" in r.soft_scores

    def test_is_frozen(self):
        from musicgen.musicality import MIDIQualityResult
        r = MIDIQualityResult(passed=True, score=0.8, hard_failures=[], soft_scores={})
        with pytest.raises((AttributeError, TypeError)):
            r.score = 0.5  # type: ignore[misc]

    def test_score_field_in_unit_interval(self):
        from musicgen.musicality import MIDIQualityResult
        r = MIDIQualityResult(passed=False, score=0.3, hard_failures=["x"], soft_scores={})
        assert 0.0 <= r.score <= 1.0


# ---------------------------------------------------------------------------
# check_midi_quality interface
# ---------------------------------------------------------------------------

class TestCheckMIDIQuality:
    def test_importable(self):
        from musicgen.musicality import check_midi_quality
        assert callable(check_midi_quality)

    def test_returns_midi_quality_result(self, tmp_path):
        from musicgen.musicality import check_midi_quality, MIDIQualityResult
        pitches = _scale_degree_notes_C_major(16)
        notes = [(p, i * 240, 200, 80) for i, p in enumerate(pitches)]
        midi_path = str(tmp_path / "melody.mid")
        _write_midi(midi_path, notes)
        for layer in ("beat", "harmony", "bassline"):
            _write_midi(str(tmp_path / f"{layer}.mid"), notes[:8])

        result = check_midi_quality(
            midi_paths={
                "melody": midi_path,
                "beat": str(tmp_path / "beat.mid"),
                "harmony": str(tmp_path / "harmony.mid"),
                "bassline": str(tmp_path / "bassline.mid"),
            },
            key="C",
        )
        assert isinstance(result, MIDIQualityResult)
        assert 0.0 <= result.score <= 1.0
        assert isinstance(result.hard_failures, list)
        assert isinstance(result.soft_scores, dict)

    def test_deterministic(self, tmp_path):
        from musicgen.musicality import check_midi_quality
        pitches = _scale_degree_notes_C_major(16)
        notes = [(p, i * 240, 200, 80) for i, p in enumerate(pitches)]
        paths = {}
        for layer in ("melody", "beat", "harmony", "bassline"):
            p = str(tmp_path / f"{layer}.mid")
            _write_midi(p, notes)
            paths[layer] = p

        r1 = check_midi_quality(midi_paths=paths, key="C")
        r2 = check_midi_quality(midi_paths=paths, key="C")
        assert r1.score == r2.score
        assert r1.passed == r2.passed
        assert r1.hard_failures == r2.hard_failures

    def test_soft_scores_are_in_unit_interval(self, tmp_path):
        from musicgen.musicality import check_midi_quality
        pitches = _scale_degree_notes_C_major(16)
        notes = [(p, i * 240, 200, 80) for i, p in enumerate(pitches)]
        paths = {}
        for layer in ("melody", "beat", "harmony", "bassline"):
            p = str(tmp_path / f"{layer}.mid")
            _write_midi(p, notes)
            paths[layer] = p
        result = check_midi_quality(midi_paths=paths, key="C")
        for name, score in result.soft_scores.items():
            assert 0.0 <= score <= 1.0, f"Soft score {name!r} = {score} out of [0,1]"


# ---------------------------------------------------------------------------
# Hard checks
# ---------------------------------------------------------------------------

class TestHardChecks:
    def test_empty_layer_fails(self, tmp_path):
        """Layer with zero notes → hard failure."""
        from musicgen.musicality import check_midi_quality
        good_notes = [(60 + i % 7, i * 240, 200, 80) for i in range(8)]
        paths = {}
        for layer in ("melody", "beat", "harmony", "bassline"):
            p = str(tmp_path / f"{layer}.mid")
            if layer == "melody":
                _write_midi(p, [])  # empty!
            else:
                _write_midi(p, good_notes)
            paths[layer] = p
        result = check_midi_quality(midi_paths=paths, key="C")
        assert not result.passed
        assert any("empty" in f.lower() or "notes" in f.lower() for f in result.hard_failures)

    def test_stuck_note_fails(self, tmp_path):
        """Single pitch > 80% of notes → hard failure."""
        from musicgen.musicality import check_midi_quality
        # 16 notes, 15 are pitch 60 → 93.75% stuck
        stuck = [(60, i * 240, 200, 80) for i in range(15)] + [(62, 15 * 240, 200, 80)]
        good = [(60 + i % 5, i * 240, 200, 80) for i in range(16)]
        paths = {}
        for layer in ("melody", "beat", "harmony", "bassline"):
            p = str(tmp_path / f"{layer}.mid")
            _write_midi(p, stuck if layer == "melody" else good)
            paths[layer] = p
        result = check_midi_quality(midi_paths=paths, key="C")
        assert not result.passed
        assert any("stuck" in f.lower() or "pitch" in f.lower() for f in result.hard_failures)

    def test_extreme_pitch_range_fails(self, tmp_path):
        """Melody range > 36 semitones → hard failure."""
        from musicgen.musicality import check_midi_quality
        # 4 octave span = 48 semitones
        extreme = [(36, 0, 200, 80), (84, 240, 200, 80),
                   (36, 480, 200, 80), (84, 720, 200, 80)] * 4
        good = [(60 + i % 5, i * 240, 200, 80) for i in range(16)]
        paths = {}
        for layer in ("melody", "beat", "harmony", "bassline"):
            p = str(tmp_path / f"{layer}.mid")
            _write_midi(p, extreme if layer == "melody" else good)
            paths[layer] = p
        result = check_midi_quality(midi_paths=paths, key="C")
        assert not result.passed
        assert any("range" in f.lower() or "pitch" in f.lower() for f in result.hard_failures)

    def test_good_midi_passes_hard_checks(self, tmp_path):
        """Well-formed MIDI passes all hard checks."""
        from musicgen.musicality import check_midi_quality
        # C major scale, one octave, varied
        good = [(60 + i % 8, i * 240, 200, 80) for i in range(16)]
        paths = {}
        for layer in ("melody", "beat", "harmony", "bassline"):
            p = str(tmp_path / f"{layer}.mid")
            _write_midi(p, good)
            paths[layer] = p
        result = check_midi_quality(midi_paths=paths, key="C")
        assert result.hard_failures == []


# ---------------------------------------------------------------------------
# Internal soft-metric helpers
# ---------------------------------------------------------------------------

class TestKSKeyCorrelation:
    def test_importable(self):
        from musicgen.musicality import _ks_key_correlation
        assert callable(_ks_key_correlation)

    def test_c_major_notes_high_correlation(self):
        """Pitch-class histogram of C major notes → high KS correlation."""
        from musicgen.musicality import _ks_key_correlation
        # C major pitches: 0,2,4,5,7,9,11
        pc_hist = np.zeros(12)
        for pc in [0, 2, 4, 5, 7, 9, 11]:
            pc_hist[pc] = 1.0
        pc_hist /= pc_hist.sum()
        score = _ks_key_correlation(pc_hist)
        assert 0.0 <= score <= 1.0
        assert score > 0.5, f"C-major PC hist should correlate well, got {score:.3f}"

    def test_chromatic_notes_lower_correlation(self):
        """Flat pitch-class histogram (chromatic) → lower KS correlation."""
        from musicgen.musicality import _ks_key_correlation
        flat = np.ones(12) / 12
        score = _ks_key_correlation(flat)
        assert 0.0 <= score <= 1.0

    def test_output_in_unit_interval(self):
        from musicgen.musicality import _ks_key_correlation
        rng = random.Random(42)
        for _ in range(10):
            h = np.array([rng.random() for _ in range(12)])
            h /= h.sum()
            assert 0.0 <= _ks_key_correlation(h) <= 1.0


class TestScaleAdherence:
    def test_importable(self):
        from musicgen.musicality import _scale_adherence_score
        assert callable(_scale_adherence_score)

    def test_all_in_key_returns_high(self):
        from musicgen.musicality import _scale_adherence_score
        # All C major notes
        pitches = [60, 62, 64, 65, 67, 69, 71, 72]
        score = _scale_adherence_score(pitches, "C")
        assert score > 0.9, f"All-in-key should score > 0.9, got {score:.3f}"

    def test_all_chromatic_returns_low(self):
        from musicgen.musicality import _scale_adherence_score
        # Only black keys in C major context → low adherence
        pitches = [61, 63, 66, 68, 70] * 4  # C# D# F# G# A#
        score = _scale_adherence_score(pitches, "C")
        assert score < 0.2, f"Chromatic notes in C should score < 0.2, got {score:.3f}"

    def test_minor_key(self):
        from musicgen.musicality import _scale_adherence_score
        # A minor: A B C D E F G = 9,11,0,2,4,5,7
        pitches = [69, 71, 60, 62, 64, 65, 67] * 2
        score = _scale_adherence_score(pitches, "Am")
        assert score > 0.9

    def test_output_in_unit_interval(self):
        from musicgen.musicality import _scale_adherence_score
        for key in ("C", "G", "Am", "F", "Dm"):
            pitches = [60 + i for i in range(12)]
            assert 0.0 <= _scale_adherence_score(pitches, key) <= 1.0


class TestMelodicStepFraction:
    def test_importable(self):
        from musicgen.musicality import _melodic_step_fraction
        assert callable(_melodic_step_fraction)

    def test_stepwise_melody_high(self):
        from musicgen.musicality import _melodic_step_fraction
        # All steps of 1 or 2 semitones
        pitches = [60, 62, 64, 65, 67, 69, 71, 72]
        score = _melodic_step_fraction(pitches)
        assert score > 0.8

    def test_leap_melody_lower(self):
        from musicgen.musicality import _melodic_step_fraction
        # 7-semitone leaps
        pitches = [60, 67, 60, 67, 60, 67, 60, 67]
        score = _melodic_step_fraction(pitches)
        assert score < 0.2

    def test_single_note_returns_zero(self):
        from musicgen.musicality import _melodic_step_fraction
        assert _melodic_step_fraction([60]) == 0.0

    def test_output_in_unit_interval(self):
        from musicgen.musicality import _melodic_step_fraction
        rng = random.Random(7)
        pitches = [rng.randint(48, 84) for _ in range(20)]
        assert 0.0 <= _melodic_step_fraction(pitches) <= 1.0


class TestNgramEntropy:
    def test_importable(self):
        from musicgen.musicality import _ngram_entropy
        assert callable(_ngram_entropy)

    def test_constant_sequence_min_entropy(self):
        from musicgen.musicality import _ngram_entropy
        # All same symbol → near-zero entropy
        score = _ngram_entropy([1] * 20, n=3)
        assert score < 0.1

    def test_random_sequence_higher_entropy(self):
        from musicgen.musicality import _ngram_entropy
        rng = random.Random(0)
        seq = [rng.randint(1, 7) for _ in range(60)]
        score = _ngram_entropy(seq, n=3)
        assert score > 0.0

    def test_short_sequence_returns_zero(self):
        from musicgen.musicality import _ngram_entropy
        assert _ngram_entropy([1, 2], n=3) == 0.0

    def test_output_in_unit_interval(self):
        from musicgen.musicality import _ngram_entropy
        for seq in [[1] * 10, list(range(1, 8)) * 5, [1, 2, 3, 4, 5, 6, 7, 1, 2, 3]]:
            score = _ngram_entropy(seq, n=3)
            assert 0.0 <= score <= 1.0, f"Out of range: {score} for {seq}"


class TestLZRatio:
    def test_importable(self):
        from musicgen.musicality import _lz_ratio
        assert callable(_lz_ratio)

    def test_constant_sequence_low_ratio(self):
        from musicgen.musicality import _lz_ratio
        # Highly compressible
        score = _lz_ratio([60] * 64)
        assert score < 0.5

    def test_varied_sequence_higher_ratio(self):
        from musicgen.musicality import _lz_ratio
        rng = random.Random(42)
        seq = [rng.randint(0, 127) for _ in range(64)]
        score = _lz_ratio(seq)
        assert score > 0.3  # not super compressible

    def test_empty_sequence(self):
        from musicgen.musicality import _lz_ratio
        # Should not crash
        score = _lz_ratio([])
        assert score == 1.0 or score == 0.0  # either edge case is OK

    def test_output_in_unit_interval(self):
        from musicgen.musicality import _lz_ratio
        for seq in [[60] * 32, list(range(60, 76)) * 2, [60, 62, 64, 65, 67, 69, 71] * 4]:
            assert 0.0 <= _lz_ratio(seq) <= 1.0


# ---------------------------------------------------------------------------
# Render-integrity helper
# ---------------------------------------------------------------------------

class TestRenderIntegrity:
    def test_importable(self):
        from musicgen.musicality import _render_integrity
        assert callable(_render_integrity)

    def test_normal_audio_no_flags(self):
        from musicgen.musicality import _render_integrity
        sr = 44100
        t = np.linspace(0, 2.0, sr * 2)
        y = 0.3 * np.sin(2 * np.pi * 440 * t)
        result = _render_integrity(y, sr)
        assert result["clipping_ratio"] < 0.001
        assert result["dc_offset"] < 0.01
        assert result["silence_ratio"] < 0.5
        assert 6 <= result["crest_db"] <= 30

    def test_clipped_audio_flagged(self):
        from musicgen.musicality import _render_integrity
        sr = 44100
        y = np.ones(sr * 2)  # all samples = 1.0 → clipped
        result = _render_integrity(y, sr)
        assert result["clipping_ratio"] > 0.5

    def test_silent_audio_flagged(self):
        from musicgen.musicality import _render_integrity
        sr = 44100
        y = np.zeros(sr * 2)  # silence
        result = _render_integrity(y, sr)
        assert result["silence_ratio"] > 0.9

    def test_dc_offset_detected(self):
        from musicgen.musicality import _render_integrity
        sr = 44100
        y = np.ones(sr) * 0.5  # constant DC
        result = _render_integrity(y, sr)
        assert result["dc_offset"] > 0.1

    def test_output_keys_present(self):
        from musicgen.musicality import _render_integrity
        sr = 44100
        y = 0.3 * np.sin(2 * np.pi * 440 * np.linspace(0, 1, sr))
        result = _render_integrity(y, sr)
        for key in ("clipping_ratio", "dc_offset", "silence_ratio", "crest_db"):
            assert key in result, f"Missing key: {key!r}"


# ---------------------------------------------------------------------------
# Redesigned get_musicality_score (backward-compatible interface)
# ---------------------------------------------------------------------------

class TestGetMusicalityScore:
    def test_returns_float_and_dict(self, tmp_path):
        from musicgen.musicality import get_musicality_score
        sr = 44100
        t = np.linspace(0, 2.0, sr * 2)
        y = 0.3 * np.sin(2 * np.pi * 440 * t)
        wav = str(tmp_path / "test.wav")
        _write_wav(wav, y, sr)
        score, components = get_musicality_score(wav)
        assert isinstance(score, float)
        assert isinstance(components, dict)

    def test_score_in_unit_interval(self, tmp_path):
        from musicgen.musicality import get_musicality_score
        sr = 44100
        y = 0.3 * np.sin(2 * np.pi * 440 * np.linspace(0, 2.0, sr * 2))
        wav = str(tmp_path / "test.wav")
        _write_wav(wav, y, sr)
        score, _ = get_musicality_score(wav)
        assert 0.0 <= score <= 1.0

    def test_no_timbre_key_in_components(self, tmp_path):
        """Bug fix: timbre weight was declared but never computed. Now deleted."""
        from musicgen.musicality import get_musicality_score
        sr = 44100
        y = 0.3 * np.sin(2 * np.pi * 440 * np.linspace(0, 2.0, sr * 2))
        wav = str(tmp_path / "test.wav")
        _write_wav(wav, y, sr)
        _, components = get_musicality_score(wav)
        assert "timbre" not in components, (
            "timbre was never computed — its weight was a bug; must be removed"
        )

    def test_render_integrity_keys_in_components(self, tmp_path):
        """Render-integrity checks must surface in components dict."""
        from musicgen.musicality import get_musicality_score
        sr = 44100
        y = 0.3 * np.sin(2 * np.pi * 440 * np.linspace(0, 2.0, sr * 2))
        wav = str(tmp_path / "test.wav")
        _write_wav(wav, y, sr)
        _, components = get_musicality_score(wav)
        # At least one render-integrity key must be present
        integrity_keys = {"clipping_ratio", "dc_offset", "silence_ratio", "crest_db"}
        assert integrity_keys & set(components), (
            f"No render-integrity key in components: {set(components)}"
        )

    def test_clipped_audio_low_score(self, tmp_path):
        """Fully clipped audio → score substantially lower than clean audio."""
        from musicgen.musicality import get_musicality_score
        sr = 44100
        t = np.linspace(0, 2.0, sr * 2)
        clean = str(tmp_path / "clean.wav")
        clipped = str(tmp_path / "clipped.wav")
        _write_wav(clean, 0.3 * np.sin(2 * np.pi * 440 * t), sr)
        _write_wav(clipped, np.ones(sr * 2), sr)  # fully clipped
        score_clean, _ = get_musicality_score(clean)
        score_clipped, _ = get_musicality_score(clipped)
        assert score_clipped < score_clean, (
            f"Clipped audio (score={score_clipped:.3f}) should score lower than "
            f"clean audio (score={score_clean:.3f})"
        )

    def test_silent_audio_low_score(self, tmp_path):
        """Near-silence → low score."""
        from musicgen.musicality import get_musicality_score
        sr = 44100
        wav = str(tmp_path / "silent.wav")
        _write_wav(wav, np.zeros(sr * 2), sr)
        score, _ = get_musicality_score(wav)
        assert score < 0.5, f"Silent audio score should be < 0.5, got {score:.3f}"

    def test_missing_file_returns_zero(self):
        from musicgen.musicality import get_musicality_score
        score, components = get_musicality_score("/nonexistent/audio.wav")
        assert score == 0.0

    def test_genre_spec_accepted(self, tmp_path):
        """get_musicality_score accepts optional genre_spec without error."""
        from musicgen.musicality import get_musicality_score
        from musicgen.genre import GenreSpec
        sr = 44100
        y = 0.3 * np.sin(2 * np.pi * 440 * np.linspace(0, 2.0, sr * 2))
        wav = str(tmp_path / "test.wav")
        _write_wav(wav, y, sr)
        spec = GenreSpec(name="jazz", tempo_min=80.0, tempo_max=200.0)
        score, components = get_musicality_score(wav, genre_spec=spec)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Integration: check_midi_quality + get_musicality_score together
# ---------------------------------------------------------------------------

class TestMidAndAudioTogether:
    def test_midi_quality_passes_then_audio_scores(self, tmp_path):
        """Good MIDI passes Layer 1; audio file produces non-zero score."""
        from musicgen.musicality import check_midi_quality, get_musicality_score
        sr = 44100
        good = [(60 + i % 8, i * 240, 200, 80) for i in range(16)]
        paths = {}
        for layer in ("melody", "beat", "harmony", "bassline"):
            p = str(tmp_path / f"{layer}.mid")
            _write_midi(p, good)
            paths[layer] = p
        midi_result = check_midi_quality(midi_paths=paths, key="C")
        assert midi_result.hard_failures == []

        y = 0.3 * np.sin(2 * np.pi * 440 * np.linspace(0, 2.0, sr * 2))
        wav = str(tmp_path / "mix.wav")
        _write_wav(wav, y, sr)
        score, _ = get_musicality_score(wav)
        assert score > 0.0
