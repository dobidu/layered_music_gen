"""Tests for src/musicgen/calibrate.py (D-50..D-54, R-P9).

All tests run without FluidSynth — either monkeypatching the renderer call
or testing the pure cache read/write paths directly.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest
import scipy.io.wavfile as wf

from musicgen.calibrate import (
    _cache_path,
    _find_any_soundfont,
    load_preroll,
    measure_preroll,
    measure_and_save_preroll,
    save_preroll,
)


class TestCachePathHelper:
    def test_cache_path_under_project_root(self, tmp_path):
        p = _cache_path(str(tmp_path))
        assert p.endswith(os.path.join(".musicgen", "fluidsynth_preroll.json"))
        assert p.startswith(str(tmp_path))


class TestSaveAndLoadPreroll:
    def test_save_creates_json_file(self, tmp_path):
        save_preroll(str(tmp_path), 0.075)
        cache = _cache_path(str(tmp_path))
        assert os.path.isfile(cache)
        data = json.loads(open(cache).read())
        assert "offset_s" in data
        assert data["offset_s"] == pytest.approx(0.075)
        assert "fluidsynth_version" in data

    def test_load_reads_cached_value(self, tmp_path):
        save_preroll(str(tmp_path), 0.123)
        result = load_preroll(str(tmp_path))
        assert result == pytest.approx(0.123)

    def test_load_returns_zero_when_no_cache(self, tmp_path, monkeypatch):
        # Monkeypatch measure_preroll to return 0.0 (avoid FluidSynth call)
        monkeypatch.setattr("musicgen.calibrate.measure_preroll", lambda p: 0.0)
        result = load_preroll(str(tmp_path))
        assert result == pytest.approx(0.0)


class TestVersionGate:
    def test_load_remeasures_on_version_mismatch(self, tmp_path, monkeypatch):
        import musicgen.calibrate as cal
        # Write cache with a different fluidsynth_version
        os.makedirs(os.path.dirname(_cache_path(str(tmp_path))), exist_ok=True)
        with open(_cache_path(str(tmp_path)), "w") as f:
            json.dump({"offset_s": 0.05, "fluidsynth_version": "old-version-1.0"}, f)

        measured = []

        def fake_measure(project_root):
            measured.append(project_root)
            return 0.0

        monkeypatch.setattr("musicgen.calibrate.measure_preroll", fake_measure)
        # FLUIDSYNTH_VERSION != "old-version-1.0" → should re-measure
        load_preroll(str(tmp_path))
        assert len(measured) >= 1


class TestMeasurePreroll:
    def test_measure_returns_zero_when_fluidsynth_absent(self, tmp_path, monkeypatch):
        monkeypatch.setattr("musicgen.calibrate.FLUIDSYNTH_VERSION", "unknown")
        result = measure_preroll(str(tmp_path))
        assert result == pytest.approx(0.0)

    def test_measure_returns_zero_when_no_soundfonts(self, tmp_path, monkeypatch):
        monkeypatch.setattr("musicgen.calibrate._find_any_soundfont", lambda p: None)
        # FLUIDSYNTH_VERSION must not be "unknown" for this test
        import musicgen.calibrate as cal
        if cal.FLUIDSYNTH_VERSION == "unknown":
            pytest.skip("FluidSynth not installed — _find_any_soundfont test irrelevant")
        result = measure_preroll(str(tmp_path))
        assert result == pytest.approx(0.0)

    def test_measure_uses_fluidsynth_and_detects_offset(self, tmp_path, monkeypatch):
        """Synthetic WAV: 100 silent samples + 100 loud samples → offset ≈ 100/44100."""
        sr = 44100
        silence = np.zeros(100, dtype=np.int16)
        signal = np.full(100, 32000, dtype=np.int16)
        synthetic = np.concatenate([silence, signal])

        def fake_fluidsynth_render(project_root, sf_path, midi_path, wav_path):
            wf.write(wav_path, sr, synthetic)

        monkeypatch.setattr("musicgen.calibrate.FLUIDSYNTH_VERSION", "test-2.3")
        monkeypatch.setattr(
            "musicgen.calibrate._find_any_soundfont", lambda p: "/fake/sound.sf2"
        )
        monkeypatch.setattr(
            "musicgen.calibrate._render_calibration_midi", fake_fluidsynth_render
        )

        result = measure_preroll(str(tmp_path))
        expected = 100 / sr
        assert result == pytest.approx(expected, abs=1e-4)

    def test_measure_handles_entirely_silent_wav(self, tmp_path, monkeypatch):
        """All-silent WAV → no pre-roll detectable → returns 0.0."""
        sr = 44100
        silent = np.zeros(44100, dtype=np.int16)

        def fake_fluidsynth_render(project_root, sf_path, midi_path, wav_path):
            wf.write(wav_path, sr, silent)

        monkeypatch.setattr("musicgen.calibrate.FLUIDSYNTH_VERSION", "test-2.3")
        monkeypatch.setattr(
            "musicgen.calibrate._find_any_soundfont", lambda p: "/fake/sound.sf2"
        )
        monkeypatch.setattr(
            "musicgen.calibrate._render_calibration_midi", fake_fluidsynth_render
        )

        result = measure_preroll(str(tmp_path))
        assert result == pytest.approx(0.0)
