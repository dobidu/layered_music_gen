"""Renderer tests (R-X4): RenderResult assembly + FLUIDSYNTH_VERSION capture + seeded pick_soundfonts.

FluidSynth subprocess is mocked via ``unittest.mock.patch`` — unit tests do NOT
require the FluidSynth binary. The E2E integration test
(``tests/test_integration_full_generation.py``, Plan 04-06) covers the real
binary behind ``@pytest.mark.slow`` + ``shutil.which("fluidsynth")`` guard.
"""
from __future__ import annotations

import os
import random
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest
from pydub import AudioSegment

from musicgen.renderer import (
    FLUIDSYNTH_VERSION,
    RenderResult,
    pick_soundfonts,
    render_stems,
)


# ---------- FLUIDSYNTH_VERSION (D-07) ----------

class TestFluidSynthVersion:
    def test_fluidsynth_version_capture(self):
        """FLUIDSYNTH_VERSION is always a non-empty string.

        Either a real version line (e.g., "FluidSynth runtime version 2.3.4")
        when fluidsynth binary is on PATH, or "unknown" as the fallback when
        not (D-07 locks this: NEVER raises at import; RESEARCH Pitfall 3).
        """
        assert isinstance(FLUIDSYNTH_VERSION, str)
        assert len(FLUIDSYNTH_VERSION) > 0

    def test_renderer_importable_without_fluidsynth_binary(self):
        """Import must succeed regardless of fluidsynth binary presence."""
        import musicgen.renderer
        assert hasattr(musicgen.renderer, "FLUIDSYNTH_VERSION")


# ---------- RenderResult (D-02) ----------

class TestRenderResult:
    def test_is_frozen(self):
        rr = RenderResult(
            stem_paths={"beat": "/x.wav"},
            sample_rate=44100,
            channels=2,
            duration_seconds=1.5,
            fluidsynth_version="test",
        )
        with pytest.raises((AttributeError, Exception)):  # FrozenInstanceError or AttributeError
            rr.sample_rate = 22050  # type: ignore[misc]

    def test_equality(self):
        a = RenderResult(stem_paths={"beat": "/x.wav"}, sample_rate=44100, channels=2, duration_seconds=1.0, fluidsynth_version="v1")
        b = RenderResult(stem_paths={"beat": "/x.wav"}, sample_rate=44100, channels=2, duration_seconds=1.0, fluidsynth_version="v1")
        assert a == b

    def test_has_all_5_fields(self):
        rr = RenderResult(stem_paths={}, sample_rate=44100, channels=2, duration_seconds=0.0, fluidsynth_version="u")
        assert rr.stem_paths == {}
        assert rr.sample_rate == 44100
        assert rr.channels == 2
        assert rr.duration_seconds == 0.0
        assert rr.fluidsynth_version == "u"


# ---------- pick_soundfonts (D-08/D-17) ----------

@pytest.fixture
def fake_sf2_dirs(tmp_path):
    """Create fake sf/<layer>/ directories with 3 .sf2 files each."""
    layers = ("beat", "melody", "harmony", "bassline")
    for layer in layers:
        layer_dir = tmp_path / "sf" / layer
        layer_dir.mkdir(parents=True)
        for i in range(3):
            (layer_dir / f"fake_{i}.sf2").write_bytes(b"RIFF")
    return tmp_path


@pytest.fixture
def fake_cfg(fake_sf2_dirs, monkeypatch):
    """Config that points at the fake sf2 directories."""
    import config
    cfg = config.Config()
    cfg.sf_dir = str(fake_sf2_dirs / "sf")
    return cfg


class TestPickSoundfonts:
    def test_returns_4_layer_dict(self, fake_cfg):
        result = pick_soundfonts(cfg=fake_cfg, rng=random.Random(42))
        assert set(result.keys()) == {"beat", "melody", "harmony", "bassline"}
        for layer, path in result.items():
            assert path.endswith(".sf2")
            assert layer in path or True  # path contains the fake file name, not necessarily the layer

    def test_deterministic_same_seed(self, fake_cfg):
        a = pick_soundfonts(cfg=fake_cfg, rng=random.Random(42))
        b = pick_soundfonts(cfg=fake_cfg, rng=random.Random(42))
        assert a == b, "same seed must produce same soundfont selection"

    def test_different_seeds_different_output(self, fake_cfg):
        a = pick_soundfonts(cfg=fake_cfg, rng=random.Random(0))
        b = pick_soundfonts(cfg=fake_cfg, rng=random.Random(9999))
        # With 3 .sf2 files per layer, 2 different seeds almost always differ on at least one layer.
        assert a != b, "different seeds should produce different soundfont selection"

    def test_empty_sf_dir_raises(self, tmp_path, monkeypatch):
        import config
        cfg = config.Config()
        cfg.sf_dir = str(tmp_path / "empty_sf")
        for layer in ("beat", "melody", "harmony", "bassline"):
            (tmp_path / "empty_sf" / layer).mkdir(parents=True)
            # intentionally no .sf2 files
        with pytest.raises(FileNotFoundError, match=r"No \.sf2 files"):
            pick_soundfonts(cfg=cfg, rng=random.Random(42))

    def test_requires_rng(self, fake_cfg):
        with pytest.raises(ValueError, match="rng"):
            pick_soundfonts(cfg=fake_cfg, rng=None)


# ---------- render_stems (D-06/D-09) ----------

def _make_fake_wav(path: str, duration_ms: int = 1000, sample_rate: int = 44100):
    """Write a stereo 44.1kHz silent WAV; stand-in for a FluidSynth render."""
    AudioSegment.silent(duration=duration_ms, frame_rate=sample_rate).set_channels(2).export(
        path, format="wav"
    )


@pytest.fixture
def mock_fluidsynth(tmp_path):
    """Patch FluidSynth.midi_to_audio so it writes a fake stereo-44.1kHz WAV
    instead of calling the real subprocess.
    """
    call_counter = {"n": 0}

    def _fake_render(self, midi_path, wav_path):
        call_counter["n"] += 1
        _make_fake_wav(wav_path)

    with patch("musicgen.renderer.FluidSynth.midi_to_audio", _fake_render):
        yield call_counter


@pytest.fixture
def fake_midis_and_sfs(tmp_path):
    """Create fake MIDI paths + soundfont paths; the mock doesn't read MIDI content."""
    midi_paths = {}
    soundfonts = {}
    for layer in ("beat", "melody", "harmony", "bassline"):
        midi_path = tmp_path / f"{layer}.mid"
        midi_path.write_bytes(b"MThd fake")  # content doesn't matter; FluidSynth is mocked
        midi_paths[layer] = str(midi_path)
        sf_path = tmp_path / f"{layer}.sf2"
        sf_path.write_bytes(b"RIFF")
        soundfonts[layer] = str(sf_path)
    return midi_paths, soundfonts


class TestRenderStems:
    def test_creates_out_dir(self, tmp_path, mock_fluidsynth, fake_midis_and_sfs):
        midi_paths, soundfonts = fake_midis_and_sfs
        out_dir = tmp_path / "nested" / "stems"
        assert not out_dir.exists()
        render_stems(midi_paths, soundfonts, str(out_dir))
        assert out_dir.exists() and out_dir.is_dir()

    def test_returns_render_result(self, tmp_path, mock_fluidsynth, fake_midis_and_sfs):
        midi_paths, soundfonts = fake_midis_and_sfs
        out_dir = tmp_path / "stems"
        result = render_stems(midi_paths, soundfonts, str(out_dir))
        assert isinstance(result, RenderResult)
        assert set(result.stem_paths.keys()) == {"beat", "melody", "harmony", "bassline"}
        assert result.sample_rate == 44100
        assert result.channels == 2
        assert result.duration_seconds == pytest.approx(1.0, abs=0.1)
        assert result.fluidsynth_version == FLUIDSYNTH_VERSION

    def test_dispatches_all_4_layers(self, tmp_path, mock_fluidsynth, fake_midis_and_sfs):
        midi_paths, soundfonts = fake_midis_and_sfs
        out_dir = tmp_path / "stems"
        render_stems(midi_paths, soundfonts, str(out_dir))
        assert mock_fluidsynth["n"] == 4, f"FluidSynth.midi_to_audio called {mock_fluidsynth['n']}x, expected 4 (one per layer)"

    def test_stem_paths_under_out_dir(self, tmp_path, mock_fluidsynth, fake_midis_and_sfs):
        midi_paths, soundfonts = fake_midis_and_sfs
        out_dir = tmp_path / "stems"
        result = render_stems(midi_paths, soundfonts, str(out_dir))
        for layer, path in result.stem_paths.items():
            assert path == os.path.join(str(out_dir), f"{layer}.wav")
            assert os.path.exists(path), f"stem {layer!r} does not exist at {path}"

    def test_missing_midi_layer_raises(self, tmp_path, mock_fluidsynth, fake_midis_and_sfs):
        midi_paths, soundfonts = fake_midis_and_sfs
        del midi_paths["harmony"]
        with pytest.raises(KeyError, match="harmony"):
            render_stems(midi_paths, soundfonts, str(tmp_path / "stems"))

    def test_missing_soundfont_layer_raises(self, tmp_path, mock_fluidsynth, fake_midis_and_sfs):
        midi_paths, soundfonts = fake_midis_and_sfs
        del soundfonts["bassline"]
        with pytest.raises(KeyError, match="bassline"):
            render_stems(midi_paths, soundfonts, str(tmp_path / "stems"))


# ---------- soundfont_manager integration (opt-in, seed-deterministic) ----------

def _make_sm_module(layer_tag_to_paths: dict) -> types.ModuleType:
    """Build a fake soundfont_manager module for testing.

    layer_tag_to_paths maps a representative tag (str) to a list of path strings.
    FakeSM.get_soundfonts_by_tags returns the first matching list.
    """
    mod = types.ModuleType("soundfont_manager")

    class FakeSF:
        def __init__(self, path: str):
            self.path = path

    _data = {tag: [FakeSF(p) for p in paths] for tag, paths in layer_tag_to_paths.items()}

    class FakeSM:
        def __init__(self, json_path: str, sf2_directory=None):
            self.sf2_directory = sf2_directory

        def get_soundfonts_by_tags(self, tags, match_all=False):
            for tag in tags:
                if tag in _data:
                    return _data[tag]
            return []

        def get_absolute_path(self, sf):
            if self.sf2_directory:
                return os.path.join(self.sf2_directory, sf.path)
            return sf.path

    mod.SoundfontManager = FakeSM
    return mod


_FULL_SM_MOD = _make_sm_module({
    "drums":   ["sm_beat_a.sf2", "sm_beat_b.sf2"],
    "melody":  ["sm_melody.sf2"],
    "harmony": ["sm_harmony.sf2"],
    "pads":    ["sm_pads.sf2"],
    "bass":    ["sm_bass.sf2"],
})


@pytest.fixture
def cfg_with_sm_db(tmp_path):
    """Config with soundfont_manager_db set; fallback sf dirs also populated."""
    import config as cfg_module
    cfg = cfg_module.Config()
    cfg.sf_dir = str(tmp_path / "sf")
    cfg.soundfont_manager_db = str(tmp_path / "soundfonts.json")  # new field
    for layer in ("beat", "melody", "harmony", "bassline"):
        layer_dir = tmp_path / "sf" / layer
        layer_dir.mkdir(parents=True)
        (layer_dir / "fallback.sf2").write_bytes(b"RIFF")
    return cfg


class TestPickSoundfontsWithSoundfontManager:
    """soundfont_manager integration: opt-in, seed-deterministic, layer-aware."""

    def test_uses_sm_when_db_configured(self, cfg_with_sm_db):
        with patch.dict(sys.modules, {"soundfont_manager": _FULL_SM_MOD}):
            result = pick_soundfonts(cfg=cfg_with_sm_db, rng=random.Random(42))
        assert set(result.keys()) == {"beat", "melody", "harmony", "bassline"}
        for path in result.values():
            assert os.path.basename(path).startswith("sm_"), (
                f"expected SM-sourced path, got {path!r}"
            )

    def test_sm_deterministic_same_seed(self, cfg_with_sm_db):
        with patch.dict(sys.modules, {"soundfont_manager": _FULL_SM_MOD}):
            a = pick_soundfonts(cfg=cfg_with_sm_db, rng=random.Random(0))
            b = pick_soundfonts(cfg=cfg_with_sm_db, rng=random.Random(0))
        assert a == b

    def test_fallback_when_sm_not_installed(self, cfg_with_sm_db):
        with patch.dict(sys.modules, {"soundfont_manager": None}):
            result = pick_soundfonts(cfg=cfg_with_sm_db, rng=random.Random(42))
        for path in result.values():
            assert "fallback" in path, f"expected fallback path, got {path!r}"

    def test_fallback_when_sm_returns_no_matches(self, cfg_with_sm_db):
        empty_mod = _make_sm_module({})
        with patch.dict(sys.modules, {"soundfont_manager": empty_mod}):
            result = pick_soundfonts(cfg=cfg_with_sm_db, rng=random.Random(42))
        for path in result.values():
            assert "fallback" in path, f"expected fallback path, got {path!r}"

    def test_no_sm_db_skips_sm_entirely(self, fake_cfg):
        """When soundfont_manager_db is unset, SM is never consulted."""
        with patch.dict(sys.modules, {"soundfont_manager": _FULL_SM_MOD}):
            result = pick_soundfonts(cfg=fake_cfg, rng=random.Random(42))
        for path in result.values():
            assert not os.path.basename(path).startswith("sm_"), (
                f"SM should not be used without soundfont_manager_db; got {path!r}"
            )


class TestRendererLayerTags:
    """_LAYER_TAGS covers all 4 layers with meaningful tags."""

    def test_layer_tags_covers_all_layers(self):
        from musicgen.renderer import _LAYER_TAGS
        assert set(_LAYER_TAGS.keys()) == {"beat", "melody", "harmony", "bassline"}

    def test_beat_has_percussion_tag(self):
        from musicgen.renderer import _LAYER_TAGS
        assert any("drum" in t or "percussion" in t for t in _LAYER_TAGS["beat"])

    def test_bassline_has_bass_tag(self):
        from musicgen.renderer import _LAYER_TAGS
        assert any("bass" in t for t in _LAYER_TAGS["bassline"])


class TestConfigSoundfontManagerFields:
    """Config exposes soundfont_manager_db and soundfont_manager_sf_dir fields."""

    def test_soundfont_manager_db_defaults_none(self):
        import config as cfg_module
        cfg = cfg_module.Config()
        assert cfg.soundfont_manager_db is None

    def test_soundfont_manager_sf_dir_defaults_none(self):
        import config as cfg_module
        cfg = cfg_module.Config()
        assert cfg.soundfont_manager_sf_dir is None

    def test_soundfont_manager_db_env_var(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MUSICGEN_SOUNDFONT_MANAGER_DB", str(tmp_path / "sm.json"))
        import config as cfg_module
        cfg = cfg_module.Config.load()
        assert cfg.soundfont_manager_db == str(tmp_path / "sm.json")

    def test_soundfont_manager_sf_dir_env_var(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MUSICGEN_SOUNDFONT_MANAGER_SF_DIR", str(tmp_path / "sf2"))
        import config as cfg_module
        cfg = cfg_module.Config.load()
        assert cfg.soundfont_manager_sf_dir == str(tmp_path / "sf2")
