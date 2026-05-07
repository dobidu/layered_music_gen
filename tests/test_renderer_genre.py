"""RED tests — Phase 5: Soundfont genre integration (v0.2).

Covers:
  - _pick_via_soundfont_manager accepts layer_tags kwarg; uses genre tags over _LAYER_TAGS
  - pick_soundfonts accepts optional genre_spec; passes soundfont_tags to SM
  - genre_spec=None → identical behavior to pre-genre (backward compat)
  - Genre soundfont_tags empty for a layer → falls back to _LAYER_TAGS for that layer
  - SM not installed + genre_spec set → directory-scan fallback unchanged
"""
from __future__ import annotations

import os
import random
import sys
import types
from unittest.mock import patch

import pytest

from musicgen.genre import GenreSpec
from musicgen.renderer import _LAYER_TAGS, pick_soundfonts, _pick_via_soundfont_manager
import config as cfg_module


# ---------------------------------------------------------------------------
# Helpers — mirrors pattern from test_renderer.py
# ---------------------------------------------------------------------------

def _make_sm(tag_to_paths: dict) -> types.ModuleType:
    mod = types.ModuleType("soundfont_manager")

    class FakeSF:
        def __init__(self, path):
            self.path = path

    _data = {tag: [FakeSF(p) for p in paths] for tag, paths in tag_to_paths.items()}

    class FakeSM:
        def __init__(self, json_path, sf2_directory=None):
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


# SM with default tags (static _LAYER_TAGS) populated
_DEFAULT_SM = _make_sm({
    "drums":   ["default_beat.sf2"],
    "melody":  ["default_melody.sf2"],
    "harmony": ["default_harmony.sf2"],
    "bass":    ["default_bass.sf2"],
})

# SM with genre-specific tags populated
_GENRE_SM = _make_sm({
    "jazz_drums":  ["jazz_beat.sf2"],
    "jazz_piano":  ["jazz_melody.sf2"],
    "jazz_chords": ["jazz_harmony.sf2"],
    "jazz_bass":   ["jazz_bass.sf2"],
    # also keep default tags to allow fallback test
    "drums":       ["default_beat.sf2"],
    "melody":      ["default_melody.sf2"],
    "harmony":     ["default_harmony.sf2"],
    "bass":        ["default_bass.sf2"],
})


@pytest.fixture
def sm_cfg(tmp_path):
    cfg = cfg_module.Config()
    cfg.sf_dir = str(tmp_path / "sf")
    cfg.soundfont_manager_db = str(tmp_path / "db.json")
    for layer in ("beat", "melody", "harmony", "bassline"):
        d = tmp_path / "sf" / layer
        d.mkdir(parents=True)
        (d / "fallback.sf2").write_bytes(b"RIFF")
    return cfg


def _jazz_genre_spec(**extra) -> GenreSpec:
    return GenreSpec(
        name="jazz",
        soundfont_tags={
            "beat":     ["jazz_drums"],
            "melody":   ["jazz_piano"],
            "harmony":  ["jazz_chords"],
            "bassline": ["jazz_bass"],
        },
        **extra,
    )


# ---------------------------------------------------------------------------
# _pick_via_soundfont_manager — layer_tags kwarg
# ---------------------------------------------------------------------------

class TestPickViaSoundfontManagerLayerTags:
    def test_accepts_layer_tags_kwarg(self, sm_cfg):
        """_pick_via_soundfont_manager must accept optional layer_tags kwarg."""
        with patch.dict(sys.modules, {"soundfont_manager": _GENRE_SM}):
            result = _pick_via_soundfont_manager(
                sm_cfg,
                random.Random(0),
                layer_tags={
                    "beat":     ["jazz_drums"],
                    "melody":   ["jazz_piano"],
                    "harmony":  ["jazz_chords"],
                    "bassline": ["jazz_bass"],
                },
            )
        assert result is not None

    def test_uses_genre_tags_not_default_tags(self, sm_cfg):
        """When layer_tags supplied, SM is queried with those tags."""
        with patch.dict(sys.modules, {"soundfont_manager": _GENRE_SM}):
            result = _pick_via_soundfont_manager(
                sm_cfg,
                random.Random(0),
                layer_tags={
                    "beat":     ["jazz_drums"],
                    "melody":   ["jazz_piano"],
                    "harmony":  ["jazz_chords"],
                    "bassline": ["jazz_bass"],
                },
            )
        assert result is not None
        assert "jazz_beat.sf2" in result["beat"]
        assert "jazz_melody.sf2" in result["melody"]
        assert "jazz_harmony.sf2" in result["harmony"]
        assert "jazz_bass.sf2" in result["bassline"]

    def test_default_tags_used_when_layer_tags_none(self, sm_cfg):
        """When layer_tags=None, static _LAYER_TAGS are used (backward compat)."""
        with patch.dict(sys.modules, {"soundfont_manager": _DEFAULT_SM}):
            result = _pick_via_soundfont_manager(
                sm_cfg,
                random.Random(0),
                layer_tags=None,
            )
        assert result is not None
        assert "default_beat.sf2" in result["beat"]

    def test_no_layer_tags_arg_backward_compat(self, sm_cfg):
        """Calling without layer_tags at all must work (backward compat)."""
        with patch.dict(sys.modules, {"soundfont_manager": _DEFAULT_SM}):
            result = _pick_via_soundfont_manager(sm_cfg, random.Random(0))
        assert result is not None

    def test_fallback_when_genre_tags_not_in_sm(self, sm_cfg):
        """When genre tags yield no SM hits → returns None (triggers dir-scan fallback)."""
        sm_no_jazz = _make_sm({
            "drums": ["default_beat.sf2"],
            "melody": ["default_melody.sf2"],
            "harmony": ["default_harmony.sf2"],
            "bass": ["default_bass.sf2"],
        })
        with patch.dict(sys.modules, {"soundfont_manager": sm_no_jazz}):
            result = _pick_via_soundfont_manager(
                sm_cfg,
                random.Random(0),
                layer_tags={
                    "beat": ["jazz_drums"],  # not in sm_no_jazz
                    "melody": ["jazz_piano"],
                    "harmony": ["jazz_chords"],
                    "bassline": ["jazz_bass"],
                },
            )
        assert result is None


# ---------------------------------------------------------------------------
# pick_soundfonts — genre_spec kwarg
# ---------------------------------------------------------------------------

class TestPickSoundfontsGenreSpec:
    def test_accepts_genre_spec_kwarg(self, sm_cfg):
        spec = _jazz_genre_spec()
        with patch.dict(sys.modules, {"soundfont_manager": _GENRE_SM}):
            result = pick_soundfonts(cfg=sm_cfg, rng=random.Random(0), genre_spec=spec)
        assert set(result.keys()) == {"beat", "melody", "harmony", "bassline"}

    def test_genre_tags_used_when_sm_active(self, sm_cfg):
        """Genre soundfont_tags override static _LAYER_TAGS in SM query."""
        spec = _jazz_genre_spec()
        with patch.dict(sys.modules, {"soundfont_manager": _GENRE_SM}):
            result = pick_soundfonts(cfg=sm_cfg, rng=random.Random(0), genre_spec=spec)
        assert "jazz_beat.sf2" in result["beat"]
        assert "jazz_melody.sf2" in result["melody"]
        assert "jazz_harmony.sf2" in result["harmony"]
        assert "jazz_bass.sf2" in result["bassline"]

    def test_none_genre_spec_backward_compat(self, sm_cfg):
        """pick_soundfonts(..., genre_spec=None) must equal pick_soundfonts(...)."""
        with patch.dict(sys.modules, {"soundfont_manager": _DEFAULT_SM}):
            r1 = pick_soundfonts(cfg=sm_cfg, rng=random.Random(5), genre_spec=None)
            r2 = pick_soundfonts(cfg=sm_cfg, rng=random.Random(5))
        assert r1 == r2

    def test_genre_spec_empty_soundfont_tags_uses_static_tags(self, sm_cfg):
        """Genre with empty soundfont_tags → static _LAYER_TAGS used (no constraint)."""
        spec = GenreSpec(name="noop", soundfont_tags={})
        with patch.dict(sys.modules, {"soundfont_manager": _DEFAULT_SM}):
            result = pick_soundfonts(cfg=sm_cfg, rng=random.Random(0), genre_spec=spec)
        assert "default_beat.sf2" in result["beat"]

    def test_genre_spec_partial_tags_layer_falls_back(self, sm_cfg):
        """Genre with soundfont_tags for only some layers → static tags for missing layers."""
        # Only beat has genre tags; others should use static _LAYER_TAGS
        spec = GenreSpec(
            name="partial",
            soundfont_tags={"beat": ["jazz_drums"]},
        )
        with patch.dict(sys.modules, {"soundfont_manager": _GENRE_SM}):
            result = pick_soundfonts(cfg=sm_cfg, rng=random.Random(0), genre_spec=spec)
        # beat uses jazz tags
        assert "jazz_beat.sf2" in result["beat"]
        # melody uses default tags
        assert "default_melody.sf2" in result["melody"]

    def test_sm_not_installed_genre_spec_falls_back_to_dir_scan(self, sm_cfg):
        """SM not installed + genre_spec → directory-scan fallback unchanged."""
        spec = _jazz_genre_spec()
        with patch.dict(sys.modules, {"soundfont_manager": None}):
            result = pick_soundfonts(cfg=sm_cfg, rng=random.Random(0), genre_spec=spec)
        for path in result.values():
            assert "fallback" in path

    def test_no_sm_db_genre_spec_uses_dir_scan(self, tmp_path):
        """No SM db configured + genre_spec → directory scan ignoring tags."""
        cfg = cfg_module.Config()
        cfg.sf_dir = str(tmp_path / "sf")
        for layer in ("beat", "melody", "harmony", "bassline"):
            d = tmp_path / "sf" / layer
            d.mkdir(parents=True)
            (d / "fallback.sf2").write_bytes(b"RIFF")
        spec = _jazz_genre_spec()
        result = pick_soundfonts(cfg=cfg, rng=random.Random(0), genre_spec=spec)
        for path in result.values():
            assert "fallback" in path
