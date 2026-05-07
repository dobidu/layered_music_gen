"""Tests for GenreSpec, load_genre, merge_genres, resolve_genres, and Config.genre.

Phase v0.2-1: GenreSpec + composition engine.

Merge semantics:
  - Hard numeric ranges (tempo, swing): intersection (max-of-mins, min-of-maxes)
  - Soft weight dicts: normalized weighted average
  - chord_type_hard_filter: union when all genres have filter, None when any is None
  - Set fields (soundfont_tags per layer, drum_pool_names): union (dedup, insertion order)
  - Config precedence: CLI > env > genre-merged > defaults
"""
from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from musicgen.genre import GenreSpec, load_genre, merge_genres, resolve_genres

import config as cfg_module
from config import Config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_spec(genres_dir, name, data):
    genre_dir = os.path.join(genres_dir, name)
    os.makedirs(genre_dir, exist_ok=True)
    with open(os.path.join(genre_dir, "spec.json"), "w") as f:
        json.dump(data, f)


# ---------------------------------------------------------------------------
# GenreSpec defaults
# ---------------------------------------------------------------------------

class TestGenreSpecDefaults:
    def test_name_required(self):
        with pytest.raises(TypeError):
            GenreSpec()

    def test_description_defaults_empty(self):
        spec = GenreSpec(name="test")
        assert spec.description == ""

    def test_tempo_defaults_full_range(self):
        spec = GenreSpec(name="test")
        assert spec.tempo_min == 60.0
        assert spec.tempo_max == 240.0

    def test_swing_defaults_full_range(self):
        spec = GenreSpec(name="test")
        assert spec.swing_min == 0.5
        assert spec.swing_max == 0.75

    def test_soft_weight_dicts_default_empty(self):
        spec = GenreSpec(name="test")
        assert spec.time_sig_weights == {}
        assert spec.scale_weights == {}
        assert spec.chord_type_weights == {}
        assert spec.inversion_weights == {}
        assert spec.layer_probs == {}
        assert spec.arrangement_weights == {}
        assert spec.fx_profile == {}

    def test_chord_type_hard_filter_defaults_none(self):
        spec = GenreSpec(name="test")
        assert spec.chord_type_hard_filter is None

    def test_soundfont_tags_defaults_empty(self):
        spec = GenreSpec(name="test")
        assert spec.soundfont_tags == {}

    def test_drum_pool_names_defaults_empty(self):
        spec = GenreSpec(name="test")
        assert spec.drum_pool_names == []

    def test_mutable_defaults_not_shared(self):
        a = GenreSpec(name="a")
        b = GenreSpec(name="b")
        a.time_sig_weights["4/4"] = 1.0
        assert "4/4" not in b.time_sig_weights


# ---------------------------------------------------------------------------
# load_genre
# ---------------------------------------------------------------------------

class TestLoadGenre:
    def test_load_minimal_spec(self, tmp_path):
        _write_spec(str(tmp_path), "test", {"name": "test", "description": "Test genre"})
        spec = load_genre("test", str(tmp_path))
        assert spec.name == "test"
        assert spec.description == "Test genre"

    def test_load_injects_name_if_missing(self, tmp_path):
        _write_spec(str(tmp_path), "jazz", {})
        spec = load_genre("jazz", str(tmp_path))
        assert spec.name == "jazz"

    def test_load_partial_spec_fills_defaults(self, tmp_path):
        _write_spec(str(tmp_path), "pop", {"tempo_min": 100.0, "tempo_max": 130.0})
        spec = load_genre("pop", str(tmp_path))
        assert spec.tempo_min == 100.0
        assert spec.tempo_max == 130.0
        assert spec.swing_min == 0.5  # default

    def test_load_all_fields(self, tmp_path):
        data = {
            "name": "jazz",
            "description": "Jazz",
            "tempo_min": 60.0,
            "tempo_max": 220.0,
            "swing_min": 0.55,
            "swing_max": 0.75,
            "time_sig_weights": {"4/4": 0.7, "3/4": 0.3},
            "scale_weights": {"dorian": 0.5, "major": 0.5},
            "chord_type_weights": {"maj7": 0.4, "m7": 0.4, "dom7": 0.2},
            "chord_type_hard_filter": ["maj7", "m7", "dom7", "m7b5"],
            "inversion_weights": {"root": 0.5, "first": 0.3, "second": 0.2},
            "layer_probs": {"beat": 0.9, "melody": 0.9, "harmony": 0.9, "bassline": 0.8},
            "soundfont_tags": {"beat": ["jazz drums", "brushes"], "melody": ["jazz piano"]},
            "drum_pool_names": ["jazz", "swing"],
            "fx_profile": {"reverb_room_size_center": 0.45},
            "arrangement_weights": {"intro": 0.2, "verse": 0.4, "chorus": 0.4},
        }
        _write_spec(str(tmp_path), "jazz", data)
        spec = load_genre("jazz", str(tmp_path))
        assert spec.tempo_min == 60.0
        assert spec.time_sig_weights == {"4/4": 0.7, "3/4": 0.3}
        assert spec.chord_type_hard_filter == ["maj7", "m7", "dom7", "m7b5"]
        assert spec.soundfont_tags == {"beat": ["jazz drums", "brushes"], "melody": ["jazz piano"]}
        assert spec.drum_pool_names == ["jazz", "swing"]

    def test_unknown_genre_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Genre spec not found"):
            load_genre("nonexistent", str(tmp_path))

    def test_unknown_field_in_json_ignored(self, tmp_path):
        _write_spec(str(tmp_path), "test", {"name": "test", "unknown_future_field": 42})
        spec = load_genre("test", str(tmp_path))
        assert spec.name == "test"


# ---------------------------------------------------------------------------
# merge_genres — hard numeric intersection
# ---------------------------------------------------------------------------

class TestMergeGenresHardBounds:
    def test_single_spec_returns_equivalent(self):
        spec = GenreSpec(name="jazz", tempo_min=60, tempo_max=220)
        result = merge_genres([spec])
        assert result.tempo_min == 60
        assert result.tempo_max == 220

    def test_tempo_intersection(self):
        jazz = GenreSpec(name="jazz", tempo_min=60, tempo_max=220)
        pop = GenreSpec(name="pop", tempo_min=80, tempo_max=130)
        result = merge_genres([jazz, pop])
        assert result.tempo_min == 80   # max of mins
        assert result.tempo_max == 130  # min of maxes

    def test_swing_intersection(self):
        a = GenreSpec(name="a", swing_min=0.50, swing_max=0.75)
        b = GenreSpec(name="b", swing_min=0.55, swing_max=0.70)
        result = merge_genres([a, b])
        assert result.swing_min == pytest.approx(0.55)
        assert result.swing_max == pytest.approx(0.70)

    def test_three_genre_intersection(self):
        a = GenreSpec(name="a", tempo_min=60, tempo_max=200)
        b = GenreSpec(name="b", tempo_min=80, tempo_max=160)
        c = GenreSpec(name="c", tempo_min=90, tempo_max=140)
        result = merge_genres([a, b, c])
        assert result.tempo_min == 90
        assert result.tempo_max == 140

    def test_empty_specs_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            merge_genres([])


# ---------------------------------------------------------------------------
# merge_genres — soft weight dicts
# ---------------------------------------------------------------------------

class TestMergeGenresSoftWeights:
    def test_equal_weight_avg_time_sig(self):
        a = GenreSpec(name="a", time_sig_weights={"4/4": 1.0, "3/4": 0.0})
        b = GenreSpec(name="b", time_sig_weights={"4/4": 0.0, "3/4": 1.0})
        result = merge_genres([a, b])
        assert result.time_sig_weights["4/4"] == pytest.approx(0.5)
        assert result.time_sig_weights["3/4"] == pytest.approx(0.5)

    def test_custom_weights_applied(self):
        a = GenreSpec(name="a", time_sig_weights={"4/4": 1.0})
        b = GenreSpec(name="b", time_sig_weights={"3/4": 1.0})
        result = merge_genres([a, b], weights=[2.0, 1.0])
        # a gets weight 2/3, b gets 1/3
        assert result.time_sig_weights["4/4"] == pytest.approx(2 / 3)
        assert result.time_sig_weights["3/4"] == pytest.approx(1 / 3)

    def test_soft_weights_sum_to_one(self):
        a = GenreSpec(name="a", scale_weights={"major": 0.6, "minor": 0.4})
        b = GenreSpec(name="b", scale_weights={"dorian": 0.5, "major": 0.5})
        result = merge_genres([a, b])
        assert sum(result.scale_weights.values()) == pytest.approx(1.0)

    def test_empty_weight_dict_treated_as_zero(self):
        a = GenreSpec(name="a", chord_type_weights={"maj7": 1.0})
        b = GenreSpec(name="b", chord_type_weights={})
        result = merge_genres([a, b])
        # b contributes 0 to all keys; a's weight carries through
        assert result.chord_type_weights["maj7"] == pytest.approx(1.0)

    def test_weights_wrong_length_raises(self):
        a = GenreSpec(name="a")
        b = GenreSpec(name="b")
        with pytest.raises(ValueError):
            merge_genres([a, b], weights=[1.0])

    def test_all_empty_soft_dicts_stay_empty(self):
        a = GenreSpec(name="a")
        b = GenreSpec(name="b")
        result = merge_genres([a, b])
        assert result.time_sig_weights == {}
        assert result.scale_weights == {}


# ---------------------------------------------------------------------------
# merge_genres — chord_type_hard_filter
# ---------------------------------------------------------------------------

class TestMergeGenresHardFilter:
    def test_both_none_stays_none(self):
        a = GenreSpec(name="a", chord_type_hard_filter=None)
        b = GenreSpec(name="b", chord_type_hard_filter=None)
        result = merge_genres([a, b])
        assert result.chord_type_hard_filter is None

    def test_one_none_result_is_none(self):
        a = GenreSpec(name="a", chord_type_hard_filter=["maj7", "m7"])
        b = GenreSpec(name="b", chord_type_hard_filter=None)
        result = merge_genres([a, b])
        assert result.chord_type_hard_filter is None

    def test_both_filtered_result_is_union(self):
        a = GenreSpec(name="a", chord_type_hard_filter=["maj7", "m7"])
        b = GenreSpec(name="b", chord_type_hard_filter=["dom7", "m7"])
        result = merge_genres([a, b])
        assert set(result.chord_type_hard_filter) == {"maj7", "m7", "dom7"}

    def test_union_deduplicates(self):
        a = GenreSpec(name="a", chord_type_hard_filter=["maj7", "m7"])
        b = GenreSpec(name="b", chord_type_hard_filter=["m7", "dom7"])
        result = merge_genres([a, b])
        assert result.chord_type_hard_filter.count("m7") == 1


# ---------------------------------------------------------------------------
# merge_genres — set fields (soundfont_tags, drum_pool_names)
# ---------------------------------------------------------------------------

class TestMergeGenresSetFields:
    def test_soundfont_tags_union_per_layer(self):
        a = GenreSpec(name="a", soundfont_tags={"beat": ["drums"], "melody": ["piano"]})
        b = GenreSpec(name="b", soundfont_tags={"beat": ["brushes"], "harmony": ["pad"]})
        result = merge_genres([a, b])
        assert set(result.soundfont_tags["beat"]) == {"drums", "brushes"}
        assert result.soundfont_tags["melody"] == ["piano"]
        assert result.soundfont_tags["harmony"] == ["pad"]

    def test_soundfont_tags_dedup(self):
        a = GenreSpec(name="a", soundfont_tags={"beat": ["drums", "percussion"]})
        b = GenreSpec(name="b", soundfont_tags={"beat": ["percussion", "brushes"]})
        result = merge_genres([a, b])
        assert result.soundfont_tags["beat"].count("percussion") == 1

    def test_drum_pool_names_union(self):
        a = GenreSpec(name="a", drum_pool_names=["jazz", "swing"])
        b = GenreSpec(name="b", drum_pool_names=["latin", "jazz"])
        result = merge_genres([a, b])
        assert set(result.drum_pool_names) == {"jazz", "swing", "latin"}

    def test_drum_pool_names_dedup(self):
        a = GenreSpec(name="a", drum_pool_names=["jazz"])
        b = GenreSpec(name="b", drum_pool_names=["jazz", "swing"])
        result = merge_genres([a, b])
        assert result.drum_pool_names.count("jazz") == 1

    def test_composed_name(self):
        a = GenreSpec(name="jazz")
        b = GenreSpec(name="latin")
        result = merge_genres([a, b])
        assert "jazz" in result.name
        assert "latin" in result.name


# ---------------------------------------------------------------------------
# resolve_genres
# ---------------------------------------------------------------------------

class TestResolveGenres:
    def test_single_genre(self, tmp_path):
        _write_spec(str(tmp_path), "jazz", {"name": "jazz", "tempo_min": 60, "tempo_max": 220})
        result = resolve_genres(["jazz"], str(tmp_path))
        assert result.name == "jazz"
        assert result.tempo_min == 60

    def test_two_genres_merged(self, tmp_path):
        _write_spec(str(tmp_path), "jazz", {"name": "jazz", "tempo_min": 60, "tempo_max": 220})
        _write_spec(str(tmp_path), "latin", {"name": "latin", "tempo_min": 90, "tempo_max": 180})
        result = resolve_genres(["jazz", "latin"], str(tmp_path))
        assert result.tempo_min == 90   # intersection
        assert result.tempo_max == 180

    def test_unknown_genre_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            resolve_genres(["nonexistent"], str(tmp_path))

    def test_custom_weights_forwarded(self, tmp_path):
        _write_spec(str(tmp_path), "a", {"name": "a", "time_sig_weights": {"4/4": 1.0}})
        _write_spec(str(tmp_path), "b", {"name": "b", "time_sig_weights": {"3/4": 1.0}})
        result = resolve_genres(["a", "b"], str(tmp_path), weights=[3.0, 1.0])
        assert result.time_sig_weights["4/4"] > result.time_sig_weights["3/4"]


# ---------------------------------------------------------------------------
# Config.genre field + Config.genres_dir + env var
# ---------------------------------------------------------------------------

class TestConfigGenreField:
    def test_genre_defaults_none(self):
        cfg = Config(global_seed=1)
        assert cfg.genre is None

    def test_genre_single(self):
        cfg = Config(global_seed=1, genre=["jazz"])
        assert cfg.genre == ["jazz"]

    def test_genre_composition(self):
        cfg = Config(global_seed=1, genre=["jazz", "latin"])
        assert cfg.genre == ["jazz", "latin"]

    def test_genres_dir_defaults_to_repo_genres(self):
        cfg = Config(global_seed=1)
        assert cfg.genres_dir.endswith("genres")
        assert os.path.isabs(cfg.genres_dir)

    def test_musicgen_genre_env_var_single(self, monkeypatch, tmp_path):
        _write_spec(str(tmp_path), "jazz", {"name": "jazz"})
        monkeypatch.setenv("MUSICGEN_GENRE", "jazz")
        monkeypatch.setenv("MUSICGEN_GENRES_DIR", str(tmp_path))
        cfg = Config.load({})
        assert cfg.genre == ["jazz"]

    def test_musicgen_genre_env_var_comma_separated(self, monkeypatch, tmp_path):
        _write_spec(str(tmp_path), "jazz", {"name": "jazz"})
        _write_spec(str(tmp_path), "latin", {"name": "latin"})
        monkeypatch.setenv("MUSICGEN_GENRE", "jazz,latin")
        monkeypatch.setenv("MUSICGEN_GENRES_DIR", str(tmp_path))
        cfg = Config.load({})
        assert cfg.genre == ["jazz", "latin"]

    def test_musicgen_genres_dir_env_var(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MUSICGEN_GENRES_DIR", str(tmp_path))
        cfg = Config.load({})
        assert cfg.genres_dir == str(tmp_path)

    def test_no_genre_leaves_config_unchanged(self, monkeypatch):
        monkeypatch.delenv("MUSICGEN_GENRE", raising=False)
        cfg = Config.load({})
        assert cfg.genre is None


# ---------------------------------------------------------------------------
# AST guard meta-test: genre.py must be covered
# ---------------------------------------------------------------------------

class TestGenreModuleCoveredByAstGuard:
    def test_genre_module_exists_in_package(self):
        import glob
        package_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.pardir, "src", "musicgen")
        )
        modules = [os.path.basename(p) for p in
                   glob.glob(os.path.join(package_dir, "*.py"))]
        assert "genre.py" in modules
