"""Tests for beat pattern reorganization (v0.2 Phase 3).

Covers:
  - load_beat_patterns: single dir, multi-dir union, dedup, missing dir, invalid sig
  - generate_beat with beat_roll_pattern_dirs: uses new loader, fallback to file dict
  - Config.beat_roll_pattern_dirs field + default
  - File naming: patterns_44.txt, patterns_34.txt, etc.
"""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path
from typing import Dict, List

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from musicgen.generators.beat import generate_beat, load_beat_patterns
from config import Config
from timesig import TimeSignatureRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_patterns(directory: str, sig_flat: str, lines: List[str]):
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"patterns_{sig_flat}.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


_SPEC_44 = TimeSignatureRegistry.lookup("4/4")
_SPEC_34 = TimeSignatureRegistry.lookup("3/4")

# Minimal valid patterns (length matches time-sig numerator)
_INTRO_A = "intro: 36, 0, 38, 0"
_INTRO_B = "intro: 36, 42, 38, 42"
_INTRO_C = "intro: 36, 42, 38, 0"
_VERSE_A = "verse: 36, 0, 38, 0"


# ---------------------------------------------------------------------------
# load_beat_patterns — single directory
# ---------------------------------------------------------------------------

class TestLoadBeatPatternsSingleDir:
    def test_loads_patterns_by_part(self, tmp_path):
        d = str(tmp_path / "genre_a")
        _write_patterns(d, "44", [_INTRO_A, _INTRO_B, _VERSE_A])
        result = load_beat_patterns("4/4", [d], _SPEC_44)
        assert "intro" in result
        assert len(result["intro"]) == 2
        assert "verse" in result
        assert len(result["verse"]) == 1

    def test_pattern_values_correct(self, tmp_path):
        d = str(tmp_path / "genre_a")
        _write_patterns(d, "44", [_INTRO_A])
        result = load_beat_patterns("4/4", [d], _SPEC_44)
        assert result["intro"][0] == [36, 0, 38, 0]

    def test_skips_missing_dir_gracefully(self, tmp_path):
        missing = str(tmp_path / "nonexistent")
        result = load_beat_patterns("4/4", [missing], _SPEC_44)
        assert result == {}

    def test_skips_missing_sig_file(self, tmp_path):
        d = str(tmp_path / "genre_a")
        _write_patterns(d, "34", ["intro: 36, 0, 38"])  # only 3/4 file
        result = load_beat_patterns("4/4", [d], _SPEC_44)
        assert result == {}

    def test_empty_dirs_list(self):
        result = load_beat_patterns("4/4", [], _SPEC_44)
        assert result == {}

    def test_invalid_pattern_length_skipped(self, tmp_path):
        d = str(tmp_path / "genre_a")
        _write_patterns(d, "44", ["intro: 36, 38"])  # only 2 notes — invalid for 4/4
        result = load_beat_patterns("4/4", [d], _SPEC_44)
        assert "intro" not in result or result.get("intro", []) == []

    def test_sig_flat_mapping(self, tmp_path):
        d = str(tmp_path / "genre_a")
        _write_patterns(d, "34", ["intro: 36, 0, 38"])
        result = load_beat_patterns("3/4", [d], _SPEC_34)
        assert "intro" in result

    def test_skips_blank_lines(self, tmp_path):
        d = str(tmp_path / "genre_a")
        _write_patterns(d, "44", ["", _INTRO_A, ""])
        result = load_beat_patterns("4/4", [d], _SPEC_44)
        assert result["intro"][0] == [36, 0, 38, 0]

    def test_skips_comment_lines(self, tmp_path):
        d = str(tmp_path / "genre_a")
        _write_patterns(d, "44", ["# this is a comment", _INTRO_A])
        result = load_beat_patterns("4/4", [d], _SPEC_44)
        assert len(result["intro"]) == 1


# ---------------------------------------------------------------------------
# load_beat_patterns — multi-directory union
# ---------------------------------------------------------------------------

class TestLoadBeatPatternsMultiDir:
    def test_union_from_two_dirs(self, tmp_path):
        d1 = str(tmp_path / "a")
        d2 = str(tmp_path / "b")
        _write_patterns(d1, "44", [_INTRO_A, _INTRO_B])
        _write_patterns(d2, "44", [_INTRO_C])
        result = load_beat_patterns("4/4", [d1, d2], _SPEC_44)
        assert len(result["intro"]) == 3

    def test_dedup_identical_patterns(self, tmp_path):
        d1 = str(tmp_path / "a")
        d2 = str(tmp_path / "b")
        _write_patterns(d1, "44", [_INTRO_A])
        _write_patterns(d2, "44", [_INTRO_A])  # same pattern
        result = load_beat_patterns("4/4", [d1, d2], _SPEC_44)
        assert len(result["intro"]) == 1  # deduped

    def test_first_dir_missing_uses_second(self, tmp_path):
        d1 = str(tmp_path / "missing")
        d2 = str(tmp_path / "b")
        _write_patterns(d2, "44", [_INTRO_A])
        result = load_beat_patterns("4/4", [d1, d2], _SPEC_44)
        assert result["intro"][0] == [36, 0, 38, 0]

    def test_three_dirs_union(self, tmp_path):
        dirs = []
        for i, line in enumerate([_INTRO_A, _INTRO_B, _INTRO_C]):
            d = str(tmp_path / f"dir{i}")
            _write_patterns(d, "44", [line])
            dirs.append(d)
        result = load_beat_patterns("4/4", dirs, _SPEC_44)
        assert len(result["intro"]) == 3

    def test_union_preserves_insertion_order_from_dirs(self, tmp_path):
        d1 = str(tmp_path / "a")
        d2 = str(tmp_path / "b")
        _write_patterns(d1, "44", [_INTRO_A])
        _write_patterns(d2, "44", [_INTRO_B])
        result = load_beat_patterns("4/4", [d1, d2], _SPEC_44)
        assert result["intro"][0] == [36, 0, 38, 0]   # from d1 first
        assert result["intro"][1] == [36, 42, 38, 42]  # from d2 second


# ---------------------------------------------------------------------------
# Config.beat_roll_pattern_dirs
# ---------------------------------------------------------------------------

class TestConfigBeatRollPatternDirs:
    def test_field_exists(self):
        cfg = Config()
        assert hasattr(cfg, "beat_roll_pattern_dirs")

    def test_default_is_list(self):
        cfg = Config()
        assert isinstance(cfg.beat_roll_pattern_dirs, list)

    def test_default_contains_genres_default(self):
        cfg = Config()
        assert any("genres" in d and "default" in d for d in cfg.beat_roll_pattern_dirs)

    def test_default_dir_is_absolute(self):
        cfg = Config()
        for d in cfg.beat_roll_pattern_dirs:
            assert os.path.isabs(d)

    def test_musicgen_beat_pattern_dirs_env_var(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MUSICGEN_BEAT_PATTERN_DIRS", str(tmp_path))
        cfg = Config.load({})
        assert str(tmp_path) in cfg.beat_roll_pattern_dirs


# ---------------------------------------------------------------------------
# generate_beat using beat_roll_pattern_dirs
# ---------------------------------------------------------------------------

class TestGenerateBeatWithPatternDirs:
    def _minimal_cfg(self, pattern_dir: str) -> Config:
        cfg = Config()
        cfg.beat_roll_pattern_dirs = [pattern_dir]
        cfg.beat_roll_pattern_files = {}  # disable fallback
        return cfg

    def test_uses_dirs_when_set(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        d = str(tmp_path / "pats")
        _write_patterns(d, "44", [_INTRO_A, _VERSE_A])
        cfg = self._minimal_cfg(d)
        midi_path, annotations = generate_beat(
            part="intro", tempo=120, time_signature="4/4",
            measures=2, name="song0-beat", swing_amount=0.5,
            rng=random.Random(1), cfg=cfg,
        )
        assert os.path.isfile(midi_path)

    def test_determinism_with_dirs(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        d = str(tmp_path / "pats")
        _write_patterns(d, "44", [_INTRO_A, _INTRO_B, _VERSE_A])
        cfg = self._minimal_cfg(d)
        args = dict(
            part="intro", tempo=120, time_signature="4/4",
            measures=2, name="det-beat", swing_amount=0.5, cfg=cfg,
        )
        mid1, _ = generate_beat(**args, rng=random.Random(7))
        b1 = Path(mid1).read_bytes()
        os.remove(mid1)
        mid2, _ = generate_beat(**args, rng=random.Random(7))
        assert b1 == Path(mid2).read_bytes()

    def test_fallback_to_file_dict_when_dirs_empty(self, tmp_path, monkeypatch):
        """beat_roll_pattern_files dict still used when beat_roll_pattern_dirs is []."""
        monkeypatch.chdir(tmp_path)
        d = str(tmp_path / "pats")
        fpath = _write_patterns(d, "44", [_INTRO_A, _VERSE_A])
        cfg = Config()
        cfg.beat_roll_pattern_dirs = []  # disable new loader
        cfg.beat_roll_pattern_files = {"4/4": fpath}
        midi_path, _ = generate_beat(
            part="intro", tempo=120, time_signature="4/4",
            measures=2, name="song1-beat", swing_amount=0.5,
            rng=random.Random(2), cfg=cfg,
        )
        assert os.path.isfile(midi_path)

    def test_default_config_finds_patterns(self, tmp_path, monkeypatch):
        """Default Config (beat_roll_pattern_dirs pointing to genres/default) finds 4/4 patterns."""
        monkeypatch.chdir(tmp_path)
        cfg = Config()  # uses default dirs → genres/default/
        midi_path, _ = generate_beat(
            part="intro", tempo=120, time_signature="4/4",
            measures=2, name="song2-beat", swing_amount=0.5,
            rng=random.Random(3), cfg=cfg,
        )
        assert os.path.isfile(midi_path)


# ---------------------------------------------------------------------------
# Pattern files exist in genres/default/
# ---------------------------------------------------------------------------

class TestGenresDefaultPatternFiles:
    """Verify that genres/default/ contains all expected pattern files."""

    REPO_ROOT = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir)
    )

    @pytest.mark.parametrize("sig_flat", ["24", "34", "44", "54", "68", "78", "128"])
    def test_pattern_file_exists(self, sig_flat):
        path = os.path.join(self.REPO_ROOT, "genres", "default", f"patterns_{sig_flat}.txt")
        assert os.path.isfile(path), f"Missing genres/default/patterns_{sig_flat}.txt"

    @pytest.mark.parametrize("sig_flat", ["24", "34", "44", "54", "68", "78", "128"])
    def test_pattern_file_non_empty(self, sig_flat):
        path = os.path.join(self.REPO_ROOT, "genres", "default", f"patterns_{sig_flat}.txt")
        if os.path.isfile(path):
            assert os.path.getsize(path) > 0
