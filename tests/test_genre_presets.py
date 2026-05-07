"""RED tests — Phase 6: Built-in genre presets (v0.2).

Covers:
  - All 8 genres have genres/<name>/spec.json that loads cleanly as GenreSpec
  - spec.json fields are within valid ranges (tempo_min < tempo_max, etc.)
  - At least one patterns_*.txt per genre for applicable time signatures
  - load_genre(name, genres_dir) works for each preset
  - Sample-level smoke: genre constrains tempo/swing in loaded spec
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from musicgen.genre import GenreSpec, load_genre

REPO_ROOT = Path(__file__).parent.parent
GENRES_DIR = REPO_ROOT / "genres"

BUILTIN_GENRES = [
    "jazz",
    "hip-hop",
    "blues",
    "pop",
    "electronic",
    "latin",
    "reggae",
    "classical",
]


# ---------------------------------------------------------------------------
# spec.json existence + parseability
# ---------------------------------------------------------------------------

class TestGenreSpecFiles:
    @pytest.mark.parametrize("genre", BUILTIN_GENRES)
    def test_spec_json_exists(self, genre):
        path = GENRES_DIR / genre / "spec.json"
        assert path.is_file(), f"Missing genres/{genre}/spec.json"

    @pytest.mark.parametrize("genre", BUILTIN_GENRES)
    def test_spec_json_parses(self, genre):
        path = GENRES_DIR / genre / "spec.json"
        data = json.loads(path.read_text())
        assert isinstance(data, dict)
        assert "name" in data

    @pytest.mark.parametrize("genre", BUILTIN_GENRES)
    def test_load_genre_returns_genre_spec(self, genre):
        spec = load_genre(genre, str(GENRES_DIR))
        assert isinstance(spec, GenreSpec)
        assert spec.name == genre

    @pytest.mark.parametrize("genre", BUILTIN_GENRES)
    def test_tempo_range_valid(self, genre):
        spec = load_genre(genre, str(GENRES_DIR))
        assert spec.tempo_min >= 40, f"{genre}: tempo_min too low"
        assert spec.tempo_max <= 300, f"{genre}: tempo_max too high"
        assert spec.tempo_min <= spec.tempo_max, f"{genre}: tempo_min > tempo_max"

    @pytest.mark.parametrize("genre", BUILTIN_GENRES)
    def test_swing_range_valid(self, genre):
        spec = load_genre(genre, str(GENRES_DIR))
        assert 0.5 <= spec.swing_min <= spec.swing_max <= 0.75

    @pytest.mark.parametrize("genre", BUILTIN_GENRES)
    def test_genre_has_description(self, genre):
        spec = load_genre(genre, str(GENRES_DIR))
        assert isinstance(spec.description, str) and len(spec.description) > 0


# ---------------------------------------------------------------------------
# Drum pattern files
# ---------------------------------------------------------------------------

class TestGenreDrumPatternFiles:
    @pytest.mark.parametrize("genre", BUILTIN_GENRES)
    def test_at_least_one_pattern_file(self, genre):
        genre_dir = GENRES_DIR / genre
        pattern_files = list(genre_dir.glob("patterns_*.txt"))
        assert len(pattern_files) >= 1, (
            f"genres/{genre}/ has no patterns_*.txt files"
        )

    @pytest.mark.parametrize("genre", BUILTIN_GENRES)
    def test_pattern_file_has_content(self, genre):
        genre_dir = GENRES_DIR / genre
        for pfile in sorted(genre_dir.glob("patterns_*.txt")):
            content = pfile.read_text().strip()
            assert len(content) > 0, f"{pfile} is empty"
            # At least one valid pattern line (part: n, n, ...)
            valid_lines = [
                l for l in content.splitlines()
                if ":" in l and not l.startswith("#")
            ]
            assert len(valid_lines) >= 1, f"{pfile} has no valid pattern lines"

    @pytest.mark.parametrize("genre", ["jazz", "hip-hop", "blues", "pop",
                                        "electronic", "latin", "reggae", "classical"])
    def test_patterns_44_exists(self, genre):
        """All genres must have 4/4 patterns (most common time signature)."""
        path = GENRES_DIR / genre / "patterns_44.txt"
        assert path.is_file(), f"Missing genres/{genre}/patterns_44.txt"


# ---------------------------------------------------------------------------
# load_genre round-trip — genre-specific field checks
# ---------------------------------------------------------------------------

class TestGenreSpecFieldChecks:
    def test_jazz_has_swing_above_default(self):
        spec = load_genre("jazz", str(GENRES_DIR))
        # Jazz should prefer higher swing (> 0.6)
        assert spec.swing_min >= 0.55, "jazz swing_min should be notably above 0.5"

    def test_jazz_has_3_4_time_sig_weight(self):
        spec = load_genre("jazz", str(GENRES_DIR))
        # Jazz commonly uses 3/4 or other non-4/4 feels
        assert spec.time_sig_weights, "jazz should have time_sig_weights set"

    def test_electronic_has_high_tempo(self):
        spec = load_genre("electronic", str(GENRES_DIR))
        assert spec.tempo_min >= 100, "electronic tempo_min should be >= 100 BPM"

    def test_hip_hop_has_low_to_mid_tempo(self):
        spec = load_genre("hip-hop", str(GENRES_DIR))
        assert spec.tempo_max <= 140, "hip-hop tempo_max should be <= 140 BPM"

    def test_classical_has_wide_tempo_range(self):
        spec = load_genre("classical", str(GENRES_DIR))
        assert spec.tempo_max - spec.tempo_min >= 40, (
            "classical should have a wide tempo range (>= 40 BPM span)"
        )

    def test_reggae_has_low_tempo(self):
        spec = load_genre("reggae", str(GENRES_DIR))
        assert spec.tempo_max <= 100, "reggae tempo_max should be <= 100 BPM"
