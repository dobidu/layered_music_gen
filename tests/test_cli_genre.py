"""RED tests — Phase 7: CLI genre support (v0.2).

Covers:
  - musicgen generate --genre jazz passes genre to Config
  - musicgen generate --genre jazz latin passes multiple genres
  - musicgen list-genres prints all 8 built-in genres with descriptions
  - genres/README.md exists and has content
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from musicgen.cli import app

runner = CliRunner()

REPO_ROOT = Path(__file__).parent.parent
BUILTIN_GENRES = ["jazz", "hip-hop", "blues", "pop", "electronic", "latin", "reggae", "classical"]


# ---------------------------------------------------------------------------
# list-genres command
# ---------------------------------------------------------------------------

class TestListGenresCommand:
    def test_list_genres_exits_zero(self):
        result = runner.invoke(app, ["list-genres"])
        assert result.exit_code == 0, result.output

    def test_list_genres_shows_all_builtin(self):
        result = runner.invoke(app, ["list-genres"])
        for genre in BUILTIN_GENRES:
            assert genre in result.output, f"list-genres missing {genre!r}"

    def test_list_genres_shows_descriptions(self):
        result = runner.invoke(app, ["list-genres"])
        # At least one description word visible
        assert any(
            kw in result.output.lower()
            for kw in ["swing", "beat", "tempo", "rhythm", "feel", "style", "bpm"]
        )

    def test_list_genres_uses_genres_dir(self, tmp_path):
        """--genres-dir shows genres from a custom directory."""
        import json
        (tmp_path / "myfunk").mkdir()
        (tmp_path / "myfunk" / "spec.json").write_text(json.dumps({
            "name": "myfunk", "description": "Funky test genre"
        }))
        result = runner.invoke(app, ["list-genres", "--genres-dir", str(tmp_path)])
        assert "myfunk" in result.output
        assert "Funky test genre" in result.output


# ---------------------------------------------------------------------------
# generate --genre flag
# ---------------------------------------------------------------------------

class TestGenerateGenreFlag:
    def test_generate_help_shows_genre_option(self):
        result = runner.invoke(app, ["generate", "--help"])
        assert "--genre" in result.output

    def test_generate_help_shows_genres_dir_option(self):
        result = runner.invoke(app, ["generate", "--help"])
        assert "--genres-dir" in result.output

    def test_generate_genre_single_in_config(self, tmp_path, monkeypatch):
        """--genre jazz sets Config.genre = ['jazz']."""
        captured = {}

        import musicgen.cli as _cli_mod

        def fake_batch(cfg):
            captured["genre"] = cfg.genre
            captured["genres_dir"] = cfg.genres_dir
            class R:
                succeeded = 0; failed = 0; skipped = 0; total = 0; duration_seconds = 0.0
            return R()

        monkeypatch.setattr(_cli_mod, "generate_batch", fake_batch)
        runner.invoke(app, [
            "generate", "--seed", "1", "--out", str(tmp_path), "--genre", "jazz",
        ])
        assert captured.get("genre") == ["jazz"], f"got genre={captured.get('genre')}"

    def test_generate_genre_multiple_in_config(self, tmp_path, monkeypatch):
        """--genre jazz --genre latin sets Config.genre = ['jazz', 'latin']."""
        captured = {}

        import musicgen.cli as _cli_mod

        def fake_batch(cfg):
            captured["genre"] = cfg.genre
            class R:
                succeeded = 0; failed = 0; skipped = 0; total = 0; duration_seconds = 0.0
            return R()

        monkeypatch.setattr(_cli_mod, "generate_batch", fake_batch)
        runner.invoke(app, [
            "generate", "--seed", "1", "--out", str(tmp_path),
            "--genre", "jazz", "--genre", "latin",
        ])
        assert "jazz" in captured.get("genre", [])
        assert "latin" in captured.get("genre", [])


# ---------------------------------------------------------------------------
# genres/README.md exists
# ---------------------------------------------------------------------------

class TestGenresReadme:
    def test_genres_readme_exists(self):
        path = REPO_ROOT / "genres" / "README.md"
        assert path.is_file(), "genres/README.md missing"

    def test_genres_readme_has_content(self):
        path = REPO_ROOT / "genres" / "README.md"
        content = path.read_text()
        assert len(content) > 200

    def test_genres_readme_documents_spec_format(self):
        path = REPO_ROOT / "genres" / "README.md"
        content = path.read_text()
        assert "spec.json" in content
        assert "tempo_min" in content or "tempo" in content.lower()

    def test_genres_readme_lists_builtin_genres(self):
        path = REPO_ROOT / "genres" / "README.md"
        content = path.read_text()
        for genre in BUILTIN_GENRES:
            assert genre in content, f"genres/README.md missing {genre!r}"
