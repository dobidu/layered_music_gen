"""
Unit tests for config.py — R-S5 (config centralization) and R-S9 (soundfont pool detection).

Covers:
- Default paths match the legacy file layout (four sf layers, four fx files,
  six beat_roll_patterns files, inst_probabilities / levels / song_structures /
  chord_patterns).
- Env-var override precedence (MUSICGEN_SF_DIR, MUSICGEN_PROJECT_ROOT).
- CLI-overrides-take-precedence-over-env (D-01/D-02).
- Soundfont pool WARNING fires below threshold (<3 .sf2 files per layer).
- Soundfont pool WARNING does NOT fire at/above threshold.
- Missing layer directory emits a WARNING (not a crash).
"""
import logging
import os

import pytest

import config


class TestConfigDefaults:
    @pytest.mark.parametrize("layer", ["beat", "melody", "harmony", "bassline"])
    def test_default_sf_layer_dir_ends_with_layer_name(self, layer):
        cfg = config.Config()
        assert cfg.sf_layer_dir(layer).endswith(os.path.join("sf", layer))

    def test_default_fx_files_has_four_layers(self):
        cfg = config.Config()
        assert set(cfg.fx_files.keys()) == {"beat", "melody", "harmony", "bassline"}

    @pytest.mark.parametrize(
        "time_signature,expected_suffix",
        [
            ("2/4",  "patterns_24.txt"),
            ("3/4",  "patterns_34.txt"),
            ("4/4",  "patterns_44.txt"),
            ("5/4",  "patterns_54.txt"),
            ("6/8",  "patterns_68.txt"),
            ("7/8",  "patterns_78.txt"),
            ("12/8", "patterns_128.txt"),
        ],
    )
    def test_beat_pattern_file_per_signature(self, time_signature, expected_suffix):
        cfg = config.Config()
        assert cfg.beat_pattern_file(time_signature).endswith(expected_suffix)

    def test_default_levels_file_name(self):
        cfg = config.Config()
        assert cfg.levels_file.endswith("levels.json")

    def test_default_song_structures_file_name(self):
        cfg = config.Config()
        assert cfg.song_structures_file.endswith("song_structures.json")

    def test_default_inst_probabilities_file_name(self):
        cfg = config.Config()
        assert cfg.inst_probabilities_file.endswith("inst_probabilities.json")

    def test_default_chord_patterns_file_name(self):
        cfg = config.Config()
        assert cfg.chord_patterns_file.endswith("chord_patterns.txt")


class TestOverridePrecedence:
    def test_env_override_applies_to_sf_dir(self, monkeypatch, tmp_path):
        fake = tmp_path / "fake_sf"
        fake.mkdir()
        monkeypatch.setenv("MUSICGEN_SF_DIR", str(fake))
        cfg = config.Config.load()
        assert cfg.sf_dir == str(fake)

    def test_cli_overrides_take_precedence_over_env(self, monkeypatch, tmp_path):
        env_path = tmp_path / "env_sf"
        env_path.mkdir()
        cli_path = tmp_path / "cli_sf"
        cli_path.mkdir()
        monkeypatch.setenv("MUSICGEN_SF_DIR", str(env_path))
        cfg = config.Config.load(cli_overrides={"sf_dir": str(cli_path)})
        assert cfg.sf_dir == str(cli_path)

    def test_load_without_overrides_returns_defaults(self, monkeypatch):
        monkeypatch.delenv("MUSICGEN_SF_DIR", raising=False)
        monkeypatch.delenv("MUSICGEN_PROJECT_ROOT", raising=False)
        cfg = config.Config.load()
        assert cfg.sf_dir == config.DEFAULT_SF_DIR


class TestSoundfontPoolDetection:
    def _make_sf_tree(self, root, counts):
        """counts: dict {layer_name: int number of .sf2 files to create}."""
        for layer, n in counts.items():
            layer_dir = root / layer
            layer_dir.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                (layer_dir / f"{layer}_{i}.sf2").write_bytes(b"")
        return root

    def test_soundfont_pool_warning_fires_below_threshold(self, tmp_path, caplog):
        root = self._make_sf_tree(tmp_path, {"beat": 2, "melody": 2, "harmony": 2, "bassline": 2})
        cfg = config.Config.load(cli_overrides={"sf_dir": str(root)})
        with caplog.at_level(logging.WARNING, logger="config"):
            cfg._emit_soundfont_pool_report()
        warning_lines = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning_lines) >= 4  # one per layer

    def test_soundfont_pool_no_warning_at_threshold(self, tmp_path, caplog):
        root = self._make_sf_tree(tmp_path, {"beat": 3, "melody": 3, "harmony": 3, "bassline": 3})
        cfg = config.Config.load(cli_overrides={"sf_dir": str(root)})
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="config"):
            cfg._emit_soundfont_pool_report()
        warning_lines = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warning_lines == []

    def test_soundfont_pool_missing_layer_logs_warning(self, tmp_path, caplog):
        root = tmp_path / "empty_sf"
        root.mkdir()
        # no layer subdirectories created
        cfg = config.Config.load(cli_overrides={"sf_dir": str(root)})
        with caplog.at_level(logging.WARNING, logger="config"):
            cfg._emit_soundfont_pool_report()
        messages = " ".join(r.message for r in caplog.records)
        assert "missing" in messages.lower() or "not found" in messages.lower() or "does not exist" in messages.lower()


class TestImportSideEffects:
    def test_importing_config_does_not_emit_warnings_at_module_load(self):
        # Regression guard: `import config` itself MUST NOT call Config.load(),
        # because load() runs _emit_soundfont_pool_report which does os.listdir.
        # Phase 1 Plan 01-01 property: imports must be side-effect-free.
        import importlib
        import config as cfg_module
        importlib.reload(cfg_module)
        # If importing triggered load(), the soundfont pool report would fire
        # and log warnings. Verify nothing emitted by checking the module has
        # a Config class but no module-level Config instance.
        assert hasattr(cfg_module, "Config")
        assert not any(
            isinstance(getattr(cfg_module, name, None), cfg_module.Config)
            for name in dir(cfg_module)
        )


# ============================================================
# Phase 5 Config extensions (D-09, D-27 — R-P6, R-P7 supporting)
# ============================================================


class TestPhase5NewFields:
    """D-09: 7 new Config fields with correct defaults."""

    def test_dataset_root_default(self):
        cfg = config.Config()
        assert cfg.dataset_root.endswith(os.path.sep + "dataset") or cfg.dataset_root.endswith("/dataset")

    def test_global_seed_default_none(self):
        """D-21: None at Config level is fine; api.generate raises on use."""
        cfg = config.Config()
        assert cfg.global_seed is None

    def test_sample_index_default_zero(self):
        cfg = config.Config()
        assert cfg.sample_index == 0

    def test_split_ratios_default(self):
        cfg = config.Config()
        assert cfg.split_ratios == (0.8, 0.1, 0.1)

    def test_sum_of_stems_epsilon_default(self):
        cfg = config.Config()
        assert cfg.sum_of_stems_epsilon == 1e-3

    def test_keep_working_dirs_default_false(self):
        cfg = config.Config()
        assert cfg.keep_working_dirs is False

    def test_workers_default_none(self):
        """D-43: workers reserved for Phase 6, defaults None."""
        cfg = config.Config()
        assert cfg.workers is None


class TestSplitRatiosValidation:
    """D-27: Config.__post_init__ rejects invalid split_ratios."""

    def test_sum_exceeds_one_raises(self):
        with pytest.raises(ValueError, match="sum"):
            config.Config(split_ratios=(0.8, 0.1, 0.5))

    def test_sum_less_than_one_raises(self):
        with pytest.raises(ValueError, match="sum"):
            config.Config(split_ratios=(0.4, 0.3, 0.1))

    def test_negative_ratio_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            config.Config(split_ratios=(0.8, -0.1, 0.3))

    def test_valid_default_accepted(self):
        """Default (0.8, 0.1, 0.1) passes validation."""
        cfg = config.Config(split_ratios=(0.8, 0.1, 0.1))
        assert cfg.split_ratios == (0.8, 0.1, 0.1)

    def test_alternative_valid_ratios(self):
        """Non-default but valid ratios accepted."""
        cfg = config.Config(split_ratios=(0.5, 0.25, 0.25))
        assert cfg.split_ratios == (0.5, 0.25, 0.25)

    def test_within_epsilon_tolerance(self):
        """Float roundoff within 1e-9 of 1.0 is accepted."""
        # (0.1 + 0.1 + 0.8 = 0.9999999999999999 in IEEE-754)
        cfg = config.Config(split_ratios=(0.1, 0.1, 0.8))
        assert cfg.split_ratios == (0.1, 0.1, 0.8)


class TestPhase6Fields:
    """Phase 6 Config field stubs (D-47, D-48, D-49)."""

    def test_output_mode_default_is_full(self):
        assert config.Config().output_mode == "full"

    def test_output_mode_invalid_raises(self):
        with pytest.raises(ValueError, match="output_mode"):
            config.Config(output_mode="invalid")

    def test_count_default_is_one(self):
        assert config.Config().count == 1

    def test_count_propagates_via_load(self):
        cfg = config.Config.load(cli_overrides={"count": 5})
        assert cfg.count == 5


class TestDatasetRootEnvVar:
    """D-09: MUSICGEN_DATASET_ROOT follows Phase 2 D-01 precedence."""

    def test_env_var_overrides_default(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MUSICGEN_DATASET_ROOT", str(tmp_path / "custom"))
        cfg = config.Config.load()
        assert cfg.dataset_root == os.path.abspath(str(tmp_path / "custom"))

    def test_env_var_normalized_to_abspath(self, tmp_path, monkeypatch):
        """T-02-01 mitigation — paths abspath-ed."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("MUSICGEN_DATASET_ROOT", "./relative_dataset")
        cfg = config.Config.load()
        assert os.path.isabs(cfg.dataset_root)
        assert cfg.dataset_root.endswith("relative_dataset")

    def test_no_env_var_uses_default(self, monkeypatch):
        monkeypatch.delenv("MUSICGEN_DATASET_ROOT", raising=False)
        cfg = config.Config.load()
        assert cfg.dataset_root.endswith("dataset")
