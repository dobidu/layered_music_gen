"""v0.3 Phase 3b — RED: quality-gate regeneration loop.

Tests cover:
  - Config.min_musicality_score (default 0.0, disabled)
  - Config.max_attempts (default 1, no re-roll)
  - Config env-var wiring: MUSICGEN_MIN_MUSICALITY_SCORE, MUSICGEN_MAX_ATTEMPTS
  - Config validation: max_attempts >= 1
  - SampleResult.attempt field (default 1)
  - Quality-gate loop: stops on pass, exhausts max_attempts
  - Each attempt uses a distinct seed
  - Manifest and sample.json carry 'attempt' field
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_pipeline_factory(scores_by_attempt: dict):
    """Return a (fake_fn, call_counter) pair.

    fake_fn matches the NEW _run_pipeline signature:
      (config, attempt_seed, rngs, working_dir, attempt=1) -> SampleResult
    """
    from musicgen import SampleResult

    call_log = {"n": 0, "seeds": [], "attempts": []}

    def fake(config, attempt_seed, rngs, working_dir, attempt=1):
        call_log["n"] += 1
        call_log["seeds"].append(attempt_seed)
        call_log["attempts"].append(attempt)
        score = scores_by_attempt.get(attempt, 0.0)
        return SampleResult(
            sample_index=config.sample_index,
            seed=attempt_seed,
            sample_dir="",
            sample_json_path="",
            mix_path="",
            stem_paths={},
            midi_paths={},
            split="train",
            status="ok",
            musicality_score=score,
            duration_seconds=10.0,
            attempt=attempt,
        )

    return fake, call_log


# ---------------------------------------------------------------------------
# Config fields
# ---------------------------------------------------------------------------

class TestQualityGateConfig:
    def test_min_musicality_score_default(self):
        """min_musicality_score=0.0 means disabled (every sample passes)."""
        from config import Config
        cfg = Config()
        assert hasattr(cfg, "min_musicality_score")
        assert cfg.min_musicality_score == 0.0

    def test_max_attempts_default(self):
        """max_attempts=1 means no re-roll — single pipeline run."""
        from config import Config
        cfg = Config()
        assert hasattr(cfg, "max_attempts")
        assert cfg.max_attempts == 1

    def test_max_attempts_validation_zero_raises(self):
        """max_attempts must be >= 1."""
        from config import Config
        with pytest.raises((ValueError, TypeError)):
            Config(max_attempts=0)

    def test_max_attempts_validation_negative_raises(self):
        from config import Config
        with pytest.raises((ValueError, TypeError)):
            Config(max_attempts=-1)

    def test_min_musicality_score_env_var(self, monkeypatch):
        monkeypatch.setenv("MUSICGEN_MIN_MUSICALITY_SCORE", "0.65")
        from config import Config
        cfg = Config.load()
        assert cfg.min_musicality_score == pytest.approx(0.65)

    def test_max_attempts_env_var(self, monkeypatch):
        monkeypatch.setenv("MUSICGEN_MAX_ATTEMPTS", "4")
        from config import Config
        cfg = Config.load()
        assert cfg.max_attempts == 4

    def test_max_attempts_env_invalid_ignored(self, monkeypatch):
        """Non-integer env value → silently ignored (keeps default)."""
        monkeypatch.setenv("MUSICGEN_MAX_ATTEMPTS", "notanumber")
        from config import Config
        cfg = Config.load()
        assert cfg.max_attempts == 1  # unchanged default


# ---------------------------------------------------------------------------
# SampleResult.attempt field
# ---------------------------------------------------------------------------

class TestSampleResultAttemptField:
    def test_attempt_field_present(self):
        from musicgen import SampleResult
        import dataclasses
        fields = {f.name for f in dataclasses.fields(SampleResult)}
        assert "attempt" in fields

    def test_attempt_defaults_to_one(self):
        """Backward-compat: existing callers that don't pass attempt get 1."""
        from musicgen import SampleResult
        r = SampleResult(
            sample_index=0, seed=1, sample_dir="", sample_json_path="",
            mix_path="", stem_paths={}, midi_paths={}, split="train",
            status="ok", musicality_score=0.5, duration_seconds=10.0,
        )
        assert r.attempt == 1

    def test_sample_result_fields_include_attempt(self):
        """Updated shape contract — 12 fields."""
        from musicgen import SampleResult
        import dataclasses
        fields = {f.name for f in dataclasses.fields(SampleResult)}
        assert fields == {
            "sample_index", "seed", "sample_dir", "sample_json_path",
            "mix_path", "stem_paths", "midi_paths", "split", "status",
            "musicality_score", "duration_seconds", "attempt",
        }


# ---------------------------------------------------------------------------
# Quality-gate loop behavior
# ---------------------------------------------------------------------------

class TestQualityGateLoop:
    def test_max_attempts_one_calls_pipeline_once(self, tmp_path, monkeypatch):
        """max_attempts=1 → one pipeline call, result.attempt == 1."""
        from musicgen import Config, generate
        fake, log = _fake_pipeline_factory({1: 0.2})
        monkeypatch.setattr("musicgen.api._run_pipeline", fake)

        cfg = Config(
            global_seed=1, sample_index=0, dataset_root=str(tmp_path),
            min_musicality_score=0.9, max_attempts=1,
        )
        result = generate(cfg)
        assert log["n"] == 1
        assert result.attempt == 1

    def test_stops_when_score_meets_threshold(self, tmp_path, monkeypatch):
        """Attempt 1 fails (0.3), attempt 2 passes (0.8 >= 0.7) → stop at 2."""
        from musicgen import Config, generate
        fake, log = _fake_pipeline_factory({1: 0.3, 2: 0.8, 3: 0.9})
        monkeypatch.setattr("musicgen.api._run_pipeline", fake)

        cfg = Config(
            global_seed=1, sample_index=0, dataset_root=str(tmp_path),
            min_musicality_score=0.7, max_attempts=5,
        )
        result = generate(cfg)
        assert log["n"] == 2
        assert result.attempt == 2
        assert result.musicality_score == pytest.approx(0.8)

    def test_exhausts_max_attempts_when_never_passes(self, tmp_path, monkeypatch):
        """Score always < threshold → runs exactly max_attempts times."""
        from musicgen import Config, generate
        fake, log = _fake_pipeline_factory({1: 0.1, 2: 0.2, 3: 0.3})
        monkeypatch.setattr("musicgen.api._run_pipeline", fake)

        cfg = Config(
            global_seed=1, sample_index=0, dataset_root=str(tmp_path),
            min_musicality_score=0.9, max_attempts=3,
        )
        result = generate(cfg)
        assert log["n"] == 3
        assert result.attempt == 3

    def test_returns_last_result_when_no_pass(self, tmp_path, monkeypatch):
        """After exhausting attempts, returns the last attempt's result."""
        from musicgen import Config, generate
        fake, log = _fake_pipeline_factory({1: 0.1, 2: 0.2, 3: 0.35})
        monkeypatch.setattr("musicgen.api._run_pipeline", fake)

        cfg = Config(
            global_seed=1, sample_index=0, dataset_root=str(tmp_path),
            min_musicality_score=0.9, max_attempts=3,
        )
        result = generate(cfg)
        assert result.musicality_score == pytest.approx(0.35)
        assert result.attempt == 3

    def test_min_score_zero_always_passes_on_first(self, tmp_path, monkeypatch):
        """Default min_musicality_score=0.0 → every score passes immediately."""
        from musicgen import Config, generate
        fake, log = _fake_pipeline_factory({1: 0.0})  # even 0.0 passes
        monkeypatch.setattr("musicgen.api._run_pipeline", fake)

        cfg = Config(
            global_seed=1, sample_index=0, dataset_root=str(tmp_path),
            max_attempts=5,
            # min_musicality_score defaults to 0.0
        )
        result = generate(cfg)
        assert log["n"] == 1
        assert result.attempt == 1

    def test_each_attempt_uses_distinct_seed(self, tmp_path, monkeypatch):
        """Seeds must differ across attempts for different generation outcomes."""
        from musicgen import Config, generate
        fake, log = _fake_pipeline_factory({1: 0.0, 2: 0.0, 3: 0.0})
        monkeypatch.setattr("musicgen.api._run_pipeline", fake)

        cfg = Config(
            global_seed=1, sample_index=0, dataset_root=str(tmp_path),
            min_musicality_score=0.9, max_attempts=3,
        )
        generate(cfg)
        assert len(log["seeds"]) == 3
        assert len(set(log["seeds"])) == 3, "All attempt seeds must be unique"

    def test_attempt_one_seed_equals_base_seed(self, tmp_path, monkeypatch):
        """Attempt 1 seed == original sample_seed (backward compat)."""
        from musicgen import Config, generate
        from musicgen.seeds import derive_sample_seed
        fake, log = _fake_pipeline_factory({1: 0.0})
        monkeypatch.setattr("musicgen.api._run_pipeline", fake)

        cfg = Config(
            global_seed=42, sample_index=7, dataset_root=str(tmp_path),
            min_musicality_score=0.9, max_attempts=1,
        )
        generate(cfg)
        expected_seed = derive_sample_seed(42, 7)
        assert log["seeds"][0] == expected_seed


# ---------------------------------------------------------------------------
# Manifest carries attempt
# ---------------------------------------------------------------------------

class TestManifestAttemptField:
    def test_manifest_entry_has_attempt(self, tmp_path, monkeypatch):
        """Manifest entry must include 'attempt' key."""
        from musicgen import Config, generate
        fake, log = _fake_pipeline_factory({1: 0.3, 2: 0.8})
        monkeypatch.setattr("musicgen.api._run_pipeline", fake)

        cfg = Config(
            global_seed=1, sample_index=0, dataset_root=str(tmp_path),
            min_musicality_score=0.7, max_attempts=3,
        )
        generate(cfg)

        manifest_path = tmp_path / "manifest.jsonl"
        assert manifest_path.is_file()
        entries = [json.loads(line) for line in manifest_path.read_text().splitlines()]
        # Exactly one manifest entry for the sample (winner only)
        ok_entries = [e for e in entries if e.get("status") == "ok"]
        assert len(ok_entries) == 1
        assert ok_entries[0]["attempt"] == 2
