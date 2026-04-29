"""Tests for src/musicgen/batch.py (D-55..D-60, R-P10/P11/P15/P16).

ProcessPoolExecutor is replaced with ThreadPoolExecutor in all tests for speed
(spawn context startup is slow in CI). This tests orchestration logic but not
inter-process isolation — the slow integration test (test_integration_batch.py)
covers the real spawn-context path.
"""
from __future__ import annotations

import dataclasses
import io
import json
import os
from concurrent.futures import ThreadPoolExecutor

import pytest

from config import Config
from musicgen.api import SampleResult
from musicgen.batch import BatchResult, generate_batch
from musicgen.manifest import ManifestWriter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_ok_result(idx: int, dataset_root: str) -> SampleResult:
    sample_dir = os.path.join(dataset_root, f"{idx:06d}")
    os.makedirs(sample_dir, exist_ok=True)
    sentinel = os.path.join(sample_dir, "sample.json")
    with open(sentinel, "w") as f:
        json.dump({"sample_index": idx, "split": "train",
                   "duration_seconds": 1.0, "musicality_score": 0.5}, f)
    return SampleResult(
        sample_index=idx, seed=0, sample_dir=sample_dir,
        sample_json_path=sentinel,
        mix_path=os.path.join(sample_dir, "mix.wav"),
        stem_paths={}, midi_paths={},
        split="train", status="ok",
        musicality_score=0.5, duration_seconds=1.0,
    )


@pytest.fixture
def patch_executor(monkeypatch):
    """Replace _make_executor with a ThreadPoolExecutor factory for test speed."""
    monkeypatch.setattr(
        "musicgen.batch._make_executor",
        lambda max_workers, mp_ctx: ThreadPoolExecutor(max_workers=max_workers),
    )


@pytest.fixture
def patch_generate(monkeypatch, tmp_path):
    """Replace generate() in batch module with a fast stub."""
    call_log = []

    def _fake_generate(config, *, manifest_writer=None):
        call_log.append(config.sample_index)
        return _make_ok_result(config.sample_index, config.dataset_root)

    monkeypatch.setattr("musicgen.batch.generate", _fake_generate)
    return call_log


@pytest.fixture
def base_config(tmp_path):
    return Config(
        global_seed=42,
        dataset_root=str(tmp_path / "dataset"),
        count=1,
        workers=1,
    )


# ---------------------------------------------------------------------------
# TestBatchResult
# ---------------------------------------------------------------------------


class TestBatchResult:
    def test_batch_result_fields(self):
        fields = {f.name for f in dataclasses.fields(BatchResult)}
        assert fields == {"total", "succeeded", "failed", "skipped", "results", "duration_seconds"}

    def test_batch_result_frozen(self, tmp_path):
        r = BatchResult(total=1, succeeded=1, failed=0, skipped=0,
                        results=(), duration_seconds=1.0)
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            r.total = 99


# ---------------------------------------------------------------------------
# TestGenerateBatch
# ---------------------------------------------------------------------------


class TestGenerateBatch:
    def test_generate_batch_count_one(self, base_config, patch_executor, patch_generate, tmp_path):
        os.makedirs(base_config.dataset_root, exist_ok=True)
        result = generate_batch(base_config)
        assert result.total == 1
        assert result.succeeded == 1
        assert result.failed == 0
        assert result.skipped == 0
        assert len(result.results) == 1

    def test_generate_batch_count_three(self, base_config, patch_executor, patch_generate, tmp_path):
        cfg = dataclasses.replace(base_config, count=3)
        os.makedirs(cfg.dataset_root, exist_ok=True)
        result = generate_batch(cfg)
        assert result.total == 3
        assert result.succeeded == 3
        assert len(patch_generate) == 3

    def test_generate_batch_skips_complete(self, base_config, patch_executor, patch_generate, tmp_path):
        """Sample 0 already done → skipped=1, generate not called for it."""
        os.makedirs(base_config.dataset_root, exist_ok=True)
        _make_ok_result(0, base_config.dataset_root)  # pre-create sentinel
        result = generate_batch(base_config)
        assert result.skipped == 1
        assert result.succeeded == 0
        assert 0 not in patch_generate

    def test_generate_batch_retries_failed(self, base_config, patch_executor, patch_generate, tmp_path):
        """Failed sample (manifest entry, no sentinel) is retried — NOT skipped."""
        os.makedirs(base_config.dataset_root, exist_ok=True)
        # Manifest entry with failed status but no sample.json
        mw = ManifestWriter(base_config.dataset_root)
        mw.append({"sample_index": 0, "status": "failed", "seed": 42,
                    "sample_seed": 0, "split": "", "path": "", "wrote_at": ""})
        result = generate_batch(base_config)
        # sample.json does NOT exist → not complete → should be retried
        assert result.succeeded == 1
        assert result.skipped == 0

    def test_generate_batch_failure_isolation(self, base_config, patch_executor, monkeypatch, tmp_path):
        """One sample fails → batch continues, succeeded=2, failed=1."""
        cfg = dataclasses.replace(base_config, count=3)
        os.makedirs(cfg.dataset_root, exist_ok=True)

        def _sometimes_fail(config, *, manifest_writer=None):
            if config.sample_index == 1:
                raise RuntimeError("Synthetic failure for index 1")
            return _make_ok_result(config.sample_index, config.dataset_root)

        monkeypatch.setattr("musicgen.batch.generate", _sometimes_fail)
        result = generate_batch(cfg)
        assert result.total == 3
        assert result.succeeded == 2
        assert result.failed == 1

    def test_generate_batch_manifest_appended(self, base_config, patch_executor, patch_generate, tmp_path):
        """After batch, manifest.jsonl has entries for each sample."""
        cfg = dataclasses.replace(base_config, count=2)
        os.makedirs(cfg.dataset_root, exist_ok=True)
        generate_batch(cfg)
        manifest = os.path.join(cfg.dataset_root, "manifest.jsonl")
        assert os.path.isfile(manifest)
        lines = [json.loads(l) for l in open(manifest) if l.strip()]
        assert len(lines) == 2


# ---------------------------------------------------------------------------
# TestProgressLog
# ---------------------------------------------------------------------------


class TestProgressLog:
    def test_batch_start_event_emitted(self, base_config, patch_executor, patch_generate, monkeypatch, tmp_path, capsys):
        os.makedirs(base_config.dataset_root, exist_ok=True)
        generate_batch(base_config)
        captured = capsys.readouterr()
        events = []
        for line in captured.err.splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        event_types = [e.get("event") for e in events]
        assert "batch_start" in event_types

    def test_batch_done_event_emitted(self, base_config, patch_executor, patch_generate, monkeypatch, tmp_path, capsys):
        cfg = dataclasses.replace(base_config, count=2)
        os.makedirs(cfg.dataset_root, exist_ok=True)
        result = generate_batch(cfg)
        captured = capsys.readouterr()
        done_events = []
        for line in captured.err.splitlines():
            try:
                e = json.loads(line)
                if e.get("event") == "batch_done":
                    done_events.append(e)
            except json.JSONDecodeError:
                pass
        assert len(done_events) == 1
        assert done_events[0].get("succeeded") == 2
