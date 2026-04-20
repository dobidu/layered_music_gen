"""Tests for musicgen.manifest.ManifestWriter (D-38, R-P5)."""
from __future__ import annotations

import json
import os
import threading

import pytest

from musicgen.manifest import ManifestWriter


class TestAppend:

    def test_single_append(self, tmp_path):
        mw = ManifestWriter(str(tmp_path))
        mw.append({"sample_index": 0, "seed": 42, "status": "ok"})
        manifest = tmp_path / "manifest.jsonl"
        assert manifest.is_file()
        lines = manifest.read_text().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["sample_index"] == 0
        assert entry["seed"] == 42
        assert entry["status"] == "ok"

    def test_multiple_sequential_appends(self, tmp_path):
        mw = ManifestWriter(str(tmp_path))
        for i in range(5):
            mw.append({"sample_index": i, "status": "ok"})
        lines = (tmp_path / "manifest.jsonl").read_text().splitlines()
        assert len(lines) == 5
        for i, line in enumerate(lines):
            assert json.loads(line)["sample_index"] == i

    def test_append_creates_dataset_dir(self, tmp_path):
        """append() creates dataset_root if missing (idempotent)."""
        new_root = tmp_path / "new_dataset_root"
        assert not new_root.exists()
        mw = ManifestWriter(str(new_root))
        mw.append({"sample_index": 0, "status": "ok"})
        assert new_root.is_dir()
        assert (new_root / "manifest.jsonl").is_file()

    def test_sort_keys_stability(self, tmp_path):
        """json.dumps(sort_keys=True) — keys appear alphabetically."""
        mw = ManifestWriter(str(tmp_path))
        mw.append({"zebra": 1, "alpha": 2, "mango": 3})
        line = (tmp_path / "manifest.jsonl").read_text().strip()
        # Keys must appear in alphabetical order in the raw JSON bytes
        assert line.index('"alpha"') < line.index('"mango"') < line.index('"zebra"')


class TestConcurrent:

    def test_concurrent_threads_produce_wellformed_lines(self, tmp_path):
        """10 threads × 100 appends → 1000 well-formed JSON lines (D-38)."""
        mw = ManifestWriter(str(tmp_path))

        def _worker(worker_id: int):
            for i in range(100):
                mw.append({"sample_index": worker_id * 100 + i, "status": "ok"})

        threads = [
            threading.Thread(target=_worker, args=(w,)) for w in range(10)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        lines = (tmp_path / "manifest.jsonl").read_text().splitlines()
        assert len(lines) == 1000
        indices = set()
        for line in lines:
            entry = json.loads(line)  # Raises if corrupted — test fails.
            indices.add(entry["sample_index"])
        assert len(indices) == 1000  # All 1000 indices present, none duplicated


class TestIsSampleComplete:

    def test_returns_false_when_no_sentinel(self, tmp_path):
        assert ManifestWriter.is_sample_complete(str(tmp_path), 0) is False

    def test_returns_true_iff_sentinel(self, tmp_path):
        """D-16: sentinel-only check, manifest is irrelevant."""
        assert not ManifestWriter.is_sample_complete(str(tmp_path), 0)
        sample_dir = tmp_path / "000000"
        sample_dir.mkdir()
        (sample_dir / "sample.json").write_text("{}")
        assert ManifestWriter.is_sample_complete(str(tmp_path), 0)

    def test_does_not_require_manifest_file(self, tmp_path):
        """is_sample_complete works even without manifest.jsonl."""
        sample_dir = tmp_path / "000000"
        sample_dir.mkdir()
        (sample_dir / "sample.json").write_text("{}")
        # NO manifest.jsonl at all:
        assert not (tmp_path / "manifest.jsonl").exists()
        assert ManifestWriter.is_sample_complete(str(tmp_path), 0)

    def test_padding_width_6(self, tmp_path):
        """D-05: zero-pad width is 6 by default."""
        sample_dir = tmp_path / "000042"
        sample_dir.mkdir()
        (sample_dir / "sample.json").write_text("{}")
        assert ManifestWriter.is_sample_complete(str(tmp_path), 42)
        # And NOT 42 without padding:
        assert not ManifestWriter.is_sample_complete(str(tmp_path), 43)

    def test_custom_pad(self, tmp_path):
        """pad parameter can be overridden for forward-compat."""
        sample_dir = tmp_path / "0000042"  # 7-digit
        sample_dir.mkdir()
        (sample_dir / "sample.json").write_text("{}")
        assert ManifestWriter.is_sample_complete(str(tmp_path), 42, pad=7)
        # Default pad=6 would look for 000042, which doesn't exist:
        assert not ManifestWriter.is_sample_complete(str(tmp_path), 42)
