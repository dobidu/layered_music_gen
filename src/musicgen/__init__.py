"""musicgen package — library entry point (Phase 5, R-P12 single-sample).

Exports:
  * ``generate(config) -> SampleResult`` — the one library call.
  * ``Config`` — dataclass with ``global_seed``, ``sample_index``,
    ``dataset_root``, ``split_ratios``, etc.
  * ``SampleResult`` — frozen dataclass with 11 fields describing a
    completed (or failed) sample.
  * ``__version__`` — resolved via ``importlib.metadata.version("musicgen")``;
    falls back to ``"0.1.0+uninstalled"`` when the package is on PYTHONPATH
    but not pip-installed (e.g. raw CI runner).

Phase 6 adds ``generate_batch`` (R-P10, R-P12) and the full typer-based CLI
(R-P13). All exports now live here.
"""
from __future__ import annotations

import importlib.metadata
import os
import sys

# config.py lives at the repo root, not inside the installed package.
# Ensure it is importable when musicgen is used as a CLI entry point or
# installed via pip (where the repo root is not automatically on sys.path).
_pkg_dir = os.path.dirname(os.path.abspath(__file__))   # src/musicgen/
_src_dir = os.path.dirname(_pkg_dir)                     # src/
_repo_root = os.path.dirname(_src_dir)                   # repo root
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from musicgen.api import Config, SampleResult, generate
from musicgen.batch import BatchResult, generate_batch

try:
    __version__ = importlib.metadata.version("musicgen")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.1.0+uninstalled"

__all__ = [
    "generate", "generate_batch",
    "Config", "SampleResult", "BatchResult",
    "__version__",
]
