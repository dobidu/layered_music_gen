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

Phase 6 will add ``generate_batch`` (R-P10) and the ``typer``-based CLI
(R-P13); the public surface is forward-compatible.
"""
from __future__ import annotations

import importlib.metadata

from musicgen.api import Config, SampleResult, generate

try:
    __version__ = importlib.metadata.version("musicgen")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.1.0+uninstalled"

__all__ = ["generate", "Config", "SampleResult", "__version__"]
