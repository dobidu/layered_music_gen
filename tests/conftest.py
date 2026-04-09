"""
Test configuration for Phase 1 pytest skeleton.

Phase 3 will introduce `pyproject.toml` and a proper `src/musicgen/` package,
at which point this conftest's sys.path shim becomes unnecessary and should
be deleted along with this file.
"""
import os
import sys

# Make the repo root importable so `import music_gen` and
# `import enhanced_duration_validator` work without an editable install.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
