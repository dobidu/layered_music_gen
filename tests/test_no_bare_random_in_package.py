"""Static guard: zero bare random.<method>() in src/musicgen/**/*.py (D-17/D-31).

Generalizes the existing scoped guards:
  - tests/test_sampler.py::test_no_bare_random_in_sampler  (sampler.py only)
  - tests/test_generators/test_no_bare_random.py           (generators/*.py only)

This package-wide version is PARAMETRIZED over every ``*.py`` under
``src/musicgen/`` (recursive; excludes ``__init__.py``), so adding any new
module automatically extends the guard — no test file edits required.

The ``random.Random`` constructor IS permitted (it's the RNG factory, not a
bare draw). Every other ``random.<attr>(...)`` call is forbidden.
"""
from __future__ import annotations

import ast
import glob
import os

import pytest

PACKAGE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, "src", "musicgen")
)


def _bare_random_calls(source: str):
    """Return ``random.<attr>(...)`` Call nodes excluding the ``random.Random``
    constructor (matches the helper in tests/test_sampler.py lines 165-180).
    """
    tree = ast.parse(source)
    hits = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "random"
            and node.func.attr != "Random"
        ):
            hits.append(node)
    return hits


def _collect_package_modules():
    """List every *.py under src/musicgen/ (recursive; excludes __init__.py)."""
    return sorted(
        p for p in glob.glob(os.path.join(PACKAGE_DIR, "**", "*.py"), recursive=True)
        if not p.endswith("__init__.py")
    )


@pytest.mark.parametrize("path", _collect_package_modules())
def test_no_bare_random_in_package_module(path):
    """Every module under src/musicgen/ must use the injected rng (D-17).

    Failing this test means a ``random.choice``, ``random.random``,
    ``random.randint``, ``random.choices``, or ``random.uniform`` call
    slipped into the package. Use ``rng.<method>(...)`` with the injected
    ``rng: random.Random`` parameter instead.
    """
    with open(path, "r") as f:
        source = f.read()
    hits = _bare_random_calls(source)
    assert hits == [], (
        f"{os.path.basename(path)}: {len(hits)} bare random.<method>() at lines "
        f"{[n.lineno for n in hits]} — use rng.<method>() per D-17."
    )


def test_package_scan_covers_all_phase4_modules():
    """Meta-test: the scan collects at least sampler + generators/ + the 4 Phase 4 modules.

    Catches the case where PACKAGE_DIR mis-resolves and the parametrize returns
    an empty list (which would trivially 'pass' the bare-random test above).
    """
    modules = _collect_package_modules()
    relative = [os.path.relpath(m, PACKAGE_DIR) for m in modules]
    # Must cover all 4 Phase 4 new modules + Phase 3 sampler + generators
    expected_present = {
        "sampler.py", "renderer.py", "mixer.py", "annotator.py", "beats.py",
        "duration_validator.py",
        os.path.join("generators", "beat.py"),
        os.path.join("generators", "chord.py"),
        os.path.join("generators", "melody.py"),
        os.path.join("generators", "bassline.py"),
    }
    missing = expected_present - set(relative)
    assert not missing, f"package scan missed expected modules: {missing}"
