"""Static guard: zero bare random.<method>() in src/musicgen/**/*.py (D-17/D-31/D-42).

Generalizes the existing scoped guards:
  - tests/test_sampler.py::test_no_bare_random_in_sampler  (sampler.py only)
  - tests/test_generators/test_no_bare_random.py           (generators/*.py only)

This package-wide version is PARAMETRIZED over every ``*.py`` under
``src/musicgen/`` (recursive; excludes ``__init__.py``), so adding any new
module automatically extends the guard — no test file edits required.

Permitted attributes (Phase 5 D-42 widened the allow-list):
  - ``random.Random``   — RNG factory (e.g. ``random.Random(seed)``).
  - ``random.getstate`` — state snapshot (used by ``seeds.save_random_state``).
  - ``random.setstate`` — state restore (used by ``seeds.save_random_state``).

Every other ``random.<attr>(...)`` call is forbidden — use the injected
``rng.<method>(...)`` instead (Phase 3 D-07 / Phase 5 D-17 contract).
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
    """Return ``random.<attr>(...)`` Call nodes excluding the permitted set.

    Permitted attrs:
      - ``Random``       — RNG factory (e.g. ``random.Random(seed)``).
      - ``getstate``     — state snapshot (used by ``seeds.save_random_state``).
      - ``setstate``     — state restore (used by ``seeds.save_random_state``).

    Every other ``random.<attr>(...)`` call is forbidden — use the injected
    ``rng.<method>(...)`` instead (Phase 3 D-07 / Phase 5 D-17 contract).
    """
    tree = ast.parse(source)
    hits = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "random"
            and node.func.attr not in {"Random", "getstate", "setstate"}
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


@pytest.mark.xfail(
    strict=False,
    reason=(
        "Phase 5 modules (seeds/writer/manifest/api/musicality) land in "
        "Waves 1-4; expected_present documents the forward guard. Plan 05-05 "
        "removes this xfail once api.py (last of the five) is created."
    ),
)
def test_package_scan_covers_all_package_modules():
    """Meta-test: the scan collects all Phase 3/4/5 package modules.

    Catches the case where PACKAGE_DIR mis-resolves and the parametrize returns
    an empty list (which would trivially 'pass' the bare-random test above).

    Phase 5 (D-42) widens `expected_present` to include the five modules that
    Waves 1-4 create: seeds.py (Wave 1), writer.py + manifest.py (Wave 3),
    api.py + musicality.py (Wave 4). Until those land, this test xfails —
    the widened set is a forward guard, not a current-state assertion.
    """
    modules = _collect_package_modules()
    relative = [os.path.relpath(m, PACKAGE_DIR) for m in modules]
    # Must cover all Phase 3 + Phase 4 + Phase 5 modules under src/musicgen/
    expected_present = {
        "sampler.py", "renderer.py", "mixer.py", "annotator.py", "beats.py",
        "duration_validator.py",
        # Phase 5 additions (added in Waves 1-4):
        "seeds.py", "writer.py", "manifest.py", "api.py", "musicality.py",
        os.path.join("generators", "beat.py"),
        os.path.join("generators", "chord.py"),
        os.path.join("generators", "melody.py"),
        os.path.join("generators", "bassline.py"),
    }
    missing = expected_present - set(relative)
    assert not missing, f"package scan missed expected modules: {missing}"
