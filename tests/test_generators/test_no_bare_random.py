"""Static guard: zero bare random.<method>() in src/musicgen/generators/*.py (R-X3 / D-07).

This test is parametrized over every ``*.py`` in ``src/musicgen/generators/``
(excluding ``__init__.py``), so adding a new generator module automatically
extends the guard — no test edit required.
"""
import ast
import glob
import os

import pytest

GENERATORS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "src", "musicgen", "generators")
)


def _bare_random_calls(source: str):
    """Return ``random.<attr>(...)`` Call nodes, excluding the ``random.Random``
    constructor (which is the RNG factory, not a bare draw)."""
    tree = ast.parse(source)
    hits = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "random"
                and node.func.attr != "Random"):
            hits.append(node)
    return hits


@pytest.mark.parametrize("path", sorted(
    p for p in glob.glob(os.path.join(GENERATORS_DIR, "*.py"))
    if not p.endswith("__init__.py")
))
def test_no_bare_random_in_generator_module(path):
    with open(path, "r") as f:
        source = f.read()
    hits = _bare_random_calls(source)
    assert hits == [], (
        f"{os.path.basename(path)}: {len(hits)} bare random.<method>() at lines "
        f"{[n.lineno for n in hits]} — use rng.<method>() per D-07."
    )
