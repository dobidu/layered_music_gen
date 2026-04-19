"""Static guard: zero bare random.<method>() in src/musicgen/**/*.py (D-17/D-31).

This file is a Wave 0 skipping stub. Plan 04-05 (Wave 5) replaces the entire
body with real tests. The stub exists so pytest collection succeeds from Wave 0
onward; collection failure breaks pre-commit AST tests that iterate tests/.
"""
import pytest

pytest.skip(
    "Plan 04-00 (Wave 0) scaffold — Plan 04-05 (Wave 5) will replace this stub.",
    allow_module_level=True,
)


def test_placeholder():
    """Placeholder; will be replaced by Plan 04-05's real tests."""
    pass
