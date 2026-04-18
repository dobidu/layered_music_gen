"""
Regression tests for music_gen.py logging migration (R-S7).

1. AST-level print() scan — no print() calls may remain in music_gen.py.
2. Import side-effect guard — importing music_gen must not emit any logs or output.
3. Module-level logger exists — logger = logging.getLogger(__name__) is present.
4. basicConfig placement — only inside if __name__ == "__main__":, never at module scope.
"""
import ast
import importlib
import logging
import os

import pytest


class TestNoPrintCallsRemain:
    """R-S7 exit criterion: zero print() calls in music_gen.py."""

    def test_no_print_calls_remain_in_music_gen(self):
        src_path = os.path.join(
            os.path.dirname(__file__), os.pardir, "music_gen.py"
        )
        with open(os.path.abspath(src_path)) as f:
            tree = ast.parse(f.read())
        print_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "print"
        ]
        assert print_calls == [], (
            f"Found {len(print_calls)} print() call(s) in music_gen.py "
            f"at line(s): {[n.lineno for n in print_calls]}"
        )


class TestImportSideEffects:
    """Plan 01-01 property: import music_gen must be side-effect-free."""

    def test_import_music_gen_does_not_emit_logs(self, caplog):
        with caplog.at_level(logging.DEBUG):
            import music_gen as _mg
            importlib.reload(_mg)
        assert caplog.records == [], (
            f"Unexpected log records on import: {caplog.records}"
        )

    def test_import_music_gen_does_not_trigger_generation(self, capsys):
        import music_gen as _mg
        importlib.reload(_mg)
        captured = capsys.readouterr()
        assert captured.out == "", (
            f"Unexpected stdout on import: {captured.out!r}"
        )
        assert captured.err == "", (
            f"Unexpected stderr on import: {captured.err!r}"
        )


class TestLoggerSetup:
    """Verify logger infrastructure in music_gen.py."""

    def test_module_level_logger_exists(self):
        src_path = os.path.join(
            os.path.dirname(__file__), os.pardir, "music_gen.py"
        )
        with open(os.path.abspath(src_path)) as f:
            source = f.read()
        # logger = logging.getLogger(__name__) must appear at module level
        assert "logger = logging.getLogger(__name__)" in source

    def test_basic_config_only_in_main_guard(self):
        """basicConfig must appear inside if __name__ == '__main__' only."""
        src_path = os.path.join(
            os.path.dirname(__file__), os.pardir, "music_gen.py"
        )
        with open(os.path.abspath(src_path)) as f:
            tree = ast.parse(f.read())

        # Find all logging.basicConfig calls
        basic_config_calls = []
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "basicConfig"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "logging"
            ):
                basic_config_calls.append(node)

        assert len(basic_config_calls) >= 1, (
            "No logging.basicConfig call found in music_gen.py"
        )

        # Verify all basicConfig calls are inside an if __name__ == "__main__" guard
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                # Check if this is an if __name__ == "__main__" guard
                test = node.test
                if (
                    isinstance(test, ast.Compare)
                    and isinstance(test.left, ast.Name)
                    and test.left.id == "__name__"
                ):
                    # All basicConfig calls should be inside this guard
                    guard_lines = range(node.lineno, node.end_lineno + 1)
                    for bc_call in basic_config_calls:
                        assert bc_call.lineno in guard_lines, (
                            f"logging.basicConfig at line {bc_call.lineno} is "
                            f"outside __main__ guard (lines {node.lineno}-{node.end_lineno})"
                        )

    def test_no_fstring_in_logger_calls(self):
        """Logger calls must use %s format args, not f-strings."""
        src_path = os.path.join(
            os.path.dirname(__file__), os.pardir, "music_gen.py"
        )
        with open(os.path.abspath(src_path)) as f:
            tree = ast.parse(f.read())

        fstring_logger_calls = []
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("debug", "info", "warning", "error", "exception")
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "logger"
                and node.args
                and isinstance(node.args[0], ast.JoinedStr)  # f-string
            ):
                fstring_logger_calls.append(node)

        assert fstring_logger_calls == [], (
            f"Found {len(fstring_logger_calls)} logger call(s) using f-strings "
            f"at line(s): {[n.lineno for n in fstring_logger_calls]}. "
            f"Use %s format args instead."
        )
