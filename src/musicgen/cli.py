"""CLI entry point (stub — real CLI lands in Phase 6 per D-18).

Provides the `musicgen` console script so `pip install -e . && musicgen --help`
works. One stub command (`info`) demonstrates the plumbing; real batch/generate
commands are out of scope for Phase 3.
"""
from __future__ import annotations

import logging
from typing import Optional

import typer

app = typer.Typer(
    help="musicgen — synthetic music dataset generator",
    no_args_is_help=True,
)


@app.command()
def info() -> None:
    """Print package metadata and a friendly pointer at Phase 6's full CLI."""
    typer.echo("musicgen 0.1.0 — Phase 3 package skeleton")
    typer.echo(
        "Real CLI (generate / batch / clean / calibrate) arrives in Phase 6. "
        "Today: `python music_gen.py` runs one smoke-test song."
    )


if __name__ == "__main__":  # direct-invocation fallback
    app()
