"""Full CLI entry point (R-P13, D-61..D-65).

Three commands:
  musicgen generate --count N --out DIR --seed S [--workers W] [--output-mode M] [-v/-q]
  musicgen clean --failed [--out DIR]
  musicgen calibrate [--out DIR] [-v]

The Phase 3 stub ``info`` command is removed; this module is the production CLI.
Entry point unchanged: ``musicgen = "musicgen.cli:app"`` in pyproject.toml.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from typing import Optional

# config.py lives at the repo root (not inside the installed package).
# When invoked as a CLI entry point, the repo root may not be on sys.path.
# Detect and add it so `from config import Config` succeeds in api.py.
_pkg_dir = os.path.dirname(os.path.abspath(__file__))      # src/musicgen/
_src_dir = os.path.dirname(_pkg_dir)                        # src/
_repo_root = os.path.dirname(_src_dir)                      # repo root
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import typer

from musicgen.api import Config
from musicgen import calibrate
from musicgen.batch import generate_batch

app = typer.Typer(
    help="musicgen — synthetic music dataset generator",
    no_args_is_help=True,
)

_VALID_OUTPUT_MODES = {"full", "mix-only", "stems-only", "midi-only"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _setup_logging(verbose: bool, quiet: bool) -> None:
    if quiet:
        level = logging.ERROR
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command()
def generate(
    seed: int = typer.Option(..., "--seed", "-s", help="Global RNG seed (required for reproducibility)."),
    count: int = typer.Option(1, "--count", "-n", help="Number of samples to generate."),
    out: str = typer.Option("./dataset", "--out", "-o", help="Dataset output directory."),
    workers: Optional[int] = typer.Option(None, "--workers", "-w", help="Parallel workers (default: os.cpu_count())."),
    output_mode: str = typer.Option("full", "--output-mode", "-m", help="full | mix-only | stems-only | midi-only."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose (DEBUG) logging."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Quiet (ERROR-only) logging."),
) -> None:
    """Generate a dataset of synthetic music samples."""
    _setup_logging(verbose, quiet)

    if output_mode not in _VALID_OUTPUT_MODES:
        typer.echo(
            f"Error: --output-mode must be one of {sorted(_VALID_OUTPUT_MODES)}, "
            f"got {output_mode!r}",
            err=True,
        )
        raise typer.Exit(code=1)

    cfg = Config.load(cli_overrides={
        "global_seed": seed,
        "count": count,
        "dataset_root": os.path.abspath(out),
        "workers": workers,
        "output_mode": output_mode,
    })

    result = generate_batch(cfg)

    typer.echo(
        f"Batch done: {result.succeeded} ok, {result.failed} failed, "
        f"{result.skipped} skipped / {result.total} total "
        f"({result.duration_seconds:.1f} s)"
    )

    if result.failed > 0:
        typer.echo(
            f"Warning: {result.failed} sample(s) failed. "
            "Run 'musicgen clean --failed' to remove partial output.",
            err=True,
        )
        raise typer.Exit(code=1)


@app.command()
def clean(
    failed: bool = typer.Option(False, "--failed", help="Remove directories for failed samples."),
    out: str = typer.Option("./dataset", "--out", "-o", help="Dataset directory to clean."),
) -> None:
    """Remove incomplete or failed sample directories."""
    if not failed:
        typer.echo(
            "Error: specify --failed to remove failed sample directories.",
            err=True,
        )
        raise typer.Exit(code=1)

    dataset_root = os.path.abspath(out)
    manifest_path = os.path.join(dataset_root, "manifest.jsonl")

    if not os.path.isfile(manifest_path):
        typer.echo("Nothing to clean (no manifest.jsonl found).")
        return

    # Build last-status-wins map from manifest.
    status_map: dict = {}
    with open(manifest_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                idx = entry.get("sample_index")
                if idx is not None:
                    status_map[idx] = entry.get("status", "")
            except json.JSONDecodeError:
                pass

    removed = 0
    for idx, status in status_map.items():
        if status == "failed":
            sample_dir = os.path.join(dataset_root, f"{idx:06d}")
            sentinel = os.path.join(sample_dir, "sample.json")
            if os.path.isdir(sample_dir) and not os.path.isfile(sentinel):
                import shutil
                shutil.rmtree(sample_dir, ignore_errors=True)
                removed += 1
                typer.echo(f"Removed: {sample_dir}")

    typer.echo(f"Cleaned {removed} failed sample director{'y' if removed == 1 else 'ies'}.")


@app.command(name="calibrate")
def calibrate_cmd(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose (DEBUG) logging."),
) -> None:
    """Measure and cache the FluidSynth pre-roll offset."""
    _setup_logging(verbose, False)
    cfg = Config.load()
    offset_s = calibrate.measure_and_save_preroll(cfg.project_root)
    typer.echo(f"Pre-roll offset: {offset_s:.6f} s")
    if offset_s == 0.0:
        typer.echo(
            "(FluidSynth absent or no soundfonts found — offset defaulted to 0.0)"
        )


if __name__ == "__main__":
    app()
