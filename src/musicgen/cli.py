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
from typing import List

from musicgen.api import Config
from musicgen import calibrate
from musicgen.batch import generate_batch
from musicgen.midi_indexer import index_midi_dataset
from musicgen.audio_indexer import index_audio_dataset

app = typer.Typer(
    help="musicgen — synthetic music dataset generator",
    no_args_is_help=True,
)

samples_app = typer.Typer(help="Commands for building and managing sample libraries.", no_args_is_help=True)
app.add_typer(samples_app, name="samples")

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


_VALID_SAMPLE_MODES = {"alongside", "substitution", "adlib", "off"}


@app.command()
def generate(
    seed: int = typer.Option(..., "--seed", "-s", help="Global RNG seed (required for reproducibility)."),
    count: int = typer.Option(1, "--count", "-n", help="Number of samples to generate."),
    out: str = typer.Option("./dataset", "--out", "-o", help="Dataset output directory."),
    workers: Optional[int] = typer.Option(None, "--workers", "-w", help="Parallel workers (default: os.cpu_count())."),
    output_mode: str = typer.Option("full", "--output-mode", "-m", help="full | mix-only | stems-only | midi-only."),
    genre: Optional[List[str]] = typer.Option(None, "--genre", "-g", help="Genre name(s) (repeatable)."),
    genres_dir: Optional[str] = typer.Option(None, "--genres-dir", help="Custom genres directory."),
    # Sample composition flags (M5)
    sample_db: Optional[str] = typer.Option(None, "--sample-db", help="SampleManager JSON database path. Enables sample composition."),
    sample_beat: str = typer.Option("alongside", "--sample-beat", help="Beat layer sample mode: alongside|substitution|adlib|off."),
    sample_bassline: str = typer.Option("alongside", "--sample-bassline", help="Bassline layer sample mode: alongside|substitution|adlib|off."),
    sample_melody: str = typer.Option("off", "--sample-melody", help="Melody layer sample mode: alongside|substitution|adlib|off."),
    sample_harmony: str = typer.Option("off", "--sample-harmony", help="Harmony layer sample mode: alongside|substitution|adlib|off."),
    sample_gain: float = typer.Option(-3.0, "--sample-gain", help="Gain (dB) applied to all sample layers."),
    sample_min_score: float = typer.Option(0.0, "--sample-min-score", help="Min musicality score for sample selection (0=disabled)."),
    # Neural backend flags (v0.5)
    chord_backend: str = typer.Option("markov", "--chord-backend", help="Chord generation backend: markov | neural."),
    melody_backend: str = typer.Option("markov", "--melody-backend", help="Melody generation backend: markov | neural."),
    models_dir: Optional[str] = typer.Option(None, "--models-dir", help="Directory containing trained .pt model files."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose (DEBUG) logging."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Quiet (ERROR-only) logging."),
) -> None:
    """Generate a dataset of synthetic music samples."""
    _setup_logging(verbose, quiet)

    if output_mode not in _VALID_OUTPUT_MODES:
        typer.echo(
            f"Error: --output-mode must be one of {sorted(_VALID_OUTPUT_MODES)}, got {output_mode!r}",
            err=True,
        )
        raise typer.Exit(code=1)

    overrides: dict = {
        "global_seed": seed,
        "count": count,
        "dataset_root": os.path.abspath(out),
        "workers": workers,
        "output_mode": output_mode,
    }
    if genre:
        overrides["genre"] = list(genre)
    if genres_dir:
        overrides["genres_dir"] = os.path.abspath(genres_dir)
    if chord_backend != "markov":
        overrides["chord_backend"] = chord_backend
    if melody_backend != "markov":
        overrides["melody_backend"] = melody_backend
    if models_dir:
        overrides["models_dir"] = os.path.abspath(models_dir)

    # Build SampleCompositionConfig if --sample-db is set.
    if sample_db:
        from musicgen.sample_composition import SampleLayerRule, SampleCompositionConfig
        _layer_modes = {
            "beat":     sample_beat,
            "bassline": sample_bassline,
            "melody":   sample_melody,
            "harmony":  sample_harmony,
        }
        for lname, lmode in _layer_modes.items():
            if lmode not in _VALID_SAMPLE_MODES:
                typer.echo(
                    f"Error: --sample-{lname} must be one of {sorted(_VALID_SAMPLE_MODES)}, got {lmode!r}",
                    err=True,
                )
                raise typer.Exit(code=1)

        layer_rules = {
            layer: SampleLayerRule(layer=layer, mode=mode, gain_db=sample_gain)
            for layer, mode in _layer_modes.items()
            if mode != "off"
        }
        overrides["sample_composition"] = SampleCompositionConfig(
            sample_db_path=os.path.abspath(sample_db),
            layer_rules=layer_rules,
            global_min_musicality=sample_min_score if sample_min_score > 0 else None,
        )

    cfg = Config.load(cli_overrides=overrides)

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


@app.command(name="list-genres")
def list_genres(
    genres_dir: Optional[str] = typer.Option(None, "--genres-dir", help="Genres directory (default: <repo>/genres/)."),
) -> None:
    """List all available genre presets with descriptions."""
    import config as cfg_module
    from musicgen.genre import load_genre

    _genres_dir = os.path.abspath(genres_dir) if genres_dir else cfg_module.DEFAULT_GENRES_DIR
    if not os.path.isdir(_genres_dir):
        typer.echo(f"Error: genres directory not found: {_genres_dir}", err=True)
        raise typer.Exit(code=1)

    genre_names = sorted(
        name for name in os.listdir(_genres_dir)
        if os.path.isdir(os.path.join(_genres_dir, name))
        and os.path.isfile(os.path.join(_genres_dir, name, "spec.json"))
    )
    if not genre_names:
        typer.echo(f"No genres found in {_genres_dir}")
        return

    typer.echo(f"Available genres ({len(genre_names)}):\n")
    for name in genre_names:
        try:
            spec = load_genre(name, _genres_dir)
            desc = spec.description or "(no description)"
        except Exception:
            desc = "(error loading spec)"
        typer.echo(f"  {name:<16} {desc}")


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


@app.command(name="index-midi")
def index_midi(
    dataset: str = typer.Option(..., "--dataset", "-d", help="musicgen dataset root directory."),
    out: str = typer.Option("./midi_db.json", "--out", "-o", help="Output MidiManager JSON database path."),
    midi_dir: Optional[str] = typer.Option(None, "--midi-dir", help="Base dir for relative MIDI paths stored in the db."),
    export_csv: Optional[str] = typer.Option(None, "--csv", help="Also export a CSV of all indexed entries."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose (DEBUG) logging."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Quiet (ERROR-only) logging."),
) -> None:
    """Index generated MIDI files into a MidiManager database (midi_file_manager)."""
    _setup_logging(verbose, quiet)

    dataset_path = os.path.abspath(dataset)
    out_path = os.path.abspath(out)
    csv_path = os.path.abspath(export_csv) if export_csv else None
    midi_dir_path = os.path.abspath(midi_dir) if midi_dir else None

    try:
        count = index_midi_dataset(
            dataset_root=dataset_path,
            out_db=out_path,
            midi_dir=midi_dir_path,
            export_csv=csv_path,
        )
    except ImportError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)
    except FileNotFoundError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Indexed {count} MIDI entries → {out_path}")
    if csv_path:
        typer.echo(f"CSV exported → {csv_path}")


@app.command(name="index-audio")
def index_audio(
    dataset: str = typer.Option(..., "--dataset", "-d", help="musicgen dataset root directory."),
    out: str = typer.Option("./audio_db.json", "--out", "-o", help="Output SampleManager JSON database path."),
    samples_dir: Optional[str] = typer.Option(None, "--samples-dir", help="Base dir for relative WAV paths stored in the db."),
    export_csv: Optional[str] = typer.Option(None, "--csv", help="Also export a CSV of all indexed entries."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose (DEBUG) logging."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Quiet (ERROR-only) logging."),
) -> None:
    """Index generated WAV stems into a SampleManager database (audio_sample_manager)."""
    _setup_logging(verbose, quiet)

    dataset_path = os.path.abspath(dataset)
    out_path = os.path.abspath(out)
    csv_path = os.path.abspath(export_csv) if export_csv else None
    samples_dir_path = os.path.abspath(samples_dir) if samples_dir else None

    try:
        count = index_audio_dataset(
            dataset_root=dataset_path,
            out_db=out_path,
            samples_dir=samples_dir_path,
            export_csv=csv_path,
        )
    except ImportError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)
    except FileNotFoundError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Indexed {count} audio entries → {out_path}")
    if csv_path:
        typer.echo(f"CSV exported → {csv_path}")


# ---------------------------------------------------------------------------
# samples subgroup (M5)
# ---------------------------------------------------------------------------

@samples_app.command(name="build")
def samples_build(
    wav_dir: str = typer.Option(..., "--dir", "-d", help="Directory containing WAV/FLAC/OGG/AIF files."),
    output: str = typer.Option(..., "--output", "-o", help="Output SampleManager JSON database path."),
    category: Optional[str] = typer.Option(None, "--category", "-c", help="Force all samples to this category (beat|bass|melody|harmony). Default: infer from filename."),
    genre: Optional[List[str]] = typer.Option(None, "--genre", "-g", help="Genre tag(s) applied to every sample (repeatable)."),
    mood: Optional[List[str]] = typer.Option(None, "--mood", help="Mood tag(s) applied to every sample (repeatable)."),
    tags: Optional[List[str]] = typer.Option(None, "--tag", "-t", help="Extra tag(s) applied to every sample (repeatable)."),
    musicality: bool = typer.Option(False, "--musicality", help="Score each sample with musicality.explain() (requires musicgen[samples])."),
    recursive: bool = typer.Option(False, "--recursive", "-r", help="Walk subdirectories."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose (DEBUG) logging."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Quiet (ERROR-only) logging."),
) -> None:
    """Annotate a WAV directory into a SampleManager JSON library for use with --sample-db."""
    _setup_logging(verbose, quiet)
    from musicgen.sample_builder import build_library

    wav_dir_abs = os.path.abspath(wav_dir)
    output_abs = os.path.abspath(output)

    if not os.path.isdir(wav_dir_abs):
        typer.echo(f"Error: directory not found: {wav_dir_abs}", err=True)
        raise typer.Exit(code=1)

    if category and category not in ("beat", "bass", "melody", "harmony"):
        typer.echo("Error: --category must be beat|bass|melody|harmony", err=True)
        raise typer.Exit(code=1)

    try:
        count = build_library(
            wav_dir=wav_dir_abs,
            output=output_abs,
            category_override=category,
            genre=list(genre) if genre else None,
            mood=list(mood) if mood else None,
            tags=list(tags) if tags else None,
            compute_musicality=musicality,
            recursive=recursive,
        )
    except ImportError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)
    except FileNotFoundError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Indexed {count} samples → {output_abs}")


# ---------------------------------------------------------------------------
# extract-sequences (v0.5 Phase 1)
# ---------------------------------------------------------------------------

@app.command(name="extract-sequences")
def extract_sequences_cmd(
    dataset: str = typer.Option(..., "--dataset", "-d", help="musicgen dataset root directory."),
    output: str = typer.Option(..., "--output", "-o", help="Output sequences JSON path."),
    min_musicality: float = typer.Option(0.0, "--min-musicality", help="Skip samples below this musicality score (0=keep all)."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose (DEBUG) logging."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Quiet (ERROR-only) logging."),
) -> None:
    """Extract chord and melody token sequences from a musicgen dataset for neural training."""
    _setup_logging(verbose, quiet)
    from musicgen.corpus_extractor import extract_sequences

    dataset_abs = os.path.abspath(dataset)
    output_abs = os.path.abspath(output)

    if not os.path.isdir(dataset_abs):
        typer.echo(f"Error: dataset directory not found: {dataset_abs}", err=True)
        raise typer.Exit(code=1)

    counts = extract_sequences(
        dataset_root=dataset_abs,
        output_path=output_abs,
        min_musicality=min_musicality,
    )
    typer.echo(
        f"Extracted {counts['chord']} chord sequences, "
        f"{counts['melody']} melody sequences → {output_abs}"
    )


# ---------------------------------------------------------------------------
# train (v0.5 Phase 2)
# ---------------------------------------------------------------------------

@app.command(name="train")
def train_cmd(
    sequences: str = typer.Option(..., "--sequences", "-s", help="sequences.json produced by extract-sequences."),
    layer: str = typer.Option(..., "--layer", "-l", help="Layer to train: chord | melody."),
    genre: Optional[List[str]] = typer.Option(None, "--genre", "-g", help="Filter to these genre(s) (repeatable). Default: all genres."),
    epochs: int = typer.Option(200, "--epochs", "-e", help="Max training epochs."),
    output_dir: str = typer.Option("./models", "--output-dir", "-o", help="Directory to write .pt model files."),
    seed: int = typer.Option(42, "--seed", help="Reproducibility seed for training."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose (DEBUG) logging."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Quiet (ERROR-only) logging."),
) -> None:
    """Train a neural chord or melody LSTM on sequences extracted from a musicgen dataset."""
    _setup_logging(verbose, quiet)

    if layer not in ("chord", "melody"):
        typer.echo("Error: --layer must be 'chord' or 'melody'", err=True)
        raise typer.Exit(code=1)

    try:
        from musicgen.neural.trainer import train, save_model
    except ImportError as exc:
        typer.echo(f"Error: torch not installed — run: pip install 'musicgen[neural]'\n{exc}", err=True)
        raise typer.Exit(code=1)

    sequences_abs = os.path.abspath(sequences)
    output_dir_abs = os.path.abspath(output_dir)

    if not os.path.isfile(sequences_abs):
        typer.echo(f"Error: sequences file not found: {sequences_abs}", err=True)
        raise typer.Exit(code=1)

    genre_filter = list(genre) if genre else None
    model_name = f"{layer}{'_' + '_'.join(sorted(genre_filter)) if genre_filter else ''}.pt"
    model_path = os.path.join(output_dir_abs, model_name)

    try:
        sampler = train(
            sequences_path=sequences_abs,
            layer=layer,
            genres=genre_filter,
            epochs=epochs,
            seed=seed,
        )
    except (ValueError, ImportError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)

    save_model(sampler, model_path)
    typer.echo(f"Trained {layer} model → {model_path}")


if __name__ == "__main__":
    app()
