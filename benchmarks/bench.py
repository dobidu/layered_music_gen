#!/usr/bin/env python3
"""
musicgen benchmark suite.

Usage:
    python benchmarks/bench.py                       # all benchmarks
    python benchmarks/bench.py --fast-only           # skip FluidSynth + batch
    python benchmarks/bench.py --no-plot             # skip chart generation
    python benchmarks/bench.py --workers 1 2 4 8     # batch worker counts
    python benchmarks/bench.py --batch-samples 16    # samples per batch run
    python benchmarks/bench.py --n 20                # repetitions per benchmark
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import psutil
import scipy.io.wavfile as wavfile

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.disable(logging.CRITICAL)

from config import Config
from musicgen.genre import GenreSpec, load_genre, merge_genres, resolve_genres
from musicgen.sampler import (
    SongParams,
    generate_random_key,
    generate_random_swing,
    generate_random_tempo,
    generate_random_time_signature,
    generate_song_arrangement,
    generate_song_measures,
)
from musicgen.seeds import derive_sample_seed, make_rngs, RNG_PARAMS, RNG_GENERATORS
from musicgen import renderer as _renderer

GENRES_DIR = str(REPO_ROOT / "genres")
BUILTIN_GENRES = ["jazz", "hip-hop", "blues", "pop", "electronic", "latin", "reggae", "classical"]
HAS_FLUIDSYNTH = shutil.which("fluidsynth") is not None


# ──────────────────────────────────────────────
# System info
# ──────────────────────────────────────────────

def get_system_info() -> Dict[str, Any]:
    cpu = platform_cpu()
    mem = psutil.virtual_memory()
    import importlib.metadata
    try:
        mgv = importlib.metadata.version("musicgen")
    except Exception:
        mgv = "unknown"
    fs_ver = "n/a"
    if HAS_FLUIDSYNTH:
        try:
            r = subprocess.run(
                ["fluidsynth", "--version"], capture_output=True, text=True, timeout=5
            )
            line = (r.stdout + r.stderr).splitlines()[0]
            fs_ver = line.strip()
        except Exception:
            fs_ver = "unknown"
    import platform as _pl
    return {
        "hostname":          _pl.node(),
        "os":                _pl.system(),
        "os_version":        _pl.version()[:60],
        "cpu":               cpu,
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "ram_gb":            round(mem.total / 1024**3, 1),
        "python":            _pl.python_version(),
        "musicgen_version":  mgv,
        "has_fluidsynth":    HAS_FLUIDSYNTH,
        "fluidsynth_version": fs_ver,
    }


def platform_cpu() -> str:
    import platform as _pl
    cpu = _pl.processor() or _pl.machine()
    # macOS: use sysctl for a friendlier name
    if _pl.system() == "Darwin":
        try:
            r = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=3,
            )
            if r.stdout.strip():
                return r.stdout.strip()
        except Exception:
            pass
    return cpu


# ──────────────────────────────────────────────
# Timer
# ──────────────────────────────────────────────

def time_fn(fn: Callable, n: int = 10, warmup: int = 2) -> Dict[str, Any]:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1_000)
    return {
        "mean_ms":   round(statistics.mean(samples), 3),
        "std_ms":    round(statistics.stdev(samples) if len(samples) > 1 else 0, 3),
        "min_ms":    round(min(samples), 3),
        "max_ms":    round(max(samples), 3),
        "median_ms": round(statistics.median(samples), 3),
        "n":         n,
    }


# ──────────────────────────────────────────────
# Mock renderer (no FluidSynth)
# ──────────────────────────────────────────────

def _make_silent_wav(path: str, duration_s: float = 4.0, sr: int = 44100) -> None:
    data = np.zeros((int(sr * duration_s), 2), dtype=np.int16)
    wavfile.write(path, sr, data)


def _mock_render_stems(midi_paths, soundfonts, out_dir, cfg=None):
    from musicgen.renderer import RenderResult
    os.makedirs(out_dir, exist_ok=True)
    stem_paths = {}
    for layer in ("beat", "melody", "harmony", "bassline"):
        path = os.path.join(out_dir, f"{layer}.wav")
        _make_silent_wav(path, duration_s=4.0)
        stem_paths[layer] = path
    return RenderResult(
        stem_paths=stem_paths,
        sample_rate=44100,
        channels=2,
        duration_seconds=4.0,
        fluidsynth_version=_renderer.FLUIDSYNTH_VERSION,
    )


def _mock_pick_soundfonts(cfg, rng):
    return {layer: "/dev/null" for layer in ("beat", "melody", "harmony", "bassline")}


def _mock_musicality(path):
    return 0.5, {"tempo": 0.5, "harmony": 0.5, "rhythm": 0.5, "timbre": 0.5, "snr": 0.5}


# ──────────────────────────────────────────────
# Individual benchmarks
# ──────────────────────────────────────────────

def bench_sampler(n: int) -> Dict:
    rng = random.Random(42)
    return {"sampler_sample": time_fn(lambda: SongParams.sample(rng), n=n)}


def bench_genre(n: int) -> Dict:
    all_specs = [load_genre(g, GENRES_DIR) for g in BUILTIN_GENRES]
    jazz = all_specs[0]
    latin = all_specs[5]

    results = {}
    results["genre_load_single"] = time_fn(
        lambda: load_genre("jazz", GENRES_DIR), n=n
    )
    results["genre_load_all_8"] = time_fn(
        lambda: [load_genre(g, GENRES_DIR) for g in BUILTIN_GENRES], n=n
    )
    results["genre_merge_2"] = time_fn(
        lambda: merge_genres([jazz, latin]), n=n
    )
    results["genre_merge_8"] = time_fn(
        lambda: merge_genres(all_specs), n=n
    )
    results["genre_resolve_2_names"] = time_fn(
        lambda: resolve_genres(["jazz", "latin"], GENRES_DIR), n=n
    )
    return results


def bench_generators(n: int) -> Dict:
    from musicgen.generators.chord import generate_chord_progression
    from musicgen.generators.melody import generate_melody
    from musicgen.generators.bassline import generate_bassline
    from musicgen.generators.beat import generate_beat

    rng = random.Random(7)
    params = SongParams.sample(rng)
    part = params.song_unique_parts[0]
    time_sig = params.signatures_per_part[part]
    measures = params.measures_per_part[part]
    key = params.key
    tempo = params.tempo
    swing = params.swing_amount
    cfg = Config()

    results = {}

    with tempfile.TemporaryDirectory() as td:
        name_part = os.path.join(td, f"bench-{part}")

        def _chord():
            r = random.Random(1)
            cp, _ = generate_chord_progression(
                key, tempo, time_sig, measures, name_part, part,
                cfg.chord_patterns_file, r,
            )
            return cp

        results["chord_progression"] = time_fn(_chord, n=n)

        chord_prog, _ = generate_chord_progression(
            key, tempo, time_sig, measures, name_part, part,
            cfg.chord_patterns_file, random.Random(1),
        )

        # melody has a pre-existing Markov zero-weight bug on some seeds
        try:
            mel, _ = generate_melody(key, tempo, time_sig, measures,
                                     name_part, part, chord_prog, random.Random(2))

            def _melody():
                r = random.Random(2)
                return generate_melody(key, tempo, time_sig, measures,
                                       name_part, part, chord_prog, r)
            results["melody"] = time_fn(_melody, n=n)
        except (ValueError, Exception):
            melody_fallback = []
            mel = melody_fallback

        def _bassline():
            r = random.Random(3)
            return generate_bassline(key, tempo, time_sig, measures,
                                     name_part, part, chord_prog, mel, r)

        results["bassline"] = time_fn(_bassline, n=n)

        def _beat():
            r = random.Random(4)
            return generate_beat(part, tempo, time_sig, measures,
                                 name_part, swing, r, cfg=cfg)

        results["beat"] = time_fn(_beat, n=n)

    return results


def bench_full_midi_pipeline(n: int) -> Dict:
    """Time generating all MIDI for one song (all generators, all parts)."""
    from musicgen.generators.chord import generate_chord_progression
    from musicgen.generators.melody import generate_melody
    from musicgen.generators.bassline import generate_bassline
    from musicgen.generators.beat import generate_beat

    cfg = Config()

    GOOD_SEED = 3  # verified to avoid melody Markov zero-weight bug

    def _pipeline():
        with tempfile.TemporaryDirectory() as td:
            rng_p = random.Random(GOOD_SEED)
            rng_g = random.Random(GOOD_SEED + 1)
            params = SongParams.sample(rng_p)
            key = params.key
            tempo = params.tempo
            swing = params.swing_amount
            for part in params.song_unique_parts:
                ts = params.signatures_per_part[part]
                ms = params.measures_per_part[part]
                np_ = os.path.join(td, part)
                cp, _ = generate_chord_progression(
                    key, tempo, ts, ms, np_, part, cfg.chord_patterns_file, rng_g)
                try:
                    mel, _ = generate_melody(key, tempo, ts, ms, np_, part, cp, rng_g)
                except (ValueError, Exception):
                    mel = []
                generate_bassline(key, tempo, ts, ms, np_, part, cp, mel, rng_g)
                generate_beat(part, tempo, ts, ms, np_, swing, rng_g, cfg=cfg)

    return {"full_midi_pipeline": time_fn(_pipeline, n=n, warmup=1)}


def bench_genre_overhead(n: int) -> Dict:
    """Compare full MIDI pipeline: no genre vs 1 genre vs 2 merged."""
    from musicgen.generators.chord import generate_chord_progression
    from musicgen.generators.melody import generate_melody
    from musicgen.generators.bassline import generate_bassline
    from musicgen.generators.beat import generate_beat

    cfg = Config()

    GOOD_SEED = 3

    def _pipeline(genre_spec=None):
        with tempfile.TemporaryDirectory() as td:
            rng_p = random.Random(GOOD_SEED)
            rng_g = random.Random(GOOD_SEED + 1)
            params = SongParams.sample(rng_p, genre_spec=genre_spec)
            key = params.key
            tempo = params.tempo
            swing = params.swing_amount
            for part in params.song_unique_parts:
                ts = params.signatures_per_part[part]
                ms = params.measures_per_part[part]
                np_ = os.path.join(td, part)
                cp, _ = generate_chord_progression(
                    key, tempo, ts, ms, np_, part, cfg.chord_patterns_file, rng_g)
                try:
                    mel, _ = generate_melody(key, tempo, ts, ms, np_, part, cp, rng_g)
                except (ValueError, Exception):
                    mel = []
                generate_bassline(key, tempo, ts, ms, np_, part, cp, mel, rng_g)
                generate_beat(part, tempo, ts, ms, np_, swing, rng_g, cfg=cfg)

    jazz = load_genre("jazz", GENRES_DIR)
    latin = load_genre("latin", GENRES_DIR)
    merged = merge_genres([jazz, latin])

    results = {}
    results["genre_overhead_none"] = time_fn(lambda: _pipeline(None), n=n, warmup=1)
    results["genre_overhead_1"] = time_fn(lambda: _pipeline(jazz), n=n, warmup=1)
    results["genre_overhead_2_merged"] = time_fn(lambda: _pipeline(merged), n=n, warmup=1)
    return results


def bench_test_suite() -> Dict:
    """Time the fast test suite (subprocess)."""
    cmd = [sys.executable, "-m", "pytest", "-m", "not slow", "-q", "--tb=no", "--no-header"]
    t0 = time.perf_counter()
    r = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    elapsed_ms = (time.perf_counter() - t0) * 1_000
    # parse test count from output
    test_count = 0
    for line in r.stdout.splitlines():
        if "passed" in line:
            try:
                test_count = int(line.strip().split()[0])
            except ValueError:
                pass
    return {
        "test_suite_fast": {
            "mean_ms": round(elapsed_ms, 1),
            "n": 1,
            "test_count": test_count,
            "exit_code": r.returncode,
        }
    }


def bench_single_sample_mocked(n: int) -> Dict:
    """Full generate() pipeline with mocked renderer (no FluidSynth needed)."""
    import musicgen.api as _api

    orig_render = _api.renderer.render_stems
    orig_pick = _api.renderer.pick_soundfonts
    orig_score = _api.musicality.get_musicality_score

    _api.renderer.render_stems = _mock_render_stems
    _api.renderer.pick_soundfonts = _mock_pick_soundfonts
    _api.musicality.get_musicality_score = _mock_musicality

    try:
        def _run():
            with tempfile.TemporaryDirectory() as td:
                cfg = Config.load(cli_overrides={
                    "global_seed": 42,
                    "sample_index": 0,
                    "dataset_root": td,
                })
                from musicgen import generate
                generate(cfg)

        result = time_fn(_run, n=n, warmup=1)
    finally:
        _api.renderer.render_stems = orig_render
        _api.renderer.pick_soundfonts = orig_pick
        _api.musicality.get_musicality_score = orig_score

    return {"single_sample_mocked": result}


def bench_single_sample_mocked_genre(n: int) -> Dict:
    """Full generate() mocked — with genre=jazz for comparison."""
    import musicgen.api as _api

    orig_render = _api.renderer.render_stems
    orig_pick = _api.renderer.pick_soundfonts
    orig_score = _api.musicality.get_musicality_score

    _api.renderer.render_stems = _mock_render_stems
    _api.renderer.pick_soundfonts = _mock_pick_soundfonts
    _api.musicality.get_musicality_score = _mock_musicality

    try:
        def _run():
            with tempfile.TemporaryDirectory() as td:
                cfg = Config.load(cli_overrides={
                    "global_seed": 42, "sample_index": 0,
                    "dataset_root": td, "genre": ["jazz"],
                })
                from musicgen import generate
                generate(cfg)

        result = time_fn(_run, n=n, warmup=1)
    finally:
        _api.renderer.render_stems = orig_render
        _api.renderer.pick_soundfonts = orig_pick
        _api.musicality.get_musicality_score = orig_score

    return {"single_sample_mocked_genre": result}


def bench_single_sample_full(n: int) -> Dict:
    """Full generate() — requires FluidSynth + sf2 files."""
    from musicgen import generate

    def _run():
        with tempfile.TemporaryDirectory() as td:
            cfg = Config.load(cli_overrides={
                "global_seed": 42, "sample_index": 0, "dataset_root": td,
            })
            generate(cfg)

    return {"single_sample_full": time_fn(_run, n=n, warmup=0)}


def bench_batch_scaling(worker_counts: List[int], samples_per_run: int) -> Dict:
    """generate_batch timing across worker counts. Requires FluidSynth."""
    from musicgen import generate_batch

    results = {}
    for w in worker_counts:
        n_samples = max(samples_per_run, w * 2)
        with tempfile.TemporaryDirectory() as td:
            cfg = Config.load(cli_overrides={
                "global_seed": 1, "count": n_samples,
                "dataset_root": td, "workers": w,
            })
            t0 = time.perf_counter()
            br = generate_batch(cfg)
            elapsed_s = time.perf_counter() - t0

        key = f"batch_workers_{w}"
        results[key] = {
            "workers": w,
            "samples": n_samples,
            "succeeded": br.succeeded,
            "failed": br.failed,
            "elapsed_s": round(elapsed_s, 3),
            "samples_per_sec": round(br.succeeded / elapsed_s, 3),
            "samples_per_hour": round(br.succeeded / elapsed_s * 3600, 1),
        }
    return results


def bench_memory_single() -> Dict:
    """Peak RSS during one mocked generate()."""
    import musicgen.api as _api

    orig_render = _api.renderer.render_stems
    orig_pick = _api.renderer.pick_soundfonts
    orig_score = _api.musicality.get_musicality_score
    _api.renderer.render_stems = _mock_render_stems
    _api.renderer.pick_soundfonts = _mock_pick_soundfonts
    _api.musicality.get_musicality_score = _mock_musicality

    proc = psutil.Process()
    baseline_mb = proc.memory_info().rss / 1024**2

    try:
        with tempfile.TemporaryDirectory() as td:
            cfg = Config.load(cli_overrides={
                "global_seed": 42, "sample_index": 0, "dataset_root": td,
            })
            from musicgen import generate
            generate(cfg)
        peak_mb = proc.memory_info().rss / 1024**2
    finally:
        _api.renderer.render_stems = orig_render
        _api.renderer.pick_soundfonts = orig_pick
        _api.musicality.get_musicality_score = orig_score

    return {
        "memory_single_mocked": {
            "baseline_mb": round(baseline_mb, 1),
            "after_mb":    round(peak_mb, 1),
            "delta_mb":    round(peak_mb - baseline_mb, 1),
        }
    }


# ──────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────

def run(args: argparse.Namespace) -> Dict:
    n = args.n
    results: Dict[str, Any] = {}

    def section(label: str, fn: Callable) -> None:
        print(f"  {label}...", end=" ", flush=True)
        t0 = time.perf_counter()
        data = fn()
        elapsed = time.perf_counter() - t0
        results.update(data)
        print(f"done ({elapsed:.1f}s)")

    print("\n── Fast benchmarks (no FluidSynth) ──")
    section("sampler",            lambda: bench_sampler(n))
    section("genre system",       lambda: bench_genre(n))
    section("generators",         lambda: bench_generators(n))
    section("full MIDI pipeline", lambda: bench_full_midi_pipeline(max(n // 2, 5)))
    section("genre overhead",     lambda: bench_genre_overhead(max(n // 2, 5)))
    section("single sample (mocked)",       lambda: bench_single_sample_mocked(max(n // 4, 3)))
    section("single sample mocked + genre", lambda: bench_single_sample_mocked_genre(max(n // 4, 3)))
    section("memory (mocked)",    lambda: bench_memory_single())

    if not args.fast_only:
        section("test suite",     lambda: bench_test_suite())

    if HAS_FLUIDSYNTH and not args.fast_only:
        print("\n── FluidSynth benchmarks ──")
        section("single sample (real)",
                lambda: bench_single_sample_full(max(args.n // 5, 2)))
        if args.workers:
            worker_counts = args.workers
        else:
            cpu = psutil.cpu_count(logical=False) or 4
            worker_counts = sorted(set(w for w in [1, 2, 4, cpu] if w <= cpu))
        print(f"  batch scaling (workers={worker_counts}, samples={args.batch_samples})...",
              end=" ", flush=True)
        t0 = time.perf_counter()
        results.update(bench_batch_scaling(worker_counts, args.batch_samples))
        print(f"done ({time.perf_counter()-t0:.1f}s)")
    elif not HAS_FLUIDSYNTH:
        print("\n  [FluidSynth not found — skipping synthesis benchmarks]")

    return results


# ──────────────────────────────────────────────
# Save + plot
# ──────────────────────────────────────────────

def save_results(system_info: Dict, benchmarks: Dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    hostname = system_info["hostname"].replace(" ", "_")
    filename = out_dir / f"{hostname}_{ts}.json"
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "system":     system_info,
        "benchmarks": benchmarks,
    }
    filename.write_text(json.dumps(payload, indent=2))
    return filename


def main():
    parser = argparse.ArgumentParser(description="musicgen benchmark suite")
    parser.add_argument("--fast-only", action="store_true",
                        help="skip FluidSynth and test-suite benchmarks")
    parser.add_argument("--no-plot", action="store_true",
                        help="skip chart generation")
    parser.add_argument("--n", type=int, default=15,
                        help="repetitions per fast benchmark (default: 15)")
    parser.add_argument("--workers", type=int, nargs="+",
                        help="worker counts for batch scaling")
    parser.add_argument("--batch-samples", type=int, default=8,
                        help="samples per batch worker-scaling run (default: 8)")
    parser.add_argument("--compare", nargs=2, metavar="JSON",
                        help="skip benchmarks, just plot two existing result files")
    args = parser.parse_args()

    figures_dir = REPO_ROOT / "benchmarks" / "figures"
    results_dir = REPO_ROOT / "benchmarks" / "results"

    if args.compare:
        if not args.no_plot:
            _plot(args.compare[0], args.compare[1], figures_dir)
        return

    print("musicgen benchmark suite")
    print("=" * 50)

    system_info = get_system_info()
    print(f"Host:      {system_info['hostname']}")
    print(f"CPU:       {system_info['cpu']}")
    print(f"Cores:     {system_info['cpu_count_physical']}P / {system_info['cpu_count_logical']}L")
    print(f"RAM:       {system_info['ram_gb']} GB")
    print(f"FluidSynth: {'yes — ' + system_info['fluidsynth_version'] if system_info['has_fluidsynth'] else 'not found'}")

    benchmarks = run(args)

    result_file = save_results(system_info, benchmarks, results_dir)
    print(f"\nResults saved → {result_file}")

    if not args.no_plot:
        _plot(str(result_file), None, figures_dir)
        print(f"Charts saved → {figures_dir}/")


def _plot(file_a: str, file_b: Optional[str], figures_dir: Path):
    plot_script = REPO_ROOT / "benchmarks" / "plot.py"
    cmd = [sys.executable, str(plot_script), file_a]
    if file_b:
        cmd.extend(["--compare", file_b])
    cmd.extend(["--out", str(figures_dir)])
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
