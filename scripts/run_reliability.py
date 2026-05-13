#!/usr/bin/env python3
"""Run E1 reliability experiment and write results to JSON + human summary.

Usage:
    python3 scripts/run_reliability.py               # all three types, paper defaults
    python3 scripts/run_reliability.py --quick       # small n for smoke test
    python3 scripts/run_reliability.py --type det    # determinism only
    python3 scripts/run_reliability.py --type rinv   # render invariance only
    python3 scripts/run_reliability.py --type sdist  # seed distribution only
    python3 scripts/run_reliability.py --n-samples 10 --n-calls 5 --n-seeds 20
"""
import os
import pathlib
import sys

_VENV_PYTHON = pathlib.Path(__file__).resolve().parents[1] / ".venv" / "bin" / "python3"
_IN_VENV = str(_VENV_PYTHON) == sys.executable or \
    os.environ.get("VIRTUAL_ENV") == str(_VENV_PYTHON.parent.parent)
if not _IN_VENV:
    if _VENV_PYTHON.is_file():
        print(f"[auto] re-executing with venv Python: {_VENV_PYTHON}", flush=True)
        os.execv(str(_VENV_PYTHON), [str(_VENV_PYTHON)] + sys.argv)
    else:
        print("ERROR: .venv not found."); sys.exit(1)

import argparse
import dataclasses
import json
import time
import warnings

warnings.filterwarnings("ignore")

from musicgen.eval.reliability import (
    run_determinism_test,
    run_render_invariance_test,
    run_seed_distribution_test,
    _ALL_GENRES,
    _DETERMINISM_SIGMA_MAX,
    _RENDER_SIGMA_MAX,
)

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_OUT_DIR = _REPO_ROOT / "eval_results"


def _asdict(obj):
    if dataclasses.is_dataclass(obj):
        return {k: _asdict(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, list):
        return [_asdict(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _asdict(v) for k, v in obj.items()}
    return obj


def print_det(r):
    status = "PASS" if r.passed else "FAIL"
    print(f"\n[Type 1 — Determinism]  {status}")
    print(f"  n_samples={r.n_samples}  n_calls={r.n_calls}  criterion: σ < {_DETERMINISM_SIGMA_MAX:.0e}")
    print(f"  max_σ = {r.max_sigma:.2e}  ({'OK' if r.passed else 'EXCEEDS CRITERION'})")
    if not r.passed:
        bad = [(i, s) for i, s in enumerate(r.all_sigma) if s >= _DETERMINISM_SIGMA_MAX]
        print(f"  failing samples: {bad[:5]}")


def print_rinv(r):
    status = "PASS" if r.passed else "FAIL"
    print(f"\n[Type 2 — Render Invariance]  {status}")
    print(f"  n_samples={r.n_samples}  n_renders={r.n_renders}  criterion: genre mean σ < {_RENDER_SIGMA_MAX}")
    print(f"  max_σ = {r.max_sigma:.4f}")
    for genre, sig in sorted(r.sigma_by_genre.items()):
        flag = "" if sig < _RENDER_SIGMA_MAX else "  ← EXCEEDS"
        print(f"    {genre:<14}  mean σ = {sig:.4f}{flag}")


def print_sdist(r):
    print(f"\n[Type 3 — Seed Distribution]  (characterisation, no pass/fail)")
    print(f"  n_seeds={r.n_seeds} per genre")
    print(f"  {'genre':<14}  {'mean':>6}  {'median':>7}  {'σ':>6}  {'IQR':>6}  {'p10':>6}  {'p90':>6}")
    for e in r.entries:
        print(f"  {e.genre:<14}  {e.mean:6.3f}  {e.median:7.3f}  {e.sigma:6.3f}  "
              f"{e.iqr:6.3f}  {e.p10:6.3f}  {e.p90:6.3f}")


def main():
    parser = argparse.ArgumentParser(description="E1 reliability experiment")
    parser.add_argument("--type", choices=["det", "rinv", "sdist", "all"], default="all")
    parser.add_argument("--quick", action="store_true",
                        help="Small n for smoke test (5 samples, 5 calls, 20 seeds)")
    parser.add_argument("--n-samples", type=int, help="Samples for det/rinv")
    parser.add_argument("--n-calls",   type=int, help="Calls per sample for det")
    parser.add_argument("--n-renders", type=int, help="Renders per sample for rinv")
    parser.add_argument("--n-seeds",   type=int, help="Seeds per genre for sdist")
    parser.add_argument("--genres", nargs="+", default=_ALL_GENRES)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", default=str(_OUT_DIR), help="Output directory")
    args = parser.parse_args()

    # defaults: quick vs paper-scale
    if args.quick:
        n_samples = args.n_samples or 5
        n_calls   = args.n_calls   or 5
        n_renders = args.n_renders or 5
        n_seeds   = args.n_seeds   or 20
    else:
        n_samples = args.n_samples or 50
        n_calls   = args.n_calls   or 20
        n_renders = args.n_renders or 20
        n_seeds   = args.n_seeds   or 400

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_det   = args.type in ("det",   "all")
    run_rinv  = args.type in ("rinv",  "all")
    run_sdist = args.type in ("sdist", "all")

    report = {"genres": args.genres, "seed": args.seed}
    t_total = time.monotonic()

    if run_det:
        print(f"Running Type 1 — Determinism  (n_samples={n_samples}, n_calls={n_calls}) …", flush=True)
        t0 = time.monotonic()
        r = run_determinism_test(n_samples=n_samples, n_calls=n_calls,
                                  seed=args.seed, genres=args.genres)
        print(f"  done in {time.monotonic()-t0:.1f}s", flush=True)
        print_det(r)
        report["determinism"] = _asdict(r)

    if run_rinv:
        print(f"\nRunning Type 2 — Render Invariance  (n_samples={n_samples}, n_renders={n_renders}) …", flush=True)
        t0 = time.monotonic()
        r = run_render_invariance_test(n_samples=n_samples, n_renders=n_renders,
                                        seed=args.seed, genres=args.genres)
        print(f"  done in {time.monotonic()-t0:.1f}s", flush=True)
        print_rinv(r)
        report["render_invariance"] = _asdict(r)

    if run_sdist:
        print(f"\nRunning Type 3 — Seed Distribution  (n_seeds={n_seeds} per genre) …", flush=True)
        t0 = time.monotonic()
        r = run_seed_distribution_test(n_seeds=n_seeds, genres=args.genres,
                                        base_seed=args.seed)
        print(f"  done in {time.monotonic()-t0:.1f}s", flush=True)
        print_sdist(r)
        report["seed_distribution"] = _asdict(r)

    report["total_elapsed_s"] = round(time.monotonic() - t_total, 1)

    out_json = out_dir / "e1_reliability.json"
    out_json.write_text(json.dumps(report, indent=2))
    print(f"\nReport → {out_json}")


if __name__ == "__main__":
    main()
