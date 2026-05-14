#!/usr/bin/env python3
"""E2 construct validity experiment — gate discrimination between good and bad audio.

Usage:
    python3 scripts/run_construct_validity.py --bad-type both --n-good 50 --n-bad 20
    python3 scripts/run_construct_validity.py --bad-type synthetic --quick
    python3 scripts/run_construct_validity.py --bad-type real --n-good 30 --n-bad 15

Writes:
    <out>.json   — full results
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
import json
import time
import warnings

warnings.filterwarnings("ignore")

from musicgen.eval.construct_validity import (
    SYNTHETIC_TYPES,
    REAL_TYPES,
    ConstructValidityResult,
    run_construct_validity_test,
)


def _print_results(r: ConstructValidityResult) -> None:
    print()
    print("=" * 72)
    print(f"E2 Construct Validity  —  good n={r.n_good}  mean={r.good_mean:.3f}"
          f"  σ={r.good_std:.3f}")
    print("=" * 72)
    print(f"\n{'pathology':<26}  {'n':>4}  {'mean':>6}  {'σ':>5}  {'AUROC':>6}  {'d':>5}")
    for p in r.pathologies:
        flag = "  ✓" if p.auroc >= 0.85 else ("  ~" if p.auroc >= 0.70 else "  ✗")
        print(f"  {p.name:<24}  {len(p.scores):4d}  {p.mean:6.3f}  {p.std:5.3f}"
              f"  {p.auroc:6.3f}  {p.separation:5.2f}{flag}")
    print()
    gate = "PASS" if r.overall_auroc >= 0.85 else ("MARGINAL" if r.overall_auroc >= 0.70 else "FAIL")
    print(f"Overall AUROC = {r.overall_auroc:.3f}  d = {r.overall_separation:.2f}  [{gate}]")
    print(f"Criterion: AUROC ≥ 0.85 for gate to have discriminative validity")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--bad-type", choices=["synthetic", "real", "both"], default="both")
    parser.add_argument("--n-good", type=int, default=50)
    parser.add_argument("--n-bad", type=int, default=20,
                        help="Bad samples per pathology type")
    parser.add_argument("--synthetic-signals", nargs="+", default=None,
                        choices=SYNTHETIC_TYPES,
                        help=f"Synthetic types (default: all {SYNTHETIC_TYPES})")
    parser.add_argument("--real-signals", nargs="+", default=None,
                        choices=REAL_TYPES,
                        help=f"Real degradation types (default: all {REAL_TYPES})")
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--out", default="eval_results/e2_construct_validity",
                        help="Output stem (writes <stem>.json)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick run: n-good=5, n-bad=5")
    args = parser.parse_args()

    if args.quick:
        args.n_good = 5
        args.n_bad = 5

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    print(f"E2 Construct Validity")
    print(f"  bad-type={args.bad_type}  n-good={args.n_good}  n-bad={args.n_bad}")
    print(f"  synthetic={args.synthetic_signals or 'all'}  real={args.real_signals or 'all'}")
    print()

    t0 = time.monotonic()
    r = run_construct_validity_test(
        n_good=args.n_good,
        n_bad=args.n_bad,
        bad_types=args.bad_type,
        synthetic_signals=args.synthetic_signals,
        real_signals=args.real_signals,
        base_seed=args.base_seed,
    )
    elapsed = time.monotonic() - t0

    _print_results(r)

    report = {
        "elapsed_seconds": round(elapsed, 2),
        "n_good": r.n_good,
        "good_mean": r.good_mean,
        "good_std": r.good_std,
        "good_scores": r.good_scores,
        "overall_auroc": r.overall_auroc,
        "overall_separation": r.overall_separation,
        "pathologies": [
            {
                "name": p.name,
                "n": len(p.scores),
                "mean": p.mean,
                "std": p.std,
                "auroc": p.auroc,
                "separation": p.separation,
                "scores": p.scores,
            }
            for p in r.pathologies
        ],
    }
    out_path = args.out + ".json"
    pathlib.Path(out_path).write_text(json.dumps(report, indent=2))
    print(f"\nWritten: {out_path}  ({elapsed:.0f}s)")


if __name__ == "__main__":
    main()
