#!/usr/bin/env python3
"""Probe dc_offset and crest_db distribution over N normal musicgen samples.

Usage:
    python3 scripts/probe_integrity.py [--n 100] [--seed 1] [--genres jazz pop blues]

Prints quartiles, 95th percentile, and fraction of samples that would fire
each hard penalty at the current thresholds.
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
import tempfile
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from musicgen import Config, generate
from musicgen.musicality import _render_integrity

import librosa


def main():
    parser = argparse.ArgumentParser(description="Probe dc_offset/crest_db distribution")
    parser.add_argument("--n", type=int, default=100, help="Number of samples (default: 100)")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--genres", nargs="+", default=["blues", "classical", "electronic",
                                                         "hip-hop", "jazz", "latin", "pop", "reggae"])
    parser.add_argument("--dc-threshold", type=float, default=0.001)
    parser.add_argument("--crest-lo", type=float, default=3.0)
    parser.add_argument("--crest-hi", type=float, default=30.0)
    args = parser.parse_args()

    dc_vals, crest_vals = [], []
    errors = 0

    with tempfile.TemporaryDirectory(prefix="musicgen-probe-") as tmp:
        for i in range(args.n):
            genre = args.genres[i % len(args.genres)]
            try:
                r = generate(Config(
                    global_seed=args.seed,
                    sample_index=i,
                    dataset_root=tmp,
                    genre=[genre],
                    output_mode="mix-only",
                ))
                if r.status != "ok":
                    errors += 1
                    continue
                mix_path = str(pathlib.Path(r.sample_dir) / "mix.wav")
                y, sr = librosa.load(mix_path)
                ig = _render_integrity(y, sr)
                dc_vals.append(ig["dc_offset"])
                crest_vals.append(ig["crest_db"])
                print(f"  [{i+1:3d}/{args.n}] genre={genre:<12} dc={ig['dc_offset']:.5f}  crest={ig['crest_db']:.1f} dB",
                      flush=True)
            except Exception as exc:
                errors += 1
                print(f"  [{i+1:3d}/{args.n}] ERROR: {exc}", flush=True)

    if not dc_vals:
        print("No successful samples — cannot report."); return

    dc = np.array(dc_vals)
    cr = np.array(crest_vals)

    def _report(name, vals, threshold_lo=None, threshold_hi=None, threshold_exact=None):
        pcts = np.percentile(vals, [25, 50, 75, 90, 95, 99])
        print(f"\n{name}  (n={len(vals)})")
        print(f"  min={vals.min():.5f}  p25={pcts[0]:.5f}  median={pcts[1]:.5f}  "
              f"p75={pcts[2]:.5f}  p90={pcts[3]:.5f}  p95={pcts[4]:.5f}  p99={pcts[5]:.5f}  max={vals.max():.5f}")
        if threshold_exact is not None:
            fired = np.mean(vals > threshold_exact)
            print(f"  fires (>{threshold_exact}): {fired*100:.1f}%  "
                  f"({'OK' if fired < 0.05 else 'WARN: >5% false-reject risk'})")
        if threshold_lo is not None and threshold_hi is not None:
            fired = np.mean((vals < threshold_lo) | (vals > threshold_hi))
            print(f"  fires (outside [{threshold_lo}, {threshold_hi}]): {fired*100:.1f}%  "
                  f"({'OK' if fired < 0.05 else 'WARN: >5% false-reject risk'})")

    print("\n" + "=" * 60)
    print(f"Probe results — {len(dc_vals)} ok samples, {errors} errors")
    _report("dc_offset", dc, threshold_exact=args.dc_threshold)
    _report("crest_db", cr, threshold_lo=args.crest_lo, threshold_hi=args.crest_hi)

    print("\nRecommendation:")
    dc_p95 = float(np.percentile(dc, 95))
    cr_p05 = float(np.percentile(cr, 5))
    cr_p95 = float(np.percentile(cr, 95))
    if dc_p95 > args.dc_threshold * 0.8:
        print(f"  dc_threshold: current={args.dc_threshold} may be too tight "
              f"(p95={dc_p95:.5f}). Consider raising to {dc_p95 * 2:.4f} or making soft.")
    else:
        print(f"  dc_threshold={args.dc_threshold}: OK (p95={dc_p95:.5f} is well below threshold).")
    if cr_p05 < args.crest_lo + 1.0:
        print(f"  crest_lo={args.crest_lo}: borderline (p5={cr_p05:.1f} dB). "
              f"Consider lowering to {max(0.0, cr_p05 - 0.5):.1f}.")
    else:
        print(f"  crest_lo={args.crest_lo}: OK (p5={cr_p05:.1f} dB).")
    if cr_p95 > args.crest_hi - 3.0:
        print(f"  crest_hi={args.crest_hi}: borderline (p95={cr_p95:.1f} dB). "
              f"Consider raising to {cr_p95 + 2:.0f}.")
    else:
        print(f"  crest_hi={args.crest_hi}: OK (p95={cr_p95:.1f} dB).")

    print(f"\n  Errors/skipped: {errors}/{args.n}")


if __name__ == "__main__":
    main()
