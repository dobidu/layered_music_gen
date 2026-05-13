#!/usr/bin/env python3
"""CLI wrapper for musicgen calibration: FluidSynth pre-roll + musicality threshold.

Usage:
    python3 scripts/calibrate.py               # both pre-roll + musicality
    python3 scripts/calibrate.py --preroll     # pre-roll only
    python3 scripts/calibrate.py --musicality  # musicality threshold only
    python3 scripts/calibrate.py --n 50        # larger sample (default: 20)
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

from musicgen.calibrate import (
    measure_and_save_preroll,
    load_preroll,
    run_midi_calibration,
    save_calibration,
    load_calibration,
    PREROLL_CACHE_DIR,
    PREROLL_CACHE_FILE,
)

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_CACHE_DIR = _REPO_ROOT / PREROLL_CACHE_DIR
_PREROLL_PATH = _CACHE_DIR / PREROLL_CACHE_FILE
_MUSICALITY_PATH = _CACHE_DIR / "musicality_calibration.json"


def run_preroll():
    print("--- FluidSynth pre-roll calibration ---")
    offset = measure_and_save_preroll(str(_REPO_ROOT))
    print(f"  pre_roll_offset = {offset:.6f} s  →  {_PREROLL_PATH}")


def run_musicality(n: int, seed: int):
    print(f"--- Musicality threshold calibration (n_good={n}, n_bad={n}, seed={seed}) ---")
    result = run_midi_calibration(n_good=n, n_bad=n, seed=seed)
    print(f"  good_mean       = {result.good_mean:.4f}")
    print(f"  bad_mean        = {result.bad_mean:.4f}")
    print(f"  threshold       = {result.suggested_threshold:.4f}")
    print(f"  separation_ok   = {result.separation_ok}")
    if not result.separation_ok:
        print("  WARN: good/bad distributions overlap — threshold unreliable.")
    _CACHE_DIR.mkdir(exist_ok=True)
    save_calibration(result, str(_MUSICALITY_PATH))
    print(f"  Saved →  {_MUSICALITY_PATH}")


def show_current():
    print("--- Current calibration values ---")
    if _PREROLL_PATH.exists():
        offset = load_preroll(str(_REPO_ROOT))
        print(f"  pre_roll_offset = {offset:.6f} s  ({_PREROLL_PATH})")
    else:
        print(f"  pre_roll: not calibrated ({_PREROLL_PATH} missing)")

    if _MUSICALITY_PATH.exists():
        r = load_calibration(str(_MUSICALITY_PATH))
        print(f"  musicality threshold = {r.suggested_threshold:.4f}  "
              f"(good_mean={r.good_mean:.4f}, bad_mean={r.bad_mean:.4f}, "
              f"separation_ok={r.separation_ok})")
    else:
        print(f"  musicality: not calibrated ({_MUSICALITY_PATH} missing)")


def main():
    parser = argparse.ArgumentParser(description="musicgen calibration")
    parser.add_argument("--preroll", action="store_true", help="Run pre-roll calibration only")
    parser.add_argument("--musicality", action="store_true", help="Run musicality calibration only")
    parser.add_argument("--show", action="store_true", help="Show current calibration values")
    parser.add_argument("--n", type=int, default=20, help="Samples per class for musicality (default: 20)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.show:
        show_current()
        return

    do_preroll = args.preroll or (not args.preroll and not args.musicality)
    do_musicality = args.musicality or (not args.preroll and not args.musicality)

    if do_preroll:
        run_preroll()
    if do_musicality:
        run_musicality(args.n, args.seed)

    print()
    show_current()


if __name__ == "__main__":
    main()
