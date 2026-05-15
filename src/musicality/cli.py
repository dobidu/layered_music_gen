"""CLI for musicality scoring.

Commands:
    musicality score  <file>              — score one file
    musicality explain <file>             — score + sub-scores + labels
    musicality batch  <file> [<file>...]  — score multiple files
                      --json              — output as JSON
                      --out <csv>         — write CSV
                      --genre <genre>     — tempo-range hint
                      --threshold <float> — mark pass/fail (default 0.5)

Examples::

    musicality score track.wav
    musicality explain mix.flac
    musicality batch samples/*.wav --json
    musicality batch dir/ --out scores.csv --threshold 0.6
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from pathlib import Path
from typing import List, Optional


_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".aiff", ".aif", ".m4a"}


def _collect_paths(inputs: List[str]) -> List[str]:
    paths = []
    for inp in inputs:
        p = Path(inp)
        if p.is_dir():
            for ext in _AUDIO_EXTS:
                paths.extend(str(f) for f in sorted(p.rglob(f"*{ext}")))
        elif "*" in inp or "?" in inp:
            paths.extend(sorted(glob.glob(inp)))
        else:
            paths.append(inp)
    return [p for p in paths if Path(p).suffix.lower() in _AUDIO_EXTS]


def _cmd_score(args: argparse.Namespace) -> None:
    from musicality import score as _score
    s = _score(args.file, genre=args.genre)
    if args.json:
        print(json.dumps({"path": args.file, "score": s}))
    else:
        flag = "PASS" if s >= args.threshold else "FAIL"
        print(f"{s:.4f}  [{flag}]  {args.file}")


def _cmd_explain(args: argparse.Namespace) -> None:
    from musicality import explain as _explain
    report = _explain(args.file, genre=args.genre)
    if args.json:
        print(json.dumps(report, indent=2))
        return

    s = report["score"]
    flag = "PASS" if s >= args.threshold else "FAIL"
    print(f"\nScore: {s:.4f}  [{flag}]  ({report['label']})")
    print(f"File:  {args.file}")
    print()
    print(f"  {'component':<10}  {'score':>6}  {'label':<10}  {'weight':>6}  sub-scores")
    print("  " + "-" * 70)
    for name, comp in report["components"].items():
        sub_str = "  ".join(f"{k}={v:.3f}" for k, v in comp["sub_scores"].items())
        print(f"  {name:<10}  {comp['score']:6.3f}  {comp['label']:<10}  "
              f"{comp['weight']:6.2f}  {sub_str}")
    integ = report["integrity"]
    print()
    print(f"  Integrity: clipping={integ['clipping_ratio']:.4f}  "
          f"dc={integ['dc_offset']:.4f}  "
          f"silence={integ['silence_ratio']:.3f}  "
          f"crest={integ['crest_db']:.1f}dB  "
          f"penalty={integ['penalty']:.3f}")


def _cmd_batch(args: argparse.Namespace) -> None:
    from musicality import score as _score

    paths = _collect_paths(args.files)
    if not paths:
        print("No audio files found.", file=sys.stderr)
        sys.exit(1)

    results = []
    for i, p in enumerate(paths):
        try:
            s = _score(p, genre=args.genre)
        except Exception as exc:
            s = 0.0
            print(f"  WARN: {p}: {exc}", file=sys.stderr)
        flag = "PASS" if s >= args.threshold else "FAIL"
        results.append({"path": p, "score": s, "pass": flag})
        if not args.json and (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(paths)}]", flush=True)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(f"\n{'path':<50}  {'score':>6}  flag")
        print("-" * 62)
        for r in results:
            short = r["path"][-49:] if len(r["path"]) > 49 else r["path"]
            print(f"{short:<50}  {r['score']:6.4f}  {r['pass']}")
        passed = sum(1 for r in results if r["pass"] == "PASS")
        print(f"\n{passed}/{len(results)} passed (threshold={args.threshold})")

    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["path", "score", "pass"])
            w.writeheader()
            w.writerows(results)
        print(f"Written: {args.out}")


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="musicality",
        description="Audio musicality scoring",
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    sub = parser.add_subparsers(dest="command", required=True)

    # --- score ---
    p_score = sub.add_parser("score", help="Score a single audio file")
    p_score.add_argument("file")
    p_score.add_argument("--genre", default=None)
    p_score.add_argument("--threshold", type=float, default=0.5)
    p_score.add_argument("--json", action="store_true")
    p_score.set_defaults(func=_cmd_score)

    # --- explain ---
    p_exp = sub.add_parser("explain", help="Score with sub-scores and labels")
    p_exp.add_argument("file")
    p_exp.add_argument("--genre", default=None)
    p_exp.add_argument("--threshold", type=float, default=0.5)
    p_exp.add_argument("--json", action="store_true")
    p_exp.set_defaults(func=_cmd_explain)

    # --- batch ---
    p_batch = sub.add_parser("batch", help="Score multiple files or a directory")
    p_batch.add_argument("files", nargs="+", metavar="FILE_OR_DIR")
    p_batch.add_argument("--genre", default=None)
    p_batch.add_argument("--threshold", type=float, default=0.5)
    p_batch.add_argument("--json", action="store_true")
    p_batch.add_argument("--out", default=None, metavar="CSV_PATH")
    p_batch.set_defaults(func=_cmd_batch)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
