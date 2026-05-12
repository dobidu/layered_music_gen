#!/usr/bin/env python3
"""Generate one music sample per registered genre and report results."""
import json
import os
import pathlib
import sys
import time

# Re-execute under venv if needed. Compare RAW paths (not resolved symlinks):
# venv python may symlink to the same underlying interpreter as /usr/bin/python3,
# but running .venv/bin/python3 is what activates site-packages.
_VENV_PYTHON = pathlib.Path(__file__).resolve().parents[1] / ".venv" / "bin" / "python3"
_IN_VENV = str(_VENV_PYTHON) == sys.executable or \
    os.environ.get("VIRTUAL_ENV") == str(_VENV_PYTHON.parent.parent)
if not _IN_VENV:
    if _VENV_PYTHON.is_file():
        print(f"[auto] re-executing with venv Python: {_VENV_PYTHON}", flush=True)
        os.execv(str(_VENV_PYTHON), [str(_VENV_PYTHON)] + sys.argv)
    else:
        print("ERROR: .venv not found. Run from repo root:")
        print("  python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'")
        sys.exit(1)

import argparse

from musicgen import Config, generate

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_GENRES_DIR = _REPO_ROOT / "genres"


def _discover_genres(genres_dir: pathlib.Path) -> list[str]:
    return sorted(
        d.name
        for d in genres_dir.iterdir()
        if d.is_dir() and (d / "spec.json").exists()
    )


def _play(path: str) -> None:
    import shutil
    import subprocess
    for player in ("ffplay", "aplay", "mpv", "vlc"):
        if shutil.which(player):
            flags = ["-nodisp", "-autoexit"] if player == "ffplay" else []
            subprocess.run([player] + flags + [path], check=False)
            return
    print(f"  [no audio player found — file at {path}]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate one sample per genre")
    parser.add_argument("--out", default=str(pathlib.Path.home() / "musicgen-genres"),
                        help="Output root directory (default: ~/musicgen-genres)")
    parser.add_argument("--seed", type=int, default=42, help="Global seed (default: 42)")
    parser.add_argument("--play", action="store_true", help="Play each mix after generation")
    parser.add_argument("--genres", nargs="+", metavar="GENRE",
                        help="Only generate these genres (default: all)")
    args = parser.parse_args()

    out_root = pathlib.Path(args.out)
    genres = args.genres or _discover_genres(_GENRES_DIR)

    print(f"Output root : {out_root}")
    print(f"Seed        : {args.seed}")
    print(f"Genres ({len(genres)}): {', '.join(genres)}")
    print("=" * 60)

    results = []
    for idx, genre in enumerate(genres):
        genre_out = out_root / genre
        genre_out.mkdir(parents=True, exist_ok=True)

        print(f"\n[{idx + 1}/{len(genres)}] {genre} ...", flush=True)
        t0 = time.monotonic()
        try:
            cfg = Config(
                global_seed=args.seed,
                sample_index=0,
                dataset_root=str(genre_out),
                genre=[genre],
            )
            r = generate(cfg)
            elapsed = time.monotonic() - t0

            ann = json.loads((pathlib.Path(r.sample_dir) / "sample.json").read_text())
            ms = ann.get("musicality_score")
            score = ms["score"] if isinstance(ms, dict) else (ms or float("nan"))
            print(f"  status : {r.status}  attempt={r.attempt}  score={score:.4f}  ({elapsed:.1f}s)")
            print(f"  key={ann['key']}  tempo={ann['tempo_bpm']}  ts={ann['time_signature']}")
            print(f"  mix    : {r.sample_dir}/mix.wav")

            results.append({
                "genre": genre,
                "status": r.status,
                "score": score,
                "mix": str(pathlib.Path(r.sample_dir) / "mix.wav"),
                "key": ann["key"],
                "tempo_bpm": ann["tempo_bpm"],
                "time_signature": ann["time_signature"],
                "elapsed_s": round(elapsed, 2),
            })

            if args.play and r.status == "ok":
                print(f"  playing {genre} ...", flush=True)
                _play(str(pathlib.Path(r.sample_dir) / "mix.wav"))

        except Exception as exc:
            elapsed = time.monotonic() - t0
            print(f"  ERROR: {exc}  ({elapsed:.1f}s)")
            results.append({"genre": genre, "status": "error", "error": str(exc), "elapsed_s": round(elapsed, 2)})

    # Summary
    print("\n" + "=" * 60)
    ok = [r for r in results if r.get("status") == "ok"]
    print(f"Done: {len(ok)}/{len(results)} ok\n")
    for r in results:
        status_str = r["status"].upper()
        if r["status"] == "ok":
            print(f"  {r['genre']:<14} {status_str}  score={r['score']:.4f}  {r['key']} {r['tempo_bpm']}bpm {r['time_signature']}")
        else:
            print(f"  {r['genre']:<14} {status_str}  {r.get('error', '')}")

    # Save report
    report_path = out_root / "genre_report.json"
    report_path.write_text(json.dumps(results, indent=2))
    print(f"\nReport: {report_path}")

    if args.play and ok:
        print("\nPlay all mixes? [y/N] ", end="", flush=True)
        if input().strip().lower() == "y":
            for r in ok:
                print(f"  playing {r['genre']} ...", flush=True)
                _play(r["mix"])


if __name__ == "__main__":
    main()
