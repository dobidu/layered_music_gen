#!/usr/bin/env python3
"""Component-level diagnostic for the musicality score.

Goal: explain why the global E1 score clusters around 0.39 across all
genres. Decomposes each sample's score into all sub-metrics from Layer 1
and Layer 2 separately, plus their internal components, and reports
per-metric distributions so we can see *which* sub-score is pulling
the global average down.

Three secondary questions:
  1. Are dropped samples actual pipeline exceptions or quality-gate
     rejections? (runs with gate disabled)
  2. Is scale_adherence biased low because check_midi_quality defaults
     to key="C" but the sample is in another key? (runs Layer 1 twice)
  3. Which pairs of soft metrics are highly correlated (|rho| > 0.6)?

Usage:
    python3 scripts/component_diagnostic.py --n 100 --out eval_results/diagnostic

Writes:
    <out>.json   — aggregated statistics
    <out>.csv    — one row per sample, all metrics
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
import csv
import json
import tempfile
import time
import traceback
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

warnings.filterwarnings("ignore")

# Serial-only — FluidSynth has thread/process-unsafe global state.
# ProcessPoolExecutor with FluidSynth spawned inside workers causes
# deadlocks on some Linux versions. Keep simple serial loop.
from musicgen import Config, generate
from musicgen.musicality import (
    MusicalityAnalyzer,
    _render_integrity,
    check_midi_quality,
)
import librosa


# ---------------------------------------------------------------------------
# Per-sample worker
# ---------------------------------------------------------------------------

def _read_key(sample_json_path: str) -> Optional[str]:
    try:
        data = json.loads(pathlib.Path(sample_json_path).read_text())
        for probe in (("key",), ("song_params", "key"), ("params", "key")):
            cur = data
            try:
                for k in probe:
                    cur = cur[k]
                if isinstance(cur, str) and cur:
                    return cur
            except (KeyError, TypeError):
                continue
    except Exception:
        pass
    return None


def _process_sample(r: Any, genre: Optional[str]) -> Dict[str, Any]:
    """Decompose a successfully generated sample into all sub-metrics."""
    rec: Dict[str, Any] = {
        "seed": -1,
        "status": "ok",
        "genre": genre or "UNKNOWN",
    }
    real_key = _read_key(r.sample_json_path) or "C"
    rec["real_key"] = real_key

    # Layer 1 — twice: real key and default "C"
    l1_real = check_midi_quality(r.midi_paths, key=real_key)
    l1_c = check_midi_quality(r.midi_paths, key="C")
    rec["l1_passed"] = bool(l1_real.passed)
    rec["l1_score_realkey"] = float(l1_real.score)
    rec["l1_score_keyC"] = float(l1_c.score)
    rec["l1_hard_failures"] = list(l1_real.hard_failures)
    for k, v in l1_real.soft_scores.items():
        rec[f"l1_{k}_realkey"] = float(v)
    for k, v in l1_c.soft_scores.items():
        rec[f"l1_{k}_keyC"] = float(v)

    # Layer 2 — decomposed
    y, sr = librosa.load(r.mix_path, sr=None, mono=True)
    rec["audio_length_s"] = float(len(y) / sr)

    analyzer = MusicalityAnalyzer()
    tempo_d = analyzer.analyze_tempo(y, sr, genre_spec=None)
    harm_d = analyzer.analyze_harmony(y, sr)
    rhy_d = analyzer.analyze_rhythm(y, sr)
    noi_d = analyzer.analyze_noise(y, sr)
    integ = _render_integrity(y, sr)

    for k, v in tempo_d.items():
        rec[f"l2_tempo_{k}"] = float(v)
    for k, v in harm_d.items():
        rec[f"l2_harmony_{k}"] = float(v)
    for k, v in rhy_d.items():
        rec[f"l2_rhythm_{k}"] = float(v)
    for k, v in noi_d.items():
        rec[f"l2_noise_{k}"] = float(v)
    for k, v in integ.items():
        rec[f"l2_integ_{k}"] = float(v)

    rec["l2_tempo_mean"] = float(np.mean(list(tempo_d.values())))
    rec["l2_harmony_mean"] = float(np.mean(list(harm_d.values())))
    rec["l2_rhythm_mean"] = float(np.mean(list(rhy_d.values())))
    rec["l2_noise_value"] = float(noi_d.get("music_signal_ratio", 0.0))
    rec["final_musicality_score"] = float(r.musicality_score)
    return rec


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def _summarize(values: List[float]) -> Dict[str, Any]:
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p10": float(np.percentile(arr, 10)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _per_metric_distributions(records: List[Dict]) -> Dict[str, Dict]:
    keys = set()
    for r in records:
        for k, v in r.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                keys.add(k)
    keys -= {"seed", "sample_index", "audio_length_s"}
    return {k: _summarize([r.get(k) for r in records if r.get(k) is not None])
            for k in sorted(keys)}


def _correlation_matrix(records: List[Dict], keys: List[str]) -> Dict[str, Dict[str, float]]:
    if not records or not keys:
        return {}
    cols = {k: [float(r[k]) for r in records if r.get(k) is not None] for k in keys}
    n = min(len(v) for v in cols.values())
    if n < 3:
        return {}
    M = np.array([cols[k][:n] for k in keys])
    try:
        C = np.corrcoef(M)
    except Exception:
        return {}
    return {
        ki: {kj: float(C[i, j]) if np.isfinite(C[i, j]) else 0.0
             for j, kj in enumerate(keys)}
        for i, ki in enumerate(keys)
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--genres", nargs="+",
                        default=["blues", "classical", "electronic",
                                 "hip-hop", "jazz", "latin", "pop", "reggae"])
    parser.add_argument("--out", default="eval_results/diagnostic",
                        help="Output stem (writes <stem>.json and <stem>.csv)")
    args = parser.parse_args()

    out_stem = args.out
    pathlib.Path(out_stem).parent.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    t0 = time.monotonic()

    with tempfile.TemporaryDirectory(prefix="musicgen-diag-") as tmp:
        for i in range(args.n):
            seed = args.base_seed + i
            genre = args.genres[i % len(args.genres)]
            try:
                r = generate(Config(
                    global_seed=seed,
                    sample_index=i,
                    dataset_root=str(pathlib.Path(tmp) / genre),
                    genre=[genre],
                    min_musicality_score=0.0,
                    max_attempts=1,
                ))
                if r.status != "ok":
                    failures.append({"seed": seed, "genre": genre,
                                     "status": r.status, "reason": f"status={r.status}"})
                    continue
                rec = _process_sample(r, genre)
                rec["seed"] = seed
                rec["sample_index"] = i
                records.append(rec)
            except Exception as exc:
                failures.append({"seed": seed, "genre": genre,
                                  "status": "exception",
                                  "reason": f"{type(exc).__name__}: {exc}",
                                  "traceback": traceback.format_exc()})
            elapsed = time.monotonic() - t0
            if (i + 1) % 10 == 0 or (i + 1) == args.n:
                print(f"  [{i+1}/{args.n}] {elapsed:.0f}s  ok={len(records)}  fail={len(failures)}",
                      flush=True)

    if not records:
        print("FATAL: no successful samples.", file=sys.stderr)
        for f in failures[:5]:
            print(f"  - seed={f.get('seed')}: {f.get('reason')}", file=sys.stderr)
        sys.exit(3)

    # --- CSV --------------------------------------------------------------
    all_keys = sorted(k for r in records for k, v in r.items()
                      if not isinstance(v, (list, dict, bool)))
    all_keys = sorted(set(all_keys))
    csv_path = out_stem + ".csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        w.writeheader()
        for r in records:
            w.writerow(r)

    # --- Aggregation ------------------------------------------------------
    by_metric = _per_metric_distributions(records)

    l1_soft_keys = [k for k in by_metric
                    if k.startswith("l1_") and k.endswith("_realkey")
                    and k not in ("l1_score_realkey",)]
    l2_top_keys = ["l2_tempo_mean", "l2_harmony_mean",
                   "l2_rhythm_mean", "l2_noise_value"]
    corr_l1 = _correlation_matrix(records, l1_soft_keys)
    corr_l2 = _correlation_matrix(records, l2_top_keys)

    by_genre: Dict[str, List[float]] = {}
    for r in records:
        by_genre.setdefault(r.get("genre", "UNKNOWN"), []).append(
            r["final_musicality_score"])
    genre_stats = {g: _summarize(v) for g, v in by_genre.items()}

    fail_by_reason: Dict[str, int] = {}
    for f in failures:
        key = (f.get("reason") or "unknown")[:120]
        fail_by_reason[key] = fail_by_reason.get(key, 0) + 1

    # Scale-adherence drift
    deltas = [r["l1_scale_adherence_realkey"] - r["l1_scale_adherence_keyC"]
              for r in records
              if "l1_scale_adherence_realkey" in r and "l1_scale_adherence_keyC" in r]
    drift = None
    if deltas:
        drift = {
            "mean_delta_realkey_minus_keyC": float(np.mean(deltas)),
            "fraction_realkey_higher": float(np.mean([d > 0.01 for d in deltas])),
        }

    report = {
        "n_successful": len(records),
        "n_failed": len(failures),
        "elapsed_seconds": round(time.monotonic() - t0, 2),
        "metrics": by_metric,
        "genre_final_score": genre_stats,
        "failure_counts_by_reason": fail_by_reason,
        "correlation_layer1_soft": corr_l1,
        "correlation_layer2_top": corr_l2,
        "scale_adherence_key_drift": drift,
    }
    pathlib.Path(out_stem + ".json").write_text(json.dumps(report, indent=2))

    # --- Human summary ----------------------------------------------------
    print()
    print("=" * 70)
    print(f"Component diagnostic — {len(records)} ok, {len(failures)} failed"
          f"  ({report['elapsed_seconds']:.0f}s)")
    print("=" * 70)

    print("\nFinal score by genre:")
    for g in sorted(genre_stats):
        s = genre_stats[g]
        print(f"  {g:<14} mean={s.get('mean',0):.3f}  σ={s.get('std',0):.3f}"
              f"  p10–p90={s.get('p10',0):.3f}–{s.get('p90',0):.3f}  n={s.get('n',0)}")

    if failures:
        print(f"\nFailures by reason ({len(failures)} total):")
        for reason, cnt in sorted(fail_by_reason.items(), key=lambda x: -x[1])[:8]:
            print(f"  {cnt:3d}x  {reason}")

    decomp_keys = [k for k in by_metric if (
        (k.startswith("l1_") and k.endswith("_realkey") and "_score_" not in k)
        or (k.startswith("l2_") and not k.startswith("l2_integ_")
            and k not in ("l2_tempo_mean", "l2_harmony_mean",
                          "l2_rhythm_mean", "l2_noise_value"))
    )]
    rows = [(by_metric[k]["median"], k,
             by_metric[k]["mean"], by_metric[k]["std"],
             by_metric[k]["p10"], by_metric[k]["p90"])
            for k in decomp_keys if by_metric[k].get("n", 0) > 0]
    rows.sort()
    print(f"\n{'metric':<46}  {'med':>6}  {'mean':>6}  {'std':>5}  {'p10':>6}  {'p90':>6}")
    for med, k, mean, std, p10, p90 in rows:
        flag = "  ←LOW" if med < 0.3 else ("  ←HIGH" if med > 0.85 else "")
        print(f"  {k:<44}  {med:6.3f}  {mean:6.3f}  {std:5.3f}  {p10:6.3f}  {p90:6.3f}{flag}")

    if drift:
        print("\nScale-adherence key drift:")
        print(f"  mean(real_key − key='C'): {drift['mean_delta_realkey_minus_keyC']:+.4f}")
        print(f"  fraction where real_key > C: {drift['fraction_realkey_higher']:.0%}")
        if abs(drift["mean_delta_realkey_minus_keyC"]) > 0.05:
            print("  WARN: significant drift — default key='C' mis-scores non-C samples")

    if corr_l1:
        print("\nLayer-1 soft-score correlations (|ρ| > 0.6):")
        seen: set = set()
        for a in sorted(corr_l1):
            for b in sorted(corr_l1[a]):
                if a >= b or (a, b) in seen:
                    continue
                seen.add((a, b))
                rho = corr_l1[a][b]
                if abs(rho) > 0.6:
                    sa = a.replace("l1_", "").replace("_realkey", "")
                    sb = b.replace("l1_", "").replace("_realkey", "")
                    print(f"  ρ={rho:+.2f}   {sa}  ↔  {sb}")

    print(f"\nWritten: {out_stem}.json, {out_stem}.csv")


if __name__ == "__main__":
    main()
