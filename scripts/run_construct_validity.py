#!/usr/bin/env python3
"""E2 construct validity experiment — gate discrimination between good and bad audio.

Usage:
    python3 scripts/run_construct_validity.py --bad-type both --n-good 50 --n-bad 20
    python3 scripts/run_construct_validity.py --bad-type synthetic --quick
    python3 scripts/run_construct_validity.py --bad-type real --n-good 30 --n-bad 30
    python3 scripts/run_construct_validity.py --n-good 100 --n-bad 100 --latex

Writes:
    <out>.json   — full results (paper-ready numbers)
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
    EXCLUDED_FROM_CRITERION,
    EXCLUSION_NOTES,
    ConstructValidityResult,
    PathologyResult,
    run_construct_validity_test,
)


# ---------------------------------------------------------------------------
# Console table
# ---------------------------------------------------------------------------

def _print_table(r: ConstructValidityResult) -> None:
    print()
    print("=" * 80)
    print(
        f"E2 Construct Validity  —  good n={r.n_good}"
        f"  mean={r.good_mean:.3f}  σ={r.good_std:.3f}"
    )
    print("=" * 80)
    hdr = f"  {'pathology':<26}  {'n':>4}  {'mean ± σ':>12}  {'AUROC [95% CI]':>20}  note"
    print(hdr)
    print("  " + "-" * 76)

    for p in r.pathologies:
        mean_str = f"{p.mean:.3f} ± {p.std:.3f}"
        ci_str = f"{p.auroc:.3f} [{p.ci_low:.3f}–{p.ci_high:.3f}]"
        note = _note(p)
        pass_flag = "excl" if p.excluded else \
                    ("✓" if p.auroc >= 0.85 else ("~" if p.auroc >= 0.70 else "✗"))
        print(f"  {p.name:<26}  {len(p.scores):4d}  {mean_str:>12}  {ci_str:>20}  {pass_flag} {note}")

    print()
    oci = r.overall_auroc_ci
    cci = r.criterion_auroc_ci
    n_excl = sum(1 for p in r.pathologies if p.excluded)
    n_all = len(r.pathologies)
    gate = "PASS" if r.criterion_auroc >= 0.85 else \
           ("MARGINAL" if r.criterion_auroc >= 0.70 else "FAIL")

    print(f"Overall AUROC   ({n_all:2d} pathologies) = "
          f"{r.overall_auroc:.3f} [{oci[0]:.3f}–{oci[1]:.3f}]"
          f"  d={r.overall_separation:.2f}")
    print(f"Criterion AUROC ({n_all-n_excl:2d} pathologies) = "
          f"{r.criterion_auroc:.3f} [{cci[0]:.3f}–{cci[1]:.3f}]"
          f"  [{gate}]  (≥0.85 required)")

    excl = [p for p in r.pathologies if p.excluded]
    if excl:
        print()
        print("Excluded from criterion:")
        for p in excl:
            note_text = EXCLUSION_NOTES.get(p.exclusion_reason, p.exclusion_reason)
            print(f"  {p.name}: {note_text}")


def _note(p: PathologyResult) -> str:
    if p.excluded:
        tag = "†" if p.exclusion_reason == "temporal-completeness" else "‡"
        return f"excl{tag}"
    d = p.separation
    if d != d:  # NaN → std==0 → all rejected
        return f"rej={p.rejection_rate:.0%} ({int(p.rejection_rate * len(p.scores))}/{len(p.scores)})"
    return ""


# ---------------------------------------------------------------------------
# LaTeX table
# ---------------------------------------------------------------------------

def _latex_table(r: ConstructValidityResult) -> str:
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{E2 Construct validity: musicality gate AUROC vs.\ known pathologies.}",
        r"\label{tab:e2-construct-validity}",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Type & Pathology & $n$ & Mean $\pm$ SD & AUROC [95\,\% CI] \\",
        r"\midrule",
    ]

    prev_type = None
    for p in r.pathologies:
        ptype, pname = p.name.split("/", 1)
        type_str = ptype if ptype != prev_type else ""
        prev_type = ptype

        mean_str = f"${p.mean:.3f} \\pm {p.std:.3f}$"
        if p.std == 0.0:
            rej_n = int(p.rejection_rate * len(p.scores))
            mean_str = f"${p.mean:.3f}$ (rej.\\ {rej_n}/{len(p.scores)})"

        auroc_str = f"${p.auroc:.3f}\\;[{p.ci_low:.3f},\\,{p.ci_high:.3f}]$"
        if p.excluded:
            marker = r"$^\dagger$" if p.exclusion_reason == "temporal-completeness" \
                     else r"$^\ddagger$"
            auroc_str += marker

        lines.append(
            f"{type_str} & {pname} & {len(p.scores)} & {mean_str} & {auroc_str} \\\\"
        )

    n_excl = sum(1 for p in r.pathologies if p.excluded)
    n_all = len(r.pathologies)
    oci = r.overall_auroc_ci
    cci = r.criterion_auroc_ci
    lines += [
        r"\midrule",
        f"\\multicolumn{{2}}{{l}}{{Overall ({n_all} pathologies)}} & "
        f"{sum(len(p.scores) for p in r.pathologies)} & "
        f"--- & "
        f"${r.overall_auroc:.3f}\\;[{oci[0]:.3f},\\,{oci[1]:.3f}]$ \\\\",
        f"\\multicolumn{{2}}{{l}}{{\\textbf{{Criterion}} ({n_all-n_excl} pathologies)}} & "
        f"{sum(len(p.scores) for p in r.pathologies if not p.excluded)} & "
        f"--- & "
        f"$\\mathbf{{{r.criterion_auroc:.3f}}}\\;[{cci[0]:.3f},\\,{cci[1]:.3f}]$ \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}",
        r"\small",
        f"Good set: $n={r.n_good}$, mean $= {r.good_mean:.3f}$, SD $= {r.good_std:.3f}$. "
        r"AUROC via Mann–Whitney $U$; 95\,\% CI from $B=2000$ bootstrap samples. "
        r"Pass criterion: AUROC $\geq 0.85$. "
        r"$^\dagger$ Excluded (temporal completeness): gate measures per-frame audio quality; "
        r"silence distribution is out of scope. "
        r"$^\ddagger$ Excluded (confound): 4$\times$ resample conflates pitch shift "
        r"($\pm$2\,oct.) with tempo; a dedicated temporal-coherence check is out of scope.",
        r"\end{tablenotes}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# JSON serialisation
# ---------------------------------------------------------------------------

def _to_report(r: ConstructValidityResult, elapsed: float) -> dict:
    return {
        "elapsed_seconds": round(elapsed, 2),
        "n_good": r.n_good,
        "good_mean": r.good_mean,
        "good_std": r.good_std,
        "good_scores": r.good_scores,
        "overall_auroc": r.overall_auroc,
        "overall_auroc_ci": list(r.overall_auroc_ci),
        "overall_separation": r.overall_separation,
        "criterion_auroc": r.criterion_auroc,
        "criterion_auroc_ci": list(r.criterion_auroc_ci),
        "criterion_n_pathologies": r.criterion_n_pathologies,
        "pathologies": [
            {
                "name": p.name,
                "n": len(p.scores),
                "mean": p.mean,
                "std": p.std,
                "auroc": p.auroc,
                "auroc_ci_95": [p.ci_low, p.ci_high],
                "cohens_d": None if (p.separation != p.separation) else p.separation,
                "rejection_rate": p.rejection_rate,
                "excluded_from_criterion": p.excluded,
                "exclusion_reason": p.exclusion_reason,
                "scores": p.scores,
            }
            for p in r.pathologies
        ],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--bad-type", choices=["synthetic", "real", "both"], default="both")
    parser.add_argument("--n-good", type=int, default=50)
    parser.add_argument("--n-bad", type=int, default=20,
                        help="Bad samples per pathology type")
    parser.add_argument("--synthetic-signals", nargs="+", default=None,
                        choices=SYNTHETIC_TYPES)
    parser.add_argument("--real-signals", nargs="+", default=None,
                        choices=REAL_TYPES)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--bootstrap-n", type=int, default=2000,
                        help="Bootstrap iterations for CI (0 to skip)")
    parser.add_argument("--out", default="eval_results/e2_construct_validity")
    parser.add_argument("--latex", action="store_true",
                        help="Also write <out>.tex (LaTeX table)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick run: n-good=5, n-bad=5, bootstrap-n=500")
    args = parser.parse_args()

    if args.quick:
        args.n_good = 5
        args.n_bad = 5
        if args.bootstrap_n == 2000:
            args.bootstrap_n = 500

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    print(f"E2 Construct Validity")
    print(f"  bad-type={args.bad_type}  n-good={args.n_good}  n-bad={args.n_bad}"
          f"  bootstrap-n={args.bootstrap_n}")
    print(f"  synthetic={args.synthetic_signals or 'all'}"
          f"  real={args.real_signals or 'all'}")
    print()

    t0 = time.monotonic()
    r = run_construct_validity_test(
        n_good=args.n_good,
        n_bad=args.n_bad,
        bad_types=args.bad_type,
        synthetic_signals=args.synthetic_signals,
        real_signals=args.real_signals,
        base_seed=args.base_seed,
        bootstrap_n=args.bootstrap_n,
    )
    elapsed = time.monotonic() - t0

    _print_table(r)

    report = _to_report(r, elapsed)
    json_path = args.out + ".json"
    pathlib.Path(json_path).write_text(json.dumps(report, indent=2))
    print(f"\nWritten: {json_path}  ({elapsed:.0f}s)")

    if args.latex:
        tex = _latex_table(r)
        tex_path = args.out + ".tex"
        pathlib.Path(tex_path).write_text(tex)
        print(f"Written: {tex_path}")


if __name__ == "__main__":
    main()
