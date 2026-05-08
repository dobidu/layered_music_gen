#!/usr/bin/env python3
"""
musicgen benchmark chart generator.

Usage:
    python benchmarks/plot.py results/host_20260507_120000.json
    python benchmarks/plot.py results/ryzen.json --compare results/m4.json
    python benchmarks/plot.py results/ryzen.json --out benchmarks/figures/
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Presentation style ──────────────────────────────────────────────────────

BG      = "#1a1a2e"
PANEL   = "#16213e"
GREEN   = "#1DB954"
BLUE    = "#4a9eff"
ORANGE  = "#ff6b35"
GREY    = "#8892a4"
WHITE   = "#e8eaf6"
RED     = "#ff4757"

ACCENT_A = GREEN
ACCENT_B = ORANGE

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    PANEL,
    "axes.edgecolor":    GREY,
    "axes.labelcolor":   WHITE,
    "axes.titlecolor":   WHITE,
    "text.color":        WHITE,
    "xtick.color":       WHITE,
    "ytick.color":       WHITE,
    "grid.color":        GREY,
    "grid.alpha":        0.3,
    "font.family":       "sans-serif",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "legend.facecolor":  PANEL,
    "legend.edgecolor":  GREY,
    "figure.dpi":        150,
})


# ── Helpers ──────────────────────────────────────────────────────────────────

def load(path: str) -> Dict:
    return json.loads(Path(path).read_text())


def label_for(data: Dict) -> str:
    s = data["system"]
    cpu = s["cpu"]
    # shorten CPU string
    for drop in ["(TM)", "(R)", "Processor", "CPU"]:
        cpu = cpu.replace(drop, "")
    cpu = cpu.strip()
    if len(cpu) > 30:
        cpu = cpu[:28] + "…"
    return cpu or s["hostname"]


def ms_val(d: Dict, key: str, sub: str = "mean_ms") -> Optional[float]:
    entry = d["benchmarks"].get(key)
    if entry is None:
        return None
    return entry.get(sub)


def annotate_bar(ax, bars, fmt="{:.1f}", color=WHITE, fontsize=9, offset=2):
    for bar in bars:
        h = bar.get_height()
        if h and h > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + offset,
                fmt.format(h),
                ha="center", va="bottom", color=color, fontsize=fontsize,
            )


def annotate_hbar(ax, bars, fmt="{:.1f}", color=WHITE, fontsize=9, offset=1):
    for bar in bars:
        w = bar.get_width()
        if w and w > 0:
            ax.text(
                w + offset,
                bar.get_y() + bar.get_height() / 2,
                fmt.format(w),
                ha="left", va="center", color=color, fontsize=fontsize,
            )


def save(fig, path: Path, name: str):
    path.mkdir(parents=True, exist_ok=True)
    out = path / name
    fig.savefig(out, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  saved → {out}")


# ── Chart functions ───────────────────────────────────────────────────────────

def chart_component_breakdown(data: Dict, out: Path):
    """Horizontal bar: mean_ms for each pipeline component."""
    bm = data["benchmarks"]
    keys = [
        ("sampler_sample",        "SongParams.sample()"),
        ("chord_progression",     "Chord generation"),
        ("melody",                "Melody generation"),
        ("bassline",              "Bassline generation"),
        ("beat",                  "Beat (MIDI)"),
        ("full_midi_pipeline",    "Full MIDI pipeline (1 song)"),
        ("single_sample_mocked",  "Full pipeline\n(mocked renderer)"),
    ]
    labels, values, errs = [], [], []
    for k, lbl in keys:
        entry = bm.get(k)
        if entry:
            labels.append(lbl)
            values.append(entry["mean_ms"])
            errs.append(entry.get("std_ms", 0))

    if not values:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [BLUE] * (len(values) - 1) + [GREEN]
    bars = ax.barh(labels, values, xerr=errs, color=colors,
                   error_kw={"ecolor": GREY, "capsize": 3}, height=0.6)
    annotate_hbar(ax, bars, fmt="{:.1f} ms", offset=max(values) * 0.01)
    ax.set_xlabel("Time (ms, lower is better)")
    ax.set_title("Pipeline Component Timing — single call, mean ± std")
    ax.invert_yaxis()
    ax.grid(axis="x")
    fig.tight_layout()
    save(fig, out, "01_component_breakdown.png")


def chart_genre_overhead(data: Dict, out: Path):
    """Bar chart: no genre / 1 genre / 2 merged — full MIDI pipeline time."""
    bm = data["benchmarks"]
    keys = [
        ("genre_overhead_none",     "No genre"),
        ("genre_overhead_1",        "1 genre (jazz)"),
        ("genre_overhead_2_merged", "2 genres merged\n(jazz + latin)"),
    ]
    labels, values, errs = [], [], []
    for k, lbl in keys:
        entry = bm.get(k)
        if entry:
            labels.append(lbl)
            values.append(entry["mean_ms"])
            errs.append(entry.get("std_ms", 0))

    if not values:
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = [BLUE, GREEN, ORANGE][:len(values)]
    bars = ax.bar(labels, values, yerr=errs, color=colors,
                  error_kw={"ecolor": GREY, "capsize": 4}, width=0.5)
    annotate_bar(ax, bars, fmt="{:.1f} ms", offset=max(values) * 0.01)
    ax.set_ylabel("Time (ms, lower is better)")
    ax.set_title("Genre System Overhead — full MIDI pipeline, mean ± std")
    ax.grid(axis="y")
    fig.tight_layout()
    save(fig, out, "02_genre_overhead.png")


def chart_genre_load(data: Dict, out: Path):
    """Bar chart: time to load/merge genres."""
    bm = data["benchmarks"]
    keys = [
        ("genre_load_single",    "Load 1 genre"),
        ("genre_load_all_8",     "Load all 8 genres"),
        ("genre_merge_2",        "Merge 2 genres"),
        ("genre_merge_8",        "Merge 8 genres"),
        ("genre_resolve_2_names","Resolve 2 names\n(load + merge)"),
    ]
    labels, values = [], []
    for k, lbl in keys:
        entry = bm.get(k)
        if entry:
            labels.append(lbl)
            values.append(entry["mean_ms"])

    if not values:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, values, color=GREEN, width=0.5)
    annotate_bar(ax, bars, fmt="{:.2f} ms", offset=max(values) * 0.01)
    ax.set_ylabel("Time (ms, lower is better)")
    ax.set_title("Genre System — load & merge timing")
    ax.grid(axis="y")
    fig.tight_layout()
    save(fig, out, "03_genre_load_merge.png")


def chart_batch_scaling(data: Dict, out: Path):
    """Line + bar: throughput vs worker count."""
    bm = data["benchmarks"]
    batch_keys = sorted(
        [k for k in bm if k.startswith("batch_workers_")],
        key=lambda k: bm[k]["workers"],
    )
    if not batch_keys:
        return

    workers  = [bm[k]["workers"] for k in batch_keys]
    sph      = [bm[k]["samples_per_hour"] for k in batch_keys]
    sps      = [bm[k]["samples_per_sec"] for k in batch_keys]

    # ideal linear scaling from workers=1
    baseline_sps = sps[0]
    ideal_sph = [baseline_sps * w * 3600 for w in workers]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: samples/hour vs workers
    ax1.plot(workers, ideal_sph, "--", color=GREY, label="Linear ideal", linewidth=1.5)
    ax1.plot(workers, sph, "o-", color=GREEN, label="Actual", linewidth=2, markersize=8)
    for w, s in zip(workers, sph):
        ax1.annotate(f"{s:.0f}", (w, s), textcoords="offset points",
                     xytext=(5, 6), color=WHITE, fontsize=9)
    ax1.set_xlabel("Workers")
    ax1.set_ylabel("Samples / hour")
    ax1.set_title("Batch Throughput vs Workers")
    ax1.set_xticks(workers)
    ax1.legend()
    ax1.grid()

    # Right: parallel efficiency
    efficiency = [s / (baseline_sps * w) * 100 for w, s in zip(workers, sps)]
    bars = ax2.bar([str(w) for w in workers], efficiency, color=BLUE, width=0.5)
    ax2.axhline(100, color=GREY, linestyle="--", linewidth=1, label="100% efficiency")
    annotate_bar(ax2, bars, fmt="{:.0f}%", offset=0.5)
    ax2.set_xlabel("Workers")
    ax2.set_ylabel("Parallel efficiency (%)")
    ax2.set_title("Parallel Efficiency vs Workers")
    ax2.set_ylim(0, 115)
    ax2.legend()
    ax2.grid(axis="y")

    fig.suptitle("Batch Generation Scaling", fontsize=14, fontweight="bold", color=WHITE)
    fig.tight_layout()
    save(fig, out, "04_batch_scaling.png")


def chart_throughput_summary(data: Dict, out: Path):
    """Single infographic: key throughput numbers."""
    bm = data["benchmarks"]
    sys = data["system"]

    # collect numbers
    sampler_us = ms_val(data, "sampler_sample")
    midi_ms = ms_val(data, "full_midi_pipeline")
    mock_ms = ms_val(data, "single_sample_mocked")

    batch_keys = sorted(
        [k for k in bm if k.startswith("batch_workers_")],
        key=lambda k: bm[k]["workers"],
    )
    max_sph = None
    max_workers = None
    if batch_keys:
        best = max(batch_keys, key=lambda k: bm[k]["samples_per_hour"])
        max_sph = bm[best]["samples_per_hour"]
        max_workers = bm[best]["workers"]

    full_ms = ms_val(data, "single_sample_full")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")

    title = f"{label_for(data)}  ·  {sys['ram_gb']} GB RAM"
    ax.text(0.5, 0.97, title, ha="center", va="top",
            fontsize=12, color=GREY, transform=ax.transAxes)
    ax.text(0.5, 0.88, "musicgen — performance summary", ha="center", va="top",
            fontsize=16, fontweight="bold", color=WHITE, transform=ax.transAxes)

    metrics = []
    if sampler_us is not None:
        metrics.append((f"{sampler_us/1000*1e6:.0f} μs", "SongParams.sample()\n(sampler throughput)"))
    if midi_ms is not None:
        metrics.append((f"{midi_ms:.0f} ms", "Full MIDI generation\n(1 song, all parts)"))
    if mock_ms is not None:
        metrics.append((f"{mock_ms/1000:.2f} s", "Full pipeline\n(mocked synthesis)"))
    if full_ms is not None:
        metrics.append((f"{full_ms/1000:.1f} s", "Full pipeline\n(real FluidSynth)"))
    if max_sph is not None:
        metrics.append((f"{max_sph:,.0f}/hr", f"Batch throughput\n({max_workers} workers)"))

    n = len(metrics)
    xs = np.linspace(0.1, 0.9, n) if n > 1 else [0.5]
    for x, (val, lbl) in zip(xs, metrics):
        ax.text(x, 0.55, val, ha="center", va="center",
                fontsize=22, fontweight="bold", color=GREEN, transform=ax.transAxes)
        ax.text(x, 0.32, lbl, ha="center", va="center",
                fontsize=9, color=GREY, transform=ax.transAxes,
                multialignment="center")

    # divider line
    line = plt.Line2D([0.05, 0.95], [0.18, 0.18], transform=ax.transAxes,
                      color=GREY, linewidth=0.5, alpha=0.5)
    ax.add_line(line)

    ts_entry = bm.get("test_suite_fast")
    footer = ""
    if ts_entry:
        tc = ts_entry.get("test_count", "?")
        ts_s = ts_entry.get("mean_ms", 0) / 1000
        footer += f"{tc} fast tests in {ts_s:.1f}s   ·   "
    footer += f"Python {sys['python']}   ·   musicgen {sys['musicgen_version']}"
    ax.text(0.5, 0.08, footer, ha="center", va="center",
            fontsize=9, color=GREY, transform=ax.transAxes)

    fig.tight_layout()
    save(fig, out, "05_summary.png")


def chart_comparison(data_a: Dict, data_b: Dict, out: Path):
    """Grouped bar: compare two machines across key benchmarks."""
    label_a = label_for(data_a)
    label_b = label_for(data_b)

    comparisons = [
        ("sampler_sample",       "SongParams.sample()\n(μs)",     1000),
        ("full_midi_pipeline",   "MIDI pipeline\n(ms)",           1),
        ("genre_merge_2",        "merge_genres × 2\n(ms)",        1),
        ("single_sample_mocked", "Full pipeline\nmocked (s)",     1 / 1000),
    ]
    if any(bm_key in data_a["benchmarks"] or bm_key in data_b["benchmarks"]
           for bm_key, _, _ in [("single_sample_full", "", 1)]):
        comparisons.append(("single_sample_full", "Full pipeline\nFluidSynth (s)", 1 / 1000))

    labels, vals_a, vals_b = [], [], []
    for key, lbl, scale in comparisons:
        va = ms_val(data_a, key)
        vb = ms_val(data_b, key)
        if va is not None or vb is not None:
            labels.append(lbl)
            vals_a.append((va or 0) * scale)
            vals_b.append((vb or 0) * scale)

    if not labels:
        print("  [comparison] no overlapping benchmarks found")
        return

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(9, len(labels) * 2), 5))
    bars_a = ax.bar(x - width / 2, vals_a, width, color=ACCENT_A, label=label_a)
    bars_b = ax.bar(x + width / 2, vals_b, width, color=ACCENT_B, label=label_b)

    for bars, vals in [(bars_a, vals_a), (bars_b, vals_b)]:
        for bar, v in zip(bars, vals):
            if v:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals_a + vals_b) * 0.01,
                    f"{v:.3g}",
                    ha="center", va="bottom", color=WHITE, fontsize=8,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Time (see axis label per metric)")
    ax.set_title("Cross-Platform Comparison")
    ax.legend()
    ax.grid(axis="y")
    fig.tight_layout()
    save(fig, out, "06_comparison.png")


def chart_batch_comparison(data_a: Dict, data_b: Dict, out: Path):
    """Batch throughput comparison across worker counts for two machines."""
    def get_scaling(data):
        bm = data["benchmarks"]
        keys = sorted(
            [k for k in bm if k.startswith("batch_workers_")],
            key=lambda k: bm[k]["workers"],
        )
        return [(bm[k]["workers"], bm[k]["samples_per_hour"]) for k in keys]

    scaling_a = get_scaling(data_a)
    scaling_b = get_scaling(data_b)

    if not scaling_a and not scaling_b:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    if scaling_a:
        wa, sa = zip(*scaling_a)
        ax.plot(wa, sa, "o-", color=ACCENT_A, label=label_for(data_a), linewidth=2, markersize=8)
    if scaling_b:
        wb, sb = zip(*scaling_b)
        ax.plot(wb, sb, "s-", color=ACCENT_B, label=label_for(data_b), linewidth=2, markersize=8)

    ax.set_xlabel("Workers")
    ax.set_ylabel("Samples / hour")
    ax.set_title("Batch Throughput — Cross-Platform Scaling")
    ax.legend()
    ax.grid()
    all_workers = sorted(set((scaling_a and [w for w, _ in scaling_a] or []) +
                              (scaling_b and [w for w, _ in scaling_b] or [])))
    if all_workers:
        ax.set_xticks(all_workers)
    fig.tight_layout()
    save(fig, out, "07_batch_comparison.png")


def chart_test_suite(data: Dict, out: Path):
    """Simple infographic for test suite timing."""
    entry = data["benchmarks"].get("test_suite_fast")
    if not entry:
        return

    tc = entry.get("test_count", 0)
    elapsed_s = entry.get("mean_ms", 0) / 1000

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axis("off")
    ax.text(0.5, 0.75, f"{tc}", ha="center", va="center",
            fontsize=52, fontweight="bold", color=GREEN, transform=ax.transAxes)
    ax.text(0.5, 0.45, "fast tests", ha="center", va="center",
            fontsize=14, color=WHITE, transform=ax.transAxes)
    ax.text(0.5, 0.2, f"in {elapsed_s:.1f} s  (pytest -m 'not slow')",
            ha="center", va="center", fontsize=10, color=GREY, transform=ax.transAxes)
    fig.tight_layout()
    save(fig, out, "08_test_suite.png")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="musicgen benchmark charts")
    parser.add_argument("results", help="primary result JSON file")
    parser.add_argument("--compare", metavar="JSON",
                        help="second result JSON for cross-platform comparison")
    parser.add_argument("--out", default="benchmarks/figures",
                        help="output directory for PNG files")
    args = parser.parse_args()

    out = Path(args.out)
    data_a = load(args.results)
    label = label_for(data_a)

    print(f"\nGenerating charts for: {label}")

    chart_component_breakdown(data_a, out)
    chart_genre_overhead(data_a, out)
    chart_genre_load(data_a, out)
    chart_batch_scaling(data_a, out)
    chart_throughput_summary(data_a, out)
    chart_test_suite(data_a, out)

    if args.compare:
        data_b = load(args.compare)
        print(f"Comparing with:       {label_for(data_b)}")
        chart_comparison(data_a, data_b, out)
        chart_batch_comparison(data_a, data_b, out)

    print(f"\nAll charts written to {out}/")


if __name__ == "__main__":
    main()
