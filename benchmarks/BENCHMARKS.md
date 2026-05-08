# musicgen — Benchmark Guide

How to run the performance benchmark suite, collect cross-platform results, generate publication-ready charts, and report metrics in a paper.

---

## 1. Prerequisites

### Both platforms

```bash
git clone https://github.com/dobidu/layered_music_gen.git
cd layered_music_gen
python -m venv .venv
source .venv/bin/activate          # Windows/WSL: .venv\Scripts\activate
pip install -e '.[dev]'
pip install psutil                 # not in default deps; needed for memory metrics
```

### FluidSynth (required for synthesis benchmarks)

```bash
# Ubuntu / WSL
sudo apt-get install fluidsynth

# macOS
brew install fluidsynth
```

Verify: `fluidsynth --version`

### SoundFont files

Place `.sf2` files in `sf/<layer>/` (at least one per layer):

```
sf/beat/        *.sf2
sf/melody/      *.sf2
sf/harmony/     *.sf2
sf/bassline/    *.sf2
```

Benchmarks that do **not** require FluidSynth run regardless of SF2 availability.

---

## 2. Benchmark categories

| Category | FluidSynth needed | What it measures |
|---|---|---|
| **Fast / Python-only** | No | Sampler, genre system, MIDI generators, full pipeline w/ mocked renderer |
| **Synthesis** | Yes | Real end-to-end single-sample generation (FluidSynth synthesis included) |
| **Batch scaling** | Yes | Worker count vs throughput; parallel efficiency |
| **Memory** | No | Peak RSS delta during mocked pipeline |
| **Test suite** | No | Fast `pytest` execution time |

---

## 3. Running on each machine

### 3.1 Recommended run (full benchmark)

```bash
# with FluidSynth + sf2 files present
python benchmarks/bench.py --n 30 --batch-samples 16
```

`--n 30` runs each fast benchmark 30 times (mean ± std). For the paper, `--n 50` is preferred — takes ~5 min on the fast suite alone.

### 3.2 Fast-only (no FluidSynth required)

```bash
python benchmarks/bench.py --fast-only --n 50
```

Skips: `single_sample_full`, `batch_workers_*`, `test_suite_fast` (test suite is optional but slow).

### 3.3 Custom worker counts

```bash
# explicit worker list for batch scaling
python benchmarks/bench.py --workers 1 2 4 8 12 --batch-samples 24
```

Use `--batch-samples` ≥ 2× the largest worker count to avoid worker-starvation artifacts.

### 3.4 Output

Results land in `benchmarks/results/<hostname>_<timestamp>.json`.  
Charts land in `benchmarks/figures/` immediately after the run.

---

## 4. Cross-platform collection

Run on each machine, copy the JSON result file off the machine:

```bash
# from WSL / Linux (Ryzen)
cp benchmarks/results/<ryzen_file>.json /mnt/d/paper/ryzen.json

# from macOS (M4) — scp or AirDrop to same directory
scp benchmarks/results/<m4_file>.json user@host:/path/paper/m4.json
```

### Generate comparison charts

```bash
# run from either machine with both files accessible
python benchmarks/plot.py paper/ryzen.json --compare paper/m4.json --out paper/figures/
```

Produces: `06_comparison.png`, `07_batch_comparison.png` (in addition to per-machine charts).

### Generate per-machine charts only

```bash
python benchmarks/plot.py paper/ryzen.json --out paper/figures/ryzen/
python benchmarks/plot.py paper/m4.json    --out paper/figures/m4/
```

---

## 5. Result JSON structure

```json
{
  "timestamp": "2026-05-08T02:13:05Z",
  "system": {
    "hostname":           "Desktop-Bidu",
    "cpu":                "AMD Ryzen 9 9900X3D",
    "cpu_count_physical": 12,
    "cpu_count_logical":  24,
    "ram_gb":             32.0,
    "python":             "3.12.3",
    "musicgen_version":   "0.2.0",
    "has_fluidsynth":     true,
    "fluidsynth_version": "FluidSynth runtime 2.3.4"
  },
  "benchmarks": {
    "sampler_sample": {
      "mean_ms": 0.029, "std_ms": 0.01,
      "min_ms": 0.021,  "max_ms": 0.056,
      "median_ms": 0.025, "n": 50
    },
    "full_midi_pipeline":    { ... },
    "genre_overhead_none":   { ... },
    "genre_overhead_1":      { ... },
    "genre_overhead_2_merged": { ... },
    "single_sample_full":    { "mean_ms": ..., "n": 5 },
    "batch_workers_1":  { "workers": 1,  "samples": 16, "elapsed_s": ...,
                          "samples_per_sec": ..., "samples_per_hour": ... },
    "batch_workers_4":  { ... },
    "batch_workers_12": { ... },
    "memory_single_mocked": { "baseline_mb": 142, "after_mb": 143, "delta_mb": 1 },
    "test_suite_fast": { "mean_ms": 3800, "test_count": 1047, "exit_code": 0 }
  }
}
```

All timing fields are in milliseconds. `n` is the number of repetitions used to compute statistics.

---

## 6. Key metrics and how to report them

### 6.1 Throughput — samples per hour

Derived from `batch_workers_N.samples_per_hour`. This is the headline number for a dataset generation paper.

**Recommended table format:**

| Platform | Workers | Samples / hour | Parallel efficiency |
|---|---|---|---|
| Ryzen 9 9900X3D (Linux) | 1 | X | 100% (baseline) |
| Ryzen 9 9900X3D (Linux) | 4 | X | Y% |
| Ryzen 9 9900X3D (Linux) | 12 | X | Y% |
| Apple M4 (macOS) | 1 | X | 100% (baseline) |
| Apple M4 (macOS) | 8 | X | Y% |

Parallel efficiency = `(sps_N / (sps_1 × N)) × 100`.  
Compute from `samples_per_sec` fields in the JSON.

### 6.2 End-to-end single-sample time

Field: `single_sample_full.mean_ms ± std_ms`

Report as: **X.X ± Y.Y s** (convert ms → s for readability).  
This decomposes into:
- Python overhead: `single_sample_mocked.mean_ms` (mocked renderer)
- FluidSynth synthesis: `single_sample_full.mean_ms − single_sample_mocked.mean_ms`

**Sentence template for paper:**  
> "Single-sample generation takes X.X ± Y.Y s on [platform], of which Z% is FluidSynth synthesis and the remainder is pure Python (sampler, generators, FX, writer)."

### 6.3 Genre system overhead

Fields: `genre_overhead_none`, `genre_overhead_1`, `genre_overhead_2_merged` (all in ms).

Overhead % = `(genre_1.mean_ms / none.mean_ms − 1) × 100`

**Sentence template:**  
> "Enabling the genre system adds X% overhead to MIDI generation (1 genre: Y ms vs baseline Z ms). Merging two genres costs W ms — still negligible relative to FluidSynth synthesis time."

### 6.4 Python pipeline decomposition

For the component breakdown chart (`01_component_breakdown.png`), report individual components:

| Component | Time (ms) | % of Python pipeline |
|---|---|---|
| SongParams.sample() | 0.03 | <1% |
| Chord progression | 4.3 | ~6% |
| Bassline | 1.0 | ~1% |
| Beat (MIDI) | 0.3 | <1% |
| Full MIDI pipeline | 68 | 100% |
| Full pipeline (mocked) | 8 | — |

### 6.5 Memory footprint

Field: `memory_single_mocked.delta_mb`

**Sentence template:**  
> "Peak RSS increase during sample generation is X MB (Python pipeline only; FluidSynth subprocess memory is separate)."

For FluidSynth memory, monitor with:
```bash
/usr/bin/time -v python benchmarks/bench.py --fast-only --n 1 --no-plot 2>&1 | grep "Maximum resident"
```

### 6.6 Test suite

Field: `test_suite_fast.test_count`, `test_suite_fast.mean_ms`

> "The fast test suite (N=1047 unit tests) completes in X.X s on [platform], enabling rapid iteration without FluidSynth."

---

## 7. Statistical guidance

### Minimum repetitions for the paper

| Benchmark | Minimum `--n` | Rationale |
|---|---|---|
| Sampler, genre | 50 | Fast (<1 ms); need large n for stable std |
| Generators, MIDI pipeline | 30 | Moderate variance from RNG draws |
| Mocked pipeline | 20 | Involves I/O (tempfile); 20 sufficient |
| Real synthesis | 5 | Slow (seconds); 5 enough for std estimate |
| Batch scaling | 3 runs each | Restart process between runs to flush OS cache |

### What to report

Always report: **mean ± std**, **n**, **machine spec**.  
For throughput: report **median** (more robust to OS scheduling jitter) alongside mean.  
For batch scaling: report **samples/hour** (practical unit) + **parallel efficiency**.

### Outlier handling

The benchmark uses wall-clock time via `time.perf_counter`. On heavily loaded machines, occasional OS preemption inflates `max_ms`. Use **median** as the primary metric for paper tables. Flag any `max_ms / median_ms > 3` as suspect — re-run after closing background processes.

### Warm-up

`bench.py` runs 2 warm-up iterations before timing (configurable in `time_fn`). This primes Python's JIT, filesystem cache, and import machinery. Results are therefore **steady-state** throughput, not cold-start.

---

## 8. Controlled experiment checklist

Run these steps on both machines before the benchmark run used for the paper:

```
□ Close all non-essential applications
□ Disable Turbo Boost / Boost mode for stable clock:
    Linux: echo "1" | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo
    macOS: no standard tool; note if boost is on in system section
□ Plug in / connect power (laptops/M4 Mac)
□ Wait 60 s after boot before running
□ Run bench.py twice; use second run (cache warm)
□ Record ambient temperature and CPU temp if available:
    Linux: sensors | grep -E "Core|Package"
    macOS: sudo powermetrics --samplers smc -n 1 | grep "CPU die"
□ Note FluidSynth version: fluidsynth --version
□ Note OS version: uname -a  (Linux) or sw_vers (macOS)
□ Confirm sf2 pool size: ls sf/beat/ | wc -l  (affects soundfont selection timing)
```

---

## 9. Figures reference

| File | Content | Paper section |
|---|---|---|
| `01_component_breakdown.png` | Horizontal bar: timing per pipeline component | System description / Implementation |
| `02_genre_overhead.png` | Grouped bar: no genre / 1 genre / 2 merged | Genre system evaluation |
| `03_genre_load_merge.png` | Bar: genre load and merge timing | Genre system evaluation |
| `04_batch_scaling.png` | Line + efficiency bar: samples/hour vs workers | Scalability / Experiments |
| `05_summary.png` | Infographic: headline numbers | Abstract / Introduction |
| `06_comparison.png` | Grouped bar: Ryzen vs M4 key metrics | Cross-platform evaluation |
| `07_batch_comparison.png` | Line: throughput curves, both platforms | Scalability / Experiments |
| `08_test_suite.png` | Card: test count + execution time | Reproducibility / Quality |

All charts are 150 DPI PNG. For camera-ready submission, re-export at 300 DPI:

```bash
# edit plot.py line: figure.dpi → 300, then re-run
python benchmarks/plot.py results/ryzen.json --out figures_hires/
```

Or convert existing PNGs:
```bash
for f in benchmarks/figures/*.png; do
    convert "$f" -resample 300 "hires/$(basename $f)"
done
```

---

## 10. Reporting checklist for the paper

- [ ] State FluidSynth version and binary source (affects WAV reproducibility)
- [ ] State Python version and OS on each platform
- [ ] State `--n` (repetitions) used for each benchmark category
- [ ] Report mean ± std for all timing metrics
- [ ] Report samples/hour at 1 worker and at max workers for both platforms
- [ ] Report parallel efficiency at max workers
- [ ] Note that synthesis benchmarks require pinned FluidSynth binary for WAV reproducibility (MIDI and `sample.json` are unconditionally bit-identical)
- [ ] Mention mocked-renderer baseline clarifies Python vs synthesis split
- [ ] Note the melody Markov zero-weight bug (pre-existing, v0.3 target) affects generator benchmark only — `generate()` handles it via `SampleResult.status="failed"`

---

## 11. Re-running plots from existing results

```bash
# single machine, regenerate all charts
python benchmarks/plot.py benchmarks/results/Desktop-Bidu_20260508_021305.json

# two machines, comparison charts
python benchmarks/plot.py benchmarks/results/ryzen_<ts>.json \
    --compare benchmarks/results/m4_<ts>.json \
    --out paper/figures/
```

No re-running of benchmarks needed — the JSON is the source of truth.
