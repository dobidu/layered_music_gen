# Phase 6: Productize II — Research

**Researched:** 2026-04-28
**Domain:** FluidSynth pre-roll measurement, ProcessPoolExecutor multiprocessing safety, typer CLI patterns, OutputMode conditional writes
**Confidence:** HIGH on FluidSynth pre-roll (measurement algorithm is deterministic), HIGH on ProcessPoolExecutor spawn context (documented stdlib behavior), HIGH on typer (well-documented library, Phase 3 already has the app stub), MEDIUM on inter-process manifest safety (chosen architecture avoids the hard problem by main-process-only writes).

---

## Summary

Phase 6 has four distinct technical domains. Three are well-understood: typer CLI patterns are documented and Phase 3 already has a working stub; OutputMode filtering is a conditional-write concern in `writer.py` with no subtle edge cases; `ProcessPoolExecutor` with spawn context is the standard Python multiprocessing pattern. The one domain with implementation nuance is **FluidSynth pre-roll measurement** (timing of a synthesizer's startup silence) and **multiprocessing safety** (fork vs. spawn on Linux, inter-process file locking). Both are addressed by decisions in CONTEXT.md.

---

## 1. FluidSynth Pre-roll: What It Is and How to Measure It

### What is pre-roll?

FluidSynth — like most audio software synthesizers — renders a short silence at the start of output before the first MIDI note is synthesized. This is a **startup latency / pre-roll** caused by:
- Audio driver buffer fill time (FluidSynth fills its internal ring buffer before producing output)
- Synthesizer reverb tail initialization (some soundfonts trigger an initial reverb flush)
- The `-F` (fast render) mode's buffer alignment

**Typical values:** 50–200 ms on standard hardware. Values above 500 ms are unusual and likely indicate a problem with the soundfont or audio subsystem.

### Why it matters for beat annotation

`musicgen.beats.extract_beat_times` reads beat onset times from MIDI ticks. These are the IDEAL beat times (what the MIDI says). When FluidSynth renders the audio, the audio is **delayed by the pre-roll offset**: the actual audio beat at time `t` seconds is at `t + pre_roll_offset_s` in the rendered WAV file.

Effect: without calibration, `beat_times = [0.0, 0.5, 1.0, 1.5]` while the actual audio has the beat at `[0.12, 0.62, 1.12, 1.62]` (for a 120ms pre-roll). This makes the beat annotations incorrect for downstream beat detection and audio→MIDI transcription models (R-P9's motivation).

### Measurement algorithm

```
1. Create a 1-note MIDI: middle C (note=60), velocity=64, 1 beat (480 ticks at 120 BPM, duration=480 ticks), standard 4/4.
2. Synthesize via FluidSynth: FluidSynth.midi_to_audio(midi_path, wav_path) using the first available .sf2 in any sf/ layer dir.
3. Read the WAV: scipy.io.wavfile.read(wav_path) → (sample_rate, int16 ndarray).
4. Convert to float32 normalized: signal = data.astype(np.float32) / 32768.0
5. Find first non-silent sample: first_nonsilent = np.where(np.abs(signal) > SILENCE_THRESHOLD)[0]
   - If empty: entirely silent → return 0.0 (FluidSynth absent or soundfont problem)
   - Else: offset_s = first_nonsilent[0] / sample_rate
6. Validate: if offset_s > 1.0 s → log warning (suspicious but use as-measured)
7. Return offset_s
```

**SILENCE_THRESHOLD = 1e-4** (`≈ -80 dBFS`). This is well above digital noise floor (~−120 dBFS for float32) but well below any audible signal. The calibration MIDI uses a loud velocity (64/127 = ~50%) so the note attack is well above this threshold.

**Why not use librosa?** `scipy.io.wavfile.read` returns the raw ndarray directly (no resampling, no conversion). `librosa.load` resamples to 22050 Hz by default — we want the raw sample rate for the offset calculation. `scipy` is already a dep and is more explicit here.

**Why a single-note MIDI and not the existing pipeline?** Isolation: we want to measure FluidSynth's latency, not confound it with the sample's musical content. A standard reference MIDI (always the same) gives a deterministic, reproducible measurement.

### Applying the offset

In `writer.write_sample` (after the annotation deep-copy, before path rewrite):
```python
beat_times = [t - offset_s for t in annotation["beat_times"] if t - offset_s >= 0.0]
downbeat_times = [t - offset_s for t in annotation["downbeat_times"] if t - offset_s >= 0.0]
```
Times that would become negative (very early beats before the pre-roll ends) are dropped. This is correct: the audio starts at `offset_s`, so anything before that point has no audio counterpart.

The `pre_roll_offset_seconds` field in `sample.json` stores the raw measured value (what was subtracted), for auditability.

---

## 2. ProcessPoolExecutor: Multiprocessing Gotchas on Linux

### Fork vs. Spawn

Python's default `multiprocessing` start method on Linux is `fork`. **Fork is unsafe** for this project because:

1. **FluidSynth subprocesses**: `renderer.py` uses `midi2audio.FluidSynth.midi_to_audio` which spawns a FluidSynth subprocess. Forking a process that has already spawned a subprocess can leave inconsistent state in the child.
2. **pedalboard threading**: `pedalboard` (the FX library in `mixer.py`) uses internal C++ threads. Forking after pedalboard is initialized can deadlock in the child process.
3. **threading.Lock in manifest.py**: `ManifestWriter` holds a `threading.Lock`. Forking a process that holds a lock in another thread causes deadlock in the child.

**Solution:** `multiprocessing.get_context("spawn")` forces subprocess-style process creation (the child imports the module fresh, no shared parent state). This is the default on macOS since Python 3.8 and on Windows. For Linux it must be explicit.

```python
mp_ctx = multiprocessing.get_context("spawn")
executor = ProcessPoolExecutor(max_workers=N, mp_context=mp_ctx)
```

**Cost:** spawn is slower than fork for startup (~0.3–1s per worker process). At 10k samples with 4 workers, the 4 processes start once and are reused — the startup cost is paid once, not per sample. Negligible at our scale.

### Config pickling

`Config` is a `@dataclass` with only stdlib-typed fields (str, int, float, bool, dict, tuple, Optional). Python's default pickle protocol handles dataclasses natively. **No custom `__reduce__` or `__getstate__` needed.**

One gotcha: if `Config` had any field containing a non-picklable object (like a `threading.Lock`), pickle would fail. Our Phase 6 design specifically avoids passing locks through Config (D-58 decision: main-process-only manifest writes via `_NullManifestWriter`).

Verification:
```python
import pickle
from config import Config
cfg = Config(global_seed=42, sample_index=0)
assert pickle.loads(pickle.dumps(cfg)) == cfg  # must pass
```

### Worker function must be at module level

`ProcessPoolExecutor.submit(fn, *args)` pickles `fn` by name. Lambda functions and inner functions (closures) are NOT picklable because they have no module-level name. The worker function `_worker` in `batch.py` must be defined at module level.

```python
# CORRECT: module-level function (picklable)
def _worker(global_seed, sample_index, config):
    ...
    return generate(...)

# INCORRECT: inner function (not picklable)
def generate_batch(config):
    def _worker(sample_index):  # closure — cannot be pickled
        ...
    executor.submit(_worker, idx)  # FAILS with pickle error
```

### `as_completed` for progress

```python
futures_map = {
    executor.submit(_worker, global_seed, idx, config): idx
    for idx in range(count)
    if not ManifestWriter.is_sample_complete(dataset_root, idx)
}
for future in as_completed(futures_map):
    idx = futures_map[future]
    try:
        result = future.result()
    except Exception as exc:
        # Handle worker crash
```

`as_completed` yields futures in completion order (not submission order). This is correct for progress reporting — we want to log results as they arrive.

### Re-raising exceptions from workers

`future.result()` re-raises any exception the worker raised. In our design, `_worker` calls `generate()` which has a broad try/except and returns a `SampleResult(status="failed")` rather than raising. So `future.result()` should rarely raise. The `except Exception` block in `as_completed` is defense against unexpected crashes (e.g., OOM in the child, serialization errors).

---

## 3. Typer CLI Patterns

### App structure

```python
import typer
app = typer.Typer(help="...", no_args_is_help=True)

@app.command()
def generate(...) -> None:
    ...

@app.command()
def clean(...) -> None:
    ...

@app.command()
def calibrate(...) -> None:
    ...
```

`no_args_is_help=True` means `musicgen` (with no args) prints help instead of running anything — idiomatic for a multi-command CLI.

### Required vs. Optional parameters

- `seed: int = typer.Option(..., "--seed", "-s", help="...")` — `...` (Ellipsis) makes it required. Typer exits with an error if it's not provided.
- `workers: Optional[int] = typer.Option(None, "--workers", ...)` — `None` default makes it optional.

### CliRunner for tests

```python
from typer.testing import CliRunner
from musicgen.cli import app

runner = CliRunner()

def test_generate_exits_zero(tmp_path):
    result = runner.invoke(app, ["generate", "--count", "1", "--out", str(tmp_path), "--seed", "42"])
    assert result.exit_code == 0
```

`CliRunner.invoke` captures stdout/stderr and does not actually exit the process — safe to call in tests. Does NOT fork; runs the command in-process. This means the CLI test does not need FluidSynth if `generate_batch` is mocked.

### Exit codes

- `raise typer.Exit(code=0)` — explicit success
- `raise typer.Exit(code=1)` — explicit failure (used when `result.failed > 0`)
- Typer returns exit code 0 if the function returns normally
- Exception in the function → exit code 1 (typer default)

---

## 4. OutputMode: Filtering Without Changing the Pipeline

### Why filter in writer, not annotator

The annotator is a pure function (Phase 4 D-14): `(SongParams, render_results, ...) -> dict`. It has no knowledge of the filesystem or output mode. Keeping it pure preserves its unit-testability without filesystem setup.

The writer already owns all filesystem writes. Adding an `output_mode` parameter to `write_sample` keeps the filter logic in one place.

### stem_paths / midi_paths when omitted

When `output_mode = "mix-only"`, the writer does not write `stems/` or `midi/`. The `sample.json` `stems` and `midi` keys become empty dicts `{}`. This is the correct behavior per D-66/D-67: `sample.json` reflects what was ACTUALLY written, not what could be written.

Downstream consumers (ML loaders) should handle `sample.json["stems"] == {}` gracefully. The `mix` key is always present when `output_mode` includes the mix.

### Sum-of-stems assertion guard

The sum-of-stems assertion (`_assert_sum_of_stems`) requires BOTH the stems AND the mix to be present. It should only run when `output_mode in ("full",)` (both stems and mix are written). For `stems-only` (no mix), `mix-only` (no stems), and `midi-only` (neither), the assertion is skipped. This is correct per the assertion's purpose: verify that the mix equals the sum of stems. If we only write stems or only write the mix, the assertion cannot be computed.

### Performance

For `mix-only` mode, the full pipeline still runs (renderer renders all stems, mixer applies FX to all layers). The stems are not persisted to the dataset dir but ARE used to assemble the mix. This is by design: the mix is always derived from all stems regardless of output mode; we just choose not to write the intermediate stem files to the dataset.

An optimization that writes only specific stems (skipping unused renders) is a v0.2+ concern that would require changes to the renderer and mixer — out of scope for Phase 6.

---

## 5. Manifest Safety Under Multiprocessing

### The problem

`manifest.jsonl` is opened in append mode by each `generate()` call. In a `ProcessPoolExecutor`, multiple workers can call `generate()` concurrently. Each worker opens the same file in `"a"` mode. On POSIX systems, writes of size ≤ `PIPE_BUF` (4096 bytes) to a file opened in append mode are **atomic** at the OS level (guaranteed by POSIX). Our manifest lines are ~200 bytes — well under `PIPE_BUF`.

**However**: Python's `json.dumps` + `write` + `flush` + `os.fsync` is a multi-step sequence. Between `write` and `fsync`, another process could write to the file. The actual file content is still correct (POSIX append atomicity), but `os.fsync` in one process may block on another process's write. This is safe — it just means one fsync waits for another to complete.

**Chosen solution (D-58):** Main-process-only manifest writes. Workers use `_NullManifestWriter` (discards appends). Main process writes after each `future.result()` in `as_completed`. This completely sidesteps the concurrent-write concern at the cost of: (a) manifest entries being written in completion order (not submission order), and (b) if the main process crashes between `future.result()` and the manifest append, that sample's entry is lost from the manifest (the sample.json sentinel exists, so resume still works — the sample is complete, just not recorded in the manifest).

**Acceptable for v0.1 at 10k scale.** A crash-safe manifest writer (write-ahead log + compaction) is v0.2+ scope.
