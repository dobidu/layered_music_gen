#!/usr/bin/env python3
"""
musicgen local verification script — v0.3.0

Run from anywhere inside the repo — no venv activation needed:
    python3 scripts/verify_local.py [--out path/to/report.json]

The script auto-detects the .venv at the repo root and re-executes itself
with the venv Python if the current interpreter doesn't have musicgen installed.

Writes two files when done:
    local_test_report.json   — machine-readable (share with Claude for analysis)
    local_test_report.md     — human-readable summary

No FluidSynth required for most checks. The script auto-detects capabilities
and skips sections that need FluidSynth or .sf2 files.
"""
import os
import sys
from pathlib import Path as _Path

# ---------------------------------------------------------------------------
# Auto-reexec with venv Python before any imports that might fail.
# Strategy: compare RAW paths (not resolved symlinks). Python uses the
# executable path to locate pyvenv.cfg and populate site-packages —
# running .venv/bin/python3 picks up venv packages even when it symlinks
# to the same underlying interpreter as /usr/bin/python3.
# ---------------------------------------------------------------------------
_SCRIPT_ROOT = _Path(__file__).resolve().parent.parent
_VENV_PYTHON = _SCRIPT_ROOT / ".venv" / "bin" / "python3"
_RUNNING_INSIDE_VENV = str(_VENV_PYTHON) == sys.executable or \
    os.environ.get("VIRTUAL_ENV") == str(_SCRIPT_ROOT / ".venv")

if not _RUNNING_INSIDE_VENV:
    if _VENV_PYTHON.is_file():
        print(f"[auto] re-executing with venv Python: {_VENV_PYTHON}")
        os.execv(str(_VENV_PYTHON), [str(_VENV_PYTHON)] + sys.argv)
        # execv replaces the current process — nothing below runs
    else:
        print("ERROR: .venv not found. Run these commands from the repo root first:")
        print(f"  python3 -m venv .venv")
        print(f"  .venv/bin/pip install -e '.[dev]'")
        sys.exit(1)

import argparse
import dataclasses
import hashlib
import json
import shutil
import struct
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Report datastructure
# ---------------------------------------------------------------------------

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"


def _result(status: str, **kwargs) -> dict:
    return {"status": status, **kwargs}


# ---------------------------------------------------------------------------
# Environment detection
# ---------------------------------------------------------------------------

def detect_environment() -> dict:
    env = {}
    env["python_version"] = sys.version.split()[0]
    env["platform"] = sys.platform

    try:
        import musicgen
        env["musicgen_version"] = musicgen.__version__
    except Exception as e:
        env["musicgen_version"] = f"ERROR: {e}"

    env["fluidsynth_available"] = shutil.which("fluidsynth") is not None
    env["ffmpeg_available"] = shutil.which("ffmpeg") is not None

    # Check sf2 pool
    sf2_ok = True
    sf2_detail = {}
    try:
        import config as cfg_mod
        cfg = cfg_mod.Config()
        for layer in ("beat", "melody", "harmony", "bassline"):
            sf_dir = cfg.sf_layer_dir(layer)
            files = [f for f in os.listdir(sf_dir) if f.endswith(".sf2")] if os.path.isdir(sf_dir) else []
            sf2_detail[layer] = len(files)
            if not files:
                sf2_ok = False
    except Exception as e:
        sf2_ok = False
        sf2_detail["error"] = str(e)

    env["sf2_pools_ready"] = sf2_ok
    env["sf2_pool_counts"] = sf2_detail

    env["full_pipeline_available"] = env["fluidsynth_available"] and sf2_ok and env["ffmpeg_available"]
    return env


# ---------------------------------------------------------------------------
# Section 1: Fast pytest suite
# ---------------------------------------------------------------------------

def run_fast_tests() -> dict:
    print("\n[1/7] Fast test suite (pytest -m 'not slow') ...")
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-m", "not slow", "-q", "--tb=short"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    duration = time.time() - t0
    stdout = proc.stdout + proc.stderr

    # Parse pytest summary line
    passed = failed = errors = warnings = 0
    for line in stdout.splitlines():
        if " passed" in line or " failed" in line or " error" in line:
            import re
            m = re.findall(r"(\d+) (passed|failed|error)", line)
            for count, label in m:
                if label == "passed":
                    passed = int(count)
                elif label == "failed":
                    failed = int(count)
                elif label == "error":
                    errors = int(count)

    status = PASS if proc.returncode == 0 else FAIL
    print(f"    {status}: {passed} passed, {failed} failed, {errors} errors in {duration:.1f}s")
    if proc.returncode != 0:
        # Print last 20 lines of output for context
        lines = stdout.strip().splitlines()
        for line in lines[-20:]:
            print(f"    {line}")

    return _result(
        status,
        passed=passed,
        failed=failed,
        errors=errors,
        duration_s=round(duration, 2),
        returncode=proc.returncode,
    )


# ---------------------------------------------------------------------------
# Section 2: Calibration harness
# ---------------------------------------------------------------------------

def run_calibration_check() -> dict:
    print("\n[2/7] Calibration harness (Layer 1 only, no FluidSynth) ...")
    try:
        from musicgen.calibrate import run_midi_calibration, suggest_threshold

        with tempfile.TemporaryDirectory() as tmp:
            result = run_midi_calibration(n_good=20, n_bad=20, seed=42, tmp_dir=tmp)

        good_mean = round(result.good_mean, 4)
        bad_mean = round(result.bad_mean, 4)
        threshold = round(result.suggested_threshold, 4)
        separation_ok = result.separation_ok

        checks = {
            "good_mean_above_bad_mean": result.good_mean > result.bad_mean,
            "good_mean_above_0.5": result.good_mean > 0.5,
            "bad_mean_below_0.3": result.bad_mean < 0.3,
            "separation_ok": separation_ok,
            "threshold_in_unit_interval": 0.0 <= threshold <= 1.0,
            "threshold_between_bad_and_good": result.bad_mean < threshold < result.good_mean,
        }
        all_pass = all(checks.values())
        status = PASS if all_pass else FAIL
        print(f"    {status}: good_mean={good_mean}, bad_mean={bad_mean}, threshold={threshold}, separation_ok={separation_ok}")
        if not all_pass:
            for k, v in checks.items():
                if not v:
                    print(f"    FAIL check: {k}")

        return _result(
            status,
            good_mean=good_mean,
            bad_mean=bad_mean,
            suggested_threshold=threshold,
            separation_ok=separation_ok,
            good_scores=result.good_scores,
            bad_scores=result.bad_scores,
            checks=checks,
        )
    except Exception as e:
        print(f"    FAIL: {e}")
        return _result(FAIL, error=str(e))


# ---------------------------------------------------------------------------
# Section 3: Layer 1 hard checks
# ---------------------------------------------------------------------------

def _write_minimal_midi(path: str, notes: list, ticks_per_beat: int = 480) -> None:
    """Write a minimal SMF type-0 MIDI file."""
    import struct

    tempo = 500000  # 120 BPM
    events = [(0, bytes([0xFF, 0x51, 0x03]) + tempo.to_bytes(3, "big"))]
    for pitch, start, dur, vel in notes:
        events.append((start, bytes([0x90, pitch, vel])))
        events.append((start + dur, bytes([0x80, pitch, 0])))
    events.append((ticks_per_beat * 8, bytes([0xFF, 0x2F, 0x00])))
    events.sort(key=lambda e: e[0])

    def var_len(n):
        r = [n & 0x7F]
        n >>= 7
        while n:
            r.insert(0, (n & 0x7F) | 0x80)
            n >>= 7
        return bytes(r)

    track = b""
    prev = 0
    for tick, msg in events:
        delta = tick - prev
        prev = tick
        track += var_len(delta) + msg

    header = struct.pack(">4sIHHH", b"MThd", 6, 0, 1, ticks_per_beat)
    trk = struct.pack(">4sI", b"MTrk", len(track)) + track
    Path(path).write_bytes(header + trk)


def run_layer1_hard_checks() -> dict:
    print("\n[3/7] Layer 1 hard checks ...")
    try:
        from musicgen.musicality import check_midi_quality
        import random

        checks = {}
        filler = [(60 + i % 5, i * 240, 200, 80) for i in range(16)]

        with tempfile.TemporaryDirectory() as tmp:
            def make_paths(name, melody_notes):
                paths = {}
                for layer in ("beat", "harmony", "bassline"):
                    p = os.path.join(tmp, f"{name}-{layer}.mid")
                    _write_minimal_midi(p, filler)
                    paths[layer] = p
                p = os.path.join(tmp, f"{name}-melody.mid")
                _write_minimal_midi(p, melody_notes)
                paths["melody"] = p
                return paths

            # 3a: empty melody
            paths = make_paths("empty", [])
            r = check_midi_quality(paths, key="C")
            checks["empty_layer_caught"] = not r.passed and r.score == 0.0
            checks["empty_hard_failure_message"] = any("empty" in f for f in r.hard_failures)
            print(f"    empty: passed={r.passed}, score={r.score}, failures={r.hard_failures}")

            # 3b: stuck note (15/16 same pitch)
            stuck = [(60, i * 240, 200, 80) for i in range(15)] + [(62, 15 * 240, 200, 80)]
            paths = make_paths("stuck", stuck)
            r = check_midi_quality(paths, key="C")
            checks["stuck_note_caught"] = not r.passed and r.score == 0.0
            checks["stuck_hard_failure_message"] = any("stuck" in f for f in r.hard_failures)
            print(f"    stuck: passed={r.passed}, score={r.score}, failures={r.hard_failures}")

            # 3c: extreme pitch range (> 36 semitones)
            extreme = [(36, 0, 200, 80), (84, 240, 200, 80)] * 8
            paths = make_paths("extreme", extreme)
            r = check_midi_quality(paths, key="C")
            checks["extreme_range_caught"] = not r.passed and r.score == 0.0
            checks["extreme_hard_failure_message"] = any("extreme" in f for f in r.hard_failures)
            print(f"    extreme: passed={r.passed}, score={r.score}, failures={r.hard_failures}")

            # 3d: good melody passes all hard checks
            good_notes = [(60 + (i % 8) * 2, i * 240, 200, 80) for i in range(16)]
            paths = make_paths("good", good_notes)
            r = check_midi_quality(paths, key="C")
            checks["good_melody_passes"] = r.passed
            checks["good_melody_score_positive"] = r.score > 0.0
            print(f"    good:   passed={r.passed}, score={round(r.score, 4)}, soft={r.soft_scores}")

        all_pass = all(checks.values())
        status = PASS if all_pass else FAIL
        print(f"    {status}: {sum(checks.values())}/{len(checks)} checks passed")
        if not all_pass:
            for k, v in checks.items():
                if not v:
                    print(f"    FAIL check: {k}")

        return _result(status, checks=checks)
    except Exception as e:
        print(f"    FAIL: {e}")
        import traceback
        return _result(FAIL, error=str(e), traceback=traceback.format_exc())


# ---------------------------------------------------------------------------
# Section 4: Layer 1 soft metrics
# ---------------------------------------------------------------------------

def run_layer1_soft_metrics() -> dict:
    print("\n[4/7] Layer 1 soft metrics (expected ranges) ...")
    try:
        from musicgen.musicality import (
            _ks_key_correlation,
            _scale_adherence_score,
            _melodic_step_fraction,
            _ngram_entropy,
            _lz_ratio,
        )
        import numpy as np

        checks = {}

        # C-major scale pitches — should score well on all soft metrics
        c_major = [60, 62, 64, 65, 67, 69, 71, 72, 71, 69, 67, 65, 64, 62, 60, 62] * 2

        pc_hist = np.zeros(12)
        for p in c_major:
            pc_hist[p % 12] += 1
        pc_hist /= pc_hist.sum()

        ks = _ks_key_correlation(pc_hist)
        checks["ks_c_major_high"] = ks > 0.7
        print(f"    KS correlation (C major):    {ks:.4f}  (expect > 0.7)")

        adh = _scale_adherence_score(c_major, "C")
        checks["scale_adherence_c_major_perfect"] = adh == 1.0
        print(f"    Scale adherence (C major):   {adh:.4f}  (expect 1.0)")

        step = _melodic_step_fraction(c_major)
        checks["step_fraction_c_major_high"] = step > 0.6
        print(f"    Melodic step fraction:       {step:.4f}  (expect > 0.6)")

        ent = _ngram_entropy(c_major, n=3)
        checks["ngram_entropy_in_range"] = 0.0 < ent <= 1.0
        print(f"    N-gram entropy (trigram):    {ent:.4f}  (expect in (0, 1])")

        lz = _lz_ratio(c_major)
        checks["lz_ratio_in_unit_interval"] = 0.0 <= lz <= 1.0
        print(f"    LZ compression ratio:        {lz:.4f}  (expect in [0, 1])")

        # Random pitches — should score worse on KS and scale adherence
        rng_pitches = [36 + (i * 7 + 3) % 60 for i in range(32)]
        pc_rand = np.zeros(12)
        for p in rng_pitches:
            pc_rand[p % 12] += 1
        pc_rand /= pc_rand.sum()

        ks_rand = _ks_key_correlation(pc_rand)
        checks["ks_random_lower_than_c_major"] = ks_rand < ks
        print(f"    KS correlation (random):     {ks_rand:.4f}  (expect < {ks:.4f})")

        # Stuck note — should have low entropy
        stuck = [60] * 15 + [62]
        ent_stuck = _ngram_entropy(stuck, n=3)
        checks["ngram_entropy_stuck_lower"] = ent_stuck < ent
        print(f"    N-gram entropy (stuck):      {ent_stuck:.4f}  (expect < {ent:.4f})")

        # Highly repetitive — should compress well (low lz_ratio)
        repetitive = [60, 62] * 32
        lz_rep = _lz_ratio(repetitive)
        checks["lz_ratio_repetitive_lower"] = lz_rep < lz
        print(f"    LZ ratio (repetitive):       {lz_rep:.4f}  (expect < {lz:.4f})")

        all_pass = all(checks.values())
        status = PASS if all_pass else FAIL
        print(f"    {status}: {sum(checks.values())}/{len(checks)} checks passed")
        if not all_pass:
            for k, v in checks.items():
                if not v:
                    print(f"    FAIL check: {k}")

        return _result(
            status,
            checks=checks,
            metric_values={
                "ks_c_major": round(ks, 4),
                "ks_random": round(ks_rand, 4),
                "scale_adherence_c_major": round(adh, 4),
                "step_fraction": round(step, 4),
                "ngram_entropy_c_major": round(ent, 4),
                "ngram_entropy_stuck": round(ent_stuck, 4),
                "lz_ratio_c_major": round(lz, 4),
                "lz_ratio_repetitive": round(lz_rep, 4),
            },
        )
    except Exception as e:
        print(f"    FAIL: {e}")
        import traceback
        return _result(FAIL, error=str(e), traceback=traceback.format_exc())


# ---------------------------------------------------------------------------
# Section 5: Full pipeline (requires FluidSynth + sf2)
# ---------------------------------------------------------------------------

def run_full_pipeline(env: dict) -> dict:
    if not env["full_pipeline_available"]:
        reason_parts = []
        if not env["fluidsynth_available"]:
            reason_parts.append("FluidSynth not on PATH")
        if not env["sf2_pools_ready"]:
            reason_parts.append("sf2 pool empty (" + ", ".join(
                f"{k}: {v} files" for k, v in env["sf2_pool_counts"].items()
                if isinstance(v, int) and v == 0
            ) + ")")
        if not env["ffmpeg_available"]:
            reason_parts.append("ffmpeg not on PATH")
        reason = "; ".join(reason_parts)
        print(f"\n[5/7] Full pipeline: SKIP ({reason})")
        return _result(SKIP, reason=reason)

    print("\n[5/7] Full pipeline: generate 2 samples ...")
    try:
        from musicgen import generate, Config

        with tempfile.TemporaryDirectory(prefix="musicgen-verify-") as tmp:
            # Generate sample 0
            t0 = time.time()
            r0 = generate(Config(global_seed=1, sample_index=0, dataset_root=tmp))
            dur0 = time.time() - t0
            print(f"    sample 0: status={r0.status}, attempt={r0.attempt}, "
                  f"score={round(r0.musicality_score, 4)}, duration={round(dur0, 1)}s")

            # Generate sample 1
            t0 = time.time()
            r1 = generate(Config(global_seed=1, sample_index=1, dataset_root=tmp))
            dur1 = time.time() - t0
            print(f"    sample 1: status={r1.status}, attempt={r1.attempt}, "
                  f"score={round(r1.musicality_score, 4)}, duration={round(dur1, 1)}s")

            checks = {}
            checks["sample0_status_ok"] = r0.status == "ok"
            checks["sample1_status_ok"] = r1.status == "ok"
            checks["sample0_attempt_is_1"] = r0.attempt == 1
            checks["sample1_attempt_is_1"] = r1.attempt == 1
            checks["sample0_score_in_unit_interval"] = 0.0 <= r0.musicality_score <= 1.0
            checks["sample1_score_in_unit_interval"] = 0.0 <= r1.musicality_score <= 1.0

            if r0.status != "ok":
                print(f"    FAIL: generate() returned status={r0.status!r} for sample 0")
                return _result(FAIL, error=f"generate() status={r0.status!r}", checks=checks)

            # Verify file layout for sample 0
            s0_dir = Path(r0.sample_dir)
            expected_files = [
                "sample.json",
                "mix.wav",
                "stems/beat.wav", "stems/melody.wav", "stems/harmony.wav", "stems/bassline.wav",
                "midi/beat.mid", "midi/melody.mid", "midi/harmony.mid", "midi/bassline.mid",
            ]
            layout_ok = all((s0_dir / f).is_file() for f in expected_files)
            checks["layout_10_files_present"] = layout_ok
            if not layout_ok:
                missing = [f for f in expected_files if not (s0_dir / f).is_file()]
                print(f"    missing files: {missing}")

            # sample.json fields
            import json as _json
            annotation = _json.loads((s0_dir / "sample.json").read_text())
            checks["sample_json_has_seed"] = "seed" in annotation
            checks["sample_json_has_split"] = annotation.get("split") in ("train", "valid", "test")
            checks["sample_json_has_musicality_score"] = "musicality_score" in annotation
            checks["sample_json_has_version"] = "musicgen_version" in annotation
            print(f"    sample.json: seed={annotation.get('seed')}, split={annotation.get('split')}, "
                  f"version={annotation.get('musicgen_version')}")

            # manifest.jsonl
            manifest_path = Path(tmp) / "manifest.jsonl"
            checks["manifest_exists"] = manifest_path.is_file()
            if manifest_path.is_file():
                entries = [_json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
                ok_entries = [e for e in entries if e.get("status") == "ok"]
                checks["manifest_has_2_ok_entries"] = len(ok_entries) == 2
                checks["manifest_entries_have_attempt"] = all("attempt" in e for e in ok_entries)
                print(f"    manifest: {len(entries)} entries, {len(ok_entries)} ok, "
                      f"attempt fields: {[e.get('attempt') for e in ok_entries]}")

            all_pass = all(checks.values())
            status = PASS if all_pass else FAIL
            print(f"    {status}: {sum(checks.values())}/{len(checks)} checks passed")

            return _result(
                status,
                checks=checks,
                sample0={"status": r0.status, "attempt": r0.attempt,
                          "musicality_score": round(r0.musicality_score, 4),
                          "duration_s": round(dur0, 2)},
                sample1={"status": r1.status, "attempt": r1.attempt,
                          "musicality_score": round(r1.musicality_score, 4),
                          "duration_s": round(dur1, 2)},
                annotation_sample={k: annotation.get(k) for k in
                                    ("seed", "split", "tempo_bpm", "key", "time_signature",
                                     "musicgen_version", "fluidsynth_version")},
            )
    except Exception as e:
        print(f"    FAIL: {e}")
        import traceback
        return _result(FAIL, error=str(e), traceback=traceback.format_exc())


# ---------------------------------------------------------------------------
# Section 6: Determinism (requires FluidSynth + sf2)
# ---------------------------------------------------------------------------

def run_determinism_check(env: dict) -> dict:
    if not env["full_pipeline_available"]:
        print(f"\n[6/7] Determinism check: SKIP (FluidSynth/sf2/ffmpeg not available)")
        return _result(SKIP, reason="full pipeline not available")

    print("\n[6/7] Determinism check ...")
    try:
        from musicgen import generate, Config

        import json as _json

        with tempfile.TemporaryDirectory(prefix="musicgen-det-a-") as tmp_a, \
             tempfile.TemporaryDirectory(prefix="musicgen-det-b-") as tmp_b:

            cfg_a = Config(global_seed=42, sample_index=7, dataset_root=tmp_a)
            cfg_b = Config(global_seed=42, sample_index=7, dataset_root=tmp_b)

            r_a = generate(cfg_a)
            r_b = generate(cfg_b)

            checks = {}
            checks["both_ok"] = r_a.status == "ok" and r_b.status == "ok"

            if not checks["both_ok"]:
                failed = [f"r_a={r_a.status!r}", f"r_b={r_b.status!r}"]
                print(f"    FAIL: generate() non-ok — {', '.join(failed)}")
                return _result(FAIL, error=f"generate() status non-ok: {failed}", checks=checks)

            def sha256(path):
                h = hashlib.sha256()
                h.update(Path(path).read_bytes())
                return h.hexdigest()

            # MIDI bit-identity
            for layer in ("beat", "melody", "harmony", "bassline"):
                mid_a = str(Path(r_a.sample_dir) / "midi" / f"{layer}.mid")
                mid_b = str(Path(r_b.sample_dir) / "midi" / f"{layer}.mid")
                match = sha256(mid_a) == sha256(mid_b)
                checks[f"midi_{layer}_bit_identical"] = match
                print(f"    midi/{layer}.mid: {'MATCH' if match else 'MISMATCH'}")

            # sample.json bit-identity
            json_a = (Path(r_a.sample_dir) / "sample.json").read_text()
            json_b = (Path(r_b.sample_dir) / "sample.json").read_text()
            json_match = json_a == json_b
            checks["sample_json_bit_identical"] = json_match
            print(f"    sample.json:  {'MATCH' if json_match else 'MISMATCH'}")

            # seeds match
            checks["seeds_match"] = r_a.seed == r_b.seed
            checks["seeds_not_global_seed"] = r_a.seed != 42
            print(f"    seed: {r_a.seed} == {r_b.seed}: {'MATCH' if r_a.seed == r_b.seed else 'MISMATCH'}")

            all_pass = all(checks.values())
            status = PASS if all_pass else FAIL
            print(f"    {status}: {sum(checks.values())}/{len(checks)} checks passed")

            return _result(
                status,
                checks=checks,
                seed_a=r_a.seed,
                seed_b=r_b.seed,
                global_seed=42,
                sample_index=7,
            )
    except Exception as e:
        print(f"    FAIL: {e}")
        import traceback
        return _result(FAIL, error=str(e), traceback=traceback.format_exc())


# ---------------------------------------------------------------------------
# Section 7: Slow pytest (requires FluidSynth + sf2)
# ---------------------------------------------------------------------------

def run_slow_tests(env: dict) -> dict:
    if not env["full_pipeline_available"]:
        print(f"\n[7/7] Slow tests: SKIP (FluidSynth/sf2/ffmpeg not available)")
        return _result(SKIP, reason="full pipeline not available")

    print("\n[7/7] Slow tests (pytest -m slow) ...")
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-m", "slow", "-v", "--tb=short"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    duration = time.time() - t0
    stdout = proc.stdout + proc.stderr

    passed = failed = errors = 0
    import re
    for line in stdout.splitlines():
        if " passed" in line or " failed" in line or " error" in line:
            for count, label in re.findall(r"(\d+) (passed|failed|error)", line):
                if label == "passed":
                    passed = int(count)
                elif label == "failed":
                    failed = int(count)
                elif label == "error":
                    errors = int(count)

    status = PASS if proc.returncode == 0 else FAIL
    print(f"    {status}: {passed} passed, {failed} failed, {errors} errors in {duration:.1f}s")
    if proc.returncode != 0:
        lines = stdout.strip().splitlines()
        for line in lines[-30:]:
            print(f"    {line}")

    return _result(
        status,
        passed=passed,
        failed=failed,
        errors=errors,
        duration_s=round(duration, 2),
        returncode=proc.returncode,
    )


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------

def write_json_report(report: dict, path: Path) -> None:
    path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n    JSON report: {path}")


def write_markdown_report(report: dict, path: Path) -> None:
    env = report["environment"]
    results = report["results"]
    summary = report["summary"]

    lines = [
        "# musicgen local verification report",
        "",
        f"**Generated:** {report['timestamp']}",
        f"**musicgen version:** {env.get('musicgen_version', '?')}",
        f"**Python:** {env.get('python_version', '?')}  |  **Platform:** {env.get('platform', '?')}",
        f"**FluidSynth:** {'✓' if env['fluidsynth_available'] else '✗'}  |  "
        f"**ffmpeg:** {'✓' if env['ffmpeg_available'] else '✗'}  |  "
        f"**sf2 pools:** {'✓ all layers' if env['sf2_pools_ready'] else '✗ incomplete'}",
        "",
        "---",
        "",
        "## Summary",
        "",
        f"| Sections | Count |",
        f"|---|---|",
        f"| Passed | **{summary['passed']}** |",
        f"| Failed | **{summary['failed']}** |",
        f"| Skipped | {summary['skipped']} |",
        f"| Total | {summary['total']} |",
        "",
        "---",
        "",
        "## Section results",
        "",
    ]

    ICONS = {PASS: "✅", FAIL: "❌", SKIP: "⏭️"}

    section_names = {
        "fast_tests": "Fast test suite",
        "calibration": "Calibration harness",
        "layer1_hard_checks": "Layer 1 hard checks",
        "layer1_soft_metrics": "Layer 1 soft metrics",
        "full_pipeline": "Full pipeline (FluidSynth required)",
        "determinism": "Determinism check (FluidSynth required)",
        "slow_tests": "Slow test suite (FluidSynth required)",
    }

    for key, name in section_names.items():
        r = results.get(key, {})
        st = r.get("status", "?")
        icon = ICONS.get(st, "❓")
        lines.append(f"### {icon} {name}: {st}")
        lines.append("")

        if key == "fast_tests" and st != SKIP:
            lines.append(f"- Tests: {r.get('passed', 0)} passed, {r.get('failed', 0)} failed, "
                          f"{r.get('errors', 0)} errors")
            lines.append(f"- Duration: {r.get('duration_s', '?')}s")

        elif key == "calibration" and st not in (SKIP, FAIL):
            lines.append(f"- good_mean: `{r.get('good_mean')}`")
            lines.append(f"- bad_mean: `{r.get('bad_mean')}`")
            lines.append(f"- suggested_threshold: `{r.get('suggested_threshold')}`")
            lines.append(f"- separation_ok: `{r.get('separation_ok')}`")

        elif key in ("layer1_hard_checks", "layer1_soft_metrics") and st not in (SKIP,):
            checks = r.get("checks", {})
            for ck, cv in checks.items():
                lines.append(f"- {'✅' if cv else '❌'} `{ck}`")
            if key == "layer1_soft_metrics":
                vals = r.get("metric_values", {})
                if vals:
                    lines.append("")
                    lines.append("**Metric values:**")
                    for k, v in vals.items():
                        lines.append(f"- `{k}`: `{v}`")

        elif key == "full_pipeline" and st not in (SKIP,):
            for s_key in ("sample0", "sample1"):
                s = r.get(s_key, {})
                lines.append(f"- {s_key}: status=`{s.get('status')}`, attempt=`{s.get('attempt')}`, "
                              f"score=`{s.get('musicality_score')}`, duration=`{s.get('duration_s')}s`")
            ann = r.get("annotation_sample", {})
            if ann:
                lines.append(f"- annotation: tempo=`{ann.get('tempo_bpm')}` BPM, "
                              f"key=`{ann.get('key')}`, "
                              f"time_sig=`{ann.get('time_signature')}`, "
                              f"split=`{ann.get('split')}`")
            checks = r.get("checks", {})
            for ck, cv in checks.items():
                lines.append(f"- {'✅' if cv else '❌'} `{ck}`")

        elif key == "determinism" and st not in (SKIP,):
            lines.append(f"- global_seed=`{r.get('global_seed')}`, sample_index=`{r.get('sample_index')}`")
            lines.append(f"- derived seed: `{r.get('seed_a')}`")
            checks = r.get("checks", {})
            for ck, cv in checks.items():
                lines.append(f"- {'✅' if cv else '❌'} `{ck}`")

        elif key == "slow_tests" and st not in (SKIP,):
            lines.append(f"- Tests: {r.get('passed', 0)} passed, {r.get('failed', 0)} failed, "
                          f"{r.get('errors', 0)} errors")
            lines.append(f"- Duration: {r.get('duration_s', '?')}s")

        elif st == SKIP:
            lines.append(f"- Reason: {r.get('reason', 'N/A')}")

        elif st == FAIL and "error" in r:
            lines.append(f"- Error: `{r['error']}`")

        lines.append("")

    path.write_text("\n".join(lines))
    print(f"    Markdown report: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="musicgen local verification")
    parser.add_argument("--out", default=None, help="Output path prefix (default: <repo root>/local_test_report)")
    args = parser.parse_args()

    out_prefix = Path(args.out) if args.out else ROOT / "local_test_report"

    print("=" * 60)
    print("musicgen local verification — v0.3.0")
    print("=" * 60)

    import datetime
    timestamp = datetime.datetime.now().isoformat()

    print("\nDetecting environment ...")
    env = detect_environment()
    print(f"  musicgen version : {env['musicgen_version']}")
    print(f"  Python           : {env['python_version']}")
    print(f"  FluidSynth       : {'yes' if env['fluidsynth_available'] else 'NOT FOUND'}")
    print(f"  ffmpeg           : {'yes' if env['ffmpeg_available'] else 'NOT FOUND'}")
    print(f"  sf2 pools ready  : {'yes' if env['sf2_pools_ready'] else 'NO'} {env['sf2_pool_counts']}")
    print(f"  full pipeline    : {'AVAILABLE' if env['full_pipeline_available'] else 'UNAVAILABLE (will skip FluidSynth sections)'}")

    results = {}
    results["fast_tests"]         = run_fast_tests()
    results["calibration"]        = run_calibration_check()
    results["layer1_hard_checks"] = run_layer1_hard_checks()
    results["layer1_soft_metrics"]= run_layer1_soft_metrics()
    results["full_pipeline"]      = run_full_pipeline(env)
    results["determinism"]        = run_determinism_check(env)
    results["slow_tests"]         = run_slow_tests(env)

    counts = {"passed": 0, "failed": 0, "skipped": 0, "total": len(results)}
    for r in results.values():
        st = r.get("status", "?")
        if st == PASS:
            counts["passed"] += 1
        elif st == FAIL:
            counts["failed"] += 1
        else:
            counts["skipped"] += 1

    report = {
        "timestamp": timestamp,
        "environment": env,
        "results": results,
        "summary": counts,
    }

    overall = "ALL PASS" if counts["failed"] == 0 else f"{counts['failed']} SECTION(S) FAILED"
    print("\n" + "=" * 60)
    print(f"RESULT: {overall}")
    print(f"  {counts['passed']} passed  |  {counts['failed']} failed  |  {counts['skipped']} skipped")
    print("=" * 60)

    json_path = Path(str(out_prefix) + ".json")
    md_path = Path(str(out_prefix) + ".md")
    write_json_report(report, json_path)
    write_markdown_report(report, md_path)

    print("\nTo share with Claude for analysis, paste the contents of:")
    print(f"  {json_path}")

    sys.exit(0 if counts["failed"] == 0 else 1)


if __name__ == "__main__":
    main()
