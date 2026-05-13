"""Determinism regression test (D-28/D-29/D-30/D-41, R-P8/R-Q3).

Two classes:

  * :class:`TestDeterminismGoldens` — @pytest.mark.slow, parametrized over
    6 artifacts. Under ``pytest --regen-goldens`` writes SHA-256 fixtures;
    otherwise asserts against them. MIDI + ``sample.json`` goldens must
    pass unconditionally; ``mix.wav`` golden is gated by ``fluidsynth
    --version`` matching ``fixtures/determinism/fluidsynth_version.txt``
    (R-P8 "bit-identical WAV only under pinned binary").

  * :class:`TestSameProcessStability` — fast, no FluidSynth. Runs
    ``generate`` twice in one process with monkeypatched renderer +
    musicality stubs; hashes both ``sample.json`` files; asserts
    byte-identity. Catches wall-clock/entropy leaks independent of
    FluidSynth (D-30).

To capture or regenerate goldens on a pinned-FluidSynth host::

    .venv/bin/pytest -m slow --regen-goldens tests/test_determinism_golden.py

After regeneration, commit ``tests/fixtures/determinism/*.sha256`` and
``tests/fixtures/determinism/fluidsynth_version.txt``.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures" / "determinism"
_LAYERS = ("beat", "melody", "harmony", "bassline")

# Artifact-name → attribute-on-SampleResult resolution map.
_ARTIFACTS = ("mix", "midi_beat", "midi_melody", "midi_harmony", "midi_bassline", "sample")


# ---------------- Helpers ----------------


def _sha256_of(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _current_fluidsynth_version_line() -> str:
    """First line of ``fluidsynth --version`` stdout (or stderr if empty)."""
    try:
        result = subprocess.run(
            ["fluidsynth", "--version"],
            capture_output=True, text=True, timeout=5,
        )
        output = result.stdout if result.stdout.strip() else result.stderr
        lines = output.splitlines()
        return lines[0] if lines else ""
    except Exception:
        return ""


def _fluidsynth_version_matches_golden() -> bool:
    """True iff fluidsynth --version first line == fluidsynth_version.txt content."""
    version_file = FIXTURES / "fluidsynth_version.txt"
    if not version_file.exists():
        return False
    expected = version_file.read_text().strip()
    actual = _current_fluidsynth_version_line().strip()
    return actual == expected


def _artifact_path(result, artifact: str) -> str:
    """Resolve artifact name to a path attribute on the SampleResult."""
    if artifact == "mix":
        return result.mix_path
    if artifact == "sample":
        return result.sample_json_path
    if artifact.startswith("midi_"):
        layer = artifact.removeprefix("midi_")
        return result.midi_paths[layer]
    raise ValueError(f"unknown artifact {artifact!r}")


# ---------------- Skip gates (reused from test_integration_full_generation) ----------------


fluidsynth_available = shutil.which("fluidsynth") is not None


def _all_sf2_layers_have_files() -> bool:
    try:
        import config as _cfg_mod
        _cfg = _cfg_mod.Config()
        for layer in _LAYERS:
            sf_dir = _cfg.sf_layer_dir(layer)
            if not os.path.isdir(sf_dir):
                return False
            files = [f for f in os.listdir(sf_dir) if f.endswith(".sf2")]
            if not files:
                return False
        return True
    except Exception:
        return False


sf2_pool_ready = _all_sf2_layers_have_files()


# ============================================================
# TestDeterminismGoldens — slow, FluidSynth-bound
# ============================================================


@pytest.mark.slow
@pytest.mark.skipif(
    not fluidsynth_available,
    reason="fluidsynth binary not on PATH — skipping determinism goldens",
)
@pytest.mark.skipif(
    not sf2_pool_ready,
    reason="one or more sf/<layer>/ dirs is empty — skipping determinism goldens",
)
class TestDeterminismGoldens:
    """D-29: parametrized SHA-256 goldens, --regen-goldens flag."""

    @pytest.mark.parametrize("artifact", _ARTIFACTS)
    def test_sha256_matches_golden(self, request, tmp_path, artifact):
        """For each artifact: compute SHA-256 and compare against fixture."""
        from musicgen import Config, generate

        # Run generate with the canonical golden seed + sample_index.
        result = generate(Config(
            global_seed=1, sample_index=0, dataset_root=str(tmp_path),
        ))
        assert result.status == "ok", (
            f"generate() failed: cannot compute goldens against a failed run "
            f"(status={result.status}, seed={result.seed})"
        )

        path = _artifact_path(result, artifact)
        actual = _sha256_of(path)
        golden_path = FIXTURES / f"expected_{artifact}.sha256"

        regen = request.config.getoption("--regen-goldens")

        if regen:
            # Capture: write hash + (on first artifact only) fluidsynth_version.
            FIXTURES.mkdir(parents=True, exist_ok=True)
            golden_path.write_text(actual + "\n")
            if artifact == "mix":
                fv = _current_fluidsynth_version_line()
                (FIXTURES / "fluidsynth_version.txt").write_text(fv + "\n")
            return  # --regen-goldens → test always passes after writing

        # Assert mode.
        if not golden_path.exists():
            pytest.skip(
                f"Golden expected_{artifact}.sha256 not captured yet. "
                f"Run `.venv/bin/pytest -m slow --regen-goldens "
                f"tests/test_determinism_golden.py` on a pinned-FluidSynth host."
            )

        expected = golden_path.read_text().strip()

        # FluidSynth version gate — only applies to the mix artifact.
        if artifact == "mix" and not _fluidsynth_version_matches_golden():
            pytest.xfail(
                f"fluidsynth version mismatch — expected "
                f"{(FIXTURES / 'fluidsynth_version.txt').read_text().strip()!r}, "
                f"got {_current_fluidsynth_version_line()!r}. Regenerate goldens "
                f"with --regen-goldens or install the pinned binary. "
                f"(R-P8: WAV bit-identity only under pinned FluidSynth.)"
            )

        assert actual == expected, (
            f"{artifact} hash mismatch — expected {expected!r}, got {actual!r}. "
            f"This indicates a non-determinism regression. If intentional, "
            f"run with --regen-goldens."
        )


# ============================================================
# TestSameProcessStability — fast, no FluidSynth (D-30)
# ============================================================


class TestSameProcessStability:
    """D-30: byte-stable sample.json across two in-process generate() calls.

    Catches non-determinism in OUR code (datetime.now, os.urandom, etc.)
    without depending on FluidSynth or soundfonts. Monkeypatches renderer
    and musicality to return deterministic stubs.
    """

    def test_generate_sample_json_stable_same_process(self, tmp_path, monkeypatch):
        """Two back-to-back generate() calls produce byte-identical sample.json."""
        from musicgen import Config, generate

        # -------- Monkeypatch renderer.render_stems --------
        # Goal: produce deterministic silent WAVs matching the shape the
        # rest of the pipeline expects (stereo 44.1kHz).
        from pydub import AudioSegment

        def _fake_render_stems(midi_paths, soundfonts, out_dir, cfg=None):
            from musicgen.renderer import RenderResult
            os.makedirs(out_dir, exist_ok=True)
            stems_dir = os.path.join(out_dir, "stems")
            os.makedirs(stems_dir, exist_ok=True)
            stem_paths = {}
            for layer in _LAYERS:
                wav_path = os.path.join(stems_dir, f"{layer}.wav")
                AudioSegment.silent(
                    duration=500, frame_rate=44100,
                ).set_channels(2).export(wav_path, format="wav")
                stem_paths[layer] = wav_path
            return RenderResult(
                stem_paths=stem_paths, sample_rate=44100, channels=2,
                duration_seconds=0.5, fluidsynth_version="stub",
            )

        monkeypatch.setattr("musicgen.api.renderer.render_stems", _fake_render_stems)

        # -------- Monkeypatch renderer.pick_soundfonts --------
        # Rule 3 blocking fix: the plan's spec missed this — pick_soundfonts
        # is called BEFORE render_stems and raises FileNotFoundError when the
        # sf/<layer>/ dirs are empty (dev-machine default). Stub must still
        # draw from rng[RNG_SOUNDFONTS] to keep the RNG draw order identical
        # to the real pipeline (D-19).
        def _fake_pick_soundfonts(cfg=None, rng=None, genre_spec=None):
            if rng is None:
                raise ValueError("pick_soundfonts requires an injected rng (D-17)")
            # One rng.choice per layer matching real pick_soundfonts draw count.
            candidates = ["stub_a.sf2", "stub_b.sf2", "stub_c.sf2"]
            return {layer: f"/stub/{rng.choice(candidates)}" for layer in _LAYERS}

        monkeypatch.setattr("musicgen.api.renderer.pick_soundfonts", _fake_pick_soundfonts)

        # -------- Monkeypatch musicality.get_musicality_score --------
        # Return deterministic score + components.
        def _fake_musicality(wav_path):
            return (0.5, {"rhythm": 0.5, "harmony": 0.5})

        monkeypatch.setattr("musicgen.api.musicality.get_musicality_score", _fake_musicality)

        # -------- Run generate twice with distinct dataset_roots --------
        cfg_a = Config(global_seed=1, sample_index=0, dataset_root=str(tmp_path / "a"))
        cfg_b = Config(global_seed=1, sample_index=0, dataset_root=str(tmp_path / "b"))

        r1 = generate(cfg_a)
        r2 = generate(cfg_b)

        # If pipeline failed for any reason, test cannot prove stability.
        assert r1.status == "ok", f"first generate() failed: {r1.status}"
        assert r2.status == "ok", f"second generate() failed: {r2.status}"

        # -------- Hash both sample.json files and assert equality --------
        hash_a = hashlib.sha256(Path(r1.sample_json_path).read_bytes()).hexdigest()
        hash_b = hashlib.sha256(Path(r2.sample_json_path).read_bytes()).hexdigest()

        assert hash_a == hash_b, (
            f"sample.json byte-stability broken in-process: "
            f"{hash_a!r} != {hash_b!r}. "
            f"Check for wall-clock ({'datetime.now'!r}), entropy "
            f"({'os.urandom'!r}, unseeded random), or non-stable iteration "
            f"order leaks in annotator/writer."
        )
