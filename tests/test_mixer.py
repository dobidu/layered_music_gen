"""Mixer tests (R-X5): seeded-RNG determinism + silent-stem channel parity + D-11 FX-on-all-layers + R-S4 gain/pan preservation.

Uses in-memory AudioSegments + tmp_path-written WAVs as fixtures. Does not
require FluidSynth (renderer is mocked in tests/test_renderer.py; these tests
consume fake RenderResults directly).
"""
from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from unittest.mock import patch

import pytest
from pydub import AudioSegment

from musicgen.mixer import (
    _lin_to_db,
    _make_silent_stem,
    MixResult,
    apply_fx_to_layer,
    build_fx_boards,
    compute_layer_mask,
    concat_parts,
    mix_part,
    pedalboard_info_json,
)
from musicgen.renderer import RenderResult


# ---------- _lin_to_db (R-S4 preservation) ----------

class TestLinToDb:
    def test_unity(self):
        assert _lin_to_db(1.0) == 0.0

    def test_half(self):
        assert _lin_to_db(0.5) == pytest.approx(20 * math.log10(0.5), abs=1e-9)

    def test_clamp_zero_does_not_raise(self):
        result = _lin_to_db(0.0)
        assert result == pytest.approx(20 * math.log10(1e-6), abs=1e-6)
        assert result <= -100.0

    def test_clamp_negative_does_not_raise(self):
        # Negative input is clamped to 1e-6 — levels.json should not have negatives but be defensive
        _lin_to_db(-0.5)  # must not raise


# ---------- _make_silent_stem (D-12 + RESEARCH correction #2) ----------

class TestMakeSilentStem:
    def test_default_channels_2(self):
        """D-12 + RESEARCH correction #2: default must be stereo (channels=2)."""
        seg = _make_silent_stem(1000)
        assert seg.channels == 2, f"expected stereo, got channels={seg.channels}"

    def test_default_frame_rate_44100(self):
        """D-12 + RESEARCH correction #2: default frame rate must be 44100 Hz."""
        seg = _make_silent_stem(1000)
        assert seg.frame_rate == 44100, f"expected 44100 Hz, got {seg.frame_rate}"

    def test_duration(self):
        seg = _make_silent_stem(1500)
        assert seg.duration_seconds == pytest.approx(1.5, abs=0.01)

    def test_custom_sample_rate(self):
        seg = _make_silent_stem(1000, sample_rate=22050, channels=1)
        assert seg.frame_rate == 22050
        assert seg.channels == 1


# ---------- build_fx_boards (D-10/D-17) ----------

@pytest.fixture
def fake_fx_cfg(tmp_path):
    """Create 4 minimal FX JSON files and a config pointing at them.

    Uses probability=0.5 + tiny value_range to make the seeded-determinism
    contract meaningful — probability=1.0 would always produce the same
    pedalboard and mask the RNG-threading bug we're guarding against.
    """
    import config as config_module
    fx_spec = {
        "compressor": {"probability": 0.5, "value_range": {"threshold_db": [-30, -10], "ratio": [1.5, 4.0]}},
        "gain": {"probability": 0.5, "value_range": {"gain_db": [-6, 6]}},
        "chorus": {"probability": 0.5, "value_range": {"rate_hz": [0.5, 2.0], "depth": [0.1, 0.5]}},
        "ladder_filter": {"probability": 0.5, "value_range": {"cutoff_hz": [200, 2000]}},
        "phaser": {"probability": 0.5, "value_range": {"rate_hz": [0.1, 2.0]}},
        "delay": {"probability": 0.5, "value_range": {"delay_seconds": [0.1, 0.5], "feedback": [0.1, 0.5]}},
        "reverb": {"probability": 0.5, "value_range": {"room_size": [0.1, 0.9]}},
    }
    cfg = config_module.Config()
    fx_files = {}
    for layer in ("beat", "melody", "harmony", "bassline"):
        fx_path = tmp_path / f"{layer}_fx.json"
        with open(fx_path, "w") as f:
            json.dump(fx_spec, f)
        fx_files[layer] = str(fx_path)
    cfg.fx_files = fx_files
    return cfg


class TestBuildFxBoards:
    def test_returns_4_layer_dict(self, fake_fx_cfg):
        from pedalboard import Pedalboard
        boards = build_fx_boards(cfg=fake_fx_cfg, rng=random.Random(42))
        assert set(boards.keys()) == {"beat", "melody", "harmony", "bassline"}
        for layer, board in boards.items():
            assert isinstance(board, Pedalboard), f"layer {layer}: expected Pedalboard, got {type(board)}"

    @pytest.mark.parametrize("seed", [0, 42, 12345])
    def test_deterministic_same_seed(self, fake_fx_cfg, seed):
        """D-10/D-17: same seed → same pedalboard_info_json for all 4 layers."""
        a = build_fx_boards(cfg=fake_fx_cfg, rng=random.Random(seed))
        b = build_fx_boards(cfg=fake_fx_cfg, rng=random.Random(seed))
        for layer in ("beat", "melody", "harmony", "bassline"):
            assert pedalboard_info_json(a[layer]) == pedalboard_info_json(b[layer]), (
                f"seed={seed} layer={layer}: non-deterministic FX params"
            )

    def test_different_seeds_differ(self, fake_fx_cfg):
        """RNG is actually being used — very different seeds should produce different output."""
        a = build_fx_boards(cfg=fake_fx_cfg, rng=random.Random(0))
        b = build_fx_boards(cfg=fake_fx_cfg, rng=random.Random(9999))
        differ = any(
            pedalboard_info_json(a[layer]) != pedalboard_info_json(b[layer])
            for layer in ("beat", "melody", "harmony", "bassline")
        )
        assert differ, "expected different seeds to produce different FX boards on at least one layer"

    def test_requires_rng(self, fake_fx_cfg):
        """D-17 guard: build_fx_boards must raise ValueError when rng=None."""
        with pytest.raises(ValueError, match="rng"):
            build_fx_boards(cfg=fake_fx_cfg, rng=None)


# ---------- compute_layer_mask (D-13/D-17) ----------

class TestComputeLayerMask:
    def test_deterministic_same_seed(self):
        """D-13/D-17: same seed → identical layer mask."""
        proba = {p: {l: "0.5" for l in ("beat", "melody", "harmony", "bassline")} for p in ("intro", "verse")}
        a = compute_layer_mask(["intro", "verse"], proba, random.Random(42))
        b = compute_layer_mask(["intro", "verse"], proba, random.Random(42))
        assert a == b

    def test_structure(self):
        """Returns nested {part: {layer: bool}} with all 4 layer keys."""
        proba = {"intro": {l: "0.5" for l in ("beat", "melody", "harmony", "bassline")}}
        result = compute_layer_mask(["intro"], proba, random.Random(0))
        assert set(result.keys()) == {"intro"}
        assert set(result["intro"].keys()) == {"beat", "melody", "harmony", "bassline"}
        for v in result["intro"].values():
            assert isinstance(v, bool)

    def test_proba_1_all_true(self):
        """All-1.0 probabilities must yield every layer True."""
        proba = {"intro": {l: "1.0" for l in ("beat", "melody", "harmony", "bassline")}}
        result = compute_layer_mask(["intro"], proba, random.Random(42))
        assert all(result["intro"].values())

    def test_proba_0_all_false(self):
        """All-0.0 probabilities must yield every layer False."""
        proba = {"intro": {l: "0.0" for l in ("beat", "melody", "harmony", "bassline")}}
        result = compute_layer_mask(["intro"], proba, random.Random(42))
        assert not any(result["intro"].values())


# ---------- MixResult (D-02) ----------

class TestMixResult:
    def test_is_frozen(self):
        """D-02: MixResult must be a frozen dataclass — field assignment must raise."""
        mr = MixResult(mix_path="/x", stem_paths={}, part_layers={}, soundfonts={}, pedalboards={}, transitions=[])
        with pytest.raises((AttributeError, Exception)):
            mr.mix_path = "/y"  # type: ignore[misc]


# ---------- D-11 FX-on-all-layers regression guard ----------

@pytest.fixture
def fake_render_result(tmp_path):
    """Write 4 fake stereo-44.1kHz WAVs and build a RenderResult."""
    stem_paths = {}
    for layer in ("beat", "melody", "harmony", "bassline"):
        path = tmp_path / f"{layer}.wav"
        AudioSegment.silent(duration=500, frame_rate=44100).set_channels(2).export(str(path), format="wav")
        stem_paths[layer] = str(path)
    return RenderResult(
        stem_paths=stem_paths,
        sample_rate=44100,
        channels=2,
        duration_seconds=0.5,
        fluidsynth_version="test",
    )


@pytest.fixture
def fake_levels():
    return {
        p: {l: {"volume": 1.0, "panning": 0.0} for l in ("beat", "melody", "harmony", "bassline")}
        for p in ("intro", "verse", "chorus", "bridge", "outro")
    }


@pytest.fixture
def fake_fx_boards(fake_fx_cfg):
    return build_fx_boards(cfg=fake_fx_cfg, rng=random.Random(7))


class TestFxAppliedToAllLayers:
    def test_fx_applied_to_all_4_layers_regardless_of_mask(
        self, tmp_path, fake_render_result, fake_levels, fake_fx_boards
    ):
        """D-11 regression guard: apply_fx_to_layer MUST be called 4 times even
        when layer_mask is ALL False. The music_gen.py:276 TODO to optimize is
        OUT OF SCOPE — moving apply_fx inside the if-branch would change the
        RNG draw count and break the Phase 5 golden-seed baseline.
        """
        # Mask ALL layers off — should STILL trigger 4 apply_fx_to_layer calls.
        layer_mask_for_part = {l: False for l in ("beat", "melody", "harmony", "bassline")}

        call_counter = {"n": 0}
        original_apply_fx = apply_fx_to_layer

        def _counting_apply_fx(wav_file, board):
            call_counter["n"] += 1
            return original_apply_fx(wav_file, board)

        with patch("musicgen.mixer.apply_fx_to_layer", _counting_apply_fx):
            mix_part(
                render_result=fake_render_result,
                levels=fake_levels,
                fx_boards=fake_fx_boards,
                layer_mask_for_part=layer_mask_for_part,
                part="intro",
                out_dir=str(tmp_path / "mix"),
                soundfonts={l: f"/fake/{l}.sf2" for l in ("beat", "melody", "harmony", "bassline")},
            )

        assert call_counter["n"] == 4, (
            f"D-11 violation: apply_fx_to_layer called {call_counter['n']} times with all-False mask; "
            "expected 4. FX must be applied to ALL layers unconditionally."
        )


# ---------- R-S4 gain/pan preservation ----------

class TestApplyGainPanPreservation:
    def test_mix_part_completes_with_full_and_quiet_levels(
        self, tmp_path, fake_render_result, fake_fx_boards
    ):
        """R-S4: apply_gain(_lin_to_db(v)) must not raise and must route levels.json
        values through the gain path. Silent fake stems mean we can't assert on
        output RMS, but we can assert both mix WAVs exist (path correct) and no
        exception is raised.
        """
        levels_full = {
            "intro": {l: {"volume": 1.0, "panning": 0.0} for l in ("beat", "melody", "harmony", "bassline")},
        }
        levels_quiet = {
            "intro": {l: {"volume": 0.01, "panning": 0.0} for l in ("beat", "melody", "harmony", "bassline")},
        }
        layer_mask_on = {l: True for l in ("beat", "melody", "harmony", "bassline")}
        soundfonts = {l: f"/fake/{l}.sf2" for l in ("beat", "melody", "harmony", "bassline")}

        mr_full = mix_part(
            render_result=fake_render_result,
            levels=levels_full,
            fx_boards=fake_fx_boards,
            layer_mask_for_part=layer_mask_on,
            part="intro",
            out_dir=str(tmp_path / "mix_full"),
            soundfonts=soundfonts,
        )
        mr_quiet = mix_part(
            render_result=fake_render_result,
            levels=levels_quiet,
            fx_boards=fake_fx_boards,
            layer_mask_for_part=layer_mask_on,
            part="intro",
            out_dir=str(tmp_path / "mix_quiet"),
            soundfonts=soundfonts,
        )
        # Both should produce a mix WAV at the expected path.
        assert os.path.exists(mr_full.mix_path), "mix_part with volume=1.0 did not write output WAV"
        assert os.path.exists(mr_quiet.mix_path), "mix_part with volume=0.01 did not write output WAV"

    def test_apply_gain_call_count_in_source(self):
        """Static check: mixer.py's mix_part body must contain >= 4 apply_gain
        calls and >= 4 pan() calls (one per layer) — guards against someone
        'simplifying' the R-S4 fix back into a no-op (PITFALLS P-B regression).
        """
        src = Path(__file__).resolve().parent.parent / "src" / "musicgen" / "mixer.py"
        content = src.read_text()
        assert content.count("apply_gain(") >= 4, (
            f"R-S4 regression risk: expected >= 4 apply_gain( calls in mixer.py, "
            f"found {content.count('apply_gain(')}"
        )
        assert content.count(".pan(") >= 4, (
            f"R-S4 regression risk: expected >= 4 .pan( calls in mixer.py, "
            f"found {content.count('.pan(')}"
        )


# ---------- mix_part + MixResult shape ----------

class TestMixPart:
    def test_returns_mix_result(self, tmp_path, fake_render_result, fake_levels, fake_fx_boards):
        layer_mask = {l: True for l in ("beat", "melody", "harmony", "bassline")}
        soundfonts = {l: f"/fake/{l}.sf2" for l in ("beat", "melody", "harmony", "bassline")}
        result = mix_part(
            render_result=fake_render_result,
            levels=fake_levels,
            fx_boards=fake_fx_boards,
            layer_mask_for_part=layer_mask,
            part="intro",
            out_dir=str(tmp_path / "mix"),
            soundfonts=soundfonts,
        )
        assert isinstance(result, MixResult)
        assert os.path.exists(result.mix_path)
        assert set(result.stem_paths.keys()) == {"beat", "melody", "harmony", "bassline"}
        assert result.part_layers == layer_mask

    def test_silent_stem_for_masked_off_layer(self, tmp_path, fake_render_result, fake_levels, fake_fx_boards):
        """D-12: layers masked-off get a silent stem stub at stereo 44.1kHz (RESEARCH correction #2)."""
        layer_mask = {"beat": True, "melody": False, "harmony": False, "bassline": False}
        soundfonts = {l: f"/fake/{l}.sf2" for l in ("beat", "melody", "harmony", "bassline")}
        result = mix_part(
            render_result=fake_render_result,
            levels=fake_levels,
            fx_boards=fake_fx_boards,
            layer_mask_for_part=layer_mask,
            part="intro",
            out_dir=str(tmp_path / "mix"),
            soundfonts=soundfonts,
        )
        # All 4 stem paths exist; 3 of them are "_silent" stubs.
        # Check only the basename — the pytest tmp_path dirname may itself contain
        # "_silent" (from the test name) which would give false positives.
        silent_count = sum(
            1 for p in result.stem_paths.values()
            if "_silent" in os.path.basename(p)
        )
        assert silent_count == 3, (
            f"expected 3 silent-stem stubs for 3 masked-off layers, got {silent_count}"
        )
        # Silent stems are stereo 44.1kHz per RESEARCH correction #2.
        for layer, path in result.stem_paths.items():
            if "_silent" in os.path.basename(path):
                seg = AudioSegment.from_wav(path)
                assert seg.channels == 2, f"silent stem {layer} not stereo"
                assert seg.frame_rate == 44100, f"silent stem {layer} not 44100 Hz"


# ---------- concat_parts ----------

class TestConcatParts:
    def test_orders_correctly(self, tmp_path):
        """Two parts concatenate in order; output duration == sum of parts."""
        p1 = tmp_path / "p1.wav"
        p2 = tmp_path / "p2.wav"
        AudioSegment.silent(duration=500, frame_rate=44100).set_channels(2).export(str(p1), format="wav")
        AudioSegment.silent(duration=700, frame_rate=44100).set_channels(2).export(str(p2), format="wav")
        out = tmp_path / "song.wav"
        result_path = concat_parts([str(p1), str(p2)], str(out))
        assert result_path == str(out)
        final = AudioSegment.from_wav(str(out))
        assert final.duration_seconds == pytest.approx(1.2, abs=0.05)

    def test_empty_raises(self, tmp_path):
        with pytest.raises(ValueError, match="no part mix paths"):
            concat_parts([], str(tmp_path / "out.wav"))

    def test_creates_parent_dir(self, tmp_path):
        """concat_parts creates intermediate directories if needed."""
        p1 = tmp_path / "p1.wav"
        AudioSegment.silent(duration=500, frame_rate=44100).set_channels(2).export(str(p1), format="wav")
        out = tmp_path / "nested" / "deep" / "song.wav"
        concat_parts([str(p1)], str(out))
        assert out.exists()
