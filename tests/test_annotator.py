"""Annotator tests (R-X6): fixture-driven pure-function contract with D-15/D-16 semantics.

Builds synthetic SongParams + RenderResult + MixResult + beat/downbeat lists +
stub musicality dict, calls annotate(...), asserts returned dict shape against
the R-P4 schema. Includes a no-I/O contract test (monkeypatch open to raise).
"""
from __future__ import annotations

import builtins
import copy
from unittest.mock import patch

import pytest

from musicgen.annotator import annotate
from musicgen.sampler import SongParams
from musicgen.renderer import RenderResult
from musicgen.mixer import MixResult


# ---------- Fixtures ----------

@pytest.fixture
def minimal_song_params():
    """Single-part 4/4 SongParams for fixture simplicity."""
    return SongParams(
        key="Am",
        tempo=120,
        time_signature_base="4/4",
        time_signature_variation=1.0,
        swing_amount=0.66,
        signatures_per_part={"intro": "4/4"},
        measures_per_part={"intro": 2},
        song_unique_parts=["intro"],
        song_arrangement=["intro"],
    )


@pytest.fixture
def minimal_render_results():
    return {
        "intro": RenderResult(
            stem_paths={
                "beat": "/tmp/intro_beat.wav",
                "melody": "/tmp/intro_melody.wav",
                "harmony": "/tmp/intro_harmony.wav",
                "bassline": "/tmp/intro_bassline.wav",
            },
            sample_rate=44100,
            channels=2,
            duration_seconds=4.0,
            fluidsynth_version="FluidSynth runtime version 2.3.4",
        ),
    }


@pytest.fixture
def minimal_mix_results():
    return {
        "intro": MixResult(
            mix_path="/tmp/intro_mix.wav",
            stem_paths={
                "beat": "/tmp/intro/beat_fx.wav",
                "melody": "/tmp/intro/melody_silent.wav",
                "harmony": "/tmp/intro/harmony_fx.wav",
                "bassline": "/tmp/intro/bassline_fx.wav",
            },
            part_layers={"beat": True, "melody": False, "harmony": True, "bassline": True},
            soundfonts={
                "beat": "/sf/beat/fake.sf2",
                "melody": "/sf/melody/fake.sf2",
                "harmony": "/sf/harmony/fake.sf2",
                "bassline": "/sf/bassline/fake.sf2",
            },
            pedalboards={"beat": [], "melody": [], "harmony": [], "bassline": []},
            transitions=[["intro", 0.0], ["end", 4.0]],
        ),
    }


@pytest.fixture
def minimal_beat_times():
    return {"intro": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]}


@pytest.fixture
def minimal_downbeat_times():
    return {"intro": [0.0, 2.0]}


@pytest.fixture
def minimal_musicality():
    return {"score": 0.75, "components": {"rhythm": 0.8, "melody": 0.7}}


@pytest.fixture
def minimal_chord_progressions():
    return {"intro": ["Am", "F", "C", "G"]}


@pytest.fixture
def minimal_midi_paths():
    return {
        "intro": {
            "beat": "/tmp/intro-beat.mid",
            "melody": "/tmp/intro-melody.mid",
            "harmony": "/tmp/intro-harmony.mid",
            "bassline": "/tmp/intro-bassline.mid",
        },
    }


@pytest.fixture
def annotate_kwargs(
    minimal_song_params, minimal_render_results, minimal_mix_results,
    minimal_beat_times, minimal_downbeat_times, minimal_musicality,
    minimal_chord_progressions, minimal_midi_paths,
):
    return dict(
        song_params=minimal_song_params,
        render_results=minimal_render_results,
        mix_results=minimal_mix_results,
        beat_times=minimal_beat_times,
        downbeat_times=minimal_downbeat_times,
        musicality=minimal_musicality,
        chord_progressions=minimal_chord_progressions,
        midi_paths=minimal_midi_paths,
        mix_path="/tmp/song.wav",
        fluidsynth_version="FluidSynth runtime version 2.3.4",
    )


# ---------- Shape (D-15) ----------

class TestAnnotateShape:
    def test_returns_dict(self, annotate_kwargs):
        result = annotate(**annotate_kwargs)
        assert isinstance(result, dict)

    def test_phase4_fields_filled(self, annotate_kwargs):
        result = annotate(**annotate_kwargs)
        phase4_fields = [
            "key", "mode", "tempo_bpm", "time_signature", "time_signatures_per_part",
            "measures_per_part", "swing", "song_arrangement", "chord_progression",
            "active_layers", "soundfonts", "fx_params", "beat_times", "downbeat_times",
            "musicality_score", "duration_seconds", "fluidsynth_version",
            "mix", "stems", "midi",
        ]
        for field in phase4_fields:
            assert field in result, f"R-P4 field {field!r} missing from annotate output"
            assert result[field] is not None, f"R-P4 field {field!r} is None; Phase 4 must fill it"

    def test_key_and_tempo_threaded(self, annotate_kwargs):
        result = annotate(**annotate_kwargs)
        assert result["key"] == "Am"
        assert result["tempo_bpm"] == 120

    def test_swing_threaded(self, annotate_kwargs):
        result = annotate(**annotate_kwargs)
        assert result["swing"] == 0.66

    def test_time_signature_base(self, annotate_kwargs):
        result = annotate(**annotate_kwargs)
        assert result["time_signature"] == "4/4"
        assert result["time_signatures_per_part"] == {"intro": "4/4"}

    def test_measures_per_part(self, annotate_kwargs):
        result = annotate(**annotate_kwargs)
        assert result["measures_per_part"] == {"intro": 2}

    def test_chord_progression_per_part(self, annotate_kwargs):
        result = annotate(**annotate_kwargs)
        assert result["chord_progression"] == {"intro": ["Am", "F", "C", "G"]}

    def test_soundfonts_from_first_mix_result(self, annotate_kwargs):
        result = annotate(**annotate_kwargs)
        assert set(result["soundfonts"].keys()) == {"beat", "melody", "harmony", "bassline"}

    def test_active_layers_per_part(self, annotate_kwargs):
        result = annotate(**annotate_kwargs)
        assert result["active_layers"] == {"intro": {"beat": True, "melody": False, "harmony": True, "bassline": True}}

    def test_beat_times_per_part(self, annotate_kwargs):
        result = annotate(**annotate_kwargs)
        assert result["beat_times"]["intro"] == [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]

    def test_downbeat_times_per_part(self, annotate_kwargs):
        result = annotate(**annotate_kwargs)
        assert result["downbeat_times"]["intro"] == [0.0, 2.0]

    def test_mix_path_threaded(self, annotate_kwargs):
        result = annotate(**annotate_kwargs)
        assert result["mix"] == "/tmp/song.wav"

    def test_midi_paths_per_part(self, annotate_kwargs):
        result = annotate(**annotate_kwargs)
        assert set(result["midi"]["intro"].keys()) == {"beat", "melody", "harmony", "bassline"}

    def test_fluidsynth_version_threaded(self, annotate_kwargs):
        result = annotate(**annotate_kwargs)
        assert result["fluidsynth_version"] == "FluidSynth runtime version 2.3.4"


# ---------- D-16 None semantics ----------

class TestTbdFieldsAreNone:
    def test_seed_is_none(self, annotate_kwargs):
        result = annotate(**annotate_kwargs)
        assert "seed" in result
        assert result["seed"] is None

    def test_musicgen_version_is_none(self, annotate_kwargs):
        result = annotate(**annotate_kwargs)
        assert "musicgen_version" in result
        assert result["musicgen_version"] is None

    def test_split_is_none(self, annotate_kwargs):
        result = annotate(**annotate_kwargs)
        assert "split" in result
        assert result["split"] is None

    def test_pre_roll_offset_seconds_is_none(self, annotate_kwargs):
        result = annotate(**annotate_kwargs)
        assert "pre_roll_offset_seconds" in result
        assert result["pre_roll_offset_seconds"] is None

    def test_seed_can_be_threaded_via_kwarg(self, annotate_kwargs):
        """Phase 5 will pass seed; the kwarg routing is already wired."""
        result = annotate(**{**annotate_kwargs, "seed": 42})
        assert result["seed"] == 42

    def test_split_can_be_threaded_via_kwarg(self, annotate_kwargs):
        result = annotate(**{**annotate_kwargs, "split": "train"})
        assert result["split"] == "train"


# ---------- analysis_failed (D-16 clarification) ----------

class TestAnalysisFailedKey:
    def test_omitted_on_success_default(self, annotate_kwargs):
        result = annotate(**annotate_kwargs)
        assert "analysis_failed" not in result

    def test_omitted_when_explicit_false(self, annotate_kwargs):
        """D-16 clarification: do NOT emit {"analysis_failed": False}."""
        result = annotate(**{**annotate_kwargs, "analysis_failed": False})
        assert "analysis_failed" not in result

    def test_present_and_true_when_explicit_true(self, annotate_kwargs):
        result = annotate(**{**annotate_kwargs, "analysis_failed": True})
        assert "analysis_failed" in result
        assert result["analysis_failed"] is True


# ---------- Mode derivation ----------

class TestModeDerivation:
    @pytest.mark.parametrize("key,expected_mode", [
        ("A", "major"),
        ("Am", "minor"),
        ("C", "major"),
        ("C#", "major"),
        ("C#m", "minor"),
        ("F#m", "minor"),
        ("G", "major"),
        ("D#m", "minor"),
    ])
    def test_mode_from_key(self, annotate_kwargs, key, expected_mode):
        sp = annotate_kwargs["song_params"]
        new_sp = SongParams(
            key=key, tempo=sp.tempo, time_signature_base=sp.time_signature_base,
            time_signature_variation=sp.time_signature_variation, swing_amount=sp.swing_amount,
            signatures_per_part=sp.signatures_per_part, measures_per_part=sp.measures_per_part,
            song_unique_parts=sp.song_unique_parts, song_arrangement=sp.song_arrangement,
        )
        result = annotate(**{**annotate_kwargs, "song_params": new_sp})
        assert result["mode"] == expected_mode


# ---------- song_arrangement shape ----------

class TestSongArrangement:
    def test_is_list_of_dicts(self, annotate_kwargs):
        result = annotate(**annotate_kwargs)
        sa = result["song_arrangement"]
        assert isinstance(sa, list)
        assert all(isinstance(entry, dict) for entry in sa)

    def test_each_entry_has_part_start_end(self, annotate_kwargs):
        result = annotate(**annotate_kwargs)
        for entry in result["song_arrangement"]:
            assert set(entry.keys()) == {"part", "start_seconds", "end_seconds"}

    def test_length_matches_arrangement(self, annotate_kwargs):
        result = annotate(**annotate_kwargs)
        assert len(result["song_arrangement"]) == len(annotate_kwargs["song_params"].song_arrangement)

    def test_start_seconds_monotonic(self, annotate_kwargs):
        # Build a 3-part arrangement
        sp = annotate_kwargs["song_params"]
        new_sp = SongParams(
            key=sp.key, tempo=sp.tempo, time_signature_base=sp.time_signature_base,
            time_signature_variation=sp.time_signature_variation, swing_amount=sp.swing_amount,
            signatures_per_part={"intro": "4/4", "verse": "4/4", "chorus": "4/4"},
            measures_per_part={"intro": 2, "verse": 4, "chorus": 4},
            song_unique_parts=["intro", "verse", "chorus"],
            song_arrangement=["intro", "verse", "chorus"],
        )
        new_mr = {
            p: MixResult(
                mix_path=f"/tmp/{p}.wav",
                stem_paths={l: f"/tmp/{p}/{l}.wav" for l in ("beat", "melody", "harmony", "bassline")},
                part_layers={l: True for l in ("beat", "melody", "harmony", "bassline")},
                soundfonts={l: f"/sf/{l}/fake.sf2" for l in ("beat", "melody", "harmony", "bassline")},
                pedalboards={l: [] for l in ("beat", "melody", "harmony", "bassline")},
                transitions=[[p, 0.0], ["end", 2.0]],
            )
            for p in ("intro", "verse", "chorus")
        }
        new_rr = {
            p: RenderResult(
                stem_paths={l: f"/tmp/{p}_{l}.wav" for l in ("beat", "melody", "harmony", "bassline")},
                sample_rate=44100, channels=2, duration_seconds=2.0, fluidsynth_version="v",
            )
            for p in ("intro", "verse", "chorus")
        }
        new_bt = {p: [] for p in ("intro", "verse", "chorus")}
        new_dbt = {p: [] for p in ("intro", "verse", "chorus")}
        new_cp = {p: [] for p in ("intro", "verse", "chorus")}
        new_midi = {
            p: {l: f"/tmp/{p}-{l}.mid" for l in ("beat", "melody", "harmony", "bassline")}
            for p in ("intro", "verse", "chorus")
        }
        result = annotate(
            song_params=new_sp, render_results=new_rr, mix_results=new_mr,
            beat_times=new_bt, downbeat_times=new_dbt,
            musicality={"score": 0.5, "components": {}},
            chord_progressions=new_cp, midi_paths=new_midi,
            mix_path="/tmp/full.wav", fluidsynth_version="v",
        )
        starts = [entry["start_seconds"] for entry in result["song_arrangement"]]
        assert starts == sorted(starts), f"start_seconds not monotonic: {starts}"
        # Chained: each end_seconds == next start_seconds
        for i in range(len(result["song_arrangement"]) - 1):
            cur_end = result["song_arrangement"][i]["end_seconds"]
            next_start = result["song_arrangement"][i + 1]["start_seconds"]
            assert cur_end == next_start, f"arrangement gap at index {i}: {cur_end} vs {next_start}"

    def test_duration_equals_final_end(self, annotate_kwargs):
        result = annotate(**annotate_kwargs)
        if result["song_arrangement"]:
            assert result["duration_seconds"] == result["song_arrangement"][-1]["end_seconds"]


# ---------- Purity contract (D-14: zero I/O) ----------

class TestAnnotatorIsPure:
    def test_does_not_call_open(self, annotate_kwargs):
        """D-14: annotate() must not call builtins.open — no file I/O allowed."""
        def _fail_open(*args, **kwargs):
            raise AssertionError("annotate() called open() — D-14 purity violation")

        with patch("builtins.open", _fail_open):
            result = annotate(**annotate_kwargs)
            assert isinstance(result, dict)
        # Restoration is automatic via the context manager.

    def test_does_not_mutate_inputs(self, annotate_kwargs):
        """Annotator must not mutate input dicts (defensive copies expected)."""
        # Deep-copy inputs before call
        original_mix_results = copy.deepcopy(annotate_kwargs["mix_results"])
        original_chord_progressions = copy.deepcopy(annotate_kwargs["chord_progressions"])
        original_beat_times = copy.deepcopy(annotate_kwargs["beat_times"])
        annotate(**annotate_kwargs)
        # Verify no mutation
        assert annotate_kwargs["mix_results"] == original_mix_results
        assert annotate_kwargs["chord_progressions"] == original_chord_progressions
        assert annotate_kwargs["beat_times"] == original_beat_times


# ---------- Determinism ----------

class TestAnnotatorDeterminism:
    def test_same_inputs_same_output(self, annotate_kwargs):
        a = annotate(**annotate_kwargs)
        b = annotate(**annotate_kwargs)
        assert a == b


# ---------- stems/midi structure ----------

class TestStemsAndMidi:
    def test_stems_shape(self, annotate_kwargs):
        result = annotate(**annotate_kwargs)
        assert "intro" in result["stems"]
        assert set(result["stems"]["intro"].keys()) == {"beat", "melody", "harmony", "bassline"}

    def test_midi_shape(self, annotate_kwargs):
        result = annotate(**annotate_kwargs)
        assert "intro" in result["midi"]
        assert set(result["midi"]["intro"].keys()) == {"beat", "melody", "harmony", "bassline"}
