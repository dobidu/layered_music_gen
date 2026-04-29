# Phase 4: Renderer + Mixer + Annotator + Beats Extraction — Pattern Map

**Mapped:** 2026-04-19
**Files analyzed:** 13 new/modified files
**Analogs found:** 12 / 13 (1 file — `tests/test_integration_full_generation.py` — has no existing E2E analog)

---

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `src/musicgen/renderer.py` | service | request-response (subprocess) | `src/musicgen/generators/beat.py` | role-match |
| `src/musicgen/mixer.py` | service | transform (audio pipeline) | `music_gen.py:130-345` (extraction source) | exact |
| `src/musicgen/annotator.py` | service | transform (pure function) | `src/musicgen/sampler.py` | role-match |
| `src/musicgen/beats.py` | utility | transform (MIDI-tick) | `beat_anotator.py:6-35` + `src/musicgen/generators/beat.py` | exact |
| `src/musicgen/generators/beat.py` | utility | transform | self (modification: re-export alias) | exact |
| `music_gen.py` | orchestrator | request-response | self (modification: collapse) | exact |
| `pyproject.toml` | config | — | self (modification) | exact |
| `tests/test_renderer.py` | test | request-response | `tests/test_generators/test_beat.py` | role-match |
| `tests/test_mixer.py` | test | transform | `tests/test_generators/test_bassline.py` | role-match |
| `tests/test_annotator.py` | test | transform | `tests/test_sampler.py` | exact |
| `tests/test_beats.py` | test | transform | `tests/test_generators/test_beat.py` | exact |
| `tests/test_no_bare_random_in_package.py` | test (static) | — | `tests/test_generators/test_no_bare_random.py` + `tests/test_sampler.py::test_no_bare_random_in_sampler` | exact |
| `tests/test_integration_full_generation.py` | test (E2E) | request-response | — | no analog |

---

## Pattern Assignments

### `src/musicgen/renderer.py` (service, request-response)

**Analog:** `src/musicgen/generators/beat.py` (module shape + cfg-with-fallback + rng-threaded signature); `music_gen.py:265-274` (FluidSynth invocation)

**Imports + logger pattern** (`generators/beat.py:1-29`):
```python
"""Beat generator (extracted from music_gen.py per Plan 03-04 / R-X3).
...
Design:
  D-02 — cfg fallback: ``_beat_cfg = cfg if cfg is not None else config.Config()``.
  D-07 — Zero bare ``random.<method>`` calls — all draws use injected ``rng``.
  D-22 — Takes per-field fields, not SongParams.
...
"""
import logging
import os
import random
from typing import List, Optional, Tuple

import config
from timesig import TimeSignatureRegistry

logger = logging.getLogger(__name__)
```

**FLUIDSYNTH_VERSION capture pattern** (D-07 — locked in CONTEXT.md; `music_gen.py:4` for the FluidSynth import):
```python
import subprocess
import logging

logger = logging.getLogger(__name__)

try:
    _result = subprocess.run(
        ["fluidsynth", "--version"],
        capture_output=True, text=True, timeout=5,
    )
    FLUIDSYNTH_VERSION: str = _result.stdout.splitlines()[0]
except Exception:
    FLUIDSYNTH_VERSION = "unknown"
    logger.warning("Could not capture FluidSynth version; renderer importable on CI without binary")
```

**FluidSynth invocation pattern** (`music_gen.py:265-274`):
```python
beat_wav = 'beat' + "-" + str(part_counter) + "-" + part + ".wav"
beat_wav = os.path.join(name, beat_wav)
FluidSynth(beat_soundfont).midi_to_audio(beat_filename[part], beat_wav)
melo_wav = 'melody' + "-" + str(part_counter) + "-" + part + ".wav"
melo_wav = os.path.join(name, melo_wav)
FluidSynth(melody_soundfont).midi_to_audio(melo_filename[part], melo_wav)
harm_wav = 'harmony' + "-" + str(part_counter) + "-" + part + ".wav"
harm_wav = os.path.join(name, harm_wav)
FluidSynth(harmony_soundfont).midi_to_audio(harm_filename[part], harm_wav)
bass_wav = 'bassline' + "-" + str(part_counter) + "-" + part + ".wav"
bass_wav = os.path.join(name, bass_wav)
FluidSynth(bassline_soundfont).midi_to_audio(bass_filename[part], bass_wav)
```

**RNG-threaded soundfont selection pattern** (`music_gen.py:117-120` — the source to REPLACE; D-08):
```python
# CURRENT (bare random.choice — to be replaced):
def get_random_sound_font(directory_path):
    sound_fonts = [f for f in os.listdir(directory_path) if f.endswith('.sf2')]
    file_return = random.choice(sound_fonts)
    return os.path.join(directory_path, file_return)

# NEW pick_soundfonts pattern (D-08/D-17):
def pick_soundfonts(cfg: config.Config = None, rng: random.Random = None) -> Dict[str, str]:
    _cfg = cfg if cfg is not None else config.Config()
    soundfonts = {}
    for layer in ("beat", "melody", "harmony", "bassline"):
        sf_dir = _cfg.sf_layer_dir(layer)
        sf2_files = [f for f in os.listdir(sf_dir) if f.endswith(".sf2")]
        soundfonts[layer] = os.path.join(sf_dir, rng.choice(sf2_files))
    return soundfonts
```

**Cfg-with-fallback pattern** (`generators/beat.py:84-85`):
```python
_beat_cfg = cfg if cfg is not None else config.Config()
```

**RenderResult frozen dataclass** (D-02; matches `sampler.py:223-293` SongParams shape):
```python
from dataclasses import dataclass
from typing import Dict

@dataclass(frozen=True)
class RenderResult:
    """Per-part stem render outputs (R-X4).

    D-02: frozen dataclass; D-07: FLUIDSYNTH_VERSION captured at module load.
    """
    stem_paths: Dict[str, str]       # layer → wav path
    sample_rate: int                  # 44100 (FluidSynth default)
    channels: int                     # 2 (FluidSynth renders stereo)
    duration_seconds: float           # length of each stem (all equal per part)
    fluidsynth_version: str           # from module-level FLUIDSYNTH_VERSION
```

**ThreadPoolExecutor parallel dispatch pattern** (D-06; no existing analog — use stdlib):
```python
from concurrent.futures import ThreadPoolExecutor, as_completed

def render_stems(
    midi_paths: Dict[str, str],
    soundfonts: Dict[str, str],
    out_dir: str,
    cfg: config.Config = None,
) -> RenderResult:
    _cfg = cfg if cfg is not None else config.Config()
    os.makedirs(out_dir, exist_ok=True)
    stem_paths: Dict[str, str] = {}

    def _render_one(layer: str) -> tuple:
        wav_path = os.path.join(out_dir, f"{layer}.wav")
        FluidSynth(soundfonts[layer]).midi_to_audio(midi_paths[layer], wav_path)
        return layer, wav_path

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(_render_one, layer): layer
                   for layer in ("beat", "melody", "harmony", "bassline")}
        for future in as_completed(futures):
            layer, wav_path = future.result()
            stem_paths[layer] = wav_path
    # ... read duration/sample_rate from first stem, return RenderResult(...)
```

---

### `src/musicgen/mixer.py` (service, transform)

**Analog:** `music_gen.py:130-345` (primary extraction source)

**Imports + logger pattern** (match `generators/beat.py:1-29` shape; add pydub/pedalboard):
```python
"""Mixer module — FX application, pydub overlay, part concatenation (R-X5).

Design:
  D-10 — create_effect, generate_pedalboard, apply_fx_to_layer, pedalboard_info_json
          moved from music_gen.py; _lin_to_db unnested to module scope.
  D-11 — FX applied to ALL four layers regardless of layer mask (RNG draw order
          preservation; the music_gen.py:276 TODO is OUT OF SCOPE).
  D-12 — Silent-stem fallback: AudioSegment.silent(...).set_channels(2).
  D-17 — Zero bare random.* calls; all draws via injected rng parameter.
"""
import json
import logging
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

from pedalboard import Pedalboard, Compressor, Gain, Chorus, LadderFilter, Phaser, Delay, Reverb
from pedalboard.io import AudioFile
from pydub import AudioSegment

import config
from musicgen.renderer import RenderResult

logger = logging.getLogger(__name__)
```

**_lin_to_db helper** (`music_gen.py:285-286` — unnested to module scope per RESEARCH anti-patterns):
```python
def _lin_to_db(v: float) -> float:
    """Convert linear amplitude to dB. Clamps to avoid log(0). (R-S4 fix)"""
    return 20.0 * math.log10(max(float(v), 1e-6))
```

**create_effect pattern** (`music_gen.py:130-139` — replace bare random.*):
```python
# CURRENT (bare random.* — source):
def create_effect(effect_class, parameters):
    probability = parameters['probability']
    value_range = parameters['value_range']
    if random.random() < probability:
        kwargs = {param: random.uniform(value_range[param][0], value_range[param][1])
                  for param in value_range}
        return effect_class(**kwargs)
    return None

# NEW signature with injected rng (D-17):
def _create_effect(effect_class, parameters, rng: random.Random):
    probability = parameters['probability']
    value_range = parameters['value_range']
    if rng.random() < probability:
        kwargs = {param: rng.uniform(value_range[param][0], value_range[param][1])
                  for param in value_range}
        return effect_class(**kwargs)
    return None
```

**generate_pedalboard / build_fx_boards pattern** (`music_gen.py:141-160`):
```python
# CURRENT (extraction source):
def generate_pedalboard(effect_params_file):
    with open(effect_params_file, 'r') as json_file:
        effect_params = json.load(json_file)
    effects = [
        (Compressor, effect_params['compressor']),
        (Gain, effect_params['gain']),
        (Chorus, effect_params['chorus']),
        (LadderFilter, effect_params['ladder_filter']),
        (Phaser, effect_params['phaser']),
        (Delay, effect_params['delay']),
        (Reverb, effect_params['reverb']),
    ]
    board = Pedalboard([effect for effect in (create_effect(effect_class, parameters)
                                              for effect_class, parameters in effects)
                        if effect is not None])
    return board

# NEW build_fx_boards signature (D-10/D-17):
def build_fx_boards(cfg: config.Config = None, rng: random.Random = None) -> Dict[str, Pedalboard]:
    _cfg = cfg if cfg is not None else config.Config()
    return {
        layer: _generate_pedalboard(_cfg.fx_files[layer], rng)
        for layer in ("beat", "melody", "harmony", "bassline")
    }
```

**apply_fx_to_layer pattern** (`music_gen.py:162-171` — lifts verbatim, no rng draws):
```python
def apply_fx_to_layer(wav_file: str, board: Pedalboard) -> str:
    with AudioFile(wav_file) as af:
        with AudioFile(wav_file + '_fx.wav', 'w', af.samplerate, af.num_channels) as of:
            while af.tell() < af.frames:
                chunk = af.read(af.samplerate)
                effected = board(chunk, af.samplerate, reset=False)
                of.write(effected)
    return wav_file + '_fx.wav'
```

**pedalboard_info_json pattern** (`music_gen.py:173-190` — lifts verbatim):
```python
def pedalboard_info_json(board: Pedalboard) -> list:
    pedals_and_parameters = []
    for pedal in board:
        attributes = dir(pedal)
        parameters = [attr for attr in attributes if not attr.startswith("_") and not callable(getattr(pedal, attr))]
        pedal_info = {"name": pedal.__class__.__name__, "parameters": {}}
        for parameter in parameters:
            value = getattr(pedal, parameter)
            pedal_info['parameters'][str(parameter)] = str(value)
        pedals_and_parameters.append(pedal_info)
    return pedals_and_parameters
```

**compute_layer_mask pattern** (`music_gen.py:245-253` — replace bare random.*):
```python
# CURRENT (bare random.random() x4 per part — source):
for part in song_unique_parts:
    beat_proba    = float(inst_proba[part]['beat'])
    melody_proba  = float(inst_proba[part]['melody'])
    harmony_proba = float(inst_proba[part]['harmony'])
    bassline_proba = float(inst_proba[part]['bassline'])
    beat_part_mix[part]     = (random.random() <= beat_proba)
    melody_part_mix[part]   = (random.random() <= melody_proba)
    harmony_part_mix[part]  = (random.random() <= harmony_proba)
    bassline_part_mix[part] = (random.random() <= bassline_proba)

# NEW compute_layer_mask (D-13/D-17):
def compute_layer_mask(
    song_unique_parts: List[str],
    inst_proba: dict,
    rng: random.Random,
) -> Dict[str, Dict[str, bool]]:
    return {
        part: {
            layer: rng.random() <= float(inst_proba[part][layer])
            for layer in ("beat", "melody", "harmony", "bassline")
        }
        for part in song_unique_parts
    }
```

**gain/pan application pattern — R-S4 fix** (`music_gen.py:287-294`):
```python
beat = beat.apply_gain(_lin_to_db(levels[part]['beat']['volume']))
melody = melody.apply_gain(_lin_to_db(levels[part]['melody']['volume']))
harmony = harmony.apply_gain(_lin_to_db(levels[part]['harmony']['volume']))
bassline = bassline.apply_gain(_lin_to_db(levels[part]['bassline']['volume']))
beat = beat.pan(float(levels[part]['beat']['panning']))
melody = melody.pan(float(levels[part]['melody']['panning']))
harmony = harmony.pan(float(levels[part]['harmony']['panning']))
bassline = bassline.pan(float(levels[part]['bassline']['panning']))
```

**Silent-stem fallback — stereo match required** (D-12; Pitfall 1 fix from RESEARCH.md):
```python
# AudioSegment.silent() defaults to MONO 11025 Hz — must explicitly set stereo
# to match FluidSynth's 2-channel 44100 Hz output (RESEARCH.md Pitfall 1).
def _make_silent_stem(duration_ms: int, sample_rate: int = 44100, channels: int = 2) -> AudioSegment:
    silent = AudioSegment.silent(duration=duration_ms, frame_rate=sample_rate)
    return silent.set_channels(channels)
```

**FX-on-ALL-layers unconditionally + overlay only masked layers** (`music_gen.py:277-320`; D-11):
```python
# FX applied to all 4 — do NOT move inside if-branch (D-11: RNG order locked)
beat    = AudioSegment.from_wav(apply_fx_to_layer(beat_wav,    beat_board))
melody  = AudioSegment.from_wav(apply_fx_to_layer(melo_wav,   melody_board))
harmony = AudioSegment.from_wav(apply_fx_to_layer(harm_wav,   harmony_board))
bassline = AudioSegment.from_wav(apply_fx_to_layer(bass_wav,  bassline_board))
# ... apply gain/pan to all 4 ...
mix = AudioSegment.silent(duration=beat.duration_seconds * 1000)
# THEN overlay only masked-in layers:
if beat_part_mix[part]:
    mix = mix.overlay(beat)
if melody_part_mix[part]:
    mix = mix.overlay(melody)
if harmony_part_mix[part]:
    mix = mix.overlay(harmony)
if bassline_part_mix[part]:
    mix = mix.overlay(bassline)
```

**concat_parts pattern** (`music_gen.py:332-340`):
```python
# CURRENT (extraction source):
song = AudioSegment.from_wav(song_parts[0])
for part_wav in song_parts[1:]:
    song += AudioSegment.from_wav(part_wav)
song_file_wav = name + '.wav'
song_file_wav = os.path.join(name, song_file_wav)
song.export(song_file_wav, format='wav')
```

**MixResult frozen dataclass** (D-02):
```python
@dataclass(frozen=True)
class MixResult:
    """Per-part mix outputs (R-X5).

    D-02: frozen dataclass.
    """
    mix_path: str
    stem_paths: Dict[str, str]        # layer → post-FX wav path (incl. silent stubs)
    part_layers: Dict[str, bool]      # layer → included in mix
    soundfonts: Dict[str, str]        # layer → sf2 path
    pedalboards: Dict[str, list]      # layer → pedalboard_info_json output
    transitions: list                 # [[part, start_s], ..., ["end", end_s]]
```

---

### `src/musicgen/annotator.py` (service, transform — pure function)

**Analog:** `src/musicgen/sampler.py` (pure-function style; no I/O inside; same import shape)

**Imports + logger pattern** (`sampler.py:1-24`):
```python
"""Annotator module — pure function producing the Phase 4 R-P4 schema subset (R-X6).

Design:
  D-14 — annotate() is a pure function; zero I/O inside. Caller writes the JSON.
  D-15 — Phase-4-filled fields computed; Phase-5 TBD fields present as None (not missing).
  D-16 — TBD-flag representation: None (not "TBD" string, not absent key).
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

from musicgen.sampler import SongParams
from musicgen.renderer import RenderResult
from musicgen.mixer import MixResult

logger = logging.getLogger(__name__)
```

**Pure function signature** (D-14; locked in CONTEXT.md):
```python
def annotate(
    song_params: SongParams,
    render_results: Dict[str, RenderResult],   # per-part
    mix_results: Dict[str, MixResult],         # per-part
    beat_times: Dict[str, List[float]],        # per-part
    downbeat_times: Dict[str, List[float]],    # per-part
    musicality: dict,                          # {"score": float, "components": {...}}
    *,
    fluidsynth_version: str,
    musicgen_version: Optional[str] = None,    # Phase 5 fills
    seed: Optional[int] = None,                # Phase 5 fills
    split: Optional[str] = None,               # Phase 5 fills (R-P6)
) -> dict:
    """Produce the R-P4 annotation dict for one sample (Phase 4 subset).

    Phase-4-filled fields are non-None. Phase-5 TBD fields are present as None.
    Caller is responsible for json.dump (Phase 5 writer owns lifecycle).

    Args:
        song_params: Frozen SongParams from sampler.
        render_results: Dict of per-part RenderResult from renderer.
        mix_results: Dict of per-part MixResult from mixer.
        beat_times: Dict of per-part beat timestamps from beats.extract_beat_times.
        downbeat_times: Dict of per-part downbeat timestamps from beats.extract_downbeat_times.
        musicality: Dict with "score" (float) and "components" (dict) from musicality_score.
        fluidsynth_version: renderer.FLUIDSYNTH_VERSION captured at import.
        musicgen_version: None this phase; Phase 5 fills from R-Q4.
        seed: None this phase; Phase 5 fills from R-P7.
        split: None this phase; Phase 5 fills from R-P6.

    Returns:
        Plain dict ready for json.dump. Phase-5 TBD fields are present as None.
    """
```

**TBD-fields-as-None pattern** (D-16):
```python
# Phase-5 TBD fields — present as None, NOT absent, NOT "TBD" string
return {
    # Phase 4 FILLS:
    "key": song_params.key,
    "mode": "minor" if song_params.key.endswith("m") else "major",
    "tempo_bpm": song_params.tempo,
    # ... all Phase-4 fields ...
    "fluidsynth_version": fluidsynth_version,
    # Phase 5 FILLS (None this phase):
    "seed": seed,
    "musicgen_version": musicgen_version,
    "split": split,
    "pre_roll_offset_seconds": None,
    # analysis_failed: omitted entirely on success (D-16)
}
```

---

### `src/musicgen/beats.py` (utility, transform)

**Analog:** `beat_anotator.py:6-35` (primary extraction source — being deleted); `src/musicgen/generators/beat.py:32-38` (beat_duration source for D-21)

**Imports pattern** (match `generators/beat.py:1-29`; add mido):
```python
"""Beats module — MIDI-tick beat and downbeat extraction (R-X7).

Replaces beat_anotator.py (D-03/D-19). Uses mido MIDI-tick derivation so beat
timestamps are automatically swing-aware (swing is baked into MIDI onset times
by generators/beat.py:calculate_swing_offset at write time).

Design:
  D-19 — extract_beat_times uses mido.MidiFile + mido.tick2second.
  D-20 — extract_downbeat_times uses time-grid derivation, NOT stride slice
          (RESEARCH.md Pitfall 2: beat_times is sparse for patterns with zero slots).
  D-21 — beat_duration primary definition here; generators/beat.py keeps re-export alias.
"""
import logging
from typing import List

import mido

import config
from timesig import TimeSignatureRegistry

logger = logging.getLogger(__name__)
```

**beat_duration function** (`beat_anotator.py:6-12` + `generators/beat.py:32-38` — identical bodies; D-21 puts primary here):
```python
def beat_duration(signature: str, tempo: int) -> float:
    """Calculates the duration of a beat based on the time signature and BPM."""
    numerator, denominator = map(int, signature.split('/'))
    beat_length = 60 / tempo  # Duration of a quarter note
    return beat_length * (4 / denominator)
```

**extract_beat_times pattern** (`beat_anotator.py:21-35` — rename + add start_offset_seconds param):
```python
def extract_beat_times(
    midi_path: str,
    tempo: int,
    start_offset_seconds: float,
) -> List[float]:
    """Extract beat timestamps from MIDI note_on events.

    Swing is already baked into MIDI onset times (generators/beat.py applies
    calculate_swing_offset at write time), so tick extraction is
    automatically swing-aware (D-19).

    Args:
        midi_path: Path to the beat MIDI file.
        tempo: BPM (integer).
        start_offset_seconds: Part start time in the full mix (from MixResult.transitions).

    Returns:
        Sorted list of beat timestamps in seconds.
    """
    midi = mido.MidiFile(midi_path)
    beats = []
    time_elapsed = 0.0
    ticks_per_beat = midi.ticks_per_beat
    tempo_us = mido.bpm2tempo(tempo)
    for msg in midi:  # merged tracks; msg.time is delta ticks
        time_elapsed += mido.tick2second(msg.time, ticks_per_beat, tempo_us)
        if msg.type == 'note_on' and msg.velocity > 0:
            beats.append(round(time_elapsed + start_offset_seconds, 3))
    return sorted(beats)
```

**extract_downbeat_times — time-grid approach** (RESEARCH.md Pitfall 2 fix; D-20 corrected):
```python
def extract_downbeat_times(
    beat_times: List[float],          # kept for API compatibility (D-20)
    time_signature: str,
    measures: int,
    start_offset_seconds: float,
    tempo: int,
) -> List[float]:
    """Derive downbeat timestamps as a pure time grid (one per measure).

    Does NOT stride-slice beat_times — beat_times is sparse for patterns
    with zero-valued slots (RESEARCH.md Pitfall 2 verified against all 6
    beat pattern files: 4/4 intro pattern '0,42,38,0' has only 2 non-zero
    entries; 12/8 has 9-10 non-zero per measure, not 12).

    Args:
        beat_times: Per-part beat timestamps (kept for API compat; not sliced).
        time_signature: e.g. "4/4".
        measures: Number of measures for this part.
        start_offset_seconds: Part start time in the full mix.
        tempo: BPM integer.

    Returns:
        List of one downbeat timestamp per measure.
    """
    spec = TimeSignatureRegistry.lookup(time_signature)
    beat_slot_s = beat_duration(time_signature, tempo)
    measure_duration_s = spec.numerator * beat_slot_s
    return [
        round(start_offset_seconds + i * measure_duration_s, 3)
        for i in range(measures)
    ]
```

---

### `src/musicgen/generators/beat.py` — beat_duration re-export alias (D-21)

**Modification:** Add import at top + re-export alias. Keep `calculate_swing_offset` and `generate_beat` unchanged.

**Re-export alias pattern** (add after existing `beat_duration` function body at line 38):
```python
# D-21: beat_duration primary definition moves to musicgen.beats this phase.
# Import and re-export so existing callers (generators, tests) are unaffected.
# The function body above is kept as a compatibility shim until Phase 5.
# Once all callers import from musicgen.beats, remove the body and keep only:
#   from musicgen.beats import beat_duration  # noqa: F401
```

Or (if planner prefers the immediate import approach):
```python
# Remove the function body for beat_duration from generators/beat.py and replace with:
from musicgen.beats import beat_duration  # noqa: F401  (D-21 re-export alias)
```

---

### `music_gen.py` — orchestrator collapse (D-23/D-24)

**Analog:** Self (modification; target: ~180 lines from 523)

**Functions DELETED** (extraction targets — identified at source lines):

| Function | Lines | Destination |
|----------|-------|-------------|
| `save_beat_annotations` | 98–107 | Deleted — Phase 5 writer owns lifecycle |
| `read_instrument_probabilities` | 112–115 | Moved to `mixer.py` / `config.Config.load_inst_probabilities()` (already exists) |
| `get_random_sound_font` | 117–120 | Replaced by `renderer.pick_soundfonts(cfg, rng)` |
| `get_levels` | 122–125 | Moved to `mixer.py` / `config.Config.load_levels()` (already exists) |
| `create_effect` | 130–139 | Moved to `mixer._create_effect(...)` |
| `generate_pedalboard` | 141–160 | Moved to `mixer.build_fx_boards(...)` |
| `apply_fx_to_layer` | 162–171 | Moved to `mixer.apply_fx_to_layer(...)` |
| `pedalboard_info_json` | 173–190 | Moved to `mixer.pedalboard_info_json(...)` |
| `mix_and_save` | 193–345 | Deleted entirely (D-23); responsibilities distributed |

**Functions KEPT** (lines that survive the collapse):

| Function | Lines | Status |
|----------|-------|--------|
| `verify_pattern_for_time_signature` | 53–56 | Keep |
| `verify_beat_pattern` | 59–62 | Keep |
| `calculate_measures_for_time_signature` | 64–66 | Keep |
| `get_midi_time_signature_values` | 72–76 | Keep |
| `get_note_duration` | 78–81 | Keep |
| `get_note_durations` | 83–86 | Keep |
| `get_melody_durations` | 88–91 | Keep |
| `create_song` | 352–433 | Slimmed to orchestration (calls renderer/mixer/beats/annotator) |
| `generate_song_parts` | 435–469 | Unchanged |
| `generate_song` | 471–511 | Unchanged |
| `if __name__ == "__main__"` | 515–523 | Keep (smoke-test entry point, D-24) |

**New imports to add at top of collapsed music_gen.py** (D-23; after existing imports survive):
```python
from musicgen import renderer, mixer, annotator, beats
```

**Collapsed create_song shape** (`music_gen.py:352` — replaces 81-line body with ~40-line orchestration; from RESEARCH.md code example):
```python
def create_song(
    key: str,
    tempo: int,
    song_signatures: Dict[str, str],
    measures: Dict[str, int],
    name: str,
    chord_pat_file: str,
    swing_amount: float,
    cfg: config.Config = None,
) -> Dict:
    _cfg = cfg if cfg is not None else config.Config()

    # R-S3: arrangement computed ONCE upstream (already in place)
    _structures_file = _cfg.song_structures_file
    song_unique_parts, song_arrangement = generate_song_arrangement(
        _rng, structures_file=_structures_file
    )

    # Step 1: soundfont selection (renderer, rng-aware — replaces 4× get_random_sound_font)
    soundfonts = renderer.pick_soundfonts(_cfg, _rng)

    # Step 2: generators (unchanged from Phase 3)
    ha, ba, me, be, _ = generate_song_parts(
        key=key, tempo=tempo, song_signatures=song_signatures,
        song_measures=measures, name=name, chord_pat_file=chord_pat_file,
        swing_amount=swing_amount, cfg=cfg,
    )

    # Step 3: FX boards + layer mask (mixer, rng-aware)
    fx_boards = mixer.build_fx_boards(_cfg, _rng)
    inst_proba = _cfg.load_inst_probabilities()
    layer_mask = mixer.compute_layer_mask(song_unique_parts, inst_proba, _rng)
    levels = _cfg.load_levels()

    # Step 4: per-part render + mix (serial parts, parallel stems via ThreadPoolExecutor)
    render_results, mix_results, beat_times_dict, downbeat_times_dict = {}, {}, {}, {}
    start_offset = 0.0
    for part in song_arrangement:
        midi_paths = {
            "beat": be[part], "melody": me[part],
            "harmony": ha[part], "bassline": ba[part],
        }
        out_dir = os.path.join(name, f"{name}-{part}", "stems")
        render_results[part] = renderer.render_stems(midi_paths, soundfonts, out_dir, cfg=_cfg)
        mix_results[part] = mixer.mix_part(
            render_results[part], levels, fx_boards, layer_mask[part], part, out_dir,
            soundfonts=soundfonts,
        )
        # beats use part start offset from transitions (D-22)
        beat_times_dict[part] = beats.extract_beat_times(be[part], tempo, start_offset)
        downbeat_times_dict[part] = beats.extract_downbeat_times(
            beat_times_dict[part], song_signatures[part], measures[part], start_offset, tempo
        )
        start_offset += render_results[part].duration_seconds

    # Step 5: concatenate parts
    final_wav = mixer.concat_parts(
        list(mix_results.values()), os.path.join(name, name + ".wav")
    )

    # Step 6: musicality scoring (stays at root, D-04)
    score, component_scores = musicality_score.get_musicality_score(final_wav)
    musicality = {"score": float(score), "components": {k: float(v) for k, v in component_scores.items()}}

    # Step 7: annotation (pure function)
    song_params = SongParams(
        key=key, tempo=tempo, time_signature_base=song_signatures.get("verse", "4/4"),
        time_signature_variation=1.0, swing_amount=swing_amount,
        signatures_per_part=song_signatures, measures_per_part=measures,
        song_unique_parts=song_unique_parts, song_arrangement=song_arrangement,
    )
    annotation = annotator.annotate(
        song_params=song_params,
        render_results=render_results,
        mix_results=mix_results,
        beat_times=beat_times_dict,
        downbeat_times=downbeat_times_dict,
        musicality=musicality,
        fluidsynth_version=renderer.FLUIDSYNTH_VERSION,
    )

    # Step 8: write JSON (Phase 5 writer takes over this lifecycle)
    json_file = os.path.join(name, name + '.json')
    with open(json_file, 'w') as outfile:
        json.dump(annotation, outfile, indent=4)

    return annotation
```

**Logger + __main__ guard must survive collapse** (`music_gen.py:41,515-523`):
```python
logger = logging.getLogger(__name__)  # line 41 — must remain for test_music_gen_logging.py

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    cfg = config.Config.load()
    for i in range(1):
        generate_song(i, cfg)
```

---

### `pyproject.toml` — add mido + pytest markers (RESEARCH findings)

**Modification 1: add mido to runtime dependencies** (RESEARCH.md — mido not installed, not a transitive dep):
```toml
# Add to [project] dependencies list:
"mido>=1.3.3",
```

**Modification 2: add markers to pytest ini_options** (RESEARCH.md §Dependency Verification 5):
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
markers = [
    "slow: FluidSynth rendering tests — requires fluidsynth binary on PATH (deselect with '-m not slow')",
    "integration: end-to-end tests requiring system dependencies",
]
```

---

### `tests/test_renderer.py` (test, request-response)

**Analog:** `tests/test_generators/test_beat.py` (seeded-determinism structure + tmp_path/monkeypatch)

**Test scaffold pattern** (`test_beat.py:1-22`):
```python
"""Renderer tests (R-X4): RenderResult assembly + FLUIDSYNTH_VERSION capture.

FluidSynth subprocess is mocked — unit tests do NOT require the binary.
The E2E integration test (test_integration_full_generation.py) covers the real binary.
"""
from __future__ import annotations

import os
import random
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from musicgen.renderer import (
    FLUIDSYNTH_VERSION,
    RenderResult,
    pick_soundfonts,
    render_stems,
)
```

**Mocked FluidSynth pattern** (D-28; no existing analog — use unittest.mock):
```python
def _make_fake_wav(path: str, duration_ms: int = 1000, sample_rate: int = 44100):
    """Create a minimal WAV file for testing without FluidSynth."""
    from pydub import AudioSegment
    AudioSegment.silent(duration=duration_ms, frame_rate=sample_rate).set_channels(2).export(path, format="wav")

@pytest.fixture
def mock_fluid_synth(tmp_path):
    """Patch FluidSynth.midi_to_audio to create a fake WAV instead of calling FluidSynth."""
    def _fake_render(self, midi_path, wav_path):
        _make_fake_wav(wav_path)

    with patch("musicgen.renderer.FluidSynth.midi_to_audio", _fake_render):
        yield
```

**Seeded-determinism test pattern** (`test_beat.py:41-60`):
```python
class TestRenderStems:
    def test_render_stems_returns_render_result(self, tmp_path, mock_fluid_synth, ...):
        ...

    def test_fluidsynth_version_capture(self):
        # Must be a string, never raises at import
        assert isinstance(FLUIDSYNTH_VERSION, str)
        # On CI without FluidSynth binary the fallback must be "unknown"
        # (or a valid version string if binary present)
```

---

### `tests/test_mixer.py` (test, transform)

**Analog:** `tests/test_generators/test_bassline.py` (seeded-determinism + bytes-equal pattern) + `tests/test_sampler.py` (frozen dataclass assertions)

**Test scaffold pattern** (`test_bassline.py:1-16`):
```python
"""Mixer tests (R-X5): seeded-RNG determinism + silent-stem channel parity + D-11 guard.

Tests chdir into tmp_path and use in-memory AudioSegments as stem fixtures.
"""
from __future__ import annotations

import os
import random

import pytest
from pydub import AudioSegment

from musicgen.mixer import (
    MixResult,
    build_fx_boards,
    compute_layer_mask,
    mix_part,
    concat_parts,
)
```

**Seeded-determinism test pattern** (`test_bassline.py:20-37`):
```python
class TestBuildFxBoardsDeterministic:
    @pytest.mark.parametrize("seed", [0, 42, 12345])
    def test_same_seed_same_fx_params(self, seed):
        boards_a = build_fx_boards(rng=random.Random(seed))
        boards_b = build_fx_boards(rng=random.Random(seed))
        # pedalboard_info_json output must be equal for same seed
        from musicgen.mixer import pedalboard_info_json
        for layer in ("beat", "melody", "harmony", "bassline"):
            assert pedalboard_info_json(boards_a[layer]) == pedalboard_info_json(boards_b[layer])
```

**Silent-stem stereo assertion** (RESEARCH.md Pitfall 1 guard):
```python
def test_silent_stem_channels(tmp_path):
    """D-12: silent stems must be stereo 44100 Hz to match FluidSynth render output."""
    from musicgen.mixer import _make_silent_stem
    seg = _make_silent_stem(duration_ms=1000)
    assert seg.channels == 2
    assert seg.frame_rate == 44100
```

---

### `tests/test_annotator.py` (test, transform)

**Analog:** `tests/test_sampler.py` (fixture-driven pure-function tests; frozen dataclass assertions)

**Test scaffold pattern** (`test_sampler.py:1-35`):
```python
"""Annotator tests (R-X6): fixture-driven pure-function contract.

Builds a handcrafted SongParams + synthetic RenderResult + MixResult + beat/downbeat
lists + stub musicality dict → calls annotate() → asserts golden dict shape.
No file I/O inside the annotator (D-14).
"""
from __future__ import annotations

import pytest

from musicgen.sampler import SongParams
from musicgen.renderer import RenderResult
from musicgen.mixer import MixResult
from musicgen.annotator import annotate
```

**Fixture pattern** (`test_sampler.py:44-84` SongParams shape assertions):
```python
@pytest.fixture
def minimal_song_params():
    return SongParams(
        key="Am", tempo=120, time_signature_base="4/4",
        time_signature_variation=1.0, swing_amount=0.5,
        signatures_per_part={"intro": "4/4"}, measures_per_part={"intro": 1},
        song_unique_parts=["intro"], song_arrangement=["intro"],
    )

@pytest.fixture
def minimal_render_result():
    return RenderResult(
        stem_paths={"beat": "/tmp/beat.wav", "melody": "/tmp/melody.wav",
                    "harmony": "/tmp/harmony.wav", "bassline": "/tmp/bassline.wav"},
        sample_rate=44100, channels=2, duration_seconds=2.0,
        fluidsynth_version="FluidSynth runtime version 2.3.4",
    )
```

**TBD-fields-are-None assertion** (D-16):
```python
def test_tbd_fields_are_none(minimal_song_params, minimal_render_result, minimal_mix_result):
    result = annotate(
        minimal_song_params, {"intro": minimal_render_result},
        {"intro": minimal_mix_result}, {"intro": []}, {"intro": []},
        {"score": 0.5, "components": {}},
        fluidsynth_version="unknown",
    )
    assert result["seed"] is None
    assert result["musicgen_version"] is None
    assert result["split"] is None
    assert result["pre_roll_offset_seconds"] is None
    assert "analysis_failed" not in result  # omitted on success (D-16)
```

---

### `tests/test_beats.py` (test, transform)

**Analog:** `tests/test_generators/test_beat.py` (per-function determinism + parametrize over swing values)

**Test scaffold pattern** (`test_beat.py:1-22`):
```python
"""Beats tests (R-X7): extract_beat_times + extract_downbeat_times + swing cases.

Swing test cases: 0.5 (straight), 0.66 (moderate), 0.75 (heavy).
Uses midiutil to synthesize known MIDI fixtures in tmp_path.
"""
from __future__ import annotations

import random
from pathlib import Path

import pytest
from midiutil import MIDIFile

from musicgen.beats import extract_beat_times, extract_downbeat_times, beat_duration
from musicgen.generators.beat import generate_beat
```

**Grid assertion pattern** (`test_beat.py:25-38` beat_duration assertions):
```python
def test_extract_beat_times_44_at_120bpm(tmp_path, monkeypatch):
    """4/4 at 120 BPM → beats at [0.0, 0.5, 1.0, 1.5] for one measure of 4 kicks."""
    monkeypatch.chdir(tmp_path)
    # ... synthesize a 4-kick MIDI using midiutil ...
    result = extract_beat_times(str(midi_path), tempo=120, start_offset_seconds=0.0)
    assert result == pytest.approx([0.0, 0.5, 1.0, 1.5], abs=0.01)
```

**Swing cases parametrize pattern** (`test_beat.py:42-60`):
```python
@pytest.mark.parametrize("swing_amount", [0.5, 0.66, 0.75])
def test_swing_off_beats_later_than_straight(tmp_path, monkeypatch, swing_amount):
    monkeypatch.chdir(tmp_path)
    # generate_beat with seeded rng to get deterministic MIDI
    mid_path, _ = generate_beat(
        part="verse", tempo=120, time_signature="4/4", measures=2,
        name="song-verse", swing_amount=swing_amount, rng=random.Random(42),
    )
    times = extract_beat_times(str(mid_path), tempo=120, start_offset_seconds=0.0)
    assert times == sorted(times), "beat times must be monotonic"
    # For swing > 0.5, off-beat (odd-index) times must be later than straight grid
    if swing_amount > 0.5:
        straight_grid = [i * 0.5 for i in range(len(times))]
        # off-beats (odd indices) are swung later; on-beats (even) are straight
        for i in range(1, len(times), 2):
            if i < len(straight_grid):
                assert times[i] > straight_grid[i] - 0.001
```

**Downbeat count assertion** (D-20):
```python
def test_downbeat_count_equals_measures(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    measures = 3
    # ... generate MIDI with known measures ...
    beat_times = extract_beat_times(...)
    downbeats = extract_downbeat_times(beat_times, "4/4", measures, 0.0, 120)
    assert len(downbeats) == measures  # one per measure, not beat_times slice
```

---

### `tests/test_no_bare_random_in_package.py` (test, static guard)

**Analog:** `tests/test_generators/test_no_bare_random.py` (parametrize pattern) + `tests/test_sampler.py:165-196` (_bare_random_calls helper)

**Full pattern** (`test_no_bare_random.py:1-44`):
```python
"""Static guard: zero bare random.<method>() in src/musicgen/**/*.py (D-17/D-31).

Extended from test_generators/test_no_bare_random.py to cover the full package.
Parametrized over every *.py in src/musicgen/ (including sub-packages) excluding
__init__.py, so adding any new module auto-extends the guard.
"""
import ast
import glob
import os

import pytest

PACKAGE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, "src", "musicgen")
)


def _bare_random_calls(source: str):
    """Return random.<attr>(...) Call nodes, excluding the random.Random constructor.

    Source: tests/test_sampler.py:_bare_random_calls (identical helper).
    The random.Random() constructor is the RNG factory, not a bare draw (D-17).
    """
    tree = ast.parse(source)
    hits = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "random"
                and node.func.attr != "Random"):  # random.Random() constructor is PERMITTED
            hits.append(node)
    return hits


@pytest.mark.parametrize("path", sorted(
    p for p in glob.glob(os.path.join(PACKAGE_DIR, "**", "*.py"), recursive=True)
    if not p.endswith("__init__.py")
))
def test_no_bare_random_in_package_module(path):
    with open(path, "r") as f:
        source = f.read()
    hits = _bare_random_calls(source)
    assert hits == [], (
        f"{os.path.basename(path)}: {len(hits)} bare random.<method>() at lines "
        f"{[n.lineno for n in hits]} — use rng.<method>() per D-17."
    )
```

---

### `tests/test_integration_full_generation.py` (test E2E, request-response)

**No existing analog** — first E2E test in the suite.

**@pytest.mark.slow marker pattern** (from RESEARCH.md STACK.md reference; pyproject.toml markers now declared):
```python
"""Integration test (R-X8): full pipeline smoke test with real FluidSynth binary.

Marked @pytest.mark.slow — skipped when FluidSynth is not on PATH.
Run with: pytest tests/test_integration_full_generation.py -m slow -x
Skip in CI: pytest -m "not slow"
"""
import shutil
import random

import pytest

# Guard: skip entire module if FluidSynth binary is absent (D-30)
fluidsynth_available = shutil.which("fluidsynth") is not None


@pytest.mark.slow
@pytest.mark.skipif(not fluidsynth_available, reason="fluidsynth binary not on PATH")
class TestFullGenerationPipeline:
    def test_one_part_smoke(self, tmp_path, monkeypatch):
        """One-part 1-measure 4/4 song through full pipeline; seed=42."""
        monkeypatch.chdir(tmp_path)
        rng = random.Random(42)
        # Build minimal SongParams, call generators → renderer → mixer → beats → annotator
        # Assert: 4 stem WAVs + 1 mix WAV + 4 MIDI files exist
        # Assert: annotation dict has all Phase-4 filled fields non-None
        # Assert: annotation dict has TBD fields as None
        # MIDI is bit-identical across two runs with same seed (WAV golden test is Phase 5)
        ...
```

---

## Shared Patterns

### RNG Threading (D-17/D-18)
**Source:** `src/musicgen/sampler.py:27-48` and `src/musicgen/generators/beat.py:58-67`
**Apply to:** All new modules with random draws — `renderer.pick_soundfonts`, `mixer.build_fx_boards`, `mixer._create_effect`, `mixer.compute_layer_mask`

Every function that draws random values takes an explicit `rng: random.Random` parameter. The module-level `_rng = random.Random()` in `music_gen.py:46` is the single threaded RNG for Phase 4. `random.Random()` constructor calls are permitted; all other `random.*` method calls are forbidden under `src/musicgen/`.

```python
# PERMITTED (module-level factory in music_gen.py — the single source):
_rng = random.Random()

# PERMITTED (rng-threaded draw):
rng.choice(sf2_files)
rng.random()
rng.uniform(lo, hi)

# FORBIDDEN anywhere under src/musicgen/:
random.choice(...)  # bare draw
random.random()     # bare draw
random.uniform(...) # bare draw
```

### Cfg-with-fallback (D-25)
**Source:** `src/musicgen/generators/beat.py:85`; `src/musicgen/sampler.py:269`
**Apply to:** All new module functions that accept `cfg`

```python
_cfg = cfg if cfg is not None else config.Config()
```

### Module-level logger
**Source:** `src/musicgen/sampler.py:24`; `src/musicgen/generators/beat.py:29`; `music_gen.py:41`
**Apply to:** All new modules

```python
logger = logging.getLogger(__name__)
```

Logging level semantics (Phase 2 D-07, carry forward):
- `logger.debug(...)` — internal state dumps (pedalboard params, per-beat timestamps)
- `logger.info(...)` — part-by-part progress milestones (rendering part X of N, mix complete)
- `logger.warning(...)` — recoverable oddities (FLUIDSYNTH_VERSION capture failed, soundfont pool thin)
- `logger.error(...)` — failures (not used at module load — never raise at import)

### Frozen Dataclass Output Contract (D-02)
**Source:** `src/musicgen/sampler.py:223-293` (SongParams)
**Apply to:** `RenderResult` in `renderer.py`; `MixResult` in `mixer.py`

```python
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class ResultType:
    """Docstring with D-reference and R-X reference.

    D-02: frozen dataclass.
    """
    field_name: FieldType
    ...
```

### Seeded-determinism test class structure
**Source:** `tests/test_sampler.py:44-65`; `tests/test_generators/test_bassline.py:19-48`
**Apply to:** All new test files with RNG-bearing functions

```python
@pytest.mark.parametrize("seed", [0, 42, 12345])
def test_same_seed_same_output(self, seed, ...):
    result_a = function_under_test(..., rng=random.Random(seed))
    result_b = function_under_test(..., rng=random.Random(seed))
    assert result_a == result_b
```

### Google-style docstrings with Args/Returns
**Source:** `src/musicgen/generators/beat.py:58-82`; `src/musicgen/sampler.py:162-204`
**Apply to:** All new public functions

```python
def function_name(param: Type, ...) -> ReturnType:
    """One-line summary (R-X ref).

    Extended description referencing the D-decision that governs this function.

    Args:
        param: Description.
        rng: Injected random.Random (required; D-17).
        cfg: Optional Config override (D-25 fallback pattern).

    Returns:
        Description.
    """
```

### AST bare-random guard helper (_bare_random_calls)
**Source:** `tests/test_sampler.py:165-180`; `tests/test_generators/test_no_bare_random.py:18-29`
**Apply to:** `tests/test_no_bare_random_in_package.py`

The `node.func.attr != "Random"` exclusion is critical — it permits `random.Random()` constructor calls. Copy verbatim from either source.

---

## No Analog Found

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| `tests/test_integration_full_generation.py` | test (E2E) | request-response | No E2E test exists in the suite yet; first `@pytest.mark.slow` integration test. Use `shutil.which("fluidsynth")` guard + `@pytest.mark.skipif` per RESEARCH.md D-30 pattern. |

---

## Deleted Files (no pattern work needed)

| File | Reason | Verification |
|------|--------|-------------|
| `beat_anotator.py` | D-03: zero importers; logic moves to `beats.py`. | RESEARCH.md Pitfall 7: `grep -r "beat_anotator" --include="*.py"` returns only self-reference. Safe to delete; 371-test baseline unaffected. |

---

## Metadata

**Analog search scope:** `/home/bidu/musicgen/src/musicgen/`, `/home/bidu/musicgen/tests/`, `/home/bidu/musicgen/music_gen.py`, `/home/bidu/musicgen/beat_anotator.py`
**Files scanned:** 10 source files read in full
**Pattern extraction date:** 2026-04-19

**Critical non-obvious findings carried into patterns:**
1. `AudioSegment.silent()` defaults to MONO 11025 Hz — `_make_silent_stem` must call `.set_channels(2)` with `frame_rate=44100` (RESEARCH.md Pitfall 1, verified).
2. D-20 stride-slice downbeat approach is WRONG for 4/4 patterns with zero slots and 12/8 — `extract_downbeat_times` must use the time-grid approach with `TimeSignatureRegistry.lookup().numerator * beat_slot_s` (RESEARCH.md Pitfall 2, verified against all 6 pattern files).
3. `mido` is NOT installed and is NOT a transitive dep of `midi2audio` — must add `"mido>=1.3.3"` to `pyproject.toml` before any Wave 1 work (RESEARCH.md Dependency Verification 1).
4. `@pytest.mark.slow` and `@pytest.mark.integration` are NOT declared in `pyproject.toml` yet — add `markers` list before Wave 6 (RESEARCH.md Dependency Verification 5).
5. FX must be applied to ALL 4 layers unconditionally BEFORE overlay masking (D-11) — do not move `apply_fx_to_layer` inside the `if layer_mask[layer]` branch.
