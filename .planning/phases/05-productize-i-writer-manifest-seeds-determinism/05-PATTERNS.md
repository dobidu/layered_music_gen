# Phase 5: Productize I — writer, manifest, seed discipline, determinism - Pattern Map

**Mapped:** 2026-04-19
**Files analyzed:** 17 (8 source + 9 test/fixture)
**Analogs found:** 15 exact / 2 no-analog (manifest stdlib, musicality move)

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `src/musicgen/seeds.py` | utility (pure functions + context manager) | transform | `src/musicgen/beats.py` | exact (same module shape: `from __future__ annotations`, stdlib-only, module-level `logger`, pure helpers returning primitives) |
| `src/musicgen/writer.py` | service (file I/O + validator) | batch + transform | `src/musicgen/mixer.py` (`concat_parts` + `_make_silent_stem` + `mix_part`) | exact (pydub AudioSegment concat, `os.makedirs` + write, MixResult consumer) |
| `src/musicgen/manifest.py` | service (append-under-lock) | event-driven | `src/musicgen/config.py` (`Config` class with stdlib I/O + optional constructor arg) | role-match (no prior manifest/lock abstraction in codebase) |
| `src/musicgen/api.py` | orchestrator / library entry point | request-response | `music_gen.py` (`create_song` + `generate_song_parts` + `generate_song` — being deleted) | exact (source-to-replace: body moves into `api.generate`) |
| `src/musicgen/musicality.py` | service (module move, no edits) | request-response | `src/musicgen/duration_validator.py` (moved in Plan 03-02 from `enhanced_duration_validator.py` via `git mv`) | exact precedent (identical refactor shape) |
| `src/musicgen/__init__.py` | config (package init) | — | `src/musicgen/__init__.py` (current empty shell) | exact (rewrite on existing file) |
| `music_gen.py` | config (smoke-test wrapper) | request-response | `music_gen.py` `if __name__ == "__main__"` block (lines 192-199) | exact (stays; body above gets deleted) |
| `config.py` | config (add 7 fields + env var) | — | `config.py` existing `Config` dataclass + `load()` precedence | exact (extend, don't rewrite) |
| `tests/conftest.py` | test (pytest plugin hook) | — | *(no existing conftest.py — first one since Phase 3 deleted the old)* | no-analog (stdlib pytest_addoption pattern) |
| `tests/test_seeds.py` | test (pure function) | transform | `tests/test_sampler.py` (`TestSamplerFreeFunctions` — seeded-determinism pure-function pattern) | exact |
| `tests/test_writer.py` | test (file I/O with tmp_path) | batch | `tests/test_mixer.py` (`TestMixPart`, `TestFxAppliedToAllLayers` — synthesized WAV fixtures + tmp_path) | exact |
| `tests/test_manifest.py` | test (threading correctness) | event-driven | `tests/test_mixer.py` (tmp_path I/O) + stdlib `threading.Thread` idiom | role-match |
| `tests/test_split.py` | test (pure hash + empirical ratios) | transform | `tests/test_sampler.py` (pure-function determinism + many-seed aggregate test) | role-match |
| `tests/test_api.py` | test (integration, mixed fast/slow) | request-response | `tests/test_integration_full_generation.py` (@pytest.mark.slow + fluidsynth-skip-gate pattern) | exact |
| `tests/test_determinism_golden.py` | test (golden SHA-256 regression) | transform | `tests/test_integration_full_generation.py::TestMidiReproducibility::test_same_seed_produces_same_midi` (bit-identity under seed) | exact |
| `tests/fixtures/determinism/` | test fixture | — | none (new fixture directory) | no-analog |
| `pyproject.toml` | config | — | — | exact (verify, no edits) |

## Pattern Assignments

### `src/musicgen/seeds.py` (utility, pure functions + context manager)

**Analog:** `src/musicgen/beats.py`

**Imports pattern** (beats.py lines 17-26 — match shape exactly):
```python
"""Beats module — MIDI-tick beat and downbeat extraction (R-X7).
...
"""
from __future__ import annotations

import logging
from typing import List

import mido

from timesig import TimeSignatureRegistry

logger = logging.getLogger(__name__)
```

**For `seeds.py`, mirror exactly:**
```python
"""Seeds module — pure-function RNG hierarchy (R-P7, D-17/D-18/D-20).
...
"""
from __future__ import annotations

import contextlib
import hashlib
import logging
import random
from typing import Dict

logger = logging.getLogger(__name__)

RNG_PARAMS = "params"
RNG_GENERATORS = "generators"
RNG_SOUNDFONTS = "soundfonts"
RNG_FX = "fx"
RNG_MIX = "mix"
```

**Pure-function contract** (beats.py:29-44 — `beat_duration` is the closest sibling: pure primitive in/primitive out, no I/O, short docstring with Args/Returns):
```python
def beat_duration(signature: str, tempo: int) -> float:
    """Return the duration of one beat slot in seconds for (signature, tempo).
    ...
    Args:
        signature: Time signature string like ``"4/4"`` or ``"6/8"``.
        tempo: BPM, integer.

    Returns:
        Duration of one beat slot in seconds (float).
    """
    numerator, denominator = map(int, signature.split('/'))
    beat_length = 60 / tempo
    return beat_length * (4 / denominator)
```

Copy this docstring/signature shape for `derive_sample_seed`, `assign_split`. Body verbatim from CONTEXT D-17, D-18, D-26 and RESEARCH.md Pattern 1 (lines 177-228).

**Context manager pattern** — no in-repo analog; follow RESEARCH.md Pattern 1 code block (`@contextlib.contextmanager` decorator with try/finally) verbatim per CONTEXT D-20.

**AST-guard compliance:** No bare `random.<method>` calls. Only `random.Random(seed_xor_const)` constructors — **permitted** by `tests/test_no_bare_random_in_package.py:39` (`node.func.attr != "Random"`). `save_random_state()` uses `random.getstate()` / `random.setstate(state)` — these ARE bare `random.*` calls, but `getstate`/`setstate` are not `choice/random/randint/choices/uniform` (the grep-spot-check list in CONTEXT D-42). The AST guard blocks ALL non-Random attrs though; **planner action:** either widen the AST guard exclusion list OR document that `seeds.py` is the sole intentional exception via a `# noqa`-style sentinel. Recommended: widen `tests/test_no_bare_random_in_package.py` to also permit `getstate`/`setstate` (this is a minor plan edit scoped to that test file).

---

### `src/musicgen/writer.py` (service, batch + transform)

**Analog:** `src/musicgen/mixer.py` (multiple helpers)

**Imports pattern** (mixer.py lines 30-47):
```python
from __future__ import annotations

import json
import logging
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

from pedalboard import ...
from pydub import AudioSegment

import config
from musicgen.renderer import RenderResult

logger = logging.getLogger(__name__)
```

**For `writer.py`, mirror:**
```python
from __future__ import annotations

import copy
import json
import logging
import os
import shutil
from typing import Dict, List, Tuple

import mido
import numpy as np
import scipy.io.wavfile as wf
from pydub import AudioSegment

logger = logging.getLogger(__name__)
```

**Core stem-concat pattern** (mixer.py:418-446 — `concat_parts`). **Copy the `AudioSegment.from_wav + AudioSegment.from_wav` idiom verbatim** per CONTEXT D-06:
```python
def concat_parts(part_mix_paths: List[str], out_path: str) -> str:
    """Concatenate per-part mix WAVs into one final mix WAV.
    ...
    Raises:
        ValueError: if ``part_mix_paths`` is empty.
    """
    if not part_mix_paths:
        raise ValueError("concat_parts: no part mix paths provided; cannot assemble final mix")

    song = AudioSegment.from_wav(part_mix_paths[0])
    for part_wav in part_mix_paths[1:]:
        song += AudioSegment.from_wav(part_wav)

    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    song.export(out_path, format='wav')
    return out_path
```

`_concat_layer_stems(stem_paths_per_part, layer, out_path)` (D-06) is **the same loop, one per layer** — duplicate the 4-line body inside a per-layer loop.

**Atomic-write + makedirs pattern** (mixer.py:343 + 441-443):
```python
os.makedirs(out_dir, exist_ok=True)  # at start of function
# ... build artifacts ...
parent = os.path.dirname(out_path)
if parent:
    os.makedirs(parent, exist_ok=True)
```

**Error raising pattern** (mixer.py:181, 433):
```python
if rng is None:
    raise ValueError("build_fx_boards requires an injected rng (D-17)")
# ... and ...
if not part_mix_paths:
    raise ValueError("concat_parts: no part mix paths provided; cannot assemble final mix")
```
Writer's `_assert_sum_of_stems` uses the same bare-raise style per CONTEXT D-24 and RESEARCH Open Question #5 (raise `AssertionError` on sum-of-stems violation; api.py catches).

**MIDI concat pattern** — no in-repo analog; follow RESEARCH.md lines 497-554 (absolute-tick walk via `mido.MidiFile` read + `mido.MidiTrack` write) verbatim. This is the subtlest code in the phase (RESEARCH Pitfall 1).

**Sum-of-stems assertion pattern** — no in-repo analog; follow RESEARCH.md lines 556-584 (`scipy.io.wavfile.read` → `int32` cast → element-wise sum → `np.max(np.abs(...)) / 32768.0`). Mandatory int32 promotion per RESEARCH Pitfall 2.

**Atomic sentinel (rename) pattern** — no in-repo analog; follow RESEARCH.md Pattern 2 lines 286-290 (`open(<tmp>, "w")` → `json.dump(..., sort_keys=True, indent=2, separators=(",", ": "))` → `os.rename(<tmp>, <final>)`). D-23 pins the JSON args for byte-stability.

**Path-rewrite deep-copy pattern** — follow CONTEXT D-12 (deep-copy annotation dict before rewriting `mix`, `stems`, `midi` to per-sample-dir-relative). Use `copy.deepcopy(annotation)`; then `final_annotation["mix"] = "mix.wav"`, `final_annotation["stems"] = {l: f"stems/{l}.wav" ...}`, `final_annotation["midi"] = {l: f"midi/{l}.mid" ...}`.

---

### `src/musicgen/manifest.py` (service, event-driven under lock)

**Analog:** `src/musicgen/config.py` (closest: class with stdlib I/O, optional constructor arg)

**No in-repo analog for locks or JSONL append.** Follow RESEARCH.md Pattern 3 (lines 302-328) verbatim. The critical primitives:

**Class-with-optional-arg pattern** (config.py:46-58 — `@dataclass` + `field(default_factory=...)`):
```python
@dataclass
class Config:
    project_root: str = DEFAULT_PROJECT_ROOT
    sf_dir: str = DEFAULT_SF_DIR
    # ...
```
For `ManifestWriter`, **do NOT use `@dataclass`** — the `lock` parameter's default (`threading.Lock()`) must be constructed per-instance (mutable default pitfall). Use `__init__` explicitly per RESEARCH.md Pattern 3.

**Imports** (follow RESEARCH.md Pattern 3):
```python
from __future__ import annotations

import json
import logging
import os
import threading
from typing import ContextManager, Optional

logger = logging.getLogger(__name__)
```

**Append-under-lock core** (RESEARCH.md Pattern 3 verbatim):
```python
def append(self, entry: dict) -> None:
    os.makedirs(self.dataset_root, exist_ok=True)
    line = json.dumps(entry, sort_keys=True) + "\n"
    with self.lock:
        with open(self.manifest_path, "a") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
```
`sort_keys=True` is mandatory for byte-stability (CONTEXT D-15). `os.fsync` on `f.fileno()` per RESEARCH A8 (POSIX `O_APPEND` atomicity).

**Sentinel-only `is_sample_complete` staticmethod** (RESEARCH.md Pattern 3 verbatim) — checks `os.path.exists(<sample_dir>/sample.json)` only; **never** reads the manifest (CONTEXT D-16: manifest is a projection, not the truth).

---

### `src/musicgen/api.py` (orchestrator, request-response)

**Analog:** `music_gen.py:create_song` (lines 51-140) + `music_gen.py:generate_song_parts` (lines 143-168) + `music_gen.py:generate_song` (lines 170-189). **This is source-to-replace** per CONTEXT D-34.

**Imports pattern** (music_gen.py lines 1-15 — extract and dedupe into `api.py`):
```python
from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import logging
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import config
from musicgen.sampler import (
    SongParams, generate_random_key, generate_random_tempo,
    generate_random_time_signature, generate_random_swing,
    generate_song_measures, generate_song_arrangement, validate_measures_dict,
)
from musicgen.generators.chord import generate_chord_progression
from musicgen.generators.melody import generate_melody
from musicgen.generators.bassline import generate_bassline
from musicgen.generators.beat import generate_beat
from musicgen import renderer, mixer, annotator, beats, writer, musicality
from musicgen.seeds import (
    derive_sample_seed, make_rngs, assign_split, save_random_state,
    RNG_PARAMS, RNG_GENERATORS, RNG_SOUNDFONTS, RNG_FX, RNG_MIX,
)
from musicgen.manifest import ManifestWriter

logger = logging.getLogger(__name__)
```

**Orchestration core** — adapt from `music_gen.py:create_song` lines 56-140 with **three mechanical substitutions**:

1. Replace `_cfg = cfg if cfg is not None else config.Config()` with `_cfg = config` (already a `Config` — api takes `Config` not an override dict).
2. Replace every `_rng` with the appropriate `rngs[RNG_XXX]` per CONTEXT D-19:
   - `generate_song_arrangement(_rng, ...)` → `generate_song_arrangement(rngs[RNG_PARAMS], ...)`
   - `renderer.pick_soundfonts(_cfg, _rng)` → `renderer.pick_soundfonts(_cfg, rngs[RNG_SOUNDFONTS])`
   - `generate_chord_progression(..., _rng)` → `generate_chord_progression(..., rngs[RNG_GENERATORS])`
   - `generate_melody(..., _rng)` → `generate_melody(..., rngs[RNG_GENERATORS])`
   - `generate_bassline(..., _rng)` → `generate_bassline(..., rngs[RNG_GENERATORS])`
   - `generate_beat(..., _rng, ...)` → `generate_beat(..., rngs[RNG_GENERATORS], ...)`
   - `mixer.build_fx_boards(_cfg, _rng)` → `mixer.build_fx_boards(_cfg, rngs[RNG_FX])`
   - `mixer.compute_layer_mask(..., _rng)` → `mixer.compute_layer_mask(..., rngs[RNG_MIX])`
   - Sampler params calls in `generate_song` (key, tempo, time_signature, swing, measures) → `rngs[RNG_PARAMS]`
3. Wrap musicality scoring in `with save_random_state():` (CONTEXT D-20 defense-in-depth):
   ```python
   with save_random_state():
       score, component_scores = musicality.get_musicality_score(final_wav)
   ```

**Inner MIDI-generation helper pattern** (music_gen.py:143-168 → `api._generate_all_midi` per CONTEXT D-34):
```python
def _generate_all_midi(
    rngs: Dict[str, random.Random], key: str, tempo: int,
    song_signatures: Dict[str, str], song_measures: Dict[str, int],
    name: str, chord_pat_file: str, swing_amount: float,
    cfg: config.Config,
) -> Tuple[Dict, Dict, Dict, Dict, Dict, Dict]:
    """Per-part MIDI generation — extracted from music_gen.py:143-168."""
    # Body verbatim from music_gen.py with _rng → rngs[RNG_GENERATORS].
```

**`Config` + `SampleResult` dataclass pattern** — analog is `RenderResult` (renderer.py:70-96) and `MixResult` (mixer.py:271-295):
```python
@dataclass(frozen=True)
class RenderResult:
    """Per-part stem render outputs (R-X4).
    ...
    Attributes:
        stem_paths: Dict mapping layer name ... -> absolute path...
        sample_rate: ...
    """
    stem_paths: Dict[str, str]
    sample_rate: int
    channels: int
    duration_seconds: float
    fluidsynth_version: str
```
For `SampleResult`, copy this shape — `@dataclass(frozen=True)` with 11 fields per CONTEXT D-02. **Note:** CONTEXT D-02 says `slots=True`, RESEARCH CONVENTIONS note says existing Phase 4 dataclasses (`RenderResult`, `MixResult`) do NOT use slots — planner picks one per Open Question in CONTEXT "Claude's Discretion". Consistency with Phase 4 wins unless a reason to deviate emerges.

**`importlib.metadata.version` + fallback pattern** — no in-repo analog; follow RESEARCH Pitfall 4 (lines 449-455) verbatim:
```python
try:
    MUSICGEN_VERSION = importlib.metadata.version("musicgen")
except importlib.metadata.PackageNotFoundError:
    MUSICGEN_VERSION = "0.1.0+uninstalled"
```

**`tempfile.mkdtemp` + `shutil.rmtree(..., ignore_errors=True)` pattern** — no in-repo analog; follow RESEARCH.md System Architecture Diagram lines 120-123.

**Input validation pattern** (mixer.py:180-181 — raise with explicit reason):
```python
if rng is None:
    raise ValueError("build_fx_boards requires an injected rng (D-17)")
```
For `api.generate`: `if config.global_seed is None: raise ValueError("global_seed is required for deterministic generation; pass config.global_seed explicitly")` per CONTEXT D-21.

---

### `src/musicgen/musicality.py` (service, module move)

**Analog:** `src/musicgen/duration_validator.py` (moved in Phase 3 Plan 03-02 from `enhanced_duration_validator.py` via `git mv`)

**Precedent command** (from `.planning/phases/03.../03-02-PLAN.md` line 148-152):
```bash
cd /home/bidu/musicgen && git mv enhanced_duration_validator.py src/musicgen/duration_validator.py
```

**For Phase 5, mirror exactly:**
```bash
cd /home/bidu/musicgen && git mv musicality_score.py src/musicgen/musicality.py
```

**Import-site rewrite pattern** (from 03-02-PLAN.md line 157: replace `from enhanced_duration_validator import ...` with `from musicgen.duration_validator import ...`):
For Phase 5, the only import site is `music_gen.py:3` (`import musicality_score, config`). This import **goes away** with CONTEXT D-34's `create_song` deletion; api.py introduces `from musicgen import musicality` (never `import musicality_score`).

**Verification pattern** (03-02-PLAN.md line 177):
```bash
grep -rn "musicality_score" --include="*.py" --exclude-dir=.venv --exclude-dir=__pycache__ --exclude-dir=.planning /home/bidu/musicgen/
```
Must return zero hits after Phase 5 (CONTEXT D-34 deletes the only caller; no back-compat shim per D-03).

**No back-compat shim** — 03-02-PLAN.md D-10 explicitly rejected the shim at `enhanced_duration_validator.py`; CONTEXT D-03 rejects one at `musicality_score.py` for the same reason.

---

### `src/musicgen/__init__.py` (config, rewrite)

**Analog:** existing `src/musicgen/__init__.py` (3 lines, empty shell — explicitly reserved for Phase 5 per line 2 comment: `# Phase 5 will add \`from musicgen.sampler import SongParams\` and the \`generate\` / \`generate_batch\` library entry points.`)

**Pattern to copy:** CONTEXT D-35 verbatim:
```python
"""musicgen package — library entry point (Phase 5, R-P12 single-sample)."""
from __future__ import annotations

import importlib.metadata

from musicgen.api import generate, Config, SampleResult

try:
    __version__ = importlib.metadata.version("musicgen")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.1.0+uninstalled"

__all__ = ["generate", "Config", "SampleResult", "__version__"]
```

---

### `music_gen.py` (config, smoke-test wrapper)

**Analog:** current `music_gen.py` lines 192-199 (the `if __name__ == "__main__":` block)

**Pattern to copy** — the `if __name__ == "__main__":` block stays (CONTEXT D-33); rewrite body from `generate_song(i, cfg)` to:
```python
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    from musicgen import generate, Config
    result = generate(Config(global_seed=1, sample_index=0))
    logger.info("Sample %d (seed=%d) written to %s — status=%s",
                result.sample_index, result.seed, result.sample_dir, result.status)
```

**Delete:** lines 51-189 (create_song, generate_song_parts, generate_song, `musicality_score` import). Keep the delegation helpers (lines 22-49) per CONTEXT D-34 ("stays for one more phase"). Keep the `_rng = random.Random()` line (20) or delete — planner's call; deleting is cleaner since no remaining caller uses `_rng` after D-34. Integration test migration (`test_integration_full_generation.py`) must land in the same plan.

---

### `config.py` (config, extend with 7 fields + env var)

**Analog:** existing `config.py` `Config` dataclass (lines 46-58) + `load()` (lines 68-100)

**Field-addition pattern** (config.py:47-58 — current fields):
```python
@dataclass
class Config:
    project_root: str = DEFAULT_PROJECT_ROOT
    sf_dir: str = DEFAULT_SF_DIR
    sf_layers: Tuple[str, ...] = DEFAULT_SF_LAYERS
    fx_files: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_FX_FILES))
    # ...
```

**Add these fields** (CONTEXT D-09, D-25, D-27):
```python
    dataset_root: str = field(default_factory=lambda: os.path.join(DEFAULT_PROJECT_ROOT, "dataset"))
    global_seed: Optional[int] = None
    sample_index: int = 0
    split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1)
    sum_of_stems_epsilon: float = 1e-3
    keep_working_dirs: bool = False
    workers: Optional[int] = None  # Phase 6 reserved
```

**Post-init validation pattern** — no existing in-repo analog on `config.Config` (the class is non-frozen `@dataclass`). Follow CONTEXT D-21 and D-27:
```python
    def __post_init__(self):
        # D-27: split ratios validation
        if abs(sum(self.split_ratios) - 1.0) > 1e-9:
            raise ValueError(
                f"split_ratios must sum to 1.0, got {sum(self.split_ratios)}"
            )
        if any(r < 0 for r in self.split_ratios):
            raise ValueError(f"split_ratios must be non-negative, got {self.split_ratios}")
```
NOTE: D-21's `global_seed is None` check lives in `api.generate`, not `Config.__post_init__` (a `Config()` with no seed is valid until you call `generate`).

**Env-var override pattern** (config.py:81-86 — the existing `MUSICGEN_SF_DIR` / `MUSICGEN_PROJECT_ROOT` block):
```python
sf_env = os.environ.get("MUSICGEN_SF_DIR")
if sf_env:
    cfg.sf_dir = os.path.abspath(sf_env)
```

**Copy and extend** for `MUSICGEN_DATASET_ROOT`:
```python
dataset_env = os.environ.get("MUSICGEN_DATASET_ROOT")
if dataset_env:
    cfg.dataset_root = os.path.abspath(dataset_env)
```

---

### `tests/conftest.py` (test, pytest plugin hook)

**Analog:** none in current repo (CONTEXT note: first conftest.py since Phase 3 deleted the old one).

**Pattern to copy** (RESEARCH.md Pattern 5 / CONTEXT D-32):
```python
"""Pytest plugin hooks for Phase 5 determinism testing (D-32)."""
from __future__ import annotations


def pytest_addoption(parser):
    parser.addoption(
        "--regen-goldens",
        action="store_true",
        default=False,
        help="Regenerate determinism fixtures (tests/fixtures/determinism/*.sha256) "
             "instead of asserting against them. Use when intentionally changing "
             "RNG order or FluidSynth version.",
    )
```

Minimal — one hook, no fixtures. Matches pytest 8.x convention.

---

### `tests/test_seeds.py` (test, pure function, transform)

**Analog:** `tests/test_sampler.py` — specifically `TestSamplerFreeFunctions` (lines 90-130) and the determinism-across-seeds pattern.

**Class-based pure-function test pattern** (test_sampler.py:90-98):
```python
class TestSamplerFreeFunctions:
    """Seeded-determinism contract for the 7 extracted sampler free functions."""

    @pytest.mark.parametrize("seed", [0, 1, 42])
    def test_generate_random_key_deterministic(self, seed):
        a = generate_random_key(random.Random(seed))
        b = generate_random_key(random.Random(seed))
        assert a == b
```

**For `test_seeds.py`, mirror:**
```python
from __future__ import annotations

import random
from typing import Dict

import pytest

from musicgen.seeds import (
    derive_sample_seed, make_rngs, save_random_state,
    RNG_PARAMS, RNG_GENERATORS, RNG_SOUNDFONTS, RNG_FX, RNG_MIX,
)


class TestDeriveSampleSeed:

    @pytest.mark.parametrize("global_seed,sample_index", [(0, 0), (1, 0), (42, 5), (42, 100)])
    def test_deterministic(self, global_seed, sample_index):
        a = derive_sample_seed(global_seed, sample_index)
        b = derive_sample_seed(global_seed, sample_index)
        assert a == b

    def test_different_indices_collide_less(self):
        """100 distinct indices for one global_seed produce 100 distinct seeds."""
        seeds = {derive_sample_seed(42, i) for i in range(100)}
        assert len(seeds) == 100, f"collision: got {len(seeds)} distinct seeds for 100 indices"


class TestMakeRngs:

    def test_five_domains(self):
        rngs = make_rngs(12345)
        assert set(rngs.keys()) == {RNG_PARAMS, RNG_GENERATORS, RNG_SOUNDFONTS, RNG_FX, RNG_MIX}

    def test_domain_independence(self):
        """Draws from different domains are uncorrelated (pair-wise corr < 0.1)."""
        # ... 1000 draws per domain, assert correlation coefficient
```

**Fixture-free, no-I/O pattern** (matches beats.py analog — pure functions need nothing but `random.Random`).

---

### `tests/test_writer.py` (test, file I/O with tmp_path)

**Analog:** `tests/test_mixer.py` — specifically `TestMixPart` fixtures (lines 184-199) + tmp_path pattern.

**Synthesized-fixture pattern** (test_mixer.py:184-198):
```python
@pytest.fixture
def fake_render_result(tmp_path):
    """Write 4 fake stereo-44.1kHz WAVs and build a RenderResult."""
    stem_paths = {}
    for layer in ("beat", "melody", "harmony", "bassline"):
        path = tmp_path / f"{layer}.wav"
        AudioSegment.silent(duration=500, frame_rate=44100).set_channels(2).export(str(path), format="wav")
        stem_paths[layer] = str(path)
    return RenderResult(
        stem_paths=stem_paths, sample_rate=44100, channels=2,
        duration_seconds=0.5, fluidsynth_version="test",
    )
```

**For `test_writer.py`, adapt to generate per-part per-layer WAVs + MIDIs:**
```python
@pytest.fixture
def fake_working_dir(tmp_path):
    """Write per-part per-layer WAVs + MIDIs that concatenate to a known mix."""
    working = tmp_path / "working"
    working.mkdir()
    # For each part in ["intro", "verse"]:
    #   Write 4 layer stems (one-second AudioSegment.silent stereo)
    #   Write 4 layer MIDIs (midiutil one-note-each or empty with end_of_track)
    # Return Dict[part, Dict[layer, str]]
```

**Synthesized-mix fixture for sum-of-stems** — **new pattern**: build four stems that sum cleanly to the mix (use `AudioSegment.silent` + a known tone per layer, overlay to produce the mix). Fault-injection test zeroes one stem to prove the assertion fires. See RESEARCH.md Code Examples "Sum-of-stems assertion" lines 556-584.

**Fault-injection pattern** (test_mixer.py:226-247 — `_counting_apply_fx` mock via `patch`):
```python
call_counter = {"n": 0}
original_apply_fx = apply_fx_to_layer

def _counting_apply_fx(wav_file, board):
    call_counter["n"] += 1
    return original_apply_fx(wav_file, board)

with patch("musicgen.mixer.apply_fx_to_layer", _counting_apply_fx):
    # ...
```

**For writer, fault-inject by swapping one stem file with a different-content WAV before calling `write_sample`, expect `AssertionError` from `_assert_sum_of_stems`.**

**Sentinel-order test** — read CONTEXT D-04 / RESEARCH.md Pattern 2. Mock `_assert_sum_of_stems` to raise, then assert `<sample_dir>/sample.json` does NOT exist (other files may).

**R-relative path test** — after `write_sample`, `json.load(sample.json)` must have `mix == "mix.wav"` (not an absolute path), `stems == {"beat": "stems/beat.wav", ...}`, `midi == {"beat": "midi/beat.mid", ...}` per CONTEXT D-11.

---

### `tests/test_manifest.py` (test, threading correctness)

**Analog:** `tests/test_mixer.py` (tmp_path I/O pattern) + stdlib `threading.Thread`.

**Imports** (copy test_mixer.py:8-16 tmp_path pattern):
```python
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import pytest

from musicgen.manifest import ManifestWriter
```

**tmp_path single-append pattern:**
```python
class TestAppend:
    def test_single_append(self, tmp_path):
        mw = ManifestWriter(str(tmp_path))
        mw.append({"sample_index": 0, "seed": 42, "status": "ok"})
        manifest = tmp_path / "manifest.jsonl"
        assert manifest.is_file()
        lines = manifest.read_text().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["sample_index"] == 0
```

**Concurrent-thread pattern** — no in-repo analog; follow CONTEXT D-38:
```python
def test_concurrent_appends_well_formed(self, tmp_path):
    """10 threads × 100 appends = 1000 well-formed JSON lines (D-38)."""
    mw = ManifestWriter(str(tmp_path))

    def _worker(worker_id: int):
        for i in range(100):
            mw.append({"sample_index": worker_id * 100 + i, "status": "ok"})

    threads = [threading.Thread(target=_worker, args=(w,)) for w in range(10)]
    for t in threads: t.start()
    for t in threads: t.join()

    lines = (tmp_path / "manifest.jsonl").read_text().splitlines()
    assert len(lines) == 1000
    for line in lines:
        entry = json.loads(line)  # Raises if corrupted; test fails.
        assert "sample_index" in entry
```

**`is_sample_complete` sentinel-only pattern** (CONTEXT D-16):
```python
def test_is_sample_complete_true_iff_sentinel(self, tmp_path):
    assert not ManifestWriter.is_sample_complete(str(tmp_path), 0)
    sample_dir = tmp_path / "000000"
    sample_dir.mkdir()
    (sample_dir / "sample.json").write_text("{}")
    assert ManifestWriter.is_sample_complete(str(tmp_path), 0)
    (sample_dir / "sample.json").unlink()
    assert not ManifestWriter.is_sample_complete(str(tmp_path), 0)
```

---

### `tests/test_split.py` (test, pure hash + empirical ratios)

**Analog:** `tests/test_sampler.py` (pure-function determinism pattern) + `tests/test_config.py` (validation error test).

**Pure-function determinism pattern** (test_sampler.py:93-97) — mirror for `assign_split`:
```python
class TestAssignSplit:
    @pytest.mark.parametrize("sample_seed", [1, 42, 99, 12345])
    def test_deterministic(self, sample_seed):
        a = assign_split(sample_seed, (0.8, 0.1, 0.1))
        b = assign_split(sample_seed, (0.8, 0.1, 0.1))
        assert a == b

    def test_returns_valid_label(self):
        label = assign_split(1, (0.8, 0.1, 0.1))
        assert label in ("train", "valid", "test")
```

**Empirical-ratio pattern** — no existing analog; follow CONTEXT D-39:
```python
def test_ratios_10k_seeds(self):
    from collections import Counter
    labels = Counter()
    for i in range(10000):
        seed = derive_sample_seed(42, i)
        labels[assign_split(seed, (0.8, 0.1, 0.1))] += 1
    assert 7840 <= labels["train"] <= 8160, f"train count {labels['train']} out of 80%±2%"
    assert 800 <= labels["valid"] <= 1200
    assert 800 <= labels["test"] <= 1200
```

**ValueError-on-invalid-ratio pattern** (test_config.py pattern + CONTEXT D-27):
```python
def test_invalid_ratios_sum_raises(self):
    with pytest.raises(ValueError, match="sum"):
        config.Config(split_ratios=(0.8, 0.1, 0.5))

def test_negative_ratio_raises(self):
    with pytest.raises(ValueError, match="non-negative"):
        config.Config(split_ratios=(0.8, -0.1, 0.3))
```

---

### `tests/test_api.py` (test, integration fast + slow)

**Analog:** `tests/test_integration_full_generation.py` (lines 36-74 + 85-213 for slow cases; fast cases use mocking).

**Skip-gate pattern** (test_integration_full_generation.py:38-74) — reuse verbatim for slow cases:
```python
fluidsynth_available = shutil.which("fluidsynth") is not None


def _all_sf2_layers_have_files() -> bool:
    try:
        import config as _cfg_mod
        _cfg = _cfg_mod.Config()
        for layer in ("beat", "melody", "harmony", "bassline"):
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
```

**For `test_api.py`, SPLIT into two modules or two classes:**
- `TestApiFast` — no marker. `test_config_global_seed_required`, `test_generate_resume_short_circuits` (with mocked pipeline).
- `TestApiSlow` — `@pytest.mark.skipif(not fluidsynth_available, ...)` + `@pytest.mark.slow`. `test_generate_produces_layout`, `test_generate_twice_idempotent`.

**Fast-case test: global_seed validation:**
```python
class TestApiFast:
    def test_config_global_seed_required(self, tmp_path):
        from musicgen import generate, Config
        cfg = Config(global_seed=None, sample_index=0, dataset_root=str(tmp_path))
        with pytest.raises(ValueError, match="global_seed"):
            generate(cfg)
```

**Fast-case resume short-circuit test:**
```python
def test_generate_resume_short_circuits(self, tmp_path, monkeypatch):
    """Pre-create sample.json → generate returns without running pipeline."""
    from musicgen import generate, Config
    # Pre-write the sentinel
    sample_dir = tmp_path / "000000"
    sample_dir.mkdir()
    (sample_dir / "sample.json").write_text(json.dumps({
        "seed": 12345, "split": "train",
        "musicality_score": {"score": 0.5, "components": {}},
        "duration_seconds": 10.0,
    }))
    # Patch renderer to assert never called
    call_count = {"n": 0}
    def _poison(*a, **kw):
        call_count["n"] += 1
        raise RuntimeError("pipeline must not run on resume")
    monkeypatch.setattr("musicgen.api.renderer.render_stems", _poison)
    cfg = Config(global_seed=1, sample_index=0, dataset_root=str(tmp_path))
    result = generate(cfg)
    assert call_count["n"] == 0
    assert result.status == "ok"
```

**Slow-case pipeline test** — adapt from `test_integration_full_generation.py:88-213`, but call `musicgen.generate(Config(...))` instead of `music_gen.create_song(...)`. Assert the 10 expected files exist:
```python
@pytest.mark.slow
class TestApiSlow:
    def test_generate_produces_layout(self, tmp_path):
        from musicgen import generate, Config
        result = generate(Config(
            global_seed=1, sample_index=0, dataset_root=str(tmp_path),
        ))
        sample_dir = Path(result.sample_dir)
        assert (sample_dir / "sample.json").is_file()
        assert (sample_dir / "mix.wav").is_file()
        for layer in ("beat", "melody", "harmony", "bassline"):
            assert (sample_dir / "stems" / f"{layer}.wav").is_file()
            assert (sample_dir / "midi" / f"{layer}.mid").is_file()
```

---

### `tests/test_determinism_golden.py` (test, golden SHA-256 regression)

**Analog:** `tests/test_integration_full_generation.py::TestMidiReproducibility::test_same_seed_produces_same_midi` (lines 216-245) — same-seed bit-identity via `read_bytes()` comparison.

**Bit-identity pattern** (test_integration_full_generation.py:224-245):
```python
def _run(name: str, seed: int):
    music_gen._rng.seed(seed)
    music_gen.create_song(...)
    part_dir = Path(tmp_path) / name / f"{name}-intro"
    return {
        layer: (part_dir / f"{name}-intro-{layer}.mid").read_bytes()
        for layer in ("beat", "melody", "harmony", "bassline")
    }

a = _run("rep1", seed=42)
b = _run("rep2", seed=42)
for layer in ("beat", "melody", "harmony", "bassline"):
    assert a[layer] == b[layer], f"MIDI reproducibility broken for layer {layer!r}"
```

**For `test_determinism_golden.py`, extend with SHA-256 + `--regen-goldens` flag support** (RESEARCH.md Pattern 5, lines 344-364):
```python
@pytest.mark.slow
@pytest.mark.parametrize("artifact", [
    "mix", "midi_beat", "midi_melody", "midi_harmony", "midi_bassline", "sample",
])
def test_sha256_matches_golden(request, tmp_path, artifact):
    # Skip gate: fluidsynth + sf2 pool (reuse test_integration_full_generation pattern)
    # + fluidsynth_version match (D-29)
    from musicgen import generate, Config
    result = generate(Config(global_seed=1, sample_index=0, dataset_root=str(tmp_path)))

    # Build path + compute SHA-256
    artifact_paths = {
        "mix": result.mix_path,
        "midi_beat": result.midi_paths["beat"],
        # ...
        "sample": result.sample_json_path,
    }
    actual = hashlib.sha256(Path(artifact_paths[artifact]).read_bytes()).hexdigest()

    golden_path = FIXTURES / f"expected_{artifact}.sha256"
    if request.config.getoption("--regen-goldens"):
        golden_path.write_text(actual + "\n")
    else:
        expected = golden_path.read_text().strip()
        assert actual == expected, f"{artifact} hash mismatch: {actual} != {expected}"
```

**FluidSynth version skip gate** — new pattern per CONTEXT D-29; a module-level `pytestmark` that reads `fixtures/determinism/fluidsynth_version.txt` and compares to `subprocess.run(["fluidsynth", "--version"])`. MIDI + sample hashes must pass unconditionally; **only** the `mix` hash is version-guarded per R-P8.

**Same-process byte-stability test (D-30)** — fast, no-FluidSynth:
```python
def test_generate_sample_json_stable_same_process(tmp_path, monkeypatch):
    """Two back-to-back calls produce byte-identical sample.json (D-30)."""
    # Monkeypatch renderer + musicality to return deterministic stubs (avoid FluidSynth)
    # Run generate twice, hash both sample.json files, compare.
```
See RESEARCH.md Code Examples "SHA-256 stability cross-check" lines 587-603.

---

## Shared Patterns

### Module Header + Logging

**Source:** `src/musicgen/mixer.py:1-47` (canonical pattern for all Phase 4+ modules)

**Apply to:** `seeds.py`, `writer.py`, `manifest.py`, `api.py`, `musicality.py`

```python
"""<Module> — <one-line purpose> (R-P<n>).

<Multi-line design-rationale docstring with D-XX references.>
...
"""
from __future__ import annotations

import logging
# ... stdlib imports ...
# ... third-party imports ...
# ... local imports (config, musicgen.XXX) ...

logger = logging.getLogger(__name__)
```

`logger.info` for milestones, `logger.debug` for details, `logger.warning` for recoverable oddities, `logger.error` for failures (Phase 2 D-07 convention).

### Frozen Dataclass Shape

**Source:** `src/musicgen/renderer.py:70-96` (`RenderResult`) — confirmed by `src/musicgen/mixer.py:271-295` (`MixResult`)

**Apply to:** `api.py::SampleResult`

```python
@dataclass(frozen=True)
class <Result>:
    """<Purpose> (<R-Xn>).

    Produced by :func:`<producer>`. Consumed by :func:`<consumer>`.

    Attributes:
        field_one: <description>.
        field_two: <description>.
    """
    field_one: <type>
    field_two: <type>
    # ...
```

**Note on `slots=True`:** CONTEXT D-02 calls for `slots=True`; existing Phase 4 dataclasses do NOT use slots. Planner picks per Phase 5 "Claude's Discretion" — consistency with Phase 4 wins unless proven otherwise.

### RNG Injection (Phase 3 D-07 invariant)

**Source:** `src/musicgen/mixer.py:95-117` (`_create_effect`), `src/musicgen/mixer.py:240-266` (`compute_layer_mask`)

**Apply to:** `api.py` orchestration loop — every `rng: random.Random` parameter gets the right `rngs[RNG_XXX]` per CONTEXT D-19:

```python
def _create_effect(effect_class, parameters: dict, rng: random.Random):
    probability = parameters['probability']
    ...
    if rng.random() < probability:
        kwargs = {param: rng.uniform(...) for param in value_range}
        return effect_class(**kwargs)
    return None
```

Zero bare `random.<method>(...)` — enforced by `tests/test_no_bare_random_in_package.py`. New modules in Phase 5 are automatically scanned (meta-test at line 71 asserts the glob picks up all expected modules; extend to include `seeds.py`, `writer.py`, `manifest.py`, `api.py`, `musicality.py`).

### cfg Fallback (Phase 4 D-25)

**Source:** `src/musicgen/mixer.py:180-183` (`build_fx_boards`) — and `src/musicgen/generators/beat.py` (mentioned in beat.py header)

**Apply to:** Any public function taking `cfg: config.Config = None`:

```python
def build_fx_boards(
    cfg: Optional[config.Config] = None,
    rng: Optional[random.Random] = None,
) -> Dict[str, Pedalboard]:
    if rng is None:
        raise ValueError("build_fx_boards requires an injected rng (D-17)")
    _cfg = cfg if cfg is not None else config.Config()
    return {layer: _generate_pedalboard(_cfg.fx_files[layer], rng) for layer in _LAYERS}
```

**Note:** `api.generate(config: Config)` takes `Config` positionally (no None fallback — it's required). This is a deliberate departure: the library entry point owns its own config; every helper inside takes the resolved `_cfg`.

### Input Validation (raise-with-message)

**Source:** `src/musicgen/mixer.py:181` and `mixer.py:433`

**Apply to:** All public APIs in `seeds.py`, `writer.py`, `manifest.py`, `api.py`:

```python
if rng is None:
    raise ValueError("build_fx_boards requires an injected rng (D-17)")
# ... and ...
if not part_mix_paths:
    raise ValueError("concat_parts: no part mix paths provided; cannot assemble final mix")
```

Specifically for `api.generate`: `if config.global_seed is None: raise ValueError("global_seed is required for deterministic generation; pass config.global_seed explicitly")` per CONTEXT D-21.

### AST Guard Coverage

**Source:** `tests/test_no_bare_random_in_package.py` (existing).

**Apply to:** extend the `expected_present` set in the meta-test at line 81 to include:
```python
expected_present = {
    "sampler.py", "renderer.py", "mixer.py", "annotator.py", "beats.py",
    "duration_validator.py",
    # Phase 5 additions:
    "seeds.py", "writer.py", "manifest.py", "api.py", "musicality.py",
    os.path.join("generators", "beat.py"),
    # ... existing generators ...
}
```
The parametrized test `test_no_bare_random_in_package_module` auto-picks up new modules via `glob` — no change needed there. Only the meta-test's `expected_present` set is updated.

**Exception for `seeds.py`:** `save_random_state()` uses `random.getstate()`/`random.setstate(state)`. The guard blocks ALL `random.<attr>` except `random.Random`. Two options (planner picks):
- **A:** Extend the guard to also allow `getstate`, `setstate` (one-line change in `_bare_random_calls`).
- **B:** Document `seeds.py` as the sole exception and add it to an allow-list.

Option A is cleaner (the functions are documented stdlib primitives, not randomness consumers).

### pytest `@pytest.mark.slow` Skip Gate

**Source:** `tests/test_integration_full_generation.py:36-74` (module-level `pytestmark` with fluidsynth + sf2 skip).

**Apply to:** `tests/test_determinism_golden.py` + the slow half of `tests/test_api.py`.

```python
fluidsynth_available = shutil.which("fluidsynth") is not None
# ... _all_sf2_layers_have_files() helper ...
sf2_pool_ready = _all_sf2_layers_have_files()

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not fluidsynth_available, reason="fluidsynth binary not on PATH"),
    pytest.mark.skipif(not sf2_pool_ready, reason="one or more sf/<layer>/ dirs is empty"),
]
```

For `test_determinism_golden.py`, **add a THIRD skip gate** per CONTEXT D-29:
```python
def _fluidsynth_version_matches_golden() -> bool:
    version_file = FIXTURES / "fluidsynth_version.txt"
    if not version_file.exists():
        return False
    expected = version_file.read_text().strip()
    try:
        result = subprocess.run(
            ["fluidsynth", "--version"], capture_output=True, text=True, timeout=5,
        )
        output = result.stdout if result.stdout.strip() else result.stderr
        first_line = output.splitlines()[0] if output.splitlines() else ""
        return first_line == expected
    except Exception:
        return False
```
The `mix.wav` SHA-256 test `xfail`s when this returns False; MIDI + `sample.json` hashes must pass regardless (they're FluidSynth-independent per R-P8).

---

## No Analog Found

Files with no close match in the codebase — planner should use RESEARCH.md code examples:

| File/Function | Role | Data Flow | Reason |
|---------------|------|-----------|--------|
| `manifest.ManifestWriter` class + `threading.Lock` | service | event-driven | No prior lock/append abstraction in the codebase; RESEARCH.md Pattern 3 is the canonical source |
| `writer._concat_layer_midis` (absolute-tick walk) | utility | transform | No existing MIDI-concat code; RESEARCH.md Code Examples lines 497-554 is the canonical source (mido absolute-tick walk) |
| `writer._assert_sum_of_stems` (int32 accumulator) | validator | transform | No existing stem-sum assertion; RESEARCH.md Code Examples lines 556-584 is the canonical source |
| `seeds.save_random_state()` context manager | utility | transform | No `@contextlib.contextmanager` usage in `src/musicgen/`; RESEARCH.md Pattern 1 lines 220-227 is the canonical source |
| `tests/conftest.py` `pytest_addoption` hook | test plugin | — | No existing conftest.py; RESEARCH.md Pattern 5 lines 346-351 is the canonical source |
| `tests/fixtures/determinism/*.sha256` + `fluidsynth_version.txt` | test fixture | — | No existing golden fixtures; CONTEXT D-28 specifies the exact 7-file layout |
| `importlib.metadata.version("musicgen")` with `PackageNotFoundError` fallback | utility | transform | No existing version-resolution code; RESEARCH.md Pitfall 4 lines 449-455 is the canonical source |
| `tempfile.mkdtemp(prefix="musicgen-")` + `shutil.rmtree(..., ignore_errors=True)` working dir | utility | file-I/O | No existing temp-dir workflow; RESEARCH.md System Architecture Diagram is the source |

## Metadata

**Analog search scope:** `src/musicgen/`, `tests/`, `config.py`, `music_gen.py`, `musicality_score.py`, `.planning/phases/03-*/`, `.planning/phases/04-*/`
**Files scanned:** 18 source files + 6 test files + 2 prior-phase PLAN files
**Pattern extraction date:** 2026-04-19
**Cross-reference depth:** Full 05-CONTEXT.md (43 decisions D-01..D-43) + full 05-RESEARCH.md scans targeted at Patterns 1-5, Code Examples, and Pitfalls 1-8
