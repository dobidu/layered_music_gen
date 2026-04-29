"""Config module for musicgen — owns all filesystem paths and override layers.

D-01/D-02: three-layer precedence — CLI args > env vars > hardcoded defaults.
D-03: wraps existing JSON files; no new config file format.
D-09: soundfont pool detection fires at config load time (informational only).

Phase 2 (R-S5, R-S9). Phase 6 will populate `cli_overrides` from the typer CLI.
"""
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# --- Hardcoded defaults (lowest precedence layer per D-02) ---
DEFAULT_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SF_DIR = os.path.join(DEFAULT_PROJECT_ROOT, "sf")
DEFAULT_SF_LAYERS: Tuple[str, ...] = ("beat", "melody", "harmony", "bassline")

DEFAULT_FX_FILES: Dict[str, str] = {
    "beat":     os.path.join(DEFAULT_PROJECT_ROOT, "beat_fx.json"),
    "melody":   os.path.join(DEFAULT_PROJECT_ROOT, "melody_fx.json"),
    "harmony":  os.path.join(DEFAULT_PROJECT_ROOT, "harmony_fx.json"),
    "bassline": os.path.join(DEFAULT_PROJECT_ROOT, "bassline_fx.json"),
}

DEFAULT_INST_PROBABILITIES_FILE = os.path.join(DEFAULT_PROJECT_ROOT, "inst_probabilities.json")
DEFAULT_LEVELS_FILE              = os.path.join(DEFAULT_PROJECT_ROOT, "levels.json")
DEFAULT_SONG_STRUCTURES_FILE     = os.path.join(DEFAULT_PROJECT_ROOT, "song_structures.json")
DEFAULT_CHORD_PATTERNS_FILE      = os.path.join(DEFAULT_PROJECT_ROOT, "chord_patterns.txt")

DEFAULT_BEAT_ROLL_PATTERN_FILES: Dict[str, str] = {
    "2/4":  os.path.join(DEFAULT_PROJECT_ROOT, "beat_roll_patterns_24.txt"),
    "3/4":  os.path.join(DEFAULT_PROJECT_ROOT, "beat_roll_patterns_34.txt"),
    "4/4":  os.path.join(DEFAULT_PROJECT_ROOT, "beat_roll_patterns_44.txt"),
    "6/8":  os.path.join(DEFAULT_PROJECT_ROOT, "beat_roll_patterns_68.txt"),
    "7/8":  os.path.join(DEFAULT_PROJECT_ROOT, "beat_roll_patterns_78.txt"),
    "12/8": os.path.join(DEFAULT_PROJECT_ROOT, "beat_roll_patterns_128.txt"),
}

SOUNDFONT_POOL_WARN_THRESHOLD = 3  # D-09: warn when a layer has fewer than 3 .sf2 files


@dataclass
class Config:
    project_root: str = DEFAULT_PROJECT_ROOT
    sf_dir: str = DEFAULT_SF_DIR
    sf_layers: Tuple[str, ...] = DEFAULT_SF_LAYERS
    fx_files: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_FX_FILES))
    inst_probabilities_file: str = DEFAULT_INST_PROBABILITIES_FILE
    levels_file: str = DEFAULT_LEVELS_FILE
    song_structures_file: str = DEFAULT_SONG_STRUCTURES_FILE
    chord_patterns_file: str = DEFAULT_CHORD_PATTERNS_FILE
    beat_roll_pattern_files: Dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_BEAT_ROLL_PATTERN_FILES)
    )

    # --- Phase 5 fields (D-09, D-21, D-25, D-27) ---
    dataset_root: str = field(
        default_factory=lambda: os.path.join(DEFAULT_PROJECT_ROOT, "dataset")
    )
    global_seed: Optional[int] = None
    sample_index: int = 0
    split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1)
    sum_of_stems_epsilon: float = 1e-3
    keep_working_dirs: bool = False
    workers: Optional[int] = None

    # --- Phase 6 fields (D-47, D-48, D-49) ---
    output_mode: str = "full"   # R-P14: full | mix-only | stems-only | midi-only
    count: int = 1              # R-P12: number of samples for generate_batch

    _VALID_OUTPUT_MODES = frozenset({"full", "mix-only", "stems-only", "midi-only"})

    def __post_init__(self):
        """Phase 5+6 field validation (D-27, D-47, D-48, D-49)."""
        if abs(sum(self.split_ratios) - 1.0) > 1e-9:
            raise ValueError(
                f"split_ratios must sum to 1.0, got "
                f"{sum(self.split_ratios)} for {self.split_ratios}"
            )
        if any(r < 0 for r in self.split_ratios):
            raise ValueError(
                f"split_ratios must be non-negative, got {self.split_ratios}"
            )
        if self.output_mode not in self._VALID_OUTPUT_MODES:
            raise ValueError(
                f"output_mode must be one of {sorted(self._VALID_OUTPUT_MODES)}, "
                f"got {self.output_mode!r}"
            )
        if self.count < 1:
            raise ValueError(f"count must be >= 1, got {self.count}")

    def sf_layer_dir(self, layer: str) -> str:
        """Return the on-disk directory for a single soundfont layer."""
        return os.path.join(self.sf_dir, layer)

    def beat_pattern_file(self, time_signature: str) -> str:
        """Return the beat-roll pattern file path for a time signature string like '4/4'."""
        return self.beat_roll_pattern_files[time_signature]

    @classmethod
    def load(cls, cli_overrides: Optional[Dict[str, object]] = None) -> "Config":
        """Load Config applying D-01/D-02 precedence: CLI > env > defaults.

        Phase 2 callers pass cli_overrides=None; only env + defaults apply.
        Phase 6 typer CLI will build a dict from parsed args and pass it here:
            cfg = Config.load(cli_overrides={"sf_dir": args.sf_dir, ...})

        Fires the D-09 soundfont pool report before returning.
        """
        cfg = cls()

        # env-var layer (D-02 middle layer)
        sf_env = os.environ.get("MUSICGEN_SF_DIR")
        if sf_env:
            cfg.sf_dir = os.path.abspath(sf_env)  # T-02-01 mitigation: normalize path
        root_env = os.environ.get("MUSICGEN_PROJECT_ROOT")
        if root_env:
            cfg.project_root = os.path.abspath(root_env)
        dataset_env = os.environ.get("MUSICGEN_DATASET_ROOT")
        if dataset_env:
            cfg.dataset_root = os.path.abspath(dataset_env)  # T-02-01 mitigation
        output_mode_env = os.environ.get("MUSICGEN_OUTPUT_MODE")
        if output_mode_env:
            cfg.output_mode = output_mode_env  # validated in __post_init__ on next load
        count_env = os.environ.get("MUSICGEN_COUNT")
        if count_env:
            try:
                cfg.count = int(count_env)
            except ValueError:
                logger.warning("MUSICGEN_COUNT is not an integer: %r", count_env)

        # cli layer (D-02 top layer; framework-agnostic — avoids typer dep in Phase 2)
        if cli_overrides:
            for key, value in cli_overrides.items():
                if value is None:
                    continue
                if not hasattr(cfg, key):
                    continue
                if isinstance(value, str) and key in ("sf_dir", "project_root"):
                    value = os.path.abspath(value)  # T-02-01 mitigation
                setattr(cfg, key, value)

        cfg._emit_soundfont_pool_report()  # D-09
        return cfg

    def _emit_soundfont_pool_report(self) -> None:
        """Log soundfont counts per layer. WARNING when below D-09 threshold."""
        for layer in self.sf_layers:
            layer_dir = self.sf_layer_dir(layer)
            try:
                entries = os.listdir(layer_dir)
            except FileNotFoundError:
                logger.warning("Soundfont layer directory missing: %s", layer_dir)
                continue
            except PermissionError:
                logger.warning(
                    "Soundfont layer directory permission denied: %s", layer_dir
                )
                continue
            count = sum(1 for e in entries if e.endswith(".sf2"))
            if count < SOUNDFONT_POOL_WARN_THRESHOLD:
                logger.warning(
                    "Soundfont pool thin for layer %s: %d .sf2 files in %s "
                    "(expected >= %d)",
                    layer, count, layer_dir, SOUNDFONT_POOL_WARN_THRESHOLD,
                )
            else:
                logger.info(
                    "Soundfont pool for layer %s: %d .sf2 files", layer, count
                )

    # --- JSON loader helpers (thin wrappers over existing files; D-03) ---

    def load_levels(self) -> dict:
        with open(self.levels_file) as f:
            return json.load(f)

    def load_inst_probabilities(self) -> dict:
        with open(self.inst_probabilities_file) as f:
            return json.load(f)

    def load_song_structures(self) -> dict:
        with open(self.song_structures_file) as f:
            return json.load(f)
