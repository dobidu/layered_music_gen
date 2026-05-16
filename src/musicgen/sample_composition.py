"""M3 — SampleCompositionConfig: rules for mixing real audio samples into the pipeline.

Three mixing modes per layer:
  alongside   — sample overlaid on top of the FluidSynth-rendered mix
  substitution — sample replaces the FluidSynth stem before mix_part()
  adlib        — one-shot sample placed at a specific beat offset

SampleCompositionConfig is wired into Config as an optional field. When None,
the pipeline runs exactly as before (no behavioural change).

Depends on audio_sample_manager (pip install musicgen[samples]).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_VALID_MODES = frozenset({"alongside", "substitution", "adlib"})
_VALID_LAYERS = frozenset({"beat", "bassline", "melody", "harmony"})

# musicgen layer name → audio_sample_manager category string
LAYER_TO_CATEGORY: Dict[str, str] = {
    "beat":     "beat",
    "bassline": "bass",
    "melody":   "melody",
    "harmony":  "harmony",
}


@dataclass
class SampleLayerRule:
    """Mixing rule for one layer.

    Attributes:
        layer: musicgen layer name (beat | bassline | melody | harmony).
        mode: alongside | substitution | adlib.
        loop_align_to_measure: when True, tiled loop starts at measure boundary.
        oneshot_at_beat: for adlib mode, 0-indexed beat to place the one-shot.
        max_bpm_stretch_pct: reject samples whose BPM requires more than this
            percent of time-stretching (e.g. 10 → ±10%).
        min_musicality_score: per-layer quality gate; overrides
            SampleCompositionConfig.global_min_musicality when set.
        gain_db: gain applied to sample before mixing (negative = duck).
        tags: sample-manager tag filter.
        genre: sample-manager genre filter.
        mood: sample-manager mood filter.
    """
    layer: str
    mode: str
    loop_align_to_measure: bool = True
    oneshot_at_beat: Optional[int] = None
    max_bpm_stretch_pct: float = 10.0
    min_musicality_score: Optional[float] = None
    gain_db: float = -3.0
    tags: Optional[List[str]] = None
    genre: Optional[List[str]] = None
    mood: Optional[List[str]] = None

    def __post_init__(self) -> None:
        if self.mode not in _VALID_MODES:
            raise ValueError(f"mode must be one of {sorted(_VALID_MODES)}, got {self.mode!r}")
        if self.layer not in _VALID_LAYERS:
            raise ValueError(f"layer must be one of {sorted(_VALID_LAYERS)}, got {self.layer!r}")
        if self.mode == "adlib" and self.oneshot_at_beat is None:
            raise ValueError("adlib mode requires oneshot_at_beat to be set")


@dataclass
class SampleCompositionConfig:
    """Top-level config for sample composition.

    Attributes:
        sample_db_path: path to the SampleManager JSON database file.
        layer_rules: dict of layer name → SampleLayerRule (only configured
            layers are active; missing layers use FluidSynth as normal).
        global_min_musicality: floor musicality score for all layers; a layer
            rule's min_musicality_score overrides this when set.
        allow_transposition: pass-through to SampleSelector.select_for_layer().
        allow_time_stretching: pass-through to SampleSelector.select_for_layer().
    """
    sample_db_path: str
    layer_rules: Dict[str, SampleLayerRule] = field(default_factory=dict)
    global_min_musicality: Optional[float] = None
    allow_transposition: bool = True
    allow_time_stretching: bool = True

    def effective_min_musicality(self, layer: str) -> Optional[float]:
        """Return per-layer min score if set, else global, else None."""
        rule = self.layer_rules.get(layer)
        if rule is not None and rule.min_musicality_score is not None:
            return rule.min_musicality_score
        return self.global_min_musicality


def select_samples(
    sc_cfg: SampleCompositionConfig,
    composition_key: Optional[str],
    composition_bpm: Optional[float],
    genre: Optional[List[str]] = None,
) -> Dict[str, object]:
    """Select one sample per active layer rule.

    Returns dict mapping musicgen layer name → SampleMetadata.
    Layers with no suitable sample are omitted (callers should handle absence).

    Guards the import of audio_sample_manager — ImportError is caught and
    logged so that a missing [samples] extra degrades gracefully to no-op.
    """
    try:
        from sample_manager import SampleManager   # type: ignore[import]
        from sample_selector import SampleSelector  # type: ignore[import]
    except ImportError:
        logger.warning(
            "audio_sample_manager not installed; sample composition disabled. "
            "Install with: pip install 'musicgen[samples]'"
        )
        return {}

    try:
        mgr = SampleManager(sc_cfg.sample_db_path)
    except Exception as exc:
        logger.error("Failed to load sample database %s: %s", sc_cfg.sample_db_path, exc)
        return {}

    selector = SampleSelector(mgr)
    selected: Dict[str, object] = {}

    for layer, rule in sc_cfg.layer_rules.items():
        sm_layer = LAYER_TO_CATEGORY.get(layer, layer)  # "bassline" → "bass"
        min_score = sc_cfg.effective_min_musicality(layer)

        # BPM bounds check: skip samples outside max_bpm_stretch_pct window
        bpm_for_selection = composition_bpm
        allow_stretch = sc_cfg.allow_time_stretching
        if not allow_stretch and composition_bpm is not None:
            # Hard filter: only samples within the allowed stretch window
            pass  # SampleSelector handles allow_time_stretching=False already

        try:
            sample = selector.select_for_layer(
                layer=sm_layer,
                key=composition_key,
                bpm=bpm_for_selection,
                genre=rule.genre or genre,
                mood=rule.mood,
                tags=rule.tags,
                allow_transposition=sc_cfg.allow_transposition,
                allow_time_stretching=allow_stretch,
                min_musicality_score=min_score,
            )
        except Exception as exc:
            logger.warning("Sample selection failed for layer %s: %s", layer, exc)
            sample = None

        if sample is None:
            logger.info("No sample found for layer %s — layer uses FluidSynth only", layer)
            continue

        # Check BPM stretch budget
        if (
            composition_bpm is not None
            and sample.bpm is not None
            and sample.bpm > 0
            and rule.max_bpm_stretch_pct > 0
        ):
            stretch_pct = abs(sample.bpm - composition_bpm) / composition_bpm * 100
            if stretch_pct > rule.max_bpm_stretch_pct:
                logger.info(
                    "Layer %s: sample BPM %.1f vs composition %.1f (%.1f%% > %.1f%% limit) — skipped",
                    layer, sample.bpm, composition_bpm, stretch_pct, rule.max_bpm_stretch_pct,
                )
                continue

        selected[layer] = sample
        logger.info(
            "Layer %s: selected sample id=%s bpm=%s key=%s score=%s",
            layer,
            getattr(sample, "id", "?"),
            getattr(sample, "bpm", "?"),
            getattr(sample, "key", "?"),
            getattr(sample, "musicality_score", "?"),
        )

    return selected
