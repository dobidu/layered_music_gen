"""GenreSpec dataclass + genre composition engine.

v0.2 Phase 1: genre-aware generation system.

Genre files live at genres/<name>/spec.json (repo root).
Config.genre: Optional[List[str]] — list of genre names to compose.

Merge semantics:
  - Hard numeric ranges (tempo, swing): intersection (max-of-mins, min-of-maxes)
  - Soft weight dicts: normalized weighted average (missing key treated as 0)
  - chord_type_hard_filter: union when ALL genres have explicit filter; None when any is None
  - soundfont_tags per layer: union (dedup, insertion order from first genre)
  - drum_pool_names: union (dedup, insertion order)
  - layer_probs, fx_profile, arrangement_weights: weighted average per key

Precedence when active: CLI > env > genre-merged > Config defaults.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class GenreSpec:
    name: str
    description: str = ""

    # Hard bounds — sampler clamps draws to these ranges
    tempo_min: float = 60.0
    tempo_max: float = 240.0
    swing_min: float = 0.5
    swing_max: float = 0.75

    # Soft weight dicts — empty dict = uniform / no constraint on this dimension
    time_sig_weights: Dict[str, float] = field(default_factory=dict)
    scale_weights: Dict[str, float] = field(default_factory=dict)
    chord_type_weights: Dict[str, float] = field(default_factory=dict)
    inversion_weights: Dict[str, float] = field(default_factory=dict)
    layer_probs: Dict[str, float] = field(default_factory=dict)
    arrangement_weights: Dict[str, float] = field(default_factory=dict)
    fx_profile: Dict[str, float] = field(default_factory=dict)

    # Hard filter — None = all chord types allowed; list = only these types allowed
    chord_type_hard_filter: Optional[List[str]] = None

    # Set fields — empty = no constraint
    soundfont_tags: Dict[str, List[str]] = field(default_factory=dict)
    drum_pool_names: List[str] = field(default_factory=list)

    # Markov chord transition matrix (v0.3 Phase 1).
    # Format: {"order": 1|2, "init_probs": {chord: weight},
    #          "transitions": {key: {chord: weight}}}
    # Single-chord keys ("I") = 1st-order; "prev,curr" keys = 2nd-order.
    # None = use chord_patterns.txt (backward compat).
    chord_transition_matrix: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_GENRESPEC_FIELDS = {f.name for f in GenreSpec.__dataclass_fields__.values()}  # type: ignore[attr-defined]


def _weighted_avg_dict(
    dicts: List[Dict[str, float]],
    norm_weights: List[float],
) -> Dict[str, float]:
    """Weighted average over the union of keys; renormalized to sum=1 if non-zero."""
    if not any(d for d in dicts):
        return {}
    keys: set = set()
    for d in dicts:
        keys.update(d)
    result: Dict[str, float] = {}
    for k in keys:
        result[k] = sum(d.get(k, 0.0) * w for d, w in zip(dicts, norm_weights))
    total = sum(result.values())
    if total > 0:
        result = {k: v / total for k, v in result.items()}
    return result


def _union_list(lists: List[List], /) -> List:
    """Ordered union — first occurrence wins for dedup."""
    seen: set = set()
    out = []
    for lst in lists:
        for item in lst:
            if item not in seen:
                out.append(item)
                seen.add(item)
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_genre(name: str, genres_dir: str) -> GenreSpec:
    """Load a GenreSpec from genres/<name>/spec.json.

    Raises FileNotFoundError if the spec file does not exist.
    Unknown JSON keys are silently ignored (forward compatibility).
    """
    spec_path = os.path.join(genres_dir, name, "spec.json")
    if not os.path.isfile(spec_path):
        raise FileNotFoundError(f"Genre spec not found: {spec_path!r}")
    with open(spec_path) as f:
        data: Dict[str, Any] = json.load(f)
    data.setdefault("name", name)
    known = {k: v for k, v in data.items() if k in _GENRESPEC_FIELDS}
    transitions_path = os.path.join(genres_dir, name, "chord_transitions.json")
    if os.path.isfile(transitions_path):
        with open(transitions_path) as f:
            known["chord_transition_matrix"] = json.load(f)
    return GenreSpec(**known)


def merge_genres(
    specs: List[GenreSpec],
    weights: Optional[List[float]] = None,
) -> GenreSpec:
    """Merge a list of GenreSpecs into a single composite spec.

    weights: optional per-spec blend weights (default: equal). Must match len(specs).
    At least one spec required.
    """
    if not specs:
        raise ValueError("merge_genres requires at least one GenreSpec")
    if len(specs) == 1:
        return specs[0]

    n = len(specs)
    if weights is None:
        weights = [1.0] * n
    if len(weights) != n:
        raise ValueError(
            f"weights length {len(weights)} must match specs length {n}"
        )
    total_w = sum(weights)
    norm_w = [w / total_w for w in weights]

    # Hard numeric intersection
    tempo_min = max(s.tempo_min for s in specs)
    tempo_max = min(s.tempo_max for s in specs)
    swing_min = max(s.swing_min for s in specs)
    swing_max = min(s.swing_max for s in specs)

    # Soft weight dicts: weighted average
    time_sig_weights = _weighted_avg_dict([s.time_sig_weights for s in specs], norm_w)
    scale_weights = _weighted_avg_dict([s.scale_weights for s in specs], norm_w)
    chord_type_weights = _weighted_avg_dict([s.chord_type_weights for s in specs], norm_w)
    inversion_weights = _weighted_avg_dict([s.inversion_weights for s in specs], norm_w)
    layer_probs = _weighted_avg_dict([s.layer_probs for s in specs], norm_w)
    arrangement_weights = _weighted_avg_dict([s.arrangement_weights for s in specs], norm_w)
    fx_profile = _weighted_avg_dict([s.fx_profile for s in specs], norm_w)

    # chord_type_hard_filter: None if any genre has None; union otherwise
    filters = [s.chord_type_hard_filter for s in specs]
    if any(f is None for f in filters):
        chord_type_hard_filter = None
    else:
        chord_type_hard_filter = _union_list(filters)  # type: ignore[arg-type]

    # soundfont_tags: per-layer union
    all_layers: set = set()
    for s in specs:
        all_layers.update(s.soundfont_tags.keys())
    soundfont_tags: Dict[str, List[str]] = {}
    for layer in all_layers:
        soundfont_tags[layer] = _union_list(
            [s.soundfont_tags.get(layer, []) for s in specs]
        )

    # drum_pool_names: ordered union
    drum_pool_names = _union_list([s.drum_pool_names for s in specs])

    return GenreSpec(
        name="+".join(s.name for s in specs),
        description="Composed: " + ", ".join(s.name for s in specs),
        tempo_min=tempo_min,
        tempo_max=tempo_max,
        swing_min=swing_min,
        swing_max=swing_max,
        time_sig_weights=time_sig_weights,
        scale_weights=scale_weights,
        chord_type_weights=chord_type_weights,
        chord_type_hard_filter=chord_type_hard_filter,
        inversion_weights=inversion_weights,
        layer_probs=layer_probs,
        arrangement_weights=arrangement_weights,
        fx_profile=fx_profile,
        soundfont_tags=soundfont_tags,
        drum_pool_names=drum_pool_names,
        chord_transition_matrix=None,  # matrices not merged; fallback to pattern file
    )


def resolve_genres(
    genre_names: List[str],
    genres_dir: str,
    weights: Optional[List[float]] = None,
) -> GenreSpec:
    """Load and merge genres by name. Convenience wrapper over load_genre + merge_genres."""
    specs = [load_genre(name, genres_dir) for name in genre_names]
    return merge_genres(specs, weights)
