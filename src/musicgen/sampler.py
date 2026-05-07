"""Sampler module — pure-function song-level parameter sampling (R-X2, D-20/D-21/D-22).

All sampler functions take an explicit ``rng: random.Random`` parameter per D-07.
No bare ``random.*`` anywhere in this module — verified by the AST scan test in
``tests/test_sampler.py::test_no_bare_random_in_sampler``.

``SongParams`` is a frozen dataclass embedding arrangement + per-part signatures/measures
(D-20). Its ``sample`` classmethod draws all fields using the given rng in the exact
same order as today's ``generate_song`` so Phase 5's golden test baselines against the
current behavior (RESEARCH Risk #2).
"""
from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import config
from timesig import TimeSignatureRegistry
from musicgen.genre import GenreSpec

logger = logging.getLogger(__name__)


def generate_random_key(rng: random.Random, genre_spec: Optional[GenreSpec] = None) -> str:
    """Generate a weighted random musical key.

    One ``rng.random()`` draw, 24-branch threshold loop with explicit fallback.
    When ``genre_spec.scale_weights`` is non-empty, uses weighted draw from
    those keys instead (soft constraint — any key still reachable, but weights shift).

    Sources:
      * https://www.digitaltrends.com/music/whats-the-most-popular-music-key-spotify/
      * https://web.archive.org/web/20190426230344/https://insights.spotify.com/us/2015/05/06/most-popular-keys-on-spotify/
      * https://forum.bassbuzz.com/t/most-used-keys-on-spotify/5886
    """
    if genre_spec is not None and genre_spec.scale_weights:
        weights = genre_spec.scale_weights
        candidates = [k for k, w in weights.items() if w > 0]
        if candidates:
            w_values = [weights[k] for k in candidates]
            return rng.choices(candidates, weights=w_values, k=1)[0]

    key_ranges = [
        (0.107, 'G'), (0.209, 'C'), (0.296, 'D'), (0.357, 'A'), (0.417, 'C#'), (0.47, 'F'),
        (0.518, 'Am'), (0.561, 'G#'), (0.603, 'Em'), (0.645, 'Bm'), (0.681, 'E'), (0.716, 'A#'),
        (0.748, 'A#m'), (0.778, 'Fm'), (0.805, 'F#'), (0.831, 'B'), (0.857, 'Gm'), (0.883, 'Dm'),
        (0.908, 'F#m'), (0.932, 'D#'), (0.956, 'Cm'), (0.977, 'C#m'), (0.989, 'G#m'), (1.0, 'D#m'),
    ]
    dice = rng.random()
    for prob, key in key_ranges:
        if dice < prob:
            return key
    # Explicit fallback in case float-rounding leaves dice >= final prob
    return key_ranges[-1][1]


def generate_random_tempo(rng: random.Random, genre_spec: Optional[GenreSpec] = None) -> int:
    """Generate a weighted random tempo (BPM).

    Two draws always: ``rng.random()`` for bucket, ``rng.randint(min, max)`` inside.
    When ``genre_spec`` is set, clamps the result to [tempo_min, tempo_max] (hard bounds).

    Source: https://blog.musiio.com/2021/08/19/which-musical-tempos-are-people-streaming-the-most/
    """
    tempo_ranges = [
        (0.0183, 60, 70), (0.0454, 70, 80), (0.1849, 80, 90), (0.3721, 90, 100),
        (0.4817, 100, 110), (0.5747, 110, 120), (0.7048, 120, 130), (0.7917, 130, 140),
        (0.8958, 140, 150), (0.9739, 150, 160), (1.0, 160, 170),
    ]
    if genre_spec is not None:
        tmin = int(genre_spec.tempo_min)
        tmax = int(genre_spec.tempo_max)
        return rng.randint(tmin, tmax)

    dice = rng.random()
    for prob, min_tempo, max_tempo in tempo_ranges:
        if dice < prob:
            return rng.randint(min_tempo, max_tempo)
    # Explicit fallback in case float-rounding leaves dice >= final prob
    return rng.randint(*tempo_ranges[-1][1:])


def generate_random_time_signature(rng: random.Random, genre_spec: Optional[GenreSpec] = None) -> str:
    """Delegate to :meth:`TimeSignatureRegistry.sample_random` per R-S6 / D-09.

    When ``genre_spec.time_sig_weights`` is non-empty, uses those weights for a
    weighted draw instead of the registry's default distribution.

    Fixes Pitfall 5 in the pre-refactor code (missing return on threshold-loop
    fallthrough) because the registry implementation uses ``random.choices`` —
    see Plan 02-02 / A3 RNG-order commitment.
    """
    if genre_spec is not None and genre_spec.time_sig_weights:
        weights = genre_spec.time_sig_weights
        candidates = [ts for ts, w in weights.items() if w > 0]
        if candidates:
            w_values = [weights[ts] for ts in candidates]
            return rng.choices(candidates, weights=w_values, k=1)[0]
    return TimeSignatureRegistry.sample_random(rng)


def generate_random_swing(rng: random.Random, genre_spec: Optional[GenreSpec] = None) -> float:
    """Generate a weighted random swing value in ``[0.5, 0.75]``.

    Two draws always: ``rng.choices(...)`` for bucket + ``rng.uniform(-0.02, 0.02)``
    for variation. Favors musically appropriate values.

    When ``genre_spec`` is set, clamps result to [swing_min, swing_max] (hard bounds).

      * ``0.5``  = no swing   (30%)
      * ``0.66`` = traditional jazz (50%)
      * ``0.75`` = extreme swing (20%)
    """
    swing_weights = [
        (0.5, 0.3),   # 30% no swing
        (0.66, 0.5),  # 50% traditional jazz
        (0.75, 0.2),  # 20% extreme swing
    ]

    base_swing = rng.choices(
        [s[0] for s in swing_weights],
        weights=[s[1] for s in swing_weights],
    )[0]

    variation = rng.uniform(-0.02, 0.02)
    raw = base_swing + variation

    if genre_spec is not None:
        return min(genre_spec.swing_max, max(genre_spec.swing_min, raw))
    return min(0.75, max(0.5, raw))


def time_signature_alternative(base_time_signature: str, rng: random.Random) -> str:
    """Generate a variation of the given time signature.

    Delegates to :meth:`TimeSignatureRegistry.lookup` per R-S6; falls back to
    ``"4/4"`` when no alternatives exist for the base signature.
    """
    spec = TimeSignatureRegistry.lookup(base_time_signature)
    return rng.choice(spec.alternatives) if spec.alternatives else "4/4"


def generate_song_measures(
    time_signature: str,
    time_signature_variation: float,
    rng: random.Random,
) -> Tuple[Dict[str, int], Dict[str, str]]:
    """Generate per-part measures + per-part time-signatures for a song.

    Per-iteration RNG draws (preserved verbatim from music_gen.py:849-880 —
    RESEARCH.md §Sampler Extraction RNG trace):

      * 5x ``rng.choice([8/16 or 16/32])`` for intro/verse/chorus/bridge/outro
        base_lengths.
      * 1x ``rng.random()`` gate against ``time_signature_variation``.
      * (if gate passes) 5x ``rng.choice([ts, time_signature_alternative(ts, rng)])``
        — note that each inner ``time_signature_alternative`` call consumes 1
        additional ``rng.choice`` draw.
    """
    base_lengths = {
        'intro':  rng.choice([8, 16]),
        'verse':  rng.choice([16, 32]),
        'chorus': rng.choice([16, 32]),
        'bridge': rng.choice([8, 16]),
        'outro':  rng.choice([8, 16]),
    }

    # Sets time signatures for each part
    if rng.random() < time_signature_variation:
        signatures = {
            'intro':  rng.choice([time_signature, time_signature_alternative(time_signature, rng)]),
            'verse':  rng.choice([time_signature, time_signature_alternative(time_signature, rng)]),
            'chorus': rng.choice([time_signature, time_signature_alternative(time_signature, rng)]),
            'bridge': rng.choice([time_signature, time_signature_alternative(time_signature, rng)]),
            'outro':  rng.choice([time_signature, time_signature_alternative(time_signature, rng)]),
        }
    else:
        signatures = {part: time_signature for part in base_lengths.keys()}

    # Adjusts number of bars based on time signature
    measures = {
        part: TimeSignatureRegistry.lookup(signatures[part]).measures_for(length)
        for part, length in base_lengths.items()
    }

    return measures, signatures


def generate_song_arrangement(
    rng: random.Random,
    structures_file: Optional[str] = None,
) -> Tuple[List[str], List[str]]:
    """Generate a musical arrangement based on common music structures.

    Args:
        rng: Random.Random instance used for the arrangement choice (D-07).
        structures_file: Path to the JSON file containing song structures.
            Defaults to ``config.DEFAULT_SONG_STRUCTURES_FILE`` when None.

    Returns:
        Tuple of (unique parts, complete arrangement).
    """
    if structures_file is None:
        structures_file = config.DEFAULT_SONG_STRUCTURES_FILE
    try:
        if not os.path.exists(structures_file):
            raise FileNotFoundError(f"Structures file not found: {structures_file}")

        with open(structures_file, 'r') as f:
            data = json.load(f)

        if 'common_structures' not in data:
            raise KeyError("Missing 'common_structures' in JSON file")

        structures = data['common_structures']

        if not structures or not isinstance(structures, list):
            raise ValueError("Invalid or empty structures list")

        result = rng.choice(structures)

        # Generate a list of unique elements in the arrangement
        unique_elements = list(set(result))

        return unique_elements, result

    except (json.JSONDecodeError, FileNotFoundError, KeyError, ValueError):
        # Default structure in case of error
        default_structure = ['intro', 'verse', 'chorus', 'outro']
        logger.warning("Using default structure due to error", exc_info=True)
        return list(set(default_structure)), default_structure


def validate_measures_dict(
    measures: Dict[str, int],
    signatures: Dict[str, str],
) -> bool:
    """Validate the number of measures for each part based on its time signature.

    Cross-signature function — iterates parts and delegates per-sig validation
    to :class:`TimeSignatureRegistry`. No RNG.
    """
    for part, measure_count in measures.items():
        spec = TimeSignatureRegistry.lookup(signatures[part])
        if not spec.measure_count_valid(measure_count):
            return False
    return True


@dataclass(frozen=True)
class SongParams:
    """Frozen song-level parameters (D-20).

    Construct via :meth:`SongParams.sample` — do NOT call the constructor
    directly except in tests. The ``sample`` classmethod draws all fields in
    the canonical RNG order (D-21, RESEARCH Risk #2).
    """

    key: str
    tempo: int
    time_signature_base: str
    time_signature_variation: float
    swing_amount: float
    signatures_per_part: Dict[str, str]
    measures_per_part: Dict[str, int]
    song_unique_parts: List[str]
    song_arrangement: List[str]

    @classmethod
    def sample(
        cls,
        rng: random.Random,
        cfg: Optional[config.Config] = None,
        *,
        time_signature_variation: float = 1.0,
        genre_spec: Optional[GenreSpec] = None,
    ) -> "SongParams":
        """Draw all song-level parameters in the canonical order (D-21).

        RNG draw order (must match ``music_gen.py`` ``generate_song`` verbatim —
        RESEARCH Risk #2):

          1. :func:`generate_random_key` (rng, genre_spec)
          2. :func:`generate_random_tempo` (rng, genre_spec)
          3. :func:`generate_random_time_signature` (rng, genre_spec)
          4. :func:`generate_random_swing` (rng, genre_spec)
          5. :func:`generate_song_arrangement` (rng, structures_file=cfg.song_structures_file)
          6. LOOP UNTIL :func:`validate_measures_dict`:
                 :func:`generate_song_measures` (ts_base, ts_var, rng)

        Args:
            rng: Random.Random instance threaded through every draw.
            cfg: Optional :class:`config.Config`; when None a default instance is built.
            time_signature_variation: Probability (0.0–1.0) that per-part time
                signatures vary from the base. Keyword-only.
            genre_spec: Optional merged :class:`GenreSpec` for genre constraints. Keyword-only.
        """
        _cfg = cfg if cfg is not None else config.Config()
        key = generate_random_key(rng, genre_spec)
        tempo = generate_random_tempo(rng, genre_spec)
        time_sig_base = generate_random_time_signature(rng, genre_spec)
        swing = generate_random_swing(rng, genre_spec)
        unique_parts, arrangement = generate_song_arrangement(
            rng, structures_file=_cfg.song_structures_file,
        )
        while True:
            measures, signatures = generate_song_measures(
                time_sig_base, time_signature_variation, rng,
            )
            if validate_measures_dict(measures, signatures):
                break
        return cls(
            key=key,
            tempo=tempo,
            time_signature_base=time_sig_base,
            time_signature_variation=time_signature_variation,
            swing_amount=swing,
            signatures_per_part=signatures,
            measures_per_part=measures,
            song_unique_parts=unique_parts,
            song_arrangement=arrangement,
        )
