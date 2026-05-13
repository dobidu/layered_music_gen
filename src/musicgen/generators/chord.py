"""Chord-progression generator (extracted from music_gen.py per Plan 03-04 / R-X3).

Generates a chord progression MIDI file for one song part, given a key, tempo,
time signature, pattern file, and an injected ``rng: random.Random``. All RNG
draws go through ``rng`` so Phase 5 can feed per-sample deterministic RNGs
without rewriting this generator.

v0.2 Phase 2: extended chord vocabulary (types + inversions), genre-constrained.
  - CHORD_TYPES: 13 chord types (triads, 7ths, 9ths, sus, dim)
  - INVERSION_NAMES: root / first / second / third
  - _pick_chord_type(rng, base_symbol, genre_spec): hard filter + soft weight draw
  - _pick_inversion(rng, genre_spec): weighted inversion draw (default: root)
  - _build_chord_voicing(numeral, chord_type, inversion, key): MIDI note list
  - generate_chord_progression gains optional genre_spec kwarg; backward compat preserved

Design:
  D-06 — Uses TimeSignatureRegistry.lookup(...) directly.
  D-07 — Zero bare random.<method> calls — all draws use injected rng.
  D-22 — Takes per-part fields, not SongParams.
  D-23 — music21 roman.RomanNumeral audited clean (does not mutate global random).
"""
import logging
import os
import random
from typing import Dict, List, Optional, Tuple

from midiutil import MIDIFile
from music21 import roman, scale, pitch  # Plan 01-03 narrow-import commitment (S3)

from musicgen.duration_validator import DurationValidator
from timesig import TimeSignatureRegistry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Chord vocabulary
# ---------------------------------------------------------------------------

# Semitone intervals above root for each chord type (relative to root=0).
# All voicings placed in octave starting at root; inversions applied after.
_CHORD_INTERVALS: Dict[str, List[int]] = {
    "maj":   [0, 4, 7],
    "min":   [0, 3, 7],
    "dom7":  [0, 4, 7, 10],
    "maj7":  [0, 4, 7, 11],
    "m7":    [0, 3, 7, 10],
    "m7b5":  [0, 3, 6, 10],
    "dim7":  [0, 3, 6, 9],
    "sus2":  [0, 2, 7],
    "sus4":  [0, 5, 7],
    "9":     [0, 4, 7, 10, 14],
    "maj9":  [0, 4, 7, 11, 14],
    "m9":    [0, 3, 7, 10, 14],
    "add9":  [0, 4, 7, 14],
}

CHORD_TYPES: List[str] = list(_CHORD_INTERVALS)
INVERSION_NAMES: List[str] = ["root", "first", "second", "third"]

# Octave 4 base (C4 = MIDI 60); root note derived from roman numeral + key
_C4 = 60
_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_ENHARMONIC = {
    # single flats → sharps
    "Cb": "B",  "Db": "C#", "Eb": "D#", "Fb": "E",
    "Gb": "F#", "Ab": "G#", "Bb": "A#",
    # double sharps (x / ##)
    "C##": "D", "D##": "E", "E##": "F#", "F##": "G",
    "G##": "A", "A##": "B", "B##": "C#",
    # double flats
    "Cbb": "A#", "Dbb": "C", "Ebb": "D", "Fbb": "D#",
    "Gbb": "F",  "Abb": "G", "Bbb": "A",
}

# Default type weight when no genre spec supplied — uniform over triads + dom7
# (keeps backward-compat feel while exposing the mechanism)
_DEFAULT_TYPE_WEIGHTS: Dict[str, float] = {"maj": 0.4, "min": 0.4, "dom7": 0.2}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _root_midi(numeral: str, key: str) -> int:
    """Return MIDI note number of chord root in octave 4."""
    rn = roman.RomanNumeral(numeral.strip(), key)
    root_name = rn.root().name.replace("-", "b").replace("♭", "b").replace("~", "")
    root_name = _ENHARMONIC.get(root_name, root_name)
    semitone = _NOTE_NAMES.index(root_name)
    return _C4 + semitone


def _build_chord_voicing(
    numeral: str,
    chord_type: str,
    inversion: str,
    key: str,
) -> List[int]:
    """Build a list of MIDI note numbers for a chord voicing.

    Inversion is applied by rotating the note stack: the lowest n notes are
    moved up one octave, where n = inversion index (0=root, 1=first, 2=second,
    3=third). Third inversion falls back to second for triads (< 4 notes).
    All notes guaranteed in MIDI range 0–127.
    """
    root = _root_midi(numeral, key)
    intervals = _CHORD_INTERVALS.get(chord_type, _CHORD_INTERVALS["maj"])
    notes = [root + i for i in intervals]

    inv_index = INVERSION_NAMES.index(inversion) if inversion in INVERSION_NAMES else 0
    # Clamp third inversion to second for chords with fewer than 4 notes
    inv_index = min(inv_index, len(notes) - 1)

    # Rotate: move bottom inv_index notes up one octave
    for i in range(inv_index):
        notes[i] += 12

    # Re-sort ascending after rotation
    notes.sort()

    # Clamp to MIDI range
    while notes and notes[-1] > 127:
        notes = [n - 12 for n in notes]
    while notes and notes[0] < 0:
        notes = [n + 12 for n in notes]

    return notes


def _pick_chord_type(
    rng: random.Random,
    base_symbol: str,
    genre_spec,
) -> str:
    """Pick a chord type given genre constraints.

    Hard filter: if genre_spec.chord_type_hard_filter is set, only those types
    are candidates. Soft weights: genre_spec.chord_type_weights shifts the
    probability distribution. Falls back to _DEFAULT_TYPE_WEIGHTS when no spec.
    """
    if genre_spec is None:
        weights = _DEFAULT_TYPE_WEIGHTS
        candidates = list(weights)
    else:
        hard_filter = genre_spec.chord_type_hard_filter
        candidates = list(hard_filter) if hard_filter is not None else CHORD_TYPES
        weights_raw = genre_spec.chord_type_weights
        if weights_raw:
            weights = {c: weights_raw.get(c, 0.0) for c in candidates}
        else:
            weights = {c: 1.0 for c in candidates}

    # Remove zero-weight candidates
    weights = {c: w for c, w in weights.items() if w > 0}
    if not weights:
        weights = {c: 1.0 for c in candidates} if candidates else {"maj": 1.0}

    population = list(weights)
    w_values = [weights[c] for c in population]
    return rng.choices(population, weights=w_values, k=1)[0]


def _sample_chord_markov(
    history: List[str],
    matrix: Dict,
    rng: random.Random,
) -> str:
    """Sample next chord from a Markov transition matrix.

    Boundary handling:
    - step 0 (empty history): draw from init_probs
    - step 1 (1 chord in history): look up single-chord key; fallback to uniform over init_probs
    - step ≥ 2: look up "prev,curr" key (2nd-order); fallback to single-chord key (1st-order);
      fallback to uniform over init_probs

    matrix dict keys: "order", "init_probs", "transitions"
    """
    transitions: Dict[str, Dict[str, float]] = matrix.get("transitions", {})
    init_probs: Dict[str, float] = matrix.get("init_probs", {})

    def _draw(dist: Dict[str, float]) -> str:
        dist = {k: v for k, v in dist.items() if v > 0}
        if not dist:
            dist = {k: 1.0 for k in init_probs} if init_probs else {"I": 1.0}
        population = list(dist)
        return rng.choices(population, weights=[dist[k] for k in population], k=1)[0]

    if not history:
        return _draw(init_probs)

    curr = history[-1]
    if len(history) >= 2:
        prev = history[-2]
        key_2nd = f"{prev},{curr}"
        if key_2nd in transitions:
            return _draw(transitions[key_2nd])

    if curr in transitions:
        return _draw(transitions[curr])

    return _draw(init_probs)


def _pick_inversion(rng: random.Random, genre_spec) -> str:
    """Pick an inversion name. Defaults to 'root' when no genre spec."""
    if genre_spec is None or not genre_spec.inversion_weights:
        return "root"
    iw = genre_spec.inversion_weights
    candidates = [n for n in INVERSION_NAMES if iw.get(n, 0.0) > 0]
    if not candidates:
        return "root"
    w_values = [iw[c] for c in candidates]
    return rng.choices(candidates, weights=w_values, k=1)[0]


# ---------------------------------------------------------------------------
# Public generator
# ---------------------------------------------------------------------------

def generate_chord_progression(
    key: str,
    tempo: int,
    time_signature: str,
    measures: int,
    name: str,
    part: str,
    pattern_file: str,
    rng: random.Random,
    genre_spec=None,
) -> Tuple[List[str], str]:
    """Generate a chord progression MIDI file for one song part.

    genre_spec: optional GenreSpec — when supplied, chord type and inversion
    are drawn per-chord according to genre constraints. When None, behavior is
    identical to pre-v0.2 (backward compat).

    v0.3 Phase 1: if genre_spec.chord_transition_matrix is set, the chord
    sequence is generated via a Markov chain instead of the pattern file.
    The pattern list returned is one chord per measure (len == measures).
    """
    validator = DurationValidator()
    mf = MIDIFile(1)
    track = 0
    time = 0

    mf.addTrackName(track, time, "Chord Progression")
    mf.addTempo(track, time, tempo)
    spec = TimeSignatureRegistry.lookup(time_signature)
    numerator = spec.numerator
    midi_denominator = spec.midi_denominator_power
    mf.addTimeSignature(track, time, numerator, midi_denominator, 24, 8)

    base_duration = validator.get_suggested_duration(time_signature, "chord")
    chord_duration = validator.get_valid_duration(
        base_duration,
        time_signature,
        validator._analyze_time_signature(time_signature).beats_per_measure,
        "chord",
    )
    if chord_duration <= 0:
        raise ValueError(
            f"Invalid chord duration calculated for time signature {time_signature}"
        )

    # Markov path: genre_spec has a chord_transition_matrix — generate one
    # chord per measure via Markov sampling; skip pattern file entirely.
    matrix = getattr(genre_spec, "chord_transition_matrix", None) if genre_spec is not None else None
    if matrix is not None:
        chord_pattern: List[str] = []
        for _ in range(measures):
            chord_symbol = _sample_chord_markov(chord_pattern, matrix, rng)
            chord_pattern.append(chord_symbol)
            chord_type = _pick_chord_type(rng, chord_symbol, genre_spec)
            inversion = _pick_inversion(rng, genre_spec)
            notes = _build_chord_voicing(chord_symbol.strip(), chord_type, inversion, key)
            for note in notes:
                mf.addNote(track, 0, note, len(chord_pattern) - 1, chord_duration, 100)
        current_time = measures * chord_duration
    else:
        # Pattern-file path (pre-v0.3 behavior).
        chord_patterns: Dict[str, List] = {}
        with open(pattern_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if ":" not in line:
                    continue
                part_name, pattern = line.split(":", 1)
                pattern_chords = pattern.split(",")
                if spec.verify_chord_pattern_length(len(pattern_chords)):
                    chord_patterns.setdefault(part_name, []).append(pattern_chords)

        if part not in chord_patterns or not chord_patterns[part]:
            base_pattern = ["I"] if numerator in [2, 3] else ["I", "IV", "V", "vi"]
            chord_patterns[part] = [base_pattern]

        chord_pattern = rng.choice(chord_patterns[part])
        # music21 global-random audit (Phase 3, D-23): music21 9.9.1 roman.RomanNumeral,
        # scale.MajorScale/MinorScale, pitch.Pitch do NOT mutate random.getstate().
        # Verified 2026-04-18. If this changes, tests/test_music21_isolation.py fails.
        #
        # Backward compat (genre_spec=None): use music21 pitches directly — zero new
        # RNG draws, bit-identical to pre-v0.2 output (preserves determinism goldens).
        # Genre path: _pick_chord_type + _pick_inversion consume RNG draws per chord.
        current_time = 0
        for _ in range(measures):
            for symbol in chord_pattern:
                if genre_spec is None:
                    chord = roman.RomanNumeral(symbol.strip(), key)
                    notes = [p.midi for p in chord.pitches]
                else:
                    chord_type = _pick_chord_type(rng, symbol, genre_spec)
                    inversion = _pick_inversion(rng, genre_spec)
                    notes = _build_chord_voicing(symbol.strip(), chord_type, inversion, key)
                for note in notes:
                    mf.addNote(track, 0, note, current_time, chord_duration, 100)
                current_time += chord_duration

    directory = name.split("-")[0]
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = os.path.join(directory, name + "-chord_progression.mid")

    with open(filename, "wb") as outf:
        mf.writeFile(outf)

    logger.debug("Chord progression: %s | type picks via genre=%s", chord_pattern,
                 genre_spec.name if genre_spec else "none")

    return chord_pattern, filename
