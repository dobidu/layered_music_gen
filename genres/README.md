# Genres

musicgen's genre system constrains generation parameters — tempo, swing, time signature, chord vocabulary, FX profile, and soundfont selection — to produce stylistically coherent samples. Genres compose: multiple active genres are merged into a single `GenreSpec` before generation.

---

## Built-in genres

| Genre | Tempo (BPM) | Swing | Time signatures | Key characteristics |
|---|---|---|---|---|
| **jazz** | 80–200 | 0.60–0.75 | 4/4, 3/4, 6/8, 12/8 | Ride-dominant patterns, maj7/m7/dim7 chords, syncopation |
| **hip-hop** | 70–110 | 0.50–0.65 | 4/4 dominant | Heavy kick-snare, minor-heavy keys, compressed sound |
| **blues** | 60–130 | 0.55–0.70 | 4/4, 12/8 | 12-bar dominant 7ths, shuffle feel, guitar timbre |
| **pop** | 90–140 | 0.50–0.55 | 4/4 dominant | Clean patterns, major-key bias, snare on 2 & 4 |
| **electronic** | 110–160 | 0.50–0.55 | 4/4 dominant | Four-on-floor kick, synth leads/pads/bass, heavy FX |
| **latin** | 90–140 | 0.50–0.60 | 4/4, 3/4, 6/8 | Clave syncopation, percussion tags, wide chord voicings |
| **reggae** | 60–90 | 0.50–0.58 | 4/4 dominant | One-drop (kick on beat 3), offbeat skank, bass-heavy |
| **classical** | 50–160 | 0.50–0.52 | 4/4, 3/4, 2/4, 6/8 | Wide dynamic range, orchestral timbres, varied meters |

---

## CLI usage

```bash
# Single genre
musicgen generate --seed 42 --genre jazz

# Genre composition (merged spec)
musicgen generate --seed 42 --genre jazz --genre latin

# List available genres
musicgen list-genres

# List genres from a custom directory
musicgen list-genres --genres-dir /path/to/my-genres
```

---

## `spec.json` format

Each genre lives in `genres/<name>/spec.json`. All fields are optional except `name`; omitted fields use `GenreSpec` defaults.

```json
{
  "name": "my_genre",
  "description": "One-line description shown by list-genres.",

  "tempo_min": 90.0,
  "tempo_max": 140.0,

  "swing_min": 0.50,
  "swing_max": 0.65,

  "time_sig_weights": {
    "4/4": 0.8,
    "3/4": 0.2
  },

  "scale_weights": {
    "Am": 0.3,
    "Dm": 0.2,
    "G": 0.15,
    "C": 0.15,
    "Em": 0.2
  },

  "chord_type_weights": {
    "min": 0.35,
    "maj": 0.25,
    "m7": 0.2,
    "dom7": 0.15,
    "add9": 0.05
  },

  "inversion_weights": {
    "root": 0.6,
    "first": 0.25,
    "second": 0.15
  },

  "layer_probs": {
    "beat": 1.0,
    "melody": 0.85,
    "harmony": 0.9,
    "bassline": 0.95
  },

  "fx_profile": {
    "reverb": 0.6,
    "chorus": 0.3,
    "compressor": 0.8,
    "delay": 0.2
  },

  "chord_type_hard_filter": null,

  "soundfont_tags": {
    "beat": ["drums", "percussion"],
    "melody": ["melody", "lead"],
    "harmony": ["pads", "harmony"],
    "bassline": ["bass"]
  },

  "drum_pool_names": ["my_drum_pool"]
}
```

### Field reference

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | required | Genre identifier (must match directory name) |
| `description` | `str` | `""` | One-line description for `list-genres` |
| `tempo_min` | `float` | `60.0` | Hard lower bound on generated tempo (BPM) |
| `tempo_max` | `float` | `240.0` | Hard upper bound on generated tempo (BPM) |
| `swing_min` | `float` | `0.5` | Hard lower bound on swing amount |
| `swing_max` | `float` | `0.75` | Hard upper bound on swing amount |
| `time_sig_weights` | `Dict[str, float]` | `{}` | Soft weights per time signature; empty = registry defaults |
| `scale_weights` | `Dict[str, float]` | `{}` | Soft weights per key/scale; empty = Spotify-derived defaults |
| `chord_type_weights` | `Dict[str, float]` | `{}` | Soft weights for chord type selection |
| `inversion_weights` | `Dict[str, float]` | `{}` | Soft weights for chord inversion (`root`/`first`/`second`/`third`) |
| `layer_probs` | `Dict[str, float]` | `{}` | Probability each layer is active per part |
| `fx_profile` | `Dict[str, float]` | `{}` | Multiplier on each FX effect's probability |
| `chord_type_hard_filter` | `List[str]` or `null` | `null` | If set, only these chord types are used |
| `soundfont_tags` | `Dict[str, List[str]]` | `{}` | Tags passed to SoundfontManager per layer |
| `drum_pool_names` | `List[str]` | `[]` | Named drum pool identifiers for future use |

---

## Drum pattern files

Each genre directory can contain `patterns_<sig>.txt` files where `<sig>` is the time signature flat (e.g. `patterns_44.txt` for 4/4, `patterns_34.txt` for 3/4). Patterns from all active genres are **unioned** and deduplicated — a sample drawn with `--genre jazz --genre latin` has access to both genres' drum vocabularies.

Pattern file format (one pattern per line):

```
# Comment lines are ignored
intro: 36, 42, 38, 42
verse: 36, 0, 38, 42
chorus_roll: 49, 38, 51, 42
```

Parts: `intro`, `verse`, `chorus`, `bridge`, `outro` (and `<part>_roll` for the final bar).

Common MIDI drum note numbers:
- `36` = Kick drum
- `38` = Snare
- `42` = Hi-hat (closed)
- `46` = Hi-hat (open)
- `49` = Crash cymbal
- `51` = Ride cymbal
- `39` = Hand clap
- `0` = Rest

---

## Genre composition

When multiple genres are specified, they are merged into a single `GenreSpec`:

- **Hard ranges** (`tempo_min/max`, `swing_min/max`): intersected — the merged spec is the overlap region
- **Soft weight dicts** (`time_sig_weights`, `scale_weights`, `chord_type_weights`, etc.): normalized weighted average
- **`chord_type_hard_filter`**: `None` if any genre has `null` (permissive wins); union of lists if all genres restrict
- **`soundfont_tags`**: union per layer (deduped, first-genre order preserved)
- **`drum_pool_names`**: union (deduped)

---

## Writing a custom genre

1. Create `genres/my_genre/spec.json` with at minimum `{"name": "my_genre"}`.
2. Optionally add `genres/my_genre/patterns_44.txt` (and other time signatures).
3. Test with `musicgen list-genres` — your genre should appear.
4. Generate: `musicgen generate --seed 1 --genre my_genre`.
