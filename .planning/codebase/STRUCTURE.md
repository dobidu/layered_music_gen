# Structure

## Directory layout

```
musicgen/
├── music_gen.py                    # Main orchestrator (1161 lines, ~44KB)
├── beat_anotator.py                # Beat timing annotation utility (112 lines)
├── enhanced_duration_validator.py  # Time-signature-aware duration validator (171 lines)
├── musicality_score.py             # Audio analysis / scoring (263 lines)
│
├── requirements.txt                # Python dependencies
├── README.md
├── LICENSE
├── .gitignore
│
├── chord_patterns.txt              # Chord progression patterns
├── beats_annotations.txt
├── beat_roll_patterns_24.txt       # Drum patterns per time signature
├── beat_roll_patterns_34.txt
├── beat_roll_patterns_44.txt
├── beat_roll_patterns_54.txt
├── beat_roll_patterns_68.txt
├── beat_roll_patterns_78.txt
├── beat_roll_patterns_128.txt
│
├── song_structures.json            # Song arrangements (intro/verse/chorus/bridge/outro)
├── inst_probabilities.json         # Per-part layer inclusion probabilities
├── levels.json                     # Per-layer volume + panning per part
├── soundfonts.json
├── beat_fx.json                    # Pedalboard FX chain specs per layer
├── melody_fx.json
├── harmony_fx.json
├── bassline_fx.json
│
└── sf/                             # Soundfont assets, organized by layer
    ├── beat/
    ├── melody/
    ├── harmony/
    ├── bassline/
    └── soundfonts.txt
```

## Key locations

| Looking for... | Look in |
|---|---|
| Top-level entry | `music_gen.py:1158-1161` (bare `for` loop) |
| Song generation orchestrator | `music_gen.py:1117` `generate_song(id)` |
| Per-part component generation | `music_gen.py:1095` `create_song(...)` loop |
| Chord progression generation | `music_gen.py` `generate_chord_progression(...)` |
| Melody generation | `music_gen.py` `generate_melody(...)` |
| Bass line generation | `music_gen.py` `generate_bassline(...)` |
| Beat generation | `music_gen.py` `generate_beat(...)` |
| Mixing pipeline | `music_gen.py:758` `mix_and_save(...)` |
| Random parameter sampling | `music_gen.py:903` `generate_random_key()` and following |
| Pedalboard FX builder | `music_gen.py` `generate_pedalboard(json_file)` |
| Soundfont selection | `music_gen.py` `get_random_sound_font(dir)` |
| Note duration validation | `enhanced_duration_validator.py` `DurationValidator` |
| Audio quality metrics | `musicality_score.py` `MusicalityAnalyzer` |
| Time signature chord pattern check | `music_gen.py:22` `verify_pattern_for_time_signature` |
| Beat pattern validity check | `music_gen.py:42` `verify_beat_pattern` |

## Output layout (runtime)

Generated songs are written to a directory named with a timestamp + UUID prefix, truncated to 20 characters (`music_gen.py:1142-1143`). Inside that directory:

- `<layer>-<n>-<part>.mid` — one MIDI per layer per arrangement slot
- `<layer>-<n>-<part>.wav` — rendered audio per layer per arrangement slot (intermediate)
- `<song>-<n>.wav` — mixed audio per arrangement slot (deleted after concat, see `music_gen.py:898`)
- `<song>.wav` — final concatenated song

## Naming conventions

- **Files (Python):** `snake_case.py`. One file mixes a typo: `beat_anotator.py` (single `n`).
- **Functions:** `snake_case`. Action verbs (`generate_*`, `verify_*`, `validate_*`, `apply_*`, `get_*`, `read_*`).
- **Classes:** `PascalCase` — only `DurationValidator`, `NoteValue`, `MusicalityAnalyzer`.
- **Constants/config files:** lowercase JSON/TXT at repo root, grouped by purpose (`*_fx.json`, `beat_roll_patterns_<sig>.txt`).
- **Soundfont directories:** `sf/<layer>/` where `<layer>` ∈ `{beat, melody, harmony, bassline}`.
- **Time signature encoding:** strings like `"4/4"`, `"6/8"`, parsed inline via `map(int, ts.split('/'))`. Pattern files use `<num><denom>` (e.g. `44`, `68`, `128`).

## Module boundaries

- `music_gen.py` is a **god module**. It owns parameter sampling, all four generators, FX construction, mixing, and orchestration.
- `enhanced_duration_validator.py` and `musicality_score.py` are the only files that look like proper modules (class-based, focused responsibility).
- `beat_anotator.py` is standalone — not imported anywhere by `music_gen.py`.

## Configuration surface

Adding a new time signature requires touching all of:

1. A new `beat_roll_patterns_<sig>.txt` file
2. The hardcoded branches in `verify_pattern_for_time_signature` (`music_gen.py:22`)
3. The hardcoded branches in `verify_beat_pattern` (`music_gen.py:42`)
4. `calculate_measures_for_time_signature` (`music_gen.py:54`)
5. The time signature probability table inside `generate_random_time_signature`

This is the single biggest structural friction point in the codebase.
