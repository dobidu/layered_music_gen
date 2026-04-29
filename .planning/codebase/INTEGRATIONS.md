# External Integrations

**Analysis Date:** 2026-04-08

## Audio Libraries & APIs

**MIDI Generation:**
- midiutil - Creates MIDI files from programmatic note data
  - Client: `MIDIFile` class from midiutil
  - Usage: `/home/openclaw/musicgen/music_gen.py` functions (lines 1-200)
  - Integration: Generates chord progressions, melodies, basslines, and beats as MIDI tracks

**Music Theory:**
- music21 - Provides musical scale, chord, and note analysis
  - Client: `from music21 import *` (imported globally)
  - Usage: Throughout `/home/openclaw/musicgen/music_gen.py` for note generation and manipulation

**MIDI to Audio Synthesis:**
- midi2audio - Converts MIDI files to WAV using FluidSynth
  - Client: `FluidSynth` class from midi2audio
  - Usage: `apply_fx_to_layer()` and `mix_and_save()` functions in `/home/openclaw/musicgen/music_gen.py` (lines 727-900)
  - Dependency: Requires FluidSynth binary installed on system
  - Soundfonts: Uses .sf2 SoundFont files from `/home/openclaw/musicgen/sf/` directory

## Soundfont Integration

**SoundFont Management:**
- SoundFont files (.sf2 format) stored in: `/home/openclaw/musicgen/sf/`
- Catalog: `/home/openclaw/musicgen/soundfonts.json` defines available soundfonts by layer
- Selection: `get_random_sound_font()` function randomly selects soundfonts per layer (melody, harmony, bassline, beat)
- Audio synthesis: FluidSynth renders selected soundfonts with MIDI note data

**SoundFont Sources:**
- User-supplied: Place .sf2 files in appropriate subdirectory (sf/melody/, sf/harmony/, sf/bassline/, sf/beat/)
- Download sources documented in `/home/openclaw/musicgen/sf/soundfonts.txt`:
  - Zanderjaz.com
  - TriSamples.com
  - ProducersBuzz.com
  - GitHub free-soundfonts collection
  - Archive.org 500 Soundfonts collection

## Audio Processing & Effects

**Real-time Effects:**
- pedalboard - Audio effects processing chain
  - Location: `/home/openclaw/musicgen/music_gen.py` (lines 695-760)
  - Effects available:
    - Compressor (threshold_db, ratio, attack_ms, release_ms)
    - Gain (gain_db)
    - Chorus (depth, feedback, rate_hz)
    - LadderFilter (cutoff_hz, resonance)
    - Phaser (depth, feedback, rate_hz, centre_frequency_hz)
    - Delay (delay_seconds, mix)
    - Reverb (room_size)
  - Configuration: Per-layer effect probability and parameter ranges in JSON files:
    - `beat_fx.json`
    - `melody_fx.json`
    - `harmony_fx.json`
    - `bassline_fx.json`

**Audio Feature Analysis:**
- librosa - Audio signal analysis
  - Functions: Beat tracking, chroma feature extraction, onset strength detection
  - Usage: `musicality_score.py` module for analyzing generated audio quality
  - Integration points:
    - `librosa.beat.beat_track()` - Tempo and beat analysis
    - `librosa.feature.chroma_cqt()` - Harmonic content analysis
    - `librosa.feature.tonnetz()` - Tonal feature extraction
    - `librosa.onset.onset_strength()` - Rhythm clarity measurement

**Audio Format Conversion:**
- pydub - Audio file manipulation
  - Client: `AudioSegment` class
  - Usage: `mix_and_save()` function for combining tracks and exporting
  - Supported formats: WAV, MP3, OGG (via ffmpeg/libav)

## Data Storage

**Configuration Files (Read-only):**
- JSON files: Song structures, effect parameters, instrument probabilities, volume/pan levels
  - `song_structures.json` - Song arrangement templates
  - `inst_probabilities.json` - Layer selection probabilities
  - `levels.json` - Volume and panning per section
  - `soundfonts.json` - Soundfont catalog

**Pattern Files (Read-only):**
- Text files with chord and beat patterns
  - `chord_patterns.txt` - Chord progressions
  - `beat_roll_patterns_*.txt` - Beat roll patterns (separated by time signature)
  - `beats_annotations.txt` - Beat annotation templates

**Output Storage:**
- Local file system only
- MIDI files: Generated in memory, saved to disk
- WAV files: Generated from MIDI synthesis, saved per track and mixed
- File naming: Uses timestamp and UUID for uniqueness (e.g., `20260408180500_abc12-20`)

## Internal Audio Processing Pipeline

**Generation Flow:**
1. Generate chord progression as MIDI (`generate_chord_progression()`)
2. Generate melody MIDI from chord progression (`generate_melody()`)
3. Generate bassline MIDI from chord progression and melody (`generate_bassline()`)
4. Generate beat MIDI with drum patterns (`generate_beat()`)
5. Convert MIDI tracks to WAV using FluidSynth with selected soundfonts (`apply_fx_to_layer()`)
6. Apply audio effects from pedalboard per track
7. Mix all tracks together with volume/panning from `levels.json` (`mix_and_save()`)

**Musicality Analysis:**
- Custom module: `/home/openclaw/musicgen/musicality_score.py`
- Analyzes generated audio using librosa for:
  - Tempo stability and reasonableness
  - Harmonic clarity and consonance
  - Rhythmic coherence
  - Timbre characteristics
  - Noise/artifact detection

## Validation

**Time Signature Validation:**
- Custom module: `/home/openclaw/musicgen/enhanced_duration_validator.py`
- Validates note durations against time signature requirements
- Supports:
  - Simple meters (2/4, 3/4, 4/4)
  - Compound meters (6/8, 9/8, 12/8)
  - Duration caching for performance

**Pattern Validation:**
- Functions: `verify_pattern_for_time_signature()`, `verify_beat_pattern()`, `validate_measures()`
- Ensures generated patterns conform to time signature rules

## Logging & Monitoring

**Logging Framework:**
- python-json-logger - Structured JSON logging (if configured)
- Standard Python logging module for internal analysis warnings/errors
- Musicality analyzer logs analysis steps and errors to logger

## Dependencies at System Level

**Required System Binaries:**
- FluidSynth - MIDI to audio synthesis engine (invoked via midi2audio)
- libmagic - File type detection (for python-magic)
- ffmpeg/libav - Audio codec support (for pydub format conversion)

---

*Integration audit: 2026-04-08*
