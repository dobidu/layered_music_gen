# Technology Stack

**Analysis Date:** 2026-04-08

## Languages

**Primary:**
- Python 3.x - All core music generation and processing logic

## Runtime

**Environment:**
- Python 3.x (specified in `/home/openclaw/musicgen/requirements.txt`)

**Package Manager:**
- pip (standard Python package manager)
- Lockfile: Not present (uses requirements.txt)

## Frameworks

**Audio Processing:**
- pydub 0.25.1+ - Audio file manipulation and format conversion
- librosa 0.9.2+ - Audio feature extraction and analysis (beat tracking, chromagram, onset detection)
- pedalboard 1.0.0+ - Real-time audio effects processing (Compressor, Gain, Chorus, LadderFilter, Phaser, Delay, Reverb)
- midi2audio 0.1.1+ - MIDI to WAV conversion using FluidSynth

**MIDI & Music Theory:**
- midiutil 1.2.1+ - MIDI file creation and manipulation
- music21 7.3.3+ - Music theory library for note handling and musical analysis

**Analysis:**
- musicality_score - Custom module (`/home/openclaw/musicgen/musicality_score.py`) for analyzing musical quality (tempo, harmony, rhythm, timbre analysis)

## Key Dependencies

**Critical:**
- numpy 1.20.0+ - Numerical computing (arrays, mathematical operations)
- scipy 1.7.0+ - Scientific computing (beat analysis, signal processing, entropy calculations)
- midiutil 1.2.1+ - MIDI file generation and track manipulation
- music21 7.3.3+ - Musical note handling, scale generation, chord theory
- librosa 0.9.2+ - Audio feature extraction (beat tracking, chroma features, onset strength)
- pydub 0.25.1+ - Audio format conversion and mixing
- pedalboard 1.0.0+ - Audio effects and digital signal processing

**Performance:**
- numba 0.56.4+ - JIT compilation for performance-critical numerical code
- llvmlite 0.39.1+ - LLVM bindings for numba acceleration

**Utility:**
- python-json-logger 2.0.7+ - Structured JSON logging
- typing-extensions 4.4.0+ - Type hints for Python versions before 3.10
- uuid 1.30+ - UUID generation for unique identifiers
- python-magic 0.4.27+ - File type detection via libmagic

## Configuration

**Environment:**
- Uses JSON configuration files for song structures, effect parameters, and soundfont selections
- Environment variables not required (all configuration file-based)

**Build:**
- No build process; pure Python script execution
- Main entry point: `/home/openclaw/musicgen/music_gen.py`

## Configuration Files

**Song Structure:**
- `song_structures.json` - Predefined song arrangement templates (intro, verse, chorus, bridge, outro sequences)

**Effect Parameters:**
- `beat_fx.json` - Beat track effect configurations (Compressor, Gain, Chorus, LadderFilter, Phaser, Delay, Reverb)
- `melody_fx.json` - Melody effect configurations
- `harmony_fx.json` - Harmony effect configurations
- `bassline_fx.json` - Bass effect configurations

**Instrument & Level Control:**
- `inst_probabilities.json` - Probability weights for instrument selection per section (intro/verse/chorus/bridge/outro)
- `levels.json` - Volume and panning settings for each instrument layer per section

**Pattern Data:**
- `chord_patterns.txt` - Predefined chord progression patterns by time signature
- `beat_roll_patterns_*.txt` - Beat roll patterns for different time signatures (2/4, 3/4, 4/4, 5/4, 6/8, 7/8, 12/8, 128th note variations)

**Soundfont Index:**
- `soundfonts.json` - Catalog of available SoundFont (.sf2) files organized by layer (melody, harmony, bassline, beat)
- `/home/openclaw/musicgen/sf/` - Directory structure containing SoundFont files:
  - `sf/melody/` - Melodic instrument soundfonts
  - `sf/harmony/` - Harmonic instrument soundfonts
  - `sf/bassline/` - Bass instrument soundfonts
  - `sf/beat/` - Drum/percussion soundfonts

## Platform Requirements

**Development:**
- Python 3.7+ (minimum)
- pip package manager
- MIDI to WAV conversion requires FluidSynth binary (via midi2audio)
- libmagic system library (for python-magic file type detection)

**Production:**
- Same as development
- FluidSynth binary must be installed on system for MIDI audio synthesis
- File system access for soundfont files and output generation

---

*Stack analysis: 2026-04-08*
