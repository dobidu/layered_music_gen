# Layered Random Music - MIR Dataset Generator

A Python-based system for generating synthesized musical pieces for Music Information Retrieval (MIR) research and development. This tool creates complete musical compositions with multiple layers (melody, harmony, bassline, and percussion) following musical theory principles and conventions.

## Features

- Multi-layer music generation (melody, harmony, bass, and percussion)
- Support for multiple time signatures (4/4, 3/4, 2/4, 6/8, 5/4, 7/8, 12/8)
- Dynamic song structure generation
- Intelligent musical arrangement
- MIDI and WAV file generation
- Audio effects processing through virtual pedalboard
- Musicality scoring system
- Support for various musical scales and modes
- Markov chain-based melodic generation
- Automated mixing and mastering

## Requirements

- Python 3.8+
- librosa
- numpy
- scipy
- midiutil
- music21
- pydub
- pedalboard
- FluidSynth
- SoundFont (.sf2) files

## Installation

1. Clone the repository:
```bash
git clone https://github.com/dobidu/layered_music_gen.git
```

2. Install the required Python packages:
```bash
pip install -r requirements.txt
```

3. Install FluidSynth:
```bash
# Ubuntu/Debian
sudo apt-get install fluidsynth

# macOS
brew install fluidsynth

# Windows
# Download and install from https://www.fluidsynth.org/
```

4. Place your SoundFont (.sf2) files in the appropriate directories:
```
sf/
├── beat/
├── melody/
├── harmony/
└── bassline/
```

## Usage

To generate a single music piece:

```python
python music_gen.py
```

The generator creates:
- MIDI files for each musical layer
- WAV files for the final mix
- JSON metadata file with musical parameters and analysis

## Configuration Files

- `song_structures.json`: Defines possible song arrangements
- `chord_patterns.txt`: Contains chord progressions for different song parts
- `beat_roll_patterns_*.txt`: Defines rhythm patterns for different time signatures
- `inst_probabilities.json`: Controls instrument layer probabilities
- `*_fx.json`: Effect parameters for each instrument layer

## Musical Features

### Time Signatures
- 4/4 (Common time)
- 3/4 (Waltz time)
- 2/4 (March time)
- 6/8 (Compound duple)
- 5/4 (Quintuple)
- 7/8 (Septuple)
- 12/8 (Compound quadruple)

### Song Parts
- Intro
- Verse
- Chorus
- Bridge
- Outro

### Audio Effects
- Compression
- Reverb
- Delay
- Chorus
- Phaser
- Filter
- Gain

## Musicality Analysis

The system includes a sophisticated musicality scoring system that analyzes:
- Tempo characteristics
- Harmonic content
- Rhythmic features
- Timbre quality
- Signal-to-noise ratio

## Output

Each generated piece includes:
- Layer-specific MIDI files
- Mixed WAV file
- JSON metadata with:
  - Musical parameters
  - Song structure
  - Effect settings
  - Musicality scores
  - Timing information

## Contributing

Contributions are welcome! Please feel free to submit pull requests.

## License

See LICENSE

## Acknowledgments

- Built with music21 and librosa
- Uses FluidSynth for MIDI synthesis
- Pedalboard for audio effects processing

