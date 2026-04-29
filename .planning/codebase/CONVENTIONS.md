# Coding Conventions

**Analysis Date:** 2026-04-08

## Naming Patterns

**Files:**
- Lowercase with underscores: `music_gen.py`, `beat_anotator.py`, `enhanced_duration_validator.py`, `musicality_score.py`
- Purpose-based naming clearly indicates file function

**Functions:**
- Lowercase with underscores (snake_case): `verify_pattern_for_time_signature()`, `generate_chord_progression()`, `extract_midi_beats()`, `calculate_musicality()`
- Descriptive verb-based names starting with action: `verify_*`, `generate_*`, `extract_*`, `calculate_*`, `analyze_*`, `create_*`, `get_*`
- Helper functions follow convention: `get_note_duration()`, `get_levels()`, `get_random_sound_font()`

**Variables:**
- Lowercase with underscores: `time_signature`, `chord_pattern`, `part_counter`, `song_parts`
- Single letters used sparingly in mathematical contexts: `y`, `sr` (in librosa context), `i`, `e`
- Descriptive names preferred: `theoretical_beats`, `midi_beats`, `remaining_beats` rather than `t_beats`, `m_beats`
- Dictionary keys use lowercase: `'tempo'`, `'harmony'`, `'rhythm'`, `'noise'`

**Types:**
- Use UPPERCASE for Enum values: `NoteValue.WHOLE`, `NoteValue.QUARTER`, `NoteValue.EIGHTH`
- Class names use PascalCase: `DurationValidator`, `MusicalityAnalyzer`, `TimeSignatureInfo`

## Code Style

**Formatting:**
- No formal linter (eslintrc, black config, or prettier config) detected
- Imports organized with standard library first, then third-party, then local modules (lines 1-19 in `music_gen.py`)
- Type hints used throughout: `def verify_pattern_for_time_signature(chord_pattern: List[str], time_signature: str) -> bool`
- Functions have docstrings using triple-quoted format

**Linting:**
- No formal linting tool detected (no `.flake8`, `.pylintrc`, `pylint.ini`, or `pyproject.toml`)
- Code style is consistent but not enforced through tooling

## Import Organization

**Order:**
1. Standard library imports: `json`, `os`, `random`, `time`, `uuid`, `datetime`
2. Third-party packages: `numpy`, `midiutil`, `music21`, `librosa`, `pydub`, `pedalboard`, `scipy`, `mido`
3. Type hints: `from typing import Tuple, Dict, List, Optional`
4. Local module imports: `import musicality_score`, `from enhanced_duration_validator import DurationValidator, NoteValue`

**Path Aliases:**
- No path aliases detected (`@` or absolute imports not used in analyzed files)
- Relative imports used: `from enhanced_duration_validator import DurationValidator, NoteValue`

## Error Handling

**Patterns:**
- Explicit exceptions raised for invalid states: `raise ValueError(f"Unsupported time signature denominator: {denominator}")` (line 102 in `music_gen.py`)
- Try-except blocks used selectively: `beat_anotator.py` line 625 wraps JSON parsing in try-except
- Broad exception catching: `except (json.JSONDecodeError, FileNotFoundError, KeyError, ValueError) as e` (line 647 in `music_gen.py`)
- Guard clauses for existence checks: `if not os.path.exists(directory)` followed by `os.makedirs(directory)`
- Early returns on invalid conditions: `if not filename: raise ValueError(...)`
- Validation functions return boolean tuples: `compare_beats()` returns `(bool, str)` with validation result and message
- Warning messages printed to console for non-fatal issues: `print(f"Warning: Part '{part}' - {message}")` in `beat_anotator.py`

## Logging

**Framework:** 
- Logging module used selectively: configured in `musicality_score.py` (line 12-14) and `enhanced_duration_validator.py` (line 6)
- Logger instantiation: `self.logger = logging.getLogger(__name__)` (line 14 in `musicality_score.py`, line 36 in `enhanced_duration_validator.py`)
- Fallback to `print()` statements for general output in `music_gen.py` and `beat_anotator.py`

**Patterns:**
- ERROR level for exceptions: `self.logger.error(f"Error in tempo analysis: {str(e)}")` (line 67 in `musicality_score.py`)
- WARNING level for validation failures: `self.logger.warning(f"{layer_type} layer duration does not complete full measures")` (line 152 in `enhanced_duration_validator.py`)
- Print statements used for progress/status: `print(f"Annotations saved to: {output_file}")`, `print("Mixing part: " + part + ...)`
- Informational prints for user feedback: `print("Song saved as: " + song_file_wav)`

## Comments

**When to Comment:**
- Comments provided for complex logic: "Para compassos compostos (6/8, 12/8)" (line 28 in `music_gen.py`)
- Comments explain business logic: "# Doubling for compound measures" (line 59 in `music_gen.py`)
- Comments in Portuguese found alongside English code (language mixing observed)
- Comments sparse for straightforward operations

**JSDoc/TSDoc:**
- Google-style docstrings used: triple quotes with description line and argument documentation (e.g., `beat_anotator.py` lines 21-27)
- Format: description, blank line, Args section with parameter names and types
- Example from `enhanced_duration_validator.py` lines 91-99:
```python
def get_valid_duration(self, duration: float, time_signature: str, 
                      remaining_beats: float, layer_type: str) -> float:
    """
    Retorna uma duração válida para o contexto específico.
    
    Args:
        duration: Duração proposta
        time_signature: Assinatura de tempo
        remaining_beats: Beats restantes no compasso
        layer_type: Tipo de camada ('melody', 'chord', 'bass', 'beat')
    """
```

## Function Design

**Size:**
- Functions range from 10-40 lines typically
- Large functions exist: `generate_chord_progression()` ~80 lines, `generate_melody()` ~110 lines, `generate_beat()` ~110 lines, `mix_and_save()` ~150+ lines

**Parameters:**
- Functions accept 4-9 parameters typically: `generate_chord_progression(key, tempo, time_signature, measures, name, part, pattern_file)`
- Tuple unpacking used for return values: `numerator, denominator = map(int, time_signature.split('/'))`
- Optional parameters used implicitly through default behavior, not None defaults (exception: `Optional` type hints used)

**Return Values:**
- Single returns common: `return len(chord_pattern) in [2, 3, 6]`
- Tuple returns for multiple values: `return numerator, midi_denominator` (line 104 in `music_gen.py`)
- Tuple returns with validation: `return True, "Beats are aligned."` (line 52 in `beat_anotator.py`)
- Dictionary returns for complex data: `return {'stability': tempo_stability, 'reasonableness': tempo_reasonableness, 'clarity': tempo_clarity}` (line 61 in `musicality_score.py`)

## Module Design

**Exports:**
- Main function convention used: `if __name__ == "__main__"` block (line 106 in `beat_anotator.py`, line 263 in `musicality_score.py`)
- Classes exported implicitly through module imports: `from enhanced_duration_validator import DurationValidator, NoteValue`
- Utility functions used as module-level helpers

**Barrel Files:**
- No barrel files detected
- Direct imports from modules used: `import musicality_score`, `from enhanced_duration_validator import DurationValidator, NoteValue`

---

*Convention analysis: 2026-04-08*
