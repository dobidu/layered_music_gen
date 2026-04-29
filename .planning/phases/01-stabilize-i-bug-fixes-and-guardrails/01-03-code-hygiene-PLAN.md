---
phase: 01-stabilize-i-bug-fixes-and-guardrails
plan: 03
type: execute
wave: 2
depends_on: [01]
files_modified:
  - music_gen.py
  - musicality_score.py
  - requirements.txt
autonomous: true
requirements: [R-S2, R-S7, R-S8]
must_haves:
  truths:
    - "`from music21 import *` no longer appears in music_gen.py"
    - "All music21 symbols used (`roman`, `scale`, `pitch`) are imported explicitly"
    - "musicality_score.py has zero bare `except:` blocks; every Exception handler uses `logger.exception`"
    - "Dead imports (`glob`, `Pool`, `cpu_count`, `time`) are removed (or `time` is kept only if still used)"
    - "Dead variables (`ha`, `ba`, `me`, `be`, `now`, `beat_annotations`) are removed where unused"
    - "`uuid` line is removed from requirements.txt"
  artifacts:
    - path: "music_gen.py"
      provides: "Cleaned imports + explicit music21 symbols"
    - path: "musicality_score.py"
      provides: "Narrowed exception handlers with logger.exception"
    - path: "requirements.txt"
      provides: "uuid stub removed"
  key_links:
    - from: "music_gen.py imports"
      to: "music21 sub-modules used (roman, scale, pitch)"
      via: "explicit `from music21 import roman, scale, pitch`"
      pattern: "from music21 import roman.*scale.*pitch"
---

<objective>
Pay down the catalogued code hygiene debt for Phase 1: explicit `music21` imports, narrowed exception handlers in `musicality_score.py`, dead-code removal in `music_gen.py`, and removal of the bogus `uuid` PyPI line.

Purpose: Phase 2 (config + timesig registry + logging migration) and Phase 3 (package extraction) cannot cleanly grep for symbol usage while `from music21 import *` is in place. Bare-ish exception handlers in `musicality_score.py` will silently swallow scoring failures during the determinism work in Phase 5 if not narrowed now.

Output: A `music_gen.py` with explicit imports and no dead symbols, a `musicality_score.py` with narrow exception handlers, and a `requirements.txt` without the stdlib `uuid` line.

**Wave note:** This plan runs in wave 2 because it edits `music_gen.py` regions within ~15 lines of Plan 01's edits (both touch the `create_song` body around lines 1022-1046). Sequencing after Plan 01 eliminates merge-conflict risk. In wave 2 it runs in parallel with 01-02 and 01-04 (disjoint file regions / disjoint files).
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/REQUIREMENTS.md
@.planning/codebase/CONVENTIONS.md
@music_gen.py
@musicality_score.py
@requirements.txt

<interfaces>
**music21 symbols actually used in music_gen.py** (verified by grep — these are the only ones):

- `roman.RomanNumeral` (lines 225, 280, 386)
- `scale.MinorScale`, `scale.MajorScale` (lines 273, 275, 379, 381)
- `pitch.Pitch` (line 451)

Nothing else from `music21` is referenced. Note: `chord.pitches` and `note.midi` in the code are attributes of objects (`RomanNumeral.pitches` returns `Pitch` objects) — those are NOT module-level imports. They are method/attribute accesses.

**Dead imports to remove from music_gen.py (lines 1-19):**
- Line 8: `import time` — confirm by grep that `time.` is used. If `time.time()` appears (it does, around line 1028), keep this one.
- Line 12: `import glob` — verify no `glob.` calls remain.
- Line 13: `from multiprocessing import Pool, cpu_count` — verify no `Pool(` or `cpu_count(` calls remain.

**Dead variables to remove from music_gen.py:**
- `now = datetime.now()` at line 1141 (line 1142 immediately calls `datetime.now()` again — `now` is unused).
- `beat_annotations` in `generate_song_parts` (line 1092) — verify it's never read after assignment except as the dict that gets returned and then discarded.
- `ha`, `ba`, `me`, `be` in `create_song` (lines 1022-1025) — these initial empty-dict assignments are immediately overwritten on line 1033 by the tuple unpack. Delete the four pre-assignment lines.

**Bare exception handlers in musicality_score.py:**

Reading the file shows the handlers are `except Exception as e:` (NOT bare `except:`). They are at lines 66, 94, 173, 205, 239. The pattern is uniform:

```python
        except Exception as e:
            self.logger.error(f"Error in tempo analysis: {str(e)}")
            return {'stability': 0, 'reasonableness': 0, 'clarity': 0}
```

The R-S7 fix is twofold:
1. Narrow `Exception` to the specific exceptions librosa/numpy actually raise. Use `(librosa.LibrosaError, ValueError, RuntimeError, IndexError)` as the union for the analysis methods. For `calculate_musicality` (line 239), include `FileNotFoundError, OSError, librosa.LibrosaError, ValueError, RuntimeError`.
2. Replace `self.logger.error(f"... {str(e)}")` with `self.logger.exception("...")` so a stack trace is captured. Drop the `as e` binding since `logger.exception` reads it from `sys.exc_info()` automatically.
</interfaces>
</context>

<tasks>

<task type="auto" tdd="false">
  <name>Task 1: Replace `from music21 import *` and remove dead imports/variables in music_gen.py</name>
  <files>music_gen.py</files>
  <read_first>
    - music_gen.py lines 1-20 (imports)
    - music_gen.py lines 1022-1030 (create_song dead vars)
    - music_gen.py lines 1140-1145 (the unused `now` variable)
    - music_gen.py lines 1082-1115 (generate_song_parts and beat_annotations)
    - .planning/codebase/CONVENTIONS.md
  </read_first>
  <action>
**Edit A — imports block (lines 1-19):**

Replace the current import block:

```python
from midiutil import MIDIFile
from music21 import *
from pydub import AudioSegment
from midi2audio import FluidSynth
from datetime import datetime
from pedalboard import Pedalboard, Compressor, Gain, Chorus, LadderFilter, Phaser, Delay, Reverb
from pedalboard.io import AudioFile
import time
import json
import random
import os
import glob
from multiprocessing import Pool, cpu_count
import uuid
import musicality_score
from enhanced_duration_validator import DurationValidator, NoteValue

from typing import Tuple, Dict, List, Optional
import math
```

with:

```python
from midiutil import MIDIFile
from music21 import roman, scale, pitch
from pydub import AudioSegment
from midi2audio import FluidSynth
from datetime import datetime
from pedalboard import Pedalboard, Compressor, Gain, Chorus, LadderFilter, Phaser, Delay, Reverb
from pedalboard.io import AudioFile
import time
import json
import random
import os
import uuid
import musicality_score
from enhanced_duration_validator import DurationValidator, NoteValue

from typing import Tuple, Dict, List, Optional
import math
```

Changes: `from music21 import *` → `from music21 import roman, scale, pitch`. Removed: `import glob`, `from multiprocessing import Pool, cpu_count`. Kept: `import time` (used by `time.time()` in `create_song`), `import uuid` (still used by `uuid.uuid4()` at line 1142 — Plan 5 will remove that, not this plan).

**Edit B — remove dead `now` variable (line 1141):**

In `generate_song`, replace:

```python
    # Generates unique name for the song
    now = datetime.now()
    song_name = f"{datetime.now().strftime('%Y%m%d%H%M%S%f')}_{uuid.uuid4()}"
    song_name = song_name[:20]  # 20 chars
```

with:

```python
    # Generates unique name for the song
    song_name = f"{datetime.now().strftime('%Y%m%d%H%M%S%f')}_{uuid.uuid4()}"
    song_name = song_name[:20]  # 20 chars
```

(Just delete the `now = datetime.now()` line. The truncation bug at `[:20]` is left alone — Phase 5 fixes it via index-based naming per R-P1.)

**Edit C — remove dead `ha`, `ba`, `me`, `be` pre-assignments in create_song (lines 1022-1025):**

In `create_song`, delete these four lines:

```python
    ha = {}
    ba = {}
    me = {}
    be = {}
```

They are immediately overwritten by the tuple unpack on the next call (`ha, ba, me, be, an = generate_song_parts(...)`). Leave the tuple unpack call alone.

**Edit D — `beat_annotations` in generate_song_parts:**

Look at lines 1082-1115. The local name `beat_annotations` is built in the loop and returned as the 5th element of the tuple. Trace the call site at line 1033 — it's unpacked into `an`, and `an` is never used in `create_song`. Two options:

- **Option 1 (preferred — minimal):** Leave the function signature alone (don't break Phase 2's logging migration). Just rename the unused unpack target from `an` to `_` in `create_song` line 1033 to make it explicit:

```python
    ha, ba, me, be, _ = generate_song_parts(
```

That's the only change. Do NOT remove `beat_annotations` from `generate_song_parts` itself — Phase 4 will rebuild beat annotation properly.

Apply Option 1.

**Wave 2 sequencing note:** This plan depends on Plan 01 landing first. After Plan 01's edits, the `create_song` body at lines ~1022-1046 already has the arrangement insertion and the `mix_and_save` call wired. Your edits here only touch:
- lines 1-19 (imports block — untouched by Plan 01)
- the four `ha/ba/me/be = {}` lines (lines 1022-1025 — untouched by Plan 01)
- the `ha, ba, me, be, an = generate_song_parts(...)` tuple unpack line (rename `an` → `_`)
- the `now = datetime.now()` dead line (far from Plan 01's edits)

Read those regions fresh after Plan 01 has landed; line numbers may have shifted by a handful.
  </action>
  <verify>
    <automated>cd /home/openclaw/musicgen &amp;&amp; python -c "
import re
src = open('music_gen.py').read()
assert 'from music21 import *' not in src, 'wildcard music21 import still present'
assert 'from music21 import roman, scale, pitch' in src, 'explicit music21 import missing'
assert 'import glob' not in src, 'glob import still present'
assert 'from multiprocessing' not in src, 'multiprocessing import still present'
# now = datetime.now() should be gone (but datetime.now() in the f-string stays)
assert not re.search(r'^\\s*now\\s*=\\s*datetime\\.now\\(\\)', src, re.M), 'dead `now` variable still present'
# ha = {} ba = {} etc gone
assert not re.search(r'^\\s*ha\\s*=\\s*\\{\\}', src, re.M), 'dead `ha = {}` still present'
assert not re.search(r'^\\s*ba\\s*=\\s*\\{\\}', src, re.M), 'dead `ba = {}` still present'
print('OK: imports cleaned, dead vars removed')
" &amp;&amp; python -c "import music_gen; print('import OK')"</automated>
  </verify>
  <acceptance_criteria>
    - `grep -n "from music21 import \*" music_gen.py` returns nothing
    - `grep -n "from music21 import roman, scale, pitch" music_gen.py` returns exactly one match
    - `grep -n "^import glob" music_gen.py` returns nothing
    - `grep -n "from multiprocessing" music_gen.py` returns nothing
    - `grep -nE "^\s*now\s*=\s*datetime\.now\(\)" music_gen.py` returns nothing
    - `grep -nE "^\s*(ha|ba|me|be)\s*=\s*\{\}" music_gen.py` returns nothing (the four pre-assigns are gone)
    - `grep -n "ha, ba, me, be, _ = generate_song_parts" music_gen.py` returns one match
    - `python -c "import music_gen"` exits 0 with no generation output
  </acceptance_criteria>
  <done>music_gen.py uses explicit music21 imports, dead imports and variables are removed. R-S2 ✓ and R-S8 (music_gen.py portion) ✓.</done>
</task>

<task type="auto" tdd="false">
  <name>Task 2: Narrow exception handlers in musicality_score.py and switch to logger.exception</name>
  <files>musicality_score.py</files>
  <read_first>
    - musicality_score.py (full file — 264 lines, small)
    - .planning/research/PITFALLS.md (P-6 section)
  </read_first>
  <action>
There are five `except Exception as e:` blocks in `musicality_score.py` at lines 66, 94, 173, 205, 239. Replace each one as follows.

**Block at line 66 (analyze_tempo):**

```python
        except Exception as e:
            self.logger.error(f"Error in tempo analysis: {str(e)}")
            return {'stability': 0, 'reasonableness': 0, 'clarity': 0}
```

→

```python
        except (ValueError, RuntimeError, IndexError, FloatingPointError) as exc:
            self.logger.exception("Error in tempo analysis: %s", exc)
            return {'stability': 0, 'reasonableness': 0, 'clarity': 0}
```

**Block at line 94 (analyze_harmony):**

```python
        except Exception as e:
            self.logger.error(f"Error in harmony analysis: {str(e)}")
            return {'key_clarity': 0, 'stability': 0, 'consonance': 0}
```

→

```python
        except (ValueError, RuntimeError, IndexError, FloatingPointError) as exc:
            self.logger.exception("Error in harmony analysis: %s", exc)
            return {'key_clarity': 0, 'stability': 0, 'consonance': 0}
```

**Block at line 173 (analyze_rhythm):**

```python
        except Exception as e:
            self.logger.error(f"Error in rhythm analysis: {str(e)}")
            return {
                'regularity': 0,
                'strength': 0,
                'pattern': 0,
                'density': 0,
                'complexity': 0
            }
```

→

```python
        except (ValueError, RuntimeError, IndexError, FloatingPointError) as exc:
            self.logger.exception("Error in rhythm analysis: %s", exc)
            return {
                'regularity': 0,
                'strength': 0,
                'pattern': 0,
                'density': 0,
                'complexity': 0
            }
```

**Block at line 205 (analyze_noise):**

```python
        except Exception as e:
            self.logger.error(f"Error in noise analysis: {str(e)}")
            return {'noise_level': 1.0, 'music_signal_ratio': 0.0}
```

→

```python
        except (ValueError, RuntimeError, IndexError, FloatingPointError) as exc:
            self.logger.exception("Error in noise analysis: %s", exc)
            return {'noise_level': 1.0, 'music_signal_ratio': 0.0}
```

**Block at line 239 (calculate_musicality — outermost, includes file I/O):**

```python
        except Exception as e:
            self.logger.error(f"Error processing file {filename}: {str(e)}")
            return 0.0, {}
```

→

```python
        except (FileNotFoundError, OSError, ValueError, RuntimeError) as exc:
            self.logger.exception("Error processing file %s: %s", filename, exc)
            return 0.0, {}
```

Notes:
- We deliberately do NOT add `librosa.LibrosaError` because not all librosa versions expose that name. The `(ValueError, RuntimeError, IndexError, FloatingPointError)` union covers what librosa actually raises in practice.
- `logger.exception` automatically logs the traceback at ERROR level — no separate `traceback.format_exc()` needed.
- The fallback return values are unchanged so the contract for callers stays the same. Phase 5 will add an explicit `analysis_failed: true` flag in `sample.json` per R-P4.

Do NOT touch any other code in this file. Do NOT touch the `__main__` block at the bottom.
  </action>
  <verify>
    <automated>cd /home/openclaw/musicgen &amp;&amp; python -c "
src = open('musicality_score.py').read()
# No more 'except Exception'
assert 'except Exception' not in src, 'Bare/broad except Exception still present'
# logger.exception used
assert src.count('self.logger.exception') == 5, f'Expected 5 logger.exception calls, found {src.count(\"self.logger.exception\")}'
# logger.error format-string of exceptions should be gone (we replaced all 5)
assert 'self.logger.error(f\"Error in' not in src, 'Old logger.error f-strings still present'
print('OK: 5 narrowed handlers, all using logger.exception')
" &amp;&amp; python -c "import musicality_score; print('import OK')"</automated>
  </verify>
  <acceptance_criteria>
    - `grep -c "except Exception" musicality_score.py` returns `0`
    - `grep -c "self.logger.exception" musicality_score.py` returns `5`
    - `grep -c "self.logger.error(f\"Error in" musicality_score.py` returns `0`
    - `python -c "import musicality_score"` exits 0
    - The five exception types lists each include at least 3 specific exception classes (no single-`Exception` lists)
  </acceptance_criteria>
  <done>R-S7 (the `musicality_score.py` portion) is satisfied. Failures during scoring will now log full tracebacks. Phase 1 logging-migration coverage for `music_gen.py`'s `print` calls is intentionally deferred to Phase 2 per the ROADMAP.</done>
</task>

<task type="auto" tdd="false">
  <name>Task 3: Remove `uuid` line from requirements.txt</name>
  <files>requirements.txt</files>
  <read_first>
    - requirements.txt (full file)
  </read_first>
  <action>
`uuid` is a Python 3 standard library module. The `uuid>=1.30` PyPI entry is a stub package that does nothing useful and confuses dependency tooling. Remove it.

In `requirements.txt`, delete the line:

```
uuid>=1.30
```

Leave every other line — including the `# Utility Libraries` comment and the lines around it — unchanged.
  </action>
  <verify>
    <automated>cd /home/openclaw/musicgen &amp;&amp; ! grep -E "^uuid" requirements.txt &amp;&amp; echo "OK: uuid removed from requirements.txt"</automated>
  </verify>
  <acceptance_criteria>
    - `grep -E "^uuid" requirements.txt` returns nothing
    - `grep -c "python-json-logger" requirements.txt` is still `1` (other lines untouched)
    - `import uuid` still works in Python (because it's stdlib): `python -c "import uuid; print(uuid.uuid4())"` exits 0
  </acceptance_criteria>
  <done>R-S8 (requirements.txt portion) is satisfied. The misleading PyPI stub is gone.</done>
</task>

</tasks>

<verification>
After this plan:
1. `grep -n "from music21 import \*" music_gen.py` returns nothing.
2. `grep -c "except Exception" musicality_score.py` returns `0`.
3. `grep -c "logger.exception" musicality_score.py` returns `5`.
4. `grep -E "^uuid" requirements.txt` returns nothing.
5. `python -c "import music_gen"` exits 0 with no generation output.
6. `python -c "import musicality_score"` exits 0.
</verification>

<success_criteria>
- R-S2 ✓ (explicit music21 imports)
- R-S7 ✓ (musicality_score exception handlers narrowed; print → logging in music_gen.py is deferred to Phase 2)
- R-S8 ✓ (dead imports + variables + uuid stub removed)
</success_criteria>

<output>
Create `.planning/phases/01-stabilize-i-bug-fixes-and-guardrails/01-03-SUMMARY.md` with:
- Final import block of music_gen.py
- List of removed symbols (`glob`, `Pool`, `cpu_count`, `now`, `ha`, `ba`, `me`, `be`)
- Exact diff for each of the 5 exception handler rewrites in musicality_score.py
- Confirmation that uuid line is gone from requirements.txt
- Note: `print` → `logging` migration in music_gen.py is intentionally deferred to Phase 2 per ROADMAP
</output>
