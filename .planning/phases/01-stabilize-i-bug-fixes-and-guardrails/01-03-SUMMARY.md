---
phase: 01-stabilize-i-bug-fixes-and-guardrails
plan: 03
subsystem: hygiene
tags: [refactor, imports, exceptions, dead-code, R-S2, R-S7, R-S8]
requirements: [R-S2, R-S7, R-S8]
requires:
  - "Plan 01-01 (importability + arrangement fix)"
provides:
  - "Explicit music21 symbol set (roman, scale, pitch) — greppable for Phase 2/3"
  - "Narrow exception handlers in musicality_score.py with full stack-trace logging"
  - "Clean requirements.txt (no stdlib uuid stub)"
affects:
  - "music_gen.py imports and create_song/generate_song bodies"
  - "musicality_score.py exception handlers (5 blocks)"
  - "requirements.txt"
tech-stack:
  added: []
  patterns:
    - "Explicit named imports over wildcard imports"
    - "logger.exception(...) for capturing tracebacks in except blocks"
    - "Narrow exception tuples instead of broad Exception"
key-files:
  created: []
  modified:
    - music_gen.py
    - musicality_score.py
    - requirements.txt
decisions:
  - "music21 symbols used are exactly {roman, scale, pitch}; verified by grep"
  - "Kept `import time` (used by time.time() in create_song) and `import uuid` (still used by uuid.uuid4() in generate_song — removal deferred to Phase 5 R-P1)"
  - "Exception unions use (ValueError, RuntimeError, IndexError, FloatingPointError) for analysis methods; outer calculate_musicality adds FileNotFoundError, OSError"
  - "Did NOT add librosa.LibrosaError because not all librosa versions expose that symbol"
  - "Did NOT remove `beat_annotations` from generate_song_parts — the caller now unpacks it into `_` (Option 1 from plan). Phase 4 will rebuild beat annotation properly"
metrics:
  duration: "~5m"
  tasks_completed: 3
  files_modified: 3
  completed: 2026-04-08
---

# Phase 01 Plan 03: Code Hygiene Summary

One-liner: Explicit music21 imports, narrow exception handlers with `logger.exception`, dead-code removal, and drop of the stdlib `uuid` PyPI stub.

## What Changed

### music_gen.py — imports block

**Final import block (lines 1-17):**

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

Removed symbols:
- `from music21 import *` (replaced with explicit `roman, scale, pitch`)
- `import glob` (no `glob.` calls remained)
- `from multiprocessing import Pool, cpu_count` (no `Pool(` or `cpu_count(` calls)

Kept on purpose:
- `import time` — still used by `time.time()` in `create_song`
- `import uuid` — still used by `uuid.uuid4()` in `generate_song`; Phase 5 R-P1 will replace the whole naming scheme

### music_gen.py — dead variables

Removed:
- `ha = {}`, `ba = {}`, `me = {}`, `be = {}` pre-assignments in `create_song` (overwritten immediately by the `generate_song_parts` tuple unpack on the next call)
- `now = datetime.now()` line in `generate_song` (was never read; the next line calls `datetime.now()` again for the f-string)

Renamed:
- Tuple unpack `ha, ba, me, be, an = generate_song_parts(...)` → `ha, ba, me, be, _ = generate_song_parts(...)` to make the unused `beat_annotations` slot explicit. Function signature of `generate_song_parts` left untouched per plan Option 1 (Phase 4 will rebuild beat annotation).

### musicality_score.py — exception handler rewrites

All five broad `except Exception as e` blocks narrowed and converted to `logger.exception`. Fallback return values unchanged.

**Block 1 — analyze_tempo (line 66):**

```diff
-        except Exception as e:
-            self.logger.error(f"Error in tempo analysis: {str(e)}")
+        except (ValueError, RuntimeError, IndexError, FloatingPointError) as exc:
+            self.logger.exception("Error in tempo analysis: %s", exc)
             return {'stability': 0, 'reasonableness': 0, 'clarity': 0}
```

**Block 2 — analyze_harmony (line 94):**

```diff
-        except Exception as e:
-            self.logger.error(f"Error in harmony analysis: {str(e)}")
+        except (ValueError, RuntimeError, IndexError, FloatingPointError) as exc:
+            self.logger.exception("Error in harmony analysis: %s", exc)
             return {'key_clarity': 0, 'stability': 0, 'consonance': 0}
```

**Block 3 — analyze_rhythm (line 173):**

```diff
-        except Exception as e:
-            self.logger.error(f"Error in rhythm analysis: {str(e)}")
+        except (ValueError, RuntimeError, IndexError, FloatingPointError) as exc:
+            self.logger.exception("Error in rhythm analysis: %s", exc)
             return {
                 'regularity': 0,
                 'strength': 0,
                 'pattern': 0,
                 'density': 0,
                 'complexity': 0
             }
```

**Block 4 — analyze_noise (line 205):**

```diff
-        except Exception as e:
-            self.logger.error(f"Error in noise analysis: {str(e)}")
+        except (ValueError, RuntimeError, IndexError, FloatingPointError) as exc:
+            self.logger.exception("Error in noise analysis: %s", exc)
             return {'noise_level': 1.0, 'music_signal_ratio': 0.0}
```

**Block 5 — calculate_musicality (line 239, outermost, includes file I/O):**

```diff
-        except Exception as e:
-            self.logger.error(f"Error processing file {filename}: {str(e)}")
+        except (FileNotFoundError, OSError, ValueError, RuntimeError) as exc:
+            self.logger.exception("Error processing file %s: %s", filename, exc)
             return 0.0, {}
```

### requirements.txt — uuid stub removal

Removed the line:

```
uuid>=1.30
```

`uuid` is a Python 3 stdlib module; the PyPI entry is a misleading stub. All other lines (including `python-json-logger>=2.0.7` and `typing-extensions>=4.4.0`) untouched.

## Verification

Acceptance checks performed:

| Check | Result |
|-------|--------|
| `grep -n "from music21 import \*" music_gen.py` | empty (OK) |
| `grep -n "from music21 import roman, scale, pitch" music_gen.py` | 1 match (OK) |
| `grep -n "^import glob" music_gen.py` | empty (OK) |
| `grep -n "from multiprocessing" music_gen.py` | empty (OK) |
| `grep -nE "^\s*now\s*=\s*datetime\.now\(\)" music_gen.py` | empty (OK) |
| `grep -nE "^\s*(ha\|ba\|me\|be)\s*=\s*\{\}" music_gen.py` | empty (OK) |
| `grep -n "ha, ba, me, be, _ = generate_song_parts" music_gen.py` | 1 match (OK) |
| `grep -c "except Exception" musicality_score.py` | 0 (OK) |
| `grep -c "logger.exception" musicality_score.py` | 5 (OK) |
| `grep -E "^uuid" requirements.txt` | empty (OK) |
| `python3 -m py_compile music_gen.py musicality_score.py` | clean (OK) |
| `python3 -c "import uuid; print(uuid.uuid4())"` | prints UUID (stdlib works) |

Note: full `python -c "import music_gen"` runtime import-check not executed because `music21` is not installed in this sandbox. Syntax compile (`py_compile`) passes cleanly, which is the check available here.

## Deviations from Plan

None — plan executed exactly as written. All three tasks applied the exact edits described in `<action>` sections. No Rule 1/2/3 auto-fixes were needed, and no Rule 4 architectural checkpoints were hit.

## Deferred (per plan)

- `print` → `logging` migration across `music_gen.py`'s 32+ `print` calls is intentionally deferred to Phase 2 per the ROADMAP (the R-S7 plan explicitly limits Phase 1 scope to `musicality_score.py` handlers).
- `beat_annotations` removal from `generate_song_parts` signature — deferred to Phase 4 which rebuilds beat annotation properly.
- `song_name[:20]` truncation bug at `generate_song` — Phase 5 R-P1 fixes via index-based naming.
- `uuid` runtime import removal — follows from R-P1 naming work in Phase 5.

## Requirements Closed

- **R-S2** ✓ — Explicit music21 imports in `music_gen.py`.
- **R-S7** ✓ — `musicality_score.py` portion: all five broad handlers narrowed and using `logger.exception`. `music_gen.py` print-to-logging deferred to Phase 2 per scope.
- **R-S8** ✓ — Dead imports (`glob`, `Pool`, `cpu_count`), dead variables (`ha`, `ba`, `me`, `be`, `now`), and the `uuid` PyPI stub all removed.

## Commits

| Task | Commit | Description |
|------|--------|-------------|
| 1 | `08dbb55` | refactor(01-03): explicit music21 imports and dead-code removal |
| 2 | `350bf60` | refactor(01-03): narrow exception handlers in musicality_score |
| 3 | `72d0329` | chore(01-03): remove stdlib uuid stub from requirements.txt |

## Self-Check: PASSED

Files modified exist:
- FOUND: music_gen.py
- FOUND: musicality_score.py
- FOUND: requirements.txt

Commits exist:
- FOUND: 08dbb55
- FOUND: 350bf60
- FOUND: 72d0329
