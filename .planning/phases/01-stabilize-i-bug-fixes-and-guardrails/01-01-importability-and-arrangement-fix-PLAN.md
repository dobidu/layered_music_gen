---
phase: 01-stabilize-i-bug-fixes-and-guardrails
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - music_gen.py
autonomous: true
requirements: [R-S1, R-S3]
must_haves:
  truths:
    - "`python -c 'import music_gen'` exits 0 and prints nothing from generation"
    - "`generate_song_arrangement` is called exactly once per song, before MIDI generation"
    - "`mix_and_save` receives arrangement as parameters; never re-rolls it"
  artifacts:
    - path: "music_gen.py"
      provides: "Importable module + arrangement-once contract"
      contains: "if __name__ == \"__main__\":"
  key_links:
    - from: "create_song"
      to: "mix_and_save"
      via: "song_unique_parts, song_arrangement passed as positional arguments"
      pattern: "mix_and_save\\([^)]*song_arrangement"
---

<objective>
Make `music_gen.py` importable without side effects, and fix the arrangement re-roll bug (PITFALLS P-A) by computing arrangement once in `create_song` and threading it into `mix_and_save`.

Purpose: Establishes the determinism baseline that every downstream phase depends on. Without these fixes the module cannot be imported for tests, and the audio mix can describe a different song structure than the MIDI files in the same output directory.

Output: A `music_gen.py` that can be imported safely and where MIDI + audio always describe the same arrangement.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/REQUIREMENTS.md
@.planning/research/PITFALLS.md
@music_gen.py

<interfaces>
Current signatures (from music_gen.py):

```python
# Line 615
def generate_song_arrangement(structures_file: str = 'song_structures.json') -> Tuple[List[str], List[str]]:
    # returns (song_unique_parts, song_arrangement)

# Line 758 — current (BUGGY)
def mix_and_save(harm_filename, bass_filename, melo_filename, beat_filename, name):
    song_unique_parts, song_arrangement = generate_song_arrangement()  # ← BUG: re-rolled here
    ...

# Line 1004
def create_song(key, tempo, song_signatures, measures, name, chord_pat_file, swing_amount) -> Dict:
    # currently calls generate_song_parts then mix_and_save without arrangement

# Line 1044 — current call site
wav_name, arrangement, transitions, soundfonts, pedalboards, part_layers = mix_and_save(
    ha, ba, me, be, song_name
)

# Line 1158-1161 — current top-level execution (must be guarded)
for i in range(1):
    generate_song(i)
```

`mix_and_save` already returns `arrangement` as the second return value — keep that contract; the value just comes from the parameter now instead of from a local re-roll.
</interfaces>
</context>

<tasks>

<task type="auto" tdd="false">
  <name>Task 1: Wrap top-of-file execution in __main__ guard</name>
  <files>music_gen.py</files>
  <read_first>
    - music_gen.py (lines 1158-1161 specifically; also lines 1-20 imports)
    - .planning/codebase/TESTING.md
  </read_first>
  <action>
At the very bottom of `music_gen.py` (currently lines 1158-1161), replace:

```python
# Example usage

for i in range(1):
    generate_song(i)
```

with:

```python
# Example usage

if __name__ == "__main__":
    for i in range(1):
        generate_song(i)
```

Do NOT change anything else in this task. Do NOT touch imports. Do NOT touch the loop body. The only change is wrapping the loop in `if __name__ == "__main__":` and indenting the existing two lines by four spaces.
  </action>
  <verify>
    <automated>cd /home/openclaw/musicgen &amp;&amp; python -c "import music_gen" 2>&amp;1 | tee /tmp/import_check.log; grep -q "Generating song" /tmp/import_check.log &amp;&amp; echo "FAIL: generation triggered on import" &amp;&amp; exit 1 || echo "OK: import is side-effect-free"</automated>
  </verify>
  <acceptance_criteria>
    - `grep -n 'if __name__ == "__main__":' music_gen.py` matches a line at or after line 1158
    - `python -c "import music_gen"` exits 0
    - `python -c "import music_gen"` produces no `Generating song` output (i.e. `generate_song` is NOT invoked at import time)
    - The `for i in range(1): generate_song(i)` loop body is still present, just nested under the guard
  </acceptance_criteria>
  <done>music_gen.py is importable without triggering generation. The exit-criterion-1 check from the ROADMAP passes.</done>
</task>

<task type="auto" tdd="false">
  <name>Task 2: Fix arrangement re-roll (P-A) — compute once in create_song, pass to mix_and_save</name>
  <files>music_gen.py</files>
  <read_first>
    - music_gen.py lines 615-660 (generate_song_arrangement signature/body)
    - music_gen.py lines 758-770 (mix_and_save head — the buggy re-call)
    - music_gen.py lines 1004-1080 (create_song body)
    - music_gen.py lines 1117-1156 (generate_song body)
    - .planning/research/PITFALLS.md (P-A section)
  </read_first>
  <action>
Three coordinated edits in `music_gen.py`. Make them all in one task.

**Edit A — `mix_and_save` signature + remove the re-roll (around line 758):**

Replace:

```python
def mix_and_save(harm_filename, bass_filename, melo_filename, beat_filename, name):
    # TODO: only render and mix the parts that are used in the song arrangement
    song_unique_parts, song_arrangement = generate_song_arrangement()
    print("Song arrangement: "+ str(song_arrangement) + "\n")
```

with:

```python
def mix_and_save(harm_filename, bass_filename, melo_filename, beat_filename, name,
                 song_unique_parts, song_arrangement):
    # Arrangement is now produced once upstream (see create_song) and threaded through.
    # See PITFALLS P-A / R-S3: do NOT call generate_song_arrangement() here; doing so
    # re-rolls RNG and can decouple the rendered audio from the MIDI structure.
    # TODO (later phase): only render and mix the parts that are used in the song arrangement
    print("Song arrangement: " + str(song_arrangement) + "\n")
```

Leave the rest of the function body unchanged. The local variables `song_unique_parts` and `song_arrangement` keep the same names so the rest of the function works as-is.

**Edit B — `create_song` produces the arrangement and forwards it (around line 1044):**

In `create_song`, immediately BEFORE the `# Generates the musical components` block (around line 1031), add:

```python
    # Compute arrangement ONCE for the whole song (R-S3 / PITFALLS P-A).
    # Must happen before generate_song_parts so that all downstream RNG draws
    # (soundfont selection, FX, layer probabilities) sit deterministically after it.
    song_unique_parts, song_arrangement = generate_song_arrangement()
```

Then replace the existing `mix_and_save(...)` call (currently around line 1044):

```python
    wav_name, arrangement, transitions, soundfonts, pedalboards, part_layers = mix_and_save(
        ha, ba, me, be, song_name
    )
```

with:

```python
    wav_name, arrangement, transitions, soundfonts, pedalboards, part_layers = mix_and_save(
        ha, ba, me, be, song_name,
        song_unique_parts, song_arrangement,
    )
```

`mix_and_save` already returns `song_arrangement` as the second return value — that contract is unchanged; the value just now comes from the parameter instead of from a local re-roll. The `arrangement` variable in `create_song` will therefore equal the `song_arrangement` we just passed in. That is intentional.

**Edit C — verify there are no other call sites:**

After making edits A and B, run `grep -n "mix_and_save(" music_gen.py` and confirm there is exactly ONE call site (in `create_song`) plus the `def`. If a stray second call exists, update it to pass the new arguments.

Then run `grep -n "generate_song_arrangement" music_gen.py` and confirm exactly TWO matches: the `def` at line ~615, and the new call inside `create_song`. The line inside `mix_and_save` MUST be gone.
  </action>
  <verify>
    <automated>cd /home/openclaw/musicgen &amp;&amp; python -c "import music_gen" &amp;&amp; python -c "
import music_gen, inspect
sig = inspect.signature(music_gen.mix_and_save)
params = list(sig.parameters.keys())
assert 'song_unique_parts' in params, f'song_unique_parts missing from mix_and_save: {params}'
assert 'song_arrangement' in params, f'song_arrangement missing from mix_and_save: {params}'
src = inspect.getsource(music_gen.mix_and_save)
assert 'generate_song_arrangement(' not in src, 'mix_and_save still calls generate_song_arrangement!'
src2 = inspect.getsource(music_gen.create_song)
assert 'generate_song_arrangement(' in src2, 'create_song does not call generate_song_arrangement!'
print('OK: arrangement is computed in create_song and passed into mix_and_save')
"</automated>
  </verify>
  <acceptance_criteria>
    - `grep -c "generate_song_arrangement" music_gen.py` returns exactly `2` (one `def`, one call in `create_song`)
    - `grep -n "generate_song_arrangement" music_gen.py` does NOT show any line in the 758-770 range
    - `grep -n "def mix_and_save" music_gen.py` shows the new signature including `song_unique_parts, song_arrangement`
    - `grep -n "mix_and_save(" music_gen.py` shows exactly one call site (in `create_song`) and it passes 7 positional arguments
    - `python -c "import music_gen; import inspect; assert 'song_arrangement' in inspect.signature(music_gen.mix_and_save).parameters"` exits 0
    - `python -c "import music_gen"` exits 0 with no generation output
  </acceptance_criteria>
  <done>The arrangement is generated exactly once per song, in `create_song`, and passed into `mix_and_save` as parameters. PITFALLS P-A is closed and R-S3 is satisfied.</done>
</task>

</tasks>

<verification>
After both tasks:
1. `python -c "import music_gen"` exits 0 with no stdout from generation logic.
2. `grep -n 'generate_song_arrangement' music_gen.py | wc -l` returns `2`.
3. `grep -n 'if __name__ == "__main__":' music_gen.py` matches at the bottom of the file.
4. The module is in a state where Plan 02 (gain/pan fix in `mix_and_save`) and Plan 04 (pytest skeleton) can proceed.
</verification>

<success_criteria>
- music_gen.py imports cleanly without side effects (R-S1 ✓)
- Arrangement re-roll bug fixed; arrangement flows from `create_song` → `mix_and_save` (R-S3 ✓, PITFALLS P-A closed)
- ROADMAP exit criterion #1 satisfied
</success_criteria>

<output>
After completion, create `.planning/phases/01-stabilize-i-bug-fixes-and-guardrails/01-01-SUMMARY.md` documenting:
- Exact diff applied for the guard
- Exact diff applied for the arrangement threading
- Confirmation that import is side-effect-free
- Confirmation that grep shows exactly 2 occurrences of `generate_song_arrangement`
- Note that the RNG sequence changed (arrangement is now drawn earlier than before) — this is expected and is the prerequisite for any future seed-stability contract
</output>
