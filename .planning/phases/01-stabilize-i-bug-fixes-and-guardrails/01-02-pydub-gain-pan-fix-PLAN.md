---
phase: 01-stabilize-i-bug-fixes-and-guardrails
plan: 02
type: execute
wave: 2
depends_on: [01]
files_modified:
  - music_gen.py
autonomous: true
requirements: [R-S4]
must_haves:
  truths:
    - "`levels.json` volume values affect the final mix audio"
    - "`levels.json` panning values affect the final mix audio"
    - "No `.volume = ` assignment exists on AudioSegment objects in mix_and_save"
    - "Every `.pan(...)` call has its return value captured back into the segment variable"
  artifacts:
    - path: "music_gen.py"
      provides: "Working gain + pan application in mix_and_save"
      contains: "apply_gain"
  key_links:
    - from: "levels.json values"
      to: "AudioSegment objects (beat, melody, harmony, bassline)"
      via: "apply_gain(...) and segment.pan(...) capture-return pattern"
      pattern: "= (beat|melody|harmony|bassline)\\.(apply_gain|pan)\\("
---

<objective>
Fix the pydub gain/pan no-op bug (PITFALLS P-B / R-S4) at `music_gen.py:845-852`. Currently `levels.json` has zero effect on any output because the code assigns to a read-only `.volume` property and discards the return of `.pan()`. After this fix, `levels.json` values are actually applied.

Purpose: Without this fix, every Phase 1 output is missing two of its specified label dimensions (gain + pan), and the source-separation contract (sum-of-stems = mix) will be impossible to verify in Phase 5 because the per-stem gain/pan would not match.

Output: A `mix_and_save` whose audio output reflects `levels.json`.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/REQUIREMENTS.md
@.planning/research/PITFALLS.md
@music_gen.py

<interfaces>
Current buggy code (music_gen.py:844-852):

```python
        # Volume and panning for each layer
        beat.volume = float(levels[part]['beat']['volume'])
        melody.volume = float(levels[part]['melody']['volume'])
        harmony.volume = float(levels[part]['harmony']['volume'])
        bassline.volume = float(levels[part]['bassline']['volume'])
        beat.pan(float(levels[part]['beat']['panning']))
        melody.pan(float(levels[part]['melody']['panning']))
        harmony.pan(float(levels[part]['harmony']['panning']))
        bassline.pan(float(levels[part]['bassline']['panning']))
```

Bugs:
1. `AudioSegment.volume` is a read-only property — assignment is silently ignored.
2. `AudioSegment.pan()` returns a NEW segment; the return is discarded so the original `beat`/`melody`/etc. variables are unchanged.

pydub APIs to use:
- `AudioSegment.apply_gain(gain_db: float) -> AudioSegment` — returns a new segment with gain applied (in dB).
- `AudioSegment.pan(pan: float) -> AudioSegment` — returns a new segment with pan applied (-1.0 = full left, 0 = center, 1.0 = full right).

Important interpretation note for `levels.json`: the existing JSON `volume` values are scalar values (the original code treated them as a multiplier-like attribute). pydub's `apply_gain` expects DECIBELS. Inspect `levels.json` first to see the actual range of values used. Two cases:
- If values look like dB (e.g. -6.0, -3.0, 0.0): pass them directly to `apply_gain`.
- If values look like linear amplitude (e.g. 0.5, 1.0, 0.8): convert with `20 * math.log10(value)` (already imported as `math`).

Decide based on what the file actually contains; record the choice in the SUMMARY.
</interfaces>
</context>

<tasks>

<task type="auto" tdd="false">
  <name>Task 1: Inspect levels.json to determine gain unit (dB vs linear)</name>
  <files>levels.json (read-only)</files>
  <read_first>
    - levels.json (full file)
    - .planning/research/PITFALLS.md (P-B section)
  </read_first>
  <action>
Read `levels.json` in full. Determine the value range of the `volume` fields:

- If values are in `[0.0, 1.0]` or visibly linear (e.g. `0.5`, `0.8`, `1.0`) → unit is LINEAR amplitude → conversion needed.
- If values are in roughly `[-30.0, 6.0]` or include negative numbers like `-6.0`, `-3.0` → unit is DECIBELS → use directly.

Document the finding in a one-line comment you'll embed in Task 2's edit. Do NOT modify any file in this task.
  </action>
  <verify>
    <automated>cd /home/openclaw/musicgen &amp;&amp; python -c "
import json
with open('levels.json') as f:
    levels = json.load(f)
# Collect all volume values
vols = []
for part, layers in levels.items():
    for layer, params in layers.items():
        if 'volume' in params:
            vols.append(float(params['volume']))
print('Volume range:', min(vols), 'to', max(vols))
print('Sample values:', vols[:8])
print('Likely unit:', 'dB' if min(vols) &lt; -1 or max(vols) &lt; 0 else 'linear (needs 20*log10 conversion)')
"</automated>
  </verify>
  <acceptance_criteria>
    - Command above prints the volume range and a "Likely unit:" verdict
    - Verdict is one of `dB` or `linear (needs 20*log10 conversion)`
    - Decision recorded in working memory for Task 2
  </acceptance_criteria>
  <done>The unit of `volume` values in `levels.json` is identified (dB or linear). This determines the exact code Task 2 writes.</done>
</task>

<task type="auto" tdd="false">
  <name>Task 2: Replace .volume assignments + capture .pan() returns in mix_and_save</name>
  <files>music_gen.py</files>
  <read_first>
    - music_gen.py lines 838-855 (the buggy block)
    - levels.json (re-read to confirm Task 1 finding)
    - .planning/research/PITFALLS.md (P-B section)
  </read_first>
  <action>
At `music_gen.py:844-852` (the block currently containing `.volume =` assignments and discarded `.pan(...)` calls), replace the entire block with one of the two versions below depending on Task 1's finding.

**Version A — if `levels.json` values are in DECIBELS:**

```python
        # Volume and panning for each layer (R-S4 / PITFALLS P-B fix).
        # `.volume =` was a no-op (read-only property); `.pan()` returns a new
        # segment so its return must be captured. Values in levels.json are dB.
        beat = beat.apply_gain(float(levels[part]['beat']['volume']))
        melody = melody.apply_gain(float(levels[part]['melody']['volume']))
        harmony = harmony.apply_gain(float(levels[part]['harmony']['volume']))
        bassline = bassline.apply_gain(float(levels[part]['bassline']['volume']))
        beat = beat.pan(float(levels[part]['beat']['panning']))
        melody = melody.pan(float(levels[part]['melody']['panning']))
        harmony = harmony.pan(float(levels[part]['harmony']['panning']))
        bassline = bassline.pan(float(levels[part]['bassline']['panning']))
```

**Version B — if `levels.json` values are LINEAR amplitudes:**

```python
        # Volume and panning for each layer (R-S4 / PITFALLS P-B fix).
        # `.volume =` was a no-op (read-only property); `.pan()` returns a new
        # segment so its return must be captured. Values in levels.json are
        # linear amplitudes; convert to dB via 20*log10(v) (clamped to avoid log(0)).
        def _lin_to_db(v: float) -> float:
            return 20.0 * math.log10(max(float(v), 1e-6))
        beat = beat.apply_gain(_lin_to_db(levels[part]['beat']['volume']))
        melody = melody.apply_gain(_lin_to_db(levels[part]['melody']['volume']))
        harmony = harmony.apply_gain(_lin_to_db(levels[part]['harmony']['volume']))
        bassline = bassline.apply_gain(_lin_to_db(levels[part]['bassline']['volume']))
        beat = beat.pan(float(levels[part]['beat']['panning']))
        melody = melody.pan(float(levels[part]['melody']['panning']))
        harmony = harmony.pan(float(levels[part]['harmony']['panning']))
        bassline = bassline.pan(float(levels[part]['bassline']['panning']))
```

`math` is already imported at the top of `music_gen.py` (line 19), so no new import is needed for Version B.

Do NOT change anything else in `mix_and_save`. Specifically, leave the `mix.overlay(...)` calls below this block alone.
  </action>
  <verify>
    <automated>cd /home/openclaw/musicgen &amp;&amp; python -c "
import re
src = open('music_gen.py').read()
# No more bare .volume = assignments
bad = re.findall(r'\\b(beat|melody|harmony|bassline)\\.volume\\s*=', src)
assert not bad, f'Still has .volume = assignments: {bad}'
# All .pan(...) calls in mix_and_save must be capture-return
# Find the mix_and_save function body and check
import ast
tree = ast.parse(src)
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name == 'mix_and_save':
        body_src = ast.get_source_segment(src, node)
        # Find naked '.pan(' calls (not assigned)
        # Heuristic: every .pan( occurrence in this function should be preceded by '= ' on the same line
        for line in body_src.splitlines():
            if '.pan(' in line and 'def ' not in line:
                assert '=' in line.split('.pan(')[0], f'Naked .pan() call: {line.strip()}'
        # apply_gain should appear at least 4 times (one per layer)
        assert body_src.count('apply_gain') &gt;= 4, f'Expected &gt;=4 apply_gain calls, found {body_src.count(\"apply_gain\")}'
        print('OK: mix_and_save has 4+ apply_gain calls and no naked .pan() calls')
        break
else:
    raise AssertionError('mix_and_save not found')
" &amp;&amp; python -c "import music_gen; print('import OK')"</automated>
  </verify>
  <acceptance_criteria>
    - `grep -nE "(beat|melody|harmony|bassline)\.volume\s*=" music_gen.py` returns nothing (no matches)
    - `grep -c "apply_gain" music_gen.py` returns at least `4`
    - Every `.pan(` call inside `mix_and_save` is on the right-hand side of `=` (capture-return pattern)
    - `python -c "import music_gen"` exits 0 with no generation output
    - The block contains the comment referencing R-S4 / PITFALLS P-B
  </acceptance_criteria>
  <done>`levels.json` now affects the rendered audio. R-S4 is satisfied and PITFALLS P-B is closed. Phase 1 ROADMAP exit criterion #2 ("seeded run produces mix audio that reflects levels.json") is unblocked.</done>
</task>

</tasks>

<verification>
After this plan:
1. `grep -nE "(beat|melody|harmony|bassline)\.volume\s*=" music_gen.py` returns nothing.
2. `grep -c apply_gain music_gen.py` is ≥ 4.
3. `python -c "import music_gen"` still exits 0 (regression check vs Plan 01).
4. Manual smoke (out-of-band, optional): running `python music_gen.py` still produces a `.wav`. The audio will sound different from any pre-fix output — this is expected.
</verification>

<success_criteria>
- `.volume =` assignments removed (R-S4 ✓)
- `.pan()` returns captured (R-S4 ✓)
- Module still imports cleanly (no regression on Plan 01)
- PITFALLS P-B closed
</success_criteria>

<output>
Create `.planning/phases/01-stabilize-i-bug-fixes-and-guardrails/01-02-SUMMARY.md` containing:
- Which version (A or B) was used and the actual `levels.json` value range observed
- Exact diff of the replaced block
- Confirmation that `grep -nE "\\.volume\\s*=" music_gen.py` is empty
- Note: every existing seed will now produce different audio. Phase 5 will pin a "post-fix golden" for the determinism regression test.
</output>
