# Pitfalls Research

**Scope:** Critical risks in synthetic music dataset generation, with specific manifestations in this codebase.

**Confidence:** HIGH for code-verified bugs (P-A, P-B confirmed via direct Grep); MEDIUM for FluidSynth cross-platform determinism (empirical).

## Confirmed bugs in current code (verified by grep)

### P-A — `mix_and_save` re-rolls arrangement randomness after MIDI is generated — CRITICAL

At `music_gen.py:760` inside `mix_and_save`, `generate_song_arrangement()` is called fresh. The MIDI files were generated for parts enumerated earlier in the pipeline, but the mixing loop iterates this new arrangement. Even if the set of unique parts matches, any random draws consumed inside `generate_song_arrangement` shift downstream soundfont/FX RNG — meaning the dataset has a hidden coupling between arrangement generation and render choices that breaks reproducibility by construction. At worst, the mixed audio describes a different structure than the MIDI in the same output directory.

**Fix:** pass arrangement as a parameter; never re-call inside the renderer/mixer.
**Phase:** Stabilize.

### P-B — `pydub` volume/pan is broken → `levels.json` has zero effect today — CRITICAL

`music_gen.py:845-852`:

```python
beat.volume = float(levels[part]['beat']['volume'])   # .volume is read-only — assignment is a no-op
beat.pan(float(levels[part]['beat']['panning']))      # .pan returns a new AudioSegment; return discarded
```

All four layers. Mix levels and panning from `levels.json` are currently **not applied** to any output. Fixing this will change the audio for every existing seed.

**Fix:** use `AudioSegment.apply_gain(db)` and `segment = segment.pan(x)` — capture the return.
**Phase:** Stabilize.

**Downstream implication:** source-separation ground truth (sum-of-stems = mix) cannot be validated until these bugs are fixed *and* stem persistence is added in the same phase.

## Critical pitfalls

### P-1 — FluidSynth is not bit-reproducible across versions/platforms

FP arithmetic, DSP coefficients, and buffer sizes differ between FluidSynth releases. "Same seed = same WAV" is false unless the binary is pinned.

- **Mitigation:** pin FluidSynth binary version (Dockerfile or install script); SHA-256 regression test in CI against a canonical render; contract becomes "MIDI + metadata deterministic; audio deterministic only under pinned binary."
- **Warning signs:** WAV SHA-256 differs between machines after OS update; CI passes locally, fails in Docker.
- **Phase:** Productize. Open question already flagged in PROJECT.md Key Decisions.

### P-2 — Stems not persisted; sum-of-stems ≠ mix

Per-layer WAVs are intermediate render artifacts deleted before the function returns (`music_gen.py:898`). No source-separation ground truth can be produced until stems are persisted *and* P-B is fixed so the pydub gain/pan actually apply identically to stems and mix.

- **Mitigation:** persist post-FX stems; add post-generation assertion `max(|sum(stems) − mix|) < epsilon`; write silent stems for absent layers so the identity holds for every sample.
- **Phase:** Stabilize (bug fix) + Productize (persistence + assertion).

### P-3 — Beat annotations drift from rendered audio (swing + FluidSynth pre-roll)

Two independent failure modes:

1. FluidSynth adds ~10–50 ms of startup silence; MIDI timestamps don't match WAV onsets.
2. `beat_anotator.py` generates a theoretical straight-grid and validates it against the swung MIDI. With any swing > 0 it always warns, and the straight-grid timestamps would be wrong as annotations.

- **Mitigation:** derive beat timestamps from MIDI ticks (not theoretical grid); empirically measure FluidSynth pre-roll for the pinned binary; apply the offset to all audio-anchored annotations; log the offset in `sample.json`.
- **Warning sign:** `beat_anotator.py` prints "Misalignment detected" on every swung sample.
- **Phase:** Productize.

### P-4 — RNG leakage in multiprocessing → identical samples

`Pool`/`ProcessPoolExecutor` workers fork with the parent's `random` state. Without re-seeding, all N workers would generate the same key/tempo/arrangement/soundfonts.

- **Mitigation:** derive per-sample seeds via `sha256(global_seed, sample_index)`; workers seed their local `random.Random` instances on entry; global `random` is never touched.
- **Warning sign:** all samples in batch share the same key/arrangement; changing `--workers N` changes the dataset.
- **Phase:** Productize (before wiring multiprocessing).

### P-5 — Narrow soundfont pool → trained models overfit the synth

Small `sf/<layer>/` directories mean every sample uses 1–3 timbres. Models learn timbre identity, not musical content, and fail on real audio.

- **Mitigation:** monitor soundfont usage distribution at generation time; fail if any single soundfont exceeds 20% of samples; record distribution in manifest.
- **Warning sign:** `sf/<layer>/` has < 3 files; all annotations reference 2–3 soundfonts; model accuracy drops sharply on real audio.
- **Phase:** Stabilize (detect), Extend (fix).

## Important pitfalls

### P-6 — Bare `except:` blocks swallow failures silently

`musicality_score.py:66, 94, 173, 205, 239` catch everything and return zeros. A corrupt sample silently enters the dataset with `musicality_score: 0` and no failure flag.

- **Mitigation:** narrow exceptions + `logger.exception`; add `analysis_failed: true` field to `sample.json` when scoring raises; post-generation validation (file exists, non-zero size, loadable).
- **Phase:** Stabilize.

### P-7 — UUID truncation → collision under parallelism

`music_gen.py:1142-1143`:

```python
song_name = f"{datetime.now().strftime('%Y%m%d%H%M%S%f')}_{uuid.uuid4()}"
song_name = song_name[:20]  # 20 chars — UUID is entirely dropped
```

Two workers starting in the same microsecond produce the same directory name and overwrite each other.

- **Mitigation:** move to index-based directory naming (`00042/`) per ARCHITECTURE.md; remove the truncation; assert directory doesn't exist before creating it.
- **Phase:** Stabilize (fix) + Productize (index-based layout).

### P-8 — MIDI-to-WAV onset offset from FluidSynth buffer pre-roll

FluidSynth renders in buffer chunks; the first buffer may precede the first MIDI event by a few ms. Transcription labels end up systematically offset from audio onsets.

- **Mitigation:** measure pre-roll empirically for the pinned version; store as dataset-level metadata; apply correction to all MIDI-aligned annotations.
- **Phase:** Productize (couples to P-1 and P-3).

### P-9 — Soundfont licensing blocks public dataset distribution

"Free to download" ≠ "free to redistribute in a dataset." Many free soundfont packs have restrictive or unclear licenses.

- **Mitigation:** audit each `.sf2` file; maintain `sf/LICENSE_AUDIT.md`; for public releases use only CC0/MIT fonts (MuseScore General HQ, GeneralUser GS).
- **Scope:** blocks audio redistribution, not generation. MIDI + annotations are always publishable.
- **Phase:** deferred until the project considers a public release.

## Nice-to-know

### P-10 — Flat directory with 10k subdirectories stresses some filesystems

- **Mitigation:** two-level sharding `dataset/<hex-prefix>/<sample-id>/`; pre-flight disk space check in CLI.
- **Phase:** Productize (critical only at 100k+).

### P-11 — Intermediate files not cleaned up on failure

- **Mitigation:** `try/finally` around per-sample generation; `manifest.jsonl` status field; `musicgen clean --failed` subcommand.
- **Phase:** Productize (part of resumable runs).

## Phase mapping

| Phase | Must address |
|---|---|
| **Stabilize** | P-A, P-B, P-6, P-7 (fix), P-5 (detect) |
| **Productize** | P-1, P-2, P-3, P-4, P-7 (layout), P-8, P-10, P-11 |
| **Extend** | P-5 (broaden soundfont pool) |
| **Research / public release** | P-9 (license audit + soundfont replacement) |

## Open questions

1. FluidSynth version currently installed — determine before pinning.
2. Is FluidSynth pre-roll offset consistent across renders on the same binary? (Empirical.)
3. Does `music21` or `pedalboard` touch global `random` state? (Audit.)
4. For absent layers, is an explicit silent stem file required, or can the sum assertion account for omitted tracks? (Decide during Productize.)
