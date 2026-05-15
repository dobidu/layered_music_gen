# musicgen Musicality Scoring — Audit & Recommendations

**Scope:** review of the current `MusicalityAnalyzer`, identification of failure modes for the v0.3 quality gate, and a concrete two-layer architecture (MIDI pre-filter + audio post-filter) with calibration methodology.

**Validation note:** citations below are limited to claims I am confident are correctly attributed. Where I previously cited specific journal/year combinations I was not certain about, I have either dropped the citation or attributed the work to the author/group without a precise venue. Engineering recommendations are flagged as such where they are judgment calls rather than literature claims.

---

## 1. What musicality dimensions matter for a synthetic-data quality gate

The framing matters. The goal is not to identify "great" music — the goal is to reject samples that would poison a training distribution. That changes what a good gate looks like: **high recall on pathologies, high precision on the obviously-fine middle**, and don't waste budget separating "good" from "very good."

For this generator specifically (Markov chains over Roman numerals + scale degrees, genre-constrained tempo/swing, FluidSynth render), the failure modes that actually occur are predictable:

- **Tonal incoherence** — Markov-chain melodies drifting outside the implied key, or chord-tone/melody mismatch. This is the dominant failure mode of low-order Markov generators.
- **Rhythmic incoherence** — onsets not aligned to the metric grid; rare in a deterministic seq generator unless the renderer or beat-track layer misfires, but worth a cheap check.
- **Degenerate repetition** — Markov chains have a known failure mode of entering short cycles (e.g. I–V–I–V for 60 seconds, or scale degree 1–2–1–2 indefinitely). Symbolically obvious, audio-invisible.
- **Degenerate non-repetition** — the opposite: every bar different, no motivic identity. The lower entropy bound matters as much as the upper.
- **Pitch range pathologies** — unplayable octaves, melody crossing bass, single pitch held for the entire sample.
- **Density pathologies** — empty bars, or 32nd-note machine-gun runs that no human-trained model should learn from.
- **Render-stage failures** — clipping, FluidSynth voice exhaustion (dropped notes), DC offset, near-silence from soundfont mismatch.

The current scorer barely catches any of these. It measures *audio-surface* properties (chroma stability, onset CV, ZCR) that mostly correlate with "is this a music signal at all." For your use case, **the symbolic layer should do most of the work**, because it directly inspects the structures the generator produces, while the audio layer should mainly catch render failures and confirm the symbolic verdict.

This reframes the gate: Layer 1 (MIDI) is the actual quality filter; Layer 2 (audio) is a render-integrity check plus a cheap sanity confirmation.

---

## 2. Symbolic (MIDI) metrics — state of the art

The relevant tooling clusters around a few well-known projects: **jSymbolic2** (Cory McKay, ~1500 symbolic features, Java); **MusPy** (Dong et al., ISMIR 2020, Python-native); **MGEval** (Yang & Lerch), which proposes the now-standard *intra-set distance distribution* methodology for evaluating generative music models; and **mir_eval** for reference-comparison metrics. For tonal tension, the foundational work is Lerdahl's *Tonal Pitch Space* (Oxford, 2001) and Chew's spiral array; Herremans & Chew's tension ribbon is the most usable computational version.

Concrete metrics, in roughly the order I'd weight them for this gate:

### 2.1 Pitch-class entropy and key-profile divergence
Compute the pitch-class histogram (weighted by duration). For a sample in a known key, compute KL divergence (or the simpler max-correlation) against the Krumhansl–Schmuckler key profiles (Krumhansl & Kessler 1982; Krumhansl 1990, *Cognitive Foundations of Musical Pitch*, Oxford). Low divergence ⇒ tonal. High divergence ⇒ atonal-by-accident. **music21 implements this directly via `analysis.discrete.KrumhanslSchmuckler`.** This is the single best symbolic check for tonal coherence and it costs microseconds.

### 2.2 Melodic interval distribution
Histogram of signed semitone intervals between consecutive melody notes. Two checks:
- *Fraction of intervals ≤ 12 semitones* — should be near 1.0; large leaps are rare in real melodies (Huron, *Sweet Anticipation*, MIT Press 2006, Ch. 5).
- *Fraction of stepwise motion (|interval| ≤ 2)* — typically 0.5–0.75 in tonal melodies. This is jSymbolic's "Melodic Interval Histogram" family and is trivial on a music21 stream.

### 2.3 Pitch range and tessitura
Max minus min MIDI number per layer, plus the interquartile range. **Hard rejects:** range > 36 semitones (3 octaves) for a 30-second melody is almost always a bug; range = 0 is a stuck note. Soft penalty outside genre-typical ranges.

### 2.4 Onset grid alignment
For each note onset, compute distance to nearest 16th-note grid point in beats; take the mean. For a deterministic generator this should be ~0; nonzero indicates a tick-rounding bug. Free correctness check.

### 2.5 Note density
Notes per second, per layer. Genre-bounded (engineering judgment — adjust per genre): bebop melody ~6–10 n/s, ballad ~1–2 n/s, EDM bassline ~2–4 n/s. This is genre-aware where the current tempo clip is not.

### 2.6 Repetition structure (three complementary measures)

**(a) N-gram entropy** on melodic intervals or scale-degree sequences. Compute Shannon entropy over the empirical bigram/trigram distribution. Real music sits in a band: too low ⇒ degenerate repetition; too high ⇒ random walk. The IDyOM line of work (Marcus Pearce's PhD thesis and subsequent papers with Geraint Wiggins) makes this a principled measure of melodic predictability with strong correlations to listener surprise judgments.

**(b) LZ compression ratio** on the symbol sequence. The Manaris group has shown that compression ratio correlates with perceived musicality on large corpora. For a quality gate, the useful signal is the tails — ratios below ~0.3 (extreme repetition) or above ~0.85 (incompressible noise) flag pathologies. Use stdlib `zlib.compress` on the byte-encoded sequence.

**(c) Bar-level pitch-class self-similarity matrix.** For each bar, compute a 12-d pitch-class vector; compute pairwise cosine similarity. The mean off-diagonal value tells you whether bars resemble each other at all. Foote's work on self-similarity matrices (1999–2000) is the original reference; for short clips this is more reliable than block-level structure detection.

### 2.7 Harmonic tension (lightweight Lerdahl-style)
The full tonal pitch space is overkill for a gate. The lightweight proxy is *chord-distance in pitch space* between consecutive chords: for each pair of adjacent Roman-numeral chords, compute the minimum voice-leading distance (sum of absolute semitone moves under optimal voice assignment). Penalize sequences where every transition exceeds ~5 semitones — that's "harmonic noise." Since you're already storing Roman numerals, this is a dictionary lookup.

### 2.8 Scale adherence / chromaticism ratio
Fraction of melody notes whose pitch class is in the diatonic scale of the active key. In tonal genres should be > 0.85; in jazz with chromaticism, > 0.70. Below 0.5 indicates the Markov chain has wandered off-key. `music21.key.Key.getScale().pitches` gives this directly.

### 2.9 Melody/bass crossing
Fraction of timesteps where the melody pitch is below the bass pitch. Should be near 0; nonzero is a layering bug.

### 2.10 What I would not bother with
- Full tonal-pitch-space attraction model — too expensive for a gate.
- Contour-class histograms — too coarse for short clips.
- Full IDyOM information-content estimate — principled but requires a trained corpus model. The simpler n-gram entropy above captures most of the discriminating signal for a gate (estimated; not formally measured here).

### 2.11 On Zipf's law
The Manaris group has reported Zipf-like distributions across pitch, duration, and interval frequencies in tonal corpora, and proposed Zipf-slope features. **Honest read of the literature:** it's a real statistical regularity but a *weak filter* on its own — many bad generators also produce Zipf-ish distributions because Zipf's law is a generic property of mixture distributions. Include the slope as one feature among many, not as a primary gate.

---

## 3. Audio-side improvements

The audio layer should be smaller and more targeted than what you have.

### 3.1 Replace the tonnetz corrcoef "consonance" measure entirely
It measures frame-to-frame stationarity, not consonance. Two principled replacements:

- **Chroma–key correlation:** compute mean chroma over the clip, correlate against the 24 Krumhansl–Schmuckler major/minor key profiles, take the max. This is essentially librosa's recommended key-detection idiom and gives a real "tonal clarity" score in [0, 1]. Available via `librosa.feature.chroma_cqt` plus a 12-element dot product.
- **Roughness/dissonance via spectral peaks:** Sethares' sensory dissonance model (1993, JASA) gives a computable dissonance curve from pairs of spectral peaks. Essentia has a reference `Dissonance` algorithm. If you don't want to add Essentia, a serviceable proxy is *spectral inharmonicity* via librosa's `piptrack` plus a pairwise roughness sum (~30 lines of code).

### 3.2 Genre-aware tempo
Replace the hard 60–180 clip with the per-genre bounds from `GenreSpec`. The score becomes: estimate tempo via `librosa.beat.beat_track`, score 1.0 if inside `GenreSpec.tempo_range`, linearly decay outside. Also: compute the tempogram (`librosa.feature.tempogram`) and check that the dominant tempo periodicity has a strong, narrow peak — bimodal tempograms (around half/double) are fine, but flat ones indicate beat-tracking failure.

### 3.3 Render-integrity checks (the highest-value audio-side additions)
These catch failures the symbolic layer cannot see:

- *True peak / clipping:* count samples within 0.01 of ±1.0. Anything > 0.1% is a clipping artifact.
- *DC offset:* `abs(mean(y))`; should be < 0.001.
- *Silence ratio:* fraction of frames below −60 dBFS RMS. > 30% indicates render failure or extreme dynamics.
- *EBU R128 integrated loudness:* use `pyloudnorm` (small dependency, well-maintained, BS.1770 compliant). Target: between −30 and −10 LUFS. Outside that range usually means a soundfont/velocity bug.
- *Crest factor* (peak / RMS): < 6 dB suggests over-compression or DC; > 30 dB suggests sparse transients on near-silence.

### 3.4 MFCC for music-vs-noise
The current ZCR + flatness + contrast triple is fine for gross noise detection on FluidSynth output. A stronger discriminator (MFCC mean/variance into a tiny binary classifier) is unlikely to be worth the added complexity for this signal type. **Engineering judgment: leave alone unless false-negative rate proves to be a problem.**

### 3.5 Drop ZCR from the noise score
It's largely redundant with spectral flatness for synthesized signals and adds noise to the metric.

---

## 4. Perceptual research basis — calibrated honesty

A blunt summary of what the literature actually supports for the kind of decisions you're making:

- **Zipf-style power laws** (Manaris group): real but weak as a quality signal. Useful as a feature, not a gate.
- **IDyOM-style melodic expectation** (Pearce, Wiggins, and collaborators): the strongest validated framework for melodic predictability, with reasonable correlations against listener surprise judgments. The full system requires a trained model. Lightweight n-gram entropy approximations (Section 2.6a) capture much of the discriminating signal — sufficient for a gate, not sufficient for ranking.
- **Tension–release curves** (Lerdahl & Krumhansl on tonal-pitch-space tension; Farbood on quantitative tension models; Herremans & Chew's tension ribbon): validated against listener judgments. For a gate, you don't need the curve shape — you just need to flag samples with no tension variation (flat ribbon = boring) or runaway tension (monotonically increasing = no resolution).
- **Compression / entropy as quality:** validated as a *coarse* filter (Manaris). Useful for the tails, unreliable in the middle.
- **Krumhansl–Schmuckler key profiles:** extensively validated, robust on short excerpts, and the right tool for tonal-coherence detection at this scale.

The honest meta-point: psychoacoustic literature is much better at telling you what *correlates* with perceived musicality than at giving you a single number that *is* it. The gate should be a **bank of independent threshold checks** rather than a single composite score, because a sample can fail in ways that average out in a weighted sum. (A clipped sample with a great melody scores fine on a weighted sum.)

---

## 5. Recommended two-layer architecture

Structure this as a **rejection cascade** rather than a weighted-sum scorer. Each check is a hard or soft gate; soft scores aggregate at the end. This makes failures interpretable and tuning local.

### 5.1 Layer 1 — symbolic (pre-render, target < 5 ms)

Operate on the multi-track `mido.MidiFile`. Extract per-track note lists (pitch, onset_beat, duration_beat, velocity).

| Check | Type | Threshold (starting values) | Purpose |
|---|---|---|---|
| Has notes in every active layer | Hard | ≥ 4 notes/layer over full clip | Catches dropped layers |
| Pitch range per layer | Hard | ≤ 36 st melody, ≤ 24 st bass | Catches octave bugs |
| Stuck-note check | Hard | Max single-pitch run < 80% of layer | Catches degenerate Markov |
| Melody/bass crossing | Hard | < 5% of timesteps | Catches layering bugs |
| Onset grid deviation | Hard | mean dev < 1/64 note | Render-prep sanity |
| Pitch-class KL vs key profile | Soft | score = exp(−KL) | Tonal coherence |
| Scale adherence | Soft | linear in [0.5, 1.0] → [0, 1] | Genre-conditional floor |
| Melodic-interval reasonableness | Soft | linear in step-fraction [0.3, 0.8] | Per Huron |
| Note-density vs genre band | Soft | triangular around genre center | Uses GenreSpec |
| N-gram entropy of scale degrees | Soft | triangular around [2.0, 4.0] bits | Trigram, drops the tails |
| LZ compression ratio | Soft | triangular around [0.4, 0.8] | Drops repetition extremes |
| Bar-level self-similarity mean | Soft | triangular around [0.2, 0.6] | Some repetition, not all |
| Voice-leading distance (chords) | Soft | linear in [8, 2] semitones avg → [0, 1] | Harmonic smoothness |

Aggregate the soft scores with equal weight initially, with optional 2× weight on tonal-coherence and scale-adherence (engineering judgment — these tend to dominate audible quality). If any hard check fails OR aggregated soft score < `τ₁`, re-roll. This whole layer is well under 5 ms on a 60-second MIDI file.

### 5.2 Layer 2 — audio (post-render, target ~50–150 ms)

| Check | Type | Threshold |
|---|---|---|
| Clipping ratio | Hard | < 0.1% samples ≥ 0.99 |
| DC offset | Hard | abs(mean) < 0.001 |
| Silence ratio | Hard | < 30% frames below −60 dBFS |
| Integrated loudness (LUFS) | Hard | −30 ≤ LUFS ≤ −10 |
| Crest factor | Hard | 6 dB ≤ CF ≤ 30 dB |
| Tempo within genre band | Soft | 1.0 inside, linear decay outside |
| Beat strength / pulse clarity | Soft | onset autocorrelation peak height |
| Chroma–key correlation | Soft | max corr against 24 KS profiles |
| Spectral flatness (music-ness) | Soft | linear in [0.05, 0.3], inverted |
| Tempogram peak sharpness | Soft | inverse of half-height width |

Aggregate soft scores equal-weighted. Combined gate decision:

`pass = Layer1_hard ∧ Layer2_hard ∧ Layer1_soft ≥ τ₁ ∧ Layer2_soft ≥ τ₂`

### 5.3 Cost analysis
Layer 1 runs pre-render: three re-rolls cost 3 × MIDI gen + 3 × ~5 ms ≈ MIDI-gen time. Only samples passing Layer 1 hit FluidSynth, so audio cost is incurred at most once per accepted sample (or N times in the rare event Layer 2 hard checks reject — but those are render bugs, so in practice ≈ 1×). This sits comfortably inside the < 5 s budget for 3 re-rolls.

---

## 6. Calibration without human labels

You have a deterministic generator and full control over the input distribution. Use it.

**Reference-set bootstrapping with adversarial examples:**

1. **Generate a "reference good" set** (~500–1000 samples) using your normal generator at default settings across all genres. Compute every soft metric on every sample. You now have an empirical distribution per metric.

2. **Generate a "known bad" set** by deliberately breaking the generator: uniform-random pitch (kills tonal coherence), single-note Markov (kills entropy), zero-variance bars (kills self-similarity), wrong-key forced (kills scale adherence), 4× note density, etc. Maybe 100 samples per pathology.

3. **Per-metric thresholds:** choose each soft-score mapping to minimize expected loss assuming the bad set is what you want to reject. Concretely, set each metric's soft mapping so the reference set's median maps to ≈ 0.8 and the bad set's median maps to ≈ 0.3. This gives you a principled separation without picking magic numbers — they fall out of the empirical distributions.

4. **Aggregate threshold τ:** pick the value that achieves your target true-rejection rate on the bad set (say 95%) while accepting ≥ 95% of the reference good set. Plot the ROC. If the curves don't separate well, your metrics aren't discriminative enough — go fix metrics, not τ.

5. **Sanity-check with held-out genres:** train thresholds on jazz+pop, validate on blues+hip-hop. If thresholds generalize, you're done; if not, make them genre-conditional from `GenreSpec`.

This eliminates the magic 1.1 / 0.9 calibration factors entirely — they are replaced by learned per-metric soft mappings with a defensible derivation.

**Stronger validity option:** the MGEval methodology (Yang & Lerch) is *intra-set distance distribution* matching, which is the right tool if you also have a real-music reference corpus. If you have even a small MIDI corpus per genre, you can additionally check that your generator's metric distributions overlap with real music's, which is a stronger validity claim than "not as bad as the deliberately-broken generator." Worth doing once you have time; not blocking.

---

## 7. Python implementation — what's in the stack, what's new

| Metric | Library | In stack? |
|---|---|---|
| Krumhansl–Schmuckler key correlation | music21 `analysis.discrete` | ✅ |
| Pitch-class histogram, KL divergence | music21 + numpy/scipy | ✅ |
| Melodic intervals, scale adherence | music21 streams | ✅ |
| Pitch range, density, melody/bass crossing | mido + numpy | ✅ |
| N-gram entropy | `scipy.stats.entropy` on `Counter` | ✅ |
| LZ compression ratio | stdlib `zlib` | ✅ |
| Bar-level self-similarity | numpy cosine similarity | ✅ |
| Voice-leading distance | hand-coded on Roman numerals | ✅ |
| Chroma–key correlation | librosa + numpy | ✅ |
| Tempogram peak analysis | `librosa.feature.tempogram` | ✅ |
| Beat strength, onset autocorr | librosa | ✅ |
| Clipping, DC, silence, crest | numpy on raw WAV | ✅ |
| **Integrated LUFS (EBU R128)** | **`pyloudnorm`** | **❌ — recommended add** |
| Sethares roughness (optional) | `essentia` or DIY | ❌ — DIY is fine |
| MusPy convenience features (optional) | `muspy` | ❌ — not needed if you implement directly |

**Only one new dependency I'd argue is worth it: `pyloudnorm`.** It's small, BS.1770-compliant, well-maintained. Rolling your own R128 integrated loudness with K-weighting and gating is more work than it sounds. Everything else can be implemented with what you already have. MusPy has nice helpers but they're thin wrappers over music21/mido. jSymbolic is Java and not worth a JVM dependency for this use case.

**Determinism:** every metric above is deterministic given input. Watch out for one thing — librosa's beat tracker has a `tightness` parameter and uses internal heuristics that *are* deterministic but version-sensitive. **Pin librosa exactly in `requirements.txt`**, or you'll see scores drift across environments.

---

## 8. Pushback on the original framing

Three things worth questioning in the prompt itself:

### 8.1 "Score in [0, 1] with weights summing to 1.0"
I'd argue against this as the primary representation. A composite score with a hard threshold is *less informative* than a vector of independent gate decisions, and it makes you blind to which dimension caused a rejection. Log the per-metric scores; gate on the conjunction; treat the aggregated soft score as a tiebreaker, not the primary signal. This also makes the timbre-weight class of bugs impossible going forward — you can't silently drop a metric if each metric has its own gate.

### 8.2 Render-then-gate vs seed-aware re-roll
Even with Layer 1 in place, consider whether the gate should reject *and re-seed* deterministically rather than re-roll randomly. With a deterministic seeded generator, three re-rolls with new seeds is fine, but if the same seed family produces correlated failures (e.g. a particular Markov state pair always cycles), you'll waste budget. Track which (seed, genre) combinations get rejected and bias future seed selection — or, more straightforwardly, expand the seed search space for genres with high rejection rates. This is a generator-design question more than a scoring question, but it interacts with how aggressive you can make the gate.

### 8.3 The timbre bug
Don't "fix" it by computing a timbre score and adding it back. Timbre at the FluidSynth-render stage is determined by soundfont choice, not by anything generative; scoring it tells you about your soundfont, not your sample. **Just delete the field from the weights dict.**

---

## 9. Implementation order (suggested)

A concrete ordering that front-loads the highest-value work:

1. **Delete the timbre weight** (one line). The bug stops being a bug.
2. **Implement Layer 1 standalone** — start with key-profile correlation, scale adherence, pitch range, stuck-note check, n-gram entropy, LZ ratio. These six metrics will catch the majority of pathologies.
3. **Build the reference-good and known-bad generation harnesses.** ~1 day of work.
4. **Plot per-metric distributions over both sets.** This tells you immediately whether each metric is discriminating. Drop or fix metrics that don't separate the sets.
5. **Derive soft mappings and τ₁** from the plotted distributions (Section 6).
6. **Replace the audio layer** — keep only render-integrity checks (clipping, DC, silence, LUFS, crest), genre-aware tempo, chroma–key correlation, tempogram peak. Delete the tonnetz corrcoef and most of the existing analyzers.
7. **Add `pyloudnorm` and pin librosa.**
8. **Optional later:** MGEval-style intra-set distance distribution matching against a real-music reference corpus, once a small per-genre MIDI corpus is available.

---

## 10. Citations

Citations limited to those I am confident are correctly attributed:

- **Krumhansl & Kessler 1982**, *Psychological Review* — key profiles
- **Krumhansl 1990**, *Cognitive Foundations of Musical Pitch*, Oxford University Press
- **Huron 2006**, *Sweet Anticipation*, MIT Press — melodic expectation, step/leap statistics
- **Pearce, M. T., PhD thesis (2005), City University London** — IDyOM
- **Pearce & Wiggins** — subsequent IDyOM evaluation papers (multiple, 2006–2012; check current bibliography for exact venue when citing in a paper)
- **Lerdahl 2001**, *Tonal Pitch Space*, Oxford University Press — chord-distance metric
- **Lerdahl & Krumhansl** — tension-model validation work in *Music Perception* (mid-2000s; verify exact year before citing in a paper)
- **Farbood** — quantitative tension models (early 2010s; verify exact year/venue)
- **Herremans & Chew** — tension ribbon, computable implementation of Lerdahl-style tension
- **Chew, E.** — spiral array model (early 2000s)
- **Sethares 1993**, *Journal of the Acoustical Society of America* — sensory dissonance
- **Foote, J.** — self-similarity matrices for music structure (1999–2000, ACM Multimedia / ICME)
- **Manaris et al.** — Zipf-law studies in music, including *Computer Music Journal* publications (mid-2000s onward)
- **McKay, C.** — jSymbolic / jSymbolic2 symbolic feature toolbox
- **Dong et al. 2020**, ISMIR — MusPy
- **Yang & Lerch** — MGEval methodology (intra-set distance distributions)
- **ITU-R BS.1770 / EBU R128** — integrated loudness standard

For any of the citations above where I've flagged uncertainty about year or venue, do a quick lookup before including in a formal write-up. The work itself is real; the bibliographic precision is what I'm hedging on.

---

## Appendix A — Concrete formulas and pseudocode

### A.1 Pitch-class histogram + KS key correlation
```python
import numpy as np

# Krumhansl-Schmuckler major and minor profiles (Krumhansl & Kessler 1982)
KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                     2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                     2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

def pc_histogram(notes_with_durations):
    h = np.zeros(12)
    for pitch, dur in notes_with_durations:
        h[pitch % 12] += dur
    s = h.sum()
    return h / s if s > 0 else h

def ks_key_correlation(pc_hist):
    """Return (best_key_index_0_to_23, best_correlation)."""
    best = (-1, -1.0)
    for tonic in range(12):
        for prof_idx, prof in enumerate((KS_MAJOR, KS_MINOR)):
            rotated = np.roll(prof, tonic)
            corr = np.corrcoef(pc_hist, rotated)[0, 1]
            key_idx = tonic + 12 * prof_idx
            if corr > best[1]:
                best = (key_idx, corr)
    return best
```

### A.2 N-gram entropy on scale degrees
```python
from collections import Counter
import numpy as np

def ngram_entropy(symbols, n=3):
    if len(symbols) < n:
        return 0.0
    grams = [tuple(symbols[i:i+n]) for i in range(len(symbols) - n + 1)]
    counts = Counter(grams)
    total = sum(counts.values())
    probs = np.array([c / total for c in counts.values()])
    return float(-np.sum(probs * np.log2(probs)))
```

### A.3 LZ compression ratio
```python
import zlib

def lz_ratio(symbols):
    """Compression ratio of a symbol sequence; lower = more compressible."""
    raw = bytes(symbols) if all(0 <= s < 256 for s in symbols) else \
          ' '.join(map(str, symbols)).encode()
    if not raw:
        return 1.0
    compressed = zlib.compress(raw, level=9)
    return len(compressed) / len(raw)
```

### A.4 Bar-level self-similarity
```python
import numpy as np

def bar_pc_vectors(bars):
    """bars: list of lists of (pitch, duration). Returns (n_bars, 12) matrix."""
    M = np.zeros((len(bars), 12))
    for i, bar in enumerate(bars):
        for p, d in bar:
            M[i, p % 12] += d
        s = M[i].sum()
        if s > 0:
            M[i] /= s
    return M

def mean_off_diagonal_similarity(M):
    if M.shape[0] < 2:
        return 0.0
    # cosine similarity matrix
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Mn = M / norms
    S = Mn @ Mn.T
    n = S.shape[0]
    mask = ~np.eye(n, dtype=bool)
    return float(S[mask].mean())
```

### A.5 Render-integrity checks
```python
import numpy as np

def render_integrity(y, sr):
    """Return dict of checks; each value is ready to threshold."""
    peak = float(np.max(np.abs(y)))
    rms = float(np.sqrt(np.mean(y**2)))
    return {
        'clipping_ratio': float(np.mean(np.abs(y) >= 0.99)),
        'dc_offset': float(abs(np.mean(y))),
        'silence_ratio': float(np.mean(_frame_rms(y, sr) < 1e-3)),  # ~-60 dBFS
        'crest_db': 20 * np.log10(peak / rms + 1e-12),
        'peak': peak,
        'rms': rms,
    }

def _frame_rms(y, sr, frame_ms=20):
    n = int(sr * frame_ms / 1000)
    if n <= 0 or len(y) < n:
        return np.array([np.sqrt(np.mean(y**2) + 1e-12)])
    trimmed = y[: (len(y) // n) * n].reshape(-1, n)
    return np.sqrt(np.mean(trimmed**2, axis=1) + 1e-12)
```

### A.6 LUFS via pyloudnorm
```python
import pyloudnorm as pyln

def integrated_lufs(y, sr):
    meter = pyln.Meter(sr)  # BS.1770 K-weighted
    return float(meter.integrated_loudness(y))
```

---

## Appendix B — Summary of changes vs current implementation

| Current | Recommendation |
|---|---|
| `weights['timbre'] = 0.15` (never computed) | **Delete the key.** |
| Hard 60–180 BPM clip | Genre-aware via `GenreSpec.tempo_range` |
| `corrcoef(tonnetz[i], tonnetz[i+1])` "consonance" | Replace with chroma–key correlation against KS profiles |
| `calibration = {...}` with magic 1.1 / 0.9 factors | Replace with empirical per-metric soft mappings derived from reference + adversarial sets |
| Audio-only scoring after FluidSynth render | Add Layer 1 MIDI pre-filter; only render if Layer 1 passes |
| Single weighted-sum score in [0, 1] | Bank of independent hard + soft checks; aggregate only as tiebreaker |
| ZCR + flatness + contrast for noise | Drop ZCR; keep flatness; add render-integrity (clip/DC/silence/LUFS/crest) |
| No melodic, harmonic, or structural symbolic checks | Add KS key correlation, scale adherence, melodic intervals, n-gram entropy, LZ ratio, bar self-similarity, voice-leading distance |
| No genre awareness in scoring | Genre-conditional thresholds throughout, sourced from existing `GenreSpec` |
