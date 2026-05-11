# Musicality Scoring in musicgen

## Abstract

musicgen's quality gate must reject samples that would contaminate a training distribution — not rank good music from very good music. This document presents the two-layer scoring architecture introduced in v0.3: a fast symbolic (MIDI) pre-filter that catches the structural pathologies Markov generators actually produce, followed by a lightweight audio integrity layer that catches render failures the symbolic layer cannot see. We describe each metric's theoretical basis, its implementation, and the calibration methodology used to derive defensible thresholds without human labels.

---

## 1. Introduction

Synthetic music generators produce a predictable class of failures that audio-surface metrics largely miss. A sample with a melody that repeats scale degree 1–2–1–2 for 60 seconds sounds like music to a zero-crossing-rate detector. A clipped render with a beautiful harmonic progression scores well on chroma stability. The goal of a training-data quality gate is precisely to catch what audio metrics miss: structural degeneration at the generator level and render-level artifacts at the synthesis level.

This shifts the design principle. The symbolic layer — operating on MIDI before FluidSynth runs — does the bulk of the quality work, because it has direct access to the structures the generator produces. The audio layer narrows to render integrity confirmation. Together they form a **rejection cascade**: if a sample fails any hard check, it is rejected immediately; only samples passing all hard checks reach the soft-score aggregation.

The framing follows the practical guidance of Yang and Lerch [16] on the evaluation of generative music models: intra-set distance distributions and pathology recall matter more than a single composite score when the downstream consumer is a machine learning training pipeline.

---

## 2. Failure Modes of Markov Generators

Markov-chain generators over Roman numerals and scale degrees produce a predictable failure taxonomy:

| Failure class | Description | Detectable in |
|---|---|---|
| Tonal drift | Melody wanders outside the implied key | MIDI (KS correlation, scale adherence) |
| Degenerate repetition | Chain enters short cycle (e.g. I–V–I–V or scale deg. 1–2–1–2) | MIDI (n-gram entropy, LZ ratio) |
| Degenerate non-repetition | Every bar different; no motivic identity | MIDI (n-gram entropy, LZ ratio) |
| Stuck note | Single pitch dominates a layer | MIDI (hard check) |
| Extreme pitch range | Octave bugs, melody crossing bass | MIDI (hard check) |
| Empty layer | Layer generates no notes | MIDI (hard check) |
| Render clipping | FluidSynth velocity or gain bug | Audio |
| Near-silence | Soundfont mismatch, voice exhaustion | Audio |
| DC offset | Synthesis artifact | Audio |

A single composite score hides which dimension caused a failure and can mask category-level pathologies through compensation. The architecture below treats each check as an independent gate.

---

## 3. Two-Layer Quality-Gate Architecture

```
MIDI files
    │
    ▼
┌─────────────────────────────┐
│  Layer 1 — Symbolic         │  < 5 ms, pre-render
│  Hard checks:               │
│   • empty layer             │  → passed=False, score=0.0
│   • stuck pitch (> 80%)     │
│   • extreme range (> 36 st) │
│  Soft metrics:              │
│   • KS key correlation      │  → score ∈ [0, 1]
│   • scale adherence         │
│   • melodic step fraction   │
│   • n-gram entropy          │
│   • LZ compression ratio    │
└──────────┬──────────────────┘
           │ passed=True
           ▼
      FluidSynth render
           │
           ▼
┌─────────────────────────────┐
│  Layer 2 — Audio integrity  │  ~ 50–150 ms, post-render
│  (embedded in MusicalityAnalyzer)        │
│  Integrity penalty:         │
│   • clipping ratio          │  → penalty applied to score
│   • silence ratio           │
│  Audio analysis:            │
│   • tempo (stability +      │
│     reasonableness + clarity│
│   • harmony (KS via chroma, │
│     key clarity, stability) │
│   • rhythm (regularity +    │
│     strength + pattern +    │
│     density)                │
│   • noise (spectral metrics)│
└──────────┬──────────────────┘
           │
           ▼
    musicality_score ∈ [0, 1]
```

The quality-gate loop in `generate()` retries `_run_pipeline` up to `Config.max_attempts` times, using a distinct seed per attempt (`attempt_seed = sample_seed + (attempt − 1)`), and stops when `musicality_score ≥ Config.min_musicality_score`. Setting `min_musicality_score = 0.0` (the default) disables the gate entirely.

---

## 4. Layer 1: Symbolic (MIDI) Analysis

Layer 1 operates on the multi-track MIDI files produced by the generators, before FluidSynth renders any audio. It reads note-on events via `mido.MidiFile` and costs under 5 ms for a typical 60-second sample.

### 4.1 Hard Checks

Hard checks cause immediate rejection (`passed=False`, `score=0.0`) without consulting soft metrics.

**Empty layer.** Every active layer must contain at least one note-on event with `velocity > 0`. An empty layer indicates a generator or MIDI write failure.

**Stuck pitch.** If a single pitch class accounts for more than 80 % of melody note-on events, the melody Markov chain has entered a degenerate absorbing state. Threshold: `dominant_fraction > 0.80`.

**Extreme pitch range.** If `max(pitches) − min(pitches) > 36` semitones (three octaves) for the melody layer, the generator has produced an unplayable range almost certainly caused by an octave encoding bug.

### 4.2 Soft Symbolic Metrics

Soft metrics are computed on the melody layer and aggregated by equal-weight mean into `MIDIQualityResult.score`.

#### 4.2.1 Krumhansl–Schmuckler Key-Profile Correlation

The KS algorithm [1, 2] measures tonal coherence by correlating the pitch-class histogram of a piece against empirically validated probe-tone profiles for all 24 major and minor keys.

**Profiles** (from Krumhansl & Kessler [1]):

```
Major: [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
Minor: [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
```

The pitch-class histogram is computed from melody note-on events (unweighted by duration at Layer 1). For each of the 24 key profiles (12 major rotations × 12 minor rotations), Pearson correlation is computed against the histogram. The maximum correlation `r*` is mapped to `[0, 1]` via `(r* + 1) / 2`.

High score → melody distributes pitch classes consistent with a tonal key. Low score → pitch classes spread uniformly or contrary to any key (atonal drift).

**Implementation:** `musicgen.musicality._ks_key_correlation(pc_hist)`

#### 4.2.2 Scale Adherence

Scale adherence measures the fraction of melody pitches whose pitch class falls within the diatonic scale of the specified key:

```
score = |{p ∈ pitches : p % 12 ∈ scale_pcs(key)}| / |pitches|
```

For tonal genres this should exceed 0.85; below 0.5 indicates the Markov chain has wandered off-key [2, §6.4]. Jazz and blues genres tolerate more chromaticism. Scale adherence requires the key to be known at query time; `check_midi_quality` accepts a `key` argument (default `"C"`).

**Implementation:** `musicgen.musicality._scale_adherence_score(pitches, key)`

#### 4.2.3 Melodic Step Fraction

Huron's analysis of melodic expectation [3, Ch. 5] documents that tonal melodies strongly favour conjunct (stepwise) motion. The fraction of consecutive melodic intervals with absolute semitone distance ≤ 2 serves as a structural sanity check:

```
step_fraction = |{i : |pitch[i+1] − pitch[i]| ≤ 2}| / (|pitches| − 1)
```

Typical values in tonal music: 0.5–0.75. Values near 0 indicate a generator that only produces large leaps; values near 1 indicate a stuck-note-like pattern of adjacent half-steps.

**Implementation:** `musicgen.musicality._melodic_step_fraction(pitches)`

#### 4.2.4 N-gram Entropy

Shannon entropy of the trigram distribution over melody pitches measures structural variety. The IDyOM framework [4] (and its successor evaluations [5]) establishes entropy over n-gram symbol distributions as a principled proxy for melodic predictability, correlating well with listener surprise judgments.

The implementation computes the empirical trigram distribution from melody pitch values, computes raw Shannon entropy, and normalises by the log of the number of distinct trigrams observed:

```
H_raw = −Σ p(t) log₂ p(t)
H_norm = H_raw / log₂(|trigrams|)    [clipped to [0, 1]]
```

Extreme-low entropy: degenerate repetition. Extreme-high entropy: random walk / atonal noise. Healthy generation sits in the middle band.

**Implementation:** `musicgen.musicality._ngram_entropy(symbols, n=3)`

#### 4.2.5 LZ Compression Ratio

The Lempel–Ziv compression ratio of the melody pitch sequence is a model-free proxy for structural regularity. Manaris et al. [13] validated compression ratio as a coarse discriminator between tonal music corpora and noise. Its value for a quality gate lies at the distribution tails: extreme compressibility (ratio near 0) reveals cyclic repetition that entropy alone might underweight; near-incompressibility (ratio near 1) reveals random-pitch sequences.

The pitch sequence is serialised to bytes (`pitch % 256`) and compressed with `zlib.compress`. The ratio `len(compressed) / len(raw)` is clipped to `[0, 1]`.

**Implementation:** `musicgen.musicality._lz_ratio(symbols)`

### 4.3 Scoring Aggregation

`check_midi_quality` returns a `MIDIQualityResult` dataclass with:

- `passed`: `True` iff no hard check failed
- `score`: `0.0` if `passed=False`; otherwise the equal-weight mean of all computed soft scores
- `hard_failures`: list of descriptive strings for each failed hard check
- `soft_scores`: dict mapping metric name to individual score

The aggregate score is a tiebreaker and calibration aid, not the primary gate. The `passed` flag is the authoritative gate signal.

---

## 5. Layer 2: Audio Analysis

Layer 2 runs after FluidSynth renders the audio. Its primary role is render-integrity checking; the musical analysis sub-scores provide a secondary quality signal.

### 5.1 Render Integrity Checks

These four metrics detect FluidSynth synthesis failures that the symbolic layer cannot anticipate.

**Clipping ratio.** Fraction of samples with absolute value ≥ 0.99. Values above 0.1 % indicate a velocity or gain bug producing output near ±1.0.

**DC offset.** Absolute mean of the audio signal. Values above 0.001 indicate a DC bias in the synthesis output, which causes aliasing after resampling and degrades downstream spectral analysis.

**Silence ratio.** Fraction of samples with absolute value below 0.01 (≈ −40 dBFS). Values above 0.5 indicate a near-silent render — typically caused by a soundfont mismatch or FluidSynth voice exhaustion.

**Crest factor (dB).** `20 log₁₀(peak / RMS)`. For a healthy music signal: 3–30 dB. Below 3 dB suggests severe clipping or a DC-dominated signal. Above 30 dB suggests extremely sparse transients on near-silence.

The integrity penalty applied to the final score is:

```python
clipping_penalty = clip(clipping_ratio * 2.0, 0, 1)
silence_penalty  = clip((silence_ratio − 0.5) * 2.0, 0, 1)
integrity_penalty = max(clipping_penalty, silence_penalty)
final_score = raw_score * (1.0 − integrity_penalty)
```

**Implementation:** `musicgen.musicality._render_integrity(y, sr)`

### 5.2 Audio Musical Analysis

`MusicalityAnalyzer` computes four sub-dimensions on the loaded audio waveform via librosa.

**Tempo** (weight 0.30). Genre-aware: if a `GenreSpec` is available, the scored range is `[tempo_min, tempo_max]`; otherwise `[40, 240]` BPM. Three sub-metrics: beat stability (inverse CV of beat intervals), tempo reasonableness (proximity to genre centre), onset clarity (mean / max onset strength). Estimated via `librosa.beat.beat_track`.

**Harmony** (weight 0.30). Three sub-metrics computed from the CQT chroma (`librosa.feature.chroma_cqt`, 36 bins/octave): KS correlation of the mean chroma vector against all 24 key profiles (same implementation as Layer 1), key clarity (max of mean chroma), harmonic stability (1 − mean frame-to-frame chroma change). The KS correlation replaces the former tonnetz corrcoef, which measured frame stationarity rather than tonal coherence.

**Rhythm** (weight 0.25). Four sub-metrics via onset envelope and `librosa.beat.beat_track`: beat regularity (inverse CV of beat intervals), beat strength (mean peak onset above background), pattern score (onset autocorrelation smoothness), and density score (beats per second proximity to 1.5 Hz). Sub-metrics are pre-weighted (0.35/0.30/0.20/0.15) before aggregation.

**Noise** (weight 0.15). Three spectral quality indicators: ZCR-derived score (1 − mean ZCR), spectral flatness score (1 − mean flatness), and spectral contrast score (mean / max contrast). The combined noise sub-score is the mean of all three.

### 5.3 Score Composition

```
raw_score   = Σ weight_k × mean(sub-scores_k)
final_score = raw_score × (1 − integrity_penalty)
```

Weights sum to 1.0 (`tempo=0.30`, `harmony=0.30`, `rhythm=0.25`, `noise=0.15`). Integrity penalty is applied multiplicatively: a heavily clipped sample cannot score above `1 − penalty` regardless of musical quality.

**Implementation:** `musicgen.musicality.MusicalityAnalyzer`, `get_musicality_score(filename, genre_spec=None)`

---

## 6. Quality-Gate Integration

The quality gate is wired into `musicgen.api.generate()` via two `Config` fields:

| Field | Default | Env var | Effect |
|---|---|---|---|
| `min_musicality_score` | `0.0` | `MUSICGEN_MIN_MUSICALITY_SCORE` | Gate disabled at 0.0. Samples with `score ≥ threshold` pass. |
| `max_attempts` | `1` | `MUSICGEN_MAX_ATTEMPTS` | Number of re-roll attempts before accepting the last result. Must be ≥ 1. |

`SampleResult` carries an `attempt` field (default `1`) indicating which attempt produced the accepted result. The manifest entry includes the `attempt` field.

Seed derivation across attempts preserves backward compatibility:

```python
attempt_seed = sample_seed + (attempt − 1)
# attempt=1 → attempt_seed == sample_seed  (no change from pre-v0.3)
```

---

## 7. Calibration Methodology

Calibration derives an empirical `min_musicality_score` threshold without human labels. The approach — reference-set bootstrapping with adversarial examples — exploits full control over the generator's input distribution.

### 7.1 Reference-Good Set

The reference-good set consists of step-wise C-major melodies: all notes drawn from the C major scale (`[C4, D4, E4, F4, G4, A4, B4, C5]`) in ascending order, one octave range, randomised velocities (70–100). These are structurally sound by construction: high scale adherence, high step fraction, moderate entropy, low LZ ratio.

Non-melody layers (beat, harmony, bassline) receive a C-pentatonic filler pattern across all samples, ensuring no hard failures in those layers.

### 7.2 Adversarial Set

Four failure variants are cycled across the adversarial set:

| Variant | Description | Primary pathology |
|---|---|---|
| 0 | Empty melody (0 notes) | Hard failure: empty layer |
| 1 | Stuck note (15/16 same pitch) | Hard failure: stuck pitch |
| 2 | Extreme range (MIDI 36–84, 4-octave span) | Hard failure: extreme range |
| 3 | Chromatic noise (random semitones 36–96) | Soft failure: low KS, low scale adherence |

Variants 0–2 exercise the hard-check path (score = 0.0). Variant 3 exercises the soft-metric path and validates that the metrics collectively distinguish random-pitch sequences from structured melodies.

### 7.3 Threshold Derivation

`suggest_threshold(good_scores, bad_scores)` uses a non-parametric separation test:

```python
p10_good = np.percentile(good_scores, 10)
p90_bad  = np.percentile(bad_scores,  90)

if p10_good > p90_bad:
    # Distributions clearly separated
    threshold = (p10_good + p90_bad) / 2.0
else:
    # Overlapping — conservative fallback
    threshold = np.percentile(good_scores, 25)
```

The result is validated by `separation_ok`, which requires `good_mean − bad_mean ≥ 0.2`. When separation is not achieved, the metric set should be revisited rather than adjusting the threshold.

**Implementation:** `musicgen.calibrate.run_midi_calibration`, `musicgen.calibrate.suggest_threshold`

The `musicgen calibrate` CLI command runs the full calibration harness and prints the recommended threshold. Results are persisted via `save_calibration` / `load_calibration` for reproducibility.

---

## 8. Implementation Map

```
src/musicgen/
├── musicality.py       — Layer 1 + Layer 2 scoring
│   ├── MIDIQualityResult       dataclass
│   ├── check_midi_quality()    Layer 1 entry point
│   ├── _ks_key_correlation()   KS soft metric
│   ├── _scale_adherence_score()
│   ├── _melodic_step_fraction()
│   ├── _ngram_entropy()
│   ├── _lz_ratio()
│   ├── _render_integrity()     Layer 2 integrity
│   ├── MusicalityAnalyzer      Layer 2 audio analysis
│   └── get_musicality_score()  public Layer 2 entry point
└── calibrate.py        — calibration harness
    ├── CalibrationResult       dataclass
    ├── run_midi_calibration()
    ├── suggest_threshold()
    ├── save_calibration()
    └── load_calibration()
```

Layer 1 entry: `check_midi_quality(midi_paths: Dict[str, str], key: str) → MIDIQualityResult`

Layer 2 entry: `get_musicality_score(filename: str, genre_spec=None) → (float, Dict[str, float])`

Both are called from `musicgen.api._run_pipeline`. Layer 1 runs pre-render; Layer 2 runs after `renderer.render_stems` produces `mix.wav`. The `get_musicality_score` call is wrapped in `save_random_state()` to guard against global-state leakage from librosa's internal random use.

---

## 9. Design Decisions and Scope

**Why a bank of checks rather than a weighted sum?** A composite score hides the failure dimension and allows compensation: a severely clipped sample with a well-structured melody can average to a passing score. Independent gates make each failure visible and tunable. The aggregate soft score serves as a tiebreaker when re-rolling, not as the primary gate criterion.

**Why Layer 1 rather than audio-only?** Audio metrics measure surface properties. They detect "is this a music signal?" but not "did the Markov chain enter a degenerate cycle?" Pre-render symbolic checks cost < 5 ms and catch the dominant generator failure modes. Only structurally passing samples incur the FluidSynth render cost.

**What is not implemented:** Integrated loudness (EBU R128 / ITU-R BS.1770) via `pyloudnorm` is the highest-value missing metric for render-integrity; it would replace the crest-factor check as the primary loudness gate. Bar-level pitch-class self-similarity (Foote [12]) and voice-leading distance (Lerdahl [6]) are implemented in the audit pseudocode but deferred — they require bar-segmented MIDI parsing not yet in the pipeline. Full IDyOM-style information-content estimation [4, 5] requires a trained corpus model; the n-gram entropy approximation captures the discriminating signal for a gate at lower cost. The Sethares roughness model [11] for sensory dissonance was evaluated but its discriminating signal for FluidSynth-synthesised music is low.

**Timbre.** The pre-v0.3 implementation carried a `weights['timbre'] = 0.15` slot that was never computed, causing a systematic bias in all scores. It was deleted in v0.3 Phase 3a. Timbre at the FluidSynth stage is determined by soundfont selection, not by the generative process; scoring it measures the soundfont pool, not the sample.

---

## 10. Literature Notes on Perceptual Validity

The metrics implemented here connect to perceptual research as follows:

- **KS key profiles** [1, 2] are the most extensively validated tool for tonal-coherence detection at scale and cost. The probe-tone methodology has been replicated across languages and cultures and is the standard implementation in music analysis libraries.
- **Melodic step preference** [3] is a well-documented statistical regularity in tonal music across cultures, not a cultural norm. The step-fraction threshold is a soft gate, not a hard rule.
- **N-gram entropy / IDyOM** [4, 5]: the full IDyOM system predicts listener surprise judgments with high accuracy using variable-order Markov models trained on music corpora. The n-gram entropy approximation captures the coarse signal (repetition tails) but not the nuanced predictability curve. It is sufficient for a gate; it is not sufficient for quality ranking.
- **Compression ratio** [13]: validated as a coarse filter on large corpora. Reliable at the distribution tails (extreme repetition, random noise); unreliable in the middle band. Implemented as a supporting signal.
- **Lerdahl tension model** [6, 7] and **Farbood** [8]: validated against listener tension judgments; the full computation is overkill for a gate. Voice-leading distance is a lightweight proxy deferred to future work.

The honest meta-point from Yang and Lerch [16]: no single composite score is musicality. The goal of this gate is pathology recall, not quality ranking.

---

## References

[1] Krumhansl, C. L. and Kessler, E. J. (1982). Tracing the dynamic changes in perceived tonal organization in a spatial representation of musical keys. *Psychological Review*, 89(4):334–368.

[2] Krumhansl, C. L. (1990). *Cognitive Foundations of Musical Pitch*. Oxford University Press.

[3] Huron, D. (2006). *Sweet Anticipation: Music and the Psychology of Expectation*. MIT Press.

[4] Pearce, M. T. (2005). *The Construction and Evaluation of Statistical Models of Melodic Structure in Music Perception and Composition*. PhD thesis, City University London.

[5] Pearce, M. T. and Wiggins, G. A. (2012). Auditory expectation: the information dynamics of music perception and cognition. *Topics in Cognitive Science*, 4(4):625–652.

[6] Lerdahl, F. (2001). *Tonal Pitch Space*. Oxford University Press.

[7] Lerdahl, F. and Krumhansl, C. L. (2007). Modeling tonal tension. *Music Perception*, 24(4):329–366.

[8] Farbood, M. M. (2012). A parametric, temporal model of musical tension. *Music Perception*, 29(4):387–428.

[9] Herremans, D. and Chew, E. (2016). Tension ribbons: Quantifying and visualising tonal tension. In *Proceedings of the 2nd International Conference on Technologies for Music Notation and Representation (TENOR)*.

[10] Chew, E. (2002). The spiral array: An algorithm for determining key boundaries. In *Proceedings of the 2nd International Conference on Music and Artificial Intelligence (ICMAI)*, Springer LNCS 2445, pp. 18–31.

[11] Sethares, W. A. (1993). Local consonance and the relationship between timbre and scale. *Journal of the Acoustical Society of America*, 94(3):1218–1228.

[12] Foote, J. (1999). Visualizing music and audio using self-similarity. In *Proceedings of the 7th ACM International Conference on Multimedia*, pp. 77–80.

[13] Manaris, B., Romero, J., Machado, P., Krehbiel, D., Hirzel, T., Pharr, W., and Davis, R. B. (2005). Zipf's law, music classification, and aesthetics. *Computer Music Journal*, 29(1):55–69.

[14] McKay, C. and Fujinaga, I. (2006). jSymbolic: A feature extractor for MIDI files. In *Proceedings of the International Computer Music Conference (ICMC)*, pp. 302–305.

[15] Dong, H.-W., Chen, K., McAuley, J., and Berg-Kirkpatrick, T. (2020). MusPy: A toolkit for symbolic music generation. In *Proceedings of the 21st International Society for Music Information Retrieval Conference (ISMIR)*, pp. 101–108.

[16] Yang, L.-C. and Lerch, A. (2020). On the evaluation of generative models in music. *Neural Computing and Applications*, 32(9):4773–4784.

[17] ITU-R. (2015). *BS.1770-4: Algorithms to measure audio programme loudness and true-peak audio level*. International Telecommunication Union.

[18] EBU. (2014). *EBU R 128: Loudness Normalisation and Permitted Maximum Level of Audio Signals*. European Broadcasting Union.
