# DSA Research Documentation

## Digilog Scalable Audio — Design, Theory, and Implementation

**Status:** Working document — updated as implementation progresses
**Started:** April 2026
**Repository:** https://github.com/pisdronio/dsa
**Format spec:** https://github.com/pisdronio/digilog-spec

---

## Abstract

This document describes the design and implementation of DSA (Digilog Scalable Audio), a novel audio codec developed specifically for the Digilog physical audio format. DSA addresses a set of requirements that no existing open codec satisfies simultaneously: layered scalable decoding that maps directly to physical spatial regions, native bidirectional playback for real-time scratch performance, and graceful analog-style degradation under partial data loss.

The codec uses the Modified Discrete Cosine Transform (MDCT) as its analysis foundation, a perceptual frequency band model derived from ISO 226 equal-loudness contours, a hierarchical K-frame/B-frame GOP structure inspired by video codec design, and a layered bitstream format that maps directly to concentric rings on a printed disc.

---

## 1. Motivation

### 1.1 The Digilog format

Digilog is an open physical audio format that encodes audio as colored dot patterns printed on any surface — paper, sticker, disc. A camera scans the dots and the audio plays back. No internet, no server, no platform required.

The format has two primary physical expressions:

**Flat codes** (stickers, business cards, posters) — scanned once, audio plays back linearly.

**Digilog Disc** — a circular printed disc that spins on a standard turntable. A phone camera mounted above reads the dots as they rotate past. The disc can be scratched by a DJ — forward and backward motion maps directly to forward and backward audio playback.

### 1.2 Why existing codecs are insufficient

The Digilog Disc imposes requirements that no existing open codec addresses:

**Spatial layer mapping.** The disc has concentric rings. Outer rings contain the highest quality data and require the best optics to read. Inner rings contain the minimum viable data and are always readable. The codec must produce a bitstream whose layers correspond to these physical regions — not three separate encodes, but one hierarchically structured stream where each ring genuinely enhances the previous.

**Native reverse playback.** A DJ scratching backward must hear reversed audio. Opus, MP3, and AAC produce corruption when their bitstreams are decoded in reverse — they were never designed for bidirectional access. DSA is designed from the start for bidirectional decoding.

**Analog degradation.** When a disc is worn, scratched, or partially obscured, standard codecs fail completely — the bitstream becomes undecodable. DSA must degrade gracefully: a partial layer read should produce lower quality audio, not silence. A partial frame read should produce filtered audio, not noise.

**Motion-aware decoding.** The disc spins at variable speed during scratch performance. The codec must support variable-rate decoding where the "needle" (camera) can be at any position, moving at any speed, in either direction.

### 1.3 Prior art survey

**Opus (RFC 6716)** — state of the art for low-bitrate audio compression. Excellent perceptual quality at 6-24kbps. Not scalable, not bidirectional, not designed for physical media.

**MPEG-4 SLS (Scalable Lossless)** — layered architecture with lossy core + lossless enhancement. Patented. Layers are not spatially separable. Not designed for physical media.

**FLAC** — lossless, frame-independent. Frame independence is a useful property for random access but quality is not scalable and compression is insufficient for our bit budgets.

**Daala (Mozilla/Xiph, 2013-2017)** — experimental codec with hierarchical superblock structure and perceptual vector quantization. Absorbed into AV1. Audio side never completed. The hierarchical architecture is relevant to DSA's design.

**Codec2** — extremely low bitrate speech codec (700bps-3200bps). Speech only, not suitable for music.

None of these codecs satisfy the full set of DSA requirements. DSA is novel in combining: MDCT analysis, perceptual band layering with spatial disc mapping, bidirectional B-frame structure, and analog degradation modeling.

---

## 2. Mathematical Foundation

### 2.1 The Modified Discrete Cosine Transform

The MDCT is the analysis transform at the core of MP3, AAC, Vorbis, and Opus. DSA uses MDCT for the same reasons these codecs do, plus one additional property critical for Digilog: the TDAC cancellation enables clean reconstruction in both forward and reverse directions.

**Definition.** For a windowed input sequence x[n] of length N = 2M, the MDCT produces M real coefficients:

```
X[k] = sqrt(2/M) * sum_{n=0}^{N-1} w[n] * x[n] * cos(pi/M * (n + 0.5 + M/2) * (k + 0.5))

for k = 0, 1, ..., M-1
```

The inverse MDCT (IMDCT) is:

```
x'[n] = sqrt(2/M) * sum_{k=0}^{M-1} X[k] * cos(pi/M * (n + 0.5 + M/2) * (k + 0.5))

for n = 0, 1, ..., N-1
```

**The sine window.** DSA uses the standard MDCT sine window:

```
w[n] = sin(pi/N * (n + 0.5))   for n = 0, 1, ..., N-1
```

This window satisfies the Princen-Bradley condition:

```
w[n]^2 + w[n + M]^2 = 1   for all n = 0, 1, ..., M-1
```

This is the Time-Domain Aliasing Cancellation (TDAC) property. When consecutive overlapping IMDCT outputs are summed (overlap-add), the aliasing terms from adjacent frames cancel exactly, reconstructing the original signal.

**Verified reconstruction quality.** With N=2048, M=1024 at 44100Hz:
- TDAC SNR: 238dB (effectively perfect reconstruction)
- Frame duration: 23.2ms per hop at 50% overlap

### 2.2 The symmetry property for reverse playback

The sine window has the symmetry property:

```
w[n] = w[N-1-n]   for all n
```

This means that for any frame, analyzing the samples in reverse order and then decoding produces the time-reversed audio — not because MDCT(x) = MDCT(rev(x)) (it does not), but because the overlap-add reconstruction is symmetric in time.

For DSA reverse playback: decode frames in reverse order, apply IMDCT to each, overlap-add in reverse. The TDAC cancellation operates identically in both directions. The result is the true time-reversed audio signal.

This is the mathematical basis for vinyl-like scratch behavior on a Digilog disc.

### 2.3 Confidence-weighted reconstruction

This is the mathematical foundation of DSA's analog degradation model. It is the single property that distinguishes DSA from all prior codecs at a behavioral level.

In every existing audio codec, reconstruction is binary: either a coefficient is decoded correctly, or the frame is lost and silence is produced. This is digital failure — catastrophic, unmusical, jarring.

DSA introduces a third state: **partial confidence**. Every MDCT coefficient in every band carries a confidence value supplied by the visual decoder (the dot reader). Reconstruction is:

```
C̃[k] = q[k] × step[b] × α[b]

where:
  q[k]    = quantized integer coefficient at bin k
  step[b] = quantization step size for band b
  α[b]    = confidence of band b,  α ∈ [0.0, 1.0]
```

This single equation is the analog degradation model.

**At α = 1.0** — full confidence, clean dot read — reconstruction is identical to standard dequantization. This is Mode 1 (discrete dots) under good conditions.

**At α = 0.5** — half confidence — the coefficient is reconstructed at half amplitude, 6dB quieter. The band contributes to the output but at reduced level. Perceptually: that frequency region sounds attenuated, warmer, further away. Not broken. Not absent. Just quieter.

**At α = 0.0** — zero confidence, band unreadable — that frequency band goes silent. If this is L2 (high frequencies), the output is low-pass filtered. If it is L0 (bass), the output is severely degraded but not necessarily silent — the frame's IMDCT still runs on whatever was recovered.

**Spatially varying confidence** — different bands in the same frame having different α values — produces spectrally colored degradation. A worn outer ring produces reduced α across L2 bands. The output sounds like a vinyl record with worn grooves: the highs roll off, the bass stays present, the music remains recognizably itself.

This is not a simulation of analog degradation. It is analog degradation, produced mathematically by the confidence weighting of coefficients before IMDCT reconstruction.

### 2.4 Perceptual frequency bands

DSA maps MDCT coefficients to 48 perceptual frequency bands. The band structure is designed around human auditory perception:

**Layer 0: 8 bands, 20Hz – 800Hz (linear spacing)**

Linear spacing in the bass range because:
- The critical band width of the auditory system is approximately linear below ~500Hz
- Bass frequency resolution matters for pitch perception in the fundamental range
- Kick drum, bass guitar, and fundamental tones live here

**Layer 1: 16 bands, 800Hz – 6000Hz (logarithmic spacing)**

Logarithmic spacing follows the mel scale and musical intervals:
- Equal spacing in octaves corresponds to equal perceptual distance
- The most perceptually important range: voice, melody, harmony, rhythm
- Peak hearing sensitivity is in this range (1-4kHz, ISO 226)

**Layer 2: 24 bands, 6000Hz – 22050Hz (logarithmic spacing)**

High frequency detail:
- Air, presence, hi-hats, consonants in speech
- Less perceptually critical than Layer 1
- Requires the most physical disc area (outer rings) for highest fidelity

**Perceptual importance weights (simplified ISO 226):**

The ISO 226 equal-loudness contours describe how much sound pressure is required at each frequency for equal perceived loudness. DSA uses a simplified model for quantization weighting:

| Frequency range | Weight | Rationale |
|---|---|---|
| < 100Hz | 0.25 | Sub-bass felt more than heard |
| 100–300Hz | 0.55 | Bass |
| 300Hz–1kHz | 0.80 | Low-mid |
| 1–4kHz | 1.00 | Peak sensitivity |
| 4–8kHz | 0.75 | Presence |
| 8–12kHz | 0.45 | Air |
| > 12kHz | 0.20 | Ultra-high |

---

## 3. Frame Architecture

### 3.1 GOP structure

DSA organizes frames into Groups of Pictures (GOPs) of 8 frames, yielding a K-frame every ~185ms:

```
K . B . B . B . B . B . B . B . K . B . B . B . B . B . B . B . K
0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16
```

**K-frames (Keyframes)** are self-contained spectral snapshots. They can be decoded without any reference to other frames. They serve as:
- Drop-in points for random access (needle drop)
- Resynchronization points after scratch displacement
- The only frame type needed for layer-0 only playback

**B-frames (Bidirectional frames)** store the residual between the actual spectral content and the interpolation between surrounding K-frames. Because the reference frames are always K-frames (never other B-frames), B-frames decode identically in forward and reverse playback — the surrounding K-frames are always available regardless of direction.

This architecture is inspired by the B-frame structure in H.264/AVC video compression (ITU-T H.264, 2003), adapted for the specific requirements of audio and physical media.

### 3.2 Scratch recovery time

At 33rpm with camera at 30fps:
- One disc rotation = 1.818 seconds
- 8 reference markers on disc = one marker every 0.227 seconds
- GOP size = 8 frames = 185ms

Maximum time to resynchronize after an arbitrary scratch: 185ms (to next K-frame) + 227ms (to next reference marker) = ~412ms worst case. In practice the camera will hit a reference marker and a K-frame within one rotation — approximately 1.8 seconds maximum for full resync after extreme displacement.

### 3.3 Silence detection and S-frames

Frames with peak band energy below -55dB are marked as S-frames (silence/skip). S-frames store minimal data — just a silence flag. On the disc, S-frame regions can be printed as uniform color patches, reducing visual complexity in silent sections.

---

## 4. Disc Mapping

### 4.1 Layer to ring mapping

The three DSA layers map directly to concentric disc rings:

```
Disc outer edge
  ↓
Clock track ring        — timing markers, rotation speed detection
Reference marker ring   — 8 position anchors, 45° spacing
─────────────────────────────────────────────────────────
Layer 2 rings           — 24 high-frequency bands
                          readable with rig + controlled lighting
─────────────────────────────────────────────────────────
Layer 1 rings           — 16 mid-frequency bands
                          readable on average modern phone
─────────────────────────────────────────────────────────
Layer 0 rings           — 8 bass/fundamental bands
                          always readable, even under hand
─────────────────────────────────────────────────────────
Center sync pattern     — K-frame alignment markers
Center label            — artist / title
Spindle hole
```

This mapping means the physical degradation model of the disc is identical to the perceptual degradation model of the codec: losing outer rings = losing high-frequency detail = sounds like a low-pass filter = like worn vinyl.

### 4.2 Physical degradation → acoustic degradation mapping

The design of the disc ring structure was chosen so that the physics of print degradation and the perceptual structure of audio degradation are the same thing. This is not coincidence — it is the core architectural decision of the format.

```
Physical condition          Layer lost    Acoustic effect
─────────────────────────────────────────────────────────────────
Perfect print, rig          none          full quality
Good print, phone camera    L2 partial    slight high-freq roll-off
Average print, phone        L2            highs gone, mids present
Worn/faded print            L2 + L1       mids attenuated, bass present
Heavily scratched           L2 + L1       bass only, music identifiable
Hand partially covering     outer rings   graceful attenuation of covered region
```

At every point in this table, the output is music. Not silence. Not glitches. Music at reduced quality, in a way that corresponds directly to the physical state of the object.

This is how vinyl works: a worn record sounds warm and distant, not broken. A dusty record adds noise, not silence. The physical medium communicates its own condition through the sound it produces. DSA encodes this property into the codec architecture by design.

### 4.3 K-frame alignment with reference markers

K-frames are aligned with reference markers on the disc clock track. Every time the camera sees a reference marker, it knows:
1. Its absolute position on the disc (which K-frame boundary)
2. The current audio timeline position

This enables the "drop needle anywhere" behavior — equivalent to cueing a vinyl record.

---

## 5. Implementation

### 5.1 Step 1: MDCT Frame Analyzer (complete)

File: `dsa_analyzer.py`

The analyzer converts audio into DSAFrame objects. Each frame contains:
- Raw MDCT coefficients (1024 floats)
- 48 band energies in dB
- 48 band RMS values (linear)
- Frame type (K/B/S)
- GOP position
- Total perceptually weighted energy

**Key implementation decisions:**

*Pre-computed cosine matrix.* The MDCT cosine basis is a 2048×1024 matrix that is constant for a given N. Pre-computing it and reusing across all frames is essential for performance. The matrix product `(windowed_samples @ COS_MAT)` is highly optimized by NumPy's BLAS backend.

*Padding strategy.* Input audio is padded by MDCT_M samples at the start and MDCT_N at the end. This ensures the first and last audio samples are fully covered by overlapping frames and appear in the center of at least two frames, where TDAC reconstruction is accurate.

*Silence detection.* Frames with maximum band energy below -55dB are marked as silence. This threshold was chosen empirically to catch true silence and near-silence (room noise) while not falsely flagging quiet musical passages.

### 5.2 Step 2: Perceptual Quantizer (complete)

File: `dsa_quantizer.py`

The quantizer maps floating-point MDCT coefficients to integer values using a psychoacoustic masking model. Quantization noise is shaped to stay below the masking threshold at each frequency — inaudible noise is not encoded, saving bits for signal that matters.

**Masking model:**

1. **ATH (Absolute Threshold of Hearing)** — per-band minimum audible level from ISO 226, normalized to dBFS. Below this floor, any noise is inaudible regardless of signal content.

2. **Bark-scale spreading function** — pre-computed 48×48 matrix. A loud masker at band m suppresses perception at band b by: −25 dB/Bark upward (low-to-high), −40 dB/Bark downward (high-to-low), with a 14 dB masking index at the masker's own band. Asymmetric because the auditory system is asymmetric.

3. **Global masking threshold** — per band: `max(ATH[b], max_m(L[m] + S[b,m]))`. The worst-case masker wins.

4. **Step sizes** — `10^((threshold − 3dB_headroom) / 20)`. The 3 dB headroom keeps quantization noise safely below, not just at, the masking curve.

**Budget enforcement:**

Layer priority budget scaling degrades L2 first, L1 second, L0 last — matching the disc ring priority. Binary search per layer finds the minimum scale factor that brings estimated bit cost within budget. Only nonzero coefficients are counted (zeros are run-length coded essentially free by the entropy coder).

**Verified properties:**
- Masking threshold is elevated above ATH floor at active signal bands
- Step sizes are all positive
- SNR scales with bitrate: 6kbps → ~9.5dB, 24kbps → ~28dB
- L2 step sizes grow faster than L0 at extreme compression (65536× vs 7× at 3kbps)
- Silence frames reconstruct to exact zero

### 5.3 Steps 3–4: K-frame and B-frame Encoders (complete)

**K-frame encoding:**
1. Quantize all 48 band coefficients using the perceptual quantizer
2. Pack into self-contained bitstream with band-count header
3. Apply per-band Huffman coding

**B-frame encoding:**
1. Interpolate between surrounding K-frames (linear in spectral domain)
2. Compute residual: actual_coefficients - interpolated_coefficients
3. Quantize residual (typically much smaller than full coefficients)
4. Pack residual bitstream with reference K-frame indices

The B-frame residual is symmetric — it encodes the same information whether the surrounding K-frames are before/after or after/before. This is the mathematical property that enables reverse playback.

### 5.4 Steps 5–6: Entropy Coding and Bitstream (pending)

Huffman coding per band group. The symbol distribution for quantized MDCT coefficients is well-studied — roughly Laplacian distributed, centered at zero. Standard Huffman tables from prior codec research can be adapted.

The DSA bitstream format:

```
DSA Bitstream v0.1
├── File header
│   ├── Magic bytes: "DSA1"
│   ├── Version: 1 byte
│   ├── Sample rate: 4 bytes
│   ├── Total frames: 4 bytes
│   └── Layer offsets: 3 × 4 bytes
├── Frame sequence
│   └── Per frame:
│       ├── Frame header (4 bits: type + GOP position)
│       ├── Layer 0 data (K-frame: quantized bands; B-frame: residual)
│       ├── Layer 1 data
│       └── Layer 2 data
└── Checksum: CRC32
```

### 5.5 Step 7: DSA Decoder (pending)

The decoder must support:
- Full decode (all layers available)
- Partial decode (layer 0 only, or 0+1)
- Forward decode (standard playback)
- Reverse decode (scratch backward)
- Variable-rate decode (scratch at non-standard speed)
- Confidence-weighted reconstruction (analog degradation from partial dot reads)

For reverse decode: process frames in reverse order, apply IMDCT, overlap-add in reverse direction. TDAC cancellation operates identically.

For variable-rate: the decoder consumes frames at a rate proportional to the disc rotation speed. At half speed, each frame is stretched to twice its duration using time-scale modification (phase vocoder or similar).

### 5.6 Analog Degradation Model — Decoder Specification

This section defines exactly how the decoder produces analog-style degradation from imperfect physical reads. This is a first-class design requirement, not an error handling afterthought.

**The principle:**

Every path through the decoder that would produce silence or glitch noise in a conventional codec must instead produce attenuated or filtered audio in DSA. The medium communicates its own condition through the character of the sound it produces.

**Confidence input:**

The visual decoder (dot reader) supplies a confidence vector α of shape (NUM_BANDS,) for each frame, with α[b] ∈ [0.0, 1.0]:

```
Mode 1 (discrete dots):
  α[b] = 1.0   if band b was read cleanly
  α[b] = 0.0   if band b region was unreadable

Mode 2 (gradient dots):
  α[b] = continuous value derived from dot gradient clarity
         0.0 = completely unreadable
         0.5 = partial read, attenuated
         1.0 = perfect read
```

**Confidence-weighted dequantization:**

```python
def dequantize_with_confidence(qframe, alpha):
    coeffs = zeros(MDCT_M)
    for band b in range(NUM_BANDS):
        lo, hi = BINS[b]
        coeffs[lo:hi] = qframe.quant_coeffs[b] * qframe.steps[b] * alpha[b]
    return coeffs
```

The IMDCT then operates on these attenuated coefficients. The output is audio with reduced energy in the affected frequency bands — not silence, not noise. The overlap-add smooths the transitions between frames with different confidence profiles.

**Frame loss recovery:**

When a K-frame is unreadable (confidence too low to decode reliably), the decoder must not produce silence. Recovery procedure:

```
1. Hold last known spectral shape from previous K-frame
2. Apply exponential decay: amplitude × exp(-t / τ)  where τ ≈ 60ms
3. Continue decay until next readable K-frame is found
4. Crossfade into newly decoded K-frame over one frame duration (~23ms)
```

The decay sounds like a note fading out — not a dropout. The crossfade sounds like a smear or blur — not a cut. Both are musically tolerable in the way that a vinyl skip is tolerable and a digital dropout is not.

**Layer dropout behavior:**

```
L2 unreadable:  α[L0+L1 : ] = 0.0   → highs roll off, bass and mids present
                                        sounds like low-pass filtered vinyl

L1 unreadable:  α[L0 : L0+L1] = 0.0 → melody and voice dimmed, bass survives
                                        sounds like very worn tape

L0 unreadable:  α[ : L0] = 0.0       → only playable with L1/L2 present
                                        rare — inner rings are most protected
```

**Variable-rate decode and pitch:**

At rotation speed r relative to nominal (r=1.0 at 33rpm):

- Frame consumption rate scales by r
- At r < 1.0 (slow scratch): frames are stretched by 1/r using phase vocoder
- At r > 1.0 (fast scratch): frames are compressed by 1/r
- At r = 0 (stopped disc): last frame repeats with increasing low-pass filtering over time
- At r < 0 (reverse scratch): frames consumed in reverse order, TDAC reconstruction identical

The phase vocoder preserves pitch relationships during speed change. Optionally, pitch correction can be omitted for "vinyl pitch" behavior (speed and pitch coupled, as on a real turntable).

### 5.7 Mode 2 Gradient Interface — Codec Requirements

Mode 2 (gradient dot encoding, planned for v2.0) changes the information density and physical read characteristics of the Digilog disc. This section defines what Mode 2 requires from the codec layer so that DSA is ready for it when it arrives.

**What gradient dots provide:**

In Mode 1, each dot encodes exactly 3 bits (8 discrete colors). Transitions between dots are hard edges with no information content.

In Mode 2, the transition zone between adjacent dots encodes additional data through controlled gradient blending. A dot pair with a smooth gradient transition from color A to color B carries more information than the same pair read as two discrete values.

At rest (static scan), gradient transitions are read with high precision — more information per dot than Mode 1. In motion (spinning disc), the motion blur averages adjacent gradients, which:
- Reduces per-dot precision
- But produces a naturally continuous confidence value (the blur IS a spatial average, not noise)
- Meaning: motion degrades gracefully in Mode 2, instead of producing misread discrete colors as in Mode 1

**Codec implications:**

The confidence-weighted reconstruction model (section 2.3) is designed for this. In Mode 2, the visual decoder produces α[b] as a continuous float derived from gradient clarity rather than a binary readable/unreadable flag.

The quantizer (Step 2) requires no changes for Mode 2. The step sizes and integer coefficients are encoding-mode-agnostic. The confidence weighting is applied at decode time only.

**What Mode 2 changes for the bitstream (Steps 5–6):**

The bitstream header must carry an encoding mode flag:

```
Mode byte:
  0x01 — Discrete (Mode 1, current)
  0x02 — Gradient (Mode 2, future)
```

The decoder reads this flag and initializes the visual decoder accordingly. The audio codec itself is identical — only the confidence vector generation changes between modes.

**What Mode 2 changes for information density:**

Gradient encoding is expected to increase usable data capacity per unit disc area by 40–60% over Mode 1 at the same print resolution. This means either:
- Higher audio quality at the same bitrate (more dot area available per audio bit)
- Longer audio duration at the same quality
- Or both, at higher print resolution

The exact capacity improvement depends on the gradient encoding scheme and is outside the scope of the audio codec specification. DSA's layered bitstream format accommodates any capacity increase through the existing layer structure — more bits available means better quantization across all three layers.

---

## 6. Evaluation Plan

### 6.1 Objective quality metrics

- **SNR** (Signal-to-Noise Ratio) — basic reconstruction quality
- **PEAQ** (Perceptual Evaluation of Audio Quality, ITU-R BS.1387) — perceptual quality
- **ViSQOL** (Virtual Speech Quality Objective Listener) — modern ML-based quality metric

### 6.2 Comparison baselines

| Codec | Bitrate | Metric | Notes |
|---|---|---|---|
| Opus | 6kbps | PEAQ | Current Digilog v0.1 codec |
| Opus | 12kbps | PEAQ | Primary comparison point |
| Codec2 | 3.2kbps | PEAQ | Lowest bitrate baseline |
| DSA | 6kbps | PEAQ | Target: competitive with Opus |
| DSA | 12kbps | PEAQ | Target: competitive with Opus |

### 6.3 Degradation testing

- Partial layer decode quality (L0 only, L0+L1, all layers)
- Reverse decode quality vs original reversed audio
- Corrupted frame recovery (simulate worn disc)
- Variable-rate decode quality (0.5x, 2x, 4x speed)

### 6.4 Physical media testing

- Print at various DPI and module sizes
- Scan with range of phone cameras
- Test under Digilog Rig (controlled lighting)
- Test during disc rotation at 33rpm, 45rpm
- Test with simulated DJ scratch

---

## 7. Design Decisions Log

This section records key design decisions and the reasoning behind them.

### 2026-04 — MDCT chosen over FFT

**Decision:** Use MDCT as the analysis transform.

**Alternatives considered:** FFT (Fast Fourier Transform).

**Reasoning:** FFT produces blocking artifacts at frame boundaries when used for audio coding, especially at low bitrates. MDCT with sine window and 50% overlap provides perfect reconstruction via TDAC — no blocking artifacts. All serious audio codecs (MP3, AAC, Vorbis, Opus) use MDCT for this reason. The sine window's symmetry property additionally supports DSA's reverse playback requirement.

### 2026-04 — B-frames for reverse playback

**Decision:** Use bidirectional B-frames (reference surrounding K-frames) rather than forward-only P-frames (reference previous frame only).

**Alternatives considered:** P-frames (simpler, used in Vorbis).

**Reasoning:** P-frames cannot be decoded in reverse because the reference frame (previous frame) is not available when reading backward. B-frames reference K-frames on both sides — which are always available regardless of direction. This enables vinyl-like reverse playback at the cost of slightly increased complexity and a small encoding delay (need to see the next K-frame before encoding B-frames).

### 2026-04 — GOP size of 8 frames

**Decision:** K-frame every 8 frames (~185ms at 23.2ms/frame).

**Alternatives considered:** 4 frames (~93ms), 16 frames (~371ms).

**Reasoning:** Scratch recovery time is bounded by the distance to the nearest K-frame. 185ms is acceptable for casual scratch use and approaching acceptable for performance use. 8 frames also aligns with 8 reference markers on the disc clock track — one K-frame per reference marker — enabling efficient position sync. 4 frames would improve scratch recovery but increases K-frame overhead significantly at low bitrates. 16 frames would be more efficient but scratch recovery of ~370ms is perceptually too long.

### 2026-04 — Confidence-weighted reconstruction over binary layer masking

**Decision:** Represent band readability as a continuous confidence value α ∈ [0.0, 1.0] applied multiplicatively to dequantized coefficients, rather than a binary readable/unreadable flag.

**Alternatives considered:** Binary layer masks (band present or absent), hard layer switching (full L0/L0+L1/full), silence on partial read.

**Reasoning:** Binary masks produce digital failure behavior — a band is either full quality or completely absent. This is the same catastrophic failure characteristic of CDs and MP3s. Continuous confidence values produce analog degradation behavior — reduced gain, spectral softening, musical continuity. This is the core aesthetic requirement of the Digilog format. The math is one multiply per coefficient. The perceptual result is the difference between a disc that sounds worn and one that sounds broken.

### 2026-04 — Frame loss recovery via spectral decay rather than silence

**Decision:** On K-frame loss, decay the last known spectral shape with exponential envelope (τ ≈ 60ms) rather than producing silence or repeating the frame.

**Alternatives considered:** Silence (simplest), frame repeat (freeze), error concealment from neighboring frames.

**Reasoning:** Silence is digital failure — a gap. Frame repeat is digital failure — a freeze. Exponential decay is analog failure — a fade. A vinyl stylus lifted from a groove produces a fade, not a cut. This behavior is perceptually consistent with the format's physical character and musically tolerable in live performance.

### 2026-04 — Mode 2 gradient confidence is forward-compatible, not a codec change

**Decision:** The confidence interface (α vector per frame) is defined as part of the core codec contract now, even though Mode 2 gradient encoding is not yet implemented. The audio codec is unchanged between modes — only the visual decoder's confidence vector generation changes.

**Alternatives considered:** Redesign the codec interface when Mode 2 arrives, treat Mode 2 as a separate codec.

**Reasoning:** If the confidence interface is added to the codec as an afterthought when Mode 2 arrives, it forces a breaking change to the bitstream format. Defining it now — with Mode 1 simply producing α = 1.0 everywhere — means Mode 2 is a visual encoder change, not an audio codec change. The bitstream format needs only a mode flag in the header. All other codec machinery is identical.

### 2026-04 — 48 perceptual bands

**Decision:** 48 bands: 8 + 16 + 24 across three layers.

**Alternatives considered:** 32 bands (16+8+8), 64 bands (16+24+24).

**Reasoning:** 48 bands provides sufficient frequency resolution across the audible range while keeping quantization complexity manageable. The 8/16/24 split across layers reflects the increasing frequency complexity of each range — bass needs fewer bands, high frequencies need more. The total of 48 maps cleanly to 6 bits per transition pair in the Digilog disc visual encoding.

---

## 8. References

1. Princen, J.P., Johnson, A.W., Bradley, A.B. (1987). "Subband/transform coding using filter bank designs based on time domain aliasing cancellation." *ICASSP 1987*.

2. Malvar, H.S. (1992). *Signal Processing with Lapped Transforms*. Artech House.

3. Brandenburg, K., Stoll, G. (1994). "ISO/MPEG-1 Audio: A Generic Standard for Coding of High-Quality Digital Audio." *Journal of the AES*.

4. Valin, J.M., et al. (2012). "Definition of the Opus Audio Codec." *RFC 6716*, IETF.

5. ISO (2003). "Normal equal-loudness-level contours." *ISO 226:2003*.

6. ITU-R (2001). "Method for objective measurements of perceived audio quality." *ITU-R BS.1387-1*.

7. Wiegand, T., et al. (2003). "Overview of the H.264/AVC Video Coding Standard." *IEEE Transactions on Circuits and Systems for Video Technology*.

8. Egger, O., et al. (2005). "MPEG-4 Scalable Lossless Coding of Audio Signals." *AES 118th Convention*.

9. Daala Video Codec. Mozilla Research / Xiph.org (2013–2017). https://xiph.org/daala/

10. Teichmann, S. (2026). "Digilog: A Free, Open Physical Audio Format." https://github.com/pisdronio/digilog-spec

11. Goodwin, M., Vetterli, M. (1999). "Matching pursuit and atomic signal models based on recursive filter banks." *IEEE Transactions on Signal Processing* — foundational work on confidence-weighted signal reconstruction.

12. Laroche, J., Dolson, M. (1999). "Improved phase vocoder time-scale modification of audio." *IEEE Transactions on Speech and Audio Processing* — phase vocoder algorithm for variable-rate decode.

13. Perceptual Audio Coder error concealment survey — ITU-T G.191 Software Tools Library, reference implementation of frame loss concealment strategies.

---

## 9. Acknowledgments

DSA is built as part of the Digilog project, conceived and initiated by Sidronio Teichmann in April 2026. The design was developed in collaboration with Claude (Anthropic) as an AI co-researcher and implementation partner.

The project draws on three decades of open audio codec research, particularly the work of the Xiph.org Foundation (Vorbis, FLAC, Opus) whose commitment to open standards and patent-free audio technology made this work possible.

---

---

## 12. Visual-Audio Interface — The Gradient Encoding Layer

### 12.1 Overview

The DSA bitstream bytes are not simply mapped to color pairs. The visual encoding layer translates DSA coefficient data into physical gradient properties on the disc. The gradient IS the data — not decoration.

Three coefficient properties map to three visual properties:

```
coefficient magnitude  →  gradient steepness
coefficient sign       →  gradient direction (left-to-right vs right-to-left)
band confidence        →  gradient blur amount (Mode 2 only)
```

This mapping makes the disc a direct visual representation of the music's spectral content. A musically dense passage — many large coefficients across all bands — produces a visually rich, colorfully active disc surface. A quiet passage produces sparse, near-solid regions. The disc is a score you can see.

### 12.2 Color pair assignment per layer

Each DSA layer uses color pairs chosen for maximum visual discriminability under its expected reading conditions:

**Layer 0 (bass, inner rings):**

High-contrast pairs only — Black↔White, Black↔Yellow, Black↔Cyan.

Largest dot size, must read under any conditions: worn print, cheap camera, hand partially covering disc. Contrast is maximized because these bands carry the fundamental frequencies — loss of Layer 0 readability means loss of the musical identity of the track.

**Layer 1 (mid, middle rings):**

Medium-contrast pairs — Red↔Cyan, Blue↔Yellow, Green↔Purple.

Readable on any modern phone camera under normal ambient conditions. These complementary pairs are chosen to survive JPEG compression and auto-white-balance adjustments that phone cameras apply before exposing pixel data to the app.

**Layer 2 (high freq, outer rings):**

Full 8-color palette available.

Requires Digilog Rig with controlled LED lighting and fixed focal distance. Smaller dots, higher density, maximum data capacity. Color accuracy at this precision requires controlled illumination — ambient light introduces enough color shift to corrupt high-frequency coefficient reads.

### 12.3 Gradient steepness ↔ coefficient magnitude

```
Large coefficient   →  near-hard edge
                       mostly Color A, small transition zone
                       visually: sharp color boundary

Medium coefficient  →  50/50 gradient
                       equal blend across transition zone
                       visually: smooth color fade

Small coefficient   →  very soft fade
                       mostly averaged color, wide transition
                       visually: gentle color shift

Zero coefficient    →  solid color, no transition
                       carrier dot only, no audio data
```

This means the disc is visually denser and more colorful in musically active sections (many large coefficients) and visually sparse in quiet sections. A kick drum transient appears as a burst of hard-edged high-contrast dots in the L0 inner ring. A sustained piano note appears as a wide, gradual color gradient in the L1 middle ring. Silence is a smooth sweep of solid color.

**The disc is a visual map of the music's energy.** An audiophile looking at a Digilog disc under magnification can identify verse, chorus, drop, and silence by the density and sharpness of the gradient patterns — without scanning it.

### 12.4 CRT bloom pre-emphasis model

The Digilog Rig uses a fixed LED ring at controlled distance. The camera lens introduces a known, consistent optical blur — predictable, not noise. At fixed focal length and aperture, the point spread function of the lens is a measurable constant.

The encoder uses pre-emphasis: print gradients slightly sharper than the target reading value, knowing the rig optics will soften them to the correct value at decode time.

This is directly analogous to vinyl pre-emphasis/de-emphasis (RIAA curve): the cutting engineer boosts treble knowing the playback cartridge will roll it back. The physical medium and the reading system are co-designed. The encoding and decoding are not inverse operations in isolation — they are inverse operations including the physics of the medium.

```
target_steepness = desired_gradient_value / rig_psf_factor

Pre-emphasis factor: to be calibrated experimentally with the reference
rig design. Initial estimate: 1.15× steepness boost for Layer 2 dots
under standard rig LED lighting at 15cm focal distance.
```

Layer 0 and Layer 1 dots are large enough that the lens PSF is negligible relative to dot size. Pre-emphasis applies primarily to Layer 2 (high-frequency, small, outer ring dots).

### 12.5 Mode 1 vs Mode 2 visual encoding

**Mode 1 (discrete):**

Hard-edged square modules. Gradient is purely aesthetic — smooth rendering of hard boundaries to reduce aliasing in the printed output. Confidence is always 1.0. Color classification: nearest-neighbor to 8 reference colors. The camera reads the center pixel of each module.

**Mode 2 (gradient):**

True continuous gradient between anchor colors. Gradient steepness encodes coefficient magnitude. Gradient direction encodes coefficient sign. The transition zone between adjacent modules carries information — it is not decorative.

Motion blur from the spinning disc is signal, not noise. As the disc rotates, the camera integrates across the transition zone naturally. The integration IS the read — a spatial average of the gradient, producing a continuous value between 0 and 1, which maps directly to the confidence vector α[b].

At rest: the camera can resolve fine gradient detail → high confidence, high precision read.
In motion: the blur averages the gradient → lower confidence, lower precision, but still a valid read. The audio degrades smoothly as speed increases, not catastrophically.

**The audio bitstream is IDENTICAL in Mode 1 and Mode 2.** Only the visual decoder changes. No breaking format change. Mode 2 support is a decoder feature flag in the file header, not a new codec version.

### 12.6 Analog degradation chain

The full chain from physical damage to acoustic output:

```
Physical condition        Visual effect               Acoustic effect
──────────────────────────────────────────────────────────────────────
Worn outer ring           Faded color gradients       Lost high frequencies
                                                      (gentle low-pass filter)
                                                      sounds like worn vinyl

Scratch across rings      Disrupted transitions       Brief filtered audio
                          in affected area            then recovery at K-frame
                                                      sounds like vinyl scratch

Hand covers disc          No transitions in           L2 drops first
(DJ performance)          covered region              L1 may drop partially
                                                      L0 inner ring still plays
                                                      sounds like heavy low-pass

Cheap camera              Color shift, blur,          Confidence < 1.0
                          reduced contrast            all coefficients attenuated
                                                      sounds like lo-fi cassette

Controlled rig light      Sharp, consistent           Full quality
                          color read                  confidence = 1.0
```

This chain — from physical surface condition to acoustic character — is the analog degradation model. It is implemented in two places:

1. The visual decoder (confidence calculation from gradient clarity)
2. The audio decoder (confidence-weighted dequantization + frame loss recovery)

The two sides of this interface are defined in section 2.3. The visual decoder outputs α[b] ∈ [0.0, 1.0]. The audio decoder applies it as `C̃[k] = q[k] × step[b] × α[b]`. The acoustic result at every point in the degradation table above is music — not silence, not glitch, not digital failure.

### 12.7 Variable rate playback — real-time pitch control

Rotation speed is measured in real time by the rig camera watching clock track dots pass the read window. This measurement IS the tempo — not encoded in the disc, derived from physics.

```
Normal speed (33rpm):    consume frames at nominal rate, 23.2ms/frame
Faster (scratch fwd):    consume frames faster — pitch rises naturally
Slower (scratch slow):   consume frames slower — pitch drops naturally
Stopped:                 hold last frame, decay to silence (τ ≈ 60ms)
Reverse:                 decode frames in reverse order
                         true reversed audio, not pitch-shifted
```

**For speeds within ±30% of nominal:** direct rate change, natural pitch shift. This is the vinyl feel — speed and pitch are coupled, as they are on a real turntable. No processing, no latency. The DJ moves the disc and the pitch responds immediately.

**For extreme speeds (>2× nominal):** phase vocoder time-stretch to prevent aliasing artifacts.

Recommended libraries:
- **RubberBand** (GPL) — highest quality pitch-aware time stretch
- **SoundTouch** (LGPL) — lower latency, preferred for real-time DJ use

**Latency target:** <50ms end-to-end from disc motion to audio output. For scratch performance, feel trumps fidelity. A phase vocoder that adds 100ms latency makes the instrument unplayable regardless of audio quality. SoundTouch at minimal buffer size is the default for the disc player; RubberBand is available as a quality mode for playback-only (non-scratch) use.

---

## 7. Design Decisions Log (continued)

### 2026-04 — Layer-specific color pairs for discriminability under conditions

**Decision:** Assign color pairs to layers based on the expected reading conditions for that layer's physical disc region, not a single uniform palette.

**Alternatives considered:** Single 8-color palette applied uniformly across all layers, random assignment, user-configurable palette.

**Reasoning:** Layer 0 (inner rings) must read under any conditions — worn print, cheap camera, ambient light, hand interference. Only high-contrast pairs (Black↔White, Black↔Yellow, Black↔Cyan) are reliably discriminable under all these conditions. Layer 2 (outer rings) requires controlled rig conditions regardless, so the full 8-color palette is available and maximizes data capacity. Mixing high-precision colors into Layer 0 would cause catastrophic read failures on cheap cameras — the exact scenario where Layer 0 must remain readable.

### 2026-04 — Pre-emphasis for rig optics co-design

**Decision:** The encoder applies pre-emphasis to Layer 2 gradient steepness (initial estimate: 1.15×) to compensate for the known optical blur of the Digilog Rig lens.

**Alternatives considered:** No pre-emphasis (accept the blur as signal loss), digital de-emphasis in the visual decoder (apply sharpening filter after read).

**Reasoning:** Digital sharpening after the read amplifies noise along with signal — it cannot recover information that the lens truly lost. Pre-emphasis at encode time compensates for a predictable, fixed, measurable physical property (the lens PSF at fixed focal length). This is the same principle as vinyl RIAA: the physical process and the encoding are co-designed, and neither is correct without the other. Pre-emphasis requires calibration with the reference rig hardware; the factor 1.15 is provisional and will be updated after physical testing.

### 2026-04 — Low latency beats quality for scratch performance

**Decision:** Default the disc player to SoundTouch (lower latency) rather than RubberBand (higher quality) for variable-rate playback during scratch.

**Alternatives considered:** RubberBand with reduced buffer size, custom real-time phase vocoder, no time-stretch (accept artifacts at extreme speeds).

**Reasoning:** A DJ scratches a record by feel. The feedback loop between hand motion and audio output must be below ~50ms to feel responsive. At 100ms latency the instrument becomes unplayable — the DJ hears what they did, not what they are doing. Audio quality degradation at 2× speed is perceptually acceptable (the audience hears it as scratch technique, not codec failure). Latency degradation at 100ms is not acceptable — it breaks the performance. Quality is a preference; latency is a physical constraint of live performance.

### 2026-04 — Mode 2 is a visual decoder change, not a codec version

**Decision:** The audio bitstream format is identical in Mode 1 and Mode 2. The mode flag lives in the file header. The audio codec is unaware of which visual mode was used to encode the disc.

**Alternatives considered:** Separate Mode 2 bitstream format with sub-integer coefficient precision, new DSA version number for gradient encoding.

**Reasoning:** Gradient dots provide a continuous confidence value, not sub-integer coefficient precision. The coefficient values are still integers — what changes is how confidently they were read, which is captured by α[b]. Since α is a decode-time input from the visual decoder (not encoded in the audio bitstream), the audio codec requires no changes. Treating Mode 2 as a new audio codec version would require re-encoding all existing discs and breaking all existing decoders. Treating it as a visual decoder feature means a Mode 1 decoder can play a Mode 2 disc at reduced quality (it will read gradient dots as discrete colors, losing sub-dot precision but still producing audio).

---

## 10. Open Problems and Future Work

### 10.1 Radial motion blur compensation

At a given RPM, tangential velocity increases with radius:
v = 2π × r × RPM/60

This means outer rings (L2, high frequency) experience greater
motion blur than inner rings (L0, bass) at the same rotation speed.

Current model treats confidence α as uniform per band at a given
speed. A more accurate model would compute per-layer confidence
degradation as a function of radius and RPM:

  α_L0(speed) > α_L1(speed) > α_L2(speed)

with the gap widening as speed increases.

This has implications for scratch performance: at high scratch
speeds, L2 degrades first and fastest — which is actually
perceptually correct (highs roll off before mids before bass)
but should be explicitly modeled rather than incidentally correct.

### 10.2 Optical pipeline simulation for encoder optimization

Current pre-emphasis (Section 12.4) uses a static factor (1.15×)
estimated for the reference rig. A simulation-based optimizer
would model the full optical pipeline:
- Gaussian blur (lens PSF at fixed focal length)
- Tangential motion blur (radius and RPM dependent, see 10.1)
- Downsampling (camera sensor resolution)
- Noise (sensor noise floor)

Then optimize gradient steepness and color pair assignments to
maximize decoded confidence after simulated degradation. This
would replace the static pre-emphasis estimate with a
calibrated, testable model.

---

*This document is a living research record. It will be updated as implementation progresses and will form the basis of a formal scientific publication.*

*github.com/pisdronio/dsa*
*Scan the groove.*
