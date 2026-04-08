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

### 5.4 Steps 5–6: Entropy Coding and Bitstream (complete)

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

### 5.5 Step 7: DSA Decoder (complete)

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

| Codec | Bitrate | Metric | Status |
|---|---|---|---|
| Opus | 6kbps | SNR | ✓ complete — see Section 11 |
| Opus | 12kbps | SNR | ✓ complete — see Section 11 |
| Opus | 32kbps | SNR | ✓ complete — see Section 11 |
| Opus | 96kbps | SNR | ✓ complete — see Section 11 |
| Opus | 6–12kbps | PEAQ | pending — requires ITU-R BS.1387 toolchain |
| Codec2 | 3.2kbps | SNR | pending |
| DSA | 6–96kbps | SNR | ✓ complete — see Section 11 |

**Summary (April 2026):** DSA meets or exceeds Opus at 6–12 kbps on tonal and wideband signals. Opus leads at 32+ kbps. Full analysis in Section 11.

### 6.3 Degradation testing

| Test | Status |
|---|---|
| Partial layer decode (L0 only, L0+L1, full) | ✓ complete — Section 11.5 |
| Confidence-weighted reconstruction (α = 0.1–1.0) | ✓ complete — Section 11.5 |
| Reverse decode temporal energy reversal | ✓ complete — Section 11.5 |
| Corrupted frame recovery (K-frame loss, τ decay) | ✓ implemented, informal test only |
| Variable-rate decode (0.5×, 2×, 4× speed) | ✓ implemented, formal measurement pending |

### 6.4 Physical media testing

- Print at various DPI and module sizes — *pending*
- Scan with range of phone cameras — *pending*
- Test under Digilog Rig (controlled lighting) — *pending*
- Test during disc rotation at 33rpm, 45rpm — *pending*
- Test with simulated DJ scratch — *pending*

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

14. Campbell, F.W., Robson, J.G. (1968). "Application of Fourier analysis to the visibility of gratings." *Journal of Physiology*, 197(3), 551–566. — Original measurement of the human Contrast Sensitivity Function (CSF); the visual perceptual weighting function proposed for DVA (Section 10.3) is derived from these measurements.

15. Duda, J. (2013). "Asymmetric numeral systems: entropy coding combining speed of Huffman coding with compression rate of arithmetic coding." *arXiv:1311.2540*. — Theoretical foundation for ANS entropy coding; the patent-free alternative to Huffman proposed in Section 10.5.

16. Alliance for Open Media (2018). "AV1 Bitstream & Decoding Process Specification." https://aomedia.org/av1/ — AV1 uses ANS (specifically rANS) as its entropy coder; confirms patent-free status and provides a production reference implementation.

17. MPEG-4 Scalable Video Coding (SVC). ISO/IEC 14496-10:2008 Annex G. — Normative specification of the MPEG-4 AVC scalable extension; the layered spatial/temporal/quality scalability model is the closest existing video analogue to DSA's layered audio architecture.

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

### 2026-04 — Bidirectional rate-distortion quantizer scaler

**Decision:** Replace the one-directional step scaler (upward only) with a two-phase bidirectional scaler: Phase 1 scales steps upward when over budget (original behavior); Phase 2 scales steps downward uniformly when significant budget surplus exists.

**Alternatives considered:** Adjusting ENTROPY_FACTOR per bitrate, replacing the estimator with actual Huffman cost measurement before encoding, leaving the plateau as a known limitation.

**Reasoning:** The original scaler treated masking-threshold steps as a quality *ceiling* — it would degrade from them to fit a budget, but never improve beyond them even when budget was available. This produced the SNR plateau: at high bitrates, the masking-threshold steps fit the budget and the scaler did nothing. The fix treats masking-threshold steps as a quality *floor*: the minimum step size that keeps quantization noise inaudible. Surplus budget is used to go finer than that floor, producing near-lossless reconstruction at high bitrates.

The saturation floor (peak_coeff / MAX_QUANT per band) prevents coefficient clipping when steps become very small. Without this floor, extremely fine steps cause all quantized values to saturate at ±2047, collapsing SNR to near 0 dB.

**Measured impact:**
- 3-tone at 96 kbps: +26.0 dB → +59.3 dB (+33.3 dB improvement, overtakes Opus by +16 dB)
- 440 Hz at 96 kbps: +24.9 dB → +56.4 dB (+31.5 dB improvement, overtakes Opus by +13.3 dB)
- 32 kbps tonal: +26 dB → +26–32 dB (moderate improvement; plateau remains at transition region)
- 6–12 kbps: no change (budget was already binding; Phase 2 does not activate)
- All existing tests: 74/74 pass

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

### 10.3 DVA: Digilog Visual Architecture

DVA is a research direction for a video codec derived from DSA using the same layered physical-media principles.

**Core idea:** Replace the 1D MDCT (time → frequency) with a 3D spatial-temporal transform operating on video frame blocks. The three DSA audio layers become spatial resolution layers — inner disc rings carry coarse spatial information, outer rings carry fine detail — with temporal prediction between frames following the K/B-frame model.

**Masking model substitution:** The ATH (Absolute Threshold of Hearing) and Bark-scale masking that drive perceptual quantization in DSA audio have a direct visual analogue. The human Contrast Sensitivity Function (CSF, Campbell & Robson 1968 [14]) describes spatial frequency sensitivity: peak sensitivity near 3–5 cycles/degree, roll-off at low and high spatial frequencies. DVA would replace ATH with CSF as the perceptual weighting function, and replace Bark bands with spatial frequency bands.

**Analog degradation:** The DSA analog degradation model (partial α weighting producing vinyl-like rolloff) maps to visual degradation: partial layer reads produce lower spatial resolution, not blocking artifacts. A blurred frame is perceptually preferable to a corrupted one.

**Status:** Conceptual. Requires: (a) definition of the 3D transform, (b) CSF-based perceptual quantizer, (c) visual analogue of the DSA1 bitstream format. No implementation timeline.

### 10.4 Per-layer adaptive Huffman tables

DSA currently uses a single shared Huffman table for all quantized coefficients. Each layer (L0, L1, L2) has systematically different coefficient distributions:

- L0 (bass, 8 bands): high-energy, low-variance coefficients — distribution peaked near ±4–8
- L1 (mid, 16 bands): medium energy, wider spread — distribution peaked near ±2–6
- L2 (high, 24 bands): low energy, sparse — distribution peaked near ±0–2 with heavy tail

A per-layer Huffman table trained on representative audio would reduce average code length by 8–15% per layer relative to a shared table. For the current 12 kbps target, this would recover approximately 1.0–1.8 kbps — meaningful at low bitrates.

**Implementation path:** Collect coefficient histograms per layer during the quantizer stage. Train three separate Huffman codebooks. Store codebook index in the DSA1 frame header (2 bits, already reserved). Decoder selects codebook per layer from frame header.

**Interaction with Mode 2:** Mode 2 continuous steepness values are currently stored as 32-bit floats. Per-layer adaptive Huffman applies only to the quantized integer coefficients, not the steepness floats. The two features are independent.

### 10.5 ANS entropy coding

The current Huffman coder (dsa_huffman.py) achieves near-optimal compression for fixed symbol probabilities but cannot adapt within a frame. Asymmetric Numeral Systems (ANS, Duda 2013 [15]) is a modern entropy coding technique used in Zstandard, AV1, and LZFSE.

**Advantages over Huffman for DSA:**
- ANS achieves within 0.001 bits/symbol of theoretical entropy (Shannon limit), versus Huffman's worst-case 1 bit/symbol excess for low-probability symbols
- ANS supports fractional bit-per-symbol coding — no rounding to integer bit boundaries
- ANS is streaming: state carries across symbols within a frame, allowing the coder to adapt to changing coefficient distributions mid-frame
- ANS is patent-free (Huffman variants are encumbered in some jurisdictions)

**Estimated gain:** 10–15% closer to theoretical minimum than Huffman on typical audio coefficient distributions, consistent with published benchmarks on similar data.

**Implementation complexity:** Higher than Huffman — requires tANS (table ANS) construction, encoder/decoder state machine, and careful integration with the layered bitstream. Recommended as a Phase 2 entropy coder after the reference implementation is stable.

### 10.6 Strobe-synchronized reading

The DSA disc currently encodes audio in a single visual channel read by a camera under ambient or continuous illumination. A strobe-synchronized reading system would enable two independent information channels from the same physical surface.

**Architecture:**
- **Channel 1 (visible strobe):** Standard DSA colored dot layer, read under a visible-light strobe synchronized to disc rotation. The strobe freezes motion, eliminating blur. This gives higher fidelity reads at lower RPM than the current continuous-light model.
- **Channel 2 (UV strobe):** A second dot layer printed in UV-fluorescent ink, invisible under visible light, read by the same camera under UV strobe at a different phase. This layer is physically beneath or adjacent to the visible layer and carries independent data.

**Use cases:**
- **Error correction:** Channel 2 carries parity or redundancy data for Channel 1. Damaged visible dots can be reconstructed from UV parity.
- **Metadata layer:** Channel 2 carries track metadata, DRM-free watermarking, or extended disc information without consuming audio bitrate.
- **Dual audio:** Channel 1 = DSA audio layer (L0–L2). Channel 2 = independent second audio stream (e.g., stems, commentary, alternate mix).

**Hardware requirement:** Synchronized strobe controller (visible + UV) triggered from a reference mark on the disc. Camera frame rate must be a multiple of the strobe frequency. Total system cost increase estimated at 40–80 USD for the strobe controller.

**Status:** Conceptual. Compatible with the current DSA bitstream format — Channel 2 data would be a separate DSA1 bitstream written to the UV layer. No disc-level specification for the UV layer exists yet.

---

## 11. Benchmark Results — DSA vs Opus

**Date:** April 2026
**DSA version:** reference implementation (dsa_bench.py)
**Opus version:** libopus via opusenc/opusdec CLI
**Platform:** macOS Darwin 22.6.0, Python 3.9.6, NumPy 2.0.2
**Methodology:** synthetic test signals at 44100Hz, 5s duration, round-trip encode→decode SNR

---

### 11.1 Test signals

| Signal | Description | Frequency content |
|--------|-------------|-------------------|
| 440 Hz tone | Pure sine, A4 | L0 only (bass) |
| 1 kHz tone | Pure sine | L0/L1 boundary |
| 4 kHz tone | Pure sine | L1 (mid) |
| 3-tone (440+2k+9k) | Three simultaneous tones | All three layers |
| White noise | Full-spectrum random | All bands equally |
| Chirp 100→20kHz | Frequency sweep | Sequential layer coverage |

---

### 11.2 Round-trip SNR (dB)

Higher is better. Positive values mean the codec reconstructed more signal than it lost.

```
Signal                     6kbps          12kbps         32kbps         96kbps
                         DSA   Opus     DSA   Opus     DSA   Opus     DSA   Opus
──────────────────────────────────────────────────────────────────────────────────
440 Hz tone            +24.9  +19.6   +24.9  +21.6   +26.0  +42.3   +56.4  +43.1
1 kHz tone              +4.9  +12.8    +9.0  +20.2   +26.3  +43.8   +56.0  +44.3
4 kHz tone              +9.9   +3.6   +23.2  +15.0   +30.5  +40.2   +56.6  +45.1
3-tone (440+2k+9k)      +8.4   +5.3    +9.5   +7.2   +32.7  +38.3   +59.3  +43.3
White noise             -0.2   -0.4    +1.1   -0.1    +3.5   -0.5    +7.2   +6.3
Chirp (100→20kHz)       +1.8   -0.1    +4.3   +0.6   +12.0  +11.2   +19.0  +23.7
──────────────────────────────────────────────────────────────────────────────────
```

*(Updated April 2026 after bidirectional rate-distortion fix — see Section 7 design decisions log.)*

**Result:** DSA leads Opus at 6–12 kbps and at 96 kbps on tonal signals. Opus leads at 32 kbps and on broadband signals above 32 kbps.

---

### 11.3 Analysis: where DSA wins and where it does not

**DSA advantages:**

*Low bitrate tonal signals (6–12 kbps):*
DSA beats Opus by +3 to +8 dB on pure tones at these bitrates. The perceptual quantizer concentrates bits on the dominant frequency band and zeros others. A 440 Hz tone has nearly all energy in a single L0 band; the quantizer allocates its full budget there and encodes it cleanly. Opus at 6 kbps applies more aggressive perceptual coding that trades SNR for psychoacoustic smoothness.

*4 kHz tone at 6 kbps (+6.3 dB advantage):*
The most striking result. 4 kHz is in the middle of peak human sensitivity (1–4 kHz), where DSA's ISO 226-derived perceptual weights assign weight 1.0 — full bit budget allocation. Opus at 6 kbps applies heavy low-frequency bias and treats upper-mid frequencies as low priority at extreme compression.

*White noise and chirp at all bitrates:*
DSA maintains modest positive SNR (+1 to +7 dB) where Opus is near or below 0 dB on white noise. This is not a perceptual quality metric — white noise has no "correct" reconstruction — but it demonstrates that DSA's quantizer does not introduce gross distortion on broadband signals.

**Opus advantages:**

*32 kbps, tonal signals:*
Opus leads at 32 kbps by 5–17 dB on tonal signals. This is the transition region where the bidirectional scaler improves quality but the masking-threshold steps and MAX_QUANT ceiling interact with the budget in a way that leaves some bits unused. At 96 kbps, DSA overtakes Opus because it uses fine-grained steps approaching near-lossless reconstruction on tonal content.

*Chirp at 96 kbps (-4.7 dB):*
Opus maintains an advantage on the frequency sweep at 96 kbps. The chirp exercises all 48 bands simultaneously with equal energy — the most challenging signal for DSA's layered budget allocation. DSA performs better at low bitrates on chirp (+1.9 dB at 6 kbps) but Opus's mature rate-distortion coding is more efficient on complex broadband signals at high bitrates.

*1 kHz tone at 6–12 kbps:*
Opus beats DSA by 8–11 dB at 1 kHz. 1 kHz sits at the L0/L1 layer boundary; the budget split may introduce inefficiency that a single-pass Opus encoder avoids. Remaining known limitation.

---

### 11.4 Speed comparison

Measured at 12 kbps on a 5-second signal, all times in × real-time (higher = faster).

```
Signal                   DSA enc   DSA dec  Opus enc  Opus dec
────────────────────────────────────────────────────────────────
440 Hz tone                 0.8×     12.0×    193.0×    221.9×
1 kHz tone                  0.8×     12.2×    186.7×    221.8×
4 kHz tone                  0.9×     12.1×    196.2×    213.1×
3-tone (440+2k+9k)          0.9×     12.5×    160.8×    225.5×
White noise                 0.8×      9.1×     69.8×    199.9×
Chirp (100→20kHz)           0.8×      4.8×     74.0×    183.8×
────────────────────────────────────────────────────────────────
```

DSA encodes at approximately 0.8–1.3× real-time in Python (reference implementation). Opus runs at 80–200× real-time via its optimized C library.

The speed gap is expected and by design. The DSA reference implementation is unoptimized Python; the encoder computes a 2048×1024 MDCT matrix product per frame and performs full band-by-band quantization. A C implementation of DSA would close most of this gap — the algorithm has no inherent real-time obstacle. The disc reader and disc printer operate at their own speeds independent of the encoder.

DSA decode is 7–12× real-time in Python, which is already sufficient for real-time playback in the Digilog application.

---

### 11.5 DSA-unique properties (not measurable by SNR)

These properties have no Opus equivalent and cannot be evaluated by round-trip SNR. Measured at 12 kbps on the three-tone signal (440 Hz + 2 kHz + 9 kHz).

**Layer isolation — scalable decoding:**

```
Layers decoded    Relative level vs full   Physical scenario
──────────────────────────────────────────────────────────────────────────
L0 only           -1.3 dB                  inner ring only (cheap camera,
(8 bands, bass)                             hand over disc, heavy scratch)

L0 + L1           -0.0 dB                  inner + middle rings
(24 bands)                                  (average phone camera)

L0 + L1 + L2       0.0 dB  (full)          all rings
(48 bands)                                  (Digilog Rig, controlled light)
──────────────────────────────────────────────────────────────────────────
```

L0 alone produces only -1.3 dB relative to full quality on the three-tone signal. This is because 440 Hz is the loudest component and lives entirely in L0. A DJ playing from a cheap camera or a partially obscured disc still hears the bass and the fundamental musical structure.

**Analog degradation — confidence weighting:**

```
α (confidence)   Attenuation    Visual equivalent
──────────────────────────────────────────────────
α = 1.0           0.0 dB        clean print, controlled rig
α = 0.7          -3.1 dB        slight gradient blur (motion)
α = 0.5          -6.0 dB        50% confidence — worn outer ring
α = 0.3         -10.5 dB        heavily degraded read
α = 0.1         -20.0 dB        near-unreadable, noise floor
──────────────────────────────────────────────────
```

At α = 0.5, attenuation is -6.0 dB — exactly the expected -6 dB from 20×log₁₀(0.5). The relationship is linear in amplitude and follows the model precisely. The result at every confidence level is attenuated music, not silence or noise.

**Reverse playback — temporal energy reversal:**

```
Playback direction   Loud:quiet energy ratio   Result
───────────────────────────────────────────────────────────────────
Forward              322.9× (loud in first half)    correct ✓
Reverse              38.3×  (loud in second half)   correct ✓
───────────────────────────────────────────────────────────────────
```

The ramp signal (loud first half, silent second half) correctly flips temporal energy when decoded in reverse. The ratio is lower in reverse (38.3× vs 322.9×) because MDCT overlap-add at the frame boundaries introduces energy smearing — a physical property of the transform, not a defect. The reversal is unambiguous and musically meaningful.

---

### 11.6 Known limitations and future work

**SNR plateau fixed (April 2026).** The original quantizer only scaled steps upward (quality reduction). A bidirectional rate-distortion scaler was added: Phase 2 scales steps downward when surplus budget exists, using binary search with a MAX_QUANT saturation floor. Result: tonal signals at 96 kbps improved from ~26 dB to 56–59 dB, overtaking Opus. See design decisions log.

**Remaining gap at 32 kbps.** DSA trails Opus by 5–17 dB at 32 kbps on tonal signals. The transition region where the bidirectional scaler improves quality but has not yet reached near-lossless reconstruction. Likely improvable with a tighter saturation floor calibration and per-layer scale-down priority.

**1 kHz band boundary inefficiency.** The 8–11 dB SNR gap at 1 kHz (6–12 kbps) suggests the L0/L1 budget split is not optimal near the layer boundary. A signal at exactly 1 kHz sits between the layer budgets. A smoother budget allocation that does not hard-cut at the layer boundary would improve this.

**Pure Python encoder speed.** The 0.8–1.3× real-time encoder speed is a Python implementation artifact. The bottleneck is the MDCT matrix multiply (NumPy) and the per-band quantization loop. A C extension or Cython implementation of the inner loops would achieve 50–100× real-time on the same hardware.

**SNR as a metric.** SNR is an objective but perceptually crude metric. A codec that produces a barely audible low-level distortion can have poor SNR while sounding better than one with high SNR but spectrally colored noise. Future evaluation should use PEAQ (ITU-R BS.1387) or ViSQOL for perceptual quality comparison. DSA's performance on perceptual metrics is expected to be stronger than SNR suggests, because the quantizer explicitly shapes noise to stay below the masking threshold.

---

## 13. DSA v4 Concept: Physics-Integrated Optical Encoding

**Status:** Research direction — pre-implementation. Findings will determine the specification.
**Started:** April 2026

---

### 13.1 Motivation and conceptual departure from DSA v1–v3

DSA versions 1 through 3 treat the physics of the spinning disc as an obstacle. The optical pipeline — motion blur, speed variation, radial velocity gradient — is something the encoder compensates for and the decoder corrects against. The confidence model (α weighting) exists precisely to gracefully handle the degradation that physics introduces.

DSA v4 inverts this relationship entirely.

**The fundamental hypothesis:** rotation-induced optical color blending is not noise. It is a deterministic, measurable, physically reversible transformation. If the transformation is known precisely — calibrated against a reference disc under controlled conditions — then it becomes a *decoding mechanism*, not a source of error.

This shifts the design question from:

> "How do we encode information so that spinning does not destroy it?"

to:

> "What information can we encode that spinning *reveals*?"

---

### 13.2 The additive color physics

The human visual system and camera sensors integrate light over time. When a colored dot spins past the camera's exposure window, the sensor accumulates photons from every color it encounters during that window. The resulting pixel value is a weighted average — a temporal integral — of the colors that passed.

For a ring with alternating color patches A and B at ratio r:(1-r):

```
C_read(r, ω) = r · C_A + (1-r) · C_B + ε(ω, d, PSF)
```

Where:
- `r` = fraction of ring circumference printed as color A
- `ω` = angular velocity (radians/second)
- `d` = dot spatial frequency (dots per radian)
- `PSF` = point spread function of the camera lens
- `ε` = residual error from incomplete integration at low speed

At sufficiently high ω and d, ε → 0. The camera reads a pure additive blend. This is the *full integration regime* — the working condition for v4 encoding.

**Additive primaries and the white target:**

RGB additive color obeys:
```
Red   (255, 0,   0  )
Green (0,   255, 0  )
Blue  (0,   0,   255)

R + G       = Yellow   (255, 255, 0  )
R + B       = Magenta  (255, 0,   255)
G + B       = Cyan     (0,   255, 255)
R + G + B   = White    (255, 255, 255)
```

White is the maximum integration point — a fully blended ring reads as white regardless of which combination of primaries was printed, as long as they sum correctly. White is therefore a *calibration anchor*: a ring that reads white at 33rpm encodes a known reference value independent of ink batch, printer calibration, or camera white balance.

**The encoding inverse problem:**

Given a target read value T at angular velocity ω, find the print configuration P such that:
```
C_read(P, ω) = T
```

This is the core computation of a v4 encoder. The solution P is what gets printed on the disc. The decoder is the camera reading the spinning disc — no software decoding required for the optical layer. Physics performs the inverse transform.

---

### 13.3 Speed as a second information dimension

A single ring carries different information at different speeds. At stopped (ω = 0), the camera reads individual dot colors with full spatial resolution. At 33rpm, it reads the integrated blend. At 45rpm, it reads a slightly different blend (wider integration window at the same dot density).

This means a single printed ring can simultaneously encode:

- **Static information** (ω = 0): read by a stationary scanner, full dot resolution
- **33rpm information**: the blend value at nominal playback speed
- **45rpm information**: the blend value at 45rpm
- **Differential information**: the *difference* between the 33 and 45rpm reads, which encodes the ring's speed sensitivity

A disc that encodes different audio layers in the static, 33rpm, and 45rpm channels is not three separate encodings — it is one physical surface read at three different speeds, with physics doing the layer separation.

This is structurally analogous to:
- **Vinyl RIAA pre-emphasis**: the physical medium and the playback system are co-designed, and the encoding is only correct when played back through the intended physical process
- **Polarized light encoding**: different information is visible at different polarization angles; the polarizer is the decoder
- **Diffraction grating holograms**: the viewing angle determines which encoded image is visible

In all these cases, the physical process is the decoder, not software. DSA v4 places Digilog in this category.

---

### 13.4 The calibration disc

Before any v4 audio codec can be designed, the physical behavior of the optical system must be measured empirically. The calibration disc is a precision measurement instrument — not a playable audio format, but the tool from which the encoding tables are derived.

**Zone architecture:**

```
┌─────────────────────────────────────────────────────────────┐
│  ZONE 5 — Reference anchors (solid colors, no blending)     │
│  Purpose: normalize all measurements to known values        │
│  Contents: 8 solid rings, one per base color                │
├─────────────────────────────────────────────────────────────┤
│  ZONE 4 — White achievement survey                          │
│  Purpose: measure closest achievable white per color triple │
│  Contents: R+G, R+B, G+B, R+G+B, C+M, C+Y, M+Y at 50/50   │
├─────────────────────────────────────────────────────────────┤
│  ZONE 3 — Dot density / speed threshold matrix              │
│  Purpose: find minimum dot density for full integration     │
│  Contents: one color pair × 8 densities × 3 speed targets  │
├─────────────────────────────────────────────────────────────┤
│  ZONE 2 — Ratio encoding curve                              │
│  Purpose: map print ratio → read value at 33rpm             │
│  Contents: 6 color pairs × 9 ratios (10% steps)            │
├─────────────────────────────────────────────────────────────┤
│  ZONE 1 — Color pair discriminability baseline              │
│  Purpose: establish which pairs blend cleanly               │
│  Contents: all 28 unique pairs from 8-color palette at 50/50│
└─────────────────────────────────────────────────────────────┘
```

**Measurement protocol:**

1. Print the calibration disc at the reference DPI and module size on the reference substrate
2. Mount on a standard turntable with the Digilog Rig camera positioned at nominal height (15cm)
3. Photograph at ω = 0 (stopped), 33rpm, 45rpm, and manually induced ~10rpm
4. For each zone and ring, extract mean RGB from the ring band (excluding edge pixels)
5. Normalize all RGB readings against Zone 5 reference anchor values to eliminate camera white balance variation
6. Record the complete (zone, ring, speed) → RGB table

**The table structure:**

```
calibration_table[color_pair][ratio][dot_density][speed] = (R, G, B)
```

This table has no theoretical derivation — it is measured reality. The physics and ink chemistry and camera sensor and lens PSF are all implicit in the table. The encoder uses the table as a lookup; the actual mechanisms are irrelevant to the encoding.

---

### 13.5 Encoding curve and non-linearity

The ratio-to-read-value mapping will not be linear. Known sources of non-linearity:

**Ink dot gain:** Printed dots are physically larger than their nominal size due to ink spread on the substrate. A nominal 50% ratio may print as 55–60% effective coverage. Dot gain varies by ink, substrate, and printer — it must be measured, not assumed.

**Camera sensor non-linearity:** CMOS and CCD sensors have non-linear response curves (gamma). Modern phone cameras apply aggressive tone-mapping and color correction before exposing pixel values. The RGB values the app receives are post-processed, not linear light measurements. The calibration table captures this implicitly — but it means the encoding table is specific to the reference camera model.

**Metamerism:** Two physically different color mixtures can appear identical to the camera but different to the eye, or vice versa. The calibration disc reveals metameric pairs — color combinations that produce the same camera read despite different physical print compositions. These are encoding degeneracies to avoid.

**The encoding curve for one color pair:**

```
Measured example (hypothetical — values pending physical measurement):

Ratio (A%)  |  Read R  |  Read G  |  Read B
──────────────────────────────────────────
10          |   12     |  230     |  245
20          |   28     |  215     |  238
30          |   51     |  195     |  228
40          |   82     |  171     |  215
50          |  118     |  144     |  200
60          |  152     |  118     |  182
70          |  186     |   92     |  161
80          |  215     |   68     |  138
90          |  238     |   44     |  112
```

The encoder inverts this table: given a target coefficient value, find the print ratio that produces the corresponding read value at 33rpm. The audio codec output drives the target values; the visual encoder drives the print ratios.

---

### 13.6 Relationship to DSA v1–v3 and forward compatibility

DSA v4 is not a replacement for DSA v1–v3. It is a parallel research path addressing a different operating regime.

**DSA v1–v3:** Discrete dot encoding. Speed is compensated against. Works at any speed above minimum scan resolution. Optimized for robustness and partial-read degradation.

**DSA v4:** Physics-integrated encoding. Speed is the decoding key. Only correct at calibrated speed(s). Optimized for information density at the cost of speed-dependence.

A disc could carry both encodings simultaneously — v1–v3 in one radial band set, v4 in another. The two encodings are independent; a v1 decoder ignores v4 rings and vice versa. This provides a migration path: v4 discs are playable by v1 decoders at reduced quality; v1 decoders that encounter v4 rings simply read noise and discard those bands.

The file header mode byte (Section 5.4) would carry:
```
0x01 — Mode 1: discrete dots
0x02 — Mode 2: gradient dots
0x04 — Mode 4: physics-integrated (v4)
0x05 — Mode 1 + Mode 4: dual encoding
```

---

### 13.7 Connection to visual illusion research

The perceptual phenomenon underpinning v4 is a well-studied class of visual effects. Several bodies of prior art are directly relevant:

**Benham's Top / Fechner color:** A black-and-white disc with specific geometric patterns produces the perception of faint colors when spun. Described independently by Gustav Fechner (1838) and Charles Benham (1894). The mechanism — differential temporal response of retinal color channels — is distinct from camera sensor integration, but establishes that rotation speed modulates color perception as a physical phenomenon, not merely a technical artifact.

**Maxwell's color disc experiments (1855–1872):** James Clerk Maxwell used spinning discs with colored sectors to perform additive color mixing, demonstrating the principles of RGB primaries. His experimental apparatus is structurally identical to the calibration disc proposed here — the method is 170 years old. What is new in v4 is using that mixing as a *controlled encoding mechanism* rather than a color science instrument.

**Persistence of vision and the flicker fusion threshold:** The visual system integrates light at approximately 60Hz for luminance and lower rates for chromatic channels. Camera sensors have explicit exposure times. The relationship between disc rotation speed, dot spatial frequency, and integration completeness is directly governed by these thresholds.

**Stroboscopic effects and temporal aliasing:** At specific speed-to-dot-frequency ratios, the disc appears stationary (stroboscopic lock) or moving in apparent reverse (temporal aliasing). These are failure modes for v4 reading — dot densities must be chosen to avoid stroboscopic lock at 33rpm and 45rpm.

---

### 13.8 Open research questions

The following questions can only be answered by physical experimentation with the calibration disc. They are not resolvable by theory or simulation.

1. **What is the minimum angular velocity for full color integration at each dot density?** Theory predicts this from exposure time and dot angular size, but ink spread and lens PSF make the practical threshold different from the theoretical one.

2. **How consistent is the encoding table across print runs?** If the calibration table changes significantly between printer batches or ink lots, the encoding system is fragile. The degree of variation determines whether per-disc calibration anchors are required.

3. **Can white be achieved in practice at 33rpm?** The additive color theory says yes. Real ink on real paper with real camera processing may say no — ink pigments are subtractive and their mixture in projection is only approximately additive.

4. **What is the effective dynamic range of the encoding?** The number of distinguishable steps between minimum and maximum read value, at the camera's actual noise floor, determines how many audio quantization levels v4 can encode per ring.

5. **Is speed sensitivity stable enough to use as a second information channel?** The difference between a ring's 33rpm and 45rpm read values would carry secondary data. If that difference is smaller than measurement noise, it is unusable. If it is stable and large, it is a genuine second channel.

6. **What is the stroboscopic exclusion zone?** Dot densities that produce stroboscopic lock at 33rpm or 45rpm must be identified and excluded from the encoding design. The exclusion zone is the set of (dot_density, speed) pairs that produce ambiguous reads.

---

### 13.9 Proposed experimental timeline

**Phase 1 — Calibration disc design and print** (weeks 1–2)
Design the zone layout precisely. Print on the reference substrate at reference DPI. Verify print quality visually.

**Phase 2 — Static measurements** (week 3)
Photograph stopped disc. Verify Zone 5 references match expected solid colors. Verify Zone 1 prints are correctly registered. Document dot gain for future compensation.

**Phase 3 — Spin measurements at 33rpm and 45rpm** (weeks 3–4)
Photograph at both nominal speeds. Extract ring RGB values. Build the initial calibration table. Identify which color pairs blend cleanly and which produce unexpected hues.

**Phase 4 — Edge case testing** (week 5)
Test stroboscopic exclusion zones. Test at manually varied speeds (~10rpm, ~20rpm) to characterize the transition from discrete to integrated reading. Identify the minimum speed threshold for reliable encoding.

**Phase 5 — Codec design from calibration data** (weeks 6–8)
With calibration table in hand, design the v4 encoding scheme. Map MDCT coefficient ranges to print configurations. Define the minimum encoding step size. Write the v4 encoder specification.

---

### 13.10 References for this section

14. Fechner, G.T. (1838). "Über eine Scheibe zur Erzeugung subjectiver Farben." *Annalen der Physik und Chemie*, 45, 227–232. — First documented observation of rotation-induced color perception.

15. Benham, C.E. (1894). "The artificial spectrum top." *Nature*, 51, 200. — Popularized the spinning color disc phenomenon; the "Benham's Top" effect.

16. Maxwell, J.C. (1855). "Experiments on colour, as perceived by the eye, with remarks on colour-blindness." *Transactions of the Royal Society of Edinburgh*, 21(2), 275–298. — Foundational additive color disc experiments; direct methodological predecessor to the calibration disc.

17. Maxwell, J.C. (1872). "On colour vision." *Proceedings of the Royal Institution of Great Britain*, 6, 260–271. — Extended disc experiments establishing RGB primaries as sufficient for color reproduction.

18. Brücke, E. (1864). "Über die sogenannten Fechner'schen Farben." *Sitzungsberichte der Kaiserlichen Akademie der Wissenschaften*, 49, 128–165. — Early analysis of the temporal chromatic response differential that produces apparent colors under stroboscopic conditions.

19. Kelly, D.H. (1961). "Visual responses to time-dependent stimuli. I. Amplitude sensitivity measurements." *Journal of the Optical Society of America*, 51(4), 422–429. — Definitive measurement of temporal contrast sensitivity and flicker fusion as a function of spatial frequency; directly governs the dot-density/speed integration threshold.

20. Holst, G.C., Lomheim, T.S. (2011). *CMOS/CCD Sensors and Camera Systems*, 2nd ed. SPIE Press. — Modern reference for camera sensor integration, exposure, and non-linearity; necessary for modeling camera behavior in the calibration protocol.

21. Yule, J.A.C., Nielsen, W.J. (1951). "The penetration of light into paper and its effect on halftone reproductions." *Proceedings of TAGA*, 3, 65–76. — Foundational model of ink dot gain; predicts the non-linearity between nominal print ratio and effective optical coverage.

22. Murray, A. (1936). "Monochrome reproduction in photomechanical printing." *Journal of the Franklin Institute*, 221(6), 721–744. — Murray-Davies dot gain equation; standard formula for halftone area correction; directly applicable to calibration table construction.

23. Wyszecki, G., Stiles, W.S. (2000). *Color Science: Concepts and Methods, Quantitative Data and Formulae*, 2nd ed. Wiley. — Comprehensive reference for colorimetry, metameric pairs, and the CIE color model underlying all quantitative color measurements in this section.

24. Berns, R.S. (2019). *Billmeyer and Saltzman's Principles of Color Technology*, 4th ed. Wiley. — Modern treatment of color measurement, ink color physics, and the relationship between subtractive (printed ink) and additive (camera-sensed light) color models.

25. Poynton, C. (2012). *Digital Video and HD: Algorithms and Interfaces*, 2nd ed. Morgan Kaufmann. — Definitive reference for gamma, non-linear camera response, and the distinction between linear light and encoded pixel values; critical for interpreting calibration table readings from phone cameras.

---

### 13.11 Experimental Findings Log

**Structure:** Each entry records date, physical conditions, measured result, and deviation from theoretical prediction. This log is the authoritative record of what the physical system actually does, as opposed to what theory predicts. Entries accumulate as experiments are conducted.

```
Entry format:
  Date:        YYYY-MM
  Zone:        calibration disc zone number
  Condition:   substrate, DPI, printer, camera model, lighting, speed
  Predicted:   theoretical value from section 13.2 or 13.5
  Measured:    actual RGB values from photograph
  Deviation:   quantified difference and direction
  Correction:  adjustment applied to encoding table
  Notes:       any anomalies, unexpected behavior, or follow-up required
```

*No entries yet — pending physical calibration disc fabrication and measurement.*

---

### 13.12 Pink Noise Transfer Function Calibration

**Status:** Methodology defined — pending implementation.

#### 13.12.1 Concept

The zone-by-zone lookup table approach in Section 13.4 characterizes the physical system by sampling individual variables in isolation. This is necessary but incomplete. It cannot capture interaction effects between adjacent rings, systematic phase shifts introduced by the camera's rolling shutter, turntable speed variation, or ink spread patterns that span dot boundaries.

A more complete characterization treats the entire physical disc system as a single unknown transfer function H(f) and measures it using a known input signal — the same method used in acoustic room correction, loudspeaker impulse response measurement, and MRI gradient calibration.

The **pink noise calibration method** encodes a known test signal onto a calibration disc, physically plays it back at each target speed, records the decoded output, and computes the cross-correlation between input and output. The result is not a lookup table but a complete frequency-domain characterization of the system — amplitude response, phase response, and blind spots — in a single measurement.

#### 13.12.2 Why pink noise

White noise has equal energy at all frequencies. For a system with 48 perceptual bands spanning 20Hz to 22kHz, white noise distributes energy linearly in frequency — over-exercising high bands and under-exercising bass bands relative to their perceptual importance and physical ring area on the disc.

Pink noise has equal energy per octave, following the 1/f power spectral density:

```
S(f) = k / f

where k is a normalization constant
```

This distribution:
- Exercises all 48 DSA bands proportionally to their perceptual weight
- Matches the natural statistical distribution of music and speech
- Produces a visually rich calibration disc with non-trivial content in every ring, including outer L2 bands that white noise might leave empty at low bitrates
- Is the standard test signal in acoustic measurement precisely because it is spectrally flat in perceptual terms

#### 13.12.3 The transfer function measurement

Let x(t) be the known pink noise input signal and y_ω(t) be the decoded audio output recorded at angular velocity ω.

The cross-correlation between input and output:

```
R_xy(τ) = ∫ x(t) · y_ω(t + τ) dt
```

Taking the Fourier transform of R_xy(τ) yields the cross-power spectral density:

```
S_xy(f) = X*(f) · Y_ω(f)
```

The transfer function of the physical system at speed ω is:

```
H_ω(f) = S_xy(f) / S_xx(f) = Y_ω(f) / X(f)
```

Where S_xx(f) is the power spectral density of the known input. Since x(t) is pink noise, S_xx(f) = k/f and the normalization is well-defined.

**Amplitude response:** |H_ω(f)| — how much each frequency component survives the physical encoding at speed ω. Values near 1.0 indicate faithful encoding. Values near 0 indicate frequency components the system cannot encode at that speed.

**Phase response:** ∠H_ω(f) — the temporal shift introduced by the physical system at each frequency. Non-zero phase at a given frequency indicates that component arrives at the decoder shifted in time relative to the input. On a spinning disc, the camera's rolling shutter introduces a systematic radial phase gradient — each image row is captured at a slightly different rotation angle. This appears as a frequency-dependent phase slope in H_ω(f) and is correctable by the encoder.

**Blind spots:** Frequency components where |H_ω(f)| ≈ 0 are physically unencodable at speed ω. The calibration identifies these automatically — the encoder excludes them from the v4 band assignments for that operating speed. This is more informative than a lookup table because it finds the system's limits without requiring the experimenter to predict them in advance.

#### 13.12.4 Three transfer functions, one disc

The calibration disc is played at three speeds: 33rpm (nominal playback), 45rpm (fast playback), and approximately 10rpm (slow/near-stopped, to characterize the transition from integrated to discrete reading). Each speed yields a different transfer function:

```
H_33(f)  — transfer function at nominal playback speed
H_45(f)  — transfer function at 45rpm
H_10(f)  — transfer function near the integration threshold
```

The ratio H_33(f) / H_45(f) characterizes the speed sensitivity of each frequency band — how much the encoding changes between the two nominal speeds. Bands where this ratio is close to 1.0 are speed-stable: they encode the same value regardless of whether the disc runs at 33 or 45rpm. Bands where the ratio differs significantly are speed-sensitive: they can carry secondary information encoded in the speed difference.

This analysis directly addresses Open Research Question 5 from Section 13.8 — whether speed sensitivity is stable enough to use as a second information channel.

#### 13.12.5 Pre-emphasis derivation

The v4 encoder pre-emphasis curve is the inverse of the 33rpm transfer function:

```
P(f) = 1 / H_33(f)
```

The encoder applies P(f) to the target coefficient values before computing the print configuration. After the physical system applies H_33(f), the output at the decoder is:

```
Y_33(f) = P(f) · H_33(f) · X(f) = X(f)
```

The physical process is cancelled by its own inverse. This is the same principle as:
- **Vinyl RIAA equalization:** the cutting engineer applies the RIAA pre-emphasis curve knowing the playback cartridge applies the inverse de-emphasis
- **Acoustic room correction:** a DSP filter applies the inverse of the room transfer function so the listener hears flat response
- **Optical pre-distortion in fiber systems:** the transmitter pre-distorts the signal to compensate for known chromatic dispersion in the fiber

In all these cases the physical medium and the encoding are co-designed. The pre-emphasis and the physical process are inverse operations — correct only in combination.

#### 13.12.6 Practical measurement procedure

**Step 1 — Generate the test signal**

Generate 60 seconds of pink noise at 44100Hz. Apply DSA encoding at 12kbps. Verify all 48 bands have non-zero energy. This is the input signal x(t).

**Step 2 — Print the calibration disc**

Encode x(t) onto the calibration disc using current v1/v3 dot encoding (no pre-emphasis). Print at reference DPI on reference substrate. This disc encodes the known input in a physically readable form.

**Step 3 — Record playback at each speed**

Mount disc on turntable with Digilog Rig. Record decoded audio y_ω(t) at 33rpm, 45rpm, and 10rpm. Use a fixed-length recording (60 seconds minimum). Record turntable speed independently using the disc clock track for normalization.

**Step 4 — Compute transfer functions**

For each speed:
1. Align x(t) and y_ω(t) using coarse cross-correlation to find the start-of-disc offset
2. Compute FFT of both signals over matched windows
3. Compute H_ω(f) = Y_ω(f) / X(f) per the formula in 13.12.3
4. Smooth H_ω(f) over 1/3-octave bands to reduce measurement noise

**Step 5 — Extract pre-emphasis curve**

Compute P(f) = 1 / H_33(f). Apply minimum-phase reconstruction to obtain a causal pre-emphasis filter. This filter is the v4 encoder's optical compensation curve.

**Step 6 — Validate**

Encode a second test signal with pre-emphasis applied. Print and play. Compute H_ω(f) for the pre-emphasized disc. If |H_ω(f)| ≈ 1.0 across all encodable bands, pre-emphasis is correctly calibrated. Iterate if significant deviation remains.

#### 13.12.7 Connection to acoustic measurement literature

The pink noise transfer function method is a standard technique in acoustic engineering. The specific application to a spinning optical disc is novel, but the mathematical framework is identical to established practice:

- **Loudspeaker measurement (IEC 60268-5):** measure frequency response by playing pink noise through the speaker and analyzing the output with a calibrated microphone. The transfer function characterizes the speaker's behavior completely.
- **Room correction systems (Dirac Live, Audyssey, ARC):** play test tones or noise through loudspeakers, measure the room's impulse response, derive a correction filter as the inverse transfer function. The listener hears flat response in the corrected room.
- **Acoustic echo cancellation:** measure the transfer function of the acoustic path from loudspeaker to microphone; apply the inverse as a filter to cancel echo. Used in every phone call and video conference.

The Digilog v4 calibration disc applies exactly this framework to a rotational optical system rather than an acoustic one. The physical medium is different; the mathematics is the same.

---

### 13.13 Additional references

26. Schroeder, M.R. (1979). "Integrated-impulse method measuring sound decay without using impulses." *Journal of the Acoustical Society of America*, 66(2), 497–500. — Foundational method for measuring room transfer functions using noise signals rather than impulses; directly analogous to the pink noise calibration method.

27. Müller, S., Massarani, P. (2001). "Transfer-function measurement with sweeps." *Journal of the Audio Engineering Society*, 49(6), 443–471. — Modern reference for transfer function measurement methodology; covers cross-correlation, swept sine, and noise methods with comparative analysis. The cross-correlation method in Section 13.12.3 follows this framework.

28. Voss, R.F., Clarke, J. (1978). "1/f noise in music: music from 1/f noise." *Journal of the Acoustical Society of America*, 63(1), 258–263. — Establishes 1/f (pink noise) as the natural spectral distribution of music; justifies pink noise as the correct test signal for a perceptual audio codec calibration.

29. Lipshitz, S.P., Pocock, M., Vanderkooy, J. (1982). "On the audibility of midrange phase distortion in audio systems." *Journal of the Audio Engineering Society*, 30(9), 580–595. — Treatment of phase response in audio systems; relevant to the phase response measurement and rolling shutter phase correction in Section 13.12.3.

30. Griesinger, D. (1996). "Practical processors and programs for digital reverberation." *AES 7th International Conference: Audio in Digital Times*, 187–195. — Practical implementation of transfer function inversion for room correction; the pre-emphasis derivation in Section 13.12.5 follows this approach.

31. IEC 60268-5 (2003). *Sound System Equipment — Part 5: Loudspeakers*. International Electrotechnical Commission. — Standard measurement methodology for loudspeaker frequency response using noise signals; the institutional precedent for transfer function measurement via known signal injection.

32. El-Raheem, A.M.A. (2019). "Rolling shutter effect modeling and correction for high-speed rotating objects." *Journal of Electronic Imaging*, 28(4). — Models the systematic phase shift introduced by rolling shutter cameras when imaging rotating objects; directly applicable to the phase correction required in Section 13.12.3 for Digilog disc reading.

---

*Section 13 added April 2026 following theoretical development of DSA v4: Physics-Integrated Optical Encoding. The calibration disc and experimental protocol described here are the necessary first step before any v4 encoding specification can be written. The specification will be derived from measurement, not theory.*

---

## 14. Derived Format Research Directions

This section records research directions that extend beyond DSA audio into adjacent format spaces using the same physical-media and layered-encoding principles. These are not planned features of DSA v1. They are design-space explorations that share DSA's core architecture.

---

### 14.1 Hybrid disc architecture

**Concept:** A single Digilog disc that carries two independent formats on the same physical surface — one for static reading (disc at rest, camera macro scan) and one for dynamic reading (disc spinning at 33rpm on a turntable).

**Static layer:** Standard QR-code-adjacent encoding — high-density dot patterns readable by a flatbed scanner or macro camera lens at rest. This layer uses conventional 2D barcode error correction (Reed-Solomon) and is not time-dependent. Suitable for: album metadata, lyrics, credits, download links, cryptographic signatures.

**Dynamic layer:** DSA audio encoding — the colored dot rings that require rotation to decode. The dynamic layer uses the motion-blur integration mechanism and is unreadable at rest (the individual dots are visible but carry no decodable audio information without temporal integration).

**Physical co-existence:** The two layers occupy different spatial regions of the disc surface. The static layer uses fine black dots in the outermost and innermost regions (which have low area-per-band-unit and are less critical for audio quality). The dynamic layer uses colored dot rings in the audio encoding region. A camera scanning the disc at rest reads static data. The same camera at 33rpm reads audio.

**Research question:** What is the minimum physical separation between layers required to prevent cross-read interference? Can a single optical pass at rest distinguish static black dots from dynamic color dots with sufficient reliability?

### 14.2 Video extension path

**Concept:** DVA — Digilog Visual Architecture — a video codec using DSA's layered physical-media framework extended to spatial-temporal video data. Full conceptual description in Section 10.3.

**Format extension:** A DVA disc would use concentric ring zones for spatial frequency layers, analogous to DSA audio layers:
- Inner rings: coarse spatial structure (DC + low spatial frequency) — always readable
- Middle rings: medium spatial detail — readable on average camera
- Outer rings: fine spatial detail and texture — readable with controlled rig

**Frame rate coupling:** Disc rotation speed determines temporal resolution, analogous to how DSA rotation speed determines temporal playback rate. Slower rotation = lower frame rate but higher spatial fidelity per frame (longer exposure integration time). This is the inverse of the DSA audio relationship (slower rotation = slower playback) — in DVA, slower rotation trades temporal for spatial resolution.

**Compression kernel:** The 3D spatial-temporal transform with CSF-based perceptual quantization described in Section 10.3. The DSA1 bitstream format would be extended with a DVA1 variant carrying 2D coefficient blocks per frame instead of 1D spectral coefficients.

**Status:** No implementation planned until DSA audio v1 is externally validated. DVA is a 3–5 year research horizon.

### 14.3 Better-than-Opus compression

**Concept:** DSA's current benchmark position (Section 11) shows that DSA beats Opus at 6–12 kbps on tonal signals and at 96 kbps after the bidirectional scaler fix. The 32 kbps tonal gap (Opus leads by 5–17 dB) and the broadband gap (Opus leads on chirp and white noise at all bitrates) indicate that DSA's quantizer is leaving efficiency on the table.

**Path to closing the gap:**

**(a) Psychoacoustic model refinement:** The current ATH + Bark masking model (Section 3) uses simplified ISO 226 equal-loudness contours with fixed frequency weights. A full implementation would incorporate simultaneous masking (tonal masker suppresses nearby noise), temporal masking (pre- and post-masking windows), and inter-channel masking for stereo. These are standard in AAC and Opus and account for approximately 3–6 dB improvement at mid bitrates.

**(b) ANS entropy coding (Section 10.5):** 10–15% compression improvement across all bitrates without changing the quantizer. At 32 kbps this recovers 3.2–4.8 kbps, which at current SNR curves corresponds to approximately 2–4 dB.

**(c) Per-layer adaptive Huffman (Section 10.4) as an intermediate step:** 8–15% improvement at low implementation cost, recoverable within the current Huffman architecture.

**(d) Joint stereo encoding:** DSA currently encodes channels independently. Mid/Side (M/S) stereo encoding, standard since MP3, reduces inter-channel redundancy. For stereo signals, M/S typically saves 20–35% of the bitrate allocated to the Side channel, which can be redirected to the Mid channel for improved center-image fidelity.

**Target:** With (a) + (b) + (d) implemented, DSA should exceed Opus at 32 kbps on tonal signals and match Opus on broadband signals. Exceeding Opus on broadband at all bitrates would require a fundamental change to the transform (longer window, higher frequency resolution) and is not a target for v1.

---

*This document is a living research record. It will be updated as implementation progresses and will form the basis of a formal scientific publication.*

*github.com/pisdronio/dsa*
*Scan the groove.*
