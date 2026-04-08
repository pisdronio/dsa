# DSA Format Specification

**DSA — Digilog Scalable Audio**
**Version:** 1.0
**Status:** Draft
**Repository:** https://github.com/pisdronio/dsa
**Format spec:** https://github.com/pisdronio/digilog-spec
**License:** Creative Commons Attribution-ShareAlike 4.0 (CC-BY-SA 4.0)

---

## About this document

This is the normative specification for the DSA audio format. It defines the bitstream layout, frame structure, quantization model, and layered encoding rules that any conforming DSA encoder or decoder must implement.

The reference implementation is at https://github.com/pisdronio/dsa (Python, GPL v3).
The scientific documentation is in `RESEARCH.md` in the same repository.

---

## 1. Overview

DSA is an audio codec for the Digilog physical audio format. It encodes audio as a layered bitstream that maps directly to concentric rings on a printed disc:

```
Layer 0  →  inner disc rings   (8 bands,  20Hz – 800Hz)    always readable
Layer 1  →  middle rings       (16 bands, 800Hz – 6kHz)    readable on phone camera
Layer 2  →  outer rings        (24 bands, 6kHz – 22kHz)    requires Digilog Rig
```

Core properties:
- Native bidirectional playback (reverse = true reversed audio, not corruption)
- Layered scalable decoding (each layer genuinely enhances the previous)
- Analog degradation model (partial reads → attenuated audio, not silence)
- Variable-speed playback with pitch coupling (vinyl feel)

---

## 2. Transform

### 2.1 MDCT parameters

```
N  = 2048   window length (samples)
M  = 1024   output coefficients (N/2)
HOP = 1024  hop size (50% overlap)
```

Sample rate: 44100 Hz
Frame duration: HOP / sample_rate = 23.2 ms

### 2.2 Window

Sine window:

```
w[n] = sin(π/N × (n + 0.5))    for n = 0, 1, ..., N−1
```

Satisfies the Princen-Bradley (TDAC) condition:

```
w[n]² + w[n + M]² = 1    for all n = 0, ..., M−1
```

### 2.3 Forward MDCT

```
X[k] = sqrt(2/M) × Σ_{n=0}^{N−1}  w[n] × x[n] × cos(π/M × (n + 0.5 + M/2) × (k + 0.5))

for k = 0, 1, ..., M−1
```

### 2.4 Inverse MDCT

```
x'[n] = sqrt(2/M) × Σ_{k=0}^{M−1}  X[k] × cos(π/M × (n + 0.5 + M/2) × (k + 0.5))

for n = 0, 1, ..., N−1
```

### 2.5 Overlap-add reconstruction

Output is reconstructed by overlap-add of consecutive IMDCT frames with HOP=M shift:

```
out[i×HOP : i×HOP + N] += imdct(coeffs[i])
```

TDAC guarantees perfect reconstruction in the valid interior region (SNR ≥ 238 dB).

---

## 3. Frequency Bands

48 perceptual bands split across three layers:

```
Layer 0:   8 bands   20Hz –   800Hz   linear spacing    (indices 0–7)
Layer 1:  16 bands  800Hz –  6000Hz   log spacing       (indices 8–23)
Layer 2:  24 bands  6000Hz – 22050Hz  log spacing       (indices 24–47)
```

### 3.1 Band boundaries (Hz)

```
Layer 0 — linear, 8 bands:
  Band  0:    20 –   120 Hz
  Band  1:   120 –   220 Hz
  Band  2:   220 –   320 Hz
  Band  3:   320 –   420 Hz
  Band  4:   420 –   520 Hz
  Band  5:   520 –   620 Hz
  Band  6:   620 –   720 Hz
  Band  7:   720 –   800 Hz

Layer 1 — log, 16 bands:
  Band  8:   800 –   900 Hz
  Band  9:   900 –  1013 Hz
  Band 10:  1013 –  1139 Hz
  Band 11:  1139 –  1282 Hz
  Band 12:  1282 –  1442 Hz
  Band 13:  1442 –  1622 Hz
  Band 14:  1622 –  1825 Hz
  Band 15:  1825 –  2054 Hz
  Band 16:  2054 –  2311 Hz
  Band 17:  2311 –  2600 Hz
  Band 18:  2600 –  2926 Hz
  Band 19:  2926 –  3294 Hz
  Band 20:  3294 –  3708 Hz
  Band 21:  3708 –  4174 Hz
  Band 22:  4174 –  4699 Hz
  Band 23:  4699 –  6000 Hz

Layer 2 — log, 24 bands:
  Band 24:  6000 –  6415 Hz
  Band 25:  6415 –  6860 Hz
  Band 26:  6860 –  7335 Hz
  Band 27:  7335 –  7843 Hz
  Band 28:  7843 –  8388 Hz
  Band 29:  8388 –  8972 Hz
  Band 30:  8972 –  9591 Hz
  Band 31:  9591 – 10260 Hz
  Band 32: 10260 – 10970 Hz
  Band 33: 10970 – 11729 Hz
  Band 34: 11729 – 12542 Hz
  Band 35: 12542 – 13411 Hz
  Band 36: 13411 – 14342 Hz
  Band 37: 14342 – 15338 Hz
  Band 38: 15338 – 16403 Hz
  Band 39: 16403 – 17545 Hz
  Band 40: 17545 – 18764 Hz
  Band 41: 18764 – 20068 Hz
  Band 42: 20068 – 21460 Hz
  Band 43: 21460 – 22050 Hz
  Bands 44–47: remaining bins to Nyquist
```

Exact MDCT bin ranges are derived from the band Hz boundaries:
`bin = round(freq_hz × N / sample_rate)`

### 3.2 Perceptual weights

Simplified ISO 226 equal-loudness weighting applied to quantization step allocation:

```
< 100 Hz:       0.25   sub-bass
100 – 300 Hz:   0.55   bass
300 Hz – 1 kHz: 0.80   low-mid
1 – 4 kHz:      1.00   peak sensitivity (maximum weight)
4 – 8 kHz:      0.75   presence
8 – 12 kHz:     0.45   air
> 12 kHz:       0.20   ultra-high
```

---

## 4. Frame Structure

### 4.1 Frame types

| Type | Symbol | Description |
|------|--------|-------------|
| Keyframe | K | Self-contained spectral snapshot. Decodable without reference to other frames. |
| Bidirectional | B | Residual relative to interpolation between surrounding K-frames. |
| Silence | S | Energy below silence threshold (−55 dBFS). Minimal data. |

### 4.2 GOP structure

Group of Pictures: 8 frames (~185ms at 23.2ms/frame).

```
K  B  B  B  B  B  B  B  K  B  B  B  ...
0  1  2  3  4  5  6  7  8  9  10 11
↑                       ↑
Keyframe               Keyframe
(resync point)         (resync point)
```

K-frames occur at frame indices that are multiples of 8.
B-frames at positions 1–7 within each GOP reference the K-frames at positions 0 and 8.

### 4.3 Silence threshold

A frame is classified as silence (S-frame) when:

```
max(energies_db) < −55 dBFS
```

S-frames store a silence flag only. Their MDCT coefficients are all zero.

### 4.4 K-frame encoding

1. Apply perceptual quantization to all 48 bands
2. Entropy-code with static Huffman table per band group
3. Pack into self-contained K-frame record

### 4.5 B-frame encoding

1. Compute spectral interpolation between surrounding K-frames:
   ```
   α = gop_pos / GOP_SIZE    (0 < α < 1)
   interp[k] = (1 − α) × K0_coeffs[k] + α × K1_coeffs[k]
   ```
2. Compute residual: `residual[k] = actual_coeffs[k] − interp[k]`
3. Quantize residual (typically much smaller than full coefficients)
4. Entropy-code and pack with references to surrounding K-frame indices

B-frames decode identically in forward and reverse order because both surrounding K-frames are always available regardless of playback direction.

---

## 5. Perceptual Quantization

### 5.1 Absolute Threshold of Hearing (ATH)

Per-band ATH in dBFS (normalized to 96 dBSPL for 16-bit audio):

```
ATH_dBSPL(f_kHz) = 3.64×f^−0.8 − 6.5×exp(−0.6×(f−3.3)²) + 0.001×f⁴

ATH_dBFS[b] = ATH_dBSPL(fc[b]) − 96
```

where `fc[b]` is the center frequency of band b in kHz.

### 5.2 Masking thresholds

Bark-scale simultaneous masking using asymmetric spreading function:

```
S[b, m] = −MI − 25 × dz    if dz ≥ 0  (upward masking)
S[b, m] = −MI − 40 × |dz|  if dz < 0  (downward masking)

where MI = 14 dB  (masking index)
      dz = Bark(b) − Bark(m)
      Bark(f) = 13×arctan(0.00076f) + 3.5×arctan((f/7500)²)
```

Global masking threshold per band:

```
T[b] = max(ATH[b],  max_m( L[m] + S[b, m] ))
```

### 5.3 Step sizes

```
step[b] = 10^((T[b] − 3dB) / 20)
```

The 3 dB headroom keeps quantization noise safely below (not merely at) the masking threshold.

### 5.4 Bidirectional rate-distortion budget enforcement

Layer degradation priority: L2 first (outer rings), L1 second, L0 last (inner rings).

**Phase 1 — over budget:** Binary search for minimum upward scale factor per layer that brings estimated bit cost within budget. Applied L2 → L1 → L0 until cost ≤ budget.

**Phase 2 — under budget:** When surplus budget > 5% of total, binary search for a uniform downward scale factor across all bands that consumes the budget. Lower bound: `peak_coeff[b] / (MAX_QUANT × step[b])` per band, to prevent quantized value saturation at MAX_QUANT = 2047.

### 5.5 Quantization

Uniform scalar quantization per band:

```
q[k] = clip(round(coeff[k] / step[b]), −MAX_QUANT, +MAX_QUANT)
```

where MAX_QUANT = 2047 and b is the band containing bin k.

---

## 6. Analog Degradation Model

### 6.1 Confidence vector

The visual decoder supplies a per-band confidence vector α of shape (48,) with α[b] ∈ [0.0, 1.0]:

```
Mode 1 (discrete dots):
  α[b] = 1.0   band b read cleanly
  α[b] = 0.0   band b region unreadable

Mode 2 (gradient dots):
  α[b] = continuous value from gradient clarity
         0.0 = completely unreadable
         1.0 = perfect read
```

### 6.2 Confidence-weighted dequantization

```
C̃[k] = q[k] × step[b] × α[b]
```

At α = 1.0: standard dequantization.
At α = 0.5: −6 dB attenuation (exactly 20×log₁₀(0.5)).
At α = 0.0: band contributes zero to IMDCT reconstruction.

### 6.3 Layer selection

Layer selection is implemented as a confidence mask:

```
L0 excluded:  α[0:8]    = 0.0
L1 excluded:  α[8:24]   = 0.0
L2 excluded:  α[24:48]  = 0.0
```

### 6.4 K-frame loss recovery

When a K-frame is unreadable, the decoder holds the last valid spectral shape and applies exponential decay:

```
decay_per_frame = exp(−frame_ms / (τ × 1000))
                = exp(−23.2 / 60000)
                ≈ 0.679

substitute_K[n_frames_lost] = last_valid_K × 0.679^n_frames_lost
```

τ = 60 ms. Acoustic result: a note fading out, not a digital dropout.

---

## 7. Reverse Playback

Process frames in reverse index order, then apply IMDCT and overlap-add in that order:

```
# Forward
for i in 0, 1, 2, ..., N−1:
    out[i×HOP : i×HOP + N] += imdct(coeffs[i])

# Reverse
for i in N−1, N−2, ..., 0:
    out[j×HOP : j×HOP + N] += imdct(coeffs[i])
    j += 1
```

The sine window satisfies `w[n] = w[N−1−n]`, so TDAC cancellation operates identically in reverse. The result is the true time-reversed audio signal at frame-level temporal granularity (23.2 ms per frame).

---

## 8. Variable-Speed Playback

Speed multiplier `s` relative to nominal (s = 1.0 at standard playback):

```
s > 1.0   faster, pitch rises  (vinyl scratch forward)
s < 1.0   slower, pitch drops  (vinyl slow-down)
s = 0.0   stopped: exponential amplitude decay, τ = 60 ms
s < 0.0   reverse at |s|
```

Pitch is coupled to speed (natural vinyl feel). Output sample count:

```
n_output = round(n_decoded / s)
```

Implemented via linear resampling. For real-time DJ use, SoundTouch (LGPL) is recommended for latency < 50 ms. For playback-only use, RubberBand (GPL) provides higher quality.

---

## 9. Entropy Coding

Static Huffman tables per band group. Symbol: quantized integer magnitude.
Zero-valued coefficients are run-length coded. Sign bit stored separately.

Huffman table derivation: Laplacian distribution with parameter estimated from the perceptual band energy. See reference implementation (`dsa_huffman.py`) for the static table definition.

---

## 10. Bitstream Format — DSA1

### 10.1 File layout

```
DSA1 File
├── Header              (32 bytes)
├── Frame index         (4 bytes × n_frames)
├── Layer 0 block       (Huffman-coded K/B-frame data, L0 bands)
├── Layer 1 block       (L1 bands)
├── Layer 2 block       (L2 bands)
└── CRC32               (4 bytes, covers all preceding bytes)
```

### 10.2 Header (32 bytes)

| Offset | Size | Field | Value |
|--------|------|-------|-------|
| 0 | 4 | Magic | `0x44 0x53 0x41 0x31` ("DSA1") |
| 4 | 1 | Version | `0x01` |
| 5 | 1 | Mode | `0x01` = discrete, `0x02` = gradient |
| 6 | 2 | Reserved | `0x00 0x00` |
| 8 | 4 | Sample rate | uint32 LE (e.g. 44100) |
| 12 | 4 | Frame count | uint32 LE |
| 16 | 4 | Nominal bitrate | uint32 LE (kbps) |
| 20 | 4 | L0 block offset | uint32 LE (bytes from file start) |
| 24 | 4 | L1 block offset | uint32 LE |
| 28 | 4 | L2 block offset | uint32 LE |

### 10.3 Mode byte values

| Value | Name | Description |
|-------|------|-------------|
| `0x01` | MODE_DISCRETE | Discrete dot encoding. Visual decoder produces α ∈ {0, 1}. |
| `0x02` | MODE_GRADIENT | Gradient dot encoding. Visual decoder produces α ∈ [0, 1]. |
| `0x04` | MODE_PHYSICS | Physics-integrated v4 (reserved, not yet specified). |

### 10.4 Frame index

One 4-byte uint32 LE per frame. Value = byte offset of the frame's data within its layer block.

### 10.5 Frame record layout

**K-frame:**
```
[frame_type: 2 bits = 0b00]
[gop_pos:    4 bits = 0]
[reserved:   2 bits]
[per-layer data blocks, one per included layer]
  [band_count: 1 byte]
  [step_exp:   1 byte per band (quantization step exponent)]
  [huffman:    variable-length Huffman-coded coefficients]
```

**B-frame:**
```
[frame_type: 2 bits = 0b01]
[gop_pos:    4 bits = 1–7]
[reserved:   2 bits]
[k0_ref:     2 bytes = frame index of preceding K-frame]
[per-layer residual data blocks]
```

**S-frame:**
```
[frame_type: 2 bits = 0b10]
[gop_pos:    4 bits]
[reserved:   2 bits]
(no data — all coefficients implicitly zero)
```

### 10.6 CRC32

CRC32 (ISO 3309 polynomial) of all bytes from file offset 0 through the end of the L2 block. Stored as 4-byte little-endian uint32 at the end of the file.

---

## 11. Disc Layout — GradientDot Interface

DSA disc encoding maps MDCT coefficients to physical gradient dots on the Digilog disc surface.

### 11.1 GradientDot record

Each coefficient produces one GradientDot:

| Field | Type | Description |
|-------|------|-------------|
| `frame_idx` | int | Source frame index |
| `band_idx` | int | Band index 0–47 |
| `coeff_idx` | int | Coefficient position within band |
| `layer` | int | 0, 1, or 2 |
| `color_a` | str | Primary color (layer-assigned) |
| `color_b` | str | Secondary color (layer-assigned) |
| `steepness` | float | Gradient steepness ∈ [0, 1] — encodes coefficient magnitude |
| `direction` | int | +1 or −1 — encodes coefficient sign |

### 11.2 Color pair assignment

Layer-specific color pairs for maximum visual discriminability under expected reading conditions:

```
Layer 0 (inner rings, always readable):
  High-contrast pairs: Black↔White, Black↔Yellow, Black↔Cyan
  Cycling over band index within layer.

Layer 1 (middle rings, phone camera):
  Complementary pairs: Red↔Cyan, Blue↔Yellow, Green↔Purple
  Cycling over band index within layer.

Layer 2 (outer rings, Digilog Rig):
  Full 8-color palette, all pairs available.
  8 pairs cycling over band index within layer.
```

### 11.3 Steepness encoding

**Mode 1 (discrete):**

```
steepness = |q_int| / 31
```

32 discrete levels (0, 1/31, 2/31, ..., 1). Layer 2 pre-emphasis: `min(1.0, steepness × 1.15)`.

**Mode 2 (gradient):**

```
steepness = |c_float| / band_peak
```

where `band_peak` = maximum absolute coefficient value in that band across all frames.
Layer 2 pre-emphasis: `min(1.0, steepness × 1.15)`.

### 11.4 Layer 2 pre-emphasis

Factor: 1.15×. Compensates for lens PSF blur at 15 cm focal distance under Digilog Rig LED illumination. Layer 0 and Layer 1 dots are large enough that PSF blur is negligible.

---

## 12. Conformance

A conforming DSA decoder MUST:

1. Implement MDCT/IMDCT with N=2048, M=1024, sine window
2. Decode K-frames without reference to any other frame
3. Decode B-frames using the two surrounding K-frames
4. Apply the confidence-weighted dequantization model: `C̃[k] = q[k] × step[b] × α[b]`
5. Implement K-frame loss recovery with τ ≈ 60 ms exponential decay
6. Decode in reverse order when `reverse=True`
7. Verify CRC32 before decoding

A conforming DSA decoder SHOULD:

- Support layer selection (L0 only, L0+L1, all layers)
- Support variable-speed playback
- Support analog degradation input (continuous α ∈ [0, 1])

A conforming DSA encoder MUST:

1. Produce valid DSA1 file headers with correct magic bytes and CRC32
2. Align K-frames to GOP boundaries (multiples of 8)
3. Encode K-frames as self-contained records
4. Encode B-frames with residuals relative to surrounding K-frame interpolation
5. Apply the perceptual masking model for quantization step selection
6. Enforce layer priority degradation order (L2 first, L0 last)

---

## 13. Reference Implementation

Python reference implementation: https://github.com/pisdronio/dsa

```
dsa_analyzer.py   — MDCT, band layout, frame analysis
dsa_quantizer.py  — perceptual quantizer, masking model
dsa_encoder.py    — K/B-frame encoders
dsa_huffman.py    — static Huffman table
dsa_bitstream.py  — DSA1 file format
dsa_decoder.py    — decoder, reverse playback, variable speed
dsa_disc.py       — GradientDot disc layout interface
dsa_cli.py        — unified command-line tool
```

Test suite: `tests/` — 74 tests, all pass on the reference implementation.

---

## 14. License

This specification is licensed under Creative Commons Attribution-ShareAlike 4.0 International (CC-BY-SA 4.0).

You are free to implement this specification in any software, hardware, or physical medium. Implementations may be proprietary. Modifications to the specification itself must be shared under the same license.

Reference implementation: GPL v3 (https://github.com/pisdronio/dsa/blob/main/LICENSE)

---

*DSA is part of the Digilog open physical audio format.*
*github.com/pisdronio/dsa*
*Scan the groove.*
