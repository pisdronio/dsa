#!/usr/bin/env python3
"""
DSA — Digilog Scalable Audio
Step 2: Perceptual Quantizer

Noise-shaping quantizer using a psychoacoustic masking model:
  - Absolute threshold of hearing (ATH, derived from ISO 226)
  - Simultaneous masking with Bark-scale frequency spreading
  - Asymmetric spreading function (Zwicker): upward -25 dB/Bark, downward -40 dB/Bark
  - Per-layer bit budget priority (L0 most protected, L2 degraded first)
  - Variable bit budget for different disc capacities / bitrates

License: GPL v3 — github.com/pisdronio/digilog-spec
"""

import numpy as np
import sys
from dataclasses import dataclass
from dsa_analyzer import (
    DSAFrame, BANDS, BINS, WEIGHTS, MDCT_M, NUM_BANDS,
    L0, L1, L2, SAMPLE_RATE, FRAME_MS, SILENCE_DB,
    mdct, imdct, coeffs_to_bands, DSAAnalyzer,
)

# ─── Constants ────────────────────────────────────────────────────────────────

MIN_STEP         = 1e-7     # minimum quantization step (avoid divide-by-zero)
MAX_QUANT        = 2047     # max quantized integer (±2047, 12-bit signed)
SMR_HEADROOM_DB  = 3.0      # dB safety margin below masking threshold
DEFAULT_BITRATE  = 12       # kbps

# Huffman/arithmetic coding compresses nonzero MDCT magnitudes by roughly 2×
# over naive sign+magnitude coding (geometric/Laplacian distribution).
# The budget estimator below counts only nonzero coefficients; we scale the
# raw per-frame budget up by this factor to account for that compression.
# Zero runs are coded essentially for free by the entropy coder (Step 5).
ENTROPY_FACTOR      = 2.0
BITS_PER_BAND_HDR   = 2.0   # per-band overhead (band type / count header)

# ─── Band center frequencies ──────────────────────────────────────────────────

_FC = np.array([(lo + hi) / 2.0 for lo, hi in BANDS])  # (NUM_BANDS,)

# ─── Absolute Threshold of Hearing (ATH) ──────────────────────────────────────

def _ath_db(f_hz: float) -> float:
    """
    Absolute threshold of hearing in dBSPL.
    Painter & Spanias (2000), from ISO 226 equal-loudness contours.
    """
    f = max(f_hz, 20.0) / 1000.0           # Hz → kHz
    return (3.64 * f**-0.8
            - 6.5 * np.exp(-0.6 * (f - 3.3)**2)
            + 0.001 * f**4)

# ATH normalized to dBFS: 0 dBFS ≈ 96 dBSPL for 16-bit audio.
_ATH_DBFS = np.array([_ath_db(fc) for fc in _FC]) - 96.0   # (NUM_BANDS,)

# ─── Bark scale ───────────────────────────────────────────────────────────────

def _hz_to_bark(f: float) -> float:
    """Zwicker (1961) Bark approximation."""
    return 13.0 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500.0)**2)

_BARK = np.array([_hz_to_bark(fc) for fc in _FC])           # (NUM_BANDS,)

# ─── Spreading function (pre-computed) ────────────────────────────────────────

def _build_spread() -> np.ndarray:
    """
    S[b, m] = masking contribution (dB) of masker at band m on masked band b,
              relative to the masker's own level. Always ≤ 0.

    Asymmetric Zwicker spreading:
      dz = Bark(b) - Bark(m)
      dz > 0  (upward masking, b above m):   -MI - 25 * dz  dB
      dz < 0  (downward masking, b below m): -MI - 40 * |dz|  dB
      dz = 0  (same band):                   -MI  dB

    Masking index MI = 14 dB (threshold is 14 dB below the masker in the same band).
    """
    MI  = 14.0
    dz  = _BARK[:, np.newaxis] - _BARK[np.newaxis, :]   # (b, m), vectorised
    S   = np.where(dz >= 0,
                   -MI - 25.0 * dz,
                   -MI + 40.0 * dz)   # dz<0 → -MI - 40*|dz|
    return S

_SPREAD = _build_spread()   # (NUM_BANDS, NUM_BANDS)

# ─── Masking threshold ────────────────────────────────────────────────────────

def masking_thresholds(energies_db: np.ndarray) -> np.ndarray:
    """
    Compute per-band global masking threshold.

    Each band acts as a potential masker. Its contribution to every other band
    is its level plus the spreading function. The threshold at each band is the
    max of all masker contributions and the absolute threshold of hearing.

    Args:
        energies_db: (NUM_BANDS,) band energies in dBFS.

    Returns:
        (NUM_BANDS,) masking threshold in dBFS — quantization noise must stay below this.
    """
    L = np.maximum(energies_db, SILENCE_DB)          # clamp to floor
    # contrib[b, m] = L[m] + S[b, m]: masker m's contribution to band b
    contrib = L[np.newaxis, :] + _SPREAD             # (NUM_BANDS, NUM_BANDS)
    simultaneous = np.max(contrib, axis=1)           # (NUM_BANDS,)
    return np.maximum(_ATH_DBFS, simultaneous)

# ─── Step sizes ───────────────────────────────────────────────────────────────

def threshold_to_steps(threshold_db: np.ndarray) -> np.ndarray:
    """
    Convert per-band masking threshold (dBFS) to quantization step sizes
    (linear amplitude). Quantization noise must stay below the threshold.
    """
    noise_db = threshold_db - SMR_HEADROOM_DB
    return np.maximum(10.0 ** (noise_db / 20.0), MIN_STEP)

# ─── Bit estimation ───────────────────────────────────────────────────────────

def _band_bits(coeffs: np.ndarray, step: float) -> float:
    """
    Bit estimate for one band's quantized coefficients.

    Counts only nonzero quantized levels (zero-runs are entropy-coded for nearly
    free in Step 5 via run-length encoding). Each nonzero level k costs
    log2(k+1) + 1 bits (magnitude + sign). A small per-band header is added.

    This approximates the post-entropy-coding bit cost and avoids the ~10×
    overestimate of peak-based or 1-bit-per-zero approaches, which would cause
    the budget scaler to crush all coefficients to zero.
    """
    if step <= MIN_STEP or len(coeffs) == 0:
        return BITS_PER_BAND_HDR
    q       = np.round(np.abs(coeffs) / step)
    nonzero = q[q > 0]
    if len(nonzero) == 0:
        return BITS_PER_BAND_HDR
    bits = np.log2(nonzero + 1.0) + 1.0    # sign (1) + magnitude (log2(k+1))
    return float(np.sum(bits)) + BITS_PER_BAND_HDR

def _total_bits(coeffs_bands: list, steps: np.ndarray) -> float:
    return sum(_band_bits(coeffs_bands[b], steps[b]) for b in range(NUM_BANDS))

def _scale_to_budget(
    coeffs_bands: list,
    steps: np.ndarray,
    bit_budget: float,
) -> np.ndarray:
    """
    Scale step sizes upward (reducing quality) until estimated bit cost fits
    within bit_budget * ENTROPY_FACTOR.

    Degrades layers in priority order — L2 first (outer disc rings, high freq),
    L1 second, L0 last (inner rings, always-readable bass layer).

    Uses binary search per layer for speed.
    """
    steps      = steps.copy()
    eff_budget = bit_budget * ENTROPY_FACTOR

    layer_ranges = [
        list(range(L0 + L1, NUM_BANDS)),   # L2 — degraded first
        list(range(L0, L0 + L1)),           # L1
        list(range(0, L0)),                 # L0 — most protected
    ]

    for band_indices in layer_ranges:
        if _total_bits(coeffs_bands, steps) <= eff_budget:
            break
        lo_s, hi_s = 1.0, 65536.0
        for _ in range(24):                 # 24 iterations → sub-ppm precision
            mid   = (lo_s + hi_s) / 2.0
            trial = steps.copy()
            for b in band_indices:
                trial[b] = steps[b] * mid
            if _total_bits(coeffs_bands, trial) <= eff_budget:
                hi_s = mid
            else:
                lo_s = mid
        for b in band_indices:
            steps[b] *= hi_s

    return steps

# ─── QuantizedFrame ───────────────────────────────────────────────────────────

@dataclass
class QuantizedFrame:
    """
    Quantized representation of one DSA frame.

    Stores integer MDCT coefficients (per band) alongside the step sizes used,
    so dequantization is fully self-contained. This is the output of Step 2 and
    the input to Steps 3–4 (K/B-frame encoding) and Step 5 (entropy coding).
    """
    frame_type:   str
    frame_idx:    int
    gop_pos:      int
    is_silence:   bool
    quant_coeffs: list          # list[np.ndarray[int16]], one array per band
    steps:        np.ndarray    # (NUM_BANDS,) step sizes (float64, linear)
    bits_l0:      int = 0       # uncompressed bit estimate, layer 0
    bits_l1:      int = 0       # uncompressed bit estimate, layer 1
    bits_l2:      int = 0       # uncompressed bit estimate, layer 2

    @property
    def total_bits(self) -> int:
        return self.bits_l0 + self.bits_l1 + self.bits_l2

# ─── DSAQuantizer ─────────────────────────────────────────────────────────────

class DSAQuantizer:
    """
    Perceptual quantizer for DSA frames.

    For each frame:
      1. Compute masking thresholds from band energies (ATH + simultaneous masking)
      2. Derive initial step sizes (quantization noise stays below masking threshold)
      3. Scale steps to fit the frame's bit budget, degrading L2 → L1 → L0
      4. Quantize each band's MDCT coefficients with uniform scalar quantization

    Usage:
        q  = DSAQuantizer(bitrate_kbps=12)
        qf = q.quantize(frame)        # DSAFrame → QuantizedFrame
        c  = q.dequantize(qf)         # QuantizedFrame → float64 MDCT coefficients
    """

    def __init__(self, bitrate_kbps: float = DEFAULT_BITRATE):
        self.bitrate_kbps  = bitrate_kbps
        self.bits_per_frame = int(bitrate_kbps * 1000.0 * (FRAME_MS / 1000.0))

    def quantize(self, frame: DSAFrame) -> QuantizedFrame:
        """Quantize a DSAFrame into integer MDCT coefficients."""
        if frame.is_silence:
            return QuantizedFrame(
                frame_type=frame.frame_type,
                frame_idx=frame.frame_idx,
                gop_pos=frame.gop_pos,
                is_silence=True,
                quant_coeffs=[np.zeros(0, dtype=np.int16)] * NUM_BANDS,
                steps=np.ones(NUM_BANDS, dtype=np.float64),
            )

        # 1. Masking threshold per band
        thresh = masking_thresholds(frame.energies_db)

        # 2. Initial step sizes from thresholds
        steps = threshold_to_steps(thresh)

        # 3. Per-band coefficient slices
        coeffs_bands = [frame.coeffs[lo:hi] for lo, hi in BINS]

        # 4. Adjust steps to fit bit budget (layer-priority degradation)
        steps = _scale_to_budget(coeffs_bands, steps, self.bits_per_frame)

        # 5. Quantize
        quant_coeffs = []
        bl = [0, 0, 0]
        for i, bc in enumerate(coeffs_bands):
            q = np.clip(np.round(bc / steps[i]),
                        -MAX_QUANT, MAX_QUANT).astype(np.int16)
            quant_coeffs.append(q)
            layer = 0 if i < L0 else (1 if i < L0 + L1 else 2)
            bl[layer] += int(_band_bits(bc, steps[i]))

        return QuantizedFrame(
            frame_type=frame.frame_type,
            frame_idx=frame.frame_idx,
            gop_pos=frame.gop_pos,
            is_silence=False,
            quant_coeffs=quant_coeffs,
            steps=steps,
            bits_l0=bl[0],
            bits_l1=bl[1],
            bits_l2=bl[2],
        )

    def dequantize(self, qframe: QuantizedFrame) -> np.ndarray:
        """Reconstruct float64 MDCT coefficients from a QuantizedFrame."""
        coeffs = np.zeros(MDCT_M, dtype=np.float64)
        if qframe.is_silence:
            return coeffs
        for i, (lo, hi) in enumerate(BINS):
            if len(qframe.quant_coeffs[i]):
                coeffs[lo:hi] = qframe.quant_coeffs[i].astype(np.float64) * qframe.steps[i]
        return coeffs

    def roundtrip_snr(self, frame: DSAFrame) -> float:
        """SNR (dB) of the quantize → dequantize roundtrip on MDCT coefficients."""
        qf      = self.quantize(frame)
        rec     = self.dequantize(qf)
        sig_rms = np.sqrt(np.mean(frame.coeffs**2))
        err_rms = np.sqrt(np.mean((frame.coeffs - rec)**2))
        if err_rms < 1e-12:
            return 200.0
        return 20.0 * np.log10(sig_rms / err_rms)

# ─── Verification ─────────────────────────────────────────────────────────────

def verify() -> bool:
    """Verification tests for the perceptual quantizer."""
    analyzer = DSAAnalyzer()
    all_pass = True

    # Test signal: three tones spanning all three layers
    t      = np.arange(MDCT_M * 8) / SAMPLE_RATE
    sig    = (0.5 * np.sin(2 * np.pi * 440  * t)   # 440 Hz  → L0 bass
            + 0.3 * np.sin(2 * np.pi * 2000 * t)   # 2 kHz   → L1 mid
            + 0.2 * np.sin(2 * np.pi * 9000 * t))  # 9 kHz   → L2 high
    frames = analyzer.analyze_samples(sig)
    frame  = frames[1]   # skip first frame (window boundary)

    # ── Test 1: masking raises threshold at active bands ──────────────────────
    thresh  = masking_thresholds(frame.energies_db)
    # The band containing 440Hz has active signal — its threshold must be above
    # the ATH floor (masking from the tone should elevate it).
    tone_band     = next(i for i, (lo, hi) in enumerate(BANDS) if lo <= 440 < hi)
    ath_at_tone   = float(_ATH_DBFS[tone_band])
    thresh_at_tone = float(thresh[tone_band])
    t1_pass       = thresh_at_tone > ath_at_tone
    print(f"  Masking elevates threshold:  {thresh_at_tone:.1f}dB > ATH {ath_at_tone:.1f}dB  "
          f"{'PASS ✓' if t1_pass else 'FAIL'}")
    all_pass = all_pass and t1_pass

    # ── Test 2: step sizes are all positive ───────────────────────────────────
    steps   = threshold_to_steps(thresh)
    t2_pass = bool(np.all(steps > 0))
    print(f"  Step sizes positive:         {'PASS ✓' if t2_pass else 'FAIL'}")
    all_pass = all_pass and t2_pass

    # ── Test 3: roundtrip SNR at multiple bitrates ────────────────────────────
    # Thresholds reflect realistic expectations for a 3-tone test signal.
    # At 6 kbps the budget is very tight; main tones should survive but with
    # coarse quantization. At 24+ kbps quality is limited by masking thresholds.
    # At 12 kbps the 9 kHz L2 component is intentionally degraded first;
    # overall SNR reflects that trade-off rather than a masking model failure.
    snr_floor = {6: 5.0, 12: 8.0, 24: 15.0, 48: 15.0}
    for kbps in (6, 12, 24, 48):
        q   = DSAQuantizer(kbps)
        snr = q.roundtrip_snr(frame)
        qf  = q.quantize(frame)
        ok  = snr >= snr_floor[kbps]
        print(f"  Roundtrip SNR @ {kbps:>2d}kbps:      {snr:5.1f}dB  "
              f"budget={q.bits_per_frame}b  "
              f"used={qf.total_bits}b (L0={qf.bits_l0} L1={qf.bits_l1} L2={qf.bits_l2})  "
              f"{'PASS ✓' if ok else 'FAIL'}")
        all_pass = all_pass and ok

    # ── Test 4: quality scales with bitrate ───────────────────────────────────
    snr_lo  = DSAQuantizer(6).roundtrip_snr(frame)
    snr_hi  = DSAQuantizer(48).roundtrip_snr(frame)
    t4_pass = snr_hi > snr_lo
    print(f"  Quality scales w/ bitrate:   48kbps({snr_hi:.1f}dB) > 6kbps({snr_lo:.1f}dB)  "
          f"{'PASS ✓' if t4_pass else 'FAIL'}")
    all_pass = all_pass and t4_pass

    # ── Test 5: layer priority — L2 step sizes grow faster than L0 ───────────
    # Use a very tight budget (3 kbps) to force degradation.
    # Verify that the budget scaler enlarged L2 steps more than L0 steps,
    # i.e. the L0 quantization is finer (smaller step) relative to L2.
    q_loose = DSAQuantizer(96)    # nearly unconstrained — baseline step sizes
    q_tight = DSAQuantizer(3)
    qf_loose = q_loose.quantize(frame)
    qf_tight = q_tight.quantize(frame)
    # Scale factor applied: tight_step[b] / loose_step[b]
    scale_l0 = float(np.mean(qf_tight.steps[:L0] / qf_loose.steps[:L0]))
    scale_l2 = float(np.mean(qf_tight.steps[L0+L1:] / qf_loose.steps[L0+L1:]))
    t5_pass  = scale_l2 >= scale_l0   # L2 got scaled more aggressively
    print(f"  L2 degraded before L0 @3kbps: L2_scale={scale_l2:.1f}x >= L0_scale={scale_l0:.1f}x  "
          f"{'PASS ✓' if t5_pass else 'FAIL'}")
    all_pass = all_pass and t5_pass

    # ── Test 6: silence frames reconstruct to zero ────────────────────────────
    silent = DSAFrame('S', 99, 3,
                      np.zeros(MDCT_M),
                      np.full(NUM_BANDS, SILENCE_DB),
                      np.zeros(NUM_BANDS), 0.0, True)
    q   = DSAQuantizer()
    rec = q.dequantize(q.quantize(silent))
    t6_pass = bool(np.allclose(rec, 0.0))
    print(f"  Silence → zero reconstruction: {'PASS ✓' if t6_pass else 'FAIL'}")
    all_pass = all_pass and t6_pass

    return all_pass

# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print(f"\n  DSA — Digilog Scalable Audio")
    print(f"  Step 2: Perceptual Quantizer")
    print(f"  {'─'*48}")
    print(f"  Model:   ATH (ISO 226) + simultaneous masking")
    print(f"  Spread:  Bark scale, asymmetric (↑-25dB/Bark, ↓-40dB/Bark)")
    print(f"  Budget:  per-layer priority (L2 → L1 → L0)")
    print(f"  Bands:   {NUM_BANDS}  L0:{L0} L1:{L1} L2:{L2}")
    print(f"  {'─'*48}")

    print(f"\n  Running verification tests...")
    ok = verify()

    if not ok:
        print(f"\n  ✗ Some tests failed.")
        sys.exit(1)

    print(f"\n  All tests passed ✓")

    if len(sys.argv) > 1:
        path = sys.argv[1]
        kbps = float(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_BITRATE
        print(f"\n  Analyzing: {path}  @ {kbps}kbps")
        analyzer = DSAAnalyzer()
        frames, sr, dur = analyzer.analyze_file(path)
        q = DSAQuantizer(kbps)

        snrs, total_bits = [], 0
        for frame in frames:
            if not frame.is_silence:
                snrs.append(q.roundtrip_snr(frame))
                total_bits += q.quantize(frame).total_bits

        if snrs:
            print(f"\n  Quantizer results:")
            print(f"    Frames analyzed:  {len(snrs)}")
            print(f"    SNR mean:         {np.mean(snrs):.1f} dB")
            print(f"    SNR min/max:      {np.min(snrs):.1f} / {np.max(snrs):.1f} dB")
            print(f"    Total bits (est): {total_bits:,}")
            print(f"    Effective kbps:   {total_bits / dur / 1000:.1f}")
    else:
        print(f"\n  Run with audio to quantize:")
        print(f"  python3 dsa_quantizer.py your_song.mp3 [bitrate_kbps]")

    print(f"\n  Ready for Step 3: K-frame encoder ✓")
    print(f"\n  Scan the groove.\n")
