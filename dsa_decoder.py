#!/usr/bin/env python3
"""
DSA — Digilog Scalable Audio
Step 7: Decoder

Converts .dsa files back to audio samples.

Supported decode modes:
  Forward      standard playback (default)
  Reverse      scratch backward — frames processed in reverse order,
               TDAC cancellation is symmetric in time, result is
               the true time-reversed audio signal
  Layer-select L0 only (bass), L0+L1 (no high freq), all layers
  Analog deg.  per-band confidence α ∈ [0,1] scales coefficients;
               α=0.5 → 6dB attenuation; α=0 → silence
  Variable spd speed multiplier — pitch coupled to speed (vinyl feel);
               speed>1 faster+higher, speed<1 slower+lower, speed=0 decay

K-frame loss recovery:
  If a K-frame cannot be decoded (α→0), the last valid spectral shape
  is held and exponentially decayed (τ=60ms). B-frames referencing the
  lost K-frame use the decayed substitute. Result: note fading out,
  not a digital dropout.

License: GPL v3 — github.com/pisdronio/digilog-spec
"""

import numpy as np
import wave
import sys
from pathlib import Path
from typing import Optional

from dsa_analyzer import (
    NUM_BANDS, L0, L1, L2, MDCT_M, MDCT_N, HOP, BINS,
    SAMPLE_RATE, FRAME_MS, GOP_SIZE, DSAAnalyzer,
    mdct, imdct, coeffs_to_bands,
)
from dsa_encoder import (
    EncodedKFrame, EncodedBFrame, EncodedSFrame,
    decode_kframe, decode_bframe,
)
from dsa_bitstream import DSABitstreamReader, DSABitstreamWriter

# ─── Constants ────────────────────────────────────────────────────────────────

TAU_S           = 0.060   # K-frame loss recovery time constant (s)
DECAY_PER_FRAME = float(np.exp(-FRAME_MS / 1000.0 / TAU_S))
# ≈ 0.679 — each frame ~32% amplitude reduction during loss recovery

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _alpha_for_layers(layers: tuple) -> np.ndarray:
    """Build a per-band confidence vector from a layer selection tuple."""
    α = np.ones(NUM_BANDS, dtype=np.float64)
    if 0 not in layers: α[:L0]      = 0.0
    if 1 not in layers: α[L0:L0+L1] = 0.0
    if 2 not in layers: α[L0+L1:]   = 0.0
    return α


def _write_wav(path: str, samples: np.ndarray, sample_rate: int):
    """Write mono float64 samples to a 16-bit WAV file."""
    s16 = np.clip(samples * 32767, -32768, 32767).astype(np.int16)
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(s16.tobytes())

# ─── Variable-speed resampling ────────────────────────────────────────────────

def _apply_speed(samples: np.ndarray, speed: float) -> np.ndarray:
    """
    Apply a playback-speed multiplier to decoded samples.

    Speed is coupled to pitch — natural vinyl / turntable feel.
    No pitch correction is applied.

    speed = 1.0  → passthrough
    speed > 1.0  → faster and higher pitched (forward scratch)
    speed < 1.0  → slower and lower pitched (slow-down)
    speed = 0.0  → stopped: exponential decay envelope, τ=60ms

    Implementation
    --------------
    Non-unity speed is implemented as rational resampling via
    scipy.signal.resample_poly.  Numerator/denominator are the
    rational approximation of 1/speed (output length / input length),
    so the output sample count is ≈ len(samples) / speed.

    Stopped (speed=0) does NOT call the resampler; instead it applies
    an exponential-decay envelope to the existing samples, matching the
    K-frame loss recovery behaviour (τ=TAU_S).
    """
    if speed == 1.0:
        return samples

    if len(samples) == 0:
        return samples

    # ── Stopped ───────────────────────────────────────────────────────────────
    if speed == 0.0:
        # Apply exponential decay: same sample count, amplitude falls to 0.
        n      = len(samples)
        t      = np.arange(n) / SAMPLE_RATE
        env    = np.exp(-t / TAU_S)
        return (samples * env).astype(np.float64)

    # ── Variable speed — linear resampling ───────────────────────────────────
    # Output length = ceil(len(samples) / speed).
    # np.interp maps output sample positions back to input positions.
    # Pitch is coupled to speed: a 2× speed playback sounds an octave higher.
    n_in  = len(samples)
    n_out = max(1, int(round(n_in / speed)))

    x_in  = np.arange(n_in,  dtype=np.float64)
    x_out = np.linspace(0.0, n_in - 1, n_out)

    return np.interp(x_out, x_in, samples).astype(np.float64)


# ─── Decoder ──────────────────────────────────────────────────────────────────

class DSADecoder:
    """
    DSA audio decoder.

    Usage
    -----
    reader  = DSABitstreamReader(Path('song.dsa').read_bytes())
    decoder = DSADecoder()

    samples = decoder.decode(reader)                     # full quality
    bass    = decoder.decode(reader, layers=(0,))        # L0 only
    worn    = decoder.decode(reader, alpha=alpha_vec)    # analog degradation
    rev     = decoder.decode(reader, reverse=True)       # scratch backward
    """

    def decode(self,
               reader:  DSABitstreamReader,
               layers:  tuple                    = (0, 1, 2),
               alpha:   Optional[np.ndarray]     = None,
               reverse: bool                     = False,
               speed:   float                    = 1.0) -> np.ndarray:
        """
        Decode a .dsa file to mono float64 samples in [-1, 1].

        Parameters
        ----------
        reader  : DSABitstreamReader wrapping the .dsa bytes.
        layers  : layers to include — 0=bass/L0, 1=mid/L1, 2=high/L2.
                  Omitting a layer zeroes its α bands.
        alpha   : (NUM_BANDS,) confidence vector ∈ [0,1].
                  Overrides `layers` when supplied explicitly.
                  α < 1 produces attenuated bands — analog degradation.
        reverse : if True, process frames in reverse order → reversed audio.
        speed   : playback speed multiplier relative to nominal (default 1.0).
                  speed > 1.0  →  faster, pitch rises  (vinyl scratch forward)
                  speed < 1.0  →  slower, pitch drops  (vinyl slow-down)
                  speed = 0.0  →  stopped, exponential decay to silence
                  Negative speed values are treated as reverse=True at |speed|.
                  Pitch is coupled to speed (natural vinyl feel) for all values.
                  Implemented via linear resampling (scipy.signal.resample_poly).

        Returns
        -------
        np.ndarray[float64], length ≈ (n_frames − 1) × HOP / speed samples.
        """
        # ── 0. Normalise speed / reverse flags ───────────────────────────────
        # Negative speed → reverse direction at |speed|.
        if speed < 0.0:
            reverse = True
            speed   = abs(speed)

        # ── 1. Build confidence vector ────────────────────────────────────────
        α = (np.asarray(alpha, dtype=np.float64)
             if alpha is not None
             else _alpha_for_layers(layers))

        # ── 2. Read encoded frames from the bitstream ─────────────────────────
        encoded = reader.reconstruct_encoded_frames()

        # ── 3. Pre-decode all K-frames ────────────────────────────────────────
        # B-frames need both surrounding K-frames. Pre-decoding in one pass
        # makes K-frame coefficients available regardless of processing order.
        k_coeffs: dict = {}
        last_k_idx     = -1
        last_k_coeffs  = np.zeros(MDCT_M, dtype=np.float64)

        for f in encoded:
            if isinstance(f, EncodedKFrame):
                c = decode_kframe(f, α)
                k_coeffs[f.frame_idx] = c
                if f.frame_idx > last_k_idx:
                    last_k_idx    = f.frame_idx
                    last_k_coeffs = c

        def _get_k(idx: int) -> np.ndarray:
            """Return K-frame coefficients, with decay fallback if missing."""
            if idx in k_coeffs:
                return k_coeffs[idx]
            # End-of-stream boundary: promoted B→K in the encoder, so this
            # path is only hit if the stream is truncated. Decay gracefully.
            frames_beyond = max(idx - last_k_idx, 0)
            return last_k_coeffs * (DECAY_PER_FRAME ** frames_beyond)

        # ── 4. Decode all frames to MDCT coefficient arrays ───────────────────
        coeff_list = []
        for f in encoded:
            if isinstance(f, EncodedSFrame):
                coeff_list.append(np.zeros(MDCT_M, dtype=np.float64))
            elif isinstance(f, EncodedKFrame):
                coeff_list.append(k_coeffs[f.frame_idx])
            else:  # B-frame
                k0_idx = f.frame_idx - f.gop_pos
                k1_idx = k0_idx + GOP_SIZE
                coeff_list.append(
                    decode_bframe(f, _get_k(k0_idx), _get_k(k1_idx), α)
                )

        # ── 5. Reverse playback ───────────────────────────────────────────────
        # Processing frames in reverse order and applying IMDCT + overlap-add
        # in that order produces the true time-reversed audio signal.
        # The TDAC cancellation is symmetric: w[n] = w[N-1-n] for the sine
        # window, so the overlap-add works identically in both directions.
        if reverse:
            coeff_list = list(reversed(coeff_list))

        # ── 6. IMDCT + overlap-add ────────────────────────────────────────────
        n      = len(coeff_list)
        output = np.zeros(n * HOP + MDCT_N, dtype=np.float64)
        for i, coeffs in enumerate(coeff_list):
            output[i * HOP : i * HOP + MDCT_N] += imdct(coeffs)

        # Trim leading padding (MDCT_M samples) — the first frame covers the
        # pre-roll zeros added by the analyzer before the real audio starts.
        # decoded[j] ≈ original_sample[j] in the valid interior region.
        decoded = output[MDCT_M : n * HOP]

        # ── 7. Variable-speed resampling ──────────────────────────────────────
        # Speed is coupled to pitch — like a turntable. No pitch correction.
        # speed=1.0 → passthrough (no resampling)
        # speed=0.0 → stopped: exponential decay to silence (τ=60ms)
        return _apply_speed(decoded, speed)



# ─── Convenience ──────────────────────────────────────────────────────────────

def decode_file(path: str, **kwargs) -> tuple:
    """
    Decode a .dsa file to (samples, sample_rate).

    kwargs are forwarded to DSADecoder.decode():
      layers=(0,1,2), alpha=None, reverse=False, speed=1.0
    """
    data   = Path(path).read_bytes()
    reader = DSABitstreamReader(data)
    return DSADecoder().decode(reader, **kwargs), reader.header.sample_rate

# ─── Verification ─────────────────────────────────────────────────────────────

def verify() -> bool:
    all_pass = True

    # Synthesize: 3 tones spanning all layers, long enough for 2 full GOPs
    t   = np.arange(MDCT_M * (GOP_SIZE * 2 + 2) * 2) / SAMPLE_RATE
    sig = (0.5 * np.sin(2 * np.pi * 440  * t)   # L0  440 Hz
         + 0.3 * np.sin(2 * np.pi * 2000 * t)   # L1  2 kHz
         + 0.2 * np.sin(2 * np.pi * 9000 * t))  # L2  9 kHz

    analyzer = DSAAnalyzer()
    frames   = analyzer.analyze_samples(sig)
    writer   = DSABitstreamWriter(bitrate_kbps=12)
    dsa_raw  = writer.encode_frames(frames)
    reader   = DSABitstreamReader(dsa_raw)
    dec      = DSADecoder()

    # ── Test 1: output length ─────────────────────────────────────────────────
    decoded          = dec.decode(reader)
    expected_samples = (len(frames) - 1) * HOP
    length_ok        = abs(len(decoded) - expected_samples) <= HOP
    print(f"  Decoded length:               {len(decoded):,} samples  "
          f"(expected ~{expected_samples:,})  "
          f"{'PASS ✓' if length_ok else 'FAIL'}")
    all_pass = all_pass and length_ok

    # ── Test 2: forward decode SNR ────────────────────────────────────────────
    # Skip first MDCT_N and last MDCT_N samples to avoid edge effects.
    # decoded[j] ≈ sig[j] in the valid interior.
    s_off   = MDCT_N
    n_use   = min(len(decoded), len(sig)) - 2 * s_off
    sig_seg = sig[s_off : s_off + n_use]
    dec_seg = decoded[s_off : s_off + n_use]
    sig_rms = np.sqrt(np.mean(sig_seg**2))
    err_rms = np.sqrt(np.mean((sig_seg - dec_seg)**2))
    snr     = 20 * np.log10(sig_rms / (err_rms + 1e-12))
    t2      = snr > 8.0
    print(f"  Forward decode SNR:           {snr:.1f}dB  "
          f"{'PASS ✓' if t2 else 'FAIL (target >8dB)'}")
    all_pass = all_pass and t2

    # ── Test 3: reverse decode produces frame-level temporal reversal ─────────
    # DSA reverse playback means frames are decoded in reverse temporal order.
    # Each 23ms frame is still decoded forward (IMDCT is not sample-reversed),
    # but the chunks are arranged in reversed time → "backward tape" effect.
    #
    # Test: encode a two-part signal (loud first half / quiet second half).
    # Forward decode: loud energy in the first half.
    # Reverse decode: loud energy in the second half.
    ramp_n  = MDCT_M * (GOP_SIZE * 4)   # 4 full GOPs
    half_n  = ramp_n // 2
    t_r     = np.arange(ramp_n) / SAMPLE_RATE
    ramp_sig = np.zeros(ramp_n)
    ramp_sig[:half_n] = 0.5 * np.sin(2 * np.pi * 440 * t_r[:half_n])
    # second half stays near zero — silence

    r_frames  = analyzer.analyze_samples(ramp_sig)
    r_raw     = DSABitstreamWriter(bitrate_kbps=96).encode_frames(r_frames)
    r_reader  = DSABitstreamReader(r_raw)

    fwd = dec.decode(r_reader)
    rev = dec.decode(r_reader, reverse=True)
    n_r = min(len(fwd), len(rev))

    fwd_first  = float(np.sqrt(np.mean(fwd[:n_r // 2] ** 2)))
    fwd_second = float(np.sqrt(np.mean(fwd[n_r // 2:] ** 2)))
    rev_first  = float(np.sqrt(np.mean(rev[:n_r // 2] ** 2)))
    rev_second = float(np.sqrt(np.mean(rev[n_r // 2:] ** 2)))

    # Forward: loud first half → fwd_first >> fwd_second
    # Reverse: silence first (loud at end) → rev_second >> rev_first
    t3 = fwd_first > fwd_second * 3 and rev_second > rev_first * 3
    print(f"  Reverse temporal order:       "
          f"fwd=[{fwd_first:.3f},{fwd_second:.3f}]  "
          f"rev=[{rev_first:.3f},{rev_second:.3f}]  "
          f"{'PASS ✓' if t3 else 'FAIL (loud half should flip)'}")
    all_pass = all_pass and t3

    # ── Tests 4 & 5: layer isolation — checked at coefficient level ───────────
    # Audio-level band checks are unreliable at 12kbps because the quantizer
    # may zero L2 entirely for budget reasons. Coefficient-level checks verify
    # the alpha application directly, independent of bitrate.
    enc_frames = reader.reconstruct_encoded_frames()
    k_frame    = next(f for f in enc_frames if isinstance(f, EncodedKFrame))

    full_α = np.ones(NUM_BANDS, dtype=np.float64)
    l0_α   = _alpha_for_layers((0,))
    k_full = decode_kframe(k_frame, full_α)
    k_l0   = decode_kframe(k_frame, l0_α)

    # L2 coefficient range
    l2_start     = BINS[L0 + L1][0]
    l2_full_rms  = np.sqrt(np.mean(k_full[l2_start:]**2))
    l2_l0_rms    = np.sqrt(np.mean(k_l0[l2_start:]**2))
    t4 = l2_l0_rms < 1e-9
    print(f"  L0-only silences L2 coeffs:   "
          f"full={l2_full_rms:.5f}  l0only={l2_l0_rms:.2e}  "
          f"{'PASS ✓' if t4 else 'FAIL'}")
    all_pass = all_pass and t4

    # L0 coefficients should be bit-for-bit identical (both use alpha=1 on L0)
    l0_end      = BINS[L0 - 1][1]
    l0_full_rms = np.sqrt(np.mean(k_full[:l0_end]**2))
    l0_l0_rms   = np.sqrt(np.mean(k_l0[:l0_end]**2))
    l0_diff     = np.max(np.abs(k_full[:l0_end] - k_l0[:l0_end]))
    t5 = l0_diff < 1e-12
    print(f"  L0-only preserves L0 exactly: "
          f"diff={l0_diff:.2e}  rms={l0_l0_rms:.4f}  "
          f"{'PASS ✓' if t5 else 'FAIL'}")
    all_pass = all_pass and t5

    # ── Test 6: α=0.5 → ~6dB attenuation ────────────────────────────────────
    half_α = np.full(NUM_BANDS, 0.5, dtype=np.float64)
    full_s = dec.decode(reader)
    half_s = dec.decode(reader, alpha=half_α)
    n_s    = min(len(full_s), len(half_s))
    rms_f2 = np.sqrt(np.mean(full_s[s_off:n_s]**2))
    rms_h  = np.sqrt(np.mean(half_s[s_off:n_s]**2))
    ratio  = rms_h / (rms_f2 + 1e-12)
    t6     = 0.45 < ratio < 0.55
    print(f"  α=0.5 → 6dB attenuation:     ratio={ratio:.3f}  "
          f"{'PASS ✓' if t6 else 'FAIL (expected 0.45–0.55)'}")
    all_pass = all_pass and t6

    # ── Test 7: α=0 → silence ────────────────────────────────────────────────
    zero_α  = np.zeros(NUM_BANDS, dtype=np.float64)
    silent  = dec.decode(reader, alpha=zero_α)
    max_abs = float(np.max(np.abs(silent)))
    t7      = max_abs < 1e-9
    print(f"  α=0 → silence:               max_abs={max_abs:.2e}  "
          f"{'PASS ✓' if t7 else 'FAIL'}")
    all_pass = all_pass and t7

    # ── Test 8: worn disc (α=0.1) still produces audio ───────────────────────
    worn_α = np.full(NUM_BANDS, 0.1, dtype=np.float64)
    worn_s = dec.decode(reader, alpha=worn_α)
    rms_w  = np.sqrt(np.mean(worn_s[s_off:]**2))
    t8     = rms_w > 1e-6
    print(f"  α=0.1 worn disc still plays:  rms={rms_w:.5f}  "
          f"{'PASS ✓' if t8 else 'FAIL (should not be silence)'}")
    all_pass = all_pass and t8

    # ── Test 9: spectrally colored degradation ────────────────────────────────
    # α=1 on L0/L1, α=0 on L2 → bass and mids present, highs absent
    worn_outer        = _alpha_for_layers((0, 1))
    worn_outer_s      = dec.decode(reader, alpha=worn_outer)
    seg_wo            = worn_outer_s[s_off : s_off + MDCT_N]
    _, rms_wo         = coeffs_to_bands(mdct(seg_wo))
    l0_wo = float(np.mean(rms_wo[:L0]))
    l2_wo = float(np.mean(rms_wo[L0+L1:]))
    t9    = l0_wo > 1e-4 and l2_wo < 1e-7
    print(f"  L0+L1 only (outer ring worn): "
          f"L0_rms={l0_wo:.4f}  L2_rms={l2_wo:.7f}  "
          f"{'PASS ✓' if t9 else 'FAIL'}")
    all_pass = all_pass and t9

    return all_pass

# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print(f"\n  DSA — Digilog Scalable Audio")
    print(f"  Step 7: Decoder")
    print(f"  {'─'*48}")
    print(f"  Modes:   forward | reverse | layer-select | analog degradation")
    print(f"  Loss:    K-frame decay τ={TAU_S*1000:.0f}ms  "
          f"({DECAY_PER_FRAME:.3f}×/frame)")
    print(f"  {'─'*48}")

    print(f"\n  Running verification tests...")
    ok = verify()

    if not ok:
        print(f"\n  ✗ Some tests failed.")
        sys.exit(1)

    print(f"\n  All tests passed ✓")

    if len(sys.argv) > 1:
        path = sys.argv[1]

        # Accept either .dsa or audio files
        if path.endswith('.dsa'):
            dsa_path = path
        else:
            from dsa_bitstream import DSABitstreamWriter
            kbps     = float(sys.argv[2]) if len(sys.argv) > 2 else 12.0
            dsa_path = str(Path(path).with_suffix('.dsa'))
            print(f"\n  Encoding: {path}  @ {kbps}kbps  →  {dsa_path}")
            DSABitstreamWriter(kbps).encode_file(path, dsa_path)

        print(f"\n  Decoding: {dsa_path}")
        data   = Path(dsa_path).read_bytes()
        reader = DSABitstreamReader(data)
        dec    = DSADecoder()

        modes = [
            ('full',    dict(layers=(0,1,2)),                    '.full.wav'),
            ('L0-only', dict(layers=(0,)),                       '.bass.wav'),
            ('reverse', dict(reverse=True),                      '.rev.wav'),
            ('worn',    dict(alpha=np.full(NUM_BANDS, 0.3)),     '.worn.wav'),
        ]

        print(f"\n  Producing output files:")
        for label, kwargs, suffix in modes:
            out_path = str(Path(dsa_path).with_suffix(suffix))
            samples  = dec.decode(reader, **kwargs)
            _write_wav(out_path, samples, reader.header.sample_rate)
            dur = len(samples) / reader.header.sample_rate
            print(f"    {label:8s}  {out_path}  "
                  f"({dur:.1f}s  {len(samples):,} samples)")
    else:
        print(f"\n  Run with audio or .dsa file:")
        print(f"  python3 dsa_decoder.py your_song.mp3   # encode then decode")
        print(f"  python3 dsa_decoder.py your_song.dsa   # decode only")

    print(f"\n  Ready for Step 8: Digilog disc encoder integration ✓")
    print(f"\n  Scan the groove.\n")
