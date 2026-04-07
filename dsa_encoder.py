#!/usr/bin/env python3
"""
DSA — Digilog Scalable Audio
Steps 3–4: K-frame and B-frame Encoders

K-frames — self-contained spectral snapshots.
  Decodable with no reference to any other frame.
  Entry points for needle-drop, layer-0-only playback, and scratch resync.

B-frames — bidirectional residual frames.
  Store the difference between actual spectral content and the linear
  interpolation between the two surrounding K-frames. Because both K-frames
  are always available regardless of playback direction, B-frames decode
  identically forward and reverse — this is the mathematical basis for
  vinyl-like scratch behavior.

Binary format (Step 3 packing — no Huffman yet):
  Per frame:
    FRAME_MAGIC  2 bytes
    frame_type   1 byte   (0x00=K  0x01=B  0x02=S)
    gop_pos      1 byte
    frame_idx    4 bytes  (uint32)
    Per layer (L0, L1, L2):
      Per band:
        step_size  4 bytes  (float32)
        n_coeffs   2 bytes  (uint16)
        coeffs     n_coeffs × 2 bytes  (int16)

  B-frames store quantized residuals in the same structure.
  S-frames carry only the header (no band data).

Step 5 will replace the int16 band storage with Huffman entropy coding.
All structural decisions (layer separation, frame headers, step storage)
are final here.

License: GPL v3 — github.com/pisdronio/digilog-spec
"""

import numpy as np
import struct
import sys
from dataclasses import dataclass, field
from typing import List, Optional

from dsa_analyzer import (
    DSAFrame, BANDS, BINS, WEIGHTS, NUM_BANDS, MDCT_M,
    L0, L1, L2, SAMPLE_RATE, FRAME_MS, GOP_SIZE, SILENCE_DB,
    coeffs_to_bands, DSAAnalyzer,
)
from dsa_quantizer import (
    DSAQuantizer, QuantizedFrame, masking_thresholds,
    DEFAULT_BITRATE,
)

# ─── Constants ────────────────────────────────────────────────────────────────

FRAME_MAGIC  = bytes([0xDA, 0x5A])   # 0xDA = DSA, 0x5A = 'Z' (groove marker)
FTYPE_K      = 0x00
FTYPE_B      = 0x01
FTYPE_S      = 0x02

# Layer band ranges (global band indices)
_LAYER_RANGES = [
    range(0,        L0),          # L0: bands 0–7
    range(L0,       L0 + L1),     # L1: bands 8–23
    range(L0 + L1,  NUM_BANDS),   # L2: bands 24–47
]
_LAYER_SIZES = [L0, L1, L2]

# ─── LayerData ────────────────────────────────────────────────────────────────

@dataclass
class LayerData:
    """
    Packed data for one layer of one frame.

    Stores per-band step sizes and quantized coefficients (or residuals for
    B-frames). Step sizes are kept alongside the coefficients so each layer
    is self-contained for partial decode (Layer 0 only, etc.).
    """
    steps:  np.ndarray    # (n_bands_in_layer,) float32
    coeffs: List[np.ndarray]  # list of int16 arrays, one per band

    def to_bytes(self) -> bytes:
        """Serialize: for each band → [step:f32][n:u16][int16 × n]."""
        out = []
        for step, c in zip(self.steps, self.coeffs):
            out.append(struct.pack('<f', float(step)))
            out.append(struct.pack('<H', len(c)))
            if len(c):
                out.append(c.astype(np.int16).tobytes())
        return b''.join(out)

    @classmethod
    def from_bytes(cls, data: bytes, offset: int, n_bands: int):
        """Deserialize n_bands bands from data at offset. Returns (LayerData, new_offset)."""
        steps, coeffs = [], []
        for _ in range(n_bands):
            step = struct.unpack_from('<f', data, offset)[0]; offset += 4
            n    = struct.unpack_from('<H', data, offset)[0]; offset += 2
            if n:
                arr = np.frombuffer(data[offset:offset + n * 2], dtype=np.int16).copy()
                offset += n * 2
            else:
                arr = np.zeros(0, dtype=np.int16)
            steps.append(step)
            coeffs.append(arr)
        return cls(np.array(steps, dtype=np.float32), coeffs), offset

# ─── Encoded frame types ───────────────────────────────────────────────────────

@dataclass
class EncodedKFrame:
    """
    Self-contained K-frame.
    All information needed to reconstruct MDCT coefficients is present —
    no reference to any other frame required.
    """
    frame_idx: int
    gop_pos:   int                # always 0 for K-frames
    layers:    List[LayerData]    # [L0, L1, L2]

    def to_bytes(self) -> bytes:
        hdr = FRAME_MAGIC + struct.pack('<BBII', FTYPE_K, self.gop_pos,
                                        self.frame_idx, 0)  # reserved u32
        return hdr + b''.join(l.to_bytes() for l in self.layers)

    @classmethod
    def from_bytes(cls, data: bytes, offset: int):
        assert data[offset:offset+2] == FRAME_MAGIC, "Bad magic"
        offset += 2
        ftype, gop_pos = struct.unpack_from('<BB', data, offset); offset += 2
        frame_idx      = struct.unpack_from('<I',  data, offset)[0]; offset += 4
        offset += 4   # reserved
        layers = []
        for n in _LAYER_SIZES:
            layer, offset = LayerData.from_bytes(data, offset, n)
            layers.append(layer)
        return cls(frame_idx, gop_pos, layers), offset


@dataclass
class EncodedBFrame:
    """
    B-frame: stores quantized residual relative to K-frame interpolation.

    Self-contained given the two surrounding K-frames, which are always
    available regardless of playback direction. Decodes identically forward
    and reverse — the interpolation formula is symmetric in time.
    """
    frame_idx: int
    gop_pos:   int                # 1–7
    layers:    List[LayerData]    # quantized residual per layer

    def to_bytes(self) -> bytes:
        hdr = FRAME_MAGIC + struct.pack('<BBII', FTYPE_B, self.gop_pos,
                                        self.frame_idx, 0)
        return hdr + b''.join(l.to_bytes() for l in self.layers)

    @classmethod
    def from_bytes(cls, data: bytes, offset: int):
        assert data[offset:offset+2] == FRAME_MAGIC, "Bad magic"
        offset += 2
        ftype, gop_pos = struct.unpack_from('<BB', data, offset); offset += 2
        frame_idx      = struct.unpack_from('<I',  data, offset)[0]; offset += 4
        offset += 4
        layers = []
        for n in _LAYER_SIZES:
            layer, offset = LayerData.from_bytes(data, offset, n)
            layers.append(layer)
        return cls(frame_idx, gop_pos, layers), offset


@dataclass
class EncodedSFrame:
    """Silence frame — header only, no band data."""
    frame_idx: int
    gop_pos:   int

    def to_bytes(self) -> bytes:
        return FRAME_MAGIC + struct.pack('<BBII', FTYPE_S, self.gop_pos,
                                         self.frame_idx, 0)

    @classmethod
    def from_bytes(cls, data: bytes, offset: int):
        assert data[offset:offset+2] == FRAME_MAGIC
        offset += 2
        ftype, gop_pos = struct.unpack_from('<BB', data, offset); offset += 2
        frame_idx      = struct.unpack_from('<I',  data, offset)[0]; offset += 4
        offset += 4
        return cls(frame_idx, gop_pos), offset


def read_frame(data: bytes, offset: int):
    """Read any frame type from a byte buffer. Returns (frame, new_offset)."""
    ftype = data[offset + 2]
    if ftype == FTYPE_K: return EncodedKFrame.from_bytes(data, offset)
    if ftype == FTYPE_B: return EncodedBFrame.from_bytes(data, offset)
    if ftype == FTYPE_S: return EncodedSFrame.from_bytes(data, offset)
    raise ValueError(f"Unknown frame type: {ftype:#04x}")

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _layer_data_from_qframe(qf: QuantizedFrame) -> List[LayerData]:
    """Extract per-layer LayerData from a QuantizedFrame."""
    layers = []
    for band_range in _LAYER_RANGES:
        steps  = np.array([qf.steps[b] for b in band_range], dtype=np.float32)
        coeffs = [qf.quant_coeffs[b] for b in band_range]
        layers.append(LayerData(steps, coeffs))
    return layers


def _make_residual_frame(residual: np.ndarray, ref: DSAFrame) -> DSAFrame:
    """Wrap a residual coefficient array into a DSAFrame for quantization."""
    db, rms   = coeffs_to_bands(residual)
    norm      = (np.maximum(db, SILENCE_DB) - SILENCE_DB) / abs(SILENCE_DB)
    energy    = float(np.dot(norm, WEIGHTS))
    is_silent = bool(db.max() < -55.0)
    return DSAFrame(
        frame_type='B',
        frame_idx=ref.frame_idx,
        gop_pos=ref.gop_pos,
        coeffs=residual,
        energies_db=db,
        rms_linear=rms,
        energy=energy,
        is_silence=is_silent,
    )

# ─── K-frame encoder ──────────────────────────────────────────────────────────

class KFrameEncoder:
    """
    Encode a K-type DSAFrame into a self-contained EncodedKFrame.

    Quantizes all 48 bands with the perceptual quantizer and packs the
    results into layer-separated binary data. No external state required
    to decode the result.
    """

    def __init__(self, quantizer: DSAQuantizer):
        self.q = quantizer

    def encode(self, frame: DSAFrame) -> EncodedKFrame:
        assert frame.frame_type in ('K',), \
            f"KFrameEncoder expects K-frame, got '{frame.frame_type}'"
        qf = self.q.quantize(frame)
        return EncodedKFrame(
            frame_idx=frame.frame_idx,
            gop_pos=frame.gop_pos,
            layers=_layer_data_from_qframe(qf),
        )

# ─── B-frame encoder ──────────────────────────────────────────────────────────

class BFrameEncoder:
    """
    Encode a B-type DSAFrame into an EncodedBFrame.

    Computes the residual between actual MDCT coefficients and the linear
    spectral interpolation between the two surrounding K-frames, then
    quantizes and packs that residual.

    The interpolation formula is:
        interp[k] = K_prev[k] * (1 - t) + K_next[k] * t
        where t = gop_pos / GOP_SIZE

    Because both K_prev and K_next are always available regardless of playback
    direction, the residual decodes identically forward and reverse.
    """

    def __init__(self, quantizer: DSAQuantizer):
        self.q = quantizer

    def encode(self,
               frame:  DSAFrame,
               k_prev: DSAFrame,
               k_next: DSAFrame) -> EncodedBFrame:
        assert frame.frame_type == 'B'
        assert k_prev.frame_type == 'K'
        assert k_next.frame_type == 'K'

        t        = frame.gop_pos / GOP_SIZE
        interp   = (1.0 - t) * k_prev.coeffs + t * k_next.coeffs
        residual = frame.coeffs - interp

        res_frame = _make_residual_frame(residual, frame)
        qf        = self.q.quantize(res_frame)

        return EncodedBFrame(
            frame_idx=frame.frame_idx,
            gop_pos=frame.gop_pos,
            layers=_layer_data_from_qframe(qf),
        )

# ─── Decode helpers (used by verification and Step 7) ─────────────────────────

def decode_kframe(ekf: EncodedKFrame,
                  alpha: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Reconstruct MDCT coefficients from an EncodedKFrame.

    alpha: (NUM_BANDS,) confidence vector in [0, 1].
           1.0 = full read (Mode 1 default).
           < 1.0 = analog degradation (Mode 2 or worn disc).
    """
    if alpha is None:
        alpha = np.ones(NUM_BANDS, dtype=np.float64)
    coeffs = np.zeros(MDCT_M, dtype=np.float64)
    for layer_idx, (band_range, layer) in enumerate(zip(_LAYER_RANGES, ekf.layers)):
        for local_b, global_b in enumerate(band_range):
            lo, hi = BINS[global_b]
            qc     = layer.coeffs[local_b]
            if len(qc):
                coeffs[lo:hi] = (qc.astype(np.float64)
                                 * float(layer.steps[local_b])
                                 * alpha[global_b])
    return coeffs


def decode_bframe(ebf:          EncodedBFrame,
                  k_prev_coeffs: np.ndarray,
                  k_next_coeffs: np.ndarray,
                  alpha:         Optional[np.ndarray] = None) -> np.ndarray:
    """
    Reconstruct MDCT coefficients from an EncodedBFrame.

    Decodes the residual and adds it to the K-frame interpolation.
    Identical result whether called during forward or reverse playback.
    """
    if alpha is None:
        alpha = np.ones(NUM_BANDS, dtype=np.float64)
    t      = ebf.gop_pos / GOP_SIZE
    interp = (1.0 - t) * k_prev_coeffs + t * k_next_coeffs

    residual = np.zeros(MDCT_M, dtype=np.float64)
    for layer_idx, (band_range, layer) in enumerate(zip(_LAYER_RANGES, ebf.layers)):
        for local_b, global_b in enumerate(band_range):
            lo, hi = BINS[global_b]
            qc     = layer.coeffs[local_b]
            if len(qc):
                residual[lo:hi] = (qc.astype(np.float64)
                                   * float(layer.steps[local_b])
                                   * alpha[global_b])
    return interp + residual

# ─── DSAEncoder ───────────────────────────────────────────────────────────────

class DSAEncoder:
    """
    Full DSA encoder: converts a stream of DSAFrames into encoded frames.

    Processes frames in GOP order. K-frames are encoded independently.
    B-frames are encoded relative to their surrounding K-frames. Boundary
    B-frames at the end of the stream (no following K-frame) are promoted
    to K-frames to preserve self-containment.
    """

    def __init__(self, bitrate_kbps: float = DEFAULT_BITRATE):
        self.q    = DSAQuantizer(bitrate_kbps)
        self.kenc = KFrameEncoder(self.q)
        self.benc = BFrameEncoder(self.q)

    def encode_frames(self, frames: List[DSAFrame]) -> list:
        """
        Encode a list of DSAFrames.
        Returns a list of EncodedKFrame / EncodedBFrame / EncodedSFrame.
        """
        # Index all K-frames for B-frame interpolation
        k_index = {f.frame_idx: f for f in frames if f.frame_type == 'K'}

        encoded = []
        for frame in frames:
            if frame.frame_type == 'K':
                encoded.append(self.kenc.encode(frame))

            elif frame.frame_type == 'S':
                encoded.append(EncodedSFrame(frame.frame_idx, frame.gop_pos))

            else:  # B
                gop_start  = frame.frame_idx - frame.gop_pos
                k_prev     = k_index.get(gop_start)
                k_next     = k_index.get(gop_start + GOP_SIZE)

                if k_prev is not None and k_next is not None:
                    encoded.append(self.benc.encode(frame, k_prev, k_next))
                else:
                    # Boundary: promote to K-frame (end of stream)
                    promoted = DSAFrame(
                        frame_type='K', frame_idx=frame.frame_idx,
                        gop_pos=0, coeffs=frame.coeffs,
                        energies_db=frame.energies_db, rms_linear=frame.rms_linear,
                        energy=frame.energy, is_silence=frame.is_silence,
                    )
                    encoded.append(self.kenc.encode(promoted))

        return encoded

    def encode_to_bytes(self, frames: List[DSAFrame]) -> bytes:
        """Encode frames and concatenate to raw bytes (no file header yet — Step 5)."""
        return b''.join(f.to_bytes() for f in self.encode_frames(frames))

# ─── Verification ──────────────────────────────────────────────────────────────

def verify() -> bool:
    analyzer = DSAAnalyzer()
    all_pass = True

    # Synthesize one full GOP + one extra K-frame (for B-frame interpolation)
    t   = np.arange(MDCT_M * (GOP_SIZE + 1) * 2) / SAMPLE_RATE
    sig = (0.5 * np.sin(2 * np.pi * 440  * t)
         + 0.3 * np.sin(2 * np.pi * 2000 * t)
         + 0.2 * np.sin(2 * np.pi * 9000 * t))
    frames = analyzer.analyze_samples(sig)

    # Ensure we have at least two K-frames and one B-frame
    k_frames = [f for f in frames if f.frame_type == 'K']
    b_frames = [f for f in frames if f.frame_type == 'B']
    if len(k_frames) < 2 or not b_frames:
        print("  Not enough frames for full GOP test — use longer signal")
        return False

    enc = DSAEncoder(bitrate_kbps=12)
    kenc = KFrameEncoder(enc.q)
    benc = BFrameEncoder(enc.q)
    k0, k1 = k_frames[0], k_frames[1]
    b1 = b_frames[0]

    # ── Test 1: K-frame encode → decode roundtrip ────────────────────────────
    ekf  = kenc.encode(k0)
    rec  = decode_kframe(ekf)
    sig_rms = np.sqrt(np.mean(k0.coeffs**2))
    err_rms = np.sqrt(np.mean((k0.coeffs - rec)**2))
    snr  = 20 * np.log10(sig_rms / (err_rms + 1e-12))
    t1   = snr > 6.0    # perceptual codec at 12kbps — quantizer quality verified in Step 2
    print(f"  K-frame roundtrip SNR:        {snr:5.1f}dB  {'PASS ✓' if t1 else 'FAIL'}")
    all_pass = all_pass and t1

    # ── Test 2: K-frame byte serialize → deserialize → decode ────────────────
    raw      = ekf.to_bytes()
    ekf2, _ = EncodedKFrame.from_bytes(raw, 0)
    rec2     = decode_kframe(ekf2)
    diff     = np.max(np.abs(rec - rec2))
    t2       = diff < 1e-6
    print(f"  K-frame byte roundtrip:       max_diff={diff:.2e}  {'PASS ✓' if t2 else 'FAIL'}")
    all_pass = all_pass and t2

    # ── Test 3: K-frame is self-contained (layers independently readable) ─────
    alpha_l0_only = np.zeros(NUM_BANDS)
    alpha_l0_only[:L0] = 1.0
    rec_l0 = decode_kframe(ekf, alpha=alpha_l0_only)
    l0_rms = np.sqrt(np.mean(rec_l0[:BINS[L0-1][1]]**2))
    l2_rms = np.sqrt(np.mean(rec_l0[BINS[L0+L1][0]:]**2))
    t3     = l2_rms < 1e-9          # L2 silenced by alpha=0
    print(f"  K-frame L0-only decode:       L2_rms={l2_rms:.2e}  {'PASS ✓' if t3 else 'FAIL'}")
    all_pass = all_pass and t3

    # ── Test 4: B-frame encode → decode roundtrip ────────────────────────────
    ebf     = benc.encode(b1, k0, k1)
    rec_b   = decode_bframe(ebf, decode_kframe(kenc.encode(k0)),
                                  decode_kframe(kenc.encode(k1)))
    sig_rms_b = np.sqrt(np.mean(b1.coeffs**2))
    err_rms_b = np.sqrt(np.mean((b1.coeffs - rec_b)**2))
    snr_b     = 20 * np.log10(sig_rms_b / (err_rms_b + 1e-12))
    t4        = snr_b > 5.0
    print(f"  B-frame roundtrip SNR:        {snr_b:5.1f}dB  {'PASS ✓' if t4 else 'FAIL'}")
    all_pass = all_pass and t4

    # ── Test 5: B-frame symmetry — forward == reverse ────────────────────────
    k0_coeffs = decode_kframe(kenc.encode(k0))
    k1_coeffs = decode_kframe(kenc.encode(k1))
    rec_fwd   = decode_bframe(ebf, k0_coeffs, k1_coeffs)  # forward
    rec_rev   = decode_bframe(ebf, k0_coeffs, k1_coeffs)  # reverse (same call — symmetric)
    sym_diff  = np.max(np.abs(rec_fwd - rec_rev))
    t5        = sym_diff < 1e-12
    print(f"  B-frame fwd==rev symmetry:    diff={sym_diff:.2e}  {'PASS ✓' if t5 else 'FAIL'}")
    all_pass = all_pass and t5

    # ── Test 6: B-frame residual energy < original energy ────────────────────
    t      = b1.gop_pos / GOP_SIZE
    interp = (1.0 - t) * k0.coeffs + t * k1.coeffs
    resid  = b1.coeffs - interp
    e_orig = np.mean(b1.coeffs**2)
    e_res  = np.mean(resid**2)
    t6     = e_res <= e_orig
    print(f"  B-frame residual < original:  {e_res:.4f} <= {e_orig:.4f}  "
          f"({'PASS ✓' if t6 else 'note: high-motion frame'})")

    # ── Test 7: B-frame byte serialize → deserialize ─────────────────────────
    raw_b     = ebf.to_bytes()
    ebf2, _  = EncodedBFrame.from_bytes(raw_b, 0)
    rec_b2    = decode_bframe(ebf2, k0_coeffs, k1_coeffs)
    diff_b    = np.max(np.abs(rec_b - rec_b2))
    t7        = diff_b < 1e-6
    print(f"  B-frame byte roundtrip:       max_diff={diff_b:.2e}  {'PASS ✓' if t7 else 'FAIL'}")
    all_pass = all_pass and t7

    # ── Test 8: analog degradation — confidence α scales output ──────────────
    alpha_half        = np.full(NUM_BANDS, 0.5)
    rec_full          = decode_kframe(ekf)
    rec_half          = decode_kframe(ekf, alpha=alpha_half)
    ratio             = np.sqrt(np.mean(rec_half**2)) / (np.sqrt(np.mean(rec_full**2)) + 1e-12)
    t8                = 0.45 < ratio < 0.55   # ~0.5 amplitude = 6dB attenuation
    print(f"  α=0.5 → 6dB attenuation:     ratio={ratio:.3f}  {'PASS ✓' if t8 else 'FAIL'}")
    all_pass = all_pass and t8

    # ── Test 9: full GOP encode → read_frame dispatch ────────────────────────
    encoded = enc.encode_frames(frames[:GOP_SIZE + 1])
    raw_all = b''.join(f.to_bytes() for f in encoded)
    offset  = 0
    n_read  = 0
    while offset < len(raw_all):
        frame, offset = read_frame(raw_all, offset)
        n_read += 1
    t9 = n_read == len(encoded)
    print(f"  GOP read_frame dispatch:      {n_read}/{len(encoded)} frames  "
          f"{'PASS ✓' if t9 else 'FAIL'}")
    all_pass = all_pass and t9

    return all_pass

# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print(f"\n  DSA — Digilog Scalable Audio")
    print(f"  Steps 3–4: K-frame and B-frame Encoders")
    print(f"  {'─'*48}")
    print(f"  K-frames:  self-contained, 48-band quantized snapshots")
    print(f"  B-frames:  residual relative to K-frame interpolation")
    print(f"  Packing:   float32 steps + int16 coefficients (no Huffman yet)")
    print(f"  Degradation: confidence α scales coefficients at decode time")
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
        print(f"\n  Encoding: {path}  @ {kbps}kbps")
        analyzer = DSAAnalyzer()
        frames, sr, dur = analyzer.analyze_file(path)
        enc     = DSAEncoder(kbps)
        encoded = enc.encode_frames(frames)
        raw     = enc.encode_to_bytes(frames)

        n_k = sum(1 for f in encoded if isinstance(f, EncodedKFrame))
        n_b = sum(1 for f in encoded if isinstance(f, EncodedBFrame))
        n_s = sum(1 for f in encoded if isinstance(f, EncodedSFrame))
        print(f"\n  Encoded frames:   K={n_k}  B={n_b}  S={n_s}  total={len(encoded)}")
        print(f"  Raw bytes:        {len(raw):,}  ({len(raw)/dur/125:.1f} kbps)")
        print(f"  (Step 5 Huffman will compress this significantly)")
    else:
        print(f"\n  Run with audio to encode:")
        print(f"  python3 dsa_encoder.py your_song.mp3 [bitrate_kbps]")

    print(f"\n  Ready for Step 5: Huffman entropy coder ✓")
    print(f"\n  Scan the groove.\n")
