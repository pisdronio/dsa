#!/usr/bin/env python3
"""
DSA — Digilog Scalable Audio
Step 5: Huffman Entropy Coder

Static Huffman coding for quantized MDCT integer coefficients.
Replaces raw int16 band storage from Steps 3-4.
Matches ENTROPY_FACTOR = 2.0 assumed by dsa_quantizer.py.

Symbol set (34 symbols):
  0 – 31  : coefficient magnitude (direct)
  32      : ESC — magnitude >= 32, followed by 12-bit literal
  33      : EOB — end of band, remaining coefficients are zero

Sign bit (0=positive, 1=negative) appended after each nonzero code.

Band wire format (per band within a layer):
  [step:f32][huff_n:u16][huff_data × huff_n bytes]
  huff_n == 0 means all-zero band (silence/unvoiced).
  Coefficient count per band is fixed by BINS — not stored on wire.

License: GPL v3 — github.com/pisdronio/digilog-spec
"""

import heapq
import struct
import numpy as np
import sys
from typing import Union

from dsa_analyzer import (
    NUM_BANDS, L0, L1, L2, MDCT_M, BINS, SAMPLE_RATE, FRAME_MS,
    DSAAnalyzer, SILENCE_DB,
)
from dsa_quantizer import DSAQuantizer, DEFAULT_BITRATE
from dsa_encoder import (
    LayerData, EncodedKFrame, EncodedBFrame, EncodedSFrame,
    _LAYER_RANGES, _LAYER_SIZES, FRAME_MAGIC, FTYPE_K, FTYPE_B, FTYPE_S,
    DSAEncoder, decode_kframe, decode_bframe,
)

# ─── Symbol definitions ────────────────────────────────────────────────────────

MAX_DIRECT = 31    # magnitudes 0-31 coded directly
SYM_ESC    = 32    # escape: next 12 bits = magnitude >= 32
SYM_EOB    = 33    # end-of-band: remaining coefficients are zero
N_SYMBOLS  = 34

# Laplacian distribution parameter — fitted to 12 kbps DSA audio coefficients.
# P(magnitude = k) ∝ exp(-LAMBDA * k). At 12 kbps roughly 30-40% of
# quantized coefficients are zero; this value gives appropriate code lengths.
LAMBDA = 0.4

# Fixed coefficient count per band (derived from global BINS constant).
# These are the lengths of the int16 arrays stored in LayerData.coeffs.
_BAND_SIZES = [hi - lo for lo, hi in BINS]   # (NUM_BANDS,)

_LAYER_BAND_SIZES = [
    _BAND_SIZES[:L0],
    _BAND_SIZES[L0: L0 + L1],
    _BAND_SIZES[L0 + L1:],
]

# ─── Static Huffman table (built once at import) ───────────────────────────────

def _build_table():
    """
    Build static Huffman encode/decode tables from Laplacian probabilities.

    Returns:
        encode: dict[int, (code_int, n_bits)]  — symbol → bit representation
        decode_root: nested (left, right) | int  — tree for bit-by-bit decode
    """
    # Symbol probabilities
    probs = {}
    for k in range(MAX_DIRECT + 1):
        probs[k] = float(np.exp(-LAMBDA * k))

    # ESC: geometric tail for magnitudes > MAX_DIRECT
    tail = float(np.exp(-LAMBDA * (MAX_DIRECT + 1)) / (1.0 - np.exp(-LAMBDA) + 1e-15))
    probs[SYM_ESC] = max(tail, 1e-9)

    # EOB: approximately once per band — give it the weight of magnitude 4
    probs[SYM_EOB] = float(np.exp(-LAMBDA * 4))

    total = sum(probs.values())
    for k in probs:
        probs[k] /= total

    # Build Huffman tree with min-heap.
    # heap entry: (prob, tiebreak_counter, node)
    # node: int (leaf = symbol) or list [left_node, right_node] (internal)
    heap, cnt = [], 0
    for sym, p in probs.items():
        heapq.heappush(heap, (p, cnt, sym))
        cnt += 1

    while len(heap) > 1:
        p1, _, n1 = heapq.heappop(heap)
        p2, _, n2 = heapq.heappop(heap)
        heapq.heappush(heap, (p1 + p2, cnt, [n1, n2]))
        cnt += 1

    _, _, root = heap[0]

    encode = {}   # sym → (code_int, n_bits)

    def _extract(node, code, bits):
        """Populate encode table and return decode tree node."""
        if isinstance(node, int):
            encode[node] = (code, max(bits, 1))
            return node       # leaf: symbol int
        left, right = node
        lt = _extract(left,  code << 1,       bits + 1)
        rt = _extract(right, (code << 1) | 1, bits + 1)
        return (lt, rt)      # internal: (left_subtree, right_subtree)

    decode_root = _extract(root, 0, 0)
    return encode, decode_root


_ENCODE, _DECODE_ROOT = _build_table()

# ─── Bit I/O ──────────────────────────────────────────────────────────────────

class _BitWriter:
    """MSB-first bit stream writer."""
    __slots__ = ('_buf', '_byte', '_fill')

    def __init__(self):
        self._buf  = bytearray()
        self._byte = 0
        self._fill = 0

    def write(self, value: int, n_bits: int):
        for i in range(n_bits - 1, -1, -1):
            self._byte = (self._byte << 1) | ((value >> i) & 1)
            self._fill += 1
            if self._fill == 8:
                self._buf.append(self._byte)
                self._byte = self._fill = 0

    def flush(self) -> bytes:
        if self._fill:
            self._buf.append(self._byte << (8 - self._fill))
            self._byte = self._fill = 0
        return bytes(self._buf)


class _BitReader:
    """MSB-first bit stream reader. Returns 0 beyond end of data (safe padding)."""
    __slots__ = ('_data', '_bp', '_pp')

    def __init__(self, data: bytes):
        self._data = data
        self._bp   = 0   # byte index
        self._pp   = 0   # bit index within byte (0 = MSB)

    def read_bit(self) -> int:
        if self._bp >= len(self._data):
            return 0
        b = (self._data[self._bp] >> (7 - self._pp)) & 1
        self._pp += 1
        if self._pp == 8:
            self._pp = 0
            self._bp += 1
        return b

    def read(self, n_bits: int) -> int:
        v = 0
        for _ in range(n_bits):
            v = (v << 1) | self.read_bit()
        return v


def _decode_sym(r: _BitReader) -> int:
    """Walk the Huffman decode tree one bit at a time; return symbol."""
    node = _DECODE_ROOT
    while isinstance(node, tuple):
        node = node[r.read_bit()]
    return node

# ─── Band encode / decode ──────────────────────────────────────────────────────

def encode_band(coeffs: np.ndarray) -> bytes:
    """
    Encode one band's int16 coefficients to Huffman bytes.

    - Emits EOB once all remaining coefficients are zero (saves tail bits).
    - Returns empty bytes for all-zero bands (no data written).
    - ESC + 12-bit literal for magnitudes > MAX_DIRECT.
    """
    if len(coeffs) == 0 or not np.any(coeffs):
        return b''

    # Last nonzero position — EOB is emitted after it
    nz_idx  = np.flatnonzero(coeffs)
    last_nz = int(nz_idx[-1])

    w = _BitWriter()
    for i, c in enumerate(coeffs):
        if i > last_nz:
            code, nbits = _ENCODE[SYM_EOB]
            w.write(code, nbits)
            break

        mag = abs(int(c))

        if mag <= MAX_DIRECT:
            code, nbits = _ENCODE[mag]
            w.write(code, nbits)
        else:
            code, nbits = _ENCODE[SYM_ESC]
            w.write(code, nbits)
            w.write(mag, 12)

        if mag:
            w.write(0 if c > 0 else 1, 1)   # sign: 0=positive, 1=negative

    return w.flush()


def decode_band(data: bytes, n_coeffs: int) -> np.ndarray:
    """
    Decode Huffman bytes back to int16 array of length n_coeffs.
    Remaining positions after EOB (or end of data) are zero.
    """
    out = np.zeros(n_coeffs, dtype=np.int16)
    if not data:
        return out
    r = _BitReader(data)
    for i in range(n_coeffs):
        sym = _decode_sym(r)
        if sym == SYM_EOB:
            break
        if sym == SYM_ESC:
            mag = r.read(12)
        else:
            mag = sym
        if mag:
            sign = r.read(1)
            out[i] = np.int16(-mag if sign else mag)
    return out

# ─── Layer encode / decode ─────────────────────────────────────────────────────

def encode_layer_huffman(layer: LayerData) -> bytes:
    """
    Encode a LayerData to bytes with Huffman-coded coefficients.

    Per-band format: [step:f32][huff_n:u16][huff_data × huff_n]
    huff_n == 0 → all-zero band (no huff_data bytes follow).
    Step sizes are preserved as float32 (not entropy-coded — one per band).
    """
    out = []
    for step, c in zip(layer.steps, layer.coeffs):
        out.append(struct.pack('<f', float(step)))
        hb = encode_band(c) if len(c) else b''
        out.append(struct.pack('<H', len(hb)))
        if hb:
            out.append(hb)
    return b''.join(out)


def decode_layer_huffman(data: bytes, offset: int,
                         band_sizes: list) -> tuple:
    """
    Decode a Huffman-encoded layer from data at offset.

    band_sizes: list of int — coefficient count per band (derived from BINS).
    Returns (LayerData, new_offset).
    """
    steps, coeffs = [], []
    for n_c in band_sizes:
        step = struct.unpack_from('<f', data, offset)[0]; offset += 4
        hb_n = struct.unpack_from('<H', data, offset)[0]; offset += 2
        if hb_n:
            c = decode_band(data[offset: offset + hb_n], n_c)
            offset += hb_n
        else:
            c = np.zeros(n_c, dtype=np.int16)
        steps.append(step)
        coeffs.append(c)
    return LayerData(np.array(steps, dtype=np.float32), coeffs), offset

# ─── Frame encode / decode ────────────────────────────────────────────────────

def encode_frame_huffman(frame) -> bytes:
    """
    Serialize any encoded frame (K/B/S) using Huffman-coded layer data.
    Replaces frame.to_bytes() for the Huffman bitstream.
    S-frames carry only a header; their to_bytes() is reused unchanged.
    """
    if isinstance(frame, EncodedSFrame):
        return frame.to_bytes()

    ftype = FTYPE_K if isinstance(frame, EncodedKFrame) else FTYPE_B
    hdr   = FRAME_MAGIC + struct.pack('<BBII', ftype, frame.gop_pos,
                                      frame.frame_idx, 0)
    return hdr + b''.join(
        encode_layer_huffman(layer)
        for layer in frame.layers
    )


def decode_frame_huffman(data: bytes, offset: int) -> tuple:
    """
    Deserialize any frame from a Huffman bitstream.
    Replaces read_frame() for the Huffman bitstream.
    Returns (frame, new_offset).
    """
    assert data[offset: offset + 2] == FRAME_MAGIC, \
        f"Bad frame magic at offset {offset}"
    offset += 2
    ftype, gop_pos = struct.unpack_from('<BB', data, offset); offset += 2
    frame_idx      = struct.unpack_from('<I',  data, offset)[0]; offset += 4
    offset += 4   # reserved uint32

    if ftype == FTYPE_S:
        return EncodedSFrame(frame_idx, gop_pos), offset

    layers = []
    for band_sizes in _LAYER_BAND_SIZES:
        layer, offset = decode_layer_huffman(data, offset, list(band_sizes))
        layers.append(layer)

    if ftype == FTYPE_K:
        return EncodedKFrame(frame_idx, gop_pos, layers), offset
    return EncodedBFrame(frame_idx, gop_pos, layers), offset

# ─── High-level encoder wrapper ───────────────────────────────────────────────

class DSAHuffmanEncoder:
    """
    Full-pipeline encoder producing Huffman-coded frame bytes.

    Chains: DSAAnalyzer → DSAQuantizer → K/B-frame encoders → Huffman coding.
    Output is a raw byte sequence of Huffman-encoded frames with no file header.
    Step 6 (dsa_bitstream.py) wraps this output with the DSA1 file header,
    layer offset table, and CRC32 checksum.
    """

    def __init__(self, bitrate_kbps: float = DEFAULT_BITRATE):
        self.enc = DSAEncoder(bitrate_kbps)

    def encode_frames(self, frames: list) -> bytes:
        """DSAFrame list → Huffman-coded bytes."""
        encoded = self.enc.encode_frames(frames)
        return b''.join(encode_frame_huffman(f) for f in encoded)

    def encode_file(self, path: str) -> tuple:
        """
        Load audio file and encode to Huffman bytes.
        Returns (bytes, n_frames, duration_s).
        """
        analyzer = DSAAnalyzer()
        frames, sr, dur = analyzer.analyze_file(path)
        return self.encode_frames(frames), len(frames), dur

# ─── Verification ─────────────────────────────────────────────────────────────

def verify() -> bool:
    all_pass = True

    # ── Test 1: Huffman table — all symbols have codes ────────────────────────
    missing = [s for s in range(N_SYMBOLS) if s not in _ENCODE]
    t1 = len(missing) == 0
    print(f"  Huffman table complete:       {N_SYMBOLS} symbols  "
          f"{'PASS ✓' if t1 else f'FAIL missing={missing}'}")
    all_pass = all_pass and t1

    # Code lengths: zero gets shortest, ESC gets longest
    zero_len = _ENCODE[0][1]
    eob_len  = _ENCODE[SYM_EOB][1]
    esc_len  = _ENCODE[SYM_ESC][1]
    t1b = zero_len <= 3 and esc_len >= 8
    print(f"  Code lengths (0={zero_len}b EOB={eob_len}b ESC={esc_len}b): "
          f"{'PASS ✓' if t1b else 'FAIL (zero should be short, ESC long)'}")
    all_pass = all_pass and t1b

    # ── Test 2: BitWriter / BitReader roundtrip ───────────────────────────────
    w = _BitWriter()
    test_vals = [(0b10110, 5), (0b1, 1), (0b11001100, 8), (0b101, 3)]
    for v, n in test_vals:
        w.write(v, n)
    raw = w.flush()
    r   = _BitReader(raw)
    rt  = [(r.read(n), n) for _, n in test_vals]
    t2  = all(r_v == o_v for (r_v, _), (o_v, _) in zip(rt, test_vals))
    print(f"  BitWriter/BitReader roundtrip: {'PASS ✓' if t2 else 'FAIL'}")
    all_pass = all_pass and t2

    # ── Test 3: encode_band / decode_band roundtrip ───────────────────────────
    rng = np.random.default_rng(7)
    for desc, arr in [
        ("all-zero",       np.zeros(40, dtype=np.int16)),
        ("sparse",         np.array([0,0,5,-3,0,0,0,1,0,0,0,0,0,-7,0,0,0,2,0,0],
                                    dtype=np.int16)),
        ("dense small",    rng.integers(-4, 5, 64).astype(np.int16)),
        ("large values",   np.array([127, -512, 2047, -2047, 0, 1],
                                    dtype=np.int16)),
    ]:
        enc_b = encode_band(arr)
        rec   = decode_band(enc_b, len(arr))
        ok    = np.array_equal(arr, rec)
        ratio = len(arr) * 2 / (len(enc_b) + 1e-9)
        print(f"  Band roundtrip ({desc:12s}): {'PASS ✓' if ok else 'FAIL'}  "
              f"{len(enc_b)}B vs {len(arr)*2}B raw  ({ratio:.1f}×)")
        all_pass = all_pass and ok

    # ── Test 4: encode/decode layer roundtrip ─────────────────────────────────
    # Build a synthetic QuantizedFrame for L0
    from dsa_quantizer import QuantizedFrame
    n_bands_l0 = L0
    band_sizes  = _LAYER_BAND_SIZES[0]
    fake_steps  = np.ones(n_bands_l0, dtype=np.float32) * 0.01
    fake_coeffs = [rng.integers(-10, 11, sz).astype(np.int16)
                   for sz in band_sizes]
    layer = LayerData(fake_steps, fake_coeffs)
    layer_bytes = encode_layer_huffman(layer)
    layer_rec, end_off = decode_layer_huffman(layer_bytes, 0, list(band_sizes))

    ok_steps  = np.allclose(layer.steps, layer_rec.steps, atol=1e-5)
    ok_coeffs = all(np.array_equal(a, b)
                    for a, b in zip(layer.coeffs, layer_rec.coeffs))
    t4 = ok_steps and ok_coeffs
    print(f"  Layer roundtrip (L0):         {'PASS ✓' if t4 else 'FAIL'}")
    all_pass = all_pass and t4

    # ── Test 5: full K-frame encode → Huffman → decode ────────────────────────
    t    = np.arange(MDCT_M * 10) / SAMPLE_RATE
    sig  = (0.5 * np.sin(2 * np.pi * 440  * t)
          + 0.3 * np.sin(2 * np.pi * 2000 * t)
          + 0.2 * np.sin(2 * np.pi * 9000 * t))
    analyzer = DSAAnalyzer()
    frames   = analyzer.analyze_samples(sig)
    enc      = DSAEncoder(bitrate_kbps=12)
    k_frames = [f for f in frames if f.frame_type == 'K']
    b_frames = [f for f in frames if f.frame_type == 'B']

    from dsa_encoder import KFrameEncoder, BFrameEncoder
    kenc = KFrameEncoder(enc.q)
    benc = BFrameEncoder(enc.q)
    k0, k1 = k_frames[0], k_frames[1]
    b1     = b_frames[0]

    ekf      = kenc.encode(k0)
    huff_raw = encode_frame_huffman(ekf)
    ekf_rec, _ = decode_frame_huffman(huff_raw, 0)
    rec_coeffs = decode_kframe(ekf_rec)
    ref_coeffs = decode_kframe(ekf)
    diff = np.max(np.abs(rec_coeffs - ref_coeffs))
    t5 = diff < 1e-6
    print(f"  K-frame Huffman roundtrip:    max_diff={diff:.2e}  "
          f"{'PASS ✓' if t5 else 'FAIL'}")
    all_pass = all_pass and t5

    # ── Test 6: B-frame Huffman roundtrip ─────────────────────────────────────
    ebf      = benc.encode(b1, k0, k1)
    huff_braw = encode_frame_huffman(ebf)
    ebf_rec, _ = decode_frame_huffman(huff_braw, 0)
    k0c = decode_kframe(kenc.encode(k0))
    k1c = decode_kframe(kenc.encode(k1))
    rec_b  = decode_bframe(ebf_rec, k0c, k1c)
    ref_b  = decode_bframe(ebf,     k0c, k1c)
    diff_b = np.max(np.abs(rec_b - ref_b))
    t6 = diff_b < 1e-6
    print(f"  B-frame Huffman roundtrip:    max_diff={diff_b:.2e}  "
          f"{'PASS ✓' if t6 else 'FAIL'}")
    all_pass = all_pass and t6

    # ── Test 7: full GOP — multi-frame stream roundtrip ───────────────────────
    henc     = DSAHuffmanEncoder(bitrate_kbps=12)
    raw_huff = henc.encode_frames(frames[:9])
    offset   = 0
    n_read   = 0
    while offset < len(raw_huff):
        frm, offset = decode_frame_huffman(raw_huff, offset)
        n_read += 1
    t7 = n_read == 9
    print(f"  GOP stream roundtrip:         {n_read}/9 frames decoded  "
          f"{'PASS ✓' if t7 else 'FAIL'}")
    all_pass = all_pass and t7

    # ── Test 8: compression ratio ≥ 1.5× vs raw int16 ────────────────────────
    enc_raw = enc.encode_to_bytes(frames[:9])   # raw Steps 3-4 format
    ratio   = len(enc_raw) / (len(raw_huff) + 1e-9)
    t8 = ratio >= 1.5
    print(f"  Compression vs raw int16:     {ratio:.2f}×  "
          f"(raw={len(enc_raw):,}B  huffman={len(raw_huff):,}B)  "
          f"{'PASS ✓' if t8 else 'FAIL (target ≥ 1.5×)'}")
    all_pass = all_pass and t8

    return all_pass

# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print(f"\n  DSA — Digilog Scalable Audio")
    print(f"  Step 5: Huffman Entropy Coder")
    print(f"  {'─'*48}")
    print(f"  Distribution: Laplacian  λ={LAMBDA}")
    print(f"  Symbols:      {N_SYMBOLS}  (mag 0-{MAX_DIRECT}, ESC, EOB)")
    print(f"  Bit order:    MSB-first")
    print(f"  {'─'*48}")

    # Show code table summary
    avg_bits = sum(_ENCODE[k][1] * float(np.exp(-LAMBDA * k))
                   for k in range(min(8, MAX_DIRECT + 1)))
    print(f"\n  Code table sample:")
    for sym, label in [(0,'zero'),(1,'±1'),(2,'±2'),(4,'±4'),
                       (8,'±8'),(SYM_EOB,'EOB'),(SYM_ESC,'ESC')]:
        code, nbits = _ENCODE[sym]
        print(f"    {label:6s}  {nbits:2d} bits  {code:0{nbits}b}")

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
        henc = DSAHuffmanEncoder(kbps)
        raw_huff, n_frames, dur = henc.encode_file(path)

        # Compare with raw Steps 3-4 output
        analyzer = DSAAnalyzer()
        frames, sr, _ = analyzer.analyze_file(path)
        enc_raw  = DSAEncoder(kbps).encode_to_bytes(frames)
        ratio    = len(enc_raw) / (len(raw_huff) + 1e-9)

        print(f"\n  Huffman results:")
        print(f"    Frames:        {n_frames:,}")
        print(f"    Raw bytes:     {len(enc_raw):,}  ({len(enc_raw)/dur/125:.1f} kbps)")
        print(f"    Huffman bytes: {len(raw_huff):,}  ({len(raw_huff)/dur/125:.1f} kbps)")
        print(f"    Compression:   {ratio:.2f}×")
    else:
        print(f"\n  Run with audio to encode:")
        print(f"  python3 dsa_huffman.py your_song.mp3 [bitrate_kbps]")

    print(f"\n  Ready for Step 6: Layered bitstream packer ✓")
    print(f"\n  Scan the groove.\n")
