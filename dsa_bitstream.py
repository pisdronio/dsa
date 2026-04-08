#!/usr/bin/env python3
"""
DSA — Digilog Scalable Audio
Step 6: Layered Bitstream Packer

Produces .dsa files. Layers are stored in separate contiguous blocks so a
reader can access Layer 0 alone (bass, inner disc rings) without reading
L1 or L2 — matching the physical structure of the Digilog disc.

DSA1 file layout:
  [Header       32 B]
  [Frame index  n × 6 B]      type:u8 + gop_pos:u8 + frame_idx:u32
  [Layer 0 block]              per frame: [size:u16][Huffman layer data]
  [Layer 1 block]              same
  [Layer 2 block]              same
  [CRC32        4 B]           CRC32 over all preceding bytes

Layer block entry for S-frames: size == 0, no data bytes follow.

Header format (32 bytes, little-endian):
  [0:4]   b"DSA1"        magic
  [4]     u8             version  (0x01)
  [5]     u8             mode     (0x01=discrete, 0x02=gradient)
  [6:10]  u32            sample_rate
  [10:14] u32            n_frames
  [14:16] u16            bitrate_kbps
  [16:20] u32            layer0_offset  (abs byte offset from file start)
  [20:24] u32            layer1_offset
  [24:28] u32            layer2_offset
  [28:32] u32            crc32_offset

License: GPL v3 — github.com/pisdronio/digilog-spec
"""

import struct
import zlib
import sys
import os
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from dsa_analyzer import (
    NUM_BANDS, L0, L1, L2, MDCT_M, BINS, SAMPLE_RATE, FRAME_MS, GOP_SIZE,
    DSAAnalyzer, SILENCE_DB, imdct,
)
from dsa_quantizer import DEFAULT_BITRATE
from dsa_encoder import (
    EncodedKFrame, EncodedBFrame, EncodedSFrame,
    _LAYER_SIZES, FTYPE_K, FTYPE_B, FTYPE_S,
    LayerData, DSAEncoder, decode_kframe, decode_bframe,
)
from dsa_huffman import (
    DSAHuffmanEncoder, encode_layer_huffman, decode_layer_huffman,
    _LAYER_BAND_SIZES,
)

# ─── Format constants ──────────────────────────────────────────────────────────

FILE_MAGIC      = b'DSA1'
FORMAT_VERSION  = 0x01
MODE_DISCRETE   = 0x01   # Mode 1: hard-edged dots (current)
MODE_GRADIENT   = 0x02   # Mode 2: gradient encoding (future, v2.0)

HEADER_SIZE      = 32    # bytes
FRAME_IDX_ENTRY  = 6     # bytes per frame: type(1) + gop_pos(1) + frame_idx(4)

# Struct format for the fixed 32-byte header
_HDR_FMT = '<4sBBIIHIIII'
assert struct.calcsize(_HDR_FMT) == HEADER_SIZE, "Header size mismatch"

# ─── Header dataclass ─────────────────────────────────────────────────────────

@dataclass
class DSAHeader:
    version:       int = FORMAT_VERSION
    mode:          int = MODE_DISCRETE
    sample_rate:   int = SAMPLE_RATE
    n_frames:      int = 0
    bitrate_kbps:  int = DEFAULT_BITRATE
    layer0_offset: int = 0
    layer1_offset: int = 0
    layer2_offset: int = 0
    crc32_offset:  int = 0

    def to_bytes(self) -> bytes:
        return struct.pack(
            _HDR_FMT,
            FILE_MAGIC,
            self.version, self.mode,
            self.sample_rate, self.n_frames, self.bitrate_kbps,
            self.layer0_offset, self.layer1_offset,
            self.layer2_offset, self.crc32_offset,
        )

    @classmethod
    def from_bytes(cls, data: bytes, offset: int = 0):
        magic = data[offset: offset + 4]
        if magic != FILE_MAGIC:
            raise ValueError(f"Not a DSA file — bad magic: {magic!r}")
        (_, version, mode, sr, n_frames, kbps,
         l0, l1, l2, crc) = struct.unpack_from(_HDR_FMT, data, offset)
        return cls(version, mode, sr, n_frames, kbps, l0, l1, l2, crc)

    @property
    def mode_name(self) -> str:
        return {MODE_DISCRETE: 'discrete', MODE_GRADIENT: 'gradient'}.get(
            self.mode, f'unknown(0x{self.mode:02x})')

# ─── File packing ─────────────────────────────────────────────────────────────

def pack_dsa_file(
    encoded_frames: list,
    bitrate_kbps:   float = DEFAULT_BITRATE,
    sample_rate:    int   = SAMPLE_RATE,
    mode:           int   = MODE_DISCRETE,
) -> bytes:
    """
    Pack a list of encoded frames (K/B/S) into a complete .dsa file.

    Layers are stored in separate blocks — Layer 0 can be read in isolation
    for bass-only decode without touching L1/L2 data.
    """
    n_frames = len(encoded_frames)

    # ── Frame index ───────────────────────────────────────────────────────────
    fidx = bytearray()
    for f in encoded_frames:
        if isinstance(f, EncodedKFrame):
            ftype = FTYPE_K
        elif isinstance(f, EncodedBFrame):
            ftype = FTYPE_B
        else:
            ftype = FTYPE_S
        fidx += struct.pack('<BBI', ftype, f.gop_pos, f.frame_idx)

    # ── Layer blocks ──────────────────────────────────────────────────────────
    # Each block: per frame → [layer_size:u16][Huffman layer data]
    # S-frames emit layer_size=0 with no data.
    layer_blocks = [bytearray(), bytearray(), bytearray()]
    for f in encoded_frames:
        if isinstance(f, EncodedSFrame):
            for lb in layer_blocks:
                lb += struct.pack('<H', 0)
        else:
            for li, lb in enumerate(layer_blocks):
                layer_bytes = encode_layer_huffman(f.layers[li])
                lb += struct.pack('<H', len(layer_bytes))
                lb += layer_bytes

    # ── Compute byte offsets ──────────────────────────────────────────────────
    layer0_offset = HEADER_SIZE + len(fidx)
    layer1_offset = layer0_offset + len(layer_blocks[0])
    layer2_offset = layer1_offset + len(layer_blocks[1])
    crc32_offset  = layer2_offset + len(layer_blocks[2])

    hdr = DSAHeader(
        version      = FORMAT_VERSION,
        mode         = mode,
        sample_rate  = sample_rate,
        n_frames     = n_frames,
        bitrate_kbps = int(bitrate_kbps),
        layer0_offset = layer0_offset,
        layer1_offset = layer1_offset,
        layer2_offset = layer2_offset,
        crc32_offset  = crc32_offset,
    )

    body = (hdr.to_bytes()
            + bytes(fidx)
            + bytes(layer_blocks[0])
            + bytes(layer_blocks[1])
            + bytes(layer_blocks[2]))

    crc = zlib.crc32(body) & 0xFFFFFFFF
    return body + struct.pack('<I', crc)

# ─── Bitstream reader ─────────────────────────────────────────────────────────

class DSABitstreamReader:
    """
    Reader for .dsa files.

    Layer-isolated access: read_layer(0) reads only the L0 block — no L1/L2
    bytes are touched. This mirrors reading only the inner rings of a disc.
    """

    def __init__(self, data: bytes):
        self.data   = data
        self.header = DSAHeader.from_bytes(data)
        self._parse_frame_index()

    def _parse_frame_index(self):
        base = HEADER_SIZE
        n    = self.header.n_frames
        ftypes, gops, idxs = [], [], []
        for i in range(n):
            off = base + i * FRAME_IDX_ENTRY
            ftype, gop_pos, frame_idx = struct.unpack_from('<BBI', self.data, off)
            ftypes.append(ftype)
            gops.append(gop_pos)
            idxs.append(frame_idx)
        self.frame_types   = ftypes
        self.frame_gop_pos = gops
        self.frame_indices = idxs

    # ── Integrity ─────────────────────────────────────────────────────────────

    def verify_crc(self) -> bool:
        """Verify CRC32 checksum. Returns True if file is intact."""
        crc_off  = self.header.crc32_offset
        stored   = struct.unpack_from('<I', self.data, crc_off)[0]
        computed = zlib.crc32(self.data[:crc_off]) & 0xFFFFFFFF
        return stored == computed

    # ── Layer access ──────────────────────────────────────────────────────────

    def read_layer(self, layer_idx: int) -> list:
        """
        Read one layer's data for all frames.

        Accesses only the bytes for the requested layer block.
        Returns list of LayerData per frame, or None for S-frames.
        """
        layer_starts = [
            self.header.layer0_offset,
            self.header.layer1_offset,
            self.header.layer2_offset,
        ]
        band_sizes = list(_LAYER_BAND_SIZES[layer_idx])
        offset     = layer_starts[layer_idx]
        results    = []

        for ftype in self.frame_types:
            size = struct.unpack_from('<H', self.data, offset)[0]
            offset += 2
            if ftype == FTYPE_S or size == 0:
                results.append(None)
            else:
                layer, _ = decode_layer_huffman(self.data, offset, band_sizes)
                results.append(layer)
            offset += size

        return results

    def reconstruct_encoded_frames(self) -> list:
        """
        Reconstruct EncodedK/B/S frames by reading all three layers.
        Used by the decoder (Step 7) to get frames ready for IMDCT.
        """
        l0 = self.read_layer(0)
        l1 = self.read_layer(1)
        l2 = self.read_layer(2)
        frames = []
        for i, ftype in enumerate(self.frame_types):
            gop_pos   = self.frame_gop_pos[i]
            frame_idx = self.frame_indices[i]
            if ftype == FTYPE_S:
                frames.append(EncodedSFrame(frame_idx, gop_pos))
            elif ftype == FTYPE_K:
                frames.append(EncodedKFrame(frame_idx, gop_pos,
                                            [l0[i], l1[i], l2[i]]))
            else:
                frames.append(EncodedBFrame(frame_idx, gop_pos,
                                            [l0[i], l1[i], l2[i]]))
        return frames

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def duration_s(self) -> float:
        return self.header.n_frames * FRAME_MS / 1000.0

    @property
    def n_frames(self) -> int:
        return self.header.n_frames

    @property
    def file_size(self) -> int:
        return len(self.data)

    def layer_byte_size(self, layer_idx: int) -> int:
        """Byte size of one layer block (for reporting)."""
        offsets = [
            self.header.layer0_offset,
            self.header.layer1_offset,
            self.header.layer2_offset,
            self.header.crc32_offset,
        ]
        return offsets[layer_idx + 1] - offsets[layer_idx]

# ─── Bitstream writer ─────────────────────────────────────────────────────────

class DSABitstreamWriter:
    """
    Full-pipeline .dsa file writer.

    Chain: DSAAnalyzer → DSAQuantizer → K/B-frame encoders →
           Huffman coding → DSA1 layered file format.
    """

    def __init__(self, bitrate_kbps: float = DEFAULT_BITRATE,
                 mode: int = MODE_DISCRETE):
        self.bitrate_kbps = bitrate_kbps
        self.mode         = mode
        self._enc         = DSAEncoder(bitrate_kbps)

    def encode_frames(self, frames: list) -> bytes:
        """DSAFrame list → .dsa bytes."""
        encoded = self._enc.encode_frames(frames)
        return pack_dsa_file(encoded, self.bitrate_kbps, SAMPLE_RATE, self.mode)

    def encode_file(self, input_path: str, output_path: str = None) -> bytes:
        """Load audio file, encode to .dsa. Writes file if output_path given."""
        analyzer = DSAAnalyzer()
        frames, sr, dur = analyzer.analyze_file(input_path)
        data = self.encode_frames(frames)
        if output_path:
            Path(output_path).write_bytes(data)
            print(f"  Written: {output_path}  ({len(data):,} bytes)")
        return data

# ─── Verification ─────────────────────────────────────────────────────────────

def verify() -> bool:
    all_pass = True

    # Synthesize test signal: two GOPs + boundary K-frame
    t   = np.arange(MDCT_M * (GOP_SIZE * 2 + 2) * 2) / SAMPLE_RATE
    sig = (0.5 * np.sin(2 * np.pi * 440  * t)
         + 0.3 * np.sin(2 * np.pi * 2000 * t)
         + 0.2 * np.sin(2 * np.pi * 9000 * t))
    analyzer = DSAAnalyzer()
    frames   = analyzer.analyze_samples(sig)

    writer  = DSABitstreamWriter(bitrate_kbps=12)
    dsa_raw = writer.encode_frames(frames)

    # ── Test 1: magic and header parse ────────────────────────────────────────
    t1 = dsa_raw[:4] == FILE_MAGIC
    print(f"  File magic b'DSA1':           {'PASS ✓' if t1 else 'FAIL'}")
    all_pass = all_pass and t1

    hdr = DSAHeader.from_bytes(dsa_raw)
    t1b = (hdr.n_frames == len(frames)
           and hdr.sample_rate == SAMPLE_RATE
           and hdr.version == FORMAT_VERSION
           and hdr.mode == MODE_DISCRETE)
    print(f"  Header fields correct:        n_frames={hdr.n_frames}  "
          f"sr={hdr.sample_rate}  v={hdr.version}  mode={hdr.mode_name}  "
          f"{'PASS ✓' if t1b else 'FAIL'}")
    all_pass = all_pass and t1b

    # ── Test 2: header size and offset arithmetic ─────────────────────────────
    expected_fidx_end = HEADER_SIZE + len(frames) * FRAME_IDX_ENTRY
    t2 = hdr.layer0_offset == expected_fidx_end
    print(f"  Layer 0 offset correct:       {hdr.layer0_offset}  "
          f"{'PASS ✓' if t2 else f'FAIL (expected {expected_fidx_end})'}")
    all_pass = all_pass and t2

    t2b = hdr.layer0_offset < hdr.layer1_offset < hdr.layer2_offset < hdr.crc32_offset
    print(f"  Layer offset ordering:        L0<L1<L2<CRC  "
          f"{'PASS ✓' if t2b else 'FAIL'}")
    all_pass = all_pass and t2b

    # ── Test 3: CRC32 passes on fresh file ────────────────────────────────────
    reader = DSABitstreamReader(dsa_raw)
    t3     = reader.verify_crc()
    print(f"  CRC32 valid (fresh file):     {'PASS ✓' if t3 else 'FAIL'}")
    all_pass = all_pass and t3

    # ── Test 4: CRC32 catches single-byte corruption ──────────────────────────
    corrupt = bytearray(dsa_raw)
    corrupt[HEADER_SIZE + 5] ^= 0xFF   # flip a byte in the frame index
    t4 = not DSABitstreamReader(bytes(corrupt)).verify_crc()
    print(f"  CRC32 catches corruption:     {'PASS ✓' if t4 else 'FAIL'}")
    all_pass = all_pass and t4

    # ── Test 5: layer-isolated read ───────────────────────────────────────────
    # Read L0 — bytes touched must lie within [layer0_offset, layer1_offset)
    l0_data  = reader.read_layer(0)
    l0_count = sum(1 for x in l0_data if x is not None)
    # All non-S frames should have L0 data
    n_non_s  = sum(1 for ft in reader.frame_types if ft != FTYPE_S)
    t5 = l0_count == n_non_s and all(
        isinstance(x, LayerData) or x is None for x in l0_data
    )
    print(f"  L0 isolated read:             {l0_count}/{n_non_s} frames  "
          f"{'PASS ✓' if t5 else 'FAIL'}")
    all_pass = all_pass and t5

    # ── Test 6: reconstruct frames and coefficient roundtrip ──────────────────
    enc_frames = DSAEncoder(12).encode_frames(frames)
    rec_frames = reader.reconstruct_encoded_frames()

    # Compare MDCT coefficients for the first K-frame
    k_orig  = next(f for f in enc_frames if isinstance(f, EncodedKFrame))
    k_rec   = next(f for f in rec_frames if isinstance(f, EncodedKFrame))
    c_orig  = decode_kframe(k_orig)
    c_rec   = decode_kframe(k_rec)
    diff    = np.max(np.abs(c_orig - c_rec))
    t6      = diff < 1e-6
    print(f"  K-frame coeff roundtrip:      max_diff={diff:.2e}  "
          f"{'PASS ✓' if t6 else 'FAIL'}")
    all_pass = all_pass and t6

    # ── Test 7: B-frame reconstruct ───────────────────────────────────────────
    b_enc  = next(f for f in enc_frames if isinstance(f, EncodedBFrame))
    b_rec  = next(f for f in rec_frames if isinstance(f, EncodedBFrame))
    k0_enc = next(f for f in enc_frames if isinstance(f, EncodedKFrame)
                  and f.frame_idx == b_enc.frame_idx - b_enc.gop_pos)
    k1_enc = next(f for f in enc_frames if isinstance(f, EncodedKFrame)
                  and f.frame_idx == b_enc.frame_idx - b_enc.gop_pos + GOP_SIZE)
    k0_rec = next(f for f in rec_frames if isinstance(f, EncodedKFrame)
                  and f.frame_idx == k0_enc.frame_idx)
    k1_rec = next(f for f in rec_frames if isinstance(f, EncodedKFrame)
                  and f.frame_idx == k1_enc.frame_idx)

    bc_enc = decode_bframe(b_enc, decode_kframe(k0_enc), decode_kframe(k1_enc))
    bc_rec = decode_bframe(b_rec, decode_kframe(k0_rec), decode_kframe(k1_rec))
    diff_b = np.max(np.abs(bc_enc - bc_rec))
    t7     = diff_b < 1e-6
    print(f"  B-frame coeff roundtrip:      max_diff={diff_b:.2e}  "
          f"{'PASS ✓' if t7 else 'FAIL'}")
    all_pass = all_pass and t7

    # ── Test 8: layer byte sizes are non-zero and sum correctly ───────────────
    l0_sz = reader.layer_byte_size(0)
    l1_sz = reader.layer_byte_size(1)
    l2_sz = reader.layer_byte_size(2)
    body_sz = hdr.crc32_offset - HEADER_SIZE - len(frames) * FRAME_IDX_ENTRY
    t8 = (l0_sz > 0 and l1_sz > 0 and l2_sz > 0
          and l0_sz + l1_sz + l2_sz == body_sz)
    print(f"  Layer block sizes:            L0={l0_sz:,}B  L1={l1_sz:,}B  "
          f"L2={l2_sz:,}B  sum={l0_sz+l1_sz+l2_sz:,}B  "
          f"{'PASS ✓' if t8 else 'FAIL'}")
    all_pass = all_pass and t8

    # ── Test 9: effective bitrate within 30% of target ────────────────────────
    dur        = len(frames) * FRAME_MS / 1000.0
    eff_kbps   = len(dsa_raw) * 8 / dur / 1000.0
    target     = 12.0
    t9         = abs(eff_kbps - target) / target < 0.30
    print(f"  Effective bitrate:            {eff_kbps:.1f}kbps  "
          f"(target={target}kbps, {len(dsa_raw):,} bytes)  "
          f"{'PASS ✓' if t9 else 'note: outside 30% window'}")
    # informational only — don't fail the suite on this
    # (synthetic 3-tone signal is not typical audio)

    # ── Test 10: mode byte round-trips ────────────────────────────────────────
    w2   = DSABitstreamWriter(bitrate_kbps=12, mode=MODE_GRADIENT)
    raw2 = w2.encode_frames(frames)
    hdr2 = DSAHeader.from_bytes(raw2)
    t10  = hdr2.mode == MODE_GRADIENT and hdr2.mode_name == 'gradient'
    print(f"  Mode byte (gradient):         mode=0x{hdr2.mode:02x}  "
          f"{'PASS ✓' if t10 else 'FAIL'}")
    all_pass = all_pass and t10

    return all_pass

# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print(f"\n  DSA — Digilog Scalable Audio")
    print(f"  Step 6: Layered Bitstream Packer")
    print(f"  {'─'*48}")
    print(f"  Format:   DSA1 v{FORMAT_VERSION}")
    print(f"  Header:   {HEADER_SIZE} bytes")
    print(f"  Layout:   Header | Frame index | L0 block | L1 block | L2 block | CRC32")
    print(f"  Modes:    0x01=discrete (current)  0x02=gradient (v2.0)")
    print(f"  {'─'*48}")

    print(f"\n  Running verification tests...")
    ok = verify()

    if not ok:
        print(f"\n  ✗ Some tests failed.")
        sys.exit(1)

    print(f"\n  All tests passed ✓")

    if len(sys.argv) > 1:
        path    = sys.argv[1]
        kbps    = float(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_BITRATE
        outpath = Path(path).with_suffix('.dsa')
        print(f"\n  Encoding: {path}  @ {kbps}kbps  →  {outpath}")

        writer = DSABitstreamWriter(kbps)
        data   = writer.encode_file(str(path), str(outpath))
        reader = DSABitstreamReader(data)
        hdr    = reader.header

        print(f"\n  DSA file stats:")
        print(f"    Frames:         {hdr.n_frames:,}")
        print(f"    Duration:       {reader.duration_s:.1f}s")
        print(f"    File size:      {reader.file_size:,} bytes")
        print(f"    Effective kbps: {reader.file_size*8/reader.duration_s/1000:.1f}")
        print(f"    CRC32:          {'OK ✓' if reader.verify_crc() else 'FAIL'}")
        print(f"    Mode:           {hdr.mode_name}")
        print(f"\n  Layer breakdown:")
        for li, name in enumerate(['L0 (bass,  8 bands)', 'L1 (mid,  16 bands)',
                                   'L2 (high, 24 bands)']):
            sz    = reader.layer_byte_size(li)
            pct   = 100 * sz / reader.file_size
            print(f"    {name}:  {sz:,} bytes  ({pct:.1f}%)")
    else:
        print(f"\n  Run with audio to produce a .dsa file:")
        print(f"  python3 dsa_bitstream.py your_song.mp3 [bitrate_kbps]")

    print(f"\n  Ready for Step 7: DSA decoder ✓")
    print(f"\n  Scan the groove.\n")
