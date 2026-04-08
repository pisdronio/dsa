#!/usr/bin/env python3
"""
DSA — Digilog Scalable Audio
dsa_reader.py — Disc Dot Reader (virtual / simulated)

Reads a rendered disc PNG and extracts the encoded gradient data,
simulating what a camera-based reader does with a physical disc.

For each (frame, band) arc cell, samples two pixels at arc positions
25% and 75%, projects each onto the expected color_a→color_b axis,
and recovers steepness, direction, and read confidence α.

Output:
    - Per-band average confidence vector → analog degradation input for decoder
    - Per-frame/band confidence matrix   → future per-frame decoding
    - Accuracy statistics vs original disc.json
    - Confidence map PNG                 → visual showing read quality
    - Optionally: decoded WAV with confidence weighting

Read geometry (two samples per arc):
    left  sample: arc_pos = 0.25   theta = (fi + 0.25) × frame_angle
    right sample: arc_pos = 0.75   theta = (fi + 0.75) × frame_angle

    t     = dot(C − ca, cb − ca) / |cb − ca|²   (blend factor, 0..1)
    diff  = t_right − t_left
    steepness = 2 × |diff|
    direction = sign(diff)  if |diff| > threshold else 0
    α     = 1 − residual / |cb − ca|             (color match quality)

Usage:
    python3 dsa_reader.py disc.png disc.disc.json
    python3 dsa_reader.py disc.png disc.disc.json --decode song.dsa --out recovered.wav
    python3 dsa_reader.py disc.png disc.disc.json --confidence-map conf.png

License: GPL v3 — github.com/pisdronio/dsa
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

try:
    from PIL import Image
except ImportError:
    sys.exit("Pillow required — pip install Pillow")

# ─── Disc geometry (mm) ───────────────────────────────────────────────────────

DISC_DIAMETER_MM  = 290.0
OUTER_AUDIO_MM    = 141.0
INNER_AUDIO_MM    = 62.0
NUM_BANDS         = 48

# Direction detection threshold — |diff| below this is treated as zero.
# Must be well below half the minimum Mode 1 diff:
#   min_steepness = 1/31 ≈ 0.032  →  min_diff = 0.5 × 0.032 = 0.016
#   threshold = 0.008  (half of min_diff — safe margin above sampling noise)
DIRECTION_THRESHOLD = 0.008

# ─── Palette ──────────────────────────────────────────────────────────────────

PALETTE = {
    'black':  np.array([0,   0,   0],   dtype=np.float64),
    'white':  np.array([255, 255, 255], dtype=np.float64),
    'red':    np.array([220, 50,  50],  dtype=np.float64),
    'green':  np.array([50,  180, 50],  dtype=np.float64),
    'blue':   np.array([50,  50,  220], dtype=np.float64),
    'yellow': np.array([240, 220, 0],   dtype=np.float64),
    'cyan':   np.array([0,   210, 210], dtype=np.float64),
    'purple': np.array([160, 50,  200], dtype=np.float64),
}


def _color_to_blend(color: np.ndarray,
                    ca: np.ndarray,
                    cb: np.ndarray) -> tuple[float, float]:
    """
    Project a sampled RGB color onto the ca→cb axis.

    Returns
    -------
    t     : blend factor in [0, 1]
    alpha : confidence — 1.0 = perfect match, 0.0 = completely off-axis
    """
    v       = cb - ca
    denom   = np.dot(v, v)
    if denom < 1e-6:
        return 0.0, 1.0   # degenerate pair (same color) — always confident

    t       = np.dot(color - ca, v) / denom
    t       = float(np.clip(t, 0.0, 1.0))

    # Residual: how far is the actual color from the projected point
    projected   = ca + t * v
    residual    = float(np.linalg.norm(color - projected))
    max_dist    = float(np.linalg.norm(v))
    confidence  = max(0.0, 1.0 - residual / (max_dist + 1e-9))

    return t, confidence


def _sample_pixel(img_arr: np.ndarray, ctr: int,
                  r_px: float, theta: float) -> np.ndarray:
    """
    Bilinear sample of img_arr at polar coordinates (r_px, theta).
    theta: 0 = top (12 o'clock), clockwise.
    Returns float64 RGB array.
    """
    x = ctr + r_px * math.sin(theta)
    y = ctr - r_px * math.cos(theta)

    # Bilinear interpolation
    x0, y0 = int(x), int(y)
    x1, y1 = x0 + 1, y0 + 1
    h, w   = img_arr.shape[:2]

    x0 = max(0, min(x0, w - 1))
    x1 = max(0, min(x1, w - 1))
    y0 = max(0, min(y0, h - 1))
    y1 = max(0, min(y1, h - 1))

    fx = x - int(x)
    fy = y - int(y)

    c00 = img_arr[y0, x0].astype(np.float64)
    c10 = img_arr[y0, x1].astype(np.float64)
    c01 = img_arr[y1, x0].astype(np.float64)
    c11 = img_arr[y1, x1].astype(np.float64)

    return (c00 * (1 - fx) * (1 - fy) +
            c10 * fx       * (1 - fy) +
            c01 * (1 - fx) * fy       +
            c11 * fx       * fy)


class DSADiscReader:
    """
    Reads a rendered DSA disc image and extracts gradient encoding data.

    Parameters
    ----------
    image_path  : path to disc PNG (from dsa_render.py)
    layout_path : path to .disc.json (provides color pair assignments)
    """

    def __init__(self, image_path: str, layout_path: str):
        print(f"  Loading image  {Path(image_path).name} ...", end=' ', flush=True)
        pil         = Image.open(image_path).convert('RGB')
        self.img    = np.array(pil, dtype=np.uint8)
        self.size   = pil.width          # assumes square image
        self.ctr    = self.size // 2
        print(f"done  ({pil.width}×{pil.height}px)")

        print(f"  Loading layout {Path(layout_path).name} ...", end=' ', flush=True)
        with open(layout_path) as f:
            self.layout = json.load(f)
        self.n_frames   = self.layout['n_frames']
        self.n_bands    = self.layout['n_bands']
        self.bp_map     = {b['band']: (b['color_a'], b['color_b'])
                           for b in self.layout['band_pairs']}
        print(f"done  ({self.n_frames} frames, {self.n_bands} bands)")

        # Pixel geometry
        mm_to_px            = self.size / DISC_DIAMETER_MM
        self.inner_audio_px = INNER_AUDIO_MM * mm_to_px
        self.ring_width_px  = ((OUTER_AUDIO_MM - INNER_AUDIO_MM) * mm_to_px) / NUM_BANDS
        self.frame_angle    = 2.0 * math.pi / self.n_frames

        # Spiral geometry (set via set_spiral)
        self.spiral         = False
        self.frames_per_rev = self.n_frames
        self.n_revolutions  = 1
        self.sub_band_px    = self.ring_width_px

    def set_spiral(self, rpm: float) -> None:
        """Configure spiral reading mode to match dsa_render.py --spiral --rpm."""
        fps_actual          = self.n_frames / self.layout['duration_s']
        self.frames_per_rev = max(1, round(fps_actual / (rpm / 60.0)))
        self.n_revolutions  = self.n_frames // self.frames_per_rev
        self.sub_band_px    = self.ring_width_px / self.n_revolutions
        self.spiral         = True
        angle_per_lf        = 2.0 * math.pi / self.frames_per_rev
        print(f"  Spiral mode: {rpm} RPM, {self.frames_per_rev} frames/rev, "
              f"{self.n_revolutions} revolutions, sub-band {self.sub_band_px:.2f}px")

    def read(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Read all (frame, band) cells from the disc image.

        Returns
        -------
        steep_read  : (n_frames, n_bands) float64 — measured steepness
        dir_read    : (n_frames, n_bands) int8     — measured direction
        conf_frame  : (n_frames, n_bands) float64  — per-cell confidence α
        alpha       : (n_bands,) float64            — per-band mean confidence
        """
        n_frames = self.n_frames
        n_bands  = self.n_bands
        img      = self.img
        ctr      = self.ctr

        steep_read = np.zeros((n_frames, n_bands), dtype=np.float64)
        dir_read   = np.zeros((n_frames, n_bands), dtype=np.int8)
        conf_frame = np.zeros((n_frames, n_bands), dtype=np.float64)

        print(f"  Reading {n_frames} × {n_bands} cells ...", end=' ', flush=True)

        for b in range(n_bands):
            ca_name, cb_name = self.bp_map[b]
            ca = PALETTE[ca_name]
            cb = PALETTE[cb_name]

            for fi in range(n_frames):
                if not self.spiral:
                    # Single-revolution: all frames at band midpoint radius
                    r_px        = self.inner_audio_px + (b + 0.5) * self.ring_width_px
                    theta_left  = (fi + 0.25) * self.frame_angle
                    theta_right = (fi + 0.75) * self.frame_angle
                else:
                    # Spiral: revolution determines sub-band radius,
                    #         local frame determines angle within revolution
                    rev         = fi // self.frames_per_rev
                    rev         = min(rev, self.n_revolutions - 1)
                    f_local     = fi % self.frames_per_rev
                    r_outer_b   = self.inner_audio_px + (b + 1) * self.ring_width_px
                    r_px        = r_outer_b - (rev + 0.5) * self.sub_band_px
                    angle_lf    = 2.0 * math.pi / self.frames_per_rev
                    theta_left  = (f_local + 0.25) * angle_lf
                    theta_right = (f_local + 0.75) * angle_lf

                c_left  = _sample_pixel(img, ctr, r_px, theta_left)
                c_right = _sample_pixel(img, ctr, r_px, theta_right)

                t_left,  al = _color_to_blend(c_left,  ca, cb)
                t_right, ar = _color_to_blend(c_right, ca, cb)

                diff = t_right - t_left
                s    = min(1.0, 2.0 * abs(diff))

                if abs(diff) < DIRECTION_THRESHOLD:
                    d = 0
                elif diff > 0:
                    d = 1
                else:
                    d = -1

                steep_read[fi, b] = s
                dir_read[fi, b]   = d
                conf_frame[fi, b] = (al + ar) / 2.0

            if (b + 1) % 12 == 0:
                print(f"{b+1}..", end='', flush=True)

        print(" done")

        alpha = conf_frame.mean(axis=0)   # per-band mean confidence
        return steep_read, dir_read, conf_frame, alpha

    def compare(self,
                steep_read: np.ndarray,
                dir_read:   np.ndarray) -> dict:
        """
        Compare read values against the original disc.json.

        Returns a dict of accuracy metrics.
        """
        orig_steep = np.zeros((self.n_frames, self.n_bands), dtype=np.float64)
        orig_dir   = np.zeros((self.n_frames, self.n_bands), dtype=np.int8)

        for fd in self.layout['frames']:
            fi = fd['frame_idx']
            if fd['silence']:
                continue
            for bd in fd['bands']:
                b               = bd['band']
                orig_steep[fi, b] = bd['steepness']
                orig_dir[fi, b]   = bd['direction']

        steep_err   = np.abs(steep_read - orig_steep)
        dir_match   = (dir_read == orig_dir)

        return {
            'steepness_mae':      float(steep_err.mean()),
            'steepness_max_err':  float(steep_err.max()),
            'direction_accuracy': float(dir_match.mean()),
            'direction_errors':   int((~dir_match).sum()),
            'n_cells':            self.n_frames * self.n_bands,
        }

    def confidence_map(self,
                       conf_frame: np.ndarray,
                       out_path: str) -> None:
        """
        Save a confidence map PNG:
        x = frames, y = bands, color = confidence (green=1.0, red=0.0).
        """
        from PIL import Image as PILImage

        cell_h  = 8
        img_w   = self.n_frames
        img_h   = self.n_bands * cell_h

        arr = np.zeros((img_h, img_w, 3), dtype=np.uint8)

        for b in range(self.n_bands):
            row_top = (self.n_bands - 1 - b) * cell_h
            row_bot = row_top + cell_h
            α_row   = conf_frame[:, b]  # (n_frames,)

            # Green (α=1) → Red (α=0)
            r = ((1.0 - α_row) * 220).astype(np.uint8)
            g = (α_row          * 180).astype(np.uint8)
            b_ = np.zeros_like(r)

            arr[row_top:row_bot, :, 0] = r[np.newaxis, :]
            arr[row_top:row_bot, :, 1] = g[np.newaxis, :]
            arr[row_top:row_bot, :, 2] = b_[np.newaxis, :]

        PILImage.fromarray(arr).save(out_path)
        print(f"  Confidence map → {out_path}")


def main():
    p = argparse.ArgumentParser(
        description='Read a DSA disc image and extract gradient encoding data')
    p.add_argument('image',         help='Disc PNG (from dsa_render.py)')
    p.add_argument('layout',        help='.disc.json layout file')
    p.add_argument('--decode',      type=str,   default=None,
                   help='.dsa bitstream to decode with read confidence')
    p.add_argument('--out',         type=str,   default=None,
                   help='Output WAV path (requires --decode)')
    p.add_argument('--confidence-map', type=str, default=None, metavar='PATH',
                   help='Save confidence map PNG to PATH')
    p.add_argument('--spiral',      action='store_true',
                   help='Read spiral disc geometry (must match --spiral used during render)')
    p.add_argument('--rpm',         type=float, default=33.0,
                   help='RPM used when rendering the spiral disc (default: 33)')
    args = p.parse_args()

    print()
    print("  DSA — Digilog Scalable Audio")
    print("  Disc Dot Reader")
    print("  ─────────────────────────────────────────────────")
    print()

    reader = DSADiscReader(args.image, args.layout)
    if args.spiral:
        reader.set_spiral(args.rpm)

    print()
    steep_read, dir_read, conf_frame, alpha = reader.read()

    print()
    print("  ── Read accuracy (vs original disc.json) ─────")
    metrics = reader.compare(steep_read, dir_read)
    print(f"  Steepness MAE:        {metrics['steepness_mae']:.6f}")
    print(f"  Steepness max error:  {metrics['steepness_max_err']:.6f}")
    print(f"  Direction accuracy:   {metrics['direction_accuracy']*100:.2f}%  "
          f"({metrics['n_cells'] - metrics['direction_errors']}"
          f"/{metrics['n_cells']} cells correct)")
    print()
    print("  ── Per-band mean confidence (α) ──────────────")
    print("  L0 (bass):  ", ' '.join(f"{alpha[b]:.2f}" for b in range(8)))
    print("  L1 (mid):   ", ' '.join(f"{alpha[b]:.2f}" for b in range(8, 16)))
    print("              ", ' '.join(f"{alpha[b]:.2f}" for b in range(16, 24)))
    print("  L2 (high):  ", ' '.join(f"{alpha[b]:.2f}" for b in range(24, 36)))
    print("              ", ' '.join(f"{alpha[b]:.2f}" for b in range(36, 48)))
    print()
    print(f"  Overall mean α: {alpha.mean():.4f}")

    if args.confidence_map:
        reader.confidence_map(conf_frame, args.confidence_map)

    if args.decode:
        from pathlib import Path as _Path
        import scipy.io.wavfile as wav
        from dsa_bitstream import DSABitstreamReader
        from dsa_decoder import DSADecoder

        dsa_bytes = _Path(args.decode).read_bytes()
        br        = DSABitstreamReader(dsa_bytes)
        decoder   = DSADecoder()
        samples   = decoder.decode(br, alpha=alpha)

        out_path  = args.out or str(_Path(args.decode).with_suffix('.reader.wav'))
        s16       = np.clip(samples * 32767, -32768, 32767).astype(np.int16)
        wav.write(out_path, 44100, s16)
        print(f"  Decoded WAV → {out_path}  ({len(samples)/44100:.1f}s)")

    print()
    print("  Scan the groove.")
    print()


if __name__ == '__main__':
    main()
