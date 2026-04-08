#!/usr/bin/env python3
"""
DSA — Digilog Scalable Audio
dsa_strip.py — Unrolled Disc Strip Viewer

Renders a .disc.json as a flat 2D image:
    x-axis = time (frames, left → right)
    y-axis = frequency band (0=bass/bottom → 47=high/top)
    color  = gradient color for that (frame, band) cell

This is what the camera would read if the disc were unrolled into a strip.
It is a visual spectrogram of the DSA encoding — useful for verifying
that spectral content, color assignments, and gradient directions are correct.

Layer boundaries are marked:
    L0 bands  0– 7  (bass,  bottom)
    L1 bands  8–23  (mid,   middle)
    L2 bands 24–47  (high,  top)

Usage:
    python3 dsa_strip.py song.disc.json
    python3 dsa_strip.py song.disc.json --height 4 --out strip.png
    python3 dsa_strip.py song.disc.json --start 0 --end 200  # first 200 frames

License: GPL v3 — github.com/pisdronio/dsa
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    sys.exit("Pillow required — pip install Pillow")

from dsa_color import lerp_lab

# ─── Layer boundaries ─────────────────────────────────────────────────────────

L0_BANDS  = range(0,  8)   # bass
L1_BANDS  = range(8,  24)  # mid
L2_BANDS  = range(24, 48)  # high

# ─── Palette ──────────────────────────────────────────────────────────────────

PALETTE = {
    'black':  np.array([0,   0,   0],   dtype=np.float32),
    'white':  np.array([255, 255, 255], dtype=np.float32),
    'red':    np.array([220, 50,  50],  dtype=np.float32),
    'green':  np.array([50,  180, 50],  dtype=np.float32),
    'blue':   np.array([50,  50,  220], dtype=np.float32),
    'yellow': np.array([240, 220, 0],   dtype=np.float32),
    'cyan':   np.array([0,   210, 210], dtype=np.float32),
    'purple': np.array([160, 50,  200], dtype=np.float32),
}

# Label colors for layer separators
L0_COLOR  = (60,  60,  60)
L1_COLOR  = (60,  60,  60)
L2_COLOR  = (60,  60,  60)
SEP_COLOR = (80,  80,  80)


def _band_color(steepness: float, direction: int,
                ca: np.ndarray, cb: np.ndarray) -> np.ndarray:
    """
    Single representative color for a (frame, band) cell.
    Uses the midpoint of the arc (arc_pos = 0.5) and LAB interpolation
    for perceptually uniform gradient representation.
    """
    if direction == 0 or steepness < 0.001:
        return ca
    t = 0.5 * steepness
    if direction < 0:
        t = steepness - t
    t = max(0.0, min(1.0, t))
    return lerp_lab(ca, cb, np.array(t)).astype(np.float32)


def render_strip(layout_path: str,
                 cell_h: int    = 8,
                 frame_start: int = 0,
                 frame_end: int   = None,
                 out_path: str    = None) -> str:
    """
    Render a .disc.json to a flat strip PNG.

    Parameters
    ----------
    layout_path  : path to .disc.json
    cell_h       : pixel height of each band row (default 8)
    frame_start  : first frame to render (default 0)
    frame_end    : last frame (exclusive); None = all frames
    out_path     : output PNG path; defaults to <layout>.strip.png
    """

    # ── Load ──────────────────────────────────────────────────────────────────
    print(f"  Loading {Path(layout_path).name} ...", end=' ', flush=True)
    with open(layout_path) as f:
        layout = json.load(f)

    n_frames   = layout['n_frames']
    n_bands    = layout['n_bands']
    duration_s = layout['duration_s']
    frames     = layout['frames']
    bp_map     = {b['band']: (b['color_a'], b['color_b']) for b in layout['band_pairs']}

    if frame_end is None:
        frame_end = n_frames
    frame_end = min(frame_end, n_frames)
    n_render  = frame_end - frame_start

    print(f"done  ({n_frames} frames total, rendering {n_render})")

    # ── Build color table ─────────────────────────────────────────────────────
    print("  Building color table...", end=' ', flush=True)

    # color_table[frame_idx, band_idx] = RGB uint8
    color_table = np.zeros((n_frames, n_bands, 3), dtype=np.uint8)

    # Default: solid color_a for each band
    for b in range(n_bands):
        color_table[:, b] = PALETTE[bp_map[b][0]].astype(np.uint8)

    for fd in frames:
        fi = fd['frame_idx']
        if fd['silence']:
            continue
        for bd in fd['bands']:
            b   = bd['band']
            ca  = PALETTE[bp_map[b][0]]
            cb  = PALETTE[bp_map[b][1]]
            col = _band_color(bd['steepness'], bd['direction'], ca, cb)
            color_table[fi, b] = col.astype(np.uint8)

    print("done")

    # ── Render ────────────────────────────────────────────────────────────────
    # Image layout:
    #   width  = n_render frames (1px per frame)
    #   height = n_bands × cell_h + separator lines
    #
    # Bands are drawn bottom-to-top: band 0 (bass) at bottom, band 47 (high) at top.
    # Layer separators (2px) between L0/L1 and L1/L2.

    SEP_PX   = 2    # separator line thickness
    img_w    = n_render
    img_h    = n_bands * cell_h + 2 * SEP_PX  # two separators

    print(f"  Strip size: {img_w}×{img_h} px  ({n_render} frames × {n_bands} bands × {cell_h}px/band)")

    img = np.zeros((img_h, img_w, 3), dtype=np.uint8)

    # Draw bands bottom-to-top
    # Band row mapping (bottom = band 0):
    #   band b → rows  [img_h - (b+1)*cell_h - sep_offset : img_h - b*cell_h - sep_offset]
    # With separators inserted between L0/L1 (after band 7) and L1/L2 (after band 23):

    def band_row_top(b: int) -> int:
        """Top pixel row of band b (0=bass at bottom → 47=high at top)."""
        # separators occupy 2px between L0/L1 and L1/L2
        sep_below = 0
        if b >= 8:   sep_below += SEP_PX
        if b >= 24:  sep_below += SEP_PX
        # distance from bottom: b bands × cell_h + separators already added below
        bottom_offset = b * cell_h + sep_below
        return img_h - bottom_offset - cell_h

    for b in range(n_bands):
        row_top = band_row_top(b)
        row_bot = row_top + cell_h
        # color column slice: shape (n_render, 3)
        colors = color_table[frame_start:frame_end, b, :]  # (n_render, 3)
        # broadcast across cell_h rows
        img[row_top:row_bot, :, :] = colors[np.newaxis, :, :]

    # Separator lines (dark gray)
    sep0_top = band_row_top(8) + cell_h     # between L0 (band 7) and L1 (band 8)
    sep1_top = band_row_top(24) + cell_h    # between L1 (band 23) and L2 (band 24)
    img[sep0_top : sep0_top + SEP_PX, :] = SEP_COLOR
    img[sep1_top : sep1_top + SEP_PX, :] = SEP_COLOR

    # ── Add labels (PIL) ──────────────────────────────────────────────────────
    pil_img  = Image.fromarray(img)
    draw     = ImageDraw.Draw(pil_img)

    # Time markers every ~1 second (44100/1024 ≈ 43 frames/second)
    frames_per_sec = 44100 / 1024
    t = frame_start
    while t < frame_end:
        x = t - frame_start
        draw.line([(x, 0), (x, img_h)], fill=(50, 50, 50), width=1)
        t += int(frames_per_sec)

    # ── Save ──────────────────────────────────────────────────────────────────
    if out_path is None:
        out_path = str(Path(layout_path).with_suffix('.strip.png'))

    print(f"  Saving → {out_path} ...", end=' ', flush=True)
    pil_img.save(out_path)
    size_kb = Path(out_path).stat().st_size / 1024
    print(f"done  ({size_kb:.0f} KB)")
    return out_path


def main():
    p = argparse.ArgumentParser(
        description='Render a DSA .disc.json as a flat 2D strip (unrolled disc)')
    p.add_argument('layout',
                   help='Path to .disc.json')
    p.add_argument('--height',  type=int, default=8,
                   help='Pixel height per band row (default: 8)')
    p.add_argument('--start',   type=int, default=0,
                   help='First frame to render (default: 0)')
    p.add_argument('--end',     type=int, default=None,
                   help='Last frame (exclusive); default: all')
    p.add_argument('--out',     type=str, default=None,
                   help='Output PNG path')
    args = p.parse_args()

    print()
    print("  DSA — Digilog Scalable Audio")
    print("  Unrolled Disc Strip Viewer")
    print("  ─────────────────────────────────────────────────")
    print(f"  Layout:  {args.layout}")
    print()

    out = render_strip(args.layout,
                       cell_h=args.height,
                       frame_start=args.start,
                       frame_end=args.end,
                       out_path=args.out)
    print()
    print("  L0 (bass):  bottom rows  — black/white/yellow/cyan")
    print("  L1 (mid):   middle rows  — red/cyan, blue/yellow, green/purple")
    print("  L2 (high):  top rows     — full palette")
    print("  Vertical lines = 1-second markers")
    print()


if __name__ == '__main__':
    main()
