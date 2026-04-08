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
    Single representative color for a (frame, band) cell — visualization mode.
    Uses the midpoint of the arc (arc_pos = 0.5) and LAB interpolation.
    NOTE: This loses direction information — both +1 and -1 produce the same
    midpoint color. Use _band_gradient_row() for readable encoding.
    """
    if direction == 0 or steepness < 0.001:
        return ca
    t = 0.5 * steepness
    t = max(0.0, min(1.0, t))
    return lerp_lab(ca, cb, np.array(t)).astype(np.float32)


def _band_gradient_row(steepness: float, direction: int,
                       ca: np.ndarray, cb: np.ndarray,
                       cell_w: int) -> np.ndarray:
    """
    Render one frame-cell as a cell_w-pixel horizontal gradient — readable mode.

    Encodes direction and steepness in a way that matches the disc arc encoding:
      direction=+1:  left→right transition from ca toward cb (t: 0 → steepness)
      direction=-1:  left→right transition from cb-side toward ca (t: steepness → 0)
      direction= 0:  solid ca

    Sampling at 25% and 75% x-positions recovers steepness and direction via
    the same formula as dsa_reader.py:
      diff = t_right − t_left  →  steepness = 2|diff|,  direction = sign(diff)

    Returns (cell_w, 3) uint8 array.
    """
    row = np.zeros((cell_w, 3), dtype=np.float32)
    if direction == 0 or steepness < 0.001:
        row[:] = ca
        return row.astype(np.uint8)

    for xi in range(cell_w):
        f = xi / (cell_w - 1) if cell_w > 1 else 0.5   # 0.0 → 1.0
        if direction > 0:
            t = f * steepness          # left=0(ca) → right=steepness
        else:
            t = (1.0 - f) * steepness  # left=steepness(toward cb) → right=0(ca)
        t = max(0.0, min(1.0, t))
        row[xi] = lerp_lab(ca, cb, np.array(t)).astype(np.float32)

    return row.astype(np.uint8)


def render_strip(layout_path: str,
                 cell_h: int      = 8,
                 cell_w: int      = 1,
                 frame_start: int = 0,
                 frame_end: int   = None,
                 out_path: str    = None,
                 fiducials: bool  = False,
                 strip_dpi: int   = 300) -> str:
    """
    Render a .disc.json to a flat strip PNG.

    Parameters
    ----------
    layout_path  : path to .disc.json
    cell_h       : pixel height of each band row (default 8)
    cell_w       : pixel width per frame column (default 1 = visualization only).
                   Set ≥4 to render readable color gradients within each cell —
                   required for dsa_camera.py to detect steepness and direction.
                   Recommended: 4 (compact), 8 (comfortable margin).
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

    # ── Build colour data ─────────────────────────────────────────────────────
    readable = cell_w > 1

    if readable:
        print(f"  Mode: READABLE gradient  (cell_w={cell_w}px per frame — encodes direction)")
        # gradient_table[frame_idx, band_idx] = (cell_w, 3) uint8
        gradient_table = {}
        for b in range(n_bands):
            ca = PALETTE[bp_map[b][0]]
            cb = PALETTE[bp_map[b][1]]
            gradient_table[b] = {}
            # default: solid ca
            default_row = np.tile(ca.astype(np.uint8), (cell_w, 1))
            for fi in range(n_frames):
                gradient_table[b][fi] = default_row

        print("  Building gradient table...", end=' ', flush=True)
        for fd in frames:
            fi = fd['frame_idx']
            if fd['silence']:
                continue
            for bd in fd['bands']:
                b  = bd['band']
                ca = PALETTE[bp_map[b][0]]
                cb = PALETTE[bp_map[b][1]]
                gradient_table[b][fi] = _band_gradient_row(
                    bd['steepness'], bd['direction'], ca, cb, cell_w)
        print("done")
    else:
        print("  Mode: VISUALIZATION flat colour  (cell_w=1 — direction not encoded)")
        print("  Building color table...", end=' ', flush=True)
        color_table = np.zeros((n_frames, n_bands, 3), dtype=np.uint8)
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
    #   width  = n_render × cell_w  (cell_w px per frame)
    #   height = n_bands × cell_h + separator lines
    #
    # Bands are drawn bottom-to-top: band 0 (bass) at bottom, band 47 (high) at top.
    # Layer separators (2px) between L0/L1 and L1/L2.

    SEP_PX   = 2    # separator line thickness
    img_w    = n_render * cell_w
    img_h    = n_bands * cell_h + 2 * SEP_PX  # two separators

    print(f"  Strip size: {img_w}×{img_h} px  "
          f"({n_render} frames × {cell_w}px/frame × {n_bands} bands × {cell_h}px/band)")

    img = np.zeros((img_h, img_w, 3), dtype=np.uint8)

    def band_row_top(b: int) -> int:
        """Top pixel row of band b (0=bass at bottom → 47=high at top)."""
        sep_below = 0
        if b >= 8:   sep_below += SEP_PX
        if b >= 24:  sep_below += SEP_PX
        bottom_offset = b * cell_h + sep_below
        return img_h - bottom_offset - cell_h

    for b in range(n_bands):
        row_top = band_row_top(b)
        row_bot = row_top + cell_h

        if readable:
            # Each frame occupies cell_w columns; fill with gradient row
            for i, fi in enumerate(range(frame_start, frame_end)):
                x0 = i * cell_w
                x1 = x0 + cell_w
                row = gradient_table[b][fi]         # (cell_w, 3)
                img[row_top:row_bot, x0:x1, :] = row[np.newaxis, :, :]
        else:
            # 1px per frame — broadcast flat colour
            colors = color_table[frame_start:frame_end, b, :]  # (n_render, 3)
            img[row_top:row_bot, :, :] = colors[np.newaxis, :, :]

    # Separator lines (dark gray)
    sep0_top = band_row_top(8) + cell_h
    sep1_top = band_row_top(24) + cell_h
    img[sep0_top : sep0_top + SEP_PX, :] = SEP_COLOR
    img[sep1_top : sep1_top + SEP_PX, :] = SEP_COLOR

    # ── Add labels (PIL) ──────────────────────────────────────────────────────
    pil_img  = Image.fromarray(img)
    draw     = ImageDraw.Draw(pil_img)

    # Time markers every ~1 second (44100/1024 ≈ 43 frames/second)
    frames_per_sec = 44100 / 1024
    t = frame_start
    while t < frame_end:
        x = int(t - frame_start) * cell_w
        draw.line([(x, 0), (x, img_h)], fill=(50, 50, 50), width=1)
        t += int(frames_per_sec)

    # ── Fiducial registration marks ───────────────────────────────────────────
    if fiducials:
        px_per_mm   = strip_dpi / 25.4
        corner_mm   = 5.0
        corner_px   = int(round(corner_mm * px_per_mm))
        scalebar_mm = 10.0
        scalebar_px = int(round(scalebar_mm * px_per_mm))
        scalebar_h  = max(4, int(round(1.0 * px_per_mm)))
        border_px   = max(20, scalebar_h + 8)

        # Clamp scale bar width so it fits between left corner and right corner
        old_w, old_h = pil_img.size
        new_w = old_w + 2 * border_px
        new_h = old_h + 2 * border_px
        scalebar_px = min(scalebar_px, new_w - 2 * corner_px - 8)

        # Expand canvas
        bordered = Image.new('RGB', (new_w, new_h), (255, 255, 255))
        bordered.paste(pil_img, (border_px, border_px))
        draw2 = ImageDraw.Draw(bordered)

        # Corner squares (5mm, at each corner — extend into border region)
        cp = corner_px
        draw2.rectangle([0,          0,          cp - 1,      cp - 1     ], fill=(0, 0, 0))
        draw2.rectangle([new_w - cp, 0,          new_w - 1,   cp - 1     ], fill=(0, 0, 0))
        draw2.rectangle([0,          new_h - cp, cp - 1,      new_h - 1  ], fill=(0, 0, 0))
        draw2.rectangle([new_w - cp, new_h - cp, new_w - 1,   new_h - 1  ], fill=(0, 0, 0))

        # 10mm scale bar in bottom border, left of center
        bar_x0 = cp + 4
        bar_x1 = bar_x0 + scalebar_px
        bar_y  = new_h - border_px + (border_px - scalebar_h) // 2
        draw2.rectangle([bar_x0, bar_y, bar_x1, bar_y + scalebar_h - 1], fill=(0, 0, 0))
        # End-cap ticks
        tick_ext = scalebar_h
        draw2.line([(bar_x0, bar_y - tick_ext), (bar_x0, bar_y + scalebar_h + tick_ext)],
                   fill=(0, 0, 0), width=2)
        draw2.line([(bar_x1, bar_y - tick_ext), (bar_x1, bar_y + scalebar_h + tick_ext)],
                   fill=(0, 0, 0), width=2)

        pil_img = bordered
        phys_w  = old_w / px_per_mm
        phys_h  = old_h / px_per_mm
        print(f"  Fiducials: border={border_px}px, corners={corner_mm:.0f}mm ({corner_px}px), "
              f"scale bar={scalebar_mm:.0f}mm — assume {strip_dpi} DPI when printing")
        print(f"  Physical strip size: {phys_w:.0f}×{phys_h:.0f}mm "
              f"({phys_w/10:.1f}×{phys_h/10:.1f}cm) at {strip_dpi} DPI")

    # ── Save ──────────────────────────────────────────────────────────────────
    if out_path is None:
        suffix   = '.fiducials.strip.png' if fiducials else '.strip.png'
        out_path = str(Path(layout_path).with_suffix('').with_suffix('')) + suffix

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
    p.add_argument('--cell-w',  type=int, default=1,
                   help='Pixel width per frame column (default: 1 = visualization). '
                        'Set ≥4 to encode readable gradients for dsa_camera.py.')
    p.add_argument('--start',   type=int, default=0,
                   help='First frame to render (default: 0)')
    p.add_argument('--end',     type=int, default=None,
                   help='Last frame (exclusive); default: all')
    p.add_argument('--out',       type=str,  default=None,
                   help='Output PNG path')
    p.add_argument('--fiducials', action='store_true',
                   help='Add corner registration marks and 10mm scale bar for camera reader')
    p.add_argument('--strip-dpi', type=int,  default=300,
                   help='Physical DPI to assume when printing (default: 300)')
    args = p.parse_args()

    print()
    print("  DSA — Digilog Scalable Audio")
    print("  Unrolled Disc Strip Viewer")
    print("  ─────────────────────────────────────────────────")
    print(f"  Layout:  {args.layout}")
    print()

    out = render_strip(args.layout,
                       cell_h=args.height,
                       cell_w=args.cell_w,
                       frame_start=args.start,
                       frame_end=args.end,
                       out_path=args.out,
                       fiducials=args.fiducials,
                       strip_dpi=args.strip_dpi)
    print()
    print("  L0 (bass):  bottom rows  — black/white/yellow/cyan")
    print("  L1 (mid):   middle rows  — red/cyan, blue/yellow, green/purple")
    print("  L2 (high):  top rows     — full palette")
    print("  Vertical lines = 1-second markers")
    print()


if __name__ == '__main__':
    main()
