#!/usr/bin/env python3
"""
DSA — Digilog Scalable Audio
dsa_render.py — Disc Image Renderer

Converts a .disc.json layout file into a printable disc PNG.

The disc has 48 concentric audio rings (one per frequency band) divided into
arc segments — one segment per audio frame. Each arc is filled with a color
gradient whose steepness encodes coefficient magnitude and whose direction
encodes coefficient sign.

Disc geometry (290mm / ~11.4 inch):
  Outer edge      →  white background
  141–143.5mm     →  clock track (alternating B/W, 8 red reference markers)
  141–62mm        →  48 audio rings  (L0 inner/bass → L2 outer/high)
  62–55mm         →  label gap
  55mm            →  center label (gray)
  7mm             →  spindle hole (white)

Usage:
    python3 dsa_render.py song.disc.json
    python3 dsa_render.py song.disc.json --dpi 300 --out disc.png
    python3 dsa_render.py song.disc.json --dpi 600 --out disc_hires.png

License: GPL v3 — github.com/pisdronio/dsa
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

try:
    from PIL import Image, ImageDraw
except ImportError:
    sys.exit("Pillow required — pip install Pillow")

from dsa_color import rgb_to_lab, lab_to_rgb


# ─── Disc geometry (all in mm) ────────────────────────────────────────────────

DISC_DIAMETER_MM   = 290.0   # nominal 12-inch equivalent
OUTER_AUDIO_MM     = 141.0   # outer edge of audio band rings
INNER_AUDIO_MM     = 62.0    # inner edge of audio band rings
CLOCK_WIDTH_MM     = 2.5     # clock track ring width
CLOCK_SEGMENTS     = 300     # alternating B/W segments in clock track
REF_MARKERS        = 8       # red reference marker dots on clock track
LABEL_RADIUS_MM    = 55.0    # outer edge of center label circle
SPINDLE_RADIUS_MM  = 7.0     # spindle hole radius

NUM_BANDS          = 48

# ─── Fiducial marker ──────────────────────────────────────────────────────────
#
# Magenta (255,0,255) is absent from the 8-color DSA palette and unreachable
# by gradient interpolation between any band-pair endpoints.  The detector in
# dsa_reader.py isolates it with a three-channel threshold that no audio
# content can satisfy, regardless of what is encoded.
#
# Change both constants together if the fiducial color is ever revised.

FIDUCIAL_RGB       = (255, 0, 255)
FIDUCIAL_THRESHOLD = dict(r_min=200, g_max=50, b_min=200)

# Minimum physical size (mm) that survives print + phone-photo at arm's length.
FIDUCIAL_MM        = 8.0

# ─── Color palette ────────────────────────────────────────────────────────────

PALETTE: dict[str, np.ndarray] = {
    'black':  np.array([0,   0,   0],   dtype=np.float32),
    'white':  np.array([255, 255, 255], dtype=np.float32),
    'red':    np.array([220, 50,  50],  dtype=np.float32),
    'green':  np.array([50,  180, 50],  dtype=np.float32),
    'blue':   np.array([50,  50,  220], dtype=np.float32),
    'yellow': np.array([240, 220, 0],   dtype=np.float32),
    'cyan':   np.array([0,   210, 210], dtype=np.float32),
    'purple': np.array([160, 50,  200], dtype=np.float32),
}


def render_disc(layout_path: str, dpi: int = 300, out_path: str = None,
                spiral: bool = False, rpm: float = 33.0) -> str:
    """
    Render a .disc.json layout to a PNG disc image.

    Parameters
    ----------
    layout_path : path to .disc.json produced by dsa_disc.py / dsa_cli.py disc
    dpi         : output resolution in DPI (300 = good print, 600 = high-res)
    out_path    : output PNG path; defaults to layout_path with .png extension

    Returns
    -------
    Path to the saved PNG file.
    """

    # ── Load layout ───────────────────────────────────────────────────────────
    print(f"  Loading {Path(layout_path).name} ...", end=' ', flush=True)
    with open(layout_path) as f:
        layout = json.load(f)

    n_frames   = layout['n_frames']
    n_bands    = layout['n_bands']
    duration_s = layout['duration_s']
    frames     = layout['frames']
    bp_map     = {b['band']: (b['color_a'], b['color_b']) for b in layout['band_pairs']}

    print(f"done  ({n_frames} frames, {duration_s:.1f}s, {n_bands} bands)")

    # ── Build steepness / direction lookup tables ─────────────────────────────
    print("  Building lookup tables...", end=' ', flush=True)

    steep = np.zeros((n_frames, n_bands), dtype=np.float32)
    dir_  = np.zeros((n_frames, n_bands), dtype=np.int8)

    for fd in frames:
        fi = fd['frame_idx']
        if fd['silence']:
            continue
        for bd in fd['bands']:
            b             = bd['band']
            steep[fi, b]  = bd['steepness']
            dir_[fi, b]   = bd['direction']

    # Per-band color arrays: shape (n_bands, 3)
    ca_rgb = np.zeros((n_bands, 3), dtype=np.float64)
    cb_rgb = np.zeros((n_bands, 3), dtype=np.float64)
    for b in range(n_bands):
        ca_rgb[b] = PALETTE[bp_map[b][0]]
        cb_rgb[b] = PALETTE[bp_map[b][1]]

    # Pre-convert to LAB for perceptually uniform interpolation
    ca_lab = rgb_to_lab(ca_rgb)   # (n_bands, 3)
    cb_lab = rgb_to_lab(cb_rgb)   # (n_bands, 3)

    print("done")

    # ── Image dimensions ──────────────────────────────────────────────────────
    mm_to_px       = dpi / 25.4
    disc_px        = int(DISC_DIAMETER_MM * mm_to_px)
    disc_px       += disc_px % 2          # keep even
    ctr            = disc_px // 2         # center x and y (square image)

    outer_audio_px = OUTER_AUDIO_MM  * mm_to_px
    inner_audio_px = INNER_AUDIO_MM  * mm_to_px
    ring_width_px  = (outer_audio_px - inner_audio_px) / n_bands
    clock_inner_px = outer_audio_px
    clock_outer_px = (OUTER_AUDIO_MM + CLOCK_WIDTH_MM) * mm_to_px
    label_px       = LABEL_RADIUS_MM * mm_to_px
    spindle_px     = SPINDLE_RADIUS_MM * mm_to_px
    disc_r_px      = (DISC_DIAMETER_MM / 2) * mm_to_px

    arc_w_outer_px = (2 * math.pi * outer_audio_px) / n_frames
    arc_w_inner_px = (2 * math.pi * inner_audio_px) / n_frames
    arc_w_outer_mm = arc_w_outer_px / mm_to_px
    arc_w_inner_mm = arc_w_inner_px / mm_to_px

    print(f"  Canvas:    {disc_px}×{disc_px} px  ({disc_px/dpi:.2f}\" diameter at {dpi} DPI)")
    print(f"  Ring width: {ring_width_px/mm_to_px:.2f}mm  ({ring_width_px:.1f}px)")
    if not spiral:
        print(f"  Arc width:  {arc_w_outer_mm:.3f}mm outer  /  {arc_w_inner_mm:.3f}mm inner  (single revolution)")
        if arc_w_inner_mm < 0.3:
            print("  WARNING: inner arc < 0.3mm — may be too small to print/read at this DPI")

    # ── Spiral geometry ────────────────────────────────────────────────────────
    frames_per_rev = 1     # placeholders overwritten in spiral mode
    n_revolutions  = 1
    sub_band_px    = ring_width_px

    if spiral:
        if rpm < 1.0:
            sys.exit("  ERROR: --rpm must be >= 1.0")
        fps_actual     = n_frames / duration_s
        frames_per_rev = max(1, round(fps_actual / (rpm / 60.0)))
        n_revolutions  = n_frames // frames_per_rev
        if n_revolutions < 1:
            sys.exit(f"  ERROR: n_frames={n_frames} < frames_per_rev={frames_per_rev}. "
                     f"Audio too short for spiral at {rpm} RPM.")
        leftover = n_frames - n_revolutions * frames_per_rev
        sub_band_px  = ring_width_px / n_revolutions
        sub_band_mm  = sub_band_px / mm_to_px
        arc_spiral_outer_mm = (2 * math.pi * outer_audio_px / frames_per_rev) / mm_to_px
        arc_spiral_inner_mm = (2 * math.pi * inner_audio_px / frames_per_rev) / mm_to_px
        print(f"  Mode:      SPIRAL  {rpm} RPM")
        print(f"  Frames/rev:{frames_per_rev}  |  Revolutions: {n_revolutions}"
              + (f"  (+{leftover} leftover frames)" if leftover else ""))
        print(f"  Arc width:  {arc_spiral_outer_mm:.2f}mm outer  /  {arc_spiral_inner_mm:.2f}mm inner")
        print(f"  Sub-band:  {sub_band_mm:.3f}mm  ({sub_band_px:.2f}px/revolution)")
        if sub_band_px < 4.0:
            safe_rev = int(ring_width_px / 4.0)
            safe_rpm = round((fps_actual / (n_frames / safe_rev)) * 60) if safe_rev > 0 else 0
            print(f"  WARNING: sub-band {sub_band_px:.2f}px < 4px at {dpi} DPI — "
                  f"revolutions too dense to resolve. "
                  f"Max readable revolutions at {dpi} DPI: {safe_rev} (~{safe_rpm} RPM). "
                  f"Consider higher DPI or lower --rpm.")

    # ── Pixel polar coordinates ───────────────────────────────────────────────
    print("  Computing pixel geometry...", end=' ', flush=True)

    yy, xx = np.mgrid[0:disc_px, 0:disc_px].astype(np.float32)
    ddx    = xx - ctr
    ddy    = yy - ctr
    r      = np.sqrt(ddx * ddx + ddy * ddy)

    # theta: 0 = 12 o'clock (top), increases clockwise, range [0, 2π)
    theta  = np.arctan2(ddx, -ddy) % (2.0 * math.pi)

    print("done")

    # ── Render ────────────────────────────────────────────────────────────────
    print("  Rendering audio bands...", end=' ', flush=True)

    # Start with white canvas
    img = np.full((disc_px, disc_px, 3), 255, dtype=np.uint8)

    # --- Audio ring mask ---
    in_audio = (r >= inner_audio_px) & (r < outer_audio_px)

    # Band index per pixel: 0 = innermost (bass), n_bands-1 = outermost (high)
    b_idx = ((r - inner_audio_px) / ring_width_px).astype(np.int32)
    np.clip(b_idx, 0, n_bands - 1, out=b_idx)

    # Frame index and arc position per pixel
    if not spiral:
        # ── Single-revolution mode ─────────────────────────────────────────────
        frame_angle = 2.0 * math.pi / n_frames
        f_idx       = (theta / frame_angle).astype(np.int32)
        np.clip(f_idx, 0, n_frames - 1, out=f_idx)
        arc_pos     = (theta % frame_angle) / frame_angle  # 0..1 within arc
    else:
        # ── Spiral mode ────────────────────────────────────────────────────────
        # Each band's radial zone is subdivided into n_revolutions tracks.
        # Revolution 0 = outermost within the band = start of audio (vinyl-style).
        # r_from_outer = distance inward from the outer edge of the band.
        outer_edges  = (inner_audio_px
                        + (b_idx.astype(np.float32) + 1.0) * ring_width_px)
        r_from_outer = outer_edges - r
        rev_idx      = (r_from_outer / sub_band_px).astype(np.int32)
        np.clip(rev_idx, 0, n_revolutions - 1, out=rev_idx)

        angle_per_lf  = 2.0 * math.pi / frames_per_rev
        f_local_float = theta / angle_per_lf
        f_local_int   = f_local_float.astype(np.int32)
        np.clip(f_local_int, 0, frames_per_rev - 1, out=f_local_int)

        f_idx   = rev_idx * frames_per_rev + f_local_int
        np.clip(f_idx, 0, n_frames - 1, out=f_idx)
        arc_pos = f_local_float - f_local_int.astype(np.float32)  # fractional part within arc

    # Extract audio pixels
    ay, ax   = np.where(in_audio)
    fi       = f_idx[ay, ax]
    bi       = b_idx[ay, ax]
    pos      = arc_pos[ay, ax]

    # Steepness and direction per pixel
    s = steep[fi, bi]
    d = dir_[fi, bi].astype(np.float32)

    # Blend factor t ∈ [0, 1]:
    #   direction =  0 → solid color_a  (t = 0)
    #   direction = +1 → A→B gradient   (t = arc_pos × steepness)
    #   direction = -1 → B→A gradient   (t = steepness × (1 − arc_pos))
    t = np.where(d == 0,  0.0,
        np.where(d > 0,   pos * s,
                          s * (1.0 - pos))).astype(np.float32)
    np.clip(t, 0.0, 1.0, out=t)

    # Perceptually uniform color interpolation via CIELAB
    t3        = t[:, np.newaxis]                                        # (N, 1)
    mixed_lab = ca_lab[bi] + (cb_lab[bi] - ca_lab[bi]) * t3            # (N, 3)
    img[ay, ax] = lab_to_rgb(mixed_lab)                                 # (N, 3) uint8

    print("done")

    # --- Clock track (alternating B/W) ---
    print("  Rendering clock track...", end=' ', flush=True)
    clock_y, clock_x = np.where((r >= clock_inner_px) & (r < clock_outer_px))
    if len(clock_y):
        segs = (theta[clock_y, clock_x] / (2.0 * math.pi) * CLOCK_SEGMENTS
                ).astype(np.int32) % CLOCK_SEGMENTS
        even = segs % 2 == 0
        img[clock_y[even],  clock_x[even]]  = (0,   0,   0)    # black
        img[clock_y[~even], clock_x[~even]] = (255, 255, 255)   # white
    print("done")

    # --- Center label and spindle hole ---
    img[r < label_px]   = (210, 210, 210)   # light gray label
    img[r < spindle_px] = (255, 255, 255)   # white spindle hole

    # --- Disc boundary (outside disc circle → white) ---
    img[r >= disc_r_px] = (255, 255, 255)

    # --- Reference marker dots (8 red circles on clock track) ---
    pil_img = Image.fromarray(img)
    draw    = ImageDraw.Draw(pil_img)
    ref_r   = (clock_inner_px + clock_outer_px) / 2.0
    dot_r   = max(4, int(mm_to_px * 1.8))
    for i in range(REF_MARKERS):
        angle   = i * 2.0 * math.pi / REF_MARKERS
        px_x    = ctr + ref_r * math.sin(angle)
        px_y    = ctr - ref_r * math.cos(angle)
        draw.ellipse(
            [px_x - dot_r, px_y - dot_r, px_x + dot_r, px_y + dot_r],
            fill=(230, 40, 40),
        )

    # --- Fiducial corner squares ---
    # Four magenta squares in the canvas corners.  The disc is circular; the
    # image corners are always white background, so the squares never overlap
    # audio content.  Size: max(8px, FIDUCIAL_MM mm) so they survive rescaling.
    fid_px = max(8, int(FIDUCIAL_MM * mm_to_px))
    for (x0, y0) in [
        (0,              0),               # TL
        (disc_px - fid_px, 0),             # TR
        (0,              disc_px - fid_px),# BL
        (disc_px - fid_px, disc_px - fid_px),  # BR
    ]:
        draw.rectangle([x0, y0, x0 + fid_px - 1, y0 + fid_px - 1],
                       fill=FIDUCIAL_RGB)

    # ── Save ──────────────────────────────────────────────────────────────────
    if out_path is None:
        base = str(Path(layout_path).with_suffix('').with_suffix(''))  # strip .disc.json
        if spiral:
            out_path = base + f'.spiral_{int(rpm)}rpm.png'
        else:
            out_path = str(Path(layout_path).with_suffix('.png'))

    print(f"  Saving → {out_path} ...", end=' ', flush=True)
    pil_img.save(out_path, dpi=(dpi, dpi))
    size_mb = Path(out_path).stat().st_size / 1024 / 1024
    print(f"done  ({size_mb:.1f} MB)")

    return out_path


def main():
    p = argparse.ArgumentParser(
        description='Render a DSA .disc.json layout to a printable disc PNG')
    p.add_argument('layout',
                   help='Path to .disc.json file (from dsa_cli.py disc)')
    p.add_argument('--dpi',    type=int,   default=300,
                   help='Output resolution in DPI (default: 300)')
    p.add_argument('--out',    type=str,   default=None,
                   help='Output PNG path')
    p.add_argument('--spiral', action='store_true',
                   help='Spiral geometry: multiple revolutions per band (vinyl-style)')
    p.add_argument('--rpm',    type=float, default=33.0,
                   help='Disc speed for spiral mode in RPM (default: 33)')
    args = p.parse_args()

    print()
    print("  DSA — Digilog Scalable Audio")
    print("  Disc Renderer")
    print("  ─────────────────────────────────────────────────")
    print(f"  Layout:  {args.layout}")
    print(f"  DPI:     {args.dpi}")
    print()

    out = render_disc(args.layout, dpi=args.dpi, out_path=args.out,
                      spiral=args.spiral, rpm=args.rpm)

    print()
    print("  Scan the groove.")
    print()


if __name__ == '__main__':
    main()
