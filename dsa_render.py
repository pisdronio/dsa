#!/usr/bin/env python3
"""
DSA — Digilog Scalable Audio
dsa_render.py — Disc Image Renderer + Strip Render Modes

Three distinct render targets.  Never mix them.

──────────────────────────────────────────────────────────────────
DEFAULT (no --mode)   Disc image — circular 290mm disc PNG / TIFF
──────────────────────────────────────────────────────────────────
    python3 dsa_render.py song.disc.json
    python3 dsa_render.py song.disc.json --dpi 600 --output-format cmyk

──────────────────────────────────────────────────────────────────
--mode window         DEV TOOL — screen display + phone photo test
──────────────────────────────────────────────────────────────────
Renders only as many frames as fit within 100mm at 3mm/cell.
All four magenta fiducial corners must be visible in the phone frame.
100mm ÷ 3mm/cell = 33 frames ≈ 0.75s — this IS the "window".
RGB PNG only (screen display).  Do NOT print this file.

    python3 dsa_render.py song.disc.json --mode window
    → song.window.png

Workflow:
  1. Display song.window.png fullscreen on monitor
  2. Photograph monitor with phone at 20-30cm
  3. python3 dsa_reader.py photo.jpg song.disc.json --source monitor
  4. If direction accuracy >80% proceed to --mode print

──────────────────────────────────────────────────────────────────
--mode print          PRODUCTION — physical print, CMYK TIFF only
──────────────────────────────────────────────────────────────────
Renders full song.  4 inches × 2 inches per tile, tiled as needed.
300% TAC cap.  DPI metadata embedded.  Only after window test passes.

    python3 dsa_render.py song.disc.json --mode print
    → song.print.tif   (multi-tile CMYK TIFF, send directly to printer)
    python3 dsa_render.py song.disc.json --mode print --dpi 300

Workflow:
  1. Only after window mode accuracy >80%
  2. Send TIFF directly to printer — do not convert to PDF or PNG
  3. Photograph printed strip
  4. python3 dsa_reader.py photo.jpg song.disc.json --source print

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

from dsa_color import rgb_to_lab, lab_to_rgb, PALETTE_RGB, FIDUCIAL_RGB, FIDUCIAL_THRESHOLD
from dsa_strip import _band_gradient_row as _strip_gradient_row


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
# FIDUCIAL_RGB and FIDUCIAL_THRESHOLD imported from dsa_color (§18.5).
# Minimum physical size that survives print + phone-photo at arm's length.

FIDUCIAL_MM = 8.0

# ─── Color palette ────────────────────────────────────────────────────────────

# Canonical RGB from dsa_color — single source of truth (§18.5).
PALETTE: dict[str, np.ndarray] = {
    name: np.array(rgb, dtype=np.float32)
    for name, rgb in PALETTE_RGB.items()
}


# ─── Strip render constants ───────────────────────────────────────────────────

_STRIP_SEP_PX   = 2     # separator line height between layers (matches dsa_strip.py)
_STRIP_L0_END   = 8
_STRIP_L1_END   = 24
_STRIP_N_BANDS  = 48

# Window mode (MODE 1) — screen display, phone photo test
#
# Physical sizing rationale (§18.4):
#   Phone camera at 20-30cm frames roughly 150-250mm of width.
#   The strip must fit entirely within that frame so all four magenta
#   fiducials (§18.2) are visible — the reader needs all four corners
#   to compute the perspective warp.
#
#   Primary constraint: total strip width ≤ _WINDOW_TARGET_W_MM.
#   Secondary constraint: duration cap of _WINDOW_DURATION_S.
#   Frames rendered = min(frames_by_width, frames_by_duration).
#
#   Cell width is fixed at _WINDOW_MIN_CELL_MM — the minimum the phone
#   camera resolves as a distinct gradient at 20-30cm.  Width target
#   then determines how many frames fit, not the other way around.
#
_WINDOW_DPI         = 96        # standard monitor DPI
_WINDOW_TARGET_W_MM = 100.0     # target physical strip width — must fit in phone FOV
_WINDOW_DURATION_S  = 3.0       # maximum duration cap (secondary constraint)
_WINDOW_MIN_CELL_MM = 3.0       # fixed cell width — minimum phone-resolvable at 20-30cm
_WINDOW_BAND_H_MM   = 1.5       # 48 bands × 1.5mm = 72mm content height
_WINDOW_BORDER_MM   = 8.0       # border width — fiducial squares fill the corners

# Print mode (MODE 2) — CMYK TIFF, physical print production
_PRINT_DPI          = 600       # preferred; 300 accepted via --dpi
_PRINT_PAGE_W_IN    = 4.0       # tile width (inches)
_PRINT_PAGE_H_IN    = 2.0       # tile height (inches)
_PRINT_BORDER_MM    = 8.0       # border for fiducials
_PRINT_MIN_CELL_MM  = 0.3       # Digilog Rig minimum arc (§12) — ~7px at 600 DPI
_PRINT_MAX_CELL_MM  = 10.0      # cap cell width for very short clips


# ─── Strip render helpers ─────────────────────────────────────────────────────

def _render_strip_content(layout: dict, frame_start: int, n_frames: int,
                           cell_w_px: int, band_h_px: int) -> np.ndarray:
    """
    Render strip gradient content for n_frames starting at frame_start.
    No border, no fiducials.  Returns (H, W, 3) uint8 RGB array.
    """
    n_bands = layout['n_bands']
    bp_map  = {b['band']: (b['color_a'], b['color_b']) for b in layout['band_pairs']}
    img_h   = n_bands * band_h_px + 2 * _STRIP_SEP_PX
    img_w   = n_frames * cell_w_px
    img     = np.zeros((img_h, img_w, 3), dtype=np.uint8)

    # Build per-cell lookup tables for this segment
    steep = np.zeros((n_frames, n_bands), dtype=np.float32)
    dir_  = np.zeros((n_frames, n_bands), dtype=np.int8)
    for fd in layout['frames']:
        fi = fd['frame_idx'] - frame_start
        if fi < 0 or fi >= n_frames or fd['silence']:
            continue
        for bd in fd['bands']:
            b = bd['band']
            steep[fi, b] = bd['steepness']
            dir_[fi, b]  = bd['direction']

    for b in range(n_bands):
        ca_name, cb_name = bp_map[b]
        ca = PALETTE[ca_name]
        cb = PALETTE[cb_name]
        sep_below = (_STRIP_SEP_PX if b >= _STRIP_L0_END else 0) + \
                    (_STRIP_SEP_PX if b >= _STRIP_L1_END else 0)
        row_top   = img_h - b * band_h_px - sep_below - band_h_px
        for i in range(n_frames):
            row = _strip_gradient_row(float(steep[i, b]), int(dir_[i, b]),
                                      ca, cb, cell_w_px)
            img[row_top : row_top + band_h_px,
                i * cell_w_px : (i + 1) * cell_w_px] = row[np.newaxis, :, :]

    # Separator lines (dark gray) between L0/L1 and L1/L2
    sep0_top = img_h - _STRIP_L0_END * band_h_px - _STRIP_SEP_PX
    sep1_top = img_h - _STRIP_L1_END * band_h_px - 2 * _STRIP_SEP_PX
    img[sep0_top : sep0_top + _STRIP_SEP_PX, :] = (80, 80, 80)
    img[sep1_top : sep1_top + _STRIP_SEP_PX, :] = (80, 80, 80)

    return img


def _add_strip_border(content: np.ndarray, border_px: int,
                       mm_to_px: float) -> Image.Image:
    """
    Wrap strip content in a white border.  Add:
      - Magenta fiducials (FIDUCIAL_RGB) filling each corner square
      - 8-color calibration patches centred in the bottom border (§18.4)

    Both elements use colors from dsa_color only — no hardcoded RGB.

    Returns a PIL Image (RGB mode).
    """
    old_h, old_w = content.shape[:2]
    new_w = old_w + 2 * border_px
    new_h = old_h + 2 * border_px

    bordered = Image.new('RGB', (new_w, new_h), (255, 255, 255))
    bordered.paste(Image.fromarray(content), (border_px, border_px))
    draw = ImageDraw.Draw(bordered)

    # Fiducials: square = full border_px × border_px in each corner
    fid = border_px
    for x0, y0 in [(0, 0), (new_w - fid, 0),
                   (0, new_h - fid), (new_w - fid, new_h - fid)]:
        draw.rectangle([x0, y0, x0 + fid - 1, y0 + fid - 1], fill=FIDUCIAL_RGB)

    # Calibration patches: one per palette color, centred in bottom border.
    # patch size is capped so all 8 patches + gaps fit within the image width.
    n_cal      = len(PALETTE_RGB)               # 8
    gap_px     = max(1, round(mm_to_px * 1.0))  # 1mm gap between patches
    max_patch  = max(4, (new_w - (n_cal - 1) * gap_px) // n_cal)
    patch_h    = min(max(4, border_px - 4), max_patch)
    row_w      = n_cal * patch_h + (n_cal - 1) * gap_px
    cal_x0     = new_w // 2 - row_w // 2
    cal_y0  = new_h - border_px + (border_px - patch_h) // 2
    for i, (_, rgb) in enumerate(PALETTE_RGB.items()):
        px = cal_x0 + i * (patch_h + gap_px)
        draw.rectangle([px, cal_y0, px + patch_h - 1, cal_y0 + patch_h - 1], fill=rgb)

    return bordered


def _to_cmyk_tiff(pil_rgb: Image.Image, dpi: int) -> Image.Image:
    """
    Convert an RGB PIL image to a TAC-capped CMYK PIL image ready for TIFF save.
    300% total area coverage cap applied (§18.3).
    """
    arr_cmyk = np.array(pil_rgb.convert('CMYK'), dtype=np.uint16)
    total    = arr_cmyk.sum(axis=2, keepdims=True)
    cap      = 300 * 255 // 100
    scale    = np.where(total > cap, cap / (total.astype(np.float32) + 1e-6), 1.0)
    arr_cmyk = np.clip(arr_cmyk * scale, 0, 255).astype(np.uint8)
    return Image.fromarray(arr_cmyk, mode='CMYK')


def render_disc(layout_path: str, dpi: int = 300, out_path: str = None,
                spiral: bool = False, rpm: float = 33.0,
                output_format: str = 'rgb') -> str:
    """
    Render a .disc.json layout to a disc image.

    Parameters
    ----------
    layout_path   : path to .disc.json produced by dsa_disc.py / dsa_cli.py disc
    dpi           : output resolution in DPI (300 = good print, 600 = high-res)
    out_path      : output path; defaults to layout_path with appropriate extension
    output_format : 'rgb' (default) → RGB PNG; 'cmyk' → CMYK TIFF for print

    Returns
    -------
    Path to the saved file.
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
        (0,                0),
        (disc_px - fid_px, 0),
        (0,                disc_px - fid_px),
        (disc_px - fid_px, disc_px - fid_px),
    ]:
        draw.rectangle([x0, y0, x0 + fid_px - 1, y0 + fid_px - 1],
                       fill=FIDUCIAL_RGB)

    # --- Calibration patch strip (§18.4) ---
    # One solid 8×8mm patch per palette color, in PALETTE_RGB key order,
    # arranged in a horizontal row centred inside the center label area
    # (r < label_px).  The reader samples these patches to build a per-read
    # Lab affine remap that compensates for ink, paper, and camera variance.
    cal_px   = max(8, int(8.0 * mm_to_px))   # 8mm per patch
    gap_px   = max(2, int(1.0 * mm_to_px))   # 1mm gap between patches
    n_cal    = len(PALETTE_RGB)               # 8
    row_w    = n_cal * cal_px + (n_cal - 1) * gap_px
    cal_x0   = ctr - row_w // 2
    cal_y0   = ctr - cal_px // 2             # centred vertically in label
    for i, (name, rgb) in enumerate(PALETTE_RGB.items()):
        px = cal_x0 + i * (cal_px + gap_px)
        draw.rectangle([px, cal_y0, px + cal_px - 1, cal_y0 + cal_px - 1],
                       fill=rgb)

    # ── Save ──────────────────────────────────────────────────────────────────
    if out_path is None:
        base = str(Path(layout_path).with_suffix('').with_suffix(''))
        if output_format == 'cmyk':
            suffix = f'.spiral_{int(rpm)}rpm.tif' if spiral else '.cmyk.tif'
        else:
            suffix = f'.spiral_{int(rpm)}rpm.png' if spiral else '.png'
        out_path = base + suffix

    print(f"  Saving → {out_path} ...", end=' ', flush=True)
    if output_format == 'cmyk':
        # Convert RGB→CMYK.  PIL's built-in conversion applies a simple
        # channel inversion (no ICC profile); tune with a RIP for production.
        # Total ink coverage is capped at 300% to avoid paper saturation.
        pil_cmyk = pil_img.convert('CMYK')
        arr_cmyk = np.array(pil_cmyk, dtype=np.uint16)
        total    = arr_cmyk.sum(axis=2, keepdims=True)
        cap      = 300 * 255 // 100         # 300% TAC in PIL units (0-255)
        over     = total > cap
        scale    = np.where(over, cap / (total + 1e-6), 1.0)
        arr_cmyk = np.clip(arr_cmyk * scale, 0, 255).astype(np.uint8)
        from PIL import Image as _PILImage
        pil_img  = _PILImage.fromarray(arr_cmyk, mode='CMYK')
        pil_img.save(out_path, format='TIFF', dpi=(dpi, dpi))
    else:
        pil_img.save(out_path, dpi=(dpi, dpi))
    size_mb = Path(out_path).stat().st_size / 1024 / 1024
    print(f"done  ({size_mb:.1f} MB)")

    return out_path


def render_window(layout_path: str, out_path: str = None) -> str:
    """
    MODE 1 — Window test render (dev tool, screen only).

    Width target (100mm) is the primary constraint — it determines how many
    frames are rendered.  Cell is fixed at 3mm (minimum phone-resolvable at
    20-30cm); 100mm ÷ 3mm = 33 frames.  Duration (3s) is only a secondary
    cap.  This ensures all four magenta fiducial corners fit within the
    phone camera's field of view in one shot.  Do NOT print this file.

    Parameters
    ----------
    layout_path : .disc.json path
    out_path    : output PNG; defaults to <base>.window.png
    """
    print("  Mode: WINDOW — screen display / phone test  (RGB PNG, do not print)")

    with open(layout_path) as f:
        layout = json.load(f)

    n_frames   = layout['n_frames']
    duration_s = layout['duration_s']
    n_bands    = layout['n_bands']
    fps        = n_frames / duration_s

    # Frame count: width target is primary, duration cap is secondary.
    # Cell width is fixed — minimum the phone resolves at 20-30cm.
    cell_w_mm         = _WINDOW_MIN_CELL_MM
    n_frames_by_width = max(1, int(_WINDOW_TARGET_W_MM / cell_w_mm))
    n_frames_by_dur   = max(1, round(_WINDOW_DURATION_S * fps))
    n_frames_w        = min(n_frames, n_frames_by_width, n_frames_by_dur)
    mm_to_px          = _WINDOW_DPI / 25.4
    cell_w_px  = max(1, round(cell_w_mm * mm_to_px))
    band_h_px  = max(1, round(_WINDOW_BAND_H_MM * mm_to_px))
    border_px  = max(8, round(_WINDOW_BORDER_MM * mm_to_px))

    content_w_mm = n_frames_w * cell_w_mm
    content_h_mm = n_bands * _WINDOW_BAND_H_MM
    total_w_mm   = content_w_mm + 2 * _WINDOW_BORDER_MM
    total_h_mm   = content_h_mm + 2 * _WINDOW_BORDER_MM
    total_w_px   = n_frames_w * cell_w_px + 2 * border_px
    total_h_px   = n_bands * band_h_px + 2 * _STRIP_SEP_PX + 2 * border_px

    print(f"  DPI:       {_WINDOW_DPI}  (screen)")
    print(f"  Window:    {n_frames_w} frames  "
          f"({n_frames_w / fps:.2f}s at {fps:.1f} fps)  "
          f"[width cap: {_WINDOW_TARGET_W_MM:.0f}mm / {cell_w_mm:.0f}mm = "
          f"{n_frames_by_width} frames,  dur cap: {n_frames_by_dur} frames]")
    print(f"  Cell:      {cell_w_mm:.1f}mm wide × {_WINDOW_BAND_H_MM:.1f}mm tall"
          f"  ({cell_w_px}×{band_h_px}px)")
    print(f"  Content:   {content_w_mm:.1f}mm × {content_h_mm:.1f}mm")
    print(f"  Total:     {total_w_mm:.1f}mm × {total_h_mm:.1f}mm"
          f"  ({total_w_px}×{total_h_px}px)")

    print("  Rendering gradient content...", end=' ', flush=True)
    content = _render_strip_content(layout, 0, n_frames_w, cell_w_px, band_h_px)
    print("done")

    pil_img = _add_strip_border(content, border_px, mm_to_px)

    if out_path is None:
        base     = str(Path(layout_path).with_suffix('').with_suffix(''))
        out_path = base + '.window.png'

    print(f"  Saving → {out_path} ...", end=' ', flush=True)
    pil_img.save(out_path, dpi=(_WINDOW_DPI, _WINDOW_DPI))
    size_kb = Path(out_path).stat().st_size / 1024
    print(f"done  ({size_kb:.0f} KB)")

    return out_path


def render_print(layout_path: str, dpi: int = _PRINT_DPI,
                 out_path: str = None) -> str:
    """
    MODE 2 — Print production render (CMYK TIFF, physical print only).

    Renders full song as a multi-tile CMYK TIFF.  Each tile is
    4 inches × 2 inches at target DPI.  300% TAC cap enforced.
    Only use after window mode accuracy >80%.

    Parameters
    ----------
    layout_path : .disc.json path
    dpi         : output DPI (default 600; 300 accepted)
    out_path    : output TIFF; defaults to <base>.print.tif
    """
    print("  Mode: PRINT — physical production  (CMYK TIFF, send directly to printer)")
    print("  ⚠  Only run after window mode accuracy >80%.")

    with open(layout_path) as f:
        layout = json.load(f)

    n_frames   = layout['n_frames']
    duration_s = layout['duration_s']
    n_bands    = layout['n_bands']

    mm_to_px       = dpi / 25.4
    page_w_mm      = _PRINT_PAGE_W_IN * 25.4   # 101.6mm
    page_h_mm      = _PRINT_PAGE_H_IN * 25.4   # 50.8mm
    border_px      = max(8, round(_PRINT_BORDER_MM * mm_to_px))
    content_w_px   = round(page_w_mm * mm_to_px) - 2 * border_px
    content_h_px   = round(page_h_mm * mm_to_px) - 2 * border_px
    band_h_px      = max(1, (content_h_px - 2 * _STRIP_SEP_PX) // n_bands)
    # Cell width fills the page for short clips; _PRINT_MIN_CELL_MM is a floor
    # (drives multi-tile for long songs), _PRINT_MAX_CELL_MM is a ceiling.
    cell_w_mm      = max(_PRINT_MIN_CELL_MM,
                         min(_PRINT_MAX_CELL_MM, content_w_px / mm_to_px / n_frames))
    cell_w_px      = max(1, round(cell_w_mm * mm_to_px))
    frames_per_tile = max(1, content_w_px // cell_w_px)
    n_tiles        = math.ceil(n_frames / frames_per_tile)

    cell_w_mm  = cell_w_px / mm_to_px
    band_h_mm  = band_h_px / mm_to_px
    tile_w_px  = frames_per_tile * cell_w_px + 2 * border_px
    tile_h_px  = n_bands * band_h_px + 2 * _STRIP_SEP_PX + 2 * border_px

    print(f"  DPI:       {dpi}")
    print(f"  Page:      {page_w_mm:.1f}mm × {page_h_mm:.1f}mm"
          f"  ({round(page_w_mm * mm_to_px)}×{round(page_h_mm * mm_to_px)}px)")
    print(f"  Cell:      {cell_w_mm:.2f}mm × {band_h_mm:.2f}mm"
          f"  ({cell_w_px}×{band_h_px}px)")
    print(f"  Frames/tile: {frames_per_tile}"
          f"  ({frames_per_tile / (n_frames / duration_s):.1f}s per tile)")
    print(f"  Tiles:     {n_tiles}  (song: {n_frames} frames / {duration_s:.1f}s)")
    print(f"  Tile size: {tile_w_px}×{tile_h_px}px")

    tiles = []
    for t in range(n_tiles):
        frame_start    = t * frames_per_tile
        frame_end      = min(n_frames, frame_start + frames_per_tile)
        n_tile_frames  = frame_end - frame_start

        print(f"  Tile {t + 1}/{n_tiles}  frames {frame_start}–{frame_end - 1}...",
              end=' ', flush=True)
        content = _render_strip_content(layout, frame_start, n_tile_frames,
                                         cell_w_px, band_h_px)

        # Do not pad the last (partial) tile — let it be exactly
        # n_tile_frames wide so fiducials sit at the true content corners.

        pil_tile = _add_strip_border(content, border_px, mm_to_px)
        tiles.append(_to_cmyk_tiff(pil_tile, dpi))
        print("done")

    if out_path is None:
        base     = str(Path(layout_path).with_suffix('').with_suffix(''))
        out_path = base + '.print.tif'

    print(f"  Saving {n_tiles}-tile CMYK TIFF → {out_path} ...", end=' ', flush=True)
    tiles[0].save(out_path, format='TIFF', dpi=(dpi, dpi),
                  save_all=True, append_images=tiles[1:])
    size_mb = Path(out_path).stat().st_size / 1024 / 1024
    print(f"done  ({size_mb:.1f} MB)")

    return out_path


def main():
    p = argparse.ArgumentParser(
        description='Render DSA .disc.json — disc image or strip modes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    p.add_argument('layout',
                   help='Path to .disc.json file')
    p.add_argument('--mode', choices=['window', 'print'], default=None,
                   help=('"window" = 3s RGB PNG for monitor test  |  '
                         '"print" = full-song CMYK TIFF for physical print  |  '
                         'omit = disc image (default)'))
    p.add_argument('--dpi',    type=int,   default=None,
                   help=('Output DPI.  window: ignored (fixed 96).  '
                         'print: default 600.  disc: default 300.'))
    p.add_argument('--out',    type=str,   default=None,
                   help='Output file path')
    # Disc-mode-only flags (ignored in window/print modes)
    p.add_argument('--spiral', action='store_true',
                   help='[disc] Spiral geometry: multiple revolutions per band')
    p.add_argument('--rpm',    type=float, default=33.0,
                   help='[disc] Disc speed for spiral mode in RPM (default: 33)')
    p.add_argument('--output-format', choices=['rgb', 'cmyk'], default='rgb',
                   help='[disc] rgb → PNG (default);  cmyk → CMYK TIFF (§18.3)')
    args = p.parse_args()

    print()
    print("  DSA — Digilog Scalable Audio")
    print("  Renderer")
    print("  ─────────────────────────────────────────────────")
    print(f"  Layout: {args.layout}")
    print()

    if args.mode == 'window':
        if args.dpi is not None:
            print(f"  NOTE: --dpi ignored in window mode (fixed {_WINDOW_DPI} DPI for screen)")
        out = render_window(args.layout, out_path=args.out)
    elif args.mode == 'print':
        dpi = args.dpi if args.dpi is not None else _PRINT_DPI
        out = render_print(args.layout, dpi=dpi, out_path=args.out)
    else:
        dpi = args.dpi if args.dpi is not None else 300
        out = render_disc(args.layout, dpi=dpi, out_path=args.out,
                          spiral=args.spiral, rpm=args.rpm,
                          output_format=args.output_format)

    print()
    print("  Scan the groove.")
    print()


if __name__ == '__main__':
    main()
