#!/usr/bin/env python3
"""
DSA — Digilog Scalable Audio
dsa_camera.py — Physical Strip Camera Reader (Tier 1)

Reads a photograph of a printed DSA strip and extracts the encoded gradient
data, validating the visual → physical → optical → digital round-trip.

Pipeline
--------
  1. Load a photo of the printed strip (phone camera, flatbed scanner, etc.)
  2. Detect the 4 corner fiducial squares (black squares added by
     dsa_strip.py --fiducials) — or accept manual corner coordinates.
  3. Compute a perspective homography: photo coords → canonical strip space.
  4. Warp the photo to canonical strip dimensions.
  5. Sample each (frame, band) cell center in the warped image.
  6. Project sampled RGB onto the expected ca→cb color axis (same math as
     dsa_reader.py) → steepness, direction, confidence α.
  7. Report accuracy vs .disc.json and save optional overlays.

Strip geometry (matches dsa_strip.py --fiducials output)
---------------------------------------------------------
  Canonical image dimensions:
      width  = n_frames + 2 × border_px
      height = n_bands × cell_h + 2 × SEP_PX + 2 × border_px

  Content origin (top-left of actual strip data) = (border_px, border_px)

  Band row layout (bottom-up in content area):
      band b row_top (in content coords) = n_bands*cell_h + 2*SEP_PX - band_row_offset(b) - cell_h
      (see _band_row_top helper — matches dsa_strip.py exactly)

  Corner fiducials: black squares of size corner_px × corner_px at each
  corner of the full (bordered) image.

Usage
-----
    python3 dsa_camera.py photo.jpg  song.disc.json
    python3 dsa_camera.py photo.jpg  song.disc.json  --cell-h 8  --border 20
    python3 dsa_camera.py photo.jpg  song.disc.json  --save-warped warped.png
    python3 dsa_camera.py photo.jpg  song.disc.json  --out-overlay overlay.png
    python3 dsa_camera.py photo.jpg  song.disc.json  --decode song.dsa  --out recovered.wav

    # Manual corners (TL TR BR BL in photo pixel coords, space-separated x,y pairs):
    python3 dsa_camera.py photo.jpg  song.disc.json  --corners "10,12 820,8 825,610 5,615"

    # Skip OpenCV auto-detect (use Pillow only, requires --corners):
    python3 dsa_camera.py photo.jpg  song.disc.json  --corners "..."  --no-cv

License: GPL v3 — github.com/pisdronio/dsa
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

try:
    from PIL import Image as PILImage
except ImportError:
    sys.exit("Pillow required — pip install Pillow")

# ─── Strip geometry constants (must match dsa_strip.py) ───────────────────────

SEP_PX   = 2    # separator line height between layers
L0_END   = 8    # band index where L1 begins
L1_END   = 24   # band index where L2 begins

# ─── Colour palette (must match dsa_strip.py / dsa_reader.py) ─────────────────

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

DIRECTION_THRESHOLD = 0.008   # matches dsa_reader.py


# ─── Geometry helpers ──────────────────────────────────────────────────────────

def _band_row_top(b: int, n_bands: int, cell_h: int) -> int:
    """
    Top pixel row of band b within the content area (no border offset applied).
    Band 0 (bass) is at the bottom; band n_bands-1 (high) is at the top.
    Replicates dsa_strip.py band_row_top() exactly.
    """
    sep_below = 0
    if b >= L0_END:
        sep_below += SEP_PX
    if b >= L1_END:
        sep_below += SEP_PX
    bottom_offset = b * cell_h + sep_below
    content_h = n_bands * cell_h + 2 * SEP_PX
    return content_h - bottom_offset - cell_h


def _cell_sample_y(b: int, n_bands: int, cell_h: int, border_px: int) -> float:
    """Y pixel in the canonical (bordered) image to sample for band b."""
    row_top_content = _band_row_top(b, n_bands, cell_h)
    return border_px + row_top_content + cell_h / 2.0


def _cell_sample_x(fi: int, border_px: int) -> float:
    """X pixel in the canonical (bordered) image to sample for frame fi."""
    return border_px + fi + 0.5


# ─── Colour math (identical to dsa_reader.py) ─────────────────────────────────

def _color_to_blend(color: np.ndarray,
                    ca: np.ndarray,
                    cb: np.ndarray) -> tuple[float, float]:
    """
    Project RGB color onto the ca→cb axis.
    Returns (t, confidence): t∈[0,1] blend factor, confidence∈[0,1].
    """
    v     = cb - ca
    denom = np.dot(v, v)
    if denom < 1e-6:
        return 0.0, 1.0

    t         = float(np.clip(np.dot(color - ca, v) / denom, 0.0, 1.0))
    projected = ca + t * v
    residual  = float(np.linalg.norm(color - projected))
    max_dist  = float(np.linalg.norm(v))
    confidence = max(0.0, 1.0 - residual / (max_dist + 1e-9))
    return t, confidence


# ─── Fiducial detection ────────────────────────────────────────────────────────

def _detect_corners_cv(img_rgb: np.ndarray,
                        corner_fraction: float = 0.08,
                        debug: bool = False) -> np.ndarray | None:
    """
    Detect the 4 corner fiducial squares using OpenCV.

    Searches each image quadrant for the largest dark contour whose bounding
    box aspect ratio is close to 1 (square-ish).  Returns an (4, 2) float32
    array in order [TL, TR, BR, BL] of corner square *centre* points, or None
    on failure.

    corner_fraction : expected corner size as fraction of image shorter side.
                      At 300 DPI, 5 mm corner = 59 px, shorter side ≈ strip
                      height — adjust if strip is very short or very tall.
    """
    try:
        import cv2
    except ImportError:
        return None

    gray  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    h, w  = gray.shape

    # Adaptive threshold — handles uneven lighting from phone photos
    block = max(11, (min(h, w) // 20) | 1)   # odd, ≥11
    bw    = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, block, 10)

    # Minimum contour area: (corner_fraction × shorter_side)²  / 4
    min_area = ((corner_fraction * min(h, w)) ** 2) / 4.0
    max_area = ((corner_fraction * min(h, w)) ** 2) * 4.0

    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Keep only roughly square contours in the plausible size range
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        x, y, cw, ch = cv2.boundingRect(cnt)
        aspect = min(cw, ch) / max(cw, ch) if max(cw, ch) > 0 else 0
        if aspect < 0.6:   # not square enough
            continue
        cx, cy = x + cw / 2.0, y + ch / 2.0
        candidates.append((cx, cy, area))

    if debug:
        print(f"  [cv] {len(candidates)} candidate fiducials found")
        for i, (cx, cy, a) in enumerate(candidates):
            print(f"       {i}: ({cx:.0f}, {cy:.0f})  area={a:.0f}")

    if len(candidates) < 4:
        return None

    # Sort by area descending, take top 8 to be robust
    candidates.sort(key=lambda t: -t[2])
    top = candidates[:min(8, len(candidates))]

    # Assign to quadrants
    cx_mid = sum(t[0] for t in top) / len(top)
    cy_mid = sum(t[1] for t in top) / len(top)

    tl = [t for t in top if t[0] < cx_mid and t[1] < cy_mid]
    tr = [t for t in top if t[0] > cx_mid and t[1] < cy_mid]
    bl = [t for t in top if t[0] < cx_mid and t[1] > cy_mid]
    br = [t for t in top if t[0] > cx_mid and t[1] > cy_mid]

    if not (tl and tr and bl and br):
        return None

    # Pick the largest in each quadrant
    def _best(lst):
        return max(lst, key=lambda t: t[2])

    pts = np.float32([
        [_best(tl)[0], _best(tl)[1]],
        [_best(tr)[0], _best(tr)[1]],
        [_best(br)[0], _best(br)[1]],
        [_best(bl)[0], _best(bl)[1]],
    ])
    return pts


def _parse_manual_corners(corners_str: str) -> np.ndarray:
    """
    Parse --corners "x0,y0 x1,y1 x2,y2 x3,y3" → (4, 2) float32 [TL TR BR BL].
    """
    parts = corners_str.strip().split()
    if len(parts) != 4:
        sys.exit(f"--corners expects 4 'x,y' pairs, got {len(parts)}")
    pts = []
    for p in parts:
        xy = p.split(',')
        if len(xy) != 2:
            sys.exit(f"  Bad corner '{p}' — expect 'x,y'")
        pts.append([float(xy[0]), float(xy[1])])
    return np.float32(pts)


# ─── Homography & warping ──────────────────────────────────────────────────────

def _canonical_corners(n_frames: int, n_bands: int,
                        cell_h: int, border_px: int,
                        strip_dpi: int) -> np.ndarray:
    """
    Returns the expected [TL, TR, BR, BL] corner *centre* positions in the
    canonical (bordered) strip image, matching dsa_strip.py fiducial placement.
    corner_px = round(5mm × dpi/25.4) — must match dsa_strip.py exactly.
    """
    content_h  = n_bands * cell_h + 2 * SEP_PX
    new_w      = n_frames + 2 * border_px
    new_h      = content_h + 2 * border_px
    cp         = int(round(5.0 * strip_dpi / 25.4))   # 5mm corner squares
    tl = [cp / 2.0,             cp / 2.0            ]
    tr = [new_w - cp / 2.0,     cp / 2.0            ]
    br = [new_w - cp / 2.0,     new_h - cp / 2.0    ]
    bl = [cp / 2.0,             new_h - cp / 2.0    ]
    return np.float32([tl, tr, br, bl])


def _warp_strip(img_rgb: np.ndarray,
                src_corners: np.ndarray,
                dst_corners: np.ndarray,
                out_w: int, out_h: int) -> np.ndarray:
    """
    Compute perspective transform src→dst and warp img_rgb to (out_w × out_h).
    Falls back to PIL-only affine if OpenCV is unavailable (less accurate).
    """
    try:
        import cv2
        M   = cv2.getPerspectiveTransform(src_corners, dst_corners)
        img_bgr = img_rgb[:, :, ::-1]   # RGB → BGR for OpenCV
        warped_bgr = cv2.warpPerspective(img_bgr, M, (out_w, out_h),
                                          flags=cv2.INTER_LINEAR,
                                          borderMode=cv2.BORDER_REPLICATE)
        return warped_bgr[:, :, ::-1]   # BGR → RGB

    except ImportError:
        # PIL fallback: affine from 3 points (approximate, no perspective)
        # Map TL, TR, BL of src → TL, TR, BL of dst
        src3 = np.float64(src_corners[:3])
        dst3 = np.float64(dst_corners[:3])

        # Build affine [a b c; d e f] from 3-point correspondence
        A = np.column_stack([src3, np.ones(3)])
        ax = np.linalg.solve(A, dst3[:, 0])
        ay = np.linalg.solve(A, dst3[:, 1])
        # PIL transform expects (a b c d e f) in the *inverse* mapping
        # Use a simple crop + resize as a coarse fallback
        pil  = PILImage.fromarray(img_rgb)
        pil  = pil.resize((out_w, out_h), PILImage.BILINEAR)
        return np.array(pil)


# ─── Strip sampling ────────────────────────────────────────────────────────────

def _bilinear(arr: np.ndarray, x: float, y: float) -> np.ndarray:
    """Bilinear sample of (H, W, 3) uint8 array at float (x, y)."""
    h, w = arr.shape[:2]
    x0, y0 = int(x), int(y)
    x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
    x0, y0 = max(0, x0), max(0, y0)
    fx, fy = x - int(x), y - int(y)

    c00 = arr[y0, x0].astype(np.float64)
    c10 = arr[y0, x1].astype(np.float64)
    c01 = arr[y1, x0].astype(np.float64)
    c11 = arr[y1, x1].astype(np.float64)
    return (c00 * (1 - fx) * (1 - fy) +
            c10 * fx       * (1 - fy) +
            c01 * (1 - fx) * fy       +
            c11 * fx       * fy)


def read_strip(warped: np.ndarray,
               layout: dict,
               cell_h: int,
               border_px: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample the warped strip image and return read values.

    Returns
    -------
    steep_read : (n_frames, n_bands) float64
    dir_read   : (n_frames, n_bands) int8
    conf_frame : (n_frames, n_bands) float64
    alpha      : (n_bands,) float64  — per-band mean confidence
    """
    n_frames = layout['n_frames']
    n_bands  = layout['n_bands']
    bp_map   = {b['band']: (b['color_a'], b['color_b'])
                for b in layout['band_pairs']}

    steep_read = np.zeros((n_frames, n_bands), dtype=np.float64)
    dir_read   = np.zeros((n_frames, n_bands), dtype=np.int8)
    conf_frame = np.zeros((n_frames, n_bands), dtype=np.float64)

    print(f"  Sampling {n_frames} × {n_bands} cells ...", end=' ', flush=True)

    for b in range(n_bands):
        ca_name, cb_name = bp_map[b]
        ca = PALETTE[ca_name]
        cb = PALETTE[cb_name]

        # Two sample points per cell: left quarter and right quarter
        y = _cell_sample_y(b, n_bands, cell_h, border_px)

        for fi in range(n_frames):
            x_base = border_px + fi
            x_left  = x_base + 0.25
            x_right = x_base + 0.75

            c_left  = _bilinear(warped, x_left,  y)
            c_right = _bilinear(warped, x_right, y)

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
            print(f"{b + 1}..", end='', flush=True)

    print(" done")
    alpha = conf_frame.mean(axis=0)
    return steep_read, dir_read, conf_frame, alpha


# ─── Comparison ───────────────────────────────────────────────────────────────

def compare_layout(layout: dict,
                   steep_read: np.ndarray,
                   dir_read: np.ndarray) -> dict:
    """Compare camera read against original .disc.json values."""
    n_frames = layout['n_frames']
    n_bands  = layout['n_bands']

    orig_steep = np.zeros((n_frames, n_bands), dtype=np.float64)
    orig_dir   = np.zeros((n_frames, n_bands), dtype=np.int8)

    for fd in layout['frames']:
        fi = fd['frame_idx']
        if fd['silence']:
            continue
        for bd in fd['bands']:
            b = bd['band']
            orig_steep[fi, b] = bd['steepness']
            orig_dir[fi, b]   = bd['direction']

    steep_err = np.abs(steep_read - orig_steep)
    dir_match = (dir_read == orig_dir)

    return {
        'steepness_mae':      float(steep_err.mean()),
        'steepness_max_err':  float(steep_err.max()),
        'direction_accuracy': float(dir_match.mean()),
        'direction_errors':   int((~dir_match).sum()),
        'n_cells':            n_frames * n_bands,
    }


# ─── Overlays ─────────────────────────────────────────────────────────────────

def save_confidence_overlay(warped: np.ndarray,
                             conf_frame: np.ndarray,
                             layout: dict,
                             cell_h: int,
                             border_px: int,
                             out_path: str) -> None:
    """
    Save the warped strip with a semi-transparent confidence heatmap overlay.
    Green = high confidence, red = low confidence.
    """
    n_frames = layout['n_frames']
    n_bands  = layout['n_bands']

    overlay = warped.copy().astype(np.float32)

    for b in range(n_bands):
        y_top = int(_band_row_top(b, n_bands, cell_h)) + border_px
        y_bot = y_top + cell_h

        α_row = conf_frame[:, b]   # (n_frames,)
        # Blend: high α → green tint, low α → red tint
        r_tint = ((1.0 - α_row) * 220).astype(np.float32)   # red channel boost
        g_tint = (α_row * 160).astype(np.float32)            # green channel boost

        for yi in range(y_top, min(y_bot, overlay.shape[0])):
            for fi in range(n_frames):
                xi = border_px + fi
                if xi >= overlay.shape[1]:
                    break
                # Blend 40% tint onto original pixel
                overlay[yi, xi, 0] = overlay[yi, xi, 0] * 0.6 + r_tint[fi] * 0.4
                overlay[yi, xi, 1] = overlay[yi, xi, 1] * 0.6 + g_tint[fi] * 0.4
                overlay[yi, xi, 2] = overlay[yi, xi, 2] * 0.6

    # Vectorized version (faster)
    overlay2 = warped.copy().astype(np.float32)
    for b in range(n_bands):
        y_top = int(_band_row_top(b, n_bands, cell_h)) + border_px
        y_bot = min(y_top + cell_h, overlay2.shape[0])
        x0    = border_px
        x1    = min(border_px + n_frames, overlay2.shape[1])

        α_row  = conf_frame[:x1 - x0, b]
        r_tint = ((1.0 - α_row) * 220).astype(np.float32)
        g_tint = (α_row * 160).astype(np.float32)

        if y_bot <= y_top or x1 <= x0:
            continue

        strip_slice = overlay2[y_top:y_bot, x0:x1, :].copy()
        strip_slice[:, :, 0] = strip_slice[:, :, 0] * 0.6 + r_tint[np.newaxis, :] * 0.4
        strip_slice[:, :, 1] = strip_slice[:, :, 1] * 0.6 + g_tint[np.newaxis, :] * 0.4
        strip_slice[:, :, 2] = strip_slice[:, :, 2] * 0.6
        overlay2[y_top:y_bot, x0:x1, :] = strip_slice

    PILImage.fromarray(np.clip(overlay2, 0, 255).astype(np.uint8)).save(out_path)
    print(f"  Confidence overlay → {out_path}")


def save_confidence_map(conf_frame: np.ndarray,
                         layout: dict,
                         out_path: str) -> None:
    """
    Save a compact confidence heatmap (x=frames, y=bands, green=high, red=low).
    """
    n_frames = layout['n_frames']
    n_bands  = layout['n_bands']
    cell_h   = 8
    img_w    = n_frames
    img_h    = n_bands * cell_h

    arr = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    for b in range(n_bands):
        row_top = (n_bands - 1 - b) * cell_h
        row_bot = row_top + cell_h
        α_row   = conf_frame[:, b]
        r = ((1.0 - α_row) * 220).astype(np.uint8)
        g = (α_row          * 180).astype(np.uint8)
        arr[row_top:row_bot, :, 0] = r[np.newaxis, :]
        arr[row_top:row_bot, :, 1] = g[np.newaxis, :]

    PILImage.fromarray(arr).save(out_path)
    print(f"  Confidence map     → {out_path}")


# ─── Scale bar measurement ────────────────────────────────────────────────────

def measure_scale_bar(warped: np.ndarray,
                       border_px: int,
                       strip_dpi_nominal: int) -> float | None:
    """
    Estimate actual DPI from the 10mm scale bar in the bottom border.

    The scale bar is drawn left-aligned in the bottom border strip:
        bar_x0 ≈ corner_px + 4   bar_x1 = bar_x0 + scalebar_px
    At nominal DPI: scalebar_px = int(round(10mm × px_per_mm))

    This function looks for the black horizontal bar in the bottom border
    and returns measured DPI, or None if the bar cannot be found.
    """
    try:
        import cv2
    except ImportError:
        return None

    h, w = warped.shape[:2]
    # Scan in the bottom border strip
    y_start = h - border_px
    border_strip = warped[y_start:, :, :]   # (border_px, w, 3)

    gray = np.mean(border_strip, axis=2).astype(np.uint8)
    # Find darkest row in border strip
    row_means = gray.mean(axis=1)
    dark_row  = int(np.argmin(row_means))

    # Find horizontal extent of dark pixels in that row
    row = gray[dark_row, :]
    dark_mask = row < 60
    if not dark_mask.any():
        return None

    # Find contiguous dark runs
    runs = []
    in_run = False
    run_start = 0
    for i, v in enumerate(dark_mask):
        if v and not in_run:
            run_start, in_run = i, True
        elif not v and in_run:
            runs.append((run_start, i))
            in_run = False
    if in_run:
        runs.append((run_start, len(dark_mask)))

    if not runs:
        return None

    # Longest run = scale bar
    bar_start, bar_end = max(runs, key=lambda r: r[1] - r[0])
    bar_px = bar_end - bar_start

    # Nominal scale bar length: 10mm × (strip_dpi / 25.4)
    nominal_mm = 10.0
    measured_dpi = (bar_px * 25.4) / nominal_mm
    return measured_dpi


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description='DSA physical strip camera reader — Tier 1 Python',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split('License')[0].strip())

    p.add_argument('photo',   help='Photo of printed DSA strip (JPG/PNG)')
    p.add_argument('layout',  help='.disc.json layout file (from dsa_disc.py)')

    p.add_argument('--cell-h',   type=int,   default=8,
                   help='Pixel height per band row used when printing (default: 8)')
    p.add_argument('--border',   type=int,   default=None,
                   help='Border width in pixels used by dsa_strip.py --fiducials '
                        '(default: auto — max(20, scalebar_h+8) at --strip-dpi)')
    p.add_argument('--strip-dpi', type=int,  default=300,
                   help='DPI used when printing the strip (default: 300)')
    p.add_argument('--corners',  type=str,   default=None,
                   metavar='"x0,y0 x1,y1 x2,y2 x3,y3"',
                   help='Manual corner positions in photo: TL TR BR BL (overrides auto-detect)')
    p.add_argument('--no-cv',    action='store_true',
                   help='Skip OpenCV (requires --corners to be set)')
    p.add_argument('--debug-detect', action='store_true',
                   help='Print fiducial detection debug info')
    p.add_argument('--save-warped',  type=str,  default=None,
                   help='Save rectified (warped) strip image to this path')
    p.add_argument('--out-overlay',  type=str,  default=None,
                   help='Save confidence overlay on warped strip to this path')
    p.add_argument('--conf-map',     type=str,  default=None,
                   help='Save compact confidence heatmap to this path')
    p.add_argument('--out-json',     type=str,  default=None,
                   help='Save read results (confidence, steepness, direction) to JSON')
    p.add_argument('--decode',       type=str,  default=None,
                   help='.dsa bitstream to decode with camera read confidence')
    p.add_argument('--out',          type=str,  default=None,
                   help='Output WAV path (requires --decode)')
    p.add_argument('--corner-frac',  type=float, default=0.08,
                   help='Expected corner square size as fraction of image shorter side '
                        '(default: 0.08) — tune if auto-detect fails')

    args = p.parse_args()

    print()
    print("  DSA — Digilog Scalable Audio")
    print("  Physical Strip Camera Reader  (Tier 1)")
    print("  ─────────────────────────────────────────────────")
    print()

    # ── Load inputs ──────────────────────────────────────────────────────────

    print(f"  Loading photo  {Path(args.photo).name} ...", end=' ', flush=True)
    pil_photo  = PILImage.open(args.photo).convert('RGB')
    photo_rgb  = np.array(pil_photo, dtype=np.uint8)
    ph, pw     = photo_rgb.shape[:2]
    print(f"done  ({pw}×{ph}px)")

    print(f"  Loading layout {Path(args.layout).name} ...", end=' ', flush=True)
    with open(args.layout) as f:
        layout = json.load(f)
    n_frames   = layout['n_frames']
    n_bands    = layout['n_bands']
    print(f"done  ({n_frames} frames, {n_bands} bands, "
          f"{layout['duration_s']:.1f}s)")

    # ── Compute border ────────────────────────────────────────────────────────

    if args.border is not None:
        border_px = args.border
    else:
        px_per_mm   = args.strip_dpi / 25.4
        scalebar_h  = max(4, int(round(1.0 * px_per_mm)))
        border_px   = max(20, scalebar_h + 8)

    cell_h = args.cell_h
    print(f"  Strip params: cell_h={cell_h}px, border={border_px}px, "
          f"dpi={args.strip_dpi}")

    # Canonical dimensions (must match bordered image from dsa_strip.py)
    content_h  = n_bands * cell_h + 2 * SEP_PX
    canon_w    = n_frames   + 2 * border_px
    canon_h    = content_h  + 2 * border_px
    print(f"  Canonical size: {canon_w}×{canon_h}px")

    # ── Find corners ──────────────────────────────────────────────────────────

    if args.corners:
        src_corners = _parse_manual_corners(args.corners)
        print(f"  Using manual corners: {src_corners.tolist()}")
    elif args.no_cv:
        sys.exit("  --no-cv requires --corners to be set")
    else:
        print("  Detecting fiducial corners...", end=' ', flush=True)
        src_corners = _detect_corners_cv(photo_rgb,
                                          corner_fraction=args.corner_frac,
                                          debug=args.debug_detect)
        if src_corners is None:
            print("FAILED")
            print()
            print("  Could not auto-detect 4 fiducial corners.")
            print("  Try: --corners 'x0,y0 x1,y1 x2,y2 x3,y3'  (TL TR BR BL in photo pixels)")
            print("  Or:  --corner-frac 0.05  (if corners are smaller than expected)")
            print("  Or:  --debug-detect  (to see what was found)")
            sys.exit(1)
        print(f"done  → TL({src_corners[0,0]:.0f},{src_corners[0,1]:.0f})  "
              f"TR({src_corners[1,0]:.0f},{src_corners[1,1]:.0f})  "
              f"BR({src_corners[2,0]:.0f},{src_corners[2,1]:.0f})  "
              f"BL({src_corners[3,0]:.0f},{src_corners[3,1]:.0f})")

    # ── Warp to canonical space ───────────────────────────────────────────────

    # corner_px matches dsa_strip.py exactly: 5mm × (dpi / 25.4)
    # At 300 DPI → corner_px ≈ 59.  This is NOT border_px (which is ~20).
    px_per_mm  = args.strip_dpi / 25.4
    corner_px  = int(round(5.0 * px_per_mm))
    cp = corner_px

    # Map detected corner centres in photo → canonical corner centres.
    # dsa_strip.py draws each corner square flush with the image edge,
    # so the centre of TL square is at (cp/2, cp/2), etc.
    dst_corners = np.float32([
        [cp / 2.0,            cp / 2.0           ],   # TL centre
        [canon_w - cp / 2.0,  cp / 2.0           ],   # TR centre
        [canon_w - cp / 2.0,  canon_h - cp / 2.0 ],   # BR centre
        [cp / 2.0,            canon_h - cp / 2.0 ],   # BL centre
    ])
    print(f"  Corner size: {corner_px}px ({5.0:.0f}mm at {args.strip_dpi} DPI)")

    print(f"  Warping to {canon_w}×{canon_h}...", end=' ', flush=True)
    warped = _warp_strip(photo_rgb, src_corners, dst_corners, canon_w, canon_h)
    print("done")

    if args.save_warped:
        PILImage.fromarray(warped).save(args.save_warped)
        print(f"  Warped strip   → {args.save_warped}")

    # ── Optional: measure DPI from scale bar ──────────────────────────────────

    measured_dpi = measure_scale_bar(warped, border_px, args.strip_dpi)
    if measured_dpi is not None:
        ratio = measured_dpi / args.strip_dpi
        print(f"  Scale bar DPI: {measured_dpi:.0f}  (nominal {args.strip_dpi}, "
              f"ratio {ratio:.3f})")
        if abs(ratio - 1.0) > 0.15:
            print(f"  WARNING: DPI ratio {ratio:.2f} deviates >15% — "
                  f"check --strip-dpi or print magnification")
    else:
        print("  Scale bar: not measured (no OpenCV or bar not found)")

    # ── Read strip ────────────────────────────────────────────────────────────

    print()
    steep_read, dir_read, conf_frame, alpha = read_strip(
        warped, layout, cell_h, border_px)

    # ── Accuracy report ───────────────────────────────────────────────────────

    print()
    print("  ── Read accuracy (vs original disc.json) ─────")
    metrics = compare_layout(layout, steep_read, dir_read)
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

    # Grade the read quality
    mean_a = alpha.mean()
    if mean_a >= 0.90:
        grade = "EXCELLENT — good print/optical quality"
    elif mean_a >= 0.75:
        grade = "GOOD — minor degradation, decoding should work"
    elif mean_a >= 0.55:
        grade = "MARGINAL — moderate degradation, lower-frequency bands only"
    else:
        grade = "POOR — check print quality, focus, or lighting"
    print(f"  Grade: {grade}")

    # ── Outputs ───────────────────────────────────────────────────────────────

    if args.out_overlay:
        print()
        save_confidence_overlay(warped, conf_frame, layout,
                                 cell_h, border_px, args.out_overlay)

    if args.conf_map:
        save_confidence_map(conf_frame, layout, args.conf_map)

    if args.out_json:
        result = {
            'photo':          str(args.photo),
            'layout':         str(args.layout),
            'n_frames':       n_frames,
            'n_bands':        n_bands,
            'cell_h':         cell_h,
            'border_px':      border_px,
            'metrics':        metrics,
            'alpha':          alpha.tolist(),
            'mean_alpha':     float(alpha.mean()),
            'measured_dpi':   measured_dpi,
        }
        with open(args.out_json, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"  Results JSON   → {args.out_json}")

    # ── Optional decode ───────────────────────────────────────────────────────

    if args.decode:
        print()
        print(f"  Decoding {Path(args.decode).name} with camera confidence...")
        try:
            import scipy.io.wavfile as wav
            from dsa_bitstream import DSABitstreamReader
            from dsa_decoder import DSADecoder

            dsa_bytes = Path(args.decode).read_bytes()
            br        = DSABitstreamReader(dsa_bytes)
            decoder   = DSADecoder()
            samples   = decoder.decode(br, alpha=alpha)

            out_path  = args.out or str(Path(args.decode).with_suffix('.camera.wav'))
            s16       = np.clip(samples * 32767, -32768, 32767).astype(np.int16)
            wav.write(out_path, 44100, s16)
            print(f"  Decoded WAV    → {out_path}  ({len(samples)/44100:.1f}s)")
        except ImportError as e:
            print(f"  Decode skipped: {e}")

    print()
    print("  Read the groove.")
    print()


if __name__ == '__main__':
    main()
