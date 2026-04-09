#!/usr/bin/env python3
"""
DSA — Digilog Scalable Audio
dsa_monitor_test.py — DEV-ONLY monitor-display strip reader

⚠  NOT PRODUCTION CODE.  Use dsa_camera.py for printed strips.

This tool exists solely for development convenience when a printer is
unavailable.  It assumes the strip was displayed fullscreen on a monitor
and the phone photo captures the strip filling most of the frame.

Instead of detecting fiducial markers (which wash out on a backlit
monitor), it locates the white-bordered strip boundary by scanning for
the brightest horizontal band near the top and bottom of the photo, then
maps those four outer corners to canonical strip space via a full 8-DOF
PIL PERSPECTIVE warp.

No fiducial detection.  No production path.  For development Tier 1
testing only.

Usage
-----
    python3 dsa_monitor_test.py photo.jpg song.disc.json \\
        --start 0 --end 100 --cell-w 8 \\
        --save-warped warped.png --out-overlay overlay.png
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

# Import the shared pipeline from dsa_camera (warp, read, compare, overlays).
# This tool is a thin wrapper — it only replaces corner detection.
try:
    from dsa_camera import (
        SEP_PX, PALETTE,
        _warp_strip, read_strip, compare_layout,
        save_confidence_overlay, save_confidence_map,
        _band_row_top, _cell_sample_y,
    )
except ImportError as exc:
    sys.exit(f"dsa_camera.py not found in the same directory: {exc}")


# ─── Strip boundary detection ─────────────────────────────────────────────────

def _find_strip_corners(photo_rgb: np.ndarray,
                        bright_thresh: int = 210,
                        min_bright_frac: float = 0.5,
                        debug: bool = False) -> np.ndarray:
    """
    Locate the four OUTER corners of the white-bordered strip in the photo.

    Scans for the topmost and bottommost horizontal rows that contain a
    long contiguous run of bright pixels (>bright_thresh) spanning at least
    min_bright_frac × image_width — these are the top and bottom white borders.
    Within each border row, find the leftmost and rightmost bright pixel.

    A 3% vertical margin at the top and bottom of the frame is skipped to
    avoid triggering on phone bezels or screen glare at the photo edges.

    Returns (4, 2) float32 [TL, TR, BR, BL] outer corners, or raises
    RuntimeError if the strip boundary cannot be located.
    """
    gray = photo_rgb.mean(axis=2).astype(np.float32)
    ph, pw = gray.shape
    # Minimum bright-run length: fraction of image width
    min_bright_run = int(pw * min_bright_frac)
    # Skip phone-edge pixels: 5% on left/right, 3% on top/bottom
    h_margin = pw // 20
    v_margin = max(1, ph // 33)

    def _scan_direction(row_range):
        """Return (y, x_left, x_right) for first row in row_range with a
        long bright run, ignoring the leftmost/rightmost 5% as phone-edge."""
        for y in row_range:
            row = gray[y, h_margin : pw - h_margin]
            bright = (row > bright_thresh)
            # Find longest contiguous run
            run, best = 0, 0
            for v in bright:
                run = run + 1 if v else 0
                best = max(best, run)
            if best >= min_bright_run:
                xs = np.where(bright)[0] + h_margin
                return y, int(xs.min()), int(xs.max())
        return None

    top = _scan_direction(range(v_margin, ph))
    bot = _scan_direction(range(ph - 1 - v_margin, -1, -1))

    if top is None or bot is None:
        raise RuntimeError(
            "Could not locate white border of strip in photo.\n"
            "  • Ensure the photo shows the strip with its white border visible\n"
            "  • Adjust --bright-thresh if the monitor is very dark or very bright\n"
            "  • Lower --min-bright-frac if the strip is small relative to the frame")

    ty, tx_l, tx_r = top
    by, bx_l, bx_r = bot

    if debug:
        print(f"  [monitor] top border  y={ty}  x={tx_l}..{tx_r}")
        print(f"  [monitor] bot border  y={by}  x={bx_l}..{bx_r}")

    return np.float32([
        [tx_l, ty],   # TL
        [tx_r, ty],   # TR
        [bx_r, by],   # BR
        [bx_l, by],   # BL
    ])


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description='DEV-ONLY — DSA monitor-display strip reader',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)

    p.add_argument('photo',  help='Photo of monitor displaying the DSA strip')
    p.add_argument('layout', help='.disc.json layout file')

    p.add_argument('--start',   type=int, default=0,
                   help='First frame in the photographed segment (default: 0)')
    p.add_argument('--end',     type=int, default=None,
                   help='Last frame (exclusive) in the photographed segment')
    p.add_argument('--cell-h',  type=int, default=8,
                   help='Pixel height per band row (default: 8)')
    p.add_argument('--cell-w',  type=int, default=8,
                   help='Pixel width per frame column (default: 8; must match dsa_strip.py)')
    p.add_argument('--border',  type=int, default=None,
                   help='Border width in pixels (default: auto from --strip-dpi)')
    p.add_argument('--strip-dpi', type=int, default=300,
                   help='DPI used when the strip was rendered (default: 300)')
    p.add_argument('--bright-thresh', type=int, default=210,
                   help='Brightness threshold for white-border detection (default: 210)')
    p.add_argument('--min-bright-frac', type=float, default=0.5,
                   help='Minimum bright-run as fraction of image width (default: 0.5)')
    p.add_argument('--debug-detect', action='store_true',
                   help='Print border detection diagnostics')
    p.add_argument('--save-warped',  type=str, default=None,
                   help='Save perspective-corrected strip image')
    p.add_argument('--out-overlay',  type=str, default=None,
                   help='Save confidence overlay on warped strip')
    p.add_argument('--conf-map',     type=str, default=None,
                   help='Save compact confidence heatmap')
    p.add_argument('--out-json',     type=str, default=None,
                   help='Save read results to JSON')

    args = p.parse_args()

    print()
    print("  DSA — Digilog Scalable Audio")
    print("  Monitor Test Reader  ⚠  DEV ONLY — not production")
    print("  ─────────────────────────────────────────────────")
    print()

    # ── Load inputs ──────────────────────────────────────────────────────────

    print(f"  Loading photo  {Path(args.photo).name} ...", end=' ', flush=True)
    photo_rgb = np.array(PILImage.open(args.photo).convert('RGB'), dtype=np.uint8)
    ph, pw = photo_rgb.shape[:2]
    print(f"done  ({pw}×{ph}px)")

    print(f"  Loading layout {Path(args.layout).name} ...", end=' ', flush=True)
    with open(args.layout) as f:
        layout = json.load(f)
    n_bands     = layout['n_bands']
    frame_start = args.start
    frame_end   = args.end if args.end is not None else layout['n_frames']
    frame_end   = min(frame_end, layout['n_frames'])
    n_frames    = frame_end - frame_start
    print(f"done  ({layout['n_frames']} frames total, reading "
          f"{n_frames} [{frame_start}:{frame_end}], {n_bands} bands)")

    layout_slice = dict(layout)
    layout_slice['n_frames'] = n_frames
    layout_slice['frames'] = [
        dict(fd, frame_idx=fd['frame_idx'] - frame_start)
        for fd in layout['frames']
        if frame_start <= fd['frame_idx'] < frame_end
    ]

    # ── Canonical dimensions ──────────────────────────────────────────────────

    if args.border is not None:
        border_px = args.border
    else:
        px_per_mm  = args.strip_dpi / 25.4
        scalebar_h = max(4, int(round(1.0 * px_per_mm)))
        border_px  = max(20, scalebar_h + 8)

    cell_h = args.cell_h
    cell_w = args.cell_w

    content_h = n_bands * cell_h + 2 * SEP_PX
    canon_w   = n_frames * cell_w + 2 * border_px
    canon_h   = content_h         + 2 * border_px
    print(f"  Strip params: cell_h={cell_h}px cell_w={cell_w}px border={border_px}px")
    print(f"  Canonical size: {canon_w}×{canon_h}px")

    # warp_scale: keep cell_w × scale ≥ 32px so camera blur doesn't collapse gradient
    photo_ratio = pw / canon_w
    warp_scale  = max(1, int(math.ceil(32 / max(cell_w, 1))))
    warp_scale  = min(warp_scale, max(1, int(photo_ratio)))
    warp_w, warp_h = canon_w * warp_scale, canon_h * warp_scale
    if warp_scale > 1:
        print(f"  Warp scale: {warp_scale}× → {warp_w}×{warp_h}px")

    # ── Locate strip corners ──────────────────────────────────────────────────

    print("  Locating strip boundary...", end=' ', flush=True)
    try:
        src_corners = _find_strip_corners(
            photo_rgb,
            bright_thresh=args.bright_thresh,
            min_bright_frac=args.min_bright_frac,
            debug=args.debug_detect)
    except RuntimeError as exc:
        print("FAILED")
        print(f"\n  {exc}")
        sys.exit(1)

    tl, tr, br, bl = src_corners
    print(f"done  → TL({tl[0]:.0f},{tl[1]:.0f})  TR({tr[0]:.0f},{tr[1]:.0f})  "
          f"BR({br[0]:.0f},{br[1]:.0f})  BL({bl[0]:.0f},{bl[1]:.0f})")

    # Map outer strip corners → outer canonical corners (0,0)…(warp_w,warp_h).
    # This is correct for the outer-corner reference: no cp/2 offset needed
    # because we're not using fiducial centres — we're using image edges.
    dst_corners = np.float32([
        [0,      0      ],   # TL
        [warp_w, 0      ],   # TR
        [warp_w, warp_h ],   # BR
        [0,      warp_h ],   # BL
    ])

    # ── Warp ─────────────────────────────────────────────────────────────────

    print(f"  Warping to {warp_w}×{warp_h}...", end=' ', flush=True)
    warped = _warp_strip(photo_rgb, src_corners, dst_corners, warp_w, warp_h)
    print("done")

    if args.save_warped:
        PILImage.fromarray(warped).save(args.save_warped)
        print(f"  Warped strip   → {args.save_warped}")

    # ── Read strip ────────────────────────────────────────────────────────────

    # The warped image's border region is now warp_scale × border_px pixels
    # wide.  Adjust read_strip accordingly.
    print()
    steep_read, dir_read, conf_frame, alpha = read_strip(
        warped, layout_slice,
        cell_h    = cell_h * warp_scale,
        border_px = border_px * warp_scale,
        cell_w    = cell_w * warp_scale)

    # ── Report ────────────────────────────────────────────────────────────────

    results = compare_layout(layout_slice, steep_read, dir_read)

    print()
    print("  ── Read accuracy (vs original disc.json) ─────")
    print(f"  Steepness MAE:        {results['steepness_mae']:.6f}")
    print(f"  Steepness max error:  {results['steepness_max_err']:.6f}")
    n_correct = int(results['direction_accuracy'] * results['n_cells'])
    print(f"  Direction accuracy:   {results['direction_accuracy']*100:.2f}%"
          f"  ({n_correct}/{results['n_cells']} cells correct)")

    print()
    print("  ── Per-band mean confidence (α) ──────────────")
    from dsa_camera import L0_END, L1_END
    print(f"  L0 (bass):   {' '.join(f'{a:.2f}' for a in alpha[:L0_END])}")
    print(f"  L1 (mid):    {' '.join(f'{a:.2f}' for a in alpha[L0_END:L1_END])}")
    # L2 in two rows of 12
    l2 = alpha[L1_END:]
    half = len(l2) // 2
    print(f"  L2 (high):   {' '.join(f'{a:.2f}' for a in l2[:half])}")
    if half < len(l2):
        print(f"               {' '.join(f'{a:.2f}' for a in l2[half:])}")
    mean_a = float(alpha.mean())
    print()
    print(f"  Overall mean α: {mean_a:.4f}")

    if   mean_a >= 0.95: grade = "EXCELLENT — very low degradation"
    elif mean_a >= 0.90: grade = "GOOD — minor degradation, decoding should work"
    elif mean_a >= 0.80: grade = "MARGINAL — moderate degradation, lower-frequency bands only"
    else:                grade = "POOR — severe degradation"
    print(f"  Grade: {grade}")
    print()
    print("  Read the groove.")
    print()

    # ── Optional outputs ──────────────────────────────────────────────────────

    if args.out_overlay:
        save_confidence_overlay(warped, conf_frame, layout_slice,
                                cell_h * warp_scale, border_px * warp_scale,
                                cell_w * warp_scale, args.out_overlay)
        print(f"  Confidence overlay → {args.out_overlay}")

    if args.conf_map:
        save_confidence_map(conf_frame, args.conf_map)
        print(f"  Confidence map     → {args.conf_map}")

    if args.out_json:
        out = {
            'steepness_mae':      results['steepness_mae'],
            'direction_accuracy': results['direction_accuracy'],
            'mean_alpha':         mean_a,
            'alpha_per_band':     alpha.tolist(),
        }
        with open(args.out_json, 'w') as f:
            json.dump(out, f, indent=2)
        print(f"  Results JSON       → {args.out_json}")


if __name__ == '__main__':
    main()
