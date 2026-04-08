#!/usr/bin/env python3
"""
DSA — Digilog Scalable Audio
dsa_animate.py — Spinning Disc Animation

Renders a disc.json as a rotating disc video (MP4 or GIF).
A fixed red scan line represents the camera read position.
As the disc rotates, different frame arcs pass under the scan line —
visually showing how audio data is read from the physical medium.

Usage:
    python3 dsa_animate.py song.disc.json
    python3 dsa_animate.py song.disc.json --rpm 33 --fps 30 --duration 5
    python3 dsa_animate.py song.disc.json --gif --duration 3
    python3 dsa_animate.py song.disc.json --size 600 --out disc.mp4

Requires: Pillow, ffmpeg (for MP4), numpy

License: GPL v3 — github.com/pisdronio/dsa
"""

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

try:
    from PIL import Image, ImageDraw
except ImportError:
    sys.exit("Pillow required — pip install Pillow")

# ─── Disc geometry (mm) ───────────────────────────────────────────────────────

DISC_DIAMETER_MM   = 290.0
OUTER_AUDIO_MM     = 141.0
INNER_AUDIO_MM     = 62.0
CLOCK_WIDTH_MM     = 2.5
CLOCK_SEGMENTS     = 300
REF_MARKERS        = 8
LABEL_RADIUS_MM    = 55.0
SPINDLE_RADIUS_MM  = 7.0
NUM_BANDS          = 48

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


def _build_disc_image(layout: dict, size_px: int) -> Image.Image:
    """
    Render the full disc as a PIL Image at size_px × size_px.
    This is done once; frames are created by rotating this image.
    """
    n_frames  = layout['n_frames']
    n_bands   = layout['n_bands']
    frames    = layout['frames']
    bp_map    = {b['band']: (b['color_a'], b['color_b']) for b in layout['band_pairs']}

    mm_to_px       = size_px / DISC_DIAMETER_MM
    ctr            = size_px // 2
    outer_audio_px = OUTER_AUDIO_MM  * mm_to_px
    inner_audio_px = INNER_AUDIO_MM  * mm_to_px
    ring_width_px  = (outer_audio_px - inner_audio_px) / n_bands
    clock_inner_px = outer_audio_px
    clock_outer_px = (OUTER_AUDIO_MM + CLOCK_WIDTH_MM) * mm_to_px
    label_px       = LABEL_RADIUS_MM * mm_to_px
    spindle_px     = SPINDLE_RADIUS_MM * mm_to_px
    disc_r_px      = (DISC_DIAMETER_MM / 2) * mm_to_px

    # Build steepness / direction tables
    steep = np.zeros((n_frames, n_bands), dtype=np.float32)
    dir_  = np.zeros((n_frames, n_bands), dtype=np.int8)
    for fd in frames:
        fi = fd['frame_idx']
        if fd['silence']:
            continue
        for bd in fd['bands']:
            b            = bd['band']
            steep[fi, b] = bd['steepness']
            dir_[fi, b]  = bd['direction']

    ca = np.zeros((n_bands, 3), dtype=np.float32)
    cb = np.zeros((n_bands, 3), dtype=np.float32)
    for b in range(n_bands):
        ca[b] = PALETTE[bp_map[b][0]]
        cb[b] = PALETTE[bp_map[b][1]]

    # Pixel coordinate arrays
    yy, xx = np.mgrid[0:size_px, 0:size_px].astype(np.float32)
    ddx    = xx - ctr
    ddy    = yy - ctr
    r      = np.sqrt(ddx * ddx + ddy * ddy)
    theta  = np.arctan2(ddx, -ddy) % (2.0 * math.pi)

    img = np.full((size_px, size_px, 3), 255, dtype=np.uint8)

    # Audio bands
    in_audio    = (r >= inner_audio_px) & (r < outer_audio_px)
    b_idx       = ((r - inner_audio_px) / ring_width_px).astype(np.int32)
    np.clip(b_idx, 0, n_bands - 1, out=b_idx)
    frame_angle = 2.0 * math.pi / n_frames
    f_idx       = (theta / frame_angle).astype(np.int32)
    np.clip(f_idx, 0, n_frames - 1, out=f_idx)
    arc_pos     = (theta % frame_angle) / frame_angle

    ay, ax = np.where(in_audio)
    fi     = f_idx[ay, ax]
    bi     = b_idx[ay, ax]
    pos    = arc_pos[ay, ax]
    s      = steep[fi, bi]
    d      = dir_[fi, bi].astype(np.float32)
    t      = np.where(d == 0, 0.0,
             np.where(d > 0,  pos * s,
                               s * (1.0 - pos))).astype(np.float32)
    np.clip(t, 0.0, 1.0, out=t)
    t3     = t[:, np.newaxis]
    colors = (ca[bi] + (cb[bi] - ca[bi]) * t3).astype(np.uint8)
    img[ay, ax] = colors

    # Clock track
    clock_y, clock_x = np.where((r >= clock_inner_px) & (r < clock_outer_px))
    if len(clock_y):
        segs = (theta[clock_y, clock_x] / (2.0 * math.pi) * CLOCK_SEGMENTS
                ).astype(np.int32) % CLOCK_SEGMENTS
        even = segs % 2 == 0
        img[clock_y[even],  clock_x[even]]  = (0,   0,   0)
        img[clock_y[~even], clock_x[~even]] = (255, 255, 255)

    img[r < label_px]   = (210, 210, 210)
    img[r < spindle_px] = (255, 255, 255)
    img[r >= disc_r_px] = (255, 255, 255)

    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)

    # Reference markers
    ref_r  = (clock_inner_px + clock_outer_px) / 2.0
    dot_r  = max(3, int(mm_to_px * 1.8))
    for i in range(REF_MARKERS):
        angle = i * 2.0 * math.pi / REF_MARKERS
        px_x  = ctr + ref_r * math.sin(angle)
        px_y  = ctr - ref_r * math.cos(angle)
        draw.ellipse([px_x - dot_r, px_y - dot_r, px_x + dot_r, px_y + dot_r],
                     fill=(230, 40, 40))

    return pil


def _make_frame(disc_img: Image.Image, angle_deg: float,
                size_px: int, scan_angle_deg: float = 90.0) -> Image.Image:
    """
    Rotate the disc by angle_deg and draw the scan line at scan_angle_deg.
    scan_angle_deg: 0=top, 90=right, 180=bottom (clockwise from 12 o'clock).
    """
    # Rotate disc (expand=False keeps the canvas size)
    rotated = disc_img.rotate(-angle_deg, resample=Image.BILINEAR, expand=False)

    draw = ImageDraw.Draw(rotated)
    ctr  = size_px // 2
    mm_to_px = size_px / DISC_DIAMETER_MM

    # Draw scan line: radial from inner audio edge to outer clock edge
    inner_r = INNER_AUDIO_MM  * mm_to_px
    outer_r = (OUTER_AUDIO_MM + CLOCK_WIDTH_MM) * mm_to_px
    rad     = math.radians(scan_angle_deg - 90)   # convert to math angle (0=right)
    x0 = ctr + inner_r * math.cos(rad)
    y0 = ctr + inner_r * math.sin(rad)
    x1 = ctr + outer_r * math.cos(rad)
    y1 = ctr + outer_r * math.sin(rad)

    # Glowing scan line: wide white + narrow bright red
    lw = max(2, int(mm_to_px * 0.6))
    draw.line([(x0, y0), (x1, y1)], fill=(255, 255, 255), width=lw + 2)
    draw.line([(x0, y0), (x1, y1)], fill=(255, 30,  30),  width=lw)

    return rotated


def animate(layout_path: str,
            rpm: float     = 33.0,
            fps: int       = 30,
            duration_s: float = 4.0,
            size_px: int   = 700,
            out_path: str  = None,
            make_gif: bool = False) -> str:
    """
    Render a spinning disc animation.

    Parameters
    ----------
    layout_path : .disc.json file
    rpm         : rotation speed (default 33 rpm)
    fps         : animation frame rate (default 30)
    duration_s  : animation duration in seconds (default 4)
    size_px     : disc image size in pixels (default 700)
    out_path    : output path (.mp4 or .gif)
    make_gif    : force GIF output even if ffmpeg available
    """

    print(f"  Loading {Path(layout_path).name} ...", end=' ', flush=True)
    with open(layout_path) as f:
        layout = json.load(f)
    n_frames = layout['n_frames']
    print(f"done  ({n_frames} frames, {layout['duration_s']:.1f}s encoded)")

    # ── Build base disc image ─────────────────────────────────────────────────
    print(f"  Rendering disc at {size_px}px ...", end=' ', flush=True)
    disc_img = _build_disc_image(layout, size_px)
    print("done")

    # ── Generate animation frames ─────────────────────────────────────────────
    deg_per_anim_frame = 360.0 * (rpm / 60.0) / fps
    n_anim_frames      = int(duration_s * fps)

    print(f"  Generating {n_anim_frames} frames "
          f"({rpm}rpm, {fps}fps, {duration_s:.1f}s, {deg_per_anim_frame:.2f}°/frame) ...")

    frames_pil = []
    for i in range(n_anim_frames):
        angle = i * deg_per_anim_frame
        frame = _make_frame(disc_img, angle, size_px)
        frames_pil.append(frame)
        if (i + 1) % fps == 0:
            print(f"    {i+1}/{n_anim_frames}", end='\r', flush=True)

    print(f"    {n_anim_frames}/{n_anim_frames}  done          ")

    # ── Output ────────────────────────────────────────────────────────────────
    if out_path is None:
        ext      = '.gif' if make_gif else '.mp4'
        out_path = str(Path(layout_path).with_suffix(ext))

    if make_gif or out_path.endswith('.gif'):
        print(f"  Saving GIF → {out_path} ...", end=' ', flush=True)
        frames_pil[0].save(
            out_path,
            save_all=True,
            append_images=frames_pil[1:],
            loop=0,
            duration=int(1000 / fps),
            optimize=False,
        )
        size_mb = Path(out_path).stat().st_size / 1024 / 1024
        print(f"done  ({size_mb:.1f} MB)")

    else:
        # MP4 via ffmpeg (much smaller, better quality)
        _check_ffmpeg()
        with tempfile.TemporaryDirectory() as tmpdir:
            print(f"  Writing frame PNGs ...", end=' ', flush=True)
            for i, f in enumerate(frames_pil):
                f.save(os.path.join(tmpdir, f"frame_{i:05d}.png"))
            print("done")

            print(f"  Encoding MP4 → {out_path} ...", end=' ', flush=True)
            cmd = [
                'ffmpeg', '-y',
                '-framerate', str(fps),
                '-i', os.path.join(tmpdir, 'frame_%05d.png'),
                '-c:v', 'libx264',
                '-crf', '18',
                '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart',
                out_path,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"\n  ffmpeg error:\n{result.stderr}")
                sys.exit(1)
            size_mb = Path(out_path).stat().st_size / 1024 / 1024
            print(f"done  ({size_mb:.1f} MB)")

    return out_path


def _check_ffmpeg():
    result = subprocess.run(['ffmpeg', '-version'], capture_output=True)
    if result.returncode != 0:
        sys.exit("ffmpeg not found — install with: brew install ffmpeg\n"
                 "Or use --gif for GIF output (no ffmpeg required)")


def main():
    p = argparse.ArgumentParser(
        description='Render a spinning DSA disc animation (MP4 or GIF)')
    p.add_argument('layout',
                   help='Path to .disc.json')
    p.add_argument('--rpm',      type=float, default=33.0,
                   help='Rotation speed in RPM (default: 33)')
    p.add_argument('--fps',      type=int,   default=30,
                   help='Animation frame rate (default: 30)')
    p.add_argument('--duration', type=float, default=4.0,
                   help='Animation duration in seconds (default: 4)')
    p.add_argument('--size',     type=int,   default=700,
                   help='Disc size in pixels (default: 700)')
    p.add_argument('--gif',      action='store_true',
                   help='Output GIF instead of MP4 (no ffmpeg required)')
    p.add_argument('--out',      type=str,   default=None,
                   help='Output file path (.mp4 or .gif)')
    args = p.parse_args()

    print()
    print("  DSA — Digilog Scalable Audio")
    print("  Spinning Disc Animator")
    print("  ─────────────────────────────────────────────────")
    print(f"  Layout:   {args.layout}")
    print(f"  Speed:    {args.rpm} RPM")
    print(f"  Duration: {args.duration}s  at {args.fps}fps")
    print(f"  Size:     {args.size}px")
    print()

    out = animate(
        args.layout,
        rpm=args.rpm,
        fps=args.fps,
        duration_s=args.duration,
        size_px=args.size,
        out_path=args.out,
        make_gif=args.gif,
    )

    print()
    print(f"  Output: {out}")
    print("  Scan the groove.")
    print()


if __name__ == '__main__':
    main()
