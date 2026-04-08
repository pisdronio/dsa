#!/usr/bin/env python3
"""
DSA — Digilog Scalable Audio
dsa_animate.py — Strip Tape Animator

Renders the DSA disc as what the camera actually sees: a strip of color bands
scrolling past a fixed read window — the tape-head model (Section 12.8).

The disc is a storage medium. The strip is the instrument.

The camera sees a narrow fixed horizontal window. The disc rotates, advancing
the spiral groove through the window one frame at a time. The output is a
vertical waterfall of color bands: left = L0 (bass), right = L2 (high),
time scrolling upward as the disc rotates forward.

Display layout:
    ┌─────────────────────────────────────────────────┐
    │  L0 (bass)  ──────────────────  L2 (high)      │
    │  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  │  past frames
    ├──────────────────────────────────────────────── ┤  camera window (NOW)
    │  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  │  future frames
    └─────────────────────────────────────────────────┘

    Forward playback:  strip scrolls upward  (past rises, future arrives from below)
    Reverse playback:  strip scrolls downward
    Speed change:      scroll rate changes   (pitch and tempo shift together)

Also available: --mode disc — original spinning disc visualization with scan line.
This is the "storage medium" view, not the "instrument" view.

Usage:
    python3 dsa_animate.py song.disc.json                        # tape strip (default)
    python3 dsa_animate.py song.disc.json --out strip.mp4
    python3 dsa_animate.py song.disc.json --mode disc            # spinning disc
    python3 dsa_animate.py song.disc.json --mode disc --out disc.mp4
    python3 dsa_animate.py song.disc.json --gif                  # GIF output

License: GPL v3 — github.com/pisdronio/dsa
"""

import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

try:
    from PIL import Image, ImageDraw
except ImportError:
    sys.exit("Pillow required — pip install Pillow")

from dsa_color import rgb_to_lab, lab_to_rgb

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


# ─── Color table builder ──────────────────────────────────────────────────────

def _build_color_table(layout: dict) -> np.ndarray:
    """
    Build a (n_frames, n_bands, 3) uint8 color table.
    Each cell gets the representative gradient midpoint color (t=0.5 × steepness).
    """
    n_frames = layout['n_frames']
    n_bands  = layout['n_bands']
    bp_map   = {b['band']: (b['color_a'], b['color_b']) for b in layout['band_pairs']}

    ca_rgb = np.array([PALETTE[bp_map[b][0]] for b in range(n_bands)], dtype=np.float64)
    cb_rgb = np.array([PALETTE[bp_map[b][1]] for b in range(n_bands)], dtype=np.float64)
    ca_lab = rgb_to_lab(ca_rgb)
    cb_lab = rgb_to_lab(cb_rgb)

    # Default: solid color_a
    table = np.zeros((n_frames, n_bands, 3), dtype=np.uint8)
    for b in range(n_bands):
        table[:, b] = PALETTE[bp_map[b][0]].astype(np.uint8)

    for fd in layout['frames']:
        fi = fd['frame_idx']
        if fd['silence']:
            continue
        for bd in fd['bands']:
            b = bd['band']
            s = bd['steepness']
            d = bd['direction']
            if d == 0 or s < 0.001:
                continue
            t     = float(np.clip(0.5 * s if d > 0 else s - 0.5 * s, 0.0, 1.0))
            mixed = ca_lab[b] + (cb_lab[b] - ca_lab[b]) * t
            table[fi, b] = lab_to_rgb(mixed[np.newaxis])[0]

    return table


# ─── Tape strip animation ─────────────────────────────────────────────────────

def render_tape(layout_path: str,
                out_path: str      = None,
                video_fps: int     = 30,
                rpm: float         = 33.0,
                cell_w: int        = 12,
                cell_h: int        = 3,
                window_frames: int = 160,
                gif: bool          = False) -> str:
    """
    Render the tape-head strip animation.

    The strip scrolls vertically. Left = L0 bass, right = L2 high.
    The fixed yellow line is the camera window (current audio frame).
    Past frames are above, slightly dimmed. Future frames below, full brightness.
    """
    print(f"  Loading {Path(layout_path).name} ...", end=' ', flush=True)
    with open(layout_path) as f:
        layout = json.load(f)
    n_frames   = layout['n_frames']
    n_bands    = layout['n_bands']
    duration_s = layout['duration_s']
    print(f"done  ({n_frames} frames, {duration_s:.1f}s)")

    print("  Building color table...", end=' ', flush=True)
    color_table = _build_color_table(layout)
    print("done")

    audio_fps       = n_frames / duration_s
    audio_per_video = audio_fps / video_fps
    n_video_frames  = int(n_frames / audio_per_video)

    img_w   = n_bands * cell_w
    img_h   = window_frames * cell_h
    now_row = img_h // 3       # camera window at 1/3 from top

    WIN_LINE  = (255, 255, 80)
    SEP_COLOR = (60,  60,  60)

    print(f"  Window: {img_w}×{img_h}px  "
          f"({n_bands} bands × {cell_w}px, {window_frames} rows × {cell_h}px)")
    print(f"  {n_video_frames} video frames  ({n_video_frames/video_fps:.1f}s at {video_fps} fps)")

    if out_path is None:
        ext      = '.gif' if gif else '.mp4'
        base     = str(Path(layout_path).with_suffix('').with_suffix(''))
        out_path = base + '.tape' + ext

    if gif:
        _tape_gif(color_table, n_frames, n_bands, n_video_frames, audio_per_video,
                  img_w, img_h, now_row, cell_w, cell_h, WIN_LINE, SEP_COLOR,
                  out_path, video_fps)
    else:
        _tape_mp4(color_table, n_frames, n_bands, n_video_frames, audio_per_video,
                  img_w, img_h, now_row, cell_w, cell_h, WIN_LINE, SEP_COLOR,
                  out_path, video_fps)

    return out_path


def _tape_frame(color_table, n_frames, n_bands,
                current_audio, img_w, img_h, now_row,
                cell_w, cell_h, win_line, sep_color):
    """Render one video frame of the tape animation."""
    img = np.zeros((img_h, img_w, 3), dtype=np.uint8)

    # Precompute audio frame indices for all display rows (vectorized)
    dy_arr     = np.arange(img_h)
    row_offset = (dy_arr - now_row) / cell_h           # neg = past, pos = future
    fi_arr     = (current_audio + row_offset).astype(np.int32)
    valid      = (fi_arr >= 0) & (fi_arr < n_frames)

    for dy in range(img_h):
        if not valid[dy]:
            continue
        fi = fi_arr[dy]
        # Draw all 48 bands for this row
        row_colors = color_table[fi]   # (n_bands, 3)
        for b in range(n_bands):
            img[dy, b * cell_w : (b + 1) * cell_w] = row_colors[b]
        # Dim past frames slightly
        if dy < now_row:
            img[dy] = np.clip(img[dy].astype(np.int32) - 30, 0, 255).astype(np.uint8)

    # Camera window highlight
    win_h = max(2, cell_h // 2)
    img[now_row : now_row + win_h] = win_line

    # Layer separators (vertical lines between L0/L1 and L1/L2)
    for sep_b in (8, 24):
        x = sep_b * cell_w
        img[:, x - 1 : x + 1] = sep_color

    return img


def _tape_mp4(color_table, n_frames, n_bands, n_video_frames, audio_per_video,
              img_w, img_h, now_row, cell_w, cell_h, win_line, sep_color,
              out_path, video_fps):
    if not _has_ffmpeg():
        print("  ffmpeg not found — use --gif")
        return
    print(f"  Encoding MP4 → {out_path} ...", flush=True)
    cmd = ['ffmpeg', '-y',
           '-f', 'rawvideo', '-vcodec', 'rawvideo',
           '-s', f'{img_w}x{img_h}', '-pix_fmt', 'rgb24',
           '-r', str(video_fps), '-i', 'pipe:0',
           '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', out_path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        for vf in range(n_video_frames):
            f = _tape_frame(color_table, n_frames, n_bands,
                            vf * audio_per_video,
                            img_w, img_h, now_row, cell_w, cell_h,
                            win_line, sep_color)
            proc.stdin.write(f.tobytes())
            if vf % 150 == 0:
                pct = 100 * vf / n_video_frames
                print(f"    {vf}/{n_video_frames} ({pct:.0f}%)", end='\r', flush=True)
    finally:
        proc.stdin.close()
        proc.wait()
    size_mb = Path(out_path).stat().st_size / 1024 / 1024
    print(f"  Done → {out_path}  ({size_mb:.1f} MB)              ")


def _tape_gif(color_table, n_frames, n_bands, n_video_frames, audio_per_video,
              img_w, img_h, now_row, cell_w, cell_h, win_line, sep_color,
              out_path, video_fps, max_frames=120):
    step = max(1, n_video_frames // max_frames)
    print(f"  Building GIF ({min(max_frames, n_video_frames)} frames) ...",
          end=' ', flush=True)
    frames = []
    for vf in range(0, n_video_frames, step):
        f = _tape_frame(color_table, n_frames, n_bands,
                        vf * audio_per_video,
                        img_w, img_h, now_row, cell_w, cell_h,
                        win_line, sep_color)
        frames.append(Image.fromarray(f))
        if len(frames) >= max_frames:
            break
    frames[0].save(out_path, save_all=True, append_images=frames[1:],
                   loop=0, duration=int(1000 / video_fps * step))
    size_kb = Path(out_path).stat().st_size / 1024
    print(f"done  ({len(frames)} frames, {size_kb:.0f} KB) → {out_path}")


# ─── Disc spinning animation ──────────────────────────────────────────────────

def render_disc_anim(layout_path: str,
                     out_path: str  = None,
                     video_fps: int = 30,
                     rpm: float     = 33.0,
                     dpi_sim: int   = 150,
                     gif: bool      = False) -> str:
    """
    Spinning disc animation — the storage medium view.

    Shows the disc rotating with a fixed red scan line. This visualizes the
    disc as an object, not as the signal the camera reads. Use --mode tape
    for the instrument/signal view.
    """
    from dsa_render import render_disc

    print(f"  Loading {Path(layout_path).name} ...", end=' ', flush=True)
    with open(layout_path) as f:
        layout = json.load(f)
    n_frames   = layout['n_frames']
    duration_s = layout['duration_s']
    print(f"done  ({n_frames} frames, {duration_s:.1f}s)")

    print("  Rendering base disc ...", flush=True)
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        tmp_path = tmp.name
    render_disc(layout_path, dpi=dpi_sim, out_path=tmp_path)

    base_img = np.array(Image.open(tmp_path).convert('RGB'), dtype=np.uint8)
    disc_px  = base_img.shape[0]
    ctr      = disc_px // 2

    audio_fps          = n_frames / duration_s
    audio_per_video    = audio_fps / video_fps
    n_video_frames     = int(n_frames / audio_per_video)
    degrees_per_vframe = (rpm / 60.0) * 360.0 / video_fps

    if out_path is None:
        ext      = '.gif' if gif else '.mp4'
        base     = str(Path(layout_path).with_suffix('').with_suffix(''))
        out_path = base + '.disc' + ext

    def _disc_frame(angle_deg):
        pil  = Image.fromarray(base_img).rotate(angle_deg, resample=Image.BILINEAR)
        draw = ImageDraw.Draw(pil)
        draw.line([(ctr, ctr), (disc_px - 1, ctr)], fill=(220, 40, 40), width=3)
        return np.array(pil, dtype=np.uint8)

    if gif:
        step   = max(1, n_video_frames // 60)
        frames = []
        for vf in range(0, n_video_frames, step):
            frames.append(Image.fromarray(_disc_frame(vf * degrees_per_vframe)))
            if len(frames) >= 60:
                break
        frames[0].save(out_path, save_all=True, append_images=frames[1:],
                       loop=0, duration=int(1000 / video_fps * step))
        size_kb = Path(out_path).stat().st_size / 1024
        print(f"  Done → {out_path}  ({size_kb:.0f} KB)")
    else:
        if not _has_ffmpeg():
            print("  ffmpeg not found — use --gif")
            return out_path
        cmd = ['ffmpeg', '-y',
               '-f', 'rawvideo', '-vcodec', 'rawvideo',
               '-s', f'{disc_px}x{disc_px}', '-pix_fmt', 'rgb24',
               '-r', str(video_fps), '-i', 'pipe:0',
               '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', out_path]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        try:
            for vf in range(n_video_frames):
                proc.stdin.write(_disc_frame(vf * degrees_per_vframe).tobytes())
        finally:
            proc.stdin.close()
            proc.wait()
        size_mb = Path(out_path).stat().st_size / 1024 / 1024
        print(f"  Done → {out_path}  ({size_mb:.1f} MB)")

    Path(tmp_path).unlink(missing_ok=True)
    return out_path


# ─── Utilities ────────────────────────────────────────────────────────────────

def _has_ffmpeg() -> bool:
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description='DSA tape-head strip animator (default) or spinning disc animator')
    p.add_argument('layout',
                   help='Path to .disc.json')
    p.add_argument('--mode',    choices=['tape', 'disc'], default='tape',
                   help='tape = scrolling strip / instrument view (default); '
                        'disc = spinning disc / storage view')
    p.add_argument('--out',     type=str,   default=None,
                   help='Output path (.mp4 default, or .gif with --gif)')
    p.add_argument('--fps',     type=int,   default=30,
                   help='Video frame rate (default: 30)')
    p.add_argument('--rpm',     type=float, default=33.0,
                   help='Disc rotation speed in RPM (default: 33)')
    p.add_argument('--gif',     action='store_true',
                   help='Write GIF instead of MP4')
    p.add_argument('--cell-w',  type=int,   default=12,
                   help='[tape] Pixels per frequency band column (default: 12)')
    p.add_argument('--cell-h',  type=int,   default=3,
                   help='[tape] Pixels per audio frame row (default: 3)')
    p.add_argument('--window',  type=int,   default=160,
                   help='[tape] Audio frames visible in window (default: 160)')
    p.add_argument('--dpi',     type=int,   default=150,
                   help='[disc] Render DPI for base disc image (default: 150)')
    args = p.parse_args()

    print()
    print("  DSA — Digilog Scalable Audio")
    if args.mode == 'tape':
        print("  Tape-Head Strip Animator  (instrument view)")
    else:
        print("  Spinning Disc Animator  (storage medium view)")
    print("  ─────────────────────────────────────────────────")
    print(f"  Layout:  {args.layout}")
    print(f"  Mode:    {args.mode}  |  {args.rpm} RPM  |  {args.fps} fps")
    print()

    if args.mode == 'tape':
        out = render_tape(args.layout,
                          out_path=args.out,
                          video_fps=args.fps,
                          rpm=args.rpm,
                          cell_w=args.cell_w,
                          cell_h=args.cell_h,
                          window_frames=args.window,
                          gif=args.gif)
        print()
        print("  Yellow line = camera window (current frame = NOW)")
        print("  Strip scrolls up = forward playback")
        print("  Left = L0 bass  |  Right = L2 high")
        print("  Gray lines = layer boundaries (L0/L1, L1/L2)")
    else:
        out = render_disc_anim(args.layout,
                               out_path=args.out,
                               video_fps=args.fps,
                               rpm=args.rpm,
                               dpi_sim=args.dpi,
                               gif=args.gif)
        print()
        print("  Red line = camera window position (fixed)")
        print("  Disc rotates = tape advancing through read head")
        print("  Use --mode tape for the instrument/signal view")
    print()


if __name__ == '__main__':
    main()
