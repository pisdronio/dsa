#!/usr/bin/env python3
"""
DSA — Digilog Scalable Audio
dsa_degrade.py — Analog Degradation Simulator

Applies controlled degradations to a rendered disc PNG, then runs
dsa_reader.py's read logic to measure how α confidence and direction
accuracy degrade under each condition.

This validates the confidence model (Section 12.6) before any physical
print/read test. If the model holds up here, the same degradation
patterns seen on a physical disc will produce predictable α values.

Degradation modes (composable):
    blur    — Gaussian blur (sigma in pixels), simulates lens defocus
                or camera shake. Affects high-frequency rings most.
    noise   — Additive Gaussian noise (sigma in 0–255 range), simulates
                sensor noise, JPEG artifacts, or print grain.
    cast    — Multiplicative color cast [R, G, B] scale factors,
                simulates auto-white-balance error or colored lighting.
    fade    — Reduce contrast toward mid-gray (factor 0=original, 1=flat),
                simulates ink fade, worn print, or overexposure.
    jpeg    — JPEG re-compression at given quality (1–95), simulates
                phone camera or scanned disc.

Output:
    - Per-degradation accuracy table (direction %, steepness MAE, mean α)
    - Confidence map PNG for each condition (optional)
    - Summary: at what sigma/quality does each layer fail below threshold?

Failure thresholds (from Section 12.2 / operational requirements):
    L0 direction accuracy < 85%  → bass readability at risk
    L1 direction accuracy < 80%  → mid readability at risk
    L2 direction accuracy < 70%  → high freq graceful degradation expected

Usage:
    python3 dsa_degrade.py disc.png disc.disc.json
    python3 dsa_degrade.py disc.png disc.disc.json --blur 0,1,2,3,5,8
    python3 dsa_degrade.py disc.png disc.disc.json --noise 0,5,10,20,40
    python3 dsa_degrade.py disc.png disc.disc.json --jpeg 95,80,60,40,20
    python3 dsa_degrade.py disc.png disc.disc.json --all --maps /tmp/maps/

License: GPL v3 — github.com/pisdronio/dsa
"""

import argparse
import io
import json
import math
import sys
from pathlib import Path

import numpy as np

try:
    from PIL import Image, ImageFilter
except ImportError:
    sys.exit("Pillow required — pip install Pillow")

# ─── Disc geometry (mm) ───────────────────────────────────────────────────────

DISC_DIAMETER_MM  = 290.0
OUTER_AUDIO_MM    = 141.0
INNER_AUDIO_MM    = 62.0
NUM_BANDS         = 48

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

DIRECTION_THRESHOLD = 0.008

# Failure thresholds per layer
L0_THRESHOLD = 0.85
L1_THRESHOLD = 0.80
L2_THRESHOLD = 0.70


# ─── Degradation functions ────────────────────────────────────────────────────

def apply_blur(img_arr: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur with radius sigma pixels."""
    if sigma <= 0:
        return img_arr
    pil = Image.fromarray(img_arr)
    pil = pil.filter(ImageFilter.GaussianBlur(radius=sigma))
    return np.array(pil, dtype=np.uint8)


def apply_noise(img_arr: np.ndarray, sigma: float) -> np.ndarray:
    """Additive Gaussian noise, sigma in [0, 255] range."""
    if sigma <= 0:
        return img_arr
    rng   = np.random.default_rng(42)   # fixed seed for reproducibility
    noise = rng.normal(0, sigma, img_arr.shape)
    out   = np.clip(img_arr.astype(np.float64) + noise, 0, 255).astype(np.uint8)
    return out


def apply_cast(img_arr: np.ndarray, rgb_scale: tuple) -> np.ndarray:
    """Multiplicative per-channel color cast."""
    r, g, b = rgb_scale
    out = img_arr.astype(np.float64)
    out[..., 0] *= r
    out[..., 1] *= g
    out[..., 2] *= b
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_fade(img_arr: np.ndarray, factor: float) -> np.ndarray:
    """Blend toward mid-gray (contrast reduction). factor=0 = original."""
    if factor <= 0:
        return img_arr
    mid = np.full_like(img_arr, 128, dtype=np.float64)
    out = img_arr.astype(np.float64) * (1 - factor) + mid * factor
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_jpeg(img_arr: np.ndarray, quality: int) -> np.ndarray:
    """JPEG round-trip at given quality (1–95)."""
    buf = io.BytesIO()
    Image.fromarray(img_arr).save(buf, format='JPEG', quality=quality)
    buf.seek(0)
    return np.array(Image.open(buf).convert('RGB'), dtype=np.uint8)


# ─── Reader core (inline, no subprocess) ─────────────────────────────────────

def _sample_pixel(img_arr: np.ndarray, ctr: int,
                  r_px: float, theta: float) -> np.ndarray:
    x = ctr + r_px * math.sin(theta)
    y = ctr - r_px * math.cos(theta)
    x0, y0 = int(x), int(y)
    x1, y1 = x0 + 1, y0 + 1
    h, w = img_arr.shape[:2]
    x0 = max(0, min(x0, w - 1))
    x1 = max(0, min(x1, w - 1))
    y0 = max(0, min(y0, h - 1))
    y1 = max(0, min(y1, h - 1))
    fx = x - int(x)
    fy = y - int(y)
    return (img_arr[y0, x0].astype(np.float64) * (1 - fx) * (1 - fy) +
            img_arr[y0, x1].astype(np.float64) * fx       * (1 - fy) +
            img_arr[y1, x0].astype(np.float64) * (1 - fx) * fy       +
            img_arr[y1, x1].astype(np.float64) * fx       * fy)


def _color_to_blend(color, ca, cb):
    v     = cb - ca
    denom = np.dot(v, v)
    if denom < 1e-6:
        return 0.0, 1.0
    t   = float(np.clip(np.dot(color - ca, v) / denom, 0.0, 1.0))
    res = float(np.linalg.norm(color - (ca + t * v)))
    alpha = max(0.0, 1.0 - res / (float(np.linalg.norm(v)) + 1e-9))
    return t, alpha


def read_disc(img_arr: np.ndarray, layout: dict):
    """
    Run the reader on img_arr.  Returns (steep, dir, conf_frame, alpha_band).
    """
    n_frames = layout['n_frames']
    n_bands  = layout['n_bands']
    bp_map   = {b['band']: (b['color_a'], b['color_b'])
                for b in layout['band_pairs']}

    size = img_arr.shape[1]
    ctr  = size // 2
    mm_to_px      = size / DISC_DIAMETER_MM
    inner_px      = INNER_AUDIO_MM * mm_to_px
    ring_w_px     = ((OUTER_AUDIO_MM - INNER_AUDIO_MM) * mm_to_px) / NUM_BANDS
    frame_angle   = 2.0 * math.pi / n_frames

    steep = np.zeros((n_frames, n_bands), dtype=np.float64)
    dirs  = np.zeros((n_frames, n_bands), dtype=np.int8)
    conf  = np.zeros((n_frames, n_bands), dtype=np.float64)

    for b in range(n_bands):
        r_px = inner_px + (b + 0.5) * ring_w_px
        ca   = PALETTE[bp_map[b][0]]
        cb   = PALETTE[bp_map[b][1]]
        for fi in range(n_frames):
            tl = (fi + 0.25) * frame_angle
            tr = (fi + 0.75) * frame_angle
            cl = _sample_pixel(img_arr, ctr, r_px, tl)
            cr = _sample_pixel(img_arr, ctr, r_px, tr)
            tl_v, al = _color_to_blend(cl, ca, cb)
            tr_v, ar = _color_to_blend(cr, ca, cb)
            diff = tr_v - tl_v
            s    = min(1.0, 2.0 * abs(diff))
            d    = 0 if abs(diff) < DIRECTION_THRESHOLD else (1 if diff > 0 else -1)
            steep[fi, b] = s
            dirs[fi, b]  = d
            conf[fi, b]  = (al + ar) / 2.0

    alpha = conf.mean(axis=0)
    return steep, dirs, conf, alpha


def compare(layout: dict,
            steep_read: np.ndarray,
            dir_read: np.ndarray) -> dict:
    n_frames = layout['n_frames']
    n_bands  = layout['n_bands']
    orig_s   = np.zeros((n_frames, n_bands), dtype=np.float64)
    orig_d   = np.zeros((n_frames, n_bands), dtype=np.int8)

    for fd in layout['frames']:
        fi = fd['frame_idx']
        if fd['silence']:
            continue
        for bd in fd['bands']:
            b = bd['band']
            orig_s[fi, b] = bd['steepness']
            orig_d[fi, b] = bd['direction']

    # Per-layer direction accuracy
    l0 = slice(0,  8)
    l1 = slice(8,  24)
    l2 = slice(24, 48)

    def _acc(s): return float((dir_read[:, s] == orig_d[:, s]).mean())

    return {
        'dir_overall':  float((dir_read == orig_d).mean()),
        'dir_l0':       _acc(l0),
        'dir_l1':       _acc(l1),
        'dir_l2':       _acc(l2),
        'steep_mae':    float(np.abs(steep_read - orig_s).mean()),
        'mean_alpha':   0.0,   # filled by caller
    }


# ─── Sweep helpers ────────────────────────────────────────────────────────────

def _run_condition(label: str,
                   img_arr: np.ndarray,
                   layout: dict,
                   maps_dir: str = None) -> dict:
    steep, dirs, conf, alpha = read_disc(img_arr, layout)
    metrics = compare(layout, steep, dirs)
    metrics['mean_alpha'] = float(alpha.mean())
    metrics['label'] = label

    if maps_dir:
        from dsa_reader import DSADiscReader
        # Build a minimal reader-like object to call confidence_map
        # (inline to avoid circular deps)
        _save_conf_map(conf, layout['n_frames'], layout['n_bands'],
                       Path(maps_dir) / f"conf_{label}.png")

    return metrics


def _save_conf_map(conf: np.ndarray,
                   n_frames: int, n_bands: int,
                   out_path: Path) -> None:
    cell_h = 4
    arr = np.zeros((n_bands * cell_h, n_frames, 3), dtype=np.uint8)
    for b in range(n_bands):
        rt = (n_bands - 1 - b) * cell_h
        rb = rt + cell_h
        a  = conf[:, b]
        arr[rt:rb, :, 0] = ((1 - a) * 220).astype(np.uint8)
        arr[rt:rb, :, 1] = (a * 180).astype(np.uint8)
    Image.fromarray(arr).save(str(out_path))


def _print_table(results: list[dict]) -> None:
    print()
    print(f"  {'Condition':22s}  {'Overall':>8s}  {'L0':>7s}  {'L1':>7s}  {'L2':>7s}  "
          f"{'StpMAE':>7s}  {'Mean α':>7s}  Flags")
    print(f"  {'─'*22}  {'─'*8}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*5}")
    for r in results:
        flags = []
        if r['dir_l0'] < L0_THRESHOLD: flags.append('L0!')
        if r['dir_l1'] < L1_THRESHOLD: flags.append('L1!')
        if r['dir_l2'] < L2_THRESHOLD: flags.append('L2!')
        flag_str = ' '.join(flags) if flags else '—'
        print(f"  {r['label']:22s}  "
              f"{r['dir_overall']*100:7.2f}%  "
              f"{r['dir_l0']*100:6.2f}%  "
              f"{r['dir_l1']*100:6.2f}%  "
              f"{r['dir_l2']*100:6.2f}%  "
              f"{r['steep_mae']:7.4f}  "
              f"{r['mean_alpha']:7.4f}  "
              f"{flag_str}")
    print()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description='Simulate analog degradations on a DSA disc image and measure read accuracy')
    p.add_argument('image',   help='Disc PNG (from dsa_render.py)')
    p.add_argument('layout',  help='.disc.json layout file')
    p.add_argument('--blur',  type=str, default=None,
                   metavar='S[,S,...]',
                   help='Gaussian blur sigma(s) in pixels, e.g. 0,1,2,3,5')
    p.add_argument('--noise', type=str, default=None,
                   metavar='S[,S,...]',
                   help='Additive noise sigma(s) in 0–255, e.g. 0,5,10,20')
    p.add_argument('--fade',  type=str, default=None,
                   metavar='F[,F,...]',
                   help='Contrast fade factor(s) 0.0–1.0, e.g. 0,0.1,0.3,0.5')
    p.add_argument('--cast',  type=str, default=None,
                   metavar='R,G,B',
                   help='Single color cast as R,G,B scale factors, e.g. 1.2,0.9,0.8')
    p.add_argument('--jpeg',  type=str, default=None,
                   metavar='Q[,Q,...]',
                   help='JPEG quality levels, e.g. 95,80,60,40,20')
    p.add_argument('--all',   action='store_true',
                   help='Run default sweep across all degradation types')
    p.add_argument('--maps',  type=str, default=None, metavar='DIR',
                   help='Save confidence map PNGs to DIR')
    args = p.parse_args()

    print()
    print("  DSA — Analog Degradation Simulator")
    print("  ────────────────────────────────────────────────────────────────")
    print()

    # Load
    print(f"  Loading {Path(args.image).name} ...", end=' ', flush=True)
    orig_arr = np.array(Image.open(args.image).convert('RGB'), dtype=np.uint8)
    print(f"done  ({orig_arr.shape[1]}×{orig_arr.shape[0]}px)")

    print(f"  Loading {Path(args.layout).name} ...", end=' ', flush=True)
    with open(args.layout) as f:
        layout = json.load(f)
    n_frames = layout['n_frames']
    n_bands  = layout['n_bands']
    print(f"done  ({n_frames} frames, {n_bands} bands)")
    print()

    if args.maps:
        Path(args.maps).mkdir(parents=True, exist_ok=True)

    results = []

    # ── Baseline ──────────────────────────────────────────────────────────────
    print("  Baseline (no degradation)...", end=' ', flush=True)
    r = _run_condition('baseline', orig_arr, layout, args.maps)
    results.append(r)
    print(f"done  ({r['dir_overall']*100:.2f}% dir, α={r['mean_alpha']:.4f})")

    # ── Blur sweep ────────────────────────────────────────────────────────────
    blur_vals = []
    if args.blur:
        blur_vals = [float(x) for x in args.blur.split(',')]
    elif args.all:
        blur_vals = [1, 2, 3, 5, 8]

    for s in blur_vals:
        label = f"blur σ={s:.1f}px"
        print(f"  {label}...", end=' ', flush=True)
        degraded = apply_blur(orig_arr, s)
        r = _run_condition(label, degraded, layout, args.maps)
        results.append(r)
        print(f"done  ({r['dir_overall']*100:.2f}%)")

    # ── Noise sweep ───────────────────────────────────────────────────────────
    noise_vals = []
    if args.noise:
        noise_vals = [float(x) for x in args.noise.split(',')]
    elif args.all:
        noise_vals = [5, 10, 20, 40]

    for s in noise_vals:
        label = f"noise σ={s:.0f}"
        print(f"  {label}...", end=' ', flush=True)
        degraded = apply_noise(orig_arr, s)
        r = _run_condition(label, degraded, layout, args.maps)
        results.append(r)
        print(f"done  ({r['dir_overall']*100:.2f}%)")

    # ── Fade sweep ────────────────────────────────────────────────────────────
    fade_vals = []
    if args.fade:
        fade_vals = [float(x) for x in args.fade.split(',')]
    elif args.all:
        fade_vals = [0.1, 0.2, 0.3, 0.5]

    for f in fade_vals:
        label = f"fade {f*100:.0f}%"
        print(f"  {label}...", end=' ', flush=True)
        degraded = apply_fade(orig_arr, f)
        r = _run_condition(label, degraded, layout, args.maps)
        results.append(r)
        print(f"done  ({r['dir_overall']*100:.2f}%)")

    # ── JPEG sweep ────────────────────────────────────────────────────────────
    jpeg_vals = []
    if args.jpeg:
        jpeg_vals = [int(x) for x in args.jpeg.split(',')]
    elif args.all:
        jpeg_vals = [95, 80, 60, 40, 20]

    for q in jpeg_vals:
        label = f"jpeg q={q}"
        print(f"  {label}...", end=' ', flush=True)
        degraded = apply_jpeg(orig_arr, q)
        r = _run_condition(label, degraded, layout, args.maps)
        results.append(r)
        print(f"done  ({r['dir_overall']*100:.2f}%)")

    # ── Color cast (single) ───────────────────────────────────────────────────
    if args.cast or args.all:
        cast_rgb = (1.2, 0.9, 0.8) if args.all else tuple(
            float(x) for x in args.cast.split(','))
        label = f"cast R×{cast_rgb[0]} G×{cast_rgb[1]} B×{cast_rgb[2]}"
        print(f"  {label}...", end=' ', flush=True)
        degraded = apply_cast(orig_arr, cast_rgb)
        r = _run_condition(label, degraded, layout, args.maps)
        results.append(r)
        print(f"done  ({r['dir_overall']*100:.2f}%)")

    # ── Results table ─────────────────────────────────────────────────────────
    print()
    print("  ── Results ───────────────────────────────────────────────────")
    _print_table(results)

    # ── Failure summary ───────────────────────────────────────────────────────
    failures = [r for r in results if
                r['dir_l0'] < L0_THRESHOLD or
                r['dir_l1'] < L1_THRESHOLD or
                r['dir_l2'] < L2_THRESHOLD]
    if failures:
        print("  ── Failure conditions (below layer threshold) ─────────────")
        for r in failures:
            issues = []
            if r['dir_l0'] < L0_THRESHOLD:
                issues.append(f"L0={r['dir_l0']*100:.1f}% < {L0_THRESHOLD*100:.0f}%")
            if r['dir_l1'] < L1_THRESHOLD:
                issues.append(f"L1={r['dir_l1']*100:.1f}% < {L1_THRESHOLD*100:.0f}%")
            if r['dir_l2'] < L2_THRESHOLD:
                issues.append(f"L2={r['dir_l2']*100:.1f}% < {L2_THRESHOLD*100:.0f}%")
            print(f"  {r['label']:22s}  {', '.join(issues)}")
        print()
    else:
        print("  All conditions passed all layer thresholds.")
        print()


if __name__ == '__main__':
    main()
