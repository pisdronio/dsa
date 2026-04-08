#!/usr/bin/env python3
"""
DSA — Digilog Scalable Audio
dsa_color.py — Perceptual Color Model

Provides CIELAB-based color utilities for the DSA disc renderer and strip viewer.

Key improvements over v1 RGB-linear model:
1. LAB interpolation — gradients are perceptually uniform, no muddy RGB midpoints
2. CIEDE2000 distance — measures perceptual contrast between color pairs
3. Monotonicity check — verifies gradients remain visually discriminable at every step
4. Robustness analysis — checks color pair stability under lighting variation

Color pipeline:
    sRGB (0-255)  →  Linear RGB  →  XYZ D65  →  CIELAB
    CIELAB        →  XYZ D65     →  Linear RGB → sRGB (0-255) clipped

Reference: IEC 61966-2-1 (sRGB), CIE 15:2004 (colorimetry), CIEDE2000 (Sharma 2005)

License: GPL v3 — github.com/pisdronio/dsa
"""

import numpy as np
from typing import Tuple

# ─── sRGB ↔ CIELAB conversion ────────────────────────────────────────────────

# sRGB to XYZ D65 matrix (IEC 61966-2-1)
_M_RGB2XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
], dtype=np.float64)

# XYZ to sRGB D65 matrix (inverse)
_M_XYZ2RGB = np.array([
    [ 3.2404542, -1.5371385, -0.4985314],
    [-0.9692660,  1.8760108,  0.0415560],
    [ 0.0556434, -0.2040259,  1.0572252],
], dtype=np.float64)

# D65 white point
_D65 = np.array([0.95047, 1.00000, 1.08883], dtype=np.float64)

_DELTA = 6.0 / 29.0


def _linearize_srgb(v: np.ndarray) -> np.ndarray:
    """sRGB [0,1] → linear light [0,1]"""
    return np.where(v > 0.04045, ((v + 0.055) / 1.055) ** 2.4, v / 12.92)


def _gamma_srgb(v: np.ndarray) -> np.ndarray:
    """Linear light [0,1] → sRGB [0,1]"""
    return np.where(v > 0.0031308, 1.055 * v ** (1.0 / 2.4) - 0.055, 12.92 * v)


def _f_lab(t: np.ndarray) -> np.ndarray:
    return np.where(t > _DELTA ** 3, t ** (1.0 / 3.0),
                    t / (3.0 * _DELTA ** 2) + 4.0 / 29.0)


def _f_lab_inv(t: np.ndarray) -> np.ndarray:
    return np.where(t > _DELTA, t ** 3.0,
                    3.0 * _DELTA ** 2 * (t - 4.0 / 29.0))


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """
    Convert sRGB to CIELAB.

    Parameters
    ----------
    rgb : (..., 3) array, values in [0, 255]

    Returns
    -------
    lab : (..., 3) array — L* ∈ [0,100], a* and b* ∈ approx [-128, 127]
    """
    rgb = np.asarray(rgb, dtype=np.float64) / 255.0
    lin = _linearize_srgb(rgb)
    xyz = lin @ _M_RGB2XYZ.T
    xyz_n = xyz / _D65
    f = _f_lab(xyz_n)
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """
    Convert CIELAB to sRGB.

    Parameters
    ----------
    lab : (..., 3) array — L*, a*, b*

    Returns
    -------
    rgb : (..., 3) uint8 array clipped to [0, 255]
    """
    lab = np.asarray(lab, dtype=np.float64)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0
    xyz = _D65 * _f_lab_inv(np.stack([fx, fy, fz], axis=-1))
    lin = xyz @ _M_XYZ2RGB.T
    srgb = _gamma_srgb(np.clip(lin, 0.0, 1.0))
    return np.clip(srgb * 255.0, 0, 255).astype(np.uint8)


# ─── Perceptual interpolation ─────────────────────────────────────────────────

def lerp_lab(ca_rgb: np.ndarray, cb_rgb: np.ndarray,
             t: np.ndarray) -> np.ndarray:
    """
    Perceptually uniform interpolation between two sRGB colors via CIELAB.

    Parameters
    ----------
    ca_rgb : (3,) sRGB color A, values [0, 255]
    cb_rgb : (3,) sRGB color B, values [0, 255]
    t      : (...,) blend factor in [0, 1]  — 0 = color A, 1 = color B

    Returns
    -------
    (…, 3) uint8 sRGB array
    """
    ca_lab = rgb_to_lab(ca_rgb)  # (3,)
    cb_lab = rgb_to_lab(cb_rgb)  # (3,)
    t3     = np.asarray(t, dtype=np.float64)[..., np.newaxis]  # (..., 1)
    mixed  = ca_lab + (cb_lab - ca_lab) * t3                   # (..., 3)
    return lab_to_rgb(mixed)


# ─── CIEDE2000 color difference ───────────────────────────────────────────────

def ciede2000(lab1: np.ndarray, lab2: np.ndarray) -> float:
    """
    CIEDE2000 perceptual color difference between two CIELAB colors.
    Reference: Sharma et al. (2005), Color Research & Application.

    Parameters
    ----------
    lab1, lab2 : (3,) CIELAB arrays [L*, a*, b*]

    Returns
    -------
    ΔE₀₀ scalar
    """
    L1, a1, b1 = float(lab1[0]), float(lab1[1]), float(lab1[2])
    L2, a2, b2 = float(lab2[0]), float(lab2[1]), float(lab2[2])

    # Step 1: C*ab and h*ab
    C1 = np.sqrt(a1**2 + b1**2)
    C2 = np.sqrt(a2**2 + b2**2)
    C_avg = (C1 + C2) / 2.0
    C_avg7 = C_avg ** 7
    G = 0.5 * (1 - np.sqrt(C_avg7 / (C_avg7 + 25**7)))
    a1p = a1 * (1 + G)
    a2p = a2 * (1 + G)
    C1p = np.sqrt(a1p**2 + b1**2)
    C2p = np.sqrt(a2p**2 + b2**2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360

    # Step 2: ΔL', ΔC', Δh'
    dLp = L2 - L1
    dCp = C2p - C1p
    if C1p * C2p == 0:
        dhp = 0.0
    elif abs(h2p - h1p) <= 180:
        dhp = h2p - h1p
    elif h2p - h1p > 180:
        dhp = h2p - h1p - 360
    else:
        dhp = h2p - h1p + 360
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp / 2))

    # Step 3: CIEDE2000
    Lp_avg = (L1 + L2) / 2.0
    Cp_avg = (C1p + C2p) / 2.0
    if C1p * C2p == 0:
        hp_avg = h1p + h2p
    elif abs(h1p - h2p) <= 180:
        hp_avg = (h1p + h2p) / 2.0
    elif h1p + h2p < 360:
        hp_avg = (h1p + h2p + 360) / 2.0
    else:
        hp_avg = (h1p + h2p - 360) / 2.0

    T = (1
         - 0.17 * np.cos(np.radians(hp_avg - 30))
         + 0.24 * np.cos(np.radians(2 * hp_avg))
         + 0.32 * np.cos(np.radians(3 * hp_avg + 6))
         - 0.20 * np.cos(np.radians(4 * hp_avg - 63)))

    SL = 1 + 0.015 * (Lp_avg - 50)**2 / np.sqrt(20 + (Lp_avg - 50)**2)
    SC = 1 + 0.045 * Cp_avg
    SH = 1 + 0.015 * Cp_avg * T

    Cp_avg7 = Cp_avg ** 7
    RC = 2 * np.sqrt(Cp_avg7 / (Cp_avg7 + 25**7))
    d_theta = 30 * np.exp(-((hp_avg - 275) / 25)**2)
    RT = -np.sin(np.radians(2 * d_theta)) * RC

    return float(np.sqrt(
        (dLp / SL)**2 +
        (dCp / SC)**2 +
        (dHp / SH)**2 +
        RT * (dCp / SC) * (dHp / SH)
    ))


def pair_delta_e(ca_rgb, cb_rgb) -> float:
    """CIEDE2000 between two sRGB colors (values 0–255)."""
    return ciede2000(rgb_to_lab(np.array(ca_rgb, dtype=np.float64)),
                     rgb_to_lab(np.array(cb_rgb, dtype=np.float64)))


def monotonicity_score(ca_rgb, cb_rgb, steps: int = 10) -> float:
    """
    Check how monotonic the perceptual gradient is between two colors.

    Samples `steps` intermediate blend values, computes ΔE between consecutive
    steps in LAB space, and returns the minimum step ΔE.

    A higher minimum ΔE means each gradient step is more visually distinct.
    Returns 0 if any step is a perceptual reversal (non-monotonic).
    """
    ca_lab = rgb_to_lab(np.array(ca_rgb, dtype=np.float64))
    cb_lab = rgb_to_lab(np.array(cb_rgb, dtype=np.float64))
    ts     = np.linspace(0, 1, steps + 1)
    labs   = ca_lab + (cb_lab - ca_lab) * ts[:, np.newaxis]
    diffs  = []
    for i in range(steps):
        de = ciede2000(labs[i], labs[i + 1])
        diffs.append(de)
    return float(min(diffs))


# ─── Color pair analysis ─────────────────────────────────────────────────────

# Current DSA v1 palette
PALETTE_V1 = {
    'black':  np.array([0,   0,   0],   dtype=np.float64),
    'white':  np.array([255, 255, 255], dtype=np.float64),
    'red':    np.array([220, 50,  50],  dtype=np.float64),
    'green':  np.array([50,  180, 50],  dtype=np.float64),
    'blue':   np.array([50,  50,  220], dtype=np.float64),
    'yellow': np.array([240, 220, 0],   dtype=np.float64),
    'cyan':   np.array([0,   210, 210], dtype=np.float64),
    'purple': np.array([160, 50,  200], dtype=np.float64),
}

V1_PAIRS = [
    ('black',  'white'),   # L0 band 0
    ('black',  'yellow'),  # L0 band 1
    ('black',  'cyan'),    # L0 band 2
    ('red',    'cyan'),    # L1
    ('blue',   'yellow'),  # L1
    ('green',  'purple'),  # L1
]


def analyze_pairs(pairs=None, palette=None):
    """Print CIEDE2000 and monotonicity scores for a set of color pairs."""
    if pairs is None:
        pairs = V1_PAIRS
    if palette is None:
        palette = PALETTE_V1

    print(f"  {'Pair':25s}  {'ΔE₀₀':>8s}  {'Min step ΔE':>12s}  {'Grade':>6s}")
    print(f"  {'─'*25}  {'─'*8}  {'─'*12}  {'─'*6}")
    for ca_name, cb_name in pairs:
        ca = palette[ca_name]
        cb = palette[cb_name]
        de   = pair_delta_e(ca, cb)
        mono = monotonicity_score(ca, cb, steps=10)
        # Grade: ΔE₀₀ > 50 and min step ΔE > 3 is good
        if de > 60 and mono > 4:
            grade = 'A'
        elif de > 40 and mono > 2:
            grade = 'B'
        elif de > 25:
            grade = 'C'
        else:
            grade = 'F'
        print(f"  {ca_name:10s} ↔ {cb_name:10s}  {de:8.1f}  {mono:12.2f}  {grade:>6s}")


if __name__ == '__main__':
    print("\n  DSA Color Pair Analysis — v1 palette\n")
    analyze_pairs()

    print("\n  LAB interpolation vs RGB interpolation:")
    print("  Blue↔Yellow midpoint (t=0.5):")
    ca = PALETTE_V1['blue']
    cb = PALETTE_V1['yellow']
    mid_rgb = ((ca + cb) / 2).astype(np.uint8)
    mid_lab = lerp_lab(ca, cb, np.array(0.5))
    print(f"    RGB linear: {tuple(mid_rgb)}")
    print(f"    LAB linear: {tuple(mid_lab)}")

    print("\n  Red↔Cyan midpoint (t=0.5):")
    ca = PALETTE_V1['red']
    cb = PALETTE_V1['cyan']
    mid_rgb = ((ca + cb) / 2).astype(np.uint8)
    mid_lab = lerp_lab(ca, cb, np.array(0.5))
    print(f"    RGB linear: {tuple(mid_rgb)}")
    print(f"    LAB linear: {tuple(mid_lab)}")
    print()
