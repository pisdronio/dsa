#!/usr/bin/env python3
"""
DSA — Digilog Scalable Audio
dsa_palette.py — CIELAB Color Pair Optimizer

Exhaustively scores all 28 possible pairs from the 8-color DSA palette,
ranks them by perceptual quality, and recommends the best color pair
assignments for each disc layer (L0 / L1 / L2).

Quality metrics (both required):
    ΔE₀₀  — total perceptual contrast between the two endpoint colors
              (CIEDE2000, Sharma 2005). Higher = more discriminable.

    min_step_ΔE  — minimum perceptual step across 10 equispaced blend
                   positions along the LAB gradient (monotonicity score).
                   A high value means the gradient is smooth and every
                   increment is visually detectable.

Layer readability tiers (RESEARCH.md §12 / §15):

    L0  (bands  0– 7, inner rings, bass)
        Worst-case readers: worn print, phone camera with AWB, hand on disc.
        Requirement: ΔE₀₀ > 70 AND min_step > 5.0
        Must survive JPEG compression artifacts and strong color casts.

    L1  (bands  8–23, middle rings, mid)
        Any modern phone camera in normal ambient light.
        Requirement: ΔE₀₀ > 55 AND min_step > 3.5

    L2  (bands 24–47, outer rings, high)
        Digilog Rig only: controlled LED ring at fixed focal distance.
        Requirement: ΔE₀₀ > 40 AND min_step > 2.0

Optimization goal: within each layer, assign pairs to the cycled BAND_PAIRS
list to maximize the *minimum* ΔE₀₀ across bands (max-min fairness) while
keeping all pairs above the layer threshold.

Usage:
    python3 dsa_palette.py             # full report + recommendations
    python3 dsa_palette.py --apply     # update BAND_PAIRS in dsa_disc.py
    python3 dsa_palette.py --json      # emit machine-readable JSON

License: GPL v3 — github.com/pisdronio/dsa
"""

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

from dsa_color import (
    rgb_to_lab, ciede2000, monotonicity_score, PALETTE_V1,
)

# ─── Palette ──────────────────────────────────────────────────────────────────

PALETTE = PALETTE_V1   # canonical 8-color DSA palette

COLOR_NAMES = list(PALETTE.keys())   # fixed order for indexing

# ─── Layer thresholds ─────────────────────────────────────────────────────────

LAYER_THRESHOLDS = {
    'L0': {'delta_e': 70.0, 'min_step': 5.0, 'label': 'L0 (inner/bass,  worst-case reader)'},
    'L1': {'delta_e': 55.0, 'min_step': 3.5, 'label': 'L1 (middle/mid,  phone camera)     '},
    'L2': {'delta_e': 40.0, 'min_step': 2.0, 'label': 'L2 (outer/high,  rig only)         '},
}

# Layer sizes (bands per layer)
L0_SIZE = 8
L1_SIZE = 16
L2_SIZE = 24

# How many distinct pairs each layer cycles through
L0_POOL = 3   # cycles over 3 pairs across 8 bands
L1_POOL = 3   # cycles over 3 pairs across 16 bands
L2_POOL = 8   # cycles over 8 pairs across 24 bands


# ─── Score all pairs ──────────────────────────────────────────────────────────

def score_all_pairs(palette: dict = PALETTE,
                    mono_steps: int = 10) -> list[dict]:
    """
    Compute ΔE₀₀ and monotonicity score for every unique color pair.

    Returns list of dicts sorted by ΔE₀₀ descending.
    """
    results = []
    names = list(palette.keys())
    for a, b in combinations(names, 2):
        ca = palette[a]
        cb = palette[b]
        de   = ciede2000(rgb_to_lab(ca), rgb_to_lab(cb))
        mono = monotonicity_score(ca, cb, steps=mono_steps)

        # Composite score: harmonic mean of ΔE₀₀ and 10× min_step
        # (keeps balance between contrast and gradient quality)
        composite = 2 * de * (10 * mono) / (de + 10 * mono + 1e-9)

        results.append({
            'ca': a,
            'cb': b,
            'delta_e':   round(de,   2),
            'min_step':  round(mono, 3),
            'composite': round(composite, 2),
        })

    results.sort(key=lambda x: x['delta_e'], reverse=True)
    return results


def _grade(delta_e: float, min_step: float) -> str:
    if delta_e > 80 and min_step > 6:
        return 'A+'
    if delta_e > 60 and min_step > 4:
        return 'A'
    if delta_e > 45 and min_step > 2.5:
        return 'B'
    if delta_e > 30:
        return 'C'
    return 'F'


# ─── Layer-filtered pair selection ────────────────────────────────────────────

def _eligible_pairs(all_pairs: list[dict],
                    min_delta_e: float,
                    min_step: float) -> list[dict]:
    return [p for p in all_pairs
            if p['delta_e'] >= min_delta_e and p['min_step'] >= min_step]


def best_pool(all_pairs: list[dict],
              pool_size: int,
              min_delta_e: float,
              min_step: float) -> list[dict]:
    """
    Select `pool_size` pairs from the eligible set that maximize the
    minimum ΔE₀₀ within the pool (max-min fairness).

    Strategy: greedy — keep adding the next-best pair that doesn't
    drop the pool minimum below the already-selected minimum.
    """
    eligible = _eligible_pairs(all_pairs, min_delta_e, min_step)
    if not eligible:
        # Relax thresholds and take best available
        eligible = sorted(all_pairs, key=lambda x: x['delta_e'], reverse=True)

    if len(eligible) <= pool_size:
        return eligible[:pool_size]

    # Greedy max-min selection
    # Start with the top pair (highest ΔE₀₀)
    selected = [eligible[0]]
    remaining = eligible[1:]

    while len(selected) < pool_size and remaining:
        # Add the next pair that maximizes the current minimum
        best_candidate = None
        best_min = -1
        for cand in remaining:
            pool_min = min(p['delta_e'] for p in selected + [cand])
            if pool_min > best_min:
                best_min = pool_min
                best_candidate = cand
        if best_candidate:
            selected.append(best_candidate)
            remaining.remove(best_candidate)
        else:
            break

    return selected


# ─── Report ───────────────────────────────────────────────────────────────────

def print_full_report(all_pairs: list[dict]) -> None:
    print()
    print("  DSA — CIELAB Color Pair Optimizer")
    print("  ────────────────────────────────────────────────────────────────")
    print()
    print("  All 28 pairs, sorted by ΔE₀₀:")
    print()
    print(f"  {'Pair':25s}  {'ΔE₀₀':>8s}  {'Min step ΔE':>12s}  {'Composite':>10s}  Grade")
    print(f"  {'─'*25}  {'─'*8}  {'─'*12}  {'─'*10}  {'─'*5}")
    for p in all_pairs:
        pair_str = f"{p['ca']:10s} ↔ {p['cb']:10s}"
        grade = _grade(p['delta_e'], p['min_step'])
        print(f"  {pair_str:25s}  {p['delta_e']:8.1f}  {p['min_step']:12.3f}  {p['composite']:10.2f}  {grade}")

    print()
    print("  ── Layer recommendations ──────────────────────────────────────")
    print()

    for layer_name, thresh in LAYER_THRESHOLDS.items():
        pool_size = {'L0': L0_POOL, 'L1': L1_POOL, 'L2': L2_POOL}[layer_name]
        pool = best_pool(all_pairs, pool_size,
                         thresh['delta_e'], thresh['min_step'])

        min_de   = min(p['delta_e'] for p in pool) if pool else 0
        label    = thresh['label']

        print(f"  {label}  (need {pool_size} pairs)")
        print(f"  {'Pair':25s}  {'ΔE₀₀':>8s}  {'Min step':>9s}  Grade")
        print(f"  {'─'*25}  {'─'*8}  {'─'*9}  {'─'*5}")
        for p in pool:
            pair_str = f"{p['ca']:10s} ↔ {p['cb']:10s}"
            grade = _grade(p['delta_e'], p['min_step'])
            marker = '  ← best' if p is pool[0] else ''
            print(f"  {pair_str:25s}  {p['delta_e']:8.1f}  {p['min_step']:9.3f}  {grade}{marker}")

        print(f"  Pool minimum ΔE₀₀: {min_de:.1f}")
        print()


def _pool_to_tuples(pool: list[dict]) -> list[tuple[str, str]]:
    return [(p['ca'], p['cb']) for p in pool]


def build_recommendations(all_pairs: list[dict]) -> dict:
    """Return recommended pool tuples for L0, L1, L2."""
    l0 = best_pool(all_pairs, L0_POOL,
                   LAYER_THRESHOLDS['L0']['delta_e'],
                   LAYER_THRESHOLDS['L0']['min_step'])
    l1 = best_pool(all_pairs, L1_POOL,
                   LAYER_THRESHOLDS['L1']['delta_e'],
                   LAYER_THRESHOLDS['L1']['min_step'])
    l2 = best_pool(all_pairs, L2_POOL,
                   LAYER_THRESHOLDS['L2']['delta_e'],
                   LAYER_THRESHOLDS['L2']['min_step'])
    return {
        'L0': _pool_to_tuples(l0),
        'L1': _pool_to_tuples(l1),
        'L2': _pool_to_tuples(l2),
    }


# ─── Apply recommendations to dsa_disc.py ────────────────────────────────────

def apply_to_disc(recs: dict) -> None:
    """
    Rewrite the _L0_PAIRS / _L1_PAIRS / _L2_PAIRS blocks in dsa_disc.py
    with the recommended pairs.
    """
    disc_path = Path(__file__).parent / 'dsa_disc.py'
    src = disc_path.read_text()

    def _pairs_block(pairs: list[tuple[str, str]], indent: str = '    ') -> str:
        lines = [f"{indent}('{a}', '{b}')," for a, b in pairs]
        return '\n'.join(lines)

    import re

    def _replace_block(src: str, var: str, new_pairs: list) -> str:
        block = _pairs_block(new_pairs)
        pattern = rf'({re.escape(var)}\s*=\s*\[)\n(.*?)(\n\])'
        replacement = rf'\1\n{block}\3'
        return re.sub(pattern, replacement, src, flags=re.DOTALL)

    src = _replace_block(src, '_L0_PAIRS', recs['L0'])
    src = _replace_block(src, '_L1_PAIRS', recs['L1'])
    src = _replace_block(src, '_L2_PAIRS', recs['L2'])

    disc_path.write_text(src)
    print(f"  Updated {disc_path.name} with optimized pairs.")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description='Score and optimize DSA color pairs using CIEDE2000')
    p.add_argument('--apply', action='store_true',
                   help='Write recommended pairs back into dsa_disc.py')
    p.add_argument('--json', action='store_true',
                   help='Emit machine-readable JSON (all pairs + recommendations)')
    p.add_argument('--steps', type=int, default=10,
                   help='Gradient monotonicity sample steps (default: 10)')
    args = p.parse_args()

    all_pairs = score_all_pairs(PALETTE, mono_steps=args.steps)

    if args.json:
        recs = build_recommendations(all_pairs)
        out = {
            'all_pairs': all_pairs,
            'recommendations': {
                layer: [{'ca': a, 'cb': b} for a, b in pairs]
                for layer, pairs in recs.items()
            },
        }
        print(json.dumps(out, indent=2))
        return

    print_full_report(all_pairs)

    recs = build_recommendations(all_pairs)

    print("  ── Recommended BAND_PAIRS update ─────────────────────────────")
    print()
    print("  _L0_PAIRS = [")
    for a, b in recs['L0']:
        print(f"      ('{a}', '{b}'),")
    print("  ]")
    print()
    print("  _L1_PAIRS = [")
    for a, b in recs['L1']:
        print(f"      ('{a}', '{b}'),")
    print("  ]")
    print()
    print("  _L2_PAIRS = [")
    for a, b in recs['L2']:
        print(f"      ('{a}', '{b}'),")
    print("  ]")
    print()

    if args.apply:
        apply_to_disc(recs)
    else:
        print("  Run with --apply to write these pairs into dsa_disc.py")
        print()


if __name__ == '__main__':
    main()
