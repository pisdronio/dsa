#!/usr/bin/env python3
"""
DSA — Digilog Scalable Audio
Benchmark: DSA vs Opus

Measures round-trip SNR, bitrate accuracy, and encode/decode throughput
at multiple bitrates. Also reports DSA-unique properties that Opus cannot
provide (layer isolation, analog degradation, reverse playback).

Usage:
  python3 dsa_bench.py            # full benchmark, table output
  python3 dsa_bench.py --quick    # fewer signals, faster
  python3 dsa_bench.py --csv      # CSV output for plotting

Requirements:
  pip install numpy scipy
  brew install opus-tools ffmpeg   # for Opus comparison
  (DSA runs without external tools)

License: GPL v3 — github.com/pisdronio/digilog-spec
"""

import argparse
import os
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

import numpy as np

from dsa_analyzer import (
    DSAAnalyzer, NUM_BANDS, L0, L1, L2, BINS,
    MDCT_M, MDCT_N, HOP, SAMPLE_RATE, GOP_SIZE,
    mdct, coeffs_to_bands,
)
from dsa_bitstream import DSABitstreamWriter, DSABitstreamReader
from dsa_decoder import DSADecoder, _alpha_for_layers

# ─── Test signals ─────────────────────────────────────────────────────────────

DURATION_S   = 5.0   # seconds per test signal
N_SAMPLES    = int(DURATION_S * SAMPLE_RATE)


def _make_signal(kind: str) -> np.ndarray:
    t = np.arange(N_SAMPLES) / SAMPLE_RATE
    if kind == 'tone_440':
        return 0.5 * np.sin(2 * np.pi * 440 * t)
    if kind == 'tone_1k':
        return 0.5 * np.sin(2 * np.pi * 1000 * t)
    if kind == 'tone_4k':
        return 0.5 * np.sin(2 * np.pi * 4000 * t)
    if kind == 'three_tone':
        return (0.5 * np.sin(2 * np.pi * 440  * t)
              + 0.3 * np.sin(2 * np.pi * 2000 * t)
              + 0.2 * np.sin(2 * np.pi * 9000 * t))
    if kind == 'white_noise':
        rng = np.random.default_rng(42)
        return 0.3 * rng.standard_normal(N_SAMPLES)
    if kind == 'chirp':
        # Linear frequency sweep 100Hz → 20kHz
        f0, f1 = 100.0, 20000.0
        phase  = 2 * np.pi * (f0 * t + 0.5 * (f1 - f0) / DURATION_S * t ** 2)
        return 0.4 * np.sin(phase)
    raise ValueError(f"Unknown signal kind: {kind}")


SIGNAL_LABELS = {
    'tone_440':   '440 Hz tone',
    'tone_1k':    '1 kHz tone',
    'tone_4k':    '4 kHz tone',
    'three_tone': '3-tone (440+2k+9k)',
    'white_noise': 'White noise',
    'chirp':      'Chirp (100→20kHz)',
}

QUICK_SIGNALS = ['three_tone', 'white_noise']
FULL_SIGNALS  = list(SIGNAL_LABELS.keys())

BITRATES = [6, 12, 32, 96]

# ─── WAV helpers ──────────────────────────────────────────────────────────────

def _write_wav(path: str, samples: np.ndarray, sr: int = SAMPLE_RATE):
    s16 = np.clip(samples * 32767, -32768, 32767).astype(np.int16)
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(s16.tobytes())


def _read_wav(path: str) -> np.ndarray:
    with wave.open(path, 'rb') as wf:
        raw = wf.readframes(wf.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0


# ─── SNR measurement ──────────────────────────────────────────────────────────

def _snr(ref: np.ndarray, dec: np.ndarray, skip: int = MDCT_N) -> float:
    """SNR in dB, interior only (skip samples to avoid edge artifacts)."""
    n   = min(len(ref), len(dec)) - 2 * skip
    if n <= 0:
        return float('-inf')
    r   = ref[skip:skip + n]
    d   = dec[skip:skip + n]
    rms = np.sqrt(np.mean(r ** 2))
    err = np.sqrt(np.mean((r - d) ** 2))
    return 20 * np.log10(rms / (err + 1e-12))


# ─── DSA encode/decode ────────────────────────────────────────────────────────

def bench_dsa(sig: np.ndarray, bitrate_kbps: float) -> dict:
    """Full DSA encode → decode round-trip. Returns metrics dict."""
    analyzer = DSAAnalyzer()

    t0     = time.perf_counter()
    frames = analyzer.analyze_samples(sig)
    raw    = DSABitstreamWriter(bitrate_kbps=bitrate_kbps).encode_frames(frames)
    t_enc  = time.perf_counter() - t0

    reader = DSABitstreamReader(raw)

    t0      = time.perf_counter()
    decoded = DSADecoder().decode(reader)
    t_dec   = time.perf_counter() - t0

    snr         = _snr(sig, decoded)
    actual_kbps = len(raw) * 8 / DURATION_S / 1000.0
    enc_xrt     = DURATION_S / t_enc   # × real-time
    dec_xrt     = DURATION_S / t_dec

    return {
        'snr_db':      snr,
        'actual_kbps': actual_kbps,
        'enc_xrt':     enc_xrt,
        'dec_xrt':     dec_xrt,
        'bytes':       len(raw),
    }


# ─── Opus encode/decode ───────────────────────────────────────────────────────

def _opus_available() -> bool:
    try:
        r = subprocess.run(['opusenc', '--version'],
                           capture_output=True, timeout=5)
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def bench_opus(sig: np.ndarray, bitrate_kbps: float):
    """Opus encode → decode round-trip via opusenc/opusdec. Returns None if unavailable."""
    if not _opus_available():
        return None

    with tempfile.TemporaryDirectory() as td:
        wav_in  = os.path.join(td, 'input.wav')
        opus_f  = os.path.join(td, 'encoded.opus')
        wav_out = os.path.join(td, 'decoded.wav')

        _write_wav(wav_in, sig)

        t0  = time.perf_counter()
        r   = subprocess.run(
            ['opusenc', '--bitrate', str(int(bitrate_kbps)),
             '--comp', '10', wav_in, opus_f],
            capture_output=True, timeout=60
        )
        t_enc = time.perf_counter() - t0
        if r.returncode != 0:
            return None

        t0 = time.perf_counter()
        r  = subprocess.run(
            ['opusdec', '--rate', str(SAMPLE_RATE), opus_f, wav_out],
            capture_output=True, timeout=60
        )
        t_dec = time.perf_counter() - t0
        if r.returncode != 0:
            return None

        decoded     = _read_wav(wav_out)
        snr         = _snr(sig, decoded)
        actual_kbps = Path(opus_f).stat().st_size * 8 / DURATION_S / 1000.0
        enc_xrt     = DURATION_S / t_enc
        dec_xrt     = DURATION_S / t_dec

        return {
            'snr_db':      snr,
            'actual_kbps': actual_kbps,
            'enc_xrt':     enc_xrt,
            'dec_xrt':     dec_xrt,
            'bytes':       Path(opus_f).stat().st_size,
        }


# ─── DSA-unique properties ────────────────────────────────────────────────────

def bench_dsa_unique(sig: np.ndarray, bitrate_kbps: float = 12.0) -> dict:
    """Measure properties unique to DSA (not comparable to Opus)."""
    analyzer = DSAAnalyzer()
    frames   = analyzer.analyze_samples(sig)
    raw      = DSABitstreamWriter(bitrate_kbps=bitrate_kbps).encode_frames(frames)

    results = {}

    # ── Layer isolation SNR ───────────────────────────────────────────────
    for layer_set, label in [((0,), 'l0_only'), ((0, 1), 'l0_l1')]:
        reader = DSABitstreamReader(raw)
        dec    = DSADecoder().decode(reader, layers=layer_set)
        # Layer-isolated decode is intentionally lower quality — report RMS not SNR
        results[f'rms_{label}'] = float(np.sqrt(np.mean(dec[MDCT_N:] ** 2)))

    full_reader = DSABitstreamReader(raw)
    full_dec    = DSADecoder().decode(full_reader)
    results['rms_full'] = float(np.sqrt(np.mean(full_dec[MDCT_N:] ** 2)))

    # ── Degradation curve: SNR vs α ───────────────────────────────────────
    deg_snrs = {}
    for alpha_val in [1.0, 0.7, 0.5, 0.3, 0.1]:
        reader  = DSABitstreamReader(raw)
        dec     = DSADecoder().decode(reader, alpha=np.full(NUM_BANDS, alpha_val))
        rms_dec = float(np.sqrt(np.mean(dec[MDCT_N:] ** 2)))
        rms_ref = float(np.sqrt(np.mean(full_dec[MDCT_N:] ** 2)))
        attn_db = 20 * np.log10(rms_dec / (rms_ref + 1e-12))
        deg_snrs[alpha_val] = attn_db
    results['degradation_db'] = deg_snrs

    # ── Reverse playback: temporal order reversed ─────────────────────────
    # Build a two-part signal (loud first half, silent second half).
    n      = len(sig)
    half   = n // 2
    ramp   = np.zeros(n)
    t      = np.arange(half) / SAMPLE_RATE
    ramp[:half] = 0.5 * np.sin(2 * np.pi * 440 * t)
    r_frames = analyzer.analyze_samples(ramp)
    r_raw    = DSABitstreamWriter(bitrate_kbps=bitrate_kbps).encode_frames(r_frames)

    r_reader = DSABitstreamReader(r_raw)
    dec_fwd  = DSADecoder().decode(r_reader)
    dec_rev  = DSADecoder().decode(r_reader, reverse=True)
    nr       = min(len(dec_fwd), len(dec_rev))
    nh       = nr // 2

    fwd_ratio = (np.sqrt(np.mean(dec_fwd[:nh] ** 2)) /
                 (np.sqrt(np.mean(dec_fwd[nh:] ** 2)) + 1e-12))
    rev_ratio = (np.sqrt(np.mean(dec_rev[nh:] ** 2)) /
                 (np.sqrt(np.mean(dec_rev[:nh] ** 2)) + 1e-12))
    results['reverse_fwd_ratio'] = fwd_ratio   # loud:quiet in fwd (should be >> 1)
    results['reverse_rev_ratio'] = rev_ratio   # loud:quiet in rev (should be >> 1)

    return results


# ─── Report formatting ────────────────────────────────────────────────────────

def _snr_str(v) -> str:
    if v is None:
        return '   n/a'
    return f'{v:+6.1f}'


def print_snr_table(results: dict, opus_ok: bool):
    """
    results: {signal_kind: {bitrate: {'dsa': {...}, 'opus': {...}}}}
    """
    print()
    print('  Round-trip SNR (dB) — DSA vs Opus')
    print('  ' + '─' * 74)
    hdr = f'  {"Signal":<22}'
    for br in BITRATES:
        col = f'{br}kbps'
        hdr += f'  {"DSA":>6} {"Opus":>6}' if opus_ok else f'  {"DSA":>6}'
    print(hdr)
    print('  ' + '─' * 74)

    for kind, by_br in results.items():
        label = SIGNAL_LABELS.get(kind, kind)
        row   = f'  {label:<22}'
        for br in BITRATES:
            d = by_br.get(br, {})
            dsa_snr  = d.get('dsa',  {}).get('snr_db')
            opus_snr = d.get('opus', {}).get('snr_db')
            row += f'  {_snr_str(dsa_snr)}'
            if opus_ok:
                row += f' {_snr_str(opus_snr)}'
        print(row)
    print('  ' + '─' * 74)


def print_speed_table(results: dict, opus_ok: bool):
    print()
    print('  Encode / decode speed (× real-time) @ 12kbps')
    print('  ' + '─' * 56)
    print(f'  {"Signal":<22}  {"DSA enc":>8}  {"DSA dec":>8}', end='')
    if opus_ok:
        print(f'  {"Opus enc":>8}  {"Opus dec":>8}')
    else:
        print()
    print('  ' + '─' * 56)
    for kind, by_br in results.items():
        d   = by_br.get(12, {})
        dsa = d.get('dsa',  {})
        opu = d.get('opus', {})
        label = SIGNAL_LABELS.get(kind, kind)
        row = (f'  {label:<22}  '
               f'{dsa.get("enc_xrt", 0):>7.1f}×  '
               f'{dsa.get("dec_xrt", 0):>7.1f}×')
        if opus_ok:
            row += (f'  {opu.get("enc_xrt", 0):>7.1f}×  '
                    f'{opu.get("dec_xrt", 0):>7.1f}×')
        print(row)
    print('  ' + '─' * 56)


def print_unique_table(unique: dict):
    print()
    print('  DSA-unique properties (@ 12kbps, three-tone signal)')
    print('  ' + '─' * 56)

    # Layer isolation
    rms_full = unique.get('rms_full', 0)
    rms_l0   = unique.get('rms_l0_only', 0)
    rms_l0l1 = unique.get('rms_l0_l1', 0)
    db_l0    = 20 * np.log10(rms_l0   / (rms_full + 1e-12))
    db_l0l1  = 20 * np.log10(rms_l0l1 / (rms_full + 1e-12))
    print(f'  Layer isolation:')
    print(f'    L0 only  (inner ring):  {db_l0:.1f} dB vs full  '
          f'— bass always readable')
    print(f'    L0+L1    (no outer):    {db_l0l1:.1f} dB vs full  '
          f'— phone-quality read')
    print(f'    L0+L1+L2 (full):         0.0 dB  — controlled rig')

    # Degradation
    print(f'  Analog degradation (α attenuation vs full):')
    for alpha_val, attn_db in sorted(unique.get('degradation_db', {}).items(),
                                     reverse=True):
        bar_len = max(0, int((attn_db + 30) * 1.2))
        bar     = '█' * bar_len
        print(f'    α={alpha_val:.1f}  {attn_db:+6.1f} dB  {bar}')

    # Reverse
    fwd_r = unique.get('reverse_fwd_ratio', 0)
    rev_r = unique.get('reverse_rev_ratio', 0)
    fwd_ok = '✓' if fwd_r > 3 else '✗'
    rev_ok = '✓' if rev_r > 3 else '✗'
    print(f'  Reverse playback (loud:quiet energy ratio):')
    print(f'    Forward:  {fwd_r:.1f}× (loud first half)   {fwd_ok}')
    print(f'    Reverse:  {rev_r:.1f}× (loud second half)  {rev_ok}')
    print('  ' + '─' * 56)


def print_csv(results: dict, opus_ok: bool):
    header = 'signal,bitrate_kbps,codec,snr_db,actual_kbps,enc_xrt,dec_xrt'
    print(header)
    for kind, by_br in results.items():
        for br, codecs in by_br.items():
            for codec, m in codecs.items():
                if m is None:
                    continue
                print(f'{kind},{br},{codec},'
                      f'{m.get("snr_db", ""):.2f},'
                      f'{m.get("actual_kbps", ""):.2f},'
                      f'{m.get("enc_xrt", ""):.1f},'
                      f'{m.get("dec_xrt", ""):.1f}')


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_bench(signals: list, bitrates: list, csv_mode: bool = False):
    opus_ok = _opus_available()

    if not csv_mode:
        print(f'\n  DSA — Digilog Scalable Audio')
        print(f'  Benchmark: DSA vs {"Opus" if opus_ok else "Opus (not found — DSA only)"}')
        print(f'  {"─" * 48}')
        print(f'  Signals:  {", ".join(signals)}')
        print(f'  Bitrates: {", ".join(str(b) for b in bitrates)} kbps')
        print(f'  Duration: {DURATION_S:.0f}s per signal')
        print(f'  {"─" * 48}')

    results: dict = {}

    for kind in signals:
        sig       = _make_signal(kind)
        results[kind] = {}
        for br in bitrates:
            if not csv_mode:
                print(f'  {SIGNAL_LABELS[kind]:<22}  {br:>3}kbps  ', end='', flush=True)

            dsa_m  = bench_dsa(sig, br)
            opus_m = bench_opus(sig, br) if opus_ok else None

            results[kind][br] = {'dsa': dsa_m, 'opus': opus_m}

            if not csv_mode:
                dsa_snr  = dsa_m['snr_db']
                opus_snr = opus_m['snr_db'] if opus_m else None
                status = f'DSA {dsa_snr:+.1f}dB'
                if opus_snr is not None:
                    diff   = dsa_snr - opus_snr
                    status += f'  Opus {opus_snr:+.1f}dB  ({diff:+.1f}dB)'
                print(status)

    if csv_mode:
        print_csv(results, opus_ok)
        return

    print_snr_table(results, opus_ok)
    print_speed_table(results, opus_ok)

    # DSA-unique properties on three-tone signal
    three_sig  = _make_signal('three_tone')
    unique     = bench_dsa_unique(three_sig, bitrate_kbps=12.0)
    print_unique_table(unique)

    if not opus_ok:
        print()
        print('  Note: Opus not found. Install with:  brew install opus-tools')
        print('  Rerun for side-by-side comparison.')

    print()
    print('  Scan the groove.')
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DSA benchmark vs Opus')
    parser.add_argument('--quick', action='store_true',
                        help='Fewer signals, faster run')
    parser.add_argument('--csv',   action='store_true',
                        help='CSV output for plotting')
    args = parser.parse_args()

    sigs = QUICK_SIGNALS if args.quick else FULL_SIGNALS
    run_bench(sigs, BITRATES, csv_mode=args.csv)
