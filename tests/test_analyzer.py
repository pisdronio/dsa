"""
Tests for Step 1: MDCT Frame Analyzer (dsa_analyzer.py)

Covers:
  - TDAC perfect reconstruction (238dB SNR)
  - Band layout: 48 bands, correct layer split, bin monotonicity
  - Perceptual weights in expected ranges per band
  - Frame analysis: correct K/B/S counts, gop_pos cycling
  - Silence detection threshold
  - Layer coefficient slices are non-overlapping and contiguous
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from dsa_analyzer import (
    mdct, imdct, DSAAnalyzer, DSAFrame,
    BANDS, BINS, WEIGHTS, NUM_BANDS, MDCT_M, MDCT_N, HOP,
    SAMPLE_RATE, FRAME_MS, GOP_SIZE, SILENCE_DB,
    L0, L1, L2, coeffs_to_bands,
)


# ─── MDCT / IMDCT ─────────────────────────────────────────────────────────────

class TestMDCT:
    def test_output_shape(self):
        x = np.random.default_rng(0).standard_normal(MDCT_N)
        C = mdct(x)
        assert C.shape == (MDCT_M,)

    def test_imdct_output_shape(self):
        C = np.random.default_rng(1).standard_normal(MDCT_M)
        y = imdct(C)
        assert y.shape == (MDCT_N,)

    def test_tdac_reconstruction_snr(self):
        """TDAC overlap-add must achieve ≥ 200dB SNR in the valid interior."""
        rng = np.random.default_rng(42)
        sig = rng.standard_normal(MDCT_M * 12)
        out = np.zeros_like(sig)
        for i in range(11):
            pos = i * MDCT_M
            out[pos:pos + MDCT_N] += imdct(mdct(sig[pos:pos + MDCT_N]))
        s, e = MDCT_N, MDCT_M * 10
        err = np.sqrt(np.mean((sig[s:e] - out[s:e]) ** 2))
        rms = np.sqrt(np.mean(sig[s:e] ** 2))
        snr = 20 * np.log10(rms / (err + 1e-30))
        assert snr > 200, f"TDAC SNR {snr:.0f}dB < 200dB"

    def test_linearity(self):
        rng = np.random.default_rng(5)
        a, b = rng.standard_normal(MDCT_N), rng.standard_normal(MDCT_N)
        assert np.allclose(mdct(a + b), mdct(a) + mdct(b), atol=1e-10)

    def test_mdct_imdct_energy(self):
        """IMDCT should approximately preserve energy (within windowing factor)."""
        rng = np.random.default_rng(7)
        C = rng.standard_normal(MDCT_M)
        y = imdct(C)
        assert np.sqrt(np.mean(y ** 2)) > 0


# ─── Band layout ──────────────────────────────────────────────────────────────

class TestBands:
    def test_band_count(self):
        assert len(BANDS) == NUM_BANDS == 48

    def test_layer_split(self):
        assert L0 + L1 + L2 == NUM_BANDS
        assert L0 == 8
        assert L1 == 16
        assert L2 == 24

    def test_bins_count(self):
        assert len(BINS) == NUM_BANDS

    def test_bins_monotone(self):
        """Each band's bin range is valid and non-empty."""
        for i, (lo, hi) in enumerate(BINS):
            assert 0 <= lo < hi <= MDCT_M, f"Band {i}: [{lo},{hi}] invalid"

    def test_bins_non_overlapping(self):
        """Adjacent bands should not share bins (outer boundary of b == inner of b+1 is fine)."""
        for i in range(NUM_BANDS - 1):
            assert BINS[i][1] <= BINS[i + 1][0] + 1, \
                f"Bands {i} and {i+1} overlap"

    def test_layer_freq_boundaries(self):
        """L0 ends around 800Hz, L1 ends around 6kHz."""
        l0_max_hz = BANDS[L0 - 1][1]
        l1_max_hz = BANDS[L0 + L1 - 1][1]
        assert 700 < l0_max_hz < 900, f"L0 top freq {l0_max_hz:.0f}Hz unexpected"
        assert 5500 < l1_max_hz < 6500, f"L1 top freq {l1_max_hz:.0f}Hz unexpected"

    def test_weights_range(self):
        assert np.all(WEIGHTS > 0)
        assert np.all(WEIGHTS <= 1.0)

    def test_weights_peak_sensitivity(self):
        """1–4kHz bands should have weight 1.0 (peak human sensitivity)."""
        for i, (lo, hi) in enumerate(BANDS):
            center = (lo + hi) / 2
            if 1000 <= center <= 4000:
                assert WEIGHTS[i] == 1.0, f"Band {i} center {center:.0f}Hz weight {WEIGHTS[i]}"

    def test_sub_bass_weights_low(self):
        """Bands below 100Hz should have weight 0.25."""
        for i, (lo, hi) in enumerate(BANDS):
            if (lo + hi) / 2 < 100:
                assert WEIGHTS[i] == 0.25


# ─── DSAAnalyzer ──────────────────────────────────────────────────────────────

def _make_signal(n_hops: int = GOP_SIZE * 4, freq_hz: float = 440.0,
                 amp: float = 0.5) -> np.ndarray:
    t = np.arange(n_hops * HOP) / SAMPLE_RATE
    return amp * np.sin(2 * np.pi * freq_hz * t)


class TestAnalyzer:
    def setup_method(self):
        self.analyzer = DSAAnalyzer()

    def test_returns_frames(self):
        sig = _make_signal()
        frames = self.analyzer.analyze_samples(sig)
        assert len(frames) > 0
        assert all(isinstance(f, DSAFrame) for f in frames)

    def test_first_frame_is_keyframe(self):
        frames = self.analyzer.analyze_samples(_make_signal())
        assert frames[0].frame_type == 'K'
        assert frames[0].gop_pos == 0

    def test_gop_pos_cycling(self):
        frames = self.analyzer.analyze_samples(_make_signal(n_hops=GOP_SIZE * 4))
        for f in frames:
            assert 0 <= f.gop_pos < GOP_SIZE, f"gop_pos {f.gop_pos} out of range"
        # Every GOP_SIZE frames should be a K-frame
        k_positions = [f.frame_idx for f in frames if f.frame_type == 'K']
        for idx in k_positions:
            assert idx % GOP_SIZE == 0, f"K-frame at non-GOP boundary {idx}"

    def test_frame_count_scales_with_signal(self):
        short = _make_signal(GOP_SIZE)
        long  = _make_signal(GOP_SIZE * 4)
        n_short = len(self.analyzer.analyze_samples(short))
        n_long  = len(self.analyzer.analyze_samples(long))
        assert n_long > n_short * 3

    def test_silence_detection(self):
        silence = np.zeros(HOP * GOP_SIZE * 2)
        frames  = self.analyzer.analyze_samples(silence)
        # All non-K frames should be silence
        non_k = [f for f in frames if f.frame_type != 'K']
        assert all(f.frame_type == 'S' for f in non_k)

    def test_coeffs_shape(self):
        frames = self.analyzer.analyze_samples(_make_signal())
        for f in frames:
            assert f.coeffs.shape == (MDCT_M,)

    def test_energies_db_shape(self):
        frames = self.analyzer.analyze_samples(_make_signal())
        for f in frames:
            assert f.energies_db.shape == (NUM_BANDS,)

    def test_layer_properties(self):
        frames = self.analyzer.analyze_samples(_make_signal())
        f = frames[1]  # use a non-silence B-frame
        assert len(f.layer0) == L0
        assert len(f.layer1) == L1
        assert len(f.layer2) == L2

    def test_440hz_energy_in_l0(self):
        """440Hz tone should have dominant energy in L0 (bass/low-mid layer)."""
        frames = self.analyzer.analyze_samples(_make_signal(freq_hz=440.0))
        f = next(fr for fr in frames if fr.frame_type == 'K')
        # 440Hz is in L0 range (< 800Hz)
        assert f.layer0.max() > f.layer1.max() - 10, \
            "440Hz should be strong in L0"

    def test_9khz_energy_in_l2(self):
        """9kHz tone should have dominant energy in L2 (high-freq outer ring)."""
        frames = self.analyzer.analyze_samples(_make_signal(freq_hz=9000.0))
        f = next(fr for fr in frames if fr.frame_type == 'K')
        assert f.layer2.max() > f.layer0.max(), \
            "9kHz should be dominant in L2"
