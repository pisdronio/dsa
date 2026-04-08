"""
Integration tests: full encode → decode pipeline (Steps 1–8)

Covers:
  - Round-trip SNR at multiple bitrates
  - Layer isolation: L0-only, L0+L1, all layers
  - Reverse playback: temporal order is reversed
  - Analog degradation: α=0.5 → ~6dB, α=0 → silence
  - Bitstream integrity: CRC passes after encode
  - Disc layout: correct structure, steepness/direction ranges
  - Edge cases: pure silence, very short signal, all-K-frame signal
"""

import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from dsa_analyzer import (
    DSAAnalyzer, NUM_BANDS, L0, L1, L2, BINS,
    MDCT_M, MDCT_N, HOP, SAMPLE_RATE, GOP_SIZE,
    mdct, imdct, coeffs_to_bands,
)
from dsa_bitstream import DSABitstreamWriter, DSABitstreamReader
from dsa_decoder import DSADecoder, _alpha_for_layers
from dsa_disc import DSADiscEncoder, BAND_PAIRS, L2_PRE_EMPHASIS, MODE_DISCRETE, MODE_GRADIENT
from dsa_encoder import decode_kframe, EncodedKFrame


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def _tone(freq_hz: float, amp: float = 0.5, n_hops: int = GOP_SIZE * 4) -> np.ndarray:
    t = np.arange(n_hops * HOP) / SAMPLE_RATE
    return amp * np.sin(2 * np.pi * freq_hz * t)


def _three_tone(n_hops: int = GOP_SIZE * 4) -> np.ndarray:
    """Three tones spanning L0, L1, L2."""
    t = np.arange(n_hops * HOP) / SAMPLE_RATE
    return (0.5 * np.sin(2 * np.pi * 440  * t)
          + 0.3 * np.sin(2 * np.pi * 2000 * t)
          + 0.2 * np.sin(2 * np.pi * 9000 * t))


def _encode_decode(sig: np.ndarray, bitrate: float = 12.0,
                   **decode_kwargs) -> tuple:
    """Full pipeline: signal → encode → bitstream → decode → samples."""
    analyzer = DSAAnalyzer()
    frames   = analyzer.analyze_samples(sig)
    raw      = DSABitstreamWriter(bitrate_kbps=bitrate).encode_frames(frames)
    reader   = DSABitstreamReader(raw)
    decoded  = DSADecoder().decode(reader, **decode_kwargs)
    return decoded, raw, reader, frames


# ─── Round-trip quality ───────────────────────────────────────────────────────

class TestRoundTrip:
    @pytest.mark.parametrize("bitrate,min_snr", [
        (6,   5.0),
        (12,  8.0),
        (32, 12.0),
        (96, 18.0),
    ])
    def test_snr_at_bitrate(self, bitrate, min_snr):
        """Round-trip SNR must meet minimum at each bitrate."""
        sig              = _three_tone(n_hops=GOP_SIZE * 6)
        decoded, *_      = _encode_decode(sig, bitrate=bitrate)
        skip             = MDCT_N
        n                = min(len(sig), len(decoded)) - 2 * skip
        err_rms          = np.sqrt(np.mean((sig[skip:skip+n] - decoded[skip:skip+n]) ** 2))
        sig_rms          = np.sqrt(np.mean(sig[skip:skip+n] ** 2))
        snr              = 20 * np.log10(sig_rms / (err_rms + 1e-12))
        assert snr >= min_snr, f"SNR {snr:.1f}dB < {min_snr}dB at {bitrate}kbps"

    def test_output_length(self):
        sig             = _three_tone()
        decoded, _, _, frames = _encode_decode(sig)
        expected        = (len(frames) - 1) * HOP
        assert abs(len(decoded) - expected) <= HOP

    def test_output_finite(self):
        sig             = _three_tone()
        decoded, *_     = _encode_decode(sig)
        assert np.all(np.isfinite(decoded))

    def test_output_in_range(self):
        """Decoded audio should stay within a reasonable amplitude range."""
        sig             = _three_tone()
        decoded, *_     = _encode_decode(sig)
        assert np.max(np.abs(decoded)) < 5.0   # not wildly clipped


# ─── Bitstream integrity ──────────────────────────────────────────────────────

class TestBitstream:
    def test_crc_valid(self):
        sig   = _three_tone()
        _, raw, reader, _ = _encode_decode(sig)
        assert reader.verify_crc(), "CRC check failed after encode"

    def test_header_fields(self):
        sig    = _three_tone()
        _, raw, reader, frames = _encode_decode(sig, bitrate=12.0)
        hdr    = reader.header
        assert hdr.sample_rate   == SAMPLE_RATE
        assert hdr.n_frames      == len(frames)
        assert hdr.bitrate_kbps  == 12

    def test_layer_byte_sizes(self):
        sig     = _three_tone()
        _, raw, reader, _ = _encode_decode(sig)
        l0_size = reader.layer_byte_size(0)
        l1_size = reader.layer_byte_size(1)
        l2_size = reader.layer_byte_size(2)
        assert l0_size > 0
        assert l1_size > 0
        assert l2_size > 0

    def test_l0_smallest_layer(self):
        """L0 (8 bands) should occupy fewer bytes than L1 (16 bands)."""
        _, raw, reader, _ = _encode_decode(_three_tone())
        assert reader.layer_byte_size(0) < reader.layer_byte_size(1)


# ─── Layer isolation ──────────────────────────────────────────────────────────

class TestLayerIsolation:
    def _encode(self, sig, bitrate=12.0):
        frames = DSAAnalyzer().analyze_samples(sig)
        return DSABitstreamWriter(bitrate_kbps=bitrate).encode_frames(frames)

    def test_l0_only_silences_high_freq(self):
        """L0-only decode should have near-zero energy in the 9kHz band."""
        sig    = _three_tone()
        raw    = self._encode(sig)
        reader = DSABitstreamReader(raw)
        dec    = DSADecoder()

        l0_decoded = dec.decode(reader, layers=(0,))
        skip       = MDCT_N
        segment    = l0_decoded[skip:skip + MDCT_N]
        _, rms     = coeffs_to_bands(mdct(segment))

        l2_rms = float(np.mean(rms[L0 + L1:]))
        assert l2_rms < 1e-6, f"L2 energy {l2_rms:.2e} not suppressed in L0-only"

    def test_l0_only_preserves_bass(self):
        """L0-only decode should still carry 440Hz energy."""
        sig    = _three_tone()
        raw    = self._encode(sig)
        reader = DSABitstreamReader(raw)

        l0_decoded = DSADecoder().decode(reader, layers=(0,))
        skip       = MDCT_N
        segment    = l0_decoded[skip:skip + MDCT_N]
        _, rms     = coeffs_to_bands(mdct(segment))

        l0_rms = float(np.mean(rms[:L0]))
        assert l0_rms > 1e-3, f"L0 energy {l0_rms:.4f} too low in L0-only decode"

    def test_all_layers_louder_than_l0_only(self):
        """Full decode should have more energy than L0-only."""
        sig    = _three_tone()
        raw    = self._encode(sig, bitrate=96.0)  # high bitrate for real L2 content
        reader = DSABitstreamReader(raw)
        dec    = DSADecoder()

        full_rms = np.sqrt(np.mean(dec.decode(reader) ** 2))
        l0_rms   = np.sqrt(np.mean(dec.decode(reader, layers=(0,)) ** 2))
        assert full_rms > l0_rms, "Full decode should be louder than L0 only"


# ─── Analog degradation ───────────────────────────────────────────────────────

class TestAnalogDegradation:
    def _reader(self):
        sig    = _three_tone()
        frames = DSAAnalyzer().analyze_samples(sig)
        raw    = DSABitstreamWriter(bitrate_kbps=12).encode_frames(frames)
        return DSABitstreamReader(raw)

    def test_alpha_zero_is_silence(self):
        reader = self._reader()
        silent = DSADecoder().decode(reader, alpha=np.zeros(NUM_BANDS))
        assert np.max(np.abs(silent)) < 1e-9

    def test_alpha_half_is_6db_down(self):
        reader   = self._reader()
        dec      = DSADecoder()
        full     = dec.decode(reader)
        half     = dec.decode(reader, alpha=np.full(NUM_BANDS, 0.5))
        skip     = MDCT_N
        rms_full = np.sqrt(np.mean(full[skip:] ** 2))
        rms_half = np.sqrt(np.mean(half[skip:] ** 2))
        ratio    = rms_half / (rms_full + 1e-12)
        assert 0.45 < ratio < 0.55, f"α=0.5 ratio {ratio:.3f} not near 0.5"

    def test_worn_disc_still_plays(self):
        reader = self._reader()
        worn   = DSADecoder().decode(reader, alpha=np.full(NUM_BANDS, 0.1))
        assert np.sqrt(np.mean(worn[MDCT_N:] ** 2)) > 1e-5

    def test_alpha_spectrally_selective(self):
        """α=0 on L2 only should silence high freq without affecting bass."""
        reader  = self._reader()
        alpha   = _alpha_for_layers((0, 1))    # L2 = 0
        decoded = DSADecoder().decode(reader, alpha=alpha)
        skip    = MDCT_N
        seg     = decoded[skip:skip + MDCT_N]
        _, rms  = coeffs_to_bands(mdct(seg))

        l0_rms = float(np.mean(rms[:L0]))
        l2_rms = float(np.mean(rms[L0 + L1:]))
        assert l0_rms  > 1e-4, "Bass should survive L2-muted decode"
        assert l2_rms  < 1e-6, "L2 should be silent when α[L2]=0"


# ─── Reverse playback ─────────────────────────────────────────────────────────

class TestReversePlayback:
    def _reader_from_ramp(self):
        """Loud first half, silent second half — exposes temporal reversal."""
        n      = MDCT_M * (GOP_SIZE * 4)
        half   = n // 2
        sig    = np.zeros(n)
        t      = np.arange(half) / SAMPLE_RATE
        sig[:half] = 0.5 * np.sin(2 * np.pi * 440 * t)
        frames = DSAAnalyzer().analyze_samples(sig)
        raw    = DSABitstreamWriter(bitrate_kbps=96).encode_frames(frames)
        return DSABitstreamReader(raw)

    def test_reverse_flips_temporal_energy(self):
        reader = self._reader_from_ramp()
        dec    = DSADecoder()
        fwd    = dec.decode(reader)
        rev    = dec.decode(reader, reverse=True)
        n      = min(len(fwd), len(rev))
        half   = n // 2

        fwd_first  = np.sqrt(np.mean(fwd[:half]  ** 2))
        fwd_second = np.sqrt(np.mean(fwd[half:]  ** 2))
        rev_first  = np.sqrt(np.mean(rev[:half]  ** 2))
        rev_second = np.sqrt(np.mean(rev[half:]  ** 2))

        assert fwd_first  > fwd_second * 3, "Forward: loud half should be first"
        assert rev_second > rev_first  * 3, "Reverse: loud half should be second"

    def test_reverse_same_length(self):
        sig    = _three_tone()
        frames = DSAAnalyzer().analyze_samples(sig)
        raw    = DSABitstreamWriter(bitrate_kbps=12).encode_frames(frames)
        reader = DSABitstreamReader(raw)
        dec    = DSADecoder()
        fwd    = dec.decode(reader)
        rev    = dec.decode(reader, reverse=True)
        assert len(fwd) == len(rev)

    def test_reverse_not_silence(self):
        sig    = _three_tone()
        frames = DSAAnalyzer().analyze_samples(sig)
        raw    = DSABitstreamWriter(bitrate_kbps=12).encode_frames(frames)
        reader = DSABitstreamReader(raw)
        rev    = DSADecoder().decode(reader, reverse=True)
        assert np.sqrt(np.mean(rev ** 2)) > 1e-4

    def test_negative_speed_equals_reverse(self):
        """speed=-1.0 should produce same output as reverse=True."""
        sig    = _three_tone()
        frames = DSAAnalyzer().analyze_samples(sig)
        raw    = DSABitstreamWriter(bitrate_kbps=12).encode_frames(frames)
        dec    = DSADecoder()
        rev    = dec.decode(DSABitstreamReader(raw), reverse=True)
        neg    = dec.decode(DSABitstreamReader(raw), speed=-1.0)
        assert len(rev) == len(neg)
        assert np.allclose(rev, neg)


# ─── Variable-speed playback ───────────────────────────────────────────────────

class TestVariableSpeed:
    def _encode(self, n_hops=GOP_SIZE * 4, bitrate=12.0):
        sig    = _three_tone(n_hops=n_hops)
        frames = DSAAnalyzer().analyze_samples(sig)
        raw    = DSABitstreamWriter(bitrate_kbps=bitrate).encode_frames(frames)
        return raw

    def _nominal_len(self, raw):
        reader = DSABitstreamReader(raw)
        return len(DSADecoder().decode(reader))

    def test_speed_1_passthrough(self):
        raw = self._encode()
        n1  = self._nominal_len(raw)
        n2  = len(DSADecoder().decode(DSABitstreamReader(raw), speed=1.0))
        assert n1 == n2

    def test_double_speed_half_duration(self):
        raw  = self._encode()
        n1   = self._nominal_len(raw)
        n2x  = len(DSADecoder().decode(DSABitstreamReader(raw), speed=2.0))
        assert abs(n2x - n1 // 2) <= 2, f"2× speed: expected ~{n1//2}, got {n2x}"

    def test_half_speed_double_duration(self):
        raw  = self._encode()
        n1   = self._nominal_len(raw)
        nhf  = len(DSADecoder().decode(DSABitstreamReader(raw), speed=0.5))
        assert abs(nhf - n1 * 2) <= 2, f"0.5× speed: expected ~{n1*2}, got {nhf}"

    def test_stopped_same_length_decay(self):
        """speed=0 returns same number of samples but amplitude decays."""
        raw  = self._encode()
        n1   = self._nominal_len(raw)
        stop = DSADecoder().decode(DSABitstreamReader(raw), speed=0.0)
        assert len(stop) == n1
        # Energy in first quarter should exceed last quarter (exponential decay)
        q    = n1 // 4
        e_first = np.sqrt(np.mean(stop[:q] ** 2))
        e_last  = np.sqrt(np.mean(stop[-q:] ** 2))
        assert e_first > e_last * 2, "Stopped: energy should decay over time"

    def test_output_finite(self):
        for speed in [0.25, 0.5, 1.0, 1.5, 2.0, 4.0]:
            raw = self._encode()
            out = DSADecoder().decode(DSABitstreamReader(raw), speed=speed)
            assert np.all(np.isfinite(out)), f"Non-finite output at speed={speed}"

    def test_output_not_silence(self):
        for speed in [0.5, 1.0, 2.0]:
            raw = self._encode()
            out = DSADecoder().decode(DSABitstreamReader(raw), speed=speed)
            assert np.sqrt(np.mean(out ** 2)) > 1e-4, f"Silent output at speed={speed}"


# ─── Edge cases ───────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_pure_silence(self):
        """All-zero signal should encode/decode without error."""
        sig    = np.zeros(HOP * GOP_SIZE * 4)
        frames = DSAAnalyzer().analyze_samples(sig)
        raw    = DSABitstreamWriter(bitrate_kbps=12).encode_frames(frames)
        reader = DSABitstreamReader(raw)
        dec    = DSADecoder().decode(reader)
        assert np.max(np.abs(dec)) < 1e-6

    def test_single_gop(self):
        """Minimum-length signal (one GOP) encodes/decodes without error."""
        sig    = _tone(440.0, n_hops=GOP_SIZE)
        frames = DSAAnalyzer().analyze_samples(sig)
        raw    = DSABitstreamWriter(bitrate_kbps=12).encode_frames(frames)
        reader = DSABitstreamReader(raw)
        dec    = DSADecoder().decode(reader)
        assert len(dec) > 0
        assert np.all(np.isfinite(dec))

    def test_high_amplitude_signal(self):
        """Signal near clipping should encode/decode without NaN/Inf."""
        sig    = _tone(440.0, amp=0.99, n_hops=GOP_SIZE * 4)
        frames = DSAAnalyzer().analyze_samples(sig)
        raw    = DSABitstreamWriter(bitrate_kbps=12).encode_frames(frames)
        reader = DSABitstreamReader(raw)
        dec    = DSADecoder().decode(reader)
        assert np.all(np.isfinite(dec))

    def test_multiple_bitrates_produce_different_sizes(self):
        sig    = _three_tone()
        frames = DSAAnalyzer().analyze_samples(sig)
        raw_lo = DSABitstreamWriter(bitrate_kbps=6 ).encode_frames(frames)
        raw_hi = DSABitstreamWriter(bitrate_kbps=96).encode_frames(frames)
        assert len(raw_hi) > len(raw_lo)


# ─── Disc layout ──────────────────────────────────────────────────────────────

class TestDiscLayout:
    def _layout(self, bitrate=12.0, layers=(0, 1, 2)):
        sig    = _three_tone()
        frames = DSAAnalyzer().analyze_samples(sig)
        raw    = DSABitstreamWriter(bitrate_kbps=bitrate).encode_frames(frames)
        reader = DSABitstreamReader(raw)
        return DSADiscEncoder().encode(reader, layers=layers)

    def test_frame_count_matches_codec(self):
        sig    = _three_tone()
        frames = DSAAnalyzer().analyze_samples(sig)
        raw    = DSABitstreamWriter(bitrate_kbps=12).encode_frames(frames)
        reader = DSABitstreamReader(raw)
        layout = DSADiscEncoder().encode(reader)
        assert layout.n_frames == len(frames)

    def test_steepness_in_range(self):
        layout = self._layout()
        steeps = [d.steepness for f in layout.frames for d in f.dots]
        assert len(steeps) > 0
        assert all(0.0 <= s <= 1.0 for s in steeps)

    def test_direction_values(self):
        layout = self._layout()
        dirs   = {d.direction for f in layout.frames for d in f.dots}
        assert dirs.issubset({-1, 0, 1})

    def test_silence_frames_empty(self):
        layout = self._layout()
        for f in layout.frames:
            if f.is_silence:
                assert len(f.dots) == 0

    def test_k_frames_cover_all_bands(self):
        layout    = self._layout()
        k_frames  = [f for f in layout.frames if f.frame_type == 'K']
        assert len(k_frames) > 0
        all_bands = {d.band_idx for kf in k_frames for d in kf.dots}
        assert all_bands == set(range(NUM_BANDS))

    def test_l0_only_no_l1_l2_dots(self):
        layout = self._layout(layers=(0,))
        for f in layout.frames:
            for d in f.dots:
                assert d.layer == 0

    def test_band_pairs_assigned(self):
        layout = self._layout()
        for f in layout.frames:
            for d in f.dots:
                expected = BAND_PAIRS[d.band_idx]
                assert (d.color_a, d.color_b) == expected

    def test_l2_pre_emphasis_constant(self):
        """L2 pre-emphasis factor should be > 1.0."""
        assert L2_PRE_EMPHASIS > 1.0

    def test_json_serializable(self):
        layout = self._layout()
        j      = layout.to_json()
        obj    = json.loads(j)
        assert obj['n_frames']    == layout.n_frames
        assert obj['n_bands']     == NUM_BANDS
        assert len(obj['frames']) == layout.n_frames

    def test_band_summary_length(self):
        layout = self._layout()
        f      = next(fr for fr in layout.frames if not fr.is_silence)
        summ   = f.band_summary()
        assert len(summ) == NUM_BANDS

    def test_layer_dot_counts(self):
        layout        = self._layout()
        l0c, l1c, l2c = layout.layer_dot_counts()
        assert l0c > 0
        assert l1c > 0
        assert l2c > 0
        # L2 has most bands (24) so most dots
        assert l2c > l0c


class TestMode2GradientEncoding:
    """Mode 2: continuous steepness encoding via per-band normalization."""

    def _layout_m1_m2(self, layers=(0, 1, 2)):
        sig    = _three_tone(n_hops=GOP_SIZE * 6)
        frames = DSAAnalyzer().analyze_samples(sig)
        raw_m1 = DSABitstreamWriter(bitrate_kbps=32, mode=MODE_DISCRETE).encode_frames(frames)
        raw_m2 = DSABitstreamWriter(bitrate_kbps=32, mode=MODE_GRADIENT).encode_frames(frames)

        l1 = DSADiscEncoder().encode(DSABitstreamReader(raw_m1), layers=layers)
        l2 = DSADiscEncoder().encode(DSABitstreamReader(raw_m2), layers=layers)
        return l1, l2

    def test_mode2_steepness_in_range(self):
        _, l2 = self._layout_m1_m2()
        steeps = [d.steepness for f in l2.frames for d in f.dots]
        assert all(0.0 <= s <= 1.0 for s in steeps)

    def test_mode2_directions_valid(self):
        _, l2 = self._layout_m1_m2()
        dirs = {d.direction for f in l2.frames for d in f.dots}
        assert dirs.issubset({-1, 0, 1})

    def test_mode2_k_frame_steepness_continuous(self):
        """Mode 2 K-frame steepness should not be restricted to multiples of 1/31."""
        l1, l2 = self._layout_m1_m2()
        # Collect K-frame steepness values (non-zero, non-trivial)
        def k_steeps(layout):
            return [d.steepness for f in layout.frames
                    if f.frame_type == 'K' for d in f.dots
                    if 0 < d.steepness < 0.99]

        s1 = k_steeps(l1)
        s2 = k_steeps(l2)
        assert len(s1) > 0 and len(s2) > 0

        # Mode 1 K-frame steepness = q_int/31 — all values are exact multiples of 1/31
        for s in s1:
            nearest = round(s * 31) / 31
            assert abs(s - nearest) < 1e-6 or abs(s * 31 / 1.15 - round(s * 31 / 1.15)) < 1e-5, (
                f"Mode 1 K-frame steepness {s:.8f} should be q_int/31 (possibly pre-emphasized)"
            )

        # Mode 2 K-frame steepness should have values NOT restricted to multiples of 1/31
        non_quantized = sum(
            1 for s in s2
            if abs(s * 31 - round(s * 31)) > 0.01
            and abs(s * 31 / 1.15 - round(s * 31 / 1.15)) > 0.01
        )
        assert non_quantized > 0, "Mode 2 K-frame steepness should have continuous (non-quantized) values"

    def test_mode2_full_range_utilization(self):
        """Mode 2 per-band normalization should make each band use the full [0,1] steepness range."""
        _, l2 = self._layout_m1_m2()
        k_frames = [f for f in l2.frames if f.frame_type == 'K']
        assert len(k_frames) > 0
        # Collect max steepness per band across all K-frames
        band_max = {}
        for f in k_frames:
            for d in f.dots:
                if d.band_idx not in band_max or d.steepness > band_max[d.band_idx]:
                    band_max[d.band_idx] = d.steepness
        # For bands with content, at least some bands should reach near 1.0
        maxima = [v for v in band_max.values() if v > 0]
        assert len(maxima) > 0
        assert max(maxima) > 0.8, f"Mode 2 max steepness {max(maxima):.3f} should be close to 1.0"

    def test_mode2_frame_count_same_as_mode1(self):
        """Both modes should produce the same number of disc frames."""
        l1, l2 = self._layout_m1_m2()
        assert l1.n_frames == l2.n_frames

    def test_mode2_json_serializable(self):
        _, l2 = self._layout_m1_m2()
        j   = l2.to_json()
        obj = json.loads(j)
        assert obj['n_frames'] == l2.n_frames
        assert len(obj['frames']) == l2.n_frames

    def test_mode2_silence_frames_empty(self):
        _, l2 = self._layout_m1_m2()
        for f in l2.frames:
            if f.is_silence:
                assert len(f.dots) == 0
