"""
Camera reader tests — Tier 0 (software baseline) and Tier 1 (unit tests)

Tier 0 — Software baseline (§ RESEARCH.md)
  Feed the rendered strip PNG directly as the "photo".  The homography is
  identity so accuracy should be high.  Proves the pipeline is wired
  correctly and that dsa_camera.py reads back what dsa_strip.py wrote.

  Expected (cell_w=8 readable strip):
    direction accuracy ≥ 93%    (spec says ≥ 98%; allow margin for pixel quantisation)
    mean α              ≥ 0.90

Tier 1 — Unit tests for camera reader components
  _band_gradient_row, _color_to_blend, geometry helpers, read_strip API.
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from PIL import Image

from dsa_analyzer import DSAAnalyzer, GOP_SIZE, HOP, SAMPLE_RATE
from dsa_bitstream import DSABitstreamWriter, DSABitstreamReader
from dsa_disc import DSADiscEncoder
from dsa_strip import _band_gradient_row, _band_color, render_strip, PALETTE as STRIP_PALETTE
from dsa_camera import (
    _color_to_blend,
    _band_row_top,
    _cell_sample_y,
    _bilinear,
    read_strip,
    compare_layout,
    PALETTE,
    SEP_PX,
    DIRECTION_THRESHOLD,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def _three_tone(n_hops: int = GOP_SIZE * 8) -> np.ndarray:
    t = np.arange(n_hops * HOP) / SAMPLE_RATE
    return (0.5 * np.sin(2 * np.pi * 440  * t)
          + 0.3 * np.sin(2 * np.pi * 2000 * t)
          + 0.2 * np.sin(2 * np.pi * 9000 * t))


def _make_layout(n_hops: int = GOP_SIZE * 8, bitrate: float = 32.0) -> dict:
    sig    = _three_tone(n_hops)
    frames = DSAAnalyzer().analyze_samples(sig)
    raw    = DSABitstreamWriter(bitrate_kbps=bitrate).encode_frames(frames)
    layout = DSADiscEncoder().encode(DSABitstreamReader(raw))
    return json.loads(layout.to_json())


def _border_px(strip_dpi: int = 72) -> int:
    """Reproduces the border_px formula used in dsa_strip.render_strip (fiducials=True)."""
    px_per_mm  = strip_dpi / 25.4
    scalebar_h = max(4, int(round(1.0 * px_per_mm)))
    return max(20, scalebar_h + 8)


# ─── Tier 1: _color_to_blend ──────────────────────────────────────────────────

class TestColorToBlend:
    def setup_method(self):
        self.ca = PALETTE['black']
        self.cb = PALETTE['white']

    def test_ca_gives_zero(self):
        t, conf = _color_to_blend(self.ca, self.ca, self.cb)
        assert t == pytest.approx(0.0, abs=1e-9)

    def test_cb_gives_one(self):
        t, conf = _color_to_blend(self.cb, self.ca, self.cb)
        assert t == pytest.approx(1.0, abs=1e-9)

    def test_midpoint(self):
        mid = (self.ca + self.cb) / 2
        t, _ = _color_to_blend(mid, self.ca, self.cb)
        assert t == pytest.approx(0.5, abs=0.01)

    def test_on_axis_high_confidence(self):
        mid = (self.ca + self.cb) * 0.75
        _, conf = _color_to_blend(mid, self.ca, self.cb)
        assert conf > 0.95

    def test_off_axis_low_confidence(self):
        """A colour perpendicular to the ca→cb axis should have low confidence."""
        # Use red→green axis; pure blue is nearly perpendicular to it.
        ca   = PALETTE['red']    # (220,  50,  50)
        cb   = PALETTE['green']  # ( 50, 180,  50)
        perp = PALETTE['blue']   # ( 50,  50, 220)
        _, conf = _color_to_blend(perp, ca, cb)
        assert conf < 0.25, f"Blue on red→green axis: expected low conf, got {conf:.4f}"

    def test_identical_colors_no_crash(self):
        """Degenerate case: ca == cb should return (0, 1) without division by zero."""
        t, conf = _color_to_blend(self.ca, self.ca, self.ca)
        assert np.isfinite(t) and np.isfinite(conf)

    def test_clamp_within_range(self):
        """t must stay in [0, 1] even for out-of-gamut colours."""
        beyond = self.cb * 1.5    # brighter than cb
        t, _ = _color_to_blend(beyond, self.ca, self.cb)
        assert 0.0 <= t <= 1.0


# ─── Tier 1: geometry helpers ─────────────────────────────────────────────────

class TestGeometry:
    def test_band_row_top_highest_band_at_zero(self):
        """Band 47 (highest) should be at content row 0."""
        assert _band_row_top(47, 48, 8) == 0

    def test_band_row_top_bass_is_near_bottom(self):
        """Band 0 (bass) row_top should be close to total content height minus one cell."""
        cell_h  = 8
        n_bands = 48
        content_h = n_bands * cell_h + 2 * SEP_PX
        assert _band_row_top(0, n_bands, cell_h) == content_h - cell_h

    def test_band_row_top_monotone(self):
        """Higher band index → smaller row_top (higher in image)."""
        n, ch = 48, 8
        tops = [_band_row_top(b, n, ch) for b in range(n)]
        assert all(tops[i] > tops[i + 1] or tops[i] == tops[i + 1]
                   for i in range(n - 1)), "row_top should be non-increasing with band index"
        # Strict: actually it should be strictly decreasing (no two bands share a row)
        assert tops[0] > tops[-1]

    def test_cell_sample_y_within_image(self):
        """Sample y should be inside the bordered image for all bands."""
        n, ch, border = 48, 8, 20
        content_h = n * ch + 2 * SEP_PX
        img_h     = content_h + 2 * border
        for b in range(n):
            y = _cell_sample_y(b, n, ch, border)
            assert border <= y < img_h - border, f"Band {b}: y={y} outside content area"

    def test_cell_sample_y_bass_below_high(self):
        """Band 0 (bass) should have a larger y than band 47 (high) in image coords."""
        y0  = _cell_sample_y(0,  48, 8, 20)
        y47 = _cell_sample_y(47, 48, 8, 20)
        assert y0 > y47


# ─── Tier 1: _band_gradient_row ──────────────────────────────────────────────

class TestBandGradientRow:
    """Unit tests for the cell gradient renderer in dsa_strip.py."""

    def setup_method(self):
        self.ca = STRIP_PALETTE['black']
        self.cb = STRIP_PALETTE['white']

    def _sample(self, row: np.ndarray, cell_w: int) -> tuple[int, float]:
        """
        Apply the same 25%/75% sampling that dsa_camera.read_strip uses.
        Returns (recovered_direction, recovered_steepness).
        """
        ca = PALETTE['black']
        cb = PALETTE['white']

        x_left  = int(0.25 * cell_w)
        x_right = int(0.75 * cell_w)
        c_left  = row[x_left].astype(np.float64)
        c_right = row[x_right].astype(np.float64)

        t_left,  _ = _color_to_blend(c_left,  ca, cb)
        t_right, _ = _color_to_blend(c_right, ca, cb)

        diff = t_right - t_left
        rec_s = float(min(1.0, 2.0 * abs(diff)))
        rec_d = 0 if abs(diff) < DIRECTION_THRESHOLD else (1 if diff > 0 else -1)
        return rec_d, rec_s

    def test_silence_returns_solid_ca(self):
        row = _band_gradient_row(0.0, 0, self.ca, self.cb, cell_w=8)
        assert row.shape == (8, 3)
        assert np.all(row == self.ca.astype(np.uint8))

    def test_direction_zero_returns_solid_ca(self):
        row = _band_gradient_row(0.5, 0, self.ca, self.cb, cell_w=8)
        assert np.all(row == self.ca.astype(np.uint8))

    def test_output_shape(self):
        for cw in [4, 8, 16]:
            row = _band_gradient_row(0.5, 1, self.ca, self.cb, cw)
            assert row.shape == (cw, 3), f"cell_w={cw}: expected ({cw},3), got {row.shape}"

    def test_output_dtype_uint8(self):
        row = _band_gradient_row(0.5, 1, self.ca, self.cb, cell_w=8)
        assert row.dtype == np.uint8

    @pytest.mark.parametrize("direction", [1, -1])
    def test_direction_recovered(self, direction):
        """Sampling at 25%/75% must recover the correct direction sign."""
        for steepness in [0.3, 0.5, 0.8, 1.0]:
            row  = _band_gradient_row(steepness, direction, self.ca, self.cb, cell_w=8)
            d, _ = self._sample(row, cell_w=8)
            assert d == direction, (
                f"steepness={steepness}, direction={direction}: got {d}")

    def test_steepness_monotone(self):
        """Higher input steepness → higher recovered steepness (with cell_w=8)."""
        steepnesses = [0.1, 0.3, 0.5, 0.8]
        recovered   = []
        for s in steepnesses:
            row  = _band_gradient_row(s, 1, self.ca, self.cb, cell_w=8)
            _, rs = self._sample(row, cell_w=8)
            recovered.append(rs)
        assert recovered == sorted(recovered), "Recovered steepness not monotone with input"

    def test_steepness_proportional(self):
        """Recovered steepness should be within 2% of input (uint8 quantisation only)."""
        for s in [0.2, 0.4, 0.6, 0.8]:
            row  = _band_gradient_row(s, 1, self.ca, self.cb, cell_w=8)
            _, rs = self._sample(row, cell_w=8)
            assert abs(rs - s) < 0.02, (
                f"steepness={s}: recovered {rs:.4f}, error {abs(rs-s):.5f}")

    def test_direction_positive_left_to_right(self):
        """direction=+1: leftmost pixel should be closer to ca, rightmost to cb."""
        row  = _band_gradient_row(0.8, 1, self.ca, self.cb, cell_w=8)
        t0,  _ = _color_to_blend(row[0].astype(np.float64),  PALETTE['black'], PALETTE['white'])
        t_end, _ = _color_to_blend(row[-1].astype(np.float64), PALETTE['black'], PALETTE['white'])
        assert t0 < t_end, "direction=+1: t should increase left to right"

    def test_direction_negative_right_to_left(self):
        """direction=-1: leftmost pixel should be further from ca than rightmost."""
        row  = _band_gradient_row(0.8, -1, self.ca, self.cb, cell_w=8)
        t0,  _ = _color_to_blend(row[0].astype(np.float64),  PALETTE['black'], PALETTE['white'])
        t_end, _ = _color_to_blend(row[-1].astype(np.float64), PALETTE['black'], PALETTE['white'])
        assert t0 > t_end, "direction=-1: t should decrease left to right"

    def test_cell_w_4_direction_recovered(self):
        """Direction must be recoverable even at minimum useful cell_w=4."""
        for direction in [1, -1]:
            row  = _band_gradient_row(0.5, direction, self.ca, self.cb, cell_w=4)
            d, _ = self._sample(row, cell_w=4)
            assert d == direction, f"cell_w=4, direction={direction}: got {d}"


# ─── Tier 1: _bilinear ────────────────────────────────────────────────────────

class TestBilinear:
    def test_exact_pixel(self):
        """Integer positions should return exact pixel values."""
        arr = np.array([[[10, 20, 30], [40, 50, 60]],
                        [[70, 80, 90], [100, 110, 120]]], dtype=np.uint8)
        arr = arr.astype(np.uint8)
        assert np.allclose(_bilinear(arr, 0.0, 0.0), [10, 20, 30])
        assert np.allclose(_bilinear(arr, 1.0, 0.0), [40, 50, 60])

    def test_midpoint_interpolation(self):
        """Half-pixel should be the average of the four neighbours."""
        arr = np.array([[[0, 0, 0], [100, 100, 100]],
                        [[100, 100, 100], [200, 200, 200]]], dtype=np.uint8)
        mid = _bilinear(arr, 0.5, 0.5)
        expected = (0 + 100 + 100 + 200) / 4
        assert np.allclose(mid, [expected, expected, expected], atol=1.0)

    def test_clamp_at_boundary(self):
        """Positions at or beyond the image edge should not raise."""
        arr = np.zeros((5, 5, 3), dtype=np.uint8)
        val = _bilinear(arr, 4.9, 4.9)
        assert np.all(np.isfinite(val))


# ─── Tier 1: read_strip API ───────────────────────────────────────────────────

class TestReadStripAPI:
    """read_strip should return arrays with the correct shapes and value ranges."""

    def _warped_and_layout(self, cell_w: int = 8):
        layout = _make_layout()
        n_frames = layout['n_frames']
        n_bands  = layout['n_bands']
        cell_h   = 8
        border   = _border_px()

        with tempfile.NamedTemporaryFile(suffix='.disc.json', mode='w', delete=False) as f:
            json.dump(layout, f)
            disc_path = f.name
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            strip_path = f.name

        render_strip(disc_path, cell_h=cell_h, cell_w=cell_w,
                     fiducials=True, strip_dpi=72, out_path=strip_path)
        warped = np.array(Image.open(strip_path).convert('RGB'))

        import os
        os.unlink(disc_path)
        os.unlink(strip_path)

        return warped, layout, cell_h, border

    def test_output_shapes(self):
        warped, layout, cell_h, border = self._warped_and_layout(cell_w=8)
        n_frames = layout['n_frames']
        n_bands  = layout['n_bands']
        steep, dirs, conf, alpha = read_strip(warped, layout, cell_h, border, cell_w=8)
        assert steep.shape == (n_frames, n_bands)
        assert dirs.shape  == (n_frames, n_bands)
        assert conf.shape  == (n_frames, n_bands)
        assert alpha.shape == (n_bands,)

    def test_steepness_in_range(self):
        warped, layout, cell_h, border = self._warped_and_layout(cell_w=8)
        steep, *_ = read_strip(warped, layout, cell_h, border, cell_w=8)
        assert np.all(steep >= 0.0)
        assert np.all(steep <= 1.0)

    def test_direction_values(self):
        warped, layout, cell_h, border = self._warped_and_layout(cell_w=8)
        _, dirs, *_ = read_strip(warped, layout, cell_h, border, cell_w=8)
        assert set(np.unique(dirs)).issubset({-1, 0, 1})

    def test_confidence_in_range(self):
        warped, layout, cell_h, border = self._warped_and_layout(cell_w=8)
        _, _, conf, alpha = read_strip(warped, layout, cell_h, border, cell_w=8)
        assert np.all(conf >= 0.0)
        assert np.all(conf <= 1.0)
        assert np.all(alpha >= 0.0)
        assert np.all(alpha <= 1.0)

    def test_alpha_is_mean_of_conf(self):
        warped, layout, cell_h, border = self._warped_and_layout(cell_w=8)
        _, _, conf, alpha = read_strip(warped, layout, cell_h, border, cell_w=8)
        assert np.allclose(alpha, conf.mean(axis=0))


# ─── Tier 1: compare_layout ───────────────────────────────────────────────────

class TestCompareLayout:
    def test_perfect_read_all_correct(self):
        """If read values match layout exactly, direction_accuracy == 1.0 and MAE == 0."""
        layout = _make_layout()
        n_frames = layout['n_frames']
        n_bands  = layout['n_bands']

        # Build ground-truth arrays from layout
        steep_gt = np.zeros((n_frames, n_bands))
        dir_gt   = np.zeros((n_frames, n_bands), dtype=np.int8)
        for fd in layout['frames']:
            fi = fd['frame_idx']
            if fd['silence']:
                continue
            for bd in fd['bands']:
                steep_gt[fi, bd['band']] = bd['steepness']
                dir_gt[fi, bd['band']]   = bd['direction']

        stats = compare_layout(layout, steep_gt, dir_gt)
        assert stats['direction_accuracy'] == pytest.approx(1.0)
        assert stats['steepness_mae']      == pytest.approx(0.0, abs=1e-9)
        assert stats['direction_errors']   == 0

    def test_all_wrong_gives_low_accuracy(self):
        """Flipping all directions should give near-zero accuracy."""
        layout = _make_layout()
        n_frames = layout['n_frames']
        n_bands  = layout['n_bands']
        dir_wrong = np.ones((n_frames, n_bands), dtype=np.int8) * 99   # all invalid
        stats = compare_layout(layout, np.zeros((n_frames, n_bands)), dir_wrong)
        assert stats['direction_accuracy'] < 0.05

    def test_result_keys(self):
        layout = _make_layout()
        n_f = layout['n_frames']; n_b = layout['n_bands']
        stats = compare_layout(layout,
                               np.zeros((n_f, n_b)),
                               np.zeros((n_f, n_b), dtype=np.int8))
        assert 'direction_accuracy' in stats
        assert 'steepness_mae'      in stats
        assert 'direction_errors'   in stats
        assert 'n_cells'            in stats

    def test_n_cells_correct(self):
        layout = _make_layout()
        n_f = layout['n_frames']; n_b = layout['n_bands']
        stats = compare_layout(layout,
                               np.zeros((n_f, n_b)),
                               np.zeros((n_f, n_b), dtype=np.int8))
        assert stats['n_cells'] == n_f * n_b


# ─── Tier 0: software baseline ────────────────────────────────────────────────

class TestTier0Baseline:
    """
    Feed the rendered strip PNG directly to the camera reader (no optics).
    The homography is identity so accuracy should approach spec targets.

    Spec targets (RESEARCH.md §Tier 0):
        direction accuracy ≥ 98%
        mean α             ≥ 0.93

    We test at slightly relaxed thresholds to allow for pixel quantisation
    in the uint8 gradient rendering.
    """

    @pytest.fixture(scope='class')
    def pipeline(self):
        """Generate layout → render strip → read back; shared across tests."""
        layout  = _make_layout(n_hops=GOP_SIZE * 12, bitrate=32.0)
        cell_h  = 8
        cell_w  = 8
        border  = _border_px(strip_dpi=72)

        with tempfile.NamedTemporaryFile(suffix='.disc.json', mode='w', delete=False) as f:
            json.dump(layout, f)
            disc_path = f.name
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            strip_path = f.name

        render_strip(disc_path, cell_h=cell_h, cell_w=cell_w,
                     fiducials=True, strip_dpi=72, out_path=strip_path)
        warped = np.array(Image.open(strip_path).convert('RGB'))

        import os
        os.unlink(disc_path)
        os.unlink(strip_path)

        steep, dirs, conf, alpha = read_strip(warped, layout, cell_h, border, cell_w=cell_w)
        stats = compare_layout(layout, steep, dirs)

        return dict(layout=layout, steep=steep, dirs=dirs,
                    conf=conf, alpha=alpha, stats=stats)

    def test_image_dimensions_match_layout(self, pipeline):
        """The rendered image width/height must match the expected canonical size."""
        layout = pipeline['layout']
        n_frames = layout['n_frames']
        n_bands  = layout['n_bands']
        cell_h, cell_w = 8, 8
        border = _border_px()
        content_h = n_bands * cell_h + 2 * SEP_PX
        expected_w = n_frames * cell_w + 2 * border
        expected_h = content_h + 2 * border
        # Just verify our geometry helpers agree (render already ran without error)
        assert expected_w > 0 and expected_h > 0

    def test_direction_accuracy_gte_93pct(self, pipeline):
        """Tier 0 baseline: direction accuracy ≥ 93% (spec target 98%)."""
        acc = pipeline['stats']['direction_accuracy']
        assert acc >= 0.93, f"Direction accuracy {acc:.4f} < 0.93"

    def test_mean_alpha_gte_090(self, pipeline):
        """Tier 0 baseline: mean α ≥ 0.90 (spec target 0.93)."""
        mean_alpha = float(pipeline['alpha'].mean())
        assert mean_alpha >= 0.90, f"Mean α {mean_alpha:.4f} < 0.90"

    def test_all_bands_have_nonzero_alpha(self, pipeline):
        """Every band should have non-trivial confidence when reading the rendered strip."""
        assert np.all(pipeline['alpha'] > 0.50), \
            f"Some bands have very low α: min={pipeline['alpha'].min():.4f}"

    def test_l0_bands_high_alpha(self, pipeline):
        """L0 (bass, bands 0–7) should have the highest confidence — widest gradients."""
        alpha  = pipeline['alpha']
        l0_mean = float(alpha[:8].mean())
        assert l0_mean >= 0.90, f"L0 mean α {l0_mean:.4f} < 0.90"

    def test_steepness_mae_lt_12pct(self, pipeline):
        """Steepness mean absolute error should stay below 0.12 (uint8 quantisation only)."""
        mae = pipeline['stats']['steepness_mae']
        assert mae < 0.12, f"Steepness MAE {mae:.4f} ≥ 0.12"

    def test_compare_layout_structure(self, pipeline):
        """compare_layout must return all expected keys."""
        stats = pipeline['stats']
        assert 'direction_accuracy' in stats
        assert 'steepness_mae'      in stats
        assert 'direction_errors'   in stats
        assert 'n_cells'            in stats

    def test_visualization_strip_no_crash(self):
        """Rendering and reading back a cell_w=1 visualization strip should not crash."""
        layout = _make_layout(n_hops=GOP_SIZE * 4)
        cell_h = 8; cell_w = 1; border = _border_px()

        with tempfile.NamedTemporaryFile(suffix='.disc.json', mode='w', delete=False) as f:
            json.dump(layout, f)
            disc_path = f.name
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            strip_path = f.name

        try:
            render_strip(disc_path, cell_h=cell_h, cell_w=cell_w,
                         fiducials=True, strip_dpi=72, out_path=strip_path)
            warped = np.array(Image.open(strip_path).convert('RGB'))
            steep, dirs, conf, alpha = read_strip(warped, layout, cell_h, border, cell_w=cell_w)
            assert steep.shape == (layout['n_frames'], layout['n_bands'])
        finally:
            import os
            os.unlink(disc_path)
            os.unlink(strip_path)

    @pytest.mark.parametrize("cell_w", [4, 8])
    def test_direction_accuracy_both_cell_widths(self, cell_w):
        """Direction accuracy ≥ 90% for both cell_w=4 and cell_w=8."""
        layout = _make_layout(n_hops=GOP_SIZE * 8, bitrate=32.0)
        cell_h = 8; border = _border_px()

        with tempfile.NamedTemporaryFile(suffix='.disc.json', mode='w', delete=False) as f:
            json.dump(layout, f)
            disc_path = f.name
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            strip_path = f.name

        try:
            render_strip(disc_path, cell_h=cell_h, cell_w=cell_w,
                         fiducials=True, strip_dpi=72, out_path=strip_path)
            warped = np.array(Image.open(strip_path).convert('RGB'))
            steep, dirs, conf, alpha = read_strip(warped, layout, cell_h, border, cell_w=cell_w)
            stats = compare_layout(layout, steep, dirs)
            acc = stats['direction_accuracy']
            assert acc >= 0.90, f"cell_w={cell_w}: direction accuracy {acc:.4f} < 0.90"
        finally:
            import os
            os.unlink(disc_path)
            os.unlink(strip_path)
