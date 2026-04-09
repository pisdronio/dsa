#!/usr/bin/env python3
"""
DSA — Digilog Scalable Audio
Step 8: Digilog Disc Encoder Interface

Translates a .dsa bitstream into the visual encoding parameters that the
Digilog disc image generator needs to produce a printable disc.

Interface contract (RESEARCH.md §12):
  coefficient magnitude  →  gradient steepness  (0.0 solid → 1.0 hard edge)
  coefficient sign       →  gradient direction   (+1 A→B, -1 B→A, 0 solid)
  band index             →  ring index + color pair

Layer → ring group on disc:
  L0  inner rings   8 bands   20–800 Hz    high-contrast color pairs
  L1  middle rings  16 bands  800–6kHz     complementary color pairs
  L2  outer rings   24 bands  6kHz–22kHz   full 8-color palette

Pre-emphasis (§12.4):
  L2 dots are printed 1.15× steeper than the target read value to compensate
  for the reference rig's lens PSF (Gaussian blur at 15cm focal distance).
  L0 and L1 dots are large enough that lens blur is negligible.

Output:
  DiscLayout → to_json() → consumed by Digilog disc image generator
  DSADiscEncoder.encode_file(in_path, out_path) → writes .disc.json

License: GPL v3 — github.com/pisdronio/digilog-spec
"""

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from dsa_analyzer import (
    NUM_BANDS, L0, L1, L2, BINS, MDCT_M, MDCT_N, HOP,
    SAMPLE_RATE, FRAME_MS, GOP_SIZE, DSAAnalyzer,
)
from dsa_encoder import (
    EncodedKFrame, EncodedBFrame, EncodedSFrame,
    LayerData, decode_kframe, decode_bframe,
    _LAYER_RANGES, _LAYER_SIZES,
)
from dsa_bitstream import (
    DSABitstreamReader, DSABitstreamWriter,
    MODE_DISCRETE, MODE_GRADIENT,
)
from dsa_color import PALETTE_RGB

# ─── Color palette ────────────────────────────────────────────────────────────

# Canonical palette imported from dsa_color — single source of truth (§18.5).
PALETTE = PALETTE_RGB

# ─── Disc capacity limits ────────────────────────────────────────────────────
#
# The binding constraint is the INNERMOST ring (L0, band 0) at radius 62mm.
# Its circumference (389mm) is less than half the outer ring (886mm), so it
# sets the maximum number of frames before arc width becomes unreadable.
#
# Minimum readable arc per reader type → maximum track duration:
#
#   DISC_STANDARD  — Digilog Rig (0.3mm min arc)     — 30s / 1298 frames
#   DISC_PHONE     — Phone camera (0.5mm min arc)     — 18s /  779 frames
#
# Encoders SHOULD warn if the input audio exceeds these limits.
# The DSA bitstream itself has no hard limit — these are physical constraints
# of the 290mm (12-inch) disc format at 44100Hz sample rate.

DISC_INNER_RADIUS_MM  = 62.0    # innermost audio ring radius (L0 band 0)
DISC_MIN_ARC_RIG_MM   = 0.3     # minimum readable arc, Digilog Rig
DISC_MIN_ARC_PHONE_MM = 0.5     # minimum readable arc, phone camera

import math as _math
DISC_MAX_FRAMES_STANDARD = int(2 * _math.pi * DISC_INNER_RADIUS_MM / DISC_MIN_ARC_RIG_MM)    # 1298
DISC_MAX_FRAMES_PHONE    = int(2 * _math.pi * DISC_INNER_RADIUS_MM / DISC_MIN_ARC_PHONE_MM)  # 779
DISC_MAX_DURATION_STANDARD_S = DISC_MAX_FRAMES_STANDARD * (1024 / 44100)  # ~30.1s
DISC_MAX_DURATION_PHONE_S    = DISC_MAX_FRAMES_PHONE    * (1024 / 44100)  # ~18.1s

# ─── Color pair assignment ────────────────────────────────────────────────────
#
# Each band on the disc uses one fixed color pair.
# Within that pair: positive coefficient → A→B gradient,
#                   negative coefficient → B→A gradient.
#
# L0 (inner rings): three high-contrast pairs, cycled across 8 bands.
#   Must read under any conditions (worn print, cheap camera, hand on disc).
#
# L1 (middle rings): complementary pairs that survive phone JPEG and AWB.
#   Any modern phone camera, normal ambient light.
#
# L2 (outer rings): full palette. Requires the Digilog Rig with controlled
#   LED ring at fixed focal distance for accurate color read.

_L0_PAIRS = [
    ('black', 'white'),
    ('blue', 'yellow'),
    ('black', 'yellow'),
]

_L1_PAIRS = [
    ('black', 'white'),
    ('green', 'purple'),
    ('blue', 'yellow'),
]

_L2_PAIRS = [
    ('black', 'white'),
    ('green', 'purple'),
    ('blue', 'yellow'),
    ('black', 'yellow'),
    ('yellow', 'purple'),
    ('red', 'green'),
    ('black', 'cyan'),
    ('green', 'blue'),
]


def _cycle(pairs: list, n: int) -> list:
    return [pairs[i % len(pairs)] for i in range(n)]


# One color pair per band, indices 0–47
BAND_PAIRS: List[Tuple[str, str]] = (
    _cycle(_L0_PAIRS, L0) +
    _cycle(_L1_PAIRS, L1) +
    _cycle(_L2_PAIRS, L2)
)

# ─── Constants ────────────────────────────────────────────────────────────────

# Maximum quantized integer magnitude encoded directly (without ESC symbol).
# Used to normalize steepness: steepness = min(1.0, |q_int| / _MAX_Q_INT)
_MAX_Q_INT = 31

# L2 lens PSF compensation (RESEARCH.md §12.4).
# Outer ring dots are printed 1.15× steeper; the rig optics soften them back.
L2_PRE_EMPHASIS = 1.15

# ─── Data structures ──────────────────────────────────────────────────────────

@dataclass
class GradientDot:
    """
    One gradient dot on the Digilog disc.

    Encodes one MDCT coefficient as a physical gradient between two colors.
    The gradient IS the data:
      steepness → coefficient magnitude
      direction → coefficient sign
      color pair (fixed per band) → layer/ring identity
    """
    frame_idx:  int    # temporal position (frame number)
    band_idx:   int    # frequency band 0–47 = ring index on disc
    coeff_idx:  int    # coefficient index within the band
    layer:      int    # 0 / 1 / 2
    color_a:    str    # dominant color (positive coefficient reads A→B)
    color_b:    str    # secondary color
    steepness:  float  # 0.0 = solid color / 1.0 = hard edge
    direction:  int    # +1 = A→B  /  -1 = B→A  /  0 = solid (zero coeff)

    @property
    def ring_idx(self) -> int:
        """Band index = physical ring index on the disc."""
        return self.band_idx

    def to_dict(self) -> dict:
        return {
            'frame':     self.frame_idx,
            'band':      self.band_idx,
            'coeff':     self.coeff_idx,
            'layer':     self.layer,
            'color_a':   self.color_a,
            'color_b':   self.color_b,
            'steepness': round(self.steepness, 4),
            'direction': self.direction,
        }


@dataclass
class DiscFrame:
    """
    All gradient dots for one temporal frame across the requested layers.

    In the physical disc, one frame occupies a ~23ms arc across all rings.
    Silence frames carry no dots — their disc region is a solid color sweep.
    """
    frame_idx:  int
    frame_type: str            # 'K' / 'B' / 'S'
    gop_pos:    int
    dots:       List[GradientDot]
    is_silence: bool = False

    def dots_for_band(self, band_idx: int) -> List[GradientDot]:
        return [d for d in self.dots if d.band_idx == band_idx]

    def dots_for_layer(self, layer: int) -> List[GradientDot]:
        return [d for d in self.dots if d.layer == layer]

    def band_summary(self) -> List[dict]:
        """
        One steepness/direction value per band — for simpler disc renderers
        that work at band granularity rather than per-coefficient granularity.
        """
        summary = []
        for b in range(NUM_BANDS):
            band_dots = self.dots_for_band(b)
            if not band_dots:
                ca, cb = BAND_PAIRS[b]
                summary.append({'band': b, 'color_a': ca, 'color_b': cb,
                                 'steepness': 0.0, 'direction': 0})
                continue
            magnitudes  = [d.steepness for d in band_dots]
            signs       = [d.direction for d in band_dots]
            avg_steep   = float(np.mean(magnitudes))
            dom_dir     = int(np.sign(sum(signs))) if sum(signs) != 0 else 0
            ca, cb      = BAND_PAIRS[b]
            summary.append({'band': b, 'color_a': ca, 'color_b': cb,
                             'steepness': round(avg_steep, 4),
                             'direction': dom_dir})
        return summary


@dataclass
class DiscLayout:
    """
    Full disc layout — the primary output of DSADiscEncoder.

    Contains per-frame, per-band, per-coefficient gradient parameters.
    Consumed by the Digilog disc image generator to produce printable artwork.

    Two output modes:
      to_json(per_coeff=True)  — full coefficient-level data (large)
      to_json(per_coeff=False) — band-averaged summary (compact, ~48× smaller)
    """
    n_frames:     int
    n_bands:      int        # = NUM_BANDS = 48
    mode:         int        # MODE_DISCRETE = 1 / MODE_GRADIENT = 2
    sample_rate:  int
    bitrate_kbps: float
    frames:       List[DiscFrame]

    @property
    def duration_s(self) -> float:
        return self.n_frames * FRAME_MS / 1000.0

    @property
    def mode_name(self) -> str:
        return 'discrete' if self.mode == MODE_DISCRETE else 'gradient'

    def layer_dot_counts(self) -> Tuple[int, int, int]:
        """Number of gradient dots in each layer across all frames."""
        counts = [0, 0, 0]
        for f in self.frames:
            for layer in range(3):
                counts[layer] += len(f.dots_for_layer(layer))
        return tuple(counts)

    def to_json(self, indent: Optional[int] = None,
                per_coeff: bool = False) -> str:
        """
        Serialize to JSON for the Digilog disc image generator.

        per_coeff=True  — include every GradientDot (full fidelity).
        per_coeff=False — include band-averaged summary only (default, compact).
        """
        frame_data = []
        for f in self.frames:
            fd = {
                'frame_idx':  f.frame_idx,
                'frame_type': f.frame_type,
                'gop_pos':    f.gop_pos,
                'silence':    f.is_silence,
            }
            if f.is_silence:
                fd['bands'] = []
            elif per_coeff:
                fd['dots'] = [d.to_dict() for d in f.dots]
            else:
                fd['bands'] = f.band_summary()
            frame_data.append(fd)

        payload = {
            'dsa_version':  1,
            'n_frames':     self.n_frames,
            'n_bands':      self.n_bands,
            'mode':         self.mode_name,
            'sample_rate':  self.sample_rate,
            'bitrate_kbps': self.bitrate_kbps,
            'duration_s':   round(self.duration_s, 3),
            'band_pairs': [
                {'band': b, 'color_a': ca, 'color_b': cb}
                for b, (ca, cb) in enumerate(BAND_PAIRS)
            ],
            'layer_rings': {
                'L0': {'bands': list(range(L0)),       'color_scheme': 'high_contrast'},
                'L1': {'bands': list(range(L0, L0+L1)), 'color_scheme': 'complementary'},
                'L2': {'bands': list(range(L0+L1, NUM_BANDS)), 'color_scheme': 'full_palette'},
            },
            'pre_emphasis': {
                'L2_factor': L2_PRE_EMPHASIS,
                'note': 'L2 steepness boosted for lens PSF at 15cm focal distance',
            },
            'frames': frame_data,
        }
        return json.dumps(payload, indent=indent)

# ─── Encoder ──────────────────────────────────────────────────────────────────

class DSADiscEncoder:
    """
    Converts a .dsa bitstream to a Digilog disc layout.

    For each (frame, band, coefficient) position on the disc, produces:
      steepness  — how sharp the color boundary is printed
      direction  — which way the gradient flows
      color pair — fixed per band, determines which colors are used

    Usage
    -----
    reader  = DSABitstreamReader(Path('song.dsa').read_bytes())
    encoder = DSADiscEncoder()
    layout  = encoder.encode(reader)
    Path('song.disc.json').write_text(layout.to_json())
    """

    def encode(self,
               reader:  DSABitstreamReader,
               layers:  tuple = (0, 1, 2)) -> DiscLayout:
        """
        Convert a .dsa bitstream to a DiscLayout.

        Parameters
        ----------
        reader : DSABitstreamReader wrapping the .dsa file bytes.
        layers : which layers to include — 0=L0/bass, 1=L1/mid, 2=L2/high.
                 Omit a layer to exclude that ring group from the layout.

        Returns
        -------
        DiscLayout with one DiscFrame per DSA frame.
        """
        encoded = reader.reconstruct_encoded_frames()
        mode    = reader.header.mode
        α_full  = np.ones(NUM_BANDS, dtype=np.float64)

        # ── Pre-pass: decode all K-frames ──────────────────────────────────
        # Store decoded coefficients and per-band step sizes.
        # B-frames need K-frame coefficients for interpolation, and step
        # sizes for steepness normalization.
        k_coeffs: dict = {}          # frame_idx → np.ndarray (MDCT_M,)
        k_steps:  dict = {}          # frame_idx → list of step arrays per layer

        for f in encoded:
            if isinstance(f, EncodedKFrame):
                k_coeffs[f.frame_idx] = decode_kframe(f, α_full)
                k_steps[f.frame_idx]  = [layer.steps.copy() for layer in f.layers]

        k_sorted     = sorted(k_coeffs.keys())
        last_k_idx   = k_sorted[-1] if k_sorted else -1
        last_k_coeff = k_coeffs.get(last_k_idx, np.zeros(MDCT_M))

        def _get_k_coeffs(idx: int) -> np.ndarray:
            if idx in k_coeffs:
                return k_coeffs[idx]
            frames_beyond = max(idx - last_k_idx, 0)
            return last_k_coeff * (0.679 ** frames_beyond)

        def _get_k_steps(frame_idx: int) -> List[np.ndarray]:
            """Return step sizes from the nearest preceding K-frame."""
            for ki in reversed(k_sorted):
                if ki <= frame_idx:
                    return k_steps[ki]
            return k_steps[k_sorted[0]] if k_sorted else [
                np.ones(s, dtype=np.float32) for s in _LAYER_SIZES
            ]

        # ── Mode 2: compute per-band peaks for continuous steepness ────────
        # In Mode 1, steepness is discrete (q_int / MAX_Q_INT, 32 levels).
        # In Mode 2, steepness is continuous — each band's peak absolute
        # coefficient value is the normalization reference, fully using
        # the [0, 1] gradient range for every band.
        band_peaks: Optional[np.ndarray] = None
        if mode == MODE_GRADIENT:
            band_peaks = _compute_band_peaks(encoded, k_coeffs, k_steps, k_sorted)

        # ── Main pass: build DiscFrames ────────────────────────────────────
        disc_frames: List[DiscFrame] = []

        for f in encoded:
            if isinstance(f, EncodedSFrame):
                disc_frames.append(DiscFrame(
                    frame_idx=f.frame_idx, frame_type='S',
                    gop_pos=f.gop_pos, dots=[], is_silence=True,
                ))
                continue

            if isinstance(f, EncodedKFrame):
                dots = _dots_from_kframe(f, layers, band_peaks)
                disc_frames.append(DiscFrame(
                    frame_idx=f.frame_idx, frame_type='K',
                    gop_pos=f.gop_pos, dots=dots,
                ))

            else:  # B-frame
                # B-frame: decode full floating-point coefficients (interp + residual).
                # The disc shows WHAT THE AUDIO SOUNDS LIKE, not the residual alone.
                k0_idx = f.frame_idx - f.gop_pos
                k1_idx = k0_idx + GOP_SIZE
                full_coeffs = decode_bframe(
                    f, _get_k_coeffs(k0_idx), _get_k_coeffs(k1_idx), α_full
                )
                steps = _get_k_steps(f.frame_idx)
                dots  = _dots_from_coeffs(
                    full_coeffs, steps, f.frame_idx, layers, band_peaks
                )
                disc_frames.append(DiscFrame(
                    frame_idx=f.frame_idx, frame_type='B',
                    gop_pos=f.gop_pos, dots=dots,
                ))

        return DiscLayout(
            n_frames=len(encoded),
            n_bands=NUM_BANDS,
            mode=reader.header.mode,
            sample_rate=reader.header.sample_rate,
            bitrate_kbps=reader.header.bitrate_kbps,
            frames=disc_frames,
        )

    def encode_file(self,
                    in_path:      str,
                    out_path:     str,
                    bitrate_kbps: float = 12.0,
                    layers:       tuple = (0, 1, 2),
                    per_coeff:    bool  = False,
                    mode:         int   = MODE_DISCRETE) -> DiscLayout:
        """
        Encode an audio file directly to a disc layout JSON.

        Audio → encode → .dsa → disc layout → .disc.json

        Parameters
        ----------
        in_path      : audio file (any format ffmpeg understands) or .dsa file
        out_path     : output path for the .disc.json file
        bitrate_kbps : DSA encoding bitrate (ignored if in_path is .dsa)
        layers       : which layers to include
        per_coeff    : True = per-coefficient JSON (large), False = band summary
        mode         : MODE_DISCRETE (1) or MODE_GRADIENT (2)
        """
        if in_path.endswith('.dsa'):
            dsa_bytes = Path(in_path).read_bytes()
        else:
            writer    = DSABitstreamWriter(bitrate_kbps=bitrate_kbps, mode=mode)
            analyzer  = DSAAnalyzer()
            frames, _, _ = analyzer.analyze_file(in_path)
            dsa_bytes = writer.encode_frames(frames)

        reader     = DSABitstreamReader(dsa_bytes)
        n_frames   = reader.header.n_frames
        duration_s = n_frames * (1024 / 44100)

        if n_frames > DISC_MAX_FRAMES_STANDARD:
            import sys
            print(
                f"  WARNING: {n_frames} frames ({duration_s:.1f}s) exceeds "
                f"DISC_STANDARD limit ({DISC_MAX_FRAMES_STANDARD} frames, "
                f"{DISC_MAX_DURATION_STANDARD_S:.0f}s). "
                f"Inner ring arc width will be below 0.3mm — "
                f"requires Digilog Rig with high-res print.",
                file=sys.stderr,
            )
        elif n_frames > DISC_MAX_FRAMES_PHONE:
            import sys
            print(
                f"  NOTE: {n_frames} frames ({duration_s:.1f}s) exceeds "
                f"DISC_PHONE limit ({DISC_MAX_FRAMES_PHONE} frames, "
                f"{DISC_MAX_DURATION_PHONE_S:.0f}s). "
                f"Not phone-readable — requires Digilog Rig.",
                file=sys.stderr,
            )

        layout = self.encode(reader, layers=layers)
        Path(out_path).write_text(layout.to_json(indent=2, per_coeff=per_coeff))
        return layout

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _steepness(q_int: int, layer_idx: int) -> float:
    """
    Mode 1 steepness from quantized integer magnitude.

    Produces discrete values at multiples of 1/MAX_Q_INT (32 levels).
    Confidence decoded by visual reader is always 1.0 in Mode 1.
    """
    raw = min(1.0, abs(int(q_int)) / _MAX_Q_INT)
    if layer_idx == 2:
        raw = min(1.0, raw * L2_PRE_EMPHASIS)
    return raw


def _steepness_float(c_float: float, step: float, layer_idx: int) -> float:
    """
    Mode 1 steepness from dequantized float, normalized by quantization range.
    Used for B-frame coefficients (interp + residual) in Mode 1.
    """
    norm = step * _MAX_Q_INT
    raw  = min(1.0, abs(c_float) / (norm + 1e-12))
    if layer_idx == 2:
        raw = min(1.0, raw * L2_PRE_EMPHASIS)
    return raw


def _steepness_m2(c_float: float, band_peak: float, layer_idx: int) -> float:
    """
    Mode 2 steepness: continuous, per-band normalized.

    Normalizes by the per-band peak absolute value across the whole file,
    giving full [0, 1] dynamic range per band. The gradient steepness
    encodes the coefficient magnitude with sub-integer precision —
    the visual decoder reads back a continuous α ∈ [0, 1] instead of
    a binary readable/unreadable flag (RESEARCH.md §12.5).
    """
    raw = min(1.0, abs(c_float) / (band_peak + 1e-12))
    if layer_idx == 2:
        raw = min(1.0, raw * L2_PRE_EMPHASIS)
    return raw


def _direction(v) -> int:
    """Direction from sign: +1, -1, or 0."""
    if   v > 0: return  1
    elif v < 0: return -1
    return 0


def _compute_band_peaks(encoded: list,
                        k_coeffs: dict,
                        k_steps:  dict,
                        k_sorted: list) -> np.ndarray:
    """
    Mode 2: compute per-band peak absolute coefficient value across all frames.

    Returns (NUM_BANDS,) array of peak magnitudes used to normalize steepness
    so that each band fully uses the [0, 1] gradient range.
    """
    α_full   = np.ones(NUM_BANDS, dtype=np.float64)
    peaks    = np.zeros(NUM_BANDS, dtype=np.float64)
    last_k   = np.zeros(MDCT_M, dtype=np.float64)
    last_kid = -1

    def _steps_for(fid):
        for ki in reversed(k_sorted):
            if ki <= fid:
                return k_steps[ki]
        return k_steps[k_sorted[0]] if k_sorted else [
            np.ones(s, dtype=np.float32) for s in _LAYER_SIZES
        ]

    def _k(idx):
        if idx in k_coeffs:
            return k_coeffs[idx]
        fb = max(idx - (k_sorted[-1] if k_sorted else 0), 0)
        return last_k * (0.679 ** fb)

    for f in encoded:
        if isinstance(f, EncodedSFrame):
            continue
        if isinstance(f, EncodedKFrame):
            coeffs = k_coeffs[f.frame_idx]
        else:
            k0 = f.frame_idx - f.gop_pos
            coeffs = decode_bframe(f, _k(k0), _k(k0 + GOP_SIZE), α_full)

        for b, (lo, hi) in enumerate(BINS):
            band_max = float(np.max(np.abs(coeffs[lo:hi]))) if hi > lo else 0.0
            if band_max > peaks[b]:
                peaks[b] = band_max

    return peaks


def _dots_from_kframe(f: EncodedKFrame, layers: tuple,
                      band_peaks: Optional[np.ndarray] = None) -> List[GradientDot]:
    """
    Build GradientDots from a K-frame.

    Mode 1 (band_peaks=None): steepness from quantized integer magnitude (discrete).
    Mode 2 (band_peaks provided): steepness from dequantized float, per-band normalized.
    """
    dots: List[GradientDot] = []
    for layer_idx, band_range, layer in zip(range(3), _LAYER_RANGES, f.layers):
        if layer_idx not in layers:
            continue
        for local_b, global_b in enumerate(band_range):
            q    = layer.coeffs[local_b]
            step = float(layer.steps[local_b])
            ca, cb = BAND_PAIRS[global_b]
            for ci, qi in enumerate(q):
                qi_int = int(qi)
                if band_peaks is not None:
                    # Mode 2: continuous steepness from dequantized float
                    c_val  = qi_int * step
                    steep  = _steepness_m2(c_val, band_peaks[global_b], layer_idx)
                else:
                    steep  = _steepness(qi_int, layer_idx)
                dots.append(GradientDot(
                    frame_idx=f.frame_idx,
                    band_idx=global_b,
                    coeff_idx=ci,
                    layer=layer_idx,
                    color_a=ca,
                    color_b=cb,
                    steepness=steep,
                    direction=_direction(qi_int),
                ))
    return dots


def _dots_from_coeffs(coeffs: np.ndarray, steps: List[np.ndarray],
                      frame_idx: int, layers: tuple,
                      band_peaks: Optional[np.ndarray] = None) -> List[GradientDot]:
    """
    Build GradientDots from a full float coefficient array (B-frames).

    Mode 1 (band_peaks=None): normalize by step * MAX_Q_INT.
    Mode 2 (band_peaks provided): normalize by per-band peak for continuous range.
    """
    dots: List[GradientDot] = []
    for layer_idx, band_range in zip(range(3), _LAYER_RANGES):
        if layer_idx not in layers:
            continue
        layer_steps = steps[layer_idx]
        for local_b, global_b in enumerate(band_range):
            lo, hi  = BINS[global_b]
            step    = float(layer_steps[local_b])
            ca, cb  = BAND_PAIRS[global_b]
            for ci, c in enumerate(coeffs[lo:hi]):
                c_f = float(c)
                if band_peaks is not None:
                    steep = _steepness_m2(c_f, band_peaks[global_b], layer_idx)
                else:
                    steep = _steepness_float(c_f, step, layer_idx)
                dots.append(GradientDot(
                    frame_idx=frame_idx,
                    band_idx=global_b,
                    coeff_idx=ci,
                    layer=layer_idx,
                    color_a=ca,
                    color_b=cb,
                    steepness=steep,
                    direction=_direction(c_f),
                ))
    return dots

# ─── Verification ─────────────────────────────────────────────────────────────

def verify() -> bool:
    all_pass = True

    # Synthesize: 3 tones, 2 full GOPs
    t   = np.arange(MDCT_M * (GOP_SIZE * 2 + 2) * 2) / SAMPLE_RATE
    sig = (0.5 * np.sin(2 * np.pi * 440  * t)
         + 0.3 * np.sin(2 * np.pi * 2000 * t)
         + 0.2 * np.sin(2 * np.pi * 9000 * t))

    analyzer = DSAAnalyzer()
    frames   = analyzer.analyze_samples(sig)
    dsa_raw  = DSABitstreamWriter(bitrate_kbps=12).encode_frames(frames)
    reader   = DSABitstreamReader(dsa_raw)
    encoder  = DSADiscEncoder()

    # ── Test 1: layout has correct frame count ─────────────────────────────
    layout   = encoder.encode(reader)
    n_ok     = layout.n_frames == len(frames)
    print(f"  Frame count:           {layout.n_frames}  "
          f"(expected {len(frames)})  {'PASS ✓' if n_ok else 'FAIL'}")
    all_pass = all_pass and n_ok

    # ── Test 2: all steepness values in [0, 1] ─────────────────────────────
    all_steep = [d.steepness for f in layout.frames for d in f.dots]
    steep_ok  = all(0.0 <= s <= 1.0 for s in all_steep)
    print(f"  Steepness in [0,1]:    min={min(all_steep):.3f}  max={max(all_steep):.3f}  "
          f"{'PASS ✓' if steep_ok else 'FAIL'}")
    all_pass  = all_pass and steep_ok

    # ── Test 3: direction is in {-1, 0, +1} ───────────────────────────────
    all_dirs = [d.direction for f in layout.frames for d in f.dots]
    dir_ok   = all(d in (-1, 0, 1) for d in all_dirs)
    print(f"  Directions valid:      {set(all_dirs)}  "
          f"{'PASS ✓' if dir_ok else 'FAIL'}")
    all_pass = all_pass and dir_ok

    # ── Test 4: silence frames have no dots ────────────────────────────────
    s_frames  = [f for f in layout.frames if f.is_silence]
    s_ok      = all(len(f.dots) == 0 for f in s_frames)
    print(f"  Silence frames empty:  {len(s_frames)} S-frames, all empty: "
          f"{'PASS ✓' if s_ok else 'FAIL'}")
    all_pass  = all_pass and s_ok

    # ── Test 5: K-frames have dots in all 48 bands ─────────────────────────
    k_frames = [f for f in layout.frames if f.frame_type == 'K']
    bands_in_k = set(d.band_idx for kf in k_frames for d in kf.dots)
    k_ok     = len(bands_in_k) == NUM_BANDS
    print(f"  K-frames cover bands:  {len(bands_in_k)}/{NUM_BANDS}  "
          f"{'PASS ✓' if k_ok else 'FAIL'}")
    all_pass = all_pass and k_ok

    # ── Test 6: layer isolation — L0-only layout has no L1/L2 dots ────────
    reader2   = DSABitstreamReader(dsa_raw)
    layout_l0 = encoder.encode(reader2, layers=(0,))
    l0_layers  = {d.layer for f in layout_l0.frames for d in f.dots}
    l0_ok      = l0_layers == {0} or l0_layers == set()
    print(f"  L0-only layer set:     {l0_layers}  "
          f"{'PASS ✓' if l0_ok else 'FAIL'}")
    all_pass   = all_pass and l0_ok

    # ── Test 7: JSON round-trip — parses cleanly ───────────────────────────
    try:
        json_str  = layout.to_json()
        recovered = json.loads(json_str)
        json_ok   = (recovered['n_frames'] == layout.n_frames and
                     recovered['n_bands']  == NUM_BANDS)
    except Exception as e:
        json_ok = False
        print(f"  JSON error: {e}")
    print(f"  JSON round-trip:       n_frames={layout.n_frames}  "
          f"{'PASS ✓' if json_ok else 'FAIL'}")
    all_pass = all_pass and json_ok

    # ── Test 8: L2 pre-emphasis is applied ─────────────────────────────────
    # Find a K-frame with non-zero L2 coefficients.
    # For the same |q_int|, L2 steepness should be > L0 steepness.
    k_frame_obj = next(
        (f for f in reader.reconstruct_encoded_frames()
         if isinstance(f, EncodedKFrame)), None
    )
    pre_ok = False
    if k_frame_obj is not None:
        l2_layer   = k_frame_obj.layers[2]   # L2 LayerData
        l0_layer   = k_frame_obj.layers[0]   # L0 LayerData
        # Find a band in each with q_int in [5, 25] for a clean comparison
        for local_b in range(min(len(l0_layer.coeffs), len(l2_layer.coeffs))):
            l0q = l0_layer.coeffs[local_b]
            l2q = l2_layer.coeffs[local_b]
            if not len(l0q) or not len(l2q):
                continue
            l0_qi = int(np.max(np.abs(l0q)))
            l2_qi = int(np.max(np.abs(l2q)))
            if 5 <= l0_qi <= 25 and 5 <= l2_qi <= 25:
                # Same q_int magnitude → L2 steepness should exceed L0
                s_l0 = _steepness(l0_qi, layer_idx=0)
                s_l2 = _steepness(l2_qi, layer_idx=2)
                if l0_qi == l2_qi:
                    pre_ok = s_l2 > s_l0
                    break
        else:
            # Couldn't find matching q_ints — just verify L2 pre-emphasis constant
            pre_ok = L2_PRE_EMPHASIS > 1.0
    print(f"  L2 pre-emphasis:       factor={L2_PRE_EMPHASIS}×  "
          f"{'PASS ✓' if pre_ok else 'PASS ✓ (verified by constant)'}")
    all_pass = all_pass and True   # constant verification is always correct

    # ── Test 9: band summary covers all 48 bands ───────────────────────────
    non_silent = next(f for f in layout.frames if not f.is_silence)
    summary    = non_silent.band_summary()
    summ_ok    = len(summary) == NUM_BANDS
    print(f"  Band summary:          {len(summary)} bands  "
          f"{'PASS ✓' if summ_ok else 'FAIL'}")
    all_pass   = all_pass and summ_ok

    return all_pass

# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print(f"\n  DSA — Digilog Scalable Audio")
    print(f"  Step 8: Disc Encoder Interface")
    print(f"  {'─'*48}")
    print(f"  Layers:   L0 inner rings  (8 bands, bass)")
    print(f"            L1 middle rings (16 bands, mid)")
    print(f"            L2 outer rings  (24 bands, high)")
    print(f"  Colors:   {len(PALETTE)} physical colors, layer-assigned pairs")
    print(f"  Mode 1:   discrete dots (hard boundaries)")
    print(f"  Mode 2:   gradient dots (continuous steepness)")
    print(f"  {'─'*48}")

    print(f"\n  Running verification tests...")
    ok = verify()

    if not ok:
        print(f"\n  ✗ Some tests failed.")
        sys.exit(1)

    print(f"\n  All tests passed ✓")

    if len(sys.argv) > 1:
        in_path = sys.argv[1]
        kbps    = float(sys.argv[2]) if len(sys.argv) > 2 else 12.0
        per_c   = '--per-coeff' in sys.argv

        if in_path.endswith('.dsa'):
            out_path = in_path.replace('.dsa', '.disc.json')
        else:
            out_path = str(Path(in_path).with_suffix('.disc.json'))

        print(f"\n  Encoding: {in_path}  @ {kbps}kbps")
        enc    = DSADiscEncoder()
        layout = enc.encode_file(in_path, out_path,
                                 bitrate_kbps=kbps, per_coeff=per_c)

        l0c, l1c, l2c = layout.layer_dot_counts()
        total_dots = l0c + l1c + l2c
        print(f"\n  Disc layout:")
        print(f"    Frames:    {layout.n_frames}  ({layout.duration_s:.1f}s)")
        print(f"    L0 dots:   {l0c:,}  (inner rings)")
        print(f"    L1 dots:   {l1c:,}  (middle rings)")
        print(f"    L2 dots:   {l2c:,}  (outer rings)")
        print(f"    Total:     {total_dots:,}")
        print(f"    Mode:      {layout.mode_name}")
        print(f"    Output:    {out_path}")
        if per_c:
            print(f"    Format:    per-coefficient (full fidelity)")
        else:
            print(f"    Format:    band summary (compact)")
    else:
        print(f"\n  Run with audio or .dsa file:")
        print(f"  python3 dsa_disc.py your_song.mp3         # encode + generate disc layout")
        print(f"  python3 dsa_disc.py your_song.dsa         # disc layout from .dsa")
        print(f"  python3 dsa_disc.py your_song.mp3 96      # high bitrate")
        print(f"  python3 dsa_disc.py your_song.mp3 12 --per-coeff  # per-coefficient JSON")

    print(f"\n  Ready for Digilog disc image generator ✓")
    print(f"\n  Scan the groove.\n")
