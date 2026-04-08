#!/usr/bin/env python3
"""
DSA — Digilog Scalable Audio
dsa_live.py — Live Strip Reader / Single-Column Sampler

The real instrument. The camera reads one column of the strip at a time —
the tape-head slice. Everything to the left is past audio. Everything to
the right is future audio. The needle sees only NOW.

This is not an image processor. It is a single-column sampler driven by
physical motion. Speed, direction, and position all emerge from how fast
and in what direction the strip moves past the window.

Architecture
------------
  DSAStreamDecoder
    Loads a .dsa bitstream, pre-caches all K-frames. Exposes
    decode_frame(idx, alpha) → HOP audio samples via IMDCT + overlap-add.
    Maintains overlap buffer across frames for seamless OLA reconstruction.
    Handles arbitrary seek (scratch) by resetting the overlap state.

  StripColumnSampler
    Given a warped strip image (or live camera frame), extracts the
    48-band colour read for whichever column is currently in the window.
    Same colour projection math as dsa_reader.py and dsa_camera.py.

  SpeedEstimator
    Tracks tape position over time. Derives playback rate in audio frames
    per second. Converts column displacement to speed ratio.
    0.0 = stopped (needle touching, no motion) — exponential decay applies.
    Negative = reverse scratch.

  TapeDisplay
    Scrolling terminal display: shows the last N rows of colour data
    that passed through the window — exactly what §12.8 describes.
    One row per audio frame. Left = L0 (bass). Right = L2 (high).

Timecode architecture (§4.3 — built into the codec from day 1)
---------------------------------------------------------------
  DSA unifies timecode and audio into a single physical signal.
  There is no separate control track (unlike Traktor timecode vinyl).

  Speed signal    → column displacement per video frame (strip) or
                    clock-track dot spacing (disc)
  Absolute position → frame index is the column x-coordinate in the
                    canonical strip; no separate position encoding needed
  Direction        → sign of column displacement (positive = forward)
  Sync points      → K-frames every 8 frames (~185ms) allow instant
                    resync after arbitrary seek; reference markers (disc)
                    provide 8 absolute anchors per revolution

  Difference from Traktor: Traktor puts timecode ON a record whose audio
  is in a computer. DSA's physical medium IS the audio. The codec, the
  timecode, and the music are the same object.

  What Traktor has that DSA lacks (future work):
    - A pilot-tone ring readable at near-zero speed (column-displacement
      method fails at very slow scratch; need a persistent geometric anchor)
    - Per-cell redundant position counter (frame index is implicit in
      geometry, not encoded redundantly in the visual cells)

Modes
-----
  simulate   read from a rendered strip PNG + disc.json (no camera required)
             simulates the tape advancing at a controllable speed
             audio output via sounddevice (or WAV file if unavailable)

  webcam     live OpenCV VideoCapture — strip must be visible in frame
             detects strip position, tracks column displacement, decodes
             audio in real time                            [planned — stub]

  scan       read a static strip PNG from left to right at nominal speed
             equivalent to dsa_camera.py but with audio output, no UI

Usage
-----
    python3 dsa_live.py simulate strip.png song.disc.json song.dsa
    python3 dsa_live.py simulate strip.png song.disc.json song.dsa --speed 0.5
    python3 dsa_live.py simulate strip.png song.disc.json song.dsa --reverse
    python3 dsa_live.py simulate strip.png song.disc.json song.dsa --out sim.wav
    python3 dsa_live.py scan    strip.png song.disc.json song.dsa
    python3 dsa_live.py webcam            song.disc.json song.dsa  [planned]

License: GPL v3 — github.com/pisdronio/dsa
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

try:
    from PIL import Image as PILImage
except ImportError:
    sys.exit("Pillow required — pip install Pillow")

# ─── DSA imports ──────────────────────────────────────────────────────────────

from dsa_analyzer import (
    NUM_BANDS, MDCT_M, MDCT_N, HOP, SAMPLE_RATE, GOP_SIZE,
    imdct,
)
from dsa_encoder import (
    EncodedKFrame, EncodedBFrame, EncodedSFrame,
    decode_kframe, decode_bframe,
)
from dsa_bitstream import DSABitstreamReader

# ─── Colour palette (matches dsa_reader.py / dsa_camera.py) ──────────────────

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

# ─── Geometry constants (match dsa_strip.py) ──────────────────────────────────

SEP_PX  = 2
L0_END  = 8
L1_END  = 24

# ─── Terminal display ─────────────────────────────────────────────────────────

# ANSI colour blocks for 48-band terminal display.
# Map band index → ANSI 256-colour code approximation.
_ANSI_RESET = '\033[0m'

def _ansi_block(r: int, g: int, b: int) -> str:
    """Return an ANSI 256-colour background block character (2 spaces)."""
    # Convert 8-bit RGB to nearest xterm-256 colour cube
    def _to6(v: int) -> int:
        if v < 48:  return 0
        if v < 115: return 1
        return int((v - 35) / 40)

    rc, gc, bc = _to6(r), _to6(g), _to6(b)
    code = 16 + 36 * rc + 6 * gc + bc
    return f'\033[48;5;{code}m  {_ANSI_RESET}'


def _render_tape_row(band_colors: list[tuple[int, int, int]]) -> str:
    """
    Render one strip row (one audio frame) as a line of coloured terminal blocks.
    band_colors: list of 48 (R, G, B) uint8 tuples, index 0=bass (left) → 47=high (right).
    """
    return ''.join(_ansi_block(r, g, b) for r, g, b in band_colors)


# ─── DSAStreamDecoder ─────────────────────────────────────────────────────────

class DSAStreamDecoder:
    """
    Frame-by-frame streaming decoder.

    Pre-loads all K-frames from the .dsa bitstream on init (fast — small files).
    Exposes decode_frame(idx, alpha) → HOP PCM samples.
    Maintains IMDCT overlap-add state for seamless reconstruction during
    sequential playback. Seeking (scratch) clears the overlap buffer —
    a brief transient occurs, matching the physical vinyl scratch behaviour.

    Parameters
    ----------
    dsa_bytes : raw .dsa file bytes
    """

    def __init__(self, dsa_bytes: bytes):
        reader          = DSABitstreamReader(dsa_bytes)
        self.header     = reader.header
        self.n_frames   = self.header.n_frames
        self.sample_rate = self.header.sample_rate

        encoded = reader.reconstruct_encoded_frames()

        # Index encoded frames by frame_idx for O(1) access
        self._frames: dict[int, object] = {}
        for f in encoded:
            self._frames[f.frame_idx] = f

        # Pre-decode all K-frames
        self._k_coeffs: dict[int, np.ndarray] = {}
        last_k_idx    = -1
        self._last_k  = np.zeros(MDCT_M, dtype=np.float64)

        for fi, f in sorted(self._frames.items()):
            if isinstance(f, EncodedKFrame):
                c = decode_kframe(f, np.ones(NUM_BANDS))   # unit alpha — scale later
                self._k_coeffs[fi] = c
                if fi > last_k_idx:
                    last_k_idx   = fi
                    self._last_k = c

        # Overlap-add state (MDCT_M samples of tail from previous frame)
        self._ola_tail     = np.zeros(MDCT_M, dtype=np.float64)
        self._last_fi      = -1    # last decoded frame index (for seek detection)

        print(f"  DSAStreamDecoder: {self.n_frames} frames, "
              f"{self.n_frames / SAMPLE_RATE * HOP:.1f}s, "
              f"{len(self._k_coeffs)} K-frames cached")

    def _get_k(self, idx: int, α: np.ndarray) -> np.ndarray:
        """Return α-scaled K-frame coefficients, with decay fallback."""
        if idx in self._k_coeffs:
            return self._k_coeffs[idx] * α
        # Truncated stream — decay gracefully from last known K
        frames_beyond = max(0, idx - max(self._k_coeffs.keys(), default=0))
        from dsa_decoder import DECAY_PER_FRAME
        return self._last_k * α * (DECAY_PER_FRAME ** frames_beyond)

    def get_coeffs(self, frame_idx: int,
                   alpha: np.ndarray | None = None) -> np.ndarray:
        """
        Return MDCT coefficient array for frame_idx, scaled by alpha.
        alpha : (NUM_BANDS,) confidence, or None for unity.
        """
        α = alpha if alpha is not None else np.ones(NUM_BANDS, dtype=np.float64)
        fi = max(0, min(frame_idx, self.n_frames - 1))
        f  = self._frames.get(fi)

        if f is None or isinstance(f, EncodedSFrame):
            return np.zeros(MDCT_M, dtype=np.float64)

        if isinstance(f, EncodedKFrame):
            return self._k_coeffs.get(fi, np.zeros(MDCT_M)) * α

        # B-frame — needs surrounding K-frames
        k0_idx = fi - f.gop_pos
        k1_idx = k0_idx + GOP_SIZE
        return decode_bframe(f, self._get_k(k0_idx, α), self._get_k(k1_idx, α), α)

    def decode_frame(self, frame_idx: int,
                     alpha: np.ndarray | None = None,
                     allow_seek: bool = True) -> np.ndarray:
        """
        Decode one audio frame → HOP PCM float64 samples.

        If frame_idx is not sequential (seek / scratch), the overlap-add
        state is cleared first — produces a brief transient, matching the
        physical scratch artefact.

        Returns HOP samples at self.sample_rate.
        """
        # Detect seek (non-sequential or reverse jump)
        if allow_seek and self._last_fi >= 0 and abs(frame_idx - self._last_fi) > 1:
            self._ola_tail = np.zeros(MDCT_M, dtype=np.float64)

        coeffs = self.get_coeffs(frame_idx, alpha)
        full   = imdct(coeffs)           # MDCT_N = 2048 samples

        # IMDCT overlap-add: sum with stored tail, output first HOP samples
        combined  = full.copy()
        combined[:MDCT_M] += self._ola_tail

        samples        = combined[:HOP]       # output window
        self._ola_tail = combined[HOP:]       # save tail for next frame
        self._last_fi  = frame_idx

        return samples

    def reset(self):
        """Clear overlap-add state (use after a scratch jump)."""
        self._ola_tail = np.zeros(MDCT_M, dtype=np.float64)
        self._last_fi  = -1


# ─── StripColumnSampler ───────────────────────────────────────────────────────

class StripColumnSampler:
    """
    Samples one column of a warped strip image → 48-band colour read.

    The strip must already be in canonical space (matching dsa_strip.py output,
    with or without fiducial border).  For live camera input, run homography
    correction first (via dsa_camera._warp_strip) and pass the result here.

    Parameters
    ----------
    strip_arr : (H, W, 3) uint8 numpy array — the warped strip image
    layout    : .disc.json dict
    cell_h    : pixel height per band row (must match rendering params)
    border_px : border added by dsa_strip.py --fiducials (0 if no border)
    """

    def __init__(self, strip_arr: np.ndarray, layout: dict,
                 cell_h: int = 8, border_px: int = 0):
        self.strip    = strip_arr
        self.layout   = layout
        self.cell_h   = cell_h
        self.border   = border_px
        self.n_frames = layout['n_frames']
        self.n_bands  = layout['n_bands']

        # Colour pairs per band
        self.bp_map = {b['band']: (b['color_a'], b['color_b'])
                       for b in layout['band_pairs']}

        # Pre-compute sample Y position for each band
        content_h = self.n_bands * cell_h + 2 * SEP_PX
        self._sample_y = []
        for b in range(self.n_bands):
            sep_below = 0
            if b >= L0_END:  sep_below += SEP_PX
            if b >= L1_END:  sep_below += SEP_PX
            bottom_offset = b * cell_h + sep_below
            row_top = content_h - bottom_offset - cell_h
            y = border_px + row_top + cell_h / 2.0
            self._sample_y.append(y)

    def _bilinear(self, x: float, y: float) -> np.ndarray:
        arr = self.strip
        h, w = arr.shape[:2]
        x0, y0 = int(x), int(y)
        x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
        x0, y0 = max(0, x0), max(0, y0)
        fx, fy = x - int(x), y - int(y)
        return (arr[y0, x0].astype(np.float64) * (1 - fx) * (1 - fy) +
                arr[y0, x1].astype(np.float64) * fx       * (1 - fy) +
                arr[y1, x0].astype(np.float64) * (1 - fx) * fy       +
                arr[y1, x1].astype(np.float64) * fx       * fy)

    def _color_to_blend(self, color: np.ndarray,
                        ca: np.ndarray, cb: np.ndarray) -> tuple[float, float]:
        v     = cb - ca
        denom = np.dot(v, v)
        if denom < 1e-6:
            return 0.0, 1.0
        t         = float(np.clip(np.dot(color - ca, v) / denom, 0.0, 1.0))
        projected = ca + t * v
        residual  = float(np.linalg.norm(color - projected))
        max_dist  = float(np.linalg.norm(v))
        return t, max(0.0, 1.0 - residual / (max_dist + 1e-9))

    def sample_column(self, frame_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Read the 48-band colour data for one frame (one column of the strip).

        Returns
        -------
        alpha      : (n_bands,) float64 — per-band confidence
        steepness  : (n_bands,) float64 — gradient steepness
        direction  : (n_bands,) int8    — gradient direction (-1, 0, +1)
        """
        fi = max(0, min(frame_idx, self.n_frames - 1))
        x_base = self.border + fi

        alpha     = np.zeros(self.n_bands, dtype=np.float64)
        steepness = np.zeros(self.n_bands, dtype=np.float64)
        direction = np.zeros(self.n_bands, dtype=np.int8)

        for b in range(self.n_bands):
            ca_name, cb_name = self.bp_map[b]
            ca = PALETTE[ca_name]
            cb = PALETTE[cb_name]
            y  = self._sample_y[b]

            c_left  = self._bilinear(x_base + 0.25, y)
            c_right = self._bilinear(x_base + 0.75, y)

            t_left,  al = self._color_to_blend(c_left,  ca, cb)
            t_right, ar = self._color_to_blend(c_right, ca, cb)

            diff = t_right - t_left
            s    = min(1.0, 2.0 * abs(diff))

            if abs(diff) < DIRECTION_THRESHOLD:
                d = 0
            elif diff > 0:
                d = np.int8(1)
            else:
                d = np.int8(-1)

            alpha[b]     = (al + ar) / 2.0
            steepness[b] = s
            direction[b] = d

        return alpha, steepness, direction

    def band_color_at(self, frame_idx: int, band: int) -> tuple[int, int, int]:
        """Return the representative RGB uint8 colour for one (frame, band) cell."""
        fi = max(0, min(frame_idx, self.n_frames - 1))
        x  = self.border + fi + 0.5
        y  = self._sample_y[band]
        c  = self._bilinear(x, y)
        return (int(np.clip(c[0], 0, 255)),
                int(np.clip(c[1], 0, 255)),
                int(np.clip(c[2], 0, 255)))


# ─── SpeedEstimator ───────────────────────────────────────────────────────────

class SpeedEstimator:
    """
    Tracks tape position over time and estimates playback speed.

    For the strip format: position = frame_idx (column index in strip space).
    Speed = Δposition / Δtime_in_frames * (audio_fps / video_fps).

    For webcam mode: position is detected from the strip column in the camera view.
    For simulate mode: position is driven directly by the simulation parameters.
    """

    def __init__(self, audio_fps: float = SAMPLE_RATE / HOP):
        self.audio_fps  = audio_fps     # nominal audio frames per second (≈43.07)
        self._positions: list[tuple[float, float]] = []   # (time_s, frame_idx)
        self._window    = 0.25          # estimation window (seconds)

    def push(self, frame_idx: float, t: float):
        """Record a (frame_idx, time) sample."""
        self._positions.append((t, frame_idx))
        # Trim old samples outside window
        cutoff = t - self._window
        self._positions = [(ts, fi) for ts, fi in self._positions if ts >= cutoff]

    def speed(self) -> float:
        """
        Current speed ratio relative to nominal playback (1.0 = nominal).
        Positive = forward, negative = reverse, 0 = stopped.
        Returns 1.0 if not enough data.
        """
        if len(self._positions) < 2:
            return 1.0
        t0, fi0 = self._positions[0]
        t1, fi1 = self._positions[-1]
        dt = t1 - t0
        if dt < 1e-4:
            return 0.0
        delta_fi = fi1 - fi0
        # Speed: audio frames advanced per second / nominal audio fps
        return (delta_fi / dt) / self.audio_fps

    def direction(self) -> int:
        """1 = forward, -1 = reverse, 0 = stopped."""
        s = self.speed()
        if abs(s) < 0.02:
            return 0
        return 1 if s > 0 else -1


# ─── TapeDisplay ──────────────────────────────────────────────────────────────

class TapeDisplay:
    """
    Scrolling terminal tape-head display.
    Shows the last `window` rows (audio frames) that passed through the
    camera window, left=bass, right=treble.

    One character block = 2 terminal columns = one frequency band.
    48 bands × 2 = 96 terminal columns wide.
    """

    def __init__(self, window: int = 20, n_bands: int = 48):
        self.window   = window
        self.n_bands  = n_bands
        self._rows: list[list[tuple[int, int, int]]] = []
        self._frame_count = 0
        self._t_start     = time.time()

    def push_frame(self, band_colors: list[tuple[int, int, int]],
                   frame_idx: int, alpha_mean: float, speed: float):
        """Add one audio frame to the display buffer."""
        self._rows.append(band_colors)
        if len(self._rows) > self.window:
            self._rows.pop(0)
        self._frame_count += 1

    def render(self, frame_idx: int, alpha_mean: float, speed: float) -> str:
        """Build the full display string."""
        lines = []

        # Header
        elapsed = time.time() - self._t_start
        dir_sym = '▶' if speed >= 0.02 else ('◀' if speed <= -0.02 else '■')
        lines.append(
            f"\033[2J\033[H"    # clear screen
            f"  DSA — Live Strip Reader   "
            f"frame {frame_idx:>6}  "
            f"{dir_sym} {abs(speed):.2f}×   "
            f"α={alpha_mean:.3f}   "
            f"t={elapsed:.1f}s"
        )
        lines.append("  " + "─" * 96)

        # Band labels top
        l0_w = L0_END * 2
        l1_w = (L1_END - L0_END) * 2
        l2_w = (self.n_bands - L1_END) * 2
        lines.append(
            f"  {'L0 bass':<{l0_w}}{'L1 mid':<{l1_w}}{'L2 high':>{l2_w}}"
        )
        lines.append("  " + "─" * 96)

        # Tape rows — oldest at top, newest at bottom (newest = NOW)
        # Rows are ordered so the "camera window" is always the BOTTOM row.
        for i, row in enumerate(self._rows):
            is_now = (i == len(self._rows) - 1)
            prefix = "▶ " if is_now else "  "
            lines.append(prefix + _render_tape_row(row))

        lines.append("  " + "─" * 96)
        lines.append(
            f"  L0 α: {alpha_mean:.3f}  "
            f"bands: {'█' * int(alpha_mean * 48):48s}  "
        )

        return '\n'.join(lines)


# ─── Audio output ──────────────────────────────────────────────────────────────

def _try_sounddevice():
    try:
        import sounddevice as sd
        return sd
    except ImportError:
        return None


class AudioOutput:
    """
    Real-time audio output via sounddevice, or buffered WAV fallback.
    Buffers HOP samples per push(); plays back continuously.
    """

    def __init__(self, sample_rate: int, out_path: str | None = None):
        self.sample_rate  = sample_rate
        self.out_path     = out_path
        self._sd          = _try_sounddevice() if out_path is None else None
        self._buffer      = []
        self._stream      = None
        self._buf_q: list[np.ndarray] = []
        self._initialized = False

    def start(self):
        if self._sd and self.out_path is None:
            import queue
            self._q = queue.Queue(maxsize=16)

            def _callback(outdata, frames, time_info, status):
                try:
                    data = self._q.get_nowait()
                except Exception:
                    data = np.zeros(frames, dtype=np.float32)
                n = min(len(data), frames)
                outdata[:n, 0]   = data[:n].astype(np.float32)
                outdata[n:, 0]   = 0.0

            self._stream = self._sd.OutputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32',
                blocksize=HOP,
                callback=_callback,
            )
            self._stream.start()
            self._initialized = True
            print("  Audio: sounddevice stream started")
        else:
            self._initialized = True
            if self.out_path:
                print(f"  Audio: buffering to WAV → {self.out_path}")
            else:
                print("  Audio: sounddevice not available — buffering to WAV")

    def push(self, samples: np.ndarray):
        """Push HOP samples. Plays or buffers."""
        if self._stream is not None:
            try:
                self._q.put_nowait(samples)
            except Exception:
                pass   # drop frame if queue full
        else:
            self._buffer.append(samples)

    def stop(self):
        if self._stream:
            self._stream.stop()
            self._stream.close()
        if self._buffer and (self.out_path or self._sd is None):
            import scipy.io.wavfile as wav
            out = self.out_path or '/tmp/dsa_live_out.wav'
            all_samples = np.concatenate(self._buffer)
            s16 = np.clip(all_samples * 32767, -32768, 32767).astype(np.int16)
            wav.write(out, self.sample_rate, s16)
            print(f"  WAV written → {out}  ({len(all_samples)/self.sample_rate:.1f}s)")


# ─── Simulate mode ─────────────────────────────────────────────────────────────

def run_simulate(strip_path: str, layout_path: str, dsa_path: str,
                 speed: float = 1.0, reverse: bool = False,
                 cell_h: int = 8, border_px: int = 0,
                 frame_start: int = 0, frame_end: int | None = None,
                 out_wav: str | None = None,
                 display: bool = True,
                 fps: float = 30.0):
    """
    Simulate the tape advancing past the camera window at `speed` × nominal.

    speed   : 1.0 = nominal (43.07 frames/s), 2.0 = double speed + pitch, etc.
    reverse : read right-to-left (reverse scratch)
    fps     : simulated video frame rate (controls display update rate)
    """
    print()
    print("  DSA — Digilog Scalable Audio")
    print("  Live Strip Reader  [simulate mode]")
    print("  ─────────────────────────────────────────────────")
    print()

    # ── Load strip image ──────────────────────────────────────────────────────
    print(f"  Loading strip  {Path(strip_path).name} ...", end=' ', flush=True)
    pil = PILImage.open(strip_path).convert('RGB')
    strip_arr = np.array(pil, dtype=np.uint8)
    print(f"done  ({pil.width}×{pil.height}px)")

    print(f"  Loading layout {Path(layout_path).name} ...", end=' ', flush=True)
    with open(layout_path) as f:
        layout = json.load(f)
    n_frames = layout['n_frames']
    n_bands  = layout['n_bands']
    print(f"done  ({n_frames} frames, {layout['duration_s']:.1f}s)")

    print(f"  Loading .dsa   {Path(dsa_path).name} ...", end=' ', flush=True)
    dsa_bytes = Path(dsa_path).read_bytes()
    print("done")
    print()

    # ── Init subsystems ───────────────────────────────────────────────────────
    sampler  = StripColumnSampler(strip_arr, layout, cell_h=cell_h, border_px=border_px)
    decoder  = DSAStreamDecoder(dsa_bytes)
    audio    = AudioOutput(decoder.sample_rate, out_path=out_wav)
    display_ = TapeDisplay(window=20, n_bands=n_bands) if display else None
    speed_est = SpeedEstimator()

    if frame_end is None:
        frame_end = n_frames
    frame_end = min(frame_end, n_frames)

    frame_range = range(frame_start, frame_end)
    if reverse:
        frame_range = range(frame_end - 1, frame_start - 1, -1)

    # Audio frames advance faster than video frames when speed > 1
    # We generate audio for ALL frames but display only at fps
    audio_fps_nominal = SAMPLE_RATE / HOP   # ≈ 43.07

    print(f"  Speed: {speed:.2f}×  {'reverse' if reverse else 'forward'}  "
          f"{'display ON' if display else 'display OFF'}")
    print(f"  Frame range: {frame_start}–{frame_end}  ({frame_end - frame_start} frames)")
    print()

    audio.start()
    t_start  = time.time()

    for i, fi in enumerate(frame_range):
        # ── Sample the strip column ──────────────────────────────────────────
        alpha, steepness, direction = sampler.sample_column(fi)

        # ── Decode audio frame ────────────────────────────────────────────────
        samples = decoder.decode_frame(fi, alpha=alpha)

        # Variable speed: resample the HOP samples
        if abs(speed - 1.0) > 0.01:
            n_out = max(1, int(round(HOP / speed)))
            from numpy import interp
            x_in  = np.linspace(0, 1, HOP)
            x_out = np.linspace(0, 1, n_out)
            samples = interp(x_out, x_in, samples)

        audio.push(samples)

        # ── Speed estimator ───────────────────────────────────────────────────
        t_now = time.time()
        speed_est.push(fi, t_now)

        # ── Display update ─────────────────────────────────────────────────────
        if display_ is not None:
            # Display at ~fps (not every audio frame)
            display_every = max(1, int(audio_fps_nominal / fps))
            if i % display_every == 0:
                band_colors = [sampler.band_color_at(fi, b) for b in range(n_bands)]
                alpha_mean  = float(alpha.mean())
                display_.push_frame(band_colors, fi, alpha_mean, speed)
                print(display_.render(fi, alpha_mean, speed))

        # ── Timing: real-time pacing for simulation ───────────────────────────
        # Sleep to match nominal audio rate when display is on
        if display_ is not None and out_wav is None:
            expected_t = t_start + (i + 1) * HOP / (SAMPLE_RATE * speed)
            slack = expected_t - time.time()
            if slack > 0.001:
                time.sleep(slack)

    audio.stop()
    print()
    print("  Playback complete.")
    print()


# ─── Scan mode ────────────────────────────────────────────────────────────────

def run_scan(strip_path: str, layout_path: str, dsa_path: str,
             cell_h: int = 8, border_px: int = 0,
             out_wav: str | None = None):
    """
    Read the full strip from left to right at nominal speed.
    No display. Fast batch operation with audio output.
    Equivalent to dsa_camera.py but with audio output.
    """
    run_simulate(strip_path, layout_path, dsa_path,
                 speed=1.0, reverse=False,
                 cell_h=cell_h, border_px=border_px,
                 out_wav=out_wav, display=False)


# ─── Webcam mode (stub) ────────────────────────────────────────────────────────

def run_webcam(layout_path: str, dsa_path: str):
    """
    Live webcam reader — planned, not yet implemented.

    Architecture (for implementation):
      1. OpenCV VideoCapture(0) → grab video frames at camera fps
      2. For each video frame:
         a. Detect strip region (cached homography from first good frame)
         b. Detect current column: compare center column to last frame
            via optical flow or direct pixel correlation
         c. Update SpeedEstimator with (detected_frame_idx, current_time)
         d. sample_column(detected_frame_idx) → alpha
         e. decoder.decode_frame(detected_frame_idx, alpha) → samples
         f. audio.push(samples)
         g. display.push_frame(...)
      3. Physical scratch → frame_idx jumps → decoder.reset() → brief transient
    """
    print()
    print("  Webcam mode is planned but not yet implemented.")
    print()
    print("  Planned pipeline:")
    print("    1. OpenCV VideoCapture → video frames at camera fps")
    print("    2. Homography detection (one-time, from fiducial corners)")
    print("    3. Optical flow → column displacement → speed estimate")
    print("    4. StripColumnSampler.sample_column(detected_fi) → alpha")
    print("    5. DSAStreamDecoder.decode_frame(fi, alpha) → audio samples")
    print("    6. sounddevice real-time output + TapeDisplay scrolling strip")
    print()
    print("  Timecode chain:")
    print("    column displacement → SpeedEstimator → speed ratio")
    print("    column x-position   → frame_idx → absolute audio position")
    print("    (K-frame every 8 frames = resync within 185ms after any jump)")
    print()
    print("  Use 'simulate' mode to test the full pipeline without a camera.")
    print()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description='DSA live strip reader — single-column sampler, real-time audio',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split('License')[0].strip())

    sub = p.add_subparsers(dest='mode', metavar='mode')
    sub.required = True

    # ── simulate ──
    s = sub.add_parser('simulate', help='simulate tape advancing (no camera required)')
    s.add_argument('strip',   help='strip PNG (from dsa_strip.py, with or without fiducials)')
    s.add_argument('layout',  help='.disc.json layout file')
    s.add_argument('dsa',     help='.dsa bitstream file')
    s.add_argument('--speed',    type=float, default=1.0,
                   help='playback speed multiplier (default: 1.0; negative = reverse)')
    s.add_argument('--reverse',  action='store_true',
                   help='read strip right-to-left (reverse playback)')
    s.add_argument('--cell-h',   type=int,   default=8,
                   help='pixel height per band row (default: 8)')
    s.add_argument('--border',   type=int,   default=0,
                   help='fiducial border width in pixels (default: 0 = no border)')
    s.add_argument('--start',    type=int,   default=0,
                   help='first frame to read (default: 0)')
    s.add_argument('--end',      type=int,   default=None,
                   help='last frame (exclusive; default: all)')
    s.add_argument('--out',      type=str,   default=None,
                   help='save decoded audio to WAV (skips real-time pacing)')
    s.add_argument('--no-display', action='store_true',
                   help='disable terminal display (faster — audio only)')
    s.add_argument('--fps',      type=float, default=30.0,
                   help='display update rate in frames/s (default: 30)')

    # ── scan ──
    sc = sub.add_parser('scan', help='read full strip at nominal speed, output WAV')
    sc.add_argument('strip',   help='strip PNG')
    sc.add_argument('layout',  help='.disc.json layout file')
    sc.add_argument('dsa',     help='.dsa bitstream file')
    sc.add_argument('--cell-h',  type=int, default=8)
    sc.add_argument('--border',  type=int, default=0)
    sc.add_argument('--out',     type=str, default=None,
                    help='output WAV path (default: <dsa>.live.wav)')

    # ── webcam ──
    wc = sub.add_parser('webcam', help='live camera reader [planned]')
    wc.add_argument('layout',  help='.disc.json layout file')
    wc.add_argument('dsa',     help='.dsa bitstream file')

    args = p.parse_args()

    if args.mode == 'simulate':
        reverse = args.reverse or (args.speed < 0)
        speed   = abs(args.speed)
        run_simulate(args.strip, args.layout, args.dsa,
                     speed=speed, reverse=reverse,
                     cell_h=args.cell_h, border_px=args.border,
                     frame_start=args.start, frame_end=args.end,
                     out_wav=args.out,
                     display=not args.no_display,
                     fps=args.fps)

    elif args.mode == 'scan':
        out = args.out or str(Path(args.dsa).with_suffix('.live.wav'))
        run_scan(args.strip, args.layout, args.dsa,
                 cell_h=args.cell_h, border_px=args.border,
                 out_wav=out)

    elif args.mode == 'webcam':
        run_webcam(args.layout, args.dsa)


if __name__ == '__main__':
    main()
