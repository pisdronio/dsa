# DSA — Digilog Scalable Audio

**An open, layered audio codec designed for physical media and real-time motion-based playback.**

> Part of the [Digilog](https://github.com/pisdronio/digilog-spec) open physical audio format.

---

## What DSA is

DSA is an audio codec built from first principles for a specific purpose: encoding audio into a physical printed medium that can be played by scanning with a camera, including in real time while the medium is in motion — spinning on a turntable, scratched by a DJ, or hand-scanned across a surface.

No existing open codec was designed for this. DSA is.

### What makes DSA different from Opus, MP3, AAC

| Property | MP3 / AAC / Opus | DSA |
|---|---|---|
| Designed for streaming | Yes | No |
| Designed for physical media | No | Yes |
| Layered scalable decoding | No | Yes |
| Native reverse playback | No | Yes |
| Motion-aware (variable speed) | No | Yes |
| Graceful analog degradation | No | Yes |
| Maps to physical disc rings | No | Yes |

---

## The core ideas

### Layered encoding

DSA encodes audio in three independent layers that map directly to physical regions of a Digilog disc:

```
Layer 0  →  inner disc rings  →  always readable
Layer 1  →  middle rings      →  readable on average camera
Layer 2  →  outer rings       →  readable with rig + controlled light
```

A damaged disc, a cheap camera, a hand covering part of the surface — you always get what's physically readable. Layer 0 alone plays the bass and fundamental frequencies. All three layers play full quality audio.

This is not three separate encodes. It is one audio stream encoded hierarchically — each layer genuinely enhances the previous.

### Native reverse playback

DSA uses the Modified Discrete Cosine Transform (MDCT) with a symmetric sine window and a bidirectional frame structure (K-frames and B-frames). Decoding frames in reverse order produces the true time-reversed audio signal.

A DJ scratching a Digilog disc backward hears the music in reverse — like vinyl. Not corruption. Not silence. Reversed audio.

### Analog degradation model

Standard codecs fail in binary — they either decode or they don't. DSA degrades gracefully. Partial layer reads produce lower quality audio. Partial frame reads produce filtered audio. The degradation sounds like worn vinyl, not digital dropout.

---

## Technical foundation

### Transform

**MDCT** (Modified Discrete Cosine Transform)

- Window length N = 2048 samples
- Output coefficients M = 1024 (N/2)
- Hop size = 1024 samples (50% overlap)
- Frame duration = 23.2ms at 44100Hz
- Sine window — satisfies Princen-Bradley condition
- TDAC reconstruction SNR: 238dB (verified)

### Frame structure

```
GOP (Group of Pictures) = 8 frames (~185ms)

K . B . B . B . B . B . B . B . K . B . B ...
↑                               ↑
Keyframe                     Keyframe
Self-contained               Self-contained
Drop needle here             Resync point

B-frames: bidirectional — reference surrounding K-frames
          decode cleanly in forward or reverse direction
          scratch recovery: ≤185ms to nearest K-frame
```

### Frequency bands

48 perceptual bands across 3 layers:

```
Layer 0:  8 bands   20Hz –   800Hz   linear spacing    bass, fundamental
Layer 1: 16 bands  800Hz –  6000Hz   log spacing       melody, voice, rhythm
Layer 2: 24 bands  6000Hz – 22050Hz  log spacing       presence, air, detail
```

Band spacing follows human auditory perception:
- Linear below 800Hz (bass frequencies need even resolution)
- Logarithmic above 800Hz (ear perceives pitch logarithmically)

### Perceptual weighting (ISO 226 simplified)

```
< 100Hz:    weight 0.25   sub-bass, felt more than heard
100–300Hz:  weight 0.55   bass
300Hz–1kHz: weight 0.80   low-mid
1–4kHz:     weight 1.00   peak sensitivity — most important
4–8kHz:     weight 0.75   presence
8–12kHz:    weight 0.45   air
> 12kHz:    weight 0.20   very high
```

---

## Status

```
Step 1  ✓  MDCT Frame Analyzer           — complete, verified
Step 2  ✓  Perceptual Quantizer          — complete, verified
Step 3  ✓  K-frame encoder               — complete, verified
Step 4  ✓  B-frame encoder               — complete, verified
Step 5  ✓  Huffman entropy coder         — complete, verified
Step 6  ✓  Layered bitstream packer      — complete, verified
Step 7  ✓  DSA decoder                   — complete, verified
Step 8  ✓  Digilog disc encoder interface — complete, verified

Tools  ✓  Unified CLI (dsa_cli.py)        — encode / decode / disc / bench / info
       ✓  Mode 2 gradient encoding        — continuous per-band steepness
       ✓  Variable-speed decoder          — 0.25×–4× with pitch coupling
       ✓  Benchmark suite (dsa_bench.py)  — DSA vs Opus at 6–96 kbps
       ✓  Test suite (74 tests)           — unit + integration, all pass
```

---

## Repository structure

```
dsa/
├── README.md            — this file
├── LICENSE              — GPL v3
├── .gitignore
├── RESEARCH.md          — scientific documentation
├── dsa_analyzer.py      — Step 1: MDCT frame analyzer ✓
├── dsa_quantizer.py     — Step 2: perceptual quantizer ✓
├── dsa_encoder.py       — Steps 3–4: K/B-frame encoders ✓
├── dsa_huffman.py       — Step 5: Huffman entropy coder ✓
├── dsa_bitstream.py     — Step 6: layered bitstream packer ✓
├── dsa_decoder.py       — Step 7: decoder (+ variable-speed) ✓
├── dsa_disc.py          — Step 8: Digilog disc encoder interface ✓
├── dsa_cli.py           — unified CLI: encode / decode / disc / bench / info ✓
├── dsa_bench.py         — benchmark suite: DSA vs Opus ✓
└── tests/
    ├── test_analyzer.py   — Step 1 unit tests (MDCT, bands, analyzer) ✓
    └── test_pipeline.py   — integration tests, full pipeline Steps 1–8 ✓
```

---

## Getting started

### Requirements

```bash
python3 --version        # 3.10+
pip install numpy scipy  # core dependencies
apt install ffmpeg       # audio loading
```

### Unified CLI

```bash
git clone https://github.com/pisdronio/dsa
cd dsa

python3 dsa_cli.py encode  song.mp3               # encode at 12 kbps
python3 dsa_cli.py encode  song.mp3 -b 32         # 32 kbps
python3 dsa_cli.py encode  song.mp3 --mode 2      # gradient Mode 2

python3 dsa_cli.py decode  song.dsa               # decode to song.wav
python3 dsa_cli.py decode  song.dsa --reverse     # reverse playback
python3 dsa_cli.py decode  song.dsa --layers 0    # bass-only (L0)
python3 dsa_cli.py decode  song.dsa --speed 0.5   # half speed (lower pitch)
python3 dsa_cli.py decode  song.dsa --alpha 0.3   # worn-disc simulation

python3 dsa_cli.py disc    song.mp3               # disc layout JSON
python3 dsa_cli.py disc    song.mp3 -b 96 --mode 2  # high-res gradient layout

python3 dsa_cli.py info    song.dsa               # file metadata
python3 dsa_cli.py bench   --quick                # benchmark vs Opus
```

### Run the analyzer

```bash
python3 dsa_analyzer.py your_song.mp3
```

Output:

```
  DSA — Digilog Scalable Audio
  Step 1: MDCT Frame Analyzer
  ────────────────────────────────────────────────
  Transform:  MDCT  N=2048  M=1024
  Frame hop:  1024 samples  (23.2ms)
  GOP:        8 frames  (186ms per K-frame)
  Bands:      48  L0:8 L1:16 L2:24
  ────────────────────────────────────────────────

  TDAC reconstruction:  238dB  PASS ✓
  Reverse playback:     PASS ✓

  Loading: your_song.mp3
  Duration: 192.1s  |  Samples: 8,469,504
  Frames: 8,273  (23.2ms/frame)
  K=1035  B=7067  S=171
```

### Use as a library

```python
from dsa_analyzer import DSAAnalyzer, mdct, imdct

analyzer = DSAAnalyzer()
frames, sample_rate, duration = analyzer.analyze_file('song.mp3')

# Each frame has layered band energies
for frame in frames[:10]:
    print(f"Frame {frame.frame_idx} [{frame.frame_type}]")
    print(f"  Bass (L0):     {frame.layer0}")
    print(f"  Mid (L1):      {frame.layer1}")
    print(f"  Presence (L2): {frame.layer2}")
```

---

## Scientific context

DSA is developed as a component of the Digilog format — a novel physical audio format that encodes audio as printed colored dot patterns readable by a camera.

The specific requirements that motivated DSA:

1. **Layered spatial encoding** — audio layers must map to physical disc rings. Outer rings = higher quality, inner rings = always readable. No existing scalable codec was designed for this spatial constraint.

2. **Native reverse playback** — a DJ scratching a printed disc backward must hear reversed audio, not corruption. This requires symmetric frame design (B-frames) that standard codecs do not implement.

3. **Analog degradation model** — partial reads (hand covering disc, worn print, cheap camera) must produce graceful audio degradation resembling worn vinyl, not digital failure.

4. **Motion-aware decoding** — the codec must support variable-speed playback as the disc rotates at different speeds during scratch performance.

A full scientific paper documenting the design, implementation, and evaluation of DSA is in preparation. See `RESEARCH.md`.

---

## License

GNU General Public License v3.0 — see [LICENSE](./LICENSE)

DSA is free. Free to use, implement, modify, and distribute.
Any derivative must remain open. No proprietary forks.

---

## Related

- [digilog-spec](https://github.com/pisdronio/digilog-spec) — the Digilog format specification
- [Opus codec](https://opus-codec.org) — used in Digilog v0.1 before DSA is complete
- [Xiph.org](https://xiph.org) — philosophical allies, creators of Opus/Vorbis/FLAC

---

*Scan the groove.*
