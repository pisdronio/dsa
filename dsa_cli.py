#!/usr/bin/env python3
"""
dsa — Digilog Scalable Audio  unified command-line interface

Subcommands
-----------
  encode   audio/PCM → .dsa bitstream
  decode   .dsa (or audio) → WAV file(s)
  disc     audio/.dsa → Digilog disc layout JSON
  bench    benchmark DSA vs Opus at multiple bitrates
  info     print header info for a .dsa file
  camera   read a photograph of a printed strip → confidence + accuracy report

Usage
-----
  python3 dsa_cli.py encode song.mp3 -o song.dsa --bitrate 32
  python3 dsa_cli.py decode song.dsa
  python3 dsa_cli.py decode song.dsa --layers 0 --reverse --output song.rev.wav
  python3 dsa_cli.py disc   song.mp3 --bitrate 12 --mode 2 --per-coeff
  python3 dsa_cli.py bench  --quick
  python3 dsa_cli.py info   song.dsa
  python3 dsa_cli.py camera photo.jpg song.disc.json
  python3 dsa_cli.py camera photo.jpg song.disc.json --save-warped warped.png
"""

import argparse
import sys
from pathlib import Path

import numpy as np


# ─── helpers ──────────────────────────────────────────────────────────────────

def _die(msg: str):
    print(f"  error: {msg}", file=sys.stderr)
    sys.exit(1)


def _require_file(path: str):
    if not Path(path).exists():
        _die(f"file not found: {path}")


def _banner():
    print(f"\n  DSA — Digilog Scalable Audio")
    print(f"  {'─'*48}")


# ─── encode ───────────────────────────────────────────────────────────────────

def cmd_encode(args):
    from dsa_bitstream import DSABitstreamWriter, MODE_DISCRETE, MODE_GRADIENT

    _require_file(args.input)
    mode    = MODE_GRADIENT if args.mode == 2 else MODE_DISCRETE
    outpath = args.output or str(Path(args.input).with_suffix('.dsa'))

    _banner()
    print(f"  encode")
    print(f"  {'─'*48}")
    print(f"  Input:    {args.input}")
    print(f"  Output:   {outpath}")
    print(f"  Bitrate:  {args.bitrate} kbps")
    print(f"  Mode:     {'gradient (Mode 2)' if mode == MODE_GRADIENT else 'discrete (Mode 1)'}")

    writer = DSABitstreamWriter(bitrate_kbps=args.bitrate, mode=mode)
    data   = writer.encode_file(args.input, outpath)

    from dsa_bitstream import DSABitstreamReader
    reader = DSABitstreamReader(data)
    hdr    = reader.header

    print(f"\n  DSA file written:")
    print(f"    Frames:         {hdr.n_frames:,}")
    print(f"    Duration:       {reader.duration_s:.1f}s")
    print(f"    File size:      {reader.file_size:,} bytes  "
          f"({reader.file_size * 8 / reader.duration_s / 1000:.1f} kbps effective)")
    print(f"    CRC32:          {'OK ✓' if reader.verify_crc() else 'FAIL ✗'}")
    print(f"\n  Scan the groove.\n")


# ─── decode ───────────────────────────────────────────────────────────────────

def cmd_decode(args):
    from dsa_bitstream import DSABitstreamWriter, DSABitstreamReader, DEFAULT_BITRATE
    from dsa_decoder   import DSADecoder
    from dsa_analyzer  import NUM_BANDS

    _require_file(args.input)

    # Accept audio or .dsa
    in_path = args.input
    if not in_path.endswith('.dsa'):
        kbps     = args.bitrate
        dsa_path = args.output or str(Path(in_path).with_suffix('.dsa'))
        _banner()
        print(f"  encode → decode")
        print(f"  {'─'*48}")
        print(f"  Encoding {in_path}  @ {kbps}kbps  →  {dsa_path}")
        DSABitstreamWriter(kbps).encode_file(in_path, dsa_path)
        in_path = dsa_path

    data   = Path(in_path).read_bytes()
    reader = DSABitstreamReader(data)
    dec    = DSADecoder()

    # Build decode kwargs
    layers  = tuple(args.layers) if args.layers else (0, 1, 2)
    alpha   = None
    if args.alpha is not None:
        alpha = np.full(NUM_BANDS, args.alpha, dtype=np.float64)

    _banner()
    print(f"  decode")
    print(f"  {'─'*48}")
    print(f"  Input:    {in_path}  ({reader.duration_s:.1f}s)")
    print(f"  Layers:   {layers}")
    if args.reverse:
        print(f"  Mode:     reverse playback")
    if args.speed != 1.0:
        print(f"  Speed:    {args.speed}× {'(stopped — decay)' if args.speed == 0.0 else ''}")
    if alpha is not None:
        print(f"  Alpha:    {args.alpha:.2f} (analog degradation)")

    kwargs = dict(layers=layers, reverse=args.reverse, speed=args.speed)
    if alpha is not None:
        kwargs['alpha'] = alpha

    samples  = dec.decode(reader, **kwargs)
    out_path = args.output or str(Path(in_path).with_suffix('.wav'))

    from dsa_decoder import _write_wav
    _write_wav(out_path, samples, reader.header.sample_rate)
    dur = len(samples) / reader.header.sample_rate
    print(f"\n  Output:   {out_path}  ({dur:.1f}s  {len(samples):,} samples)")
    print(f"\n  Scan the groove.\n")


# ─── disc ─────────────────────────────────────────────────────────────────────

def cmd_disc(args):
    from dsa_disc import DSADiscEncoder, MODE_DISCRETE, MODE_GRADIENT

    _require_file(args.input)

    in_path = args.input
    if in_path.endswith('.dsa'):
        out_path = args.output or in_path.replace('.dsa', '.disc.json')
    else:
        out_path = args.output or str(Path(in_path).with_suffix('.disc.json'))

    _banner()
    print(f"  disc")
    print(f"  {'─'*48}")
    print(f"  Input:    {in_path}")
    print(f"  Output:   {out_path}")
    print(f"  Bitrate:  {args.bitrate} kbps")
    print(f"  Mode:     {'gradient (Mode 2)' if args.mode == 2 else 'discrete (Mode 1)'}")
    if args.layers:
        print(f"  Layers:   {tuple(args.layers)}")

    layers = tuple(args.layers) if args.layers else (0, 1, 2)
    enc    = DSADiscEncoder()
    mode   = MODE_GRADIENT if args.mode == 2 else MODE_DISCRETE

    layout = enc.encode_file(in_path, out_path,
                             bitrate_kbps=args.bitrate,
                             layers=layers,
                             per_coeff=args.per_coeff,
                             mode=mode)

    l0c, l1c, l2c = layout.layer_dot_counts()
    print(f"\n  Disc layout:")
    print(f"    Frames:    {layout.n_frames:,}  ({layout.duration_s:.1f}s)")
    print(f"    L0 dots:   {l0c:,}  (inner rings, bass)")
    print(f"    L1 dots:   {l1c:,}  (middle rings, mid)")
    print(f"    L2 dots:   {l2c:,}  (outer rings, high)")
    print(f"    Total:     {l0c + l1c + l2c:,}")
    print(f"    Mode:      {layout.mode_name}")
    print(f"    Output:    {out_path}")
    print(f"\n  Scan the groove.\n")


# ─── bench ────────────────────────────────────────────────────────────────────

def cmd_bench(args):
    from dsa_bench import QUICK_SIGNALS, FULL_SIGNALS, BITRATES, run_bench

    sigs = QUICK_SIGNALS if args.quick else FULL_SIGNALS
    run_bench(sigs, BITRATES, csv_mode=args.csv)


# ─── info ─────────────────────────────────────────────────────────────────────

def cmd_info(args):
    from dsa_bitstream import DSABitstreamReader

    _require_file(args.input)
    data   = Path(args.input).read_bytes()
    reader = DSABitstreamReader(data)
    hdr    = reader.header

    _banner()
    print(f"  info: {args.input}")
    print(f"  {'─'*48}")
    print(f"  Format:         DSA1 v{hdr.version}")
    print(f"  Frames:         {hdr.n_frames:,}")
    print(f"  Duration:       {reader.duration_s:.2f}s")
    print(f"  Sample rate:    {hdr.sample_rate} Hz")
    print(f"  Bitrate:        {hdr.bitrate_kbps} kbps (nominal)")
    print(f"  Effective:      {reader.file_size * 8 / reader.duration_s / 1000:.1f} kbps")
    print(f"  File size:      {reader.file_size:,} bytes")
    print(f"  Mode:           {hdr.mode_name}")
    print(f"  CRC32:          {'OK ✓' if reader.verify_crc() else 'FAIL ✗'}")
    print(f"\n  Layer breakdown:")
    for li, name in enumerate(['L0 (bass,   8 bands)', 'L1 (mid,   16 bands)',
                               'L2 (high,  24 bands)']):
        sz  = reader.layer_byte_size(li)
        pct = 100 * sz / reader.file_size
        print(f"    {name}:  {sz:,} bytes  ({pct:.1f}%)")
    print()


# ─── camera ───────────────────────────────────────────────────────────────────

def cmd_camera(args):
    import dsa_camera as cam

    cam.main.__module__  # ensure import worked
    # Reconstruct argv so dsa_camera.main() parses correctly
    import sys as _sys
    old_argv = _sys.argv
    _sys.argv = ['dsa_camera.py', args.photo, args.layout]
    if args.cell_h != 8:
        _sys.argv += ['--cell-h', str(args.cell_h)]
    if args.border is not None:
        _sys.argv += ['--border', str(args.border)]
    if args.strip_dpi != 300:
        _sys.argv += ['--strip-dpi', str(args.strip_dpi)]
    if args.corners:
        _sys.argv += ['--corners', args.corners]
    if args.no_cv:
        _sys.argv.append('--no-cv')
    if args.corner_frac != 0.08:
        _sys.argv += ['--corner-frac', str(args.corner_frac)]
    if args.debug_detect:
        _sys.argv.append('--debug-detect')
    if args.save_warped:
        _sys.argv += ['--save-warped', args.save_warped]
    if args.out_overlay:
        _sys.argv += ['--out-overlay', args.out_overlay]
    if args.conf_map:
        _sys.argv += ['--conf-map', args.conf_map]
    if args.out_json:
        _sys.argv += ['--out-json', args.out_json]
    if args.decode:
        _sys.argv += ['--decode', args.decode]
    if args.out:
        _sys.argv += ['--out', args.out]
    try:
        cam.main()
    finally:
        _sys.argv = old_argv


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog='dsa',
        description='DSA — Digilog Scalable Audio codec',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  dsa encode song.mp3                               encode at default 12 kbps
  dsa encode song.mp3 -b 32 -o song.dsa            32 kbps, explicit output
  dsa encode song.mp3 --mode 2                      gradient Mode 2 encoding
  dsa decode song.dsa                               decode to song.wav
  dsa decode song.dsa --reverse                     reverse playback
  dsa decode song.dsa --layers 0                    bass-only (L0)
  dsa decode song.dsa --alpha 0.3                   analog degradation simulation
  dsa disc   song.mp3                               disc layout JSON at 12 kbps
  dsa disc   song.mp3 -b 96 --mode 2                high-bitrate gradient layout
  dsa disc   song.mp3 --per-coeff                   per-coefficient JSON
  dsa bench  --quick                                fast benchmark vs Opus
  dsa bench  --csv > results.csv                    CSV output for plotting
  dsa info   song.dsa                               print .dsa file metadata
  dsa camera photo.jpg song.disc.json               read physical strip photo
  dsa camera photo.jpg song.disc.json --save-warped warped.png
  dsa camera photo.jpg song.disc.json --corners "10,12 820,8 825,610 5,615"
        """,
    )

    sub = parser.add_subparsers(dest='command', metavar='command')
    sub.required = True

    # ── encode ──
    p_enc = sub.add_parser('encode', help='encode audio to .dsa')
    p_enc.add_argument('input',            help='input audio file (mp3, wav, flac, …)')
    p_enc.add_argument('-o', '--output',   help='output .dsa path (default: <input>.dsa)')
    p_enc.add_argument('-b', '--bitrate',  type=float, default=12.0, metavar='KBPS',
                       help='target bitrate in kbps (default: 12)')
    p_enc.add_argument('--mode',           type=int, default=1, choices=[1, 2],
                       help='1=discrete (default), 2=gradient')

    # ── decode ──
    p_dec = sub.add_parser('decode', help='decode .dsa to WAV')
    p_dec.add_argument('input',            help='.dsa file (or audio to encode-then-decode)')
    p_dec.add_argument('-o', '--output',   help='output WAV path (default: <input>.wav)')
    p_dec.add_argument('-b', '--bitrate',  type=float, default=12.0, metavar='KBPS',
                       help='encode bitrate if input is audio (default: 12)')
    p_dec.add_argument('--layers',         type=int, nargs='+', choices=[0, 1, 2],
                       metavar='N', help='layers to decode: 0=bass 1=mid 2=high (default: all)')
    p_dec.add_argument('--reverse',        action='store_true',
                       help='reverse playback (frame-level temporal reversal)')
    p_dec.add_argument('--alpha',          type=float, metavar='A',
                       help='analog degradation confidence α ∈ [0,1] applied uniformly')
    p_dec.add_argument('--speed',          type=float, default=1.0, metavar='S',
                       help='playback speed multiplier (default: 1.0); '
                            '>1 faster+higher pitch, <1 slower+lower, 0 decay, '
                            'negative = reverse at |speed|')

    # ── disc ──
    p_disc = sub.add_parser('disc', help='generate Digilog disc layout JSON')
    p_disc.add_argument('input',            help='audio or .dsa file')
    p_disc.add_argument('-o', '--output',   help='output .disc.json path')
    p_disc.add_argument('-b', '--bitrate',  type=float, default=12.0, metavar='KBPS',
                        help='encode bitrate (default: 12)')
    p_disc.add_argument('--mode',           type=int, default=1, choices=[1, 2],
                        help='1=discrete (default), 2=gradient')
    p_disc.add_argument('--layers',         type=int, nargs='+', choices=[0, 1, 2],
                        metavar='N', help='layers to include (default: all)')
    p_disc.add_argument('--per-coeff',      action='store_true',
                        help='per-coefficient JSON (default: band summary)')

    # ── bench ──
    p_ben = sub.add_parser('bench', help='benchmark DSA vs Opus')
    p_ben.add_argument('--quick',  action='store_true', help='fewer signals, faster run')
    p_ben.add_argument('--csv',    action='store_true', help='CSV output for plotting')

    # ── info ──
    p_inf = sub.add_parser('info', help='print .dsa file metadata')
    p_inf.add_argument('input', help='.dsa file')

    # ── camera ──
    p_cam = sub.add_parser('camera',
                            help='read a photo of a printed strip — Tier 1 physical validation')
    p_cam.add_argument('photo',   help='photo of printed strip (JPG/PNG)')
    p_cam.add_argument('layout',  help='.disc.json layout file')
    p_cam.add_argument('--cell-h',     type=int,   default=8,
                       help='pixel height per band row used when printing (default: 8)')
    p_cam.add_argument('--border',     type=int,   default=None,
                       help='border width px (default: auto from --strip-dpi)')
    p_cam.add_argument('--strip-dpi',  type=int,   default=300,
                       help='DPI used when printing (default: 300)')
    p_cam.add_argument('--corners',    type=str,   default=None,
                       metavar='"x0,y0 x1,y1 x2,y2 x3,y3"',
                       help='manual TL TR BR BL corner positions in photo (skips auto-detect)')
    p_cam.add_argument('--no-cv',      action='store_true',
                       help='skip OpenCV (requires --corners)')
    p_cam.add_argument('--corner-frac', type=float, default=0.08,
                       help='expected corner square size as fraction of shorter image side '
                            '(default: 0.08; reduce if corners are small)')
    p_cam.add_argument('--debug-detect', action='store_true',
                       help='print fiducial detection debug info')
    p_cam.add_argument('--save-warped',  type=str,  default=None,
                       help='save rectified strip image to PATH')
    p_cam.add_argument('--out-overlay',  type=str,  default=None,
                       help='save confidence overlay on warped strip to PATH')
    p_cam.add_argument('--conf-map',     type=str,  default=None,
                       help='save compact confidence heatmap to PATH')
    p_cam.add_argument('--out-json',     type=str,  default=None,
                       help='save read results JSON to PATH')
    p_cam.add_argument('--decode',       type=str,  default=None,
                       help='.dsa bitstream to decode using camera read confidence')
    p_cam.add_argument('-o', '--out',    type=str,  default=None,
                       help='decoded WAV output path (requires --decode)')

    args = parser.parse_args()

    dispatch = {
        'encode': cmd_encode,
        'decode': cmd_decode,
        'disc':   cmd_disc,
        'bench':  cmd_bench,
        'info':   cmd_info,
        'camera': cmd_camera,
    }
    dispatch[args.command](args)


if __name__ == '__main__':
    main()
