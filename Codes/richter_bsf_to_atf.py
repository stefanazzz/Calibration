#!/usr/bin/env python3
"""
richter_bsf_to_atf.py

Convert Applied Seismology Consulting / Richter digitiser .bsf files to InSite-compatible ATF.

Key features:
- Reads Richter BSF waveform blocks (int32 LE), with per-channel 256-byte gaps (configurable).
- Extracts acquisition/header parameters from the BSF header when possible (TracePoints, TSamp, TimeUnits,
  TraceMaxVolts, and an ADC volts-per-count scale).
- Writes ATF with counts in [TraceData] and AmpToVolts set to the parsed (or user-provided) volts-per-count.
  This preserves small signals (e.g., Ch2 from Richter system) with no rounding loss.
- Optional CSV export (in volts).

Observed layout for your files:
- BSF has a global header; Ch1 waveform starts at byte offset 424 by default.
- Each channel waveform block: npts * 4 bytes (int32), then channel_gap_bytes (default 256) before next channel.

Examples:
  python richter_bsf_to_atf.py 04_4650.bsf
  python richter_bsf_to_atf.py 04_4650.bsf --channels 1,2 --csv
  python richter_bsf_to_atf.py 04_4650.bsf --outdir out --prefix myevent
  python richter_bsf_to_atf.py 04_4650.bsf --header-from-bsf
  python richter_bsf_to_atf.py 04_4650.bsf --no-header-from-bsf --fs 1e7 --scale 5.9604644775e-7

Notes on scaling:
- Many Richter files appear to store 24-bit ADC counts in an int32 container.
- A common scale is ±5 V full-scale: volts_per_count = 5 / 2^23 = 5.960464477539063e-7
- Some files also store a header scale like 5/32768 (16-bit-like). This script will use the BSF header's
  scale if it looks plausible, otherwise it falls back to --scale.

Stefan Nielsen March 2026
"""

from __future__ import annotations

import argparse
import re
import struct
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


DEFAULT_OFFSET = 424
DEFAULT_NCH = 4
# DEFAULT_NPTS = 8192 # changed to 65536 for the files acquired in Roma-INGV in April 2026 for calibration of piezo snesors:
DEFAULT_NPTS = 65536
DEFAULT_FS = 10e6
DEFAULT_CHANNEL_GAP_BYTES = 256

# Common ADC scale if counts are 24-bit signed stored in int32 and range is ±5V
DEFAULT_SCALE_V_PER_COUNT = 5.0 / (2**23)  # ~5.9604644775e-7 V/count


@dataclass
class BSFConfig:
    offset: int = DEFAULT_OFFSET
    nch: int = DEFAULT_NCH
    npts: int = DEFAULT_NPTS
    fs: float = DEFAULT_FS
    channel_gap_bytes: int = DEFAULT_CHANNEL_GAP_BYTES
    scale_v_per_count: float = DEFAULT_SCALE_V_PER_COUNT
    dtype: str = "<i4"  # int32 little-endian


@dataclass
class RichterBSFHeader:
    # timestamp (best-effort)
    dt0: Optional[datetime]

    # acquisition
    trace_points: Optional[int]
    tsamp: Optional[float]
    time_units: Optional[float]

    # scaling/display
    trace_max_volts: Optional[float]
    volts_per_count: Optional[float]


def parse_survey_datetime_ascii(blob: bytes) -> Optional[datetime]:
    """
    Extract timestamp from embedded ASCII like: surveyYYYYMMDDhhmmss
    """
    m = re.search(rb"survey(\d{14})", blob)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1).decode(), "%Y%m%d%H%M%S")
    except Exception:
        return None


def _safe_unpack_from(fmt: str, blob: bytes, offset: int):
    size = struct.calcsize(fmt)
    if offset < 0 or offset + size > len(blob):
        return None
    try:
        return struct.unpack_from(fmt, blob, offset)
    except Exception:
        return None


def read_richter_bsf_header(blob: bytes) -> RichterBSFHeader:
    """
    Best-effort parse of Richter BSF header parameters.

    For Ricther files the following offsets seems to work:
      - year/month/day/hour/min/sec: u32 at offsets 12..32
      - fractional seconds: float64 at offset 36
      - TraceMaxVolts: float64 at offset 117
      - VoltsPerCount (header scale): float64 at offset 133
      - TracePoints: u32 at offset 149
      - TSamp: float64 at offset 153
      - TimeUnits: float64 at offset 161

    If any of these are missing/unreasonable, they will be returned as None.
    """
    # Try numeric timestamp first (then fall back to ASCII survey timestamp)
    dt0 = None
    year  = _safe_unpack_from("<I", blob, 12)
    month = _safe_unpack_from("<I", blob, 16)
    day   = _safe_unpack_from("<I", blob, 20)
    hour  = _safe_unpack_from("<I", blob, 24)
    minute= _safe_unpack_from("<I", blob, 28)
    second= _safe_unpack_from("<I", blob, 32)
    frac  = _safe_unpack_from("<d", blob, 36)

    if all(v is not None for v in (year, month, day, hour, minute, second, frac)):
        y, mo, d, h, mi, s = year[0], month[0], day[0], hour[0], minute[0], second[0]
        frac_s = float(frac[0])
        # Sanity checks
        if 1970 <= y <= 2100 and 1 <= mo <= 12 and 1 <= d <= 31 and 0 <= h <= 23 and 0 <= mi <= 59 and 0 <= s <= 60:
            micros = int(round(max(0.0, min(frac_s, 0.999999999)) * 1_000_000))
            try:
                dt0 = datetime(y, mo, d, h, mi, s, micros)
            except Exception:
                dt0 = None

    if dt0 is None:
        dt0 = parse_survey_datetime_ascii(blob)

    trace_max_volts = _safe_unpack_from("<d", blob, 117)
    volts_per_count = _safe_unpack_from("<d", blob, 133)
    trace_points = _safe_unpack_from("<I", blob, 149)
    tsamp = _safe_unpack_from("<d", blob, 153)
    time_units = _safe_unpack_from("<d", blob, 161)

    def sane_f(x, lo, hi):
        return x is not None and lo <= float(x) <= hi

    def sane_i(x, lo, hi):
        return x is not None and lo <= int(x) <= hi

    tp = int(trace_points[0]) if sane_i(trace_points[0] if trace_points else None, 1, 10_000_000) else None
    ts = float(tsamp[0]) if sane_f(tsamp[0] if tsamp else None, 1e-12, 1e3) else None
    tu = float(time_units[0]) if sane_f(time_units[0] if time_units else None, 1e-12, 1e0) else None
    tmv = float(trace_max_volts[0]) if sane_f(trace_max_volts[0] if trace_max_volts else None, 1e-6, 1e6) else None
    vpc = float(volts_per_count[0]) if sane_f(volts_per_count[0] if volts_per_count else None, 1e-12, 1e0) else None

    return RichterBSFHeader(
        dt0=dt0,
        trace_points=tp,
        tsamp=ts,
        time_units=tu,
        trace_max_volts=tmv,
        volts_per_count=vpc,
    )


def read_bsf_waveforms(
    path: Path,
    cfg: BSFConfig,
    channels: List[int],
    skip_short_channels: bool = True,
) -> Tuple[np.ndarray, bytes, List[int]]:
    """
    Returns counts array shape (nch, npts) int32, raw file bytes, and channels
    that were actually read. Channels not requested are not inspected.
    """
    blob = path.read_bytes()

    bytes_per_sample = 4  # int32
    ch_data_bytes = cfg.npts * bytes_per_sample
    stride = ch_data_bytes + cfg.channel_gap_bytes

    counts = np.zeros((cfg.nch, cfg.npts), dtype=np.int32)
    read_channels: List[int] = []

    for ch_num in channels:
        ch = ch_num - 1
        start = cfg.offset + ch * stride
        end = start + ch_data_bytes
        if end > len(blob):
            msg = (
                f"File too short for channel {ch_num}: "
                f"need bytes [{start}:{end}] of {len(blob)}"
            )
            if skip_short_channels:
                print(f"Warning: {path.name}: {msg}; skipping channel {ch_num}")
                continue
            raise ValueError(msg)
        counts[ch, :] = np.frombuffer(blob[start:end], dtype=cfg.dtype, count=cfg.npts)
        read_channels.append(ch_num)

    return counts, blob, read_channels


def parse_channels(s: str, nch: int) -> List[int]:
    s = s.strip().lower()
    if s in ("all", "*"):
        return list(range(1, nch + 1))
    out: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a_i, b_i = int(a), int(b)
            out.extend(list(range(a_i, b_i + 1)))
        else:
            out.append(int(part))
    out = sorted(set(out))
    for ch in out:
        if ch < 1 or ch > nch:
            raise ValueError(f"Channel {ch} out of range 1..{nch}")
    return out


def write_atf_counts(
    out_atf: Path,
    counts: np.ndarray,
    dt0: Optional[datetime],
    trace_points: int,
    tsamp: float,
    time_units: float,
    amp_to_volts: float,
    trace_max_volts: float,
) -> None:
    """
    Writes ATF v1.00 with integer counts in [TraceData] and AmpToVolts=volts_per_count.
    This preserves low-amplitude signals.
    """
    out_atf = Path(out_atf)
    counts = np.asarray(counts, dtype=np.int64)

    # Date/Time: InSite in your example uses DD-MM-YYYY
    if dt0:
        date_str = dt0.strftime("%d-%m-%Y")
        # InSite example had 7 fractional digits; emulate by appending a trailing 0 to microseconds.
        time_str = dt0.strftime("%H:%M:%S.%f") + "0"
    else:
        date_str = "01-01-1970"
        time_str = "00:00:00.0000000"

    header1 = "ATF v1.00"
    header2 = (
        f"Date={date_str}; "
        f"Time={time_str}; "
        f"TracePoints={trace_points}; "
        f"TSamp={tsamp:.5f}; "
        f"TimeUnits= {time_units:.5e}; "
        f"AmpToVolts={amp_to_volts:.8e}; "
        f"TraceMaxVolts={trace_max_volts:.4f}; "
        f"PTime=0.00000; "
        f"STime=0.00000;"
    )

    with open(out_atf, "w") as f:
        f.write(header1 + "\n")
        f.write(header2 + "\n")
        f.write("[TraceData]\n")
        for v in counts:
            f.write(f"{int(v)}\n")

def write_atf_volts(
    out_atf: Path,
    y_volts: np.ndarray,
    dt0: Optional[datetime],
    trace_points: int,
    tsamp: float,
    time_units: float,
    trace_max_volts: float,
) -> None:
    """
    Writes ATF v1.00 with VOLTS in [TraceData] and AmpToVolts=1.0000.
    Data written in scientific notation to preserve tiny signals.
    """
    out_atf = Path(out_atf)
    y_volts = np.asarray(y_volts, dtype=np.float64)

    if dt0:
        date_str = dt0.strftime("%d-%m-%Y")
        time_str = dt0.strftime("%H:%M:%S.%f") + "0"
    else:
        date_str = "01-01-1970"
        time_str = "00:00:00.0000000"

    header1 = "ATF v1.00"
    header2 = (
        f"Date={date_str}; "
        f"Time={time_str}; "
        f"TracePoints={trace_points}; "
        f"TSamp={tsamp:.5f}; "
        f"TimeUnits= {time_units:.5e}; "
        f"AmpToVolts=1.0000; "
        f"TraceMaxVolts={trace_max_volts:.4f}; "
        f"PTime=0.00000; "
        f"STime=0.00000;"
    )

    with open(out_atf, "w") as f:
        f.write(header1 + "\n")
        f.write(header2 + "\n")
        f.write("[TraceData]\n")
        for v in y_volts:
            f.write(f"{v:.12e}\n")


def write_csv_volts(path_out: Path, counts: np.ndarray, dt: float, channels: List[int], amp_to_volts: float) -> None:
    """
    Writes CSV with columns: t_s, ch1_V, ch2_V, ...
    """
    npts = counts.shape[1]
    t = np.arange(npts, dtype=np.float64) * dt
    cols = [t]
    header = ["t_s"]
    for ch in channels:
        cols.append(counts[ch - 1].astype(np.float64) * amp_to_volts)
        header.append(f"ch{ch}_V")
    arr = np.column_stack(cols)
    np.savetxt(path_out, arr, delimiter=",", header=",".join(header), comments="")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bsf", type=str, help="Input .bsf file")
    ap.add_argument("--outdir", type=str, default=".", help="Output directory")
    ap.add_argument("--prefix", type=str, default=None, help="Output prefix (default: input stem)")
    ap.add_argument("--channels", type=str, default="all", help='Channels to export, e.g. "1,2" or "1-2" or "all"')
    ap.add_argument("--csv", action="store_true", help="Also write a CSV with selected channels (in volts)")
    ap.add_argument(
        "--strict-channels",
        action="store_true",
        help="Fail if any requested channel is too short instead of skipping that channel.",
    )

    # waveform layout
    ap.add_argument("--offset", type=int, default=DEFAULT_OFFSET, help="Byte offset to CH1 waveform start")
    ap.add_argument("--channel-gap-bytes", type=int, default=DEFAULT_CHANNEL_GAP_BYTES, help="Gap between channel waveform blocks")
    ap.add_argument("--nch", type=int, default=DEFAULT_NCH, help="Number of channels in file")
    ap.add_argument("--npts", type=int, default=DEFAULT_NPTS, help="Samples per channel")

    # acquisition/scaling defaults (used if header parsing fails or disabled)
    ap.add_argument("--fs", type=float, default=DEFAULT_FS, help="Sampling rate (Hz) used if header lacks TSamp/TimeUnits")
    ap.add_argument("--scale", type=float, default=DEFAULT_SCALE_V_PER_COUNT, help="Volts per count (AmpToVolts) fallback")
    ap.add_argument("--trace-max-volts", type=float, default=5.0, help="TraceMaxVolts fallback")

    # header parsing controls
    ap.add_argument("--header-from-bsf", dest="header_from_bsf", action="store_true", help="Parse TSamp/TimeUnits/TraceMaxVolts/scale from BSF header when possible (default)")
    ap.add_argument("--no-header-from-bsf", dest="header_from_bsf", action="store_false", help="Do not parse header; use CLI defaults")
    ap.set_defaults(header_from_bsf=True)

    args = ap.parse_args()

    bsf_path = Path(args.bsf).expanduser().resolve()

################
#   outdir = Path(args.outdir).expanduser().resolve()
#   outdir.mkdir(parents=True, exist_ok=True)
#   prefix = args.prefix if args.prefix else bsf_path.stem
################
# If not explicitly providing --outdir,
# write output in same directory as input file.
    if args.outdir == ".":
        outdir = bsf_path.parent
    else:
        outdir = Path(args.outdir).expanduser().resolve()
        outdir.mkdir(parents=True, exist_ok=True)
    
    prefix = args.prefix if args.prefix else bsf_path.stem
#################
    cfg = BSFConfig(
        offset=args.offset,
        nch=args.nch,
        npts=args.npts,
        fs=args.fs,
        channel_gap_bytes=args.channel_gap_bytes,
        scale_v_per_count=args.scale,
    )

    channels = parse_channels(args.channels, cfg.nch)
    counts, blob, channels = read_bsf_waveforms(
        bsf_path,
        cfg,
        channels,
        skip_short_channels=not args.strict_channels,
    )
    if not channels:
        raise SystemExit(f"No requested channels could be read from {bsf_path}")

    # Defaults from CLI
    dt0 = parse_survey_datetime_ascii(blob)
    trace_points = cfg.npts
    tsamp = 0.1
    time_units = 1e-6
    trace_max_volts = float(args.trace_max_volts)
    amp_to_volts = float(args.scale)

    if args.header_from_bsf:
        hdr = read_richter_bsf_header(blob)
        if hdr.dt0 is not None:
            dt0 = hdr.dt0

        if hdr.trace_points is not None:
            trace_points = hdr.trace_points

        # If TSamp/TimeUnits exist in header, prefer them.
        # Otherwise compute dt from fs but still try to match the "0.1 and 1e-6" style when dt == 1e-7.
        if hdr.tsamp is not None and hdr.time_units is not None:
            tsamp = hdr.tsamp
            time_units = hdr.time_units
        else:
            dt = 1.0 / float(cfg.fs)
            # Prefer InSite-style TSamp=0.1, TimeUnits=1e-6 when possible.
            if abs(dt - 1e-7) < 1e-12:
                tsamp, time_units = 0.1, 1e-6
            else:
                tsamp, time_units = 1.0, dt

        # TraceMaxVolts from header if plausible
        if hdr.trace_max_volts is not None:
            trace_max_volts = hdr.trace_max_volts

        # Volts-per-count: use header value only if it is plausible; otherwise use CLI scale.
        if hdr.volts_per_count is not None:
            # Accept header scale if it is within an order-of-magnitude of CLI scale or equals common 5/32768 or 5/2^23.
            v = hdr.volts_per_count
            common_16 = 5.0 / 32768.0
            common_24 = 5.0 / (2**23)
            if (0.1 * amp_to_volts <= v <= 10.0 * amp_to_volts) or abs(v - common_16) < 1e-12 or abs(v - common_24) < 1e-12:
                amp_to_volts = v

    # dt used for CSV time axis
    dt = float(tsamp) * float(time_units)

    # Write ATFs (one per channel)
    for ch in channels:
        out_atf = outdir / f"{prefix}_ch{ch:02d}.atf"
    
        # volts = counts * TraceMaxVolts / 2^53
        y_volts = counts[ch - 1].astype(np.float64) * (trace_max_volts / (2**23))
    
        write_atf_volts(
            out_atf=out_atf,
            y_volts=y_volts,
            dt0=dt0,
            trace_points=trace_points,
            tsamp=tsamp,
            time_units=time_units,
            trace_max_volts=trace_max_volts,
        )
    

    # Optional CSV (volts)
    if args.csv:
        out_csv = outdir / f"{prefix}_selected_channels.csv"
        write_csv_volts(out_csv, counts, dt, channels, amp_to_volts=amp_to_volts)
        print(f"Wrote CSV: {out_csv}")

    print(f"Done. Wrote {len(channels)} ATF file(s) to: {outdir}")


if __name__ == "__main__":
    main()
