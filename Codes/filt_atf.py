#!/usr/bin/env python3
"""
Filter a 1-column ASCII trace file that contains a header + data (one sample per line).

Assumptions / rules:
- The file has an arbitrary text header followed by numeric samples (one column).
- Sampling interval is: dt = TSamp * TimeUnits  (both read from header text).
- Applies frequency-domain filtering (FFT) with a smooth cosine taper at the cutoff(s).
- freq_min = 0 means "no high-pass" (i.e., allow DC)
- freq_max = 0 means "no low-pass" (i.e., allow up to Nyquist)
- Output keeps the original header exactly and writes one filtered sample per line.

Example:
  python filter_trace.py input.atf output.atf --freq_min 1000 --freq_max 200000
  python filter_trace.py input.atf output.atf --freq_min 0 --freq_max 50000     # low-pass
  python filter_trace.py input.atf output.atf --freq_min 2000 --freq_max 0      # high-pass
  python filter_trace.py @filt_input.txt
  python filter_trace.py --input-files file1.atf file2.atf --freq_min 20000 --freq_max 40000

Created on Wed Feb 11 15:36:30 2026

@author: stefan
"""


import argparse
import re
import sys
from pathlib import Path

import numpy as np


def parse_dt_from_header(header_lines):
    """
    Extract TSamp and TimeUnits from ATF-style header and compute dt = TSamp * TimeUnits.

    Works if keys appear anywhere in the header, e.g.
      "Date=...; TracePoints=...; TSamp=1e-07; TimeUnits=1; AmpToVolts=1;"
    """
    header_text = "\n".join(header_lines)

    num = r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"

    m_ts = re.search(rf"(?i)\bTSamp\s*=\s*{num}\b", header_text)
    m_tu = re.search(rf"(?i)\bTimeUnits\s*=\s*{num}\b", header_text)

    if not m_ts or not m_tu:
        raise ValueError(
            "Could not find BOTH 'TSamp' and 'TimeUnits' in the header text. "
            "Expected patterns like 'TSamp=1e-07' and 'TimeUnits=1'."
        )

    tsamp = float(m_ts.group(1))
    timeunits = float(m_tu.group(1))

    dt = tsamp * timeunits
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError(f"Computed dt is invalid: dt={dt} (TSamp={tsamp}, TimeUnits={timeunits})")

    return dt



def split_header_and_data(lines):
    """
    Split file into (header_lines, data_lines) where data_lines are numeric samples.
    We detect the first line that looks like a numeric (float) and treat everything after as data.
    """
    float_re = re.compile(r"^\s*[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\s*$")

    first_data_idx = None
    for i, line in enumerate(lines):
        if float_re.match(line):
            first_data_idx = i
            break

    if first_data_idx is None:
        raise ValueError("No numeric data lines found (expected one float per line).")

    header_lines = lines[:first_data_idx]
    data_lines = lines[first_data_idx:]
    return header_lines, data_lines


def cosine_taper_bandpass(freqs, fmin, fmax, trans):
    """
    Build a smooth bandpass mask H(f) (for rfft frequencies, freqs >= 0).

    fmin==0 => no high-pass
    fmax==0 => no low-pass

    trans is the transition width (Hz) used as a half-cosine ramp near cutoffs.
    """
    H = np.ones_like(freqs, dtype=float)

    # High-pass side (around fmin)
    if fmin > 0:
        f1 = max(0.0, fmin - trans)
        f2 = fmin + trans
        # Below f1 => 0, above f2 => 1, between => half-cosine ramp
        H[freqs <= f1] = 0.0
        mid = (freqs > f1) & (freqs < f2)
        if np.any(mid):
            x = (freqs[mid] - f1) / (f2 - f1)  # 0..1
            H[mid] = 0.5 - 0.5 * np.cos(np.pi * x)

    # Low-pass side (around fmax)
    if fmax > 0:
        f3 = max(0.0, fmax - trans)
        f4 = fmax + trans
        # Above f4 => 0, below f3 => 1, between => half-cosine ramp down
        H[freqs >= f4] = 0.0
        mid = (freqs > f3) & (freqs < f4)
        if np.any(mid):
            x = (freqs[mid] - f3) / (f4 - f3)  # 0..1
            H[mid] *= 0.5 + 0.5 * np.cos(np.pi * x)

    return H


def fft_filter(x, dt, freq_min, freq_max, transition_hz):
    """
    Frequency-domain filtering with smooth cosine tapers at cutoff(s).
    """
    n = len(x)
    if n < 2:
        return x.copy()

    fs = 1.0 / dt
    nyq = 0.5 * fs

    fmin = float(freq_min)
    fmax = float(freq_max)

    if fmin < 0 or fmax < 0:
        raise ValueError("freq_min and freq_max must be >= 0.")

    # Interpret zeros as "no bound"
    if fmin == 0:
        fmin = 0.0
    if fmax == 0:
        fmax = 0.0

    # Validate vs Nyquist if specified
    if fmin > nyq:
        raise ValueError(f"freq_min={fmin} exceeds Nyquist={nyq}.")
    if fmax > 0 and fmax > nyq:
        raise ValueError(f"freq_max={fmax} exceeds Nyquist={nyq}.")
    if fmax > 0 and fmin > 0 and fmin >= fmax:
        raise ValueError("Need freq_min < freq_max for a band-pass (when both are non-zero).")

    # If no filtering requested
    if (freq_min == 0) and (freq_max == 0):
        return x.copy()

    trans = float(transition_hz)
    if trans <= 0:
        # sensible default: 5% of the smallest non-zero cutoff, but at least 1 Hz
        cutoffs = [c for c in [fmin, fmax] if c and c > 0]
        base = min(cutoffs) if cutoffs else nyq
        trans = max(1.0, 0.05 * base)

    # FFT
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=dt)

    H = cosine_taper_bandpass(freqs, fmin=fmin, fmax=fmax, trans=trans)

    Y = X * H
    y = np.fft.irfft(Y, n=n)
    return y


def convert_arg_line_to_args(line):
    """
    Allow argument files with comments.

    Example input file:
        --input-files
        file1.atf             # first input file
        file2.atf             # second input file
        --freq_min 20000
        --freq_max 40000
        --transition_hz 2000
        --demean

    Run with:
        python filt_atf.py @filt_input.txt
    """
    line = line.split("#", 1)[0].strip()
    if not line:
        return []
    return line.split()


def format_khz(freq_hz):
    value = float(freq_hz) / 1000.0
    return f"{value:g}"


def filter_suffix(freq_min, freq_max):
    if freq_min == 0 and freq_max == 0:
        return "_unfiltered"
    if freq_min == 0:
        return f"_0_{format_khz(freq_max)}k"
    if freq_max == 0:
        return f"_{format_khz(freq_min)}k_nyq"
    return f"_{format_khz(freq_min)}_{format_khz(freq_max)}k"


def automatic_output_path(input_path, freq_min, freq_max):
    return input_path.with_name(f"{input_path.stem}{filter_suffix(freq_min, freq_max)}{input_path.suffix}")


def filter_file(input_path, output_path, args):
    # Read as text, preserve line endings
    text = input_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines(keepends=True)

    header_lines, data_lines = split_header_and_data(lines)
    dt = parse_dt_from_header([hl.rstrip("\r\n") for hl in header_lines])

    # Parse samples
    # (strip line endings, convert to float)
    x = np.array([float(dl.strip()) for dl in data_lines], dtype=float)

    mean0 = 0.0
    if args.demean:
        mean0 = float(np.mean(x))
        x = x - mean0

    y = fft_filter(
        x,
        dt=dt,
        freq_min=args.freq_min,
        freq_max=args.freq_max,
        transition_hz=args.transition_hz,
    )

    if args.demean:
        y = y + mean0

    # Write output: same header + one sample per line
    # Preserve original newline style: infer from the first newline in file, else default to '\n'
    newline = "\n"
    for ln in lines:
        if ln.endswith("\r\n"):
            newline = "\r\n"
            break
        if ln.endswith("\n"):
            newline = "\n"
            break

    out_lines = []
    out_lines.extend(header_lines)
    out_lines.extend([f"{val:.16e}{newline}" for val in y])

    output_path.write_text("".join(out_lines), encoding="utf-8")
    print(f"Wrote: {output_path}  (N={len(y)}, dt={dt:g}s)")


def main():
    ap = argparse.ArgumentParser(
        description="Band-pass / low-pass / high-pass filter for 1-col trace ASCII files.",
        fromfile_prefix_chars="@",
    )
    ap.convert_arg_line_to_args = convert_arg_line_to_args
    ap.add_argument(
        "input",
        nargs="?",
        type=Path,
        help="Input file for single-file mode (header + one float per line)",
    )
    ap.add_argument(
        "output",
        nargs="?",
        type=Path,
        help="Output file for single-file mode (same header, filtered samples)",
    )
    ap.add_argument(
        "--input-files",
        nargs="+",
        type=Path,
        help="Input files for multi-file mode.",
    )
    ap.add_argument(
        "--output-files",
        nargs="+",
        type=Path,
        help=(
            "Output files for multi-file mode. Optional; if omitted, names are "
            "generated by appending the filter band, e.g. _20_40k."
        ),
    )
    ap.add_argument("--freq_min", type=float, default=0.0, help="High-pass cutoff (Hz). 0 => no high-pass.")
    ap.add_argument("--freq_max", type=float, default=0.0, help="Low-pass cutoff (Hz). 0 => no low-pass.")
    ap.add_argument(
        "--transition_hz",
        type=float,
        default=0.0,
        help="Cosine-taper transition width in Hz (default: auto ~5%% of cutoff).",
    )
    ap.add_argument(
        "--demean",
        action="store_true",
        help="Remove mean before filtering (mean is added back after).",
    )
    args = ap.parse_args()

    positional_mode = args.input is not None or args.output is not None
    list_mode = args.input_files is not None or args.output_files is not None

    if positional_mode and list_mode:
        ap.error("Use either positional input/output OR --input-files/--output-files, not both")

    if list_mode:
        if args.input_files is None:
            ap.error("--input-files is required when using list mode")
        if args.output_files is not None:
            if len(args.input_files) != len(args.output_files):
                ap.error(
                    f"--input-files has {len(args.input_files)} file(s), "
                    f"but --output-files has {len(args.output_files)} file(s)"
                )
            output_files = args.output_files
        else:
            output_files = [
                automatic_output_path(input_path, args.freq_min, args.freq_max)
                for input_path in args.input_files
            ]
        pairs = list(zip(args.input_files, output_files))
    else:
        if args.input is None:
            ap.error("Supply an input file or use --input-files")
        output = args.output or automatic_output_path(args.input, args.freq_min, args.freq_max)
        pairs = [(args.input, output)]

    for input_path, output_path in pairs:
        filter_file(input_path, output_path, args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
