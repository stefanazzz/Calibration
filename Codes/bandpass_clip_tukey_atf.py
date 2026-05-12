#!/usr/bin/env python3
"""
bandpass_clip_tukey_atf.py

Read an ATF v1.00 file, bandpass filter, interactive clip selection, Tukey taper,
and write out a new ATF with the same header format and updated TracePoints.

Example:
  python bandpass_clip_tukey_atf.py stack.atf --fmin 10000 --fmax 200000 --alpha 0.1
Stefan Nielsen March 2026
"""

import argparse
import os
import re
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt
from matplotlib.widgets import SpanSelector


def parse_atf(path: str):
    """Return (line1, header_line, header_dict, data_array)."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = [ln.rstrip("\n") for ln in f]

    if len(lines) < 4:
        raise ValueError("File too short to be a valid ATF.")

    line1 = lines[0].strip()
    header_line = lines[1].strip()

    if not line1.startswith("ATF"):
        raise ValueError(f"Not an ATF file? First line: {line1!r}")

    # Find the [TraceData] marker
    try:
        i_td = next(i for i, ln in enumerate(lines) if ln.strip() == "[TraceData]")
    except StopIteration:
        raise ValueError("Could not find [TraceData] marker in ATF file.")

    data_lines = lines[i_td + 1 :]
    data = []
    for ln in data_lines:
        s = ln.strip()
        if not s:
            continue
        try:
            data.append(float(s))
        except ValueError:
            # ignore any non-numeric trailing lines (rare)
            pass

    if not data:
        raise ValueError("No numeric samples found after [TraceData].")

    # Parse header k=v; pairs (best-effort)
    hdr = {}
    for part in header_line.split(";"):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            k, v = part.split("=", 1)
            hdr[k.strip()] = v.strip()
        else:
            # sometimes bare tokens appear; ignore
            pass

    return line1, header_line, hdr, np.asarray(data, dtype=float)


def dt_from_header(hdr: dict):
    """
    ATF convention in your file:
      TSamp=0.10000; TimeUnits=1.00000e-06  => dt = TSamp * TimeUnits
    """
    if "TSamp" not in hdr or "TimeUnits" not in hdr:
        raise ValueError("Header missing TSamp and/or TimeUnits; can't compute dt.")

    tsamp = float(hdr["TSamp"])
    time_units = float(hdr["TimeUnits"])
    return tsamp * time_units


def butter_bandpass_filt(x, fs, fmin, fmax, order=4):
    if fmin <= 0 or fmax <= 0:
        raise ValueError("fmin and fmax must be > 0.")
    if fmax <= fmin:
        raise ValueError("fmax must be > fmin.")
    nyq = 0.5 * fs
    if fmax >= nyq:
        raise ValueError(f"fmax must be < Nyquist ({nyq:.3g} Hz).")

    b, a = butter(order, [fmin / nyq, fmax / nyq], btype="bandpass")
    return filtfilt(b, a, x)


def tukey_window(N: int, alpha: float):
    """
    Simple Tukey window
    alpha in [0,1]. alpha=0 -> rectangular, alpha=1 -> Hann.
    """
    if N <= 0:
        return np.array([], dtype=float)
    alpha = float(alpha)
    if alpha <= 0:
        return np.ones(N, dtype=float)
    if alpha >= 1:
        # Hann
        n = np.arange(N)
        return 0.5 * (1 - np.cos(2 * np.pi * n / (N - 1))) if N > 1 else np.ones(1)

    n = np.arange(N, dtype=float)
    w = np.ones(N, dtype=float)

    edge = int(np.floor(alpha * (N - 1) / 2.0))
    if edge < 1:
        return w

    # left taper
    nL = n[: edge + 1]
    w[: edge + 1] = 0.5 * (1 + np.cos(np.pi * (2 * nL / (alpha * (N - 1)) - 1)))

    # right taper
    nR = n[-(edge + 1) :]
    w[-(edge + 1) :] = 0.5 * (1 + np.cos(np.pi * (2 * nR / (alpha * (N - 1)) - 2 / alpha + 1)))

    return w


def update_tracepoints_in_header(header_line: str, new_n: int):
    """
    Replace TracePoints=... in the raw header line, preserving the rest.
    If TracePoints is absent, append it.
    """
    if re.search(r"\bTracePoints\s*=\s*\d+", header_line):
        return re.sub(r"(\bTracePoints\s*=\s*)\d+", rf"\g<1>{new_n}", header_line)
    # append before trailing ';' if present
    if header_line.endswith(";"):
        return header_line[:-1] + f"; TracePoints={new_n};"
    return header_line + f"; TracePoints={new_n};"


def write_atf(out_path: str, line1: str, header_line: str, y: np.ndarray):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"{line1}\n")
        f.write(f"{header_line}\n")
        f.write("[TraceData]\n")
        for v in y:
            f.write(f"{v:.15e}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("atf", help="Input ATF file (ATF v1.00)")
    ap.add_argument("--fmin", type=float, required=True, help="Bandpass low cutoff [Hz]")
    ap.add_argument("--fmax", type=float, required=True, help="Bandpass high cutoff [Hz]")
    ap.add_argument("--order", type=int, default=4, help="Butterworth order (default 4)")
    ap.add_argument("--alpha", type=float, default=0.1, help="Tukey alpha in [0,1] (default 0.1)")
    ap.add_argument("--out", default=None, help="Output ATF path (default: auto)")
    ap.add_argument("--no_interactive", action="store_true",
                    help="Skip interactive selection and use full filtered trace.")
    ap.add_argument("--t0", type=float, default=None, help="Clip start time [s] (non-interactive)")
    ap.add_argument("--t1", type=float, default=None, help="Clip end time [s] (non-interactive)")
    args = ap.parse_args()

    line1, header_line_raw, hdr, x = parse_atf(args.atf)
    dt = dt_from_header(hdr)
    fs = 1.0 / dt
    t = np.arange(len(x)) * dt

    y = butter_bandpass_filt(x, fs=fs, fmin=args.fmin, fmax=args.fmax, order=args.order)

    # Determine clip indices
    clip = {"i0": 0, "i1": len(y)}

    if args.no_interactive or (args.t0 is not None and args.t1 is not None):
        if args.t0 is not None and args.t1 is not None:
            if args.t1 <= args.t0:
                raise ValueError("--t1 must be > --t0")
            clip["i0"] = max(0, int(np.floor(args.t0 / dt)))
            clip["i1"] = min(len(y), int(np.ceil(args.t1 / dt)))
        # else full length
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
        ax1.plot(t, x, linewidth=1)
        ax1.set_title("Original")
        ax1.set_ylabel("Amplitude")

        ax2.plot(t, y, linewidth=1)
        ax2.set_title("Filtered (drag to select clip on this panel)")
        ax2.set_xlabel("Time [s]")
        ax2.set_ylabel("Amplitude")

        span_patch = {"obj": None}

        def onselect(tmin, tmax):
            if tmax <= tmin:
                return
            i0 = int(np.floor(tmin / dt))
            i1 = int(np.ceil(tmax / dt))
            i0 = max(0, min(i0, len(y) - 1))
            i1 = max(i0 + 1, min(i1, len(y)))

            clip["i0"], clip["i1"] = i0, i1

            # draw selected region
            if span_patch["obj"] is not None:
                span_patch["obj"].remove()
            span_patch["obj"] = ax2.axvspan(i0 * dt, i1 * dt, alpha=0.2)
            fig.canvas.draw_idle()

        _ = SpanSelector(
            ax2,
            onselect,
            "horizontal",
            useblit=True,
            props=dict(alpha=0.2),
            interactive=True,
            drag_from_anywhere=True,
        )

        plt.tight_layout()
        plt.show()  # close the window to continue

    y_clip = y[clip["i0"] : clip["i1"]].copy()

    # remove baseline using the first value inside the clipped interval
    y_clip -= y_clip[0]

    # remove trend between first and last point of clip
    N = len(y_clip)
    if N >= 2:
        ramp = np.linspace(0.0, y_clip[-1], N)   # line connecting first(=0) to last
        y_clip = y_clip - ramp

    # Tukey taper
    w = tukey_window(len(y_clip), args.alpha)
    y_out = y_clip * w

    # Output naming
    if args.out is None:
        base, ext = os.path.splitext(args.atf)
        out_path = f"{base}_bp_{args.fmin:g}-{args.fmax:g}Hz_clip_tukey{args.alpha:g}.atf"
    else:
        out_path = args.out

    # Update TracePoints in the header line (keep everything else the same)
    header_line_new = update_tracepoints_in_header(header_line_raw, len(y_out))

    write_atf(out_path, line1, header_line_new, y_out)
    print(f"Saved: {out_path}")
    print(f"dt = {dt:.6e} s, fs = {fs:.6g} Hz")
    print(f"clip samples: {len(y_out)} (i0={clip['i0']}, i1={clip['i1']})")


if __name__ == "__main__":
    main()

