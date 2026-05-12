#!/usr/bin/env python3
import argparse
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np


def read_atf(path: str):
    with open(path, "r", errors="ignore") as f:
        _ = f.readline().strip()
        header2 = f.readline().strip()

        def get_header_float(name: str) -> float:
            m = re.search(rf"{re.escape(name)}\s*=\s*([0-9.eE+-]+)", header2)
            if not m:
                raise ValueError(f"Missing header field '{name}' in {path}\nHeader: {header2}")
            return float(m.group(1))

        def get_header_int(name: str) -> int:
            m = re.search(rf"{re.escape(name)}\s*=\s*(\d+)", header2)
            if not m:
                raise ValueError(f"Missing header field '{name}' in {path}\nHeader: {header2}")
            return int(m.group(1))

        n = get_header_int("TracePoints")
        tsamp = get_header_float("TSamp")
        time_units = get_header_float("TimeUnits")
        amp_to_volts = get_header_float("AmpToVolts")
        dt = tsamp * time_units

        line = f.readline()
        while line and "[TraceData]" not in line:
            line = f.readline()

        if not line:
            raise ValueError(f"[TraceData] section not found in {path}")

        data = []
        for _ in range(n):
            s = f.readline()
            if not s:
                break
            s = s.strip()
            if s:
                data.append(float(s))

    x = np.asarray(data, dtype=np.float64) * amp_to_volts
    t = np.arange(len(x), dtype=np.float64) * dt
    return t, x, dt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot two normalized ATF traces with optional time-axis limits."
    )

    parser.add_argument(
        "files",
        nargs="*",
        default=["./stacked_ch2.atf", "./stacked_ch3.atf"],
        help="Two input ATF files. Default: ./stacked_ch2.atf ./stacked_ch3.atf",
    )

    parser.add_argument(
        "--tmin",
        type=float,
        default=None,
        help="Initial plot time in seconds. Default: start of processed window.",
    )

    parser.add_argument(
        "--tmax",
        type=float,
        default=None,
        help="Final plot time in seconds. Default: end of processed window.",
    )

    parser.add_argument(
        "--window-sec",
        type=float,
        default=0.000125,
        help="Length of the data window used for baseline correction and normalization. Default: 0.000125 s.",
    )

    parser.add_argument(
        "--baseline-start-sec",
        type=float,
        default=0.000005,
        help="Baseline window start time in seconds. Default: 0.000005 s.",
    )

    parser.add_argument(
        "--baseline-end-sec",
        type=float,
        default=0.000025,
        help="Baseline window end time in seconds. Default: 0.000025 s.",
    )

    parser.add_argument(
        "--amp-02",
        type=float,
        default=31.6,
        help="Amplitude multiplier for the first trace / Ch02. Default: 31.6.",
    )

    parser.add_argument(
        "--amp-03",
        type=float,
        default=2.0,
        help="Amplitude multiplier for the second trace / Ch03. Default: 2.0.",
    )

    args = parser.parse_args()

    if len(args.files) != 2:
        parser.error("provide exactly two ATF files, or no files to use the defaults")

    if args.window_sec <= 0:
        parser.error("--window-sec must be positive")

    if args.baseline_start_sec < 0 or args.baseline_end_sec < 0:
        parser.error("baseline times must be non-negative")

    if args.baseline_start_sec >= args.baseline_end_sec:
        parser.error("--baseline-start-sec must be smaller than --baseline-end-sec")

    if args.tmin is not None and args.tmin < 0:
        parser.error("--tmin must be non-negative")

    if args.tmax is not None and args.tmax <= 0:
        parser.error("--tmax must be positive")

    if args.tmin is not None and args.tmax is not None and args.tmin >= args.tmax:
        parser.error("--tmin must be smaller than --tmax")

    return args


def main():
    args = parse_args()

    paths = [os.path.abspath(f) for f in args.files]

    # Make sure the processed data window extends at least as far as the requested plot end.
    window_sec = args.window_sec
    if args.tmax is not None:
        window_sec = max(window_sec, args.tmax)

    data_list = []
    for p in paths:
        if not os.path.exists(p):
            print(f"File not found: {p}")
            sys.exit(1)

        t, x, dt = read_atf(p)
        n_window = int(np.floor(window_sec / dt))
        if n_window < 1:
            raise SystemExit("Window too small for sampling interval")

        n_window = min(n_window, len(x))
        t_trim = t[:n_window]
        x_trim = x[:n_window].astype(np.float64)

        # Baseline from baseline_start_sec to baseline_end_sec.
        n_baseline_start = int(np.floor(args.baseline_start_sec / dt))
        n_baseline_end = int(np.floor(args.baseline_end_sec / dt))

        # Clamp indices.
        n_baseline_start = max(0, n_baseline_start)
        n_baseline_end = min(len(x_trim), max(n_baseline_start + 1, n_baseline_end))

        if n_baseline_end <= n_baseline_start:
            baseline = 0.0
        else:
            baseline = np.mean(x_trim[n_baseline_start:n_baseline_end])

        x_corr = x_trim - baseline

        # Normalize to max absolute amplitude in the processed window.
        max_amp = float(np.max(np.abs(x_corr)))
        if max_amp == 0:
            x_norm = x_corr
            inv_factor = float("nan")
        else:
            x_norm = x_corr / max_amp
            inv_factor = 1.0 / max_amp

        data_list.append((t_trim, x_corr, x_norm, max_amp, inv_factor))

    # Plot: first trace on left axis, second trace on right axis.
    t0, x2_corr, x2_norm, max_amp1, invf1 = data_list[0]
    t1, x3_corr, x3_norm, max_amp2, invf2 = data_list[1]

    plot_tmin = args.tmin if args.tmin is not None else 0.0
    plot_tmax = args.tmax if args.tmax is not None else min(t0[-1], t1[-1])

    fig, ax = plt.subplots(figsize=(10, 5))
    color2 = "C0"
    color3 = "C1"

    ax.plot(t0, x2_norm, color=color2, lw=1.0, label=os.path.basename(paths[0]))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized amplitude (ch02)", color=color2)
    ax.tick_params(axis="y", labelcolor=color2)
    ax.set_xlim(plot_tmin, plot_tmax)
    ax.set_ylim(-1.05, 1.05)

    ax2 = ax.twinx()
    ax2.plot(t1, x3_norm, color=color3, lw=1.0, label=os.path.basename(paths[1]))
    ax2.set_ylabel("Normalized amplitude (ch03)", color=color3)
    ax2.tick_params(axis="y", labelcolor=color3)
    ax2.set_ylim(-1.05, 1.05)

    # Combined legend.
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, fontsize=8)

    # Display scaling factors and their combined ratio C.
    Ch02 = invf1
    Ch03 = invf2
    try:
        C = (Ch03 * args.amp_03) / (Ch02 * args.amp_02)
    except Exception:
        C = float("nan")

    txt = (
        f"Ch02 (Acc)={Ch02:.3e} ×{args.amp_02:g}\n"
        f"Ch03 (PZ)={Ch03:.3e} ×{args.amp_03:g}\n"
        f"C(V to m/s$^{{-2}}$)={C:.3e}"
    )
    ax.text(
        0.02,
        0.98,
        txt,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
    )

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
