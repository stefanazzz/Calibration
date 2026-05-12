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


def crop_to_window(t, x, tmin, tmax, label="trace"):
    """Crop one trace to the requested absolute-time window."""
    mask = (t >= tmin) & (t <= tmax)
    if not np.any(mask):
        raise SystemExit(
            f"No samples found for {label} in requested plot window {tmin:g} to {tmax:g} s"
        )
    return t[mask], x[mask]


def peak_scales_by_polarity(x):
    """
    Return positive-peak and negative-trough magnitudes for one cropped trace.

    positive_scale = max(x)
    negative_scale = abs(min(x))

    Values are clipped at zero so that a trace with no positive samples has
    positive_scale=0, and a trace with no negative samples has negative_scale=0.
    """
    positive_scale = max(float(np.max(x)), 0.0)
    negative_scale = max(abs(float(np.min(x))), 0.0)
    return positive_scale, negative_scale


def normalize_by_scale(x, scale, label="trace"):
    """Normalize a cropped trace by a supplied peak scale."""
    if scale <= 0 or not np.isfinite(scale):
        raise SystemExit(
            f"Cannot normalize {label}: selected peak scale is {scale:g}. "
            "Check the crop window and polarity."
        )
    return x / scale, 1.0 / scale


def convert_arg_line_to_args(line):
    """
    Allow argument files with comments.

    Example input file:
        stacked_ch2.atf        # first file
        stacked_ch3.atf        # second file
        --tmin 2e-5            # min time for plot
        --tmax 8e-5            # max time for plot
        --title1 Accelerometer # label for first file
        --title2 Piezo         # label for second file
        --amp-file1 31.6       # amplitude multiplier for first file
        --amp-file2 2.0        # amplitude multiplier for second file

    Run with:
        python plot_two.py @input.txt
    """
    # Remove running comments and ignore blank lines.
    line = line.split("#", 1)[0].strip()
    if not line:
        return []

    # Split on normal whitespace. For labels with spaces, quote them on the command
    # line instead, or use underscores in the input file.
    return line.split()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot two ATF traces, cropped and normalized within the requested absolute-time window, using a common peak polarity choice for both traces.",
        fromfile_prefix_chars="@",
    )
    parser.convert_arg_line_to_args = convert_arg_line_to_args

    parser.add_argument(
        "files",
        nargs="*",
        default=["./stacked_ch2.atf", "./stacked_ch3.atf"],
        help="Two input ATF files. Default: ./stacked_ch2.atf ./stacked_ch3.atf",
    )

    parser.add_argument(
        "--title1",
        default="Trace 1",
        help="Plot label/title for the first file. Default: Trace 1.",
    )

    parser.add_argument(
        "--title2",
        default="Trace 2",
        help="Plot label/title for the second file. Default: Trace 2.",
    )

    parser.add_argument(
        "--tmin",
        type=float,
        default=None,
        help="Initial time in seconds for crop, plot, and peak normalization. Default: 0.0.",
    )

    parser.add_argument(
        "--tmax",
        type=float,
        default=None,
        help="Final time in seconds for crop, plot, and peak normalization. Default: end of available data.",
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
        "--amp-file1",
        type=float,
        default=31.6,
        help="Amplitude multiplier for the first file. Default: 31.6.",
    )

    parser.add_argument(
        "--amp-file2",
        type=float,
        default=2.0,
        help="Amplitude multiplier for the second file. Default: 2.0.",
    )

    parser.add_argument(
        "--shift-file1-sec",
        type=float,
        default=0.0,
        help=(
            "Time shift applied to the first file, in seconds, before baseline "
            "correction, cropping, normalization, and plotting. "
            "Positive values move file1 later; negative values move it earlier. "
            "Default: 0.0."
        ),
    )

    parser.add_argument(
        "--invert-file1",
        action="store_true",
        help="Invert polarity of the first file before normalisation.",
    )

    parser.add_argument(
        "--invert-file2",
        action="store_true",
        help="Invert polarity of the second file before normalisation.",
    )

    parser.add_argument(
        "--norm-mode",
        choices=["global", "polarity"],
        default="global",
        help=(
            "Normalization mode:\n"
            "  global   = normalize by max absolute value across both traces (default)\n"
            "  polarity = normalize by dominant polarity (pos or neg) across both traces"
        ),
    )

    args = parser.parse_args()

    if len(args.files) != 2:
        parser.error("provide exactly two ATF files, or no files to use the defaults")


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
    trace_titles = [args.title1, args.title2]

    # Read each full trace. The crop/plot/normalization window is controlled
    # directly by --tmin and --tmax, and the x-axis remains in absolute time.
    data_list = []
    for i, p in enumerate(paths):
        if not os.path.exists(p):
            print(f"File not found: {p}")
            sys.exit(1)

        t, x, dt = read_atf(p)
        x_full = x.astype(np.float64)

        # Apply the optional time-axis shift to file1 before any processing.
        # Positive shift moves file1 later; negative shift moves file1 earlier.
        if i == 0 and args.shift_file1_sec != 0.0:
            t = t + args.shift_file1_sec

        # Baseline correction is taken from the requested absolute-time window
        # after any optional file1 time shift. It is applied before cropping.
        baseline_mask = (t >= args.baseline_start_sec) & (t <= args.baseline_end_sec)
        if not np.any(baseline_mask):
            baseline = 0.0
            print(
                f"Warning: no baseline samples found for {os.path.basename(p)} "
                f"in {args.baseline_start_sec:g}–{args.baseline_end_sec:g} s; "
                "using baseline=0.0"
            )
        else:
            baseline = float(np.mean(x_full[baseline_mask]))

        x_corr = x_full - baseline
        data_list.append((t, x_corr))

    t_file1_full, x_file1_corr_full = data_list[0]
    t_file2_full, x_file2_corr_full = data_list[1]

    plot_tmin = args.tmin if args.tmin is not None else 0.0
    plot_tmax = args.tmax if args.tmax is not None else min(t_file1_full[-1], t_file2_full[-1])

    if plot_tmin >= plot_tmax:
        raise SystemExit("Crop/plot window is empty: check --tmin and --tmax")

    if args.invert_file1:
        x_file1_corr_full = -x_file1_corr_full

    if args.invert_file2:
        x_file2_corr_full = -x_file2_corr_full

    # Crop first. The cropped interval is also the processing and plotting window.
    t_file1, x_file1_corr = crop_to_window(
        t_file1_full, x_file1_corr_full, plot_tmin, plot_tmax, label="file1"
    )
    t_file2, x_file2_corr = crop_to_window(
        t_file2_full, x_file2_corr_full, plot_tmin, plot_tmax, label="file2"
    )

    # Choose one normalization polarity for both traces.
    # Compare the strongest positive peak across both traces with the strongest
    # negative trough magnitude across both traces. Whichever side is larger is
    # used consistently for file1 and file2.
    pos_scale_file1, neg_scale_file1 = peak_scales_by_polarity(x_file1_corr)
    pos_scale_file2, neg_scale_file2 = peak_scales_by_polarity(x_file2_corr)

    global_positive_peak = max(pos_scale_file1, pos_scale_file2)
    global_negative_peak = max(neg_scale_file1, neg_scale_file2)

    if global_negative_peak > global_positive_peak:
        norm_polarity = "negative"
        scale_file1 = neg_scale_file1
        scale_file2 = neg_scale_file2
    else:
        norm_polarity = "positive"
        scale_file1 = pos_scale_file1
        scale_file2 = pos_scale_file2

    x_file1_norm, invf_file1 = normalize_by_scale(
        x_file1_corr, scale_file1, label="file1"
    )
    x_file2_norm, invf_file2 = normalize_by_scale(
        x_file2_corr, scale_file2, label="file2"
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    color1 = "C0"
    color2 = "C1"

    label1 = f"{trace_titles[0]} ({os.path.basename(paths[0])})"
    label2 = f"{trace_titles[1]} ({os.path.basename(paths[1])})"

    ax.plot(t_file1, x_file1_norm, color=color1, lw=1.0, label=label1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"Normalized amplitude ({trace_titles[0]})", color=color1)
    ax.tick_params(axis="y", labelcolor=color1)
    ax.set_xlim(plot_tmin, plot_tmax)
    ax.set_ylim(-1.05, 1.05)

    ax2 = ax.twinx()
    ax2.plot(t_file2, x_file2_norm, color=color2, lw=1.0, label=label2)
    ax2.set_ylabel(f"Normalized amplitude ({trace_titles[1]})", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_ylim(-1.05, 1.05)

    # Combined legend.
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, fontsize=8)

    # Display scaling factors and their combined ratio C.
    # These are inverse normalization factors from the plotted interval only.
    inv1 = invf_file1
    inv2 = invf_file2
    try:
        C = (inv2 * args.amp_file2) / (inv1 * args.amp_file1)
    except Exception:
        C = float("nan")

    txt = (
        f"{trace_titles[0]}={inv1:.3e} ×{args.amp_file1:g}\n"
        f"{trace_titles[1]}={inv2:.3e} ×{args.amp_file2:g}\n"
        f"C(V to m/s$^{{-2}}$)={C:.3e}\n"
        f"File1 shift={args.shift_file1_sec:.3e} s\n"
        f"Norm polarity={norm_polarity}\n"
        #f"Crop/norm window={plot_tmin:.3e}–{plot_tmax:.3e} s"
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
