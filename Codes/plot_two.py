#!/usr/bin/env python3
import argparse
import os
import re
import sys

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
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


def normalization_scales(x_file1, x_file2, norm_mode, norm_reference=None):
    """Return normalization reference text and per-trace peak scales."""
    if norm_mode == "global":
        return (
            "global abs",
            float(np.max(np.abs(x_file1))),
            float(np.max(np.abs(x_file2))),
        )

    if norm_mode == "polarity":
        pos_scale_file1, neg_scale_file1 = peak_scales_by_polarity(x_file1)
        pos_scale_file2, neg_scale_file2 = peak_scales_by_polarity(x_file2)

        if norm_reference == "positive":
            return "positive", pos_scale_file1, pos_scale_file2

        if norm_reference == "negative":
            return "negative", neg_scale_file1, neg_scale_file2

        if norm_reference is not None:
            raise SystemExit(f"Unknown normalization reference: {norm_reference}")

        global_positive_peak = max(pos_scale_file1, pos_scale_file2)
        global_negative_peak = max(neg_scale_file1, neg_scale_file2)

        if global_negative_peak > global_positive_peak:
            return "negative", neg_scale_file1, neg_scale_file2

        return "positive", pos_scale_file1, pos_scale_file2

    raise SystemExit(f"Unknown normalization mode: {norm_mode}")


def actual_amplitude_formatter(scale):
    """Return a formatter that labels normalized ticks as scaled amplitudes."""
    return FuncFormatter(lambda y, _pos: f"{y * scale:.3g}")


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
        --amp-file1 31.6       # positive amplitude divisor for first file
        --amp-file2 2.0        # positive amplitude divisor for second file

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
        help=(
            "Positive amplitude divisor applied to the first file before "
            "normalisation. Default: 31.6."
        ),
    )

    parser.add_argument(
        "--amp-file2",
        type=float,
        default=2.0,
        help=(
            "Positive amplitude divisor applied to the second file before "
            "normalisation. Default: 2.0."
        ),
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

    if not np.isfinite(args.amp_file1) or args.amp_file1 <= 0:
        parser.error("--amp-file1 must be a finite positive value")

    if not np.isfinite(args.amp_file2) or args.amp_file2 <= 0:
        parser.error("--amp-file2 must be a finite positive value")

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
    t_file1, x_file1_corr_unscaled = crop_to_window(
        t_file1_full, x_file1_corr_full, plot_tmin, plot_tmax, label="file1"
    )
    t_file2, x_file2_corr_unscaled = crop_to_window(
        t_file2_full, x_file2_corr_full, plot_tmin, plot_tmax, label="file2"
    )

    legacy_norm_polarity, legacy_scale_file1, legacy_scale_file2 = normalization_scales(
        x_file1_corr_unscaled, x_file2_corr_unscaled, args.norm_mode
    )
    legacy_c = (
        (1.0 / legacy_scale_file2 * args.amp_file2)
        / (1.0 / legacy_scale_file1 * args.amp_file1)
    )

    x_file1_corr = x_file1_corr_unscaled / args.amp_file1
    x_file2_corr = x_file2_corr_unscaled / args.amp_file2

    # Normalization.
    #
    # global:
    #   Each trace is normalized by its own maximum absolute amplitude inside
    #   the cropped/plot window. Therefore each trace reaches +/-1 in its own
    #   axis, unless it is flat.
    #
    # polarity:
    #   The polarity reference is chosen from the unscaled traces to keep the
    #   calibration ratio consistent with the previous code. The amplitudes used
    #   for normalization are still measured after amp-file scaling.
    norm_polarity, scale_file1, scale_file2 = normalization_scales(
        x_file1_corr, x_file2_corr, args.norm_mode, norm_reference=legacy_norm_polarity
    )

    x_file1_norm, _ = normalize_by_scale(
        x_file1_corr, scale_file1, label="file1"
    )
    x_file2_norm, _ = normalize_by_scale(
        x_file2_corr, scale_file2, label="file2"
    )

    ymin = min(np.min(x_file1_norm), np.min(x_file2_norm))
    ymax = max(np.max(x_file1_norm), np.max(x_file2_norm))

    # Use one symmetric normalized y-range for both axes. The tick labels are
    # converted back to per-trace amplitudes, but zero must stay coincident.
    max_abs_y = max(abs(float(ymin)), abs(float(ymax)))
    ylimit = 1.05 * max_abs_y if max_abs_y > 0 else 1.0
    ylimits = (-ylimit, ylimit)
    common_yticks = np.linspace(-ylimit, ylimit, 7)

    fig, ax = plt.subplots(figsize=(10, 5))
    color1 = "C0"
    color2 = "C1"

    label1 = f"{trace_titles[0]} ({os.path.basename(paths[0])})"
    label2 = f"{trace_titles[1]} ({os.path.basename(paths[1])})"

    ax.plot(t_file1, x_file1_norm, color=color1, lw=1.0, label=label1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"Scaled amplitude ({trace_titles[0]}: m s$^{{-2}}$)", color=color1)
    ax.yaxis.set_major_formatter(actual_amplitude_formatter(scale_file1))
    ax.tick_params(axis="y", labelcolor=color1)
    ax.set_xlim(plot_tmin, plot_tmax)
    ax.set_ylim(*ylimits)
    ax.set_yticks(common_yticks)
    ax.axhline(0.0, color="0.35", lw=0.8, alpha=0.45, zorder=0)

    ax2 = ax.twinx()
    ax2.plot(t_file2, x_file2_norm, color=color2, lw=1.0, label=label2)
    ax2.set_ylabel(f"Scaled amplitude ({trace_titles[1]}: V)", color=color2)
    ax2.yaxis.set_major_formatter(actual_amplitude_formatter(scale_file2))
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_ylim(*ylimits)
    ax2.set_yticks(common_yticks)

    # Combined legend.
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, fontsize=8)

    peak_file1 = scale_file1
    peak_file2 = scale_file2
    C = peak_file1 / peak_file2
    if not np.isclose(C, legacy_c, rtol=1e-9, atol=1e-12):
        print(
            "Warning: scaled-peak C differs from the legacy formula: "
            f"new C={C:.12e}, legacy C={legacy_c:.12e}"
        )

    txt = (
        f"Peak {trace_titles[0]}={peak_file1:.3e}\n"
        f"Peak {trace_titles[1]}={peak_file2:.3e}\n"
        f"C({trace_titles[0]}/{trace_titles[1]})={C:.3e}\n"
        f"File1 shift={args.shift_file1_sec:.3e} s\n"
        f"Baseline={args.baseline_start_sec:.3e}-{args.baseline_end_sec:.3e} s\n"
        f"Norm mode={args.norm_mode}\n"
        f"Norm reference={norm_polarity}\n"
        f"Amp scaling: /{args.amp_file1:g}, /{args.amp_file2:g}\n"
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
