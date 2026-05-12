#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import sys


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


if __name__ == '__main__':
    # default files (relative to cwd) or provide via args
    files = sys.argv[1:] if len(sys.argv) > 1 else ['./stacked_ch2.atf', './stacked_ch3.atf']
    if len(files) < 2:
        raise SystemExit('Provide two files or run from folder containing stacked_ch2.atf and stacked_ch3.atf')

    paths = [os.path.abspath(f) for f in files]

    # parameters
    window_sec = 0.000125
    baseline_sec = 0.00002
    # user amplitude multipliers
    amp_02 = 31.6
    amp_03 = 2.0

    data_list = []
    for p in paths:
        if not os.path.exists(p):
            print(f'File not found: {p}')
            sys.exit(1)
        t, x, dt = read_atf(p)
        n_window = int(np.floor(window_sec / dt))
        if n_window < 1:
            raise SystemExit('Window too small for sampling interval')
        t_trim = t[:n_window]
        x_trim = x[:n_window].astype(np.float64)

        # baseline from baseline_start_sec to baseline_end_sec
        baseline_start_sec = 0.000005
        baseline_end_sec = 0.000025
        n_baseline_start = int(np.floor(baseline_start_sec / dt))
        n_baseline_end = int(np.floor(baseline_end_sec / dt))
        # clamp indices
        n_baseline_start = max(0, n_baseline_start)
        n_baseline_end = min(len(x_trim), max(n_baseline_start + 1, n_baseline_end))
        if n_baseline_end <= n_baseline_start:
            baseline = 0.0
        else:
            baseline = np.mean(x_trim[n_baseline_start:n_baseline_end])
        x_corr = x_trim - baseline

        # normalize to max absolute amplitude in window
        max_amp = float(np.max(np.abs(x_corr)))
        if max_amp == 0:
            x_norm = x_corr
            inv_factor = float('nan')
        else:
            x_norm = x_corr / max_amp
            inv_factor = 1.0 / max_amp

        data_list.append((t_trim, x_corr, x_norm, max_amp, inv_factor))

    # plot: ch2 (first) on left, ch3 (second) on right
    t0, x2_corr, x2_norm, max_amp1, invf1 = data_list[0]
    t1, x3_corr, x3_norm, max_amp2, invf2 = data_list[1]

    fig, ax = plt.subplots(figsize=(10, 5))
    color2 = 'C0'
    color3 = 'C1'

    ax.plot(t0, x2_norm, color=color2, lw=1.0, label=os.path.basename(paths[0]))
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Normalized amplitude (ch02)', color=color2)
    ax.tick_params(axis='y', labelcolor=color2)
    ax.set_xlim(0, window_sec)
    ax.set_ylim(-1.05, 1.05)

    ax2 = ax.twinx()
    ax2.plot(t1, x3_norm, color=color3, lw=1.0, label=os.path.basename(paths[1]))
    ax2.set_ylabel('Normalized amplitude (ch03)', color=color3)
    ax2.tick_params(axis='y', labelcolor=color3)
    ax2.set_ylim(-1.05, 1.05)

    # combined legend
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, fontsize=8)

    # display scaling factors and their combined ratio C
    # invf1/invf2 are inverse normalization factors (1/max_amp)
    Ch02 = invf1
    Ch03 = invf2
    try:
        C = (Ch03 * amp_03) / (Ch02 * amp_02)
    except Exception:
        C = float('nan')

    # format: label=value (scientific, 3 sig figs) and multiplier ×amp
    txt = (
        f"Ch02 (Acc)={Ch02:.3e} ×{amp_02:g}\n"
        f"Ch03 (PZ)={Ch03:.3e} ×{amp_03:g}\n"
        f"C(V to m/s$^{{-2}}$)={C:.3e}"
    )
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, va='top', ha='left', fontsize=9,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    fig.tight_layout()
    plt.show()
