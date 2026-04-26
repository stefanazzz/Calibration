#!/usr/bin/env python3
import numpy as np
import matplotlib
matplotlib.use('Agg')
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


def read_generic(path: str):
    if path.lower().endswith('.atf'):
        return read_atf(path)
    data = np.loadtxt(path)
    if data.ndim == 1:
        x = data.astype(np.float64)
        dt = 1.0
        t = np.arange(len(x), dtype=np.float64) * dt
        return t, x, dt
    else:
        t = data[:, 0].astype(np.float64)
        x = data[:, 1].astype(np.float64)
        return t, x, None


if __name__ == '__main__':
    flist_path = 'flist.txt'
    if not os.path.exists(flist_path):
        raise SystemExit(f"{flist_path} not found in cwd={os.getcwd()}")

    files = [l.strip() for l in open(flist_path) if l.strip()]
    if not files:
        raise SystemExit("flist.txt is empty")

    fig, ax = plt.subplots(figsize=(12, 6))

    global_t_min = float('inf')
    global_t_max = float('-inf')
    global_y_min = float('inf')
    global_y_max = float('-inf')

    for f in files:
        try:
            t, x, dt = read_generic(f)
        except Exception:
            try:
                t, x, dt = read_atf(f)
            except Exception as e:
                print(f"Skipping {f}: could not read ({e})")
                continue

        ax.plot(t, x, lw=0.8, label=os.path.basename(f))

        if t.size:
            global_t_min = min(global_t_min, float(np.nanmin(t)))
            global_t_max = max(global_t_max, float(np.nanmax(t)))
        if x.size:
            global_y_min = min(global_y_min, float(np.nanmin(x)))
            global_y_max = max(global_y_max, float(np.nanmax(x)))

    if global_t_min == float('inf'):
        raise SystemExit('No data plotted')

    yrange = global_y_max - global_y_min
    pad = yrange * 0.05 if yrange > 0 else 1.0
    ax.set_xlim(global_t_min, global_t_max)
    ax.set_ylim(global_y_min - pad, global_y_max + pad)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('AU')
    ax.legend(fontsize=8, ncol=2)

    fig.tight_layout()
    outname = sys.argv[1] if len(sys.argv) > 1 else 'plot_together_saved.png'
    fig.savefig(outname, dpi=200)
    print(f"Saved plot to {outname}")
