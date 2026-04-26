#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import re
import os


def read_atf(path: str):
    with open(path, 'r', errors='ignore') as f:
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

        n = get_header_int('TracePoints')
        tsamp = get_header_float('TSamp')
        time_units = get_header_float('TimeUnits')
        amp_to_volts = get_header_float('AmpToVolts')
        dt = tsamp * time_units

        line = f.readline()
        while line and '[TraceData]' not in line:
            line = f.readline()
        if not line:
            raise ValueError('[TraceData] section not found')

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
    path = './stacked_ch2.atf'
    if not os.path.exists(path):
        raise SystemExit(f'{path} not found')

    t, x, dt = read_atf(path)
    window_sec = 0.000125
    idx = t <= window_sec
    if not np.any(idx):
        raise SystemExit('No samples in requested window')
    t_trim = t[idx]
    x_trim = x[idx]

    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(t_trim, x_trim, lw=1)
    ax.set_xlim(0, window_sec)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude (V)')
    ax.set_title(os.path.basename(path))
    fig.tight_layout()
    plt.show()
