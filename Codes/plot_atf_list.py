#!/usr/bin/env python3
"""
plot_atf_list.py
Sequential interactive plot list of .atf files

Stefan March 2026

- Reads a list of `.atf` files from a text file
- Loads each waveform (Insite ATF format)
- Displays one waveform at a time in a matplotlib window
- Allows stepping through files interactively
- Provides a button/shortcut to copy the current filename to clipboard (Linux, via xclip)

This is useful for:
- Rapid quality control of many ATF traces
- Browsing large datasets without opening multiple plots
- Identifying good/bad signals interactively
- Copying filenames of selected traces for further processing

Run:
    python plot_atf_list.py  # (defaults to flist.txt of atf to plot)
    python plot_atf_list.py my_list.txt # (custom file with list of atf)

INTERACTIVE CONTROLS
--------------------
    Click anywhere (not on buttons) → Next waveform
    Right arrow / space / enter → Next waveform
    Left arrow / backspace      → Previous waveform
    c                           → Copy current filename to clipboard

OUTPUT
------
- Interactive matplotlib window. No files are written


NOTES
-----
- Expects ATF files in the "Insite" format with header fields:
    ATF v1.00
    Date=23-04-2026; Time=10:47:54.9920000; TracePoints=65536; TSamp=0.10000; TimeUnits= 1.00000 e-06; AmpToVolts=1.0000; TraceMaxVolts=5.0000; PTime=0.00000; STime=0.00000; TracePoints, TSamp, TimeUnits, AmpToVolts
- Automatically converts data to volts using AmpToVolts
- Time axis is reconstructed from sampling interval

"""

import os
import re
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# >>> NEW IMPORTS FOR CLIPBOARD <<<
import subprocess
import shutil


# ========================
# ATF reader (your Insite format)
# ================================
def read_atf(path: str) -> tuple[np.ndarray, float]:
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
    return x, dt
#################
def read_atf_old(path: str) -> tuple[np.ndarray, float]:
    with open(path, "r", errors="ignore") as f:
        _ = f.readline()
        header2 = f.readline()

        m_tp = re.search(r"TracePoints=(\d+)", header2)
        m_ts = re.search(r"TSamp=([0-9.]+)", header2)
        m_tu = re.search(r"TimeUnits=([0-9.eE+-]+)", header2)
        m_a2v = re.search(r"AmpToVolts=([0-9.eE+-]+)", header2)
        if not (m_tp and m_ts and m_tu and m_a2v):
            raise ValueError(f"Could not parse ATF header in {path}")

        n = int(m_tp.group(1))
        tsamp = float(m_ts.group(1))
        time_units = float(m_tu.group(1))
        amp_to_volts = float(m_a2v.group(1))
        dt = tsamp * time_units

        # seek to [TraceData]
        line = f.readline()
        while line and "[TraceData]" not in line:
            line = f.readline()

        data = []
        for _ in range(n):
            s = f.readline()
            if not s:
                break
            #data.append(int(s.strip()))
            data.append(float(s.strip()))

    x = np.asarray(data, dtype=np.float64) * amp_to_volts
    return x, dt


# ==================================
# NEW: Linux clipboard using xclip
# ==================================
def copy_text_linux(text: str, also_primary: bool = True) -> tuple[bool, str]:

    if not shutil.which("xclip"):
        return False, "xclip not found (sudo apt install xclip)"

    try:
        subprocess.run(
            ["xclip", "-in", "-selection", "clipboard"],
            input=text.encode("utf-8"),
            check=True,
        )

        if also_primary:
            subprocess.run(
                ["xclip", "-in", "-selection", "primary"],
                input=text.encode("utf-8"),
                check=True,
            )

        return True, "Copied ✓ (clipboard + primary)"

    except Exception as e:
        return False, f"xclip failed: {e}"


# ==================================
# Viewer class
# ==================================
class ATFClickViewer:

    def __init__(self, files):
        self.files = files
        self.i = 0

        self.fig, self.ax = plt.subplots(figsize=(10, 4))
        plt.subplots_adjust(bottom=0.22)

        # Buttons
        ax_prev = self.fig.add_axes([0.10, 0.05, 0.12, 0.08])
        ax_next = self.fig.add_axes([0.23, 0.05, 0.12, 0.08])
        ax_copy = self.fig.add_axes([0.60, 0.05, 0.18, 0.08])

        self.btn_prev = Button(ax_prev, "Prev")
        self.btn_next = Button(ax_next, "Next")
        self.btn_copy = Button(ax_copy, "Copy filename")

        self.btn_prev.on_clicked(lambda evt: self.prev())
        self.btn_next.on_clicked(lambda evt: self.next())
        self.btn_copy.on_clicked(lambda evt: self.do_copy())

        self.status = self.fig.text(0.02, 0.94, "", fontsize=9)

        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        self.load_and_plot()

    def current_path(self):
        return self.files[self.i]

    def load_and_plot(self):
        path = self.current_path()
        self.ax.clear()

        try:
            x, dt = read_atf(path)
            t = np.arange(len(x)) * dt
            self.ax.plot(t, x, lw=0.8)
            self.ax.set_xlabel("Time (s)")
            self.ax.set_ylabel("Volts")
            self.ax.grid(True, alpha=0.3)
            self.ax.set_title(os.path.basename(path))

            self.status.set_text(
                f"[{self.i+1}/{len(self.files)}] {path}"
            )

        except Exception as e:
            self.ax.text(
                0.05, 0.5,
                f"Failed to read:\n{path}\n\n{e}",
                transform=self.ax.transAxes
            )

        self.fig.canvas.draw_idle()

    def on_click(self, event):
        if event.inaxes not in [self.btn_prev.ax, self.btn_next.ax, self.btn_copy.ax]:
            self.next()

    def on_key(self, event):
        if event.key in ("right", " ", "enter"):
            self.next()
        elif event.key in ("left", "backspace"):
            self.prev()
        elif event.key == "c":
            self.do_copy()

    def next(self):
        if self.i < len(self.files) - 1:
            self.i += 1
            self.load_and_plot()

    def prev(self):
        if self.i > 0:
            self.i -= 1
            self.load_and_plot()

    def do_copy(self):
        text = self.current_path()
        ok, msg = copy_text_linux(text, also_primary=True)
        self.status.set_text(self.status.get_text() + f" | {msg}")
        self.fig.canvas.draw_idle()


# ==================================
# Main
# ==================================
def read_flist(path="flist.txt"):
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]


def main():

    if __name__ == "__main__":
        flist = sys.argv[1] if len(sys.argv) > 1 else "flist.txt"
        files = read_flist(flist)

        #files = read_flist()
        ATFClickViewer(files)
        plt.show()


if __name__ == "__main__":
    main()
