#!/usr/bin/env python3

"""
Interactive selection and teparing of waform fragment:
1) Reads a waform from an .atf file (e.g. ínput_file.atf) 
2) Makes an interactive plot where mouse can be used to select part of the waveform
3) Applies a TUCKEY taper to the edges of the selection
4) OUtputs result to 'input_file_tapered.atf'

S Nielsen Feb 2026
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, Button

# ----------------- ATF SUPPORT -----------------
def _parse_atf_header_kv(header_lines):
    kv = {}
    for line in header_lines:
        for tok in line.strip().split(';'):
            tok = tok.strip()
            if not tok or '=' not in tok:
                continue
            k, v = tok.split('=', 1)
            kv[k.strip()] = v.strip()
    return kv

def read_atf(fname):
    """
    Read a single-column (or multi-column) ATF and return (dt, x, header_lines).
    dt = TSamp * TimeUnits (both parsed from header key=value pairs).
    Data are always returned as float.
    If multiple columns exist, the first column is used.
    """
    with open(fname, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    header_lines = []
    trace_start = None
    for i, line in enumerate(lines):
        header_lines.append(line)
        if line.strip() == "[TraceData]":
            trace_start = i + 1
            break

    if trace_start is None:
        raise ValueError("ATF file does not contain a [TraceData] block")

    kv = _parse_atf_header_kv(header_lines)
    if "TSamp" not in kv or "TimeUnits" not in kv:
        keys = ", ".join(sorted(kv.keys())) if kv else "(none parsed)"
        raise ValueError(f"TSamp or TimeUnits not found in ATF header. Parsed keys: {keys}")

    tsamp = float(kv["TSamp"])
    timeunits = float(kv["TimeUnits"])
    dt = tsamp * timeunits

    data = []
    for line in lines[trace_start:]:
        s = line.strip()
        if not s:
            continue
        # take first column if tab/space separated
        data.append(float(s.split()[0]))

    x = np.asarray(data, dtype=float).ravel()
    return dt, x, header_lines

def write_atf(outname, header_lines, x):
    """
    Write ATF preserving header verbatim up to and including [TraceData],
    then write new trace values as floats (one per line).
    """
    with open(outname, "w", encoding="utf-8") as f:
        for line in header_lines:
            f.write(line)
            if line.strip() == "[TraceData]":
                break
        for v in np.asarray(x, dtype=float).ravel():
            f.write(f"{v:.10g}\n")
# ----------------------------------------------


# -----------------------------
# Simple Tukey window
# -----------------------------
def tukey_window(n, alpha=0.05):
    if n <= 1:
        return np.ones(n)

    if alpha <= 0:
        return np.ones(n)

    if alpha >= 1:
        m = np.arange(n)
        return 0.5 * (1 - np.cos(2*np.pi*m/(n-1)))

    w = np.ones(n)
    edge = int(alpha * (n - 1) / 2)

    for i in range(edge):
        val = 0.5 * (1 + np.cos(np.pi * (2*i/(alpha*(n-1)) - 1)))
        w[i] = val
        w[-(i+1)] = val

    return w


def autoscale_y(ax, y, headroom=0.10):
    m = np.max(np.abs(y)) if y.size else 1.0
    if m <= 0:
        m = 1.0
    lim = (1.0 + headroom) * m
    ax.set_ylim(-lim, lim)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("in_path", help="Either an ATF/NPZ file, or a folder containing aligned_peakwin/")
    ap.add_argument("--file_in", default="stack_A.npz",
                    help="If in_path is a folder, input file inside aligned_peakwin/ (default stack_A.npz)")
    ap.add_argument("--alpha", type=float, default=0.05,
                    help="Tukey alpha (default 0.05)")
    args = ap.parse_args()

    in_path = os.path.abspath(args.in_path)

    # If user passed a file directly, use it. Otherwise keep original folder/aligned_peakwin logic.
    if os.path.isfile(in_path) and os.path.splitext(in_path)[1].lower() in (".npz", ".atf"):
        stack_path = in_path
    else:
        folder = in_path
        stack_path = os.path.join(folder, "aligned_peakwin", args.file_in)

    if not os.path.exists(stack_path):
        raise SystemExit(f"File not found:\n  {stack_path}")

    ext = os.path.splitext(stack_path)[1].lower()

    # ---- Load ----
    atf_header = None
    if ext == ".atf":
        dt, x0, atf_header = read_atf(stack_path)
    else:
        S = np.load(stack_path, allow_pickle=True)
        dt = float(S["dt"])
        x0 = np.asarray(S["stack"]).ravel().astype(float)

    t = np.arange(len(x0)) * dt

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.subplots_adjust(bottom=0.22)

    ax.plot(t, x0, lw=1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Drag to select window for Tukey taper (crop + save)")

    ax.set_xlim(t[0], t[-1])
    autoscale_y(ax, x0)

    # Store selection globally
    selection = {"i0": None, "i1": None}

    # ---- Selection callback ----
    def onselect(eclick, erelease):
        if eclick.xdata is None or erelease.xdata is None:
            return

        tmin = min(eclick.xdata, erelease.xdata)
        tmax = max(eclick.xdata, erelease.xdata)

        i0 = int(np.searchsorted(t, tmin))
        i1 = int(np.searchsorted(t, tmax))

        if i1 <= i0 + 1:
            return

        selection["i0"] = i0
        selection["i1"] = i1

        # Visual preview: show tapered window (cropped)
        w = tukey_window(i1 - i0, alpha=args.alpha)
        x_preview = x0[i0:i1] * w
        t_preview = t[i0:i1]

        ax.clear()
        ax.plot(t_preview, x_preview, lw=1)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title("Preview of tapered window (drag again to refine)")
        autoscale_y(ax, x_preview)

        fig.canvas.draw_idle()

    selector = RectangleSelector(
        ax, onselect,
        useblit=True,
        button=[1],
        interactive=True
    )

    # ---- Buttons ----
    ax_redo = plt.axes([0.08, 0.06, 0.12, 0.08])
    ax_save = plt.axes([0.25, 0.06, 0.12, 0.08])

    btn_redo = Button(ax_redo, "Redo")
    btn_save = Button(ax_save, "Save")

    def redo(event):
        selection["i0"], selection["i1"] = None, None
        ax.clear()
        ax.plot(t, x0, lw=1)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title("Drag to select window for Tukey taper")
        ax.set_xlim(t[0], t[-1])
        autoscale_y(ax, x0)
        fig.canvas.draw_idle()

    def save(event):
        i0 = selection["i0"]
        i1 = selection["i1"]

        if i0 is None or i1 is None:
            print("No window selected.")
            return

        w = tukey_window(i1 - i0, alpha=args.alpha)
        x_window = x0[i0:i1] * w

        base, _ = os.path.splitext(stack_path)
        out_path = base + "_tapered" + ext

        # Save ONLY tapered cropped window
        if ext == ".atf":
            write_atf(out_path, atf_header, x_window)
        else:
            np.savez(out_path, dt=dt, stack=x_window)

        print(f"Saved tapered window to:\n  {out_path}")

    btn_redo.on_clicked(redo)
    btn_save.on_clicked(save)

    plt.show()


if __name__ == "__main__":
    main()
