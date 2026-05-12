#!/usr/bin/env python3

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, Button


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
    ap.add_argument("in_folder", help="Folder containing aligned_peakwin/")
    ap.add_argument("--file_in", default="stack_A.npz",
                    help="Input file (default stack_A.npz)")
    ap.add_argument("--alpha", type=float, default=0.05,
                    help="Tukey alpha (default 0.05)")
    args = ap.parse_args()

    folder = os.path.abspath(args.in_folder)
    stack_path = os.path.join(folder, "aligned_peakwin", args.file_in)

    if not os.path.exists(stack_path):
        raise SystemExit(f"File not found:\n  {stack_path}")

    # ---- Load ----
    S = np.load(stack_path, allow_pickle=True)
    dt = float(S["dt"])
    x0 = np.asarray(S["stack"]).ravel()
    t = np.arange(len(x0)) * dt

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.subplots_adjust(bottom=0.22)

    line, = ax.plot(t, x0, lw=1)
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

        out_path = stack_path.replace(".npz", "_tapered.npz")

        # Save ONLY tapered cropped window
        np.savez(out_path, dt=dt, stack=x_window)

        print(f"Saved tapered window to:\n  {out_path}")

    btn_redo.on_clicked(redo)
    btn_save.on_clicked(save)

    plt.show()


if __name__ == "__main__":
    main()
