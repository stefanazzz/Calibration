#!/usr/bin/env python3
"""
Interactive ATF pre-processing:
1) Load two .atf files (single-column TraceData) and parse header metadata
2) Plot in two stacked axes (shared time)
3) Select CLIP interval (used for both traces)
4) Select BASELINE interval (mean used as zero-reference for both traces)
5) Optional demean/detrend (per trace)
6) Tukey taper (alpha=0.02)
7) Save each processed trace to: <orig>_prep.txt (1 column) with 1-line header

Usage examples:
  python atf_prep.py ch1.atf ch2.atf
  python atf_prep.py ch1.atf ch2.atf --demean1 --detrend2 --alpha 0.02

Controls (in the plot window):
  - Drag (left mouse) to select CLIP interval (first)
  - Drag again to select BASELINE interval (second)
  - Press 's' to save outputs once both intervals are selected
  - Press 'r' to reset selections
  - Press 'q' or close window to quit
"""

import argparse
import os
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector
from scipy.signal import detrend as sp_detrend
from scipy.signal.windows import tukey


@dataclass
class ATFTrace:
    path: str
    dt: float
    amp_to_volts: float
    data_volts: np.ndarray
    header_kv: dict


def _parse_header_kv(line: str) -> dict:
    """
    Parses a semicolon-separated key=value header line into dict.
    Example: "Date=...; Time=...; TracePoints=65536; TSamp=0.10000; TimeUnits=1.0e-6; AmpToVolts=0.039063; ..."
    """
    kv = {}
    parts = [p.strip() for p in line.strip().split(";") if p.strip()]
    for p in parts:
        if "=" in p:
            k, v = p.split("=", 1)
            kv[k.strip()] = v.strip()
    return kv


def read_atf_onecol(path: str) -> ATFTrace:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()

    if not lines or not lines[0].startswith("ATF"):
        raise ValueError(f"{path}: not an ATF file (missing 'ATF' first line).")

    # Find the "[TraceData]" line
    try:
        idx_td = next(i for i, ln in enumerate(lines) if ln.strip() == "[TraceData]")
    except StopIteration:
        raise ValueError(f"{path}: could not find [TraceData] section.")

    # The metadata line is typically line 1 (second line), but be robust:
    # search first non-empty line after the ATF version line, before [TraceData]
    meta_line = None
    for ln in lines[1:idx_td]:
        if ln.strip():
            meta_line = ln
            break
    if meta_line is None:
        raise ValueError(f"{path}: could not find metadata line before [TraceData].")

    kv = _parse_header_kv(meta_line)

    # Required fields
    try:
        tsamp = float(kv["TSamp"])
        time_units = float(kv["TimeUnits"])
        amp_to_volts = float(kv["AmpToVolts"])
    except KeyError as e:
        raise ValueError(f"{path}: missing required header key {e!s}.") from None
    except ValueError:
        raise ValueError(f"{path}: could not parse TSamp/TimeUnits/AmpToVolts as floats.") from None

    dt = tsamp * time_units

    # Load trace data: one-column numeric after [TraceData]
    # Skip empty lines
    data_lines = [ln.strip() for ln in lines[idx_td + 1 :] if ln.strip()]
    data = np.array([float(x) for x in data_lines], dtype=float)

    data_volts = data * amp_to_volts

    return ATFTrace(path=path, dt=dt, amp_to_volts=amp_to_volts, data_volts=data_volts, header_kv=kv)


def save_onecol(path_out: str, y: np.ndarray, header_one_liner: str):
    # np.savetxt will prepend "# " to header; we want a single header line anyway.
    np.savetxt(path_out, y.reshape(-1, 1), fmt="%.10e", header=header_one_liner, comments="")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("atf1", help="First .atf file")
    ap.add_argument("atf2", help="Second .atf file")
    ap.add_argument("--alpha", type=float, default=0.02, help="Tukey taper alpha (default 0.02)")
    ap.add_argument("--demean1", action="store_true", help="Demean trace 1 after baseline subtraction")
    ap.add_argument("--detrend1", action="store_true", help="Detrend trace 1 after baseline subtraction")
    ap.add_argument("--demean2", action="store_true", help="Demean trace 2 after baseline subtraction")
    ap.add_argument("--detrend2", action="store_true", help="Detrend trace 2 after baseline subtraction")
    args = ap.parse_args()

    tr1 = read_atf_onecol(args.atf1)
    tr2 = read_atf_onecol(args.atf2)

    if not np.isclose(tr1.dt, tr2.dt):
        print(f"WARNING: dt differs: {tr1.dt} vs {tr2.dt}. Using dt1 for time axis + saving.")
    dt = tr1.dt

    n = min(len(tr1.data_volts), len(tr2.data_volts))
    y1 = tr1.data_volts[:n].copy()
    y2 = tr2.data_volts[:n].copy()
    t = np.arange(n) * dt

    # ---- interactive state ----
    state = {
        "clip": None,        # (i0, i1)
        "baseline": None,    # (j0, j1)
        "phase": "clip",     # "clip" then "baseline"
        "artists": {"clip": [], "base": []},
    }

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(11, 6), constrained_layout=True)
    fig.canvas.manager.set_window_title("ATF Prep: select CLIP then BASELINE; press 's' to save")

    l1, = ax1.plot(t, y1, lw=0.9)
    l2, = ax2.plot(t, y2, lw=0.9)
    ax1.set_ylabel("Trace 1 (V)")
    ax2.set_ylabel("Trace 2 (V)")
    ax2.set_xlabel("Time (s)")
    ax1.set_title("Drag to select CLIP interval (used for both traces)")

    def _clear_spans(kind: str):
        for a in state["artists"][kind]:
            try:
                a.remove()
            except Exception:
                pass
        state["artists"][kind] = []

    def _draw_span(kind: str, x0: float, x1: float):
        # draw translucent spans on both axes
        _clear_spans(kind)
        for ax in (ax1, ax2):
            a = ax.axvspan(x0, x1, alpha=0.2)
            state["artists"][kind].append(a)

    def _x_to_idx(x: float) -> int:
        # nearest sample index, clamped
        i = int(np.round(x / dt))
        return int(np.clip(i, 0, n - 1))

    def on_select(xmin, xmax):
        if xmin == xmax:
            return
        x0, x1 = (xmin, xmax) if xmin < xmax else (xmax, xmin)
        i0, i1 = _x_to_idx(x0), _x_to_idx(x1)
        if i1 <= i0:
            return

        if state["phase"] == "clip":
            state["clip"] = (i0, i1)
            _draw_span("clip", t[i0], t[i1])
            ax1.set_title("Now drag to select BASELINE interval (mean will be zero-reference)")
            state["phase"] = "baseline"
        else:
            state["baseline"] = (i0, i1)
            _draw_span("base", t[i0], t[i1])
            ax1.set_title("Press 's' to save. Press 'r' to reset selections.")
        fig.canvas.draw_idle()

    span = SpanSelector(
        ax1,
        on_select,
        direction="horizontal",
        useblit=True,
        interactive=True,
        props=dict(alpha=0.2),
    )

    def _process_and_save():
        if state["clip"] is None or state["baseline"] is None:
            print("Need both CLIP and BASELINE selections before saving (drag twice).")
            return

        i0, i1 = state["clip"]
        j0, j1 = state["baseline"]

        # Baseline mean from original (full-length) traces over baseline interval
        b1 = float(np.mean(y1[j0:j1]))
        b2 = float(np.mean(y2[j0:j1]))

        # Clip, then baseline-subtract
        z1 = y1[i0:i1].copy() - b1
        z2 = y2[i0:i1].copy() - b2

        # Optional demean/detrend (after baseline subtraction)
        if args.demean1:
            z1 -= np.mean(z1)
        if args.detrend1:
            z1 = sp_detrend(z1, type="linear")
        if args.demean2:
            z2 -= np.mean(z2)
        if args.detrend2:
            z2 -= np.mean(z2)
        if args.detrend2:
            z2 = sp_detrend(z2, type="linear")

        # Tukey taper
        w = tukey(len(z1), alpha=args.alpha) if len(z1) > 1 else np.ones_like(z1)
        z1 *= w
        z2 *= w

        # Save with 1-line header
        def out_name(in_path: str) -> str:
            base = os.path.splitext(os.path.basename(in_path))[0]
            return base + "_prep.txt"

        hdr = (
            f"dt={dt:.10e} "
            f"clip_idx=[{i0},{i1}) baseline_idx=[{j0},{j1}) "
            f"alpha={args.alpha:g} "
            f"file1={os.path.basename(tr1.path)} file2={os.path.basename(tr2.path)} "
            f"proc1=demean:{int(args.demean1)} detrend:{int(args.detrend1)} "
            f"proc2=demean:{int(args.demean2)} detrend:{int(args.detrend2)}"
        )

        out1 = out_name(tr1.path)
        out2 = out_name(tr2.path)
        save_onecol(out1, z1, hdr)
        save_onecol(out2, z2, hdr)

        print(f"Saved:\n  {out1}  (N={len(z1)})\n  {out2}  (N={len(z2)})\nHeader:\n  {hdr}")

    def on_key(event):
        if event.key == "q":
            plt.close(fig)
        elif event.key == "r":
            state["clip"] = None
            state["baseline"] = None
            state["phase"] = "clip"
            _clear_spans("clip")
            _clear_spans("base")
            ax1.set_title("Drag to select CLIP interval (used for both traces)")
            fig.canvas.draw_idle()
            print("Reset selections.")
        elif event.key == "s":
            _process_and_save()

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()


if __name__ == "__main__":
    main()

