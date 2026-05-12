#!/usr/bin/env python3
"""
Visualise waveform from .atf files and classify as 'good' or 'bad'

Usage:
  python Codes/classify_atf.py                         # classify *.atf in current folder
  python Codes/classify_atf.py cal_no_epoxy/flat       # classify *.atf in a folder
  python Codes/classify_atf.py --list list_ch2.txt     # classify files listed in a text file
"""
import os
import glob
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

def read_atf(path: str):
    """
    Reads a Tektronix-style or Richter-Itasca-style .atf (one-column samples) with header like:
      ATF v1.00
      Date=...; TracePoints=65536; TSamp=0.10000; TimeUnits=1.00000e-006; AmpToVolts=0.039063; ...
      [TraceData]
      <samples...>

    Returns: (t_seconds, x_volts, dt_seconds)
    """
    dt = None
    amp_to_volts = 1.0
    data = []

    in_data = False
    header_line = ""

    with open(path, "r", errors="replace") as f:
        for line in f:
            s = line.strip()

            if not in_data:
                if s.startswith("Date="):
                    header_line = s
                if s == "[TraceData]":
                    in_data = True
                    # Parse header_line for TSamp, TimeUnits, AmpToVolts
                    # Example: "... TSamp=0.10000; TimeUnits=1.00000e-006; AmpToVolts=0.039063; ..."
                    parts = [p.strip() for p in header_line.split(";") if p.strip()]
                    kv = {}
                    for p in parts:
                        if "=" in p:
                            k, v = p.split("=", 1)
                            kv[k.strip()] = v.strip()

                    try:
                        tsamp = float(kv.get("TSamp"))
                        time_units = float(kv.get("TimeUnits"))
                        dt = tsamp * time_units
                    except Exception:
                        dt = None

                    try:
                        amp_to_volts = float(kv.get("AmpToVolts", "1.0"))
                    except Exception:
                        amp_to_volts = 1.0
                continue

            # Data section
            if s == "":
                continue
            try:
                data.append(float(s))
            except ValueError:
                # ignore stray non-numeric lines
                pass

    x = np.asarray(data, dtype=float) * amp_to_volts

    if dt is None:
        # Fallback: assume 1 sample unit spacing if dt missing
        dt = 1.0

    t = np.arange(x.size) * dt
    return t, x, dt

def append_line(txt_path: str, line: str):
    os.makedirs(os.path.dirname(os.path.abspath(txt_path)), exist_ok=True)
    with open(txt_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def read_list_file(list_path: str):
    """
    Read .atf paths from a plain text list.
    Blank lines and comments are ignored. Relative paths are resolved relative
    to the list file location.
    """
    list_path = Path(list_path).expanduser().resolve()
    base = list_path.parent
    files = []

    with open(list_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.split("#", 1)[0].strip()
            if not s:
                continue

            path = Path(s).expanduser()
            if not path.is_absolute():
                path = base / path
            files.append(str(path.resolve()))

    return files

def classify_folder(folder: str, pattern: str = "*.atf", listfile: str = None):
    folder = os.path.abspath(folder)
    if listfile:
        files = read_list_file(listfile)
        missing = [p for p in files if not os.path.exists(p)]
        if missing:
            raise SystemExit(
                "Missing file(s) from list:\n" + "\n".join(f"  {p}" for p in missing)
            )
    else:
        files = sorted(glob.glob(os.path.join(folder, pattern)))

    if not files:
        if listfile:
            print(f"No files found in list: {listfile}")
        else:
            print(f"No files matched {pattern} in {folder}")
        return

    good_txt = os.path.join(folder, "good.txt")
    bad_txt  = os.path.join(folder, "bad.txt")

    print(f"Found {len(files)} files.")
    print(f"Writing to:\n  {good_txt}\n  {bad_txt}")

    for i, path in enumerate(files, start=1):
        fname = os.path.basename(path)

        # Skip if already classified (optional convenience)
        already = set()
        for p in (good_txt, bad_txt):
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8", errors="replace") as f:
                    already |= {ln.strip() for ln in f if ln.strip()}
        if fname in already:
            print(f"[{i}/{len(files)}] Skipping already classified: {fname}")
            continue

        print(f"[{i}/{len(files)}] Showing: {fname}")

        t, x, dt = read_atf(path)

        fig, ax = plt.subplots(figsize=(10, 4.5))
        fig.canvas.manager.set_window_title(fname)
        ax.plot(t, x, lw=0.8)
        ax.set_title(fname)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Volts")
        ax.grid(True, alpha=0.25)

        # Buttons
        # Leave space at bottom
        fig.subplots_adjust(bottom=0.22)

        ax_good = fig.add_axes([0.30, 0.05, 0.15, 0.10])
        ax_bad  = fig.add_axes([0.55, 0.05, 0.15, 0.10])
        ax_skip  = fig.add_axes([0.75, 0.05, 0.15, 0.10])

        btn_good = Button(ax_good, "Good")
        btn_bad  = Button(ax_bad, "Bad")
        btn_skip  = Button(ax_skip, "Skip")

        decision = {"value": None}

        def on_good(event):
            decision["value"] = "good"
            append_line(good_txt, fname)
            plt.close(fig)

        def on_bad(event):
            decision["value"] = "bad"
            append_line(bad_txt, fname)
            plt.close(fig)

        def on_skip(event):
            plt.close(fig)

        btn_good.on_clicked(on_good)
        btn_bad.on_clicked(on_bad)
        btn_skip.on_clicked(on_skip)

        # If user closes window without clicking, treat as "skip" 
        def on_close(event):
            if decision["value"] is None: buff=1

        fig.canvas.mpl_connect("close_event", on_close)

        plt.show()

    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualise .atf waveforms and classify them as good or bad."
    )
    parser.add_argument(
        "folder",
        nargs="?",
        default=None,
        help="Folder containing .atf files. Default: current folder, or list file folder if --list is used.",
    )
    parser.add_argument(
        "--list",
        dest="listfile",
        default=None,
        help="Optional text file listing .atf files to classify, one path per line.",
    )
    args = parser.parse_args()

    if args.folder is None and args.listfile:
        folder = str(Path(args.listfile).expanduser().resolve().parent)
    else:
        folder = args.folder or "."

    classify_folder(folder, listfile=args.listfile)
