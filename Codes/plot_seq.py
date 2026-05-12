import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

"""
plot_seq.py
S Nielsen May 2026

Interactive visualisation tool for sequentially plotting aligned traces stored in
`.npz` files (typically produced after alignment/stacking workflows).

The script:
- Looks for files named `*_aligned.npz` inside a folder `aligned_peakwin/`
- Each file is expected to contain a trace array (default key: "A_window_aligned")
- Optionally loads a precomputed stack (e.g. median/mean) from `stack_A.npz`
- Displays a matplotlib figure where traces are added interactively

Key features:
- Click inside the plot to add traces one-by-one
- Show a stack trace first (median/mean/trimmed)
- Step backward (remove last trace)
- Reset and restart

This is useful for:
- Quality control of aligned signals
- Visual comparison of stacking behaviour
- Incremental inspection of waveform variability


USAGE
-----

1) Folder mode (default)
-----------------------
Provide a folder that contains:
    - subfolder: aligned_peakwin/
    - inside it: *_aligned.npz files (+ optional stack_A.npz)

Example:
    python plot_seq.py /path/to/data_folder

Expected structure:
    data_folder/
        aligned_peakwin/
            trace1_aligned.npz
            trace2_aligned.npz
            ...
            stack_A.npz   (optional)

--------------------------------------------------

2) List mode (alternative input)
-------------------------------
Provide a text file listing .atf (or other) file paths.

Example:
    python plot_seq.py --list my_files.txt

NOTE:
- The list is currently only used to define the working folder and filenames.
- Make sure the corresponding aligned `.npz` files exist in:
      aligned_peakwin/

--------------------------------------------------

INTERACTIVE CONTROLS
--------------------
Mouse:
    Left click      → Add next trace

Keyboard:
    n              → Add next trace (same as click)
    backspace/delete → Remove last added trace
    r              → Reset (clear all traces)

--------------------------------------------------

CONFIGURATION (inside script)
-----------------------------
You can modify behaviour via these variables:

    plot_stack_first = True
        Show stack before adding traces

    stack_kind = "median"
        Options: "mean", "median", "trimmed", "stack"

    trace_key = "A_window_aligned"
        Key used to extract trace from each .npz file

--------------------------------------------------

OUTPUT
------
- Interactive matplotlib window
- No files are written to disk

--------------------------------------------------

REQUIREMENTS
------------
- numpy
- matplotlib

--------------------------------------------------
"""


ap = argparse.ArgumentParser()
ap.add_argument("folder", nargs="?", default=None,
                help="Folder containing good.txt + atf files (default mode).")
ap.add_argument("--list", default=None,
                help="Alternative: text file listing .atf paths (one per line).")
args = ap.parse_args()

if args.list:
    a_paths = load_list_from_textfile(args.list)
    folder = os.path.dirname(os.path.abspath(args.list))
    bases = [os.path.splitext(os.path.basename(p))[0] for p in a_paths]
else:
    if not args.folder:
        raise SystemExit("Provide either a folder (with good.txt) or --list.")
    folder = os.path.abspath(args.folder)
#   a_paths = load_list_from_goodtxt(folder)
#   bases = [os.path.basename(p).replace("_01.atf", "") for p in a_paths]

aligned_dir = os.path.join(folder, "aligned_peakwin")
#os.makedirs(outdir, exist_ok=True)



# point this to the new output folder
#aligned_dir = "aligned_peakwin"   # <- was "aligned"
plot_stack_first = True           # show stack first if present
stack_kind = "median"             # "mean" | "median" | "trimmed" | "stack"
trace_key = "A_window_aligned"    # <- was "A"

# --- gather files ---
files = sorted(
    f for f in os.listdir(aligned_dir)
    if f.endswith("_aligned.npz")
)
# exclude stack files if you ever saved them with that suffix (usually they don't)
files = [f for f in files if not f.startswith("stack_")]

if not files:
    raise SystemExit(f"No *_aligned.npz found in {aligned_dir}")

# --- load stack (optional) ---
stack_path = os.path.join(aligned_dir, "stack_A.npz")
stack = None

if plot_stack_first and os.path.exists(stack_path):
    S = np.load(stack_path, allow_pickle=True)
    dt = float(S["dt"])
    if stack_kind not in S.files:
        raise KeyError(f"{stack_kind=} not found in {stack_path}. Available: {S.files}")
    stack = S[stack_kind]
else:
    # fall back to dt from first file
    D0 = np.load(os.path.join(aligned_dir, files[0]), allow_pickle=True)
    dt = float(D0["dt"])

t_ms = None  # set once we know trace length

# --- figure setup ---
fig, ax = plt.subplots(figsize=(10, 4))
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Amplitude (V)")
ax.grid(True, alpha=0.25)

lines = []          # matplotlib Line2D objects added so far
shown_files = []    # filenames added so far
i = 0               # next file index to add

# plot stack first if requested
if stack is not None:
    t_ms = np.arange(len(stack)) * dt * 1e3
    ax.plot(t_ms, stack, lw=2.0, label=f"stack ({stack_kind})")
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("Click to add aligned traces (stack shown first)")
else:
    ax.set_title("Click to add aligned traces")

def add_trace(idx: int):
    global t_ms
    fn = files[idx]
    path = os.path.join(aligned_dir, fn)
    D = np.load(path, allow_pickle=True)

    if trace_key not in D.files:
        raise KeyError(f"{trace_key=} not found in {fn}. Available: {D.files}")

    x = D[trace_key]

    if t_ms is None:
        t_ms = np.arange(len(x)) * dt * 1e3

    (ln,) = ax.plot(t_ms, x, lw=0.8, alpha=0.75)
    lines.append(ln)
    shown_files.append(fn)

    ax.set_title(f"{idx+1}/{len(files)}  added: {fn}")
    fig.canvas.draw_idle()

def pop_trace():
    if not lines:
        return
    ln = lines.pop()
    ln.remove()
    fn = shown_files.pop()
    ax.set_title(f"Removed: {fn}  (now {len(lines)}/{len(files)} traces shown)")
    fig.canvas.draw_idle()

def reset():
    while lines:
        pop_trace()
    ax.set_title("Reset. Click to add aligned traces.")
    fig.canvas.draw_idle()

def on_click(event):
    global i
    if event.inaxes != ax:
        return
    if i >= len(files):
        ax.set_title("All traces added. (Press 'r' to reset, backspace to step back)")
        fig.canvas.draw_idle()
        return
    add_trace(i)
    i += 1

def on_key(event):
    global i
    if event.key in ("backspace", "delete"):
        if i > 0:
            i -= 1
        pop_trace()
    elif event.key == "r":
        reset()
        i = 0
    elif event.key == "n":
        on_click(type("E", (), {"inaxes": ax})())

fig.canvas.mpl_connect("button_press_event", on_click)
fig.canvas.mpl_connect("key_press_event", on_key)

plt.tight_layout()
plt.show()

