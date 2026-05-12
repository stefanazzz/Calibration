#!/usr/bin/env python3
"""
plot_any_flexible.py - quick viewer for traces stored as:
  - .npz (possibly containing multiple arrays/keys)
  - .npy (single array; may be an object/dict pickled array)
  - .atf (Axon Text File; uses dt = TSamp * TimeUnits, reads [TraceData])

This is a drop-in friendly replacement for plot_any.py
but adds automatic file-type detection so it won't choke on ATF outputs.

Examples:
  python plot_any_flexible.py path/to/stack_A_tapered.npz
  python plot_any_flexible.py path/to/trace_tapered.atf
  python plot_any_flexible.py path/to/any.npy
  python plot_any_flexible.py path/to/any.npz --keys stack,mean

SAtefan March 2026
"""
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

DEFAULT_PREFERRED_KEYS = [
    # taper/stack outputs you often use
    "stack_tapered", "stack_original",
    "stack", "mean", "median", "trimmed",
    # generic
    "x", "y", "data", "signal", "trace"
]


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
    Returns: dt (float), x (1D float array)
    - dt = TSamp * TimeUnits
    - reads the first numeric column after [TraceData]
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
        data.append(float(s.split()[0]))

    x = np.asarray(data, dtype=float).ravel()
    return dt, x
# ----------------------------------------------


def _load_npy_any(path):
    """
    Load a .npy that might be:
      - a numeric ndarray
      - a 0-d object array containing a dict or similar (pickled)
    Returns either:
      - ndarray (numeric)
      - dict-like mapping of arrays
    """
    arr = np.load(path, allow_pickle=True)
    if isinstance(arr, np.ndarray) and arr.dtype == object and arr.shape == ():
        obj = arr.item()
        return obj
    return arr


def _choose_keys_from_mapping(S, keys_arg=None):
    """S is dict-like or np.lib.npyio.NpzFile."""
    files = list(S.keys()) if hasattr(S, "keys") else list(S.files)
    if keys_arg:
        keys = [k.strip() for k in keys_arg.split(",") if k.strip()]
        return keys, files

    # auto: preferred keys that exist, else all numeric 1D/2D arrays
    keys = [k for k in DEFAULT_PREFERRED_KEYS if k in files]
    if not keys:
        keys = []
        for k in files:
            try:
                a = np.asarray(S[k])
            except Exception:
                continue
            if a.dtype == object:
                continue
            if a.ndim in (1, 2) and a.size > 0:
                keys.append(k)
    return keys, files


def _plot_1d(ax, y, dt=None, t=None, label=None, lw=1):
    y = np.asarray(y).ravel()
    if t is not None:
        tt = np.asarray(t).ravel()
        if tt.size != y.size:
            tt = np.arange(y.size) * (dt if dt is not None else 1.0)
    else:
        if dt is not None:
            tt = np.arange(y.size) * dt
        else:
            tt = np.arange(y.size)
    ax.plot(tt, y, lw=lw, label=label)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="Input file: .npz / .npy / .atf")
    ap.add_argument("--keys", default=None,
                    help="Comma-separated keys to plot (for .npz or dict-like .npy). Example: stack,mean")
    ap.add_argument("--max_traces", type=int, default=50,
                    help="Max number of traces to plot if 2D array (default 50)")
    ap.add_argument("--title", default=None, help="Optional plot title")
    args = ap.parse_args()

    path = args.path
    ext = os.path.splitext(path)[1].lower()

    fig, ax = plt.subplots(figsize=(10, 5))

    # ---------- ATF ----------
    if ext == ".atf":
        dt, x = read_atf(path)
        _plot_1d(ax, x, dt=dt, label=f"trace {x.shape}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title(args.title or path)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        plt.show()
        return

    # ---------- NPZ ----------
    if ext == ".npz":
        S = np.load(path, allow_pickle=True)

        # time base if present
        t = np.asarray(S["t"]).ravel() if "t" in S.files else None
        dt = float(S["dt"]) if ("dt" in S.files and t is None) else None

        keys, files = _choose_keys_from_mapping(S, args.keys)
        if not keys:
            raise SystemExit(f"No plottable numeric arrays found. Keys: {files}")

        plotted_any = False
        for k in keys:
            if k not in S.files:
                print(f"Skip missing key: {k}")
                continue
            a = np.asarray(S[k])
            if a.dtype == object:
                print(f"Skip object key: {k}")
                continue
            if a.ndim == 0:
                print(f"Skip scalar key: {k}")
                continue

            if a.ndim == 1:
                _plot_1d(ax, a, dt=dt, t=t, label=f"{k} {a.shape}", lw=1)
                plotted_any = True

            elif a.ndim == 2:
                A = a
                # heuristic: if samples x traces, transpose
                if A.shape[0] > A.shape[1]:
                    A = A.T
                n_tr = min(A.shape[0], args.max_traces)
                n_samp = A.shape[1]
                # time base for samples
                if t is not None and t.size == n_samp:
                    tt = t
                elif dt is not None:
                    tt = np.arange(n_samp) * dt
                else:
                    tt = np.arange(n_samp)
                for i in range(n_tr):
                    ax.plot(tt, A[i], lw=0.8, alpha=0.8, label=f"{k}[{i}]")
                if A.shape[0] > n_tr:
                    print(f"{k}: plotted {n_tr}/{A.shape[0]} traces (max_traces={args.max_traces})")
                plotted_any = True
            else:
                print(f"Skip {k}: ndim={a.ndim}")

        if not plotted_any:
            raise SystemExit("Nothing was plotted (all candidates skipped).")

        ax.set_xlabel("Time (s)" if (t is not None or dt is not None) else "Sample #")
        ax.set_ylabel("Amplitude")
        ax.set_title(args.title or path)

        handles, labels = ax.get_legend_handles_labels()
        if len(labels) <= 15:
            ax.legend(loc="best", fontsize=8)
        else:
            print(f"Legend suppressed ({len(labels)} lines). Use --keys to reduce.")

        # autoscale y with 10% headroom
        ymins, ymaxs = ax.get_ylim()
        m = max(abs(ymins), abs(ymaxs))
        if m > 0:
            ax.set_ylim(-1.1*m, 1.1*m)

        fig.tight_layout()
        plt.show()
        return

    # ---------- NPY ----------
    if ext == ".npy":
        obj = _load_npy_any(path)

        # dict-like (pickled) case
        if isinstance(obj, dict):
            t = np.asarray(obj["t"]).ravel() if "t" in obj else None
            dt = float(obj["dt"]) if ("dt" in obj and t is None) else None

            keys, files = _choose_keys_from_mapping(obj, args.keys)
            if not keys:
                raise SystemExit(f"No plottable numeric arrays found. Keys: {files}")

            plotted_any = False
            for k in keys:
                if k not in obj:
                    print(f"Skip missing key: {k}")
                    continue
                a = np.asarray(obj[k])
                if a.dtype == object:
                    print(f"Skip object key: {k}")
                    continue
                if a.ndim == 0:
                    print(f"Skip scalar key: {k}")
                    continue
                if a.ndim == 1:
                    _plot_1d(ax, a, dt=dt, t=t, label=f"{k} {a.shape}", lw=1)
                    plotted_any = True
                elif a.ndim == 2:
                    A = a
                    if A.shape[0] > A.shape[1]:
                        A = A.T
                    n_tr = min(A.shape[0], args.max_traces)
                    n_samp = A.shape[1]
                    if t is not None and t.size == n_samp:
                        tt = t
                    elif dt is not None:
                        tt = np.arange(n_samp) * dt
                    else:
                        tt = np.arange(n_samp)
                    for i in range(n_tr):
                        ax.plot(tt, A[i], lw=0.8, alpha=0.8, label=f"{k}[{i}]")
                    plotted_any = True
                else:
                    print(f"Skip {k}: ndim={a.ndim}")

            if not plotted_any:
                raise SystemExit("Nothing was plotted (all candidates skipped).")

            ax.set_xlabel("Time (s)" if (t is not None or dt is not None) else "Sample #")
            ax.set_ylabel("Amplitude")
            ax.set_title(args.title or path)
            fig.tight_layout()
            plt.show()
            return

        # plain numeric ndarray case
        a = np.asarray(obj)
        if a.ndim == 0:
            raise SystemExit("NPY contains a scalar; nothing to plot.")
        if a.ndim == 1:
            _plot_1d(ax, a, dt=None, t=None, label=f"array {a.shape}", lw=1)
        elif a.ndim == 2:
            A = a
            if A.shape[0] > A.shape[1]:
                A = A.T
            n_tr = min(A.shape[0], args.max_traces)
            tt = np.arange(A.shape[1])
            for i in range(n_tr):
                ax.plot(tt, A[i], lw=0.8, alpha=0.8, label=f"arr[{i}]")
        else:
            raise SystemExit(f"NPY array ndim={a.ndim} not supported (expected 1D/2D).")

        ax.set_xlabel("Sample #")
        ax.set_ylabel("Amplitude")
        ax.set_title(args.title or path)
        fig.tight_layout()
        plt.show()
        return

    raise SystemExit("Unsupported input type. Use .npz, .npy, or .atf")


if __name__ == "__main__":
    main()
