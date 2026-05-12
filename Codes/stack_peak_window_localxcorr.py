#!/usr/bin/env python3
"""
Sync timeline of selected *.atf events in FOLDER. 
Selection list read from file "good.txt".
Stack synced events in to obtain single average trace.

Strategy:
1) Crop ABSOLUTE time window: [tmin, tmax] seconds (default 0.3–0.4 ms)
2) Pick sync point: max(abs(x)) within that window (polarity-safe)
3) Align by shifting traces so picked peak lands at a common index (median of picks)
4) Optional refinement: LOCAL cross-correlation around first packet peak vs template ("stack" or "first")
5) Stack: mean + median + optional trimmed-mean

Input:
- Provide a folder containing good.txt listing *_01.atf (and optionally *_02.atf),
  OR provide a plain file list (one .atf per line) with --list.

Outputs:
- aligned_peakwin/
    *_aligned.npz (A and maybe B)
    stack_A.npz (mean/median/trimmed stacks for A)
    stack_B.npz (if B exists)
    shifts.txt

Argument: FOLDER (mandatory) Folder containing good.txt and .atf files

Arguments (optional):
--list FILE             Text file listing .atf paths (one per line)

--tmin_ms FLOAT         Absolute window start time in milliseconds
--tmax_ms FLOAT         Absolute window end time in milliseconds

--refine_xcorr          Enable post-peak cross-correlation refinement
--reference {stack,first}
                        Reference for xcorr refinement (mean stack or first trace)
--max_lag_samp INT      Maximum allowed xcorr lag in samples (±)
--xcorr_halfwin_samp INT
                        Half-width of local xcorr window around first packet peak

--stack_method {mean,median,trimmed}
                        Stack designated as output "stack"
--trim_frac FLOAT       Fraction trimmed at each end for trimmed mean stack

--also_pair_B           Apply same alignment shifts to matching *_02.atf files

Example inline usage:
python stack_peak_window_localxcorr.py FOLDER \
  --refine_xcorr \
  --max_lag_samp 6 \
  --xcorr_halfwin_samp 20

Plotting: 
can plot the mean and the individual synced events using plot_seq.py

S. Nielsen February 2026
"""

from __future__ import annotations
import os
import re
import argparse
import numpy as np


# -----------------------------
# ATF reader (ATF is the output from Insite checchi leach)
# -----------------------------
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
########################
def read_atf_old(path: str) -> tuple[np.ndarray, float]:
    """Return (x_volts, dt_seconds)."""
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
        dt = tsamp * time_units  # seconds

        # seek to [TraceData]
        line = f.readline()
        while line and "[TraceData]" not in line:
            line = f.readline()

        data = []
        for _ in range(n):
            s = f.readline()
            if not s:
                break
            data.append(int(s.strip()))
        if len(data) != n:
            raise ValueError(f"Expected {n} samples, got {len(data)} in {path}")

    x = np.asarray(data, dtype=np.float64) * amp_to_volts
    return x, dt


def demean_early(x: np.ndarray, frac: float = 0.05) -> np.ndarray:
    n0 = max(1, int(frac * len(x)))
    return x - np.mean(x[:n0])


# -----------------------------
# Alignment helpers
# -----------------------------
def shift_by_samples(x: np.ndarray, shift: int) -> np.ndarray:
    """Integer-sample shift with zero padding."""
    x = np.asarray(x)
    y = np.zeros_like(x)
    if shift == 0:
        return x.copy()
    if shift > 0:
        y[shift:] = x[:-shift]
    else:
        s = -shift
        y[:-s] = x[s:]
    return y


def crop_abs_window(x: np.ndarray, dt: float, tmin: float, tmax: float) -> tuple[np.ndarray, int, int]:
    """
    Crop x to absolute-time window [tmin, tmax] using dt, assuming trace starts at t=0.
    Returns (xw, i0, i1) where i0/i1 are indices into full trace (i1 exclusive).
    """
    n = len(x)
    i0 = int(np.ceil(tmin / dt))
    i1 = int(np.floor(tmax / dt)) + 1  # include endpoint
    i0 = max(0, min(n, i0))
    i1 = max(0, min(n, i1))
    if i1 <= i0 + 2:
        raise ValueError(f"Window [{tmin},{tmax}] s too small/outside trace (n={n}, dt={dt}).")
    return x[i0:i1], i0, i1


def xcorr_refine_shift(ref: np.ndarray, x: np.ndarray, max_lag: int) -> int:
    """
    Integer lag maximizing cross-correlation within +/- max_lag samples.
    Returns shift to apply to x to align to ref.
    """
    r = np.asarray(ref) - np.mean(ref)
    y = np.asarray(x) - np.mean(x)

    c = np.correlate(r, y, mode="full")
    mid = len(c) // 2
    lo = mid - max_lag
    hi = mid + max_lag + 1
    c_win = c[lo:hi]
    best = int(np.argmax(c_win))
    lag = (lo + best) - mid  # +lag means x is to the right of ref
    return int(-lag)


def xcorr_refine_shift_local(ref: np.ndarray, x: np.ndarray, center_idx: int,
                             halfwin: int, max_lag: int) -> int:
    """
    LOCAL cross-correlation refinement using only a short sub-window centered at center_idx.

    This is designed for use in case the first wave packet is coherent across events,
    but later coda (reflections/refractions/source-position drift) is not. By correlating only
    a short window around the first packet peak, we avoid 'late arrivals voting' the shift.

    Returns shift to apply to x to align to ref (integer samples).
    """
    n = len(x)
    i0 = max(0, center_idx - halfwin)
    i1 = min(n, center_idx + halfwin + 1)

    # Need enough samples to correlate meaningfully
    if i1 - i0 < 8:
        return 0

    return xcorr_refine_shift(ref[i0:i1], x[i0:i1], max_lag=max_lag)


def trimmed_mean(X: np.ndarray, trim_frac: float = 0.1, axis: int = 0) -> np.ndarray:
    X = np.asarray(X)
    if not (0 <= trim_frac < 0.5):
        raise ValueError("trim_frac must be in [0, 0.5).")
    if trim_frac == 0:
        return np.mean(X, axis=axis)

    n = X.shape[axis]
    k = int(np.floor(trim_frac * n))
    if 2 * k >= n:
        raise ValueError("trim_frac too large for number of traces.")

    Xs = np.sort(X, axis=axis)
    sl = [slice(None)] * X.ndim
    sl[axis] = slice(k, n - k)
    return np.mean(Xs[tuple(sl)], axis=axis)


def stack_from_method(X: np.ndarray, method: str, trim_frac: float) -> np.ndarray:
    if method == "mean":
        return np.mean(X, axis=0)
    if method == "median":
        return np.median(X, axis=0)
    if method == "trimmed":
        return trimmed_mean(X, trim_frac=trim_frac, axis=0)
    raise ValueError("stack_method must be mean|median|trimmed")


# -----------------------------
# Main workflow
# -----------------------------
def load_list_from_goodtxt(folder: str) -> list[str]:
    good = os.path.join(folder, "good.txt")
    if not os.path.exists(good):
        raise SystemExit(f"Missing good.txt in {folder} (or pass --list).")
    names = [l.strip() for l in open(good) if l.strip()]
    a_files = sorted([n for n in names if n.endswith("_01.atf")])
    if not a_files:
        raise SystemExit("No *_01.atf listed in good.txt")
    return [os.path.join(folder, f) for f in a_files]


def load_list_from_textfile(path: str) -> list[str]:
    folder = os.path.dirname(os.path.abspath(path))
    files = []
    for l in open(path):
        l = l.strip()
        if not l:
            continue
        files.append(l if os.path.isabs(l) else os.path.join(folder, l))
    if not files:
        raise SystemExit(f"No files found in list: {path}")
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", nargs="?", default=None,
                    help="Folder containing good.txt + atf files (default mode).")
    ap.add_argument("--list", default=None,
                    help="Alternative: text file listing .atf paths (one per line).")
    ap.add_argument("--tmin_ms", type=float, default=0.3, help="Window start (ms), absolute.")
    ap.add_argument("--tmax_ms", type=float, default=0.4, help="Window end (ms), absolute.")
    ap.add_argument("--refine_xcorr", action="store_true", help="Enable xcorr refinement.")
    ap.add_argument("--max_lag_samp", type=int, default=40, help="Max xcorr lag in samples (tight!).")
    ap.add_argument("--xcorr_halfwin_samp", type=int, default=30,
                    help="Half-width (samples) of LOCAL xcorr window around first packet peak.")
    ap.add_argument("--reference", choices=["stack", "first"], default="stack",
                    help="Reference for xcorr refine.")
    ap.add_argument("--stack_method", choices=["mean", "median", "trimmed"], default="mean",
                    help="Which stack to report as 'stack'.")
    ap.add_argument("--trim_frac", type=float, default=0.1, help="Trim fraction for trimmed mean.")
    ap.add_argument("--also_pair_B", action="store_true",
                    help="If using *_01.atf, also shift matching *_02.atf if present.")
    args = ap.parse_args()

    if args.list:
        a_paths = load_list_from_textfile(args.list)
        folder = os.path.dirname(os.path.abspath(args.list))
        bases = [os.path.splitext(os.path.basename(p))[0] for p in a_paths]
    else:
        if not args.folder:
            raise SystemExit("Provide either a folder (with good.txt) or --list.")
        folder = os.path.abspath(args.folder)
        a_paths = load_list_from_goodtxt(folder)
        bases = [os.path.basename(p).replace("_01.atf", "") for p in a_paths]

    # Read first to get dt
    x0, dt = read_atf(a_paths[0])
    fs = 1.0 / dt
    tmin = args.tmin_ms * 1e-3
    tmax = args.tmax_ms * 1e-3

    outdir = os.path.join(folder, "aligned_peakwin")
    os.makedirs(outdir, exist_ok=True)

    print(f"A traces: {len(a_paths)} | dt={dt*1e9:.1f} ns | fs={fs/1e6:.3f} MHz")
    print(f"ABS window: {args.tmin_ms:.3f}–{args.tmax_ms:.3f} ms")
    print(f"Peak pick: argmax(|x|) within window")
    print(f"Refine xcorr: {args.refine_xcorr} (LOCAL halfwin={args.xcorr_halfwin_samp} samp, max_lag={args.max_lag_samp} samp, ref={args.reference})")

    # Load, demean, crop window, pick peaks
    Xw = []
    peak_idx = []
    i0s, i1s = [], []
    for p in a_paths:
        x, dtp = read_atf(p)
        if abs(dtp - dt) > 1e-15:
            raise SystemExit(f"Sampling mismatch in {p}")
        x = demean_early(x)
        xw, i0, i1 = crop_abs_window(x, dt, tmin, tmax)
        Xw.append(xw)
        i0s.append(i0); i1s.append(i1)
        peak_idx.append(int(np.argmax(np.abs(xw))))

    # Common length (trim to shortest window length)
    nmin = min(len(x) for x in Xw)
    Xw = [x[:nmin] for x in Xw]
    peak_idx = [min(pi, nmin - 1) for pi in peak_idx]

    # Peak alignment shifts (within window arrays)
    target = int(np.median(peak_idx))
    shifts_peak = np.array([target - pi for pi in peak_idx], dtype=int)

    X_peak = np.vstack([shift_by_samples(x, s) for x, s in zip(Xw, shifts_peak)])

    # Optional xcorr refinement (LOCAL around first packet peak)
    shifts_xc = np.zeros(len(a_paths), dtype=int)
    X_al = X_peak
    if args.refine_xcorr:
        ref = X_peak.mean(axis=0) if args.reference == "stack" else X_peak[0]
        Xr = []

        # Correlate ONLY around the peak-aligned target index to avoid late incoherent coda steering the shift
        center_idx = target
        halfwin = int(args.xcorr_halfwin_samp)

        for i, x in enumerate(X_peak):
            s2 = xcorr_refine_shift_local(ref, x, center_idx=center_idx,
                                          halfwin=halfwin, max_lag=args.max_lag_samp)
            shifts_xc[i] = int(s2)
            Xr.append(shift_by_samples(x, int(s2)))
        X_al = np.vstack(Xr)

    # Stacks
    stack_mean = np.mean(X_al, axis=0)
    stack_median = np.median(X_al, axis=0)
    stack_trim = trimmed_mean(X_al, trim_frac=args.trim_frac, axis=0)
    stack_primary = stack_from_method(X_al, args.stack_method, args.trim_frac)

    # Save per-event aligned windows + optional B
    # Note: we store window-aligned data (not full-length shifted trace) to keep this simple.
    for base, apath, xwin_al, sp, sx in zip(bases, a_paths, X_al, shifts_peak, shifts_xc):
        out = {"dt": dt, "fs": fs, "tmin_s": tmin, "tmax_s": tmax,
               "shift_peak_samp": int(sp), "shift_xcorr_samp": int(sx),
               "A_window_aligned": xwin_al}

        if args.also_pair_B and base.endswith("") and apath.endswith("_01.atf"):
            bpath = os.path.join(folder, base + "_02.atf")
            if os.path.exists(bpath):
                xb, dtb = read_atf(bpath)
                if abs(dtb - dt) > 1e-15:
                    raise SystemExit(f"Sampling mismatch in {bpath}")
                xb = demean_early(xb)
                xb_w, _, _ = crop_abs_window(xb, dt, tmin, tmax)
                xb_w = xb_w[:nmin]
                xb_al = shift_by_samples(xb_w, int(sp))
                if args.refine_xcorr:
                    xb_al = shift_by_samples(xb_al, int(sx))
                out["B_window_aligned"] = xb_al

        np.savez(os.path.join(outdir, base + "_aligned.npz"), **out)

    # Save stacks + metadata
    np.savez(os.path.join(outdir, "stack_A.npz"),
             dt=dt, fs=fs, tmin_s=tmin, tmax_s=tmax,
             stack_method=args.stack_method,
             mean=stack_mean, median=stack_median, trimmed=stack_trim, stack=stack_primary,
             shifts_peak=shifts_peak, shifts_xcorr=shifts_xc,
             bases=np.array(bases, dtype=object))

    # Save shifts.txt
    with open(os.path.join(outdir, "shifts.txt"), "w") as f:
        for base, sp, sx in zip(bases, shifts_peak, shifts_xc):
            f.write(f"{base}  shift_peak_samp={int(sp)}  shift_xcorr_samp={int(sx)}\n")

    print(f"\nSaved aligned windows + stacks in: {outdir}")
    print(f"  stack_A.npz includes mean/median/trimmed and chosen 'stack' ({args.stack_method})")

if __name__ == "__main__":
    main()

