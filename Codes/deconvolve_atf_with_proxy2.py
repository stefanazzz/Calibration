#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
deconvolve_atf_with_proxy2.py

- Input:  measured trace (.atf) and proxy impulse response (.npz)
- Output:
    1) <input_base>_deconvolved.atf          (ONLY the deconvolved trace)
    2) <input_base>_deconvolved_info.npz     (diagnostics + metadata)

Deconvolution is done in the frequency domain with either:
  - Tikhonov/Wiener-style regularisation: X * H* / (|H|^2 + lam^2)
  - Waterlevel: floor on |H|^2

Assumes: x_measured(t) = s_true(t) * h_proxy(t)

Created on Wed Feb 11 13:28:09 2026

@author: stefan
"""

import argparse
import os
from datetime import datetime
import numpy as np


# --- ATF reader (compatible with your classify_atf.py expectations) ---
def read_atf(path: str):
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

            if s == "":
                continue
            try:
                data.append(float(s))
            except ValueError:
                pass

    x = np.asarray(data, dtype=float) * amp_to_volts
    if dt is None:
        dt = 1.0
    t = np.arange(x.size) * dt
    return t, x, dt


def write_atf(path: str, x_volts: np.ndarray, dt_s: float, amp_to_volts: float = 1.0):
    """
    Write a minimal Tektronix-style ATF v1.00:
      - We write values in "counts" such that counts*AmpToVolts = volts.
      - Default AmpToVolts=1.0 means we write volts directly.
    This matches how read_atf() scales: x = raw * AmpToVolts.
    """
    x_volts = np.asarray(x_volts, dtype=float).ravel()

    # store as "raw" values; reader will multiply by AmpToVolts
    raw = x_volts / float(amp_to_volts)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    npts = raw.size

    # Keep it simple: use seconds as time units
    time_units = 1.0
    tsamp = float(dt_s) / time_units

    header = [
        "ATF v1.00",
        f"Date={now}; TracePoints={npts}; TSamp={tsamp:.12g}; TimeUnits={time_units:.12g}; AmpToVolts={amp_to_volts:.12g};",
        "[TraceData]",
    ]

    with open(path, "w", newline="\n") as f:
        for line in header:
            f.write(line + "\n")
        # write one sample per line
        for v in raw:
            f.write(f"{v:.12g}\n")


def choose_proxy_key(npz_obj, prefer_keys):
    keys = list(npz_obj.files)
    for k in prefer_keys:
        if k in keys:
            a = np.asarray(npz_obj[k])
            if a.dtype != object and a.ndim >= 1 and a.size > 0:
                return k
    for k in keys:
        a = np.asarray(npz_obj[k])
        if a.dtype != object and a.ndim == 1 and a.size > 0:
            return k
    raise ValueError(f"No usable 1D proxy trace found in NPZ. Keys={keys}")


def next_pow2(n: int) -> int:
    return 1 if n <= 1 else 2 ** int(np.ceil(np.log2(n)))


def interp_to_dt(t_src, y_src, dt_target):
    """
    Resample y_src(t_src) onto uniform grid with dt_target and same duration.
    """
    t_src = np.asarray(t_src, dtype=float).ravel()
    y_src = np.asarray(y_src, dtype=float).ravel()
    n_target = int(np.round((t_src[-1] - t_src[0]) / dt_target)) + 1
    t0 = t_src[0]
    t_target = t0 + np.arange(n_target) * dt_target
    y_target = np.interp(t_target, t_src, y_src, left=0.0, right=0.0)
    return t_target, y_target


def deconvolve_fft(x, h, method="tikhonov", lam=1e-6, waterlevel=1e-6, nfft=None):
    x = np.asarray(x, dtype=float)
    h = np.asarray(h, dtype=float)

    if nfft is None:
        nfft = next_pow2(max(len(x), len(h)))

    X = np.fft.rfft(x, n=nfft)
    H = np.fft.rfft(h, n=nfft)

    H_conj = np.conj(H)
    H2 = (H * H_conj).real  # |H|^2

    if method == "tikhonov":
        denom = H2 + (lam ** 2)
    elif method == "waterlevel":
        # floor as fraction of max |H|^2
        maxH2 = float(np.max(H2)) if H2.size else 0.0
        floor = (waterlevel ** 2) * maxH2 if maxH2 > 0 else (waterlevel ** 2)
        denom = np.maximum(H2, floor)
    else:
        raise ValueError("method must be 'tikhonov' or 'waterlevel'")

    S = X * H_conj / denom
    s = np.fft.irfft(S, n=nfft)

    return s[: len(x)], nfft


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_atf", help="Input trace (.atf) to deconvolve")
    ap.add_argument("proxy_npz", help="Proxy impulse response (.npz), e.g. stacked_A_tapered.npz")

    ap.add_argument("--proxy_key", default=None,
                    help="Key in proxy NPZ to use (overrides auto-pick)")
    ap.add_argument("--method", choices=["tikhonov", "waterlevel"], default="tikhonov",
                    help="Regularisation method")
    ap.add_argument("--lam", type=float, default=1e-6,
                    help="Tikhonov lambda (used if --method tikhonov)")
    ap.add_argument("--waterlevel", type=float, default=1e-6,
                    help="Waterlevel fraction (used if --method waterlevel)")

    ap.add_argument("--detrend", action="store_true",
                    help="Remove mean from input and proxy before FFT")
    ap.add_argument("--nfft", type=int, default=None,
                    help="FFT length (default: next pow2 of max(len(x),len(h)))")

    ap.add_argument("--proxy_resample", choices=["auto", "off"], default="auto",
                    help="Resample proxy to match input dt if proxy has its own time base (default auto).")

    ap.add_argument("--atf_amp_to_volts", type=float, default=1.0,
                    help="AmpToVolts written in output ATF (default 1.0 => values written in volts).")

    args = ap.parse_args()

    # --- load input ---
    t_in, x_in, dt_in = read_atf(args.input_atf)

    # --- load proxy ---
    P = np.load(args.proxy_npz, allow_pickle=True)
    prefer = ["stack_tapered", "stack_original", "stack", "mean", "median", "trimmed", "x", "y", "data", "signal", "trace"]
    if args.proxy_key:
        if args.proxy_key not in P.files:
            raise SystemExit(f"proxy_key '{args.proxy_key}' not in {args.proxy_npz}. Keys={P.files}")
        k_proxy = args.proxy_key
    else:
        k_proxy = choose_proxy_key(P, prefer)

    h = np.asarray(P[k_proxy]).ravel()

    # proxy time base if present
    t_h = None
    dt_h = None
    if "t" in P.files:
        t_h = np.asarray(P["t"]).ravel()
        if t_h.size >= 2:
            dt_h = float(np.median(np.diff(t_h)))
    elif "dt" in P.files:
        dt_h = float(P["dt"])
        t_h = np.arange(h.size) * dt_h

    x = x_in.copy()
    h_used = h.copy()

    if args.detrend:
        x = x - np.mean(x)
        h_used = h_used - np.mean(h_used)

    # optional proxy resample to match input dt
    if args.proxy_resample == "auto" and dt_h is not None and not np.isclose(dt_h, dt_in, rtol=1e-3, atol=0):
        # resample proxy onto input dt (preserve proxy duration)
        _, h_used = interp_to_dt(t_h, h_used, dt_in)

    # deconvolve
    x_deconv, nfft_used = deconvolve_fft(
        x=x,
        h=h_used,
        method=args.method,
        lam=args.lam,
        waterlevel=args.waterlevel,
        nfft=args.nfft
    )

    # output names
    base, _ = os.path.splitext(args.input_atf)
    out_atf = base + "_deconvolved.atf"
    out_npz = base + "_deconvolved_info.npz"

    # write deconvolved trace alone as ATF
    write_atf(out_atf, x_volts=x_deconv, dt_s=dt_in, amp_to_volts=float(args.atf_amp_to_volts))

    # write info NPZ
    info = {
        "t": t_in,
        "dt": dt_in,
        "x_in": x_in,
        "x_in_detrended": x if args.detrend else x_in,
        "h_proxy_original": h,
        "h_proxy_used": h_used,
        "proxy_key_used": np.array([k_proxy]),
        "x_deconv": x_deconv,
        "method": np.array([args.method]),
        "lam": float(args.lam),
        "waterlevel": float(args.waterlevel),
        "nfft": int(nfft_used),
        "input_atf": np.array([os.path.abspath(args.input_atf)]),
        "proxy_npz": np.array([os.path.abspath(args.proxy_npz)]),
        "output_atf": np.array([os.path.abspath(out_atf)]),
    }
    np.savez(out_npz, **info)

    print(f"Saved deconvolved trace: {out_atf}")
    print(f"Saved info:             {out_npz}")
    print(f"Proxy key used:         {k_proxy}")


if __name__ == "__main__":
    main()
