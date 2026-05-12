#!/usr/bin/env python3
"""
Estimate a frequency-dependent calibration from ch03 to ch02.

The calibration is computed as the transfer-function amplitude

    C(f) = |S23(f) / S33(f)|

where S23 = X_ch02 * conj(X_ch03) and S33 = X_ch03 * conj(X_ch03).
For multiple unstacked events, cross spectra are averaged before the ratio.
Band values are measured from coherent bins when possible and interpolated in
log-frequency/log-amplitude space where coherence is poor.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import re
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


def read_atf(path: Path) -> tuple[np.ndarray, float]:
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
                data.append(float(s.split()[0]))

    return np.asarray(data, dtype=np.float64) * amp_to_volts, dt


def tukey_window(n: int, alpha: float) -> np.ndarray:
    if n <= 1 or alpha <= 0:
        return np.ones(n)
    if alpha >= 1:
        return np.hanning(n)
    w = np.ones(n)
    edge = int(alpha * (n - 1) / 2)
    if edge <= 0:
        return w
    i = np.arange(edge)
    taper = 0.5 * (1 + np.cos(np.pi * (2 * i / (alpha * (n - 1)) - 1)))
    w[:edge] = taper
    w[-edge:] = taper[::-1]
    return w


def next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


def parse_bands(s: str) -> list[tuple[float, float]]:
    bands = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        a, b = part.split("-", 1)
        f0 = float(a)
        f1 = float(b)
        if f0 <= 0 or f1 <= f0:
            raise ValueError(f"Invalid frequency band: {part}")
        bands.append((f0, f1))
    if not bands:
        raise ValueError("No valid frequency bands supplied")
    return bands


def moving_average(x: np.ndarray, n: int) -> np.ndarray:
    if n <= 1:
        return x
    kernel = np.ones(n, dtype=np.float64) / float(n)
    if np.iscomplexobj(x):
        return np.convolve(x.real, kernel, mode="same") + 1j * np.convolve(
            x.imag, kernel, mode="same"
        )
    return np.convolve(x, kernel, mode="same")


def crop_and_baseline(
    x: np.ndarray,
    dt: float,
    baseline_start: float,
    baseline_end: float,
    tmin: float | None,
    tmax: float | None,
    baseline_stat: str,
) -> np.ndarray:
    t = np.arange(x.size, dtype=np.float64) * dt
    bmask = (t >= baseline_start) & (t <= baseline_end)
    if not np.any(bmask):
        raise ValueError(f"No baseline samples in {baseline_start:g}-{baseline_end:g} s")
    if baseline_stat == "median":
        baseline = float(np.median(x[bmask]))
    else:
        baseline = float(np.mean(x[bmask]))
    y = x - baseline

    if tmin is None:
        tmin = t[0]
    if tmax is None:
        tmax = t[-1]
    wmask = (t >= tmin) & (t <= tmax)
    if not np.any(wmask):
        raise ValueError(f"No samples in spectral window {tmin:g}-{tmax:g} s")
    return y[wmask]


def pair_paths_from_globs(globs: list[str]) -> list[tuple[Path, Path]]:
    if len(globs) == 1:
        return pair_paths_from_inferred_glob(globs[0])
    if len(globs) != 2:
        raise ValueError("--pairs-glob expects one inferred glob or two explicit globs")

    file1_paths = sorted(Path(p) for p in glob.glob(globs[0]))
    file2_paths = sorted(Path(p) for p in glob.glob(globs[1]))
    if len(file1_paths) != len(file2_paths):
        raise ValueError(
            f"--pairs-glob matched {len(file1_paths)} file1 path(s) and "
            f"{len(file2_paths)} file2 path(s)"
        )
    return list(zip(file1_paths, file2_paths))


def pair_paths_from_inferred_glob(ch02_glob: str) -> list[tuple[Path, Path]]:
    pairs = []
    for ch02 in sorted(Path(p) for p in glob.glob(ch02_glob)):
        name = ch02.name
        if "_ch02.atf" not in name:
            continue
        ch03 = ch02.with_name(name.replace("_ch02.atf", "_ch03.atf"))
        if ch03.exists():
            pairs.append((ch02, ch03))
        else:
            print(f"Warning: missing ch03 partner for {ch02.name}")
    return pairs


def spectra_for_pair(
    ch02_path: Path,
    ch03_path: Path,
    args: argparse.Namespace,
    nfft: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    x2, dt2 = read_atf(ch02_path)
    x3, dt3 = read_atf(ch03_path)
    if not np.isclose(dt2, dt3, rtol=0, atol=max(dt2, dt3) * 1e-9):
        raise ValueError(f"dt mismatch: {ch02_path.name} dt={dt2}, {ch03_path.name} dt={dt3}")

    y2 = crop_and_baseline(
        x2,
        dt2,
        args.baseline_start_sec,
        args.baseline_end_sec,
        args.tmin,
        args.tmax,
        args.baseline_stat,
    )
    y3 = crop_and_baseline(
        x3,
        dt3,
        args.baseline_start_sec,
        args.baseline_end_sec,
        args.tmin,
        args.tmax,
        args.baseline_stat,
    )
    y2 = y2 / args.amp_file1
    y3 = y3 / args.amp_file2

    n = min(y2.size, y3.size)
    y2 = y2[:n]
    y3 = y3[:n]
    w = tukey_window(n, args.tukey_alpha)
    y2 = y2 * w
    y3 = y3 * w

    if nfft is None:
        nfft_used = next_pow2(n)
    else:
        nfft_used = nfft
    x2f = np.fft.rfft(y2, n=nfft_used)
    x3f = np.fft.rfft(y3, n=nfft_used)
    freqs = np.fft.rfftfreq(nfft_used, d=dt2)

    s23 = x2f * np.conj(x3f)
    s22 = np.abs(x2f) ** 2
    s33 = np.abs(x3f) ** 2
    return freqs, s23, s22, s33, n


def interpolate_log_curve(freqs: np.ndarray, values: np.ndarray, good: np.ndarray) -> np.ndarray:
    out = np.full_like(values, np.nan, dtype=np.float64)
    valid = good & np.isfinite(values) & (values > 0) & (freqs > 0)
    if np.count_nonzero(valid) < 2:
        out[valid] = values[valid]
        return out
    logf = np.log10(freqs[valid])
    logv = np.log10(values[valid])
    target = (freqs > 0) & np.isfinite(values)
    out[target] = 10 ** np.interp(np.log10(freqs[target]), logf, logv)
    return out


def band_rows(
    freqs: np.ndarray,
    cal: np.ndarray,
    cal_interp: np.ndarray,
    coherence: np.ndarray,
    power_ok: np.ndarray,
    bands: Iterable[tuple[float, float]],
    coherence_min: float,
    min_good_bins: int,
) -> list[dict[str, object]]:
    rows = []
    measured_centers = []
    measured_values = []

    for f0, f1 in bands:
        band = (freqs >= f0) & (freqs < f1) & np.isfinite(cal) & (cal > 0)
        good = band & power_ok & (coherence >= coherence_min)
        if np.count_nonzero(good) >= min_good_bins:
            value = float(10 ** np.median(np.log10(cal[good])))
            status = "measured"
            measured_centers.append(math.sqrt(f0 * f1))
            measured_values.append(value)
        else:
            interp_good = band & np.isfinite(cal_interp) & (cal_interp > 0)
            value = float(10 ** np.median(np.log10(cal_interp[interp_good]))) if np.any(interp_good) else float("nan")
            status = "interpolated" if np.isfinite(value) else "unreliable"
        coh_med = float(np.median(coherence[band])) if np.any(band) else float("nan")
        rows.append(
            {
                "f0_hz": f0,
                "f1_hz": f1,
                "center_hz": math.sqrt(f0 * f1),
                "calibration_ch02_per_ch03": value,
                "coherence_median": coh_med,
                "good_bins": int(np.count_nonzero(good)),
                "total_bins": int(np.count_nonzero(band)),
                "status": status,
            }
        )
    return rows


def write_curve_csv(path: Path, freqs, cal, cal_interp, coherence, power_ok) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frequency_hz", "calibration_raw", "calibration_interp", "coherence", "power_ok"])
        for row in zip(freqs, cal, cal_interp, coherence, power_ok):
            w.writerow(row)


def write_band_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def plot_results(path: Path, freqs, cal, cal_interp, coherence, rows, args) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    positive = (freqs > 0) & np.isfinite(cal) & (cal > 0)
    ax1.loglog(freqs[positive], cal[positive], color="0.65", lw=0.8, label="Raw |H(f)|")
    good_interp = (freqs > 0) & np.isfinite(cal_interp) & (cal_interp > 0)
    ax1.loglog(freqs[good_interp], cal_interp[good_interp], color="C0", lw=1.5, label="Interpolated/smoothed")
    for row in rows:
        f0 = float(row["f0_hz"])
        f1 = float(row["f1_hz"])
        value = float(row["calibration_ch02_per_ch03"])
        if np.isfinite(value) and value > 0:
            color = "C2" if row["status"] == "measured" else "C1"
            ax1.hlines(value, f0, f1, colors=color, lw=3)
    ax1.set_ylabel("Calibration ch02 / ch03")
    ax1.grid(True, which="both", alpha=0.25)
    ax1.legend(loc="best", fontsize=8)

    valid_coh = freqs > 0
    ax2.semilogx(freqs[valid_coh], coherence[valid_coh], color="C3", lw=1.0)
    ax2.axhline(args.coherence_min, color="0.3", ls="--", lw=0.9)
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Coherence")
    ax2.grid(True, which="both", alpha=0.25)

    if args.freq_min is not None or args.freq_max is not None:
        ax2.set_xlim(args.freq_min or np.min(freqs[valid_coh]), args.freq_max or np.max(freqs))
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def convert_arg_line_to_args(line: str) -> list[str]:
    """
    Allow argument files with comments.

    Example input file:
        --stacked stacked_ch2.atf stacked_ch3.atf
        --tmin 4e-5
        # omit --tmax to use the end of each data file
        --baseline-start-sec 6e-6
        --baseline-end-sec 40e-6
        --amp-file1 31.6
        --amp-file2 2.0
        --out-prefix stacked_spectral_cal

    Run with:
        python spectral_calibrate_ch3.py @spectral_input.txt
    """
    line = line.split("#", 1)[0].strip()
    if not line:
        return []
    return line.split()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Frequency-band calibration of ch03 using ch02 as reference.",
        fromfile_prefix_chars="@",
    )
    ap.convert_arg_line_to_args = convert_arg_line_to_args
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--stacked", nargs=2, metavar=("CH02_ATF", "CH03_ATF"))
    mode.add_argument(
        "--pairs-glob",
        nargs="+",
        default=None,
        metavar="GLOB",
        help=(
            'Glob(s) for unstacked pairs. With one glob, e.g. "survey*_ch02.atf", '
            'ch03 partners are inferred. With two globs, matched files are paired '
            'in sorted order, e.g. "survey*_ch01.atf" "survey*ch04.atf".'
        ),
    )
    ap.add_argument(
        "--baseline-start-sec",
        "--baseline-start",
        dest="baseline_start_sec",
        type=float,
        default=6e-6,
        help="Baseline window start time in seconds. Default: 6e-6 s.",
    )
    ap.add_argument(
        "--baseline-end-sec",
        "--baseline-end",
        dest="baseline_end_sec",
        type=float,
        default=40e-6,
        help="Baseline window end time in seconds. Default: 40e-6 s.",
    )
    ap.add_argument("--baseline-stat", choices=["mean", "median"], default="mean")
    ap.add_argument(
        "--amp-file1",
        type=float,
        default=31.6,
        help="Amplitude divisor applied to file1/ch02 after baseline removal. Default: 31.6.",
    )
    ap.add_argument(
        "--amp-file2",
        type=float,
        default=2.0,
        help="Amplitude divisor applied to file2/ch03 after baseline removal. Default: 2.0.",
    )
    ap.add_argument("--tmin", type=float, default=40e-6, help="Start of spectral signal window")
    ap.add_argument(
        "--tmax",
        type=float,
        default=None,
        help="End of spectral signal window. Default: end of each data file.",
    )
    ap.add_argument("--tukey-alpha", type=float, default=0.10)
    ap.add_argument("--nfft", type=int, default=None)
    ap.add_argument("--smooth-hz", type=float, default=5000.0)
    ap.add_argument("--waterlevel", type=float, default=1e-5, help="Relative floor on S33")
    ap.add_argument("--coherence-min", type=float, default=0.70)
    ap.add_argument("--min-good-bins", type=int, default=5)
    ap.add_argument(
        "--bands",
        default="20000-40000,40000-80000,80000-150000,150000-300000,300000-600000",
    )
    ap.add_argument("--freq-min", type=float, default=1e4)
    ap.add_argument("--freq-max", type=float, default=8e5)
    ap.add_argument("--out-prefix", default="spectral_cal_ch03_from_ch02")
    args = ap.parse_args()

    if not np.isfinite(args.amp_file1) or args.amp_file1 <= 0:
        ap.error("--amp-file1 must be a finite positive value")
    if not np.isfinite(args.amp_file2) or args.amp_file2 <= 0:
        ap.error("--amp-file2 must be a finite positive value")

    bands = parse_bands(args.bands)
    if args.stacked:
        pairs = [(Path(args.stacked[0]), Path(args.stacked[1]))]
    else:
        try:
            pairs = pair_paths_from_globs(args.pairs_glob)
        except ValueError as exc:
            ap.error(str(exc))
    if not pairs:
        raise SystemExit("No ATF pairs found")

    s23_sum = None
    s22_sum = None
    s33_sum = None
    freqs_ref = None
    used = 0
    n_samples = []

    for ch02, ch03 in pairs:
        try:
            freqs, s23, s22, s33, n = spectra_for_pair(ch02, ch03, args, args.nfft)
        except Exception as exc:
            print(f"Warning: skipping {ch02.name}/{ch03.name}: {exc}")
            continue
        if freqs_ref is None:
            freqs_ref = freqs
            s23_sum = np.zeros_like(s23, dtype=np.complex128)
            s22_sum = np.zeros_like(s22, dtype=np.float64)
            s33_sum = np.zeros_like(s33, dtype=np.float64)
        elif freqs.size != freqs_ref.size or not np.allclose(freqs, freqs_ref):
            print(f"Warning: skipping {ch02.name}/{ch03.name}: frequency grid mismatch")
            continue
        s23_sum += s23
        s22_sum += s22
        s33_sum += s33
        used += 1
        n_samples.append(n)

    if used == 0:
        raise SystemExit("No usable pairs")

    freqs = freqs_ref
    s23 = s23_sum / used
    s22 = s22_sum / used
    s33 = s33_sum / used

    df = float(freqs[1] - freqs[0]) if freqs.size > 1 else 0.0
    smooth_bins = max(1, int(math.ceil(args.smooth_hz / df))) if df > 0 else 1
    if smooth_bins % 2 == 0:
        smooth_bins += 1

    s23s = moving_average(s23, smooth_bins)
    s22s = moving_average(s22, smooth_bins)
    s33s = moving_average(s33, smooth_bins)

    s33_floor = args.waterlevel * float(np.nanmax(np.real(s33s)))
    denom = np.maximum(np.real(s33s), s33_floor)
    h = s23s / denom
    cal = np.abs(h)
    coherence = np.abs(s23s) ** 2 / np.maximum(np.real(s22s) * np.real(s33s), np.finfo(float).tiny)
    coherence = np.clip(coherence, 0.0, 1.0)
    power_ok = np.real(s33s) >= s33_floor

    if args.freq_min is not None:
        power_ok &= freqs >= args.freq_min
    if args.freq_max is not None:
        power_ok &= freqs <= args.freq_max
    reliable = power_ok & (coherence >= args.coherence_min)
    cal_interp = interpolate_log_curve(freqs, cal, reliable)

    rows = band_rows(
        freqs,
        cal,
        cal_interp,
        coherence,
        power_ok,
        bands,
        args.coherence_min,
        args.min_good_bins,
    )

    prefix = Path(args.out_prefix)
    curve_csv = prefix.with_suffix(".curve.csv")
    band_csv = prefix.with_suffix(".bands.csv")
    png = prefix.with_suffix(".png")
    write_curve_csv(curve_csv, freqs, cal, cal_interp, coherence, power_ok)
    write_band_csv(band_csv, rows)
    plot_results(png, freqs, cal, cal_interp, coherence, rows, args)

    print(f"Used {used} pair(s); spectral samples per pair: {min(n_samples)}-{max(n_samples)}")
    print(f"df={df:.6g} Hz; smoothing={smooth_bins} bin(s) ~= {smooth_bins * df:.6g} Hz")
    print(f"Wrote: {curve_csv}")
    print(f"Wrote: {band_csv}")
    print(f"Wrote: {png}")
    print("\nBand calibration ch02/ch03:")
    for row in rows:
        value = row["calibration_ch02_per_ch03"]
        value_s = f"{value:.6e}" if np.isfinite(value) else "nan"
        print(
            f"  {row['f0_hz']:9.0f}-{row['f1_hz']:<9.0f} Hz  "
            f"{value_s}  coh={row['coherence_median']:.3f}  "
            f"good={row['good_bins']}/{row['total_bins']}  {row['status']}"
        )


if __name__ == "__main__":
    main()
