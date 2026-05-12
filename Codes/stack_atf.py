#!/usr/bin/env python3
"""
stack_atf.py

Stack (average) ATF waveforms (3-line header + one sample per line).
The waveforms are each in one .atf file
The file names are read from a list in an input file (plain text, e.g. fileanmes.txt)

Usage:
  python stack_atf.py filenames.txt
  python stack_atf.py PATH/filenames.txt

The output "stack.atf" is saved in the same folder as filenames.txt.

S Nielsen March 2026
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np

from datetime import datetime

def parse_dt_from_header_line2(line2: str) -> datetime | None:
    """
    Parse Date=DD-MM-YYYY; Time=HH:MM:SS.fffffff; from the ATF metadata line.
    Returns datetime or None if not parseable.
    """
    m_date = re.search(r"Date=(\d{2}-\d{2}-\d{4})\s*;", line2)
    m_time = re.search(r"Time=([0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?)\s*;", line2)
    if not (m_date and m_time):
        return None

    dstr = m_date.group(1)
    tstr = m_time.group(1)

    # Convert ATF's 7-digit fractional seconds to Python microseconds (6 digits) by truncation
    if "." in tstr:
        hms, frac = tstr.split(".", 1)
        frac6 = (frac + "000000")[:6]
        tstr_py = f"{hms}.{frac6}"
        fmt = "%H:%M:%S.%f"
    else:
        tstr_py = tstr
        fmt = "%H:%M:%S"

    try:
        d = datetime.strptime(dstr, "%d-%m-%Y").date()
        t = datetime.strptime(tstr_py, fmt).time()
        return datetime.combine(d, t)
    except Exception:
        return None


def set_or_append_kv(line2: str, key: str, value: str) -> str:
    """
    Set key=value in the semicolon-separated header line.
    If key exists, replace its value. Otherwise append '; key=value;'
    """
    # Replace existing
    pat = re.compile(rf"({re.escape(key)}=)[^;]*")
    if pat.search(line2):
        return pat.sub(rf"\g<1>{value}", line2)

    # Append new field (ensure it ends with a semicolon)
    s = line2.rstrip()
    if not s.endswith(";"):
        s += ";"
    return s + f" {key}={value};"



def read_list_file(list_path: Path) -> List[Path]:
    """
    Read file paths from list_path.
    - Ignores blank lines and lines starting with '#'
    - Resolves relative paths relative to list_path.parent
    """
    base = list_path.parent.resolve()
    files: List[Path] = []
    for line in list_path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        p = Path(s)
        if not p.is_absolute():
            p = (base / p)
        files.append(p.expanduser().resolve())
    return files


def read_atf_3line(path: Path) -> Tuple[List[str], np.ndarray]:
    """
    Read an ATF file assumed to be:
      line 1: "ATF v1.xx"
      line 2: metadata
      line 3: "[TraceData]"
      subsequent lines: numeric samples (float or int)

    Returns (header_lines[3], data_array float64)
    """
    lines = path.read_text().splitlines()
    if len(lines) < 4:
        raise ValueError(f"{path} is too short to be a valid 3-line-header ATF.")

    header = lines[:3]
    if not header[0].strip().lower().startswith("atf"):
        raise ValueError(f"{path}: first line does not look like an ATF header.")
    if header[2].strip() != "[TraceData]":
        raise ValueError(f"{path}: expected '[TraceData]' on line 3, got: {header[2]!r}")

    # Parse numeric samples
    try:
        data = np.array([float(x.strip()) for x in lines[3:] if x.strip() != ""], dtype=np.float64)
    except Exception as e:
        raise ValueError(f"{path}: failed to parse numeric data after header: {e}")

    return header, data


def update_tracepoints_line2(line2: str, npts: int) -> str:
    """
    Update 'TracePoints=...' in the second header line if present.
    Leaves everything else unchanged.
    """
    # Replace TracePoints=#### with TracePoints=<npts>
    return re.sub(r"(TracePoints=)\d+", rf"\g<1>{npts}", line2)


def write_atf_3line(out_path: Path, header3: List[str], data: np.ndarray, stack_number: int) -> None:
    """
    Write 3-line header ATF + samples in scientific notation, and add StackNumber.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if len(header3) != 3:
        raise ValueError("header3 must contain exactly 3 lines.")

    header3 = header3.copy()
    header3[1] = update_tracepoints_line2(header3[1], int(data.size))
    header3[1] = set_or_append_kv(header3[1], "StackNumber", str(stack_number))

    with open(out_path, "w") as f:
        f.write(header3[0].rstrip() + "\n")
        f.write(header3[1].rstrip() + "\n")
        f.write(header3[2].rstrip() + "\n")
        for v in data:
            f.write(f"{v:.12e}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Stack (average) ATF waveforms listed in a text file.")
    ap.add_argument("listfile", type=str, help="Text file containing list of .atf input files")
    ap.add_argument("--out", type=str, default="stack.atf", help="Output filename (default: stack.atf)")
    args = ap.parse_args()

    list_path = Path(args.listfile).expanduser().resolve()
    if not list_path.exists():
        raise SystemExit(f"List file not found: {list_path}")

    files = read_list_file(list_path)
    if not files:
        raise SystemExit(f"No input files found in list: {list_path}")

    # Output saved in same folder as list file
    out_path = list_path.parent / args.out

    ######################
    best_header = None
    best_dt = None

    sum_data = None
    n_used = 0
    npts_ref = None

    for fp in files:
        if not fp.exists():
            raise SystemExit(f"Missing input file: {fp}")

        header, data = read_atf_3line(fp)
        this_dt = parse_dt_from_header_line2(header[1])

        # Keep earliest header (if timestamps parse); otherwise keep first seen
        if best_header is None:
            best_header = header
            best_dt = this_dt
        else:
            if this_dt is not None and (best_dt is None or this_dt < best_dt):
                best_header = header
                best_dt = this_dt

        if sum_data is None:
            npts_ref = data.size
            sum_data = np.zeros_like(data, dtype=np.float64)
        else:
            if data.size != npts_ref:
                raise SystemExit(
                    f"Length mismatch:\n"
                    f"  reference npts={npts_ref} (from first file)\n"
                    f"  {fp} has npts={data.size}\n"
                    f"All files must have the same number of samples."
                )

        sum_data += data
        n_used += 1

    avg_data = sum_data / float(n_used)

    write_atf_3line(out_path, best_header, avg_data, stack_number=n_used)
    print(f"Done. Stacked {n_used} file(s) -> {out_path}")
    #####################


if __name__ == "__main__":
    main()

