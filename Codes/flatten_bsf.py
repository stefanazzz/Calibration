#!/usr/bin/env python3
"""
Flatten InSite ASC BSF folder structure to a single directory with timestamped names.

Example source path pattern:
  .../2026/02/17/14/42/04_5650.bsf
Becomes:
  survey20260217144204_5650.bsf

Usage examples:
  python flatten_bsf.py /data/2026 /data/flat --prefix survey
  python flatten_bsf.py /data/2026/02/17/14/42 /data/flat --prefix survey
Dry run first:
  python flatten_bsf.py /data/2026/02/17 /data/flat --dry-run
Move instead of copy:
  python flatten_bsf.py /data/2026 /data/flat --move

S. Nielsen February 2026
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path
from typing import Optional, Tuple


BSF_SUFFIX = ".bsf"

# Match ".../<YYYY>/<MM>/<DD>/<HH>/<mm>/<SS_FFFF>.bsf" anywhere within a path
# We keep it flexible: it can be nested under other directories too.
PATH_TIME_RE = re.compile(
    r"(?P<year>\d{4})[\\/]"
    r"(?P<month>\d{2})[\\/]"
    r"(?P<day>\d{2})[\\/]"
    r"(?P<hour>\d{2})[\\/]"
    r"(?P<minute>\d{2})[\\/]"
    r"(?P<secfrac>\d{2}_\d+)"
    r"\.bsf$",
    re.IGNORECASE,
)


def parse_from_path(p: Path) -> Optional[Tuple[str, str, str, str, str, str]]:
    """
    Return (YYYY, MM, DD, HH, mm, SS_FFFF) if the folder structure is present in the path.
    Otherwise return None.
    """
    s = str(p.as_posix())
    m = PATH_TIME_RE.search(s)
    if not m:
        return None
    return (
        m.group("year"),
        m.group("month"),
        m.group("day"),
        m.group("hour"),
        m.group("minute"),
        m.group("secfrac"),
    )


def safe_copy_or_move(src: Path, dst: Path, move: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if move:
        shutil.move(str(src), str(dst))
    else:
        shutil.copy2(str(src), str(dst))


def uniquify(path: Path) -> Path:
    """
    If path exists, append _0001, _0002, ... before suffix.
    """
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    for i in range(1, 100000):
        candidate = parent / f"{stem}_{i:04d}{suffix}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Too many name collisions for {path.name}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Flatten InSite ASC .bsf files into a single folder with timestamped filenames."
    )
    ap.add_argument("input_root", type=Path, help="Input directory (can be any depth: year/ or year/month/day/...)")
    ap.add_argument("output_dir", type=Path, help="Output directory to copy/move flattened files into")
    ap.add_argument("--prefix", default="survey", help="Filename prefix (default: survey)")
    ap.add_argument("--move", action="store_true", help="Move files instead of copying")
    ap.add_argument("--dry-run", action="store_true", help="Print actions but do not copy/move")
    ap.add_argument("--recursive", action="store_true", help="Force recursive search (default is recursive anyway)")
    ap.add_argument("--skip-unparsed", action="store_true",
                    help="Skip files that don't match the YYYY/MM/DD/HH/mm/SS_FFFF.bsf pattern")
    args = ap.parse_args()

    in_root = args.input_root.expanduser().resolve()
    out_dir = args.output_dir.expanduser().resolve()

    if not in_root.exists() or not in_root.is_dir():
        raise SystemExit(f"Input root does not exist or is not a directory: {in_root}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # rglob is recursive; keep --recursive for CLI clarity but not required
    bsf_files = sorted(in_root.rglob(f"*{BSF_SUFFIX}"))

    if not bsf_files:
        print(f"No {BSF_SUFFIX} files found under: {in_root}")
        return

    n_ok = 0
    n_skip = 0

    for src in bsf_files:
        parsed = parse_from_path(src)
        if not parsed:
            if args.skip_unparsed:
                n_skip += 1
                continue
            # Fallback: just use the original name (still flattened)
            new_name = f"{args.prefix}{src.name}"
        else:
            yyyy, mm, dd, HH, Min, secfrac = parsed
            new_name = f"{args.prefix}{yyyy}{mm}{dd}{HH}{Min}{secfrac}{BSF_SUFFIX}"

        dst = uniquify(out_dir / new_name)

        if args.dry_run:
            action = "MOVE" if args.move else "COPY"
            print(f"{action}: {src}  ->  {dst}")
        else:
            safe_copy_or_move(src, dst, move=args.move)

        n_ok += 1

    print(f"Done. Processed: {n_ok} file(s). Skipped (unparsed): {n_skip}. Output: {out_dir}")


if __name__ == "__main__":
    main()

