#!/usr/bin/env python3
"""Extract VTUAV test ground-truth txt files while preserving directories."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


SUBSET_ALIASES = {
    "test_st",
    "test_lt",
    "short-term",
    "long-term",
}
GT_NAMES = {"rgb.txt", "ir.txt"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy rgb.txt and ir.txt from VTUAV test_st/test_lt while preserving "
            "their paths relative to the VTUAV root."
        )
    )
    parser.add_argument(
        "source",
        type=Path,
        help="VTUAV root containing test_st/test_lt (or short-term/long-term)",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="new folder in which the extracted directory tree is created",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing destination GT file when its content differs",
    )
    return parser.parse_args()


def find_subsets(source: Path) -> list[Path]:
    aliases = {name.casefold() for name in SUBSET_ALIASES}
    return sorted(
        (path for path in source.iterdir() if path.is_dir() and path.name.casefold() in aliases),
        key=lambda path: path.name.casefold(),
    )


def same_file_contents(left: Path, right: Path) -> bool:
    if left.stat().st_size != right.stat().st_size:
        return False
    with left.open("rb") as src, right.open("rb") as dst:
        while True:
            src_chunk = src.read(1024 * 1024)
            dst_chunk = dst.read(1024 * 1024)
            if src_chunk != dst_chunk:
                return False
            if not src_chunk:
                return True


def main() -> int:
    args = parse_args()
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()

    if not source.is_dir():
        print(f"[ERROR] VTUAV source directory does not exist: {source}", file=sys.stderr)
        return 2
    if source == output or source in output.parents:
        print("[ERROR] output must not be inside the VTUAV source directory", file=sys.stderr)
        return 2

    subsets = find_subsets(source)
    if not subsets:
        expected = ", ".join(sorted(SUBSET_ALIASES))
        print(f"[ERROR] no test subset found under {source}; expected one of: {expected}", file=sys.stderr)
        return 2

    gt_files: list[Path] = []
    for subset in subsets:
        gt_files.extend(
            path
            for path in subset.rglob("*.txt")
            if path.is_file() and path.name.casefold() in GT_NAMES
        )
    gt_files.sort(key=lambda path: str(path.relative_to(source)).casefold())

    if not gt_files:
        print(f"[ERROR] no rgb.txt or ir.txt found in: {', '.join(map(str, subsets))}", file=sys.stderr)
        return 2

    copied = skipped = 0
    output.mkdir(parents=True, exist_ok=True)
    for src in gt_files:
        relative = src.relative_to(source)
        dst = output / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            if same_file_contents(src, dst):
                skipped += 1
                continue
            if not args.overwrite:
                print(
                    f"[ERROR] destination differs and already exists: {dst}\n"
                    "        Re-run with --overwrite to replace it.",
                    file=sys.stderr,
                )
                return 3
        shutil.copy2(src, dst)
        copied += 1

    print("=" * 72)
    print("VTUAV ground-truth extraction completed")
    print(f"Source       : {source}")
    print(f"Output       : {output}")
    print(f"Subsets      : {', '.join(path.name for path in subsets)}")
    print(f"GT files     : {len(gt_files)}")
    print(f"Copied       : {copied}")
    print(f"Already same : {skipped}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
