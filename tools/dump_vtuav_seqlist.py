#!/usr/bin/env python3
"""Dump the VTUAV sequence list into ``lib/train/data_specs/vtuav_<split>.txt``.

The training loader (:mod:`lib.train.dataset.vtuav`) falls back to a directory
listing when no list file exists, so this helper is optional: it pins the exact
sequence order so a run can be reproduced later.

Usage::

    python tools/dump_vtuav_seqlist.py --root /root/RGBTData/VTUAV
    python tools/dump_vtuav_seqlist.py --root /root/RGBTData/VTUAV --splits train val_st
"""
from __future__ import annotations

import argparse
import os
import sys

PRJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PRJ_ROOT not in sys.path:
    sys.path.insert(0, PRJ_ROOT)

from lib.train.dataset.vtuav import SPLIT_DIRS  # noqa: E402

DEFAULT_SPECS_DIR = os.path.join(PRJ_ROOT, "lib", "train", "data_specs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("dump VTUAV sequence list")
    parser.add_argument("--root", required=True,
                        help="VTUAV root containing train/ test_ST/ test_LT/")
    parser.add_argument("--splits", nargs="+", default=["train", "val_st", "val_lt"],
                        choices=sorted(SPLIT_DIRS))
    parser.add_argument("--out_dir", default=DEFAULT_SPECS_DIR,
                        help="destination directory (default: lib/train/data_specs)")
    parser.add_argument("--dry_run", action="store_true", help="only report, do not write")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = os.path.abspath(os.path.expanduser(args.root))
    if not os.path.isdir(root):
        print(f"[ERROR] VTUAV root not found: {root}", file=sys.stderr)
        return 2

    os.makedirs(args.out_dir, exist_ok=True)
    failed = False
    for split in args.splits:
        split_dir = os.path.join(root, SPLIT_DIRS[split])
        if not os.path.isdir(split_dir):
            print(f"[WARN] missing split directory, skipped: {split_dir}")
            failed = True
            continue

        names = sorted(name for name in os.listdir(split_dir)
                       if os.path.isdir(os.path.join(split_dir, name)))
        out_path = os.path.join(args.out_dir, f"vtuav_{split}.txt")
        print(f"{split:<7} {len(names):>4} sequences  {split_dir}  ->  {out_path}")
        if args.dry_run:
            continue
        with open(out_path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(names) + "\n")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
