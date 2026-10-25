#!/usr/bin/env python3
"""Check VTUAV frame <-> annotation alignment before/after a training run.

VTUAV annotates **one frame per 10 images** (``000000``, ``000010``, ...).  If a
folder happens to contain *every* video frame, addressing images by their sorted
ordinal puts the ground truth 10x away from the image it belongs to.  Training
still runs and the loss still goes down, but the IoU stalls around 0.15 and
never recovers -- which is very hard to spot from the logs alone.

This tool prints, per split:

* how many images and how many annotation rows each sequence has,
* the stride/offset that :mod:`lib.train.dataset.vtuav_alignment` infers,
* whether the alignment used at train/test time is identity or stride-mapped,
* for a few sequences, exactly which file each annotation row maps to.

Usage::

    python tools/check_vtuav_alignment.py --root /root/RGBTData/VTUAV
    python tools/check_vtuav_alignment.py --root /root/RGBTData/VTUAV --splits train
    python tools/check_vtuav_alignment.py --root /root/RGBTData/VTUAV --limit 5 --examples 3

Exit code 0 = all sequences align cleanly, 2 = at least one needs attention.
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
from typing import Dict, List, Tuple

PRJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PRJ_ROOT not in sys.path:
    sys.path.insert(0, PRJ_ROOT)

from lib.train.dataset.vtuav import SPLIT_DIRS                    # noqa: E402
from lib.train.dataset.vtuav_alignment import (                   # noqa: E402
    annotated_frame_ordinals,
    load_init_frame,
    parse_stem,
)

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def list_images(seq_dir: str, sub: str = "rgb") -> List[str]:
    frame_dir = os.path.join(seq_dir, sub)
    if not os.path.isdir(frame_dir):
        return []
    return sorted(name for name in os.listdir(frame_dir)
                  if name.lower().endswith(IMAGE_EXTS))


def count_gt_rows(seq_dir: str) -> int:
    path = os.path.join(seq_dir, "rgb.txt")
    if not os.path.isfile(path):
        path = os.path.join(seq_dir, "ir.txt")
    if not os.path.isfile(path):
        return 0
    rows = 0
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.strip():
                rows += 1
    return rows


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("check VTUAV frame/GT alignment")
    ap.add_argument("--root", required=True, help="VTUAV_HOME (contains train/ test_ST/ test_LT/)")
    ap.add_argument("--splits", nargs="*", default=["train", "val_st", "val_lt"],
                    help="split names or raw directories (train, val_st, val_lt, test_ST, test_LT)")
    ap.add_argument("--limit", type=int, default=0, help="inspect at most N sequences per split")
    ap.add_argument("--examples", type=int, default=2, help="rows to print per example sequence")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    init_frame = load_init_frame([os.path.join(PRJ_ROOT, "lib", "train", "dataset")])
    print("init_frame.npy   : %s"
          % (f"{len(init_frame)} sequence(s) from {os.environ.get('VTUAV_INIT_FRAME') or 'lib/train/dataset/'}"
             if init_frame else "not found (offset = first frame on disk)"))
    print("VTUAV root       : %s" % os.path.abspath(args.root))
    print()

    problems = 0
    for split in args.splits:
        split_dir = os.path.join(args.root, SPLIT_DIRS.get(split, split))
        if not os.path.isdir(split_dir):
            print(f"=== {split}: directory not found ({split_dir}) -> skipped\n")
            continue

        seq_names = sorted(n for n in os.listdir(split_dir)
                           if os.path.isdir(os.path.join(split_dir, n)))
        if args.limit:
            seq_names = seq_names[:args.limit]

        ratios: List[float] = []
        mapped = 0
        shorter = 0
        identity = 0
        odd_examples: List[Tuple[str, str]] = []

        print(f"=== {split}  ({len(seq_names)} sequence(s) inspected) ===")
        for name in seq_names:
            seq_dir = os.path.join(split_dir, name)
            images = list_images(seq_dir)
            n_img = len(images)
            n_gt = count_gt_rows(seq_dir)
            if n_img == 0 or n_gt == 0:
                problems += 1
                print(f"  [BAD ] {name:<28s} images={n_img:<6d} gt_rows={n_gt:<6d}  empty!")
                continue

            ordinals, info = annotated_frame_ordinals(
                images, n_gt, offset=init_frame.get(name), seq_name=name)
            ratio = n_img / float(n_gt)
            ratios.append(ratio)
            if info.get("scaled"):
                mapped += 1
            elif n_img < n_gt:
                shorter += 1
            else:
                identity += 1

            if info.get("scaled") or n_img < n_gt or info.get("missing"):
                if len(odd_examples) < 8:
                    odd_examples.append((name, describe_line(info)))

        if ratios:
            print(f"  images-per-gt-ratio : min={min(ratios):.2f} median={statistics.median(ratios):.2f} "
                  f"max={max(ratios):.2f}")
        print(f"  identity / stride-mapped / fewer-images-than-gt : "
              f"{identity} / {mapped} / {shorter}")
        if mapped:
            print("  -> stride mapping is REQUIRED for these sequences (dense frame folders)")
        for name, line in odd_examples:
            print(f"    {name:<28s} {line}")
            if args.examples <= 0:
                continue
            seq_dir = os.path.join(split_dir, name)
            images = list_images(seq_dir)
            n_gt = count_gt_rows(seq_dir)
            ordinals, _ = annotated_frame_ordinals(
                images, n_gt, offset=init_frame.get(name), seq_name=name)
            for row in range(min(args.examples, len(ordinals))):
                print(f"        gt row {row} -> {images[ordinals[row]]}")
        print()
        if shorter:
            problems += shorter

    print("RESULT: " + ("OK - every sequence aligns cleanly" if problems == 0
                        else f"ATTENTION - {problems} sequence(s) reported an issue"))
    if not problems:
        print("        (identity mapping everywhere: nothing to fix)")
    else:
        print("        run training/testing with this repo's loaders; they apply the")
        print("        stride mapping automatically. See docs/VTUAV.md section 1.4.")
    return 0 if problems == 0 else 2


def describe_line(info: Dict[str, object]) -> str:
    return ("images={n_images} gt={n_gt} stride={stride} offset={offset} "
            "missing={missing} ({reason})").format(**info)


if __name__ == "__main__":
    raise SystemExit(main())
