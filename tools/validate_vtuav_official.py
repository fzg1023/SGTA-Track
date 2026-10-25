#!/usr/bin/env python
"""Validate ``tools/eval_official_vtuav.py`` against the toolkit's own numbers.

The toolkit ships the reference trackers it was benchmarked with
(``toolkit/BBresults/*_<seq>.txt``) and one saved success figure,
``toolkit/figsResults/ALL_SR.fig``, whose legend labels are the MSR values the
toolkit itself computed.  This script re-derives the legend labels from the .fig
and recomputes MSR for the same trackers with our implementation, so the two can
be compared directly.

``ALL_SR.fig`` was produced by ``plot_LT.m`` (its ``algs`` list is the one with
``HMFT-LT``), i.e. it covers the LONG-TERM subset, so the default ``--subset`` is
``lt``.

    python tools/validate_vtuav_official.py
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.eval_official_vtuav import (  # noqa: E402
    load_gt,
    load_results,
    sequence_metrics,
)

#: legends in ALL_SR.fig use these names; BBresults uses these file prefixes
TRACKER_PREFIXES = {
    "DAFNet": "DAFNet",
    "ADRNet": "ADRNet",
    "FSRPN": "FSRPN",
    "mfDIMP": "mfDiMP",     # LT results are stored with the other capitalisation
    "HMFT": "HMFT",
    "HMFT-LT": "HMFT-LT",
}

LABEL_RE = re.compile(rb"[A-Za-z][A-Za-z0-9-]{1,15}\[[0-9.]+\]")


def read_fig_msr(path: str) -> Dict[str, float]:
    """Pull ``name[value]`` legend labels out of a MATLAB .fig (MSR values)."""
    try:
        from scipy.io import loadmat
    except ImportError:
        print("[WARN] scipy unavailable: cannot read the .fig, skipping the diff",
              file=sys.stderr)
        return {}

    texts: List[str] = []

    def as_text(node) -> Optional[str]:
        arr = np.asarray(node)
        if arr.dtype.kind in ("U", "S"):
            return "".join(str(x) for x in arr.ravel())
        if arr.dtype.kind in ("i", "u") and arr.dtype.itemsize <= 4:
            flat = arr.ravel()
            if flat.size == 0 or flat.size > 4096 or flat.max() > 255 or flat.min() < 9:
                return None
            return "".join(chr(int(v)) for v in flat)
        return None

    def walk(node, depth=0):
        if depth > 20 or node is None:
            return
        if isinstance(node, np.ndarray):
            if node.dtype.names:
                for name in node.dtype.names:
                    walk(node[name], depth + 1)
            elif node.dtype == object:
                for item in node.ravel()[:200]:
                    walk(item, depth + 1)
            else:
                text = as_text(node)
                if text:
                    texts.append(text)
        elif hasattr(node, "_fieldnames"):
            for field in node._fieldnames:
                walk(getattr(node, field), depth + 1)
        elif isinstance(node, (list, tuple)):
            for item in node[:200]:
                walk(item, depth + 1)

    for key, value in loadmat(path, struct_as_record=True, squeeze_me=False).items():
        if not key.startswith("__"):
            walk(value)

    values: Dict[str, float] = {}
    for text in texts:
        match = re.fullmatch(r"([A-Za-z][A-Za-z0-9-]{1,15})\[([0-9.]+)\]", text)
        if match:
            values[match.group(1)] = float(match.group(2))
    return values


def main() -> int:
    parser = argparse.ArgumentParser("validate the official VTUAV MPR/MSR implementation")
    parser.add_argument("--toolkit", default="/home/fzg/AQA/vtuavtoolkit/toolkit")
    parser.add_argument("--gt-root", default="/home/fzg/AQA/VTUAV_GT")
    parser.add_argument("--subset", default="lt", choices=["lt", "st"])
    parser.add_argument("--metric", default="MSR", choices=["MSR", "MPR"])
    parser.add_argument("--tolerance", type=float, default=5e-4,
                        help="legend labels are sprintf('%%.3f'), so 5e-4 is exact")
    args = parser.parse_args()

    if args.metric != "MSR":
        print(f"[NOTE] --metric {args.metric}: the only shipped figure is "
              f"figsResults/ALL_SR.fig (a SUCCESS plot), so only MSR has reference "
              f"values.  {args.metric} is computed from the same frame pairing, GT "
              f"parsing and centre/IoU code path, so the MSR match validates it "
              f"indirectly; the values below are informational only.",
              file=sys.stderr)

    split = "test_LT" if args.subset == "lt" else "test_ST"
    split_root = os.path.join(args.gt_root, split)
    if not os.path.isdir(split_root):
        print(f"GT split not found: {split_root}", file=sys.stderr)
        return 3
    sequences = sorted(d for d in os.listdir(split_root)
                       if os.path.isdir(os.path.join(split_root, d)))
    print(f"subset={args.subset}  sequences={len(sequences)}  gt={split_root}")

    fig = os.path.join(args.toolkit, "figsResults", "ALL_SR.fig")
    expected = read_fig_msr(fig)
    if expected:
        print(f"legend labels from {os.path.relpath(fig)}:")
        for name, value in expected.items():
            print(f"  {name:<10} {value:.3f}")
    else:
        print("[WARN] no legend labels extracted", file=sys.stderr)
    print()

    bb = os.path.join(args.toolkit, "BBresults")
    print(f"{'tracker':<10}{'prefix':<10}{'seqs':>6}{args.metric:>10}{'toolkit':>10}{'diff':>10}  verdict")
    print("-" * 68)

    failures = 0
    for name, prefix in TRACKER_PREFIXES.items():
        values: List[float] = []
        missing = 0
        for seq in sequences:
            path = os.path.join(bb, f"{prefix}_{seq}.txt")
            if not os.path.isfile(path):
                missing += 1
                continue
            gt_rgb = load_gt(os.path.join(split_root, seq, "rgb.txt"))
            gt_ir = load_gt(os.path.join(split_root, seq, "ir.txt"))
            pred = load_results(path)
            if gt_rgb is None or gt_ir is None or pred is None:
                missing += 1
                continue
            try:
                values.append(sequence_metrics(pred, gt_rgb, gt_ir)[args.metric])
            except ValueError:
                missing += 1
        if not values:
            print(f"{name:<10}{prefix:<10}{0:>6}   (no results found)")
            continue

        computed = float(np.mean(values))
        reference = expected.get(name)
        if reference is None:
            print(f"{name:<10}{prefix:<10}{len(values):>6}{computed:>10.4f}"
                  f"{'?':>10}{'?':>10}  (no legend entry)")
            continue
        if args.metric != "MSR":
            print(f"{name:<10}{prefix:<10}{len(values):>6}{computed:>10.4f}"
                  f"{'-':>10}{'-':>10}  (informational)")
            continue
        diff = computed - reference
        ok = abs(diff) <= args.tolerance
        if not ok:
            failures += 1
        print(f"{name:<10}{prefix:<10}{len(values):>6}{computed:>10.4f}"
              f"{reference:>10.3f}{diff:>+10.4f}  {'OK' if ok else 'MISMATCH'}"
              + (f"  ({missing} seq skipped)" if missing else ""))

    print("-" * 68)
    if not expected:
        print("VALIDATION INCONCLUSIVE: the reference figure could not be read")
        return 2
    if args.metric != "MSR":
        print(f"{args.metric} printed for information; only MSR has reference values")
        return 0
    if failures:
        print(f"VALIDATION FAILED: {failures} tracker(s) outside {args.tolerance}")
        return 1
    print(f"VALIDATION PASSED: every tracker reproduced the toolkit's own "
          f"{args.metric} within {args.tolerance}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
