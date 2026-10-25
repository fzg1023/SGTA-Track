#!/usr/bin/env python
"""Prove that test_sgta.official_seq_metrics() reproduces the official toolkit.

`RGBT_workspace/test_sgta.py` now computes official LasHeR PR/NPR/SR online so the
per-epoch output can show a running mean (and the training loop can print the
official trio without a second rollout).  That is only useful if the numbers
agree with `rgbt` 1.0.1, which is what this script checks on an *existing* dump
directory -- no GPU, no tracking, takes seconds.

Usage::

    python tools/check_official_metric_parity.py                        # newest run
    python tools/check_official_metric_parity.py --epoch 24
    python tools/check_official_metric_parity.py --result-dir /abs/path
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

# Must be set before test_sgta is imported (its DATASET_CFG is built at import).
os.environ.setdefault("LASHER_TEST_ROOT", "/home/fzg/data/lasher/testingset")
os.environ.setdefault("RGBT_DATA_ROOT", "/home/fzg/data/lasher/testingset")


def load_test_sgta():
    spec = importlib.util.spec_from_file_location(
        "test_sgta", os.path.join(REPO, "RGBT_workspace", "test_sgta.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", default="")
    ap.add_argument("--results-root", default=os.path.join(REPO, "RGBT_workspace", "results", "lasher"))
    ap.add_argument("--pattern", default="lasher_train_")
    ap.add_argument("--epoch", type=int, default=0, help="pick <pattern><epoch> if set")
    ap.add_argument("--tol", type=float, default=0.002, help="max |delta| accepted")
    args = ap.parse_args()

    if args.result_dir:
        result_dir = os.path.abspath(args.result_dir)
    else:
        cands = [d for d in os.listdir(args.results_root)
                 if args.pattern in d and os.path.isdir(os.path.join(args.results_root, d))]
        if not cands:
            print("no result dir found", file=sys.stderr)
            return 3
        if args.epoch:
            want = f"{args.pattern}{args.epoch}"
            cands = [d for d in cands if d == want] or cands
        cands.sort(key=lambda d: int(d.rsplit("_", 1)[-1]) if d.rsplit("_", 1)[-1].isdigit() else -1)
        result_dir = os.path.join(args.results_root, cands[-1])
    tag = os.path.basename(result_dir)
    print(f"result dir : {result_dir}")

    tl = load_test_sgta()
    cfg = tl.DATASET_CFG["lasher"]
    seq_dirs = tl.sequence_dirs("lasher")
    print(f"sequences  : {len(seq_dirs)}")

    # ---- our online implementation (same code path as the training eval) ----
    per_seq = []
    for seq_dir in seq_dirs:
        name = os.path.basename(seq_dir)
        pred_path = os.path.join(result_dir, name + ".txt")
        if not os.path.isfile(pred_path):
            print(f"[WARN] missing dump for {name}; skipped")
            continue
        preds = tl.read_predictions(pred_path)
        gts = tl.read_boxes(os.path.join(seq_dir, cfg["gt"]))
        n = min(len(preds), len(gts))
        if n == 0:
            continue
        per_seq.append(tl.official_seq_metrics(preds[:n], gts[:n]))
    if not per_seq:
        print("no sequence scored", file=sys.stderr)
        return 4
    mine = {k: float(np.mean([r[k] for r in per_seq])) for k in tl.OFFICIAL_KEYS}
    print(f"scored     : {len(per_seq)} sequences")

    # ---- official toolkit ---------------------------------------------------
    try:
        from rgbt import LasHeR
    except ImportError:
        print("cannot import rgbt -- use the mambavision_rgbt interpreter", file=sys.stderr)
        return 5
    lasher = LasHeR()
    # count frames the same way, as an extra sanity check on the dump itself
    n_frames = 0
    for seq_dir in seq_dirs:
        gt = lasher[os.path.basename(seq_dir)]
        n_frames += len(gt["visible"] if isinstance(gt, dict) else gt)
    print(f"frames     : {n_frames} (toolkit ground truth)")
    lasher(tracker_name=tag, result_path=result_dir, bbox_type="ltwh")
    theirs = {"oPR": lasher.PR(tag)[0], "oNPR": lasher.NPR(tag)[0], "oSR": lasher.SR(tag)[0]}

    # ---- compare ------------------------------------------------------------
    names = {"oPR": "PR", "oNPR": "NPR", "oSR": "SR"}
    print()
    print(f"{'metric':<6} {'ours':>10} {'toolkit':>10} {'delta':>10}")
    print("-" * 40)
    ok = True
    for key in tl.OFFICIAL_KEYS:
        d = mine[key] - theirs[key]
        flag = "" if abs(d) <= args.tol else "   <-- MISMATCH"
        if abs(d) > args.tol:
            ok = False
        print(f"{names[key]:<6} {mine[key]:10.4f} {theirs[key]:10.4f} {d:+10.4f}{flag}")
    print("-" * 40)
    print(f"tolerance  : {args.tol}")
    print("PARITY OK" if ok else "PARITY FAILED")
    return 0 if ok else 6


if __name__ == "__main__":
    sys.exit(main())
