#!/usr/bin/env python3
"""Explain a dense-protocol result: protocol artefact, or a real bug?

A dense VTUAV sweep scores **every** frame, but the ground truth only exists for
1 frame in 10.  The intermediate frames are therefore scored against a *stale*
annotation (up to half the annotation interval away), which depresses PR/SR even
for a perfect tracker.  This tool quantifies that effect on dumps you already
have -- it needs no GPU and no tracker, just the prediction files:

  A "dense"      : every frame vs its nearest annotation       (what the sweep reports)
  B "dense_ann"  : run the dense rollout, but score ONLY the annotated frames
                   (comparable with the official/sparse number)
  C gap          : A - B  =  pure ground-truth staleness penalty

It also reports how much the target moves between two annotations, which is the
mechanism behind C: if the annotation interval costs more than the 20 px PR
threshold, PR cannot stay high on the interpolated frames.

Usage::

    python tools/analyze_dense_vs_official.py \
        --result-dir RGBT_workspace/results/vtuav_st/vtuav_train_30_dense
    python tools/analyze_dense_vs_official.py --result-dir <dir> --dataset vtuav_lt --limit 20
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import statistics
import sys
from typing import Dict, List, Tuple

PRJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PRJ_ROOT not in sys.path:
    sys.path.insert(0, PRJ_ROOT)


def load_test_sgta() -> object:
    spec = importlib.util.spec_from_file_location(
        "test_sgta_mod", os.path.join(PRJ_ROOT, "RGBT_workspace", "test_sgta.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("dense vs annotation-only analysis on existing dumps")
    ap.add_argument("--result-dir", required=True,
                    help="dense dump dir, e.g. results/vtuav_st/<config>_<epoch>_dense")
    ap.add_argument("--dataset", default="vtuav_st", choices=("vtuav_st", "vtuav_lt"))
    ap.add_argument("--limit", type=int, default=0, help="analyse at most N sequences")
    ap.add_argument("--show", type=int, default=15, help="rows to print in the per-sequence table")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    tl = load_test_sgta()
    cfg = tl.DATASET_CFG[args.dataset]
    seq_dirs = tl.sequence_dirs(args.dataset)
    if not seq_dirs:
        print(f"no sequences found for {args.dataset} (check VTUAV_HOME/VTUAV_ST_ROOT)", file=sys.stderr)
        return 3

    rows: List[Dict[str, float]] = []
    skipped = 0
    for seq_dir in seq_dirs:
        name = os.path.basename(seq_dir)
        dump = os.path.join(args.result_dir, f"{name}.txt")
        if not os.path.isfile(dump):
            skipped += 1
            continue
        frames = tl.list_frames(seq_dir, cfg["rgb"])
        gts = tl.read_boxes(os.path.join(seq_dir, cfg["gt"]))
        n_img, n_gt = len(frames), len(gts)
        if n_img == 0 or n_gt == 0:
            skipped += 1
            continue
        preds = tl.read_predictions(dump)
        if len(preds) < min(n_img, n_gt):
            skipped += 1                      # partially written file
            continue

        if n_img > n_gt:
            ordinals, _ = tl.annotated_frame_ordinals(frames, n_gt, seq_name=name)
            gt_of_image = tl.dense_annotation_index(ordinals, n_img, n_gt)
        else:
            ordinals = list(range(n_img))
            gt_of_image = list(range(n_img))

        # ---- A: dense, every frame against its nearest annotation ----------
        n = min(len(preds), n_img)
        a = tl.compute_metrics(preds[:n], [gts[gt_of_image[j]] for j in range(n)], None, cfg)

        # ---- B: same rollout, scored only on the annotated frames ----------
        pairs = [(ordinals[i], i) for i in range(len(ordinals)) if ordinals[i] < n]
        b = tl.compute_metrics([preds[j] for j, _ in pairs], [gts[i] for _, i in pairs], None, cfg)

        # ---- C: how far the target moves between two annotations -----------
        shifts = []
        for i in range(1, len(ordinals)):
            g0, g1 = gts[i - 1], gts[i]
            if g0[2] > 0 and g1[2] > 0:
                shifts.append(tl.center_distance_xywh(g0, g1))
        rows.append({
            "seq": name, "n_img": n_img, "n_gt": n_gt,
            "A_PR": a.get("PR", -1), "A_SS": a.get("SS", -1),
            "B_PR": b.get("PR", -1), "B_SS": b.get("SS", -1),
            "shift_med": statistics.median(shifts) if shifts else float("nan"),
            "shift_mean": statistics.fmean(shifts) if shifts else float("nan"),
        })
        if args.limit and len(rows) >= args.limit:
            break

    if not rows:
        print("no usable dumps found in %s" % args.result_dir, file=sys.stderr)
        return 4

    def mean(key: str) -> float:
        vals = [r[key] for r in rows if r[key] >= 0]
        return statistics.fmean(vals) if vals else float("nan")

    rows.sort(key=lambda r: r["A_PR"])
    print("=" * 100)
    print("dense vs annotation-only scoring  (%d sequences, %d dump(s) missing)"
          % (len(rows), skipped))
    print("dir: %s   dataset: %s" % (os.path.abspath(args.result_dir), args.dataset))
    print("=" * 100)
    print("%-22s %7s %7s %7s %8s %8s %9s" %
          ("sequence", "imgs", "gt", "A.PR", "B.PR", "A.SS", "B.SS"))
    for r in rows[:args.show]:
        print("%-22s %7d %7d %7.3f %8.3f %8.3f %9.3f" %
              (r["seq"], r["n_img"], r["n_gt"], r["A_PR"], r["B_PR"], r["A_SS"], r["B_SS"]))
    if len(rows) > args.show:
        print("... (%d more; worst %d shown above)" % (len(rows) - args.show, args.show))

    print("-" * 100)
    print("mean over sequences")
    print("  A dense       (every frame vs nearest GT) : PR=%.4f  SS=%.4f" % (mean("A_PR"), mean("A_SS")))
    print("  B dense_ann   (same rollout, annotated only): PR=%.4f  SS=%.4f" % (mean("B_PR"), mean("B_SS")))
    print("  C = B - A (ground-truth staleness penalty) : PR=%+.4f  SS=%+.4f"
          % (mean("B_PR") - mean("A_PR"), mean("B_SS") - mean("A_SS")))
    print()
    shifts = [r["shift_med"] for r in rows if r["shift_med"] == r["shift_med"]]
    smean = [r["shift_mean"] for r in rows if r["shift_mean"] == r["shift_mean"]]
    if shifts:
        print("  annotation interval: target moves median %.1f px / mean %.1f px between"
              % (statistics.median(shifts), statistics.fmean(smean)))
        print("  two annotations -> on the interpolated frames the GT is on average off by"
              "\n  ~half of that, which is the floor of the dense PR@20px penalty.")
    over = sum(1 for r in rows if r["B_PR"] >= 0 and r["A_PR"] >= 0 and (r["B_PR"] - r["A_PR"]) > 0.05)
    print()
    print("  sequences where annotation-only scoring is >5 PR points better: %d/%d" % (over, len(rows)))
    if shifts and statistics.median(shifts) > 20:
        print("  VERDICT: the gap is consistent with stale GT (interval > 20 px); the dense")
        print("           protocol is behaving as designed, not mis-scoring frames.")
    else:
        print("  VERDICT: the gap is NOT explained by stale GT alone -- look for a real bug")
        print("           (frame pairing, GT index, or a dump written by another protocol).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
