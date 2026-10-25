#!/usr/bin/env python
"""Official VTUAV MPR / MSR evaluation (reimplementation of the vtuavtoolkit).

Why this exists
---------------
The numbers printed by ``RGBT_workspace/test_sgta.py`` (PR/NPR/SR) use the RGBT
convention.  VTUAV's own toolkit defines two *different* summary metrics and it is
those that the benchmark papers report, so they have to be recomputed on the
stored dumps with the toolkit's exact rules:

============================  =========================================================
toolkit file                  what it fixes
============================  =========================================================
``load_groundtruth_txt_info``  GT lines are ``x y w h`` (extra columns ignored)
``calcPlotErr_TPR.m``          ``res(1:10:end,:)`` -- dumps are DENSE (one row per video
                               frame) and only every 10th row is compared to GT row i
``calcRectInt.m``              IoU with the ``x + w - 1`` corner convention,
                               ``union = w1*h1 + w2*h2 - inter``
``calcPlotErr_TPR.m``          centre = ``(x + w/2, y + h/2)``, Euclidean pixel distance
``calcPlotErr_TPR.m``          per frame, overlap = max(rgb, ir) and
                               centre error = min(rgb, ir)   (best of the two modalities)
``plot_ST.m`` / ``plot_LT.m``  MPR(seq) = fraction of frames with centre error <= 20 px;
                               MSR(seq) = mean over thresholds 0, 0.05, ..., 1 of the
                               fraction of frames with overlap > threshold
                               dataset value = MEAN OVER SEQUENCES (not over frames)
============================  =========================================================

Validation: this implementation reproduces ``toolkit/figsResults/ALL_SR.fig``
(the toolkit's own LT-subset success plot) to 3 decimals on the reference
trackers shipped in ``toolkit/BBresults`` -- see
``tools/validate_vtuav_official.py``.

Usage
-----
    python tools/eval_official_vtuav.py                       # both subsets
    python tools/eval_official_vtuav.py --datasets vtuav_lt
    python tools/eval_official_vtuav.py --runs '*_dense_ann'
    python tools/eval_official_vtuav.py --image-root /root/RGBTData/VTUAV/test_ST \
        --datasets vtuav_st        # full frame-grid check (needs the images)
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# toolkit constants (do not "improve" these: they are what defines the metric)
# ---------------------------------------------------------------------------
PRECISION_THRESHOLD = 20.0                                  # plot_ST.m: pixelThreshold
SUCCESS_THRESHOLDS = np.arange(0.0, 1.0 + 0.5 * 0.05, 0.05)  # plot_ST.m: successX
TOOLKIT_SUBSAMPLE = 10                                      # calcPlotErr_TPR.m: res(1:10:end,:)

DATASET_SPLITS = {"vtuav_st": "test_ST", "vtuav_lt": "test_LT"}

#: numpy renamed trapz -> trapezoid in 2.0; test_sgta.py still calls np.trapz
_TRAPZ = getattr(np, "trapezoid", None) or np.trapz


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def load_gt(path: str) -> Optional[np.ndarray]:
    """``x y w h`` per line (only the first four numbers are read, as textscan does)."""
    if not os.path.isfile(path):
        return None
    rows: List[List[float]] = []
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 4:
                continue
            try:
                rows.append([float(v) for v in parts[:4]])
            except ValueError:
                continue
    if not rows:
        return None
    return np.asarray(rows, dtype=np.float64)


def load_results(path: str) -> Optional[np.ndarray]:
    """Dump -> ``(N, 4)`` xywh.  8 columns are treated as 4 corners (corner2rect.m)."""
    if not os.path.isfile(path):
        return None
    rows: List[List[float]] = []
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.replace(",", " ").replace("[", " ").replace("]", " ").split()
            try:
                rows.append([float(v) for v in parts])
            except ValueError:
                continue
    if not rows:
        return None
    array = np.asarray(rows, dtype=np.float64)
    if array.ndim != 2:
        return None
    if array.shape[1] == 8:
        # corner2rect.m: round([x1, y1, x2 - x1, y3 - y1])
        return np.round(np.stack(
            [array[:, 0], array[:, 1], array[:, 2] - array[:, 0], array[:, 5] - array[:, 1]],
            axis=1))
    if array.shape[1] >= 4:
        return array[:, :4]
    return None


# ---------------------------------------------------------------------------
# metric core
# ---------------------------------------------------------------------------
def _iou_rect(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """calcRectInt.m, vectorised: ``+1`` corner convention, ``union = a1 + a2 - inter``."""
    left_a, bottom_a, w_a, h_a = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    left_b, bottom_b, w_b, h_b = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    right_a, top_a = left_a + w_a - 1.0, bottom_a + h_a - 1.0
    right_b, top_b = left_b + w_b - 1.0, bottom_b + h_b - 1.0

    inter_w = np.maximum(0.0, np.minimum(right_a, right_b) - np.maximum(left_a, left_b) + 1.0)
    inter_h = np.maximum(0.0, np.minimum(top_a, top_b) - np.maximum(bottom_a, bottom_b) + 1.0)
    inter = inter_w * inter_h
    union = w_a * h_a + w_b * h_b - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, inter / union, 0.0)
    return np.clip(iou, 0.0, 1.0)


def _centre_error(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Euclidean distance between ``(x + w/2, y + h/2)`` centres, in pixels."""
    dx = (a[:, 0] + a[:, 2] / 2.0) - (b[:, 0] + b[:, 2] / 2.0)
    dy = (a[:, 1] + a[:, 3] / 2.0) - (b[:, 1] + b[:, 3] / 2.0)
    return np.sqrt(dx * dx + dy * dy)


def sequence_metrics(pred: np.ndarray, gt_rgb: np.ndarray, gt_ir: np.ndarray) -> Dict[str, float]:
    """MPR/MSR for one sequence, exactly as ``calcPlotErr_TPR`` + ``plot_ST.m`` do it.

    ``pred`` is the RAW dense dump (every video frame); the toolkit keeps row
    0, 10, 20, ... of it before comparing with the GT rows.

    Also returns three *reconciliation* values, which are NOT the official
    metric but explain the difference against ``test_sgta.py``:

    * ``MPR_nd`` / ``MSR_nd``: same rules, but frames whose RGB GT box is
      degenerate (``w <= 0 or h <= 0``) are dropped.  ``test_sgta.py``
      ``compute_metrics`` filters exactly those frames out
      (``g[2] > 0 and g[3] > 0``); the toolkit scores them, and since a
      degenerate GT box can never be matched they count as failures.
    * ``MSR_trapz``: trapezoid AUC over the 21 thresholds instead of the plain
      mean.  ``test_sgta.py`` uses ``np.trapz(curve, thresholds)``, the toolkit
      uses ``sum/21``.
    * ``MSR_nd_trapz``: both of the above at once -- this is the variant that
      reproduces ``test_sgta.py``'s MSR almost exactly.
    """
    sub = pred[0::TOOLKIT_SUBSAMPLE]
    n_gt = min(len(gt_rgb), len(gt_ir))
    n = min(len(sub), n_gt)
    if n <= 0:
        raise ValueError("empty sequence")
    sub, rgb, ir = sub[:n], gt_rgb[:n], gt_ir[:n]

    iou = np.maximum(_iou_rect(sub, rgb), _iou_rect(sub, ir))          # best modality
    centre = np.minimum(_centre_error(sub, rgb), _centre_error(sub, ir))

    keep = (rgb[:, 2] > 0) & (rgb[:, 3] > 0)                           # test_sgta.py rule
    iou_nd, centre_nd = iou[keep], centre[keep]

    mpr = float(np.mean(centre <= PRECISION_THRESHOLD))
    per_threshold = [float(np.mean(iou > thr)) for thr in SUCCESS_THRESHOLDS]
    msr = float(np.mean(per_threshold))
    if len(centre_nd):
        curve_nd = [float(np.mean(iou_nd > thr)) for thr in SUCCESS_THRESHOLDS]
        mpr_nd = float(np.mean(centre_nd <= PRECISION_THRESHOLD))
        msr_nd = float(np.mean(curve_nd))
        msr_trapz = float(_TRAPZ(np.asarray(per_threshold), SUCCESS_THRESHOLDS))
        msr_nd_trapz = float(_TRAPZ(np.asarray(curve_nd), SUCCESS_THRESHOLDS))
    else:
        mpr_nd = msr_nd = msr_trapz = msr_nd_trapz = float("nan")
    return {
        "n_frames": n,
        "MPR": mpr,
        "MSR": msr,
        "mean_iou": float(np.mean(iou)),
        "PR50": float(np.mean(centre <= 50.0)),
        "SR50": float(np.mean(iou > 0.5)),
        "n_frames_nd": int(keep.sum()),
        "MPR_nd": mpr_nd,
        "MSR_nd": msr_nd,
        "MSR_trapz": msr_trapz,
        "MSR_nd_trapz": msr_nd_trapz,
    }


def evaluate_sequence(run_dir: str, seq: str, split_root: str) -> Dict[str, object]:
    pred = load_results(os.path.join(run_dir, f"{seq}.txt"))
    gt_rgb = load_gt(os.path.join(split_root, seq, "rgb.txt"))
    gt_ir = load_gt(os.path.join(split_root, seq, "ir.txt"))
    if pred is None:
        return {"seq": seq, "status": "no_prediction"}
    if gt_rgb is None or gt_ir is None:
        return {"seq": seq, "status": "no_gt"}

    expected = int(np.ceil(len(pred) / TOOLKIT_SUBSAMPLE))
    record: Dict[str, object] = {
        "seq": seq,
        "status": "ok",
        "pred_rows": len(pred),
        "gt_rows": len(gt_rgb),
        "gt_rows_ir": len(gt_ir),
        # the toolkit pairs GT row i with dump row 10*i, so the dump must hold
        # every video frame for that to reach all gt_rows rows; a SPARSE dump
        # (only the annotated frames, e.g. `--protocol official`) does not and
        # cannot be scored with the official rule at all.
        "grid_short": expected < len(gt_rgb),
        "grid_extra": expected - len(gt_rgb),
    }
    if len(gt_rgb) != len(gt_ir):
        record["status"] = "gt_length_mismatch"
        return record
    if expected < len(gt_rgb):
        record["status"] = "sparse_dump"
        return record

    record.update(sequence_metrics(pred, gt_rgb, gt_ir))
    return record


# ---------------------------------------------------------------------------
# optional: verify the toolkit's 1:10 grid against the loader's own alignment
# ---------------------------------------------------------------------------
def check_grid_against_alignment(run_dir: str, seq: str, split_root: str,
                                 image_root: str, n_gt: int) -> Optional[bool]:
    """True when ``dump row 10*i`` is the frame our loader pairs with GT row ``i``.

    Needs the image folder (``--image-root``); returns ``None`` when unavailable.
    """
    image_dir = os.path.join(image_root, seq, "rgb")
    if not os.path.isdir(image_dir):
        return None
    try:
        from lib.train.dataset.vtuav_alignment import annotated_frame_ordinals
    except Exception:  # pragma: no cover - helper is stdlib-only, so this is rare
        return None
    names, skipped = [], []
    for name in sorted(os.listdir(image_dir)):
        (skipped if name.startswith(".") else names).append(name)
    names = [n for n in names if os.path.splitext(n)[1].lower() in
             (".jpg", ".jpeg", ".png", ".bmp")]
    if not names:
        return None
    ordinals, _ = annotated_frame_ordinals(names, n_gt, seq_name=seq)
    if len(ordinals) != n_gt:
        return False
    return all(ordinal == TOOLKIT_SUBSAMPLE * i for i, ordinal in enumerate(ordinals))


# ---------------------------------------------------------------------------
# run / dataset level
# ---------------------------------------------------------------------------
def read_summary_metrics(run_dir: str) -> Dict[str, float]:
    """The run's own summary.json ``official_metrics`` (PR/NPR/SR), for context."""
    path = os.path.join(run_dir, "summary.json")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as handle:
            summary = json.load(handle)
    except Exception:  # noqa: BLE001
        return {}
    return dict(summary.get("official_metrics") or {})


def evaluate_run(run_dir: str, dataset: str, gt_root: str,
                 image_root: Optional[str] = None) -> Dict[str, object]:
    split_root = os.path.join(gt_root, DATASET_SPLITS[dataset])
    sequences = sorted(os.path.splitext(os.path.basename(p))[0]
                       for p in glob.glob(os.path.join(run_dir, "*.txt")))
    records: List[Dict[str, object]] = []
    for seq in sequences:
        record = evaluate_sequence(run_dir, seq, split_root)
        if image_root and record.get("status") == "ok":
            aligned = check_grid_against_alignment(
                run_dir, seq, split_root, image_root, int(record["gt_rows"]))
            if aligned is not None:
                record["grid_aligned"] = aligned
        records.append(record)

    ok = [r for r in records if r.get("status") == "ok"]
    problems = [r for r in records if r.get("status") != "ok"]
    summary: Dict[str, object] = {
        "run_dir": run_dir,
        "dataset": dataset,
        "gt_root": split_root,
        "n_sequences": len(sequences),
        "n_evaluated": len(ok),
        "n_problems": len(problems),
        "toolkit": "vtuavtoolkit res(1:10:end) + max/min modality fusion",
        "precision_threshold_px": PRECISION_THRESHOLD,
        "success_thresholds": [float(t) for t in SUCCESS_THRESHOLDS],
        "aggregation": "mean over sequences",
    }
    if not ok:
        print(f"[WARN] {run_dir}: no sequence could be scored "
              f"({len(problems)} problem(s), first status: "
              f"{problems[0].get('status') if problems else 'n/a'}).  The official "
              f"rule needs a dense dump (every video frame); a sparse "
              f"`--protocol official` dump cannot be scored this way.", file=sys.stderr)
    if ok:
        summary["MPR"] = float(np.mean([r["MPR"] for r in ok]))
        summary["MSR"] = float(np.mean([r["MSR"] for r in ok]))
        summary["mean_iou"] = float(np.mean([r["mean_iou"] for r in ok]))
        summary["SR50"] = float(np.mean([r["SR50"] for r in ok]))
        summary["total_frames"] = int(sum(r["n_frames"] for r in ok))
        # reconciliation against the repo's own convention (see sequence_metrics)
        summary["reconcile"] = {
            "MPR_nd": float(np.nanmean([r["MPR_nd"] for r in ok])),
            "MSR_nd": float(np.nanmean([r["MSR_nd"] for r in ok])),
            "MSR_trapz": float(np.nanmean([r["MSR_trapz"] for r in ok])),
            "MSR_nd_trapz": float(np.nanmean([r["MSR_nd_trapz"] for r in ok])),
            "frames_dropped": int(sum(r["n_frames"] - r["n_frames_nd"] for r in ok)),
            "note": ("_nd = frames with a degenerate RGB GT box (w<=0 or h<=0) dropped, "
                     "which is what test_sgta.compute_metrics does; _trapz = trapezoid "
                     "AUC over the same 21 thresholds instead of the mean"),
        }
    if image_root:
        checked = [r for r in ok if "grid_aligned" in r]
        summary["grid_checked"] = len(checked)
        summary["grid_misaligned"] = sum(1 for r in checked if not r["grid_aligned"])
    #: ``dense_ann`` dumps hold EVERY video frame, which is what the toolkit's
    #: ``res(1:10:end)`` subsampling needs.  A sparse dump (``--protocol
    #: official``, only the annotated frames) cannot be scored here; those runs
    #: are reported as ``sparse_dump`` and excluded from the means.
    summary["problem_sequences"] = [
        {k: r[k] for k in ("seq", "status", "pred_rows", "gt_rows") if k in r}
        for r in problems[:20]
    ]
    summary["_records"] = records
    return summary


def run_is_computed(run_dir: str) -> bool:
    """True when this run's official MPR/MSR output already exists and is current.

    "Current" means both output files are there AND no dump in the directory is
    newer than the JSON: a dump that was written later means sequences were
    re-tracked (or added) after the last evaluation, so the stored numbers are
    stale and must be redone.  This is what makes ``--skip-computed`` safe to use
    as a routine "only evaluate what is new" pass.
    """
    marker = os.path.join(run_dir, "official_mpr_msr.json")
    if not os.path.isfile(marker):
        return False
    if not os.path.isfile(os.path.join(run_dir, "official_mpr_msr_per_seq.csv")):
        return False
    marker_time = os.path.getmtime(marker)
    for dump in glob.glob(os.path.join(run_dir, "*.txt")):
        if os.path.getmtime(dump) > marker_time:
            return False
    return True


def load_saved_summary(run_dir: str) -> Optional[Dict[str, object]]:
    """The stored ``official_mpr_msr.json``, shaped like :func:`evaluate_run`.

    Returns ``None`` when it cannot be read, so the caller falls back to
    recomputing instead of failing.
    """
    path = os.path.join(run_dir, "official_mpr_msr.json")
    try:
        with open(path, encoding="utf-8") as handle:
            summary = json.load(handle)
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(summary, dict) or "MPR" not in summary:
        return None
    summary.setdefault("toolkit", "vtuavtoolkit res(1:10:end) + max/min modality fusion")
    return summary


def write_run_outputs(summary: Dict[str, object]) -> Tuple[str, str]:
    run_dir = str(summary["run_dir"])
    records = summary.pop("_records")

    # `seq_name` (not `seq`): that is the key tools/bootstrap_compare_csv.py reads,
    # so `--metrics MPR MSR` works directly on these files.
    csv_path = os.path.join(run_dir, "official_mpr_msr_per_seq.csv")
    fields = ["seq_name", "status", "n_frames", "MPR", "MSR", "mean_iou", "PR50", "SR50",
              "n_frames_nd", "MPR_nd", "MSR_nd", "MSR_trapz", "MSR_nd_trapz",
              "pred_rows", "gt_rows", "grid_short", "grid_extra", "grid_aligned"]
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            row = dict(record)
            row["seq_name"] = row.pop("seq", "")
            writer.writerow(row)

    json_path = os.path.join(run_dir, "official_mpr_msr.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return csv_path, json_path


def main() -> int:
    parser = argparse.ArgumentParser(
        "Official VTUAV MPR/MSR (reimplementation of vtuavtoolkit)")
    parser.add_argument("--result-root", default="./RGBT_workspace/results")
    parser.add_argument("--gt-root", default="/home/fzg/AQA/VTUAV_GT")
    parser.add_argument("--datasets", default="vtuav_st vtuav_lt")
    parser.add_argument("--runs", default="*",
                        help="run directory glob inside each dataset dir (default all)")
    parser.add_argument("--image-root", default="",
                        help="e.g. /root/RGBTData/VTUAV/test_ST: also verify the 1:10 grid "
                             "against the loader's own frame alignment")
    parser.add_argument("--out", default="",
                        help="write the combined table to this CSV as well")
    parser.add_argument("--limit", type=int, default=0,
                        help="process at most N run directories per dataset (0 = all)")
    parser.add_argument("--reconcile", action="store_true",
                        help="also print the two variants that reproduce test_sgta.py's "
                             "own numbers (degenerate GT dropped / trapezoid AUC); these "
                             "are NOT the official metric")
    parser.add_argument("--skip-computed", action="store_true",
                        help="only evaluate run dirs that are new or whose dumps changed; "
                             "reuse the stored official_mpr_msr.json everywhere else")
    parser.add_argument("--quiet", action="store_true", help="only the final table")
    args = parser.parse_args()

    if not os.path.isdir(args.gt_root):
        print(f"GT root not found: {args.gt_root}", file=sys.stderr)
        return 3

    rows: List[Dict[str, object]] = []
    recomputed: List[str] = []
    reused: List[str] = []
    for dataset in args.datasets.split():
        if dataset not in DATASET_SPLITS:
            print(f"unknown dataset: {dataset}", file=sys.stderr)
            return 2
        dataset_dir = os.path.join(args.result_root, dataset)
        run_dirs = sorted(d for d in glob.glob(os.path.join(dataset_dir, args.runs))
                          if os.path.isdir(d))
        if not run_dirs:
            print(f"[WARN] no run directories in {dataset_dir} matching {args.runs}",
                  file=sys.stderr)
        if args.limit:
            run_dirs = run_dirs[:args.limit]
        for run_dir in run_dirs:
            name = os.path.basename(run_dir)
            summary = None
            if args.skip_computed and run_is_computed(run_dir):
                summary = load_saved_summary(run_dir)
            if summary is not None:
                reused.append(f"{dataset}/{name}")
                csv_path = os.path.join(run_dir, "official_mpr_msr_per_seq.csv")
                json_path = os.path.join(run_dir, "official_mpr_msr.json")
            else:
                summary = evaluate_run(run_dir, dataset, args.gt_root,
                                       args.image_root or None)
                csv_path, json_path = write_run_outputs(summary)
                recomputed.append(f"{dataset}/{name}")

            context = read_summary_metrics(run_dir)
            context_text = " ".join(
                f"{key}={value:.4f}" for key, value in context.items()
                if isinstance(value, (int, float)))
            row = {
                "dataset": dataset,
                "run": os.path.basename(run_dir),
                "n_seq": summary.get("n_evaluated", 0),
                "frames": summary.get("total_frames", 0),
                "MPR": summary.get("MPR"),
                "MSR": summary.get("MSR"),
                "mean_iou": summary.get("mean_iou"),
                "summary_official": context_text,
                "MPR_nd": (summary.get("reconcile") or {}).get("MPR_nd"),
                "MSR_trapz": (summary.get("reconcile") or {}).get("MSR_trapz"),
                "problems": summary.get("n_problems", 0),
                "grid_misaligned": summary.get("grid_misaligned"),
            }
            rows.append(row)

            if not args.quiet:
                marker = " [reused]" if f"{dataset}/{name}" in reused else " [recomputed]"
                print(f"--- {dataset}/{name}{marker} ---")
                print(f"    sequences evaluated : {row['n_seq']}/{summary['n_sequences']}")
                print(f"    MPR (PR@20px, official) : {row['MPR']:.4f}"
                      if row["MPR"] is not None else "    MPR : n/a")
                print(f"    MSR (success AUC, off.) : {row['MSR']:.4f}"
                      if row["MSR"] is not None else "    MSR : n/a")
                if context_text:
                    print(f"    (RGBT convention in summary.json: {context_text})")
                if args.reconcile and summary.get("reconcile"):
                    rec = summary["reconcile"]
                    print("    -- reconciliation (NOT the official number) --")
                    print(f"       degenerate RGB GT dropped ({rec['frames_dropped']} frame(s))"
                          f" -> MPR {rec['MPR_nd']:.4f}, MSR {rec['MSR_nd']:.4f}")
                    print(f"       trapezoid AUC instead of the 21-point mean"
                          f"  -> MSR {rec['MSR_trapz']:.4f}")
                    print(f"       both (== test_sgta.py's convention)"
                          f"             -> MPR {rec['MPR_nd']:.4f}, MSR {rec['MSR_nd_trapz']:.4f}")
                if summary.get("grid_checked"):
                    print(f"    frame-grid check : {summary['grid_misaligned']} of "
                          f"{summary['grid_checked']} sequences disagree with the "
                          f"loader's own alignment")
                if row["problems"]:
                    print(f"    problems : {row['problems']} sequence(s) skipped")
                    for item in summary["problem_sequences"][:5]:
                        print(f"      {item}")
                print(f"    wrote {csv_path}")
                print(f"          {json_path}")

    print()
    if recomputed or reused:
        print(f"recomputed {len(recomputed)} run(s)"
              + (f": {', '.join(recomputed)}" if recomputed else ""))
        print(f"reused     {len(reused)} run(s) already up to date"
              + (f": {', '.join(reused)}" if reused and not recomputed else ""))
        print()
    header = (f"{'dataset':<10}{'run':<40}{'seqs':>6}{'frames':>9}"
              f"{'MPR':>9}{'MSR':>9}{'meanIoU':>9}  summary.json")
    print("=" * 100)
    print("official VTUAV MPR/MSR  (summary.json column = that run's own RGBT-convention values)")
    print("=" * 100)
    print(header)
    print("-" * 100)
    for row in rows:
        def fmt(value, width=8):
            return f"{'n/a':>{width}}" if value is None else f"{value:>{width}.4f}"
        print(f"{row['dataset']:<10}{row['run']:<40}{row['n_seq']:>6}{row['frames']:>9}"
              f"{fmt(row['MPR'], 9)}{fmt(row['MSR'], 9)}{fmt(row['mean_iou'], 9)}"
              f"  {row['summary_official']}")
    print("=" * 100)

    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"combined table -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
