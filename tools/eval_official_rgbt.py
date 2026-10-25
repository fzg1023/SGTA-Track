#!/usr/bin/env python3
"""Evaluate SGTA-Track tracking results with the official RGBT toolkit (rgbt-1.0.1).

The official toolkit is used *as-is* (no patching): it is imported from
``<rgbt_root>/src`` so no ``pip install`` is required.

Input layout, exactly what ``RGBT_workspace/test_sgta.py`` writes::

    <result_root>/<dataset>/<tracker_tag>/<seq_name>.txt

Each file holds one ``x,y,w,h`` (top-left corner + width/height, i.e. ``ltwh``)
box per line with comma separators -- the native bbox type of the toolkit, so no
coordinate conversion is applied.

Metrics reported per dataset (names follow the toolkit):

    lasher   PR   precision @ 20px (sequence mean)     NPR  normalized precision @ 0.2
             SR   mean success over 21 IoU thresholds
    gtot     MPR  precision @ 5px  (frame weighted)   MSR  success AUC
    rgbt210  PR   precision @ 20px (sequence mean)     SR   success AUC
    rgbt234  MPR  precision @ 20px (sequence mean)     MSR  success AUC

NOTE on LasHeR: the official PR/NPR/SR are NOT the same quantities as this
repo's internal `test_sgta.py` metrics. The internal evaluator uses
dist/sqrt(w*h) with a 0.5 threshold for NPR and an AUC (trapz) for SR, which
inflates NPR by roughly 7 points; only the official numbers are comparable to
the literature. Use `tools/eval_official_lasher.py` to sweep a whole series of
per-epoch dumps and to see the internal-vs-official offset.

Usage::

    python tools/eval_official_rgbt.py --tracker_tag lasher_train_30
    python tools/eval_official_rgbt.py --datasets lasher --tracker_tag my_tag
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

PRJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# dataset key -> (class name, metric method names, result subdirectory)
DATASET_SPEC: Dict[str, Tuple[str, Tuple[str, str]]] = {
    "lasher": ("LasHeR", ("PR", "NPR", "SR")),
    "gtot": ("GTOT", ("MPR", "MSR")),
    "rgbt210": ("RGBT210", ("PR", "SR")),
    "rgbt234": ("RGBT234", ("MPR", "MSR")),
}

# First existing entry wins; the pip-installed `rgbt` is used as a last resort.
DEFAULT_RGBT_ROOTS = (
    os.path.expanduser("~/AQA/rgbt-1.0.1/rgbt-1.0.1"),
    "/home/sau_fzg1/Drop/rgbt-1.0.1",
)


def load_toolkit(rgbt_root: str):
    src = os.path.join(rgbt_root, "src")
    if os.path.isdir(src):
        if src not in sys.path:
            sys.path.insert(0, src)
    else:
        print(f"[WARN] toolkit src not found at {src}; "
              f"falling back to the importable `rgbt` package")
    import rgbt  # noqa: F401
    from rgbt import GTOT, LasHeR, RGBT210, RGBT234  # noqa: F401
    from rgbt.utils import RGBT_end, RGBT_start

    return {"GTOT": GTOT, "LasHeR": LasHeR, "RGBT210": RGBT210, "RGBT234": RGBT234,
            "RGBT_start": RGBT_start, "RGBT_end": RGBT_end}


def gt_lengths(dataset, seq_name: str) -> List[int]:
    """Ground-truth length per modality.

    GTOT / RGBT210 / RGBT234 store ``<seq>/{visible,infrared}.txt`` and return a
    dict; LasHeR ships a single flat ``<seq>.txt`` and returns an ndarray.
    """
    serial = dataset[seq_name]
    if isinstance(serial, dict):
        return [len(serial["visible"]), len(serial["infrared"])]
    return [len(serial)]


def build_dataset(key: str, toolkit: dict):
    """Instantiate a dataset object.

    ``RGBT_start()`` must wrap construction for GTOT: its ground truth is stored
    as ``ltrb`` and the toolkit's ``ltrb_2_ltwh`` depends on the global strict
    flag. RGBT210/RGBT234 ground truth is already ``ltwh`` so it is a no-op
    there, but the same path keeps behaviour identical across datasets.
    """
    class_name, _ = DATASET_SPEC[key]
    toolkit["RGBT_start"]()
    try:
        dataset = toolkit[class_name]()
    finally:
        toolkit["RGBT_end"]()
    return dataset


def check_coverage(dataset, result_dir: str, tag: str) -> Tuple[List[str], List[str]]:
    """Return (missing, mismatched) sequences for the tracking result directory."""
    missing: List[str] = []
    mismatched: List[str] = []
    for seq_name in dataset.seqs_name:
        path = os.path.join(result_dir, f"{seq_name}.txt")
        if not os.path.isfile(path):
            missing.append(seq_name)
            continue
        n_pred = sum(1 for line in open(path, encoding="utf-8") if line.strip())
        try:
            n_gt = min(gt_lengths(dataset, seq_name))
        except Exception:
            continue
        if n_pred != n_gt:
            mismatched.append(f"{seq_name}(pred={n_pred},gt={n_gt})")
    return missing, mismatched


def main() -> int:
    parser = argparse.ArgumentParser("official RGBT toolkit evaluation")
    parser.add_argument("--rgbt_root",
                        default=os.environ.get("RGBT_ROOT", DEFAULT_RGBT_ROOTS[0]),
                        help="root of the official rgbt toolkit (contains src/rgbt); "
                             "falls back to the installed `rgbt` package if absent")
    parser.add_argument("--result_root", default=os.path.join(PRJ_ROOT, "RGBT_workspace", "results"),
                        help="root holding <dataset>/<tracker_tag>/<seq>.txt")
    parser.add_argument("--tracker_tag", default="lasher_train_30",
                        help="result sub-directory name produced by test_sgta.py")
    parser.add_argument("--datasets", nargs="+", default=["lasher", "gtot", "rgbt210", "rgbt234"],
                        choices=sorted(DATASET_SPEC))
    parser.add_argument("--bbox_type", default="ltwh",
                        help="bbox type of the result files; test_sgta.py writes ltwh")
    parser.add_argument("--out_json", default="",
                        help="optional path for a JSON report (default: <result_root>/official_eval.json)")
    parser.add_argument("--allow_missing", action="store_true",
                        help="evaluate only the sequences that have results instead of failing")
    parser.add_argument("--check_only", action="store_true",
                        help="only load the toolkit and print dataset/GT info, then exit")
    args = parser.parse_args()

    toolkit = load_toolkit(args.rgbt_root)

    report: Dict[str, dict] = {}
    failed = False

    for key in args.datasets:
        _, metric_names = DATASET_SPEC[key]
        result_dir = os.path.join(args.result_root, key, args.tracker_tag)

        print("=" * 78)
        print(f"dataset    : {key}")
        print(f"result dir : {result_dir}")

        dataset = build_dataset(key, toolkit)
        if args.check_only:
            lengths = []
            for seq_name in dataset.seqs_name:
                try:
                    lengths.append(min(gt_lengths(dataset, seq_name)))
                except Exception as exc:  # noqa: BLE001
                    print(f"[WARN] failed to read ground truth for {seq_name}: {exc}")
            lengths.sort()
            if lengths:
                print(f"sequences  : {len(dataset.seqs_name)}  "
                      f"frames min={lengths[0]} median={lengths[len(lengths) // 2]} "
                      f"max={lengths[-1]} total={sum(lengths)}")
            continue

        if not os.path.isdir(result_dir):
            print("[ERROR] result directory missing; run test_sgta.py first")
            failed = True
            continue

        missing, mismatched = check_coverage(dataset, result_dir, args.tracker_tag)

        print(f"sequences  : {len(dataset.seqs_name)} in toolkit, "
              f"{len(dataset.seqs_name) - len(missing)} with results")
        if missing:
            preview = ", ".join(missing[:8]) + (" ..." if len(missing) > 8 else "")
            print(f"[WARN] {len(missing)} sequence(s) without results: {preview}")
        if mismatched:
            preview = ", ".join(mismatched[:8]) + (" ..." if len(mismatched) > 8 else "")
            print(f"[WARN] {len(mismatched)} frame-count mismatch(es): {preview}")

        if missing and not args.allow_missing:
            print("[SKIP] not all sequences have results; re-run test_sgta.py "
                  "(or pass --allow_missing to score a subset)")
            failed = True
            continue

        eval_seqs = [name for name in dataset.seqs_name if name not in missing]
        if not eval_seqs:
            print("[ERROR] no sequence to evaluate")
            failed = True
            continue

        # `seqs` must be forwarded explicitly: TrackerResult eagerly loads every
        # file in the sequence list, so registering without it would raise on the
        # first missing result even when --allow_missing is set.
        dataset(args.tracker_tag, result_dir, seqs=eval_seqs, bbox_type=args.bbox_type)

        entry: Dict[str, float] = {}
        for name in metric_names:
            value, _curve = getattr(dataset, name)(args.tracker_tag, seqs=eval_seqs)
            entry[name] = float(value)
            print(f"  {name:>4} = {value * 100:.2f}")

        entry["n_sequences"] = len(eval_seqs)
        entry["n_missing"] = len(missing)
        entry["n_frame_count_mismatch"] = len(mismatched)
        report[key] = entry

    if args.check_only:
        print("=" * 78)
        print("[CHECK] toolkit loaded, ground truth readable; no evaluation performed")
        return 0 if not failed else 1

    print("=" * 78)
    if report:
        summary = " | ".join(
            f"{key} " + " ".join(f"{m}={v * 100:.2f}" for m, v in entry.items()
                                 if isinstance(v, float))
            for key, entry in report.items())
        print(f"[OFFICIAL] {summary}")
    else:
        print("[OFFICIAL] no dataset evaluated")

    out_json = args.out_json or os.path.join(args.result_root, "official_eval.json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    payload = {"tracker_tag": args.tracker_tag, "bbox_type": args.bbox_type,
               "toolkit_root": args.rgbt_root, "results": report}
    with open(out_json, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    print(f"[INFO] report -> {out_json}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
