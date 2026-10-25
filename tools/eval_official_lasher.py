#!/usr/bin/env python
"""Re-evaluate SGTA result dumps with the OFFICIAL RGBT toolkit.

The official toolkit (`rgbt` 1.0.1, installed in the mambavision_rgbt env, source
also at ~/AQA/rgbt-1.0.1/rgbt-1.0.1) defines the LasHeR metrics differently from
this repo's internal evaluator (`RGBT_workspace/test_sgta.py`):

| metric | official (rgbt 1.0.1)                                  | repo internal                     |
|--------|---------------------------------------------------------|-----------------------------------|
| PR     | CLE <= 20 px, `(w-1)/2` centre convention                | CLE <= 20 px, `w/2` centre        |
| NPR    | `normalize_CLE` (x/gt_w, y/gt_h), threshold **0.2**      | dist/sqrt(w*h), threshold **0.5** |
| SR     | mean over 21 IoU thresholds in [0,1]                     | AUC (trapz) over same thresholds  |
| input  | boxes `np.round(0)` to integers, frame-0 forced to GT,   | raw float boxes, no repair        |
|        | invalid boxes repaired from the previous frame           |                                   |

So the official numbers are the ones comparable to the literature; the internal
ones are for tracking progress only.

Usage
-----
    python tools/eval_official_lasher.py                     # all matching dirs
    python tools/eval_official_lasher.py --pattern lasher_   # any prefix
    python tools/eval_official_lasher.py --allow-partial     # include partial dirs
"""
import argparse
import csv
import os
import sys
import time


def parse_epoch(name):
    """'lasher_train_12' -> 12 (or a large sentinel if absent)."""
    tail = name.rsplit("_", 1)[-1]
    try:
        return int(tail)
    except ValueError:
        return 10 ** 6


def load_internal(history, prefix):
    """epoch -> (PR, NPR, SR) from the repo's own eval_history.csv."""
    out = {}
    if not os.path.isfile(history):
        return out
    with open(history) as fh:
        for row in csv.DictReader(fh):
            tag = row.get("ckpt_tag", "")
            if not tag.startswith(prefix):
                continue
            try:
                epoch = int(tag.rsplit("_", 1)[-1])
                out[epoch] = (float(row["seq_PR"]), float(row["seq_NPS"]),
                              float(row["seq_SS"]))
            except (ValueError, KeyError):
                continue
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default=None)
    ap.add_argument("--pattern", default="lasher_")
    ap.add_argument("--internal-history", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--ns", default="",
                    help="results namespace, e.g. lasher_a0: shorthand for "
                         "--results-root/--internal-history/--out inside "
                         "RGBT_workspace/results/<ns>.  Explicit paths win.")
    ap.add_argument("--min-seqs", type=int, default=245)
    ap.add_argument("--allow-partial", action="store_true")
    args = ap.parse_args()

    # Resolve the result paths: an explicit flag wins, then --ns, then the
    # legacy shared tree (results/lasher).
    if not args.ns:
        args.ns = "lasher"
    ns_dir = os.path.join("RGBT_workspace", "results", args.ns)
    if args.results_root is None:
        args.results_root = ns_dir
    if args.internal_history is None:
        args.internal_history = os.path.join(ns_dir, "eval_history.csv")
    if args.out is None:
        args.out = os.path.join(ns_dir, "official_eval_history.csv")
    print("results root : %s" % args.results_root)

    try:
        from rgbt import LasHeR
    except ImportError:
        print("cannot import rgbt -- use the mambavision_rgbt interpreter", file=sys.stderr)
        return 3

    # ---- collect candidate result directories -------------------------------
    cands = []
    for name in os.listdir(args.results_root):
        path = os.path.join(args.results_root, name)
        if not os.path.isdir(path) or args.pattern not in name:
            continue
        n_txt = sum(1 for f in os.listdir(path) if f.endswith(".txt"))
        cands.append((parse_epoch(name), name, path, n_txt))
    cands.sort()

    done = [(e, n, p, c) for e, n, p, c in cands if c >= args.min_seqs]
    partial = [(e, n, p, c) for e, n, p, c in cands if c < args.min_seqs]
    if partial:
        print("skipping %d incomplete dir(s):" % len(partial))
        for e, n, p, c in partial:
            print("   %-34s %d/%d sequences%s"
                  % (n, c, args.min_seqs, "  (use --allow-partial)" if args.allow_partial else ""))
    if args.allow_partial:
        done += partial
    if not done:
        print("no result directories found", file=sys.stderr)
        return 4

    print("\nofficial toolkit: rgbt %s" % getattr(__import__("rgbt"), "__version__", "?"))
    lasher = LasHeR()

    def gt_frames(seq):
        """LasHeR ships a single flat GT file, but the base class may return a
        {'visible': .., 'infrared': ..} dict for other datasets."""
        gt = lasher[seq]
        return len(gt["visible"] if isinstance(gt, dict) else gt)

    print("dataset: %s | sequences: %d | frames: %d\n"
          % (lasher.name, len(lasher), sum(gt_frames(s) for s in lasher.seqs_name)))

    internal = {}
    rows = []
    header = ("epoch", "PR_off", "NPR_off", "SR_off",
              "PR_int", "NPR_int", "SR_int", "dPR", "dNPR", "dSR", "n_seq", "sec")
    print("%-6s %8s %8s %8s | %8s %8s %8s | %7s %7s %7s %6s %6s"
          % header)
    print("-" * 92)

    for epoch, name, path, n_txt in done:
        t0 = time.time()
        try:
            lasher(tracker_name=name, result_path=path, bbox_type="ltwh")
            pr, _ = lasher.PR(name)
            npr, _ = lasher.NPR(name)
            sr, _ = lasher.SR(name)
        except Exception as exc:                                  # noqa: BLE001
            print("%-6d ERROR %s: %s" % (epoch, name, exc))
            continue
        secs = time.time() - t0
        # match the internal row of THIS run (dir name without the trailing
        # _<epoch>), so ablation rows never borrow each other's numbers
        prefix = name.rsplit("_", 1)[0]
        if prefix not in internal:
            internal[prefix] = load_internal(args.internal_history, prefix)
        ipr, inpr, isr = internal[prefix].get(epoch, (float("nan"),) * 3)
        rows.append(dict(epoch=epoch, tag=name, result_dir=os.path.abspath(path),
                         n_seq=n_txt, PR=pr, NPR=npr, SR=sr,
                         PR_internal=ipr, NPR_internal=inpr, SR_internal=isr,
                         seconds=round(secs, 1)))
        print("%-6d %8.4f %8.4f %8.4f | %8.4f %8.4f %8.4f | %+7.4f %+7.4f %+7.4f %6d %6.1f"
              % (epoch, pr, npr, sr, ipr, inpr, isr, pr - ipr, npr - inpr, sr - isr,
                 n_txt, secs))
        del lasher.trackers[name]        # free memory

    if not rows:
        print("nothing evaluated", file=sys.stderr)
        return 5

    # ---- write CSV ----------------------------------------------------------
    fields = ["tag", "epoch", "n_seq", "PR", "NPR", "SR",
              "PR_internal", "NPR_internal", "SR_internal",
              "result_dir", "seconds"]
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})

    def stat(key):
        v = [r[key] for r in rows if r[key] == r[key]]
        return sum(v) / len(v) if v else float("nan")

    print("-" * 92)
    print("best official PR   : %.4f (%.2f%%) @ep%d"
          % (max(r["PR"] for r in rows), 100 * max(r["PR"] for r in rows),
             max(rows, key=lambda r: r["PR"])["epoch"]))
    print("best official NPR  : %.4f (%.2f%%)"
          % (max(r["NPR"] for r in rows), 100 * max(r["NPR"] for r in rows)))
    print("best official SR   : %.4f (%.2f%%)"
          % (max(r["SR"] for r in rows), 100 * max(r["SR"] for r in rows)))
    print("mean official      : PR %.4f  NPR %.4f  SR %.4f" % (stat("PR"), stat("NPR"), stat("SR")))
    print("mean internal      : PR %.4f  NPR %.4f  SR %.4f"
          % (stat("PR_internal"), stat("NPR_internal"), stat("SR_internal")))
    print("mean offset (off - int): PR %+.4f  NPR %+.4f  SR %+.4f"
          % (stat("PR") - stat("PR_internal"), stat("NPR") - stat("NPR_internal"),
             stat("SR") - stat("SR_internal")))
    print("\nwrote %s (%d rows)" % (args.out, len(rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
