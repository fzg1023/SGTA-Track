"""Per-sequence analysis of a lasher eval (per_seq_metrics.csv)."""
import argparse
import csv
import os

import numpy as np


def load(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    args = ap.parse_args()
    rows = load(args.csv)
    print("sequences: %d" % len(rows))

    def f(r, k):
        try:
            return float(r[k])
        except Exception:
            return -1.0

    for k in ("PR", "SS", "AO", "SR50", "SR75"):
        vals = np.array([f(r, k) for r in rows if f(r, k) >= 0])
        print("  %-5s mean=%.4f min=%.4f p10=%.4f" % (k, vals.mean(), vals.min(), np.percentile(vals, 10)))

    print("\n== worst 15 by PR (定位漂移/丢失) ==")
    for r in sorted(rows, key=lambda r: f(r, "PR"))[:15]:
        print("  %-32s PR=%.3f SS=%.3f AO=%.3f frames=%s" %
              (r["seq_name"], f(r, "PR"), f(r, "SS"), f(r, "AO"), r.get("frame_count", "?")))

    print("\n== SS 明显低于 PR 的序列(尺度/重叠问题) ==")
    gap = []
    for r in rows:
        pr, ss = f(r, "PR"), f(r, "SS")
        if pr >= 0 and ss >= 0:
            gap.append((pr - ss, r))
    for d, r in sorted(gap, reverse=True)[:15]:
        if d > 0.15:
            print("  %-32s PR=%.3f SS=%.3f gap=%.3f frames=%s" %
                  (r["seq_name"], f(r, "PR"), f(r, "SS"), d, r.get("frame_count", "?")))

    print("\n== 长序列(>2000帧)表现 ==")
    longs = [r for r in rows if int(r.get("frame_count", 0) or 0) > 2000]
    if longs:
        prs = np.array([f(r, "PR") for r in longs])
        sss = np.array([f(r, "SS") for r in longs])
        print("  n=%d  mean PR=%.4f  mean SS=%.4f" % (len(longs), prs.mean(), sss.mean()))
        for r in sorted(longs, key=lambda r: f(r, "PR"))[:10]:
            print("    %-32s PR=%.3f SS=%.3f frames=%s" %
                  (r["seq_name"], f(r, "PR"), f(r, "SS"), r.get("frame_count", "?")))


if __name__ == "__main__":
    main()
