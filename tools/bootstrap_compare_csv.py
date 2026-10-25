"""Paired bootstrap significance test using the project's own per-sequence
metrics (per_seq_metrics.csv written by RGBT_workspace/test_sgta.py).

Usage:
    python tools/bootstrap_compare_csv.py \
        --csv_a ./RGBT_workspace/results/lasher/lasher_train_10/per_seq_metrics.csv \
        --csv_b ./RGBT_workspace/results/lasher/lasher_train_15/per_seq_metrics.csv \
        --tag_a lasher_train_10 --tag_b lasher_train_15 \
        --metrics PR SS
"""
import argparse
import csv

import numpy as np


def load(path):
    rows = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows[row["seq_name"]] = row
    return rows


def bootstrap_paired(diff, seed=0, n_boot=20000):
    rng = np.random.default_rng(seed)
    n = len(diff)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = diff[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    p_less = float(np.mean(means <= 0))
    p_two = 2.0 * min(p_less, 1.0 - p_less)
    return means, lo, hi, p_less, p_two


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_a", required=True)
    ap.add_argument("--csv_b", required=True)
    ap.add_argument("--tag_a", required=True)
    ap.add_argument("--tag_b", required=True)
    ap.add_argument("--metrics", nargs="+", default=["PR", "SS"])
    args = ap.parse_args()

    a = load(args.csv_a)
    b = load(args.csv_b)
    common = sorted(set(a) & set(b))
    missing = sorted((set(a) | set(b)) - set(a) & set(b))
    print("sequences compared: %d (A=%d B=%d missing=%s)"
          % (len(common), len(a), len(b), missing or "none"))

    for metric in args.metrics:
        va = np.array([float(a[s][metric]) for s in common])
        vb = np.array([float(b[s][metric]) for s in common])
        diff = vb - va
        wins = int((diff > 1e-9).sum())
        losses = int((diff < -1e-9).sum())
        ties = len(diff) - wins - losses
        means, lo, hi, p_less, p_two = bootstrap_paired(diff)
        print("\n== %s: %s -> %s ==" % (metric, args.tag_a, args.tag_b))
        print("  seq-mean (B-A)   : %+.4f  (%.4f -> %.4f)" % (diff.mean(), va.mean(), vb.mean()))
        print("  per-seq diff std : %.4f" % diff.std())
        print("  wins/losses/ties : %d / %d / %d" % (wins, losses, ties))
        print("  95%% bootstrap CI : [%+.4f, %+.4f]" % (lo, hi))
        print("  p(diff<=0) one-sid: %.4f" % p_less)
        print("  p two-sided      : %.4f" % p_two)
        verdict = "SIGNIFICANT (p<0.05)" if p_two < 0.05 else "NOT significant"
        print("  verdict          : %s" % verdict)
        order = np.argsort(diff)
        print("  top-5 improve: " + ", ".join("%s(%+.2f)" % (common[i], diff[i]) for i in order[::-1][:5]))
        print("  top-5 degrade: " + ", ".join("%s(%+.2f)" % (common[i], diff[i]) for i in order[:5]))


if __name__ == "__main__":
    main()
