"""Paired bootstrap significance test between two checkpoints.

Uses the exact same per-sequence metric computation as RGBT_workspace/test_sgta.py
(via the `rgbt` toolkit, PR_LasHeR / SR_LasHeR), then runs a paired bootstrap
over sequences to test whether the mean per-sequence PR/SR difference is
statistically different from zero.

Usage:
    python tools/bootstrap_compare.py \
        --tag_a lasher_train_10 \
        --tag_b lasher_train_15 \
        --res_a ./RGBT_workspace/results/lasher/lasher_train_10 \
        --res_b ./RGBT_workspace/results/lasher/lasher_train_15
"""
import argparse
import os

import numpy as np


def load_per_seq(tag, result_path):
    from rgbt import LasHeR
    bench = LasHeR()
    bench(tracker_name=tag, result_path=result_path, bbox_type="ltwh")
    seqs = bench.seqs_name
    pr_val, pr = bench.PR_fun(bench, bench.trackers[tag], seqs)   # (val, (N,51))
    sr_val, sr = bench.SR_fun(bench, bench.trackers[tag], seqs)   # (val, (N,21))
    pr_seq = np.asarray(pr)[:, 20].astype(np.float64)             # fraction CLE<=20px
    sr_seq = np.asarray(sr).mean(axis=1).astype(np.float64)       # mean over IoU thresholds
    return seqs, pr_seq, sr_seq, float(pr_val), float(sr_val)


def bootstrap_paired(diff, seed=0, n_boot=20000):
    rng = np.random.default_rng(seed)
    n = len(diff)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = diff[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    p_less = float(np.mean(means <= 0))      # P(diff <= 0)
    p_two = 2.0 * min(p_less, 1.0 - p_less)  # two-sided
    return means, lo, hi, p_less, p_two


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag_a", required=True)
    ap.add_argument("--tag_b", required=True)
    ap.add_argument("--res_a", required=True)
    ap.add_argument("--res_b", required=True)
    args = ap.parse_args()

    print("loading %s ..." % args.tag_a)
    seqs_a, pr_a, sr_a, prv_a, srv_a = load_per_seq(args.tag_a, args.res_a)
    print("loading %s ..." % args.tag_b)
    seqs_b, pr_b, sr_b, prv_b, srv_b = load_per_seq(args.tag_b, args.res_b)

    assert seqs_a == seqs_b, "sequence name/order mismatch"
    n = len(seqs_a)
    print("sequences: %d" % n)

    # sanity check: reproduced overall values should match eval_history.csv
    print("\n== sanity (reproduced overall) ==")
    print("  %-32s PR=%.4f SR=%.4f" % (args.tag_a, prv_a, srv_a))
    print("  %-32s PR=%.4f SR=%.4f" % (args.tag_b, prv_b, srv_b))

    for name, va, vb in (("PR", pr_a, pr_b), ("SR", sr_a, sr_b)):
        diff = vb - va
        wins = int((diff > 1e-9).sum())
        losses = int((diff < -1e-9).sum())
        ties = n - wins - losses
        means, lo, hi, p_less, p_two = bootstrap_paired(diff)
        effect = diff.mean() / (diff.std() + 1e-12)
        print("\n== %s: %s -> %s ==" % (name, args.tag_a, args.tag_b))
        print("  seq-mean diff (B-A) : %+.4f  (%.4f -> %.4f)" % (diff.mean(), va.mean(), vb.mean()))
        print("  per-seq diff std    : %.4f" % diff.std())
        print("  effect size (d/sd)  : %+.2f" % effect)
        print("  wins/losses/ties    : %d / %d / %d" % (wins, losses, ties))
        print("  95%% bootstrap CI    : [%+.4f, %+.4f]" % (lo, hi))
        print("  p(diff<=0) one-sided: %.4f" % p_less)
        print("  p two-sided         : %.4f" % p_two)
        sig = "SIGNIFICANT (p<0.05)" if p_two < 0.05 else "NOT significant (p>=0.05)"
        print("  verdict             : %s" % sig)

        # top sequences where B improves / loses the most
        order = np.argsort(diff)
        print("  top-5 improve: " + ", ".join(
            "%s(%+.2f)" % (seqs_a[i], diff[i]) for i in order[::-1][:5]))
        print("  top-5 degrade: " + ", ".join(
            "%s(%+.2f)" % (seqs_a[i], diff[i]) for i in order[:5]))


if __name__ == "__main__":
    main()
