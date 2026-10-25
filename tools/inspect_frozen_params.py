"""Inspect frozen / dead parameters in a checkpoint pair.

Prints the FULL key names of tensors whose change (baseline -> target)
is ~zero, and (approximately, via shape matching) which optimizer states
have ~zero momentum in the target checkpoint.
"""
import argparse
from collections import defaultdict

import torch

EPS = 1e-8


def net_tensors(ck):
    net = ck.get("net", None)
    if net is None:
        raise KeyError("no net in checkpoint: %s" % list(ck.keys())[:5])
    return {k: v for k, v in net.items() if torch.is_tensor(v)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--target", required=True)
    args = ap.parse_args()

    print("loading baseline ...")
    ck_a = torch.load(args.baseline, map_location="cpu")
    ta = net_tensors(ck_a)
    opt_a = ck_a.get("optimizer", {}).get("state", {})
    del ck_a

    print("loading target ...")
    ck_b = torch.load(args.target, map_location="cpu")
    tb = net_tensors(ck_b)
    opt_b = ck_b.get("optimizer", {}).get("state", {})
    del ck_b

    print("\n== tensors with ~zero change (baseline -> target) ==")
    n_frozen = 0
    for k in sorted(set(ta) & set(tb)):
        if ta[k].shape != tb[k].shape:
            continue
        d = (tb[k].float() - ta[k].float()).abs().mean().item()
        if d < EPS:
            n_frozen += 1
            print("  [%s] %s" % ("buf?" if ("running" in k or "num_batches" in k) else "prm ", k))
    print("total frozen tensors:", n_frozen)

    # ---- approximate zero-momentum param names via shape matching ----
    print("\n== optimizer states with ~zero exp_avg (target), approx name ==")
    # net tensor order: follow insertion order of state_dict
    net_items = [(k, v) for k, v in tb.items() if v.requires_grad]
    zero_states = []
    for sid, st in opt_b.items():
        ea = st.get("exp_avg", None)
        if ea is None or float(ea.float().abs().mean()) >= 1e-12:
            continue
        zero_states.append((tuple(ea.shape), sid))
    used = set()
    for shape, sid in zero_states:
        hit = None
        for i, (k, v) in enumerate(net_items):
            if i in used:
                continue
            if tuple(v.shape) == shape:
                hit = (i, k)
                break
        if hit is not None:
            used.add(hit[0])
            print("  ~zero momentum: %s" % hit[1])
        else:
            print("  ~zero momentum: <shape %s unmatched>" % (shape,))
    print("total zero-momentum states: %d / %d" % (len(zero_states), len(opt_b)))


if __name__ == "__main__":
    main()
