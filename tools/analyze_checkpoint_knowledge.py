"""Analyze which SGTA modules actually learned knowledge in a checkpoint.

Compares two checkpoints (net state dicts) tensor-wise and groups the
changes by module. Also inspects optimizer momentum (exp_avg) of the
later checkpoint to see which parameters were still being updated.

Usage:
    python tools/analyze_checkpoint_knowledge.py \
        --baseline .../BATrack_ep0001.pth.tar \
        --target    .../BATrack_ep0010.pth.tar \
        [--third .../BATrack_ep0012.pth.tar]  # compared against --baseline too
"""
import argparse
from collections import defaultdict

import torch

GROUPS = [
    ("UGMFusion(空间Mamba融合)", ".adap_fusion."),
    ("TemporalMambaMem(时序记忆)", ".adap_predictor."),
    ("LiquidCtrl(液态门控)", ".adap_liquid_controller."),
    ("MoERouter(融合路由)", ".adap_moe."),
    ("TemplateRouter", ".templaterouter."),
    ("BoxHead", "box_head."),
    ("OtherAdap(Bi-adapter等)", "adap"),
    ("PosEmbed", "pos_embed"),
    ("BackboneRest", ""),
]


def group_of(name):
    for grp, pat in GROUPS:
        if pat and pat in name:
            return grp
    return "BackboneRest"


def net_tensors(ck):
    net = ck.get("net", None)
    if net is None:
        for k in ("model", "state_dict"):
            if k in ck:
                net = ck[k]
                break
    if net is None:
        raise KeyError("no 'net' in checkpoint; top keys: %s" % list(ck.keys())[:10])
    if hasattr(net, "items"):
        return {k: v for k, v in net.items() if torch.is_tensor(v)}
    return {}


def summarize(tensors, label):
    """per-group: n, elems, ||w||_2"""
    g = defaultdict(lambda: {"n": 0, "elems": 0, "l2": 0.0})
    for k, v in tensors.items():
        w = v.float()
        r = g[group_of(k)]
        r["n"] += 1
        r["elems"] += w.numel()
        r["l2"] += float(w.pow(2).sum())
    print("== %s 规模 ==" % label)
    for grp, _ in GROUPS:
        if grp not in g:
            continue
        r = g[grp]
        print("  %-30s tensors=%-4d elems=%-10d ||w||2=%.3f" % (grp, r["n"], r["elems"], r["l2"] ** 0.5))
    return g


def compare(tensors_a, tensors_b, label_a, label_b):
    common = set(tensors_a) & set(tensors_b)
    only_b = sorted(set(tensors_b) - set(tensors_a))
    g = defaultdict(lambda: {
        "n": 0, "elems": 0, "l2a": 0.0, "diff2": 0.0,
        "mean_abs": 0.0, "frozen": 0, "top": [],
    })
    for k in sorted(common):
        va = tensors_a[k].float()
        vb = tensors_b[k].float()
        if va.shape != vb.shape:
            continue
        d = vb - va
        grp = group_of(k)
        r = g[grp]
        r["n"] += 1
        r["elems"] += vb.numel()
        r["l2a"] += float(va.pow(2).sum())
        r["diff2"] += float(d.pow(2).sum())
        ma = float(d.abs().sum()) / d.numel()
        r["mean_abs"] += float(d.abs().sum())
        if ma < 1e-8:
            r["frozen"] += 1
        short = k.split(".")[-1]
        r["top"].append((ma, short, k))
    print("== %s -> %s 参数变化 (mean|d|, 相对L2变化) ==" % (label_a, label_b))
    for grp, _ in GROUPS:
        if grp not in g:
            continue
        r = g[grp]
        rel = (r["diff2"] ** 0.5) / (r["l2a"] ** 0.5 + 1e-12)
        print("  %-30s tensors=%-4d mean|d|=%-10.3g rel=%-8.3g frozen_tensors=%d/%d"
              % (grp, r["n"], r["mean_abs"] / max(r["elems"], 1), rel, r["frozen"], r["n"]))
        for ma, short, k in sorted(r["top"], reverse=True)[:3]:
            print("       top-changed: %-24s mean|d|=%.3g" % (short, ma))
    if only_b:
        print("  only-in-%s keys: %d (e.g. %s)" % (label_b, len(only_b), only_b[:3]))
    return g


def optimizer_activity(ck):
    opt = ck.get("optimizer", None)
    if opt is None:
        print("== optimizer: not found ==")
        return
    state = opt.get("state", {})
    if not state:
        print("== optimizer: no state ==")
        return
    steps = sorted({int(v.get("step", 0)) for v in state.values()})
    acts = []
    for v in state.values():
        ea = v.get("exp_avg", None)
        if ea is None:
            continue
        ea = ea.float()
        acts.append(float(ea.abs().mean()))
    acts.sort(reverse=True)
    n = len(acts)
    print("== optimizer state ==")
    print("  param_states=%d  steps=%s" % (len(state), steps[-5:] if steps else []))
    print("  exp_avg mean|.| over params: max=%.3g top10%%=%.3g median=%.3g min=%.3g"
          % (acts[0], acts[n // 10], acts[n // 2], acts[-1]))
    zero = sum(1 for a in acts if a < 1e-12)
    print("  params with ~zero momentum: %d/%d" % (zero, n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--third", default=None)
    args = ap.parse_args()

    print("loading baseline:", args.baseline)
    ck_a = torch.load(args.baseline, map_location="cpu")
    print("top-level keys:", sorted(ck_a.keys()))
    tensors_a = net_tensors(ck_a)
    summary_a = summarize(tensors_a, "baseline")
    del ck_a

    print("loading target:", args.target)
    ck_b = torch.load(args.target, map_location="cpu")
    tensors_b = net_tensors(ck_b)
    summarize(tensors_b, "target")
    compare(tensors_a, tensors_b, "baseline", "target")
    optimizer_activity(ck_b)
    del ck_b

    if args.third:
        print("loading third:", args.third)
        ck_c = torch.load(args.third, map_location="cpu")
        tensors_c = net_tensors(ck_c)
        compare(tensors_a, tensors_c, "baseline", "third")
        del ck_c


if __name__ == "__main__":
    main()
