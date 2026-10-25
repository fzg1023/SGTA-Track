#!/usr/bin/env python
"""Measure the Liquid update gate at test time: is it ever closed?

WHY THIS EXISTS
---------------
The v1 ablation (``lasher_ablation_a0..a4``) made ``a3`` look like the row
where "Liquid helps the most" (mean PR 0.7388 vs 0.6995 for a2 and 0.7076 for
a4).  But ``a3`` was **the only row in that batch with
``ENABLE_TEMPLATE_UPDATE: True``**, so "Liquid helps" and "dynamic template
update helps" are perfectly collinear and cannot be separated.

The gate is the ONLY Liquid output consumed at test time
(``lib/test/tracker/bat.py:276-286``)::

    template_update_enabled = sgta_enabled and ENABLE_TEMPLATE_UPDATE
    liquid_allows_update = template_update_enabled and (
        liquid_gate is None or liquid_gate >= UPDATE_GATE_THRESHOLD)

If ``gamma >= threshold`` on essentially every frame, the gate is a no-op and
behaves exactly like the unconditioned "update every 10 frames" baseline, which
would prove the v1 gain did not come from the gate.  If gamma is bimodal with
meaningful mass below the threshold, the gate is doing real work.

WHAT IS RECORDED (per frame, single process, no multiprocessing)
---------------------------------------------------------------
* ``update_gate`` -- row 0, i.e. the exact scalar the tracker consumes
* ``modal_gate`` / ``fuse_gate`` / ``motion_scale`` -- unused at test time
* ``relative_score`` (2) -- TemplateRouter output, used in the promotion test
* ``best_score_2/3`` -- the head scores compared on a 4-way forward
* ``new_template_ready`` / ``updated`` -- was a candidate available / promoted
* ``is_decision_frame`` -- ``frame_id % interval == 1`` (interval defaults to 10)

Usage::

    python tools/diag_liquid_gate.py \
        --yaml lasher_train \
        --checkpoint /home/fzg/AQA/UATrack-main/output/checkpoints/train/bat/lasher_train/BATrack_ep0014.pth.tar \
        --epoch 14 --n 20

    python tools/diag_liquid_gate.py --yaml lasher_train \
        --checkpoint output/checkpoints/train/bat/lasher_train/BATrack_ep0014.pth.tar \
        --epoch 14 --n 20
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
from typing import Dict, List

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def load_test_sgta():
    spec = importlib.util.spec_from_file_location(
        "test_sgta", os.path.join(REPO, "RGBT_workspace", "test_sgta.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_sgta_cfg(yaml_name: str) -> Dict[str, object]:
    """Read MODEL.SGTA.* from the experiment yaml (defaults match config.py)."""
    out = {"ENABLE_TEMPLATE_UPDATE": True, "UPDATE_GATE_THRESHOLD": 0.5,
           "USE_LIQUID_CONTROLLER": False, "USE_TEMPLATE_ROUTER": True}
    path = os.path.join(REPO, "experiments", "bat", yaml_name + ".yaml")
    if not os.path.isfile(path):
        print(f"[WARN] yaml not found: {path} (using defaults)")
        return out
    try:
        import yaml as _yaml
        with open(path) as fh:
            doc = _yaml.safe_load(fh) or {}
    except Exception as exc:                                     # noqa: BLE001
        print(f"[WARN] could not parse {path}: {exc}")
        return out
    sgta = ((doc.get("MODEL") or {}).get("SGTA")) or {}
    for key in list(out):
        if key in sgta:
            out[key] = sgta[key]
    out["_path"] = path
    return out


def find_module(net, class_name: str):
    for name, mod in net.named_modules():
        if mod.__class__.__name__ == class_name:
            return name, mod
    return None, None


def pct(values, q):
    return float(np.percentile(values, q)) if len(values) else float("nan")


def ascii_hist(values, bins=10, width=48):
    counts, edges = np.histogram(values, bins=np.linspace(0.0, 1.0, bins + 1))
    top = max(int(counts.max()), 1)
    lines = []
    for i, c in enumerate(counts):
        bar = "#" * int(round(width * c / top))
        lines.append("  [%.1f, %.1f) %8d %s" % (edges[i], edges[i + 1], c, bar))
    return "\n".join(lines)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser("Liquid update-gate diagnostics")
    ap.add_argument("--yaml", default="lasher_train")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--epoch", type=int, default=14)
    ap.add_argument("--sequence", default="",
                    help="comma separated sequence names; default = auto subset")
    ap.add_argument("--select", default="first",
                    choices=["first", "longest", "shortest", "random"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--max-frames", type=int, default=12000)
    ap.add_argument("--min-seq-frames", type=int, default=100)
    ap.add_argument("--max-seq-frames", type=int, default=800)
    ap.add_argument("--threshold", type=float, default=None,
                    help="override UPDATE_GATE_THRESHOLD from the yaml")
    ap.add_argument("--lasher-root", default="/home/fzg/data/lasher/testingset")
    ap.add_argument("--out-dir", default=os.path.join(REPO, "RGBT_workspace", "diag"))
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    os.environ.setdefault("LASHER_TEST_ROOT", args.lasher_root)
    os.environ.setdefault("RGBT_DATA_ROOT", args.lasher_root)

    tl = load_test_sgta()
    import lib.test.parameter.bat as bat_params
    from lib.test.tracker.bat import BATTrack

    sgta_cfg = read_sgta_cfg(args.yaml)
    threshold = args.threshold if args.threshold is not None else float(
        sgta_cfg["UPDATE_GATE_THRESHOLD"])
    update_enabled = bool(sgta_cfg["ENABLE_TEMPLATE_UPDATE"])

    # ---------------- sequence subset ----------------
    cfg = tl.DATASET_CFG["lasher"]
    all_seqs = tl.sequence_dirs("lasher")
    if args.sequence:
        wanted = {s.strip() for s in args.sequence.split(",") if s.strip()}
        all_seqs = [p for p in all_seqs if os.path.basename(p) in wanted]
    else:
        counts = np.array([tl.count_frames(p, "lasher") for p in all_seqs])
        keep = np.ones(len(all_seqs), dtype=bool)
        if args.min_seq_frames > 0:
            keep &= counts >= args.min_seq_frames
        if args.max_seq_frames > 0:
            keep &= counts <= args.max_seq_frames
        pool = np.where(keep)[0]
        if pool.size == 0:
            raise SystemExit("no sequence left after length filtering")
        if args.select == "longest":
            order = pool[np.argsort(counts[pool])[::-1]]
        elif args.select == "shortest":
            order = pool[np.argsort(counts[pool])]
        elif args.select == "random":
            order = np.random.RandomState(args.seed).permutation(pool)
        else:
            order = pool
        chosen, total = [], 0
        for idx in order:
            if len(chosen) >= args.n or total >= args.max_frames:
                break
            chosen.append(all_seqs[idx])
            total += int(counts[idx])
        all_seqs = chosen
    if not all_seqs:
        raise SystemExit("no sequence selected")

    print("=" * 86)
    print(f"checkpoint       : {args.checkpoint}")
    print(f"yaml             : {args.yaml}  ({sgta_cfg.get('_path', '?')})")
    print(f"ENABLE_TEMPLATE_UPDATE : {update_enabled}")
    print(f"UPDATE_GATE_THRESHOLD  : {threshold}")
    print(f"USE_LIQUID_CONTROLLER  : {sgta_cfg['USE_LIQUID_CONTROLLER']}")
    print(f"sequences        : {len(all_seqs)}  frames="
          f"{sum(tl.count_frames(p, 'lasher') for p in all_seqs)}")
    print("=" * 86, flush=True)

    # ---------------- build tracker ----------------
    params = bat_params.parameters(args.yaml, args.epoch)
    params.checkpoint = args.checkpoint
    tracker = BATTrack(params)
    net = tracker.network
    interval = int(getattr(tracker, "interval", 10))

    liquid_name, liquid = find_module(net, "LiquidTemplateUpdateController")
    if liquid is None:
        print("[FATAL] no LiquidTemplateUpdateController in the model -- "
              "this checkpoint cannot answer the question")
        return 3
    print(f"liquid module    : {liquid_name}")
    print(f"update interval  : every {interval} frames "
          f"(decision frames: frame_id %% {interval} == 1, frame_id != 1)")

    # ---------------- instrumentation ----------------
    # NOTE: ``nn.Module.__call__`` forwards kwargs straight to the assigned
    # callable, so the replacement must accept the exact signature the caller
    # uses: ``self.adap_liquid_controller(features, state=...)``.
    rec: Dict[str, List[dict]] = {"rows": []}
    run = {"seq": "", "frame": 0, "on": False}

    orig_liquid_forward = liquid.forward

    def liquid_forward(features, state=None, dt=1.0):
        out = orig_liquid_forward(features, state=state, dt=dt)
        if run["on"]:
            rec["rows"].append(dict(
                seq=run["seq"], frame=run["frame"],
                update_gate=float(out["update_gate"][0, 0]),
                update_gate_row1=(float(out["update_gate"][1, 0])
                                  if out["update_gate"].shape[0] > 1 else float("nan")),
                batch=int(out["update_gate"].shape[0]),
                modal_0=float(out["modal_gate"][0, 0]),
                modal_1=float(out["modal_gate"][0, 1]),
                fuse_gate=float(out["fuse_gate"][0, 0]),
                motion_scale=float(out["motion_scale"][0, 0]),
            ))
        return out

    liquid.forward = liquid_forward

    orig_net_forward = net.forward
    extra = {"row": None}

    def net_forward(*fa, **fk):
        out = orig_net_forward(*fa, **fk)
        if run["on"] and isinstance(out, dict):
            rs = out.get("relative_score")
            extra["row"] = dict(
                rel_0=(float(rs[0, 0]) if rs is not None and rs.shape[0] > 0 else float("nan")),
                rel_1=(float(rs[0, 1]) if rs is not None and rs.shape[1] > 1 else float("nan")),
                has_router=int(rs is not None),
            )
        return out

    net.forward = net_forward

    orig_track = tracker.track

    def track(image):
        prev_last = getattr(tracker, "last_template", None)
        candidate_before = getattr(tracker, "new_template", None)
        out = orig_track(image)
        updated = getattr(tracker, "last_template", None) is not prev_last
        if run["on"]:
            row = rec["rows"][-1] if rec["rows"] else None
            if row is not None and row["frame"] == run["frame"]:
                row.update(extra["row"] or dict(rel_0=float("nan"), rel_1=float("nan"),
                                               has_router=0))
                row["is_decision_frame"] = int(
                    run["frame"] % interval == 1 and run["frame"] != 1)
                row["new_template_ready"] = int(candidate_before is not None)
                row["updated"] = int(updated)
        return out

    tracker.track = track

    # ---------------- rollout ----------------
    seq_summary: List[dict] = []
    try:
        for seq_dir in all_seqs:
            seq_name = os.path.basename(seq_dir)
            rgb = tl.list_frames(seq_dir, cfg["rgb"])
            tir = tl.list_frames(seq_dir, cfg["tir"])
            gts = tl.read_boxes(os.path.join(seq_dir, cfg["gt"]))
            n = min(len(rgb), len(tir), len(gts))
            if n <= 1:
                continue
            run.update(seq=seq_name, frame=0, on=True)
            before = len(rec["rows"])
            image0 = tl.read_rgbt(rgb[0], tir[0])
            tracker.initialize(image0, {"init_bbox": list(np.asarray(gts[0], np.float32))})
            preds = [list(gts[0])]
            for idx in range(1, n):
                run["frame"] = idx
                image = tl.read_rgbt(rgb[idx], tir[idx])
                preds.append(list(tracker.track(image)["target_bbox"]))
            m = tl.compute_metrics(preds, gts[:n], None, cfg)
            rows = rec["rows"][before:]
            gammas = [r["update_gate"] for r in rows]
            dec = [r for r in rows if r.get("is_decision_frame")]
            seq_summary.append(dict(
                seq=seq_name, frames=n, PR=float(m["PR"]), SS=float(m["SS"]),
                gamma_mean=float(np.mean(gammas)) if gammas else float("nan"),
                frac_open=float(np.mean([g >= threshold for g in gammas])) if gammas else float("nan"),
                decision_frames=len(dec),
                decision_open=sum(1 for r in dec if r["update_gate"] >= threshold),
                updates=sum(r.get("updated", 0) for r in rows),
            ))
            s = seq_summary[-1]
            print(f"[seq] {seq_name:<24} frames={n:<5} PR={m['PR']:.4f} "
                  f"gamma_mean={s['gamma_mean']:.4f} open={s['frac_open']:.3f} "
                  f"dec={s['decision_frames']:<3} open={s['decision_open']:<3} "
                  f"updates={s['updates']}", flush=True)
    finally:
        liquid.forward = orig_liquid_forward
        net.forward = orig_net_forward
        tracker.track = orig_track

    # ---------------- aggregate ----------------
    rows = [r for r in rec["rows"] if "is_decision_frame" in r]
    gammas = np.array([r["update_gate"] for r in rows])
    dec = [r for r in rows if r["is_decision_frame"]]
    dec_open = [r for r in dec if r["update_gate"] >= threshold]
    summary = dict(
        checkpoint=args.checkpoint, yaml=args.yaml, epoch=args.epoch,
        update_enabled=update_enabled, threshold=threshold, interval=interval,
        n_sequences=len(seq_summary),
        n_frames=int(len(rows)),
        n_decision_frames=int(len(dec)),
        gamma=dict(mean=float(gammas.mean()) if len(gammas) else float("nan"),
                   std=float(gammas.std()) if len(gammas) else float("nan"),
                   median=pct(gammas, 50), p01=pct(gammas, 1), p05=pct(gammas, 5),
                   p25=pct(gammas, 25), p75=pct(gammas, 75), p95=pct(gammas, 95),
                   p99=pct(gammas, 99),
                   min=float(gammas.min()) if len(gammas) else float("nan"),
                   max=float(gammas.max()) if len(gammas) else float("nan")),
        frac_open_all_frames=float(np.mean([r["update_gate"] >= threshold for r in rows]))
        if rows else float("nan"),
        frac_open_decision_frames=float(len(dec_open) / len(dec)) if dec else float("nan"),
        decision_frames_blocked=int(len(dec) - len(dec_open)),
        updates_actual=int(sum(r.get("updated", 0) for r in rows)),
        updates_blocked_by_quality=int(
            len(dec_open) - sum(r.get("updated", 0) for r in dec_open)),
        per_sequence=seq_summary,
    )

    print("-" * 86)
    g = summary["gamma"]
    print(f"frames                : {summary['n_frames']}  "
          f"(decision frames: {summary['n_decision_frames']})")
    print(f"gamma  mean/median    : {g['mean']:.4f} / {g['median']:.4f}   "
          f"std={g['std']:.4f}  min={g['min']:.4f}  max={g['max']:.4f}")
    print(f"gamma  p01/p05/p25/p75/p95/p99 : "
          f"{g['p01']:.4f} / {g['p05']:.4f} / {g['p25']:.4f} / "
          f"{g['p75']:.4f} / {g['p95']:.4f} / {g['p99']:.4f}")
    print(f"gamma >= {threshold}: all frames {summary['frac_open_all_frames']:.4f}   "
          f"decision frames {len(dec_open)}/{len(dec)}   "
          f"blocked by gate: {summary['decision_frames_blocked']}")
    print(f"template updates actually performed: {summary['updates_actual']}   "
          f"(decision frames where the gate allowed an update: {len(dec_open)}, "
          f"of which {summary['updates_blocked_by_quality']} were still declined by "
          f"the Router/score test)")
    print("gamma histogram (all frames):")
    print(ascii_hist(gammas) if len(gammas) else "  (no data)")

    verdict = ("GATE IS A NO-OP (open on >=99% of frames) -> behaves like the "
               "unconditioned 'update every N frames' baseline"
               if summary["frac_open_all_frames"] >= 0.99 else
               "gate is ACTIVE (blocks a non-trivial fraction of frames)"
               if summary["frac_open_all_frames"] <= 0.95 else
               "gate is mostly open (>=95%) -> little discriminative effect")
    print(f"\nVERDICT: {verdict}")

    os.makedirs(args.out_dir, exist_ok=True)
    tag = args.tag or f"liquid_gate_{args.yaml}_ep{args.epoch:04d}"
    csv_path = os.path.join(args.out_dir, tag + "_per_frame.csv")
    with open(csv_path, "w", newline="") as fh:
        fields = ["seq", "frame", "update_gate", "update_gate_row1", "batch",
                  "modal_0", "modal_1", "fuse_gate", "motion_scale",
                  "rel_0", "rel_1", "has_router",
                  "is_decision_frame", "new_template_ready", "updated"]
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    json_path = os.path.join(args.out_dir, tag + "_summary.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nwritten: {csv_path}\nwritten: {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
