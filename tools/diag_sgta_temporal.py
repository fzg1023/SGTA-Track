#!/usr/bin/env python
"""Diagnostics for the SGTA temporal branch.

Two independent questions are answered here:

1. ``drift``  -- was the temporal branch actually optimised during training?
   Compares a trained checkpoint against the DropTrack initialization.  Needs
   only ``torch``; no dataset, no GPU, no ``mamba_ssm``.

2. ``run`` -- does the temporal branch change the tracker output at all?
   Rolls the *unmodified tracker* over LasHeR sequences and, at a sample of
   frames, performs an input-matched repeat of the network forward pass with a
   single intervention applied.  Two controls make the result interpretable:

   * ``--repeat-control`` runs the identical forward twice without touching
     anything, giving the numerical reproducibility floor (cuDNN algorithm
     selection is not bit-exact even with identical inputs);
   * every intervention is applied to a *second* forward pass whose inputs are
     byte-identical to the baseline pass, and the rollout keeps following the
     baseline output, so the measured delta is a pure per-frame influence
     measure with no trajectory drift.

   The module-level probe additionally answers, for the predictor alone,
   ``cos(pred_with_state, pred_with_zero_state)``.  A value of ~1 means the
   recurrent state does not influence the prediction at all.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import re
import sys
from collections import OrderedDict
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)


# --------------------------------------------------------------------------
# part 1: parameter drift
# --------------------------------------------------------------------------

GROUPS = OrderedDict(
    [
        ("temporal_predictor", r"^adap_predictor\."),
        ("liquid_controller", r"^adap_liquid_controller\."),
        ("template_router", r"^templaterouter\."),
        ("moe", r"^adap_moe\."),
        ("fusion_blocks_9_11", r"^blocks\.(9|10|11)\."),
        ("blocks_0_8", r"^blocks\.[0-8]\."),
        ("patch_embed", r"^patch_embed\."),
        ("head", r"^(head|box_head)\."),
        ("other", r".*"),
    ]
)


def normalize_key(key: str) -> str:
    for prefix in ("backbone.", "module."):
        if key.startswith(prefix):
            return key[len(prefix):]
    return key


def load_state(path: str) -> Dict[str, torch.Tensor]:
    obj = torch.load(path, map_location="cpu")
    for wrapper in ("net", "model", "state_dict"):
        if isinstance(obj, dict) and isinstance(obj.get(wrapper), dict):
            return obj[wrapper]
    return obj


def group_of(key: str) -> str:
    for name, pattern in GROUPS.items():
        if re.match(pattern, key):
            return name
    return "other"


def cmd_drift(args) -> None:
    init = load_state(args.init)
    print(f"init : {args.init}  ({len(init)} tensors)")

    if args.inventory:
        seen = OrderedDict()
        for key in sorted(init):
            name = normalize_key(key).split(".")[0]
            seen.setdefault(name, 0)
            seen[name] += 1
        print("  top-level modules in init:")
        for name, count in seen.items():
            print(f"    {count:>5}  {name}")

    for ckpt_path in args.ckpt:
        trained = load_state(ckpt_path)
        print("\n" + "=" * 82)
        print(f"checkpoint: {ckpt_path}  ({len(trained)} tensors)")

        if args.inventory:
            touched = sorted({normalize_key(k) for k in trained} - {normalize_key(k) for k in init})
            print(f"  tensors absent from init ({len(touched)}), grouped:")
            counts: Dict[str, int] = {}
            for key in touched:
                counts[key.split(".")[0]] = counts.get(key.split(".")[0], 0) + 1
            for name, count in sorted(counts.items()):
                print(f"    {count:>5}  {name}")

        stats: Dict[str, list] = {name: [] for name in GROUPS}
        for key, w in trained.items():
            nk = normalize_key(key)
            w0 = None
            for candidate in (key, nk, "backbone." + nk, "module." + nk):
                if candidate in init:
                    w0 = init[candidate]
                    break
            if w0 is None or not torch.is_tensor(w) or w.shape != w0.shape:
                continue
            if w.is_floating_point():
                diff = w.float() - w0.float()
                stats[group_of(nk)].append(
                    (
                        diff.abs().mean().item(),
                        diff.pow(2).sum().item(),
                        w0.float().pow(2).sum().item(),
                        w0.abs().mean().item(),
                        int(diff.abs().max().item() == 0.0),
                    )
                )

        print(f"  {'group':<22}{'#tens':>6}{'#exact0':>8}{'mean|dW|':>11}"
              f"{'mean|W0|':>11}{'rel.drift':>11}{'gain':>9}")
        for name, rows in stats.items():
            if not rows or (args.only and not re.search(args.only, name)):
                continue
            n = len(rows)
            mean_diff = sum(r[0] for r in rows) / n
            mean_abs0 = sum(r[3] for r in rows) / n
            gain = (sum(r[1] for r in rows) ** 0.5) / (sum(r[2] for r in rows) ** 0.5 + 1e-12)
            print(f"  {name:<22}{n:>6}{sum(r[4] for r in rows):>8}{mean_diff:>11.3e}"
                  f"{mean_abs0:>11.3e}{mean_diff / (mean_abs0 + 1e-12):>11.4f}{gain:>9.4f}")

        if args.verbose:
            print("\n  per-tensor detail:")
            for key in sorted(trained):
                nk = normalize_key(key)
                if args.only and not re.search(args.only, nk):
                    continue
                w0 = init.get(key, init.get(nk))
                if w0 is None or not torch.is_tensor(w0) or w0.shape != trained[key].shape:
                    continue
                if trained[key].is_floating_point():
                    d = (trained[key].float() - w0.float()).abs().mean().item()
                    rel = d / (w0.float().abs().mean().item() + 1e-12)
                    print(f"    {nk:<58}{rel:>12.4f}")


# --------------------------------------------------------------------------
# part 2: rollout with interventions
# --------------------------------------------------------------------------

MODE_ORDER = (
    "none",          # baseline, no modification
    "repeat",        # control: identical forward, twice (numerical floor)
    "pred_zero",     # zero the predicted template tokens
    "pred_copy_t2",  # replace the prediction by an existing template
    "pred_noise",    # add Gaussian noise to the prediction
    "state_zero",    # zero the recurrent memory state
    "state_random",  # replace the recurrent memory state by noise
    "state_freeze",  # keep the real prediction, freeze the state write-back
    "t1_zero",       # zero the first template tokens (position 64..127)
    "t2_zero",       # zero the read-out template tokens (position 128..191)
    "D_zero",        # remove the per-position skip path of the SSM
    "swap_t1t2",     # swap the two template tokens fed to the predictor
    "no_predictor",  # bypass the temporal predictor entirely
)

# Several primitives can be combined with "+", e.g. "D_zero+t1_zero", to test
# whether the skip path is what masks the recurrent state.
MODES = set(MODE_ORDER)

STATE: Dict[str, object] = {
    "mode": "none",
    "active": True,
    "rows": [],
    "flush_every": 200,
}


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.flatten(1), b.flatten(1), dim=-1).mean())


SKIP_KEYS = {"kind", "mode", "seq"}


def _summarize(rows: List[dict]) -> dict:
    """Mean of every numeric column, plus the sample count."""
    if not rows:
        return {}
    out: dict = {"n": len(rows)}
    for key in rows[0]:
        if key in SKIP_KEYS:
            continue
        values = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        if values:
            # the first row of a running statistic is undefined by construction
            out[key] = float(np.nanmean(np.asarray(values, dtype=float)))
    return out


def _write_csv(path: str, rows: List[dict]) -> None:
    if not rows:
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def install_predictor_patch(module) -> None:
    original = module.TemporalMambaMemory.forward

    def patched(self, t1, t2, target=None, state=None, train=True):
        raw_mode = STATE["mode"] if STATE["active"] else "none"
        mode_arg = "none" if raw_mode == "repeat" else raw_mode
        mods = {part for part in str(mode_arg).split("+") if part} or {"none"}

        handles = []
        if STATE.get("probe"):
            # capture intermediate activations so the variation killed by each
            # layer can be attributed to the SSM or to the read-out head
            STATE["captured"] = {}

            def _capture(name, tokens=None):
                def hook(module, inputs, output):
                    tensor = output[0] if isinstance(output, tuple) else output
                    if torch.is_tensor(tensor) and tensor.dim() == 3 and tokens:
                        tensor = tensor[:, -tokens:]
                    if torch.is_tensor(tensor):
                        STATE["captured"][name] = tensor.detach()
                return hook

            handles.append(self.mamba.register_forward_hook(_capture("mamba_out", 64)))
            handles.append(self.pred.register_forward_hook(_capture("head_out")))
        try:
            return _patched_forward(self, original, mods, mode_arg, t1, t2, target, state, train)
        finally:
            for handle in handles:
                handle.remove()

    def _patched_forward(self, original, mods, mode_arg, t1, t2, target, state, train):
        # Intrinsic module diagnostics are always measured on an *unmodified*
        # forward pass, so that the numbers stay comparable across
        # interventions instead of being polluted by the intervention itself.
        if STATE.get("probe") and state is not None:
            with torch.no_grad():
                pred_ref, _, _ = original(self, t1, t2, target=None, state=state, train=False)
                pred_ref_zero, _, _ = original(
                    self, t1, t2, target=None, state=torch.zeros_like(state), train=False)
            copy_t2 = F.normalize(t2, dim=-1)
            # Running statistics answer "is the prediction basically a
            # constant?".  cos(pred_t, mean of all previous preds) ~ 1 means
            # the module emits the same template token no matter what the
            # inputs are, which would explain every other observation.
            flat = pred_ref.flatten(1)
            count = STATE.get("pred_count", 0)
            total = STATE.get("pred_sum")
            cos_to_mean = float("nan")
            if count > 0 and total is not None:
                mean = F.normalize(total / count, dim=-1)
                cos_to_mean = float(F.cosine_similarity(flat, mean, dim=-1).mean())
            STATE["pred_sum"] = flat.sum(dim=0) if total is None else total + flat.sum(dim=0)
            STATE["pred_count"] = count + 1
            # Same running-mean cosine for the *inputs*: if t1/t2/state are
            # themselves nearly constant on LasHeR, then a constant prediction
            # is an input property rather than a module defect.
            input_cos = {}

            def _running_cos(name, tensor):
                """cos(tensor, running mean of previous tensors), per token.

                Reshaped to (n_tokens, D) so that a changing batch size or an
                intermediate layer with more tokens cannot break the running
                sum.
                """
                tall = tensor.reshape(-1, tensor.shape[-1]).float()
                count = STATE.get(f"{name}_count", 0)
                total = STATE.get(f"{name}_sum")
                value = float("nan")
                if count > 0 and total is not None:
                    value = float(F.cosine_similarity(
                        tall, F.normalize(total / count, dim=-1), dim=-1).mean())
                STATE[f"{name}_sum"] = tall.sum(dim=0) if total is None else total + tall.sum(dim=0)
                STATE[f"{name}_count"] = count + 1
                return value

            for name, tensor in (("t1", t1), ("t2", t2), ("state", state)):
                input_cos[f"cos_{name}_running_mean"] = _running_cos(name, tensor)
            # Layers captured by the forward hooks installed in install_hooks():
            # isolates whether the SSM or the read-out head washes out the
            # per-frame variation.
            for name, tensor in list(STATE.get("captured", {}).items()):
                if tensor is not None:
                    input_cos[f"cos_{name}_running_mean"] = _running_cos(name, tensor)
            STATE["rows"].append(dict(
                kind="predictor",
                mode=mode_arg,
                cos_pred_running_mean=cos_to_mean,
                **input_cos,
                # per-token cosine: the flattened version is dominated by the
                # per-token norm spread and is not interpretable here
                cos_pred_t2=float(F.cosine_similarity(pred_ref, t2, dim=-1).mean()),
                cos_pred_copy_t2=float(F.cosine_similarity(pred_ref, copy_t2, dim=-1).mean()),
                # how far the prediction is from simply copying an existing
                # template; ~0 would mean the Mamba head contributes nothing
                rel_dev_from_copy=float(
                    (pred_ref - copy_t2).norm() / (copy_t2.norm() + 1e-12)),
                # influence of the recurrent state on the prediction itself
                cos_pred_zerostate=_cos(pred_ref, pred_ref_zero),
                rel_pred_zerostate=float(
                    (pred_ref - pred_ref_zero).norm() / (pred_ref.norm() + 1e-12)),
                pred_norm=float(pred_ref.norm()),
                t2_norm=float(t2.norm()),
                t2_norm_per_token=float(t2.norm(dim=-1).mean()),
                state_norm=float(state.norm()),
                state_norm_per_token=float(state.norm(dim=-1).mean()),
                has_state=1,
            ))

        if "no_predictor" in mods:
            return torch.zeros_like(t2), t2.new_tensor(0.0), (
                state if state is not None else torch.zeros_like(t2))

        call_t1, call_t2, call_state = t1, t2, state
        if "swap_t1t2" in mods:
            call_t1, call_t2 = t2, t1
        if "t1_zero" in mods:
            call_t1 = torch.zeros_like(t1)
        if "t2_zero" in mods:
            call_t2 = torch.zeros_like(t2)
        if "state_zero" in mods and state is not None:
            call_state = torch.zeros_like(state)
        if "state_random" in mods and state is not None:
            call_state = torch.randn_like(state)

        # D_zero temporarily removes the per-position skip path, which is what
        # makes the SSM output a pure function of the token at that position.
        # `self.mamba.D` is an nn.Parameter, so it is zeroed in place and
        # restored afterwards.
        saved_D = None
        if "D_zero" in mods:
            saved_D = self.mamba.D.detach().clone()
            self.mamba.D.data.zero_()
        try:
            pred, loss, new_state = original(self, call_t1, call_t2, target=target,
                                            state=call_state, train=train)
        finally:
            if saved_D is not None:
                self.mamba.D.data.copy_(saved_D)

        if "pred_zero" in mods:
            pred = torch.zeros_like(pred)
        elif "pred_copy_t2" in mods:
            pred = F.normalize(call_t2, dim=-1)
        elif "pred_noise" in mods:
            pred = F.normalize(pred + 0.1 * torch.randn_like(pred), dim=-1)
        if "state_freeze" in mods:
            # keep the real prediction but never advance the recurrent state;
            # isolates the state write-back from the predicted-token effect.
            new_state = state if state is not None else torch.zeros_like(pred)

        return pred, loss, new_state

    module.TemporalMambaMemory.forward = patched


def cmd_run(args) -> None:
    os.environ.setdefault("LASHER_TEST_ROOT", args.lasher_root)
    os.environ.setdefault("RGBT_DATA_ROOT", args.lasher_root)

    spec = importlib.util.spec_from_file_location(
        "test_sgta", os.path.join(REPO, "RGBT_workspace", "test_sgta.py"))
    tl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tl)

    import lib.models.bat.sgta_modules as sgta_modules
    import lib.test.parameter.bat as bat_params
    from lib.test.tracker.bat import BATTrack

    install_predictor_patch(sgta_modules)

    cfg = tl.DATASET_CFG["lasher"]
    dataset = "lasher"
    all_seqs = tl.sequence_dirs(dataset)
    names = [os.path.basename(p) for p in all_seqs]
    if args.sequence:
        wanted = [s.strip() for s in args.sequence.split(",") if s.strip()]
        all_seqs = [p for p in all_seqs if os.path.basename(p) in wanted]
    else:
        counts = np.array([tl.count_frames(p, dataset) for p in all_seqs])
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
        else:  # "first" keeps the dataset order
            order = pool
        # A single long sequence must not eat the whole frame budget: the
        # per-sequence length cap (--max-seq-frames) keeps the subset diverse,
        # which is what the paired per-sequence comparison needs.
        chosen, total = [], 0
        for idx in order:
            if len(chosen) >= args.n or total >= args.max_frames:
                break
            chosen.append(all_seqs[idx])
            total += int(counts[idx])
        all_seqs = chosen
    if not all_seqs:
        raise SystemExit("no sequence selected")

    print("=" * 82)
    print(f"checkpoint   : {args.checkpoint}")
    print(f"yaml         : {args.yaml}")
    print(f"intervention : {args.intervention}")
    print(f"sequences    : {len(all_seqs)} (select={args.select}, max_frames={args.max_frames}, "
          f"len in [{args.min_seq_frames}, {args.max_seq_frames}])")
    total_frames = sum(tl.count_frames(p, dataset) for p in all_seqs)
    print(f"total frames : {total_frames}")
    print(f"seq lengths  : {[tl.count_frames(p, dataset) for p in all_seqs]}")
    print(f"result dir   : {args.result_dir}")
    print("=" * 82)

    params = bat_params.parameters(args.yaml, args.epoch)
    params.checkpoint = args.checkpoint
    tracker = BATTrack(params)
    network = tracker.network

    STATE["mode"] = args.intervention
    STATE["probe"] = args.probe
    STATE["rows"] = []

    original_forward = network.forward
    stride = max(args.sensitivity_stride, 1)

    def hooked_forward(*fargs, **fkwargs):
        capture = STATE.get("capture")
        frame_no = STATE.get("frame", 0)
        STATE["active"] = False
        base = original_forward(*fargs, **fkwargs)
        STATE["active"] = True
        if capture is not None and frame_no % stride == 0:
            # The second pass must not leak into the rollout: BATrack stores the
            # recurrent template state on itself, so snapshot and restore it.
            # Without this, even the "repeat" control would perturb the
            # trajectory (measured +0.7% PR on a single sequence) and could not
            # serve as a numerical-noise floor.
            state_after_base = getattr(network, "dynamic_template", None)
            with torch.no_grad():
                other = original_forward(*fargs, **fkwargs)
            network.dynamic_template = state_after_base
            b_sm = base["score_map"].float()
            o_sm = other["score_map"].float()
            b_box = base["pred_boxes"].float()
            o_box = other["pred_boxes"].float()
            capture.append(dict(
                seq=STATE.get("seq"),
                frame=frame_no,
                rel_score_map=float((o_sm - b_sm).norm() / (b_sm.norm() + 1e-12)),
                max_score_map=float((o_sm - b_sm).abs().max()),
                box_shift_px=float((o_box[..., :2] - b_box[..., :2]).norm(dim=-1).mean()
                                   * params.search_size),
            ))
        return base

    network.forward = hooked_forward

    rows: List[dict] = []
    delta_rows: List[dict] = []
    predictor_rows: List[dict] = []
    try:
        for seq_dir in all_seqs:
            seq_name = os.path.basename(seq_dir)
            rgb_paths = tl.list_frames(seq_dir, cfg["rgb"])
            tir_paths = tl.list_frames(seq_dir, cfg["tir"])
            gts = tl.read_boxes(os.path.join(seq_dir, cfg["gt"]))
            n = min(len(rgb_paths), len(tir_paths), len(gts))
            if n <= 1:
                continue
            STATE["seq"] = seq_name
            STATE["frame"] = 0
            STATE["rows"] = []
            capture: List[dict] = [] if args.intervention != "none" else None
            STATE["capture"] = capture

            image0 = tl.read_rgbt(rgb_paths[0], tir_paths[0])
            tracker.initialize(image0, {"init_bbox": list(np.asarray(gts[0], dtype=np.float32))})
            preds = [list(gts[0])]
            for idx in range(1, n):
                STATE["frame"] = idx
                image = tl.read_rgbt(rgb_paths[idx], tir_paths[idx])
                bbox = tracker.track(image)["target_bbox"]
                preds.append(list(bbox))
            metrics = tl.compute_metrics(preds, gts[:n], None, cfg)
            metrics.update(seq_name=seq_name, frames=n)
            rows.append(metrics)
            if capture:
                delta_rows.extend(capture)
            predictor_rows.extend(STATE["rows"])

            mean_pr = float(np.mean([r["PR"] for r in rows]))
            mean_ss = float(np.mean([r["SS"] for r in rows]))
            diag = ""
            if predictor_rows:
                diag = (f"  cos(pred,t2)={np.mean([d['cos_pred_t2'] for d in predictor_rows]):.4f}"
                        f" cos(pred,state0)="
                        f"{np.mean([d['cos_pred_zerostate'] for d in predictor_rows]):.4f}")
            print(f"[seq] {seq_name:<26} frames={n:<5} PR={metrics['PR']:.4f} "
                  f"SS={metrics['SS']:.4f} | running PR={mean_pr:.4f} SS={mean_ss:.4f}{diag}",
                  flush=True)
    finally:
        network.forward = original_forward

    os.makedirs(args.result_dir, exist_ok=True)
    tag = f"{args.yaml}_ep{args.epoch:04d}_{args.intervention}"
    if args.tag:
        tag = f"{tag}_{args.tag}"
    keys = ("PR", "SS", "MPR", "MSR", "AO", "SR50", "SR75", "NPS")
    summary = dict(
        yaml=args.yaml,
        checkpoint=args.checkpoint,
        epoch=args.epoch,
        intervention=args.intervention,
        probe=bool(args.probe),
        sensitivity_stride=stride,
        sequences=[r["seq_name"] for r in rows],
        frames=[r["frames"] for r in rows],
        mean={key: float(np.mean([r[key] for r in rows])) for key in keys} if rows else {},
        predictor_diag=_summarize(predictor_rows),
        frame_deltas=_summarize(delta_rows),
        metric_rows=rows,
    )
    json_path = os.path.join(args.result_dir, f"{tag}.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    if rows:
        _write_csv(os.path.join(args.result_dir, f"{tag}_per_seq.csv"), rows)
    if predictor_rows:
        _write_csv(os.path.join(args.result_dir, f"{tag}_predictor.csv"), predictor_rows)
    if delta_rows:
        _write_csv(os.path.join(args.result_dir, f"{tag}_frame_deltas.csv"), delta_rows)

    print("-" * 82)
    for key, value in summary["mean"].items():
        print(f"  {key:<6} {value:.4f}")
    print(f"  predictor diag : {summary['predictor_diag']}")
    print(f"  frame deltas   : {summary['frame_deltas']}")
    print(f"  written        : {json_path}")


def cmd_mamba(args) -> None:
    """Inspect the trained SSM parameters of the temporal predictor.

    Explains *why* the scan carries no information between positions: Mamba2
    multiplies the recurrent state by ``A_bar = exp(delta * A)`` at every step.
    With ``A = -exp(A_log)`` and delta from ``softplus(dt_bias)``, this prints
    the implied per-step decay so it can be compared with the 64/128 step
    distances that separate the memory tokens from the read-out positions.
    """
    sd = load_state(args.ckpt)
    prefix = args.prefix
    keys = sorted(k for k in sd if prefix in k)
    print(f"tensors matching {prefix!r}: {len(keys)}")
    for key in keys:
        print(f"  {key:<64}{tuple(sd[key].shape)}")

    def get(name: str):
        for key in keys:
            if key.endswith("." + name) or key.endswith(name):
                return sd[key].float()
        return None

    a_log = get("A_log")
    d = get("D")
    dt_bias = get("dt_bias")
    if a_log is None:
        print("A_log not found; nothing to analyse")
        return

    a = -torch.exp(a_log)
    print(f"\nA_log  : shape={tuple(a_log.shape)} mean={a_log.mean():.4f} "
          f"min={a_log.min():.4f} max={a_log.max():.4f}")
    print(f"A      : shape={tuple(a.shape)} (negative decay) mean={a.mean():.4f} "
          f"|A| mean={a.abs().mean():.4f} min={a.abs().min():.4f} max={a.abs().max():.4f}")
    if dt_bias is not None:
        delta = F.softplus(dt_bias.float())
        print(f"dt_bias: mean={dt_bias.mean():.4f} min={dt_bias.min():.4f} max={dt_bias.max():.4f}")
        print(f"delta=softplus(dt_bias): mean={delta.mean():.6f} "
              f"min={delta.min():.6f} max={delta.max():.6f}")
        decay = torch.exp(delta.unsqueeze(-1) * a) if a.dim() == 2 else torch.exp(delta * a)
        for steps in (1, 8, 32, 64, 128):
            frac = decay.pow(steps).mean().item()
            print(f"  state retention after {steps:>3} steps: {frac:.3e}")
    if d is not None:
        print(f"D (skip) : shape={tuple(d.shape)} mean={d.mean():.4f} "
              f"min={d.min():.4f} max={d.max():.4f}")
    for name in ("dt_proj.weight", "dt_proj.bias", "in_proj.weight", "out_proj.weight"):
        tensor = get(name)
        if tensor is not None:
            print(f"{name:<24} shape={tuple(tensor.shape)} mean={tensor.mean():+.5f} "
                  f"std={tensor.std():.5f} absmean={tensor.abs().mean():.5f}")


def cmd_tensors(args) -> None:
    """Per-tensor statistics for one sub-module of a checkpoint."""
    sd = load_state(args.ckpt)
    keys = sorted(k for k in sd if args.prefix in k)
    print(f"tensors matching {args.prefix!r}: {len(keys)}")
    print(f"  {'name':<56}{'shape':>18}{'rms':>11}{'norm':>11}{'mean':>11}{'std':>11}")
    for key in keys:
        tensor = sd[key].float()
        if tensor.numel() == 0:
            continue
        rms = tensor.pow(2).mean().sqrt().item()
        print(f"  {key.replace(args.prefix, '<>'):<56}{str(tuple(tensor.shape)):>18}"
              f"{rms:>11.4f}{tensor.norm().item():>11.4f}"
              f"{tensor.mean().item():>11.4f}{tensor.std().item() if tensor.numel() > 1 else 0:>11.4f}")


def main() -> None:
    parser = argparse.ArgumentParser("SGTA temporal-branch diagnostics")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_drift = sub.add_parser("drift", help="checkpoint vs initialization parameter drift")
    p_drift.add_argument("--init", required=True)
    p_drift.add_argument("--ckpt", nargs="+", required=True)
    p_drift.add_argument("--only", default="")
    p_drift.add_argument("--inventory", action="store_true")
    p_drift.add_argument("--verbose", action="store_true")
    p_drift.set_defaults(func=cmd_drift)

    p_run = sub.add_parser("run", help="tracker rollout with per-frame interventions")
    p_run.add_argument("--yaml", default="lasher_train")
    p_run.add_argument("--checkpoint", required=True)
    p_run.add_argument("--epoch", type=int, default=15)
    p_run.add_argument("--intervention", default="none",
                       help="one mode, or several joined by '+' "
                            f"(available: {', '.join(MODE_ORDER)})")
    p_run.add_argument("--sequence", default="")
    p_run.add_argument("--select", default="first", choices=["first", "longest", "shortest", "random"])
    p_run.add_argument("--seed", type=int, default=0)
    p_run.add_argument("--n", type=int, default=10)
    p_run.add_argument("--max-frames", type=int, default=6000)
    p_run.add_argument("--min-seq-frames", type=int, default=100)
    p_run.add_argument("--max-seq-frames", type=int, default=1200)
    p_run.add_argument("--sensitivity-stride", type=int, default=10)
    p_run.add_argument("--probe", action="store_true")
    p_run.add_argument("--tag", default="")
    p_run.add_argument("--lasher-root", default="/home/fzg/data/lasher/testingset")
    p_run.add_argument("--result-dir", default=os.path.join(REPO, "RGBT_workspace", "diag"))
    p_run.set_defaults(func=cmd_run)

    p_mamba = sub.add_parser("mamba", help="inspect the trained SSM decay parameters")
    p_mamba.add_argument("--ckpt", required=True)
    p_mamba.add_argument("--prefix", default="adap_predictor.mamba")
    p_mamba.set_defaults(func=cmd_mamba)

    p_tensors = sub.add_parser("tensors", help="per-tensor statistics for a sub-module")
    p_tensors.add_argument("--ckpt", required=True)
    p_tensors.add_argument("--prefix", default="adap_predictor")
    p_tensors.set_defaults(func=cmd_tensors)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
