#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from typing import Dict, List

import cv2
import numpy as np


PRJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PRJ_ROOT not in sys.path:
    sys.path.insert(0, PRJ_ROOT)

# Frame <-> annotation alignment.  VTUAV annotates one frame per 10 images, so a
# folder holding *every* video frame must be indexed through the annotation row
# (image = str(row*10 + init_idx)), not by the sorted ordinal -- otherwise the
# GT is 10x off and PR/SR collapse.  Identity when counts already match.
from lib.train.dataset.vtuav_alignment import (  # noqa: E402
    ALIGN_ANOMALIES,
    align_verbose,
    annotated_frame_ordinals,
    dense_annotation_index,
    describe,
    load_init_frame,
    split_frame_names,
)

_INIT_FRAME = load_init_frame([os.path.join(PRJ_ROOT, "lib", "train", "dataset")])

RGBT_ROOT = os.environ.get("RGBT_ROOT", os.environ.get("RGBT_DATA_ROOT", "/root/RGBTData"))
LASHER_TRAIN_ROOT = os.environ.get("LASHER_TRAIN_ROOT", os.path.join(RGBT_ROOT, "LasHeR", "train"))
LASHER_TEST_ROOT = os.environ.get("LASHER_TEST_ROOT", os.path.join(RGBT_ROOT, "LasHeR", "test"))
VTUAV_ROOT = os.environ.get("VTUAV_HOME", os.path.join(RGBT_ROOT, "VTUAV"))

DATASET_CFG = {
    "lasher": {
        "root": LASHER_TEST_ROOT,
        "rgb": "visible",
        "tir": "infrared",
        "gt": "visible.txt",
    },
    "lasher_val": {
        "root": LASHER_TRAIN_ROOT,
        "rgb": "visible",
        "tir": "infrared",
        "gt": "visible.txt",
        "val_list": os.path.join(PRJ_ROOT, "lib", "train", "data_specs", "lasher_val.txt"),
    },
    "vtuav_st": {
        # <root>/<seq>/{rgb,ir} + rgb.txt/ir.txt   (flattened layout, see docs/VTUAV.md)
        # A two-level layout (<root>/test_ST_00X/<seq>) is auto-detected as well.
        "root": os.environ.get("VTUAV_ST_ROOT", os.path.join(VTUAV_ROOT, "test_ST")),
        "rgb": "rgb",
        "tir": "ir",
        "gt": "rgb.txt",
        "metric": "vtuav",
        "mpr_thr": 20.0,
        "mpr_cmp": "le",
        "msr_mode": "mean",
    },
    "vtuav_lt": {
        "root": os.environ.get("VTUAV_LT_ROOT", os.path.join(VTUAV_ROOT, "test_LT")),
        "rgb": "rgb",
        "tir": "ir",
        "gt": "rgb.txt",
        "metric": "vtuav",
        "mpr_thr": 20.0,
        "mpr_cmp": "le",
        "msr_mode": "mean",
    },
    "rgbt234": {
        "root": os.environ.get("RGBT234_ROOT", os.path.join(RGBT_ROOT, "RGBT234")),
        "rgb": "visible",
        "tir": "infrared",
        "gt": "visible.txt",
        "gt_i": "infrared.txt",
        "metric": "rgbt234",
        "mpr_thr": 20.0,
        "mpr_cmp": "le",
        "msr_mode": "mean",
    },
    "rgbt210": {
        "root": os.environ.get("RGBT210_ROOT", os.path.join(RGBT_ROOT, "RGBT210")),
        "rgb": "visible",
        "tir": "infrared",
        "gt": "init.txt",
    },
    "gtot": {
        "root": os.environ.get("GTOT_ROOT", os.path.join(RGBT_ROOT, "GTOT")),
        "rgb": "v",
        "tir": "i",
        "gt": "groundTruth_v.txt",
        "gt_i": "groundTruth_i.txt",
        "xyxy_gt": True,
        "metric": "gtot",
        "mpr_thr": 5.0,
        "mpr_cmp": "lt",
        "msr_mode": "auc",
    },
}

METRICS = ["PR", "SS", "MPR", "MSR", "AO", "SR50", "SR75", "PS", "NPS"]
SUCCESS_THRESHOLDS = np.linspace(0, 1, 21)
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

# frame dirs already reported as junk-contaminated (one line, not one per frame)
_JUNK_WARNED = set()


def _warn_junk_frames(frame_dir: str, skipped: List[str]) -> None:
    """Report dropped housekeeping files once per directory.

    The interesting case is macOS' AppleDouble sidecar ``._<name>``: it is
    written next to every file when a folder is copied through a non-HFS
    filesystem, it keeps the ``.jpg`` extension, ``cv2.imread`` returns ``None``
    for it, and it sorts BEFORE the real frames -- so it is not enough to retry
    on read errors, the listing itself has to drop it.
    """
    if frame_dir in _JUNK_WARNED:
        return
    _JUNK_WARNED.add(frame_dir)
    sample = ", ".join(skipped[:3]) + ("..." if len(skipped) > 3 else "")
    print(f"[FRAMES][WARN] skipping {len(skipped)} non-frame file(s) in "
          f"{frame_dir}: {sample}", flush=True)


def list_frames(seq_dir: str, modal: str) -> List[str]:
    frame_dir = os.path.join(seq_dir, modal)
    if not os.path.isdir(frame_dir):
        return []
    names, skipped = split_frame_names(sorted(os.listdir(frame_dir)), IMAGE_EXTS)
    if skipped:
        _warn_junk_frames(frame_dir, skipped)
    return [os.path.join(frame_dir, name) for name in names]


def read_frame(path: str, is_tir: bool = False) -> np.ndarray:
    if is_tir:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise OSError(f"cv2.imread failed: {path}")
        if img.dtype == np.uint16:
            mn, mx = img.min(), img.max()
            if mx > mn:
                img = ((img.astype(np.float32) - mn) / (mx - mn + 1e-6) * 255).astype(np.uint8)
            else:
                img = np.zeros_like(img, dtype=np.uint8)
        elif img.dtype != np.uint8:
            img = img.astype(np.uint8)
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=2)
        elif img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    img = cv2.imread(path)
    if img is None:
        raise OSError(f"cv2.imread failed: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_rgbt(rgb_path: str, tir_path: str) -> np.ndarray:
    rgb = read_frame(rgb_path, is_tir=False)
    tir = read_frame(tir_path, is_tir=True)
    if rgb.shape[:2] != tir.shape[:2]:
        tir = cv2.resize(tir, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    return np.concatenate([rgb, tir], axis=2)


def read_boxes(path: str, xyxy: bool = False) -> List[List[float]]:
    boxes: List[List[float]] = []
    if not os.path.isfile(path):
        return boxes
    raw = open(path, "rb").read()
    if b"\x00" in raw:
        return boxes
    for line in raw.decode("utf-8", errors="replace").splitlines():
        line = line.strip().replace(",", " ").replace("\t", " ")
        if not line:
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            vals = [float(v) for v in parts[:4]]
        except ValueError:
            continue
        if xyxy:
            vals = [vals[0], vals[1], vals[2] - vals[0], vals[3] - vals[1]]
        boxes.append(vals)
    return boxes


def read_predictions(path: str) -> List[List[float]]:
    return read_boxes(path, xyxy=False)


def save_predictions(path: str, preds: List[List[float]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for box in preds:
            f.write(",".join(f"{v:.4f}" for v in box[:4]) + "\n")
    os.replace(tmp_path, path)


def iou_xywh(a, b) -> float:
    ax1, ay1, aw, ah = a[:4]
    bx1, by1, bw, bh = b[:4]
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0


def center_distance_xywh(a, b) -> float:
    ax, ay = a[0] + a[2] / 2, a[1] + a[3] / 2
    bx, by = b[0] + b[2] / 2, b[1] + b[3] / 2
    return float(np.hypot(ax - bx, ay - by))


def success_auc(curve: np.ndarray) -> float:
    return float(np.trapz(curve, SUCCESS_THRESHOLDS))


def compute_metrics(preds, gts, gts_i=None, metric_cfg=None) -> dict:
    metric_cfg = metric_cfg or {}
    has_dual_gt = gts_i is not None
    valid = []
    if has_dual_gt:
        for pred, gt_v, gt_i in zip(preds, gts, gts_i):
            if (
                len(pred) >= 4
                and len(gt_v) >= 4
                and len(gt_i) >= 4
                and gt_v[2] > 0
                and gt_v[3] > 0
                and gt_i[2] > 0
                and gt_i[3] > 0
            ):
                valid.append((pred, gt_v, gt_i))
    else:
        valid = [(p, g) for p, g in zip(preds, gts) if len(p) >= 4 and len(g) >= 4 and g[2] > 0 and g[3] > 0]
    if not valid:
        return {**{k: -1.0 for k in METRICS}, "n_valid": 0}

    ious, dists, ndists = [], [], []
    for item in valid:
        if has_dual_gt:
            pred, gt_v, gt_i = item
            ious.append(max(iou_xywh(pred, gt_v), iou_xywh(pred, gt_i)))
            dist = min(center_distance_xywh(pred, gt_v), center_distance_xywh(pred, gt_i))
            norm_area = max(gt_v[2] * gt_v[3], gt_i[2] * gt_i[3])
        else:
            pred, gt = item
            ious.append(iou_xywh(pred, gt))
            dist = center_distance_xywh(pred, gt)
            norm_area = gt[2] * gt[3]
        dists.append(dist)
        ndists.append(dist / float(np.sqrt(norm_area)) if norm_area > 0 else 0.0)

    iou_arr = np.asarray(ious, dtype=np.float64)
    dist_arr = np.asarray(dists, dtype=np.float64)
    ndist_arr = np.asarray(ndists, dtype=np.float64)
    if has_dual_gt:
        success_curve = np.asarray([(iou_arr > t).mean() for t in SUCCESS_THRESHOLDS], dtype=np.float64)
        mpr_thr = float(metric_cfg.get("mpr_thr", 20.0))
        if metric_cfg.get("mpr_cmp") == "lt":
            mpr = float((dist_arr < mpr_thr).mean())
        else:
            mpr = float((dist_arr <= mpr_thr).mean())
        msr_mode = metric_cfg.get("msr_mode", "mean")
        msr = float(success_curve.mean()) if msr_mode == "mean" else success_auc(success_curve)
    else:
        success_curve = np.asarray([(iou_arr >= t).mean() for t in SUCCESS_THRESHOLDS], dtype=np.float64)
        mpr = float((dist_arr <= 20.0).mean())
        msr = success_auc(success_curve)

    return {
        "PR": mpr,
        "SS": msr,
        "MPR": mpr,
        "MSR": msr,
        "AO": float(iou_arr.mean()),
        "SR50": float((iou_arr >= 0.50).mean()),
        "SR75": float((iou_arr >= 0.75).mean()),
        "PS": mpr,
        "NPS": float((ndist_arr <= 0.5).mean()),
        "n_valid": len(valid),
    }


class BATWrapper:
    def __init__(self, params):
        from lib.test.tracker.bat import BATTrack

        self.tracker = BATTrack(params)

    def initialize(self, image, bbox):
        self.tracker.initialize(image, {"init_bbox": list(np.asarray(bbox, dtype=np.float32))})

    def track(self, image):
        out = self.tracker.track(image)
        return out["target_bbox"], out.get("best_score", 0.0)


def _is_sequence_dir(path: str, cfg: dict) -> bool:
    return os.path.isdir(os.path.join(path, cfg["rgb"]))


def sequence_dirs(dataset: str) -> List[str]:
    cfg = DATASET_CFG[dataset]
    root = cfg["root"]
    if not os.path.isdir(root):
        raise FileNotFoundError(f"dataset root not found: {root}")

    candidates = [
        os.path.join(root, name)
        for name in sorted(os.listdir(root))
        if os.path.isdir(os.path.join(root, name))
    ]
    # Two-level layout, e.g. VTUAV's original release
    # <root>/test_ST_001/<seq>/{rgb,ir}: when the first level carries no
    # modality directory, descend one more level.
    if candidates and not any(_is_sequence_dir(path, cfg) for path in candidates):
        nested = []
        for parent in candidates:
            for name in sorted(os.listdir(parent)):
                child = os.path.join(parent, name)
                if _is_sequence_dir(child, cfg):
                    nested.append(child)
        candidates = nested

    seqs = [path for path in candidates if _is_sequence_dir(path, cfg)]
    val_list = cfg.get("val_list")
    if val_list and os.path.isfile(val_list):
        names = {line.strip() for line in open(val_list, encoding="utf-8") if line.strip()}
        seqs = [path for path in seqs if os.path.basename(path) in names]
    return seqs


def count_frames(seq_dir: str, dataset: str) -> int:
    return len(list_frames(seq_dir, DATASET_CFG[dataset]["rgb"]))


def round_robin_sequence_chunks(seqs: List[str], dataset: str, workers: int) -> tuple[List[List[str]], List[int]]:
    seq_items = sorted([(seq, count_frames(seq, dataset)) for seq in seqs], key=lambda item: item[1])
    chunks = [[] for _ in range(workers)]
    worker_frames = [0 for _ in range(workers)]
    for idx, (seq, frame_count) in enumerate(seq_items):
        worker_id = idx % workers
        chunks[worker_id].append(seq)
        worker_frames[worker_id] += frame_count
    return chunks, worker_frames


def _wait_message(out_q, procs, budget: float = 600.0, poll: float = 5.0):
    """Wait for a worker message, also noticing when every worker has died.

    returns ``(msg, "ok")``, ``(None, "timeout")`` or ``(None, "dead")``.

    A worker killed from outside (host/cgroup OOM) never sends anything, so a
    plain ``out_q.get(timeout=600)`` hides the failure for the full 10 minutes --
    which is the whole run when there is a single worker.  Polling liveness every
    ``poll`` seconds lets the parent give up (and report) within seconds.
    ``"dead"`` needs two consecutive empty polls, so a result still travelling
    through the queue's feeder thread is not dropped.
    """
    waited = 0.0
    empty_polls = 0
    while waited < budget:
        try:
            return out_q.get(timeout=min(poll, budget - waited)), "ok"
        except Exception:
            waited += poll
            empty_polls += 1
            if not any(p.is_alive() for p in procs):
                if empty_polls >= 2:
                    return None, "dead"
            else:
                empty_polls = 0
    return None, "timeout"


def _dead_worker_note(procs) -> str:
    """``"  dead: worker 0 exitcode=-9"`` for workers that are no longer alive.

    The exit code is the only way to tell the two failure modes apart: Python's
    ``Process.exitcode`` is ``-signal`` when the process was killed (``-9`` =
    SIGKILL, typically the host/cgroup OOM killer) and >= 0 for a normal exit or
    an uncaught exception.  A worker killed from outside never gets to send a
    ``seq_error`` message, so without this the run just looks "stuck".
    """
    dead = [(i, proc.exitcode) for i, proc in enumerate(procs) if not proc.is_alive()]
    if not dead:
        return ""
    return "  dead: " + ", ".join(f"worker {i} exitcode={code}" for i, code in dead)


def _report_lost_sequences(procs, chunks, recs) -> None:
    """List the sequences each dead worker will never report (they can be resumed)."""
    reported = {row["seq_name"] for row in recs}
    killed = 0
    for worker_id, proc in enumerate(procs):
        if proc.is_alive():
            continue
        if proc.exitcode is not None and proc.exitcode < 0:
            killed += 1
        pending = [os.path.basename(seq) for seq in chunks[worker_id]
                   if os.path.basename(seq) not in reported]
        print("[ERROR] worker %d exitcode=%s: %d sequence(s) never reported%s"
              % (worker_id, proc.exitcode, len(pending),
                 (": " + ", ".join(pending[:8]) + (" ..." if len(pending) > 8 else ""))
                 if pending else ""), flush=True)
    if killed:
        print("[ERROR] negative exit codes = the process was killed by a signal. "
              "SIGKILL (-9) almost always means the host/cgroup OOM killer (RAM, not VRAM); "
              "SIGSEGV (-11) a crash.  Check `dmesg -T | grep -i -E 'oom|killed process'` "
              "and `free -g`, then re-run with a smaller WORKERS.", flush=True)


def run_worker(worker_id: int, seqs: List[str], args_dict: dict, out_q: mp.Queue) -> None:
    try:
        import torch
        import lib.test.parameter.bat as bat_params

        num_gpus = max(torch.cuda.device_count(), 1)
        torch.cuda.set_device(worker_id % num_gpus)
        params = bat_params.parameters(args_dict["yaml_name"], args_dict["epoch"])
        params.checkpoint = args_dict["checkpoint"]
        tracker = BATWrapper(params)
        out_q.put(("ready", worker_id))
    except Exception:
        out_q.put(("init_error", worker_id, traceback.format_exc()))
        return

    dataset = args_dict["dataset"]
    cfg = DATASET_CFG[dataset]
    result_dir = args_dict["result_dir"]
    skip_existing = args_dict["skip_existing"]

    for seq_dir in seqs:
        seq_name = os.path.basename(seq_dir)
        pred_path = os.path.join(result_dir, f"{seq_name}.txt")
        t0 = time.perf_counter()
        try:
            rgb_paths = list_frames(seq_dir, cfg["rgb"])
            tir_paths = list_frames(seq_dir, cfg["tir"])
            gts = read_boxes(os.path.join(seq_dir, cfg["gt"]), xyxy=bool(cfg.get("xyxy_gt", False)))
            gts_i = None
            if cfg.get("gt_i"):
                gts_i = read_boxes(os.path.join(seq_dir, cfg["gt_i"]), xyxy=bool(cfg.get("xyxy_gt", False)))

            # One entry per annotation row -> which frame of rgb/ to feed the
            # tracker.  Identity for every dataset that ships one image per GT
            # row; a stride-10 pick for VTUAV folders holding all video frames.
            n_gt = min(len(gts), len(gts_i) if gts_i is not None else len(gts))
            if n_gt <= 0:
                raise ValueError("no valid frames or gt")
            n_img = min(len(rgb_paths), len(tir_paths))
            if n_img > n_gt:
                ordinals, align_info = annotated_frame_ordinals(
                    rgb_paths, n_gt, offset=_INIT_FRAME.get(seq_name), seq_name=seq_name)
                # Quiet by default (a dense VTUAV folder is the normal case);
                # VTUAV_ALIGN_VERBOSE=1 prints every sequence, real data problems
                # are always reported.
                if align_verbose():
                    print("[ALIGN] %s" % describe(align_info), flush=True)
                elif align_info.get("missing") or align_info.get("reason") in ALIGN_ANOMALIES:
                    print("[ALIGN][WARN] %s" % describe(align_info), flush=True)
            else:
                ordinals, align_info = list(range(n_img)), {"reason": "one_to_one"}
            # Which frames to run the tracker on, and the GT of each of them.
            #  official  - only the annotated frames (VTUAV: 1 per 10 images)
            #  dense     - every frame in rgb/, paired with the nearest
            #              annotation (GT of the frames between labels is repeated)
            #  dense_ann - every frame is TRACKED, but only the annotated frames
            #              are scored; the tracker still sees the full frame rate
            protocol = args_dict.get("protocol", "official")
            score_at_annotations = False
            if protocol in ("dense", "dense_ann") and n_img > n_gt:
                gt_of_image = dense_annotation_index(ordinals, n_img, n_gt)
                frame_idx = list(range(n_img))
                if protocol == "dense":
                    gts_eval = [gts[i] for i in gt_of_image]
                    gts_i_eval = [gts_i[i] for i in gt_of_image] if gts_i is not None else None
                else:
                    gts_eval = gts
                    gts_i_eval = gts_i
                    score_at_annotations = True
            else:
                frame_idx = list(ordinals)[:min(len(ordinals), n_gt, n_img)]
                gts_eval = gts[:len(frame_idx)]
                gts_i_eval = gts_i[:len(frame_idx)] if gts_i is not None else None
            n = len(frame_idx)
            if n <= 0:
                raise ValueError("no valid frames or gt")

            def score_pairs(preds_all):
                """(preds, gts, gts_i) handed to the metric code for this protocol."""
                if not score_at_annotations:
                    return preds_all, gts_eval, gts_i_eval
                sel = [(j, row) for row, j in enumerate(ordinals[:len(gts_eval)])
                       if j < len(preds_all)]
                return ([preds_all[j] for j, _ in sel], [gts[row] for _, row in sel],
                        [gts_i[row] for _, row in sel] if gts_i is not None else None)

            if skip_existing and os.path.isfile(pred_path):
                preds = read_predictions(pred_path)
                if len(preds) == n:
                    pm, gm, gim = score_pairs(preds)
                    metrics = compute_metrics(pm, gm, gim, cfg)
                    metrics.update(official_seq_metrics(pm, gm))
                    metrics.update(seq_name=seq_name, frame_count=n, fps=0.0,
                                   elapsed=time.perf_counter() - t0, cached=True)
                    out_q.put(("result", worker_id, metrics))
                    continue
                print(f"[CACHE] incomplete prediction, re-running {seq_name}: "
                      f"{len(preds)}/{n} frames", flush=True)

            image0 = read_rgbt(rgb_paths[frame_idx[0]], tir_paths[frame_idx[0]])
            tracker.initialize(image0, gts_eval[0])
            preds = [list(gts_eval[0])]

            track_start = time.perf_counter()
            for idx in range(1, n):
                j = frame_idx[idx]
                image = read_rgbt(rgb_paths[j], tir_paths[j])
                bbox, _ = tracker.track(image)
                preds.append(list(bbox))
            track_time = time.perf_counter() - track_start

            save_predictions(pred_path, preds)
            pm, gm, gim = score_pairs(preds)
            metrics = compute_metrics(pm, gm, gim, cfg)
            metrics.update(official_seq_metrics(pm, gm))
            metrics.update(
                seq_name=seq_name,
                frame_count=n,
                fps=float((n - 1) / max(track_time, 1e-6)),
                elapsed=float(time.perf_counter() - t0),
                cached=False,
            )
            out_q.put(("result", worker_id, metrics))
        except Exception:
            out_q.put(("seq_error", worker_id, seq_name, traceback.format_exc(), time.perf_counter() - t0))


def cumulative(valid: List[dict]) -> Dict[str, float]:
    if not valid:
        return {k: -1.0 for k in METRICS}
    return {k: float(np.mean([row[k] for row in valid])) for k in METRICS}


# ---------------------------------------------------------------------------
# Official LasHeR metrics (rgbt 1.0.1 semantics)
# ---------------------------------------------------------------------------
# These deliberately re-implement the official toolkit's definitions so that the
# live per-sequence output can show a running mean that converges to the number
# `tools/eval_official_lasher.py` reports.  Differences from the internal
# metrics above are NOT cosmetic:
#   * centre convention is (w-1)/2 rather than w/2
#   * NPR normalises each axis by the GT width/height separately (elliptical)
#     and uses threshold 0.2, not 0.5
#   * SR is the mean success over 21 thresholds in [0, 1], not the trapezoid AUC
#   * the first frame is forced to the GT box, predictions are truncated to the
#     GT length, non-positive boxes are repaired from the previous frame, and
#     frames whose GT has a non-positive entry are set to -1
# Verified against the toolkit by tools/check_official_metric_parity.py.
OFFICIAL_PR_THR = 20.0
OFFICIAL_NPR_THR = 0.2
OFFICIAL_SR_THRESHOLDS = np.linspace(0.0, 1.0, 21)


def official_seq_metrics(preds, gts) -> Dict[str, float]:
    """Per-sequence official PR / NPR / SR for LasHeR (ltwh boxes)."""
    p = np.asarray(preds, dtype=np.float64).reshape(-1, 4).copy()
    g = np.asarray(gts, dtype=np.float64).reshape(-1, 4)
    n = min(len(p), len(g))
    if n == 0:
        return {"oPR": -1.0, "oNPR": -1.0, "oSR": -1.0}
    p, g = p[:n], g[:n]
    # The toolkit loads both files through initial_result_file(), which applies
    # np.round(0).  Rounding here is what makes this implementation agree with
    # the toolkit instead of only approximately.
    p = np.round(p)
    g = np.round(g)

    # ---- official preprocessing -------------------------------------------
    p[0] = g[0]                                    # ignore the first frame
    for i in range(1, n):                          # repair invalid boxes
        if p[i, 2] <= 0 or p[i, 3] <= 0:
            p[i] = p[i - 1]

    # ---- centre location error (strict (w-1)/2 convention) ----------------
    pc_x = p[:, 0] + (p[:, 2] - 1.0) / 2.0
    pc_y = p[:, 1] + (p[:, 3] - 1.0) / 2.0
    gc_x = g[:, 0] + (g[:, 2] - 1.0) / 2.0
    gc_y = g[:, 1] + (g[:, 3] - 1.0) / 2.0
    cle = np.hypot(pc_x - gc_x, pc_y - gc_y)

    # ---- normalized CLE: each axis divided by the GT extent ---------------
    eps = 1e-8
    gt_w = np.abs(g[:, 2]) + eps
    gt_h = np.abs(g[:, 3]) + eps
    ncle = np.hypot((pc_x - gc_x) / gt_w, (pc_y - gc_y) / gt_h)

    # ---- IoU with the l+w-1 / area = w*h convention -----------------------
    x1, y1 = p[:, 0], p[:, 1]
    x2, y2 = p[:, 0] + p[:, 2] - 1.0, p[:, 1] + p[:, 3] - 1.0
    tx1, ty1 = g[:, 0], g[:, 1]
    tx2, ty2 = g[:, 0] + g[:, 2] - 1.0, g[:, 1] + g[:, 3] - 1.0
    iw = np.clip(np.minimum(x2, tx2) - np.maximum(x1, tx1) + 1.0, 0, None)
    ih = np.clip(np.minimum(y2, ty2) - np.maximum(y1, ty1) + 1.0, 0, None)
    inter = iw * ih
    union = p[:, 2] * p[:, 3] + g[:, 2] * g[:, 3] - inter
    iou = np.where(np.abs(union) > eps, inter / np.where(union == 0, eps, union), 0.0)

    # ---- frames with a non-positive GT entry count as -1 ------------------
    invalid = (g <= 0).any(axis=1)
    cle = np.where(invalid, -1.0, cle)
    ncle = np.where(invalid, -1.0, ncle)
    iou = np.where(invalid, -1.0, iou)

    pr = float(np.mean(cle <= OFFICIAL_PR_THR))
    npr = float(np.mean(ncle <= OFFICIAL_NPR_THR))
    sr = float(np.mean([np.mean(iou > t) for t in OFFICIAL_SR_THRESHOLDS]))
    return {"oPR": pr, "oNPR": npr, "oSR": sr}


OFFICIAL_KEYS = ["oPR", "oNPR", "oSR"]


def cumulative_official(valid: List[dict]) -> Dict[str, float]:
    """Running dataset-level official metrics = mean over sequences so far.

    The official toolkit averages the per-sequence values, so this running mean
    converges exactly to `lasher.PR()/NPR()/SR()` once every sequence is in.
    """
    if not valid:
        return {k: -1.0 for k in OFFICIAL_KEYS}
    return {k: float(np.mean([row.get(k, -1.0) for row in valid])) for k in OFFICIAL_KEYS}


def official_metric_source(dataset: str) -> str:
    if dataset == "gtot":
        return "frm"
    if dataset == "rgbt234":
        return "seq"
    return "seq"


def official_metric_names(dataset: str) -> List[str]:
    if dataset in ("gtot", "rgbt234"):
        return ["MPR", "MSR"]
    # LasHeR: the official RGBT benchmark reports PR (precision rate, center
    # error <= 20px), NPR (normalized precision rate, normalized error <= 0.5,
    # stored under the "NPS" key) and SR (success rate = AUC of the success
    # curve, stored under the "SS" key).
    return ["PR", "NPS", "SS"]


def write_outputs(args, save_name: str, result_dir: str, recs: List[dict], total_time: float) -> None:
    valid = [row for row in recs if row["AO"] >= 0]
    if valid:
        total_frames = sum(row["n_valid"] for row in valid)
        seq_means = {k: float(np.mean([row[k] for row in valid])) for k in METRICS}
        frm_means = {
            k: float(sum(row[k] * row["n_valid"] for row in valid) / max(total_frames, 1))
            for k in METRICS
        }
        fps_values = [row["fps"] for row in valid if row["fps"] > 0]
        mean_fps = float(np.mean(fps_values)) if fps_values else 0.0
    else:
        total_frames = 0
        seq_means = {k: -1.0 for k in METRICS}
        frm_means = {k: -1.0 for k in METRICS}
        mean_fps = 0.0

    # Dataset-level official LasHeR metrics: the toolkit averages the
    # per-sequence values, so this is exactly `lasher.PR()/NPR()/SR()`.
    seq_official = cumulative_official(valid)

    per_seq_path = os.path.join(result_dir, "per_seq_metrics.csv")
    ordered_recs = sorted(recs, key=lambda row: (row.get("frame_count", row.get("n_valid", 0)), row.get("seq_name", "")))
    fields = (["seq_name", "frame_count"] + METRICS + ["n_valid", "fps", "elapsed", "cached"]
              + OFFICIAL_KEYS + [f"cum_{k}" for k in METRICS]
              + [f"cum_{k}" for k in OFFICIAL_KEYS])
    with open(per_seq_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        running = []
        for row in ordered_recs:
            if row["AO"] >= 0:
                running.append(row)
            cum = cumulative(running)
            cum_o = cumulative_official(running)
            out = {k: row.get(k, "") for k in fields}
            out.update({f"cum_{k}": cum[k] for k in METRICS})
            out.update({f"cum_{k}": cum_o[k] for k in OFFICIAL_KEYS})
            writer.writerow(out)

    history_dir = os.path.join(args.save_dir,
                               (getattr(args, "results_ns", "") or "").strip() or args.dataset)
    os.makedirs(history_dir, exist_ok=True)
    history_path = os.path.join(history_dir, "eval_history.csv")
    hist_fields = (
        ["ckpt_tag", "checkpoint", "dataset", "n_valid", "n_total", "mean_fps"]
        + [f"seq_{k}" for k in METRICS]
        + [f"frm_{k}" for k in METRICS]
        + [f"seq_{k}" for k in OFFICIAL_KEYS]
    )
    hist_row = {
        "ckpt_tag": save_name,
        "checkpoint": args.checkpoint,
        "dataset": args.dataset,
        "n_valid": len(valid),
        "n_total": len(recs),
        "mean_fps": f"{mean_fps:.2f}",
    }
    for key in METRICS:
        hist_row[f"seq_{key}"] = f"{seq_means[key]:.4f}"
        hist_row[f"frm_{key}"] = f"{frm_means[key]:.4f}"
    for key in OFFICIAL_KEYS:
        hist_row[f"seq_{key}"] = f"{seq_official[key]:.4f}"

    write_header = not os.path.isfile(history_path)
    # The history file is SHARED by every run/config, so its schema evolves when
    # new columns are added.  Never append a wider row under a narrower header:
    # readers resolve columns by name, and a stale header would silently hide
    # the new columns.  Upgrade the header in place (old columns keep their
    # position, new ones are appended) and preserve every existing row.
    existing_header: List[str] = []
    existing_rows: List[List[str]] = []
    if not write_header:
        with open(history_path, newline="", encoding="utf-8") as f_old:
            rows_txt = list(csv.reader(f_old))
        if rows_txt:
            existing_header = rows_txt[0]
            existing_rows = rows_txt[1:]
    merged_fields = list(existing_header)
    for name in hist_fields:
        if name not in merged_fields:
            merged_fields.append(name)
    if not merged_fields:
        merged_fields = list(hist_fields)
    if existing_header != merged_fields:
        with open(history_path, "w", newline="", encoding="utf-8") as f_old:
            w_old = csv.DictWriter(f_old, fieldnames=merged_fields, extrasaction="ignore")
            w_old.writeheader()
            for row in existing_rows:
                w_old.writerow({k: (row[i] if i < len(row) else "")
                                for i, k in enumerate(existing_header)})
        if existing_header:
            print("[INFO] eval_history.csv header upgraded with: %s"
                  % ", ".join(h for h in merged_fields if h not in existing_header), flush=True)
    with open(history_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=merged_fields, extrasaction="ignore")
        writer.writerow(hist_row)

    official_source = official_metric_source(args.dataset)
    official_names = official_metric_names(args.dataset)
    official = frm_means if official_source == "frm" else seq_means

    summary = {
        "checkpoint": args.checkpoint,
        "dataset": args.dataset,
        "yaml_name": args.yaml_name,
        "epoch": args.epoch,
        "protocol": getattr(args, "protocol", "official"),
        "n_sequences": len(recs),
        "n_valid": len(valid),
        "total_frames": total_frames,
        "mean_fps": mean_fps,
        "per_sequence_report_order": "frame_count_asc_then_seq_name",
        "seq_means": seq_means,
        "frm_means": frm_means,
        "official_metric_source": official_source,
        "official_metrics": {name: official[name] for name in official_names},
        "official_toolkit_metrics": dict(seq_official),
        "total_time_min": total_time / 60.0,
    }
    summary_path = os.path.join(result_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    width = 9
    total_time_min = total_time / 60.0
    total_time_hour = total_time / 3600.0
    print("\n" + "=" * 78)
    print(f"[Summary] valid={len(valid)}/{len(recs)} frames={total_frames} mean_fps={mean_fps:.1f}")
    print(f"{'':12}" + "".join(f"{k:>{width}}" for k in METRICS))
    print(f"{'seq_mean(%)':<12}" + "".join(f"{seq_means[k] * 100:>{width}.2f}" for k in METRICS))
    print(f"{'frm_weight(%)':<12}" + "".join(f"{frm_means[k] * 100:>{width}.2f}" for k in METRICS))
    print(f"{'official(%)':<12}" + "".join(f"{official[k] * 100:>{width}.2f}" for k in official_names) + "       " + "       ".join(official_names))
    print("=" * 78)
    print("[RESULT] " + " ".join(f"{k}={seq_means[k]:.4f}" for k in METRICS))
    print("[OFFICIAL] " + " ".join(f"{k}={official[k]:.4f}" for k in official_names) + f" source={official_source}")
    # Toolkit-aligned numbers: these use the official definitions (NPR threshold
    # 0.2, threshold-averaged SR, (w-1)/2 centre) and are what the paper reports.
    print("[OFFICIAL_TOOLKIT] " + " ".join(f"{k}={seq_official[k]:.4f}" for k in OFFICIAL_KEYS)
          + " source=seq")
    print(f"[OFFICIAL_TOOLKIT_PCT] PR={seq_official['oPR'] * 100:.2f} "
          f"NPR={seq_official['oNPR'] * 100:.2f} SR={seq_official['oSR'] * 100:.2f}")
    print(f"[TIME] total_sec={total_time:.1f} total_min={total_time_min:.2f} total_hour={total_time_hour:.3f}")
    print(f"[INFO] per_seq_metrics.csv -> {per_seq_path}")
    print(f"[INFO] eval_history.csv    -> {history_path}")
    print(f"[INFO] summary.json        -> {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("SGTA-Track RGBT evaluation")
    parser.add_argument("--yaml_name", default="lasher_train")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--dataset", default="lasher")
    parser.add_argument("--save_dir", default=os.path.join(PRJ_ROOT, "RGBT_workspace", "results"))
    parser.add_argument("--results_ns", default="")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--epoch", type=int, default=30)
    parser.add_argument("--tag", default="")
    parser.add_argument("--sequence", action="append", default=[],
                        help="restrict the run to this sequence; repeat the flag to "
                             "select several (append keeps a single `--sequence NAME` "
                             "working exactly as before)")
    parser.add_argument("--sequences", default="",
                        help="comma/space separated sequence list; the convenient way "
                             "to run a subset, e.g. for a throughput measurement on a "
                             "few sequences without touching the full 220k-frame set")
    parser.add_argument("--protocol", default="official",
                        choices=("official", "dense", "dense_ann"),
                        help="official: track+score only the annotated frames (1 per 10 "
                             "on VTUAV, the benchmark protocol); dense: track every "
                             "frame and score every frame against its nearest annotation; "
                             "dense_ann: track every frame but score only the annotated "
                             "ones (full-rate rollout with a benchmark-comparable score).")
    parser.add_argument("--skip_existing", action="store_true", default=True)
    parser.add_argument("--force", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    mp.set_start_method("spawn", force=True)
    args = parse_args()
    args.dataset = args.dataset.lower()
    if args.dataset == "lasher_test":
        args.dataset = "lasher"
    if args.dataset not in DATASET_CFG:
        raise ValueError(f"unsupported dataset: {args.dataset}; choices={list(DATASET_CFG)}")
    if args.force:
        args.skip_existing = False
    if not args.checkpoint:
        args.checkpoint = os.path.join(
            PRJ_ROOT,
            "output",
            "checkpoints",
            "train",
            "bat",
            args.yaml_name,
            f"BATrack_ep{args.epoch:04d}.pth.tar",
        )
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")

    save_name = f"{args.yaml_name}_{args.epoch}{args.tag}"
    # Results namespace: the dataset name by default, or an explicit
    # subdirectory so an ablation row keeps its dumps (and its OWN
    # eval_history.csv) apart from the main tree, e.g.
    #   results/lasher_a0/<config>_<epoch>/
    results_ns = (args.results_ns or "").strip() or args.dataset
    args.results_ns = results_ns
    result_dir = os.path.join(args.save_dir, results_ns, save_name)
    os.makedirs(result_dir, exist_ok=True)

    seqs = sequence_dirs(args.dataset)
    wanted = [name.strip() for name in (args.sequence or []) if name.strip()]
    if args.sequences:
        wanted += [name for name in args.sequences.replace(",", " ").split() if name]
    if wanted:
        available = {os.path.basename(path) for path in seqs}
        missing = [name for name in wanted if name not in available]
        if missing:
            print(f"[WARN] sequence(s) not found in {args.dataset}: {missing}")
        keep = set(wanted)
        seqs = [path for path in seqs if os.path.basename(path) in keep]
    if not seqs:
        raise RuntimeError("no sequence found")

    cfg = DATASET_CFG[args.dataset]
    workers = min(max(args.workers, 1), len(seqs))
    chunks, worker_frames = round_robin_sequence_chunks(seqs, args.dataset, workers)

    print("=" * 80)
    print(f"Checkpoint : {args.checkpoint}")
    print(f"YAML       : {args.yaml_name}")
    print(f"Epoch      : {args.epoch}")
    print(f"Dataset    : {args.dataset}")
    _proto = getattr(args, "protocol", "official")
    _proto_note = {
        "official": "  (annotated frames only -- benchmark protocol)",
        "dense": "  (track+score every frame; GT = nearest annotation)",
        "dense_ann": "  (track every frame, score the annotated frames only)",
    }.get(_proto, "")
    print(f"Protocol   : {_proto}{_proto_note}")
    print(f"Results NS : {results_ns}{'  (default = dataset name)' if results_ns == args.dataset else ''}")
    print(f"Data root  : {cfg['root']}")
    print(f"Result dir : {result_dir}")
    print(f"Workers    : {workers}")
    print("Schedule   : frame_count_asc round-robin")
    for worker_id, chunk in enumerate(chunks):
        print(f"  worker {worker_id}: seqs={len(chunk)} frames={worker_frames[worker_id]}")
    print("=" * 80)

    out_q: mp.Queue = mp.Queue()
    payload = vars(args).copy()
    payload["result_dir"] = result_dir
    procs = []
    for worker_id, chunk in enumerate(chunks):
        proc = mp.Process(target=run_worker, args=(worker_id, chunk, payload, out_q), daemon=True)
        proc.start()
        procs.append(proc)

    ready = 0
    while ready < workers:
        msg, status = _wait_message(out_q, procs)
        if status == "dead":
            print("[ERROR] all workers exited before ready; abort", flush=True)
            print("[ERROR] worker exit codes:%s"
                  % "".join(" worker %d=%s" % (i, p.exitcode)
                            for i, p in enumerate(procs)), flush=True)
            raise SystemExit(2)
        if status == "timeout":
            print("[WARN] no ready message for 600s, waiting ...", flush=True)
            continue
        if msg[0] == "ready":
            ready += 1
            print(f"[INFO] worker {msg[1]} ready ({ready}/{workers})", flush=True)
        elif msg[0] == "init_error":
            print(f"[ERROR] worker {msg[1]} init failed:\n{msg[2]}", flush=True)
            ready += 1

    header = (
        f"\n{'#':<9} {'seq':<32} {'PR':>6} {'SS':>6} {'MPR':>6} {'MSR':>6} "
        f"{'AO':>6} {'SR50':>6} {'SR75':>6} {'NPS':>6} {'FPS':>7} "
        f"{'DonePR':>7} {'DoneNPR':>8} {'DoneSR':>7} "
        f"{'| oPR':>6} {'oNPR':>6} {'oSR':>6}　{'cPR':>6} {'cNPR':>6} {'cSR':>6}"
    )
    print(header, flush=True)
    print("-" * (len(header) - 1), flush=True)

    recs: List[dict] = []
    valid_running: List[dict] = []
    done = 0
    total = len(seqs)
    start = time.perf_counter()
    while done < total:
        msg, status = _wait_message(out_q, procs)
        if status == "dead":
            print("[ERROR] all %d workers exited with %d/%d results missing; stop waiting"
                  % (workers, total - done, total), flush=True)
            _report_lost_sequences(procs, chunks, recs)
            break
        if status == "timeout":
            print("[WARN] no message for 600s; workers alive: %s%s"
                  % ([p.is_alive() for p in procs], _dead_worker_note(procs)), flush=True)
            continue
        if msg[0] == "result":
            _, _, row = msg
            done += 1
            recs.append(row)
            if row["AO"] >= 0:
                valid_running.append(row)
            cum = cumulative(valid_running)
            cum_o = cumulative_official(valid_running)
            print(
                f"[{done:03d}/{total:03d}] {row['seq_name']:<32s} "
                f"{row['PR']:6.3f} {row['SS']:6.3f} {row['MPR']:6.3f} {row['MSR']:6.3f} "
                f"{row['AO']:6.3f} {row['SR50']:6.3f} {row['SR75']:6.3f} {row['NPS']:6.3f} "
                f"{row['fps']:7.1f} {cum['PR']:7.3f} {cum['NPS']:8.3f} {cum['SS']:7.3f} "
                f"{row.get('oPR', -1.0):6.3f} {row.get('oNPR', -1.0):6.3f} {row.get('oSR', -1.0):6.3f} "
                f"{cum_o['oPR']:6.3f} {cum_o['oNPR']:6.3f} {cum_o['oSR']:6.3f}"
                f"{' [CACHED]' if row.get('cached', False) else ''}",
                flush=True,
            )
        elif msg[0] == "seq_error":
            _, wid, seq_name, tb, elapsed = msg
            done += 1
            row = {
                "seq_name": seq_name,
                "frame_count": count_frames(os.path.join(DATASET_CFG[args.dataset]["root"], seq_name), args.dataset),
                **{k: -1.0 for k in METRICS},
                "n_valid": 0,
                "fps": 0.0,
                "elapsed": elapsed,
                "cached": False,
            }
            recs.append(row)
            print(f"[{done:03d}/{total:03d}] {seq_name:<32s} [ERROR] worker={wid} {tb.splitlines()[-1]}",
                  flush=True)
            print(tb, flush=True)

    for proc in procs:
        proc.join(timeout=30)

    if done < total:
        print("[ERROR] incomplete: %d/%d results (workers died or timed out). "
              "No history/summary written; re-run to resume missing sequences." % (done, total),
              flush=True)
        raise SystemExit(2)

    failed = [row["seq_name"] for row in recs if row.get("AO", -1.0) < 0.0]
    if failed:
        print("[ERROR] %d/%d sequences failed: %s" %
              (len(failed), total, ", ".join(failed)), flush=True)
        print("[ERROR] No history/summary written; stop training to release GPU memory, "
              "then re-run to resume only the missing sequences.", flush=True)
        raise SystemExit(2)

    write_outputs(args, save_name, result_dir, recs, time.perf_counter() - start)


if __name__ == "__main__":
    main()
