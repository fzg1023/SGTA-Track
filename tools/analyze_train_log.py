#!/usr/bin/env python
"""Per-epoch curves + checkpoint shortlist from a run_training log.

The trainer writes two kinds of progress lines::

    [train: E, i / N] FPS: ... Loss/total: ... IoU: ...
    [val:   E, i / M] FPS: ... Loss/total: ... IoU: ...

and, when in-training benchmark evaluation is enabled, one block per evaluated
epoch::

    [VTUAV_ST_RESULT]   tag=<config>_<epoch> PR=0.8547 NPR=0.8752 SR=0.7553
    [VTUAV_ST_OFFICIAL] tag=<config>_<epoch> PR=85.47 NPR=85.03 SR=73.70

Everything here is read-only and needs no GPU, so it can run while training is
still going.

    python tools/analyze_train_log.py                          # newest vtuav log
    python tools/analyze_train_log.py --log output/logs/bat-lasher_ablation_a1.log
    python tools/analyze_train_log.py --top 5 --json /tmp/curves.json

The shortlist is printed with the reason each epoch was picked; the per-epoch
table is written to ``--out`` as CSV.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import statistics
import sys
from typing import Dict, List, Optional, Tuple

TRAIN_RE = re.compile(r"^\[train:\s*(\d+),\s*(\d+)\s*/\s*(\d+)\](.*)$")
VAL_RE = re.compile(r"^\[val:\s*(\d+),\s*(\d+)\s*/\s*(\d+)\](.*)$")
BENCH_RE = re.compile(r"^\[([A-Z0-9_]+)_(RESULT|OFFICIAL)\]\s+tag=(\S+)\s+(.*)$")
IOU_RE = re.compile(r"\bIoU:\s*([0-9.]+)")
LOSS_RE = re.compile(r"Loss/total:\s*([0-9.]+)")
FPS_RE = re.compile(r"FPS:\s*([0-9.]+)\s*\(([0-9.]+)\)")
KV_RE = re.compile(r"([A-Za-z_]+)=([-0-9.]+)")


def parse_log(path: str) -> Tuple[Dict[int, Dict[str, List[float]]], List[Dict[str, object]]]:
    train: Dict[int, Dict[str, List[float]]] = {}
    bench: List[Dict[str, object]] = []
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = TRAIN_RE.match(line) or VAL_RE.match(line)
            if match:
                split = "train" if line.startswith("[train") else "val"
                epoch = int(match.group(1))
                iou = IOU_RE.search(line)
                loss = LOSS_RE.search(line)
                fps = FPS_RE.search(line)
                slot = train.setdefault(epoch, {"train_iou": [], "train_loss": [],
                                                "val_iou": [], "val_loss": [],
                                                "train_fps": [], "val_fps": [],
                                                "val_iters": [], "train_iters": []})
                if iou:
                    slot[f"{split}_iou"].append(float(iou.group(1)))
                if loss:
                    slot[f"{split}_loss"].append(float(loss.group(1)))
                if fps:
                    slot[f"{split}_fps"].append(float(fps.group(2)))
                slot[f"{split}_iters"].append(int(match.group(2)))
                continue

            match = BENCH_RE.match(line)
            if match:
                dataset, kind, tag, rest = match.groups()
                record: Dict[str, object] = {"dataset": dataset, "kind": kind, "tag": tag}
                for key, value in KV_RE.findall(rest):
                    record[key] = float(value)
                epoch = tag.rsplit("_", 1)[-1]
                record["epoch"] = int(epoch) if epoch.isdigit() else None
                bench.append(record)
    return train, bench


def mean(values: List[float]) -> Optional[float]:
    return statistics.fmean(values) if values else None


def tail_mean(values: List[float], fraction: float = 0.2) -> Optional[float]:
    if not values:
        return None
    keep = max(1, int(len(values) * fraction))
    return statistics.fmean(values[-keep:])


def build_rows(train: Dict[int, Dict[str, List[float]]],
               bench: List[Dict[str, object]]) -> List[Dict[str, object]]:
    bench_by_epoch: Dict[int, Dict[str, Dict[str, float]]] = {}
    for record in bench:
        if record["epoch"] is None:
            continue
        bench_by_epoch.setdefault(record["epoch"], {})[str(record["dataset"])] = record

    rows: List[Dict[str, object]] = []
    for epoch in sorted(train):
        slot = train[epoch]
        row: Dict[str, object] = {
            "epoch": epoch,
            "train_iou": mean(slot["train_iou"]),
            "train_iou_tail": tail_mean(slot["train_iou"]),
            "train_loss": mean(slot["train_loss"]),
            "val_iou": mean(slot["val_iou"]),
            "val_loss": mean(slot["val_loss"]),
            "val_iters": max(slot["val_iters"]) if slot["val_iters"] else 0,
            "train_iters": max(slot["train_iters"]) if slot["train_iters"] else 0,
        }
        for dataset, record in bench_by_epoch.get(epoch, {}).items():
            tag = "vtuav_st" if dataset.startswith("VTUAV") else dataset.lower()
            for key in ("PR", "NPR", "SR"):
                if key in record:
                    row[f"{tag}_{key}"] = float(record[key])
        rows.append(row)
    return rows


def shortlist(rows: List[Dict[str, object]], top: int, lr_drop: Optional[int]) -> List[Tuple[int, str]]:
    """Epoch -> reason, ordered and de-duplicated."""
    picked: List[Tuple[int, str]] = []

    def add(epoch: Optional[int], reason: str) -> None:
        if epoch is None:
            return
        if not any(existing == epoch for existing, _ in picked):
            picked.append((int(epoch), reason))

    with_val = [r for r in rows if r["val_iou"] is not None]
    if not with_val:
        return picked

    best_val = max(with_val, key=lambda r: r["val_iou"])
    add(best_val["epoch"], f"best val IoU {best_val['val_iou']:.4f}")

    # benchmark-evaluated epochs beat an indirect val signal when present
    bench_rows = [r for r in rows if any(str(k).endswith("_SR") for k in r)]
    if bench_rows:
        best_bench = max(bench_rows, key=lambda r: r.get("vtuav_st_SR", -1.0))
        add(best_bench["epoch"],
            f"best in-training VTUAV ST SR {best_bench.get('vtuav_st_SR'):.4f}")

    add(rows[-1]["epoch"], "last trained epoch (report this one)")
    add(rows[len(rows) // 2]["epoch"], "mid-run reference point")

    if lr_drop:
        add(lr_drop, f"last epoch before the LR drop (LR_DROP_EPOCH={lr_drop})")
        add(lr_drop + 1, "first epoch after the LR drop")

    # top-k by val IoU, so the final choice can be narrowed with real rollouts
    for row in sorted(with_val, key=lambda r: r["val_iou"], reverse=True)[:top]:
        add(row["epoch"], f"top-{top} val IoU {row['val_iou']:.4f}")
    return picked


def find_log(explicit: str) -> Optional[str]:
    if explicit:
        return explicit
    candidates: List[str] = []
    for pattern in ("output/bat-*.log", "output/logs/*.log"):
        candidates.extend(glob.glob(pattern))
    candidates = [c for c in candidates if "Zone.Identifier" not in c]
    if not candidates:
        return None
    vtuav = [c for c in candidates if "vtuav" in os.path.basename(c)]
    pool = vtuav or candidates
    return max(pool, key=os.path.getmtime)


def config_lr_drop(config: str) -> Optional[int]:
    path = f"experiments/bat/{config}.yaml"
    if not os.path.isfile(path):
        return None
    value = None
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.strip().startswith("LR_DROP_EPOCH:"):
                value = line.split(":", 1)[1].strip()
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser("per-epoch curves + checkpoint shortlist from a train log")
    parser.add_argument("--log", default="", help="defaults to the newest output/**/*vtuav*.log")
    parser.add_argument("--config", default="",
                        help="config name, to read LR_DROP_EPOCH (default: infer from the log name)")
    parser.add_argument("--top", type=int, default=3, help="how many top-val epochs to shortlist")
    parser.add_argument("--out", default="", help="write the per-epoch table here (CSV)")
    parser.add_argument("--json", default="", help="write the curves + shortlist here (JSON)")
    args = parser.parse_args()

    log = find_log(args.log)
    if not log or not os.path.isfile(log):
        print("no training log found", file=sys.stderr)
        return 3
    train, bench = parse_log(log)
    if not train:
        print(f"{log}: no [train:] lines parsed", file=sys.stderr)
        return 3
    rows = build_rows(train, bench)

    config = args.config
    if not config:
        base = os.path.basename(log)
        match = re.match(r"bat-(.+?)\.log$", base)
        config = match.group(1) if match else ""
    lr_drop = config_lr_drop(config) if config else None

    print(f"log    : {log}")
    print(f"config : {config or '(unknown)'}"
          + (f"   LR_DROP_EPOCH={lr_drop}" if lr_drop else ""))
    print(f"epochs : {rows[0]['epoch']}..{rows[-1]['epoch']} ({len(rows)} epochs logged)")

    header = (f"{'ep':>3}{'trainIoU':>10}{'trainTail':>10}{'trainLoss':>10}"
              f"{'valIoU':>9}{'valLoss':>9}{'dVal':>8}  in-training eval")
    print()
    print(header)
    print("-" * len(header))
    for index, row in enumerate(rows):
        delta = ""
        if index and rows[index - 1]["val_iou"] is not None and row["val_iou"] is not None:
            delta = f"{row['val_iou'] - rows[index - 1]['val_iou']:+.4f}"
        bench_bits = []
        for key in ("vtuav_st_PR", "vtuav_st_SR", "vtuav_lt_PR", "vtuav_lt_SR",
                    "lasher_PR", "lasher_SR"):
            if key in row:
                bench_bits.append(f"{key.split('_')[-1]}={row[key]:.4f}")
            if key == "vtuav_st_SR" and "vtuav_st_PR" in row:
                bench_bits[-2] = f"ST PR={row['vtuav_st_PR']:.4f}"
        print(f"{row['epoch']:>3}{row['train_iou']:>10.4f}{row['train_iou_tail']:>10.4f}"
              f"{row['train_loss']:>10.4f}"
              f"{(row['val_iou'] if row['val_iou'] is not None else float('nan')):>9.4f}"
              f"{(row['val_loss'] if row['val_loss'] is not None else float('nan')):>9.4f}"
              f"{delta:>8}  {' '.join(bench_bits)}")

    print()
    print("=== candidate checkpoints (evaluate these, not all of them) ===")
    picks = shortlist(rows, args.top, lr_drop)
    for order, (epoch, reason) in enumerate(picks, 1):
        row = next(r for r in rows if r["epoch"] == epoch)
        print(f"  {order}. epoch {epoch:<3} val IoU {row['val_iou']:.4f}   <- {reason}")

    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nper-epoch table -> {args.out}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump({"log": log, "config": config, "lr_drop_epoch": lr_drop,
                       "rows": rows,
                       "shortlist": [{"epoch": e, "reason": r} for e, r in picks]},
                      handle, indent=2, ensure_ascii=False)
        print(f"curves + shortlist -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
