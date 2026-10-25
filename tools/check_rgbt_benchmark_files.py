#!/usr/bin/env python3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from RGBT_workspace.test_sgta import DATASET_CFG, list_frames, read_boxes, sequence_dirs


def check_dataset(name):
    cfg = DATASET_CFG[name]
    missing = []
    seqs = sequence_dirs(name)
    for seq_dir in seqs:
        seq = Path(seq_dir).name
        rgb = list_frames(seq_dir, cfg["rgb"])
        tir = list_frames(seq_dir, cfg["tir"])
        gt = read_boxes(str(Path(seq_dir) / cfg["gt"]), xyxy=bool(cfg.get("xyxy_gt", False)))
        gt_i = None
        if cfg.get("gt_i"):
            gt_i = read_boxes(str(Path(seq_dir) / cfg["gt_i"]), xyxy=bool(cfg.get("xyxy_gt", False)))
        if not rgb or not tir or not gt or (gt_i is not None and not gt_i):
            missing.append((seq, len(rgb), len(tir), len(gt), len(gt_i) if gt_i is not None else -1))
    print(f"{name}: root={cfg['root']}")
    print(f"{name}: sequences={len(seqs)} missing={len(missing)}")
    for row in missing[:20]:
        print(f"{name}: missing seq={row[0]} rgb={row[1]} tir={row[2]} gt={row[3]} gt_i={row[4]}")
    if len(missing) > 20:
        print(f"{name}: ... {len(missing) - 20} more")
    return len(missing)


def main():
    missing = 0
    for dataset in ("gtot", "rgbt234", "rgbt210"):
        missing += check_dataset(dataset)
    raise SystemExit(1 if missing else 0)


if __name__ == "__main__":
    main()
