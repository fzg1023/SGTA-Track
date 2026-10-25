"""Rebuild per_seq_metrics.csv + summary.json from existing result txt files.

No GPU / no tracking: reads each sequence's predicted boxes and the (fixed)
GT files, recomputes metrics with the project's own code
(RGBT_workspace/test_sgta.py), and writes the same outputs as write_outputs,
including appending one corrected row to eval_history.csv.

Usage:
    python tools/recompute_per_seq.py \
        --result_dir ./RGBT_workspace/results/lasher/lasher_train_10 \
        --save_name lasher_train_10 \
        --checkpoint ./output/checkpoints/train/bat/lasher_train/BATrack_ep0010.pth.tar \
        --yaml_name lasher_train --epoch 10 \
        --save_dir ./RGBT_workspace/results
"""
import argparse
import os
import sys
import types
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "RGBT_workspace"))
import test_sgta as tl  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_dir", required=True)
    ap.add_argument("--save_name", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--yaml_name", required=True)
    ap.add_argument("--epoch", type=int, required=True)
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--results_ns", default="",
                    help="results namespace of the run whose history should be "
                         "updated (e.g. lasher_a0); empty = dataset name")
    args = ap.parse_args()

    cfg = tl.DATASET_CFG["lasher"]
    seqs = tl.sequence_dirs("lasher")
    recs = []
    for seq_dir in seqs:
        name = os.path.basename(seq_dir)
        pred_path = os.path.join(args.result_dir, name + ".txt")
        if not os.path.isfile(pred_path):
            print("missing result:", name)
            continue
        preds = tl.read_predictions(pred_path)
        gts = tl.read_boxes(os.path.join(seq_dir, cfg["gt"]),
                            xyxy=bool(cfg.get("xyxy_gt", False)))
        n = min(len(preds), len(gts))
        m = tl.compute_metrics(preds[:n], gts[:n], None, cfg)
        m.update(seq_name=name, frame_count=n, fps=0.0, elapsed=0.0, cached=True)
        recs.append(m)
    print("records:", len(recs))

    fake = types.SimpleNamespace(
        checkpoint=args.checkpoint, dataset="lasher", yaml_name=args.yaml_name,
        epoch=args.epoch, save_dir=args.save_dir, results_ns=args.results_ns)
    tl.write_outputs(fake, args.save_name, args.result_dir, recs, total_time=0.0)
    print("rebuilt per_seq_metrics.csv + summary.json; history row appended")


if __name__ == "__main__":
    main()
