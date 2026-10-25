#!/usr/bin/env python3
import argparse
import re
from pathlib import Path


LINE_RE = re.compile(
    r"\[(train|val):\s*(\d+),\s*(\d+)\s*/\s*(\d+)\].*?"
    r"Loss/total:\s*([0-9.]+).*?IoU:\s*([0-9.]+)"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log")
    args = parser.parse_args()

    lines = Path(args.log).read_text(errors="replace").splitlines()
    ends = {}
    latest = {}
    for line in lines:
        match = LINE_RE.search(line)
        if not match:
            continue
        split, epoch, step, total, loss, iou = match.groups()
        row = {
            "split": split,
            "epoch": int(epoch),
            "step": int(step),
            "total": int(total),
            "loss": float(loss),
            "iou": float(iou),
            "line": line,
        }
        latest[split] = row
        if row["step"] == row["total"]:
            ends[(split, row["epoch"])] = row

    print("epoch,train_loss,train_iou,val_loss,val_iou")
    for epoch in sorted({epoch for _, epoch in ends.keys()}):
        train = ends.get(("train", epoch))
        val = ends.get(("val", epoch))
        print(
            f"{epoch},"
            f"{train['loss'] if train else ''},"
            f"{train['iou'] if train else ''},"
            f"{val['loss'] if val else ''},"
            f"{val['iou'] if val else ''}"
        )

    print("latest")
    for split in ("train", "val"):
        row = latest.get(split)
        if row:
            print(f"{split}: epoch={row['epoch']} step={row['step']}/{row['total']} loss={row['loss']:.5f} iou={row['iou']:.5f}")

    errors = [line for line in lines if "ERROR: Could not read image" in line]
    print(f"image_read_errors={len(errors)}")
    for line in sorted(set(errors))[-10:]:
        print(line)


if __name__ == "__main__":
    main()
