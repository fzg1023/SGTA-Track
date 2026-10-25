#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Average BAT/SGTA checkpoints")
    parser.add_argument("--checkpoints", nargs="+", required=True, help="Checkpoint paths to average")
    parser.add_argument("--output", required=True, help="Output checkpoint path")
    parser.add_argument("--epoch", type=int, default=0, help="Epoch value written into output checkpoint")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_paths = [Path(path).expanduser() for path in args.checkpoints]
    missing = [str(path) for path in ckpt_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing checkpoint(s): " + ", ".join(missing))

    checkpoints = [torch.load(path, map_location="cpu") for path in ckpt_paths]
    net_types = {ckpt.get("net_type") for ckpt in checkpoints}
    if len(net_types) != 1:
        raise ValueError(f"Cannot average different net_type values: {sorted(net_types)}")

    net_keys = [set(ckpt["net"].keys()) for ckpt in checkpoints]
    if any(keys != net_keys[0] for keys in net_keys[1:]):
        common = set.intersection(*net_keys)
        missing_by_ckpt = [sorted(net_keys[0] - keys)[:5] for keys in net_keys]
        raise ValueError(f"Checkpoint net keys differ; common={len(common)} samples={missing_by_ckpt}")

    out = dict(checkpoints[-1])
    avg_net = {}
    with torch.no_grad():
        for key in sorted(net_keys[0]):
            values = [ckpt["net"][key] for ckpt in checkpoints]
            if torch.is_floating_point(values[0]):
                avg_net[key] = torch.stack([value.float() for value in values], dim=0).mean(dim=0).to(values[0].dtype)
            else:
                avg_net[key] = values[-1]

    out["net"] = avg_net
    out["epoch"] = args.epoch
    out["optimizer"] = {}
    out["averaged_from"] = [os.path.abspath(str(path)) for path in ckpt_paths]

    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    torch.save(out, tmp)
    os.replace(tmp, output)
    print(f"[OK] averaged {len(ckpt_paths)} checkpoints -> {output}")


if __name__ == "__main__":
    main()
