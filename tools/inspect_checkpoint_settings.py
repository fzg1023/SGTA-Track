#!/usr/bin/env python3
"""Print the training settings embedded in a SGTA-Track / BATrack checkpoint.

`BaseTrainer.save_checkpoint` stores the full `settings` object next to the
network weights, so the config name, cfg file and SGTA module switches can be
recovered from a checkpoint even when it was copied out of its original
`output/checkpoints/train/bat/<config>/` directory.

Usage:
    python tools/inspect_checkpoint_settings.py output/checkpoints/train/bat/BATrack_ep0015.pth.tar
    python tools/inspect_checkpoint_settings.py <ckpt> --config-only
"""
from __future__ import annotations

import argparse
import os
import sys

PRJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PRJ_ROOT not in sys.path:
    sys.path.insert(0, PRJ_ROOT)

import torch  # noqa: E402

SGTA_FLAGS = (
    "ENABLED",
    "USE_MAMBA_FUSION",
    "USE_TEMPORAL_MAMBA",
    "USE_LIQUID_CONTROLLER",
    "USE_MOE",
    "USE_TEMPLATE_ROUTER",
    "ENABLE_TEMPLATE_UPDATE",
    "USE_LEGACY_TEMPORAL_PREDICTOR",
    "MAMBA_VERSION",
    "TEMPORAL_MEMORY_TOKENS",
    "LIQUID_HIDDEN_DIM",
    "MOE_HIDDEN_DIM",
    "UPDATE_GATE_THRESHOLD",
)
TRAIN_KEYS = ("EPOCH", "BATCH_SIZE", "LR", "LR_DROP_EPOCH", "CE_START_EPOCH", "CE_WARM_EPOCH")


def describe_settings(settings) -> int:
    for name in ("config_name", "cfg_file", "project_path", "script_name", "save_dir"):
        print(f"  {name:<14}: {getattr(settings, name, None)}")

    cfg = getattr(settings, "cfg", None)
    if cfg is None:
        print("  (no cfg attached)")
        return 0

    model = getattr(cfg, "MODEL", None)
    sgta = getattr(model, "SGTA", None) if model is not None else None
    print("  MODEL.SGTA:")
    if sgta is None:
        print("    (none)")
    else:
        for key in SGTA_FLAGS:
            print(f"    {key:<30}: {getattr(sgta, key, '<unset>')}")

    train = getattr(cfg, "TRAIN", None)
    print("  TRAIN:")
    if train is None:
        print("    (none)")
    else:
        for key in TRAIN_KEYS:
            print(f"    {key:<30}: {getattr(train, key, '<unset>')}")
        print(f"    {'EVAL_LASHER_EVERY_EPOCH':<30}: {getattr(train, 'EVAL_LASHER_EVERY_EPOCH', '<unset>')}")

    backbone = getattr(model, "BACKBONE", None) if model is not None else None
    if backbone is not None:
        print("  MODEL.BACKBONE:")
        for key in ("TYPE", "CE_LOC", "CE_KEEP_RATIO"):
            print(f"    {key:<30}: {getattr(backbone, key, '<unset>')}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser("inspect SGTA-Track checkpoint settings")
    parser.add_argument("checkpoint")
    parser.add_argument("--config-only", action="store_true",
                        help="print only the config/cfg_file names (for shell use)")
    args = parser.parse_args()

    checkpoint = os.path.abspath(args.checkpoint)
    if not os.path.isfile(checkpoint):
        print(f"[ERROR] checkpoint not found: {checkpoint}", file=sys.stderr)
        return 2

    print(f"checkpoint : {checkpoint}")
    print(f"size       : {os.path.getsize(checkpoint) / 1e9:.2f} GB")
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)

    settings = ckpt.get("settings", None)
    if args.config_only:
        print(getattr(settings, "config_name", ""))
        return 0

    print(f"epoch      : {ckpt.get('epoch')}")
    print(f"actor_type : {ckpt.get('actor_type')}")
    print(f"net_type   : {ckpt.get('net_type')}")
    print(f"net params : {len(ckpt.get('net', {}))} tensors")
    print("settings:")
    if settings is None:
        print("  (not stored in this checkpoint)")
        return 1
    return describe_settings(settings)


if __name__ == "__main__":
    raise SystemExit(main())
