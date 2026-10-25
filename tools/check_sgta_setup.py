#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

from mamba_ssm import Mamba2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.config.bat.config import cfg, update_config_from_file
from lib.models.bat import build_batrack
from lib.train.admin import env_settings
import lib.train.base_functions as base_functions


def count_params(net):
    trainable = 0
    adap = 0
    head = 0
    other = 0
    for name, param in net.named_parameters():
        if not param.requires_grad:
            continue
        numel = param.numel()
        trainable += numel
        if "adap" in name:
            adap += numel
        elif name.startswith("box_head") or ".box_head" in name:
            head += numel
        else:
            other += numel
    return trainable, adap, head, other


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="lasher_train")
    args = parser.parse_args()

    update_config_from_file(f"experiments/bat/{args.config}.yaml")
    env = env_settings()
    net = build_batrack(cfg, training=True)

    base_functions.is_main_process = lambda: False
    optimizer, _ = base_functions.get_optimizer_scheduler(net, cfg)

    total = sum(param.numel() for param in net.parameters())
    trainable, adap, head, other = count_params(net)
    lrs = [group.get("lr", cfg.TRAIN.LR) for group in optimizer.param_groups]

    print(f"config={args.config}")
    print(f"lasher_train_dir={getattr(env, 'lasher_train_dir', env.lasher_dir)}")
    print(f"lasher_val_dir={getattr(env, 'lasher_val_dir', env.lasher_dir)}")
    print(f"lasher_test_dir={getattr(env, 'lasher_test_dir', env.lasher_dir)}")
    print(f"pretrain={cfg.MODEL.PRETRAIN_FILE}")
    print(f"train_mode={cfg.TRAIN.TRAIN_MODE}")
    print(f"batch_size={cfg.TRAIN.BATCH_SIZE}")
    print(f"epochs={cfg.TRAIN.EPOCH}")
    print(f"mamba2_count={sum(1 for module in net.modules() if isinstance(module, Mamba2))}")
    print(f"params_total={total}")
    print(f"params_trainable={trainable}")
    print(f"params_adap={adap}")
    print(f"params_head={head}")
    print(f"params_other={other}")
    print(f"lr_groups={lrs}")


if __name__ == "__main__":
    main()
