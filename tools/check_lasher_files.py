#!/usr/bin/env python3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.train.admin import env_settings
from lib.train.dataset.LasHeR_trainingSet import LasHeR_trainingSet
from lib.train.dataset.LasHeR_testingSet import LasHeR_testingSet


def check_dataset(name, dataset, root):
    missing = []
    for seq in dataset.sequence_list:
        seq_path = Path(root) / seq
        required = [seq_path / "init.txt", seq_path / "visible", seq_path / "infrared"]
        bad = [str(path) for path in required if not path.exists()]
        if bad:
            missing.append((seq, bad))
    print(f"{name}: root={root}")
    print(f"{name}: sequences={len(dataset.sequence_list)} missing={len(missing)}")
    for seq, bad in missing[:20]:
        print(f"{name}: missing {seq}: {bad}")
    if len(missing) > 20:
        print(f"{name}: ... {len(missing) - 20} more missing sequences")
    return len(missing)


def main():
    env = env_settings()
    train_root = getattr(env, "lasher_train_dir", env.lasher_dir)
    val_root = getattr(env, "lasher_val_dir", getattr(env, "lasher_test_dir", env.lasher_dir))
    train = LasHeR_trainingSet(train_root)
    val = LasHeR_testingSet(val_root)
    missing = check_dataset("LasHeR_train", train, train_root)
    missing += check_dataset("LasHeR_val", val, val_root)
    raise SystemExit(1 if missing else 0)


if __name__ == "__main__":
    main()
