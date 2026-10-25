import os
from pathlib import Path


class EnvironmentSettings:
    """Training-time paths.

    Only ``LasHeR`` and ``VTUAV`` are required by the SGTA-Track configs.  All
    roots are derived from ``RGBT_DATA_ROOT`` (default ``/root/RGBTData``) so a
    fork does not inherit the authors' absolute paths; the per-dataset
    environment variables take precedence when set.
    """

    def __init__(self):
        repo_root = Path(__file__).resolve().parents[3]
        data_root = Path(os.environ.get("RGBT_DATA_ROOT", "/root/RGBTData"))
        lasher_root = data_root / "LasHeR"

        self.workspace_dir = str(repo_root)    # Base directory for saving network checkpoints.
        self.tensorboard_dir = str(repo_root / "tensorboard")    # Directory for tensorboard files.
        self.pretrained_networks = str(repo_root / "pretrained_networks")
        self.got10k_val_dir = str(data_root / "got10k" / "val")
        self.lasot_lmdb_dir = str(data_root / "lasot_lmdb")
        self.got10k_lmdb_dir = str(data_root / "got10k_lmdb")
        self.trackingnet_lmdb_dir = str(data_root / "trackingnet_lmdb")
        self.coco_lmdb_dir = str(data_root / "coco_lmdb")
        self.coco_dir = str(data_root / "coco")
        self.lasot_dir = str(data_root / "lasot")
        self.got10k_dir = str(data_root / "got10k" / "train")
        self.trackingnet_dir = str(data_root / "trackingnet")
        self.depthtrack_dir = str(data_root / "depthtrack" / "train")
        # LasHeR: <root>/LasHeR/{train,test}; the train split is what
        # DATA.TRAIN (LasHeR_train) reads, the test split is LasHeR_val here and
        # the benchmark split at test time.
        self.lasher_dir = os.environ.get("LASHER_TRAIN_HOME", str(lasher_root / "train"))
        self.lasher_train_dir = self.lasher_dir
        self.lasher_val_dir = os.environ.get(
            "LASHER_TEST_ROOT",
            os.environ.get("LASHER_HOME", str(lasher_root / "test")))
        self.lasher_test_dir = self.lasher_val_dir
        self.visevent_dir = str(data_root / "visevent" / "train")
        # VTUAV root; contains train/ test_ST/ test_LT/ (see README "Data").
        self.vtuav_dir = os.environ.get("VTUAV_HOME", str(data_root / "VTUAV"))
