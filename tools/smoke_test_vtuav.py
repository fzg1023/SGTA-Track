#!/usr/bin/env python3
"""Synthetic end-to-end smoke test for the VTUAV integration.

Builds a tiny fake VTUAV tree (train / test_ST / test_LT) with real JPEGs and
GT files, then exercises:
  * lib.train.dataset.VTUAV      (training loader)
  * test_sgta.py sequence_dirs   (test-side discovery + frame counting)
  * experiments/bat/vtuav_train.yaml  (config loading)
No real VTUAV data is required.
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import cv2

PRJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PRJ_ROOT)

FAIL = []


def check(label, cond, extra=""):
    print(("  [OK]   " if cond else "  [FAIL] ") + label + (("  " + str(extra)) if extra else ""))
    if not cond:
        FAIL.append(label)


def make_sequence(seq_dir, n_frames, stride=10, init=0, gt_cols=4, occluded_frame=None,
                  junk=False):
    """One annotation per 10 images by default (stride=10 naming, only annotated
    frames on disk).  ``stride=1`` produces a DENSE folder -- every video frame
    present -- which is the layout that used to be indexed 10x off.

    ``junk=True`` also drops macOS/Windows housekeeping files into rgb/ and ir/:
    AppleDouble sidecars ``._<name>.jpg`` (same extension as a frame, sort BEFORE
    it, ``cv2.imread`` returns None) plus a ``.DS_Store``.
    """
    os.makedirs(os.path.join(seq_dir, "rgb"), exist_ok=True)
    os.makedirs(os.path.join(seq_dir, "ir"), exist_ok=True)
    rows = []
    for i in range(n_frames):
        name = f"{i * stride + init:06d}.jpg"
        img = np.full((32, 32, 3), (i * 7) % 255, dtype=np.uint8)
        cv2.imwrite(os.path.join(seq_dir, "rgb", name), img)
        cv2.imwrite(os.path.join(seq_dir, "ir", name), img)
        if occluded_frame is not None and i == occluded_frame:
            box = [0.0, 0.0, 0.0, 0.0]
        else:
            box = [float(10 + i), float(20 + i), 15.0, 25.0]
        row = box + [float(99)] * (gt_cols - 4)
        rows.append(" ".join(f"{v:g}" for v in row))
    for gt in ("rgb.txt", "ir.txt"):
        with open(os.path.join(seq_dir, gt), "w", encoding="utf-8") as handle:
            handle.write("\n".join(rows) + "\n")
    if junk:
        for name in (f"{init:06d}.jpg", f"{init + stride:06d}.jpg"):
            for modal in ("rgb", "ir"):
                with open(os.path.join(seq_dir, modal, "._" + name), "wb") as handle:
                    handle.write(b"\x00\x05\x16\x07")          # AppleDouble magic
        with open(os.path.join(seq_dir, "rgb", ".DS_Store"), "wb") as handle:
            handle.write(b"\x00\x00\x00\x01Bud1")


def make_dense_sequence(seq_dir, n_annot, per_gt=10, init=0, gt_cols=4):
    """DENSE layout: ``per_gt`` images per annotation (the VTUAV reality where
    every video frame was extracted and the GT only annotates every 10th).

    Returns the sorted expected frame names for annotation rows 0..n_annot-1.
    """
    os.makedirs(os.path.join(seq_dir, "rgb"), exist_ok=True)
    os.makedirs(os.path.join(seq_dir, "ir"), exist_ok=True)
    n_images = (n_annot - 1) * per_gt + 1
    for i in range(n_images):
        name = f"{i + init:06d}.jpg"
        img = np.full((32, 32, 3), (i * 7) % 255, dtype=np.uint8)
        cv2.imwrite(os.path.join(seq_dir, "rgb", name), img)
        cv2.imwrite(os.path.join(seq_dir, "ir", name), img)
    rows = []
    for i in range(n_annot):
        box = [float(10 + i), float(20 + i), 15.0, 25.0]
        rows.append(" ".join(f"{v:g}" for v in box + [float(99)] * (gt_cols - 4)))
    for gt in ("rgb.txt", "ir.txt"):
        with open(os.path.join(seq_dir, gt), "w", encoding="utf-8") as handle:
            handle.write("\n".join(rows) + "\n")
    return [f"{init + i * per_gt:06d}.jpg" for i in range(n_annot)]


def build_tree(root):
    # train: 8-column GT (as documented by LFSTrack), one occluded frame
    make_sequence(os.path.join(root, "train", "car_016"), 6, gt_cols=8, occluded_frame=3)
    make_sequence(os.path.join(root, "train", "car_031"), 4, gt_cols=8)
    # DENSE train sequence: every frame on disk, one annotation per 10 images.
    # This is the layout that used to mis-align GT by 10x (IoU stuck at ~0.15).
    make_dense_sequence(os.path.join(root, "train", "dense_077"), 6, per_gt=10)
    # test splits: 4-column GT
    # animal_001 additionally carries macOS/Windows housekeeping files, so the
    # frame listing has to filter them out (they are NOT frames even though
    # '._000000.jpg' ends in .jpg, and they sort before the real frames).
    for seq in ("animal_001", "car_004"):
        make_sequence(os.path.join(root, "test_ST", seq), 5, junk=(seq == "animal_001"))
    make_sequence(os.path.join(root, "test_LT", "pedestrian_144"), 5)
    # DENSE test sequence with a non-zero start index (needs init_frame offset)
    make_dense_sequence(os.path.join(root, "test_ST", "dense_009"), 5, per_gt=10, init=37)


def main():
    tmp_root = tempfile.mkdtemp(prefix="fake_vtuav_")
    try:
        build_tree(tmp_root)
        print(f"\nfake VTUAV root: {tmp_root}")

        # ---------------------------------------------------------- training
        print("\n[1] lib.train.dataset.VTUAV (training loader)")
        from lib.train.dataset import VTUAV

        ds = VTUAV(tmp_root, split="train", dtype="rgbrgb")
        check("sequence_list from directory listing",
              ds.sequence_list == ["car_016", "car_031", "dense_077"], ds.sequence_list)

        info = ds.get_sequence_info(0)
        check("bbox shape (n_frames, 4)", tuple(info["bbox"].shape) == (6, 4), tuple(info["bbox"].shape))
        check("GT first row = x,y,w,h", [round(float(v)) for v in info["bbox"][0]] == [10, 20, 15, 25],
              info["bbox"][0].tolist())
        check("8-column GT truncated to 4", info["bbox"].shape[1] == 4)
        check("occluded frame marked invalid", bool(info["valid"][3]) is False)
        check("valid count", int(info["valid"].sum()) == 5, int(info["valid"].sum()))

        frames, anno, meta = ds.get_frames(0, [0, 5], anno=info)
        check("frame stacked to (H, W, 6)", frames[0].shape == (32, 32, 6), frames[0].shape)
        check("anno_frames per key", len(anno["bbox"]) == 2)
        check("object_meta is an OrderedDict", hasattr(meta, "keys"))

        # frame ordinal indexing must survive the stride-10 naming
        p0 = ds._get_frame_path(ds._get_sequence_path(0), 0)
        p5 = ds._get_frame_path(ds._get_sequence_path(0), 5)
        check("ordinal indexing (frame 0)", os.path.basename(p0[0]) == "000000.jpg", p0[0])
        check("ordinal indexing (frame 5, stride 10)", os.path.basename(p5[0]) == "000050.jpg", p5[0])
        check("rgb/ir names identical", os.path.basename(p0[0]) == os.path.basename(p0[1]))

        # pinned sequence list path
        spec = os.path.join(tmp_root, "spec.txt")
        with open(spec, "w", encoding="utf-8") as handle:
            handle.write("car_031\n")
        ds2 = VTUAV(tmp_root, split="train", seq_list_file=spec)
        check("pinned seq_list_file honoured", ds2.sequence_list == ["car_031"], ds2.sequence_list)

        ds3 = VTUAV(tmp_root, split="val_st")
        check("val_st resolves to test_ST",
              ds3.sequence_list == ["animal_001", "car_004", "dense_009"],
              ds3.sequence_list)
        # the training loader lists rgb/ itself, so it needs the same filter
        junk_path = ds3._get_sequence_path(ds3.sequence_list.index("animal_001"))
        junk_names = ds3._frame_names(junk_path)
        check("junk: training loader ignores AppleDouble sidecars",
              len(junk_names) == 5 and junk_names[0] == "000000.jpg", junk_names)

        # ---------------------------------------- DENSE layout (the 10x trap)
        print("\n[1b] VTUAV frame <-> annotation alignment")
        dense_id = ds.sequence_list.index("dense_077")
        info_dense = ds.get_sequence_info(dense_id)
        check("dense sequence: len(valid) == number of annotations (sampler space)",
              int(info_dense["valid"].shape[0]) == 6, int(info_dense["valid"].shape[0]))
        frames_d, anno_d, _ = ds.get_frames(dense_id, [0, 1, 2], anno=info_dense)
        # each synthetic frame stores its own image index as (i*7)%255
        vals = [int(np.asarray(f).reshape(-1, 6)[0, 0]) for f in frames_d]
        expected_vals = [(i * 7) % 255 for i in (0, 10, 20)]
        check("dense: annotation rows 0/1/2 load the images of frames 0/10/20",
              vals == expected_vals, (vals, expected_vals))
        check("dense: 6-channel stack", frames_d[0].shape == (32, 32, 6), frames_d[0].shape)
        check("dense: GT row index is untouched by the mapping",
              [int(float(v[0])) for v in anno_d["bbox"]] == [10, 11, 12],
              [int(float(v[0])) for v in anno_d["bbox"]])

        from lib.train.dataset.vtuav_alignment import annotated_frame_ordinals
        ords, ainfo = annotated_frame_ordinals(
            [f"{i:06d}.jpg" for i in range(30)], n_gt=3, seq_name="toy")
        check("helper: dense 30 images / 3 gt -> rows on 0/10/20",
              ords == [0, 10, 20] and ainfo["stride"] == 10, (ords, ainfo["stride"]))
        ords, ainfo = annotated_frame_ordinals(
            [f"{i:06d}.jpg" for i in range(0, 30, 10)], n_gt=3, seq_name="toy")
        check("helper: pre-extracted (images == gt) -> identity",
              ords == [0, 1, 2] and not ainfo["scaled"], (ords, ainfo["scaled"]))
        # deliberately irregular folder (odd naming stride, 3 images, 2 rows): no
        # candidate spacing fits, so the loader must fall back to the nearest
        # frame and REPORT it instead of raising
        ords, ainfo = annotated_frame_ordinals(
            ["000000.jpg", "000007.jpg", "000030.jpg"], n_gt=2, seq_name="toy")
        check("helper: an irregular folder falls back to the nearest frame",
              len(ords) == 2 and ainfo["missing"] >= 1, (ords, ainfo["missing"]))

        # init_frame offset: the video starts at 37, not 0
        sidecar = os.path.join(tmp_root, "init_frame.json")
        with open(sidecar, "w", encoding="utf-8") as handle:
            json.dump({"dense_009": 37}, handle)
        os.environ["VTUAV_INIT_FRAME"] = sidecar
        ds_off = VTUAV(tmp_root, split="val_st")
        off_id = ds_off.sequence_list.index("dense_009")
        info_off = ds_off.get_sequence_info(off_id)
        names_off = ds_off._frame_names(ds_off._get_sequence_path(off_id))
        ords_off = ds_off._annotated_ordinals(ds_off._get_sequence_path(off_id),
                                              "dense_009", int(info_off["valid"].shape[0]))
        check("init_frame.json offset -> rows start at 000037",
              [names_off[o] for o in ords_off[:3]] == ["000037.jpg", "000047.jpg", "000057.jpg"],
              [names_off[o] for o in ords_off[:3]])
        del os.environ["VTUAV_INIT_FRAME"]

        # ------------------------------------------------------------- test
        print("\n[2] RGBT_workspace/test_sgta.py (test-side discovery)")
        os.environ["VTUAV_ST_ROOT"] = os.path.join(tmp_root, "test_ST")
        os.environ["VTUAV_LT_ROOT"] = os.path.join(tmp_root, "test_LT")
        spec_tl = importlib.util.spec_from_file_location(
            "test_sgta_mod", os.path.join(PRJ_ROOT, "RGBT_workspace", "test_sgta.py"))
        mod = importlib.util.module_from_spec(spec_tl)
        spec_tl.loader.exec_module(mod)

        check("vtuav_st in DATASET_CFG", "vtuav_st" in mod.DATASET_CFG)
        seqs = mod.sequence_dirs("vtuav_st")
        check("sequence_dirs finds flat sequences",
              [os.path.basename(s) for s in seqs] == ["animal_001", "car_004", "dense_009"],
              [os.path.basename(s) for s in seqs])
        check("count_frames", mod.count_frames(seqs[0], "vtuav_st") == 5,
              mod.count_frames(seqs[0], "vtuav_st"))
        # macOS junk: '._000000.jpg' ends in .jpg and sorts BEFORE the real
        # frames, so an unfiltered listing would feed it to cv2.imread (which
        # returns None) as the very first frame -- and shift every index by one.
        listed_rgb = mod.list_frames(seqs[0], "rgb")
        listed_ir = mod.list_frames(seqs[0], "ir")
        check("junk: list_frames drops AppleDouble/.DS_Store sidecars",
              len(listed_rgb) == 5 and os.path.basename(listed_rgb[0]) == "000000.jpg",
              [os.path.basename(p) for p in listed_rgb])
        check("junk: ir/ listing drops them too", len(listed_ir) == 5,
              [os.path.basename(p) for p in listed_ir])
        check("junk: read_rgbt on the first listed frame works",
              mod.read_rgbt(listed_rgb[0], listed_ir[0]).shape == (32, 32, 6),
              mod.read_rgbt(listed_rgb[0], listed_ir[0]).shape)
        check("GT parsed as ltwh", mod.read_boxes(os.path.join(seqs[0], "rgb.txt"))[0] ==
              [10.0, 20.0, 15.0, 25.0], mod.read_boxes(os.path.join(seqs[0], "rgb.txt"))[0])
        check("official metrics = PR/NPS/SS",
              mod.official_metric_names("vtuav_st") == ["PR", "NPS", "SS"],
              mod.official_metric_names("vtuav_st"))
        check("official source = seq", mod.official_metric_source("vtuav_st") == "seq")
        check("lt sequence_dirs", len(mod.sequence_dirs("vtuav_lt")) == 1)

        # the test side must pick the annotation's own frame too (dense_009 starts
        # the video at frame 37, so the rows map onto 000037/000047/...)
        paths = [os.path.join(seqs[2], "rgb", n) for n in
                 sorted(os.listdir(os.path.join(tmp_root, "test_ST", "dense_009", "rgb")))]
        ords_t, info_t = mod.annotated_frame_ordinals(paths, n_gt=5, seq_name="dense_009")
        check("test side: dense 41 images / 5 gt -> rows every 10 frames",
              info_t["stride"] == 10 and len(ords_t) == 5, (info_t["stride"], len(ords_t)))
        check("test side: evaluates annotation rows, not the first N frames",
              [os.path.basename(paths[o]) for o in ords_t[:3]] ==
              ["000037.jpg", "000047.jpg", "000057.jpg"],
              [os.path.basename(paths[o]) for o in ords_t[:3]])

        # nested (original release) layout must be auto-detected
        nested_root = os.path.join(tmp_root, "nested", "short-term", "test_ST_001", "animal_001")
        make_sequence(nested_root, 3)
        os.environ["VTUAV_ST_ROOT"] = os.path.join(tmp_root, "nested", "short-term")
        mod.DATASET_CFG["vtuav_st"]["root"] = os.path.join(tmp_root, "nested", "short-term")
        nested = mod.sequence_dirs("vtuav_st")
        check("nested two-level layout auto-detected", len(nested) == 1, nested)

        # ----------------------------------------------------------- config
        print("\n[3] experiments/bat/vtuav_train.yaml")
        from lib.config.bat.config import cfg, update_config_from_file
        update_config_from_file(os.path.join(PRJ_ROOT, "experiments", "bat", "vtuav_train.yaml"))
        check("train dataset name", cfg.DATA.TRAIN.DATASETS_NAME == ["VTUAV_train"],
              cfg.DATA.TRAIN.DATASETS_NAME)
        check("val dataset name", cfg.DATA.VAL.DATASETS_NAME == ["VTUAV_val_st"],
              cfg.DATA.VAL.DATASETS_NAME)
        check("all four SGTA modules on",
              all([cfg.MODEL.SGTA.ENABLED, cfg.MODEL.SGTA.USE_MAMBA_FUSION,
                   cfg.MODEL.SGTA.USE_TEMPORAL_MAMBA, cfg.MODEL.SGTA.USE_LIQUID_CONTROLLER,
                   cfg.MODEL.SGTA.USE_MOE, cfg.MODEL.SGTA.USE_TEMPLATE_ROUTER,
                   cfg.MODEL.SGTA.ENABLE_TEMPLATE_UPDATE]))
        check("legacy temporal predictor disabled",
              cfg.MODEL.SGTA.USE_LEGACY_TEMPORAL_PREDICTOR is False)
        check("EVAL_DATASET = vtuav_st", cfg.TRAIN.EVAL_DATASET == "vtuav_st", cfg.TRAIN.EVAL_DATASET)

        # -------------------------------------------------- dataset registry
        print("\n[4] dataset registries")
        import lib.train.dataset as train_ds
        check("VTUAV in lib.train.dataset registry", train_ds.VTUAV is VTUAV)
        from lib.test.evaluation import datasets as test_ds
        check("vtuav_st/vtuav_lt in test registry",
              "vtuav_st" in test_ds.dataset_dict and "vtuav_lt" in test_ds.dataset_dict)
        import lib.train.admin.local as train_local
        check("env_settings().vtuav_dir exists", hasattr(train_local.EnvironmentSettings(), "vtuav_dir"))
        import lib.test.evaluation.local as test_local
        check("test env has vtuav_path", hasattr(test_local.local_env_settings(), "vtuav_path"))

    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    print("\n" + "=" * 70)
    if FAIL:
        print(f"SMOKE TEST FAILED: {len(FAIL)} check(s): {FAIL}")
        return 1
    print("SMOKE TEST PASSED: all checks OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
