"""VTUAV dataset loader for training (RGBT: RGB-Thermal UAV tracking).

Reference implementation: ``LFSTrack/lib/train/dataset/vtuav.py``
(itself ported from OmniAdapt / HMFT).

Expected on-disk layout (``VTUAV_HOME`` or ``env_settings().vtuav_dir``)::

    <VTUAV_HOME>/train/<seq>/{rgb,ir}/<frame>.jpg     rgb.txt  ir.txt
    <VTUAV_HOME>/test_ST/<seq>/{rgb,ir}/<frame>.jpg   rgb.txt  ir.txt
    <VTUAV_HOME>/test_LT/<seq>/{rgb,ir}/<frame>.jpg   rgb.txt  ir.txt

Notes
-----
* ``rgb.txt`` / ``ir.txt`` are whitespace separated; only the first four
  columns ``(x, y, w, h)`` are used, so both the 4-column test files and the
  8-column train files documented by LFSTrack are accepted.
* Frames of a sequence are addressed by their *annotation* index, i.e. row
  ``i`` of ``rgb.txt`` is the box of the i-th **annotated** frame. The image
  that belongs to it is looked up through
  :mod:`lib.train.dataset.vtuav_alignment`, which understands both VTUAV
  layouts: a folder holding only the annotated frames (``len(images) ==
  len(gt)``) and a folder holding every video frame (``len(images) ==
  10 * len(gt)``).

  This mirrors the HMFT / LFSTrack / CADTrack reference, which addresses the
  image as ``str(frame_id * 10 + init_idx).zfill(6) + '.jpg'``. Addressing the
  image by its *sorted ordinal* instead is a silent 10x misalignment whenever
  the folder is dense: training then looks healthy except that the IoU stalls
  around 0.15 and never recovers.

  An ``init_frame.npy`` next to this module (or ``$VTUAV_INIT_FRAME``) is
  honoured when present, so numbers match the reference implementation exactly.
* The sequence list can be pinned through
  ``lib/train/data_specs/vtuav_<split>.txt`` (one name per line). When that
  file is absent the directory listing is used, so the loader works on a fresh
  download. ``tools/dump_vtuav_seqlist.py`` writes those files.
"""
from __future__ import annotations

import os
import os.path
import random
from collections import OrderedDict
from typing import Dict, List, Optional

import numpy as np
import torch

from .base_video_dataset import BaseVideoDataset
from lib.train.admin import env_settings
from lib.train.dataset.depth_utils import get_x_frame
from lib.train.dataset.vtuav_alignment import (
    ALIGN_ANOMALIES,
    align_verbose,
    annotated_frame_ordinals,
    describe,
    load_init_frame,
    split_frame_names,
)


#: public split name -> directory under ``VTUAV_HOME``
SPLIT_DIRS = {
    "train": "train",
    "val_st": "test_ST",
    "val_lt": "test_LT",
    "test_st": "test_ST",
    "test_lt": "test_LT",
}


class VTUAV(BaseVideoDataset):
    """VTUAV RGBT tracking dataset (training / validation loaders)."""

    def __init__(self, root: Optional[str] = None, split: str = "train",
                 dtype: str = "rgbrgb", seq_ids: Optional[List[int]] = None,
                 data_fraction: Optional[float] = None, attr=None,
                 seq_list_file: Optional[str] = None):
        """
        args:
            root - path to the VTUAV root (contains train/ test_ST/ test_LT/).
            split - one of ``train``, ``val_st``, ``val_lt`` (aliases
                    ``test_st`` / ``test_lt`` are accepted).
            dtype - passed to :func:`get_x_frame`; ``rgbrgb`` yields (H, W, 6).
            seq_ids - optional subset of sequence indices.
            data_fraction - optional random fraction of the sequences.
            seq_list_file - explicit sequence-list file; defaults to
                    ``data_specs/vtuav_<split>.txt`` when it exists.
        """
        split = str(split).lower()
        if split not in SPLIT_DIRS:
            raise ValueError(
                f"Unknown VTUAV split: {split!r}; expected one of {sorted(SPLIT_DIRS)}")

        root = env_settings().vtuav_dir if root is None else root
        split_dir = os.path.join(root, SPLIT_DIRS[split])
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(
                f"VTUAV split directory not found: {split_dir} "
                f"(set VTUAV_HOME / env_settings().vtuav_dir)")

        super().__init__("VTUAV", split_dir)

        self.dtype = dtype
        self.split = split
        self.attr = attr
        self._frame_names_cache: Dict[str, List[str]] = {}
        self._gt_map_cache: Dict[tuple, List[int]] = {}
        # {sequence: first annotated frame}, shipped as init_frame.npy by the
        # reference implementations; empty when the file is not available.
        self._init_frame = load_init_frame(
            [os.path.dirname(os.path.realpath(__file__))])

        self.sequence_list = self._read_sequence_list(split_dir, split, seq_list_file)

        if seq_ids is None:
            seq_ids = list(range(len(self.sequence_list)))
        self.sequence_list = [self.sequence_list[i] for i in seq_ids]

        if data_fraction is not None:
            keep = int(len(self.sequence_list) * data_fraction)
            self.sequence_list = random.sample(self.sequence_list, keep)

        print(f"Dataset: VTUAV[{split}], root: {split_dir}, "
              f"seq length: {len(self.sequence_list)}")

    # ------------------------------------------------------------------ setup
    def _read_sequence_list(self, split_dir: str, split: str,
                            seq_list_file: Optional[str]) -> List[str]:
        if seq_list_file is None:
            default = os.path.join(
                os.path.dirname(os.path.realpath(__file__)), "..",
                "data_specs", f"vtuav_{split}.txt")
            seq_list_file = default if os.path.isfile(default) else None

        if seq_list_file is not None:
            with open(seq_list_file, "r", encoding="utf-8") as handle:
                names = [line.strip() for line in handle if line.strip()]
            missing = [n for n in names if not os.path.isdir(os.path.join(split_dir, n))]
            if missing:
                raise FileNotFoundError(
                    f"{len(missing)} sequence(s) from {seq_list_file} are missing under "
                    f"{split_dir}, e.g. {missing[:3]}")
            return names

        names = [n for n in sorted(os.listdir(split_dir))
                 if os.path.isdir(os.path.join(split_dir, n))]
        if not names:
            raise RuntimeError(f"no sequence directory found under {split_dir}")
        return names

    # ------------------------------------------------------------- dataset api
    def get_name(self) -> str:
        return "vtuav"

    def has_class_info(self) -> bool:
        return True

    def has_occlusion_info(self) -> bool:
        return True

    def _get_sequence_list(self):
        return list(self.sequence_list)

    def _get_sequence_path(self, seq_id: int) -> str:
        return os.path.join(self.root, self.sequence_list[seq_id])

    # ---------------------------------------------------------------- geometry
    def _read_bb_anno(self, seq_path: str) -> torch.Tensor:
        """Read ``rgb.txt`` (fallback ``ir.txt``) as an ``(N, 4)`` tensor.

        Parsed without pandas on purpose: the GT is plain whitespace-separated
        numbers, and this keeps the loader independent of the pandas build that
        happens to be installed on the training machine.

        One row is produced per non-empty line — malformed lines become an
        all-zero (hence invalid) box instead of being dropped, so row ``i``
        always corresponds to the i-th frame and the ``valid`` mask still lines
        up with the frame ordinals.
        """
        anno_file = os.path.join(seq_path, "rgb.txt")
        if not os.path.isfile(anno_file):
            anno_file = os.path.join(seq_path, "ir.txt")
        if not os.path.isfile(anno_file):
            raise FileNotFoundError(f"missing VTUAV annotation (rgb.txt/ir.txt): {seq_path}")

        boxes: List[List[float]] = []
        with open(anno_file, "r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip().replace(",", " ").replace("\t", " ")
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 4:
                    boxes.append([0.0, 0.0, 0.0, 0.0])
                    continue
                try:
                    boxes.append([float(v) for v in parts[:4]])   # (x, y, w, h)
                except ValueError:
                    boxes.append([0.0, 0.0, 0.0, 0.0])

        if not boxes:
            return torch.zeros((0, 4), dtype=torch.float32)
        return torch.from_numpy(np.asarray(boxes, dtype=np.float32))

    def _read_target_visible(self, seq_path: str) -> torch.Tensor:
        occlusion_file = os.path.join(seq_path, "absence.label")
        cover_file = os.path.join(seq_path, "cover.label")
        if not (os.path.isfile(occlusion_file) and os.path.isfile(cover_file)):
            return self._read_bb_anno(seq_path) > 0

        occlusion = torch.ByteTensor(
            [int(line.strip().split()[0]) for line in open(occlusion_file) if line.strip()])
        cover = torch.ByteTensor(
            [int(line.strip().split()[0]) for line in open(cover_file) if line.strip()])
        return ~occlusion & (cover > 0)

    def get_sequence_info(self, seq_id: int) -> Dict[str, torch.Tensor]:
        seq_path = self._get_sequence_path(seq_id)
        bbox = self._read_bb_anno(seq_path)
        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = valid.clone().byte()
        return {"bbox": bbox, "valid": valid, "visible": visible}

    # ------------------------------------------------------------------ frames
    def _frame_names(self, seq_path: str) -> List[str]:
        names = self._frame_names_cache.get(seq_path)
        if names is None:
            rgb_dir = os.path.join(seq_path, "rgb")
            if not os.path.isdir(rgb_dir):
                raise FileNotFoundError(f"missing rgb/ directory: {rgb_dir}")
            # Drop housekeeping files (macOS AppleDouble '._<name>.jpg',
            # '.DS_Store', Windows 'Thumbs.db'): they carry an image extension
            # but are not frames, they sort BEFORE the real frames, and reading
            # one raises.  See vtuav_alignment.split_frame_names.
            names, skipped = split_frame_names(sorted(os.listdir(rgb_dir)))
            if skipped:
                sample = ", ".join(skipped[:3]) + ("..." if len(skipped) > 3 else "")
                print(f"[VTUAV][FRAMES][WARN] {os.path.basename(seq_path)}: skipping "
                      f"{len(skipped)} non-frame file(s) in rgb/: {sample}", flush=True)
            self._frame_names_cache[seq_path] = names
        return names

    def _get_frame_path(self, seq_path: str, frame_id: int):
        names = self._frame_names(seq_path)
        if frame_id < 0 or frame_id >= len(names):
            raise IndexError(f"frame {frame_id} out of range for {seq_path} ({len(names)} frames)")
        frame_name = names[frame_id]
        return (os.path.join(seq_path, "rgb", frame_name),
                os.path.join(seq_path, "ir", frame_name))

    def _annotated_ordinals(self, seq_path: str, seq_name: str, n_gt: int) -> List[int]:
        """Annotation row -> ordinal into the sorted ``rgb/`` listing.

        Identity when ``rgb/`` already holds exactly the annotated frames; a
        stride-10 mapping when the folder holds every video frame (see the
        module docstring).  Computed once per sequence and cached.
        """
        key = (seq_path, n_gt)
        mapping = self._gt_map_cache.get(key)
        if mapping is not None:
            return mapping

        names = self._frame_names(seq_path)
        mapping, info = annotated_frame_ordinals(
            names, n_gt, offset=self._init_frame.get(seq_name), seq_name=seq_name)
        self._gt_map_cache[key] = mapping

        # Quiet by default: the mapping is the normal case, not news.  Only real
        # data problems are reported, and the full per-sequence line is available
        # with VTUAV_ALIGN_VERBOSE=1 when something needs investigating.
        if align_verbose():
            print(f"[VTUAV][align] {describe(info)}")
        elif info.get("missing") or info.get("reason") in ALIGN_ANOMALIES:
            print(f"[VTUAV][align][WARN] {describe(info)}")
        return mapping

    def _get_frame(self, seq_path: str, frame_id: int) -> np.ndarray:
        rgb_frame_path, ir_frame_path = self._get_frame_path(seq_path, frame_id)
        if not os.path.isfile(ir_frame_path):
            raise FileNotFoundError(f"missing infrared frame: {ir_frame_path}")
        return get_x_frame(rgb_frame_path, ir_frame_path, dtype=self.dtype)   # (h, w, 6)

    def get_frames(self, seq_id, frame_ids, anno=None):
        """Load frames for *annotation* indices ``frame_ids``.

        ``frame_ids`` address annotation rows (that is the space the sampler
        works in: it samples within ``len(anno['valid'])``), so they are
        translated to image ordinals before touching the file system, while
        ``anno`` is indexed directly by ``frame_ids``.
        """
        seq_path = self._get_sequence_path(seq_id)
        seq_name = self.sequence_list[seq_id]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        n_gt = len(anno["bbox"]) if "bbox" in anno else len(self._frame_names(seq_path))
        ordinals = self._annotated_ordinals(seq_path, seq_name, n_gt)
        if not ordinals:
            raise IndexError(f"no frame/GT alignment for {seq_path} (gt rows={n_gt})")

        frame_list = []
        for f_id in frame_ids:
            if f_id < 0 or f_id >= len(ordinals):
                raise IndexError(
                    f"annotation index {f_id} out of range for {seq_name} "
                    f"(gt rows={len(ordinals)})")
            frame_list.append(self._get_frame(seq_path, ordinals[f_id]))

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        object_meta = OrderedDict({
            "object_class_name": None,
            "motion_class": None,
            "major_class": None,
            "root_class": None,
            "motion_adverb": None,
        })
        return frame_list, anno_frames, object_meta
