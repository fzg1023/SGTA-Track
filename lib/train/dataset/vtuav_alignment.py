"""Frame <-> annotation alignment for VTUAV.

The trap this module exists for
------------------------------
VTUAV names frames with a **stride of 10** and annotates **one frame per 10
images**::

    rgb/000000.jpg   <- annotation row 0
    rgb/000010.jpg   <- annotation row 1
    rgb/000020.jpg   <- annotation row 2
    ...

Depending on how the dataset was unpacked, ``rgb/`` may contain

* only the annotated frames  (``len(images) == len(gt)``), or
* *every* frame of the video (``len(images) ~= 10 * len(gt)``, step 1 naming).

Addressing frames by their *sorted ordinal* (``sorted(names)[i]``) is only
correct in the first case.  In the second case frame ``i`` is **10x off** from
annotation row ``i``, which shows up as a tracker that cannot learn (IoU stalls
around 0.15) while everything else looks healthy.

The HMFT / LFSTrack / CADTrack family sidesteps this by addressing the image
through the annotation index::

    frame_path = rgb/ str(frame_id * 10 + init_idx).zfill(6) + '.jpg'

with a per-sequence ``init_frame.npy``.  This module reproduces that rule but
**derives** the stride/offset from the data instead of hard-coding 10, and
degrades gracefully:

* ``len(images) == len(gt)``  -> identity mapping (stride 1, no change);
* otherwise stride := round((last_stem - first_stem) / (len(gt) - 1)) and the
  image with stem ``first_stem + stride * row`` is used;
* if ``init_frame.npy`` (or an explicit offset) is available it takes
  precedence, so results match the reference implementation exactly;
* a stem that is missing on disk falls back to the nearest available frame
  instead of raising, so one truncated folder cannot kill a training run.

Everything here is stdlib-only on purpose: ``RGBT_workspace/test_sgta.py``
imports it too, and that file must stay importable without torch.
"""
from __future__ import annotations

import bisect
import math
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

try:  # optional, only used when present next to the dataset module
    import numpy as _np  # type: ignore
except Exception:  # pragma: no cover - numpy is a hard dep of the project
    _np = None

#: files the reference implementations ship: {sequence_name: first_frame_index}
INIT_FRAME_FILENAMES = ("init_frame.npy", "init_frame.json")

#: reason codes that mean "the data is not shaped the way the loader expects"
ALIGN_ANOMALIES = ("images_lt_gt", "non_numeric_names", "degenerate_span", "empty")

#: file extensions the loaders accept as frames (lower case, leading dot)
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

#: housekeeping files that are never frames.  The important one is macOS'
#: AppleDouble sidecar ``._<name>``: it is written next to every file when a
#: folder is copied through a non-HFS filesystem, it KEEPS the ``.jpg``
#: extension (so an extension check alone lets it through) and ``cv2.imread``
#: returns ``None`` for it.  Because ``._000000.jpg`` sorts BEFORE
#: ``000000.jpg`` (``'.' < '0'``), an unfiltered listing shifts every frame
#: index by one -- which silently misaligns the GT as well, not just crashes.
JUNK_NAME_PREFIXES = (".",)
JUNK_NAME_EXACT = ("thumbs.db", "desktop.ini", "ehthumbs.db", "icon\r")


def is_junk_name(name: str) -> bool:
    """True for a known non-frame file (AppleDouble, ``.DS_Store``, ``Thumbs.db``)."""
    base = os.path.basename(name)
    if base.startswith(JUNK_NAME_PREFIXES):
        return True
    return base.lower() in JUNK_NAME_EXACT


def is_frame_name(name: str, exts: Sequence[str] = IMAGE_EXTS) -> bool:
    """True when *name* looks like a frame the loaders should try to read."""
    if is_junk_name(name):
        return False
    return os.path.splitext(name)[1].lower() in tuple(exts)


def split_frame_names(names: Sequence[str],
                      exts: Sequence[str] = IMAGE_EXTS
                      ) -> Tuple[List[str], List[str]]:
    """``(frames, skipped)`` for a directory listing, preserving input order.

    Every caller that lists a modality folder should use this before treating
    the result as frames; ``skipped`` is what to report, and a non-empty list
    means the folder carries files that are not frames.
    """
    frames: List[str] = []
    skipped: List[str] = []
    for name in names:
        (frames if is_frame_name(name, exts) else skipped).append(name)
    return frames, skipped


def filter_frame_names(names: Sequence[str],
                       exts: Sequence[str] = IMAGE_EXTS) -> List[str]:
    """Only the frame names from *names* (see :func:`split_frame_names`)."""
    return split_frame_names(names, exts)[0]


def align_verbose() -> bool:
    """True when ``VTUAV_ALIGN_VERBOSE=1`` is set in the environment.

    The alignment is silent by default: a stride mapping is the normal VTUAV
    situation, not a warning worth printing once per sequence (a training split
    has ~1000 of them).  Set the variable when investigating a dataset.
    """
    return os.environ.get("VTUAV_ALIGN_VERBOSE", "").strip().lower() in (
        "1", "true", "yes", "on")


_STEM_RE = re.compile(r"(\d+)")


def parse_stem(name: str) -> Optional[int]:
    """Return the numeric part of a frame file name, or ``None``.

    ``'000010.jpg'`` -> ``10``; ``'frame_7.png'`` -> ``7``; ``'abc.jpg'`` -> ``None``.
    """
    stem = os.path.splitext(os.path.basename(name))[0]
    if stem.isdigit():
        return int(stem)
    match = _STEM_RE.search(stem)
    if match is None:
        return None
    return int(match.group(1))


def load_init_frame(search_dirs: Sequence[str] = ()) -> Dict[str, int]:
    """Load ``{sequence: first_annotated_frame}`` from ``init_frame.npy``.

    Looks in ``search_dirs`` first (the dataset module's own directory by
    default), then in the ``VTUAV_INIT_FRAME`` environment variable if set.
    Returns an empty dict when nothing is found -- callers then fall back to the
    first frame actually present on disk.
    """
    candidates: List[str] = []
    env_path = os.environ.get("VTUAV_INIT_FRAME", "").strip()
    if env_path:
        candidates.append(env_path)
    for directory in search_dirs:
        for filename in INIT_FRAME_FILENAMES:
            candidates.append(os.path.join(directory, filename))

    for path in candidates:
        if not os.path.isfile(path):
            continue
        try:
            if path.endswith(".json"):
                import json
                with open(path, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
            else:
                if _np is None:
                    continue
                data = _np.load(path, allow_pickle=True).item()
            if isinstance(data, dict):
                return {str(k): int(v) for k, v in data.items()}
        except Exception:  # noqa: BLE001 - a broken sidecar must not be fatal
            continue
    return {}


def annotated_frame_ordinals(names: Sequence[str],
                             n_gt: int,
                             offset: Optional[int] = None,
                             seq_name: str = "") -> Tuple[List[int], Dict[str, object]]:
    """Map annotation rows to ordinals into ``names`` (already sorted).

    args:
        names    - sorted frame file names as found in ``rgb/``.
        n_gt     - number of annotation rows (GT length).
        offset    - first annotated frame's number (``init_frame.npy`` value).
                    ``None`` -> assume the first frame on disk is annotated.
        seq_name - only for diagnostics.

    returns:
        ``(ordinals, info)`` where ``ordinals[row]`` is an index into ``names``
        and ``info`` describes what was inferred (stride, scaling, missing).
    """
    info: Dict[str, object] = {
        "seq": seq_name, "n_images": len(names), "n_gt": n_gt,
        "stride": 1, "step": 1, "img_stride": 1, "offset": None,
        "scaled": False, "missing": 0,
        "reason": "",
    }
    if n_gt <= 0 or not names:
        info["reason"] = "empty"
        return [], info

    if len(names) <= n_gt:
        # Either already one image per annotation, or fewer images than GT rows
        # (truncated folder): keep the legacy ordinal behaviour and let the
        # caller clip to the shorter list.
        info["reason"] = "one_to_one" if len(names) == n_gt else "images_lt_gt"
        info["truncated"] = True
        return list(range(len(names))), info

    stems = [parse_stem(name) for name in names]
    if any(stem is None for stem in stems):
        # Unparsable names: nothing better than the ordinal mapping, but say so.
        info["reason"] = "non_numeric_names"
        info["truncated"] = True
        return list(range(n_gt)), info

    stems_int = [int(s) for s in stems]  # type: ignore[arg-type]
    first = int(offset) if offset is not None else stems_int[0]
    n_img = len(stems_int)

    # Naming stride: 1 for a folder holding every video frame, 10 for the
    # pre-extracted layout. gcd() rather than the mean so one irregular gap
    # (e.g. a dropped frame) cannot distort it.
    diffs = [b - a for a, b in zip(stems_int, stems_int[1:]) if b > a]
    img_stride = max(1, math.gcd(*diffs)) if diffs else 1

    # How many images separate two annotations.  Two estimates, because either
    # edge of the folder can be partial:
    #   span  = (n_img-1)/(n_gt-1)  exact when the folder ends ON the annotation
    #                               grid, i.e. the extraction covers exactly the
    #                               annotated range
    #   ratio = n_img/n_gt          better when the folder carries extra frames
    #                               past the last annotation
    #   (the two agree whenever n_gt is large, i.e. for every real sequence)
    grid_exact = n_gt > 1 and (n_img - 1) % (n_gt - 1) == 0
    span_est = max(1, int(round((n_img - 1) / float(n_gt - 1)))) if n_gt > 1 else 1
    ratio_est = max(1, int(round(n_img / float(n_gt))))
    primary = span_est if grid_exact else ratio_est

    # Keep a small neighbourhood of candidates and prefer the one landing on
    # real files; the first entry (the primary estimate) wins ties.
    candidates: List[int] = []
    for base in (primary, span_est, ratio_est):
        for delta in (0, -1, 1, -2, 2):
            cand = base + delta
            if 1 <= cand <= max(1, n_img) and cand not in candidates:
                candidates.append(cand)

    index_by_stem = {stem: i for i, stem in enumerate(stems_int)}
    best_step, best_missing = candidates[0], None
    for cand in candidates:
        miss = sum(1 for row in range(n_gt)
                   if (first + img_stride * cand * row) not in index_by_stem)
        # strict "<" keeps the earlier (higher-priority) candidate on a tie
        if best_missing is None or miss < best_missing:
            best_step, best_missing = cand, miss
    step = best_step

    # Annotation row r sits on the image whose number is first + row*step*stride.
    stem_step = img_stride * step
    info.update(stride=stem_step, step=step, img_stride=img_stride,
                offset=first, grid_exact=grid_exact, span_est=span_est,
                ratio_est=ratio_est, missing_candidates=best_missing,
                scaled=bool(step > 1 or n_img > n_gt or stem_step > 1))

    ordinals: List[int] = []
    missing = 0
    for row in range(n_gt):
        target = first + stem_step * row
        hit = index_by_stem.get(target)
        if hit is None:
            # nearest available frame (never crash on a gap)
            pos = bisect.bisect_left(stems_int, target)
            if pos >= len(stems_int):
                pos = len(stems_int) - 1
            elif pos > 0 and abs(stems_int[pos - 1] - target) <= abs(stems_int[pos] - target):
                pos -= 1
            hit = pos
            missing += 1
        ordinals.append(hit)
    info["missing"] = missing
    info["reason"] = "stride_mapping"
    return ordinals, info


def describe(info: Dict[str, object]) -> str:
    """One-line human readable summary of :func:`annotated_frame_ordinals`."""
    return ("seq={seq} images={n_images} gt_rows={n_gt} images_per_gt={step} "
            "naming_stride={img_stride} row_stride={stride} offset={offset} "
            "mapped={scaled} missing={missing} ({reason})").format(**info)


def identity_mapping(names: Sequence[str], n_gt: int) -> List[int]:
    """Convenience: the mapping used when no stride correction is needed."""
    n = min(len(names), n_gt)
    return list(range(n))


def dense_annotation_index(ordinals: Sequence[int], n_images: int,
                           n_gt: int) -> List[int]:
    """DENSE evaluation: which annotation row belongs to each image ordinal.

    ``ordinals`` is the sparse mapping produced by
    :func:`annotated_frame_ordinals` (annotation row -> image ordinal, strictly
    increasing).  It is inverted here: image ``j`` is scored against the box of
    the annotation that is *closest* to it, ties going to the earlier row.  This
    is what a dense-protocol VTUAV evaluation needs -- the tracker runs on every
    frame of the folder while the ground truth is only annotated every 10th.

    Returns a list of length ``n_images`` with annotation row indices.
    """
    if not ordinals or n_images <= 0:
        return []
    n_rows = min(len(ordinals), max(n_gt, 1))
    out: List[int] = []
    row = 0
    for image in range(n_images):
        # advance while the next annotation is strictly closer to this image
        while (row + 1 < n_rows
               and abs(int(ordinals[row + 1]) - image) < abs(int(ordinals[row]) - image)):
            row += 1
        out.append(row)
    return out
