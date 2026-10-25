"""Training datasets with lazy imports.

Optional datasets depend on packages such as lmdb and pycocotools. Importing
them eagerly made a LasHeR-only run require every optional dependency.
"""

from importlib import import_module


_DATASETS = {
    "Lasot": ("lasot", "Lasot"),
    "Got10k": ("got10k", "Got10k"),
    "TrackingNet": ("tracking_net", "TrackingNet"),
    "ImagenetVID": ("imagenetvid", "ImagenetVID"),
    "MSCOCO": ("coco", "MSCOCO"),
    "MSCOCOSeq": ("coco_seq", "MSCOCOSeq"),
    "Got10k_lmdb": ("got10k_lmdb", "Got10k_lmdb"),
    "Lasot_lmdb": ("lasot_lmdb", "Lasot_lmdb"),
    "ImagenetVID_lmdb": ("imagenetvid_lmdb", "ImagenetVID_lmdb"),
    "MSCOCOSeq_lmdb": ("coco_seq_lmdb", "MSCOCOSeq_lmdb"),
    "TrackingNet_lmdb": ("tracking_net_lmdb", "TrackingNet_lmdb"),
    "LasHeR": ("lasher", "LasHeR"),
    "DepthTrack": ("depthtrack", "DepthTrack"),
    "VisEvent": ("visevent", "VisEvent"),
    "VTUAV": ("vtuav", "VTUAV"),
    "LasHeR_testingSet": ("LasHeR_testingSet", "LasHeR_testingSet"),
    "LasHeR_trainingSet": ("LasHeR_trainingSet", "LasHeR_trainingSet"),
    "RGBT234": ("rgbt234", "RGBT234"),
    "RGBT210Dataset": ("rgbt210dataset", "RGBT210Dataset"),
}

__all__ = list(_DATASETS)


def __getattr__(name):
    try:
        module_name, class_name = _DATASETS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(f"{__name__}.{module_name}"), class_name)
    globals()[name] = value
    return value
