import os

from easydict import EasyDict as edict
import yaml
import os

"""
Add default config for BAT.
"""
cfg = edict()

# MODEL
cfg.MODEL = edict()
cfg.MODEL.PRETRAIN_FILE = ""
cfg.MODEL.EXTRA_MERGER = False
cfg.MODEL.RETURN_INTER = False
cfg.MODEL.RETURN_STAGES = []

# MODEL.BACKBONE
cfg.MODEL.BACKBONE = edict()
cfg.MODEL.BACKBONE.TYPE = "vit_base_patch16_224"
cfg.MODEL.BACKBONE.STRIDE = 16
cfg.MODEL.BACKBONE.MID_PE = False
cfg.MODEL.BACKBONE.SEP_SEG = False
cfg.MODEL.BACKBONE.CAT_MODE = 'direct'
cfg.MODEL.BACKBONE.MERGE_LAYER = 0
cfg.MODEL.BACKBONE.ADD_CLS_TOKEN = False
cfg.MODEL.BACKBONE.CLS_TOKEN_USE_MODE = 'ignore'

cfg.MODEL.BACKBONE.CE_LOC = []
cfg.MODEL.BACKBONE.CE_KEEP_RATIO = []
cfg.MODEL.BACKBONE.CE_TEMPLATE_RANGE = 'ALL'  # choose between ALL, CTR_POINT, CTR_REC, GT_BOX

# MODEL.HEAD
cfg.MODEL.HEAD = edict()
cfg.MODEL.HEAD.TYPE = "CENTER"
cfg.MODEL.HEAD.NUM_CHANNELS = 256

# MODEL.SGTA: State-Gated Template Adaptation (SGTA) modules
cfg.MODEL.SGTA = edict()
cfg.MODEL.SGTA.ENABLED = False
cfg.MODEL.SGTA.USE_MAMBA_FUSION = False
cfg.MODEL.SGTA.USE_TEMPORAL_MAMBA = False
cfg.MODEL.SGTA.USE_LIQUID_CONTROLLER = False
cfg.MODEL.SGTA.USE_MOE = False
cfg.MODEL.SGTA.USE_LEGACY_TEMPORAL_PREDICTOR = True
cfg.MODEL.SGTA.USE_TEMPLATE_ROUTER = True
cfg.MODEL.SGTA.ENABLE_TEMPLATE_UPDATE = True
cfg.MODEL.SGTA.MAMBA_VERSION = 2
cfg.MODEL.SGTA.MAMBA_D_STATE = 64
cfg.MODEL.SGTA.MAMBA_D_CONV = 4
cfg.MODEL.SGTA.MAMBA_EXPAND = 2
cfg.MODEL.SGTA.MAMBA_HEAD_DIM = 64
cfg.MODEL.SGTA.TEMPORAL_MEMORY_TOKENS = 64
cfg.MODEL.SGTA.TEMPORAL_LOSS_WEIGHT = 1.0
cfg.MODEL.SGTA.LIQUID_HIDDEN_DIM = 64
cfg.MODEL.SGTA.MOE_HIDDEN_DIM = 256
cfg.MODEL.SGTA.UPDATE_GATE_THRESHOLD = 0.5

# TRAIN
cfg.TRAIN = edict()
cfg.TRAIN.PROMPT = edict()
cfg.TRAIN.PROMPT.TYPE = 'bat'  # bat_12
cfg.TRAIN.LR = 0.0001
cfg.TRAIN.WEIGHT_DECAY = 0.0001
cfg.TRAIN.EPOCH = 500
cfg.TRAIN.LR_DROP_EPOCH = 400
cfg.TRAIN.BATCH_SIZE = 16
cfg.TRAIN.NUM_WORKER = 8
cfg.TRAIN.OPTIMIZER = "ADAMW"
cfg.TRAIN.TRAIN_MODE = "adapter_only"  # adapter_only, adapter_head, adapter_head_last3, full
cfg.TRAIN.BACKBONE_MULTIPLIER = 0.1
cfg.TRAIN.GIOU_WEIGHT = 2.0
cfg.TRAIN.L1_WEIGHT = 5.0
cfg.TRAIN.FOCAL_WEIGHT = 1.0
cfg.TRAIN.FREEZE_LAYERS = [0, ]
cfg.TRAIN.PRINT_INTERVAL = 50
cfg.TRAIN.VAL_EPOCH_INTERVAL = 20
cfg.TRAIN.GRAD_CLIP_NORM = 0.1
cfg.TRAIN.AMP = False
## TRAIN save cfgs
cfg.TRAIN.FIX_BN = True      ######
cfg.TRAIN.SAVE_EPOCH_INTERVAL = 1 # 1 means save model each epoch
cfg.TRAIN.SAVE_LAST_N_EPOCH = 1 # besides, last n epoch model will be saved
cfg.TRAIN.LOAD_LATEST = True

cfg.TRAIN.CE_START_EPOCH = 20  # candidate elimination start epoch
cfg.TRAIN.CE_WARM_EPOCH = 80  # candidate elimination warm up epoch
cfg.TRAIN.DROP_PATH_RATE = 0.1  # drop path rate for ViT backbone

## TRAIN full-dataset evaluation after each epoch
## After every training epoch, save the checkpoint and run a complete
## lasher testingset evaluation (PR/NPR/SR). Results are appended to
## <EVAL_LASHER_SAVE_DIR>/<EVAL_LASHER_RESULTS_NS or dataset>/eval_history.csv.
cfg.TRAIN.EVAL_LASHER_EVERY_EPOCH = False  # run full lasher test after every epoch
cfg.TRAIN.EVAL_LASHER_WORKERS = 4          # parallel processes for the test
cfg.TRAIN.EVAL_LASHER_GPU = ""             # CUDA_VISIBLE_DEVICES for the test process ("" = inherit)
cfg.TRAIN.EVAL_LASHER_SAVE_DIR = "./RGBT_workspace/results"  # result/history output dir
# Optional results SUBDIRECTORY (namespace) below EVAL_LASHER_SAVE_DIR.  Empty
# means "use the dataset name", i.e. <SAVE_DIR>/lasher/<config>_<epoch>.  Set it
# to e.g. "lasher_a0" so an ablation row keeps its dumps AND its own
# eval_history.csv in a clearly separated tree:
#   <SAVE_DIR>/lasher_a0/<config>_<epoch>/   +  <SAVE_DIR>/lasher_a0/eval_history.csv
cfg.TRAIN.EVAL_LASHER_RESULTS_NS = ""
# Checkpoints are still written every SAVE_EPOCH_INTERVAL epochs; this only
# throttles the (expensive) full-benchmark run: N means "test every N epochs".
# 1 = evaluate right after every saved checkpoint.
cfg.TRAIN.EVAL_LASHER_EPOCH_INTERVAL = 1
# Which benchmark the per-epoch hook above evaluates: lasher (default) or a
# VTUAV subset such as vtuav_st / vtuav_lt. Overridable with the EVAL_DATASET
# environment variable.
cfg.TRAIN.EVAL_DATASET = "lasher"

# TRAIN.SCHEDULER
cfg.TRAIN.SCHEDULER = edict()
cfg.TRAIN.SCHEDULER.TYPE = "step"
cfg.TRAIN.SCHEDULER.DECAY_RATE = 0.1

# DATA
cfg.DATA = edict()
cfg.DATA.SAMPLER_MODE = "causal"  # sampling methods
cfg.DATA.MEAN = [0.485, 0.456, 0.406]
cfg.DATA.STD = [0.229, 0.224, 0.225]
cfg.DATA.MAX_SAMPLE_INTERVAL = 200
# DATA.TRAIN
cfg.DATA.TRAIN = edict()
cfg.DATA.TRAIN.DATASETS_NAME = ["LASOT", "GOT10K_vottrain"]
cfg.DATA.TRAIN.DATASETS_RATIO = [1, 1]
cfg.DATA.TRAIN.SAMPLE_PER_EPOCH = 60000
# DATA.VAL
cfg.DATA.VAL = edict()
cfg.DATA.VAL.DATASETS_NAME = []
cfg.DATA.VAL.DATASETS_RATIO = [1]
cfg.DATA.VAL.SAMPLE_PER_EPOCH = 10000
# DATA.SEARCH
cfg.DATA.SEARCH = edict()
cfg.DATA.SEARCH.SIZE = 320
cfg.DATA.SEARCH.FACTOR = 5.0
cfg.DATA.SEARCH.CENTER_JITTER = 4.5
cfg.DATA.SEARCH.SCALE_JITTER = 0.5
cfg.DATA.SEARCH.NUMBER = 1
# DATA.TEMPLATE
cfg.DATA.TEMPLATE = edict()
cfg.DATA.TEMPLATE.NUMBER = 1
cfg.DATA.TEMPLATE.SIZE = 128
cfg.DATA.TEMPLATE.FACTOR = 2.0
cfg.DATA.TEMPLATE.CENTER_JITTER = 0
cfg.DATA.TEMPLATE.SCALE_JITTER = 0

# TEST
cfg.TEST = edict()
cfg.TEST.TEMPLATE_FACTOR = 2.0
cfg.TEST.TEMPLATE_SIZE = 128
cfg.TEST.SEARCH_FACTOR = 5.0
cfg.TEST.SEARCH_SIZE = 320
cfg.TEST.EPOCH = 500
# 测试期后处理调参: 1.0 = 原行为(直接乘 hann 窗); 0 = 不加窗惩罚
cfg.TEST.WINDOW_INFLUENCE = 1.0
# 尺寸惩罚系数: 0 = 不启用
cfg.TEST.PENALTY_K = 0.0


def _edict2dict(dest_dict, src_edict):
    if isinstance(dest_dict, dict) and isinstance(src_edict, dict):
        for k, v in src_edict.items():
            if not isinstance(v, edict):
                dest_dict[k] = v
            else:
                dest_dict[k] = {}
                _edict2dict(dest_dict[k], v)
    else:
        return


def gen_config(config_file):
    cfg_dict = {}
    _edict2dict(cfg_dict, cfg)
    with open(config_file, 'w') as f:
        yaml.dump(cfg_dict, f, default_flow_style=False)


def _update_config(base_cfg, exp_cfg):
    if isinstance(base_cfg, dict) and isinstance(exp_cfg, edict):
        for k, v in exp_cfg.items():
            if k in base_cfg:
                if not isinstance(v, dict):
                    base_cfg[k] = v
                else:
                    _update_config(base_cfg[k], v)
            else:
                raise ValueError("{} not exist in config.py".format(k))
    else:
        return


def update_config_from_file(filename, base_cfg=None):
    exp_config = None
    with open(filename) as f:
        exp_config = edict(yaml.safe_load(f))
        if base_cfg is not None:
            _update_config(base_cfg, exp_config)
        else:
            _update_config(cfg, exp_config)


#: environment variable -> (cfg path, parser).  Lets a run be re-tuned (batch
#: size for a smaller GPU, AMP, epoch budget, ...) without editing the yaml.
_TRAIN_ENV_OVERRIDES = {
    "BATCH_SIZE": ("TRAIN.BATCH_SIZE", int),
    "NUM_WORKER": ("TRAIN.NUM_WORKER", int),
    "EPOCH": ("TRAIN.EPOCH", int),
    "LR": ("TRAIN.LR", float),
    "AMP": ("TRAIN.AMP", lambda v: v.strip().lower() in ("1", "true", "yes", "on")),
    "TRAIN_SAMPLE_PER_EPOCH": ("DATA.TRAIN.SAMPLE_PER_EPOCH", int),
    "VAL_SAMPLE_PER_EPOCH": ("DATA.VAL.SAMPLE_PER_EPOCH", int),
    "EVAL_LASHER_EVERY_EPOCH": ("TRAIN.EVAL_LASHER_EVERY_EPOCH",
                                lambda v: v.strip().lower() in ("1", "true", "yes", "on")),
    "EVAL_LASHER_EPOCH_INTERVAL": ("TRAIN.EVAL_LASHER_EPOCH_INTERVAL", int),
    "EVAL_DATASET": ("TRAIN.EVAL_DATASET", str),
    "EVAL_LASHER_SAVE_DIR": ("TRAIN.EVAL_LASHER_SAVE_DIR", str),
    "EVAL_LASHER_RESULTS_NS": ("TRAIN.EVAL_LASHER_RESULTS_NS", str),
}


def apply_env_overrides(target=None, environ=None, verbose=True):
    """Override training config entries from environment variables.

    Mirrors ``lib/test/parameter/bat.py``, which already lets the test-time
    post-processing be swept through env vars.  Returns the list of applied
    ``"KEY=value"`` strings so callers (and tests) can report exactly what was
    changed; unknown/unparsable variables are ignored.
    """
    target = cfg if target is None else target
    environ = os.environ if environ is None else environ
    applied = []
    for name, (path, parse) in _TRAIN_ENV_OVERRIDES.items():
        raw = environ.get(name)
        if raw is None or raw == "":
            continue
        try:
            value = parse(raw)
        except (TypeError, ValueError):
            continue
        node = target
        keys = path.split(".")
        for key in keys[:-1]:
            node = node[key]
        node[keys[-1]] = value
        applied.append("{}={}".format(name, value))
    if verbose and applied:
        print("[CONFIG] env overrides: " + " ".join(applied))
    return applied
