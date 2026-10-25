from lib.test.utils import TrackerParams
import os
from lib.test.evaluation.environment import env_settings
from lib.config.bat.config import cfg, update_config_from_file


def _env_float(name, default):
    """Read a float from the environment, ignoring blank/invalid values.

    Test-time sweeps are driven by env vars (UPDATE_GATE_THRESHOLD, ...), and a
    stray ``export VAR=`` would otherwise raise in ``float('')``.
    """
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return float(str(raw).strip())
    except ValueError:
        print('[WARN] ignoring invalid %s=%r, using %r' % (name, raw, default))
        return default


def parameters(yaml_name: str, epoch=None):
    params = TrackerParams()
    prj_dir = env_settings().prj_dir
    save_dir = env_settings().save_dir
    # update default config from yaml file
    yaml_file = os.path.join(prj_dir, 'experiments/bat/%s.yaml' % yaml_name)
    update_config_from_file(yaml_file)
    cfg.MODEL.SGTA.UPDATE_GATE_THRESHOLD = _env_float(
        'UPDATE_GATE_THRESHOLD', cfg.MODEL.SGTA.UPDATE_GATE_THRESHOLD)
    params.cfg = cfg
    print("test config: ", cfg)

    # template and search region
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    # SEARCH_FACTOR 可用环境变量覆盖(裁剪尺度, 不改变 token 数);
    # SEARCH_SIZE 请勿改: 与模型的 pos_embed_x token 数绑定(测试期不做插值)。
    params.search_factor = _env_float('SEARCH_FACTOR', cfg.TEST.SEARCH_FACTOR)
    params.search_size = cfg.TEST.SEARCH_SIZE

    # 测试期后处理参数(可用环境变量覆盖, 便于参数扫描)
    params.window_influence = _env_float(
        'WINDOW_INFLUENCE', getattr(cfg.TEST, 'WINDOW_INFLUENCE', 1.0))
    params.penalty_k = _env_float(
        'PENALTY_K', getattr(cfg.TEST, 'PENALTY_K', 0.0))

    # Network checkpoint path
    params.checkpoint = os.path.join(save_dir, "checkpoints/train/bat/%s/BATrack_ep%04d.pth.tar" % (yaml_name, epoch))
    # params.checkpoint = os.path.join(prj_dir, "./output/checkpoints/train/bat/rgbt/BATrack_ep0021.pth.tar") #for lasher
    # params.checkpoint = os.path.join(prj_dir, "./output/checkpoints/train/bat/rgbt_0114/BATrack_ep0012.pth.tar") #for rgbt234
    # whether to save boxes from all queries
    params.save_all_boxes = False

    return params
