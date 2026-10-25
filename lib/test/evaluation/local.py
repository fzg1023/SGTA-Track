from lib.test.evaluation.environment import EnvSettings
import os
from pathlib import Path


def local_env_settings():
    """Dataset / output locations used by the evaluation entry points.

    Every dataset root is derived from ``RGBT_DATA_ROOT`` (default
    ``/root/RGBTData``) and can be overridden by a per-dataset environment
    variable, so no absolute path of the authors' machine leaks into a fork.
    Only ``lasher`` / ``vtuav_*`` / ``gtot`` / ``rgbt210`` / ``rgbt234`` are used
    by the SGTA-Track configs; the remaining entries keep the upstream BAT
    defaults.
    """
    settings = EnvSettings()

    repo_root = Path(__file__).resolve().parents[3]
    data_root = Path(os.environ.get("RGBT_DATA_ROOT", "/root/RGBTData"))

    def root(name, env_var=None):
        default = data_root / name
        return str(Path(os.environ.get(env_var, default))) if env_var else str(default)

    settings.davis_dir = ''
    settings.got10k_lmdb_path = root("got10k_lmdb")
    settings.got10k_path = root("got10k")
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = root("itb")
    settings.lasot_extension_subset_path_path = root("lasot_extension_subset")
    settings.lasot_lmdb_path = root("lasot_lmdb")
    settings.lasot_path = root("lasot")
    settings.network_path = str(repo_root / "output/test/networks")    # Where tracking networks are stored.
    settings.nfs_path = root("nfs")
    settings.otb_path = root("otb")
    settings.prj_dir = str(repo_root)
    settings.result_plot_path = str(repo_root / "output/test/result_plots")
    settings.results_path = str(repo_root / "output/test/tracking_results")    # Where to store tracking results
    settings.save_dir = str(repo_root / "output")
    settings.segmentation_path = str(repo_root / "output/test/segmentation_results")
    settings.tc128_path = root("TC128")
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = root("tnl2k")
    settings.tpl_path = ''
    settings.trackingnet_path = root("trackingnet")
    settings.uav_path = root("uav")
    settings.vot18_path = root("vot2018")
    settings.vot22_path = root("vot2022")
    settings.vot_path = root("VOT2019")
    # RGBT benchmarks.  LASHER_HOME / LASHER_TEST_ROOT point at the LasHeR test
    # split (<root>/<seq>/{visible,infrared}); defaults follow the standard
    # <RGBT_DATA_ROOT>/LasHeR/test layout.
    settings.gtot_path = root("GTOT", "GTOT_ROOT")
    settings.rgbt210_path = root("RGBT210", "RGBT210_ROOT")
    settings.rgbt234_path = root("RGBT234", "RGBT234_ROOT")
    settings.lasher_path = os.environ.get(
        "LASHER_TEST_ROOT",
        os.environ.get("LASHER_HOME", str(data_root / "LasHeR" / "test")))
    # VTUAV root holds train/ test_ST/ test_LT/ (see README "Data").
    settings.vtuav_path = root("VTUAV", "VTUAV_HOME")
    settings.youtubevos_dir = ''

    return settings
