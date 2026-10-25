import torch
from torch.utils.data.distributed import DistributedSampler
from lib.train.data import sampler, opencv_loader, processing, LTRLoader
import lib.train.data.transforms as tfm
from lib.utils.misc import is_main_process


def update_settings(settings, cfg):
    settings.print_interval = cfg.TRAIN.PRINT_INTERVAL
    settings.search_area_factor = {'template': cfg.DATA.TEMPLATE.FACTOR,
                                   'search': cfg.DATA.SEARCH.FACTOR}
    settings.output_sz = {'template': cfg.DATA.TEMPLATE.SIZE,
                          'search': cfg.DATA.SEARCH.SIZE}
    settings.center_jitter_factor = {'template': cfg.DATA.TEMPLATE.CENTER_JITTER,
                                     'search': cfg.DATA.SEARCH.CENTER_JITTER}
    settings.scale_jitter_factor = {'template': cfg.DATA.TEMPLATE.SCALE_JITTER,
                                    'search': cfg.DATA.SEARCH.SCALE_JITTER}
    settings.grad_clip_norm = cfg.TRAIN.GRAD_CLIP_NORM
    settings.print_stats = None
    settings.batchsize = cfg.TRAIN.BATCH_SIZE
    settings.scheduler_type = cfg.TRAIN.SCHEDULER.TYPE
    settings.fix_bn = getattr(cfg.TRAIN, "FIX_BN", False) # add for fixing base model bn layer


def names2datasets(name_list: list, settings, image_loader):
    assert isinstance(name_list, list)
    datasets = []
    lasher_train_dir = getattr(settings.env, "lasher_train_dir", settings.env.lasher_dir)
    lasher_val_dir = getattr(settings.env, "lasher_val_dir", getattr(settings.env, "lasher_test_dir", settings.env.lasher_dir))
    vtuav_dir = getattr(settings.env, "vtuav_dir", None)
    for name in name_list:
        assert name in ["LASOT", "GOT10K_vottrain", "GOT10K_votval", "GOT10K_train_full", "GOT10K_official_val", "COCO17", "VID", "TRACKINGNET",
                        "DepthTrack_train", "DepthTrack_val", "LasHeR_all", "LasHeR_train", "LasHeR_val", "VisEvent","RGBT210",
                        "VTUAV_train", "VTUAV_val_st", "VTUAV_val_lt"]
        if name == "VTUAV_train":
            from lib.train.dataset import VTUAV
            datasets.append(VTUAV(vtuav_dir, split='train', dtype='rgbrgb'))
        if name == "VTUAV_val_st":
            from lib.train.dataset import VTUAV
            datasets.append(VTUAV(vtuav_dir, split='val_st', dtype='rgbrgb'))
        if name == "VTUAV_val_lt":
            from lib.train.dataset import VTUAV
            datasets.append(VTUAV(vtuav_dir, split='val_lt', dtype='rgbrgb'))
        if name == "DepthTrack_train":
            from lib.train.dataset import DepthTrack
            datasets.append(DepthTrack(settings.env.depthtrack_dir, dtype='rgbcolormap', split='train'))
        if name == "DepthTrack_val":
            from lib.train.dataset import DepthTrack
            datasets.append(DepthTrack(settings.env.depthtrack_dir, dtype='rgbcolormap', split='val'))
        if name == "LasHeR_all":
            from lib.train.dataset import LasHeR
            datasets.append(LasHeR(settings.env.lasher_dir, dtype='rgbrgb', split='all'))
        if name == "LasHeR_train":
            from lib.train.dataset import LasHeR_trainingSet
            datasets.append(LasHeR_trainingSet(lasher_train_dir))
        if name == "LasHeR_val":
            from lib.train.dataset import LasHeR_testingSet
            datasets.append(LasHeR_testingSet(lasher_val_dir))
        if name == "VisEvent":
            from lib.train.dataset import VisEvent
            datasets.append(VisEvent(settings.env.visevent_dir, dtype='rgbrgb', split='train'))
        """Following is rgb dataset"""
        if name == "LASOT":
            if settings.use_lmdb:
                from lib.train.dataset import Lasot_lmdb
                print("Building lasot dataset from lmdb")
                datasets.append(Lasot_lmdb(settings.env.lasot_lmdb_dir, split='train', image_loader=image_loader))
            else:
                from lib.train.dataset import Lasot
                datasets.append(Lasot(settings.env.lasot_dir, split='train', image_loader=image_loader))
        if name == "GOT10K_vottrain":
            if settings.use_lmdb:
                from lib.train.dataset import Got10k_lmdb
                print("Building got10k from lmdb")
                datasets.append(Got10k_lmdb(settings.env.got10k_lmdb_dir, split='vottrain', image_loader=image_loader))
            else:
                from lib.train.dataset import Got10k
                datasets.append(Got10k(settings.env.got10k_dir, split='vottrain', image_loader=image_loader))
        if name == "GOT10K_train_full":
            if settings.use_lmdb:
                from lib.train.dataset import Got10k_lmdb
                print("Building got10k_train_full from lmdb")
                datasets.append(Got10k_lmdb(settings.env.got10k_lmdb_dir, split='train_full', image_loader=image_loader))
            else:
                from lib.train.dataset import Got10k
                datasets.append(Got10k(settings.env.got10k_dir, split='train_full', image_loader=image_loader))
        if name == "GOT10K_votval":
            if settings.use_lmdb:
                from lib.train.dataset import Got10k_lmdb
                print("Building got10k from lmdb")
                datasets.append(Got10k_lmdb(settings.env.got10k_lmdb_dir, split='votval', image_loader=image_loader))
            else:
                from lib.train.dataset import Got10k
                datasets.append(Got10k(settings.env.got10k_dir, split='votval', image_loader=image_loader))
        if name == "GOT10K_official_val":
            if settings.use_lmdb:
                raise ValueError("Not implement")
            else:
                from lib.train.dataset import Got10k
                datasets.append(Got10k(settings.env.got10k_val_dir, split=None, image_loader=image_loader))
        if name == "COCO17":
            if settings.use_lmdb:
                from lib.train.dataset import MSCOCOSeq_lmdb
                print("Building COCO2017 from lmdb")
                datasets.append(MSCOCOSeq_lmdb(settings.env.coco_lmdb_dir, version="2017", image_loader=image_loader))
            else:
                from lib.train.dataset import MSCOCOSeq
                datasets.append(MSCOCOSeq(settings.env.coco_dir, version="2017", image_loader=image_loader))
        if name == "VID":
            if settings.use_lmdb:
                from lib.train.dataset import ImagenetVID_lmdb
                print("Building VID from lmdb")
                datasets.append(ImagenetVID_lmdb(settings.env.imagenet_lmdb_dir, image_loader=image_loader))
            else:
                from lib.train.dataset import ImagenetVID
                datasets.append(ImagenetVID(settings.env.imagenet_dir, image_loader=image_loader))
        if name == "TRACKINGNET":
            if settings.use_lmdb:
                from lib.train.dataset import TrackingNet_lmdb
                print("Building TrackingNet from lmdb")
                datasets.append(TrackingNet_lmdb(settings.env.trackingnet_lmdb_dir, image_loader=image_loader))
            else:
                from lib.train.dataset import TrackingNet
                # raise ValueError("NOW WE CAN ONLY USE TRACKINGNET FROM LMDB")
                datasets.append(TrackingNet(settings.env.trackingnet_dir, image_loader=image_loader))
    return datasets


def build_dataloaders(cfg, settings):
    # Data transform
    # Note: for multimodal data, ToGrayscale and Normalize need modify
    transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.05),
                                    tfm.RandomHorizontalFlip(probability=0.5))

    transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.2),
                                    tfm.RandomHorizontalFlip_Norm(probability=0.5),
                                    tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))

    transform_val = tfm.Transform(tfm.ToTensor(),
                                  tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))

    # The tracking pairs processing module
    output_sz = settings.output_sz
    search_area_factor = settings.search_area_factor

    data_processing_train = processing.BATProcessing(search_area_factor=search_area_factor,
                                                       output_sz=output_sz,
                                                       center_jitter_factor=settings.center_jitter_factor,
                                                       scale_jitter_factor=settings.scale_jitter_factor,
                                                       mode='sequence',
                                                       transform=transform_train,
                                                       joint_transform=transform_joint,
                                                       settings=settings)

    data_processing_val = processing.BATProcessing(search_area_factor=search_area_factor,
                                                     output_sz=output_sz,
                                                     center_jitter_factor=settings.center_jitter_factor,
                                                     scale_jitter_factor=settings.scale_jitter_factor,
                                                     mode='sequence',
                                                     transform=transform_val,
                                                     joint_transform=transform_joint,
                                                     settings=settings)

    # Train sampler and loader
    settings.num_template = getattr(cfg.DATA.TEMPLATE, "NUMBER", 1)
    settings.num_search = getattr(cfg.DATA.SEARCH, "NUMBER", 1)
    sampler_mode = getattr(cfg.DATA, "SAMPLER_MODE", "causal")
    train_cls = getattr(cfg.TRAIN, "TRAIN_CLS", False)
    print("sampler_mode", sampler_mode)
    dataset_train = sampler.TrackingSampler(datasets=names2datasets(cfg.DATA.TRAIN.DATASETS_NAME, settings, opencv_loader),
                                            p_datasets=cfg.DATA.TRAIN.DATASETS_RATIO,
                                            samples_per_epoch=cfg.DATA.TRAIN.SAMPLE_PER_EPOCH,
                                            max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL, num_search_frames=settings.num_search,
                                            num_template_frames=settings.num_template, processing=data_processing_train,
                                            frame_sample_mode=sampler_mode, train_cls=train_cls)

    train_sampler = DistributedSampler(dataset_train) if settings.local_rank != -1 else None
    shuffle = False if settings.local_rank != -1 else True

    loader_train = LTRLoader('train', dataset_train, training=True, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=shuffle,
                             num_workers=cfg.TRAIN.NUM_WORKER, drop_last=True, stack_dim=1, sampler=train_sampler)

    # Validation samplers and loaders(visevent no val split)
    if cfg.DATA.VAL.DATASETS_NAME[0] is None:
        loader_val = None
    else:
        dataset_val = sampler.TrackingSampler(datasets=names2datasets(cfg.DATA.VAL.DATASETS_NAME, settings, opencv_loader),
                                            p_datasets=cfg.DATA.VAL.DATASETS_RATIO,
                                            samples_per_epoch=cfg.DATA.VAL.SAMPLE_PER_EPOCH,
                                            max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL, num_search_frames=settings.num_search,
                                            num_template_frames=settings.num_template, processing=data_processing_val,
                                            frame_sample_mode=sampler_mode, train_cls=train_cls)
        val_sampler = DistributedSampler(dataset_val) if settings.local_rank != -1 else None
        loader_val = LTRLoader('val', dataset_val, training=False, batch_size=cfg.TRAIN.BATCH_SIZE,
                            num_workers=cfg.TRAIN.NUM_WORKER, drop_last=True, stack_dim=1, sampler=val_sampler,
                            epoch_interval=cfg.TRAIN.VAL_EPOCH_INTERVAL)

    return loader_train, loader_val


def get_optimizer_scheduler(net, cfg):
    train_mode = getattr(cfg.TRAIN, "TRAIN_MODE", None)
    if train_mode is None:
        train_type = getattr(cfg.TRAIN.PROMPT, "TYPE", "")
        train_mode = "adapter_only" if "bat" in train_type else "full"
    train_mode = str(train_mode).lower()
    valid_modes = {"adapter_only", "adapter_head", "adapter_head_last3", "full"}
    if train_mode not in valid_modes:
        raise ValueError(f"Unsupported TRAIN.TRAIN_MODE: {train_mode}. Valid modes: {sorted(valid_modes)}")

    def is_adapter(name):
        return "adap" in name

    def is_head(name):
        return name.startswith("box_head") or ".box_head" in name

    def is_new_module(name):
        lower_name = name.lower()
        return is_adapter(name) or any(
            key in lower_name
            for key in ("sgta", "mamba", "liquid", "moe", "temporal")
        )

    def is_last3(name):
        return any(f"backbone.blocks.{idx}." in name for idx in (9, 10, 11)) or name.startswith("backbone.norm")

    def should_train(name):
        if train_mode == "adapter_only":
            return is_adapter(name)
        if train_mode == "adapter_head":
            return is_adapter(name) or is_head(name)
        if train_mode == "adapter_head_last3":
            return is_adapter(name) or is_head(name) or is_last3(name)
        return True

    for n, p in net.named_parameters():
        p.requires_grad = should_train(n)

    main_params = []
    backbone_params = []
    main_num = 0
    backbone_num = 0
    if is_main_process():
        print(f"TRAIN_MODE: {train_mode}")
        print("Trainable parameters are shown below.")

    for n, p in net.named_parameters():
        if not p.requires_grad:
            continue
        if is_new_module(n) or is_head(n):
            main_params.append(p)
            main_num += p.numel()
        else:
            backbone_params.append(p)
            backbone_num += p.numel()
        if is_main_process():
            print(n, p.numel())

    param_dicts = []
    if main_params:
        param_dicts.append({"params": main_params})
    if backbone_params:
        param_dicts.append({
            "params": backbone_params,
            "lr": cfg.TRAIN.LR * cfg.TRAIN.BACKBONE_MULTIPLIER,
        })

    total_num = sum(p.numel() for _, p in net.named_parameters())
    trainable_num = sum(p.numel() for _, p in net.named_parameters() if p.requires_grad)
    if is_main_process():
        print("=Total: ", total_num, "Trainable: ", trainable_num)
        print("=Main LR params(new/head): ", main_num)
        print("=Backbone low LR params: ", backbone_num)

    # print(param_dicts)
    if cfg.TRAIN.OPTIMIZER == "ADAMW":
        optimizer = torch.optim.AdamW(param_dicts, lr=cfg.TRAIN.LR,
                                      weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    else:
        raise ValueError("Unsupported Optimizer")
    if cfg.TRAIN.SCHEDULER.TYPE == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.TRAIN.LR_DROP_EPOCH)
    elif cfg.TRAIN.SCHEDULER.TYPE == "Mstep":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=cfg.TRAIN.SCHEDULER.MILESTONES,
                                                            gamma=cfg.TRAIN.SCHEDULER.GAMMA)
    else:
        # lr_scheduler = None
        raise ValueError("Unsupported scheduler")
    return optimizer, lr_scheduler
