import os

import torch

from datasets import build_dataset_from_cfg
from models import build_model_from_cfg
from utils.logger import print_log
from utils.misc import worker_init_fn


def dataset_builder(args, config):
    dataset = build_dataset_from_cfg(config._base_, config.others)
    is_train_subset = config.others.subset in ("train", "trainval")
    shuffle = is_train_subset
    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            shuffle=shuffle,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.others.bs,
            num_workers=int(args.num_workers),
            drop_last=is_train_subset,
            worker_init_fn=worker_init_fn,
            sampler=sampler,
        )
    else:
        sampler = None
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.others.bs,
            shuffle=shuffle,
            drop_last=is_train_subset,
            num_workers=int(args.num_workers),
            worker_init_fn=worker_init_fn,
        )
    return sampler, dataloader


def model_builder(config, default_args=None):
    return build_model_from_cfg(config, default_args=default_args)


def load_model(base_model, ckpt_path, logger=None):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"no checkpoint file from path {ckpt_path}")
    print_log(f"Loading weights from {ckpt_path}...", logger=logger)

    state_dict = torch.load(ckpt_path, map_location="cpu")
    if state_dict.get("model") is not None:
        base_ckpt = {
            k.replace("module.", ""): v for k, v in state_dict["model"].items()
        }
    elif state_dict.get("base_model") is not None:
        base_ckpt = {
            k.replace("module.", ""): v
            for k, v in state_dict["base_model"].items()
        }
    else:
        raise RuntimeError("mismatch of ckpt weight")

    base_model.load_state_dict(base_ckpt, strict=True)

    epoch = state_dict.get("epoch", -1)
    metrics = state_dict.get("metrics", "No Metrics")
    if not isinstance(metrics, (dict, str)) and hasattr(metrics, "state_dict"):
        metrics = metrics.state_dict()
    print_log(
        f"ckpts @ {epoch} epoch( performance = {str(metrics):s})",
        logger=logger,
    )
