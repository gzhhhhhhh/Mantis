import yaml
from easydict import EasyDict

from .logger import print_log


def log_args_to_file(args, pre="args", logger=None):
    for key, val in args.__dict__.items():
        print_log(f"{pre}.{key} : {val}", logger=logger)


def log_config_to_file(cfg, pre="cfg", logger=None):
    for key, val in cfg.items():
        if isinstance(cfg[key], EasyDict):
            print_log(f"{pre}.{key} = edict()", logger=logger)
            log_config_to_file(cfg[key], pre=pre + "." + key, logger=logger)
            continue
        print_log(f"{pre}.{key} : {val}", logger=logger)


def merge_new_config(config, new_config):
    for key, val in new_config.items():
        if not isinstance(val, dict):
            if key == "_base_":
                with open(new_config["_base_"], "r") as f:
                    try:
                        val = yaml.load(f, Loader=yaml.FullLoader)
                    except TypeError:
                        val = yaml.load(f)
                config[key] = EasyDict()
                merge_new_config(config[key], val)
            else:
                config[key] = val
                continue
        if key not in config:
            config[key] = EasyDict()
        merge_new_config(config[key], val)
    return config


def cfg_from_yaml_file(cfg_file):
    config = EasyDict()
    with open(cfg_file, "r") as f:
        try:
            new_config = yaml.load(f, Loader=yaml.FullLoader)
        except TypeError:
            new_config = yaml.load(f)
    merge_new_config(config=config, new_config=new_config)
    return config


def get_config(args, logger=None):
    config = cfg_from_yaml_file(args.config)
    print_log(f"Loaded config from {args.config}", logger=logger)
    return config
