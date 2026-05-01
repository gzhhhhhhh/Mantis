from copy import deepcopy


def _to_plain_dict(cfg):
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        plain = {}
        for key, value in cfg.items():
            plain[key] = _to_plain_dict(value)
        return plain
    return deepcopy(cfg)


def _deep_update(base, override):
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def build_peft_cfg(cfg=None):
    default_cfg = {
        'freeze_backbone': True,
        'enable_saa': False,
        'enable_dscd': False,
        'saa': {
            'rank': 8,
            'control_dim': 64,
            'target_layers': 'all',
            'modulate_dt': True,
            'modulate_B': True,
            'modulate_C': True,
            'modulate_A': False,
            'disable_fast_path': True,
        },
        'dscd': {
            'orders': ['x', 'y'],
            'proj_dim': 128,
            'feat_loss': 'cosine',
            'temperature': 2.0,
        },
    }
    return _deep_update(default_cfg, _to_plain_dict(cfg))


def build_loss_cfg(cfg=None):
    default_cfg = {
        'task_weight': 1.0,
        'feat_cons_weight': 0.0,
        'pred_cons_weight': 0.0,
        'ctrl_weight': 0.0,
        'stability_weight': 0.0,
        'moo': {
            'enable': False,
            'warmup_epochs': 0,
            'task_min': 0.6,
            'solver_steps': 32,
            'eps': 1e-12,
        },
    }
    return _deep_update(default_cfg, _to_plain_dict(cfg))


def freeze_module(module):
    if module is None:
        return
    for parameter in module.parameters():
        parameter.requires_grad = False


def unfreeze_module(module):
    if module is None:
        return
    for parameter in module.parameters():
        parameter.requires_grad = True
