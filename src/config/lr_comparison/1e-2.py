from src.config.convnet_utkface_classifier import build_config as build_base_config

def build_config():
    cfg = build_base_config()
    cfg.model.lr = 1e-2

    return cfg