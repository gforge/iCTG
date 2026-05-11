from __future__ import annotations

from pathlib import Path

from ctg_ml import multimodal_config as _config

CTG2ModelConfig = _config.MultimodalModelConfig
CTG2PathsConfig = _config.MultimodalPathsConfig
CTG2ProjectConfig = _config.MultimodalProjectConfig
CTG2RegistryConfig = _config.MultimodalRegistryConfig
CTG2SequenceConfig = _config.MultimodalSequenceConfig
CTG2SplitConfig = _config.MultimodalSplitConfig
CTG2TrainConfig = _config.MultimodalTrainConfig

__all__ = [
    "CTG2ModelConfig",
    "CTG2PathsConfig",
    "CTG2ProjectConfig",
    "CTG2RegistryConfig",
    "CTG2SequenceConfig",
    "CTG2SplitConfig",
    "CTG2TrainConfig",
    "load_ctg2_config",
]


def load_ctg2_config(path: str | Path = "configs/ctg2_multimodal.toml") -> CTG2ProjectConfig:
    """Legacy CTG2 config loader kept for old scripts."""
    return _config.load_multimodal_config(path)
