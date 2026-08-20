"""Paper-aligned KMAGNN reproduction package."""

from .config import Config, DataConfig, ModelConfig, TrainingConfig, load_config
from .train import train_kmagnn

__all__ = [
    "Config",
    "DataConfig",
    "ModelConfig",
    "TrainingConfig",
    "load_config",
    "train_kmagnn",
]

