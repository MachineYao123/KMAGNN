from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    dataset_dir: str
    split_strategy: str = "temporal"
    test_ratio: float = 0.2
    valid_ratio: float = 0.1
    require_kg: bool = True
    cold_user_max_interactions: int = 5
    cold_user_train_interactions: int = 1
    cold_item_fraction: float = 0.1
    cold_item_max_interactions: int | None = None
    kg_edge_dropout: float = 0.0
    kg_noise_ratio: float = 0.0


@dataclass
class ModelConfig:
    embedding_dim: int = 256
    num_layers: int = 3
    num_heads: int = 4
    dropout: float = 0.2
    retained_neighbors: int = 20
    max_neighbors: int | None = None
    neighbor_gamma: float = 0.6
    jaccard_mode: str = "exact"
    predictor: str = "mlp"
    max_frontier_nodes: int | None = None


@dataclass
class TrainingConfig:
    seed: int = 42
    device: str | None = None
    epochs: int = 30
    batch_size: int = 512
    lr: float = 1e-3
    lambda_cl: float = 0.3
    lambda_listnet: float = 0.1
    lambda_l2: float = 1e-5
    temperature: float = 0.2
    num_negatives: int = 7
    negative_strategy: str = "adaptive"
    hard_start_epoch: int = 11
    hard_candidate_pool_size: int = 128
    hard_pool_size: int = 32
    adaptive_min_warmup: int = 10
    adaptive_max_warmup: int = 15
    adaptive_patience: int = 3
    adaptive_min_delta: float = 1e-3
    adaptive_grad_cv_threshold: float = 0.05
    eval_every: int = 5
    eval_negatives: int = 100
    early_stop_patience: int = 25
    grad_clip: float = 1.0
    output_dir: str = "runs/default"
    num_workers: int = 0
    save_model: bool = True


@dataclass
class Config:
    data: DataConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def _merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> Config:
    raw: dict[str, Any] = {}
    if path is not None:
        with Path(path).open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    if overrides:
        raw = _merge_dict(raw, overrides)
    if "data" not in raw or "dataset_dir" not in raw["data"]:
        raise ValueError("config must provide data.dataset_dir")
    return Config(
        data=DataConfig(**raw.get("data", {})),
        model=ModelConfig(**raw.get("model", {})),
        training=TrainingConfig(**raw.get("training", {})),
    )


def config_to_dict(config: Config) -> dict[str, Any]:
    return {
        "data": config.data.__dict__,
        "model": config.model.__dict__,
        "training": config.training.__dict__,
    }

