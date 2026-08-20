from __future__ import annotations

from pathlib import Path

import torch

from .config import Config, DataConfig, ModelConfig, TrainingConfig
from .data import (
    build_collaborative_knowledge_graph,
    build_user_items,
    load_interaction_data,
    make_data_split,
    merge_user_items,
)
from .metrics import evaluate
from .model import KMAGNN
from .utils import resolve_device


def load_checkpoint_for_evaluation(checkpoint_path: str | Path, device: str | None = None):
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    cfg = checkpoint["config"]
    config = Config(
        data=DataConfig(**cfg["data"]),
        model=ModelConfig(**cfg["model"]),
        training=TrainingConfig(**cfg["training"]),
    )
    target_device = resolve_device(device or config.training.device)
    interactions, items, user_mapping, item_mapping = load_interaction_data(config.data.dataset_dir)
    train_data, valid_data, test_data = make_data_split(
        interactions,
        split_strategy=config.data.split_strategy,
        test_ratio=config.data.test_ratio,
        valid_ratio=config.data.valid_ratio,
        cold_user_max_interactions=config.data.cold_user_max_interactions,
        cold_user_train_interactions=config.data.cold_user_train_interactions,
        cold_item_fraction=config.data.cold_item_fraction,
        cold_item_max_interactions=config.data.cold_item_max_interactions,
    )
    ckg = build_collaborative_knowledge_graph(
        config.data.dataset_dir,
        train_data=train_data,
        num_users=len(user_mapping),
        num_items=len(item_mapping),
        item_mapping=item_mapping,
        max_neighbors=config.model.max_neighbors,
        seed=config.training.seed,
        require_kg=config.data.require_kg,
        kg_edge_dropout=config.data.kg_edge_dropout,
        kg_noise_ratio=config.data.kg_noise_ratio,
    )
    model = KMAGNN(
        num_users=len(user_mapping),
        num_items=len(item_mapping),
        num_nodes=ckg.num_nodes,
        num_relations=ckg.num_relations,
        neighbor_store=ckg.neighbor_store,
        embedding_dim=config.model.embedding_dim,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        dropout=config.model.dropout,
        retained_neighbors=config.model.retained_neighbors,
        neighbor_gamma=config.model.neighbor_gamma,
        jaccard_mode=config.model.jaccard_mode,
        predictor=config.model.predictor,
        max_frontier_nodes=config.model.max_frontier_nodes,
    ).to(target_device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    seen = merge_user_items(build_user_items(train_data), build_user_items(valid_data))
    return model, config, test_data, seen, items, target_device


def evaluate_checkpoint(checkpoint_path: str | Path, device: str | None = None):
    model, config, test_data, seen, items, target_device = load_checkpoint_for_evaluation(checkpoint_path, device)
    return evaluate(
        model,
        test_data,
        seen,
        items,
        device=target_device,
        eval_negatives=config.training.eval_negatives,
    )
