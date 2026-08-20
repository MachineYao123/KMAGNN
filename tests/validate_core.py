from __future__ import annotations

import csv
import math
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kmagnn.config import Config, DataConfig, ModelConfig, TrainingConfig  # noqa: E402
from kmagnn.data import (  # noqa: E402
    build_collaborative_knowledge_graph,
    load_interaction_data,
    make_data_split,
)
from kmagnn.evaluate import evaluate_checkpoint  # noqa: E402
from kmagnn.losses import joint_kmagnn_loss  # noqa: E402
from kmagnn.metrics import category_ild, tradeoff_f_score  # noqa: E402
from kmagnn.model import KMAGNN  # noqa: E402
from kmagnn.sampling import DynamicNegativeSampler  # noqa: E402
from kmagnn.train import train_kmagnn  # noqa: E402


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def build_dataset(root: Path) -> Path:
    dataset = root / "toy"
    interactions = [
        ["u1", "i1", 1],
        ["u2", "i1", 1],
        ["u2", "i2", 2],
        ["u3", "i1", 1],
        ["u3", "i2", 2],
        ["u3", "i3", 3],
        ["u3", "i4", 4],
    ]
    write_csv(dataset / "interactions.csv", ["user_id", "item_id", "timestamp"], interactions)
    write_csv(
        dataset / "items.csv",
        ["item_id", "category"],
        [["i1", "A"], ["i2", "A"], ["i3", "B"], ["i4", "C"]],
    )
    write_csv(
        dataset / "kg.csv",
        ["head", "relation", "tail"],
        [
            ["i1", "has_category", "A"],
            ["i2", "has_category", "A"],
            ["i3", "has_category", "B"],
            ["i4", "has_category", "C"],
            ["i4", "also_related_to", "i1"],
        ],
    )
    return dataset


def test_metrics() -> None:
    assert category_ild(["A", "B", "A"]) == 2 / 3
    value = tradeoff_f_score(ndcg=0.6, recall=0.8, ild=1.0, cc=0.5)
    accuracy = (0.6 + 0.8) / 2
    diversity = (1.0 + 0.5) / 2
    expected = 2 * accuracy * diversity / (accuracy + diversity + 1e-8)
    assert abs(value - expected) < 1e-12


def test_paper_temporal_split(dataset: Path) -> None:
    interactions, _, _, _ = load_interaction_data(dataset)
    train, valid, test = make_data_split(interactions, split_strategy="temporal")
    assert len(train) == 4
    assert len(valid) == 1
    assert len(test) == 2
    u1 = interactions.loc[interactions["timestamp"].eq(1), "user_id"].iloc[0]
    assert u1 in set(train["user_id"])
    assert u1 not in set(test["user_id"]) or len(interactions[interactions["user_id"].eq(u1)]) > 1


def test_ckg_and_model_forward(dataset: Path) -> None:
    interactions, _, user_mapping, item_mapping = load_interaction_data(dataset)
    train, _, _ = make_data_split(interactions, split_strategy="temporal")
    ckg = build_collaborative_knowledge_graph(
        dataset,
        train_data=train,
        num_users=len(user_mapping),
        num_items=len(item_mapping),
        item_mapping=item_mapping,
        max_neighbors=8,
        require_kg=True,
    )
    assert ckg.num_nodes >= len(user_mapping) + len(item_mapping)
    assert ckg.num_relations >= 4
    user_zero_edges = ckg.neighbor_store.adjacency[0]
    assert user_zero_edges, "interaction edge should exist for a training user"

    model = KMAGNN(
        num_users=len(user_mapping),
        num_items=len(item_mapping),
        num_nodes=ckg.num_nodes,
        num_relations=ckg.num_relations,
        neighbor_store=ckg.neighbor_store,
        embedding_dim=16,
        num_layers=1,
        num_heads=4,
        retained_neighbors=4,
        jaccard_mode="signature",
        predictor="mlp",
    )
    users = torch.tensor([0, 1], dtype=torch.long)
    candidates = torch.tensor([[1, 2, 3], [1, 3, 4]], dtype=torch.long)
    scores, user_emb, item_emb = model.score_candidates(users, candidates, return_embeddings=True)
    assert scores.shape == (2, 3)
    assert user_emb.shape[0] == 2
    assert item_emb.shape[:2] == (2, 3)
    loss, bpr, cl, listnet = joint_kmagnn_loss(scores, user_emb, item_emb)
    for value in [loss, bpr, cl, listnet]:
        assert torch.isfinite(value).item()


def test_negative_sampler_excludes_seen() -> None:
    sampler = DynamicNegativeSampler(
        train_user_items={0: {1, 2}, 1: {3}},
        item_counts={1: 5, 2: 4, 3: 3, 4: 2, 5: 1},
        num_items=5,
        num_negatives=3,
        strategy="popularity",
    )
    users = torch.tensor([0, 1], dtype=torch.long)
    negatives = sampler.sample(users)
    assert negatives.shape == (2, 3)
    assert not (set(negatives[0].tolist()) & {1, 2})
    assert not (set(negatives[1].tolist()) & {3})


def test_end_to_end_checkpoint(dataset: Path, root: Path) -> None:
    config = Config(
        data=DataConfig(dataset_dir=str(dataset), require_kg=True),
        model=ModelConfig(
            embedding_dim=16,
            num_layers=1,
            num_heads=4,
            retained_neighbors=4,
            max_neighbors=8,
            jaccard_mode="signature",
            predictor="mlp",
        ),
        training=TrainingConfig(
            epochs=1,
            batch_size=2,
            eval_every=1,
            eval_negatives=2,
            num_negatives=2,
            hard_candidate_pool_size=4,
            hard_pool_size=2,
            negative_strategy="popularity",
            output_dir=str(root / "run"),
            device="cpu",
            save_model=True,
        ),
    )
    _, results, details = train_kmagnn(config)
    assert 20 in results
    assert math.isfinite(details["best_valid_loss"])
    checkpoint = root / "run" / "kmagnn.pt"
    assert checkpoint.exists()
    reloaded = evaluate_checkpoint(checkpoint, device="cpu")
    assert set(reloaded.keys()) == set(results.keys())


def main() -> None:
    torch.set_num_threads(1)
    np.random.seed(7)
    temp = Path(tempfile.mkdtemp(prefix="kmagnn_validate_"))
    try:
        dataset = build_dataset(temp)
        test_metrics()
        test_paper_temporal_split(dataset)
        test_ckg_and_model_forward(dataset)
        test_negative_sampler_excludes_seen()
        test_end_to_end_checkpoint(dataset, temp)
        print("core validation passed")
    finally:
        shutil.rmtree(temp, ignore_errors=True)


if __name__ == "__main__":
    main()

