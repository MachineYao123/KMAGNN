from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class PairwiseInteractionDataset(Dataset):
    """Observed implicit-feedback pairs from Eq. (1)."""

    def __init__(self, interactions: pd.DataFrame):
        self.users = interactions["user_id"].astype(np.int64).to_numpy()
        self.items = interactions["item_id"].astype(np.int64).to_numpy()

    def __len__(self) -> int:
        return int(len(self.users))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.tensor(self.users[idx], dtype=torch.long), torch.tensor(self.items[idx], dtype=torch.long)


def _safe_timestamp(interactions: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" in interactions.columns:
        return interactions
    out = interactions.copy()
    out["timestamp"] = np.arange(len(out), dtype=np.int64)
    return out


def load_interaction_data(dataset_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    """Load interactions and remap ids to the internal convention used by the paper code.

    Users are mapped to 0..num_users-1. Items are mapped to 1..num_items so that
    item node ids can be computed as num_users + item_id - 1.
    """

    dataset_dir = Path(dataset_dir)
    interaction_path = dataset_dir / "interactions.csv"
    if not interaction_path.exists():
        raise FileNotFoundError(f"missing interactions.csv in {dataset_dir}")

    interactions = pd.read_csv(interaction_path)
    if "user_id" not in interactions.columns or "item_id" not in interactions.columns:
        raise ValueError("interactions.csv must contain user_id and item_id columns")

    interactions = _safe_timestamp(interactions)
    user_mapping = {raw: idx for idx, raw in enumerate(interactions["user_id"].drop_duplicates())}
    item_mapping = {raw: idx + 1 for idx, raw in enumerate(interactions["item_id"].drop_duplicates())}

    mapped = interactions.copy()
    mapped["user_id"] = mapped["user_id"].map(user_mapping).astype(int)
    mapped["item_id"] = mapped["item_id"].map(item_mapping).astype(int)
    mapped = mapped.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    items_path = dataset_dir / "items.csv"
    items = pd.read_csv(items_path) if items_path.exists() else pd.DataFrame(columns=["item_id", "category"])
    if "item_id" in items.columns and not items.empty:
        items = items.copy()
        items["item_id"] = items["item_id"].map(item_mapping)
        items = items[pd.notna(items["item_id"])].copy()
        items["item_id"] = items["item_id"].astype(int)
    return mapped, items, user_mapping, item_mapping


def train_valid_test_split(
    interactions: pd.DataFrame,
    test_ratio: float = 0.2,
    valid_ratio: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Manuscript temporal split.

    Users with at least three interactions contribute one validation and one
    test interaction. Users with exactly two interactions contribute one test
    interaction and retain one training interaction. One-interaction users are
    retained for representation learning and excluded from ranking evaluation.
    """

    if interactions.empty:
        return interactions.copy(), interactions.copy(), interactions.copy()

    train_parts, valid_parts, test_parts = [], [], []
    for _, group in interactions.groupby("user_id", sort=False):
        group = group.sort_values("timestamp")
        n = len(group)
        if n >= 3:
            train_parts.append(group.iloc[:-2])
            valid_parts.append(group.iloc[-2:-1])
            test_parts.append(group.iloc[-1:])
        elif n == 2:
            train_parts.append(group.iloc[:1])
            test_parts.append(group.iloc[1:])
        elif n == 1:
            train_parts.append(group.iloc[:1])

    empty = interactions.iloc[0:0].copy()
    return (
        pd.concat(train_parts, ignore_index=True) if train_parts else empty,
        pd.concat(valid_parts, ignore_index=True) if valid_parts else empty.copy(),
        pd.concat(test_parts, ignore_index=True) if test_parts else empty.copy(),
    )


def temporal_ratio_split(
    interactions: pd.DataFrame,
    test_ratio: float = 0.2,
    valid_ratio: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if interactions.empty:
        return interactions.copy(), interactions.copy(), interactions.copy()

    data = interactions.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    group = data.groupby("user_id", sort=False)
    pos = group.cumcount().to_numpy(dtype=np.int64)
    counts = group["item_id"].transform("size").to_numpy(dtype=np.int64)

    small = counts < 3
    train_end_small = np.maximum(counts - 1, 1)
    test_size = np.maximum(1, np.ceil(counts * float(test_ratio)).astype(np.int64))
    train_valid_len = counts - test_size
    valid_size = np.where(
        train_valid_len > 1,
        np.maximum(1, np.ceil(train_valid_len * float(valid_ratio)).astype(np.int64)),
        0,
    )
    train_end = np.where(small, train_end_small, train_valid_len - valid_size)
    valid_end = np.where(small, train_end_small, train_valid_len)

    train_mask = pos < train_end
    valid_mask = (~small) & (pos >= train_end) & (pos < valid_end)
    test_mask = np.where(small, pos >= train_end_small, pos >= train_valid_len)

    train_df = data.loc[train_mask].reset_index(drop=True)
    valid_df = data.loc[valid_mask].reset_index(drop=True)
    test_df = data.loc[test_mask].reset_index(drop=True)
    return train_df, valid_df, test_df


def user_cold_start_split(
    interactions: pd.DataFrame,
    test_ratio: float = 0.2,
    valid_ratio: float = 0.1,
    cold_user_max_interactions: int = 5,
    cold_user_train_interactions: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_parts, valid_parts, test_parts = [], [], []
    for _, group in interactions.groupby("user_id", sort=False):
        group = group.sort_values("timestamp")
        n = len(group)
        if 3 <= n <= int(cold_user_max_interactions):
            train_end = min(max(int(cold_user_train_interactions), 1), n - 2)
            valid_end = train_end + 1
            train_parts.append(group.iloc[:train_end])
            valid_parts.append(group.iloc[train_end:valid_end])
            test_parts.append(group.iloc[valid_end:])
        else:
            train, valid, test = train_valid_test_split(group, test_ratio=test_ratio, valid_ratio=valid_ratio)
            train_parts.append(train)
            valid_parts.append(valid)
            test_parts.append(test)
    empty = interactions.iloc[0:0].copy()
    return (
        pd.concat(train_parts, ignore_index=True) if train_parts else empty,
        pd.concat(valid_parts, ignore_index=True) if valid_parts else empty.copy(),
        pd.concat(test_parts, ignore_index=True) if test_parts else empty.copy(),
    )


def item_cold_start_split(
    interactions: pd.DataFrame,
    test_ratio: float = 0.2,
    valid_ratio: float = 0.1,
    cold_item_fraction: float = 0.1,
    cold_item_max_interactions: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    item_counts = interactions["item_id"].value_counts().sort_values()
    if cold_item_max_interactions is not None:
        cold_items = set(item_counts[item_counts <= int(cold_item_max_interactions)].index.astype(int).tolist())
    else:
        n_cold = max(1, int(math.ceil(len(item_counts) * float(cold_item_fraction))))
        cold_items = set(item_counts.head(n_cold).index.astype(int).tolist())

    train_parts, valid_parts, test_parts = [], [], []
    for _, group in interactions.groupby("user_id", sort=False):
        group = group.sort_values("timestamp")
        cold_rows = group[group["item_id"].isin(cold_items)]
        warm_rows = group[~group["item_id"].isin(cold_items)]
        if len(warm_rows) >= 2:
            train, valid, test = train_valid_test_split(warm_rows, test_ratio=test_ratio, valid_ratio=valid_ratio)
            train_parts.append(train)
            valid_parts.append(valid)
            test_parts.append(pd.concat([test, cold_rows], ignore_index=True))
        elif len(group) >= 3:
            train_parts.append(group.iloc[:1])
            valid_parts.append(group.iloc[1:2])
            test_parts.append(group.iloc[2:])
        elif len(group) > 0:
            train_parts.append(group.iloc[:1])
            test_parts.append(group.iloc[1:])

    empty = interactions.iloc[0:0].copy()
    return (
        pd.concat(train_parts, ignore_index=True) if train_parts else empty,
        pd.concat(valid_parts, ignore_index=True) if valid_parts else empty.copy(),
        pd.concat(test_parts, ignore_index=True) if test_parts else empty.copy(),
    )


def make_data_split(
    interactions: pd.DataFrame,
    split_strategy: str = "temporal",
    test_ratio: float = 0.2,
    valid_ratio: float = 0.1,
    cold_user_max_interactions: int = 5,
    cold_user_train_interactions: int = 1,
    cold_item_fraction: float = 0.1,
    cold_item_max_interactions: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    strategy = str(split_strategy).lower()
    if strategy == "temporal":
        return train_valid_test_split(interactions, test_ratio=test_ratio, valid_ratio=valid_ratio)
    if strategy == "temporal_ratio":
        return temporal_ratio_split(interactions, test_ratio=test_ratio, valid_ratio=valid_ratio)
    if strategy == "user_cold":
        return user_cold_start_split(
            interactions,
            test_ratio=test_ratio,
            valid_ratio=valid_ratio,
            cold_user_max_interactions=cold_user_max_interactions,
            cold_user_train_interactions=cold_user_train_interactions,
        )
    if strategy == "item_cold":
        return item_cold_start_split(
            interactions,
            test_ratio=test_ratio,
            valid_ratio=valid_ratio,
            cold_item_fraction=cold_item_fraction,
            cold_item_max_interactions=cold_item_max_interactions,
        )
    raise ValueError("split_strategy must be one of: temporal, temporal_ratio, user_cold, item_cold")


def build_user_items(interactions: pd.DataFrame) -> dict[int, set[int]]:
    out: dict[int, set[int]] = defaultdict(set)
    for user, item in zip(interactions["user_id"].to_numpy(), interactions["item_id"].to_numpy()):
        out[int(user)].add(int(item))
    return dict(out)


def merge_user_items(*dicts: dict[int, set[int]]) -> dict[int, set[int]]:
    out: dict[int, set[int]] = defaultdict(set)
    for source in dicts:
        for user, items in source.items():
            out[int(user)].update(int(item) for item in items)
    return dict(out)
