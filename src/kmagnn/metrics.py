from __future__ import annotations

import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch


def dcg_at_k(relevance, k: int) -> float:
    values = np.asarray(relevance, dtype=float)[:k]
    if values.size == 0:
        return 0.0
    return float(np.sum(values / np.log2(np.arange(2, values.size + 2))))


def ndcg_at_k(relevance, k: int) -> float:
    ideal = dcg_at_k(sorted(relevance, reverse=True), k)
    if ideal == 0.0:
        return 0.0
    return dcg_at_k(relevance, k) / ideal


def build_item_categories(items_df: pd.DataFrame) -> dict[int, str]:
    out: dict[int, str] = {}
    if "item_id" not in items_df.columns:
        return out
    category_col = "category" if "category" in items_df.columns else None
    if category_col is None:
        for fallback in ["main_cat", "category_l0", "genre", "type"]:
            if fallback in items_df.columns:
                category_col = fallback
                break
    if category_col is None:
        return out
    for _, row in items_df.iterrows():
        iid = row.get("item_id")
        if pd.notna(iid):
            value = row.get(category_col, "Unknown")
            out[int(iid)] = "Unknown" if pd.isna(value) else str(value)
    return out


def sample_eval_negatives(num_items: int, forbidden: set[int], count: int = 100) -> list[int]:
    out = []
    attempts = 0
    while len(out) < count and attempts < count * 100:
        item = random.randint(1, int(num_items))
        attempts += 1
        if item not in forbidden and item not in out:
            out.append(item)
    if len(out) < count:
        for item in range(1, int(num_items) + 1):
            if item not in forbidden and item not in out:
                out.append(item)
                if len(out) >= count:
                    break
    return out


def category_ild(categories: list[str]) -> float:
    if len(categories) <= 1:
        return 0.0
    diff = 0
    total = 0
    for i in range(len(categories)):
        for j in range(i + 1, len(categories)):
            diff += 1 if categories[i] != categories[j] else 0
            total += 1
    return diff / max(total, 1)


def tradeoff_f_score(ndcg: float, recall: float, ild: float, cc: float, eps: float = 1e-8) -> float:
    accuracy = (float(ndcg) + float(recall)) / 2.0
    diversity = (float(ild) + float(cc)) / 2.0
    return 2.0 * accuracy * diversity / (accuracy + diversity + eps)


def evaluate(
    model,
    test_data: pd.DataFrame,
    seen_user_items: dict[int, set[int]],
    items_df: pd.DataFrame,
    k_list=(5, 10, 15, 20),
    device: torch.device | str = "cpu",
    eval_negatives: int = 100,
) -> dict[int, dict[str, float]]:
    """Sampled-candidate evaluation used in Section 5.

    Each held-out positive item is ranked against sampled unobserved negatives.
    Under the default temporal split, each evaluated user contributes one
    held-out test item, matching the protocol described in the manuscript.
    """

    model.eval()
    k_list = [int(k) for k in k_list]
    item_categories = build_item_categories(items_df)
    metrics = {
        k: {"ndcg": [], "recall": [], "hr": [], "precision": [], "ild": [], "cc": [], "f_score": []}
        for k in k_list
    }
    test_user_items: dict[int, list[int]] = defaultdict(list)
    for _, row in test_data.iterrows():
        test_user_items[int(row["user_id"])].append(int(row["item_id"]))

    with torch.no_grad():
        for user, positives_for_user in test_user_items.items():
            user_test_set = set(int(item) for item in positives_for_user)
            for positive in list(dict.fromkeys(positives_for_user)):
                positive = int(positive)
                forbidden = set(seen_user_items.get(user, set())) | user_test_set
                negatives = sample_eval_negatives(model.num_items, forbidden, count=eval_negatives)
                candidates = [positive] + negatives
                user_tensor = torch.tensor([user], dtype=torch.long, device=device)
                cand_tensor = torch.tensor([candidates], dtype=torch.long, device=device)
                scores = model.predict(user_tensor, cand_tensor).float().cpu().numpy()[0]
                order = np.argsort(scores)[::-1]
                ranked_items = [int(candidates[i]) for i in order]

                for k in k_list:
                    top_k = ranked_items[:k]
                    hits = [1 if item == positive else 0 for item in top_k]
                    ndcg = ndcg_at_k(hits, k)
                    recall = 1.0 if positive in top_k else 0.0
                    precision = sum(hits) / max(k, 1)
                    categories = [item_categories.get(int(item), "Unknown") for item in top_k]
                    ild = category_ild(categories)
                    cc = len(set(categories)) / max(k, 1)
                    f_score = tradeoff_f_score(ndcg, recall, ild, cc)
                    metrics[k]["ndcg"].append(ndcg)
                    metrics[k]["recall"].append(recall)
                    metrics[k]["hr"].append(1.0 if sum(hits) > 0 else 0.0)
                    metrics[k]["precision"].append(precision)
                    metrics[k]["ild"].append(ild)
                    metrics[k]["cc"].append(cc)
                    metrics[k]["f_score"].append(f_score)

    return {
        k: {name: float(np.mean(values)) if values else 0.0 for name, values in values_by_name.items()}
        for k, values_by_name in metrics.items()
    }
