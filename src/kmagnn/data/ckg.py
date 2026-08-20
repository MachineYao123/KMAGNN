from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .kg_io import find_kg_file, infer_kg_columns, read_kg_table


class NeighborStore:
    """Adjacency lookup and Jaccard overlap for Eq. (17) and Eq. (18)."""

    def __init__(
        self,
        adjacency: list[list[tuple[int, int]]],
        num_nodes: int,
        max_neighbors: int | None = None,
        signature_dim: int = 64,
        seed: int = 42,
    ):
        self.num_nodes = int(num_nodes)
        self.max_neighbors = None if max_neighbors is None or int(max_neighbors) <= 0 else int(max_neighbors)
        self.signature_dim = int(signature_dim)
        self._neighbor_set_cache: dict[int, frozenset[int]] = {}
        self._jaccard_cache: dict[tuple[int, int], float] = {}
        self.adjacency: list[list[tuple[int, int]]] = []
        self.signatures = torch.zeros((self.num_nodes, self.signature_dim), dtype=torch.bool)

        rng = np.random.default_rng(seed)
        for node in range(self.num_nodes):
            edges = adjacency[node] if node < len(adjacency) else []
            if self.max_neighbors is not None and len(edges) > self.max_neighbors:
                idx = rng.choice(len(edges), size=self.max_neighbors, replace=False)
                edges = [edges[int(i)] for i in idx]
            clean_edges = [(int(rel), int(dst)) for rel, dst in edges]
            self.adjacency.append(clean_edges)
            buckets = [dst % self.signature_dim for _, dst in clean_edges if dst >= 0]
            if buckets:
                self.signatures[node, torch.tensor(buckets, dtype=torch.long)] = True

    def _neighbor_set(self, node: int) -> frozenset[int]:
        node = int(node)
        cached = self._neighbor_set_cache.get(node)
        if cached is not None:
            return cached
        values = frozenset(int(dst) for _, dst in self.adjacency[node])
        if len(self._neighbor_set_cache) > 200000:
            self._neighbor_set_cache.clear()
        self._neighbor_set_cache[node] = values
        return values

    def _pair_jaccard(self, left: int, right: int) -> float:
        left, right = int(left), int(right)
        if left == right:
            return 1.0
        key = (left, right) if left < right else (right, left)
        cached = self._jaccard_cache.get(key)
        if cached is not None:
            return cached
        left_set = self._neighbor_set(left)
        right_set = self._neighbor_set(right)
        union = len(left_set | right_set)
        value = 0.0 if union == 0 else len(left_set & right_set) / union
        if len(self._jaccard_cache) > 500000:
            self._jaccard_cache.clear()
        self._jaccard_cache[key] = float(value)
        return float(value)

    def lookup(self, node_ids: torch.Tensor, device: torch.device, limit: int | None = None):
        ids = node_ids.detach().to("cpu").long().clamp(min=0, max=self.num_nodes - 1)
        rows = []
        width = 0
        cap = self.max_neighbors if limit is None else int(limit)
        for node in ids.tolist():
            edges = self.adjacency[int(node)]
            if cap is not None:
                edges = edges[:cap]
            rows.append(edges)
            width = max(width, len(edges))

        nodes = torch.full((len(rows), width), -1, dtype=torch.long)
        rels = torch.zeros((len(rows), width), dtype=torch.long)
        mask = torch.zeros((len(rows), width), dtype=torch.bool)
        for row_idx, edges in enumerate(rows):
            if not edges:
                continue
            rel_values = [rel for rel, _ in edges]
            node_values = [dst for _, dst in edges]
            n = len(edges)
            rels[row_idx, :n] = torch.tensor(rel_values, dtype=torch.long)
            nodes[row_idx, :n] = torch.tensor(node_values, dtype=torch.long)
            mask[row_idx, :n] = True
        return nodes.to(device), rels.to(device), mask.to(device)

    def exact_jaccard(self, selected_nodes: torch.Tensor, selected_valid: torch.Tensor, device: torch.device):
        if selected_nodes.numel() == 0:
            return torch.zeros_like(selected_nodes, dtype=torch.float32, device=device)
        nodes = selected_nodes.detach().to("cpu").long().numpy()
        valid = selected_valid.detach().to("cpu").bool().numpy()
        out = np.zeros(nodes.shape, dtype=np.float32)
        for row_idx in range(nodes.shape[0]):
            for i, node_i in enumerate(nodes[row_idx]):
                if not valid[row_idx, i] or node_i < 0:
                    continue
                max_overlap = 0.0
                for j, node_j in enumerate(nodes[row_idx]):
                    if i == j or not valid[row_idx, j] or node_j < 0:
                        continue
                    max_overlap = max(max_overlap, self._pair_jaccard(int(node_i), int(node_j)))
                out[row_idx, i] = max_overlap
        return torch.from_numpy(out).to(device=device, non_blocking=True)

    def signature_jaccard(self, selected_nodes: torch.Tensor, selected_valid: torch.Tensor, device: torch.device):
        if selected_nodes.numel() == 0:
            return torch.zeros_like(selected_nodes, dtype=torch.float32, device=device)
        shape = selected_nodes.shape
        ids = selected_nodes.detach().to("cpu").long().clamp(min=0, max=self.num_nodes - 1)
        sig = self.signatures.index_select(0, ids.reshape(-1)).reshape(*shape, self.signature_dim)
        sig = sig.to(device=device, non_blocking=True)
        valid = selected_valid.bool() & selected_nodes.ge(0)
        sig = sig & valid.unsqueeze(-1)
        left = sig.unsqueeze(2)
        right = sig.unsqueeze(1)
        inter = (left & right).sum(dim=-1).float()
        union = (left | right).sum(dim=-1).float().clamp_min(1.0)
        jac = inter / union
        k = selected_nodes.size(1)
        eye = torch.eye(k, dtype=torch.bool, device=device).unsqueeze(0)
        pair_valid = valid.unsqueeze(2) & valid.unsqueeze(1)
        jac = jac.masked_fill(eye, 0.0)
        jac = jac.masked_fill(~pair_valid, 0.0)
        return jac.max(dim=-1).values

    def jaccard(self, selected_nodes: torch.Tensor, selected_valid: torch.Tensor, device: torch.device, mode: str):
        if mode == "exact":
            return self.exact_jaccard(selected_nodes, selected_valid, device)
        if mode == "signature":
            return self.signature_jaccard(selected_nodes, selected_valid, device)
        raise ValueError("jaccard mode must be 'exact' or 'signature'")


@dataclass
class CKGBuildResult:
    neighbor_store: NeighborStore
    num_nodes: int
    num_relations: int
    num_entities: int
    entity_mapping: dict[str, int]
    relation_mapping: dict[str, int]
    kg_edges: int


def build_collaborative_knowledge_graph(
    dataset_dir: str | Path,
    train_data: pd.DataFrame,
    num_users: int,
    num_items: int,
    item_mapping: dict,
    max_neighbors: int | None = None,
    signature_dim: int = 64,
    seed: int = 42,
    require_kg: bool = True,
    kg_edge_dropout: float = 0.0,
    kg_noise_ratio: float = 0.0,
) -> CKGBuildResult:
    """Build the CKG described in Section 4.1 of the manuscript."""

    dataset_dir = Path(dataset_dir)
    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(num_users + num_items)]
    relation_mapping = {"interact": 0, "rev_interact": 1}
    entity_mapping: dict[str, int] = {}
    item_lookup = {raw: mapped for raw, mapped in item_mapping.items()}
    item_lookup.update({str(raw): mapped for raw, mapped in item_mapping.items()})

    def item_node(item_id: int) -> int:
        return int(num_users + int(item_id) - 1)

    def ensure_node(node_id: int) -> None:
        while int(node_id) >= len(adjacency):
            adjacency.append([])

    def add_relation(name) -> int:
        key = str(name)
        if key not in relation_mapping:
            relation_mapping[key] = len(relation_mapping)
        return relation_mapping[key]

    def add_entity(value) -> int:
        key = str(value)
        if key not in entity_mapping:
            node_id = num_users + num_items + len(entity_mapping)
            entity_mapping[key] = node_id
            ensure_node(node_id)
        return entity_mapping[key]

    def raw_to_node(value, prefer_item: bool = False):
        if pd.isna(value):
            return None
        if value in item_lookup:
            return item_node(item_lookup[value])
        value_str = str(value)
        if value_str in item_lookup:
            return item_node(item_lookup[value_str])
        if prefer_item:
            try:
                mapped_id = int(value)
                if 1 <= mapped_id <= num_items:
                    return item_node(mapped_id)
            except Exception:
                pass
        return add_entity(value)

    def add_edge(head, rel: int, tail) -> None:
        if head is None or tail is None:
            return
        head, tail = int(head), int(tail)
        ensure_node(max(head, tail))
        adjacency[head].append((int(rel), tail))

    for user_id, item_id in zip(train_data["user_id"].to_numpy(), train_data["item_id"].to_numpy()):
        user_node = int(user_id)
        item = item_node(int(item_id))
        add_edge(user_node, relation_mapping["interact"], item)
        add_edge(item, relation_mapping["rev_interact"], user_node)

    kg_edges = 0
    kg_file = find_kg_file(dataset_dir)
    if require_kg and kg_file is None:
        raise FileNotFoundError(
            "KMAGNN requires an item-side KG file. Expected kg.csv, knowledge_graph.csv, "
            "kg_triplets.csv, triplets.csv, kg_final.csv/txt, or knowledge_graph.txt."
        )

    if kg_file is not None:
        kg_df = read_kg_table(kg_file)
        head_col, rel_col, tail_col = infer_kg_columns(kg_df)
        if kg_edge_dropout > 0.0:
            keep_prob = max(0.0, min(1.0, 1.0 - float(kg_edge_dropout)))
            kg_df = kg_df.sample(frac=keep_prob, random_state=seed).reset_index(drop=True)
        if kg_noise_ratio > 0.0 and not kg_df.empty:
            n_noise = int(math.ceil(len(kg_df) * float(kg_noise_ratio)))
            rng = np.random.default_rng(seed)
            noise_df = kg_df.sample(n=n_noise, replace=True, random_state=seed).copy()
            tails = kg_df[tail_col].dropna().to_numpy()
            if len(tails) > 0:
                noise_df[tail_col] = rng.choice(tails, size=n_noise, replace=True)
                kg_df = pd.concat([kg_df, noise_df], ignore_index=True)

        prefer_head_item = "item" in str(head_col).lower()
        prefer_tail_item = "item" in str(tail_col).lower()
        for head_value, rel_value, tail_value in zip(
            kg_df[head_col].to_numpy(),
            kg_df[rel_col].to_numpy(),
            kg_df[tail_col].to_numpy(),
        ):
            head = raw_to_node(head_value, prefer_item=prefer_head_item)
            tail = raw_to_node(tail_value, prefer_item=prefer_tail_item)
            rel = add_relation(f"kg:{rel_value}")
            rev_rel = add_relation(f"rev_kg:{rel_value}")
            add_edge(head, rel, tail)
            add_edge(tail, rev_rel, head)
            kg_edges += 2

    num_nodes = len(adjacency)
    neighbor_store = NeighborStore(
        adjacency=adjacency,
        num_nodes=num_nodes,
        max_neighbors=max_neighbors,
        signature_dim=signature_dim,
        seed=seed,
    )
    return CKGBuildResult(
        neighbor_store=neighbor_store,
        num_nodes=num_nodes,
        num_relations=len(relation_mapping),
        num_entities=len(entity_mapping),
        entity_mapping=entity_mapping,
        relation_mapping=relation_mapping,
        kg_edges=kg_edges,
    )

