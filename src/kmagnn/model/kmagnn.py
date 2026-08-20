from __future__ import annotations

import torch
import torch.nn as nn

from .layers import RelationAwareKGAttentionLayer, gather_by_id


class KMAGNN(nn.Module):
    """KMAGNN encoder and prediction head from Section 4 of the manuscript."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        num_nodes: int,
        num_relations: int,
        neighbor_store,
        embedding_dim: int = 256,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.2,
        retained_neighbors: int = 20,
        neighbor_gamma: float = 0.6,
        jaccard_mode: str = "exact",
        predictor: str = "mlp",
        max_frontier_nodes: int | None = None,
    ):
        super().__init__()
        self.num_users = int(num_users)
        self.num_items = int(num_items)
        self.num_nodes = int(num_nodes)
        self.num_relations = int(num_relations)
        self.embedding_dim = int(embedding_dim)
        self.num_layers = int(num_layers)
        self.predictor_type = str(predictor).lower()
        self.neighbor_store = neighbor_store
        self.retained_neighbors = int(retained_neighbors)
        self.max_frontier_nodes = None if max_frontier_nodes is None else int(max_frontier_nodes)

        self.node_embeddings = nn.Embedding(self.num_nodes, self.embedding_dim)
        self.layers = nn.ModuleList(
            [
                RelationAwareKGAttentionLayer(
                    embedding_dim=self.embedding_dim,
                    num_heads=num_heads,
                    num_relations=self.num_relations,
                    retained_neighbors=retained_neighbors,
                    gamma=neighbor_gamma,
                    dropout=dropout,
                    jaccard_mode=jaccard_mode,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.readout_dim = self.embedding_dim * (self.num_layers + 1)
        if self.predictor_type in {"mlp", "mlp1"}:
            self.predictor = nn.Sequential(
                nn.Linear(self.readout_dim * 2, self.readout_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(self.readout_dim, 1),
            )
        elif self.predictor_type == "mlp2":
            self.predictor = nn.Sequential(
                nn.Linear(self.readout_dim * 2, self.readout_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(self.readout_dim, self.readout_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(self.readout_dim // 2, 1),
            )
        elif self.predictor_type == "dot":
            self.predictor = None
        else:
            raise ValueError("predictor must be one of: mlp, mlp1, mlp2, dot")

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.node_embeddings.weight, std=0.01)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            if isinstance(module, nn.Embedding) and module is not self.node_embeddings:
                nn.init.normal_(module.weight, std=0.01)

    def item_nodes(self, item_ids: torch.Tensor) -> torch.Tensor:
        return self.num_users + item_ids.long() - 1

    def _cap_frontier(self, required: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        merged = torch.unique(torch.cat([required, candidates]))
        if self.max_frontier_nodes is None or merged.numel() <= self.max_frontier_nodes:
            return merged
        required = torch.unique(required)
        if required.numel() >= self.max_frontier_nodes:
            return required[: self.max_frontier_nodes]
        extra = merged[~torch.isin(merged, required)]
        slots = max(self.max_frontier_nodes - required.numel(), 0)
        return torch.unique(torch.cat([required, extra[:slots]]))

    def _build_frontiers(self, target_nodes: torch.Tensor) -> list[torch.Tensor]:
        device = target_nodes.device
        frontiers: list[torch.Tensor | None] = [None for _ in range(self.num_layers + 1)]
        frontiers[self.num_layers] = torch.unique(target_nodes)
        for depth in range(self.num_layers, 0, -1):
            nodes, _, mask = self.neighbor_store.lookup(
                frontiers[depth],
                device=device,
                limit=self.retained_neighbors,
            )
            candidates = nodes[mask]
            if candidates.numel() == 0:
                frontiers[depth - 1] = frontiers[depth]
            else:
                frontiers[depth - 1] = self._cap_frontier(frontiers[depth], candidates)
        return [x for x in frontiers if x is not None]

    def encode_nodes(self, node_ids: torch.Tensor) -> torch.Tensor:
        node_ids = node_ids.long().view(-1)
        unique_nodes, inverse = torch.unique(node_ids, sorted=True, return_inverse=True)
        frontiers = self._build_frontiers(unique_nodes)

        layer_nodes = [frontiers[0]]
        layer_embs = [self.node_embeddings(frontiers[0])]

        for layer_idx, layer in enumerate(self.layers, start=1):
            nodes = frontiers[layer_idx]
            prev_nodes = layer_nodes[layer_idx - 1]
            prev_embs = layer_embs[layer_idx - 1]

            center_prev, center_found = gather_by_id(nodes, prev_nodes, prev_embs)
            if not center_found.all():
                fallback = self.node_embeddings(nodes)
                center_prev = torch.where(center_found.unsqueeze(-1), center_prev, fallback)

            nbr_nodes, nbr_rels, nbr_mask = self.neighbor_store.lookup(
                nodes,
                device=nodes.device,
                limit=self.retained_neighbors,
            )
            nbr_embs, nbr_found = gather_by_id(nbr_nodes, prev_nodes, prev_embs)
            nbr_mask = nbr_mask & nbr_found
            updated = layer(center_prev, nbr_embs, nbr_rels, nbr_mask, nbr_nodes, self.neighbor_store)
            layer_nodes.append(nodes)
            layer_embs.append(updated)

        readout_parts = []
        for nodes, embs in zip(layer_nodes, layer_embs):
            part, found = gather_by_id(unique_nodes, nodes, embs)
            if not found.all():
                fallback = self.node_embeddings(unique_nodes)
                part = torch.where(found.unsqueeze(-1), part, fallback)
            readout_parts.append(part)
        readout = torch.cat(readout_parts, dim=-1)
        return readout[inverse]

    def predict_scores(self, user_emb: torch.Tensor, item_emb: torch.Tensor) -> torch.Tensor:
        if item_emb.dim() == 2:
            item_emb = item_emb.unsqueeze(1)
        user_expanded = user_emb.unsqueeze(1).expand(-1, item_emb.size(1), -1)
        if self.predictor_type == "dot":
            return (user_expanded * item_emb).sum(dim=-1)
        x = torch.cat([user_expanded, item_emb], dim=-1)
        return self.predictor(x).squeeze(-1)

    def score_candidates(
        self,
        users: torch.Tensor,
        candidate_items: torch.Tensor,
        return_embeddings: bool = False,
    ):
        users = users.long().view(-1)
        if candidate_items.dim() == 1:
            candidate_items = candidate_items.view(-1, 1)
        candidate_items = candidate_items.long()

        user_nodes = users
        item_nodes = self.item_nodes(candidate_items.reshape(-1))
        all_nodes = torch.cat([user_nodes, item_nodes], dim=0)
        all_embs = self.encode_nodes(all_nodes)
        user_emb = all_embs[: users.numel()]
        item_emb = all_embs[users.numel() :].view(users.numel(), candidate_items.size(1), -1)
        scores = self.predict_scores(user_emb, item_emb)
        if return_embeddings:
            return scores, user_emb, item_emb
        return scores

    def forward(self, users: torch.Tensor, pos_items: torch.Tensor, neg_items: torch.Tensor | None = None):
        if neg_items is None:
            return self.score_candidates(users, pos_items).squeeze(-1)
        candidates = torch.cat([pos_items.view(-1, 1), neg_items], dim=1)
        return self.score_candidates(users, candidates)

    def predict(self, users: torch.Tensor, candidate_items: torch.Tensor) -> torch.Tensor:
        return self.score_candidates(users, candidate_items)

