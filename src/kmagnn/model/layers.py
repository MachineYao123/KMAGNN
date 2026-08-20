from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def gather_by_id(query_ids: torch.Tensor, key_ids: torch.Tensor, key_embs: torch.Tensor):
    """Gather embeddings for query ids from a sparse frontier tensor."""

    original_shape = query_ids.shape
    flat = query_ids.reshape(-1)
    valid = flat.ge(0)
    out = key_embs.new_zeros((flat.numel(), key_embs.size(-1)))
    found = torch.zeros(flat.numel(), dtype=torch.bool, device=flat.device)
    if valid.any() and key_ids.numel() > 0:
        sorted_keys, order = torch.sort(key_ids)
        valid_flat = flat[valid]
        pos = torch.searchsorted(sorted_keys, valid_flat)
        in_bounds = pos.lt(sorted_keys.numel())
        pos_safe = pos.clamp(max=max(sorted_keys.numel() - 1, 0))
        matched = in_bounds & sorted_keys[pos_safe].eq(valid_flat)
        valid_positions = torch.nonzero(valid, as_tuple=False).squeeze(1)
        if matched.any():
            source_idx = order[pos_safe[matched]]
            target_idx = valid_positions[matched]
            out[target_idx] = key_embs[source_idx]
            found[target_idx] = True
    return out.reshape(*original_shape, key_embs.size(-1)), found.reshape(original_shape)


class RelationAwareKGAttentionLayer(nn.Module):
    """Relation-aware attention and gated update from the manuscript.

    Eq. (12): Q, K, V projections.
    Eq. (13)-(15): multi-head attention and residual fusion.
    Eq. (16)-(18): relevance-redundancy score, Top-K retention, Jaccard reweighting.
    Eq. (19)-(20): transformed update, gate, LayerNorm.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        num_relations: int,
        retained_neighbors: int = 20,
        gamma: float = 0.6,
        dropout: float = 0.2,
        jaccard_mode: str = "exact",
    ):
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")
        self.embedding_dim = int(embedding_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.embedding_dim // self.num_heads
        self.retained_neighbors = int(retained_neighbors)
        self.gamma = float(gamma)
        self.jaccard_mode = str(jaccard_mode)

        self.relation_embeddings = nn.Embedding(int(num_relations), self.embedding_dim)
        self.query_proj = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.key_proj = nn.Linear(self.embedding_dim * 2, self.embedding_dim, bias=False)
        self.value_proj = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.output_proj = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.transform = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.gate = nn.Linear(self.embedding_dim * 2, self.embedding_dim)
        self.layer_norm = nn.LayerNorm(self.embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def _empty_update(self, center_emb: torch.Tensor) -> torch.Tensor:
        transformed = F.relu(self.transform(center_emb))
        transformed = self.dropout(transformed)
        gate = torch.sigmoid(self.gate(torch.cat([center_emb, transformed], dim=-1)))
        return self.layer_norm(gate * transformed + (1.0 - gate) * center_emb)

    def forward(
        self,
        center_emb: torch.Tensor,
        neighbor_emb: torch.Tensor,
        neighbor_rels: torch.Tensor,
        neighbor_mask: torch.Tensor,
        neighbor_nodes: torch.Tensor,
        neighbor_store,
    ) -> torch.Tensor:
        batch_size, max_neighbors, _ = neighbor_emb.shape
        if max_neighbors == 0:
            return self._empty_update(center_emb)

        query = self.query_proj(center_emb).view(batch_size, self.num_heads, self.head_dim)
        rel_emb = self.relation_embeddings(neighbor_rels.clamp(min=0))
        key_input = torch.cat([rel_emb, neighbor_emb], dim=-1)
        key = self.key_proj(key_input).view(batch_size, max_neighbors, self.num_heads, self.head_dim)
        value = self.value_proj(neighbor_emb).view(batch_size, max_neighbors, self.num_heads, self.head_dim)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)

        logits = (query.unsqueeze(2) * key).sum(dim=-1) / math.sqrt(self.head_dim)
        logits = logits.masked_fill(~neighbor_mask.unsqueeze(1), -1e9)
        attn = torch.softmax(logits, dim=-1).masked_fill(~neighbor_mask.unsqueeze(1), 0.0)

        relevance = attn.mean(dim=1)
        semantic = F.cosine_similarity(center_emb.unsqueeze(1), neighbor_emb, dim=-1)
        score = self.gamma * relevance - (1.0 - self.gamma) * semantic
        score = score.masked_fill(~neighbor_mask, -1e9)

        retained = max(1, min(self.retained_neighbors, max_neighbors))
        top_score, top_idx = torch.topk(score, k=retained, dim=-1)
        selected_valid = top_score.gt(-1e8)
        selected_nodes = neighbor_nodes.gather(1, top_idx)
        selected_relevance = relevance.gather(1, top_idx)

        value_full = value.permute(0, 2, 1, 3).reshape(batch_size, max_neighbors, self.embedding_dim)
        selected_value = value_full.gather(1, top_idx.unsqueeze(-1).expand(-1, -1, self.embedding_dim))

        redundancy = neighbor_store.jaccard(
            selected_nodes=selected_nodes,
            selected_valid=selected_valid,
            device=center_emb.device,
            mode=self.jaccard_mode,
        )
        adjusted = selected_relevance * (1.0 - redundancy).clamp_min(0.0)
        adjusted = adjusted.masked_fill(~selected_valid, 0.0)
        adjusted = adjusted / adjusted.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        message = (selected_value * adjusted.unsqueeze(-1)).sum(dim=1)
        relation_aware = self.output_proj(message) + center_emb
        transformed = F.relu(self.transform(relation_aware))
        transformed = self.dropout(transformed)
        gate = torch.sigmoid(self.gate(torch.cat([center_emb, transformed], dim=-1)))
        updated = gate * transformed + (1.0 - gate) * center_emb
        return self.layer_norm(updated)

