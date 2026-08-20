from __future__ import annotations

import torch
import torch.nn.functional as F


def bpr_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
    """Bayesian Personalized Ranking loss from Eq. (23)."""

    return -F.logsigmoid(pos_scores.unsqueeze(1) - neg_scores).mean()


def contrastive_loss(
    user_emb: torch.Tensor,
    pos_emb: torch.Tensor,
    neg_emb: torch.Tensor,
    temperature: float = 0.2,
) -> torch.Tensor:
    """InfoNCE-style objective from Eq. (24)."""

    pos_sim = F.cosine_similarity(user_emb, pos_emb, dim=-1).unsqueeze(1)
    neg_sim = F.cosine_similarity(user_emb.unsqueeze(1), neg_emb, dim=-1)
    logits = torch.cat([pos_sim, neg_sim], dim=1) / float(temperature)
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, labels)


def listnet_loss(scores: torch.Tensor) -> torch.Tensor:
    """ListNet loss from Eq. (25), with the first candidate as the positive item."""

    target = torch.zeros_like(scores)
    target[:, 0] = 1.0
    target_prob = torch.softmax(target, dim=1)
    pred_log_prob = F.log_softmax(scores, dim=1)
    return -(target_prob * pred_log_prob).sum(dim=1).mean()


def joint_kmagnn_loss(
    scores: torch.Tensor,
    user_emb: torch.Tensor,
    item_emb: torch.Tensor,
    lambda_cl: float = 0.3,
    lambda_listnet: float = 0.1,
    temperature: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Eq. (9): BPR + lambda_1 CL + lambda_2 ListNet.

    L2 is handled by the optimizer weight_decay so that it affects all learnable
    parameters consistently.
    """

    pos_scores = scores[:, 0]
    neg_scores = scores[:, 1:]
    rec = bpr_loss(pos_scores, neg_scores)
    cl = contrastive_loss(user_emb, item_emb[:, 0, :], item_emb[:, 1:, :], temperature=temperature)
    ln = listnet_loss(scores)
    total = rec + float(lambda_cl) * cl + float(lambda_listnet) * ln
    return total, rec, cl, ln

