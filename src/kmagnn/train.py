from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import Config, config_to_dict
from .data import (
    PairwiseInteractionDataset,
    build_collaborative_knowledge_graph,
    build_user_items,
    load_interaction_data,
    make_data_split,
    merge_user_items,
)
from .losses import joint_kmagnn_loss
from .metrics import evaluate
from .model import KMAGNN
from .sampling import AdaptiveSwitchController, DynamicNegativeSampler
from .utils import ensure_dir, get_logger, resolve_device, set_seed, write_json


def _grad_norm(model: torch.nn.Module) -> float:
    total = 0.0
    for param in model.parameters():
        if param.grad is None:
            continue
        value = float(param.grad.detach().data.norm(2).item())
        total += value * value
    return float(total ** 0.5)


def validation_loss(
    model: KMAGNN,
    valid_data,
    sampler: DynamicNegativeSampler,
    device: torch.device,
    batch_size: int,
    lambda_cl: float,
    lambda_listnet: float,
    temperature: float,
) -> float:
    if valid_data.empty:
        return float("inf")
    model.eval()
    dataset = PairwiseInteractionDataset(valid_data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    total = 0.0
    steps = 0
    with torch.no_grad():
        for users, pos_items in loader:
            users = users.to(device)
            pos_items = pos_items.to(device)
            neg_np = sampler.sample(users.cpu(), model=None, epoch=1, device=device)
            neg_items = torch.from_numpy(neg_np).long().to(device)
            candidates = torch.cat([pos_items.view(-1, 1), neg_items], dim=1)
            scores, user_emb, item_emb = model.score_candidates(users, candidates, return_embeddings=True)
            loss, _, _, _ = joint_kmagnn_loss(
                scores,
                user_emb,
                item_emb,
                lambda_cl=lambda_cl,
                lambda_listnet=lambda_listnet,
                temperature=temperature,
            )
            total += float(loss.item())
            steps += 1
    return total / max(steps, 1)


def train_kmagnn(config: Config) -> tuple[KMAGNN, dict[int, dict[str, float]], dict]:
    set_seed(config.training.seed)
    output_dir = ensure_dir(config.training.output_dir)
    logger = get_logger("kmagnn_train", output_dir)
    device = resolve_device(config.training.device)
    logger.info("Starting KMAGNN reproduction training")
    logger.info("Device: %s", device)

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
    num_users = len(user_mapping)
    num_items = len(item_mapping)
    logger.info(
        "Data: users=%d items=%d interactions=%d train=%d valid=%d test=%d",
        num_users,
        num_items,
        len(interactions),
        len(train_data),
        len(valid_data),
        len(test_data),
    )

    train_user_items = build_user_items(train_data)
    valid_user_items = build_user_items(valid_data)
    eval_seen_user_items = merge_user_items(train_user_items, valid_user_items)

    ckg = build_collaborative_knowledge_graph(
        config.data.dataset_dir,
        train_data=train_data,
        num_users=num_users,
        num_items=num_items,
        item_mapping=item_mapping,
        max_neighbors=config.model.max_neighbors,
        seed=config.training.seed,
        require_kg=config.data.require_kg,
        kg_edge_dropout=config.data.kg_edge_dropout,
        kg_noise_ratio=config.data.kg_noise_ratio,
    )
    logger.info(
        "CKG: nodes=%d relations=%d entities=%d directed_kg_edges=%d",
        ckg.num_nodes,
        ckg.num_relations,
        ckg.num_entities,
        ckg.kg_edges,
    )

    model = KMAGNN(
        num_users=num_users,
        num_items=num_items,
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
    ).to(device)

    item_counts = train_data["item_id"].value_counts().to_dict()
    sampler = DynamicNegativeSampler(
        train_user_items=train_user_items,
        item_counts=item_counts,
        num_items=num_items,
        num_negatives=config.training.num_negatives,
        popularity_exponent=0.75,
        hard_start_epoch=config.training.hard_start_epoch,
        candidate_pool_size=config.training.hard_candidate_pool_size,
        hard_pool_size=config.training.hard_pool_size,
        strategy=config.training.negative_strategy,
    )
    adaptive_controller = AdaptiveSwitchController(
        min_warmup=config.training.adaptive_min_warmup,
        max_warmup=config.training.adaptive_max_warmup,
        patience=config.training.adaptive_patience,
        min_delta=config.training.adaptive_min_delta,
        grad_cv_threshold=config.training.adaptive_grad_cv_threshold,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.lr, weight_decay=config.training.lambda_l2)
    dataset = PairwiseInteractionDataset(train_data)
    loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=device.type != "cpu",
        drop_last=False,
    )

    best_valid = float("inf")
    best_epoch = 0
    best_state = None
    no_improve = 0
    history = []
    results = None
    start_time = time.perf_counter()

    for epoch in range(1, config.training.epochs + 1):
        model.train()
        epoch_start = time.perf_counter()
        sums = {"loss": 0.0, "bpr": 0.0, "cl": 0.0, "listnet": 0.0}
        grad_norms = []
        steps = 0

        for users, pos_items in loader:
            users = users.to(device, non_blocking=True)
            pos_items = pos_items.to(device, non_blocking=True)
            neg_np = sampler.sample(users, model=model, epoch=epoch, device=device)
            neg_items = torch.from_numpy(neg_np).long().to(device, non_blocking=True)
            candidates = torch.cat([pos_items.view(-1, 1), neg_items], dim=1)

            optimizer.zero_grad(set_to_none=True)
            scores, user_emb, item_emb = model.score_candidates(users, candidates, return_embeddings=True)
            loss, bpr, cl, listnet = joint_kmagnn_loss(
                scores,
                user_emb,
                item_emb,
                lambda_cl=config.training.lambda_cl,
                lambda_listnet=config.training.lambda_listnet,
                temperature=config.training.temperature,
            )
            if not torch.isfinite(loss):
                logger.warning("Skipping non-finite loss at epoch %d step %d", epoch, steps)
                continue
            loss.backward()
            norm = _grad_norm(model)
            grad_norms.append(norm)
            if config.training.grad_clip and config.training.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.training.grad_clip))
            optimizer.step()

            sums["loss"] += float(loss.item())
            sums["bpr"] += float(bpr.item())
            sums["cl"] += float(cl.item())
            sums["listnet"] += float(listnet.item())
            steps += 1

        denom = max(steps, 1)
        avg = {key: value / denom for key, value in sums.items()}
        val_loss = validation_loss(
            model,
            valid_data,
            sampler,
            device=device,
            batch_size=config.training.batch_size,
            lambda_cl=config.training.lambda_cl,
            lambda_listnet=config.training.lambda_listnet,
            temperature=config.training.temperature,
        )
        if not np.isfinite(val_loss):
            val_loss = avg["loss"]
        epoch_grad = float(np.mean(grad_norms)) if grad_norms else None

        if config.training.negative_strategy.lower() == "adaptive" and not sampler.adaptive_hard_enabled:
            enable, reason = adaptive_controller.update(epoch, val_loss, epoch_grad)
            if enable:
                sampler.set_hard_enabled(True)
                logger.info("Adaptive hard-negative mining enabled at epoch %d: %s", epoch, reason)

        if val_loss < best_valid:
            best_valid = float(val_loss)
            best_epoch = epoch
            no_improve = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1

        record = {
            "epoch": epoch,
            "loss": avg["loss"],
            "bpr_loss": avg["bpr"],
            "contrastive_loss": avg["cl"],
            "listnet_loss": avg["listnet"],
            "valid_loss": float(val_loss),
            "grad_norm": epoch_grad,
            "hard_negative_enabled": bool(
                sampler.strategy == "hard"
                or (sampler.strategy == "fixed" and epoch >= sampler.hard_start_epoch)
                or (sampler.strategy == "adaptive" and sampler.adaptive_hard_enabled)
            ),
            "epoch_seconds": float(time.perf_counter() - epoch_start),
        }
        history.append(record)
        logger.info(
            "Epoch %d/%d loss=%.4f bpr=%.4f cl=%.4f listnet=%.4f valid=%.4f hard=%s",
            epoch,
            config.training.epochs,
            record["loss"],
            record["bpr_loss"],
            record["contrastive_loss"],
            record["listnet_loss"],
            record["valid_loss"],
            record["hard_negative_enabled"],
        )

        if epoch % config.training.eval_every == 0 or epoch == config.training.epochs:
            results = evaluate(
                model,
                test_data,
                eval_seen_user_items,
                items,
                device=device,
                eval_negatives=config.training.eval_negatives,
            )
            for k in sorted(results):
                logger.info(
                    "K=%d NDCG=%.4f Recall=%.4f ILD=%.4f CC=%.4f F=%.4f",
                    k,
                    results[k]["ndcg"],
                    results[k]["recall"],
                    results[k]["ild"],
                    results[k]["cc"],
                    results[k]["f_score"],
                )

        if no_improve >= config.training.early_stop_patience:
            logger.info("Early stopping after %d epochs without validation improvement", no_improve)
            break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    results = evaluate(
        model,
        test_data,
        eval_seen_user_items,
        items,
        device=device,
        eval_negatives=config.training.eval_negatives,
    )
    details = {
        "config": config_to_dict(config),
        "best_valid_loss": best_valid,
        "best_epoch": best_epoch,
        "history": history,
        "results": results,
        "runtime_seconds": float(time.perf_counter() - start_time),
        "data": {
            "num_users": num_users,
            "num_items": num_items,
            "num_interactions": int(len(interactions)),
            "train_interactions": int(len(train_data)),
            "valid_interactions": int(len(valid_data)),
            "test_interactions": int(len(test_data)),
            "ckg_nodes": ckg.num_nodes,
            "ckg_relations": ckg.num_relations,
            "ckg_entities": ckg.num_entities,
            "ckg_directed_kg_edges": ckg.kg_edges,
        },
    }
    write_json(details, output_dir / "details.json")
    write_json({str(k): v for k, v in results.items()}, output_dir / "metrics.json")

    if config.training.save_model:
        checkpoint = {
            "state_dict": model.state_dict(),
            "config": config_to_dict(config),
            "num_users": num_users,
            "num_items": num_items,
            "num_nodes": ckg.num_nodes,
            "num_relations": ckg.num_relations,
            "user_mapping": user_mapping,
            "item_mapping": item_mapping,
            "best_valid_loss": best_valid,
            "best_epoch": best_epoch,
        }
        torch.save(checkpoint, output_dir / "kmagnn.pt")
    logger.info("Training finished; outputs saved to %s", output_dir)
    return model, results, details

