from __future__ import annotations

import random
from collections import deque

import numpy as np
import torch


class AdaptiveSwitchController:
    """Adaptive easy-to-hard trigger described in Section 4.4.2."""

    def __init__(
        self,
        min_warmup: int = 10,
        max_warmup: int = 15,
        patience: int = 3,
        min_delta: float = 1e-3,
        grad_cv_threshold: float = 0.05,
    ):
        self.min_warmup = int(min_warmup)
        self.max_warmup = int(max_warmup)
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.grad_cv_threshold = float(grad_cv_threshold)
        self.losses: list[float] = []
        self.grad_norms = deque(maxlen=max(self.patience, 2))

    def update(self, epoch: int, valid_loss: float, grad_norm: float | None = None) -> tuple[bool, str | None]:
        epoch = int(epoch)
        self.losses.append(float(valid_loss))
        if grad_norm is not None and np.isfinite(grad_norm):
            self.grad_norms.append(float(grad_norm))
        if epoch < self.min_warmup:
            return False, None
        if epoch >= self.max_warmup:
            return True, "maximum warm-up reached"

        if len(self.losses) >= self.patience + 1:
            recent = self.losses[-(self.patience + 1) :]
            small_improvements = 0
            for previous, current in zip(recent[:-1], recent[1:]):
                denom = max(abs(previous), 1e-12)
                rel_improvement = (previous - current) / denom
                if rel_improvement < self.min_delta:
                    small_improvements += 1
            if small_improvements >= self.patience:
                return True, "validation loss plateau"

        if len(self.grad_norms) >= self.patience:
            values = np.asarray(self.grad_norms, dtype=np.float64)
            cv = float(values.std() / max(values.mean(), 1e-12))
            if cv < self.grad_cv_threshold:
                return True, "gradient norm stabilized"
        return False, None


class DynamicNegativeSampler:
    """Popularity-guided easy negatives and model-aware hard negatives."""

    def __init__(
        self,
        train_user_items: dict[int, set[int]],
        item_counts: dict[int, int],
        num_items: int,
        num_negatives: int = 7,
        popularity_exponent: float = 0.75,
        hard_start_epoch: int = 11,
        candidate_pool_size: int = 128,
        hard_pool_size: int = 32,
        strategy: str = "adaptive",
    ):
        self.train_user_items = train_user_items
        self.num_items = int(num_items)
        self.num_negatives = int(num_negatives)
        self.hard_start_epoch = int(hard_start_epoch)
        self.candidate_pool_size = int(candidate_pool_size)
        self.hard_pool_size = int(hard_pool_size)
        self.strategy = str(strategy).lower()
        self.adaptive_hard_enabled = False
        valid = {"random", "popularity", "hard", "fixed", "adaptive"}
        if self.strategy not in valid:
            raise ValueError(f"negative_strategy must be one of {sorted(valid)}")

        counts = np.ones(self.num_items + 1, dtype=np.float64) * 1e-12
        for item, count in item_counts.items():
            item = int(item)
            if 1 <= item <= self.num_items:
                counts[item] += float(count)
        probs = np.power(counts[1:], float(popularity_exponent))
        probs = probs / probs.sum()
        self.alias_prob, self.alias_idx = self._build_alias_table(probs)

    @staticmethod
    def _build_alias_table(probs: np.ndarray):
        probs = np.asarray(probs, dtype=np.float64)
        n = len(probs)
        scaled = probs * n
        alias_prob = np.zeros(n, dtype=np.float32)
        alias_idx = np.zeros(n, dtype=np.int64)
        small = [i for i, p in enumerate(scaled) if p < 1.0]
        large = [i for i, p in enumerate(scaled) if p >= 1.0]
        while small and large:
            s = small.pop()
            l = large.pop()
            alias_prob[s] = scaled[s]
            alias_idx[s] = l
            scaled[l] = scaled[l] - (1.0 - scaled[s])
            if scaled[l] < 1.0:
                small.append(l)
            else:
                large.append(l)
        for idx in small + large:
            alias_prob[idx] = 1.0
            alias_idx[idx] = idx
        return alias_prob, alias_idx

    def _weighted_choice(self, size):
        kk = np.random.randint(0, self.num_items, size=size)
        accept = np.random.random(size=size) < self.alias_prob[kk]
        sampled = np.where(accept, kk, self.alias_idx[kk])
        return sampled.astype(np.int64, copy=False) + 1

    def set_hard_enabled(self, enabled: bool = True) -> None:
        self.adaptive_hard_enabled = bool(enabled)

    def _sample_random_for_user(self, user: int, count: int) -> np.ndarray:
        seen = self.train_user_items.get(int(user), set())
        out = []
        selected = set()
        attempts = 0
        while len(out) < count and attempts < count * 100:
            draw = random.randint(1, self.num_items)
            attempts += 1
            if draw not in seen and draw not in selected:
                out.append(draw)
                selected.add(draw)
        while len(out) < count:
            draw = random.randint(1, self.num_items)
            out.append(draw)
        return np.asarray(out, dtype=np.int64)

    def _filter_negative_draws(self, user: int, draws: np.ndarray, count: int) -> np.ndarray:
        seen = self.train_user_items.get(int(user), set())
        out = []
        selected = set()
        for draw in draws:
            if len(out) >= count:
                break
            draw = int(draw)
            if draw not in seen and draw not in selected:
                out.append(draw)
                selected.add(draw)
        attempts = 0
        while len(out) < count and attempts < count * 100:
            draw = random.randint(1, self.num_items)
            attempts += 1
            if draw not in seen and draw not in selected:
                out.append(draw)
                selected.add(draw)
        while len(out) < count:
            out.append(random.randint(1, self.num_items))
        return np.asarray(out, dtype=np.int64)

    def _sample_popularity_batch(self, users_np: np.ndarray, count: int) -> np.ndarray:
        draw_width = max(int(count) * 50, int(count) + 32)
        draws = self._weighted_choice((len(users_np), draw_width))
        rows = [self._filter_negative_draws(user, row, count) for user, row in zip(users_np, draws)]
        return np.stack(rows, axis=0)

    def _use_hard_negatives(self, epoch: int, model) -> bool:
        if model is None:
            return False
        if self.strategy in {"random", "popularity"}:
            return False
        if self.strategy == "hard":
            return True
        if self.strategy == "adaptive":
            return self.adaptive_hard_enabled
        return int(epoch) >= self.hard_start_epoch

    def sample(self, users: torch.Tensor, model=None, epoch: int = 1, device: torch.device | str = "cpu") -> np.ndarray:
        users_np = users.detach().cpu().numpy().astype(np.int64)
        if self.strategy == "random":
            return np.stack([self._sample_random_for_user(u, self.num_negatives) for u in users_np], axis=0)
        if not self._use_hard_negatives(epoch, model):
            return self._sample_popularity_batch(users_np, self.num_negatives)

        pool_arr = self._sample_popularity_batch(users_np, self.candidate_pool_size)
        was_training = model.training
        model.eval()
        with torch.no_grad():
            pool_tensor = torch.from_numpy(pool_arr).long().to(device)
            user_tensor = users.to(device)
            scores = model.score_candidates(user_tensor, pool_tensor).detach().cpu().numpy()
        if was_training:
            model.train()

        negatives = np.zeros((len(users_np), self.num_negatives), dtype=np.int64)
        for row_idx in range(len(users_np)):
            row_scores = scores[row_idx]
            pool = pool_arr[row_idx]
            hard_k = min(self.hard_pool_size, len(pool))
            top_idx = np.argpartition(row_scores, -hard_k)[-hard_k:]
            hard_pool = pool[top_idx]
            replace = len(hard_pool) < self.num_negatives
            negatives[row_idx] = np.random.choice(hard_pool, size=self.num_negatives, replace=replace)
        return negatives

