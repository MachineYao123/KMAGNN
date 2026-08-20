from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kmagnn import load_config, train_kmagnn  # noqa: E402


def build_overrides(args) -> dict:
    overrides: dict = {"data": {}, "model": {}, "training": {}}
    if args.dataset_dir:
        overrides["data"]["dataset_dir"] = args.dataset_dir
    if args.output_dir:
        overrides["training"]["output_dir"] = args.output_dir
    if args.device:
        overrides["training"]["device"] = args.device
    if args.epochs is not None:
        overrides["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        overrides["training"]["batch_size"] = args.batch_size
    if args.embedding_dim is not None:
        overrides["model"]["embedding_dim"] = args.embedding_dim
    if args.retained_neighbors is not None:
        overrides["model"]["retained_neighbors"] = args.retained_neighbors
    if args.gamma is not None:
        overrides["model"]["neighbor_gamma"] = args.gamma
    if args.negative_strategy:
        overrides["training"]["negative_strategy"] = args.negative_strategy
    return {k: v for k, v in overrides.items() if v}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the paper-aligned KMAGNN reproduction.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "kmagnn.yaml"))
    parser.add_argument("--dataset-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--device")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--embedding-dim", type=int)
    parser.add_argument("--retained-neighbors", type=int)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--negative-strategy", choices=["random", "popularity", "hard", "fixed", "adaptive"])
    args = parser.parse_args()
    config = load_config(args.config, overrides=build_overrides(args))
    train_kmagnn(config)


if __name__ == "__main__":
    main()

