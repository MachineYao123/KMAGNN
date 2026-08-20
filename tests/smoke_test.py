from __future__ import annotations

import csv
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kmagnn.config import Config, DataConfig, ModelConfig, TrainingConfig  # noqa: E402
from kmagnn.train import train_kmagnn  # noqa: E402


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def build_synthetic_dataset(root: Path) -> Path:
    dataset = root / "toy"
    interactions = []
    timestamp = 0
    for user in range(6):
        for item in range(user % 3, user % 3 + 4):
            interactions.append([f"u{user}", f"i{item % 8}", timestamp])
            timestamp += 1
    write_csv(dataset / "interactions.csv", ["user_id", "item_id", "timestamp"], interactions)
    write_csv(
        dataset / "items.csv",
        ["item_id", "category"],
        [[f"i{i}", f"c{i % 3}"] for i in range(8)],
    )
    kg_rows = []
    for item in range(8):
        kg_rows.append([f"i{item}", "has_category", f"c{item % 3}"])
        kg_rows.append([f"i{item}", "same_bucket", f"b{item % 2}"])
    write_csv(dataset / "kg.csv", ["head", "relation", "tail"], kg_rows)
    return dataset


def main() -> None:
    temp = Path(tempfile.mkdtemp(prefix="kmagnn_smoke_"))
    try:
        dataset_dir = build_synthetic_dataset(temp)
        config = Config(
            data=DataConfig(dataset_dir=str(dataset_dir), require_kg=True),
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
                batch_size=4,
                eval_every=1,
                eval_negatives=5,
                num_negatives=2,
                hard_candidate_pool_size=6,
                hard_pool_size=3,
                negative_strategy="popularity",
                output_dir=str(temp / "run"),
                device="cpu",
                save_model=True,
            ),
        )
        _, results, details = train_kmagnn(config)
        assert 20 in results
        assert "best_valid_loss" in details
        assert (temp / "run" / "kmagnn.pt").exists()
        print("smoke test passed")
    finally:
        shutil.rmtree(temp, ignore_errors=True)


if __name__ == "__main__":
    main()

