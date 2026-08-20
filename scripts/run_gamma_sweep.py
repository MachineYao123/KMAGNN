from __future__ import annotations

import argparse
import copy
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kmagnn import load_config, train_kmagnn  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the gamma sensitivity experiment in Section 5.4.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "kmagnn.yaml"))
    parser.add_argument("--dataset-dir")
    parser.add_argument("--output-dir", default="runs/gamma_sweep")
    parser.add_argument("--values", default="0,0.25,0.5,0.6,0.75,1.0")
    args = parser.parse_args()

    base = load_config(args.config)
    if args.dataset_dir:
        base.data.dataset_dir = args.dataset_dir
    out_root = Path(args.output_dir)
    rows = []
    for value in [float(x.strip()) for x in args.values.split(",") if x.strip()]:
        cfg = copy.deepcopy(base)
        cfg.model.neighbor_gamma = value
        cfg.training.output_dir = str(out_root / f"gamma_{value:g}")
        _, results, _ = train_kmagnn(cfg)
        row = {"gamma": value}
        for k, metric in results.items():
            row[f"recall@{k}"] = metric["recall"]
            row[f"f@{k}"] = metric["f_score"]
        rows.append(row)

    out_root.mkdir(parents=True, exist_ok=True)
    path = out_root / "gamma_sweep.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {path}")


if __name__ == "__main__":
    main()

