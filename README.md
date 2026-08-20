# KMAGNN

Paper-aligned PyTorch reproduction of KMAGNN, a collaborative knowledge graph-guided recommendation framework for balancing accuracy and diversity.

## Structure

- `src/kmagnn/data`: interaction loading, temporal splitting, and collaborative knowledge graph construction.
- `src/kmagnn/model`: relation-aware KG attention, diversity-aware neighbor reweighting, gated propagation, readout, and prediction.
- `src/kmagnn/sampling`: popularity-guided easy negatives and adaptive hard-negative mining.
- `src/kmagnn/losses.py`: BPR, InfoNCE-style contrastive loss, ListNet, and the joint KMAGNN objective.
- `src/kmagnn/metrics.py`: NDCG, Recall, category-based ILD, normalized CC, and F@K.
- `scripts/train.py`: training entry point.
- `tests`: smoke and core validation scripts.

## Data Format

Each dataset directory should contain:

```text
interactions.csv
items.csv
kg.csv
```

`interactions.csv` requires `user_id` and `item_id`; `timestamp` is optional but recommended. `items.csv` should include `item_id` and `category` for diversity metrics. `kg.csv` should contain head, relation, and tail columns.

## Run

```powershell
pip install -r requirements.txt
python scripts\train.py --dataset-dir path\to\dataset --output-dir runs\movie
```

## Validate

```powershell
python -m compileall src scripts tests
python tests\validate_core.py
python tests\smoke_test.py
```

The default temporal split and sampled-candidate evaluation follow the manuscript protocol.

