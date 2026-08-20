from __future__ import annotations

from pathlib import Path

import pandas as pd


KG_FILE_NAMES = [
    "kg.csv",
    "knowledge_graph.csv",
    "kg_triplets.csv",
    "triplets.csv",
    "kg_final.csv",
    "kg_final.txt",
    "knowledge_graph.txt",
]


def find_kg_file(dataset_dir: str | Path) -> Path | None:
    dataset_dir = Path(dataset_dir)
    for name in KG_FILE_NAMES:
        path = dataset_dir / name
        if path.exists():
            return path
    return None


def read_kg_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".txt":
        return pd.read_csv(path, sep=None, engine="python")
    return pd.read_csv(path)


def _find_column(columns: list, candidates: list[str]):
    lower = {str(c).lower(): c for c in columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


def infer_kg_columns(kg_df: pd.DataFrame):
    cols = list(kg_df.columns)
    head_col = _find_column(cols, ["head", "h", "head_id", "entity_head", "src", "source", "item_id"])
    rel_col = _find_column(cols, ["relation", "relation_id", "rel", "r", "predicate"])
    tail_col = _find_column(cols, ["tail", "t", "tail_id", "entity_tail", "dst", "target", "entity_id"])
    if head_col is None or rel_col is None or tail_col is None:
        if len(cols) < 3:
            raise ValueError("KG file must contain at least three columns for head, relation, and tail")
        head_col, rel_col, tail_col = cols[:3]
    return head_col, rel_col, tail_col

