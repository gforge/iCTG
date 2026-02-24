from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class SplitFractions:
    train_fraction: float
    val_fraction: float
    test_fraction: float


def create_stratified_splits(
    labels: pd.DataFrame,
    fractions: SplitFractions,
    random_seed: int,
) -> pd.DataFrame:
    required = {"BabyID", "apgar5", "target"}
    missing = required - set(labels.columns)
    if missing:
        msg = f"labels missing columns: {sorted(missing)}"
        raise ValueError(msg)

    baby_df = labels[["BabyID", "apgar5", "target"]].copy()
    y = baby_df["target"]

    train_ids, tmp_ids = train_test_split(
        baby_df,
        test_size=(fractions.val_fraction + fractions.test_fraction),
        stratify=y,
        random_state=random_seed,
    )

    val_share_of_tmp = fractions.val_fraction / (fractions.val_fraction + fractions.test_fraction)
    val_ids, test_ids = train_test_split(
        tmp_ids,
        test_size=(1.0 - val_share_of_tmp),
        stratify=tmp_ids["target"],
        random_state=random_seed,
    )

    train_ids = train_ids.assign(split="train")
    val_ids = val_ids.assign(split="val")
    test_ids = test_ids.assign(split="test")

    splits = pd.concat([train_ids, val_ids, test_ids], ignore_index=True)
    if splits["BabyID"].duplicated().any():
        raise ValueError("BabyID overlap detected in generated splits")
    return splits


def save_splits(splits: pd.DataFrame, path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    splits.sort_values(["split", "BabyID"]).to_csv(out_path, index=False)


def print_split_summary(splits: pd.DataFrame) -> None:
    total = len(splits)
    for split_name in ["train", "val", "test"]:
        part = splits[splits["split"] == split_name]
        pos = int(part["target"].sum())
        n = len(part)
        pct = (100.0 * pos / n) if n else 0.0
        print(f"{split_name:>5}: n={n:6d}, positives={pos:4d}, positive_rate={pct:6.3f}%")
    print(f" total: n={total}")
