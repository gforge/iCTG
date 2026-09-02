from __future__ import annotations

import numpy as np
import pandas as pd

from ctg_ml.splits import SplitFractions, create_stratified_splits

FRACTIONS = SplitFractions(train_fraction=0.6, val_fraction=0.2, test_fraction=0.2)


def _synthetic_labels(n_babies: int, n_positive: int) -> pd.DataFrame:
    apgar5 = np.full(n_babies, 9, dtype=int)
    apgar5[:n_positive] = 5
    return pd.DataFrame(
        {
            "BabyID": [f"B{i:03d}" for i in range(n_babies)],
            "apgar5": apgar5,
            "target": (apgar5 <= 6).astype(int),
        }
    )


def test_splits_partition_all_baby_ids_without_overlap() -> None:
    labels = _synthetic_labels(n_babies=40, n_positive=10)

    splits = create_stratified_splits(labels, FRACTIONS, random_seed=1)

    ids_by_split = {str(name): set(part["BabyID"]) for name, part in splits.groupby("split")}
    assert set(ids_by_split) == {"train", "val", "test"}
    assert ids_by_split["train"].isdisjoint(ids_by_split["val"])
    assert ids_by_split["train"].isdisjoint(ids_by_split["test"])
    assert ids_by_split["val"].isdisjoint(ids_by_split["test"])
    assert ids_by_split["train"] | ids_by_split["val"] | ids_by_split["test"] == set(
        labels["BabyID"]
    )
    assert len(splits) == len(labels)


def test_stratification_keeps_positives_in_every_split() -> None:
    labels = _synthetic_labels(n_babies=40, n_positive=10)

    splits = create_stratified_splits(labels, FRACTIONS, random_seed=7)

    sizes = splits.groupby("split")["target"].agg(["size", "sum"])
    assert sizes.loc["train", "size"] == 24
    assert sizes.loc["val", "size"] == 8
    assert sizes.loc["test", "size"] == 8
    # 10 positives at a 60/20/20 stratified split -> 6/2/2.
    assert sizes.loc["train", "sum"] == 6
    assert sizes.loc["val", "sum"] == 2
    assert sizes.loc["test", "sum"] == 2
