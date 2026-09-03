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


def test_siblings_stay_in_the_same_split() -> None:
    labels = _synthetic_labels(n_babies=60, n_positive=12)
    # 20 mothers with three pregnancies each, one baby with unknown mother
    labels["MotherID"] = [f"M{i % 20:02d}" for i in range(60)]
    labels.loc[0, "MotherID"] = None

    for seed in range(5):
        splits = create_stratified_splits(labels, FRACTIONS, random_seed=seed)
        assert len(splits) == 60 and not splits["BabyID"].duplicated().any()
        assert "MotherID" in splits.columns
        per_mother = splits.dropna(subset=["MotherID"]).groupby("MotherID")["split"].nunique()
        assert (per_mother == 1).all()
        assert set(splits["split"]) == {"train", "val", "test"}

    # without the column the split degrades to the BabyID level
    plain = create_stratified_splits(labels.drop(columns=["MotherID"]), FRACTIONS, random_seed=1)
    assert "MotherID" not in plain.columns and len(plain) == 60
