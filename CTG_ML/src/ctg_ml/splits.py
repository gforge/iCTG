from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# Column that identifies the mother of a pregnancy (stage 7/8 registry output). When present,
# all pregnancies of one mother land in the same split so sibling CTGs cannot leak.
GROUP_COLUMN = "MotherID"


@dataclass(frozen=True)
class SplitFractions:
    train_fraction: float
    val_fraction: float
    test_fraction: float


def _group_key(labels: pd.DataFrame, group_column: str | None) -> pd.Series:
    """One key per row: the mother when known, else the BabyID itself (its own group)."""
    baby = labels["BabyID"].astype(str)
    if group_column is None or group_column not in labels.columns:
        return baby
    group = labels[group_column]
    known = group.notna() & (group.astype(str).str.strip() != "")
    return group.astype(str).where(known, baby)


def create_stratified_splits(
    labels: pd.DataFrame,
    fractions: SplitFractions,
    random_seed: int,
    group_column: str | None = GROUP_COLUMN,
) -> pd.DataFrame:
    """Stratified train/val/test split of BabyIDs.

    The split is made on groups (``group_column``, default ``MotherID``; a BabyID without a
    known mother is its own group), stratified on whether any pregnancy of the group is
    positive, and then expanded back to BabyIDs. A registry without the group column gives
    the plain BabyID-level split.
    """
    required = {"BabyID", "apgar5", "target"}
    missing = required - set(labels.columns)
    if missing:
        msg = f"labels missing columns: {sorted(missing)}"
        raise ValueError(msg)

    baby_df = labels[["BabyID", "apgar5", "target"]].copy()
    baby_df["_group"] = _group_key(labels, group_column).to_numpy()
    groups = baby_df.groupby("_group", sort=True)["target"].max().reset_index()
    groups = groups.rename(columns={"target": "_group_target"})

    train_groups, tmp_groups = train_test_split(
        groups,
        test_size=(fractions.val_fraction + fractions.test_fraction),
        stratify=groups["_group_target"],
        random_state=random_seed,
    )

    val_share_of_tmp = fractions.val_fraction / (fractions.val_fraction + fractions.test_fraction)
    val_groups, test_groups = train_test_split(
        tmp_groups,
        test_size=(1.0 - val_share_of_tmp),
        stratify=tmp_groups["_group_target"],
        random_state=random_seed,
    )

    assignment = pd.concat(
        [
            train_groups.assign(split="train"),
            val_groups.assign(split="val"),
            test_groups.assign(split="test"),
        ],
        ignore_index=True,
    )[["_group", "split"]]

    splits = baby_df.merge(assignment, on="_group", how="left", validate="many_to_one")
    if splits["split"].isna().any():
        raise ValueError("Some BabyIDs received no split assignment")
    if splits["BabyID"].duplicated().any():
        raise ValueError("BabyID overlap detected in generated splits")
    if group_column is not None and group_column in labels.columns:
        splits[group_column] = (
            labels.set_index("BabyID").loc[splits["BabyID"], group_column].to_numpy()
        )
    return splits.drop(columns=["_group"]).reset_index(drop=True)


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
        extra = ""
        if GROUP_COLUMN in part.columns:
            extra = f", mothers={part[GROUP_COLUMN].dropna().nunique():6d}"
        print(f"{split_name:>5}: n={n:6d}, positives={pos:4d}, positive_rate={pct:6.3f}%{extra}")
    print(f" total: n={total}")
