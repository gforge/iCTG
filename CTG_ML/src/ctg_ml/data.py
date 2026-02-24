from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_registry_labels(
    registry_csv: str | Path,
    at_risk_max_apgar: int,
) -> pd.DataFrame:
    df = pd.read_csv(registry_csv, usecols=["BabyID", "apgar5"])
    df = df.dropna(subset=["BabyID", "apgar5"]).copy()
    df["apgar5"] = df["apgar5"].astype(int)

    dupes = df["BabyID"].duplicated().sum()
    if dupes:
        msg = f"registry_simple.csv contains {dupes} duplicate BabyID rows"
        raise ValueError(msg)

    df["target"] = (df["apgar5"] <= at_risk_max_apgar).astype(int)
    return df


def print_label_summary(labels: pd.DataFrame) -> None:
    counts = labels["target"].value_counts().sort_index()
    total = len(labels)
    healthy = int(counts.get(0, 0))
    at_risk = int(counts.get(1, 0))
    at_risk_pct = 100.0 * at_risk / total if total else 0.0
    print(
        f"Label rows={total:,} healthy={healthy:,} at_risk={at_risk:,} "
        f"({at_risk_pct:.3f}% positive)"
    )
