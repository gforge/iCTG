from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ctg_ml.ctg2_config import CTG2RegistryConfig


@dataclass(frozen=True)
class TabularEncoder:
    numeric_columns: list[str]
    boolean_columns: list[str]
    categorical_columns: list[str]
    numeric_medians: dict[str, float]
    categorical_levels: dict[str, list[str]]
    feature_names: list[str]


@dataclass(frozen=True)
class MultitaskTargetSpec:
    regression_names: list[str]
    binary_names: list[str]


LEAKAGE_WARNING = (
    "The following registry inputs are excluded by default because they are unavailable "
    "at real prediction time or contain direct post-birth information: "
    "etablerade_varkar_seconds, ph_navelartar, ph_navelven, ph_navel_below7."
)


def _clean_boolean_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.astype("boolean")
    lowered = series.astype(str).str.strip().str.lower()
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "ja": True,
        "nej": False,
        "nan": pd.NA,
        "none": pd.NA,
        "": pd.NA,
    }
    return lowered.map(mapping).astype("boolean")


def load_registry_for_multimodal(
    registry_csv: str,
    registry_cfg: CTG2RegistryConfig,
) -> pd.DataFrame:
    usecols = ["BabyID"]
    usecols += registry_cfg.input_numeric
    usecols += registry_cfg.input_boolean
    usecols += registry_cfg.input_categorical
    usecols += registry_cfg.input_excluded_due_to_leakage
    usecols += registry_cfg.regression_outputs
    usecols += registry_cfg.binary_outputs
    df = pd.read_csv(registry_csv, usecols=sorted(set(usecols)))
    if df["BabyID"].duplicated().any():
        dupes = int(df["BabyID"].duplicated().sum())
        raise ValueError(f"registry_final.csv contains {dupes} duplicate BabyID rows")

    for col in registry_cfg.input_numeric + registry_cfg.regression_outputs:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in registry_cfg.input_boolean + registry_cfg.binary_outputs + ["ph_navel_below7"]:
        if col in df.columns:
            df[col] = _clean_boolean_series(df[col])

    for col in registry_cfg.input_categorical:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()
            df.loc[df[col].isin(["", "<NA>", "nan", "None"]), col] = pd.NA

    if "langd_inskrivning_cm" in df.columns:
        df.loc[df["langd_inskrivning_cm"] <= 0, "langd_inskrivning_cm"] = np.nan

    return df


def load_registry_labels_v2(registry_csv: str, at_risk_max_apgar: int = 6) -> pd.DataFrame:
    df = pd.read_csv(registry_csv, usecols=["BabyID", "apgar5"])
    df = df.dropna(subset=["BabyID", "apgar5"]).copy()
    df["apgar5"] = pd.to_numeric(df["apgar5"], errors="raise").astype(int)
    if df["BabyID"].duplicated().any():
        dupes = int(df["BabyID"].duplicated().sum())
        raise ValueError(f"registry_final.csv contains {dupes} duplicate BabyID rows")
    df["target"] = (df["apgar5"] <= at_risk_max_apgar).astype(int)
    return df


def fit_tabular_encoder(
    train_df: pd.DataFrame,
    registry_cfg: CTG2RegistryConfig,
) -> TabularEncoder:
    feature_names: list[str] = []
    numeric_medians: dict[str, float] = {}
    categorical_levels: dict[str, list[str]] = {}

    for col in registry_cfg.input_numeric:
        median = float(train_df[col].median()) if train_df[col].notna().any() else 0.0
        numeric_medians[col] = median
        feature_names.append(col)
        feature_names.append(f"{col}__missing")

    for col in registry_cfg.input_boolean:
        feature_names.append(col)
        feature_names.append(f"{col}__missing")

    for col in registry_cfg.input_categorical:
        series = train_df[col].astype("string")
        series = series.fillna("__MISSING__")
        counts = series.value_counts()
        if col == "fodelseland":
            levels = counts.head(registry_cfg.country_top_k).index.tolist()
        else:
            levels = counts[counts >= registry_cfg.categorical_other_min_frequency].index.tolist()
        levels = [str(x) for x in levels if str(x) != "__MISSING__"]
        levels = sorted(levels)
        categorical_levels[col] = levels
        feature_names.append(f"{col}__missing")
        for level in levels:
            feature_names.append(f"{col}=={level}")
        feature_names.append(f"{col}__other")

    return TabularEncoder(
        numeric_columns=list(registry_cfg.input_numeric),
        boolean_columns=list(registry_cfg.input_boolean),
        categorical_columns=list(registry_cfg.input_categorical),
        numeric_medians=numeric_medians,
        categorical_levels=categorical_levels,
        feature_names=feature_names,
    )


def transform_tabular_inputs(df: pd.DataFrame, encoder: TabularEncoder) -> np.ndarray:
    n = len(df)
    features = np.zeros((n, len(encoder.feature_names)), dtype=np.float32)
    cursor = 0

    for col in encoder.numeric_columns:
        raw = pd.to_numeric(df[col], errors="coerce")
        missing = raw.isna().to_numpy(dtype=np.float32)
        filled = raw.fillna(encoder.numeric_medians[col]).to_numpy(dtype=np.float32)
        features[:, cursor] = filled
        features[:, cursor + 1] = missing
        cursor += 2

    for col in encoder.boolean_columns:
        raw = _clean_boolean_series(df[col])
        missing = raw.isna().to_numpy(dtype=np.float32)
        filled = raw.fillna(False).astype(np.float32).to_numpy()
        features[:, cursor] = filled
        features[:, cursor + 1] = missing
        cursor += 2

    for col in encoder.categorical_columns:
        raw = df[col].astype("string").fillna("__MISSING__")
        raw = raw.astype(str)
        missing = (raw == "__MISSING__").astype(np.float32).to_numpy()
        features[:, cursor] = missing
        cursor += 1
        known_levels = encoder.categorical_levels[col]
        matched_any = np.zeros(n, dtype=bool)
        for level in known_levels:
            mask = (raw == level).to_numpy()
            features[:, cursor] = mask.astype(np.float32)
            matched_any |= mask
            cursor += 1
        other_mask = (~matched_any) & (raw != "__MISSING__").to_numpy()
        features[:, cursor] = other_mask.astype(np.float32)
        cursor += 1

    if cursor != features.shape[1]:
        raise RuntimeError(f"Tabular feature assembly mismatch: cursor={cursor}, width={features.shape[1]}")
    return features


def normalize_tabular_inplace(
    train_X: np.ndarray,
    other_Xs: list[np.ndarray],
    feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    means = np.zeros(train_X.shape[1], dtype=np.float32)
    stds = np.ones(train_X.shape[1], dtype=np.float32)
    for idx, name in enumerate(feature_names):
        if name.endswith("__missing") or "==" in name or name.endswith("__other"):
            continue
        vals = train_X[:, idx]
        mean = float(vals.mean())
        std = float(vals.std())
        means[idx] = np.float32(mean)
        stds[idx] = np.float32(std if std > 1e-6 else 1.0)
        train_X[:, idx] = (train_X[:, idx] - means[idx]) / stds[idx]
        for X in other_Xs:
            X[:, idx] = (X[:, idx] - means[idx]) / stds[idx]
    return means, stds


def build_targets(
    df: pd.DataFrame,
    target_spec: MultitaskTargetSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    reg_targets = np.zeros((len(df), len(target_spec.regression_names)), dtype=np.float32)
    reg_mask = np.zeros_like(reg_targets, dtype=np.float32)
    for idx, col in enumerate(target_spec.regression_names):
        vals = pd.to_numeric(df[col], errors="coerce")
        present = vals.notna().to_numpy()
        reg_targets[:, idx] = vals.fillna(0.0).to_numpy(dtype=np.float32)
        reg_mask[:, idx] = present.astype(np.float32)

    bin_targets = np.zeros((len(df), len(target_spec.binary_names)), dtype=np.float32)
    bin_mask = np.zeros_like(bin_targets, dtype=np.float32)
    for idx, col in enumerate(target_spec.binary_names):
        vals = _clean_boolean_series(df[col])
        present = vals.notna().to_numpy()
        bin_targets[:, idx] = vals.fillna(False).astype(np.float32).to_numpy()
        bin_mask[:, idx] = present.astype(np.float32)

    return reg_targets, reg_mask, bin_targets, bin_mask


def merge_splits_with_registry(
    splits_df: pd.DataFrame,
    registry_df: pd.DataFrame,
) -> pd.DataFrame:
    merged = splits_df.merge(registry_df, on="BabyID", how="left", validate="one_to_one")
    if merged.isna().all(axis=1).any():
        raise ValueError("Merged split/registry table contains fully empty rows")
    return merged
