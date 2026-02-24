from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ctg_ml.config import load_config
from ctg_ml.data import load_registry_labels
from ctg_ml.features import build_aggregate_features
from ctg_ml.metrics import best_f1_threshold, compute_binary_metrics
from ctg_ml.splits import (
    SplitFractions,
    create_stratified_splits,
    save_splits,
)


def ensure_splits(cfg, splits_path: Path) -> pd.DataFrame:
    if splits_path.exists():
        return pd.read_csv(splits_path)

    labels = load_registry_labels(cfg.paths.registry_csv, cfg.target.at_risk_max_apgar)
    splits = create_stratified_splits(
        labels=labels,
        fractions=SplitFractions(
            train_fraction=cfg.split.train_fraction,
            val_fraction=cfg.split.val_fraction,
            test_fraction=cfg.split.test_fraction,
        ),
        random_seed=cfg.split.random_seed,
    )
    save_splits(splits, splits_path)
    return splits


def build_dataset(cfg, splits: pd.DataFrame) -> pd.DataFrame:
    labels = load_registry_labels(cfg.paths.registry_csv, cfg.target.at_risk_max_apgar)
    features = build_aggregate_features(cfg.paths.ctg_parquet, splits[["BabyID"]])
    df = splits.merge(labels, on=["BabyID", "apgar5", "target"], how="inner")
    df = df.merge(features, on="BabyID", how="inner")
    if len(df) != len(splits):
        missing = len(splits) - len(df)
        raise ValueError(f"Feature join lost {missing} babies")
    return df


def print_metrics(prefix: str, metrics: dict[str, float]) -> None:
    print(
        f"{prefix}: thr={metrics['threshold']:.3f} "
        f"ROC-AUC={metrics['roc_auc']:.4f} PR-AUC={metrics['pr_auc']:.4f} "
        f"P={metrics['precision']:.4f} R={metrics['recall']:.4f} F1={metrics['f1']:.4f} "
        f"TN={metrics['tn']} FP={metrics['fp']} FN={metrics['fn']} TP={metrics['tp']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a baseline binary classifier on aggregated CTG features.")
    parser.add_argument("--config", default="configs/default.toml")
    parser.add_argument("--splits", default=None)
    parser.add_argument("--save-model", action="store_true", help="Save trained sklearn pipeline to artifacts/")
    args = parser.parse_args()

    cfg = load_config(args.config)
    artifacts_dir = cfg.paths.artifacts_dir
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    splits_path = Path(args.splits) if args.splits else artifacts_dir / "splits.csv"

    print("Preparing splits...")
    splits = ensure_splits(cfg, splits_path)

    print("Aggregating CTG features with DuckDB (this may take a while on first run)...")
    df = build_dataset(cfg, splits)
    dataset_path = artifacts_dir / "baseline_dataset.parquet"
    df.to_parquet(dataset_path, index=False)
    print(f"Saved merged dataset to {dataset_path}")

    feature_cols = [c for c in df.columns if c not in {"BabyID", "split", "apgar5", "target"}]
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    X_train = train_df[feature_cols]
    y_train = train_df["target"].to_numpy()
    X_val = val_df[feature_cols]
    y_val = val_df["target"].to_numpy()
    X_test = test_df[feature_cols]
    y_test = test_df["target"].to_numpy()

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=cfg.baseline.logreg_c,
                    max_iter=cfg.baseline.max_iter,
                    class_weight="balanced",
                    random_state=cfg.split.random_seed,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)

    val_prob = model.predict_proba(X_val)[:, 1]
    test_prob = model.predict_proba(X_test)[:, 1]

    tuned_thr = best_f1_threshold(y_val.astype(int), val_prob.astype(float))
    val_metrics = compute_binary_metrics(y_val.astype(int), val_prob.astype(float), threshold=tuned_thr)
    test_metrics = compute_binary_metrics(y_test.astype(int), test_prob.astype(float), threshold=tuned_thr)
    test_metrics_default = compute_binary_metrics(y_test.astype(int), test_prob.astype(float), threshold=0.5)

    print_metrics("VAL (tuned on val)", val_metrics)
    print_metrics("TEST (val-tuned thr)", test_metrics)
    print_metrics("TEST (thr=0.5)", test_metrics_default)

    out_metrics = pd.DataFrame(
        [
            {"split": "val", **val_metrics},
            {"split": "test_val_tuned", **test_metrics},
            {"split": "test_default_0.5", **test_metrics_default},
        ]
    )
    metrics_path = artifacts_dir / "baseline_metrics.csv"
    out_metrics.to_csv(metrics_path, index=False)
    print(f"Saved metrics to {metrics_path}")

    if args.save_model:
        model_path = artifacts_dir / "baseline_logreg.joblib"
        joblib.dump({"model": model, "feature_cols": feature_cols}, model_path)
        print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
