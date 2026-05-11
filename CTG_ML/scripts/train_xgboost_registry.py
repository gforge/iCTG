from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from ctg_ml.metrics import best_f1_threshold, compute_binary_metrics
from ctg_ml.multimodal_config import load_multimodal_config


@dataclass(frozen=True)
class SplitData:
    X_tab: np.ndarray
    y_apgar: np.ndarray
    y_apgar_mask: np.ndarray
    y_bin: np.ndarray
    y_bin_mask: np.ndarray
    tabular_feature_names: list[str]
    apgar_target_names: list[str]
    binary_target_names: list[str]


@dataclass(frozen=True)
class BinaryTarget:
    name: str
    kind: str
    train_y: np.ndarray
    train_mask: np.ndarray
    val_y: np.ndarray
    val_mask: np.ndarray
    test_y: np.ndarray
    test_mask: np.ndarray


def load_split(path: Path) -> SplitData:
    data = np.load(path, allow_pickle=False)
    return SplitData(
        X_tab=data["X_tab"].astype(np.float32),
        y_apgar=data["y_apgar"].astype(np.int64),
        y_apgar_mask=data["y_apgar_mask"].astype(bool),
        y_bin=data["y_bin"].astype(np.float32),
        y_bin_mask=data["y_bin_mask"].astype(bool),
        tabular_feature_names=[str(x) for x in data["tabular_feature_names"].tolist()],
        apgar_target_names=[str(x) for x in data["apgar_target_names"].tolist()],
        binary_target_names=[str(x) for x in data["binary_target_names"].tolist()],
    )


def make_targets(train: SplitData, val: SplitData, test: SplitData) -> list[BinaryTarget]:
    targets: list[BinaryTarget] = []
    for idx, name in enumerate(train.apgar_target_names):
        targets.append(
            BinaryTarget(
                name=f"{name}_below7",
                kind="apgar_below7",
                train_y=(train.y_apgar[:, idx] < 7).astype(np.int32),
                train_mask=train.y_apgar_mask[:, idx],
                val_y=(val.y_apgar[:, idx] < 7).astype(np.int32),
                val_mask=val.y_apgar_mask[:, idx],
                test_y=(test.y_apgar[:, idx] < 7).astype(np.int32),
                test_mask=test.y_apgar_mask[:, idx],
            )
        )
    for idx, name in enumerate(train.binary_target_names):
        targets.append(
            BinaryTarget(
                name=name,
                kind="binary_output",
                train_y=train.y_bin[:, idx].astype(np.int32),
                train_mask=train.y_bin_mask[:, idx],
                val_y=val.y_bin[:, idx].astype(np.int32),
                val_mask=val.y_bin_mask[:, idx],
                test_y=test.y_bin[:, idx].astype(np.int32),
                test_mask=test.y_bin_mask[:, idx],
            )
        )
    return targets


def raw_feature_group(feature_name: str) -> str:
    if feature_name.startswith("ctg_embedding_"):
        return "ctg_embedding"
    if "==" in feature_name:
        return feature_name.split("==", 1)[0]
    if "__" in feature_name:
        return feature_name.split("__", 1)[0]
    return feature_name


def xgb_feature_importance(
    model: XGBClassifier,
    feature_names: list[str],
    target_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    booster = model.get_booster()
    total_gain = booster.get_score(importance_type="total_gain")
    weight = booster.get_score(importance_type="weight")
    rows = []
    for idx, feature_name in enumerate(feature_names):
        key = f"f{idx}"
        rows.append(
            {
                "target": target_name,
                "feature": feature_name,
                "raw_feature": raw_feature_group(feature_name),
                "total_gain": float(total_gain.get(key, 0.0)),
                "split_count": float(weight.get(key, 0.0)),
            }
        )
    encoded = pd.DataFrame(rows).sort_values("total_gain", ascending=False)
    grouped = (
        encoded.groupby(["target", "raw_feature"], as_index=False)
        .agg(total_gain=("total_gain", "sum"), split_count=("split_count", "sum"))
        .sort_values("total_gain", ascending=False)
    )
    denom = float(grouped["total_gain"].sum())
    grouped["gain_fraction"] = grouped["total_gain"] / denom if denom > 0 else 0.0
    return encoded, grouped


def evaluate_masked(
    y_true: np.ndarray,
    prob: np.ndarray,
    mask: np.ndarray,
    threshold: float | None = None,
) -> dict[str, float]:
    y = y_true[mask].astype(int)
    p = prob[mask].astype(float)
    prevalence = float(y.mean()) if len(y) else float("nan")
    if len(y) == 0 or len(np.unique(y)) < 2:
        return {
            "prevalence": prevalence,
            "roc_auc": float("nan"),
            "pr_auc": float("nan"),
            "threshold": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
        }
    threshold = best_f1_threshold(y, p) if threshold is None else threshold
    metrics = compute_binary_metrics(y, p, threshold)
    return {"prevalence": prevalence, **metrics}


def train_one_target(
    target: BinaryTarget,
    train: SplitData,
    val: SplitData,
    test: SplitData,
    args: argparse.Namespace,
) -> tuple[XGBClassifier | None, dict[str, object]]:
    train_mask = target.train_mask.astype(bool)
    y_train = target.train_y[train_mask].astype(int)
    if len(y_train) == 0 or len(np.unique(y_train)) < 2:
        return None, {
            "target": target.name,
            "kind": target.kind,
            "status": "skipped_single_class_train",
        }

    positives = int(y_train.sum())
    negatives = int(len(y_train) - positives)
    scale_pos_weight = negatives / max(positives, 1)

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_lambda=args.reg_lambda,
        min_child_weight=args.min_child_weight,
        scale_pos_weight=scale_pos_weight,
        tree_method=args.tree_method,
        random_state=args.seed,
        n_jobs=args.n_jobs,
    )
    model.fit(
        train.X_tab[train_mask],
        y_train,
        eval_set=[(val.X_tab[target.val_mask], target.val_y[target.val_mask].astype(int))],
        verbose=False,
    )

    val_prob = model.predict_proba(val.X_tab)[:, 1]
    test_prob = model.predict_proba(test.X_tab)[:, 1]
    val_metrics = evaluate_masked(target.val_y, val_prob, target.val_mask)
    test_metrics = evaluate_masked(
        target.test_y,
        test_prob,
        target.test_mask,
        threshold=float(val_metrics["threshold"]),
    )

    payload: dict[str, object] = {
        "target": target.name,
        "kind": target.kind,
        "status": "ok",
        "train_n": int(train_mask.sum()),
        "train_positives": positives,
        "train_prevalence": float(positives / max(int(train_mask.sum()), 1)),
        "scale_pos_weight": float(scale_pos_weight),
        "val": val_metrics,
        "test": test_metrics,
    }
    return model, payload


def build_markdown(summary: pd.DataFrame, grouped_importance: pd.DataFrame, top_k: int) -> str:
    lines = [
        "# Registry-only XGBoost",
        "",
        "This experiment trains one binary XGBoost classifier per target using only the "
        "already encoded registry/tabular features from the multimodal NPZ files.",
        "",
        "## Test Metrics",
        "",
        "| Target | Prevalence | ROC-AUC | PR-AUC | F1 threshold | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    ok = summary[summary["status"] == "ok"].copy()
    for _, row in ok.iterrows():
        lines.append(
            f"| {row['target']} | {row['test_prevalence']:.4f} | "
            f"{row['test_roc_auc']:.4f} | {row['test_pr_auc']:.4f} | "
            f"{row['test_threshold']:.4f} | {row['test_precision']:.4f} | "
            f"{row['test_recall']:.4f} | {row['test_f1']:.4f} |"
        )

    lines.extend(["", f"## Top {top_k} Grouped Feature Importances", ""])
    for target, part in grouped_importance.groupby("target", sort=False):
        lines.append(f"### {target}")
        lines.append("")
        lines.append("| Rank | Raw feature | Gain fraction | Total gain | Split count |")
        lines.append("|---:|---|---:|---:|---:|")
        for rank, (_, row) in enumerate(part.head(top_k).iterrows(), start=1):
            lines.append(
                f"| {rank} | {row['raw_feature']} | {row['gain_fraction']:.4f} | "
                f"{row['total_gain']:.4f} | {row['split_count']:.0f} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train registry-only XGBoost baselines.")
    parser.add_argument("--config", default="configs/ctg3_multimodal.toml")
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Directory containing train/val/test NPZ.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--targets", default="all", help="Comma-separated target names or 'all'.")
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample-bytree", type=float, default=0.9)
    parser.add_argument("--reg-lambda", type=float, default=1.0)
    parser.add_argument("--min-child-weight", type=float, default=1.0)
    parser.add_argument("--tree-method", default="hist")
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=51)
    parser.add_argument("--top-k", type=int, default=15)
    args = parser.parse_args()

    cfg = load_multimodal_config(args.config)
    input_dir = Path(args.input_dir) if args.input_dir else cfg.sequence.output_dir
    output_dir = (
        Path(args.output_dir) if args.output_dir else cfg.paths.artifacts_dir / "xgboost_registry"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "models").mkdir(exist_ok=True)
    (output_dir / "importance").mkdir(exist_ok=True)

    train = load_split(input_dir / "train.npz")
    val = load_split(input_dir / "val.npz")
    test = load_split(input_dir / "test.npz")
    targets = make_targets(train, val, test)
    if args.targets != "all":
        requested = {x.strip() for x in args.targets.split(",") if x.strip()}
        targets = [target for target in targets if target.name in requested]
        missing = sorted(requested - {target.name for target in targets})
        if missing:
            raise ValueError(f"Unknown targets requested: {missing}")

    print(f"Input dir:  {input_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Features:   {len(train.tabular_feature_names)}")
    print(f"Targets:    {[target.name for target in targets]}")

    summary_rows: list[dict[str, object]] = []
    all_encoded_importance: list[pd.DataFrame] = []
    all_grouped_importance: list[pd.DataFrame] = []

    for target in targets:
        print(f"\nTraining {target.name}")
        model, payload = train_one_target(target, train, val, test, args)
        summary_row = {
            "target": payload["target"],
            "kind": payload["kind"],
            "status": payload["status"],
        }
        if payload["status"] == "ok":
            val_metrics = payload["val"]
            test_metrics = payload["test"]
            summary_row.update(
                {
                    "train_n": payload["train_n"],
                    "train_positives": payload["train_positives"],
                    "train_prevalence": payload["train_prevalence"],
                    "scale_pos_weight": payload["scale_pos_weight"],
                    **{f"val_{k}": v for k, v in val_metrics.items()},
                    **{f"test_{k}": v for k, v in test_metrics.items()},
                }
            )
            assert model is not None
            safe_name = "".join(
                ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in target.name
            )
            model.save_model(output_dir / "models" / f"{safe_name}.json")
            encoded, grouped = xgb_feature_importance(
                model, train.tabular_feature_names, target.name
            )
            encoded.to_csv(output_dir / "importance" / f"{safe_name}_encoded.csv", index=False)
            grouped.to_csv(output_dir / "importance" / f"{safe_name}_grouped.csv", index=False)
            all_encoded_importance.append(encoded)
            all_grouped_importance.append(grouped)
            print(
                f"  TEST ROC-AUC={test_metrics['roc_auc']:.4f} "
                f"PR-AUC={test_metrics['pr_auc']:.4f} "
                f"prevalence={test_metrics['prevalence']:.4f}"
            )
        else:
            print(f"  skipped: {payload['status']}")
        summary_rows.append(summary_row)

    summary = pd.DataFrame(summary_rows)
    summary_path = output_dir / "xgboost_registry_summary.csv"
    summary.to_csv(summary_path, index=False)

    payload_path = output_dir / "run_config.json"
    payload_path.write_text(
        json.dumps(
            {
                "config": args.config,
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "targets": [target.name for target in targets],
                "xgboost": {
                    "n_estimators": args.n_estimators,
                    "max_depth": args.max_depth,
                    "learning_rate": args.learning_rate,
                    "subsample": args.subsample,
                    "colsample_bytree": args.colsample_bytree,
                    "reg_lambda": args.reg_lambda,
                    "min_child_weight": args.min_child_weight,
                    "tree_method": args.tree_method,
                    "n_jobs": args.n_jobs,
                    "seed": args.seed,
                },
            },
            indent=2,
        )
    )

    if all_encoded_importance:
        encoded_all = pd.concat(all_encoded_importance, ignore_index=True)
        grouped_all = pd.concat(all_grouped_importance, ignore_index=True)
        encoded_all.to_csv(output_dir / "xgboost_encoded_importance.csv", index=False)
        grouped_all.to_csv(output_dir / "xgboost_grouped_importance.csv", index=False)
        (output_dir / "xgboost_registry_summary.md").write_text(
            build_markdown(summary, grouped_all, args.top_k)
        )

    print(f"\nWrote summary: {summary_path}")
    print(f"Wrote config:  {payload_path}")


if __name__ == "__main__":
    main()
