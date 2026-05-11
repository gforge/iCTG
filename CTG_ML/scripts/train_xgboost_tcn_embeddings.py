from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from train_xgboost_registry import (
    SplitData,
    build_markdown,
    make_targets,
    train_one_target,
    xgb_feature_importance,
)

from ctg_ml.models import MultimodalMultitaskTCN
from ctg_ml.multimodal_config import load_multimodal_config


def load_npz_metadata(path: Path) -> dict[str, object]:
    data = np.load(path, allow_pickle=False)
    return {
        "sequence_channels": [str(x) for x in data["sequence_channels"].tolist()],
        "tabular_feature_names": [str(x) for x in data["tabular_feature_names"].tolist()],
        "apgar_target_names": [str(x) for x in data["apgar_target_names"].tolist()],
        "binary_target_names": [str(x) for x in data["binary_target_names"].tolist()],
        "categorical_class_counts": [int(x) for x in data["categorical_class_counts"].tolist()],
        "num_regression_outputs": int(len(data["regression_target_names"])),
    }


def normalize_sequences_inplace(X: np.ndarray, means: np.ndarray, stds: np.ndarray) -> None:
    for ch in range(min(2, X.shape[1])):
        channel = X[:, ch, :]
        finite = np.isfinite(channel)
        channel[finite] = (channel[finite] - means[ch]) / stds[ch]
        channel[~finite] = 0.0
        X[:, ch, :] = channel
    if X.shape[1] > 2:
        masks = X[:, 2:, :]
        masks[~np.isfinite(masks)] = 0.0
        X[:, 2:, :] = masks


def load_model(
    checkpoint_path: Path,
    cfg_path: str,
    metadata: dict[str, object],
    device: torch.device,
) -> tuple[MultimodalMultitaskTCN, dict[str, object]]:
    cfg = load_multimodal_config(cfg_path)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = MultimodalMultitaskTCN(
        sequence_in_channels=len(metadata["sequence_channels"]),
        tabular_in_features=len(metadata["tabular_feature_names"]),
        tcn_channels=cfg.model.tcn_channels,
        kernel_size=cfg.model.kernel_size,
        dropout=cfg.model.dropout,
        tabular_hidden_dim=cfg.model.tabular_hidden_dim,
        fusion_hidden_dim=cfg.model.fusion_hidden_dim,
        num_apgar_outputs=len(metadata["apgar_target_names"]),
        categorical_output_dims=metadata["categorical_class_counts"],
        num_regression_outputs=metadata["num_regression_outputs"],
        num_binary_outputs=len(metadata["binary_target_names"]),
    ).to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model, state


@torch.no_grad()
def extract_split_features(
    npz_path: Path,
    model: MultimodalMultitaskTCN,
    means: np.ndarray,
    stds: np.ndarray,
    mode: str,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    data = np.load(npz_path, allow_pickle=False)
    X_seq = data["X_seq"].astype(np.float32)
    normalize_sequences_inplace(X_seq, means, stds)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_seq)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    embeddings: list[np.ndarray] = []
    for (x_seq,) in loader:
        x_seq = x_seq.to(device, non_blocking=(device.type == "cuda"))
        emb = model.sequence_encoder(x_seq).detach().cpu().numpy().astype(np.float32)
        embeddings.append(emb)
    X_emb = np.concatenate(embeddings, axis=0)

    if mode == "ctg_embedding":
        X = X_emb
    elif mode == "registry_plus_ctg_embedding":
        X_tab = data["X_tab"].astype(np.float32)
        X = np.concatenate([X_tab, X_emb], axis=1)
    else:
        raise ValueError(f"Unknown feature mode: {mode}")

    target_arrays = {
        "y_apgar": data["y_apgar"].astype(np.int64),
        "y_apgar_mask": data["y_apgar_mask"].astype(bool),
        "y_bin": data["y_bin"].astype(np.float32),
        "y_bin_mask": data["y_bin_mask"].astype(bool),
    }
    return X, target_arrays


def make_feature_names(metadata: dict[str, object], mode: str) -> list[str]:
    emb_dim = int(metadata.get("embedding_dim", 0))
    emb_names = [f"ctg_embedding_{idx:03d}" for idx in range(emb_dim)]
    if mode == "ctg_embedding":
        return emb_names
    if mode == "registry_plus_ctg_embedding":
        return [*metadata["tabular_feature_names"], *emb_names]
    raise ValueError(f"Unknown feature mode: {mode}")


def make_split_data(
    X: np.ndarray,
    arrays: dict[str, np.ndarray],
    feature_names: list[str],
    metadata: dict[str, object],
) -> SplitData:
    return SplitData(
        X_tab=X,
        y_apgar=arrays["y_apgar"],
        y_apgar_mask=arrays["y_apgar_mask"],
        y_bin=arrays["y_bin"],
        y_bin_mask=arrays["y_bin_mask"],
        tabular_feature_names=feature_names,
        apgar_target_names=metadata["apgar_target_names"],
        binary_target_names=metadata["binary_target_names"],
    )


def run_xgboost(
    train: SplitData,
    val: SplitData,
    test: SplitData,
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "models").mkdir(exist_ok=True)
    (output_dir / "importance").mkdir(exist_ok=True)

    targets = make_targets(train, val, test)
    if args.targets != "all":
        requested = {x.strip() for x in args.targets.split(",") if x.strip()}
        targets = [target for target in targets if target.name in requested]
        missing = sorted(requested - {target.name for target in targets})
        if missing:
            raise ValueError(f"Unknown targets requested: {missing}")

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
                model,
                train.tabular_feature_names,
                target.name,
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
    summary_path = output_dir / "xgboost_tcn_embedding_summary.csv"
    summary.to_csv(summary_path, index=False)
    if all_encoded_importance:
        encoded_all = pd.concat(all_encoded_importance, ignore_index=True)
        grouped_all = pd.concat(all_grouped_importance, ignore_index=True)
        encoded_all.to_csv(output_dir / "xgboost_encoded_importance.csv", index=False)
        grouped_all.to_csv(output_dir / "xgboost_grouped_importance.csv", index=False)
        markdown = build_markdown(summary, grouped_all, args.top_k)
        markdown = markdown.replace(
            "# Registry-only XGBoost",
            "# XGBoost With Frozen TCN Embeddings",
        )
        markdown = markdown.replace(
            "This experiment trains one binary XGBoost classifier per target using only the "
            "already encoded registry/tabular features from the multimodal NPZ files.",
            "This experiment trains one binary XGBoost classifier per target using frozen "
            f"TCN embedding features in `{args.feature_mode}` mode.",
        )
        (output_dir / "xgboost_tcn_embedding_summary.md").write_text(
            markdown
        )
    print(f"\nWrote summary: {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train XGBoost on frozen TCN embeddings.")
    parser.add_argument("--config", default="configs/ctg3_multimodal.toml")
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Directory containing train/val/test NPZ.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="TCN checkpoint to use as feature extractor.",
    )
    parser.add_argument(
        "--feature-mode",
        choices=["ctg_embedding", "registry_plus_ctg_embedding"],
        default="registry_plus_ctg_embedding",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--targets", default="all", help="Comma-separated target names or 'all'.")
    parser.add_argument("--device", default="auto", help="auto|cpu|cuda|cuda:0")
    parser.add_argument("--embedding-batch-size", type=int, default=256)
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
    checkpoint = (
        Path(args.checkpoint)
        if args.checkpoint
        else cfg.paths.artifacts_dir / "checkpoints" / "best_multimodal_tcn_multimodal.pt"
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else cfg.paths.artifacts_dir / f"xgboost_{args.feature_mode}"
    )
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU.")
        device = torch.device("cpu")

    metadata = load_npz_metadata(input_dir / "train.npz")
    model, state = load_model(checkpoint, args.config, metadata, device)
    means = np.asarray(state["train_signal_means"], dtype=np.float32)
    stds = np.asarray(state["train_signal_stds"], dtype=np.float32)
    metadata["embedding_dim"] = int(model.sequence_encoder.out_dim)
    feature_names = make_feature_names(metadata, args.feature_mode)

    print(f"Input dir:       {input_dir}")
    print(f"Checkpoint:      {checkpoint}")
    print(f"Feature mode:    {args.feature_mode}")
    print(f"Device:          {device}")
    print(f"Embedding dim:   {metadata['embedding_dim']}")

    split_features: dict[str, SplitData] = {}
    for split_name in ["train", "val", "test"]:
        print(f"Extracting {split_name} embeddings")
        X, arrays = extract_split_features(
            input_dir / f"{split_name}.npz",
            model,
            means,
            stds,
            args.feature_mode,
            args.embedding_batch_size,
            device,
        )
        split_features[split_name] = make_split_data(X, arrays, feature_names, metadata)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "config": args.config,
                "input_dir": str(input_dir),
                "checkpoint": str(checkpoint),
                "feature_mode": args.feature_mode,
                "output_dir": str(output_dir),
                "embedding_dim": metadata["embedding_dim"],
                "targets": args.targets,
                "xgboost_seed": args.seed,
            },
            indent=2,
        )
    )
    run_xgboost(
        split_features["train"],
        split_features["val"],
        split_features["test"],
        output_dir,
        args,
    )


if __name__ == "__main__":
    main()
