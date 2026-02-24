from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ctg_ml.config import load_config
from ctg_ml.metrics import best_f1_threshold, compute_binary_metrics
from ctg_ml.models import TCNBinaryClassifier


class NPZSequenceDataset(Dataset):
    """
    Expected NPZ layout:
      - X: float32 array of shape (N, C, T)  -> channels: [FHR, toco, fhr_missing_mask]
      - y: int64/float array of shape (N,)
      - baby_ids: optional array of shape (N,)
    """

    def __init__(self, path: str | Path) -> None:
        data = np.load(path, allow_pickle=False)
        self.X = data["X"].astype(np.float32)
        self.y = data["y"].astype(np.float32)
        if self.X.ndim != 3 or self.X.shape[1] < 2:
            raise ValueError(f"Expected X shape (N,C,T) with C>=2, got {self.X.shape}")
        if len(self.X) != len(self.y):
            raise ValueError("X/y length mismatch")
        self.channel_names = None
        if "channels" in data:
            self.channel_names = [str(x) for x in data["channels"].tolist()]

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.X[idx])
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        return x, y


def compute_signal_stats(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute mean/std for the first two signal channels (FHR, toco), ignoring NaNs.
    Returns arrays of shape (2,).
    """
    means = np.zeros(2, dtype=np.float32)
    stds = np.ones(2, dtype=np.float32)
    for ch in range(2):
        vals = X[:, ch, :].reshape(-1)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            means[ch] = 0.0
            stds[ch] = 1.0
        else:
            means[ch] = np.float32(vals.mean())
            std = float(vals.std())
            stds[ch] = np.float32(std if std > 1e-6 else 1.0)
    return means, stds


def normalize_inplace(X: np.ndarray, means: np.ndarray, stds: np.ndarray) -> None:
    for ch in range(min(2, X.shape[1])):
        channel = X[:, ch, :]
        finite = np.isfinite(channel)
        channel[finite] = (channel[finite] - means[ch]) / stds[ch]
        channel[~finite] = 0.0
        X[:, ch, :] = channel

    if X.shape[1] > 2:
        mask = X[:, 2:, :]
        mask[~np.isfinite(mask)] = 1.0
        X[:, 2:, :] = mask


@torch.no_grad()
def predict_probs(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        prob = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)
        probs.append(prob)
        ys.append(y.numpy().astype(np.int32))
    return np.concatenate(ys), np.concatenate(probs)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    show_progress: bool = True,
) -> float:
    model.train()
    running_loss = 0.0
    n = 0
    iterator = tqdm(loader, desc="train", leave=False, unit="batch") if show_progress else loader
    for x, y in iterator:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        batch_size = x.size(0)
        running_loss += float(loss.detach().cpu()) * batch_size
        n += batch_size
        if show_progress:
            iterator.set_postfix(loss=f"{running_loss / max(n, 1):.4f}")
    return running_loss / max(n, 1)


@torch.no_grad()
def eval_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    show_progress: bool = False,
) -> float:
    model.eval()
    running_loss = 0.0
    n = 0
    iterator = tqdm(loader, desc="eval", leave=False, unit="batch") if show_progress else loader
    for x, y in iterator:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        batch_size = x.size(0)
        running_loss += float(loss.detach().cpu()) * batch_size
        n += batch_size
    return running_loss / max(n, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a TCN on preprocessed fixed-length CTG sequences (NPZ).")
    parser.add_argument("--config", default="configs/default.toml")
    parser.add_argument("--train-npz", help="Path to train sequence NPZ (X, y)")
    parser.add_argument("--val-npz", help="Path to val sequence NPZ (X, y)")
    parser.add_argument("--test-npz", help="Path to test sequence NPZ (X, y)")
    parser.add_argument("--no-progress", action="store_true", help="Disable batch progress bars")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seq_out_dir = cfg.sequence.output_dir
    train_npz = Path(args.train_npz) if args.train_npz else seq_out_dir / "train.npz"
    val_npz = Path(args.val_npz) if args.val_npz else seq_out_dir / "val.npz"
    test_npz = Path(args.test_npz) if args.test_npz else seq_out_dir / "test.npz"

    if not train_npz.exists() or not val_npz.exists():
        print("Missing preprocessed sequence NPZ files.")
        print("Run preprocessing first:")
        print("  uv run python scripts/preprocess_tcn.py")
        print(f"Expected at least: {train_npz} and {val_npz}")
        return

    train_ds = NPZSequenceDataset(train_npz)
    val_ds = NPZSequenceDataset(val_npz)
    test_ds = NPZSequenceDataset(test_npz) if test_npz.exists() else None
    if len(train_ds) == 0 or len(val_ds) == 0:
        print("Train/val NPZ is empty. This usually means preprocessing dropped too many sequences.")
        print("Try `uv run python scripts/preprocess_tcn.py --pad-short` or reduce window length.")
        return

    means, stds = compute_signal_stats(train_ds.X)
    normalize_inplace(train_ds.X, means, stds)
    normalize_inplace(val_ds.X, means, stds)
    if test_ds is not None:
        normalize_inplace(test_ds.X, means, stds)

    print(
        f"Train NPZ: {train_npz} X={train_ds.X.shape} positives={int((train_ds.y == 1).sum())}/{len(train_ds)}"
    )
    print(f"Val NPZ:   {val_npz} X={val_ds.X.shape} positives={int((val_ds.y == 1).sum())}/{len(val_ds)}")
    if test_ds is not None:
        print(
            f"Test NPZ:  {test_npz} X={test_ds.X.shape} positives={int((test_ds.y == 1).sum())}/{len(test_ds)}"
        )
    print(
        f"Normalization (train only): FHR mean/std={means[0]:.3f}/{stds[0]:.3f}, "
        f"toco mean/std={means[1]:.3f}/{stds[1]:.3f}"
    )

    train_loader = DataLoader(train_ds, batch_size=cfg.tcn.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg.tcn.batch_size, shuffle=False, num_workers=0)
    test_loader = (
        DataLoader(test_ds, batch_size=cfg.tcn.batch_size, shuffle=False, num_workers=0)
        if test_ds is not None
        else None
    )

    device = torch.device(args.device)
    model = TCNBinaryClassifier(
        in_channels=train_ds.X.shape[1],
        channels=cfg.tcn.channels,
        kernel_size=cfg.tcn.kernel_size,
        dropout=cfg.tcn.dropout,
    ).to(device)

    y_train = train_ds.y
    pos = float((y_train == 1).sum())
    neg = float((y_train == 0).sum())
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.tcn.learning_rate,
        weight_decay=cfg.tcn.weight_decay,
    )

    best_val_pr_auc = float("-inf")
    ckpt_dir = cfg.paths.artifacts_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "best_tcn.pt"

    history_rows: list[dict[str, float]] = []
    show_progress = not args.no_progress
    for epoch in range(1, cfg.tcn.epochs + 1):
        print(f"\nEpoch {epoch}/{cfg.tcn.epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, show_progress=show_progress)
        val_loss = eval_loss(model, val_loader, criterion, device, show_progress=False)
        y_val, val_prob = predict_probs(model, val_loader, device)
        val_thr = best_f1_threshold(y_val.astype(int), val_prob.astype(float))
        val_metrics = compute_binary_metrics(y_val.astype(int), val_prob.astype(float), threshold=val_thr)
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_roc_auc": val_metrics["roc_auc"],
                "val_pr_auc": val_metrics["pr_auc"],
                "val_f1": val_metrics["f1"],
                "val_threshold": val_thr,
            }
        )
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.5f} val_loss={val_loss:.5f} "
            f"val_PR-AUC={val_metrics['pr_auc']:.4f} val_ROC-AUC={val_metrics['roc_auc']:.4f} "
            f"val_F1={val_metrics['f1']:.4f} thr={val_thr:.3f}"
        )

        if val_metrics["pr_auc"] > best_val_pr_auc:
            best_val_pr_auc = val_metrics["pr_auc"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "val_pr_auc": val_metrics["pr_auc"],
                    "val_roc_auc": val_metrics["roc_auc"],
                    "val_loss": val_loss,
                    "val_threshold": val_thr,
                    # Store plain Python lists so torch.load(weights_only=True) remains compatible.
                    "train_signal_means": means.tolist(),
                    "train_signal_stds": stds.tolist(),
                },
                ckpt_path,
            )
            print(f"Saved checkpoint: {ckpt_path}")

    history_path = cfg.paths.artifacts_dir / "tcn_history.csv"
    import pandas as pd  # local import to avoid hard dependency at import time

    pd.DataFrame(history_rows).to_csv(history_path, index=False)
    print(f"\nSaved training history to {history_path}")

    if ckpt_path.exists():
        # Local checkpoint generated by this script. PyTorch 2.6 changed torch.load default
        # to weights_only=True, which rejects older checkpoints containing numpy arrays.
        try:
            state = torch.load(ckpt_path, map_location=device, weights_only=False)
        except TypeError:
            state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        print(
            f"Loaded best checkpoint by val PR-AUC={float(state.get('val_pr_auc', float('nan'))):.4f}"
        )

    y_val, val_prob = predict_probs(model, val_loader, device)
    final_thr = best_f1_threshold(y_val.astype(int), val_prob.astype(float))
    val_metrics = compute_binary_metrics(y_val.astype(int), val_prob.astype(float), threshold=final_thr)
    print(
        f"VAL (best ckpt, tuned thr): thr={final_thr:.3f} ROC-AUC={val_metrics['roc_auc']:.4f} "
        f"PR-AUC={val_metrics['pr_auc']:.4f} P={val_metrics['precision']:.4f} "
        f"R={val_metrics['recall']:.4f} F1={val_metrics['f1']:.4f}"
    )

    if test_loader is not None:
        y_test, test_prob = predict_probs(model, test_loader, device)
        test_metrics_tuned = compute_binary_metrics(y_test.astype(int), test_prob.astype(float), threshold=final_thr)
        test_metrics_default = compute_binary_metrics(y_test.astype(int), test_prob.astype(float), threshold=0.5)
        print(
            f"TEST (val-tuned thr): thr={final_thr:.3f} ROC-AUC={test_metrics_tuned['roc_auc']:.4f} "
            f"PR-AUC={test_metrics_tuned['pr_auc']:.4f} P={test_metrics_tuned['precision']:.4f} "
            f"R={test_metrics_tuned['recall']:.4f} F1={test_metrics_tuned['f1']:.4f} "
            f"TN={test_metrics_tuned['tn']} FP={test_metrics_tuned['fp']} "
            f"FN={test_metrics_tuned['fn']} TP={test_metrics_tuned['tp']}"
        )
        print(
            f"TEST (thr=0.5): thr=0.500 ROC-AUC={test_metrics_default['roc_auc']:.4f} "
            f"PR-AUC={test_metrics_default['pr_auc']:.4f} P={test_metrics_default['precision']:.4f} "
            f"R={test_metrics_default['recall']:.4f} F1={test_metrics_default['f1']:.4f} "
            f"TN={test_metrics_default['tn']} FP={test_metrics_default['fp']} "
            f"FN={test_metrics_default['fn']} TP={test_metrics_default['tp']}"
        )


if __name__ == "__main__":
    main()
