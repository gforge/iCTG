from __future__ import annotations

import argparse
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler, WeightedRandomSampler
from tqdm import tqdm

from ctg_ml.config import load_config
from ctg_ml.metrics import best_f1_threshold, compute_binary_metrics
from ctg_ml.models import TCNBinaryClassifier


def set_training_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    else:
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = True


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    requested = torch.device(device_arg)
    if requested.type == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU.")
        return torch.device("cpu")
    return requested


class BalancedBatchSampler(Sampler[list[int]]):
    """
    Ensures each batch contains at least `min_positives_per_batch` positive samples
    (sampled with replacement), then fills remaining slots with negatives.
    """

    def __init__(
        self,
        labels: np.ndarray,
        batch_size: int,
        min_positives_per_batch: int,
        seed: int,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if min_positives_per_batch < 0:
            raise ValueError("min_positives_per_batch must be >= 0")

        self.labels = labels.astype(np.int64)
        self.batch_size = int(batch_size)
        self.min_positives_per_batch = int(min_positives_per_batch)
        self.seed = int(seed)
        self._epoch = 0

        self.pos_indices = np.where(self.labels == 1)[0]
        self.neg_indices = np.where(self.labels == 0)[0]
        self.n_samples = len(self.labels)
        self.num_batches = int(math.ceil(self.n_samples / self.batch_size))

        if len(self.pos_indices) == 0 or len(self.neg_indices) == 0:
            raise ValueError(
                "BalancedBatchSampler requires both positive and negative samples in training data."
            )

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self._epoch)
        self._epoch += 1

        for batch_idx in range(self.num_batches):
            if batch_idx < self.num_batches - 1:
                current_bs = self.batch_size
            else:
                current_bs = self.n_samples - (self.batch_size * (self.num_batches - 1))
                current_bs = max(current_bs, 1)

            pos_target = self.min_positives_per_batch
            if current_bs > 1:
                pos_target = min(pos_target, current_bs - 1)
            pos_target = max(min(pos_target, current_bs), 0)
            neg_target = current_bs - pos_target

            pos = rng.choice(self.pos_indices, size=pos_target, replace=True) if pos_target else np.array([], dtype=np.int64)
            neg = rng.choice(self.neg_indices, size=neg_target, replace=True) if neg_target else np.array([], dtype=np.int64)
            batch = np.concatenate([pos, neg]).astype(np.int64, copy=False)
            rng.shuffle(batch)
            yield batch.tolist()


class NPZSequenceDataset(Dataset):
    """
    Expected NPZ layout:
      - X: float32 array of shape (N, C, T)  -> channels: [FHR, toco, fhr_missing_mask]
        and optionally toco/padding masks depending on preprocessing config
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
def predict_probs(
    model: nn.Module, loader: DataLoader, device: torch.device, use_amp: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for x, y in loader:
        x = x.to(device, non_blocking=(device.type == "cuda"))
        with torch.autocast(device_type=device.type, enabled=(use_amp and device.type == "cuda")):
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
    scaler: torch.cuda.amp.GradScaler | None = None,
    use_amp: bool = True,
    gradient_clip_norm: float = 0.0,
    show_progress: bool = True,
) -> float:
    model.train()
    running_loss = 0.0
    n = 0
    iterator = tqdm(loader, desc="train", leave=False, unit="batch") if show_progress else loader
    for x, y in iterator:
        x = x.to(device, non_blocking=(device.type == "cuda"))
        y = y.to(device, non_blocking=(device.type == "cuda"))
        optimizer.zero_grad(set_to_none=True)
        amp_enabled = bool(use_amp and device.type == "cuda")
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(x)
            loss = criterion(logits, y)

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            if gradient_clip_norm and gradient_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if gradient_clip_norm and gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
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
    use_amp: bool = True,
    show_progress: bool = False,
) -> float:
    model.eval()
    running_loss = 0.0
    n = 0
    iterator = tqdm(loader, desc="eval", leave=False, unit="batch") if show_progress else loader
    for x, y in iterator:
        x = x.to(device, non_blocking=(device.type == "cuda"))
        y = y.to(device, non_blocking=(device.type == "cuda"))
        with torch.autocast(device_type=device.type, enabled=(use_amp and device.type == "cuda")):
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
    parser.add_argument("--device", default="auto", help="auto|cpu|cuda|cuda:0 ...")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_training_seed(cfg.tcn.seed, cfg.tcn.deterministic)
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
    if train_ds.channel_names:
        print(f"Channels:   {train_ds.channel_names}")
    print(f"Val NPZ:   {val_npz} X={val_ds.X.shape} positives={int((val_ds.y == 1).sum())}/{len(val_ds)}")
    if test_ds is not None:
        print(
            f"Test NPZ:  {test_npz} X={test_ds.X.shape} positives={int((test_ds.y == 1).sum())}/{len(test_ds)}"
        )
    print(
        f"Normalization (train only): FHR mean/std={means[0]:.3f}/{stds[0]:.3f}, "
        f"toco mean/std={means[1]:.3f}/{stds[1]:.3f}"
    )
    print(f"Training seed: {cfg.tcn.seed} (deterministic={cfg.tcn.deterministic})")

    device = resolve_device(args.device)
    use_cuda = device.type == "cuda"
    use_amp = bool(cfg.tcn.use_amp and use_cuda)
    print(f"Device: {device} (cuda_available={torch.cuda.is_available()}, amp={use_amp})")

    train_loader_kwargs = {
        "dataset": train_ds,
        "batch_size": cfg.tcn.batch_size,
        "num_workers": 0,
        "pin_memory": use_cuda,
    }
    loader_generator = torch.Generator()
    loader_generator.manual_seed(cfg.tcn.seed)
    if cfg.tcn.use_balanced_batch_sampler:
        train_labels = train_ds.y.astype(np.int64)
        class_counts = np.bincount(train_labels, minlength=2)
        batch_sampler = BalancedBatchSampler(
            labels=train_labels,
            batch_size=cfg.tcn.batch_size,
            min_positives_per_batch=cfg.tcn.balanced_min_positives_per_batch,
            seed=cfg.tcn.seed,
        )
        train_loader = DataLoader(
            dataset=train_ds,
            batch_sampler=batch_sampler,
            num_workers=0,
            pin_memory=use_cuda,
        )
        print(
            "Train sampler: balanced "
            f"(class_counts={class_counts.tolist()}, min_pos_per_batch={cfg.tcn.balanced_min_positives_per_batch})"
        )
    elif cfg.tcn.use_weighted_sampler:
        train_labels = train_ds.y.astype(np.int64)
        class_counts = np.bincount(train_labels, minlength=2)
        class_weights = np.zeros(2, dtype=np.float64)
        for cls in (0, 1):
            if class_counts[cls] > 0:
                class_weights[cls] = 1.0 / float(class_counts[cls])
        sample_weights = class_weights[train_labels]
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(train_ds),
            replacement=True,
            generator=loader_generator,
        )
        train_loader = DataLoader(**train_loader_kwargs, sampler=sampler)
        print(
            "Train sampler: weighted "
            f"(class_counts={class_counts.tolist()}, class_weights={class_weights.tolist()})"
        )
    else:
        train_loader = DataLoader(**train_loader_kwargs, shuffle=True, generator=loader_generator)
        print("Train sampler: standard shuffle")
    val_loader = DataLoader(
        val_ds, batch_size=cfg.tcn.batch_size, shuffle=False, num_workers=0, pin_memory=use_cuda
    )
    test_loader = (
        DataLoader(test_ds, batch_size=cfg.tcn.batch_size, shuffle=False, num_workers=0, pin_memory=use_cuda)
        if test_ds is not None
        else None
    )

    model = TCNBinaryClassifier(
        in_channels=train_ds.X.shape[1],
        channels=cfg.tcn.channels,
        kernel_size=cfg.tcn.kernel_size,
        dropout=cfg.tcn.dropout,
    ).to(device)

    y_train = train_ds.y
    pos = float((y_train == 1).sum())
    neg = float((y_train == 0).sum())
    pos_weight_value = neg / max(pos, 1.0)
    if cfg.tcn.use_balanced_batch_sampler and cfg.tcn.disable_pos_weight_with_balanced_sampler:
        pos_weight_value = 1.0
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.tcn.learning_rate,
        weight_decay=cfg.tcn.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_pr_auc = float("-inf")
    best_val_pr_auc_for_stop = float("-inf")
    ckpt_dir = cfg.paths.artifacts_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "best_tcn.pt"

    history_rows: list[dict[str, float]] = []
    show_progress = not args.no_progress
    epochs_since_improve = 0
    early_cfg = cfg.tcn
    if early_cfg.early_stopping_enabled and early_cfg.epochs > 0:
        print(
            "Early stopping: enabled "
            f"(min_epochs={early_cfg.early_stopping_min_epochs}, "
            f"patience={early_cfg.early_stopping_patience}, "
            f"min_delta={early_cfg.early_stopping_min_delta})"
        )
    print(
        f"Training options: gradient_clip_norm={cfg.tcn.gradient_clip_norm}, "
        f"use_weighted_sampler={cfg.tcn.use_weighted_sampler}, "
        f"use_balanced_batch_sampler={cfg.tcn.use_balanced_batch_sampler}, "
        f"pos_weight={pos_weight_value:.4f}"
    )

    for epoch in range(1, cfg.tcn.epochs + 1):
        print(f"\nEpoch {epoch}/{cfg.tcn.epochs}")
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler=scaler,
            use_amp=use_amp,
            gradient_clip_norm=cfg.tcn.gradient_clip_norm,
            show_progress=show_progress,
        )
        val_loss = eval_loss(model, val_loader, criterion, device, use_amp=use_amp, show_progress=False)
        y_val, val_prob = predict_probs(model, val_loader, device, use_amp=use_amp)
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

        improved_for_stop = False
        if val_metrics["pr_auc"] > (best_val_pr_auc_for_stop + early_cfg.early_stopping_min_delta):
            best_val_pr_auc_for_stop = val_metrics["pr_auc"]
            epochs_since_improve = 0
            improved_for_stop = True
        else:
            epochs_since_improve += 1

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

        if early_cfg.early_stopping_enabled:
            if improved_for_stop:
                print(
                    f"Early stopping monitor: significant PR-AUC improvement "
                    f"(best={best_val_pr_auc_for_stop:.4f})"
                )
            else:
                print(
                    f"Early stopping monitor: no significant improvement "
                    f"({epochs_since_improve}/{early_cfg.early_stopping_patience})"
                )

            if (
                epoch >= early_cfg.early_stopping_min_epochs
                and epochs_since_improve >= early_cfg.early_stopping_patience
            ):
                print(
                    "Early stopping triggered "
                    f"at epoch {epoch} (best monitored val PR-AUC={best_val_pr_auc_for_stop:.4f})"
                )
                break

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

    y_val, val_prob = predict_probs(model, val_loader, device, use_amp=use_amp)
    final_thr = best_f1_threshold(y_val.astype(int), val_prob.astype(float))
    val_metrics = compute_binary_metrics(y_val.astype(int), val_prob.astype(float), threshold=final_thr)
    print(
        f"VAL (best ckpt, tuned thr): thr={final_thr:.3f} ROC-AUC={val_metrics['roc_auc']:.4f} "
        f"PR-AUC={val_metrics['pr_auc']:.4f} P={val_metrics['precision']:.4f} "
        f"R={val_metrics['recall']:.4f} F1={val_metrics['f1']:.4f}"
    )

    if test_loader is not None:
        y_test, test_prob = predict_probs(model, test_loader, device, use_amp=use_amp)
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
