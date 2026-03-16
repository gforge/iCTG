from __future__ import annotations

import argparse
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ctg_ml.ctg2_config import load_ctg2_config
from ctg_ml.metrics import compute_binary_metrics
from ctg_ml.models import MultimodalMultitaskTCN


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
    elif hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = True


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    requested = torch.device(device_arg)
    if requested.type == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU.")
        return torch.device("cpu")
    return requested


class MultimodalNPZDataset(Dataset):
    def __init__(self, path: str | Path) -> None:
        data = np.load(path, allow_pickle=False)
        self.X_seq = data["X_seq"].astype(np.float32)
        self.X_tab = data["X_tab"].astype(np.float32)
        self.y_reg = data["y_reg"].astype(np.float32)
        self.y_reg_mask = data["y_reg_mask"].astype(np.float32)
        self.y_bin = data["y_bin"].astype(np.float32)
        self.y_bin_mask = data["y_bin_mask"].astype(np.float32)
        self.sequence_channels = [str(x) for x in data["sequence_channels"].tolist()]
        self.tabular_feature_names = [str(x) for x in data["tabular_feature_names"].tolist()]
        self.regression_target_names = [str(x) for x in data["regression_target_names"].tolist()]
        self.binary_target_names = [str(x) for x in data["binary_target_names"].tolist()]
        if len(self.X_seq) != len(self.X_tab):
            raise ValueError("Sequence/tabular length mismatch")

    def __len__(self) -> int:
        return len(self.X_seq)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        return (
            torch.from_numpy(self.X_seq[idx]),
            torch.from_numpy(self.X_tab[idx]),
            torch.from_numpy(self.y_reg[idx]),
            torch.from_numpy(self.y_reg_mask[idx]),
            torch.from_numpy(self.y_bin[idx]),
            torch.from_numpy(self.y_bin_mask[idx]),
        )


def compute_signal_stats(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    means = np.zeros(2, dtype=np.float32)
    stds = np.ones(2, dtype=np.float32)
    for ch in range(2):
        vals = X[:, ch, :].reshape(-1)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        means[ch] = np.float32(vals.mean())
        std = float(vals.std())
        stds[ch] = np.float32(std if std > 1e-6 else 1.0)
    return means, stds


def normalize_sequences_inplace(X: np.ndarray, means: np.ndarray, stds: np.ndarray) -> None:
    for ch in range(min(2, X.shape[1])):
        finite = np.isfinite(X[:, ch, :])
        X[:, ch, :][finite] = (X[:, ch, :][finite] - means[ch]) / stds[ch]
        X[:, ch, :][~finite] = 0.0
    if X.shape[1] > 2:
        X[:, 2:, :][~np.isfinite(X[:, 2:, :])] = 0.0


def masked_multitask_loss(
    reg_pred: torch.Tensor,
    bin_logits: torch.Tensor,
    y_reg: torch.Tensor,
    y_reg_mask: torch.Tensor,
    y_bin: torch.Tensor,
    y_bin_mask: torch.Tensor,
    pos_weight: torch.Tensor,
    regression_weight: float,
    binary_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    reg_loss_raw = F.smooth_l1_loss(reg_pred / 10.0, y_reg / 10.0, reduction="none")
    reg_denom = y_reg_mask.sum().clamp_min(1.0)
    reg_loss = (reg_loss_raw * y_reg_mask).sum() / reg_denom

    bin_loss_raw = F.binary_cross_entropy_with_logits(
        bin_logits,
        y_bin,
        pos_weight=pos_weight,
        reduction="none",
    )
    bin_denom = y_bin_mask.sum().clamp_min(1.0)
    bin_loss = (bin_loss_raw * y_bin_mask).sum() / bin_denom

    total = (regression_weight * reg_loss) + (binary_weight * bin_loss)
    return total, reg_loss.detach(), bin_loss.detach()


@torch.no_grad()
def evaluate_dataset(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    pos_weight: torch.Tensor,
    regression_weight: float,
    binary_weight: float,
    regression_names: list[str],
    binary_names: list[str],
) -> dict[str, object]:
    model.eval()
    total_loss = 0.0
    total_items = 0
    reg_preds: list[np.ndarray] = []
    reg_true: list[np.ndarray] = []
    reg_mask: list[np.ndarray] = []
    bin_probs: list[np.ndarray] = []
    bin_true: list[np.ndarray] = []
    bin_mask: list[np.ndarray] = []

    for x_seq, x_tab, y_reg, y_reg_mask, y_bin, y_bin_mask in loader:
        x_seq = x_seq.to(device, non_blocking=(device.type == "cuda"))
        x_tab = x_tab.to(device, non_blocking=(device.type == "cuda"))
        y_reg = y_reg.to(device, non_blocking=(device.type == "cuda"))
        y_reg_mask = y_reg_mask.to(device, non_blocking=(device.type == "cuda"))
        y_bin = y_bin.to(device, non_blocking=(device.type == "cuda"))
        y_bin_mask = y_bin_mask.to(device, non_blocking=(device.type == "cuda"))
        with torch.autocast(device_type=device.type, enabled=(use_amp and device.type == "cuda")):
            pred_reg, logits_bin = model(x_seq, x_tab)
            loss, _, _ = masked_multitask_loss(
                pred_reg,
                logits_bin,
                y_reg,
                y_reg_mask,
                y_bin,
                y_bin_mask,
                pos_weight,
                regression_weight,
                binary_weight,
            )
        bs = x_seq.size(0)
        total_loss += float(loss.detach().cpu()) * bs
        total_items += bs
        reg_preds.append(pred_reg.detach().cpu().numpy())
        reg_true.append(y_reg.detach().cpu().numpy())
        reg_mask.append(y_reg_mask.detach().cpu().numpy())
        bin_probs.append(torch.sigmoid(logits_bin).detach().cpu().numpy())
        bin_true.append(y_bin.detach().cpu().numpy())
        bin_mask.append(y_bin_mask.detach().cpu().numpy())

    reg_pred_arr = np.concatenate(reg_preds, axis=0)
    reg_true_arr = np.concatenate(reg_true, axis=0)
    reg_mask_arr = np.concatenate(reg_mask, axis=0)
    bin_prob_arr = np.concatenate(bin_probs, axis=0)
    bin_true_arr = np.concatenate(bin_true, axis=0)
    bin_mask_arr = np.concatenate(bin_mask, axis=0)

    reg_metrics: dict[str, dict[str, float]] = {}
    for idx, name in enumerate(regression_names):
        valid = reg_mask_arr[:, idx] > 0
        if valid.sum() == 0:
            reg_metrics[name] = {"mae": float("nan"), "rmse": float("nan")}
            continue
        err = reg_pred_arr[valid, idx] - reg_true_arr[valid, idx]
        reg_metrics[name] = {
            "mae": float(np.abs(err).mean()),
            "rmse": float(np.sqrt(np.mean(err**2))),
        }

    bin_metrics: dict[str, dict[str, float]] = {}
    pr_values: list[float] = []
    for idx, name in enumerate(binary_names):
        valid = bin_mask_arr[:, idx] > 0
        if valid.sum() == 0:
            bin_metrics[name] = {"roc_auc": float("nan"), "pr_auc": float("nan"), "prevalence": float("nan")}
            continue
        y_true = bin_true_arr[valid, idx].astype(int)
        prob = bin_prob_arr[valid, idx].astype(float)
        prevalence = float(y_true.mean()) if len(y_true) else float("nan")
        if len(np.unique(y_true)) < 2:
            metrics = {"roc_auc": float("nan"), "pr_auc": float("nan"), "prevalence": prevalence}
        else:
            out = compute_binary_metrics(y_true, prob, threshold=0.5)
            metrics = {"roc_auc": out["roc_auc"], "pr_auc": out["pr_auc"], "prevalence": prevalence}
            if np.isfinite(out["pr_auc"]):
                pr_values.append(out["pr_auc"])
        bin_metrics[name] = metrics

    apgar5_mae = reg_metrics.get("apgar5", {}).get("mae", float("nan"))
    mean_binary_pr_auc = float(np.mean(pr_values)) if pr_values else float("nan")
    return {
        "loss": total_loss / max(total_items, 1),
        "regression": reg_metrics,
        "binary": bin_metrics,
        "apgar5_mae": apgar5_mae,
        "mean_binary_pr_auc": mean_binary_pr_auc,
    }


def format_eval(tag: str, metrics: dict[str, object]) -> None:
    print(
        f"{tag}: loss={metrics['loss']:.5f} apgar5_MAE={metrics['apgar5_mae']:.4f} "
        f"mean_binary_PR-AUC={metrics['mean_binary_pr_auc']:.4f}"
    )
    reg = metrics["regression"]
    for name, vals in reg.items():
        print(f"  {name}: MAE={vals['mae']:.4f} RMSE={vals['rmse']:.4f}")
    binary = metrics["binary"]
    for name, vals in binary.items():
        print(
            f"  {name}: prevalence={vals['prevalence']:.4f} "
            f"ROC-AUC={vals['roc_auc']:.4f} PR-AUC={vals['pr_auc']:.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a multimodal multitask TCN on CTG2 NPZ files.")
    parser.add_argument("--config", default="configs/ctg2_multimodal.toml")
    parser.add_argument("--train-npz", default=None)
    parser.add_argument("--val-npz", default=None)
    parser.add_argument("--test-npz", default=None)
    parser.add_argument("--device", default="auto", help="auto|cpu|cuda|cuda:0")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    cfg = load_ctg2_config(args.config)
    set_training_seed(cfg.train.seed, cfg.train.deterministic)

    out_dir = cfg.sequence.output_dir
    train_npz = Path(args.train_npz) if args.train_npz else out_dir / "train.npz"
    val_npz = Path(args.val_npz) if args.val_npz else out_dir / "val.npz"
    test_npz = Path(args.test_npz) if args.test_npz else out_dir / "test.npz"
    if not train_npz.exists() or not val_npz.exists() or not test_npz.exists():
        raise FileNotFoundError(
            "Missing multimodal NPZ files. Run `uv run python scripts/preprocess_ctg2_multimodal.py` first."
        )

    train_ds = MultimodalNPZDataset(train_npz)
    val_ds = MultimodalNPZDataset(val_npz)
    test_ds = MultimodalNPZDataset(test_npz)
    means, stds = compute_signal_stats(train_ds.X_seq)
    normalize_sequences_inplace(train_ds.X_seq, means, stds)
    normalize_sequences_inplace(val_ds.X_seq, means, stds)
    normalize_sequences_inplace(test_ds.X_seq, means, stds)

    device = resolve_device(args.device)
    use_cuda = device.type == "cuda"
    use_amp = bool(cfg.train.use_amp and use_cuda)

    print(
        f"Train NPZ: {train_npz} X_seq={train_ds.X_seq.shape} X_tab={train_ds.X_tab.shape} "
        f"regression_targets={train_ds.y_reg.shape[1]} binary_targets={train_ds.y_bin.shape[1]}"
    )
    print(f"Sequence channels: {train_ds.sequence_channels}")
    print(f"Tabular features:  {len(train_ds.tabular_feature_names)}")
    print(f"Regression targets: {train_ds.regression_target_names}")
    print(f"Binary targets:     {train_ds.binary_target_names}")
    print(
        f"Normalization (train only): FHR mean/std={means[0]:.3f}/{stds[0]:.3f}, "
        f"toco mean/std={means[1]:.3f}/{stds[1]:.3f}"
    )
    print(f"Training seed: {cfg.train.seed} (deterministic={cfg.train.deterministic})")
    print(f"Device: {device} (cuda_available={torch.cuda.is_available()}, amp={use_amp})")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=use_cuda,
    )
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=0, pin_memory=use_cuda)
    test_loader = DataLoader(
        test_ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=0, pin_memory=use_cuda
    )

    model = MultimodalMultitaskTCN(
        sequence_in_channels=train_ds.X_seq.shape[1],
        tabular_in_features=train_ds.X_tab.shape[1],
        tcn_channels=cfg.model.tcn_channels,
        kernel_size=cfg.model.kernel_size,
        dropout=cfg.model.dropout,
        tabular_hidden_dim=cfg.model.tabular_hidden_dim,
        fusion_hidden_dim=cfg.model.fusion_hidden_dim,
        num_regression_outputs=train_ds.y_reg.shape[1],
        num_binary_outputs=train_ds.y_bin.shape[1],
    ).to(device)

    pos_weight_values = []
    for idx in range(train_ds.y_bin.shape[1]):
        valid = train_ds.y_bin_mask[:, idx] > 0
        if valid.sum() == 0:
            pos_weight_values.append(1.0)
            continue
        positives = float(train_ds.y_bin[valid, idx].sum())
        negatives = float(valid.sum() - positives)
        pos_weight_values.append(negatives / max(positives, 1.0))
    pos_weight = torch.tensor(pos_weight_values, dtype=torch.float32, device=device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    ckpt_dir = cfg.paths.artifacts_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "best_ctg2_multimodal.pt"
    best_val_loss = float("inf")
    best_val_for_stop = float("inf")
    epochs_since_improve = 0
    history_rows: list[dict[str, float]] = []
    show_progress = not args.no_progress

    print(
        f"Training options: lr={cfg.train.learning_rate}, batch_size={cfg.train.batch_size}, "
        f"gradient_clip_norm={cfg.train.gradient_clip_norm}, regression_weight={cfg.train.regression_loss_weight}, "
        f"binary_weight={cfg.train.binary_loss_weight}"
    )
    if cfg.train.early_stopping_enabled:
        print(
            "Early stopping: enabled "
            f"(min_epochs={cfg.train.early_stopping_min_epochs}, patience={cfg.train.early_stopping_patience}, "
            f"min_delta={cfg.train.early_stopping_min_delta})"
        )

    for epoch in range(1, cfg.train.epochs + 1):
        print(f"\nEpoch {epoch}/{cfg.train.epochs}")
        model.train()
        running = 0.0
        n = 0
        iterator = tqdm(train_loader, desc="train", leave=False, unit="batch") if show_progress else train_loader
        for batch in iterator:
            x_seq, x_tab, y_reg, y_reg_mask, y_bin, y_bin_mask = batch
            x_seq = x_seq.to(device, non_blocking=use_cuda)
            x_tab = x_tab.to(device, non_blocking=use_cuda)
            y_reg = y_reg.to(device, non_blocking=use_cuda)
            y_reg_mask = y_reg_mask.to(device, non_blocking=use_cuda)
            y_bin = y_bin.to(device, non_blocking=use_cuda)
            y_bin_mask = y_bin_mask.to(device, non_blocking=use_cuda)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=(use_amp and device.type == "cuda")):
                pred_reg, logits_bin = model(x_seq, x_tab)
                loss, reg_loss, bin_loss = masked_multitask_loss(
                    pred_reg,
                    logits_bin,
                    y_reg,
                    y_reg_mask,
                    y_bin,
                    y_bin_mask,
                    pos_weight,
                    cfg.train.regression_loss_weight,
                    cfg.train.binary_loss_weight,
                )

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if cfg.train.gradient_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.train.gradient_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if cfg.train.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.train.gradient_clip_norm)
                optimizer.step()

            bs = x_seq.size(0)
            running += float(loss.detach().cpu()) * bs
            n += bs
            if show_progress:
                iterator.set_postfix(loss=f"{running / max(n, 1):.4f}", reg=f"{float(reg_loss):.4f}", bin=f"{float(bin_loss):.4f}")

        train_loss = running / max(n, 1)
        val_metrics = evaluate_dataset(
            model,
            val_loader,
            device,
            use_amp,
            pos_weight,
            cfg.train.regression_loss_weight,
            cfg.train.binary_loss_weight,
            train_ds.regression_target_names,
            train_ds.binary_target_names,
        )
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": float(val_metrics["loss"]),
                "val_apgar5_mae": float(val_metrics["apgar5_mae"]),
                "val_mean_binary_pr_auc": float(val_metrics["mean_binary_pr_auc"]),
            }
        )
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.5f} val_loss={val_metrics['loss']:.5f} "
            f"val_apgar5_MAE={val_metrics['apgar5_mae']:.4f} "
            f"val_mean_binary_PR-AUC={val_metrics['mean_binary_pr_auc']:.4f}"
        )

        if val_metrics["loss"] < (best_val_for_stop - cfg.train.early_stopping_min_delta):
            best_val_for_stop = float(val_metrics["loss"])
            epochs_since_improve = 0
            print(f"Early stopping monitor: significant val-loss improvement (best={best_val_for_stop:.5f})")
        else:
            epochs_since_improve += 1
            if cfg.train.early_stopping_enabled:
                print(
                    "Early stopping monitor: no significant improvement "
                    f"({epochs_since_improve}/{cfg.train.early_stopping_patience})"
                )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = float(val_metrics["loss"])
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "val_loss": float(val_metrics["loss"]),
                    "train_signal_means": means.tolist(),
                    "train_signal_stds": stds.tolist(),
                },
                ckpt_path,
            )
            print(f"Saved checkpoint: {ckpt_path}")

        if (
            cfg.train.early_stopping_enabled
            and epoch >= cfg.train.early_stopping_min_epochs
            and epochs_since_improve >= cfg.train.early_stopping_patience
        ):
            print(f"Early stopping triggered at epoch {epoch} (best monitored val_loss={best_val_for_stop:.5f})")
            break

    history_path = cfg.paths.artifacts_dir / "ctg2_multimodal_history.csv"
    pd.DataFrame(history_rows).to_csv(history_path, index=False)
    print(f"\nSaved training history to {history_path}")

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    print(f"Loaded best checkpoint by val_loss={state['val_loss']:.5f}")

    val_metrics = evaluate_dataset(
        model,
        val_loader,
        device,
        use_amp,
        pos_weight,
        cfg.train.regression_loss_weight,
        cfg.train.binary_loss_weight,
        train_ds.regression_target_names,
        train_ds.binary_target_names,
    )
    test_metrics = evaluate_dataset(
        model,
        test_loader,
        device,
        use_amp,
        pos_weight,
        cfg.train.regression_loss_weight,
        cfg.train.binary_loss_weight,
        train_ds.regression_target_names,
        train_ds.binary_target_names,
    )
    format_eval("VAL", val_metrics)
    format_eval("TEST", test_metrics)


if __name__ == "__main__":
    main()
