"""Masked-reconstruction pretraining of the TCN sequence encoder.

Masking/channel decision: the supervised channel layout (FHR, toco, one-hot
Hr1_SignalQuality, padding_mask) is kept *identical* and masked positions are zeroed in
the FHR/toco channels only. Adding a mask-indicator channel would change the first conv's
input width, so the pretrained weights could no longer be loaded strictly into
``MultimodalMultitaskTCN.sequence_encoder``. Zero in normalized space is exactly what
the supervised pipeline feeds for missing FHR/toco samples (``normalize_sequences_inplace``
maps non-finite values to 0), so "masked" and "missing" look the same to the encoder in
both stages.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ctg_ml.models import MaskedReconstructionTCN, MultimodalMultitaskTCN
from ctg_ml.multimodal_config import MultimodalModelConfig, MultimodalPretrainConfig
from ctg_ml.pretrain_preprocess import META_FILENAME

ENCODER_FILENAME = "encoder.pt"
METRICS_FILENAME = "pretrain_metrics.json"
SIGNAL_CHANNELS = 2  # FHR and toco are always the first two channels


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PretrainWindowDataset(Dataset):
    """Raw (unnormalized) windows from NPZ shards; NaN marks missing FHR/toco samples."""

    def __init__(self, shard_paths: list[Path], keep_baby_ids: set[str] | None = None) -> None:
        xs: list[np.ndarray] = []
        ids: list[np.ndarray] = []
        for path in shard_paths:
            data = np.load(path, allow_pickle=False)
            x = data["x"]
            baby_ids = data["baby_ids"].astype(str)
            if keep_baby_ids is not None:
                keep = np.isin(baby_ids, list(keep_baby_ids))
                x = x[keep]
                baby_ids = baby_ids[keep]
            xs.append(x)
            ids.append(baby_ids)
        if not xs:
            raise ValueError("No pretraining shards given")
        self.x = np.concatenate(xs, axis=0)
        self.baby_ids = np.concatenate(ids, axis=0)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.from_numpy(self.x[idx].astype(np.float32))


def load_shard_baby_ids(shard_paths: list[Path]) -> np.ndarray:
    ids = [np.load(p, allow_pickle=False)["baby_ids"].astype(str) for p in shard_paths]
    return np.unique(np.concatenate(ids)) if ids else np.array([], dtype=str)


def split_baby_ids(
    baby_ids: np.ndarray, val_fraction: float, seed: int
) -> tuple[set[str], set[str]]:
    """Deterministic train/val split of pretraining windows by BabyID."""
    ids = np.array(sorted(str(x) for x in baby_ids))
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n_val = int(round(len(ids) * val_fraction))
    if len(ids) > 1:
        n_val = min(max(n_val, 1), len(ids) - 1)
    else:
        n_val = 0
    return set(ids[n_val:].tolist()), set(ids[:n_val].tolist())


def prepare_batch(
    x_raw: torch.Tensor, means: torch.Tensor, stds: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize like the supervised pipeline.

    Returns ``(x_in, target, valid)``: ``x_in`` is the full channel stack with
    non-finite values set to 0, ``target`` the normalized FHR/toco and ``valid`` marks
    positions where the raw FHR/toco value was present (finite).
    """
    x = x_raw.clone()
    signal = x[:, :SIGNAL_CHANNELS, :]
    valid = torch.isfinite(signal)
    normalized = (signal - means.view(1, -1, 1)) / stds.view(1, -1, 1)
    target = torch.where(valid, normalized, torch.zeros_like(normalized))
    x[:, :SIGNAL_CHANNELS, :] = target
    rest = x[:, SIGNAL_CHANNELS:, :]
    x[:, SIGNAL_CHANNELS:, :] = torch.where(torch.isfinite(rest), rest, torch.zeros_like(rest))
    return x, target, valid


def make_span_mask(
    batch_size: int,
    n_steps: int,
    mask_ratio: float,
    span: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Boolean (batch, time) mask of random contiguous spans covering ~mask_ratio steps.

    The number of spans is chosen so that the *expected* coverage after overlaps equals
    ``mask_ratio``.
    """
    if mask_ratio <= 0.0:
        return torch.zeros(batch_size, n_steps, dtype=torch.bool)
    if mask_ratio >= 1.0:
        return torch.ones(batch_size, n_steps, dtype=torch.bool)
    span = max(1, min(int(span), n_steps))
    if span >= n_steps:
        return torch.ones(batch_size, n_steps, dtype=torch.bool)
    n_spans = max(1, int(round(math.log(1.0 - mask_ratio) / math.log(1.0 - span / n_steps))))
    starts = torch.randint(0, n_steps - span + 1, (batch_size, n_spans), generator=generator)
    positions = (starts.unsqueeze(-1) + torch.arange(span)).reshape(batch_size, -1)
    mask = torch.zeros(batch_size, n_steps, dtype=torch.bool)
    mask.scatter_(1, positions, True)
    return mask


def apply_mask(x_in: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Zero the FHR/toco channels at masked timesteps; other channels are untouched."""
    x = x_in.clone()
    x[:, :SIGNAL_CHANNELS, :] = x[:, :SIGNAL_CHANNELS, :].masked_fill(mask.unsqueeze(1), 0.0)
    return x


def masked_reconstruction_loss(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    """MSE over masked positions where the raw signal was present."""
    weight = (mask.unsqueeze(1) & valid).to(pred.dtype)
    err = F.mse_loss(pred.float(), target.float(), reduction="none")
    return (err * weight).sum() / weight.sum().clamp_min(1.0)


def encoder_input_channels(state_dict: dict[str, torch.Tensor]) -> int:
    key = "tcn.0.net.0.weight"
    if key not in state_dict:
        raise ValueError(f"Encoder state dict has no '{key}' entry; not a TCNEncoder checkpoint")
    return int(state_dict[key].shape[1])


def load_pretrained_encoder(
    model: MultimodalMultitaskTCN,
    path: str | Path,
    expected_channel_names: list[str] | None = None,
) -> dict[str, Any]:
    """Load ``sequence_encoder_state_dict`` from encoder.pt into ``model.sequence_encoder``.

    Raises ``ValueError`` with a clear message on channel-count or channel-name mismatch.
    """
    ckpt = torch.load(Path(path), map_location="cpu", weights_only=False)
    if "sequence_encoder_state_dict" not in ckpt:
        raise ValueError(f"{path} does not contain 'sequence_encoder_state_dict'")
    state_dict = ckpt["sequence_encoder_state_dict"]
    want = int(model.sequence_encoder.tcn[0].net[0].weight.shape[1])
    have = encoder_input_channels(state_dict)
    if have != want:
        raise ValueError(
            f"Pretrained encoder expects {have} input channels "
            f"({ckpt.get('channel_names')}) but the supervised model has {want} "
            f"({expected_channel_names}). Pretrain and supervised preprocessing must use the same "
            "[sequence] channel settings."
        )
    ckpt_names = ckpt.get("channel_names")
    if expected_channel_names is not None and ckpt_names is not None:
        if [str(x) for x in ckpt_names] != [str(x) for x in expected_channel_names]:
            raise ValueError(
                f"Pretrained encoder channel names {list(ckpt_names)} differ from the supervised "
                f"channels {list(expected_channel_names)}."
            )
    model.sequence_encoder.load_state_dict(state_dict, strict=True)
    return ckpt


def set_encoder_frozen(model: MultimodalMultitaskTCN, frozen: bool) -> None:
    for param in model.sequence_encoder.parameters():
        param.requires_grad_(not frozen)


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    requested = torch.device(device_arg)
    if requested.type == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU.")
        return torch.device("cpu")
    return requested


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    means: torch.Tensor,
    stds: torch.Tensor,
    pretrain_cfg: MultimodalPretrainConfig,
    use_amp: bool,
    generator: torch.Generator,
) -> float:
    model.eval()
    total = 0.0
    n = 0
    for x_raw in loader:
        x_raw = x_raw.to(device, non_blocking=(device.type == "cuda"))
        x_in, target, valid = prepare_batch(x_raw, means, stds)
        mask = make_span_mask(
            x_in.shape[0],
            x_in.shape[2],
            pretrain_cfg.mask_ratio,
            pretrain_cfg.mask_span_seconds,
            generator,
        ).to(device)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            pred = model(apply_mask(x_in, mask))
        loss = masked_reconstruction_loss(pred, target, mask, valid)
        total += float(loss) * x_raw.shape[0]
        n += x_raw.shape[0]
    return total / max(n, 1)


def run_pretraining(
    windows_dir: str | Path,
    out_dir: str | Path,
    pretrain_cfg: MultimodalPretrainConfig,
    model_cfg: MultimodalModelConfig,
    device: str = "auto",
    show_progress: bool = True,
) -> dict[str, Any]:
    """Train the masked-reconstruction model and save ``encoder.pt`` + a metrics JSON."""
    windows_dir = Path(windows_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = json.loads((windows_dir / META_FILENAME).read_text())
    shard_paths = [windows_dir / name for name in meta["shards"]]
    if not shard_paths:
        raise ValueError(f"No shards listed in {windows_dir / META_FILENAME}")
    channel_names = [str(x) for x in meta["channel_names"]]
    n_steps = int(meta["n_steps"])
    means_list = [float(x) for x in meta["normalization"]["means"]]
    stds_list = [float(x) for x in meta["normalization"]["stds"]]

    set_seed(pretrain_cfg.seed)
    dev = _resolve_device(device)
    use_amp = bool(pretrain_cfg.use_amp and dev.type == "cuda")

    train_ids, val_ids = split_baby_ids(
        load_shard_baby_ids(shard_paths), pretrain_cfg.val_fraction, pretrain_cfg.seed
    )
    train_ds = PretrainWindowDataset(shard_paths, train_ids)
    val_ds = PretrainWindowDataset(shard_paths, val_ids) if val_ids else None
    if len(train_ds) == 0:
        raise ValueError("Pretraining train split is empty")
    print(
        f"Pretraining windows: train={len(train_ds)} ({len(train_ids)} babies) "
        f"val={len(val_ds) if val_ds is not None else 0} ({len(val_ids)} babies) "
        f"channels={channel_names} n_steps={n_steps}"
    )
    print(
        f"Normalization (pretrain windows): FHR mean/std={means_list[0]:.3f}/{stds_list[0]:.3f}, "
        f"toco mean/std={means_list[1]:.3f}/{stds_list[1]:.3f}"
    )
    print(
        f"Masking: ratio={pretrain_cfg.mask_ratio} span={pretrain_cfg.mask_span_seconds}s "
        f"(zeroing FHR/toco only; channel layout identical to supervised model)"
    )
    print(f"Device: {dev} (amp={use_amp}) seed={pretrain_cfg.seed}")

    means = torch.tensor(means_list, dtype=torch.float32, device=dev)
    stds = torch.tensor(stds_list, dtype=torch.float32, device=dev)
    pin = dev.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=pretrain_cfg.batch_size, shuffle=True, num_workers=0, pin_memory=pin
    )
    val_loader = (
        DataLoader(
            val_ds, batch_size=pretrain_cfg.batch_size, shuffle=False, num_workers=0, pin_memory=pin
        )
        if val_ds is not None
        else None
    )

    model = MaskedReconstructionTCN(
        in_channels=len(channel_names),
        tcn_channels=model_cfg.tcn_channels,
        kernel_size=model_cfg.kernel_size,
        dropout=model_cfg.dropout,
        num_reconstructed_channels=SIGNAL_CHANNELS,
        decoder_hidden_dim=pretrain_cfg.decoder_hidden_dim,
        decoder_kernel_size=pretrain_cfg.decoder_kernel_size,
    ).to(dev)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=pretrain_cfg.lr, weight_decay=pretrain_cfg.weight_decay
    )
    scaler = torch.amp.GradScaler(device="cuda", enabled=use_amp)
    mask_gen = torch.Generator().manual_seed(pretrain_cfg.seed)
    val_gen_seed = pretrain_cfg.seed + 1

    encoder_path = out_dir / ENCODER_FILENAME
    metrics_path = out_dir / METRICS_FILENAME
    history: list[dict[str, float]] = []
    best_val = float("inf")
    best_epoch = 0
    since_improve = 0
    config_payload = {
        "pretrain": {
            k: str(v) if isinstance(v, Path) else v for k, v in asdict(pretrain_cfg).items()
        },
        "model": asdict(model_cfg),
    }

    for epoch in range(1, pretrain_cfg.epochs + 1):
        model.train()
        running = 0.0
        n = 0
        iterator = (
            tqdm(
                train_loader,
                desc=f"pretrain {epoch}/{pretrain_cfg.epochs}",
                leave=False,
                unit="batch",
            )
            if show_progress
            else train_loader
        )
        for x_raw in iterator:
            x_raw = x_raw.to(dev, non_blocking=pin)
            x_in, target, valid = prepare_batch(x_raw, means, stds)
            mask = make_span_mask(
                x_in.shape[0],
                x_in.shape[2],
                pretrain_cfg.mask_ratio,
                pretrain_cfg.mask_span_seconds,
                mask_gen,
            ).to(dev)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type, enabled=use_amp):
                pred = model(apply_mask(x_in, mask))
            loss = masked_reconstruction_loss(pred, target, mask, valid)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            running += float(loss.detach()) * x_raw.shape[0]
            n += x_raw.shape[0]
            if isinstance(iterator, tqdm):
                iterator.set_postfix(loss=f"{running / max(n, 1):.4f}")
        train_loss = running / max(n, 1)

        if val_loader is not None:
            val_loss = _evaluate(
                model,
                val_loader,
                dev,
                means,
                stds,
                pretrain_cfg,
                use_amp,
                torch.Generator().manual_seed(val_gen_seed),
            )
        else:
            val_loss = train_loss
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"epoch={epoch:03d} train_loss={train_loss:.5f} val_loss={val_loss:.5f}")

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            since_improve = 0
            torch.save(
                {
                    "sequence_encoder_state_dict": {
                        k: v.detach().cpu() for k, v in model.encoder.state_dict().items()
                    },
                    "normalization": {
                        "channels": ["FHR", "toco"],
                        "means": means_list,
                        "stds": stds_list,
                    },
                    "config": config_payload,
                    "channel_names": channel_names,
                    "n_steps": n_steps,
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                encoder_path,
            )
            print(f"Saved encoder: {encoder_path}")
        else:
            since_improve += 1
            if since_improve >= pretrain_cfg.early_stopping_patience:
                print(f"Early stopping at epoch {epoch} (best val_loss={best_val:.5f})")
                break

    metrics: dict[str, Any] = {
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "history": history,
        "n_train_windows": len(train_ds),
        "n_val_windows": len(val_ds) if val_ds is not None else 0,
        "n_train_babies": len(train_ids),
        "n_val_babies": len(val_ids),
        "channel_names": channel_names,
        "n_steps": n_steps,
        "normalization": {"means": means_list, "stds": stds_list},
        "encoder_path": str(encoder_path),
        "config": config_payload,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"Saved metrics: {metrics_path}")
    return metrics
