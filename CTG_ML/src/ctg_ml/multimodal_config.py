from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MultimodalPathsConfig:
    ctg_parquet: Path
    registry_csv: Path
    artifacts_dir: Path


@dataclass(frozen=True)
class MultimodalSplitConfig:
    train_fraction: float
    val_fraction: float
    test_fraction: float
    random_seed: int


@dataclass(frozen=True)
class MultimodalSequenceConfig:
    window_minutes: int
    sample_rate_hz: int
    pad_short: bool
    treat_fhr_zero_as_missing: bool
    treat_toco_zero_as_missing: bool
    include_padding_mask: bool
    include_signal_quality_channels: bool
    quality_levels: list[str]
    output_dir: Path
    chunk_vectors_per_batch: int


@dataclass(frozen=True)
class MultimodalRegistryConfig:
    input_numeric: list[str]
    input_boolean: list[str]
    input_categorical: list[str]
    input_excluded_due_to_leakage: list[str]
    categorical_other_min_frequency: int
    country_top_k: int
    apgar_outputs: list[str]
    categorical_outputs: list[str]
    continuous_outputs: list[str]
    binary_outputs: list[str]
    binary_outputs_missing_as_false: list[str]


@dataclass(frozen=True)
class MultimodalModelConfig:
    tcn_channels: list[int]
    kernel_size: int
    dropout: float
    tabular_hidden_dim: int
    fusion_hidden_dim: int


@dataclass(frozen=True)
class MultimodalTrainConfig:
    learning_rate: float
    batch_size: int
    epochs: int
    weight_decay: float
    apgar_class_weight_power: float
    regression_loss_weight: float
    binary_loss_weight: float
    gradient_clip_norm: float
    use_amp: bool
    seed: int
    deterministic: bool
    monitor_binary_tasks: list[str]
    early_stopping_enabled: bool
    early_stopping_min_epochs: int
    early_stopping_patience: int
    early_stopping_min_delta: float
    # Optional path to a self-supervised `encoder.pt` (see scripts/pretrain_tcn.py).
    # Empty string disables initialisation from a pretrained encoder.
    init_encoder: str = ""
    # Number of initial epochs during which the sequence encoder stays frozen.
    freeze_encoder_epochs: int = 0


@dataclass(frozen=True)
class MultimodalPretrainConfig:
    """Self-supervised masked-reconstruction pretraining of the TCN sequence encoder.

    Defaults keep configs without a `[pretrain]` section loadable.
    """

    pretrain_parquet: Path = Path("data/CTG3/ctg_pretrain.parquet")
    window_minutes: int = 60
    stride_minutes: float = 30.0
    min_signal_fraction: float = 0.5
    mask_ratio: float = 0.3
    mask_span_seconds: int = 30
    epochs: int = 30
    batch_size: int = 64
    lr: float = 0.001
    weight_decay: float = 0.0001
    exclude_final_window: bool = True
    seed: int = 51
    out_subdir: str = "pretrain"
    val_fraction: float = 0.1
    early_stopping_patience: int = 5
    windows_per_shard: int = 2048
    chunk_vectors_per_batch: int = 64
    decoder_hidden_dim: int = 64
    decoder_kernel_size: int = 9
    use_amp: bool = True


@dataclass(frozen=True)
class MultimodalProjectConfig:
    paths: MultimodalPathsConfig
    split: MultimodalSplitConfig
    sequence: MultimodalSequenceConfig
    registry: MultimodalRegistryConfig
    model: MultimodalModelConfig
    train: MultimodalTrainConfig
    pretrain: MultimodalPretrainConfig = MultimodalPretrainConfig()


def _load_pretrain_config(raw: dict[str, Any] | None) -> MultimodalPretrainConfig:
    d = MultimodalPretrainConfig()
    if not raw:
        return d
    return MultimodalPretrainConfig(
        pretrain_parquet=Path(str(raw.get("pretrain_parquet", d.pretrain_parquet))),
        window_minutes=int(raw.get("window_minutes", d.window_minutes)),
        stride_minutes=float(raw.get("stride_minutes", d.stride_minutes)),
        min_signal_fraction=float(raw.get("min_signal_fraction", d.min_signal_fraction)),
        mask_ratio=float(raw.get("mask_ratio", d.mask_ratio)),
        mask_span_seconds=int(raw.get("mask_span_seconds", d.mask_span_seconds)),
        epochs=int(raw.get("epochs", d.epochs)),
        batch_size=int(raw.get("batch_size", d.batch_size)),
        lr=float(raw.get("lr", d.lr)),
        weight_decay=float(raw.get("weight_decay", d.weight_decay)),
        exclude_final_window=bool(raw.get("exclude_final_window", d.exclude_final_window)),
        seed=int(raw.get("seed", d.seed)),
        out_subdir=str(raw.get("out_subdir", d.out_subdir)),
        val_fraction=float(raw.get("val_fraction", d.val_fraction)),
        early_stopping_patience=int(raw.get("early_stopping_patience", d.early_stopping_patience)),
        windows_per_shard=int(raw.get("windows_per_shard", d.windows_per_shard)),
        chunk_vectors_per_batch=int(raw.get("chunk_vectors_per_batch", d.chunk_vectors_per_batch)),
        decoder_hidden_dim=int(raw.get("decoder_hidden_dim", d.decoder_hidden_dim)),
        decoder_kernel_size=int(raw.get("decoder_kernel_size", d.decoder_kernel_size)),
        use_amp=bool(raw.get("use_amp", d.use_amp)),
    )


def load_multimodal_config(
    path: str | Path = "configs/ctg3_multimodal.toml",
) -> MultimodalProjectConfig:
    cfg_path = Path(path)
    with cfg_path.open("rb") as f:
        raw = tomllib.load(f)

    split = raw["split"]
    total = split["train_fraction"] + split["val_fraction"] + split["test_fraction"]
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Split fractions must sum to 1.0, got {total}")

    registry = raw["registry"]
    sequence = raw["sequence"]
    model = raw["model"]
    train = raw["train"]

    return MultimodalProjectConfig(
        paths=MultimodalPathsConfig(
            ctg_parquet=Path(raw["paths"]["ctg_parquet"]),
            registry_csv=Path(raw["paths"]["registry_csv"]),
            artifacts_dir=Path(raw["paths"]["artifacts_dir"]),
        ),
        split=MultimodalSplitConfig(
            train_fraction=float(split["train_fraction"]),
            val_fraction=float(split["val_fraction"]),
            test_fraction=float(split["test_fraction"]),
            random_seed=int(split["random_seed"]),
        ),
        sequence=MultimodalSequenceConfig(
            window_minutes=int(sequence["window_minutes"]),
            sample_rate_hz=int(sequence["sample_rate_hz"]),
            pad_short=bool(sequence["pad_short"]),
            treat_fhr_zero_as_missing=bool(sequence["treat_fhr_zero_as_missing"]),
            treat_toco_zero_as_missing=bool(sequence["treat_toco_zero_as_missing"]),
            include_padding_mask=bool(sequence["include_padding_mask"]),
            include_signal_quality_channels=bool(sequence["include_signal_quality_channels"]),
            quality_levels=[str(x) for x in sequence["quality_levels"]],
            output_dir=Path(sequence["output_dir"]),
            chunk_vectors_per_batch=int(sequence["chunk_vectors_per_batch"]),
        ),
        registry=MultimodalRegistryConfig(
            input_numeric=[str(x) for x in registry["input_numeric"]],
            input_boolean=[str(x) for x in registry["input_boolean"]],
            input_categorical=[str(x) for x in registry["input_categorical"]],
            input_excluded_due_to_leakage=[
                str(x) for x in registry["input_excluded_due_to_leakage"]
            ],
            categorical_other_min_frequency=int(registry["categorical_other_min_frequency"]),
            country_top_k=int(registry["country_top_k"]),
            apgar_outputs=[str(x) for x in registry["apgar_outputs"]],
            categorical_outputs=[str(x) for x in registry.get("categorical_outputs", [])],
            continuous_outputs=[str(x) for x in registry["continuous_outputs"]],
            binary_outputs=[str(x) for x in registry["binary_outputs"]],
            binary_outputs_missing_as_false=[
                str(x) for x in registry.get("binary_outputs_missing_as_false", [])
            ],
        ),
        model=MultimodalModelConfig(
            tcn_channels=[int(x) for x in model["tcn_channels"]],
            kernel_size=int(model["kernel_size"]),
            dropout=float(model["dropout"]),
            tabular_hidden_dim=int(model["tabular_hidden_dim"]),
            fusion_hidden_dim=int(model["fusion_hidden_dim"]),
        ),
        train=MultimodalTrainConfig(
            learning_rate=float(train["learning_rate"]),
            batch_size=int(train["batch_size"]),
            epochs=int(train["epochs"]),
            weight_decay=float(train["weight_decay"]),
            apgar_class_weight_power=float(train.get("apgar_class_weight_power", 0.0)),
            regression_loss_weight=float(train["regression_loss_weight"]),
            binary_loss_weight=float(train["binary_loss_weight"]),
            gradient_clip_norm=float(train["gradient_clip_norm"]),
            use_amp=bool(train["use_amp"]),
            seed=int(train["seed"]),
            deterministic=bool(train["deterministic"]),
            monitor_binary_tasks=[str(x) for x in train.get("monitor_binary_tasks", [])],
            early_stopping_enabled=bool(train["early_stopping_enabled"]),
            early_stopping_min_epochs=int(train["early_stopping_min_epochs"]),
            early_stopping_patience=int(train["early_stopping_patience"]),
            early_stopping_min_delta=float(train["early_stopping_min_delta"]),
            init_encoder=str(train.get("init_encoder", "")),
            freeze_encoder_epochs=int(train.get("freeze_encoder_epochs", 0)),
        ),
        pretrain=_load_pretrain_config(raw.get("pretrain")),
    )
