from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class CTG2PathsConfig:
    ctg_parquet: Path
    registry_csv: Path
    artifacts_dir: Path


@dataclass(frozen=True)
class CTG2SplitConfig:
    train_fraction: float
    val_fraction: float
    test_fraction: float
    random_seed: int


@dataclass(frozen=True)
class CTG2SequenceConfig:
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
class CTG2RegistryConfig:
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
class CTG2ModelConfig:
    tcn_channels: list[int]
    kernel_size: int
    dropout: float
    tabular_hidden_dim: int
    fusion_hidden_dim: int


@dataclass(frozen=True)
class CTG2TrainConfig:
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


@dataclass(frozen=True)
class CTG2ProjectConfig:
    paths: CTG2PathsConfig
    split: CTG2SplitConfig
    sequence: CTG2SequenceConfig
    registry: CTG2RegistryConfig
    model: CTG2ModelConfig
    train: CTG2TrainConfig


def load_ctg2_config(path: str | Path = "configs/ctg2_multimodal.toml") -> CTG2ProjectConfig:
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

    return CTG2ProjectConfig(
        paths=CTG2PathsConfig(
            ctg_parquet=Path(raw["paths"]["ctg_parquet"]),
            registry_csv=Path(raw["paths"]["registry_csv"]),
            artifacts_dir=Path(raw["paths"]["artifacts_dir"]),
        ),
        split=CTG2SplitConfig(
            train_fraction=float(split["train_fraction"]),
            val_fraction=float(split["val_fraction"]),
            test_fraction=float(split["test_fraction"]),
            random_seed=int(split["random_seed"]),
        ),
        sequence=CTG2SequenceConfig(
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
        registry=CTG2RegistryConfig(
            input_numeric=[str(x) for x in registry["input_numeric"]],
            input_boolean=[str(x) for x in registry["input_boolean"]],
            input_categorical=[str(x) for x in registry["input_categorical"]],
            input_excluded_due_to_leakage=[str(x) for x in registry["input_excluded_due_to_leakage"]],
            categorical_other_min_frequency=int(registry["categorical_other_min_frequency"]),
            country_top_k=int(registry["country_top_k"]),
            apgar_outputs=[str(x) for x in registry["apgar_outputs"]],
            categorical_outputs=[str(x) for x in registry.get("categorical_outputs", [])],
            continuous_outputs=[str(x) for x in registry["continuous_outputs"]],
            binary_outputs=[str(x) for x in registry["binary_outputs"]],
            binary_outputs_missing_as_false=[str(x) for x in registry.get("binary_outputs_missing_as_false", [])],
        ),
        model=CTG2ModelConfig(
            tcn_channels=[int(x) for x in model["tcn_channels"]],
            kernel_size=int(model["kernel_size"]),
            dropout=float(model["dropout"]),
            tabular_hidden_dim=int(model["tabular_hidden_dim"]),
            fusion_hidden_dim=int(model["fusion_hidden_dim"]),
        ),
        train=CTG2TrainConfig(
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
        ),
    )
