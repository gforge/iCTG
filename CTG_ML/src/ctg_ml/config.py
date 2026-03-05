from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class PathsConfig:
    ctg_parquet: Path
    registry_csv: Path
    artifacts_dir: Path


@dataclass(frozen=True)
class TargetConfig:
    at_risk_max_apgar: int


@dataclass(frozen=True)
class SplitConfig:
    train_fraction: float
    val_fraction: float
    test_fraction: float
    random_seed: int


@dataclass(frozen=True)
class BaselineConfig:
    logreg_c: float
    max_iter: int


@dataclass(frozen=True)
class SequenceConfig:
    window_minutes: int
    sample_rate_hz: int
    pad_short: bool
    treat_fhr_zero_as_missing: bool
    include_fhr_missing_mask: bool
    treat_toco_zero_as_missing: bool
    include_toco_missing_mask: bool
    include_padding_mask: bool
    output_dir: Path
    chunk_vectors_per_batch: int


@dataclass(frozen=True)
class TCNConfig:
    learning_rate: float
    batch_size: int
    epochs: int
    weight_decay: float
    dropout: float
    channels: list[int]
    kernel_size: int
    early_stopping_enabled: bool
    early_stopping_min_epochs: int
    early_stopping_patience: int
    early_stopping_min_delta: float
    gradient_clip_norm: float
    use_weighted_sampler: bool
    use_balanced_batch_sampler: bool
    balanced_min_positives_per_batch: int
    disable_pos_weight_with_balanced_sampler: bool
    seed: int
    deterministic: bool
    use_amp: bool


@dataclass(frozen=True)
class ProjectConfig:
    paths: PathsConfig
    target: TargetConfig
    split: SplitConfig
    baseline: BaselineConfig
    sequence: SequenceConfig
    tcn: TCNConfig


def load_config(path: str | Path = "configs/default.toml") -> ProjectConfig:
    cfg_path = Path(path)
    with cfg_path.open("rb") as f:
        raw = tomllib.load(f)

    split = raw["split"]
    total = split["train_fraction"] + split["val_fraction"] + split["test_fraction"]
    if abs(total - 1.0) > 1e-9:
        msg = f"Split fractions must sum to 1.0, got {total}"
        raise ValueError(msg)

    return ProjectConfig(
        paths=PathsConfig(
            ctg_parquet=Path(raw["paths"]["ctg_parquet"]),
            registry_csv=Path(raw["paths"]["registry_csv"]),
            artifacts_dir=Path(raw["paths"]["artifacts_dir"]),
        ),
        target=TargetConfig(at_risk_max_apgar=int(raw["target"]["at_risk_max_apgar"])),
        split=SplitConfig(
            train_fraction=float(split["train_fraction"]),
            val_fraction=float(split["val_fraction"]),
            test_fraction=float(split["test_fraction"]),
            random_seed=int(split["random_seed"]),
        ),
        baseline=BaselineConfig(
            logreg_c=float(raw["baseline"]["logreg_c"]),
            max_iter=int(raw["baseline"]["max_iter"]),
        ),
        sequence=SequenceConfig(
            window_minutes=int(raw["sequence"]["window_minutes"]),
            sample_rate_hz=int(raw["sequence"]["sample_rate_hz"]),
            pad_short=bool(raw["sequence"]["pad_short"]),
            treat_fhr_zero_as_missing=bool(raw["sequence"]["treat_fhr_zero_as_missing"]),
            include_fhr_missing_mask=bool(raw["sequence"]["include_fhr_missing_mask"]),
            treat_toco_zero_as_missing=bool(raw["sequence"].get("treat_toco_zero_as_missing", True)),
            include_toco_missing_mask=bool(raw["sequence"].get("include_toco_missing_mask", True)),
            include_padding_mask=bool(raw["sequence"].get("include_padding_mask", True)),
            output_dir=Path(raw["sequence"]["output_dir"]),
            chunk_vectors_per_batch=int(raw["sequence"]["chunk_vectors_per_batch"]),
        ),
        tcn=TCNConfig(
            learning_rate=float(raw["tcn"]["learning_rate"]),
            batch_size=int(raw["tcn"]["batch_size"]),
            epochs=int(raw["tcn"]["epochs"]),
            weight_decay=float(raw["tcn"]["weight_decay"]),
            dropout=float(raw["tcn"]["dropout"]),
            channels=[int(x) for x in raw["tcn"]["channels"]],
            kernel_size=int(raw["tcn"]["kernel_size"]),
            early_stopping_enabled=bool(raw["tcn"]["early_stopping_enabled"]),
            early_stopping_min_epochs=int(raw["tcn"]["early_stopping_min_epochs"]),
            early_stopping_patience=int(raw["tcn"]["early_stopping_patience"]),
            early_stopping_min_delta=float(raw["tcn"]["early_stopping_min_delta"]),
            gradient_clip_norm=float(raw["tcn"].get("gradient_clip_norm", 0.0)),
            use_weighted_sampler=bool(raw["tcn"].get("use_weighted_sampler", False)),
            use_balanced_batch_sampler=bool(raw["tcn"].get("use_balanced_batch_sampler", False)),
            balanced_min_positives_per_batch=int(raw["tcn"].get("balanced_min_positives_per_batch", 1)),
            disable_pos_weight_with_balanced_sampler=bool(
                raw["tcn"].get("disable_pos_weight_with_balanced_sampler", True)
            ),
            seed=int(raw["tcn"].get("seed", raw["split"]["random_seed"])),
            deterministic=bool(raw["tcn"].get("deterministic", False)),
            use_amp=bool(raw["tcn"].get("use_amp", True)),
        ),
    )
