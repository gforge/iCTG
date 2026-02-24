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
        ),
    )
