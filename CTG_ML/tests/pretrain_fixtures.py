"""Synthetic pretraining data shared by the pretraining tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ctg_ml.multimodal_config import MultimodalPretrainConfig, MultimodalSequenceConfig

T0 = pd.Timestamp("2024-01-01 10:00:00")


def make_seq_cfg() -> MultimodalSequenceConfig:
    return MultimodalSequenceConfig(
        window_minutes=60,  # supervised window; pretraining overrides it with its own
        sample_rate_hz=1,
        pad_short=True,
        treat_fhr_zero_as_missing=True,
        treat_toco_zero_as_missing=True,
        include_padding_mask=True,
        include_signal_quality_channels=True,
        quality_levels=["Y", "R"],
        output_dir=Path("unused"),
        chunk_vectors_per_batch=64,
    )


def make_pretrain_cfg(**overrides: object) -> MultimodalPretrainConfig:
    base: dict[str, object] = {
        "window_minutes": 1,
        "stride_minutes": 0.5,
        "min_signal_fraction": 0.5,
        "windows_per_shard": 4,
        "chunk_vectors_per_batch": 1,
    }
    base.update(overrides)
    return MultimodalPretrainConfig(**base)  # type: ignore[arg-type]


def _session_rows(
    baby_id: str,
    session_id: int,
    seconds: np.ndarray,
    start: pd.Timestamp = T0,
    final_from: int | None = None,
) -> pd.DataFrame:
    n = len(seconds)
    quality = np.where(seconds % 3 == 0, "Y", np.where(seconds % 3 == 1, "R", "G"))
    return pd.DataFrame(
        {
            "BabyID": baby_id,
            "session_id": np.int32(session_id),
            "Timestamp": start + pd.to_timedelta(seconds, unit="s"),
            "FHR": (130.0 + (seconds % 7)).astype(np.float32),
            "toco": (20.0 + (seconds % 5)).astype(np.float32),
            "Hr1_SignalQuality": quality,
            "Hr1Mode": "US",
            "TocoMode": "TOCO",
            "in_final_window": np.zeros(n, dtype=bool)
            if final_from is None
            else seconds >= final_from,
        }
    )


def synthetic_frames() -> dict[str, pd.DataFrame]:
    """Three pregnancies: A (train split), B (val split), C (unlabeled)."""
    a1 = _session_rows("A", 1, np.arange(150))  # 4 full windows
    a2 = _session_rows("A", 2, np.arange(90), start=T0 + pd.Timedelta(hours=5), final_from=70)
    b1 = _session_rows("B", 1, np.arange(150))
    # C has a 70 s gap (seconds 50..119 missing) that must be treated as missing FHR.
    c_seconds = np.concatenate([np.arange(0, 50), np.arange(120, 200)])
    c1 = _session_rows("C", 1, c_seconds)
    return {"A": pd.concat([a1, a2]), "B": b1, "C": c1}


def write_parquet(tmp_path: Path, as_directory: bool) -> Path:
    frames = synthetic_frames()
    if as_directory:
        out_dir = tmp_path / "pretrain_buckets"
        out_dir.mkdir()
        pd.concat([frames["A"], frames["B"]]).to_parquet(
            out_dir / "all_sessions_bucket_0000.parquet", index=False
        )
        frames["C"].to_parquet(out_dir / "all_sessions_bucket_0001.parquet", index=False)
        return out_dir
    path = tmp_path / "ctg_pretrain.parquet"
    pd.concat(frames.values()).to_parquet(path, index=False)
    return path


def write_splits(tmp_path: Path) -> Path:
    path = tmp_path / "splits.csv"
    pd.DataFrame({"BabyID": ["A", "B", "X"], "split": ["train", "val", "test"]}).to_csv(
        path, index=False
    )
    return path
