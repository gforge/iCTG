from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from tqdm import tqdm

from ctg_ml.ctg2_config import CTG2RegistryConfig, CTG2SequenceConfig
from ctg_ml.ctg2_registry import (
    MultitaskTargetSpec,
    build_targets,
    fit_tabular_encoder,
    load_registry_for_multimodal,
    merge_splits_with_registry,
    normalize_tabular_inplace,
    transform_tabular_inputs,
)


@dataclass(frozen=True)
class CTG2SplitBuildStats:
    split_name: str
    total_babies: int
    kept_babies: int
    dropped_short_babies: int
    min_rows: int
    median_rows: float
    max_rows: int
    seq_channels: int
    n_steps: int
    tab_features: int
    output_path: Path


SEQUENCE_SQL = """
SELECT
    c.BabyID,
    c."Timestamp" AS ts,
    CAST(c.FHR AS DOUBLE) AS fhr,
    CAST(c.toco AS DOUBLE) AS toco,
    CAST(c.Hr1_SignalQuality AS VARCHAR) AS hr1_signal_quality
FROM read_parquet(?) c
INNER JOIN split_map s ON c.BabyID = s.BabyID
ORDER BY c.BabyID, ts
"""


def sequence_channel_names(cfg: CTG2SequenceConfig) -> list[str]:
    names = ["FHR", "toco"]
    if cfg.include_signal_quality_channels:
        names.extend([f"Hr1_SignalQuality=={level}" for level in cfg.quality_levels])
    if cfg.include_padding_mask:
        names.append("padding_mask")
    return names


def _finalize_sequence(
    group: pd.DataFrame,
    cfg: CTG2SequenceConfig,
) -> tuple[np.ndarray | None, int]:
    fhr = group["fhr"].to_numpy(dtype=np.float32, copy=True)
    toco = group["toco"].to_numpy(dtype=np.float32, copy=True)
    quality = group["hr1_signal_quality"].fillna("").astype(str).to_numpy()
    raw_len = int(len(group))

    if cfg.treat_fhr_zero_as_missing:
        fhr[fhr == 0.0] = np.nan
    fhr[~np.isfinite(fhr)] = np.nan

    if cfg.treat_toco_zero_as_missing:
        toco[toco == 0.0] = np.nan
    toco[~np.isfinite(toco)] = np.nan

    n_steps = int(cfg.window_minutes * 60 * cfg.sample_rate_hz)
    if raw_len < n_steps and not cfg.pad_short:
        return None, raw_len

    channel_names = sequence_channel_names(cfg)
    channel_index = {name: idx for idx, name in enumerate(channel_names)}
    seq = np.zeros((len(channel_names), n_steps), dtype=np.float32)
    seq[channel_index["FHR"], :] = np.nan
    seq[channel_index["toco"], :] = np.nan

    if raw_len >= n_steps:
        fhr_tail = fhr[-n_steps:]
        toco_tail = toco[-n_steps:]
        quality_tail = quality[-n_steps:]
        start = 0
    else:
        pad = n_steps - raw_len
        fhr_tail = fhr
        toco_tail = toco
        quality_tail = quality
        start = pad
        if cfg.include_padding_mask:
            seq[channel_index["padding_mask"], :pad] = 1.0

    seq[channel_index["FHR"], start:] = fhr_tail
    seq[channel_index["toco"], start:] = toco_tail
    if cfg.include_signal_quality_channels:
        for level in cfg.quality_levels:
            seq[channel_index[f"Hr1_SignalQuality=={level}"], start:] = (quality_tail == level).astype(np.float32)
    return seq, raw_len


def _build_split_npz(
    ctg_parquet: Path,
    split_df: pd.DataFrame,
    seq_cfg: CTG2SequenceConfig,
    tab_X: np.ndarray,
    apgar_targets: np.ndarray,
    apgar_mask: np.ndarray,
    reg_targets: np.ndarray,
    reg_mask: np.ndarray,
    bin_targets: np.ndarray,
    bin_mask: np.ndarray,
    feature_names: list[str],
    apgar_names: list[str],
    regression_names: list[str],
    binary_names: list[str],
    out_path: Path,
) -> CTG2SplitBuildStats:
    split_df = split_df.sort_values("BabyID").reset_index(drop=True)
    row_index = {baby_id: idx for idx, baby_id in enumerate(split_df["BabyID"].astype(str))}
    total_babies = len(split_df)
    channel_names = sequence_channel_names(seq_cfg)
    n_steps = int(seq_cfg.window_minutes * 60 * seq_cfg.sample_rate_hz)

    X_seq = np.full((total_babies, len(channel_names), n_steps), np.nan, dtype=np.float32)
    X_tab = np.zeros((total_babies, tab_X.shape[1]), dtype=np.float32)
    y_apgar = np.zeros((total_babies, apgar_targets.shape[1]), dtype=np.int64)
    y_apgar_mask = np.zeros((total_babies, apgar_mask.shape[1]), dtype=np.float32)
    y_reg = np.zeros((total_babies, reg_targets.shape[1]), dtype=np.float32)
    y_reg_mask = np.zeros((total_babies, reg_mask.shape[1]), dtype=np.float32)
    y_bin = np.zeros((total_babies, bin_targets.shape[1]), dtype=np.float32)
    y_bin_mask = np.zeros((total_babies, bin_mask.shape[1]), dtype=np.float32)
    baby_ids = np.empty((total_babies,), dtype=f"<U{int(split_df['BabyID'].str.len().max())}")

    con = duckdb.connect(database=":memory:")
    kept = 0
    dropped_short = 0
    row_counts: list[int] = []
    pbar = tqdm(total=total_babies, desc=f"{split_df['split'].iloc[0]} preprocess", unit="baby")

    def store_one(baby_id: str, group: pd.DataFrame) -> None:
        nonlocal kept, dropped_short
        seq, raw_len = _finalize_sequence(group, seq_cfg)
        row_counts.append(raw_len)
        pbar.update(1)
        if seq is None:
            dropped_short += 1
            return
        src = row_index[baby_id]
        X_seq[kept] = seq
        X_tab[kept] = tab_X[src]
        y_apgar[kept] = apgar_targets[src]
        y_apgar_mask[kept] = apgar_mask[src]
        y_reg[kept] = reg_targets[src]
        y_reg_mask[kept] = reg_mask[src]
        y_bin[kept] = bin_targets[src]
        y_bin_mask[kept] = bin_mask[src]
        baby_ids[kept] = baby_id
        kept += 1

    try:
        con.register("split_map", split_df[["BabyID"]])
        res = con.execute(SEQUENCE_SQL, [str(ctg_parquet)])
        carry: pd.DataFrame | None = None
        while True:
            chunk = res.fetch_df_chunk(vectors_per_chunk=seq_cfg.chunk_vectors_per_batch)
            if chunk is None or chunk.empty:
                break
            if carry is not None and not carry.empty:
                chunk = pd.concat([carry, chunk], ignore_index=True)
                carry = None

            last_baby = str(chunk["BabyID"].iloc[-1])
            is_last = chunk["BabyID"].astype(str) == last_baby
            carry = chunk.loc[is_last].copy()
            full_chunk = chunk.loc[~is_last]
            if full_chunk.empty:
                continue
            for baby_id, group in full_chunk.groupby("BabyID", sort=False):
                store_one(str(baby_id), group)

        if carry is not None and not carry.empty:
            store_one(str(carry["BabyID"].iloc[0]), carry)
    finally:
        pbar.close()
        con.close()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        X_seq=X_seq[:kept],
        X_tab=X_tab[:kept],
        y_apgar=y_apgar[:kept],
        y_apgar_mask=y_apgar_mask[:kept],
        y_reg=y_reg[:kept],
        y_reg_mask=y_reg_mask[:kept],
        y_bin=y_bin[:kept],
        y_bin_mask=y_bin_mask[:kept],
        baby_ids=baby_ids[:kept],
        sequence_channels=np.array(channel_names),
        tabular_feature_names=np.array(feature_names),
        apgar_target_names=np.array(apgar_names),
        regression_target_names=np.array(regression_names),
        binary_target_names=np.array(binary_names),
        n_steps=np.array([n_steps], dtype=np.int32),
    )

    row_arr = np.asarray(row_counts, dtype=np.int32) if row_counts else np.array([0], dtype=np.int32)
    return CTG2SplitBuildStats(
        split_name=str(split_df["split"].iloc[0]),
        total_babies=total_babies,
        kept_babies=int(kept),
        dropped_short_babies=int(dropped_short),
        min_rows=int(row_arr.min()),
        median_rows=float(np.median(row_arr)),
        max_rows=int(row_arr.max()),
        seq_channels=len(channel_names),
        n_steps=n_steps,
        tab_features=int(X_tab.shape[1]),
        output_path=out_path,
    )


def build_ctg2_multimodal_npz_files(
    ctg_parquet: str | Path,
    registry_csv: str | Path,
    splits_csv: str | Path,
    output_dir: str | Path,
    seq_cfg: CTG2SequenceConfig,
    registry_cfg: CTG2RegistryConfig,
) -> list[CTG2SplitBuildStats]:
    if registry_cfg.input_excluded_due_to_leakage:
        print(
            "The following registry inputs are excluded by default because they are unavailable "
            "at real prediction time or contain direct post-birth information: "
            + ", ".join(registry_cfg.input_excluded_due_to_leakage)
            + "."
        )
    splits_df = pd.read_csv(splits_csv, usecols=["BabyID", "split"])
    registry_df = load_registry_for_multimodal(str(registry_csv), registry_cfg)
    merged = merge_splits_with_registry(splits_df, registry_df)

    required = {"BabyID", "split"}
    if missing := (required - set(merged.columns)):
        raise ValueError(f"Merged split/registry table missing columns: {sorted(missing)}")

    target_spec = MultitaskTargetSpec(
        apgar_names=list(registry_cfg.apgar_outputs),
        continuous_names=list(registry_cfg.continuous_outputs),
        binary_names=list(registry_cfg.binary_outputs),
    )

    train_df = merged[merged["split"] == "train"].copy()
    encoder = fit_tabular_encoder(train_df, registry_cfg)

    split_arrays: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]] = {}
    for split_name in ["train", "val", "test"]:
        split_part = merged[merged["split"] == split_name].copy().sort_values("BabyID").reset_index(drop=True)
        X_tab = transform_tabular_inputs(split_part, encoder)
        y_apgar, y_apgar_mask, y_reg, y_reg_mask, y_bin, y_bin_mask = build_targets(split_part, target_spec)
        split_arrays[split_name] = (X_tab, y_apgar, y_apgar_mask, y_reg, y_reg_mask, y_bin, y_bin_mask, split_part)

    train_X_tab = split_arrays["train"][0]
    other_tabs = [split_arrays["val"][0], split_arrays["test"][0]]
    normalize_tabular_inplace(train_X_tab, other_tabs, encoder.feature_names)

    out_dir = Path(output_dir)
    stats: list[CTG2SplitBuildStats] = []
    for split_name in ["train", "val", "test"]:
        X_tab, y_apgar, y_apgar_mask, y_reg, y_reg_mask, y_bin, y_bin_mask, split_part = split_arrays[split_name]
        out_path = out_dir / f"{split_name}.npz"
        stats.append(
            _build_split_npz(
                ctg_parquet=Path(ctg_parquet),
                split_df=split_part,
                seq_cfg=seq_cfg,
                tab_X=X_tab,
                apgar_targets=y_apgar,
                apgar_mask=y_apgar_mask,
                reg_targets=y_reg,
                reg_mask=y_reg_mask,
                bin_targets=y_bin,
                bin_mask=y_bin_mask,
                feature_names=encoder.feature_names,
                apgar_names=target_spec.apgar_names,
                regression_names=target_spec.continuous_names,
                binary_names=target_spec.binary_names,
                out_path=out_path,
            )
        )
    return stats
