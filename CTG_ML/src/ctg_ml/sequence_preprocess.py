from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from tqdm import tqdm


@dataclass(frozen=True)
class SequenceBuildConfig:
    window_minutes: int
    sample_rate_hz: int
    pad_short: bool
    treat_fhr_zero_as_missing: bool
    include_fhr_missing_mask: bool
    chunk_vectors_per_batch: int = 64

    @property
    def n_steps(self) -> int:
        return int(self.window_minutes * 60 * self.sample_rate_hz)


@dataclass(frozen=True)
class SplitBuildStats:
    split_name: str
    total_babies: int
    kept_babies: int
    dropped_short_babies: int
    min_rows: int
    median_rows: float
    max_rows: int
    channels: int
    n_steps: int
    output_path: Path


SEQUENCE_SQL = """
SELECT
    c.BabyID,
    c."Timestamp" AS ts,
    CAST(c.FHR AS DOUBLE) AS fhr,
    CAST(c.toco AS DOUBLE) AS toco
FROM read_parquet(?) c
INNER JOIN split_map s ON c.BabyID = s.BabyID
ORDER BY c.BabyID, ts
"""


def _finalize_one_sequence(
    baby_id: str,
    group: pd.DataFrame,
    label_by_baby: dict[str, int],
    apgar_by_baby: dict[str, int],
    cfg: SequenceBuildConfig,
) -> tuple[np.ndarray | None, float | None, int]:
    fhr = group["fhr"].to_numpy(dtype=np.float32, copy=True)
    toco = group["toco"].to_numpy(dtype=np.float32, copy=True)
    raw_len = int(len(group))

    fhr_missing = ~np.isfinite(fhr)
    if cfg.treat_fhr_zero_as_missing:
        fhr_missing |= (fhr == 0.0)

    # Keep non-finite toco as missing numerically (NaN), but no explicit mask channel by default.
    toco_missing = ~np.isfinite(toco)

    fhr = fhr.astype(np.float32, copy=False)
    toco = toco.astype(np.float32, copy=False)
    fhr[fhr_missing] = np.nan
    toco[toco_missing] = np.nan

    n_steps = cfg.n_steps
    if raw_len < n_steps and not cfg.pad_short:
        return None, None, raw_len

    channels = 2 + int(cfg.include_fhr_missing_mask)
    seq = np.full((channels, n_steps), np.nan, dtype=np.float32)

    if raw_len >= n_steps:
        fhr_tail = fhr[-n_steps:]
        toco_tail = toco[-n_steps:]
        fhr_missing_tail = fhr_missing[-n_steps:].astype(np.float32)
        start = 0
    else:
        # Left-pad shorter sequences so the end of the sequence stays aligned to birth.
        pad = n_steps - raw_len
        fhr_tail = fhr
        toco_tail = toco
        fhr_missing_tail = fhr_missing.astype(np.float32)
        start = pad
        if cfg.include_fhr_missing_mask:
            seq[2, :pad] = 1.0

    seq[0, start:] = fhr_tail
    seq[1, start:] = toco_tail
    if cfg.include_fhr_missing_mask:
        seq[2, start:] = fhr_missing_tail

    y = float(label_by_baby[baby_id])
    _ = apgar_by_baby  # reserved for future multi-class/regression outputs
    return seq, y, raw_len


def _process_split_chunked(
    ctg_parquet: Path,
    split_df: pd.DataFrame,
    cfg: SequenceBuildConfig,
    out_path: Path,
) -> SplitBuildStats:
    split_df = split_df.sort_values("BabyID").reset_index(drop=True)
    total_babies = len(split_df)
    label_by_baby = dict(zip(split_df["BabyID"], split_df["target"].astype(int), strict=True))
    apgar_by_baby = dict(zip(split_df["BabyID"], split_df["apgar5"].astype(int), strict=True))

    channels = 2 + int(cfg.include_fhr_missing_mask)
    X = np.full((total_babies, channels, cfg.n_steps), np.nan, dtype=np.float32)
    y = np.zeros((total_babies,), dtype=np.float32)
    baby_ids = np.empty((total_babies,), dtype=f"<U{int(split_df['BabyID'].str.len().max())}")

    con = duckdb.connect(database=":memory:")
    kept = 0
    dropped_short = 0
    row_counts: list[int] = []
    pbar = tqdm(total=total_babies, desc=f"{split_df['split'].iloc[0]} preprocess", unit="baby")

    try:
        con.register("split_map", split_df[["BabyID"]])
        res = con.execute(SEQUENCE_SQL, [str(ctg_parquet)])
        carry: pd.DataFrame | None = None

        while True:
            chunk = res.fetch_df_chunk(vectors_per_chunk=cfg.chunk_vectors_per_batch)
            if chunk is None or chunk.empty:
                break
            chunk = chunk.rename(columns={"FHR": "fhr", "toco": "toco", "Timestamp": "ts"})
            # The SQL already aliases lowercase columns; this makes the function robust to driver behavior.
            if "fhr" not in chunk.columns and "FHR" in chunk.columns:
                chunk = chunk.rename(columns={"FHR": "fhr"})
            if "toco" not in chunk.columns and "TOCO" in chunk.columns:
                chunk = chunk.rename(columns={"TOCO": "toco"})

            if carry is not None and not carry.empty:
                chunk = pd.concat([carry, chunk], ignore_index=True)
                carry = None

            last_baby = chunk["BabyID"].iloc[-1]
            is_last = chunk["BabyID"] == last_baby
            carry = chunk.loc[is_last].copy()
            full_chunk = chunk.loc[~is_last]

            if full_chunk.empty:
                continue

            for baby_id, group in full_chunk.groupby("BabyID", sort=False):
                baby_id = str(baby_id)
                seq, label, raw_len = _finalize_one_sequence(baby_id, group, label_by_baby, apgar_by_baby, cfg)
                row_counts.append(raw_len)
                if seq is None:
                    dropped_short += 1
                    pbar.update(1)
                    continue
                X[kept] = seq
                y[kept] = label
                baby_ids[kept] = baby_id
                kept += 1
                pbar.update(1)

        if carry is not None and not carry.empty:
            baby_id = str(carry["BabyID"].iloc[0])
            seq, label, raw_len = _finalize_one_sequence(baby_id, carry, label_by_baby, apgar_by_baby, cfg)
            row_counts.append(raw_len)
            if seq is None:
                dropped_short += 1
            else:
                X[kept] = seq
                y[kept] = label
                baby_ids[kept] = baby_id
                kept += 1
            pbar.update(1)
    finally:
        pbar.close()
        con.close()

    X = X[:kept]
    y = y[:kept]
    baby_ids = baby_ids[:kept]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        X=X,
        y=y.astype(np.float32),
        baby_ids=baby_ids,
        n_steps=np.array([cfg.n_steps], dtype=np.int32),
        channels=np.array(["FHR", "toco"] + (["fhr_missing_mask"] if cfg.include_fhr_missing_mask else [])),
    )

    row_arr = np.asarray(row_counts, dtype=np.int32) if row_counts else np.array([0], dtype=np.int32)
    return SplitBuildStats(
        split_name=str(split_df["split"].iloc[0]),
        total_babies=total_babies,
        kept_babies=int(kept),
        dropped_short_babies=int(dropped_short),
        min_rows=int(row_arr.min()),
        median_rows=float(np.median(row_arr)),
        max_rows=int(row_arr.max()),
        channels=channels,
        n_steps=cfg.n_steps,
        output_path=out_path,
    )


def build_tcn_npz_files(
    ctg_parquet: str | Path,
    splits_csv: str | Path,
    output_dir: str | Path,
    cfg: SequenceBuildConfig,
) -> list[SplitBuildStats]:
    ctg_parquet = Path(ctg_parquet)
    splits_df = pd.read_csv(splits_csv)
    required = {"BabyID", "split", "target", "apgar5"}
    missing = required - set(splits_df.columns)
    if missing:
        raise ValueError(f"splits csv missing columns: {sorted(missing)}")

    out_dir = Path(output_dir)
    stats: list[SplitBuildStats] = []
    for split_name in ["train", "val", "test"]:
        split_part = splits_df[splits_df["split"] == split_name].copy()
        if split_part.empty:
            continue
        out_path = out_dir / f"{split_name}.npz"
        split_stats = _process_split_chunked(ctg_parquet=ctg_parquet, split_df=split_part, cfg=cfg, out_path=out_path)
        stats.append(split_stats)
    return stats
