from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd


AGG_FEATURE_SQL = """
WITH raw AS (
    SELECT
        c.BabyID,
        c."Timestamp" AS ts,
        CAST(c.FHR AS DOUBLE) AS fhr,
        CAST(c.toco AS DOUBLE) AS toco
    FROM read_parquet(?) c
    INNER JOIN split_map s ON c.BabyID = s.BabyID
),
ordered AS (
    SELECT * FROM raw ORDER BY BabyID, ts
),
diffed AS (
    SELECT
        BabyID,
        ts,
        fhr,
        toco,
        ABS(fhr - LAG(fhr) OVER (PARTITION BY BabyID ORDER BY ts)) AS fhr_abs_diff,
        ABS(toco - LAG(toco) OVER (PARTITION BY BabyID ORDER BY ts)) AS toco_abs_diff
    FROM ordered
)
SELECT
    BabyID,
    COUNT(*) AS n_rows,
    epoch(max(ts)) - epoch(min(ts)) AS duration_seconds,
    AVG(fhr) AS fhr_mean,
    STDDEV_SAMP(fhr) AS fhr_std,
    MIN(fhr) AS fhr_min,
    MAX(fhr) AS fhr_max,
    AVG(CASE WHEN fhr = 0 THEN 1 ELSE 0 END) AS fhr_zero_frac,
    AVG(CASE WHEN fhr > 0 THEN fhr END) AS fhr_nonzero_mean,
    STDDEV_SAMP(CASE WHEN fhr > 0 THEN fhr END) AS fhr_nonzero_std,
    quantile_cont(fhr, 0.10) AS fhr_q10,
    quantile_cont(fhr, 0.50) AS fhr_q50,
    quantile_cont(fhr, 0.90) AS fhr_q90,
    AVG(toco) AS toco_mean,
    STDDEV_SAMP(toco) AS toco_std,
    MIN(toco) AS toco_min,
    MAX(toco) AS toco_max,
    quantile_cont(toco, 0.10) AS toco_q10,
    quantile_cont(toco, 0.50) AS toco_q50,
    quantile_cont(toco, 0.90) AS toco_q90,
    AVG(fhr_abs_diff) AS fhr_abs_diff_mean,
    AVG(toco_abs_diff) AS toco_abs_diff_mean
FROM diffed
GROUP BY BabyID
"""


def build_aggregate_features(
    ctg_parquet: str | Path,
    split_map: pd.DataFrame,
) -> pd.DataFrame:
    con = duckdb.connect(database=":memory:")
    try:
        con.register("split_map", split_map[["BabyID"]])
        features = con.execute(AGG_FEATURE_SQL, [str(ctg_parquet)]).df()
    finally:
        con.close()

    features = features.sort_values("BabyID").reset_index(drop=True)
    numeric_cols = [c for c in features.columns if c != "BabyID"]
    features[numeric_cols] = features[numeric_cols].astype("float32")
    features["n_rows"] = features["n_rows"].astype("int32")
    return features
