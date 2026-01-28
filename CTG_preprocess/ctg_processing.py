from __future__ import annotations

import base64
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

FOUR_HZ = "250ms"
ONE_HZ = "1s"
FHR_MIN = 30
FHR_MAX = 240


def format_patient_id(pn: str) -> str:
    pn = pn.strip()
    if not pn:
        return ""
    if "-" in pn:
        return pn
    if len(pn) >= 9:
        return f"{pn[:8]}-{pn[8:]}"
    return pn


def _decode_toco_values(value: object) -> list[float]:
    if value is None:
        return [np.nan] * 4
    if isinstance(value, float) and np.isnan(value):
        return [np.nan] * 4

    raw: bytes
    try:
        if isinstance(value, bytes):
            raw = value if len(value) == 4 else base64.b64decode(value)
        else:
            raw = base64.b64decode(str(value))
    except (ValueError, TypeError):
        return [np.nan] * 4

    if len(raw) != 4:
        return [np.nan] * 4

    return [float(b) for b in raw]


def _row_to_4hz(row: pd.Series) -> tuple[list[pd.Timestamp], list[float], list[float]]:
    base_ts = row["Timestamp"]
    if pd.isna(base_ts):
        return [], [], []

    fhr_values = [
        row.get("Hr1_0"),
        row.get("Hr1_1"),
        row.get("Hr1_2"),
        row.get("Hr1_3"),
    ]
    toco_values = _decode_toco_values(row.get("Toco_Values"))

    timestamps = [
        base_ts + pd.Timedelta(milliseconds=250 * i) for i in range(4)
    ]
    return timestamps, fhr_values, toco_values


def _build_4hz_series(df: pd.DataFrame) -> pd.DataFrame:
    timestamps: list[pd.Timestamp] = []
    fhr: list[float] = []
    toco: list[float] = []

    for _, row in df.iterrows():
        ts_values, fhr_values, toco_values = _row_to_4hz(row)
        timestamps.extend(ts_values)
        fhr.extend(fhr_values)
        toco.extend(toco_values)

    if not timestamps:
        return pd.DataFrame(columns=["Timestamp", "FHR", "toco"])

    out = pd.DataFrame(
        {"Timestamp": timestamps, "FHR": fhr, "toco": toco}
    ).sort_values("Timestamp")
    out["FHR"] = pd.to_numeric(out["FHR"], errors="coerce")
    out["toco"] = pd.to_numeric(out["toco"], errors="coerce")
    return out.reset_index(drop=True)


def _build_1hz_series(df: pd.DataFrame, mode: str = "mean") -> pd.DataFrame:
    timestamps: list[pd.Timestamp] = []
    fhr: list[float] = []
    toco: list[float] = []

    for _, row in df.iterrows():
        base_ts = row["Timestamp"]
        if pd.isna(base_ts):
            continue

        fhr_values = [
            row.get("Hr1_0"),
            row.get("Hr1_1"),
            row.get("Hr1_2"),
            row.get("Hr1_3"),
        ]
        fhr_values = pd.to_numeric(pd.Series(fhr_values), errors="coerce")

        toco_values = pd.to_numeric(
            pd.Series(_decode_toco_values(row.get("Toco_Values"))),
            errors="coerce",
        )

        if mode == "first":
            fhr_value = fhr_values.iloc[0]
            toco_value = toco_values.iloc[0]
        else:
            fhr_value = fhr_values.mean(skipna=True)
            toco_value = toco_values.mean(skipna=True)

        timestamps.append(base_ts)
        fhr.append(float(fhr_value) if pd.notna(fhr_value) else np.nan)
        toco.append(float(toco_value) if pd.notna(toco_value) else np.nan)

    if not timestamps:
        return pd.DataFrame(columns=["Timestamp", "FHR", "toco"])

    out = pd.DataFrame(
        {"Timestamp": timestamps, "FHR": fhr, "toco": toco}
    ).sort_values("Timestamp")
    out["FHR"] = pd.to_numeric(out["FHR"], errors="coerce")
    out["toco"] = pd.to_numeric(out["toco"], errors="coerce")
    return out.reset_index(drop=True)


def load_ctg_data(
    parquet_paths: str | Path | Iterable[str | Path] | ds.Dataset,
    pn: str,
    birth_day: date | None,
    sample_rate_hz: int = 4,
    downsample_mode: str = "mean",
) -> pd.DataFrame | None:
    patient_id = format_patient_id(pn)
    if not patient_id:
        return None

    columns = [
        "PatientID",
        "Timestamp",
        "Hr1_0",
        "Hr1_1",
        "Hr1_2",
        "Hr1_3",
        "Toco_Values",
    ]

    dataset = (
        parquet_paths
        if isinstance(parquet_paths, ds.Dataset)
        else ds.dataset(parquet_paths, format="parquet")
    )
    filter_expr = ds.field("PatientID") == patient_id

    if birth_day is not None and "Timestamp" in dataset.schema.names:
        ts_field = dataset.schema.field("Timestamp")
        if pa.types.is_timestamp(ts_field.type):
            start_dt = datetime.combine(birth_day - timedelta(days=1), time.min)
            end_dt = datetime.combine(birth_day + timedelta(days=1), time.min)
            filter_expr = filter_expr & (ds.field("Timestamp") >= start_dt)
            filter_expr = filter_expr & (ds.field("Timestamp") < end_dt)

    table = dataset.to_table(columns=columns, filter=filter_expr)
    if table.num_rows == 0:
        return None

    df = table.to_pandas()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

    if birth_day is not None:
        start_dt = datetime.combine(birth_day - timedelta(days=1), time.min)
        end_dt = datetime.combine(birth_day + timedelta(days=1), time.min)
        df = df[(df["Timestamp"] >= start_dt) & (df["Timestamp"] < end_dt)]

    if df.empty:
        return None

    if sample_rate_hz == 1:
        return _build_1hz_series(df, mode=downsample_mode)

    return _build_4hz_series(df)


def _apply_hampel_filter(
    df: pd.DataFrame,
    sample_interval_seconds: float,
    window_seconds: float = 2.0,
    n_sigmas: float = 3.0,
) -> None:
    window_points = max(3, int(round(window_seconds / sample_interval_seconds)) * 2 + 1)
    fhr = df["FHR"].where(~df["filtered"])
    median = fhr.rolling(window=window_points, center=True).median()
    mad = (fhr - median).abs().rolling(window=window_points, center=True).median()
    threshold = n_sigmas * 1.4826 * mad
    outliers = (fhr - median).abs() > threshold
    outliers = outliers & fhr.notna() & threshold.notna()
    df.loc[outliers, "filtered"] = True


def _apply_spike_filter(df: pd.DataFrame, jump_threshold: float = 20.0) -> None:
    fhr = df["FHR"]
    prev_diff = (fhr - fhr.shift(1)).abs()
    next_diff = (fhr - fhr.shift(-1)).abs()
    spike = (prev_diff > jump_threshold) & (next_diff > jump_threshold)
    spike = spike & fhr.notna()
    df.loc[spike, "filtered"] = True


def _interpolate_short_gaps(
    df: pd.DataFrame, sample_interval_seconds: float, max_gap_seconds: float = 2.0
) -> None:
    max_gap_points = max(1, int(round(max_gap_seconds / sample_interval_seconds)))
    fhr = df["FHR"].where(~df["filtered"])
    interpolated = fhr.interpolate(
        method="linear",
        limit=max_gap_points,
        limit_area="inside",
    )
    filled = df["filtered"] & interpolated.notna()
    df.loc[filled, "FHR"] = interpolated[filled]
    df.loc[filled, "filtered"] = False


def _kalman_smooth(df: pd.DataFrame) -> None:
    process_var = 1.0
    measurement_var = 4.0

    estimate = None
    estimate_var = 1.0

    smoothed: list[float] = []
    for value, filtered in zip(df["FHR"], df["filtered"], strict=False):
        if estimate is None:
            if filtered or pd.isna(value):
                smoothed.append(np.nan)
                continue
            estimate = float(value)
            smoothed.append(estimate)
            continue

        estimate_var += process_var

        if filtered or pd.isna(value):
            smoothed.append(np.nan)
            continue

        kalman_gain = estimate_var / (estimate_var + measurement_var)
        estimate = estimate + kalman_gain * (float(value) - estimate)
        estimate_var = (1.0 - kalman_gain) * estimate_var
        smoothed.append(estimate)

    df["FHR"] = smoothed


def filter_ctg_data(
    df: pd.DataFrame, min_data: float = 0.5, sample_rate_hz: int = 4
) -> pd.DataFrame | None:
    if df.empty:
        return None

    df = df.copy()
    df["FHR"] = pd.to_numeric(df["FHR"], errors="coerce").astype(float)
    if "toco" in df.columns:
        df["toco"] = pd.to_numeric(df["toco"], errors="coerce")

    if df["Timestamp"].duplicated().any():
        df = df.groupby("Timestamp", as_index=False).mean(numeric_only=True)

    df["filtered"] = False

    out_of_range = df["FHR"].isna() | (df["FHR"] < FHR_MIN) | (df["FHR"] > FHR_MAX)
    df.loc[out_of_range, "filtered"] = True

    valid_mask = ~df["filtered"]
    if not valid_mask.any():
        return None

    final_ts = df.loc[valid_mask, "Timestamp"].max()
    if sample_rate_hz == 1:
        freq = ONE_HZ
        expected_points = 3600
        sample_interval = 1.0
    else:
        freq = FOUR_HZ
        expected_points = 14400
        sample_interval = 0.25

    full_index = pd.date_range(end=final_ts, periods=expected_points, freq=freq)

    df = df.set_index("Timestamp").reindex(full_index)
    df.index.name = "Timestamp"

    df["filtered"] = df["filtered"].fillna(True).astype(bool)

    _apply_hampel_filter(df, sample_interval)
    _apply_spike_filter(df)
    _interpolate_short_gaps(df, sample_interval)
    _kalman_smooth(df)

    valid_ratio = (~df["filtered"]).mean()
    if valid_ratio < min_data:
        return None

    return df.reset_index()
