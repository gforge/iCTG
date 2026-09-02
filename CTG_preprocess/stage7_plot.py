from __future__ import annotations

import argparse
from pathlib import Path
from typing import NamedTuple

import duckdb
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

from config import DEFAULT_STAGE7_CTG_PARQUET, DEFAULT_STAGE7_REGISTRY_CSV


class _BandStyle(NamedTuple):
    """Fill styling for a signal-quality band (used for both the span and its legend entry)."""

    color: str
    alpha: float
    label: str


def _safe(path: Path) -> str:
    return str(path).replace("'", "''")


def _pick_random_babies(
    con: duckdb.DuckDBPyConnection,
    ctg_path: Path,
    registry_path: Path,
    count: int,
    apgar: int | None = None,
) -> list[str]:
    del ctg_path
    safe_reg = _safe(registry_path)
    if apgar is None:
        rows = con.execute(
            f"""
            SELECT DISTINCT BabyID
            FROM read_csv_auto('{safe_reg}', header=true)
            WHERE BabyID IS NOT NULL
            ORDER BY random()
            LIMIT {int(count)}
            """
        ).fetchall()
    else:
        rows = con.execute(
            f"""
            SELECT DISTINCT BabyID
            FROM read_csv_auto('{safe_reg}', header=true)
            WHERE BabyID IS NOT NULL
              AND TRY_CAST(apgar5 AS INTEGER) = {int(apgar)}
            ORDER BY random()
            LIMIT {int(count)}
            """
        ).fetchall()

    if not rows:
        if apgar is None:
            raise RuntimeError("No BabyID found in CTG parquet.")
        raise RuntimeError(f"No BabyID found with apgar5={apgar}.")
    return [row[0] for row in rows]


def _load_apgar(con: duckdb.DuckDBPyConnection, registry_path: Path, baby_id: str) -> str:
    safe_reg = _safe(registry_path)
    safe_baby = baby_id.replace("'", "''")
    row = con.execute(
        f"SELECT apgar5 FROM read_csv_auto('{safe_reg}', header=true) WHERE BabyID = '{safe_baby}' LIMIT 1"
    ).fetchone()
    if not row:
        return "unknown"
    return str(row[0])


def _ctg_columns(con: duckdb.DuckDBPyConnection, ctg_path: Path) -> set[str]:
    safe_ctg = _safe(ctg_path)
    return {
        row[0]
        for row in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{safe_ctg}')").fetchall()
    }


def _load_ctg(con: duckdb.DuckDBPyConnection, ctg_path: Path, baby_id: str):
    safe_ctg = _safe(ctg_path)
    safe_baby = baby_id.replace("'", "''")
    columns = _ctg_columns(con, ctg_path)
    quality_select = (
        "Hr1_SignalQuality AS signal_quality"
        if "Hr1_SignalQuality" in columns
        else "CAST(NULL AS VARCHAR) AS signal_quality"
    )
    return con.execute(
        f"""
        SELECT Timestamp, FHR, toco, {quality_select}
        FROM read_parquet('{safe_ctg}')
        WHERE BabyID = '{safe_baby}'
        ORDER BY Timestamp
        """
    ).df()


def _set_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _prepare_window(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    df = df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df["signal_quality"] = df["signal_quality"].astype("string").str.upper().str.strip()

    anchor = df["Timestamp"].max()
    start = anchor - pd.Timedelta(hours=1)
    df_window = df[(df["Timestamp"] >= start) & (df["Timestamp"] <= anchor)].copy()
    if df_window.empty:
        df_window = df.copy()
        start = df["Timestamp"].min()
        anchor = df["Timestamp"].max()

    df_window["minutes_before_end"] = (df_window["Timestamp"] - anchor).dt.total_seconds() / 60.0
    return df_window, start, anchor


def _quality_intervals(
    df_window: pd.DataFrame, qualities: set[str]
) -> list[tuple[float, float, str]]:
    if "signal_quality" not in df_window.columns or df_window["signal_quality"].isna().all():
        return []

    work = df_window[["Timestamp", "minutes_before_end", "signal_quality"]].copy()
    work = work.sort_values("Timestamp")
    work["signal_quality"] = work["signal_quality"].astype("string").str.upper().str.strip()
    work["is_target"] = work["signal_quality"].isin(qualities)
    gaps = work["Timestamp"].diff().dt.total_seconds().fillna(0)
    run_break = (
        work["is_target"].ne(work["is_target"].shift(fill_value=False))
        | work["signal_quality"].ne(work["signal_quality"].shift(fill_value=""))
        | (gaps > 2)
    )
    work["run_id"] = pd.Series(run_break.to_numpy(dtype=bool), index=work.index).cumsum()

    intervals: list[tuple[float, float, str]] = []
    for (_, quality), group in work[work["is_target"]].groupby(["run_id", "signal_quality"]):
        start = float(group["minutes_before_end"].iloc[0])
        end = float(group["minutes_before_end"].iloc[-1])
        if end <= start:
            end = start + 1 / 60.0
        else:
            end += 1 / 60.0
        intervals.append((start, min(end, 0.0), str(quality)))
    return intervals


def _add_quality_bands(axes: list[plt.Axes], df_window: pd.DataFrame, mode: str) -> list[Patch]:
    if mode == "none":
        return []

    qualities = {"R"} if mode == "bad" else {"R", "Y"}
    intervals = _quality_intervals(df_window, qualities)
    if not intervals:
        return []

    colors = {
        "R": _BandStyle(color="#dc2626", alpha=0.12, label="Poor signal quality (R)"),
        "Y": _BandStyle(color="#f59e0b", alpha=0.10, label="Intermediate signal quality (Y)"),
    }
    seen: set[str] = set()
    for start, end, quality in intervals:
        if quality not in colors:
            continue
        band = colors[quality]
        for ax in axes:
            ax.axvspan(start, end, color=band.color, alpha=band.alpha, linewidth=0)
        seen.add(quality)

    return [
        Patch(
            facecolor=colors[quality].color,
            alpha=colors[quality].alpha,
            label=colors[quality].label,
        )
        for quality in ("R", "Y")
        if quality in seen
    ]


def _line_arrays_with_gaps(
    df_window: pd.DataFrame,
    column: str,
    max_gap_seconds: float = 2.0,
    mask_nonpositive: bool = True,
) -> tuple[list[float], list[float]]:
    values = pd.to_numeric(df_window[column], errors="coerce")
    if mask_nonpositive:
        values = values.mask(values <= 0)

    timestamps = pd.to_datetime(df_window["Timestamp"])
    minutes = pd.to_numeric(df_window["minutes_before_end"], errors="coerce")
    gaps = timestamps.diff().dt.total_seconds()

    x_values: list[float] = []
    y_values: list[float] = []
    previous_x: float | None = None
    for idx, (minute, value) in enumerate(zip(minutes, values, strict=True)):
        current_x = float(minute)
        if idx and gaps.iloc[idx] > max_gap_seconds and previous_x is not None:
            x_values.append((previous_x + current_x) / 2)
            y_values.append(float("nan"))

        x_values.append(current_x)
        y_values.append(float(value) if pd.notna(value) else float("nan"))
        previous_x = current_x

    return x_values, y_values


def _plot_ctg(
    df_window: pd.DataFrame,
    apgar: str,
    out_path: Path,
    quality_bands: str,
    title: str,
    show_apgar: bool,
) -> None:
    _set_publication_style()

    fig, axes_arr = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(11.2, 5.8),
        gridspec_kw={"height_ratios": [1.35, 1.0], "hspace": 0.08},
        constrained_layout=True,
    )
    axes = list(axes_arr)
    legend_handles = _add_quality_bands(axes, df_window, quality_bands)

    fhr_x, fhr_y = _line_arrays_with_gaps(df_window, "FHR")
    axes[0].plot(
        fhr_x,
        fhr_y,
        color="#1f77b4",
        linewidth=0.9,
    )
    axes[0].set_ylabel("FHR (bpm)")

    toco_x, toco_y = _line_arrays_with_gaps(df_window, "toco")
    axes[1].plot(
        toco_x,
        toco_y,
        color="#b45309",
        linewidth=0.9,
    )
    axes[1].set_ylabel("Uterine activity (TOCO, a.u.)")
    axes[1].set_xlabel("Time before end of CTG segment (min)")
    axes[1].set_xlim(-60, 0)

    for ax in axes:
        ax.grid(True, color="#e5e7eb", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", colors="#374151")

    title_parts = [title]
    if show_apgar:
        title_parts.append(f"Apgar after 5 min: {apgar}")
    fig.suptitle(" | ".join(title_parts), x=0.02, ha="left", fontsize=12, weight="bold")

    if legend_handles:
        axes[0].legend(handles=legend_handles, loc="upper right", frameon=False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _indexed_output_path(base_path: Path, index: int, total: int, apgar: str) -> Path:
    if total == 1:
        return base_path

    apgar_label = "".join(ch if ch.isalnum() else "_" for ch in str(apgar)) or "unknown"
    if base_path.suffix:
        return base_path.with_name(
            f"{base_path.stem}_{index:02d}_apgar{apgar_label}{base_path.suffix}"
        )
    return base_path / f"ctg_trace_{index:02d}_apgar{apgar_label}.png"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot an anonymized Stage 7 CTG trace with optional signal-quality bands."
    )
    parser.add_argument("--ctg", type=str, default=DEFAULT_STAGE7_CTG_PARQUET)
    parser.add_argument("--registry", type=str, default=DEFAULT_STAGE7_REGISTRY_CSV)
    parser.add_argument("--baby-id", type=str, default=None)
    parser.add_argument("--apgar", type=int, default=None)
    parser.add_argument("--out", type=str, default="stage7_plot.png")
    parser.add_argument(
        "--random-count",
        type=int,
        default=1,
        help=(
            "Generate N random traces. Can be combined with --apgar to sample from a specific "
            "Apgar score. Cannot be combined with --baby-id."
        ),
    )
    parser.add_argument(
        "--quality-bands",
        choices=["none", "bad", "all"],
        default="bad",
        help="Background shading for Hr1_SignalQuality: none, bad=R only, all=R and Y.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Final-hour CTG segment",
        help="Figure title. BabyID is never included in the figure by default.",
    )
    parser.add_argument(
        "--hide-apgar", action="store_true", help="Do not include Apgar in the figure title."
    )
    args = parser.parse_args()

    ctg_path = Path(args.ctg)
    registry_path = Path(args.registry)
    out_path = Path(args.out)

    if args.random_count < 1:
        raise ValueError("--random-count must be at least 1.")
    if args.baby_id and args.random_count != 1:
        raise ValueError("--random-count cannot be combined with --baby-id.")

    con = duckdb.connect()

    if args.baby_id:
        baby_ids = [args.baby_id]
    elif args.apgar is not None:
        baby_ids = _pick_random_babies(con, ctg_path, registry_path, args.random_count, args.apgar)
    else:
        baby_ids = _pick_random_babies(con, ctg_path, registry_path, args.random_count)

    for index, baby_id in enumerate(baby_ids, start=1):
        apgar = _load_apgar(con, registry_path, baby_id)
        df = _load_ctg(con, ctg_path, baby_id)

        if df.empty:
            raise RuntimeError(f"No CTG rows found for BabyID {baby_id}.")

        indexed_out_path = _indexed_output_path(out_path, index, len(baby_ids), apgar)
        df_window, _, _ = _prepare_window(df)
        _plot_ctg(
            df_window=df_window,
            apgar=apgar,
            out_path=indexed_out_path,
            quality_bands=args.quality_bands,
            title=args.title,
            show_apgar=not args.hide_apgar,
        )
        print(f"Saved plot: {indexed_out_path}")


if __name__ == "__main__":
    main()
