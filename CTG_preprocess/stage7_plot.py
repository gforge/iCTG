from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import pandas as pd

from config import DEFAULT_STAGE7_CTG_PARQUET, DEFAULT_STAGE7_REGISTRY_CSV


def _safe(path: Path) -> str:
    return str(path).replace("'", "''")


def _pick_random_baby(con: duckdb.DuckDBPyConnection, ctg_path: Path) -> str:
    safe_ctg = _safe(ctg_path)
    row = con.execute(
        f"SELECT BabyID FROM read_parquet('{safe_ctg}') GROUP BY BabyID ORDER BY random() LIMIT 1"
    ).fetchone()
    if not row:
        raise RuntimeError("No BabyID found in CTG parquet.")
    return row[0]


def _pick_random_baby_with_apgar(
    con: duckdb.DuckDBPyConnection, ctg_path: Path, registry_path: Path, apgar: int
) -> str:
    safe_ctg = _safe(ctg_path)
    safe_reg = _safe(registry_path)
    row = con.execute(
        f"""
        SELECT r.BabyID
        FROM read_csv_auto('{safe_reg}', header=true) r
        JOIN (SELECT DISTINCT BabyID FROM read_parquet('{safe_ctg}')) c USING (BabyID)
        WHERE r.apgar5 = {int(apgar)}
        ORDER BY random()
        LIMIT 1
        """
    ).fetchone()
    if not row:
        raise RuntimeError(f"No BabyID found with apgar5={apgar}.")
    return row[0]


def _load_apgar(con: duckdb.DuckDBPyConnection, registry_path: Path, baby_id: str) -> str:
    safe_reg = _safe(registry_path)
    safe_baby = baby_id.replace("'", "''")
    row = con.execute(
        f"SELECT apgar5 FROM read_csv_auto('{safe_reg}', header=true) WHERE BabyID = '{safe_baby}' LIMIT 1"
    ).fetchone()
    if not row:
        return "unknown"
    return str(row[0])


def _load_ctg(con: duckdb.DuckDBPyConnection, ctg_path: Path, baby_id: str):
    safe_ctg = _safe(ctg_path)
    safe_baby = baby_id.replace("'", "''")
    return con.execute(
        f"SELECT Timestamp, FHR, toco FROM read_parquet('{safe_ctg}') WHERE BabyID = '{safe_baby}' ORDER BY Timestamp"
    ).df()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Stage 7 CTG for a BabyID with Apgar score.")
    parser.add_argument("--ctg", type=str, default=DEFAULT_STAGE7_CTG_PARQUET)
    parser.add_argument("--registry", type=str, default=DEFAULT_STAGE7_REGISTRY_CSV)
    parser.add_argument("--baby-id", type=str, default=None)
    parser.add_argument("--apgar", type=int, default=None)
    parser.add_argument("--out", type=str, default="stage7_plot.png")
    args = parser.parse_args()

    ctg_path = Path(args.ctg)
    registry_path = Path(args.registry)

    con = duckdb.connect()

    if args.baby_id:
        baby_id = args.baby_id
    elif args.apgar is not None:
        baby_id = _pick_random_baby_with_apgar(con, ctg_path, registry_path, args.apgar)
    else:
        baby_id = _pick_random_baby(con, ctg_path)
    apgar = _load_apgar(con, registry_path, baby_id)
    df = _load_ctg(con, ctg_path, baby_id)

    if df.empty:
        raise RuntimeError(f"No CTG rows found for BabyID {baby_id}.")

    anchor = df["Timestamp"].max()
    start = anchor - pd.Timedelta(hours=1)
    df_window = df[(df["Timestamp"] >= start) & (df["Timestamp"] <= anchor)]
    if df_window.empty:
        df_window = df
        start = df["Timestamp"].min()
        anchor = df["Timestamp"].max()

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(12, 6))
    axes[0].plot(df_window["Timestamp"], df_window["FHR"], linestyle="-", marker=".", markersize=2)
    axes[0].set_ylabel("FHR")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df_window["Timestamp"], df_window["toco"], linestyle="-", marker=".", markersize=2, color="tab:orange")
    axes[1].set_ylabel("Toco")
    axes[1].grid(True, alpha=0.3)

    axes[1].set_xlim(start, anchor)
    fig.suptitle(f"BabyID: {baby_id} | apgar5: {apgar}")
    fig.autofmt_xdate()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    main()
